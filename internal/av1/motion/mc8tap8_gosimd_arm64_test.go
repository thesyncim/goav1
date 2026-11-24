// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"bytes"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestMC8EightTapMatchesPureGo drives the eight-tap put and prep kernels
// (a sharp axis, the other axis any kind) against the pure-Go convolves at
// every phase pair, interior and edge-clamped sources and extreme patterns,
// and requires the kernels to accept every multiple-of-8 width.
func TestMC8EightTapMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	sizes := [][2]int{{8, 8}, {8, 1}, {8, 2}, {8, 5}, {16, 7}, {16, 16}, {32, 32}, {24, 8}, {64, 64}, {128, 128}, {8, 32}, {64, 16}, {16, 64}}
	others := []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpBilinear, InterpMultiTapSharp}
	const rw, rh = 200, 180
	var scratch ConvolveScratch
	cscratch := new(CompoundConvolveScratch)
	for pattern := range 5 {
		ref := mc8TestRef(rng, rw, rh, rw+24, pattern)
		for _, sz := range sizes {
			w, h := sz[0], sz[1]
			for iter := range 48 {
				filters := InterpFilters{X: InterpMultiTapSharp, Y: others[rng.Intn(len(others))]}
				if iter&1 != 0 {
					filters.X, filters.Y = filters.Y, filters.X
				}
				subX, subY := rng.Intn(16), rng.Intn(16)
				switch iter % 6 {
				case 0:
					subX = 0
				case 1:
					subY = 0
				}
				if subX == 0 && subY == 0 {
					subX = 1 + rng.Intn(15)
				}
				var refX, refY int
				switch iter % 4 {
				case 0:
					refX = 4 + rng.Intn(rw-w-12)
					refY = 4 + rng.Intn(rh-h-12)
				case 1:
					refX = -rng.Intn(w + 6)
					refY = -rng.Intn(h + 6)
				case 2:
					refX = rw - w + rng.Intn(w+6)
					refY = rh - h + rng.Intn(h+6)
				default:
					refX = rng.Intn(rw+2*w) - w - 3
					refY = rng.Intn(rh+2*h) - h - 3
				}
				kx, ky := mc8Kind(filters.X, w), mc8Kind(filters.Y, h)
				eight := mc8NeedsEightTaps(subX, subY, kx, ky)

				dw, dh := w+16, h+8
				want := frame.Plane{Pix: make([]byte, dw*dh), Stride: dw, Width: dw, Height: dh}
				for i := range want.Pix {
					want.Pix[i] = byte(i * 7)
				}
				got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: dw, Width: dw, Height: dh}
				mc8RefPut(want, ref, 3, 2, refX, refY, w, h, w, h, subX, subY, filters)
				if !Put8(&got, &ref, 3, 2, refX, refY, w, h, w, h, subX, subY, filters, &scratch) {
					t.Fatalf("Put8 declined w=%d h=%d filters=%v sub=%d,%d eight=%v", w, h, filters, subX, subY, eight)
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("put pattern %d w=%d h=%d filters=%v sub=%d,%d ref=%d,%d: mismatch", pattern, w, h, filters, subX, subY, refX, refY)
				}

				wantP := make([]uint16, w*h+8)
				for i := range wantP {
					wantP[i] = uint16(i * 13)
				}
				gotP := slices.Clone(wantP)
				mc8RefPrep(wantP, ref, refX, refY, w, h, subX, subY, filters)
				if !mc8PrepFast(gotP[:w*h], &ref, refX, refY, w, h, w, h, subX, subY, filters, cscratch) {
					t.Fatalf("mc8PrepFast declined w=%d h=%d filters=%v sub=%d,%d", w, h, filters, subX, subY)
				}
				if !slices.Equal(gotP, wantP) {
					for i := range wantP {
						if gotP[i] != wantP[i] {
							t.Fatalf("prep pattern %d w=%d h=%d filters=%v sub=%d,%d ref=%d,%d: first mismatch at x=%d y=%d got %d want %d", pattern, w, h, filters, subX, subY, refX, refY, i%w, i/w, gotP[i], wantP[i])
						}
					}
				}
			}
		}
	}
}

func TestMC8EightTapNoAllocs(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	ref := mc8TestRef(rng, 96, 96, 96, 0)
	dst := frame.Plane{Pix: make([]byte, 32*32), Stride: 32, Width: 32, Height: 32}
	out := make([]uint16, 32*32)
	var scratch ConvolveScratch
	cscratch := new(CompoundConvolveScratch)
	f := InterpFilters{X: InterpMultiTapSharp, Y: InterpMultiTapSharp}
	if n := testing.AllocsPerRun(20, func() {
		Put8(&dst, &ref, 0, 0, 30, 30, 32, 32, 32, 32, 5, 9, f, &scratch)
		Put8(&dst, &ref, 0, 0, -3, 80, 32, 32, 32, 32, 5, 9, f, &scratch)
		mc8PrepFast(out, &ref, 30, 30, 32, 32, 32, 32, 7, 3, f, cscratch)
	}); n != 0 {
		t.Fatalf("allocs = %v", n)
	}
}
