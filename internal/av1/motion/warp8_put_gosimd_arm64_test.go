// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func warpPutPredict(simd bool, dst, ref frame.Plane, matX, matY, w, h int, m [6]int32, a, b, g, d int16, ssX, ssY bool) error {
	saved := warpGoSIMDEnabled
	warpGoSIMDEnabled = simd
	defer func() { warpGoSIMDEnabled = saved }()
	return PredictWarpedPlaneBlockToScratchBitDepth(dst, ref, 1, 8, matX, matY, w, h, m, a, b, g, d, ssX, ssY)
}

// TestWarpPut8GoSIMDMatchesScalar compares the single-reference 8-bit warp with
// and without the Go SIMD kernels (the int16-intermediate 8x8 block against
// the scalar int32 path) over random warp parameters, block sizes, interior,
// edge and outside positions, subsampling and extreme sample patterns.
func TestWarpPut8GoSIMDMatchesScalar(t *testing.T) {
	if !warpGoSIMDEnabled {
		t.Skip("Go SIMD warp disabled")
	}
	rng := rand.New(rand.NewSource(11))
	sizes := [][2]int{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 64}, {128, 128}, {8, 32}, {32, 8}, {4, 4}, {12, 8}, {20, 12}}
	used, fell := 0, 0
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		iters := 160
		if w*h >= 64*64 {
			iters = 24
		}
		for iter := range iters {
			pattern := iter % 4
			ssX, ssY := iter%3 == 1, iter%5 == 2
			rw, rh := 24+rng.Intn(160), 24+rng.Intn(160)
			ref := warpCompoundRef(rng, rw, rh, pattern)
			matX := rng.Intn(rw+64) - 32
			matY := rng.Intn(rh+64) - 32
			span, div := 8192, 8
			if iter%7 == 0 {
				span, div = 1<<15, 1
			}
			r := func() int32 { return int32(rng.Intn(span) - span/2) }
			matrix := [6]int32{
				int32(rng.Intn(1<<20)) - 1<<19,
				int32(rng.Intn(1<<20)) - 1<<19,
				1<<16 + r(), r(), r(), 1<<16 + r(),
			}
			alpha, beta := warpReduce(int(r())/div), warpReduce(int(r())/div)
			gamma, delta := warpReduce(int(r())/div), warpReduce(int(r())/div)
			if iter%6 == 3 {
				gamma = 0
			}
			if iter%11 == 5 {
				alpha, beta, gamma, delta = 0, 0, 0, 0
			}
			want := frame.Plane{Pix: make([]byte, w*h), Stride: w, Width: w, Height: h}
			got := frame.Plane{Pix: make([]byte, w*h), Stride: w, Width: w, Height: h}
			werr := warpPutPredict(false, want, ref, matX, matY, w, h, matrix, alpha, beta, gamma, delta, ssX, ssY)
			gerr := warpPutPredict(true, got, ref, matX, matY, w, h, matrix, alpha, beta, gamma, delta, ssX, ssY)
			name := fmt.Sprintf("%dx%d iter=%d", w, h, iter)
			if (werr == nil) != (gerr == nil) {
				t.Fatalf("%s: error mismatch %v / %v", name, werr, gerr)
			}
			for k := range w * h {
				if got.Pix[k] != want.Pix[k] {
					t.Fatalf("%s: sample (%d,%d) got %d want %d", name, k%w, k/w, got.Pix[k], want.Pix[k])
				}
			}
			if w >= 8 && h >= 8 {
				sx, sy := 0, 0
				if ssX {
					sx = 1
				}
				if ssY {
					sy = 1
				}
				ix4, sx4, iy4, sy4 := warpBlockOrigin(matY, matX, matrix, int(alpha), int(beta), int(gamma), int(delta), sx, sy)
				var probe [64]byte
				if warpPut8x8Dispatch(probe[:], 8, ref, ix4, sx4, iy4, sy4, int(alpha), int(beta), int(gamma), int(delta)) {
					used++
				} else {
					fell++
				}
			}
		}
	}
	if used == 0 || fell == 0 {
		t.Fatalf("kernel coverage: used=%d fell back=%d", used, fell)
	}
}

// TestWarpPut8GoSIMDAllPhases drives every filter row through both passes
// with extreme sample patterns, gamma zero and nonzero.
func TestWarpPut8GoSIMDAllPhases(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for pattern := range 4 {
		ref := warpCompoundRef(rng, 48, 48, pattern)
		for hp := range warpedFilter {
			for _, vp := range []int{hp, 0, len(warpedFilter) - 1, 64} {
				for _, gamma := range []int{0} {
					sx4 := (hp - warpedPixelPrecShifts) << warpedDiffPrecBits
					sy4 := (vp - warpedPixelPrecShifts) << warpedDiffPrecBits
					var tmp warpTmp
					warpHorizontal8Resident(&tmp, ref, 20, sx4, 20, sy4, 0, 0, round0Bits, 8+filterBits-1)
					want := frame.Plane{Pix: make([]byte, 64), Stride: 8, Width: 8, Height: 8}
					warpVertical8Full(want, &tmp, 0, 0, 0, 0, sy4, gamma, 0, round1Bits, 8+2*filterBits-round0Bits)
					var got [64]byte
					if !warpPut8x8Dispatch(got[:], 8, ref, 20, sx4, 20, sy4, 0, 0, gamma, 0) {
						t.Fatalf("pattern=%d phases=%d/%d: kernel declined", pattern, hp, vp)
					}
					for k := range got {
						if got[k] != want.Pix[k] {
							t.Fatalf("pattern=%d phases=%d/%d gamma=%d: sample %d got %d want %d", pattern, hp, vp, gamma, k, got[k], want.Pix[k])
						}
					}
				}
			}
		}
	}
}

func TestWarpPut8GoSIMDNoAllocs(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	ref := warpCompoundRef(rng, 64, 64, 0)
	matrix := [6]int32{0, 0, 1<<16 + 300, 200, -100, 1<<16 - 300}
	a, b, g, d := warpReduce(300), warpReduce(200), warpReduce(-100), warpReduce(-300)
	dst := frame.Plane{Pix: make([]byte, 16*16), Stride: 16, Width: 16, Height: 16}
	for _, pos := range [][2]int{{24, 24}, {0, 0}, {56, 56}, {-20, 30}} {
		n := testing.AllocsPerRun(50, func() {
			_ = PredictWarpedPlaneBlockToScratchBitDepth(dst, ref, 1, 8, pos[0], pos[1], 16, 16, matrix, a, b, g, d, false, false)
		})
		if n != 0 {
			t.Fatalf("pos %v: %v allocs per call", pos, n)
		}
	}
}

func BenchmarkWarpPut8(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	ref := warpCompoundRef(rng, 256, 256, 0)
	matrix := [6]int32{64 << 16, 64 << 16, 1<<16 + 900, -700, 500, 1<<16 - 400}
	dst := frame.Plane{Pix: make([]byte, 16*16), Stride: 16, Width: 16, Height: 16}
	for _, g := range []int{500, 0} {
		alpha, beta, gamma, delta := warpReduce(900), warpReduce(-700), warpReduce(g), warpReduce(-400)
		b.Run(fmt.Sprintf("int16/gamma=%d", g), func(b *testing.B) {
			for b.Loop() {
				_ = PredictWarpedPlaneBlockToScratchBitDepth(dst, ref, 1, 8, 64, 64, 16, 16, matrix, alpha, beta, gamma, delta, false, false)
			}
		})
	}
}
