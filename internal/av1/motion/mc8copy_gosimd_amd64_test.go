// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"bytes"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestAMD64CopyFastMatchesClampedCopy checks the 8- and 16-bit put copies
// against the clamped reference copy over every AV1 block width (and a few
// widths that are not powers of two), in and out of the plane: they must
// decline exactly the blocks whose source leaves the plane and write exactly
// the block otherwise.
func TestAMD64CopyFastMatchesClampedCopy(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	const rw, rh = 200, 180
	for _, bps := range []int{1, 2} {
		rs := bps * (rw + 24)
		ref := frame.Plane{Pix: make([]byte, rs*rh), Stride: rs, Width: rw, Height: rh}
		for i := range ref.Pix {
			ref.Pix[i] = byte(rng.Intn(256))
		}
		for _, sz := range [][2]int{{4, 4}, {4, 1}, {4, 16}, {8, 8}, {12, 4}, {16, 16}, {24, 8}, {32, 32}, {48, 4}, {64, 64}, {96, 2}, {128, 128}} {
			w, h := sz[0], sz[1]
			for iter := range 40 {
				refX := rng.Intn(rw+2*w) - w
				refY := rng.Intn(rh+2*h) - h
				dw, dh := w+16, h+4
				want := frame.Plane{Pix: make([]byte, bps*(dw+3)*dh), Stride: bps * (dw + 3), Width: dw, Height: dh}
				for i := range want.Pix {
					want.Pix[i] = byte(i * 3)
				}
				got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: want.Stride, Width: dw, Height: dh}
				if err := copyPlaneBlockClamped(want, ref, bps, 7, 3, refX, refY, w, h); err != nil {
					t.Fatal(err)
				}
				inside := refX >= 0 && refY >= 0 && refX+w <= rw && refY+h <= rh
				var ok bool
				if bps == 1 {
					ok = mc8CopyFast(&got, &ref, 7, 3, refX, refY, w, h)
				} else {
					ok = mc16CopyFast(&got, &ref, 7, 3, refX, refY, w, h)
				}
				if ok != inside {
					t.Fatalf("bps %d %dx%d iter %d ref=%d,%d: handled=%v want %v", bps, w, h, iter, refX, refY, ok, inside)
				}
				if inside && !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("bps %d %dx%d iter %d ref=%d,%d: mismatch", bps, w, h, iter, refX, refY)
				}
			}
		}
	}
}
