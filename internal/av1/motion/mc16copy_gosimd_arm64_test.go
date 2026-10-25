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

func TestMC16CopyFastMatchesClampedCopy(t *testing.T) {
	rng := rand.New(rand.NewSource(27))
	const rw, rh, rs = 200, 180, 224
	ref := frame.Plane{Pix: make([]byte, 2*rs*rh), Stride: 2 * rs, Width: rw, Height: rh}
	for i := range ref.Pix {
		ref.Pix[i] = byte(rng.Intn(256))
	}
	for _, sz := range [][2]int{{4, 4}, {4, 16}, {8, 8}, {12, 4}, {16, 16}, {24, 8}, {32, 32}, {64, 64}, {128, 128}, {4, 1}} {
		w, h := sz[0], sz[1]
		for iter := range 50 {
			refX := rng.Intn(rw+2*w) - w
			refY := rng.Intn(rh+2*h) - h
			dw, dh := w+16, h+4
			want := frame.Plane{Pix: make([]byte, 2*(dw+3)*dh), Stride: 2 * (dw + 3), Width: dw, Height: dh}
			for i := range want.Pix {
				want.Pix[i] = byte(i * 3)
			}
			got := frame.Plane{Pix: slices.Clone(want.Pix), Stride: want.Stride, Width: dw, Height: dh}
			if err := copyPlaneBlockClamped(want, ref, 2, 7, 3, refX, refY, w, h); err != nil {
				t.Fatal(err)
			}
			inside := refX >= 0 && refY >= 0 && refX+w <= rw && refY+h <= rh
			if mc16CopyFast(&got, &ref, 7, 3, refX, refY, w, h) != inside {
				t.Fatalf("%dx%d iter %d ref=%d,%d: handled=%v want %v", w, h, iter, refX, refY, !inside, inside)
			}
			if inside && !bytes.Equal(got.Pix, want.Pix) {
				t.Fatalf("%dx%d iter %d ref=%d,%d: mismatch", w, h, iter, refX, refY)
			}
		}
	}
}
