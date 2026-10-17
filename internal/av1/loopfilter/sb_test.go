// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// sbTestPlane fills a plane with smooth content plus noise so every filter
// width (including the flat 13-tap path) is exercised.
func sbTestPlane(rng *rand.Rand, w, h int) []byte {
	pix := make([]byte, w*h)
	for y := 0; y < h; y++ {
		base := rng.Intn(256)
		for x := 0; x < w; x++ {
			v := base
			switch rng.Intn(4) {
			case 0:
				v += rng.Intn(3) - 1
			case 1:
				v += rng.Intn(9) - 4
			case 2:
				v = rng.Intn(256)
			}
			if x%16 >= 8 && rng.Intn(3) == 0 {
				v += 20
			}
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			pix[y*w+x] = uint8(v)
		}
	}
	return pix
}

func sbTestMasks(rng *rand.Rand, luma bool, units int) [3]uint32 {
	var vm [3]uint32
	for n := 0; n < units; n++ {
		bit := uint32(1) << uint(n)
		switch rng.Intn(5) {
		case 0:
		case 1:
			vm[0] |= bit
		case 2, 3:
			vm[1] |= bit
		case 4:
			if luma {
				vm[2] |= bit
			} else {
				vm[1] |= bit
			}
		}
	}
	return vm
}

func sbTestLevels(rng *rand.Rand, n int) [][4]uint8 {
	lvl := make([][4]uint8, n)
	for i := range lvl {
		for c := 0; c < 4; c++ {
			if rng.Intn(6) != 0 {
				lvl[i][c] = uint8(rng.Intn(MaxLevel + 1))
			}
		}
	}
	return lvl
}

// TestFilterSB8MatchesScalar checks the dispatched superblock filters against
// the unit-by-unit dav1d C reference on random content, masks and levels.
func TestFilterSB8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const w, h = 160, 160
	for iter := 0; iter < 4000; iter++ {
		luma := rng.Intn(2) == 0
		sharp := uint8(rng.Intn(MaxSharpness + 1))
		lut := NewSBLUT(sharp)
		pix := sbTestPlane(rng, w, h)
		lvlStride := w / 4
		lvl := sbTestLevels(rng, lvlStride*(h/4))
		comp := rng.Intn(4)
		units := 1 + rng.Intn(32)
		vm := sbTestMasks(rng, luma, units)
		vertical := rng.Intn(2) == 0
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		var ws, gs SBStats
		if vertical {
			// Column x in 4..(w-8) so every width's taps fit; rows from y0.
			x := 4 * (2 + rng.Intn(w/4-4))
			if vm[2] != 0 {
				x = 16 * (1 + rng.Intn(w/16-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*w + x
			lvlOff := (y0/4)*lvlStride + x/4
			ws = filterSBVertical8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs = FilterSBVertical8(got, off, w, w-x, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (2 + rng.Intn(h/4-4))
			if vm[2] != 0 {
				y = 16 * (1 + rng.Intn(h/16-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*w + x0
			lvlOff := (y/4)*lvlStride + x0/4
			ws = filterSBHorizontal8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs = FilterSBHorizontal8(got, off, w, w-x0, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		if ws != gs {
			t.Fatalf("iter %d: stats %+v, want %+v", iter, gs, ws)
		}
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("iter %d vertical=%v luma=%v vm=%x: first diff at (%d,%d): got %d want %d", iter, vertical, luma, vm, i%w, i/w, got[i], want[i])
				}
			}
		}
	}
}

// TestFilterSB8ScalarMatchesEdgeKernels checks the C reference against the
// package's existing per-edge kernels, unit by unit, for every width.
func TestFilterSB8ScalarMatchesEdgeKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const w, h = 64, 64
	for iter := 0; iter < 3000; iter++ {
		sharp := uint8(rng.Intn(MaxSharpness + 1))
		lut := NewSBLUT(sharp)
		level := uint8(1 + rng.Intn(MaxLevel))
		th, err := ThresholdsForLevel(level, sharp)
		if err != nil {
			t.Fatal(err)
		}
		widths := []int{4, 6, 8, 16}
		wd := widths[rng.Intn(len(widths))]
		pix := sbTestPlane(rng, w, h)
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		edge := EdgeVertical
		x, y := 16+4*rng.Intn(8), 16+4*rng.Intn(8)
		if rng.Intn(2) == 0 {
			edge = EdgeHorizontal
		}
		plane := frame.Plane{Pix: want, Stride: w, Width: w, Height: h}
		kw := uint8(wd)
		if wd == 16 {
			kw = 14
		}
		if err := FilterEdgeByWidth(kw, plane, 1, 8, edge, int32(x), int32(y), 4, th); err != nil {
			t.Fatal(err)
		}
		if edge == EdgeVertical {
			lpfUnit8(got, y*w+x, w, 1, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		} else {
			lpfUnit8(got, y*w+x, 1, w, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("iter %d: width %d edge %v level %d sharp %d mismatch", iter, wd, edge, level, sharp)
		}
	}
}
