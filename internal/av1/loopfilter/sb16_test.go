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

// sbTestPlane16 builds a little-endian uint16 plane at the given bit depth
// from 8x8 tiles that are either flat (one background shared by every flat
// tile, noise below the scaled flatness threshold), lightly textured, or
// random, so every filter width including the 13-tap flat path fires across
// both edge directions.
func sbTestPlane16(rng *rand.Rand, w, h int, bitDepth uint8) []byte {
	shift := uint(bitDepth - 8)
	maxPix := 1<<bitDepth - 1
	flatF := 1 << shift
	bg := rng.Intn(maxPix + 1 - flatF)
	tw, th := (w+7)/8, (h+7)/8
	modes := make([]int, tw*th)
	for i := range modes {
		modes[i] = rng.Intn(5)
	}
	pix := make([]byte, 2*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var s int
			switch modes[(y/8)*tw+x/8] {
			case 0, 1, 2:
				s = bg + rng.Intn(flatF)
			case 3:
				s = bg + rng.Intn(12*flatF) - 6*flatF
			default:
				s = rng.Intn(maxPix + 1)
			}
			if rng.Intn(256) == 0 {
				s = maxPix
			}
			s = min(max(s, 0), maxPix)
			pix[2*(y*w+x)] = byte(s)
			pix[2*(y*w+x)+1] = byte(s >> 8)
		}
	}
	return pix
}

// TestFilterSB16MatchesScalar checks the dispatched high-bit-depth superblock
// filters against the unit-by-unit dav1d C reference.
func TestFilterSB16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const w, h = 160, 160
	const stride = 2 * w
	for iter := 0; iter < 4000; iter++ {
		bitDepth := uint8(10)
		if rng.Intn(2) == 0 {
			bitDepth = 12
		}
		luma := rng.Intn(2) == 0
		lut := NewSBLUT(uint8(rng.Intn(MaxSharpness + 1)))
		pix := sbTestPlane16(rng, w, h, bitDepth)
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
			x := 4 * (1 + rng.Intn(w/4-1))
			if vm[2] != 0 {
				x = 8 * (1 + rng.Intn(w/8-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*stride + 2*x
			lvlOff := (y0/4)*lvlStride + x/4
			ws = filterSBVertical16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs = FilterSBVertical16(got, off, stride, 2*(w-x), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (1 + rng.Intn(h/4-1))
			if vm[2] != 0 {
				y = 8 * (1 + rng.Intn(h/8-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*stride + 2*x0
			lvlOff := (y/4)*lvlStride + x0/4
			ws = filterSBHorizontal16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			gs = FilterSBHorizontal16(got, off, stride, 2*(w-x0), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		if ws != gs {
			t.Fatalf("iter %d: stats %+v, want %+v", iter, gs, ws)
		}
		if !bytes.Equal(got, want) {
			for i := 0; i < len(got); i += 2 {
				if got[i] != want[i] || got[i+1] != want[i+1] {
					px := i / 2
					t.Fatalf("iter %d vertical=%v luma=%v bd=%d vm=%x: first diff at (%d,%d): got %d want %d", iter, vertical, luma, bitDepth, vm, px%w, px/w, ld16(got, i), ld16(want, i))
				}
			}
		}
	}
}

// TestFilterSB16ScalarMatchesEdgeKernels checks the high-bit-depth C
// reference against the package's per-edge kernels for every width.
func TestFilterSB16ScalarMatchesEdgeKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const w, h = 64, 64
	const stride = 2 * w
	for iter := 0; iter < 3000; iter++ {
		bitDepth := uint8(10)
		if rng.Intn(2) == 0 {
			bitDepth = 12
		}
		sharp := uint8(rng.Intn(MaxSharpness + 1))
		lut := NewSBLUT(sharp)
		level := uint8(1 + rng.Intn(MaxLevel))
		th, err := ThresholdsForLevel(level, sharp)
		if err != nil {
			t.Fatal(err)
		}
		widths := []int{4, 6, 8, 16}
		wd := widths[rng.Intn(len(widths))]
		pix := sbTestPlane16(rng, w, h, bitDepth)
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		edge := EdgeVertical
		x, y := 16+4*rng.Intn(8), 16+4*rng.Intn(8)
		if rng.Intn(2) == 0 {
			edge = EdgeHorizontal
		}
		plane := frame.Plane{Pix: want, Stride: stride, Width: w, Height: h}
		kw := uint8(wd)
		if wd == 16 {
			kw = 14
		}
		if err := FilterEdgeByWidth(kw, plane, 2, bitDepth, edge, int32(x), int32(y), 4, th); err != nil {
			t.Fatal(err)
		}
		off := y*stride + 2*x
		if edge == EdgeVertical {
			lpfUnit16(got, off, stride, 2, bitDepth, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		} else {
			lpfUnit16(got, off, 2, stride, bitDepth, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("iter %d: width %d edge %v level %d sharp %d bd %d mismatch", iter, wd, edge, level, sharp, bitDepth)
		}
	}
}
