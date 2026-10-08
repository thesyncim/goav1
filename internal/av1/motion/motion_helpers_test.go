// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (amd64 || arm64) && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func itoaW(w int) string {
	switch w {
	case 8:
		return "8"
	case 16:
		return "16"
	case 32:
		return "32"
	default:
		return "64"
	}
}

func randPlaneHBD(rng *rand.Rand, side int, max uint16) frame.Plane {
	p, _ := testPlane(side, side, 2, side*2)
	for i := 0; i+1 < len(p.Pix); i += 2 {
		v := uint16(rng.Intn(int(max) + 1))
		p.Pix[i] = byte(v)
		p.Pix[i+1] = byte(v >> 8)
	}
	return p
}

func diffPlanesHBD(t *testing.T, got, want frame.Plane, w, h int, tag string, bd uint8) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := y*got.Stride + x*2
			g := uint16(got.Pix[o]) | uint16(got.Pix[o+1])<<8
			ow := y*want.Stride + x*2
			e := uint16(want.Pix[ow]) | uint16(want.Pix[ow+1])<<8
			if g != e {
				t.Fatalf("%s bd=%d w=%d h=%d (%d,%d): GoSIMD=%d PureGo=%d", tag, bd, w, h, x, y, g, e)
			}
		}
	}
}

func convolve2DKernelTables() [][16][filterTaps]int16 {
	return [][16][filterTaps]int16{
		subpelFilters8,
		subpelFilters8Smooth,
		subpelFilters8Sharp,
		bilinearFilters,
	}
}

func avx2FilterTables() [][16][filterTaps]int16 {
	return [][16][filterTaps]int16{
		subpelFilters8,
		subpelFilters8Smooth,
		subpelFilters8Sharp,
		bilinearFilters,
	}
}

func randPlane(rng *rand.Rand, side, bps int) frame.Plane {
	p, _ := testPlane(side, side, bps, side*bps)
	for i := range p.Pix {
		p.Pix[i] = byte(rng.Intn(256))
	}
	return p
}
