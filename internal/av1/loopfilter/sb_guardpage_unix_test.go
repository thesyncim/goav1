// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package loopfilter

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
)

// TestFilterSBStaysInsideGuardedPlane runs the superblock filters on planes
// whose pixels and level caches end at (or start right after) an inaccessible
// page, with the edges placed as close to the plane borders as each filter
// width allows, and checks them against the unit-by-unit C reference.
func TestFilterSBStaysInsideGuardedPlane(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for iter := range 3000 {
		hbd := iter%2 == 1
		bps := 1
		if hbd {
			bps = 2
		}
		bd := uint8(8)
		if hbd {
			bd = uint8(10 + 2*rng.Intn(2))
		}
		luma := rng.Intn(2) == 0
		vertical := rng.Intn(2) == 0
		before := rng.Intn(2) == 0
		units := 1 + rng.Intn(32)
		vm := sbTestMasks(rng, luma, units)
		if vm[0]|vm[1]|vm[2] == 0 {
			continue
		}
		reach := 4
		if vm[2] != 0 {
			reach = 8
		}
		lut := NewSBLUT(uint8(rng.Intn(MaxSharpness + 1)))
		comp := rng.Intn(4)
		// The plane is exactly as wide and tall as the edges need: the edge
		// sits reach samples from the near border and the far border is as
		// close past it as the driver's footprint allows.
		var w, h, off, lvlOff, lvlStride, lvlN int
		if vertical {
			w = 2*reach + 4*rng.Intn(5)
			h = 4 * units
			// The SIMD driver reads sixteen samples from p3 (twelve past the
			// edge at widths below 16), the edge kernels only the taps.
			x := reach
			if !before {
				x = max(reach, w-max(reach, 12))
			}
			off, lvlStride = x, w/4+1
			lvlOff = x / 4
			lvlN = lvlOff + (units-1)*lvlStride + 1
		} else {
			w = 4 * units
			h = 2*reach + 4*rng.Intn(5)
			y := reach
			if !before {
				y = h - reach
			}
			off, lvlStride = y*w, w/4
			lvlOff = (y / 4) * lvlStride
			lvlN = lvlOff + units
		}
		stride := w * bps
		n := (h-1)*stride + w*bps
		var pix []byte
		var lvl [][4]uint8
		if before {
			pix = guardpage.Before[byte](t, n)
			lvl = guardpage.Before[[4]uint8](t, lvlN)
		} else {
			pix = guardpage.After[byte](t, n)
			lvl = guardpage.After[[4]uint8](t, lvlN)
		}
		if hbd {
			for i := 0; i < n; i += 2 {
				v := rng.Intn(1 << bd)
				pix[i], pix[i+1] = byte(v), byte(v>>8)
			}
		} else {
			copy(pix, sbTestPlane(rng, n, 1))
		}
		copy(lvl, sbTestLevels(rng, lvlN))
		want := bytes.Clone(pix)
		bOff := off * bps
		rowBytes := stride - (bOff % stride)
		name := fmt.Sprintf("iter %d hbd=%v vertical=%v luma=%v before=%v units=%d w=%d h=%d vm=%x", iter, hbd, vertical, luma, before, units, w, h, vm)
		var ws, gs SBStats
		guardpage.Check(t, name, func() {
			switch {
			case vertical && !hbd:
				ws = filterSBVertical8Scalar(want, bOff, stride, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
				gs = FilterSBVertical8(pix, bOff, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			case !hbd:
				ws = filterSBHorizontal8Scalar(want, bOff, stride, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
				gs = FilterSBHorizontal8(pix, bOff, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			case vertical:
				ws = filterSBVertical16Scalar(want, bOff, stride, bd, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
				gs = FilterSBVertical16(pix, bOff, stride, rowBytes, bd, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			default:
				ws = filterSBHorizontal16Scalar(want, bOff, stride, bd, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
				gs = FilterSBHorizontal16(pix, bOff, stride, rowBytes, bd, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			}
		})
		if ws != gs || !bytes.Equal(pix, want) {
			t.Fatalf("%s: guarded filter differs from the C reference (stats %+v want %+v)", name, gs, ws)
		}
	}
}
