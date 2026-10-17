// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"math/rand"
	"testing"
)

// benchmarkFilterSB8 filters every 4-pixel column (vertical) or row
// (horizontal) of a 128x128 luma region with dense random masks, the shape of
// one mask region in the frame apply.
func benchmarkFilterSB8(b *testing.B, vertical bool) {
	rng := rand.New(rand.NewSource(9))
	const w, h = 160, 160
	pix := sbTestPlane(rng, w, h)
	lvlStride := w / 4
	lvl := sbTestLevels(rng, lvlStride*(h/4))
	lut := NewSBLUT(0)
	var masks [32][3]uint32
	for i := range masks {
		masks[i] = sbTestMasks(rng, true, 32)
		masks[i][2] &= 0x11111111 // wd16 only on 16-aligned units
	}
	work := append([]byte(nil), pix...)
	for b.Loop() {
		copy(work, pix)
		for i := 1; i < 32; i++ {
			if vertical {
				x := 4 + 4*i
				FilterSBVertical8(work, 8*w+x, w, w-x, masks[i], true, lvl, 2*lvlStride+x/4, lvlStride, 0, &lut)
			} else {
				y := 8 + 4*i
				FilterSBHorizontal8(work, y*w+8, w, w-8, masks[i], true, lvl, (y/4)*lvlStride+2, lvlStride, 1, &lut)
			}
		}
	}
}

func BenchmarkFilterSBVertical8(b *testing.B)   { benchmarkFilterSB8(b, true) }
func BenchmarkFilterSBHorizontal8(b *testing.B) { benchmarkFilterSB8(b, false) }
