// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkPaletteFillMatchesPerSampleStore(t *testing.T) {
	for _, bps := range []int{1, 2} {
		for _, size := range []uint8{2, 5, 8} {
			for _, dims := range [][2]int{{4, 4}, {8, 4}, {16, 16}, {12, 20}, {64, 64}} {
				w, h := dims[0], dims[1]
				const pw, ph = 80, 72
				mk := func() frame.Plane {
					p := frame.Plane{Pix: make([]byte, pw*ph*bps), Stride: pw * bps, Width: pw, Height: ph}
					for i := range p.Pix {
						p.Pix[i] = byte(i * 13)
					}
					return p
				}
				var colors [tile.PaletteMaxSize]uint16
				for i := range colors {
					colors[i] = uint16(17*i+3) << (8 * (bps - 1))
				}
				mapStride := w + 8
				idx := make([]uint8, mapStride*(h+4))
				for i := range idx {
					idx[i] = uint8(i*7) % size
				}
				got, want := mk(), mk()
				if err := frameWorkPaletteFill(got, bps, 3, 5, w, h, idx, mapStride, 4, 2, &colors, size); err != nil {
					t.Fatalf("bps%d size%d %dx%d: %v", bps, size, w, h, err)
				}
				for r := range h {
					for c := range w {
						frameWorkStoreSample(want, bps, 3+c, 5+r, colors[idx[(2+r)*mapStride+4+c]])
					}
				}
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("bps%d size%d %dx%d: mismatch", bps, size, w, h)
				}
				idx[(2+h-1)*mapStride+4+w-1] = size
				if err := frameWorkPaletteFill(mk(), bps, 3, 5, w, h, idx, mapStride, 4, 2, &colors, size); err == nil {
					t.Fatalf("bps%d size%d %dx%d: out-of-palette index accepted", bps, size, w, h)
				}
			}
		}
	}
}
