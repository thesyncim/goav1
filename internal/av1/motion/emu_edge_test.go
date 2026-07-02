// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import (
	"math/rand"
	"testing"
)

// TestEmuEdge8MatchesClampedReference proves emuEdge8 materializes exactly
// the samples a per-pixel clamped load would produce (dav1d src/mc_tmpl.c
// emu_edge_c semantics) for windows overhanging any combination of plane
// edges, including windows entirely outside the plane.
func TestEmuEdge8MatchesClampedReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xed6e))
	for trial := 0; trial < 2000; trial++ {
		iw := 1 + rng.Intn(64)
		ih := 1 + rng.Intn(64)
		stride := iw + rng.Intn(8)
		ref, _ := testPlane(iw, ih, 1, stride)
		for i := range ref.Pix {
			ref.Pix[i] = byte(rng.Intn(256))
		}
		bw := 1 + rng.Intn(emuEdgeStride)
		bh := 1 + rng.Intn(emuEdgeRows)
		// Window origin anywhere from fully above/left to fully below/right.
		x := rng.Intn(iw+2*bw+8) - bw - 4
		y := rng.Intn(ih+2*bh+8) - bh - 4

		var edge [emuEdgeStride * emuEdgeRows]byte
		emuEdge8(bw, bh, ref, x, y, edge[:], emuEdgeStride)

		for yy := 0; yy < bh; yy++ {
			sy := clampInt(y+yy, 0, ih-1)
			for xx := 0; xx < bw; xx++ {
				sx := clampInt(x+xx, 0, iw-1)
				want := ref.Pix[sy*ref.Stride+sx]
				got := edge[yy*emuEdgeStride+xx]
				if got != want {
					t.Fatalf("trial=%d plane=%dx%d window=%dx%d@(%d,%d) sample=(%d,%d): emu=%d clamped=%d",
						trial, iw, ih, bw, bh, x, y, xx, yy, got, want)
				}
			}
		}
	}
}
