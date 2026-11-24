// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func put8UVTestPlane(rng *rand.Rand, w, h, stride int) frame.Plane {
	pix := make([]byte, stride*h)
	for i := range pix {
		pix[i] = byte(rng.Intn(256))
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

// TestPut8UVMatchesPerPlanePut8 checks the paired chroma put against one Put8
// per plane: whenever Put8UV reports success both planes hold exactly the
// per-plane samples, and on planes of one geometry it succeeds exactly when
// both per-plane puts do. The cases cover every kernel shape (copy, h, v, hv,
// widths 2 to 64, the eight-tap sharp filter), sources inside the plane and
// clamped at each edge (the two planes then share the edge buffer in turn),
// destinations at and past the plane edge, and reference planes whose
// geometries differ.
func TestPut8UVMatchesPerPlanePut8(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const pw, ph = 96, 80
	type geom struct{ w, h, stride int }
	for _, g := range []struct {
		name   string
		u, v   geom
		dv, du geom
	}{
		{"same", geom{pw, ph, 112}, geom{pw, ph, 112}, geom{pw, ph, 128}, geom{pw, ph, 128}},
		{"refstride", geom{pw, ph, 112}, geom{pw, ph, 120}, geom{pw, ph, 128}, geom{pw, ph, 128}},
		{"dstsize", geom{pw, ph, 112}, geom{pw, ph, 112}, geom{pw, ph - 8, 128}, geom{pw, ph, 128}},
	} {
		refU := put8UVTestPlane(rng, g.u.w, g.u.h, g.u.stride)
		refV := put8UVTestPlane(rng, g.v.w, g.v.h, g.v.stride)
		for _, sz := range [][2]int{{2, 4}, {2, 16}, {4, 4}, {4, 16}, {8, 8}, {8, 32}, {16, 16}, {32, 8}, {64, 64}} {
			w, h := sz[0], sz[1]
			for _, f := range []InterpFilters{RegularFilters, {X: InterpEightTapSmooth, Y: InterpBilinear}, {X: InterpMultiTapSharp, Y: InterpEightTapRegular}} {
				for _, sub := range [][2]int{{0, 0}, {5, 0}, {0, 11}, {5, 11}, {8, 8}} {
					for _, pos := range [][4]int{{8, 8, 20, 12}, {0, 0, -5, -3}, {pw - w, ph - h, pw - w + 3, ph - h + 2}, {pw - w + 4, 0, 10, 10}, {16, 8, -40, ph + 10}} {
						dx, dy, rx, ry := pos[0], pos[1], pos[2], pos[3]
						name := fmt.Sprintf("%s/%dx%d/f%d%d/s%d,%d/d%d,%d/r%d,%d", g.name, w, h, f.X, f.Y, sub[0], sub[1], dx, dy, rx, ry)
						var s1, s2 ConvolveScratch
						wantU := put8UVTestPlane(rng, g.du.w, g.du.h, g.du.stride)
						wantV := put8UVTestPlane(rng, g.dv.w, g.dv.h, g.dv.stride)
						gotU := frame.Plane{Pix: bytes.Clone(wantU.Pix), Stride: wantU.Stride, Width: wantU.Width, Height: wantU.Height}
						gotV := frame.Plane{Pix: bytes.Clone(wantV.Pix), Stride: wantV.Stride, Width: wantV.Width, Height: wantV.Height}
						okU := Put8(&wantU, &refU, dx, dy, rx, ry, w, h, w, h, sub[0], sub[1], f, &s1)
						okV := Put8(&wantV, &refV, dx, dy, rx, ry, w, h, w, h, sub[0], sub[1], f, &s1)
						ok := Put8UV(&gotU, &gotV, &refU, &refV, dx, dy, rx, ry, w, h, w, h, sub[0], sub[1], f, &s2)
						if g.name == "same" && ok != (okU && okV) {
							t.Fatalf("%s: Put8UV ok=%v, per-plane %v/%v", name, ok, okU, okV)
						}
						if !ok {
							continue
						}
						if !okU || !okV {
							t.Fatalf("%s: Put8UV wrote a block a per-plane put declines (%v/%v)", name, okU, okV)
						}
						if !bytes.Equal(gotU.Pix, wantU.Pix) || !bytes.Equal(gotV.Pix, wantV.Pix) {
							t.Fatalf("%s: samples differ from the per-plane puts", name)
						}
					}
				}
			}
		}
	}
}
