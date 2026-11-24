// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestPredictIntra8MatchesPlaneEntry checks the decoder's 8-bit entry against
// PredictIntraPlaneBlock for every mode, block shape and DC edge availability,
// with the edges read from the same plane.
func TestPredictIntra8MatchesPlaneEntry(t *testing.T) {
	sizes := []int{4, 8, 16, 32, 64}
	modes := []IntraMode{IntraModeDC, IntraModeVertical, IntraModeHorizontal, IntraModePaeth, IntraModeSmooth, IntraModeSmoothVertical, IntraModeSmoothHorizontal}
	const x, y, stride, rows = 3, 2, 160, 72
	seed := uint32(1)
	for _, w := range sizes {
		for _, h := range sizes {
			if w > 4*h || h > 4*w {
				continue
			}
			for _, mode := range modes {
				for avail := 0; avail < 4; avail++ {
					haveTop, haveLeft := avail&1 != 0, avail&2 != 0
					if mode != IntraModeDC && (!haveTop || !haveLeft) {
						continue
					}
					for pattern := 0; pattern < 4; pattern++ {
						pix := make([]byte, stride*rows)
						for i := range pix {
							switch pattern {
							case 0:
								seed = seed*1664525 + 1013904223
								pix[i] = byte(seed >> 24)
							case 1:
								pix[i] = 0xff
							case 2:
								pix[i] = byte(i & 1 * 0xff)
							default:
								pix[i] = byte(i * 37)
							}
						}
						want := append([]byte(nil), pix...)
						plane := frame.Plane{Pix: want, Stride: stride, Width: stride, Height: rows}
						var edges IntraEdges
						above := make([]uint16, w)
						left := make([]uint16, h)
						off := y*stride + x
						for i := range above {
							above[i] = uint16(want[off-stride+i])
						}
						for i := range left {
							left[i] = uint16(want[off-1+i*stride])
						}
						if haveTop {
							edges.Above, edges.AboveAvailable = above, true
						}
						if haveLeft {
							edges.Left, edges.LeftAvailable = left, true
						}
						edges.AboveLeft, edges.AboveLeftAvailable = uint16(want[off-stride-1]), true
						if err := PredictIntraPlaneBlock(plane, 1, 8, x, y, w, h, mode, edges); err != nil {
							t.Fatal(err)
						}
						var top, leftIn []byte
						if haveTop {
							top = pix[off-stride:]
						}
						if haveLeft {
							leftIn = make([]byte, h)
							for i := range leftIn {
								leftIn[i] = pix[off-1+i*stride]
							}
						}
						scratch := make([]uint16, w+h)
						end := off + (h-1)*stride + w
						if err := PredictIntra8(pix[off:end], stride, w, h, mode, top, leftIn, pix[off-stride-1], scratch); err != nil {
							t.Fatal(err)
						}
						if !bytes.Equal(pix, want) {
							t.Fatalf("%s", fmt.Sprintf("mode %d %dx%d avail %d pattern %d: output differs", mode, w, h, avail, pattern))
						}
					}
				}
			}
		}
	}
}

func TestPredictIntra8RejectsInvalidShapes(t *testing.T) {
	pix := make([]byte, 64*64)
	edge := make([]byte, 64)
	wide := make([]uint16, 128)
	for _, c := range []struct{ w, h, stride int }{{0, 8, 64}, {8, 0, 64}, {12, 8, 64}, {128, 8, 128}, {8, 8, 4}, {64, 64, 63}} {
		if err := PredictIntra8(pix, c.stride, c.w, c.h, IntraModeSmooth, pix, edge, 0, wide); err == nil {
			t.Fatalf("%dx%d stride %d accepted", c.w, c.h, c.stride)
		}
	}
	if err := PredictIntra8(pix, 64, 8, 8, IntraModeVertical, nil, edge, 0, wide); err == nil {
		t.Fatal("vertical without top accepted")
	}
	if err := PredictIntra8(pix, 64, 8, 8, IntraModeDC, pix[:4], nil, 0, wide); err == nil {
		t.Fatal("DC with short top accepted")
	}
	if err := PredictIntra8(pix, 64, 8, 8, IntraModeSmooth, pix, edge, 0, wide[:15]); err == nil {
		t.Fatal("smooth with short scratch accepted")
	}
}

// BenchmarkPredictIntra8 times the decoder's 8-bit entry per mode and square
// size, the per-call figure compared against dav1d's ipred_*_8bpc_neon.
func BenchmarkPredictIntra8(b *testing.B) {
	pix := make([]byte, 160*72)
	for i := range pix {
		pix[i] = byte(i*37 + 11)
	}
	left := make([]byte, 64)
	for i := range left {
		left[i] = byte(i*53 + 5)
	}
	scratch := make([]uint16, 128)
	off := 2*160 + 3
	names := []string{"dc", "v", "h", "paeth", "smooth", "smooth_v", "smooth_h"}
	for mode, name := range names {
		for _, n := range []int{4, 8, 16, 32, 64} {
			b.Run(fmt.Sprintf("%s/%dx%d", name, n, n), func(b *testing.B) {
				for b.Loop() {
					_ = PredictIntra8(pix[off:off+(n-1)*160+n], 160, n, n, IntraMode(mode), pix[off-160:], left[:n], 9, scratch)
				}
			})
		}
	}
}

func TestPredictIntra8Allocs(t *testing.T) {
	pix := make([]byte, 160*72)
	left := make([]byte, 64)
	scratch := make([]uint16, 128)
	off := 2*160 + 3
	allocs := testing.AllocsPerRun(50, func() {
		for _, mode := range []IntraMode{IntraModeDC, IntraModeVertical, IntraModeHorizontal, IntraModePaeth, IntraModeSmooth, IntraModeSmoothVertical, IntraModeSmoothHorizontal} {
			_ = PredictIntra8(pix[off:off+15*160+16], 160, 16, 16, mode, pix[off-160:], left[:16], 0, scratch)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs = %v", allocs)
	}
}
