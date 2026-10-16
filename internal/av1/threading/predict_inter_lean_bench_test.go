// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func benchInterFrame(b *testing.B, format frame.Format) *frame.Frame {
	b.Helper()
	layout, err := frame.RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	out, err := frame.Bind(make([]byte, layout.Size), format)
	if err != nil {
		b.Fatal(err)
	}
	return &out
}

// BenchmarkFrameWorkPredictBlockInterWarm measures one whole inter block (all
// three 4:2:0 planes) through the decoder's predictBlockInterPtr route with
// the job geometry cache installed, as the residual loop runs it: small blocks
// are dominated by the per-plane dispatch, not the kernels.
func BenchmarkFrameWorkPredictBlockInterWarm(b *testing.B) {
	format := frame.Format{Width: 128, Height: 128, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	for _, tc := range []struct {
		name     string
		size     tile.BlockSize
		w4       uint8
		compound bool
		mv       motion.Vector
	}{
		{"8x8_hv", tile.BlockSize8x8, 2, false, motion.Vector{Col: 5, Row: -3}},
		{"8x8_copy", tile.BlockSize8x8, 2, false, motion.Vector{Col: 16, Row: -8}},
		{"16x16_hv", tile.BlockSize16x16, 4, false, motion.Vector{Col: 5, Row: -3}},
		{"8x8_compound", tile.BlockSize8x8, 2, true, motion.Vector{Col: 5, Row: -3}},
		{"16x16_compound", tile.BlockSize16x16, 4, true, motion.Vector{Col: 5, Row: -3}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			output := benchInterFrame(b, format)
			last := benchInterFrame(b, format)
			bwd := benchInterFrame(b, format)
			for i := range last.Y.Pix {
				last.Y.Pix[i] = byte(i * 7)
				bwd.Y.Pix[i] = byte(i * 5)
			}
			ctx := testCompoundInterPredictionBatch(output, last, bwd)
			var geom frameWorkJobGeometryCache
			ctx.geomCache = &geom
			visit := testInterPredictionVisit(tc.mv)
			visit.Block = tile.BlockVisit{
				MICol: 8, MIRow: 8, MIColEnd: 8 + uint16(tc.w4), MIRowEnd: 8 + uint16(tc.w4),
				X4: 8, Y4: 8, Size: tc.size, VisibleW4: tc.w4, VisibleH4: tc.w4,
				HaveTop: true, HaveLeft: true,
			}
			if tc.compound {
				refs := tile.InterReferencesResult{Compound: true, Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameBWD}}
				visit.Prediction.InterReferences = refs
				visit.Prediction.InterMotion.References = refs
				visit.Prediction.InterMotion.MV = [2]motion.Vector{tc.mv, {Col: -7, Row: 9}}
				visit.Prediction.CompoundBlend = tile.CompoundBlendResult{Type: tile.CompoundTypeAverage}
			}
			scratch := new(FrameWorkInterPredictionScratch)
			if err := ctx.predictBlockInterPtr(0, &visit, scratch); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			for b.Loop() {
				if err := ctx.predictBlockInterPtr(0, &visit, scratch); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
