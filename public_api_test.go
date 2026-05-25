package goav1_test

import (
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDecoderPostFilterTypesAreNameable(t *testing.T) {
	var buffers av1.DecoderFrameWorkRestorationFrameBuffers
	ctx := av1.DecoderFrameWorkPostFilterContext{
		RestorationFrameBuffers: &buffers,
	}
	if ctx.RestorationFrameBuffers != &buffers {
		t.Fatal("restoration frame buffers did not round-trip through public context")
	}

	edge := av1.DecoderFrameWorkLoopFilterPostFilterEdge{
		Plane:     av1.LoopFilterPlaneY,
		Edge:      av1.LoopFilterEdgeVertical,
		Transform: av1.TileTransformSize4x4,
	}
	if edge.Plane != av1.LoopFilterPlaneY || edge.Edge != av1.LoopFilterEdgeVertical {
		t.Fatalf("edge=%+v", edge)
	}

	records := [3][]av1.TileRestorationUnitRecord{
		{
			{
				Rect: av1.TileRestorationUnitRect{X1: 4, Y1: 4},
				Unit: av1.TileRestorationUnit{
					Type:   av1.RestorationWiener,
					Wiener: av1.RestorationWienerInfo{},
				},
			},
		},
	}
	boundaries := [3]av1.TileRestorationStripeBoundaries{}
	scratch := av1.TileRestorationUnitRecordBoundaryScratch{
		Unit: av1.TileRestorationUnitScratch{},
	}
	restorationReq := av1.DecoderFrameWorkRestorationPostFilterRequest{
		Records:    records,
		Boundaries: boundaries,
		Scratch:    scratch,
	}
	req := av1.DecoderFrameWorkPostFilterRequest{
		LoopFilter: av1.DecoderFrameWorkLoopFilterPostFilterRequest{
			Edges: []av1.DecoderFrameWorkLoopFilterPostFilterEdge{edge},
		},
		Restoration: restorationReq,
	}
	if req.Restoration.Records[0][0].Unit.Type != av1.RestorationWiener {
		t.Fatalf("restoration request=%+v", req.Restoration)
	}

	var loopResult av1.DecoderFrameWorkLoopFilterPostFilterApplyResult
	var restorationResult av1.TileRestorationFrameApplyResult
	result := av1.DecoderFrameWorkCallerPostFilterResult{
		FrameWorkPostFilterResult: av1.DecoderFrameWorkPostFilterResult{
			LoopFilter:  loopResult,
			Restoration: restorationResult,
		},
	}
	if !result.Completed.Empty() {
		t.Fatalf("result=%+v", result)
	}
	var callerRunner av1.DecoderFrameWorkCallerPostFilterRunner
	if callerRunner.Result.Completed != result.Completed {
		t.Fatalf("caller runner=%+v result=%+v", callerRunner, result)
	}
	if callerScratch, err := callerRunner.ScratchLen(ctx); err != nil || callerScratch.SuperRes.OutputFrame != 0 {
		t.Fatalf("caller runner scratch=%+v err=%v", callerScratch, err)
	}
	var supportedRunner av1.DecoderFrameWorkSupportedPostFilterRunner
	if supportedScratch, err := supportedRunner.ScratchLen(ctx); err != nil || supportedScratch.LoopFilter.Edges != 0 {
		t.Fatalf("supported runner scratch=%+v err=%v", supportedScratch, err)
	}

	var postScratch av1.DecoderFrameWorkPostFilterScratchSize
	postScratch.Restoration = av1.DecoderFrameWorkRestorationPostFilterScratchSize{
		Samples: av1.TileRestorationFrameSampleScratchSize{},
		Apply:   av1.TileRestorationUnitRecordBoundaryScratchSize{},
	}
	_ = postScratch
}
