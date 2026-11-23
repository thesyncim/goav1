package goav1

import (
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalloopfilter "github.com/thesyncim/goav1/internal/av1/loopfilter"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicDecoderPostFilterTypesAreNameable(t *testing.T) {
	var buffers internalthreading.FrameWorkRestorationFrameBuffers
	ctx := internaldecoder.FrameWorkPostFilterContext{
		RestorationFrameBuffers: &buffers,
	}
	if ctx.RestorationFrameBuffers != &buffers {
		t.Fatal("restoration frame buffers did not round-trip through public context")
	}
	supportedStages, err := ctx.SupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	unsupportedStages, err := ctx.UnsupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	if !supportedStages.Empty() || !unsupportedStages.Empty() {
		t.Fatalf("supported=%b unsupported=%b", supportedStages, unsupportedStages)
	}
	if err := ctx.RequireSupportedPostFilters(); err != nil {
		t.Fatalf("RequireSupportedPostFilters err=%v", err)
	}

	edge := internaldecoder.FrameWorkLoopFilterPostFilterEdge{
		Plane:     internalloopfilter.PlaneY,
		Edge:      internalloopfilter.EdgeVertical,
		Transform: internaltile.TransformSize4x4,
	}
	if edge.Plane != internalloopfilter.PlaneY || edge.Edge != internalloopfilter.EdgeVertical {
		t.Fatalf("edge=%+v", edge)
	}

	records := [3][]internaltile.RestorationUnitRecord{
		{
			{
				Rect: internaltile.RestorationUnitRect{X1: 4, Y1: 4},
				Unit: internaltile.RestorationUnit{
					Type:   internalparser.RestorationWiener,
					Wiener: internalrestoration.WienerInfo{},
				},
			},
		},
	}
	boundaries := [3]internaltile.RestorationStripeBoundaries{}
	scratch := internaltile.RestorationUnitRecordBoundaryScratch{
		Unit: internaltile.RestorationUnitScratch{},
	}
	restorationReq := internaldecoder.FrameWorkRestorationPostFilterRequest{
		Records:    records,
		Boundaries: boundaries,
		Scratch:    scratch,
	}
	req := internaldecoder.FrameWorkPostFilterRequest{
		LoopFilter: internaldecoder.FrameWorkLoopFilterPostFilterRequest{
			Edges: []internaldecoder.FrameWorkLoopFilterPostFilterEdge{edge},
		},
		Restoration: restorationReq,
	}
	if req.Restoration.Records[0][0].Unit.Type != internalparser.RestorationWiener {
		t.Fatalf("restoration request=%+v", req.Restoration)
	}

	var loopResult internaldecoder.FrameWorkLoopFilterPostFilterApplyResult
	var restorationResult internaltile.RestorationFrameApplyResult
	result := internaldecoder.FrameWorkCallerPostFilterResult{
		FrameWorkPostFilterResult: internaldecoder.FrameWorkPostFilterResult{
			LoopFilter:  loopResult,
			Restoration: restorationResult,
		},
	}
	if !result.Completed.Empty() {
		t.Fatalf("result=%+v", result)
	}
	var callerRunner internaldecoder.FrameWorkCallerPostFilterRunner
	if callerRunner.Result.Completed != result.Completed {
		t.Fatalf("caller runner=%+v result=%+v", callerRunner, result)
	}
	if callerScratch, err := callerRunner.ScratchLen(ctx); err != nil || callerScratch.SuperRes.OutputFrame != 0 {
		t.Fatalf("caller runner scratch=%+v err=%v", callerScratch, err)
	}
	var supportedRunner internaldecoder.FrameWorkSupportedPostFilterRunner
	if supportedScratch, err := supportedRunner.ScratchLen(ctx); err != nil || supportedScratch.LoopFilter.Edges != 0 {
		t.Fatalf("supported runner scratch=%+v err=%v", supportedScratch, err)
	}

	var postScratch internaldecoder.FrameWorkPostFilterScratchSize
	postScratch.Restoration = internaldecoder.FrameWorkRestorationPostFilterScratchSize{
		Samples: internaltile.RestorationFrameSampleScratchSize{},
		Apply:   internaltile.RestorationUnitRecordBoundaryScratchSize{},
	}
	postScratch.LoopFilter.Edges = 2
	postScratch.SuperRes.OutputFrame = 16
	combinedScratch := postScratch.Max(internaldecoder.FrameWorkPostFilterScratchSize{
		LoopFilter: internaldecoder.FrameWorkLoopFilterPostFilterScratchSize{Edges: 4},
		SuperRes:   internaldecoder.FrameWorkSuperResPostFilterScratchSize{OutputFrame: 8},
	})
	if combinedScratch.LoopFilter.Edges != 4 || combinedScratch.SuperRes.OutputFrame != 16 {
		t.Fatalf("combined scratch=%+v", combinedScratch)
	}
	_ = postScratch
}
