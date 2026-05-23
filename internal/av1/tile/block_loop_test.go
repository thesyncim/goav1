package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestDecodeBlockLoopReadsPrefixDeltaAndSegments(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 8), Job{Offset: 0, Size: 8}, DecodeOptions{BaseQIdx: 64}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	var scratch BlockLoopScratch
	currentSeg := make([]uint8, 16*16)
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   16,
			MIRowEnd:   16,
		},
		SkipMode: parser.SkipModeParams{Allowed: true, Enabled: true},
		CDEF:     parser.CDEFParams{Bits: 2, StrengthCount: 4},
		Delta: parser.DeltaParams{
			DeltaQPresent: true,
			DeltaQResLog2: 1,
		},
		SBSizeMIB:         16,
		CurrentSegmentMap: currentSeg,
		SegmentMapStride:  16,
	}

	var visits []BlockLoopVisit
	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Delta:     deltaCDFs,
	}, &scratch, req, func(visit BlockLoopVisit) error {
		visits = append(visits, visit)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockLoopStats{PartitionReads: 1, Blocks: 1, Prefixes: 1, DeltaReads: 1}) {
		t.Fatalf("stats=%+v", stats)
	}
	if len(visits) != 1 {
		t.Fatalf("visit count=%d", len(visits))
	}
	visit := visits[0]
	if visit.Block.Size != BlockSize64x64 || visit.Block.VisibleW4 != 16 || visit.Block.VisibleH4 != 16 {
		t.Fatalf("block=%+v", visit.Block)
	}
	if visit.SegmentID != 0 || visit.Segment.RefFrame != -1 || visit.SegmentPredicted {
		t.Fatalf("segment id=%d segment=%+v predicted=%v", visit.SegmentID, visit.Segment, visit.SegmentPredicted)
	}
	if visit.Prefix.SkipMode || visit.Prefix.SkipTransform || visit.Prefix.CDEFIndex != 0 {
		t.Fatalf("prefix=%+v", visit.Prefix)
	}
	if visit.Delta.MICol != 0 || visit.Delta.MIRow != 0 || !visit.Delta.FullSuperblock || visit.Delta.SBSizeMIB != 16 {
		t.Fatalf("delta=%+v", visit.Delta)
	}
	if state.CurrentBaseQIdx != 64 {
		t.Fatalf("base q=%d want 64", state.CurrentBaseQIdx)
	}
}

func TestDecodeBlockLoopSegmentationPreskipAndMaps(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 8), Job{Offset: 0, Size: 8}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, _ := mustBlockLoopCDFs(t)
	seg := parser.SegmentationParams{
		Enabled:   true,
		UpdateMap: true,
		Data: parser.SegmentationData{
			LastActiveID: 2,
			Preskip:      true,
		},
	}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i].RefFrame = -1
	}
	seg.Data.Segments[0].Skip = true

	currentSeg := make([]uint8, 16*16)
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   16,
			MIRowEnd:   16,
		},
		Segmentation:      seg,
		SBSizeMIB:         16,
		CurrentSegmentMap: currentSeg,
		SegmentMapStride:  16,
	}

	var scratch BlockLoopScratch
	var got BlockLoopVisit
	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, &scratch, req, func(visit BlockLoopVisit) error {
		got = visit
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockLoopStats{PartitionReads: 1, Blocks: 1, SegmentIDs: 1, Prefixes: 1}) {
		t.Fatalf("stats=%+v", stats)
	}
	if got.SegmentID != 0 || !got.Segment.Skip || !got.Prefix.SkipTransform {
		t.Fatalf("visit=%+v", got)
	}
	for i, id := range currentSeg {
		if id != 0 {
			t.Fatalf("segment map[%d]=%d want 0", i, id)
		}
	}
	if scratch.Mode.AboveSegmentPred[0] != 0 || scratch.Mode.LeftSkip[0] != 1 {
		t.Fatalf("mode context segpred=%d leftskip=%d", scratch.Mode.AboveSegmentPred[0], scratch.Mode.LeftSkip[0])
	}
}

func TestDecodeBlockLoopReadsIntraPredictionModes(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 8), Job{Offset: 0, Size: 8}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	var intraCDFs IntraModeCDFs
	if err := intraCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   16,
			MIRowEnd:   16,
		},
		SBSizeMIB:             16,
		FrameType:             parser.FrameTypeKey,
		DecodePredictionModes: true,
	}

	var scratch BlockLoopScratch
	var got BlockLoopVisit
	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Intra:     &intraCDFs,
		Delta:     deltaCDFs,
	}, &scratch, req, func(visit BlockLoopVisit) error {
		got = visit
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockLoopStats{PartitionReads: 1, Blocks: 1, Prefixes: 1, PredictionModes: 1, IntraModes: 1}) {
		t.Fatalf("stats=%+v", stats)
	}
	if !got.Prediction.Valid || !got.Prediction.Intra || got.Prediction.LumaMode != IntraModeDC {
		t.Fatalf("prediction=%+v", got.Prediction)
	}
	if scratch.Mode.AboveIntra[0] != 1 || scratch.Mode.LeftIntra[0] != 1 ||
		scratch.Mode.AboveMode[0] != IntraModeDC || scratch.Mode.LeftMode[0] != IntraModeDC {
		t.Fatalf("mode context aboveIntra=%d leftIntra=%d aboveMode=%d leftMode=%d",
			scratch.Mode.AboveIntra[0], scratch.Mode.LeftIntra[0], scratch.Mode.AboveMode[0], scratch.Mode.LeftMode[0])
	}
	if got := intraCDFs.KeyframeYMode[0][0].Values()[int(intraModeCount)]; got != 1 {
		t.Fatalf("keyframe ymode count=%d want 1", got)
	}
}

func TestDecodeBlockLoopReadsInterReferences(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}, Job{Offset: 0, Size: 8}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	var intraCDFs IntraModeCDFs
	if err := intraCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var interCDFs InterRefCDFs
	if err := interCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   16,
			MIRowEnd:   16,
		},
		SBSizeMIB:             16,
		FrameType:             parser.FrameTypeInter,
		ReferenceMode:         parser.ReferenceModeSingle,
		DecodePredictionModes: true,
	}

	var scratch BlockLoopScratch
	var visits []BlockLoopVisit
	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Intra:     &intraCDFs,
		InterRef:  &interCDFs,
		Delta:     deltaCDFs,
	}, &scratch, req, func(visit BlockLoopVisit) error {
		visits = append(visits, visit)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Blocks == 0 ||
		stats.PredictionModes != stats.Blocks ||
		stats.InterEntries != stats.Blocks ||
		stats.InterReferences != stats.Blocks ||
		stats.IntraModes != 0 {
		t.Fatalf("stats=%+v", stats)
	}
	for i, visit := range visits {
		if !visit.Prediction.Valid || visit.Prediction.Intra || !visit.Prediction.InterReferencesValid {
			t.Fatalf("visit %d prediction=%+v", i, visit.Prediction)
		}
		refs := visit.Prediction.InterReferences
		if refs.Compound || !refs.Ref[0].Valid() || refs.Ref[1] != ReferenceFrameNone {
			t.Fatalf("visit %d refs=%+v", i, refs)
		}
	}
	refs := visits[0].Prediction.InterReferences
	if scratch.Mode.AboveIntra[0] != 0 || scratch.Mode.LeftIntra[0] != 0 ||
		scratch.Mode.AboveRef[0][0] != refs.Ref[0] || scratch.Mode.LeftRef[0][0] != refs.Ref[0] {
		t.Fatalf("mode context intra=(%d,%d) refs=(%d,%d) want ref %d",
			scratch.Mode.AboveIntra[0], scratch.Mode.LeftIntra[0],
			scratch.Mode.AboveRef[0][0], scratch.Mode.LeftRef[0][0], refs.Ref[0])
	}
}

func TestDecodeBlockLoopReadsForcedInterModeAndRefMVStack(t *testing.T) {
	var state DecodeState
	payload := make([]byte, 8)
	if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	seg := parser.SegmentationParams{
		Enabled: true,
		Data: parser.SegmentationData{
			LastActiveID: 0,
		},
	}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i].RefFrame = -1
	}
	seg.Data.Segments[0].GlobalMV = true
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   16,
			MIRowEnd:   16,
		},
		Segmentation:          seg,
		SBSizeMIB:             16,
		FrameType:             parser.FrameTypeInter,
		DecodePredictionModes: true,
		DecodeInterModes:      true,
		DecodeMotionVectors:   true,
		GlobalMVs: [referenceFrameCount]motion.Vector{
			ReferenceFrameLast: {Row: 4, Col: -6},
		},
	}

	var scratch BlockLoopScratch
	var got BlockLoopVisit
	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Delta:     deltaCDFs,
	}, &scratch, req, func(visit BlockLoopVisit) error {
		got = visit
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockLoopStats{
		PartitionReads:  1,
		Blocks:          1,
		Prefixes:        1,
		PredictionModes: 1,
		InterEntries:    1,
		InterReferences: 1,
		InterModes:      1,
		RefMVStacks:     1,
		MotionVectors:   1,
	}) {
		t.Fatalf("stats=%+v", stats)
	}
	if !got.Prediction.InterModeValid || got.Prediction.InterMode.Mode != InterModeGlobalMV {
		t.Fatalf("inter mode=%+v valid=%v", got.Prediction.InterMode, got.Prediction.InterModeValid)
	}
	if !got.Prediction.ReferenceMVStackValid || !got.Prediction.ReferenceMVStack.Stack.SingleRefValid {
		t.Fatalf("ref mv stack=%+v valid=%v", got.Prediction.ReferenceMVStack, got.Prediction.ReferenceMVStackValid)
	}
	nearest, near, err := got.Prediction.ReferenceMVStack.Stack.BestSingleReferenceMVs(false, false)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != (motion.Vector{Row: 4, Col: -6}) || near != (motion.Vector{Row: 4, Col: -6}) {
		t.Fatalf("global fallback nearest=%+v near=%+v", nearest, near)
	}
	if got.Prediction.DRLIndexValid || got.Prediction.InterMVReferencesValid {
		t.Fatalf("unexpected drl/mv refs prediction=%+v", got.Prediction)
	}
	if !got.Prediction.InterMotionValid || got.Prediction.InterMotion.MV[0] != (motion.Vector{Row: 4, Col: -6}) {
		t.Fatalf("inter motion=%+v valid=%v", got.Prediction.InterMotion, got.Prediction.InterMotionValid)
	}
	if scratch.Mode.AboveRef[0][0] != ReferenceFrameLast || scratch.Mode.LeftRef[0][0] != ReferenceFrameLast ||
		scratch.Mode.AboveMotionValid[0] != 1 || scratch.Mode.LeftMotionValid[0] != 1 {
		t.Fatalf("marked refs above=%d left=%d motion=(%d,%d)",
			scratch.Mode.AboveRef[0][0], scratch.Mode.LeftRef[0][0],
			scratch.Mode.AboveMotionValid[0], scratch.Mode.LeftMotionValid[0])
	}
}

func TestDecodeBlockPredictionModeReadsInterModeDRLAndStack(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00, 0x00, 0x00}, Job{Offset: 0, Size: 4}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var interModeCDFs InterModeCDFs
	if err := interModeCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	aboveMotion := InterMotionResult{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: 3, Col: 5}},
	}
	leftMotion := InterMotionResult{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: -7, Col: 9}},
	}
	var ctx BlockModeContext
	seedAboveMotion(&ctx, 0, BlockSize8x8, aboveMotion)
	seedLeftMotion(&ctx, 0, BlockSize8x8, leftMotion)

	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{InterMode: &interModeCDFs}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeInter,
		Segmentation:          parser.SegmentationParams{Enabled: true},
		DecodePredictionModes: true,
		DecodeInterModes:      true,
	}, BlockVisit{
		Size:     BlockSize16x16,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	}, BlockModeResult{}, parser.SegmentData{RefFrame: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Valid || result.Intra || !result.InterReferencesValid || result.InterReferences != refs {
		t.Fatalf("prediction refs=%+v", result)
	}
	if !result.InterModeValid || result.InterMode.Mode != InterModeNewMV {
		t.Fatalf("inter mode=%+v valid=%v", result.InterMode, result.InterModeValid)
	}
	if !result.ReferenceMVStackValid || result.ReferenceMVStack.Stack.Count < 2 {
		t.Fatalf("ref mv stack=%+v valid=%v", result.ReferenceMVStack, result.ReferenceMVStackValid)
	}
	if !result.DRLIndexValid || result.DRLIndex != 0 {
		t.Fatalf("drl index=%d valid=%v", result.DRLIndex, result.DRLIndexValid)
	}
	if !result.InterMVReferencesValid || result.InterMVReferences.Residual[0] != (motion.Vector{Row: 3, Col: 5}) {
		t.Fatalf("mv refs=%+v valid=%v", result.InterMVReferences, result.InterMVReferencesValid)
	}
	if ctx.AboveMotionValid[0] != 0 || ctx.LeftMotionValid[0] != 0 {
		t.Fatalf("motion context should wait for residuals above=%d left=%d", ctx.AboveMotionValid[0], ctx.LeftMotionValid[0])
	}
	if got := interModeCDFs.NewMV[4].Values()[2]; got != 1 {
		t.Fatalf("newmv cdf count=%d want 1", got)
	}
	if got := interModeCDFs.DRL[0].Values()[2]; got != 1 {
		t.Fatalf("drl cdf count=%d want 1", got)
	}
}

func TestDecodeBlockPredictionModeInterModeRequiresCDFs(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	var ctx BlockModeContext
	seedAboveMotion(&ctx, 0, BlockSize8x8, InterMotionResult{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: 3, Col: 5}},
	})

	_, err := state.decodeBlockPredictionMode(BlockLoopCDFs{}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeInter,
		Segmentation:          parser.SegmentationParams{Enabled: true},
		DecodePredictionModes: true,
		DecodeInterModes:      true,
	}, BlockVisit{
		Size:    BlockSize16x16,
		X4:      0,
		Y4:      0,
		HaveTop: true,
	}, BlockModeResult{}, parser.SegmentData{RefFrame: 1})
	if !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestDecodeBlockPredictionModeReadsMotionVectorResidual(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00, 0x00, 0x00}, Job{Offset: 0, Size: 4}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var interModeCDFs InterModeCDFs
	if err := interModeCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var mvCDFs MVCDFs
	if err := mvCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	var ctx BlockModeContext
	seedAboveMotion(&ctx, 0, BlockSize8x8, InterMotionResult{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: 3, Col: 5}},
	})
	seedLeftMotion(&ctx, 0, BlockSize8x8, InterMotionResult{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: -7, Col: 9}},
	})

	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{InterMode: &interModeCDFs, MV: &mvCDFs}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeInter,
		Segmentation:          parser.SegmentationParams{Enabled: true},
		DecodePredictionModes: true,
		DecodeInterModes:      true,
		DecodeMotionVectors:   true,
	}, BlockVisit{
		Size:     BlockSize16x16,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	}, BlockModeResult{}, parser.SegmentData{RefFrame: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !result.InterMotionValid || result.InterMotion.MV[0] != (motion.Vector{Row: 3, Col: 5}) {
		t.Fatalf("inter motion=%+v valid=%v", result.InterMotion, result.InterMotionValid)
	}
	if !result.MVResidualValid[0] || result.MVResiduals[0].Joint != MVJointZero {
		t.Fatalf("mv residuals=%+v valid=%v", result.MVResiduals, result.MVResidualValid)
	}
	if ctx.AboveMotionValid[0] != 1 || ctx.LeftMotionValid[0] != 1 {
		t.Fatalf("motion context above=%d left=%d", ctx.AboveMotionValid[0], ctx.LeftMotionValid[0])
	}
}

func TestDecodeBlockPredictionModeReadsInterIntraAndMotionMode(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00, 0x00, 0x00, 0x00}, Job{Offset: 0, Size: 5}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var interModeCDFs InterModeCDFs
	if err := interModeCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var mvCDFs MVCDFs
	if err := mvCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var blendCDFs CompoundBlendCDFs
	if err := blendCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var motionCDFs MotionModeCDFs
	if err := motionCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	var ctx BlockModeContext
	seedAboveMotion(&ctx, 0, BlockSize8x8, InterMotionResult{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNewMV},
		MV:         [2]motion.Vector{{Row: 3, Col: 5}},
	})
	seedLeftMotion(&ctx, 0, BlockSize8x8, InterMotionResult{
		References: refs,
		Mode:       InterModeResult{Mode: InterModeNearMV},
		MV:         [2]motion.Vector{{Row: -7, Col: 9}},
	})

	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{
		InterMode: &interModeCDFs,
		MV:        &mvCDFs,
		Blend:     &blendCDFs,
		Motion:    &motionCDFs,
	}, &ctx, BlockLoopRequest{
		FrameType:                parser.FrameTypeInter,
		Segmentation:             parser.SegmentationParams{Enabled: true},
		DecodePredictionModes:    true,
		DecodeInterModes:         true,
		DecodeMotionVectors:      true,
		DecodeInterIntra:         true,
		DecodeMotionModes:        true,
		EnableInterIntraCompound: true,
		SwitchableMotionMode:     true,
		OverlappableNeighbors:    1,
		EnableMaskedCompound:     true,
		EnableDistWtdCompound:    true,
		AllowHighPrecisionMV:     true,
		AllowWarpedMotion:        false,
		EnableOrderHint:          true,
		OrderHintBits:            4,
		CurrentOrderHint:         8,
		ReferenceOrderHints:      [referenceFrameCount]uint32{ReferenceFrameLast: 4},
		GlobalMotionTypes:        [referenceFrameCount]parser.GlobalMotionType{ReferenceFrameLast: parser.GlobalMotionTranslation},
		ScaledReferences:         [referenceFrameCount]bool{},
		DecodeCompoundBlend:      true,
		GlobalMVs:                [referenceFrameCount]motion.Vector{},
		RefSignBias:              [referenceFrameCount]bool{},
		ForceIntegerMV:           false,
		AllowIntrabc:             false,
		ReferenceMode:            parser.ReferenceModeSingle,
		SkipModeRefs:             [2]ReferenceFrame{},
		NumProjRef:               0,
	}, BlockVisit{
		Size:     BlockSize16x16,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	}, BlockModeResult{}, parser.SegmentData{RefFrame: 1})
	if err != nil {
		t.Fatal(err)
	}
	if !result.InterIntraValid || result.InterIntra.Enabled {
		t.Fatalf("inter-intra=%+v valid=%v", result.InterIntra, result.InterIntraValid)
	}
	if !result.MotionModeValid || result.MotionMode != MotionModeTranslation {
		t.Fatalf("motion mode=%d valid=%v", result.MotionMode, result.MotionModeValid)
	}
	if result.CompoundBlendValid {
		t.Fatalf("single-ref compound blend valid: %+v", result.CompoundBlend)
	}
	if got := blendCDFs.InterIntra[yModeSizeContext[BlockSize16x16]].Values()[2]; got != 1 {
		t.Fatalf("inter-intra cdf count=%d want 1", got)
	}
	if got := motionCDFs.OBMC[BlockSize16x16].Values()[2]; got != 1 {
		t.Fatalf("obmc cdf count=%d want 1", got)
	}
}

func TestDecodeBlockPredictionModeReadsCompoundBlend(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var blendCDFs CompoundBlendCDFs
	if err := blendCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext
	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{Blend: &blendCDFs}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeInter,
		DecodePredictionModes: true,
		DecodeInterModes:      true,
		DecodeMotionVectors:   true,
		DecodeCompoundBlend:   true,
		SkipModeRefs:          [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameAltref},
		EnableDistWtdCompound: true,
		EnableOrderHint:       true,
		OrderHintBits:         4,
		CurrentOrderHint:      8,
		ReferenceOrderHints: [referenceFrameCount]uint32{
			ReferenceFrameLast:   4,
			ReferenceFrameAltref: 12,
		},
	}, BlockVisit{
		Size: BlockSize16x16,
		X4:   0,
		Y4:   0,
	}, BlockModeResult{SkipMode: true}, parser.SegmentData{RefFrame: -1})
	if err != nil {
		t.Fatal(err)
	}
	if !result.CompoundBlendValid || result.CompoundBlend.Type != CompoundTypeAverage || result.CompoundBlend.CompoundIndex != 1 {
		t.Fatalf("compound blend=%+v valid=%v", result.CompoundBlend, result.CompoundBlendValid)
	}
	if !result.InterMotionValid || !result.InterReferences.Compound {
		t.Fatalf("motion=%+v refs=%+v", result.InterMotion, result.InterReferences)
	}
	if ctx.AboveCompIndex[0] != 1 || ctx.LeftCompIndex[0] != 1 {
		t.Fatalf("compound index context above=%d left=%d want 1", ctx.AboveCompIndex[0], ctx.LeftCompIndex[0])
	}
}

func TestDecodeBlockLoopPredictionModesRequireIntraCDFs(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 8), Job{Offset: 0, Size: 8}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, _ := mustBlockLoopCDFs(t)
	_, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
	}, &BlockLoopScratch{}, BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   16,
			MIRowEnd:   16,
		},
		SBSizeMIB:             16,
		FrameType:             parser.FrameTypeKey,
		DecodePredictionModes: true,
	}, func(BlockLoopVisit) error { return nil })
	if !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestDecodeBlockLoopCopiesPreviousSegmentationWhenMapNotUpdated(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 8), Job{Offset: 0, Size: 8}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, _ := mustBlockLoopCDFs(t)
	seg := parser.SegmentationParams{
		Enabled: true,
		Data: parser.SegmentationData{
			LastActiveID: 3,
		},
	}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i].RefFrame = -1
	}
	seg.Data.Segments[2].GlobalMV = true
	prevSeg := make([]uint8, 16*16)
	for i := range prevSeg {
		prevSeg[i] = 2
	}
	curSeg := make([]uint8, 16*16)

	var got BlockLoopVisit
	_, err := state.DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, &BlockLoopScratch{}, BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   16,
			MIRowEnd:   16,
		},
		Segmentation:       seg,
		SBSizeMIB:          16,
		CurrentSegmentMap:  curSeg,
		PreviousSegmentMap: prevSeg,
		SegmentMapStride:   16,
	}, func(visit BlockLoopVisit) error {
		got = visit
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.SegmentID != 2 || !got.Segment.GlobalMV {
		t.Fatalf("visit segment id=%d segment=%+v", got.SegmentID, got.Segment)
	}
	for i, id := range curSeg {
		if id != 2 {
			t.Fatalf("curSeg[%d]=%d want 2", i, id)
		}
	}
}

func TestDecodeBlockLoopRejectsInvalidInputs(t *testing.T) {
	var state DecodeState
	partitionCDFs, modeCDFs, _ := mustBlockLoopCDFs(t)
	if _, err := (*DecodeState)(nil).DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, &BlockLoopScratch{}, BlockLoopRequest{}, func(BlockLoopVisit) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := state.Reset(make([]byte, 4), Job{Offset: 0, Size: 4}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	validReq := BlockLoopRequest{
		Walk:      BlockWalkRequest{Root: BlockLevel64x64, MIColEnd: 16, MIRowEnd: 16},
		SBSizeMIB: 16,
	}
	if _, err := state.DecodeBlockLoop(BlockLoopCDFs{Mode: &modeCDFs}, &BlockLoopScratch{}, validReq, func(BlockLoopVisit) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("missing partition cdfs err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, nil, validReq, func(BlockLoopVisit) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil scratch err=%v want %v", err, ErrInvalidDecodeState)
	}
	errBoom := errors.New("boom")
	if _, err := state.DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, &BlockLoopScratch{}, validReq, func(BlockLoopVisit) error { return errBoom }); !errors.Is(err, errBoom) {
		t.Fatalf("visitor err=%v want %v", err, errBoom)
	}
}

func FuzzDecodeBlockLoop(f *testing.F) {
	f.Add([]byte{0x00, 0x00, 0x00, 0x00}, uint8(BlockLevel64x64), uint8(16), uint8(16), false)
	f.Add([]byte{0xff, 0xff, 0xff, 0xff}, uint8(BlockLevel64x64), uint8(16), uint8(16), true)
	f.Add([]byte{0xa5, 0x5a, 0x00, 0xff}, uint8(BlockLevel32x32), uint8(8), uint8(8), false)

	f.Fuzz(func(t *testing.T, payload []byte, rawRoot uint8, rawCols uint8, rawRows uint8, delta bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		root := BlockLevel(rawRoot % uint8(blockLevelCount))
		if root < BlockLevel64x64 {
			root = BlockLevel64x64
		}
		rootSize := root.Size4x4()
		cols := uint32((rawCols % rootSize) + 1)
		rows := uint32((rawRows % rootSize) + 1)
		cols = (cols + 1) &^ 1
		rows = (rows + 1) &^ 1
		if cols == 0 || rows == 0 {
			return
		}

		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{BaseQIdx: 64}); err != nil {
			t.Fatal(err)
		}
		partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
		var intraCDFs IntraModeCDFs
		if err := intraCDFs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var interCDFs InterRefCDFs
		if err := interCDFs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var interModeCDFs InterModeCDFs
		if err := interModeCDFs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var mvCDFs MVCDFs
		if err := mvCDFs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var motionCDFs MotionModeCDFs
		if err := motionCDFs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var blendCDFs CompoundBlendCDFs
		if err := blendCDFs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		orderBits := uint8(rawRows%8 + 1)
		orderLimit := uint32(1) << orderBits
		req := BlockLoopRequest{
			Walk: BlockWalkRequest{
				Root:       root,
				MIColStart: 0,
				MIRowStart: 0,
				MIColEnd:   cols,
				MIRowEnd:   rows,
			},
			SBSizeMIB:             root.Size4x4(),
			FrameType:             parser.FrameType(rawRoot & 1),
			AllowIntrabc:          rawRows&1 != 0,
			ReferenceMode:         parser.ReferenceMode(rawCols % 3),
			DecodePredictionModes: rawRoot&0x80 != 0,
			DecodeInterModes:      rawRoot&0xc0 == 0xc0,
			DecodeMotionVectors:   rawRoot&0xe0 == 0xe0,
			DecodeInterIntra:      rawRoot&0xf0 == 0xf0,
			DecodeMotionModes:     rawRoot&0xf0 == 0xf0,
			DecodeCompoundBlend:   rawRoot&0xf0 == 0xf0,
			GlobalMVs: [referenceFrameCount]motion.Vector{
				ReferenceFrameLast:    {Row: int32(int8(rawRows)), Col: int32(int8(rawCols))},
				ReferenceFrameGolden:  {Row: -int32(int8(rawRows)), Col: int32(int8(rawCols))},
				ReferenceFrameAltref2: {Row: int32(int8(rawRoot)), Col: -int32(int8(rawCols))},
			},
			GlobalMotionTypes: [referenceFrameCount]parser.GlobalMotionType{
				ReferenceFrameLast:   parser.GlobalMotionType(rawRoot % 4),
				ReferenceFrameAltref: parser.GlobalMotionType(rawRows % 4),
			},
			RefSignBias: [referenceFrameCount]bool{
				ReferenceFrameGolden: rawRoot&1 != 0,
				ReferenceFrameAltref: rawRows&1 != 0,
			},
			ReferenceOrderHints: [referenceFrameCount]uint32{
				ReferenceFrameLast:   uint32(rawCols) % orderLimit,
				ReferenceFrameGolden: uint32(rawRows) % orderLimit,
				ReferenceFrameAltref: uint32(rawRoot) % orderLimit,
			},
			ScaledReferences: [referenceFrameCount]bool{
				ReferenceFrameLast2: rawCols&1 != 0,
				ReferenceFrameBWD:   rawRows&1 != 0,
			},
			EnableInterIntraCompound: rawCols&0x80 != 0,
			SwitchableMotionMode:     rawRows&0x80 != 0,
			AllowWarpedMotion:        rawRoot&0x20 != 0,
			OverlappableNeighbors:    int(rawCols & 1),
			NumProjRef:               int(rawRows & 1),
			EnableMaskedCompound:     rawRoot&0x10 != 0,
			EnableDistWtdCompound:    rawCols&0x40 != 0,
			EnableOrderHint:          true,
			OrderHintBits:            orderBits,
			CurrentOrderHint:         uint32(rawRoot) % orderLimit,
		}
		if delta {
			req.Delta = parser.DeltaParams{DeltaQPresent: true}
		}
		stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
			Partition: &partitionCDFs,
			Mode:      &modeCDFs,
			Intra:     &intraCDFs,
			InterRef:  &interCDFs,
			InterMode: &interModeCDFs,
			MV:        &mvCDFs,
			Motion:    &motionCDFs,
			Blend:     &blendCDFs,
			Delta:     deltaCDFs,
		}, &BlockLoopScratch{}, req, func(visit BlockLoopVisit) error {
			if visit.Block.MIColEnd > cols || visit.Block.MIRowEnd > rows {
				t.Fatalf("visit beyond region=%+v cols=%d rows=%d", visit.Block, cols, rows)
			}
			return nil
		})
		if err != nil {
			return
		}
		if stats.Blocks == 0 || stats.PartitionReads == 0 || stats.Prefixes != stats.Blocks {
			t.Fatalf("bad stats=%+v", stats)
		}
	})
}

func mustBlockLoopCDFs(t *testing.T) (PartitionCDFs, BlockModeCDFs, DeltaCDFs) {
	t.Helper()
	var partitionCDFs PartitionCDFs
	if err := partitionCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var modeCDFs BlockModeCDFs
	if err := modeCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var q entropy.CDF
	if err := q.InitDefaultDelta(); err != nil {
		t.Fatal(err)
	}
	var lf entropy.CDF
	if err := lf.InitDefaultDelta(); err != nil {
		t.Fatal(err)
	}
	var multi [FrameLoopFilterCount]entropy.CDF
	deltaCDFs := DeltaCDFs{Q: &q, LF: &lf}
	for i := 0; i < FrameLoopFilterCount; i++ {
		if err := multi[i].InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		deltaCDFs.LFMulti[i] = &multi[i]
	}
	return partitionCDFs, modeCDFs, deltaCDFs
}
