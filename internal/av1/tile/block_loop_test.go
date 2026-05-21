package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
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
		}
		if delta {
			req.Delta = parser.DeltaParams{DeltaQPresent: true}
		}
		stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
			Partition: &partitionCDFs,
			Mode:      &modeCDFs,
			Intra:     &intraCDFs,
			InterRef:  &interCDFs,
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
