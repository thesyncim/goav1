package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
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

func TestDecodeBlockLoopBeforeSuperblockRunsBeforePartition(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 8), Job{Offset: 0, Size: 8}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	var scratch BlockLoopScratch
	var roots []BlockLoopSuperblockVisit
	var bitsBeforePartition []int
	initialBits := state.Reader.BitsRead()
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   32,
			MIRowEnd:   16,
		},
		SBSizeMIB: 16,
		BeforeSuperblock: func(visit BlockLoopSuperblockVisit) error {
			roots = append(roots, visit)
			bitsBeforePartition = append(bitsBeforePartition, state.Reader.BitsRead())
			return nil
		},
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
	if stats.Blocks != 2 || stats.PartitionReads != 2 || len(visits) != 2 {
		t.Fatalf("stats=%+v visits=%d", stats, len(visits))
	}
	if len(roots) != 2 ||
		roots[0] != (BlockLoopSuperblockVisit{MICol: 0, MIRow: 0, SBSizeMIB: 16}) ||
		roots[1] != (BlockLoopSuperblockVisit{MICol: 16, MIRow: 0, SBSizeMIB: 16}) {
		t.Fatalf("roots=%+v", roots)
	}
	if len(bitsBeforePartition) != 2 || bitsBeforePartition[0] != initialBits || bitsBeforePartition[1] <= bitsBeforePartition[0] {
		t.Fatalf("bits before partition=%v", bitsBeforePartition)
	}
}

func TestDecodeBlockLoopPreservesNeighborAvailabilityAcrossRoots(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 16), Job{Offset: 0, Size: 16}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   32,
			MIRowEnd:   16,
		},
		SBSizeMIB: 16,
	}

	var scratch BlockLoopScratch
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
	if stats.Blocks != 2 || stats.PartitionReads != 2 {
		t.Fatalf("stats=%+v want two root blocks", stats)
	}
	if len(visits) != 2 {
		t.Fatalf("visits=%d want 2", len(visits))
	}
	if visits[0].Block.HaveLeft || visits[1].Block.HaveTop || !visits[1].Block.HaveLeft {
		t.Fatalf("neighbor flags first=%+v second=%+v", visits[0].Block, visits[1].Block)
	}
	if visits[1].Block.X4 != 0 || visits[1].Block.MICol != 16 {
		t.Fatalf("second root x4/mi=%+v", visits[1].Block)
	}
}

func TestBlockLoopContextCarrierRoundTripsRootEdges(t *testing.T) {
	var scratch BlockLoopScratch
	scratch.Partition.Above[2] = 9
	scratch.Partition.Left[3] = 7
	scratch.Mode.AboveSkip[0] = 1
	scratch.Mode.AboveRef[1][2] = ReferenceFrameBWD
	scratch.Mode.AboveInterMotion[4] = InterMotionResult{MV: [2]motion.Vector{{Row: 3, Col: -5}}}
	scratch.Mode.AboveMotionValid[4] = 1
	scratch.Mode.LeftSkip[5] = 1
	scratch.Mode.LeftRef[0][6] = ReferenceFrameGolden
	scratch.Mode.LeftCompIndex[6] = 1
	scratch.Mode.LeftInterMotion[6] = InterMotionResult{MV: [2]motion.Vector{{Row: -2, Col: 7}}}
	scratch.Mode.LeftMotionValid[6] = 1
	scratch.CoeffCtx.Above[2][8] = 13
	scratch.CoeffCtx.Left[1][9] = 11

	carrier := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, 2)}
	if err := blockLoopStoreRootContext(&scratch, &carrier, 1, 16); err != nil {
		t.Fatal(err)
	}

	scratch = BlockLoopScratch{}
	scratch.CDEF.Read[0] = true
	if err := blockLoopLoadRootContext(&scratch, &carrier, 1, true, true, 16); err != nil {
		t.Fatal(err)
	}
	if scratch.CDEF.Read[0] {
		t.Fatal("CDEF context persisted across root load")
	}
	if scratch.Partition.Above[2] != 9 || scratch.Partition.Left[3] != 7 {
		t.Fatalf("partition context above=%d left=%d", scratch.Partition.Above[2], scratch.Partition.Left[3])
	}
	if scratch.Mode.AboveSkip[0] != 1 ||
		scratch.Mode.AboveRef[1][2] != ReferenceFrameBWD ||
		scratch.Mode.AboveInterMotion[4].MV[0] != (motion.Vector{Row: 3, Col: -5}) ||
		scratch.Mode.AboveMotionValid[4] != 1 ||
		scratch.Mode.LeftSkip[5] != 1 ||
		scratch.Mode.LeftRef[0][6] != ReferenceFrameGolden ||
		scratch.Mode.LeftCompIndex[6] != 1 ||
		scratch.Mode.LeftInterMotion[6].MV[0] != (motion.Vector{Row: -2, Col: 7}) ||
		scratch.Mode.LeftMotionValid[6] != 1 {
		t.Fatalf("mode context did not round-trip: %+v", scratch.Mode)
	}
	if scratch.CoeffCtx.Above[2][8] != 13 || scratch.CoeffCtx.Left[1][9] != 11 {
		t.Fatalf("coeff context above=%d left=%d", scratch.CoeffCtx.Above[2][8], scratch.CoeffCtx.Left[1][9])
	}

	if err := blockLoopLoadRootContext(&scratch, &carrier, 1, false, false, 16); err != nil {
		t.Fatal(err)
	}
	if scratch.Partition.Above[2] != 0 || scratch.Partition.Left[3] != 0 ||
		scratch.Mode.AboveSkip[0] != 0 || scratch.Mode.LeftSkip[5] != 0 ||
		scratch.CoeffCtx.Above[2][8] != 0 || scratch.CoeffCtx.Left[1][9] != 0 {
		t.Fatalf("boundary load kept neighbor contexts: partition=%+v mode=%+v coeff=%+v", scratch.Partition, scratch.Mode, scratch.CoeffCtx)
	}
}

// TestBlockLoopLoadFirstRootIndependentOfCarrierLength locks the boundary
// load (haveTop=false, haveLeft=false) at the first root column to produce
// scratch state independent of (a) the carrier Above slice length and (b)
// any prior content in the carrier's right-neighbor slots. The frame-edge
// (0,0) root SB never reads carrier data via the haveTop/haveLeft gates, so
// the resulting scratch must match the single-SB (len=1, zero carrier) case
// regardless of multi-SB layouts. Regression coverage for the hypothesis
// that multi-SB rootCols allocation could leak past the carrier's
// rootColIndex=0 slot into the first block's tx_size / coefficient context.
func TestBlockLoopLoadFirstRootIndependentOfCarrierLength(t *testing.T) {
	baseline := func() BlockLoopScratch {
		var s BlockLoopScratch
		if err := blockLoopLoadRootContext(&s, nil, 0, false, false, 16); err != nil {
			t.Fatal(err)
		}
		return s
	}

	want := baseline()

	for _, n := range []int{1, 2, 3, 8} {
		n := n
		t.Run("len="+itoaTest(n), func(t *testing.T) {
			carrier := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, n)}
			// Seed every slot with non-zero state. If the load path were to
			// reach past rootColIndex=0 via the haveTop=false gate it would
			// pull this data into scratch and the comparison below would
			// reject it.
			for i := range carrier.Above {
				slot := &carrier.Above[i]
				slot.Partition[0] = 1
				slot.mode.Skip[0] = 1
				slot.mode.Intra[0] = 1
				slot.mode.Mode[0] = IntraModeDC
				slot.mode.BlockSize[0] = BlockSize32x32
				slot.mode.MotionValid[0] = 1
				slot.Coeff[0][0] = 13
				slot.Coeff[1][0] = 17
				slot.Coeff[2][0] = 19
			}
			carrier.Left.Partition[0] = 5
			carrier.Left.mode.Skip[0] = 1
			carrier.Left.mode.Mode[0] = IntraModePaeth
			carrier.Left.mode.BlockSize[0] = BlockSize64x64
			carrier.Left.Coeff[0][0] = 21

			var scratch BlockLoopScratch
			if err := blockLoopLoadRootContext(&scratch, &carrier, 0, false, false, 16); err != nil {
				t.Fatal(err)
			}
			// Partition / Mode / CoeffCtx must all match the single-SB baseline.
			if scratch.Partition != want.Partition {
				t.Errorf("len=%d: Partition differs from baseline", n)
			}
			if scratch.CoeffCtx != want.CoeffCtx {
				t.Errorf("len=%d: CoeffCtx differs from baseline", n)
			}
			if scratch.Mode != want.Mode {
				t.Errorf("len=%d: Mode differs from baseline", n)
			}
			if scratch.CDEF != want.CDEF {
				t.Errorf("len=%d: CDEF differs from baseline", n)
			}
		})
	}
}

func itoaTest(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func TestDecodeBlockLoopContextCarrierStoresRootEdges(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 16), Job{Offset: 0, Size: 16}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	seg := parser.SegmentationParams{Enabled: true}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i].RefFrame = -1
	}
	seg.Data.Segments[0].Skip = true
	carrier := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, 2)}
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel64x64,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   32,
			MIRowEnd:   16,
		},
		ContextCarrier: &carrier,
		Segmentation:   seg,
		SBSizeMIB:      16,
	}

	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Delta:     deltaCDFs,
	}, &BlockLoopScratch{}, req, func(BlockLoopVisit) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if stats.Blocks != 2 || stats.Prefixes != 2 {
		t.Fatalf("stats=%+v want two roots", stats)
	}
	if carrier.Above[0].mode.Skip[0] != 1 || carrier.Above[1].mode.Skip[0] != 1 || carrier.Left.mode.Skip[0] != 1 {
		t.Fatalf("carrier skip contexts above0=%d above1=%d left=%d", carrier.Above[0].mode.Skip[0], carrier.Above[1].mode.Skip[0], carrier.Left.mode.Skip[0])
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
	if !got.Prediction.Valid || !got.Prediction.Intra || got.Prediction.LumaMode != IntraModeDC ||
		!got.Prediction.ChromaModeValid || got.Prediction.ChromaMode != ChromaIntraModeDC {
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
	if got := intraCDFs.UVMode[0][IntraModeDC].Values()[int(chromaIntraModeCount-1)]; got != 1 {
		t.Fatalf("uvmode count=%d want 1", got)
	}
}

func TestDecodeBlockLoopReadsCoefficients(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 64), Job{Offset: 0, Size: 64}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	var intraCDFs IntraModeCDFs
	if err := intraCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel16x16,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   4,
			MIRowEnd:   4,
		},
		SBSizeMIB:             4,
		FrameType:             parser.FrameTypeKey,
		DecodePredictionModes: true,
		DecodeCoefficients:    true,
		Color:                 parser.ColorConfig{MonoChrome: true},
		TransformMode:         parser.TransformModeLargest,
		LumaTransformType:     transform.TypeDCTDCT,
	}

	var scratch BlockLoopScratch
	var got BlockLoopVisit
	coeffVisits := 0
	req.CoeffVisitor = func(parent BlockLoopVisit, block BlockCoeffBlock) error {
		coeffVisits++
		if parent.Block.Size != BlockSize16x16 || parent.Prefix.SkipTransform {
			t.Fatalf("parent visit=%+v", parent)
		}
		if block.Plane != 0 || block.Block.Size != TransformSize16x16 || block.Transform != transform.TypeDCTDCT {
			t.Fatalf("coeff block=%+v", block)
		}
		assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
		return nil
	}
	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Intra:     &intraCDFs,
		Transform: &transformCDFs,
		Coeff:     &coeffCDFs,
		Delta:     deltaCDFs,
	}, &scratch, req, func(visit BlockLoopVisit) error {
		got = visit
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockLoopStats{
		PartitionReads:      1,
		Blocks:              1,
		Prefixes:            1,
		PredictionModes:     1,
		IntraModes:          1,
		CoefficientBlocks:   1,
		CoefficientTXBs:     1,
		CoefficientNonZero:  1,
		CoefficientEOBTotal: 1,
	}) {
		t.Fatalf("stats=%+v", stats)
	}
	if coeffVisits != 1 {
		t.Fatalf("coeff visits=%d want 1", coeffVisits)
	}
	if !got.CoefficientsValid || got.Coefficients.Tree.Y != TransformSize16x16 || got.Coefficients.TotalStats().TXBs != 1 {
		t.Fatalf("coefficients valid=%v result=%+v", got.CoefficientsValid, got.Coefficients)
	}
	if scratch.CoeffCtx.Above[0][0] == 0 || scratch.CoeffCtx.Left[0][0] == 0 {
		t.Fatalf("coefficient context not marked above=%d left=%d", scratch.CoeffCtx.Above[0][0], scratch.CoeffCtx.Left[0][0])
	}
}

func TestDecodeBlockLoopCoefficientVisitorError(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 64), Job{Offset: 0, Size: 64}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	var intraCDFs IntraModeCDFs
	if err := intraCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
	errBoom := errors.New("boom")
	_, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Intra:     &intraCDFs,
		Transform: &transformCDFs,
		Coeff:     &coeffCDFs,
		Delta:     deltaCDFs,
	}, &BlockLoopScratch{}, BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel16x16,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   4,
			MIRowEnd:   4,
		},
		SBSizeMIB:             4,
		FrameType:             parser.FrameTypeKey,
		DecodePredictionModes: true,
		DecodeCoefficients:    true,
		Color:                 parser.ColorConfig{MonoChrome: true},
		TransformMode:         parser.TransformModeLargest,
		LumaTransformType:     transform.TypeDCTDCT,
		CoeffVisitor: func(BlockLoopVisit, BlockCoeffBlock) error {
			return errBoom
		},
	}, func(BlockLoopVisit) error { return nil })
	if !errors.Is(err, errBoom) {
		t.Fatalf("err=%v want %v", err, errBoom)
	}
}

func TestDecodeBlockLoopCoefficientSelectorDoesNotRequirePredictionModes(t *testing.T) {
	var state DecodeState
	if err := state.Reset(make([]byte, 64), Job{Offset: 0, Size: 64}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partitionCDFs, modeCDFs, deltaCDFs := mustBlockLoopCDFs(t)
	transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)

	beforeCoefficients := false
	coeffVisits := 0
	stats, err := state.DecodeBlockLoop(BlockLoopCDFs{
		Partition: &partitionCDFs,
		Mode:      &modeCDFs,
		Transform: &transformCDFs,
		Coeff:     &coeffCDFs,
		Delta:     deltaCDFs,
	}, &BlockLoopScratch{}, BlockLoopRequest{
		Walk: BlockWalkRequest{
			Root:       BlockLevel16x16,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   4,
			MIRowEnd:   4,
		},
		SBSizeMIB:          4,
		DecodeCoefficients: true,
		BeforeCoefficients: func(visit BlockLoopVisit) error {
			if visit.Prediction.Valid {
				t.Fatalf("prediction unexpectedly decoded: %+v", visit.Prediction)
			}
			beforeCoefficients = true
			return nil
		},
		CoeffRequest: func(visit BlockLoopVisit) (BlockCoeffRequest, error) {
			if !beforeCoefficients {
				t.Fatal("coefficient selector ran before before-coefficients hook")
			}
			return BlockCoeffRequest{
				Transform: TransformTreeRequest{
					Size:          visit.Block.Size,
					X4:            visit.Block.X4,
					Y4:            visit.Block.Y4,
					VisibleW4:     visit.Block.VisibleW4,
					VisibleH4:     visit.Block.VisibleH4,
					Color:         parser.ColorConfig{MonoChrome: true},
					TransformMode: parser.TransformModeLargest,
					Inter:         true,
					SkipTransform: visit.Prefix.SkipTransform,
				},
				LumaType: transform.TypeDCTDCT,
			}, nil
		},
		CoeffVisitor: func(parent BlockLoopVisit, block BlockCoeffBlock) error {
			coeffVisits++
			if !beforeCoefficients || parent.Prediction.Valid {
				t.Fatalf("parent before=%v prediction=%+v", beforeCoefficients, parent.Prediction)
			}
			assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
			return nil
		},
	}, func(visit BlockLoopVisit) error {
		if !visit.CoefficientsValid || visit.Prediction.Valid {
			t.Fatalf("visit=%+v", visit)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !beforeCoefficients || coeffVisits != 1 {
		t.Fatalf("before=%v coeffVisits=%d", beforeCoefficients, coeffVisits)
	}
	if stats.PredictionModes != 0 || stats.CoefficientBlocks != 1 || stats.CoefficientTXBs != 1 {
		t.Fatalf("stats=%+v", stats)
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
	currentMVEntries, err := ReferenceMVFrameEntries(16, 16)
	if err != nil {
		t.Fatal(err)
	}
	var currentMVFrame ReferenceMVFrame
	if err := currentMVFrame.Init(16, 16, make([]ReferenceMVEntry, currentMVEntries)); err != nil {
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
		Segmentation:          seg,
		SBSizeMIB:             16,
		FrameType:             parser.FrameTypeInter,
		DecodePredictionModes: true,
		DecodeInterModes:      true,
		DecodeMotionVectors:   true,
		GlobalMVs: [referenceFrameCount]motion.Vector{
			ReferenceFrameLast: {Row: 4, Col: -6},
		},
		CurrentMVFrame: &currentMVFrame,
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
	for i, entry := range currentMVFrame.Entries {
		want := ReferenceMVEntry{
			Ref:   ReferenceFrameLast,
			MV:    motion.Vector{Row: 4, Col: -6},
			Valid: true,
		}
		if entry != want {
			t.Fatalf("current MV entry %d=%+v want %+v", i, entry, want)
		}
	}
	if scratch.Mode.AboveRef[0][0] != ReferenceFrameLast || scratch.Mode.LeftRef[0][0] != ReferenceFrameLast ||
		scratch.Mode.AboveMotionValid[0] != 1 || scratch.Mode.LeftMotionValid[0] != 1 {
		t.Fatalf("marked refs above=%d left=%d motion=(%d,%d)",
			scratch.Mode.AboveRef[0][0], scratch.Mode.LeftRef[0][0],
			scratch.Mode.AboveMotionValid[0], scratch.Mode.LeftMotionValid[0])
	}
}

func TestBlockReferenceGlobalMVsForBlockPortsLibaomCenterMV(t *testing.T) {
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	block := BlockVisit{MICol: 4, MIRow: 8, Size: BlockSize16x16}
	var global [referenceFrameCount]parser.WarpedMotionParams
	global[ReferenceFrameLast] = parser.DefaultWarpedMotionParams()
	global[ReferenceFrameLast].Type = parser.GlobalMotionRotZoom
	global[ReferenceFrameLast].Matrix = [6]int32{
		8192, -4096,
		1<<16 + 2048, 1024,
		-1024, 1<<16 + 2048,
	}

	got := blockReferenceGlobalMVsForBlock(refs, [referenceFrameCount]motion.Vector{}, global, true, false, block)
	if got[0] != (motion.Vector{Row: 6, Col: 12}) {
		t.Fatalf("rotzoom global mv=%+v want row=6 col=12", got[0])
	}

	got = blockReferenceGlobalMVsForBlock(refs, [referenceFrameCount]motion.Vector{}, global, true, true, block)
	if got[0] != (motion.Vector{Row: 8, Col: 8}) {
		t.Fatalf("integer rotzoom global mv=%+v want row=8 col=8", got[0])
	}

	global[ReferenceFrameLast] = parser.DefaultWarpedMotionParams()
	global[ReferenceFrameLast].Type = parser.GlobalMotionTranslation
	global[ReferenceFrameLast].Matrix[0] = 13 << 13
	global[ReferenceFrameLast].Matrix[1] = -7 << 13
	got = blockReferenceGlobalMVsForBlock(refs, [referenceFrameCount]motion.Vector{}, global, false, true, block)
	if got[0] != (motion.Vector{Row: 16, Col: -8}) {
		t.Fatalf("translation global mv=%+v want row=16 col=-8", got[0])
	}

	fallback := [referenceFrameCount]motion.Vector{ReferenceFrameLast: {Row: 3, Col: -5}}
	got = blockReferenceGlobalMVsForBlock(refs, fallback, [referenceFrameCount]parser.WarpedMotionParams{}, true, false, block)
	if got[0] != fallback[ReferenceFrameLast] {
		t.Fatalf("fallback global mv=%+v want %+v", got[0], fallback[ReferenceFrameLast])
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
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: 1}, &PaletteModeScratch{})
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
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: 1}, &PaletteModeScratch{})
	if !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("err=%v want %v", err, entropy.ErrInvalidCDF)
	}
}

func TestDecodeBlockPredictionModeReadsIntrabcWithoutInterSyntax(t *testing.T) {
	var state DecodeState
	if err := state.Reset([]byte{0xff}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var intraCDFs IntraModeCDFs
	if err := intraCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var ctx BlockModeContext

	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{Intra: &intraCDFs}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeKey,
		AllowIntrabc:          true,
		DecodePredictionModes: true,
	}, BlockVisit{
		Size: BlockSize16x16,
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: -1}, &PaletteModeScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Valid || result.Intra || !result.Intrabc || !result.IntrabcValid {
		t.Fatalf("intrabc prediction result=%+v", result)
	}
	if result.InterReferencesValid || result.InterMotionValid {
		t.Fatalf("intrabc should not read inter syntax without motion-vector decode: %+v", result)
	}
	if !result.InterModeValid || result.InterMode.Mode != InterModeNewMV {
		t.Fatalf("intrabc mode=%+v valid=%v want NEWMV", result.InterMode, result.InterModeValid)
	}
}

func TestIntrabcPredictedMVUsesNeighborMotionBeforeFallback(t *testing.T) {
	var ctx BlockModeContext
	neighbor := InterMotionResult{
		Mode: InterModeResult{Mode: InterModeNewMV},
		MV:   [2]motion.Vector{{Row: -512}},
	}
	if err := ctx.MarkIntrabcMotion(BlockSize16x16, 0, 0, neighbor); err != nil {
		t.Fatal(err)
	}
	got, err := intrabcPredictedMV(&ctx, BlockLoopRequest{
		Walk:      BlockWalkRequest{MIColEnd: 32, MIRowEnd: 64},
		SBSizeMIB: 16,
	}, BlockVisit{
		MICol: 4, MIRow: 32,
		X4: 4, Y4: 0, Size: BlockSize16x16, HaveLeft: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != neighbor.MV[0] {
		t.Fatalf("predicted intrabc mv=%+v want %+v", got, neighbor.MV[0])
	}
}

func TestIntrabcPredictedMVUsesOuterGridBeforeFallback(t *testing.T) {
	var ctx BlockModeContext
	outer := InterMotionResult{
		Mode: InterModeResult{Mode: InterModeNewMV},
		MV:   [2]motion.Vector{{Row: -512}},
	}
	dims, ok := BlockSize8x8.Dimensions()
	if !ok {
		t.Fatal("bad block size")
	}
	ctx.markGridInterMotion(BlockSize8x8, 2, 2, outer, dims)
	got, err := intrabcPredictedMV(&ctx, BlockLoopRequest{
		Walk:      BlockWalkRequest{MIColEnd: 32, MIRowEnd: 64},
		SBSizeMIB: 16,
	}, BlockVisit{
		MICol: 4, MIRow: 32,
		X4: 4, Y4: 4,
		Size:    BlockSize8x8,
		HaveTop: true, HaveLeft: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != outer.MV[0] {
		t.Fatalf("predicted intrabc mv=%+v want outer %+v", got, outer.MV[0])
	}
}

// TestIntrabcDiagonalCornerCarriesAcrossSBRow verifies that the diagonal SB
// up-and-to-the-left of an intrabc block contributes its bottom-right corner
// MV via the carrier's PendingDiagonal/Diagonal double-buffer. This is the
// (mi_col-1, mi_row-1) cell libaom's setup_ref_mv_list reads when handling
// libaom_av1_8-bit_intra-only_intrabc_extreme_dv at MI(240,240): the only
// scan position covering the 8x16 neighbor at MI(238,236) is the outer
// block lookup at (X4-1, Y4-1), which falls in the SB diagonally up-left.
// Without the diagonal carrier the SB-row-left store overwrites that cell
// and the predicted DV falls back to (-512, 0), producing wrong pixels.
func TestIntrabcDiagonalCornerCarriesAcrossSBRow(t *testing.T) {
	dvWant := motion.Vector{Row: -1408, Col: 2112}
	carrier := &BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, 2)}
	ensureIntrabcDiagonalCarriers(carrier)
	// Prepare a synthetic prior-SB grid with a non-zero MV at the
	// bottom-right cell (the (sbSize-1, sbSize-1) slot) and capture it
	// into PendingDiagonal[1] (the slot the next-row SB at column 1
	// will consume after promotion).
	var sbMode BlockModeContext
	sbDims, ok := BlockSize8x16.Dimensions()
	if !ok {
		t.Fatal("bad block size")
	}
	sbMode.markGridInterMotion(BlockSize8x16, 14, 12, InterMotionResult{
		Mode: InterModeResult{Mode: InterModeNewMV},
		MV:   [2]motion.Vector{dvWant},
	}, sbDims)
	captureDiagonalCornerToPending(carrier, 1, &sbMode, 16)
	promotePendingDiagonalCarriers(carrier)

	// Now load into a scratch context as the next-row SB at column 1
	// would do, and run the diagonal lookup at (x=-1, y=-1).
	var scratch BlockLoopScratch
	if err := blockLoopLoadRootContext(&scratch, carrier, 1, true, true, 16); err != nil {
		t.Fatal(err)
	}
	got, _, ok := scratch.Mode.crossSBIntrabcGridInterMotion(ReferenceMVStackRequest{
		HaveTop:  true,
		HaveLeft: true,
	}, -1, -1)
	if !ok {
		t.Fatalf("diagonal lookup failed: SBDiagonalMotionValidGrid empty")
	}
	if got.MV[0] != dvWant {
		t.Fatalf("diagonal MV got=%+v want=%+v", got.MV[0], dvWant)
	}
}

// TestIntrabcPredictedMVUsesDiagonalCornerBeforeFallback covers the full
// intrabcPredictedMV path: when the only intrabc candidate is at the cell
// diagonally up-left of the current SB, IntrabcReferenceDVStack's outer
// (-1, -1) block scan must reach it and intrabcPredictedMV must return it
// instead of falling through to the SB-fallback heuristic. Mirrors the
// libaom_av1_8-bit_intra-only_intrabc_extreme_dv block at MI(240,240) whose
// diagonal up-left 8x16 intrabc neighbor at MI(238,236) yielded the
// libaom-computed dv_ref=(-1408, 2112).
func TestIntrabcPredictedMVUsesDiagonalCornerBeforeFallback(t *testing.T) {
	dvWant := motion.Vector{Row: -1408, Col: 2112}
	const cols = 16
	carrier := &BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, cols)}
	ensureIntrabcDiagonalCarriers(carrier)
	var sbMode BlockModeContext
	if err := sbMode.MarkIntrabcMotion(BlockSize8x16, 14, 12, InterMotionResult{
		Mode: InterModeResult{Mode: InterModeNewMV},
		MV:   [2]motion.Vector{dvWant},
	}); err != nil {
		t.Fatal(err)
	}
	// Stage the corner into PendingDiagonal[15] then promote (mirrors the
	// row boundary between the prior SB row (containing SB col 14, the
	// diagonal SB) and the current row).
	captureDiagonalCornerToPending(carrier, 15, &sbMode, 16)
	promotePendingDiagonalCarriers(carrier)

	var scratch BlockLoopScratch
	if err := blockLoopLoadRootContext(&scratch, carrier, 15, true, true, 16); err != nil {
		t.Fatal(err)
	}
	got, err := intrabcPredictedMV(&scratch.Mode, BlockLoopRequest{
		Walk: BlockWalkRequest{
			MIColEnd: 512, MIRowEnd: 512,
		},
		SBSizeMIB:  16,
		Monochrome: true,
	}, BlockVisit{
		MICol: 240, MIRow: 240,
		X4: 0, Y4: 0,
		Size:     BlockSize16x8,
		HaveTop:  true,
		HaveLeft: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != dvWant {
		t.Fatalf("predicted intrabc mv=%+v want diagonal corner %+v", got, dvWant)
	}
}

func TestIntrabcDVValidMatchesLibaomTileAndWavefrontGuards(t *testing.T) {
	req := BlockLoopRequest{
		Walk: BlockWalkRequest{
			MIColEnd: 160,
			MIRowEnd: 64,
		},
		SBSizeMIB:  16,
		Color:      parser.ColorConfig{SubsamplingX: true, SubsamplingY: true},
		Monochrome: false,
	}
	validBlock := BlockVisit{MICol: 0, MIRow: 32, Size: BlockSize16x16}
	if !intrabcDVValid(motion.Vector{Row: -512}, req, validBlock) {
		t.Fatal("full-SB upward intrabc DV should be valid")
	}
	invalidTop := BlockVisit{MICol: 148, MIRow: 16, Size: BlockSize16x8}
	if intrabcDVValid(motion.Vector{Row: -520}, req, invalidTop) {
		t.Fatal("DV with source top above the tile should be invalid")
	}
	if intrabcDVValid(motion.Vector{Row: -511}, req, validBlock) {
		t.Fatal("sub-pixel intrabc DV should be invalid")
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
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: 1}, &PaletteModeScratch{})
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
		EnableMaskedCompound:     true,
		EnableDistWtdCompound:    true,
		AllowHighPrecisionMV:     true,
		AllowWarpedMotion:        true,
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
	}, BlockVisit{
		Size:     BlockSize16x16,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: 1}, &PaletteModeScratch{})
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
	if got := motionCDFs.MotionMode[BlockSize16x16].Values()[3]; got != 1 {
		t.Fatalf("motion mode cdf count=%d want 1", got)
	}
	if got := motionCDFs.OBMC[BlockSize16x16].Values()[2]; got != 0 {
		t.Fatalf("obmc cdf count=%d want 0", got)
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
	}, BlockModeResult{SkipMode: true}, 0, parser.SegmentData{RefFrame: -1}, &PaletteModeScratch{})
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
	badCarrierReq := validReq
	badCarrierReq.Walk.MIColEnd = 32
	badCarrierReq.ContextCarrier = &BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, 1)}
	if _, err := state.DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, &BlockLoopScratch{}, badCarrierReq, func(BlockLoopVisit) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("short context carrier err=%v want %v", err, ErrInvalidDecodeState)
	}
	badReq := validReq
	badReq.DecodeCoefficients = true
	if _, err := state.DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, &BlockLoopScratch{}, badReq, func(BlockLoopVisit) error { return nil }); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("coefficients without prediction err=%v want %v", err, ErrInvalidDecodeState)
	}
	errBoom := errors.New("boom")
	beforeReq := validReq
	beforeReq.BeforeSuperblock = func(BlockLoopSuperblockVisit) error { return errBoom }
	if stats, err := state.DecodeBlockLoop(BlockLoopCDFs{Partition: &partitionCDFs, Mode: &modeCDFs}, &BlockLoopScratch{}, beforeReq, func(BlockLoopVisit) error { return nil }); !errors.Is(err, errBoom) || stats != (BlockLoopStats{}) {
		t.Fatalf("before superblock stats=%+v err=%v want zero stats, %v", stats, err, errBoom)
	}
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
		transformCDFs, coeffCDFs := mustBlockCoeffCDFs(t)
		orderBits := uint8(rawRows%8 + 1)
		orderLimit := uint32(1) << orderBits
		decodePredictionModes := rawRoot&0x80 != 0
		decodeCoefficients := decodePredictionModes && rawRoot&0xf8 == 0xf8
		transformMode := parser.TransformMode(rawRows % 3)
		if decodeCoefficients && root == BlockLevel64x64 {
			transformMode = parser.TransformMode4x4Only
		}
		transformType := transform.Type(rawCols % uint8(transform.TypeCount))
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
			DecodePredictionModes: decodePredictionModes,
			DecodeInterModes:      rawRoot&0xc0 == 0xc0,
			DecodeMotionVectors:   rawRoot&0xe0 == 0xe0,
			DecodeInterIntra:      rawRoot&0xf0 == 0xf0,
			DecodeMotionModes:     rawRoot&0xf0 == 0xf0,
			DecodeCompoundBlend:   rawRoot&0xf0 == 0xf0,
			DecodeCoefficients:    decodeCoefficients,
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
			NumProjRef:               int(rawRows & 1),
			EnableMaskedCompound:     rawRoot&0x10 != 0,
			EnableDistWtdCompound:    rawCols&0x40 != 0,
			EnableOrderHint:          true,
			OrderHintBits:            orderBits,
			CurrentOrderHint:         uint32(rawRoot) % orderLimit,
			Color: parser.ColorConfig{
				MonoChrome:   rawCols&0x20 == 0,
				SubsamplingX: rawCols&1 != 0,
				SubsamplingY: rawRows&1 != 0,
			},
			TransformMode:       transformMode,
			Lossless:            rawRows&0x40 != 0,
			LumaTransformType:   transformType,
			ChromaTransformType: [2]transform.Type{transformType, transformType},
		}
		if req.DecodeCoefficients {
			req.CoeffVisitor = func(parent BlockLoopVisit, block BlockCoeffBlock) error {
				if !parent.Prediction.Valid {
					t.Fatalf("coeff parent missing prediction: %+v", parent)
				}
				assertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
				return nil
			}
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
			Transform: &transformCDFs,
			Coeff:     &coeffCDFs,
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
		if stats.CoefficientTXBs != stats.CoefficientNonZero+stats.CoefficientAllZero {
			t.Fatalf("bad coefficient stats=%+v", stats)
		}
		if stats.CoefficientBlocks > stats.Blocks {
			t.Fatalf("coefficient blocks exceed visited blocks: %+v", stats)
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

// TestDecodeBlockPredictionModePopulatesGlobalWarpForGlobalMVBlock ensures
// libaom's av1_init_warp_params() promotion is mirrored: a forced
// InterModeGlobalMV block whose frame-level warp model is non-translational
// and whose min-side is at least 8 luma samples receives
// result.GlobalWarpedMotionValid=true with shear params computed from the
// frame-level global motion matrix, not from a local neighbor projection.
// Without the promotion the cdf_update vector's first inter frame produces
// frame-wide ±1..±2 sample drift versus libaom across affine GM content.
func TestDecodeBlockPredictionModePopulatesGlobalWarpForGlobalMVBlock(t *testing.T) {
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
	var ctx BlockModeContext
	seg := parser.SegmentationParams{Enabled: true}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i].RefFrame = -1
	}
	seg.Data.Segments[0].GlobalMV = true
	global := [referenceFrameCount]parser.WarpedMotionParams{}
	global[ReferenceFrameLast] = parser.DefaultWarpedMotionParams()
	global[ReferenceFrameLast].Type = parser.GlobalMotionRotZoom
	// A symmetric rotzoom whose alpha/beta/gamma/delta all fit inside
	// is_affine_shear_allowed bounds. The matrix is intentionally
	// non-translational so the promotion path triggers.
	global[ReferenceFrameLast].Matrix = [6]int32{
		8192, -4096,
		1<<16 + 2048, 1024,
		-1024, 1<<16 + 2048,
	}
	gmTypes := [referenceFrameCount]parser.GlobalMotionType{
		ReferenceFrameLast: parser.GlobalMotionRotZoom,
	}
	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{
		InterMode: &interModeCDFs,
		MV:        &mvCDFs,
		Blend:     &blendCDFs,
		Motion:    &motionCDFs,
	}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeInter,
		Segmentation:          seg,
		DecodePredictionModes: true,
		DecodeInterModes:      true,
		DecodeMotionVectors:   true,
		DecodeMotionModes:     true,
		AllowHighPrecisionMV:  true,
		AllowWarpedMotion:     true,
		GlobalMotion:          global,
		GlobalMotionTypes:     gmTypes,
		ReferenceMode:         parser.ReferenceModeSingle,
	}, BlockVisit{
		Size:     BlockSize16x16,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: 1, GlobalMV: true}, &PaletteModeScratch{})
	if err != nil {
		t.Fatalf("decodeBlockPredictionMode: %v", err)
	}
	if !result.InterModeValid || result.InterMode.Mode != InterModeGlobalMV {
		t.Fatalf("inter mode=%+v valid=%v want GLOBALMV", result.InterMode, result.InterModeValid)
	}
	if !result.MotionModeValid || result.MotionMode != MotionModeTranslation {
		t.Fatalf("motion mode=%d valid=%v want SIMPLE_TRANSLATION (1)", result.MotionMode, result.MotionModeValid)
	}
	if !result.GlobalWarpedMotionValid {
		t.Fatalf("global warped motion not populated: %+v", result.GlobalWarpedMotion)
	}
	if result.GlobalWarpedMotion.Params.Type != parser.GlobalMotionRotZoom {
		t.Fatalf("global warped motion type=%v want RotZoom", result.GlobalWarpedMotion.Params.Type)
	}
	if result.GlobalWarpedMotion.Params.Matrix != global[ReferenceFrameLast].Matrix {
		t.Fatalf("global warped motion matrix=%v want %v", result.GlobalWarpedMotion.Params.Matrix, global[ReferenceFrameLast].Matrix)
	}
	// alpha/beta/gamma/delta should equal the values produced by
	// warpShearParams on the same matrix; this guards against accidental
	// future changes to the shear-params derivation.
	expected, ok := warpShearParams(WarpedMotionModel{Params: global[ReferenceFrameLast]})
	if !ok {
		t.Fatalf("test setup matrix not shear-allowed; pick a different rotzoom")
	}
	if result.GlobalWarpedMotion.Alpha != expected.Alpha ||
		result.GlobalWarpedMotion.Beta != expected.Beta ||
		result.GlobalWarpedMotion.Gamma != expected.Gamma ||
		result.GlobalWarpedMotion.Delta != expected.Delta {
		t.Fatalf("shear params got alpha=%d beta=%d gamma=%d delta=%d want alpha=%d beta=%d gamma=%d delta=%d",
			result.GlobalWarpedMotion.Alpha, result.GlobalWarpedMotion.Beta,
			result.GlobalWarpedMotion.Gamma, result.GlobalWarpedMotion.Delta,
			expected.Alpha, expected.Beta, expected.Gamma, expected.Delta)
	}
}

// TestDecodeBlockPredictionModeSkipsGlobalWarpForTranslationGM verifies the
// promotion is gated on a non-translational global motion type. A GLOBALMV
// block whose frame-level GM is TRANSLATION must not receive a populated
// GlobalWarpedMotion - it stays on the regular sub-pel translation path.
func TestDecodeBlockPredictionModeSkipsGlobalWarpForTranslationGM(t *testing.T) {
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
	var ctx BlockModeContext
	seg := parser.SegmentationParams{Enabled: true}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i].RefFrame = -1
	}
	seg.Data.Segments[0].GlobalMV = true
	global := [referenceFrameCount]parser.WarpedMotionParams{}
	global[ReferenceFrameLast] = parser.DefaultWarpedMotionParams()
	global[ReferenceFrameLast].Type = parser.GlobalMotionTranslation
	global[ReferenceFrameLast].Matrix[0] = 13 << 13
	global[ReferenceFrameLast].Matrix[1] = -7 << 13
	gmTypes := [referenceFrameCount]parser.GlobalMotionType{
		ReferenceFrameLast: parser.GlobalMotionTranslation,
	}
	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{
		InterMode: &interModeCDFs,
		MV:        &mvCDFs,
		Blend:     &blendCDFs,
		Motion:    &motionCDFs,
	}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeInter,
		Segmentation:          seg,
		DecodePredictionModes: true,
		DecodeInterModes:      true,
		DecodeMotionVectors:   true,
		DecodeMotionModes:     true,
		AllowHighPrecisionMV:  true,
		AllowWarpedMotion:     true,
		GlobalMotion:          global,
		GlobalMotionTypes:     gmTypes,
		ReferenceMode:         parser.ReferenceModeSingle,
	}, BlockVisit{
		Size:     BlockSize16x16,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: 1, GlobalMV: true}, &PaletteModeScratch{})
	if err != nil {
		t.Fatalf("decodeBlockPredictionMode: %v", err)
	}
	if !result.InterModeValid || result.InterMode.Mode != InterModeGlobalMV {
		t.Fatalf("inter mode=%+v valid=%v want GLOBALMV", result.InterMode, result.InterModeValid)
	}
	if result.GlobalWarpedMotionValid {
		t.Fatalf("global warped motion should not be populated for translational GM: %+v", result.GlobalWarpedMotion)
	}
}

// TestDecodeBlockPredictionModeSkipsGlobalWarpForSmallBlock verifies the
// promotion is gated on a block min-side of at least 8 luma samples. A
// GLOBALMV BLOCK_4X4 (or any block under 8 on either side) with a rotzoom
// global motion stays on the regular sub-pel translation path, matching
// libaom's is_global_mv_block(...) block_size_allowed condition.
func TestDecodeBlockPredictionModeSkipsGlobalWarpForSmallBlock(t *testing.T) {
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
	var ctx BlockModeContext
	seg := parser.SegmentationParams{Enabled: true}
	for i := range seg.Data.Segments {
		seg.Data.Segments[i].RefFrame = -1
	}
	seg.Data.Segments[0].GlobalMV = true
	global := [referenceFrameCount]parser.WarpedMotionParams{}
	global[ReferenceFrameLast] = parser.DefaultWarpedMotionParams()
	global[ReferenceFrameLast].Type = parser.GlobalMotionRotZoom
	global[ReferenceFrameLast].Matrix = [6]int32{
		8192, -4096,
		1<<16 + 2048, 1024,
		-1024, 1<<16 + 2048,
	}
	gmTypes := [referenceFrameCount]parser.GlobalMotionType{
		ReferenceFrameLast: parser.GlobalMotionRotZoom,
	}
	result, err := state.decodeBlockPredictionMode(BlockLoopCDFs{
		InterMode: &interModeCDFs,
		MV:        &mvCDFs,
		Blend:     &blendCDFs,
		Motion:    &motionCDFs,
	}, &ctx, BlockLoopRequest{
		FrameType:             parser.FrameTypeInter,
		Segmentation:          seg,
		DecodePredictionModes: true,
		DecodeInterModes:      true,
		DecodeMotionVectors:   true,
		DecodeMotionModes:     true,
		AllowHighPrecisionMV:  true,
		AllowWarpedMotion:     true,
		GlobalMotion:          global,
		GlobalMotionTypes:     gmTypes,
		ReferenceMode:         parser.ReferenceModeSingle,
	}, BlockVisit{
		Size:     BlockSize4x4,
		X4:       0,
		Y4:       0,
		HaveTop:  true,
		HaveLeft: true,
	}, BlockModeResult{}, 0, parser.SegmentData{RefFrame: 1, GlobalMV: true}, &PaletteModeScratch{})
	if err != nil {
		t.Fatalf("decodeBlockPredictionMode: %v", err)
	}
	if result.GlobalWarpedMotionValid {
		t.Fatalf("global warped motion should not be populated for 4x4 block: %+v", result.GlobalWarpedMotion)
	}
}
