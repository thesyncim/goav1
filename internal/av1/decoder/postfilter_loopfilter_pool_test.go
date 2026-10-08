package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// TestLoopFilterPooledMatchesSerial proves the pooled mask-driven loop-filter
// apply (applyLoopFilterEdgesMaybePooled with a multi-lane pool) is byte-identical
// to the single-shot serial mask apply, for several worker counts. The frame is
// tall enough (4 mask region-rows) that the region-row band split engages, and
// carries chroma so the plane-parallel horizontal pass is exercised. Identical
// pixels across worker counts is the worker-invariance guarantee the postfilter
// pool rests on (mirrors TestCDEFPostFilterPooledMatchesSerial).
func TestLoopFilterPooledMatchesSerial(t *testing.T) {
	const width, height = 256, 512 // 4:2:0 luma 64x128 MI, SB128H == 4 region-rows
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	event := lfMaskApplyEvent(seq, size, lf)
	format := frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	// Uniform grid of 16x16 inter blocks (4 MI) covering the whole frame.
	var recs []lfMaskApplyRecord
	for row := 0; row < height/4; row += 4 {
		for col := 0; col < width/4; col += 4 {
			recs = append(recs, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4)})
		}
	}
	plainRecs := make([]threading.FrameWorkLoopFilterBlockRecord, len(recs))
	for i := range recs {
		plainRecs[i] = recs[i].rec
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, plainRecs...)

	// Serial reference: the known-good single-shot mask apply.
	serialOut := testFrameWorkCDEFFrame(t, format)
	lfMaskApplyFillFrame(serialOut)
	serialMasks := buildLoopFilterMasksFromRecords(t, event, size, recs, 4)
	serialCtx := FrameWorkPostFilterContext{Event: event, Output: serialOut, LoopFilterMap: &filterMap, LoopFilterMasks: serialMasks}
	if _, err := serialCtx.ApplyLoopFilterEdgesFromMasks(serialMasks, filterMap); err != nil {
		t.Fatalf("serial mask apply: %v", err)
	}

	for _, workers := range []int{2, 3, 4, 8} {
		pool, err := threading.NewPool(workers)
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		pooledOut := testFrameWorkCDEFFrame(t, format)
		lfMaskApplyFillFrame(pooledOut)
		pooledMasks := buildLoopFilterMasksFromRecords(t, event, size, recs, 4)
		pooledCtx := FrameWorkPostFilterContext{Event: event, Output: pooledOut, LoopFilterMap: &filterMap, LoopFilterMasks: pooledMasks, pool: pool}
		res, err := pooledCtx.applyLoopFilterEdgesMaybePooled(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
		if err != nil {
			pool.Close()
			t.Fatalf("workers=%d pooled apply: %v", workers, err)
		}
		if res.Applied == 0 {
			pool.Close()
			t.Fatalf("workers=%d degenerate: pooled apply filtered no edges", workers)
		}
		lfMaskApplyAssertFramesEqual(t, serialOut, pooledOut)
		pool.Close()
	}
}
