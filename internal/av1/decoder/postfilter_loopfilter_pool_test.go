package decoder

import (
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// TestLoopFilterPooledMatchesSerial proves the pooled mask-driven loop-filter
// apply is byte-identical to the single-shot serial mask apply across worker
// counts, region-column counts, and plane gating. It also compares all result
// counters and plan stats so column bands cannot lose or double-count edges.
func TestLoopFilterPooledMatchesSerial(t *testing.T) {
	for _, tc := range []struct {
		name         string
		width        int
		height       int
		bitDepth     uint8
		subsamplingX bool
		subsamplingY bool
		mono         bool
		chromaGated  bool
		mixed        bool
	}{
		{name: "one_column_420", width: 128, height: 512, bitDepth: 8, subsamplingX: true, subsamplingY: true},
		{name: "two_columns_partial_region_420", width: 136, height: 256, bitDepth: 8, subsamplingX: true, subsamplingY: true},
		{name: "three_columns_mixed_420", width: 384, height: 256, bitDepth: 8, subsamplingX: true, subsamplingY: true, mixed: true},
		{name: "four_columns_chroma_gated_420", width: 512, height: 256, bitDepth: 8, subsamplingX: true, subsamplingY: true, chromaGated: true},
		{name: "four_columns_mono", width: 512, height: 256, bitDepth: 8, mono: true},
		{name: "three_columns_10bit_444", width: 384, height: 256, bitDepth: 10},
		{name: "two_columns_12bit_444", width: 136, height: 256, bitDepth: 12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			size := parser.FrameSize{
				CodedWidth: uint32(tc.width), UpscaledWidth: uint32(tc.width),
				Height: uint32(tc.height), SuperResDenominator: 8,
			}
			seq := lfMaskApply420Sequence()
			seq.ColorConfig.BitDepth = tc.bitDepth
			seq.ColorConfig.SubsamplingX = tc.subsamplingX
			seq.ColorConfig.SubsamplingY = tc.subsamplingY
			lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
			if tc.mono {
				seq.ColorConfig.MonoChrome = true
				seq.ColorConfig.SubsamplingX = false
				seq.ColorConfig.SubsamplingY = false
				lf.LevelU = 0
				lf.LevelV = 0
			}
			if tc.chromaGated {
				lf.LevelU = 0
				lf.LevelV = 0
			}
			event := lfMaskApplyEvent(seq, size, lf)
			format := frame.Format{
				Width: tc.width, Height: tc.height, BitDepth: tc.bitDepth,
				MonoChrome: tc.mono, SubsamplingX: tc.subsamplingX, SubsamplingY: tc.subsamplingY, Align: 64,
			}

			cols, rows, err := frameWorkLoopFilterMapGrid(size)
			if err != nil {
				t.Fatalf("loop-filter map grid: %v", err)
			}
			var recs []lfMaskApplyRecord
			for row := 0; row < rows; row += 4 {
				for col := 0; col < cols; col += 4 {
					rec := testFrameWorkLoopFilterPostFilterRecordAt(col, row, min(cols, col+4), min(rows, row+4))
					if !tc.mono && !tc.subsamplingX && !tc.subsamplingY {
						rec.TransformTree.UV = tile.TransformSize16x16
					}
					if tc.mixed {
						switch (row/4 + col/4) % 3 {
						case 1:
							rec.SkipTransform = true
						case 2:
							rec.TransformTree = tile.TransformTreeResult{
								Y: tile.TransformSize16x16, UV: tile.TransformSize8x8,
								HasUV: true, Variable: true, Split: [2]uint16{1, 0},
							}
						}
					}
					recs = append(recs, lfMaskApplyRecord{rec: rec})
				}
			}
			plainRecs := make([]threading.FrameWorkLoopFilterBlockRecord, len(recs))
			for i := range recs {
				plainRecs[i] = recs[i].rec
			}
			filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, plainRecs...)

			serialOut := testFrameWorkCDEFFrame(t, format)
			lfMaskApplyFillFrame(serialOut)
			serialMasks := buildLoopFilterMasksFromRecords(t, event, size, recs, 4)
			serialCtx := FrameWorkPostFilterContext{Event: event, Output: serialOut, LoopFilterMap: &filterMap, LoopFilterMasks: serialMasks}
			serialResult, err := serialCtx.ApplyLoopFilterEdgesFromMasks(serialMasks, filterMap)
			if err != nil {
				t.Fatalf("serial mask apply: %v", err)
			}

			for _, workers := range []int{2, 3, 4, 8} {
				t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
					pool, err := threading.NewPool(workers)
					if err != nil {
						t.Fatalf("create worker pool: %v", err)
					}
					defer pool.Close()
					pooledOut := testFrameWorkCDEFFrame(t, format)
					lfMaskApplyFillFrame(pooledOut)
					pooledMasks := buildLoopFilterMasksFromRecords(t, event, size, recs, 4)
					pooledCtx := FrameWorkPostFilterContext{Event: event, Output: pooledOut, LoopFilterMap: &filterMap, LoopFilterMasks: pooledMasks, pool: pool}
					res, err := pooledCtx.applyLoopFilterEdgesMaybePooled(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
					if err != nil {
						t.Fatalf("pooled apply: %v", err)
					}
					if res.Applied == 0 {
						t.Fatal("degenerate: pooled apply filtered no edges")
					}
					if res != serialResult {
						t.Fatalf("pooled result=%+v, want serial result=%+v", res, serialResult)
					}
					if tc.chromaGated && (res.PlaneEdges[1] != 0 || res.PlaneEdges[2] != 0) {
						t.Fatalf("chroma-gated result has chroma edges: %v", res.PlaneEdges)
					}
					lfMaskApplyAssertFramesEqual(t, serialOut, pooledOut)
				})
			}
		})
	}
}
