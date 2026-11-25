package decoder

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/filmgrain"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
	"github.com/thesyncim/goav1/internal/av1/rtp"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"testing"
	"unsafe"
)

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Event", size: unsafe.Sizeof(Event{}), max: 5632},
		// The compact loop-filter lifecycle pointer is included in 566800.
		// Owned MV storage adds 20 bytes of handles, refcounts and flags plus
		// 4 bytes of alignment on 64-bit targets. Its nine backing slices
		// replace the previous eight slot slices and one current-frame slice.
		// The loop-filter mask handle adds level-table, tile-edge and noskip
		// pointers plus a single-tile flag.
		// The three restoration stripe-boundary handles each add a pixel-row
		// (Bytes) flag, 8 bytes with alignment.
		{name: "FrameWorkState", size: unsafe.Sizeof(FrameWorkState{}), max: 582784}, // +24: the shared path's current MV grid; +13320: an intra block copy MV context (DV) in each saved CDF set; +8: the loop-filter map's NoSkip grid pointer (encoder skip-only frames); +1632: frame-thread MV-grid handles and the extra CDF pool pointer; +920: segmentation-map store (dav1d segmap refs per slot and in-flight frame)
		{name: "FrameWorkLoopFilterPostFilterEdge", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterEdge{}), max: 16},
		{name: "FrameWorkLoopFilterPostFilterLevelStats", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterLevelStats{}), max: 12},
		{name: "FrameWorkLoopFilterPostFilterPlan", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterPlan{}), max: 136},
		{name: "FrameWorkLoopFilterPostFilterApplyResult", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterApplyResult{}), max: 176},
		{name: "frameWorkLoopFilterPlanningContext", size: unsafe.Sizeof(frameWorkLoopFilterPlanningContext{}), max: 64},
		{name: "FrameWorkLoopFilterBlockRecord", size: unsafe.Sizeof(threading.FrameWorkLoopFilterBlockRecord{}), max: 34},
		{name: "frameWorkLoopFilterLumaPreviousCache", size: unsafe.Sizeof(frameWorkLoopFilterLumaPreviousCache{}), max: 74},
		{name: "frameWorkLoopFilterChromaPreviousCache", size: unsafe.Sizeof(frameWorkLoopFilterChromaPreviousCache{}), max: 48},
		{name: "FrameWorkCDEFPostFilterResult", size: unsafe.Sizeof(FrameWorkCDEFPostFilterResult{}), max: 12},
		{name: "FrameWorkSuperResPostFilterPlanePlan", size: unsafe.Sizeof(FrameWorkSuperResPostFilterPlanePlan{}), max: 24},
		{name: "FrameWorkSuperResPostFilterPlan", size: unsafe.Sizeof(FrameWorkSuperResPostFilterPlan{}), max: 160},
		{name: "FrameWorkSuperResPostFilterResult", size: unsafe.Sizeof(FrameWorkSuperResPostFilterResult{}), max: 544},
		{name: "FrameWorkFilmGrainPostFilterPlanePlan", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterPlanePlan{}), max: 16},
		{name: "FrameWorkFilmGrainPostFilterPlan", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterPlan{}), max: 264},
		{name: "FrameWorkFilmGrainPostFilterResult", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterResult{}), max: 400},
		{name: "FrameWorkFilmGrainPostFilterLumaGrain", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterLumaGrain{}), max: 32},
		{name: "FrameWorkFilmGrainPostFilterChromaGrain", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterChromaGrain{}), max: 32},
		{name: "FrameWorkFilmGrainPostFilterLumaRow", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterLumaRow{}), max: 16},
		{name: "FrameWorkFilmGrainPostFilterChromaRow", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterChromaRow{}), max: 20},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}

// lfmask_apply_diff_test.go is the fast differential gate for the mask-driven
// loop-filter apply. For a synthetic decoded frame it applies deblocking two
// ways into two independent plane copies -- (a) the production edge-list
// ApplyLoopFilterEdges, (b) the mask-driven ApplyLoopFilterEdgesFromMasks -- and
// asserts the filtered planes are byte-identical. It runs in milliseconds and is
// the iteration gate for the apply-path swap.

type lfMaskApplyRecord struct {
	rec   threading.FrameWorkLoopFilterBlockRecord
	intra bool
}

// buildLoopFilterMasksFromRecords replays the decode-time deblocking mask build
// (threading.FrameWorkLoopFilterMasks.build) over records supplied in decode
// (Z) order, carrying the above/left tx-context arrays exactly as the tile block
// walk does. It fills only edge geometry (nil level cache); the apply resolves
// levels from the map.
func buildLoopFilterMasksFromRecords(t testing.TB, event Event, size parser.FrameSize, records []lfMaskApplyRecord, sbLog2 int) *threading.FrameWorkLoopFilterMasks {
	t.Helper()
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatalf("grid: %v", err)
	}
	color := event.SequenceHeader.ColorConfig
	layout := lfmask.Layout{Mono: color.MonoChrome}
	if color.SubsamplingX {
		layout.SSHor = 1
	}
	if color.SubsamplingY {
		layout.SSVer = 1
	}
	hasChroma := !color.MonoChrome
	sb128w, sb128h, maskCount := threading.FrameWorkLoopFilterMaskShape(cols, rows)
	handle := &threading.FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, maskCount),
		LevelCache: make([]uint8, 4*(cols*rows)),
		Cols:       cols,
		Rows:       rows,
		SB128W:     sb128w,
		SB128H:     sb128h,
		Layout:     layout,
		HasChroma:  hasChroma,
	}

	var builder lfmask.Builder
	aY := make([]uint8, cols)
	for i := range aY {
		aY[i] = 2
	}
	var aUV []uint8
	if hasChroma {
		aUV = make([]uint8, cols)
		for i := range aUV {
			aUV[i] = 1
		}
	}
	var lY [32]uint8
	var lUV [32]uint8
	resetLeft := func() {
		for i := range lY {
			lY[i] = 2
		}
		if hasChroma {
			for i := range lUV {
				lUV[i] = 1
			}
		}
	}
	started := false
	curSBRow := 0
	var nilLC lfmask.LevelCache

	for i := range records {
		r := &records[i]
		bx := int(r.rec.Block.MICol)
		by := int(r.rec.Block.MIRow)
		sbRow := by >> sbLog2
		if !started || sbRow != curSBRow {
			resetLeft()
			curSBRow = sbRow
			started = true
		}
		by4 := by & 31
		ay := aY[bx:]
		ly := lY[by4:]
		var auv, luv []uint8
		if hasChroma && r.rec.TransformTree.HasUV {
			dims, _ := r.rec.Block.Size.Dimensions()
			if (int(dims.W4) > layout.SSHor || bx&1 != 0) && (int(dims.H4) > layout.SSVer || by&1 != 0) {
				auv = aUV[bx>>layout.SSHor:]
				luv = lUV[by4>>layout.SSVer:]
			}
		}
		m := &handle.Masks[(by>>5)*sb128w+(bx>>5)]
		if r.intra {
			builder.CreateIntra(m, nilLC, lfmask.Levels{}, bx, by, cols, rows,
				r.rec.Block.Size, r.rec.TransformTree.Y, r.rec.TransformTree.UV, layout, ay, ly, auv, luv)
			continue
		}
		skip := 0
		if r.rec.SkipTransform {
			skip = 1
		}
		builder.CreateInter(m, nilLC, lfmask.Levels{}, bx, by, cols, rows, skip,
			r.rec.Block.Size, r.rec.TransformTree.Y, r.rec.TransformTree.Split, r.rec.TransformTree.UV,
			layout, ay, ly, auv, luv)
	}
	return handle
}

func lfMaskApplyFillPlane(plane frame.Plane, bytesPerSample int, bitDepth uint8, salt int) {
	maxv := (1 << uint(bitDepth)) - 1
	base := 1 << uint(bitDepth-1)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			// Smoothly varying content (small per-sample steps) so the deblock
			// filter actually engages and filter-width/level differences change
			// pixels -- large gradients would make every filter a no-op and mask
			// class bugs invisible.
			v := base + ((x*3 + y*5 + salt*11) % 13) - 6
			if v < 0 {
				v = 0
			}
			if v > maxv {
				v = maxv
			}
			off := y*plane.Stride + x*bytesPerSample
			if bytesPerSample == 1 {
				plane.Pix[off] = byte(v)
			} else {
				plane.Pix[off] = byte(v)
				plane.Pix[off+1] = byte(v >> 8)
			}
		}
	}
}

func lfMaskApplyFillFrame(f *frame.Frame) {
	bps := f.Layout.BytesPerSample
	bd := f.Format.BitDepth
	lfMaskApplyFillPlane(f.Y, bps, bd, 0)
	if !f.Format.MonoChrome {
		lfMaskApplyFillPlane(f.U, bps, bd, 1)
		lfMaskApplyFillPlane(f.V, bps, bd, 2)
	}
}

func lfMaskApplyAssertFramesEqual(t *testing.T, a, b *frame.Frame) {
	t.Helper()
	planes := [][2]frame.Plane{{a.Y, b.Y}}
	if !a.Format.MonoChrome {
		planes = append(planes, [2]frame.Plane{a.U, b.U}, [2]frame.Plane{a.V, b.V})
	}
	names := []string{"Y", "U", "V"}
	bps := a.Layout.BytesPerSample
	for pi, pp := range planes {
		pa, pb := pp[0], pp[1]
		if pa.Width != pb.Width || pa.Height != pb.Height {
			t.Fatalf("plane %s shape mismatch", names[pi])
		}
		for y := 0; y < pa.Height; y++ {
			for x := 0; x < pa.Width; x++ {
				off := y*pa.Stride + x*bps
				offb := y*pb.Stride + x*bps
				for k := 0; k < bps; k++ {
					if pa.Pix[off+k] != pb.Pix[offb+k] {
						t.Fatalf("plane %s differs at x=%d y=%d byte=%d: edge-list=%d mask=%d",
							names[pi], x, y, k, pa.Pix[off+k], pb.Pix[offb+k])
					}
				}
			}
		}
	}
}

func runMaskApplyDiff(t *testing.T, event Event, size parser.FrameSize, format frame.Format, records []lfMaskApplyRecord, sbLog2 int) {
	t.Helper()
	recs := make([]threading.FrameWorkLoopFilterBlockRecord, len(records))
	for i := range records {
		recs[i] = records[i].rec
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, recs...)

	frameA := testFrameWorkCDEFFrame(t, format)
	frameB := testFrameWorkCDEFFrame(t, format)
	lfMaskApplyFillFrame(frameA)
	lfMaskApplyFillFrame(frameB)

	// (a) production edge-list apply into frameA.
	ctxA := FrameWorkPostFilterContext{Event: event, Output: frameA, LoopFilterMap: &filterMap}
	scratch, err := ctxA.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatalf("scratch len: %v", err)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges)
	scheduleScratch := make([]uint32, scratch.Schedule)
	resA, err := ctxA.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges, Schedule: scheduleScratch})
	if err != nil {
		t.Fatalf("edge-list apply: %v", err)
	}

	// (b) mask-driven apply into frameB.
	masks := buildLoopFilterMasksFromRecords(t, event, size, records, sbLog2)
	ctxB := FrameWorkPostFilterContext{Event: event, Output: frameB, LoopFilterMap: &filterMap}
	if _, err := ctxB.ApplyLoopFilterEdgesFromMasks(masks, filterMap); err != nil {
		t.Fatalf("mask apply: %v", err)
	}
	if resA.Applied == 0 {
		t.Fatalf("degenerate: edge-list apply filtered no edges")
	}
	lfMaskApplyAssertFramesEqual(t, frameA, frameB)
}

func lfMaskApplyEvent(seq parser.SequenceHeader, size parser.FrameSize, lf parser.LoopFilterParams) Event {
	return Event{SequenceHeader: seq, FrameSize: size, LoopFilter: lf}
}

func lfMaskApply420Sequence() parser.SequenceHeader {
	seq := testSequence()
	seq.ColorConfig.SubsamplingX = true
	seq.ColorConfig.SubsamplingY = true
	return seq
}

// --- test cases -----------------------------------------------------------

func TestMaskApplyDiffUniformGrid(t *testing.T) {
	// A full 64x64 frame tiled with 16x16 inter blocks (fixed 16x16 luma tx,
	// 8x8 chroma). Every internal block edge is in scope.
	size := parser.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	var recs []lfMaskApplyRecord
	for row := 0; row < 16; row += 4 {
		for col := 0; col < 16; col += 4 {
			recs = append(recs, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4)})
		}
	}
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format, recs, 4)
}

func TestMaskApplyDiffVariableTransform(t *testing.T) {
	// One 16x16 inter block filling a 16x16 frame with a split transform tree.
	size := parser.FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 16, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{24, 24}, LevelU: 20, LevelV: 20, Sharpness: 0}
	rec := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	rec.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	format := frame.Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format, []lfMaskApplyRecord{{rec: rec}}, 4)
}

func TestMaskApplyDiffUnequalTransforms(t *testing.T) {
	// Two horizontally adjacent inter blocks with different transform sizes so
	// the shared vertical boundary strength is genuinely min(prev, cur).
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{28, 28}, LevelU: 24, LevelV: 24, Sharpness: 2}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	format := frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format,
		[]lfMaskApplyRecord{{rec: left}, {rec: right}}, 4)
}

func TestMaskApplyDiffSkipTransform(t *testing.T) {
	// Left block is skip (no inner tx edges), right has a variable tree. The
	// shared boundary flows through the carried left context.
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{18, 22}, LevelU: 14, LevelV: 14, Sharpness: 1}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SkipTransform = true
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	format := frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format,
		[]lfMaskApplyRecord{{rec: left}, {rec: right}}, 4)
}

func TestMaskApplyDiffStackedBlocks(t *testing.T) {
	// Two vertically stacked inter blocks with differing transforms; the shared
	// horizontal boundary resolves through the carried above context.
	size := parser.FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 32, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{30, 26}, LevelU: 22, LevelV: 20, Sharpness: 0}
	top := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	top.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	bottom.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	format := frame.Format{Width: 16, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format,
		[]lfMaskApplyRecord{{rec: top}, {rec: bottom}}, 4)
}

func TestMaskApplyDiffLevelFromPrevious(t *testing.T) {
	// Right block resolves to loop-filter level 0 (segment delta), left to 8;
	// the shared vertical boundary must fall back to the left side's level.
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	event := lfMaskApplyEvent(seq, size, parser.LoopFilterParams{LevelY: [2]uint8{8, 8}, LevelU: 8, LevelV: 8})
	event.Segmentation.Enabled = true
	event.Segmentation.Data.Segments[1].DeltaLFYV = -63
	event.Segmentation.Data.Segments[1].DeltaLFYH = -63
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SegmentID = 0
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.SegmentID = 1
	format := frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, event, size, format, []lfMaskApplyRecord{{rec: left}, {rec: right}}, 4)
}

func TestMaskApplyDiffHighBitDepth10(t *testing.T) {
	// 10-bit 32x32 frame; a 2x2 grid of 16x16 inter blocks.
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 32, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	seq.ColorConfig.BitDepth = 10
	lf := parser.LoopFilterParams{LevelY: [2]uint8{26, 26}, LevelU: 20, LevelV: 20, Sharpness: 1}
	var recs []lfMaskApplyRecord
	for row := 0; row < 8; row += 4 {
		for col := 0; col < 8; col += 4 {
			recs = append(recs, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4)})
		}
	}
	format := frame.Format{Width: 32, Height: 32, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format, recs, 4)
}

func TestMaskApplyDiffOddCropRunsAcrossFormats(t *testing.T) {
	// Odd crops leave MI(4)-aligned pixels and MI(8)-aligned allocation padding
	// around the final deblock edges. Exercise every supported bit depth against
	// each chroma layout to check both ordinary and padded edge lines.
	size := parser.FrameSize{CodedWidth: 34, UpscaledWidth: 34, Height: 34, SuperResDenominator: 8}
	for _, bitDepth := range []uint8{8, 10, 12} {
		for _, subsampling := range []struct {
			name string
			x, y bool
		}{
			{name: "420", x: true, y: true},
			{name: "422", x: true},
			{name: "444"},
		} {
			t.Run(fmt.Sprintf("%dbit/%s", bitDepth, subsampling.name), func(t *testing.T) {
				seq := testSequence()
				seq.ColorConfig.BitDepth = bitDepth
				seq.ColorConfig.SubsamplingX = subsampling.x
				seq.ColorConfig.SubsamplingY = subsampling.y
				lf := parser.LoopFilterParams{LevelY: [2]uint8{30, 24}, LevelU: 26, LevelV: 22, Sharpness: 2}
				cols, rows, err := frameWorkLoopFilterMapGrid(size)
				if err != nil {
					t.Fatal(err)
				}
				var recs []lfMaskApplyRecord
				for row := 0; row < rows; row += 4 {
					for col := 0; col < cols; col += 4 {
						recs = append(recs, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(
							col, row, minInt(cols, col+4), minInt(rows, row+4),
						)})
					}
				}
				format := frame.Format{
					Width: 34, Height: 34, BitDepth: bitDepth,
					SubsamplingX: subsampling.x, SubsamplingY: subsampling.y, Align: 64,
				}
				runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format, recs, 4)
			})
		}
	}
}

func TestMaskApplyDiffOddCropAcrossMaskRegions(t *testing.T) {
	// 226 is both a near-MI(8)-boundary crop and wider than one 128x128 mask
	// region. The final width-14 edge at x=224 uses allocated padding, while the
	// scan also crosses the mask-region boundary at x=128.
	size := parser.FrameSize{CodedWidth: 226, UpscaledWidth: 226, Height: 226, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	seq.ColorConfig.BitDepth = 12
	lf := parser.LoopFilterParams{LevelY: [2]uint8{30, 24}, LevelU: 26, LevelV: 22, Sharpness: 2}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	var recs []lfMaskApplyRecord
	for row := 0; row < rows; row += 4 {
		for col := 0; col < cols; col += 4 {
			recs = append(recs, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(
				col, row, minInt(cols, col+4), minInt(rows, row+4),
			)})
		}
	}
	format := frame.Format{Width: 226, Height: 226, BitDepth: 12, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format, recs, 4)
}

func TestMaskApplyFromMasksAllocs(t *testing.T) {
	size := parser.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	event := lfMaskApplyEvent(seq, size, parser.LoopFilterParams{LevelY: [2]uint8{24, 24}, LevelU: 20, LevelV: 20})
	var recs []lfMaskApplyRecord
	for row := 0; row < 16; row += 4 {
		for col := 0; col < 16; col += 4 {
			recs = append(recs, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4)})
		}
	}
	records := make([]threading.FrameWorkLoopFilterBlockRecord, len(recs))
	for i := range recs {
		records[i] = recs[i].rec
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)
	masks := buildLoopFilterMasksFromRecords(t, event, size, recs, 4)
	output := testFrameWorkCDEFFrame(t, frame.Format{
		Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64,
	})
	lfMaskApplyFillFrame(output)
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	if _, err := ctx.ApplyLoopFilterEdgesFromMasks(masks, filterMap); err != nil {
		t.Fatalf("warm apply: %v", err)
	}
	var applyErr error
	allocs := testing.AllocsPerRun(100, func() {
		_, applyErr = ctx.ApplyLoopFilterEdgesFromMasks(masks, filterMap)
	})
	if applyErr != nil {
		t.Fatalf("apply: %v", applyErr)
	}
	if allocs != 0 {
		t.Fatalf("warm mask apply allocs/run = %v, want 0", allocs)
	}
}

func TestMaskApplyDiffChromaLarge(t *testing.T) {
	// 32x32 frame, single 32x32 inter block with 8x8 chroma tx: exercises the
	// interior chroma edges (2x2 grid of chroma transforms).
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 32, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{22, 22}, LevelU: 30, LevelV: 28, Sharpness: 0}
	rec := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 8, 8)
	rec.Block.Size = tile.BlockSize32x32
	rec.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize8x8, HasUV: true}
	format := frame.Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format, []lfMaskApplyRecord{{rec: rec}}, 4)
}

func TestMaskApplyDiffChromaSmallNeighbor(t *testing.T) {
	// Reproduces the chroma block-boundary width divergence: the left block uses
	// a 4x4 chroma transform (class 0 / width 4) and the right block a larger
	// chroma transform. The shared vertical chroma boundary must resolve to
	// min(class(right), class(left)) = class 0 / width 4 via the carried left
	// chroma context. Frame 24x8 (chroma 12x4) fully tiled; boundary is interior.
	size := parser.FrameSize{CodedWidth: 24, UpscaledWidth: 24, Height: 8, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 24, LevelV: 22, Sharpness: 1}
	// Left: 8x8 luma at MI(0,0) with 4x4 chroma tx (chroma 4x4 -> class 0).
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 2, 2)
	left.Block.Size = tile.BlockSize8x8
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	// Right: 16x8 luma at MI(2,0) with 8x4 chroma tx (chroma 8x4 -> class 1).
	right := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 6, 2)
	right.Block.Size = tile.BlockSize16x8
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x8, UV: tile.TransformSize8x4, HasUV: true}
	format := frame.Format{Width: 24, Height: 8, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	runMaskApplyDiff(t, lfMaskApplyEvent(seq, size, lf), size, format,
		[]lfMaskApplyRecord{{rec: left}, {rec: right}}, 4)
}

func TestMaskApplyDiffChromaSharedSubBlock(t *testing.T) {
	// Two vertically stacked 16x4 luma blocks (1 MI tall each) at rows 0 and 1
	// share a single chroma row in 4:2:0. Per AV1 (dav1d has_chroma), only the
	// odd sub-block (row 1) carries the shared chroma; the even one must not emit
	// a chroma edge nor update the chroma neighbour context. A left 8x8 block
	// with a 4x4 chroma transform (class 0) sets the shared boundary width, so
	// the boundary must resolve to width 4 -- not the wider class the even
	// sub-block would otherwise inject into the mask.
	size := parser.FrameSize{CodedWidth: 24, UpscaledWidth: 24, Height: 8, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 26, LevelV: 24, Sharpness: 1}
	// Left 8x8 luma at MI(0,0), covers rows 0-1, chroma 4x4 (class 0).
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 2, 2)
	left.Block.Size = tile.BlockSize8x8
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	// Top 16x4 luma at MI(2,0) (even row) and bottom 16x4 at MI(2,1) (odd row),
	// both with an 8x4 chroma transform (class 1 vertical) and HasUV set.
	top := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 6, 1)
	top.Block.Size = tile.BlockSize16x4
	top.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x4, UV: tile.TransformSize8x4, HasUV: true}
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(2, 1, 6, 2)
	bottom.Block.Size = tile.BlockSize16x4
	bottom.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x4, UV: tile.TransformSize8x4, HasUV: true}
	format := frame.Format{Width: 24, Height: 8, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	recs := []lfMaskApplyRecord{{rec: left}, {rec: top}, {rec: bottom}}
	event := lfMaskApplyEvent(seq, size, lf)
	runMaskApplyDiff(t, event, size, format, recs, 4)

	// Direct guard on the built mask: the shared vertical chroma boundary at
	// chroma col 1 (region 0, pos 1) must carry a single strength class. If the
	// even sub-block also emitted chroma, both the class-0 (min with the left
	// 4x4 transform) and class-1 lanes would be set and the scan would pick the
	// wider class -- the exact deblock divergence this fixes.
	masks := buildLoopFilterMasksFromRecords(t, event, size, recs, 4)
	region := &masks.Masks[0]
	var class0, class1 uint16
	for row := 0; row < 4; row++ {
		class0 |= region.UV[0][1][0][0] & (1 << uint(row))
		class1 |= region.UV[0][1][1][0] & (1 << uint(row))
	}
	if class1 != 0 {
		t.Fatalf("shared chroma boundary has class-1 (width-6) bits set: class0=%#x class1=%#x; the even sub-block leaked a chroma edge", class0, class1)
	}
	if class0 == 0 {
		t.Fatalf("shared chroma boundary has no class-0 edge; test setup is degenerate")
	}
}

var _ = loopfilter.PlaneY

// lfmask_diff_test.go cross-validates the internal/av1/lfmask port (dav1d's
// per-superblock bitmask edge planner) against the production
// LoopFilterPostFilterPlan edge builder. Both implement the same AV1 deblocking
// decisions; this test asserts the lfmask build+scan reproduces the production
// per-4x4 edge geometry, width class, and level on synthetic block layouts.

// lumaWidthByClass maps a dav1d strength class (imin(2, log2 of the tx dimension
// perpendicular to the edge)) to the production pixel filter width. Production
// stores 4/8/14 (frameWorkLoopFilterWidthByTX); dav1d's 4<<idx (4/8/16) is the
// same filter selection under a different encoding.
var lumaWidthByClass = [3]uint8{4, 8, 14}

type lfDiffCell struct {
	width uint8
	level uint8
}

// productionLumaCells expands the production run-merged luma edge list for one
// edge direction to a per-4x4 cell map keyed by (x4, y4). Frame-boundary edges
// (x4==0 for vertical, y4==0 for horizontal) are dropped to match dav1d's
// apply-time have_left / have_top skip.
func productionLumaCells(t *testing.T, ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatalf("scratch len: %v", err)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	cells := map[[2]int]lfDiffCell{}
	for _, e := range edges[:plan.StoredEdges] {
		if e.Plane != loopfilter.PlaneY || e.Edge != edge {
			continue
		}
		for i := 0; i < int(e.Length4); i++ {
			// A vertical edge is a vertical line at fixed X4 spanning rows; a
			// horizontal edge is a horizontal line at fixed Y4 spanning columns.
			x4, y4 := int(e.X4), int(e.Y4)
			if edge == loopfilter.EdgeVertical {
				y4 += i
			} else {
				x4 += i
			}
			if edge == loopfilter.EdgeVertical && x4 == 0 {
				continue
			}
			if edge == loopfilter.EdgeHorizontal && y4 == 0 {
				continue
			}
			cells[[2]int{x4, y4}] = lfDiffCell{width: e.Width, level: e.Level}
		}
	}
	return cells
}

// lfmaskLumaCells builds lfmask masks from the records in the given (decode)
// order and scans one luma edge direction into a per-4x4 cell map keyed by
// (x4, y4), dropping frame-boundary cells to match production.
func lfmaskLumaCells(t *testing.T, event Event, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatalf("grid: %v", err)
	}
	if cols > 32 || rows > 32 {
		t.Fatalf("test only covers a single 128x128 region, got %dx%d", cols, rows)
	}
	levelCtx := frameWorkLoopFilterLevelContextFor(&event)

	var builder lfmask.Builder
	var mask lfmask.FilterMask
	lc := lfmask.LevelCache{Cells: make([]uint8, 4*(cols*rows)), Stride: cols}
	aY := make([]uint8, cols)
	for i := range aY {
		aY[i] = 2
	}
	lY := make([]uint8, 32)
	for i := range lY {
		lY[i] = 2
	}

	for i := range records {
		rec := &records[i]
		levels, err := frameWorkResolveLoopFilterRecordLevels(&levelCtx, rec)
		if err != nil {
			t.Fatalf("resolve levels: %v", err)
		}
		lv := lfmask.Levels{
			YVert: levels[loopfilter.PlaneY][loopfilter.EdgeVertical],
			YHorz: levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal],
		}
		bx := int(rec.Block.MICol)
		by := int(rec.Block.MIRow)
		skip := 0
		if rec.SkipTransform {
			skip = 1
		}
		builder.CreateInter(&mask, lc, lv, bx, by, cols, rows, skip,
			rec.Block.Size, rec.TransformTree.Y, rec.TransformTree.Split, rec.TransformTree.UV,
			lfmask.Layout{SSHor: 1, SSVer: 1}, aY, lY, nil, nil)
	}

	cells := map[[2]int]lfDiffCell{}
	if edge == loopfilter.EdgeVertical {
		// dir 0 (columns): position = column (x4), scan bits = rows (y4).
		for col := 1; col < cols; col++ {
			lfmask.ScanLuma(&mask.Y[0][col], 0, rows, func(off, widthClass int) {
				L := lc.Cells[4*(off*cols+col)+int(loopfilter.EdgeVertical)]
				if L == 0 {
					L = lc.Cells[4*(off*cols+col-1)+int(loopfilter.EdgeVertical)]
				}
				if L == 0 {
					return
				}
				cells[[2]int{col, off}] = lfDiffCell{width: lumaWidthByClass[widthClass], level: L}
			})
		}
	} else {
		// dir 1 (rows): position = row (y4), scan bits = columns (x4).
		for row := 1; row < rows; row++ {
			lfmask.ScanLuma(&mask.Y[1][row], 0, cols, func(off, widthClass int) {
				L := lc.Cells[4*(row*cols+off)+int(loopfilter.EdgeHorizontal)]
				if L == 0 {
					L = lc.Cells[4*((row-1)*cols+off)+int(loopfilter.EdgeHorizontal)]
				}
				if L == 0 {
					return
				}
				cells[[2]int{off, row}] = lfDiffCell{width: lumaWidthByClass[widthClass], level: L}
			})
		}
	}
	return cells
}

func assertCellsEqual(t *testing.T, prod, got map[[2]int]lfDiffCell) {
	t.Helper()
	if len(prod) == len(got) {
		equal := true
		for k, v := range prod {
			if got[k] != v {
				equal = false
				break
			}
		}
		if equal {
			return
		}
	}
	keys := map[[2]int]struct{}{}
	for k := range prod {
		keys[k] = struct{}{}
	}
	for k := range got {
		keys[k] = struct{}{}
	}
	var sorted [][2]int
	for k := range keys {
		sorted = append(sorted, k)
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i][0] != sorted[j][0] {
			return sorted[i][0] < sorted[j][0]
		}
		return sorted[i][1] < sorted[j][1]
	})
	var diff string
	for _, k := range sorted {
		p, pok := prod[k]
		g, gok := got[k]
		if pok && gok && p == g {
			continue
		}
		diff += fmt.Sprintf("  cell(x4=%d,y4=%d): production=%+v(present=%v) lfmask=%+v(present=%v)\n",
			k[0], k[1], p, pok, g, gok)
	}
	t.Fatalf("edge mismatch (production vs lfmask):\n%s", diff)
}

func lfDiffEvent(size parser.FrameSize) Event {
	return Event{
		SequenceHeader: testSequence(),
		FrameSize:      size,
		LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 8}, LevelU: 8, LevelV: 8},
	}
}

// chromaWidthByClass maps a dav1d chroma strength class (!!log2dim) to the
// production chroma pixel width (frameWorkLoopFilterWidthByTX chroma column:
// 4 for the 4x4 class, 6 otherwise).
var chromaWidthByClass = [2]uint8{4, 6}

// productionChromaCells expands the production run-merged chroma edge list for
// one plane (U or V) and edge direction to a per-4x4 chroma cell map, dropping
// chroma frame-boundary cells.
func productionChromaCells(t *testing.T, ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, plane loopfilter.Plane, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatalf("scratch len: %v", err)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	cells := map[[2]int]lfDiffCell{}
	for _, e := range edges[:plan.StoredEdges] {
		if e.Plane != plane || e.Edge != edge {
			continue
		}
		for i := 0; i < int(e.Length4); i++ {
			x4, y4 := int(e.X4), int(e.Y4)
			if edge == loopfilter.EdgeVertical {
				y4 += i
			} else {
				x4 += i
			}
			if edge == loopfilter.EdgeVertical && x4 == 0 {
				continue
			}
			if edge == loopfilter.EdgeHorizontal && y4 == 0 {
				continue
			}
			cells[[2]int{x4, y4}] = lfDiffCell{width: e.Width, level: e.Level}
		}
	}
	return cells
}

// lfmaskChromaCells builds lfmask masks (with chroma) from records in decode
// order and scans one chroma plane/edge direction into a per-4x4 chroma cell map.
// Test coverage is 4:2:0 (ss_hor=ss_ver=1), so the mask lo/hi lane split is at
// 16>>1 == 8 bits in both directions.
func lfmaskChromaCells(t *testing.T, event Event, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge) map[[2]int]lfDiffCell {
	t.Helper()
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatalf("grid: %v", err)
	}
	if cols > 32 || rows > 32 {
		t.Fatalf("test only covers a single 128x128 region, got %dx%d", cols, rows)
	}
	levelCtx := frameWorkLoopFilterLevelContextFor(&event)

	var builder lfmask.Builder
	var mask lfmask.FilterMask
	lc := lfmask.LevelCache{Cells: make([]uint8, 4*(cols*rows)), Stride: cols}
	aY := make([]uint8, cols)
	auv := make([]uint8, cols)
	for i := range aY {
		aY[i] = 2
		auv[i] = 1
	}
	lY := make([]uint8, 32)
	luv := make([]uint8, 32)
	for i := range lY {
		lY[i] = 2
		luv[i] = 1
	}

	for i := range records {
		rec := &records[i]
		levels, err := frameWorkResolveLoopFilterRecordLevels(&levelCtx, rec)
		if err != nil {
			t.Fatalf("resolve levels: %v", err)
		}
		lv := lfmask.Levels{
			YVert: levels[loopfilter.PlaneY][loopfilter.EdgeVertical],
			YHorz: levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal],
			U:     levels[loopfilter.PlaneU][loopfilter.EdgeVertical],
			V:     levels[loopfilter.PlaneV][loopfilter.EdgeVertical],
		}
		bx := int(rec.Block.MICol)
		by := int(rec.Block.MIRow)
		skip := 0
		if rec.SkipTransform {
			skip = 1
		}
		builder.CreateInter(&mask, lc, lv, bx, by, cols, rows, skip,
			rec.Block.Size, rec.TransformTree.Y, rec.TransformTree.Split, rec.TransformTree.UV,
			lfmask.Layout{SSHor: 1, SSVer: 1}, aY, lY, auv, luv)
	}

	levelIdx := 2 // level cache index: 2 = U, 3 = V
	if plane == loopfilter.PlaneV {
		levelIdx = 3
	}
	ccols := (cols + 1) >> 1
	crows := (rows + 1) >> 1
	cells := map[[2]int]lfDiffCell{}
	const laneBits = 8 // 16 >> ss for 4:2:0
	if edge == loopfilter.EdgeVertical {
		for ccol := 1; ccol < ccols; ccol++ {
			lfmask.ScanChroma(&mask.UV[0][ccol], 0, crows, laneBits, func(off, widthClass int) {
				L := lc.Cells[4*(off*cols+ccol)+levelIdx]
				if L == 0 {
					L = lc.Cells[4*(off*cols+ccol-1)+levelIdx]
				}
				if L == 0 {
					return
				}
				cells[[2]int{ccol, off}] = lfDiffCell{width: chromaWidthByClass[widthClass], level: L}
			})
		}
	} else {
		for crow := 1; crow < crows; crow++ {
			lfmask.ScanChroma(&mask.UV[1][crow], 0, ccols, laneBits, func(off, widthClass int) {
				L := lc.Cells[4*(crow*cols+off)+levelIdx]
				if L == 0 {
					L = lc.Cells[4*((crow-1)*cols+off)+levelIdx]
				}
				if L == 0 {
					return
				}
				cells[[2]int{off, crow}] = lfDiffCell{width: chromaWidthByClass[widthClass], level: L}
			})
		}
	}
	return cells
}

func runChromaDiff(t *testing.T, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge) {
	t.Helper()
	event := lfDiffEvent(size)
	ctx := FrameWorkPostFilterContext{Event: event}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)
	prod := productionChromaCells(t, ctx, filterMap, plane, edge)
	got := lfmaskChromaCells(t, event, size, records, plane, edge)
	if len(prod) == 0 {
		t.Fatalf("production produced no chroma edges for plane=%d edge=%d; test setup is degenerate", plane, edge)
	}
	assertCellsEqual(t, prod, got)
}

// runLumaDiff builds the production and lfmask per-4x4 luma cell maps for one
// edge direction and asserts they are identical.
func runLumaDiff(t *testing.T, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge) {
	t.Helper()
	runLumaDiffWithEvent(t, lfDiffEvent(size), size, records, edge)
}

func runLumaDiffWithEvent(t *testing.T, event Event, size parser.FrameSize, records []threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge) {
	t.Helper()
	ctx := FrameWorkPostFilterContext{Event: event}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)
	prod := productionLumaCells(t, ctx, filterMap, edge)
	got := lfmaskLumaCells(t, event, size, records, edge)
	if len(prod) == 0 {
		t.Fatalf("production produced no luma edges for edge=%d; test setup is degenerate", edge)
	}
	assertCellsEqual(t, prod, got)
}

func TestLFMaskDiffLumaVerticalLevelFromPrevious(t *testing.T) {
	// Right block resolves to loop-filter level 0 (segment delta), left block to 8.
	// The shared vertical boundary edge must fall back to the previous (left) side
	// for its level (AV1 "level from previous"), exactly as production does. The
	// mask geometry is level-independent; the fallback happens during the scan.
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	event := lfDiffEvent(size)
	event.Segmentation.Enabled = true
	event.Segmentation.Data.Segments[1].DeltaLFYV = -63 // drives level_y_v to 0
	event.Segmentation.Data.Segments[1].DeltaLFYH = -63

	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SegmentID = 0
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.SegmentID = 1
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runLumaDiffWithEvent(t, event, size, recs, loopfilter.EdgeVertical)
}

func TestLFMaskDiffLumaSingleSplitBlock(t *testing.T) {
	// One 16x16 inter block filling a 16x16 frame, transform tree split to four
	// 8x8 sub-transforms. Only inner tx edges exist (block edges are the frame
	// boundary at x4=0 / y4=0, dropped on both sides).
	size := parser.FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 16, SuperResDenominator: 8}
	rec := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	rec.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	recs := []threading.FrameWorkLoopFilterBlockRecord{rec}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffLumaTwoBlocksDecodeOrder(t *testing.T) {
	// Two horizontally adjacent inter blocks. The left block's right-edge tx must
	// flow through the carried l[] context so block2's left-edge strength resolves
	// as imin(block2 tx, block1 tx) exactly as the production neighbour lookup does.
	// Records are supplied in decode order (left then right).
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SkipTransform = true
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffLumaTwoBlocksUnequalTx(t *testing.T) {
	// Left block uses a small (8x8) fixed transform, right block a large (16x16)
	// one, so the shared boundary strength is genuinely min(prev,cur) rather than
	// a single class. Exercises the l[] carry across differing transform sizes.
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 16, SuperResDenominator: 8}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffLumaVerticalStackedBlocks(t *testing.T) {
	// Two vertically stacked inter blocks with differing transforms. The upper
	// block's bottom-edge tx must flow through the carried a[] (above) context so
	// the lower block's top horizontal edge resolves as min(prev,cur). Records in
	// decode order (top then bottom).
	size := parser.FrameSize{CodedWidth: 16, UpscaledWidth: 16, Height: 32, SuperResDenominator: 8}
	top := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	top.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true, Variable: true, Split: [2]uint16{1, 0}}
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	bottom.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{top, bottom}
	runLumaDiff(t, size, recs, loopfilter.EdgeVertical)
	runLumaDiff(t, size, recs, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffChromaSingleBlock(t *testing.T) {
	// One 32x32 inter block filling a 32x32 frame with an 8x8 chroma transform, so
	// the 16x16 chroma area (4:2:0) tiles into a 2x2 grid of 8x8 chroma transforms
	// with interior chroma edges (the block's own boundary edges are at the frame
	// boundary and dropped on both sides).
	size := parser.FrameSize{CodedWidth: 32, UpscaledWidth: 32, Height: 32, SuperResDenominator: 8}
	rec := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 8, 8)
	rec.Block.Size = tile.BlockSize32x32
	rec.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize8x8, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{rec}
	runChromaDiff(t, size, recs, loopfilter.PlaneU, loopfilter.EdgeVertical)
	runChromaDiff(t, size, recs, loopfilter.PlaneU, loopfilter.EdgeHorizontal)
	runChromaDiff(t, size, recs, loopfilter.PlaneV, loopfilter.EdgeVertical)
	runChromaDiff(t, size, recs, loopfilter.PlaneV, loopfilter.EdgeHorizontal)
}

func TestLFMaskDiffChromaTwoBlocksDecodeOrder(t *testing.T) {
	// Two horizontally adjacent 32x32 inter blocks; the shared chroma boundary
	// resolves through the carried chroma left context. 4:2:0.
	size := parser.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 32, SuperResDenominator: 8}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 8, 8)
	left.Block.Size = tile.BlockSize32x32
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	right := testFrameWorkLoopFilterPostFilterRecordAt(8, 0, 16, 8)
	right.Block.Size = tile.BlockSize32x32
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize32x32, UV: tile.TransformSize16x16, HasUV: true}
	recs := []threading.FrameWorkLoopFilterBlockRecord{left, right}
	runChromaDiff(t, size, recs, loopfilter.PlaneU, loopfilter.EdgeVertical)
	runChromaDiff(t, size, recs, loopfilter.PlaneV, loopfilter.EdgeVertical)
}

func TestFrameWorkStateCompactLoopFilterLifecycle(t *testing.T) {
	t.Run("reset deactivates owned state", func(t *testing.T) {
		state, compact, masks, cdef, buffers := compactFrameWorkStateFixture(t)
		if err := SetOwnedFrameWorkLoopFilterSideData(state, cdef, masks, compact, buffers); err != nil {
			t.Fatal(err)
		}
		state.Reset()
		if threading.FrameWorkLoopFilterCompactStateActive(compact) {
			t.Fatal("Reset left the owned compact state active")
		}
		if state.loopFilterCompactState != nil {
			t.Fatal("Reset retained the compact-state link")
		}
	})

	t.Run("rejected public rebind preserves owned state", func(t *testing.T) {
		state, compact, masks, cdef, buffers := compactFrameWorkStateFixture(t)
		if err := SetOwnedFrameWorkLoopFilterSideData(state, cdef, masks, compact, buffers); err != nil {
			t.Fatal(err)
		}
		if err := state.SetSideData(threading.FrameWorkCDEFIndexMap{}, threading.FrameWorkLoopFilterMap{}, buffers); err == nil {
			t.Fatal("SetSideData accepted invalid public views")
		}
		if !threading.FrameWorkLoopFilterCompactStateActive(compact) || state.loopFilterCompactState != compact {
			t.Fatal("rejected SetSideData invalidated the previous owned binding")
		}
	})

	t.Run("successful generic rebind deactivates owned state", func(t *testing.T) {
		state, compact, masks, cdef, buffers := compactFrameWorkStateFixture(t)
		if err := SetOwnedFrameWorkLoopFilterSideData(state, cdef, masks, compact, buffers); err != nil {
			t.Fatal(err)
		}
		lfMap := threading.FrameWorkLoopFilterMap{Records: make([]threading.FrameWorkLoopFilterBlockRecord, 16), Stride: 4, Rows: 4}
		if err := state.SetLoopFilterMap(lfMap); err != nil {
			t.Fatal(err)
		}
		if threading.FrameWorkLoopFilterCompactStateActive(compact) || state.loopFilterCompactState != nil {
			t.Fatal("SetLoopFilterMap retained the owned compact binding")
		}
		if !state.loopFilterMapValid || state.loopFilterMap.Records == nil {
			t.Fatal("SetLoopFilterMap did not install the public map")
		}
	})
}

func compactFrameWorkStateFixture(t *testing.T) (*FrameWorkState, *threading.FrameWorkLoopFilterCompactState, threading.FrameWorkLoopFilterMasks, threading.FrameWorkCDEFIndexMap, threading.FrameWorkRestorationFrameBuffers) {
	t.Helper()
	state := &FrameWorkState{active: true}
	masks := threading.FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, 1),
		LevelCache: make([]uint8, 4*(16)),
		Cols:       4,
		Rows:       4,
		SB128W:     1,
		SB128H:     1,
		Layout:     lfmask.Layout{Mono: true},
	}
	compact := &threading.FrameWorkLoopFilterCompactState{}
	if err := threading.BindFrameWorkLoopFilterCompactState(&masks, compact, make([]byte, 8), parser.LoopFilterParams{}, parser.SegmentationParams{}, parser.DeltaParams{}, true); err != nil {
		t.Fatal(err)
	}
	cdef := threading.FrameWorkCDEFIndexMap{Index: make([]uint8, 1), Read: make([]bool, 1), Stride: 1, Rows: 1}
	buffers := threading.FrameWorkRestorationFrameBuffers{Plan: tile.RestorationFramePlan{Planes: 1}}
	return state, compact, masks, cdef, buffers
}

func TestCDEFPrimaryDirectionGateUsesRawStrengths(t *testing.T) {
	for _, tc := range []struct {
		name       string
		yStrength  uint8
		uvStrength uint8
		plane      int
		forceLuma  bool
		want       bool
	}{
		{name: "luma secondary only", yStrength: 2, forceLuma: true, want: false},
		{name: "chroma primary needs luma", uvStrength: 3 << 2, forceLuma: true, want: true},
		{name: "uv secondary only", uvStrength: 2, forceLuma: true, want: false},
		{name: "luma primary", yStrength: 1 << 2, forceLuma: true, want: true},
		{name: "luma primary when variance-adjusted strength may be zero", yStrength: 1 << 2, want: true},
		{name: "monochrome ignores uv primary", uvStrength: 1 << 2, want: false},
		{name: "u chroma primary", uvStrength: 1 << 2, plane: 1, want: true},
		{name: "u chroma secondary only", uvStrength: 2, plane: 1, want: false},
		{name: "v chroma primary", uvStrength: 1 << 2, plane: 2, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			params := parser.CDEFParams{
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{tc.yStrength},
				UVStrength:    [parser.MaxCDEFStrengths]uint8{tc.uvStrength},
			}
			if got := frameWorkCDEFPlaneNeedsPrimaryDirection(params, 0, tc.plane, tc.forceLuma); got != tc.want {
				t.Fatalf("needs direction=%v want %v (Y=%d UV=%d plane=%d force=%v)", got, tc.want, tc.yStrength, tc.uvStrength, tc.plane, tc.forceLuma)
			}
		})
	}
}

func TestCDEFHBDInPlaceMatchesSnapshot(t *testing.T) {
	tests := []struct {
		name             string
		width, height    int
		bitDepth         uint8
		subsamplingX     bool
		subsamplingY     bool
		mono             bool
		secondaryOnly    bool
		unalignedY       bool
		oddSampleStrideY bool
	}{
		{name: "10bit_420_odd_crop", width: 137, height: 131, bitDepth: 10, subsamplingX: true, subsamplingY: true},
		{name: "12bit_422_odd_crop", width: 137, height: 131, bitDepth: 12, subsamplingX: true},
		{name: "10bit_444_partial_units", width: 73, height: 69, bitDepth: 10},
		{name: "12bit_monochrome", width: 137, height: 131, bitDepth: 12, mono: true},
		{name: "10bit_422_secondary_only", width: 137, height: 131, bitDepth: 10, subsamplingX: true, secondaryOnly: true},
		{name: "12bit_444_secondary_only", width: 137, height: 131, bitDepth: 12, secondaryOnly: true},
		{name: "12bit_monochrome_secondary_only", width: 137, height: 131, bitDepth: 12, mono: true, secondaryOnly: true},
		{name: "10bit_420_unaligned_luma", width: 137, height: 131, bitDepth: 10, subsamplingX: true, subsamplingY: true, unalignedY: true},
		{name: "10bit_420_odd_sample_stride", width: 137, height: 131, bitDepth: 10, subsamplingX: true, subsamplingY: true, oddSampleStrideY: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCDEFHBDInPlaceDifferential(t, tt.width, tt.height, tt.bitDepth, tt.subsamplingX, tt.subsamplingY, tt.mono, tt.secondaryOnly, tt.unalignedY, tt.oddSampleStrideY)
		})
	}
}

func runCDEFHBDInPlaceDifferential(t *testing.T, width, height int, bitDepth uint8, subsamplingX, subsamplingY, mono, secondaryOnly, unalignedY, oddSampleStrideY bool) {
	t.Helper()
	seq := testSequence()
	seq.EnableCDEF = true
	strengths := parser.CDEFParams{
		Damping: 5, StrengthCount: 2,
		YStrength:  [parser.MaxCDEFStrengths]uint8{9<<2 | 1, 15<<2 | 3},
		UVStrength: [parser.MaxCDEFStrengths]uint8{5<<2 | 2, 11<<2 | 1},
	}
	if secondaryOnly {
		strengths.YStrength = [parser.MaxCDEFStrengths]uint8{3, 1}
		strengths.UVStrength = [parser.MaxCDEFStrengths]uint8{2, 3}
	}
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth: uint32(width), UpscaledWidth: uint32(width), Height: uint32(height),
			SuperResDenominator: 8,
		},
		CDEF: strengths,
	}
	format := frame.Format{
		Width: width, Height: height, BitDepth: bitDepth,
		SubsamplingX: subsamplingX, SubsamplingY: subsamplingY,
		MonoChrome: mono, Align: 64,
	}
	build := func() (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		out := testFrameWorkCDEFFrame(t, format)
		fillCDEFHBDPlaneAllocation(out.Y, int(bitDepth), 1)
		if !mono {
			fillCDEFHBDPlaneAllocation(out.U, int(bitDepth), 2)
			fillCDEFHBDPlaneAllocation(out.V, int(bitDepth), 3)
		}
		if unalignedY {
			out.Y = cdefHBDPlaneWithUnalignedPix(out.Y)
		}
		if oddSampleStrideY {
			out.Y = cdefHBDPlaneWithOddSampleStride(out.Y)
		}
		if unalignedY && !frameWorkCDEFHBDPlaneViewAvailable(out.Y) {
			t.Fatal("byte-addressed in-place walk should accept unaligned luma")
		}
		if oddSampleStrideY {
			if out.Y.Stride/2%2 == 0 {
				t.Fatalf("luma stride has an even sample count: byte stride=%d", out.Y.Stride)
			}
			if !frameWorkCDEFHBDPlaneViewAvailable(out.Y) {
				t.Fatal("odd sample stride should remain eligible for the in-place uint16 view")
			}
		}
		ctx := (FrameWorkPostFilterContext{Event: event, Output: out}).WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
		for i := range req.IndexMap.Index {
			req.IndexMap.Index[i] = uint8(i % 2)
			req.IndexMap.Read[i] = i%9 != 4
		}
		if secondaryOnly {
			poisonCDEFHBDGrids(req)
		}
		for i := range req.SampleScratch {
			for j := range req.SampleScratch[i] {
				req.SampleScratch[i][j] = 0x6bad
			}
		}
		return out, ctx, req
	}

	streamOut, streamCtx, streamReq := build()
	snapshotOut, snapshotCtx, snapshotReq := build()
	initialPlanes := [3][]byte{
		append([]byte(nil), streamOut.Y.Pix...),
		append([]byte(nil), streamOut.U.Pix...),
		append([]byte(nil), streamOut.V.Pix...),
	}
	var skipMap threading.FrameWorkLoopFilterMap
	miCols := (width + 3) / 4
	miRows := (height + 3) / 4
	skipMap.Stride = uint16(miCols)
	skipMap.Rows = uint16(miRows)
	skipMap.Records = make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows)
	for i := range skipMap.Records {
		skipMap.Records[i].Valid = true
		skipMap.Records[i].SkipTransform = i%13 == 5 || i%17 == 3
	}
	streamCtx.LoopFilterMap = &skipMap
	snapshotCtx.LoopFilterMap = &skipMap

	applyStream := func() (FrameWorkCDEFPostFilterResult, error) {
		copy(streamOut.Y.Pix, initialPlanes[0])
		copy(streamOut.U.Pix, initialPlanes[1])
		copy(streamOut.V.Pix, initialPlanes[2])
		return streamCtx.ApplyCDEFPostFilter(streamReq)
	}
	streamResult, err := applyStream()
	if err != nil {
		t.Fatalf("in-place ApplyCDEFPostFilter: %v", err)
	}
	if streamResult.Blocks == 0 || streamResult.Units == 0 {
		t.Fatalf("in-place path did no filtering: %+v", streamResult)
	}
	allocs := testing.AllocsPerRun(5, func() {
		result, err := applyStream()
		if err != nil || result != streamResult {
			t.Fatalf("warmed in-place apply result=%+v err=%v, want %+v", result, err, streamResult)
		}
	})
	if allocs != 0 {
		t.Fatalf("warmed HBD in-place apply allocated %g times", allocs)
	}
	if err := snapshotCtx.LoadCDEFPostFilterSamples(snapshotReq); err != nil {
		t.Fatalf("snapshot load: %v", err)
	}
	_, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	var snapshotResult FrameWorkCDEFPostFilterResult
	for row := 0; row < rows; row++ {
		bandResult, err := snapshotCtx.ApplyCDEFPostFilterUnitRows(snapshotReq, row, row+1)
		if err != nil {
			t.Fatalf("snapshot row %d: %v", row, err)
		}
		snapshotResult.Units += bandResult.Units
		snapshotResult.Blocks += bandResult.Blocks
		if bandResult.Planes > snapshotResult.Planes {
			snapshotResult.Planes = bandResult.Planes
		}
	}
	if streamResult != snapshotResult {
		t.Fatalf("in-place counts=%+v snapshot counts=%+v", streamResult, snapshotResult)
	}
	if secondaryOnly {
		assertCDEFHBDGridsRemainPoisoned(t, streamReq)
		assertCDEFHBDGridsRemainPoisoned(t, snapshotReq)
	}
	for plane, pair := range [][2]frame.Plane{{streamOut.Y, snapshotOut.Y}, {streamOut.U, snapshotOut.U}, {streamOut.V, snapshotOut.V}} {
		if !bytes.Equal(pair[0].Pix, pair[1].Pix) {
			i := firstCDEFHBDByteDiff(pair[0].Pix, pair[1].Pix)
			if i >= len(pair[0].Pix) || i >= len(pair[1].Pix) {
				t.Fatalf("plane %d byte lengths differ: in-place=%d snapshot=%d", plane, len(pair[0].Pix), len(pair[1].Pix))
			}
			t.Fatalf("plane %d differs at byte %d: in-place=%d snapshot=%d", plane, i, pair[0].Pix[i], pair[1].Pix[i])
		}
	}
	if len(streamReq.DirectionGrid) != len(snapshotReq.DirectionGrid) ||
		len(streamReq.VarianceGrid) != len(snapshotReq.VarianceGrid) {
		t.Fatalf("grid lengths differ: stream=%d/%d snapshot=%d/%d", len(streamReq.DirectionGrid), len(streamReq.VarianceGrid), len(snapshotReq.DirectionGrid), len(snapshotReq.VarianceGrid))
	}
	for i := range streamReq.DirectionGrid {
		if streamReq.DirectionGrid[i] != snapshotReq.DirectionGrid[i] || streamReq.VarianceGrid[i] != snapshotReq.VarianceGrid[i] {
			t.Fatalf("direction/variance grids differ at unit %d", i)
		}
	}
}

func cdefHBDPlaneWithUnalignedPix(plane frame.Plane) frame.Plane {
	backing := make([]byte, len(plane.Pix)+1)
	copy(backing[1:], plane.Pix)
	plane.Pix = backing[1:]
	return plane
}

func cdefHBDPlaneWithOddSampleStride(plane frame.Plane) frame.Plane {
	oldStride := plane.Stride
	strideSamples := oldStride / 2
	if strideSamples&1 == 0 {
		strideSamples++
	}
	plane.Stride = strideSamples * 2
	allocatedRows := len(plane.Pix) / oldStride
	tailBytes := len(plane.Pix) - allocatedRows*oldStride
	pix := make([]byte, allocatedRows*plane.Stride+tailBytes)
	rowBytes := minInt(oldStride, plane.Stride)
	for y := 0; y < allocatedRows; y++ {
		oldStart := y * oldStride
		newStart := y * plane.Stride
		copy(pix[newStart:newStart+rowBytes], plane.Pix[oldStart:oldStart+rowBytes])
	}
	copy(pix[allocatedRows*plane.Stride:], plane.Pix[allocatedRows*oldStride:])
	plane.Pix = pix
	return plane
}

const cdefHBDDirectionPoison uint8 = 0xa7
const cdefHBDVariancePoison int32 = 0x1234567

func poisonCDEFHBDGrids(req FrameWorkCDEFPostFilterRequest) {
	for i := range req.DirectionGrid {
		for by := range req.DirectionGrid[i] {
			for bx := range req.DirectionGrid[i][by] {
				req.DirectionGrid[i][by][bx] = cdefHBDDirectionPoison
				req.VarianceGrid[i][by][bx] = cdefHBDVariancePoison
			}
		}
	}
}

func assertCDEFHBDGridsRemainPoisoned(t *testing.T, req FrameWorkCDEFPostFilterRequest) {
	t.Helper()
	for i := range req.DirectionGrid {
		for by := range req.DirectionGrid[i] {
			for bx := range req.DirectionGrid[i][by] {
				if req.DirectionGrid[i][by][bx] != cdefHBDDirectionPoison || req.VarianceGrid[i][by][bx] != cdefHBDVariancePoison {
					t.Fatalf("secondary-only CDEF changed grid unit=%d block=(%d,%d): dir=%d variance=%d", i, by, bx, req.DirectionGrid[i][by][bx], req.VarianceGrid[i][by][bx])
				}
			}
		}
	}
}

func fillCDEFHBDPlaneAllocation(plane frame.Plane, bitDepth, seed int) {
	stride := plane.Stride / 2
	maxSample := (1 << bitDepth) - 1
	for y := 0; y < len(plane.Pix)/plane.Stride; y++ {
		for x := 0; x < stride; x++ {
			value := (x*37 + y*53 + (x^y)*17 + seed*101) & maxSample
			binary.LittleEndian.PutUint16(plane.Pix[y*plane.Stride+x*2:], uint16(value))
		}
	}
}

func firstCDEFHBDByteDiff(a, b []byte) int {
	for i := range minInt(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	return minInt(len(a), len(b))
}

// Frozen original complete unit staging walk from 5484d71c. Assembly and
// filter helpers used here retain their original implementation.
func frameWorkApplyCDEFPlaneRowsU8FrozenSparse(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols int, rows int, rowStart int, rowEnd int, dst frame.Plane, byteScratch []byte, boundary FrameWorkCDEFPostFilterU8BandBoundary, input []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane int, xDec int, yDec int, forceLumaDirections bool, unitMasks *frameWorkCDEFUnitMaskCache) (uint32, uint32, error) {
	var units uint32
	var blocksTotal uint32
	width := dst.Width
	height := dst.Height
	stride := dst.Stride
	if width <= 0 || height <= 0 || stride < width ||
		(height-1)*stride+width > len(dst.Pix) {
		return 0, 0, frame.ErrInvalidPlane
	}
	if rowEnd < 0 || rowEnd > rows {
		rowEnd = rows
	}
	if rowStart < 0 || rowStart >= rowEnd {
		if rowStart == rowEnd {
			return 0, 0, nil
		}
		return 0, 0, threading.ErrInvalidBatch
	}
	unitSizeX := cdef.BlockSize >> xDec
	unitSizeY := cdef.BlockSize >> yDec
	blockWidth := 8 >> xDec
	blockHeight := 8 >> yDec
	indexStride := int(indexMap.Stride)

	// Pre-filter backups (dav1d cdef_apply_tmpl.c): two ping-pong line
	// buffers of 2 rows x plane width (backup2lines / cdef_line[tf]) carved
	// from the sample scratch arena, and an 8-column x unit-height left
	// backup (backup2x8 / lr_bak) on the stack.
	if len(byteScratch) < CDEFU8LineScratchLen(width) {
		return 0, 0, threading.ErrInvalidBatch
	}
	lineBuf := byteScratch[:4*width]
	if rowStart > 0 {
		top := boundary.Top[plane]
		need := cdef.VerticalBorder * width
		if len(top) < need {
			return 0, 0, frame.ErrShortBuffer
		}
		topSlot := ((rowStart ^ 1) & 1) * cdef.VerticalBorder * width
		copy(lineBuf[topSlot:topSlot+need], top[:need])
	}
	var bottomBoundary []byte
	if rowEnd < rows {
		bottomBoundary = boundary.Bottom[plane]
		if len(bottomBoundary) < cdef.VerticalBorder*width {
			return 0, 0, frame.ErrShortBuffer
		}
	}
	var leftStorage [cdef.HorizontalBorder * cdef.BlockSize]byte

	for unitRow := rowStart; unitRow < rowEnd; unitRow++ {
		unitY := (unitRow * cdef.BlockSize) >> yDec
		// Back up the 2 pre-filter rows the NEXT unit row reads as its top
		// halo before any unit in this row overwrites them.
		if unitRow+1 < rowEnd {
			nextTopY := unitY + unitSizeY
			save := lineBuf[(unitRow&1)*2*width:]
			for r := 0; r < cdef.VerticalBorder; r++ {
				y := nextTopY - cdef.VerticalBorder + r
				if y < 0 || y >= height {
					continue
				}
				copy(save[r*width:r*width+width], dst.Pix[y*stride:y*stride+width])
			}
		}
		topBuf := lineBuf[((unitRow^1)&1)*2*width:]
		bottomBuf := []byte(nil)
		if unitRow+1 == rowEnd {
			bottomBuf = bottomBoundary
		}

		prevFiltered := false
		for unitCol := range cols {
			mapOffset := unitRow*indexStride + unitCol
			if !indexMap.Read[mapOffset] {
				prevFiltered = false
				continue
			}
			strengthIndex := indexMap.Index[mapOffset]
			if int(strengthIndex) >= int(params.StrengthCount) || int(strengthIndex) >= parser.MaxCDEFStrengths {
				return units, blocksTotal, threading.ErrInvalidBatch
			}
			index := int(strengthIndex)
			packed := params.YStrength[index]
			if plane != 0 {
				packed = params.UVStrength[index]
			}
			needPrimaryDirection := frameWorkCDEFPlaneNeedsPrimaryDirection(params, index, plane, forceLumaDirections)
			directionOnly := plane == 0 && forceLumaDirections && packed == 0 && params.UVStrength[index]>>2 != 0
			if packed == 0 && !directionOnly {
				prevFiltered = false
				continue
			}
			unitX := (unitCol * cdef.BlockSize) >> xDec
			unitW := minInt(unitSizeX, width-unitX)
			unitH := minInt(unitSizeY, height-unitY)
			if unitW <= 0 || unitH <= 0 {
				prevFiltered = false
				continue
			}
			var blocks []cdef.BlockPosition
			if unitMasks != nil {
				blocks = frameWorkCDEFBlockPositionsFromMask(blockStorage, unitMasks.mask(unitRow, unitCol), unitW, unitH, blockWidth, blockHeight)
			} else {
				blocks = frameWorkCDEFBlockPositionsFiltered(blockStorage, unitW, unitH, blockWidth, blockHeight, skipMap, unitRow, unitCol)
			}
			if len(blocks) == 0 {
				prevFiltered = false
				continue
			}
			unitIndex := unitRow*cols + unitCol
			unitDirections := directions
			unitVariances := variances
			if unitIndex < len(directionGrid) && unitIndex < len(varianceGrid) {
				unitDirections = &directionGrid[unitIndex]
				unitVariances = &varianceGrid[unitIndex]
			}
			cdefPlane := cdef.PlaneY
			if plane == 1 {
				cdefPlane = cdef.PlaneU
			} else if plane == 2 {
				cdefPlane = cdef.PlaneV
			}
			filterParams, err := cdef.FrameFilterParamsFromStrength(cdefPlane, xDec, yDec, packed, int(params.Damping), 0)
			if err != nil {
				return units, blocksTotal, err
			}
			unitOrigin := unitY*stride + unitX
			if directionOnly {
				// Directions come from the uint8 frame; no tap buffer needed.
				if err := cdef.FilterFrameBlocksU8Trusted(dst.Pix[unitOrigin:], stride, input[:0], 0, blocks, unitDirections, unitVariances, filterParams); err != nil {
					return units, blocksTotal, err
				}
				prevFiltered = false
				continue
			}
			if frameWorkCDEFCanUseSecondaryByteUnit(cdef.U8SecondaryOnlyByteInteriorAvailable(), needPrimaryDirection, filterParams, unitX, unitY, unitW, unitH, width, height, blockWidth, blockHeight, cdefDebugUnit(plane, unitRow, unitCol)) {
				byteInput, byteOK := byteScratch[4*width:4*width+cdefU8TapBytes], true
				if byteOK && frameWorkAssembleCDEFSecondaryByteInputU8(byteInput, dst, topBuf, bottomBuf, leftStorage[:], prevFiltered, unitX, unitY, unitW, unitH) == nil {
					// Preserve the previous left halo until the byte filter accepts
					// the unit. Its entry validates the complete read window before
					// writing, so a decline can safely fall through to the wide path.
					oldLeftStorage := leftStorage
					nextUnitX := unitX + unitSizeX
					if nextUnitX < width {
						for r := 0; r < unitH; r++ {
							rowOff := (unitY+r)*stride + nextUnitX - cdef.HorizontalBorder
							copy(leftStorage[r*cdef.HorizontalBorder:(r+1)*cdef.HorizontalBorder], dst.Pix[rowOff:rowOff+cdef.HorizontalBorder])
						}
					}
					if cdef.FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(dst.Pix[unitOrigin:], stride, byteInput, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, unitW, unitH, blocks, filterParams) {
						prevFiltered = true
						units++
						blocksTotal += uint32(len(blocks))
						continue
					}
					leftStorage = oldLeftStorage
				}
			}
			if err := frameWorkAssembleCDEFInputU8(input, dst, topBuf, bottomBuf, leftStorage[:], prevFiltered, unitRow, unitX, unitY, unitW, unitH); err != nil {
				return units, blocksTotal, err
			}
			// Back up the pre-filter columns the NEXT unit reads as its left
			// halo (dav1d backup2x8 / lr_bak). This must happen after this
			// unit's tap assembly consumed the PREVIOUS backup (dav1d
			// ping-pongs lr_bak[bit] for the same reason) and before the
			// kernels overwrite the frame.
			nextUnitX := unitX + unitSizeX
			if nextUnitX < width {
				for r := 0; r < unitH; r++ {
					rowOff := (unitY+r)*stride + nextUnitX - cdef.HorizontalBorder
					copy(leftStorage[r*cdef.HorizontalBorder:(r+1)*cdef.HorizontalBorder], dst.Pix[rowOff:rowOff+cdef.HorizontalBorder])
				}
			}
			if cdefDebugUnit(plane, unitRow, unitCol) {
				cdefDebugLogUnit(plane, unitRow, unitCol, packed, filterParams, *unitDirections, *unitVariances, input, len(blocks))
			}
			var filterErr error
			if needPrimaryDirection {
				filterErr = cdef.FilterFrameBlocksU8Trusted(dst.Pix[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
			} else {
				filterErr = cdef.FilterFrameBlocksU8TrustedNoPrimaryDirection(dst.Pix[unitOrigin:], stride, input, cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, blocks, unitDirections, unitVariances, filterParams)
			}
			if filterErr != nil {
				return units, blocksTotal, filterErr
			}
			prevFiltered = true
			units++
			blocksTotal += uint32(len(blocks))
		}
	}
	return units, blocksTotal, nil
}

func TestCDEFSparseStageRectFrozen(t *testing.T) {
	cases := 0
	for _, shape := range [][2]int{{8, 8}, {4, 4}, {4, 8}} {
		bw, bh := shape[0], shape[1]
		for _, dims := range [][2]int{{3 * 8 * bw, 3 * 8 * bh}, {17 * bw, 17 * bh}, {16*bw + 1, 16*bh + 1}, {8 * bw, 8 * bh}} {
			w, h := dims[0], dims[1]
			stride := w + 13
			pix := make([]byte, stride*h)
			for i := range pix {
				pix[i] = byte((i*79 ^ (i >> 2)) & 255)
			}
			plane := frame.Plane{Pix: pix, Width: w, Height: h, Stride: stride}
			for uy := 0; uy < h; uy += 8 * bh {
				for ux := 0; ux < w; ux += 8 * bw {
					uw, uh := min(8*bw, w-ux), min(8*bh, h-uy)
					cols, rows := uw/bw, uh/bh
					for _, span := range [][4]int{{0, 0, 1, 1}, {cols - 1, rows - 1, cols, rows}, {0, 0, cols, 1}, {0, rows - 1, cols, rows}, {cols / 2, rows / 2, cols, rows}, {0, 0, cols, rows}} {
						blocks := []cdef.BlockPosition{{BX: uint8(span[0]), BY: uint8(span[1])}, {BX: uint8(span[2] - 1), BY: uint8(span[3] - 1)}}
						rect := frameWorkCDEFSparseStageRect(blocks, uw, uh, bw, bh)
						if !rect.active {
							continue
						}
						for mode := range 4 {
							top, bottom, left := make([]byte, 2*w), make([]byte, 2*w), make([]byte, cdef.HorizontalBorder*uh)
							for i := range top {
								top[i] = byte(i*13 + 31)
							}
							for i := range bottom {
								bottom[i] = byte(i*7 + 199)
							}
							for i := range left {
								left[i] = byte(i*29 + 53)
							}
							if mode&1 == 0 {
								bottom = nil
							}
							leftBackup := mode&2 != 0
							want := make([]uint16, cdef.InputBufferSize)
							storage := make([]uint16, cdef.InputBufferSize+2)
							for i := range storage {
								storage[i] = 0xaffe
							}
							got := storage[1 : len(storage)-1]
							if err := frameWorkAssembleCDEFInputU8(want, plane, top, bottom, left, leftBackup, uy/(8*bh), ux, uy, uw, uh); err != nil {
								t.Fatal(err)
							}
							if !frameWorkAssembleCDEFSparseInputU8(got, plane, top, bottom, left, leftBackup, uy/(8*bh), ux, uy, uw, uh, rect) {
								t.Fatal("valid sparse wide stage declined")
							}
							assertCDEFSparseRegion(t, got, want, rect, 8)
							if storage[0] != 0xaffe || storage[len(storage)-1] != 0xaffe {
								t.Fatal("wide canary")
							}
							if ux >= 2 && uy >= 2 && ux+uw+2 <= w && uy+uh+2 <= h {
								wb, gb := make([]byte, cdef.InputBufferSize), make([]byte, cdef.InputBufferSize)
								for i := range gb {
									gb[i] = 0xe7
								}
								if err := frameWorkAssembleCDEFSecondaryByteInputU8(wb, plane, top, bottom, left, leftBackup, ux, uy, uw, uh); err != nil {
									t.Fatal(err)
								}
								if !frameWorkAssembleCDEFSparseByteInputU8(gb, plane, top, bottom, left, leftBackup, ux, uy, uw, uh, rect) {
									t.Fatal("valid sparse byte stage declined")
								}
								for i := range gb {
									x, y := i%cdef.BStride-cdef.HorizontalBorder, i/cdef.BStride-cdef.VerticalBorder
									inside := x >= int(rect.x0)-2 && x < int(rect.x1)+2 && y >= int(rect.y0)-2 && y < int(rect.y1)+2
									if inside && gb[i] != wb[i] || !inside && gb[i] != 0xe7 {
										t.Fatalf("byte bounds/copy mismatch at %d", i)
									}
								}
							}
							cases++
						}
					}
				}
			}
		}
	}
	t.Logf("full-stage oracle and untouched-region matches: %d", cases)
}

func assertCDEFSparseRegion(t *testing.T, got, want []uint16, rect frameWorkCDEFStageRect, halo int) {
	t.Helper()
	for i := range got {
		x, y := i%cdef.BStride-cdef.HorizontalBorder, i/cdef.BStride-cdef.VerticalBorder
		inside := x >= int(rect.x0)-halo && x < int(rect.x1)+halo && y >= int(rect.y0)-2 && y < int(rect.y1)+2
		if inside && got[i] != want[i] || !inside && got[i] != 0xaffe {
			t.Fatalf("staging/canary at %d x=%d y=%d: got=%x want=%x inside=%t rect=%+v", i, x, y, got[i], want[i], inside, rect)
		}
	}
}

func TestCDEFSparseFullPlaneSnapshot(t *testing.T) {
	patterns := []uint64{1, 1 << 63, 0x00003c3c3c3c0000, 0x00000000ffffffff, 0x8040201008040201, 0x0101010101010101, 0x8080808080808080, 0xffffffffffffffff, 0}
	for _, shape := range [][2]int{{1, 1}, {1, 0}, {0, 0}} {
		for _, strength := range [][2]uint8{{2, 2}, {9 << 2, 5 << 2}, {9<<2 | 2, 5<<2 | 1}, {0, 5<<2 | 2}} {
			t.Run(fmt.Sprintf("%dx%d-%d-%d", shape[0], shape[1], strength[0], strength[1]), func(t *testing.T) {
				params := parser.CDEFParams{Damping: 4, StrengthCount: 1}
				params.YStrength[0], params.UVStrength[0] = strength[0], strength[1]
				runCDEFU8WalkDifferential(t, cdefU8WalkCase{lumaW: 192, lumaH: 192, xDecC: shape[0], yDecC: shape[1], params: params, readProb: 100, withSkip: true, activeMasks: patterns, poisonInput: true, lowRange: true})
			})
		}
	}
}

// Compare the entire byte scratch and error/panic outcome with the frozen
// original walk, including a byte plane and boundaries that overlap the byte
// tap buffer. The candidate must take its old path.
func TestCDEFSparseAliasesAndErrorsFrozen(t *testing.T) {
	const tapAt = 4 * 192
	for _, mode := range []string{"disjoint", "input-plane", "input-top", "input-bottom", "band-first", "band-middle", "band-last", "short-input", "short-lines", "bad-index", "bad-plane"} {
		t.Run(mode, func(t *testing.T) {
			run := func(frozen bool) (backing []byte, pix []byte, outcome string) {
				defer func() {
					if p := recover(); p != nil {
						outcome = fmt.Sprint(p)
					}
				}()
				const width, height, stride = 192, 192, 208
				backing = make([]byte, stride*height+CDEFU8LineScratchLen(width))
				for i := range backing {
					backing[i] = byte(i*31 + 7)
				}
				pix = make([]byte, stride*height)
				for i := range pix {
					pix[i] = byte((i * 29) ^ (i >> 3))
				}
				input := make([]uint16, cdef.InputBufferSize)
				for i := range input {
					input[i] = uint16(i*31 + 7)
				}
				lines := backing[:CDEFU8LineScratchLen(width)]
				if mode == "input-plane" {
					// The plane starts inside the byte tap buffer.
					pix = backing[tapAt : tapAt+len(pix)]
				}
				if mode == "short-input" {
					input = input[:3]
				}
				if mode == "short-lines" {
					lines = lines[:3]
				}
				plane := frame.Plane{Pix: pix, Width: width, Height: height, Stride: stride}
				if mode == "bad-plane" {
					plane.Stride = 2
				}
				indices := threading.FrameWorkCDEFIndexMap{Stride: 3, Rows: 3, Index: make([]uint8, 9), Read: []bool{true, true, true, true, true, true, true, true, true}}
				if mode == "bad-index" {
					indices.Index[2] = 9
				}
				params := parser.CDEFParams{Damping: 4, StrengthCount: 1}
				params.YStrength[0] = 2
				recs := make([]threading.FrameWorkLoopFilterBlockRecord, (width/4)*(height/4))
				for i := range recs {
					recs[i].Valid = true
					recs[i].SkipTransform = true
				}
				for uy := range 3 {
					for ux := range 3 {
						x, y := 4, 4
						if mode == "band-first" || mode == "band-middle" || mode == "band-last" {
							x, y = 14, 14
							if ux == 0 {
								y = 0
							}
							if ux == 1 {
								x = 0
							}
						}
						recs[(uy*16+y)*(width/4)+ux*16+x].SkipTransform = false
					}
				}
				skip := &FrameWorkLoopFilterMap{Records: recs, Stride: width / 4, Rows: height / 4}
				var blocks [64]cdef.BlockPosition
				var dirs cdef.DirectionGrid
				var vars cdef.VarianceGrid
				apply := frameWorkApplyCDEFPlaneRowsU8
				if frozen {
					apply = frameWorkApplyCDEFPlaneRowsU8FrozenSparse
				}
				rowStart, rowEnd := 0, 3
				var boundary FrameWorkCDEFPostFilterU8BandBoundary
				if mode == "band-first" || mode == "band-middle" || mode == "band-last" {
					switch mode {
					case "band-first":
						rowStart, rowEnd = 0, 1
					case "band-middle":
						rowStart, rowEnd = 1, 2
					case "band-last":
						rowStart, rowEnd = 2, 3
					}
					if rowStart > 0 {
						boundary.Top[0] = make([]byte, 2*width)
						for r := range 2 {
							copy(boundary.Top[0][r*width:][:width], pix[((rowStart*64)-2+r)*stride:][:width])
						}
					}
					if rowEnd < 3 {
						boundary.Bottom[0] = make([]byte, 2*width)
						for r := range 2 {
							copy(boundary.Bottom[0][r*width:][:width], pix[(rowEnd*64+r)*stride:][:width])
						}
					}
				}
				if mode == "input-top" || mode == "input-bottom" {
					rowStart, rowEnd = 1, 2
					boundary.Top[0] = make([]byte, 2*width)
					boundary.Bottom[0] = make([]byte, 2*width)
					view := lines[tapAt : tapAt+2*width]
					if mode == "input-top" {
						boundary.Top[0] = view
					} else {
						boundary.Bottom[0] = view
					}
				}
				u, b, err := apply(params, indices, skip, 3, 3, rowStart, rowEnd, plane, lines, boundary, input, blocks[:], &dirs, &vars, nil, nil, 0, 0, 0, false, nil)
				outcome = fmt.Sprintf("%d/%d/%v/%v/%v", u, b, err, dirs, vars)
				return
			}
			wb, wp, we := run(true)
			gb, gp, ge := run(false)
			if we != ge || !slices.Equal(wp, gp) {
				t.Fatalf("outcome/pixels differ: %s vs %s", we, ge)
			}
			// Independent input is temporary staging: unused samples are left
			// untouched, including when a later unit reports an invalid index.
			// Line backups/canaries still match; alias routes match everything.
			if mode != "input-plane" && mode != "input-top" && mode != "input-bottom" {
				wb, gb = append(wb[:tapAt:tapAt], wb[tapAt+cdefU8TapBytes:]...), append(gb[:tapAt:tapAt], gb[tapAt+cdefU8TapBytes:]...)
			}
			if !slices.Equal(wb, gb) {
				t.Fatal("alias/line/canary backing mutation differs")
			}
		})
	}
}

func TestCDEFSparseDeclineBeforeMutation(t *testing.T) {
	const width, height, stride = 192, 192, 208
	plane := frame.Plane{Pix: make([]byte, stride*height), Width: width, Height: height, Stride: stride}
	for _, mode := range []string{"short-input", "short-top", "short-bottom", "short-left", "inactive", "empty-rect", "wide-rect", "bad-plane"} {
		t.Run(mode, func(t *testing.T) {
			input := make([]uint16, cdef.InputBufferSize)
			bytes := make([]byte, 2*cdef.InputBufferSize)
			for i := range input {
				input[i] = 0xaced
			}
			for i := range bytes {
				bytes[i] = 0xda
			}
			top, bottom, left := make([]byte, 2*width), make([]byte, 2*width), make([]byte, 8*64)
			r := frameWorkCDEFStageRect{8, 8, 24, 24, true}
			p := plane
			switch mode {
			case "short-input":
				input, bytes = input[:3], bytes[:3]
			case "short-top":
				top = top[:1]
			case "short-bottom":
				bottom = bottom[:1]
			case "short-left":
				left = left[:1]
			case "inactive":
				r.active = false
			case "empty-rect":
				r.x0 = r.x1
			case "wide-rect":
				r.x1 = 65
			case "bad-plane":
				p.Stride = 1
			}
			if frameWorkAssembleCDEFSparseInputU8(input, p, top, bottom, left, true, 1, 64, 64, 64, 64, r) || frameWorkAssembleCDEFSparseByteInputU8(bytes, p, top, bottom, left, true, 64, 64, 64, 64, r) {
				t.Fatal("invalid stage accepted")
			}
			for _, x := range input {
				if x != 0xaced {
					t.Fatal("wide mutation before fallback")
				}
			}
			for _, x := range bytes {
				if x != 0xda {
					t.Fatal("byte mutation before fallback")
				}
			}
		})
	}
}

func TestCDEFSparseHelpersNoAllocs(t *testing.T) {
	const width, height, stride = 192, 192, 208
	plane := frame.Plane{Pix: make([]byte, stride*height), Width: width, Height: height, Stride: stride}
	input := make([]uint16, cdef.InputBufferSize)
	bytes := make([]byte, 2*cdef.InputBufferSize)
	top, bottom, left := make([]byte, 2*width), make([]byte, 2*width), make([]byte, 8*64)
	blocks := []cdef.BlockPosition{{BX: 2, BY: 3}, {BX: 4, BY: 5}}
	if n := testing.AllocsPerRun(100, func() {
		r := frameWorkCDEFSparseStageRect(blocks, 64, 64, 8, 8)
		if !frameWorkCDEFSparseScratchIndependent(bytes, plane.Pix, top, top, bottom) || !frameWorkAssembleCDEFSparseInputU8(input, plane, top, bottom, left, true, 1, 64, 64, 64, 64, r) || !frameWorkAssembleCDEFSparseByteInputU8(bytes, plane, top, bottom, left, true, 64, 64, 64, 64, r) {
			panic("valid stage declined")
		}
	}); n != 0 {
		t.Fatalf("allocations=%v", n)
	}
}

func TestCDEFSparseFullPlaneNoAllocs(t *testing.T) {
	for _, strength := range []uint8{2, 38} {
		params := parser.CDEFParams{Damping: 4, StrengthCount: 1}
		params.YStrength[0] = strength
		masks := []uint64{1, 1 << 63, 0x00003c3c3c3c0000, 0x00000000ffffffff, 0x00003c3c3c3c0000, 0x0101010101010101, 0x8080808080808080, ^uint64(0), 0}
		tc := cdefU8WalkCase{lumaW: 192, lumaH: 192, params: params, activeMasks: masks}
		if n := cdefPlaneWalkU8Allocs(t, tc); n != 0 {
			t.Fatalf("strength=%d allocations=%v", strength, n)
		}
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterRejectsEarlierLoopFilter(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1}},
			CDEF: parser.CDEFParams{
				Damping:       5,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{8},
			},
		},
		Output: output,
	}
	_, err := ctx.ApplyCDEFPostFilter(FrameWorkCDEFPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplyCDEFPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterSkipsZeroStrengths(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 32})
	output.Y.Pix[0] = 0x5a
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{Bits: 1, StrengthCount: 2},
		},
		Output: output,
	}
	result, err := ctx.ApplyCDEFPostFilter(FrameWorkCDEFPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkCDEFPostFilterResult{}) {
		t.Fatalf("result=%+v want zero", result)
	}
	if output.Y.Pix[0] != 0x5a {
		t.Fatalf("output sample=%d want 0x5a", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersLuma(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Units != 1 || result.Blocks == 0 {
		t.Fatalf("result=%+v", result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("CDEF did not change luma samples")
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterDefaultsIndexMapFromContext(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	cdefMap := req.IndexMap
	ctx.CDEFIndexMap = &cdefMap
	req.IndexMap = FrameWorkCDEFIndexMap{}
	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Units != 1 || result.Blocks == 0 {
		t.Fatalf("result=%+v", result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("CDEF did not change luma samples")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRunsCDEF(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	cdefMap := req.IndexMap
	ctx.CDEFIndexMap = &cdefMap
	req.IndexMap = FrameWorkCDEFIndexMap{}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if size.CDEF.Input != cdef.InputBufferSize || size.CDEF.UnitDst != cdef.InputBufferSize {
		t.Fatalf("CDEF scratch=%+v", size.CDEF)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterCDEF || result.CDEF.Planes != 1 || result.CDEF.Units != 1 {
		t.Fatalf("result=%+v", result)
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("supported postfilter pipeline did not change luma samples")
	}
}

func TestFrameWorkSupportedPostFilterRunnerApplyRunsCDEF(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	cdefMap := req.IndexMap
	ctx.CDEFIndexMap = &cdefMap
	req.IndexMap = FrameWorkCDEFIndexMap{}
	postReq := FrameWorkPostFilterRequest{CDEF: req}
	size, err := ctx.SupportedPostFilterScratchLen(postReq)
	if err != nil {
		t.Fatal(err)
	}
	if size.CDEF.Input != cdef.InputBufferSize || size.LoopFilter != (FrameWorkLoopFilterPostFilterScratchSize{}) ||
		size.Restoration != (FrameWorkRestorationPostFilterScratchSize{}) ||
		size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("supported scratch=%+v", size)
	}
	next, result, err := ctx.ApplySupportedPostFilters(postReq)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterCDEF || next.RemainingPostFilters() != 0 {
		t.Fatalf("supported result=%+v remaining=%b", result, next.RemainingPostFilters())
	}
	if next.DetachedPostFilterOutput() {
		t.Fatal("supported postfilters returned detached output")
	}
	if err := next.RequirePublishablePostFilterOutput(); err != nil {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v", err)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("supported postfilters did not change luma samples")
	}
}

func TestFrameWorkPostFilterContextSupportedPostFilterCapabilities(t *testing.T) {
	cdefEvent := Event{
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{8},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: cdefEvent}
	supported, err := ctx.SupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	unsupported, err := ctx.UnsupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	if supported != FrameWorkPostFilterCDEF || !unsupported.Empty() {
		t.Fatalf("supported=%b unsupported=%b", supported, unsupported)
	}
	if err := ctx.RequireSupportedPostFilters(); err != nil {
		t.Fatalf("RequireSupportedPostFilters err=%v", err)
	}

	superResCtx := FrameWorkPostFilterContext{
		Event: Event{
			CDEF:      cdefEvent.CDEF,
			FrameSize: parser.FrameSize{SuperResEnabled: true},
		},
	}
	supported, err = superResCtx.SupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	unsupported, err = superResCtx.UnsupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	if supported != FrameWorkPostFilterCDEF || unsupported != FrameWorkPostFilterSuperRes {
		t.Fatalf("superres supported=%b unsupported=%b", supported, unsupported)
	}
	if err := superResCtx.RequireSupportedPostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireSupportedPostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}

	tailCtx := superResCtx.WithCompletedPostFilters(FrameWorkPostFilterCDEF)
	supported, err = tailCtx.SupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	unsupported, err = tailCtx.UnsupportedPostFilters()
	if err != nil {
		t.Fatal(err)
	}
	if !supported.Empty() || unsupported != FrameWorkPostFilterSuperRes {
		t.Fatalf("tail supported=%b unsupported=%b", supported, unsupported)
	}
}

func TestFrameWorkCallerPostFilterRunnerAllocs(t *testing.T) {
	var ctx FrameWorkPostFilterContext
	var req FrameWorkPostFilterRequest
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ctx.CallerPostFilterScratchLen(req); err != nil {
			t.Fatal(err)
		}
		next, result, err := ctx.ApplyCallerPostFilters(req)
		if err != nil {
			t.Fatal(err)
		}
		if err := next.RequireNoRemainingPostFilters(); err != nil {
			t.Fatal(err)
		}
		if next.RemainingPostFilters() != 0 || result.Completed != 0 {
			t.Fatalf("caller context=%+v result=%+v", next, result)
		}
	})
	if allocs != 0 {
		t.Fatalf("caller postfilters allocated: %f", allocs)
	}
}

func TestFrameWorkCDEFPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{8, 4, 2},
		Dst:           [3]int{7, 3, 1},
		DirectionGrid: 5,
		VarianceGrid:  5,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	indexMap := FrameWorkCDEFIndexMap{
		Index:  make([]uint8, 6),
		Read:   make([]bool, 6),
		Stride: 3,
		Rows:   2,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	req, err := size.BindRequest(indexMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.SampleScratch[0]) != size.Samples[0] || len(req.DstScratch[1]) != size.Dst[1] ||
		len(req.DirectionGrid) != size.DirectionGrid || len(req.VarianceGrid) != size.VarianceGrid ||
		len(req.InputScratch) != size.Input || len(req.UnitDstScratch) != size.UnitDst {
		t.Fatalf("request lengths samples=%v dst=%v direction=%d variance=%d input=%d unitDst=%d",
			[3]int{len(req.SampleScratch[0]), len(req.SampleScratch[1]), len(req.SampleScratch[2])},
			[3]int{len(req.DstScratch[0]), len(req.DstScratch[1]), len(req.DstScratch[2])},
			len(req.DirectionGrid), len(req.VarianceGrid), len(req.InputScratch), len(req.UnitDstScratch))
	}
	if req.IndexMap.Stride != indexMap.Stride || req.IndexMap.Rows != indexMap.Rows ||
		len(req.IndexMap.Index) != len(indexMap.Index) || len(req.IndexMap.Read) != len(indexMap.Read) {
		t.Fatalf("index map=%+v want %+v", req.IndexMap, indexMap)
	}
}

func TestFrameWorkCDEFPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{4, 3, 2},
		Dst:           [3]int{4, 3, 2},
		DirectionGrid: 2,
		VarianceGrid:  2,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	tests := []struct {
		name     string
		size     FrameWorkCDEFPostFilterScratchSize
		samples  [3][]uint16
		dst      [3][]uint16
		dir      []cdef.DirectionGrid
		variance []cdef.VarianceGrid
		input    []uint16
		unitDst  []uint16
	}{
		{name: "sample", size: size, samples: [3][]uint16{samples[0][:3], samples[1], samples[2]}, dst: dst, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst},
		{name: "dst", size: size, samples: samples, dst: [3][]uint16{dst[0], dst[1][:2], dst[2]}, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst},
		{name: "direction", size: size, samples: samples, dst: dst, dir: directionGrid[:1], variance: varianceGrid, input: input, unitDst: unitDst},
		{name: "variance", size: size, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid[:1], input: input, unitDst: unitDst},
		{name: "input", size: size, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid, input: input[:cdef.InputBufferSize-1], unitDst: unitDst},
		{name: "unit-dst", size: size, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst[:cdef.InputBufferSize-1]},
		{name: "negative", size: FrameWorkCDEFPostFilterScratchSize{Samples: [3]int{-1}}, samples: samples, dst: dst, dir: directionGrid, variance: varianceGrid, input: input, unitDst: unitDst},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.size.BindRequest(FrameWorkCDEFIndexMap{}, tt.samples, tt.dst, tt.dir, tt.variance, tt.input, tt.unitDst); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("BindRequest err=%v want %v", err, frame.ErrShortBuffer)
			}
		})
	}
}

func TestFrameWorkCDEFPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{4096, 1024, 1024},
		Dst:           [3]int{4096, 1024, 1024},
		DirectionGrid: 16,
		VarianceGrid:  16,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(FrameWorkCDEFIndexMap{}, samples, dst, directionGrid, varianceGrid, input, unitDst); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsUnsupportedBeforeMutation(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 32})
	output.Y.Pix[0] = 0x44
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResEnabled:     true,
				SuperResDenominator: 16,
			},
			CDEF: parser.CDEFParams{
				Damping:       5,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			},
		},
		Output: output,
	}
	if _, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{}); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("SupportedPostFilterScratchLen err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if next.RemainingPostFilters() != ctx.RemainingPostFilters() || result != (FrameWorkPostFilterResult{}) {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if output.Y.Pix[0] != 0x44 {
		t.Fatalf("output sample=%d want 0x44", output.Y.Pix[0])
	}
}

// TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersSVCEnhancementLayer
// regression-tests CDEF on a 1280x720 frame (an SVC enhancement-layer-sized
// resolution where the unit grid is 20x12 and the chroma grid is 640x360).
// CDEF must derive every dimension from the per-frame Event/Output, never from
// a hardcoded global resolution. This guards against future regressions where
// CDEF assumes a single global frame size and trips when the enhancement layer
// decodes at a larger size than the base layer.
func TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersSVCEnhancementLayer(t *testing.T) {
	const width = 1280
	const height = 720
	const expectedLumaCols = (width + cdef.BlockSize - 1) / cdef.BlockSize  // 20
	const expectedLumaRows = (height + cdef.BlockSize - 1) / cdef.BlockSize // 12
	const expectedUnits = expectedLumaCols * expectedLumaRows               // 240

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{
		Width: width, Height: height, BitDepth: 8,
		SubsamplingX: true, SubsamplingY: true, Align: 32,
	})
	testFillFrameWorkCDEFPlane(output.Y)
	testFillFrameWorkCDEFPlane(output.U)
	testFillFrameWorkCDEFPlane(output.V)
	beforeY := testCopyFrameWorkCDEFPlane(output.Y)
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}

	// Verify the unit grid derives from the per-frame size, not a hardcoded
	// global resolution. An SVC base layer of 640x360 would yield 10x6=60
	// units; the enhancement layer must report 20x12=240.
	cols, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatalf("frameWorkCDEFUnitGrid err=%v", err)
	}
	if cols != expectedLumaCols || rows != expectedLumaRows {
		t.Fatalf("unit grid=%dx%d want %dx%d", cols, rows, expectedLumaCols, expectedLumaRows)
	}

	// Sanity-check scratch sizing scales with the frame's plane dimensions.
	size, err := ctx.CDEFPostFilterScratchLen()
	if err != nil {
		t.Fatalf("CDEFPostFilterScratchLen err=%v", err)
	}
	if size.Samples[0] < width*height || size.Dst != [3]int{} {
		t.Fatalf("luma scratch=%d dst=%v want samples >= %d and no dst scratch", size.Samples[0], size.Dst, width*height)
	}
	chromaSize := (width / 2) * (height / 2)
	if size.Samples[1] < chromaSize || size.Samples[2] < chromaSize {
		t.Fatalf("chroma sample scratch=%d/%d want >= %d", size.Samples[1], size.Samples[2], chromaSize)
	}
	if size.DirectionGrid < expectedUnits || size.VarianceGrid < expectedUnits {
		t.Fatalf("direction/variance grid=%d/%d want >= %d", size.DirectionGrid, size.VarianceGrid, expectedUnits)
	}
	// Per-unit scratch buffers stay bounded by the CDEF unit (64x64) plus
	// borders; they must not scale with frame resolution.
	if size.Input != cdef.InputBufferSize || size.UnitDst != cdef.InputBufferSize {
		t.Fatalf("input/unitDst scratch=%d/%d want %d", size.Input, size.UnitDst, cdef.InputBufferSize)
	}

	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	if req.IndexMap.Stride != expectedLumaCols || req.IndexMap.Rows != expectedLumaRows {
		t.Fatalf("indexMap stride/rows=%d/%d want %d/%d",
			req.IndexMap.Stride, req.IndexMap.Rows, expectedLumaCols, expectedLumaRows)
	}

	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatalf("ApplyCDEFPostFilter err=%v", err)
	}
	if result.Planes != 3 {
		t.Fatalf("planes=%d want 3", result.Planes)
	}
	// All 240 units have non-zero strengths and are marked read, so every
	// unit should run on every plane.
	if result.Units != expectedUnits*3 {
		t.Fatalf("units=%d want %d", result.Units, expectedUnits*3)
	}
	if result.Blocks == 0 {
		t.Fatalf("blocks=0; want > 0")
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, beforeY) {
		t.Fatal("CDEF did not change luma samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("CDEF did not change U samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("CDEF did not change V samples")
	}
}

func TestFrameWorkPostFilterContextApplyCDEFPostFilterFiltersChromaWithLumaDirectionPass(t *testing.T) {
	const width = 128
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			UVStrength:    [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	testFillFrameWorkCDEFPlane(output.U)
	testFillFrameWorkCDEFPlane(output.V)
	beforeY := testCopyFrameWorkCDEFPlane(output.Y)
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	result, err := ctx.ApplyCDEFPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 3 || result.Units != 4 || result.Blocks == 0 {
		t.Fatalf("result=%+v", result)
	}
	if testFrameWorkCDEFPlaneChanged(output.Y, beforeY) {
		t.Fatal("direction-only luma pass changed luma samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("CDEF did not change U samples")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("CDEF did not change V samples")
	}
}

func BenchmarkFrameWorkCDEFPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{1920 * 1080, 960 * 540, 960 * 540},
		Dst:           [3]int{1920 * 1080, 960 * 540, 960 * 540},
		DirectionGrid: 120 * 68,
		VarianceGrid:  120 * 68,
		Input:         cdef.InputBufferSize,
		UnitDst:       cdef.InputBufferSize,
	}
	indexMap := FrameWorkCDEFIndexMap{
		Index:  make([]uint8, size.DirectionGrid),
		Read:   make([]bool, size.DirectionGrid),
		Stride: 120,
		Rows:   68,
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(indexMap, samples, dst, directionGrid, varianceGrid, input, unitDst); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkCDEFFrame(t testing.TB, format frame.Format) *frame.Frame {
	t.Helper()
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	output, err := frame.Bind(make([]byte, layout.Size), format)
	if err != nil {
		t.Fatal(err)
	}
	return &output
}

func testFrameWorkCDEFPostFilterRequest(t testing.TB, ctx FrameWorkPostFilterContext, event Event) FrameWorkCDEFPostFilterRequest {
	t.Helper()
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
			FrameSize: event.FrameSize,
			CDEF:      event.CDEF,
		},
	}
	_, _, length, err := batch.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := batch.BindCDEFIndexMap(make([]uint8, length), make([]bool, length))
	if err != nil {
		t.Fatal(err)
	}
	for i := range cdefMap.Read {
		cdefMap.Read[i] = true
	}
	size, err := ctx.CDEFPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size)
	req, err := size.BindRequest(cdefMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func testFrameWorkCDEFScratchStorage(size FrameWorkCDEFPostFilterScratchSize) ([3][]uint16, [3][]uint16, []cdef.DirectionGrid, []cdef.VarianceGrid, []uint16, []uint16) {
	var samples [3][]uint16
	var dst [3][]uint16
	for plane := range 3 {
		if size.Samples[plane] > 0 {
			samples[plane] = make([]uint16, size.Samples[plane])
		}
		if size.Dst[plane] > 0 {
			dst[plane] = make([]uint16, size.Dst[plane])
		}
	}
	return samples, dst,
		make([]cdef.DirectionGrid, maxInt(size.DirectionGrid, 0)),
		make([]cdef.VarianceGrid, maxInt(size.VarianceGrid, 0)),
		make([]uint16, maxInt(size.Input, 0)),
		make([]uint16, maxInt(size.UnitDst, 0))
}

func testFillFrameWorkCDEFPlane(plane frame.Plane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte((x*37 + y*53 + (x^y)*17) & 0xff)
		}
	}
}

func testCopyFrameWorkCDEFPlane(plane frame.Plane) []byte {
	out := make([]byte, plane.Width*plane.Height)
	for y := 0; y < plane.Height; y++ {
		copy(out[y*plane.Width:(y+1)*plane.Width], plane.Pix[y*plane.Stride:y*plane.Stride+plane.Width])
	}
	return out
}

func testFrameWorkCDEFPlaneChanged(plane frame.Plane, before []byte) bool {
	for y := 0; y < plane.Height; y++ {
		row := before[y*plane.Width : (y+1)*plane.Width]
		for x := 0; x < plane.Width; x++ {
			if plane.Pix[y*plane.Stride+x] != row[x] {
				return true
			}
		}
	}
	return false
}

func TestFrameWorkFillCDEFInputSentinelsFillsHaloOnly(t *testing.T) {
	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = 0x7777
	}
	const unitW = 32
	const unitH = 32
	if err := frameWorkFillCDEFInputSentinels(input, unitW, unitH); err != nil {
		t.Fatalf("frameWorkFillCDEFInputSentinels: %v", err)
	}

	fillW := unitW + 2*cdef.HorizontalBorder
	fillH := unitH + 2*cdef.VerticalBorder
	for row := range fillH {
		for col := range fillW {
			got := input[row*cdef.BStride+col]
			want := uint16(0x7777)
			if row < cdef.VerticalBorder || row >= cdef.VerticalBorder+unitH ||
				col < cdef.HorizontalBorder || col >= cdef.HorizontalBorder+unitW {
				want = cdef.VeryLarge
			}
			if got != want {
				t.Fatalf("input row=%d col=%d got=%d want=%d", row, col, got, want)
			}
		}
	}
	fillLen := (fillH-1)*cdef.BStride + fillW
	if got := input[fillLen]; got != 0x7777 {
		t.Fatalf("inactive next row got=%d want marker", got)
	}
}

// TestFrameWorkCopyCDEFInputBottomEdgeReplicatesLastRow pins libaom's CDEF
// bot_linebuf behavior for chroma planes whose visible height ends inside a
// non-final CDEF unit. The chroma 33-tall plane in the libaom 66x66 vector hits
// CDEF unit row 0 with unit height 32: the kernel's bottom directional taps at
// by=7 (rows 28..31) need samples at rows 32 and 33 of the input buffer. Row 32
// is visible (real sample); row 33 in libaom comes from the YV12 buffer's
// aligned-but-out-of-crop padding (uv_height vs uv_crop_height), which holds
// the last visible row replicated by alignment/decode writes. Our SamplePlane
// is exactly the visible window, so frameWorkCopyCDEFInput synthesizes the
// extension by replicating the last visible row into the bottom border. The
// final CDEF unit row must keep the VeryLarge sentinel just like libaom's
// fill_rect(...CDEF_VERY_LARGE) when fbr == nvfb - 1.
func TestFrameWorkCopyCDEFInputBottomEdgeReplicatesLastRow(t *testing.T) {
	const planeHeight = 33
	const planeWidth = 33
	const unitX = 0
	const unitY = 0
	const unitW = 32
	const unitH = 32

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(100 + y)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Visible rows 0..32 must hold the real samples (last visible row is 32).
	for y := range planeHeight {
		for x := range planeWidth {
			if got, want := at(y, x), uint16(100+y); got != want {
				t.Fatalf("input row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
	// Rows past the visible plane within the CDEF VerticalBorder must replicate
	// the last visible row instead of leaving VeryLarge — otherwise the kernel's
	// directional taps at by=7 would diverge from libaom's bot_linebuf reads.
	const lastVisible = planeHeight - 1
	for y := planeHeight; y < unitY+unitH+cdef.VerticalBorder; y++ {
		for x := range planeWidth {
			if got, want := at(y, x), uint16(100+lastVisible); got != want {
				t.Fatalf("bottom-extension row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestFrameWorkCopyCDEFInputRightEdgeReplicatesLastColumn pins libaom's CDEF
// av1_cdef_copy_sb8_16 read behavior for non-rightmost superblocks whose visible
// plane width ends inside the central copy's hsize+CDEF_HBORDER reach. The
// chroma 33-wide plane in the libaom 66x66 vector hits unit (0,0) with unit
// width 32: the kernel's directional secondary taps at bx=7 (cols 28..31) need
// samples at cols 32 and 33 of the input buffer. Col 32 is visible (real
// sample); col 33 in libaom comes from the YV12 buffer's
// aligned-but-out-of-crop padding (uv_width vs uv_crop_width), which holds the
// last visible column replicated by alignment/decode writes. Our SamplePlane is
// exactly the visible window, so frameWorkCopyCDEFInput synthesizes the
// extension by replicating the last visible column into the right border. The
// final CDEF unit column must keep the VeryLarge sentinel just like libaom's
// fill_rect(...CDEF_VERY_LARGE) frame_boundary[RIGHT] overlay when fbc ==
// nhfb - 1.
func TestFrameWorkCopyCDEFInputRightEdgeReplicatesLastColumn(t *testing.T) {
	const planeHeight = 33
	const planeWidth = 33
	const unitX = 0
	const unitY = 0
	const unitW = 32
	const unitH = 32

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(300 + x)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Visible cols 0..32 must hold the real samples (last visible col is 32).
	// Rows past the visible plane within the CDEF VerticalBorder are independently
	// covered by TestFrameWorkCopyCDEFInputBottomEdgeReplicatesLastRow.
	for y := range planeHeight {
		for x := range planeWidth {
			if got, want := at(y, x), uint16(300+x); got != want {
				t.Fatalf("input row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
	// Cols past the visible plane within the CDEF HorizontalBorder must replicate
	// the last visible column instead of leaving VeryLarge — otherwise the kernel's
	// directional taps at bx=7 would diverge from libaom's av1_cdef_copy_sb8_16
	// read into out-of-crop frame buffer padding. The bottom-row replication
	// extends the visible rows below the plane into wantSrcY1 first; the column
	// replication then carries the last visible column across both the visible
	// rows and the bottom-extension rows so that the bottom-right corner is also
	// populated with the last visible sample (libaom-equivalent nearest-pixel
	// 2D edge extension for past-visible CDEF input).
	const lastVisible = planeWidth - 1
	wantSrcY1 := unitY + unitH + cdef.VerticalBorder
	for y := range wantSrcY1 {
		for x := planeWidth; x < unitX+unitW+cdef.HorizontalBorder; x++ {
			if got, want := at(y, x), uint16(300+lastVisible); got != want {
				t.Fatalf("right-extension row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestFrameWorkCopyCDEFInputFinalColumnKeepsVeryLarge pins the libaom branch in
// cdef_prepare_fb that fills the right border with VERY_LARGE when the unit is
// the final superblock column (fbc == nhfb - 1) — i.e., the unit's right edge
// coincides with the plane's right edge. Replicating the last visible column in
// that case would diverge from libaom's frame_boundary[RIGHT] overlay and
// regress vectors whose rightmost CDEF unit has visible content (e.g. SB(1,0)
// of the 66x66 Y plane).
func TestFrameWorkCopyCDEFInputFinalColumnKeepsVeryLarge(t *testing.T) {
	const planeHeight = 66
	const planeWidth = 66
	// Final luma CDEF unit at column 1: unitX=64 unitW=2, ends exactly at plane
	// right edge. This matches the libaom fbc == nhfb - 1 branch.
	const unitX = 64
	const unitY = 0
	const unitW = 2
	const unitH = 64

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(400 + x)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Right border cols (relative offsets unitW..unitW+HorizontalBorder-1) must
	// remain VeryLarge to match libaom's fill_rect(...VERY_LARGE) branch on the
	// rightmost superblock unit.
	for y := range unitH {
		for x := unitW; x < unitW+cdef.HorizontalBorder; x++ {
			if got := at(y, x); got != cdef.VeryLarge {
				t.Fatalf("final-column right border row=%d col=%d got=%d want VeryLarge", y, x, got)
			}
		}
	}
}

// TestFrameWorkCopyCDEFInputFinalUnitKeepsVeryLarge pins the libaom branch in
// cdef_prepare_fb that fills the bottom border with VERY_LARGE when the unit is
// the final superblock row (fbr == nvfb - 1) — i.e., the unit's bottom edge
// coincides with the plane's bottom edge. Replicating the last visible row in
// that case would diverge from libaom and regress the Y plane on 66x66.
func TestFrameWorkCopyCDEFInputFinalUnitKeepsVeryLarge(t *testing.T) {
	const planeHeight = 66
	const planeWidth = 66
	// Final luma CDEF unit at row 1: unitY=64 unitH=2, ends exactly at plane
	// bottom. This matches the libaom fbr == nvfb - 1 branch.
	const unitX = 0
	const unitY = 64
	const unitW = 64
	const unitH = 2

	src := frame.SamplePlane{
		Pix:    make([]uint16, planeWidth*planeHeight),
		Stride: planeWidth,
		Width:  planeWidth,
		Height: planeHeight,
	}
	for y := range planeHeight {
		for x := range planeWidth {
			src.Pix[y*src.Stride+x] = uint16(200 + y)
		}
	}

	input := make([]uint16, cdef.InputBufferSize)
	for i := range input {
		input[i] = cdef.VeryLarge
	}
	if err := frameWorkCopyCDEFInput(input, src, unitX, unitY, unitW, unitH); err != nil {
		t.Fatalf("frameWorkCopyCDEFInput: %v", err)
	}

	originRow := cdef.VerticalBorder
	originCol := cdef.HorizontalBorder
	at := func(row, col int) uint16 {
		return input[(originRow+row)*cdef.BStride+originCol+col]
	}

	// Bottom border rows (relative offsets unitH..unitH+VerticalBorder-1)
	// must remain VeryLarge to match libaom's fill_rect(...VERY_LARGE) branch.
	for y := unitH; y < unitH+cdef.VerticalBorder; y++ {
		for x := range unitW {
			if got := at(y, x); got != cdef.VeryLarge {
				t.Fatalf("final-unit bottom border row=%d col=%d got=%d want VeryLarge", y, x, got)
			}
		}
	}
}

// TestCDEFPostFilterBandedMatchesWhole proves the unit-row banded apply
// reproduces the whole-frame application exactly: bands over disjoint row
// ranges against one shared sample snapshot (each with its own input and
// unit scratch) write the same output the single-call path does.
func TestCDEFPostFilterBandedMatchesWhole(t *testing.T) {
	const width, height = 256, 192
	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{CodedWidth: uint32(width), UpscaledWidth: uint32(width), Height: uint32(height), SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 2,
			YStrength:     [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	build := func() (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, frame.Format{
			Width: width, Height: height, BitDepth: 8,
			SubsamplingX: true, SubsamplingY: true, Align: 64,
		})
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{
			FrameWorkFrameContext: threading.FrameWorkFrameContext{
				Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
				FrameSize: event.FrameSize,
				CDEF:      event.CDEF,
			},
		}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		idx := make([]uint8, length)
		for i := range idx {
			idx[i] = uint8(i % 2)
		}
		cdefMap, err := batch.BindCDEFIndexMap(idx, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range cdefMap.Read {
			cdefMap.Read[i] = true
		}
		scratch, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(scratch)
		req, err := scratch.BindRequest(cdefMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return output, ctx, req
	}

	wholeOut, wholeCtx, wholeReq := build()
	if _, err := wholeCtx.ApplyCDEFPostFilter(wholeReq); err != nil {
		t.Fatal(err)
	}

	bandOut, bandCtx, bandReq := build()
	if err := bandCtx.LoadCDEFPostFilterSamples(bandReq); err != nil {
		t.Fatal(err)
	}
	_, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	for r0 := 0; r0 < rows; r0 += 1 { // one band per unit row, separate scratch
		band := bandReq
		band.InputScratch = make([]uint16, len(bandReq.InputScratch))
		band.UnitDstScratch = make([]uint16, len(bandReq.UnitDstScratch))
		if _, err := bandCtx.ApplyCDEFPostFilterUnitRows(band, r0, r0+1); err != nil {
			t.Fatalf("band %d: %v", r0, err)
		}
	}
	for _, planes := range [][2]frame.Plane{{wholeOut.Y, bandOut.Y}, {wholeOut.U, bandOut.U}, {wholeOut.V, bandOut.V}} {
		w, b := planes[0], planes[1]
		for y := 0; y < w.Height; y++ {
			for x := 0; x < w.Width; x++ {
				if w.Pix[y*w.Stride+x] != b.Pix[y*b.Stride+x] {
					t.Fatalf("banded differs at (%d,%d)", x, y)
				}
			}
		}
	}
}

func TestCDEFPostFilterBandedU8MatchesSnapshot(t *testing.T) {
	runCDEFPostFilterBandedU8MatchesSnapshot(t, 256, 512, [2]uint8{9, 31}, [2]uint8{5, 17})
}

func TestCDEFPostFilterBandedU8SecondaryOnlyMatchesSnapshot(t *testing.T) {
	// Four unit rows and columns provide a real interior unit while exercising
	// the immutable top/bottom boundary snapshots between one-row bands.
	runCDEFPostFilterBandedU8MatchesSnapshot(t, 256, 256, [2]uint8{2, 2}, [2]uint8{2, 2})
}

func runCDEFPostFilterBandedU8MatchesSnapshot(t *testing.T, width int, height int, yStrength [2]uint8, uvStrength [2]uint8) {
	t.Helper()
	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{CodedWidth: uint32(width), UpscaledWidth: uint32(width), Height: uint32(height), SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 2,
			YStrength:     [parser.MaxCDEFStrengths]uint8{yStrength[0], yStrength[1]},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{uvStrength[0], uvStrength[1]},
		},
	}
	build := func() (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, frame.Format{
			Width: width, Height: height, BitDepth: 8,
			SubsamplingX: true, SubsamplingY: true, Align: 64,
		})
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{
			FrameWorkFrameContext: threading.FrameWorkFrameContext{
				Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
				FrameSize: event.FrameSize,
				CDEF:      event.CDEF,
			},
		}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		idx := make([]uint8, length)
		for i := range idx {
			idx[i] = uint8(i % 2)
		}
		cdefMap, err := batch.BindCDEFIndexMap(idx, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range cdefMap.Read {
			cdefMap.Read[i] = true
		}
		scratch, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(scratch)
		req, err := scratch.BindRequest(cdefMap, samples, dst, directionGrid, varianceGrid, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return output, ctx, req
	}

	snapshotOut, snapshotCtx, snapshotReq := build()
	if err := snapshotCtx.LoadCDEFPostFilterSamples(snapshotReq); err != nil {
		t.Fatal(err)
	}
	_, rows, err := frameWorkCDEFUnitGrid(event.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	for r0 := 0; r0 < rows; r0++ {
		band := snapshotReq
		band.InputScratch = make([]uint16, len(snapshotReq.InputScratch))
		band.UnitDstScratch = make([]uint16, len(snapshotReq.UnitDstScratch))
		if _, err := snapshotCtx.ApplyCDEFPostFilterUnitRows(band, r0, r0+1); err != nil {
			t.Fatalf("snapshot band %d: %v", r0, err)
		}
	}

	u8Out, u8Ctx, u8Req := build()
	boundaries := make([]FrameWorkCDEFPostFilterU8BandBoundary, rows)
	for r0 := 0; r0 < rows; r0++ {
		boundaries[r0] = testCDEFU8BandBoundary(width)
		if err := u8Ctx.LoadCDEFPostFilterU8BandBoundary(&boundaries[r0], r0, r0+1); err != nil {
			t.Fatalf("load u8 boundary %d: %v", r0, err)
		}
	}
	for r0 := 0; r0 < rows; r0++ {
		band := u8Req
		band.SampleScratch = testCDEFU8LineScratch(width)
		band.InputScratch = make([]uint16, len(u8Req.InputScratch))
		if _, err := u8Ctx.ApplyCDEFPostFilterUnitRowsU8(band, boundaries[r0], r0, r0+1); err != nil {
			t.Fatalf("u8 band %d: %v", r0, err)
		}
	}
	for _, planes := range [][2]frame.Plane{{snapshotOut.Y, u8Out.Y}, {snapshotOut.U, u8Out.U}, {snapshotOut.V, u8Out.V}} {
		w, b := planes[0], planes[1]
		for y := 0; y < w.Height; y++ {
			for x := 0; x < w.Width; x++ {
				if w.Pix[y*w.Stride+x] != b.Pix[y*b.Stride+x] {
					t.Fatalf("u8 banded differs at (%d,%d)", x, y)
				}
			}
		}
	}
}

func testCDEFU8BandBoundary(width int) FrameWorkCDEFPostFilterU8BandBoundary {
	var boundary FrameWorkCDEFPostFilterU8BandBoundary
	widths := [3]int{width, width / 2, width / 2}
	for plane, planeWidth := range widths {
		need := cdef.VerticalBorder * planeWidth
		boundary.Top[plane] = make([]byte, need)
		boundary.Bottom[plane] = make([]byte, need)
	}
	return boundary
}

func testCDEFU8LineScratch(width int) [3][]uint16 {
	return [3][]uint16{
		make([]uint16, 2*width),
		make([]uint16, width),
		make([]uint16, width),
	}
}

// TestCDEFPostFilterPooledMatchesSerial proves the pooled row-band CDEF apply
// (applyCDEFPostFilterMaybePooled with a multi-lane pool) is byte-identical to
// the serial whole-frame apply, for several worker counts. The frame is tall
// enough (8 CDEF unit rows) that the min-rows-per-band gate lets the pooled path
// engage. This is the worker-count-invariance guarantee the postfilter pool
// rests on.
func TestCDEFPostFilterPooledMatchesSerial(t *testing.T) {
	const width, height = 256, 512 // 4x8 CDEF unit grid
	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 2,
			YStrength:     [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength:    [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	build := func(pool *threading.Pool) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, frame.Format{
			Width: width, Height: height, BitDepth: 8,
			SubsamplingX: true, SubsamplingY: true, Align: 64,
		})
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output, pool: pool}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{
			FrameWorkFrameContext: threading.FrameWorkFrameContext{
				Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
				FrameSize: event.FrameSize,
				CDEF:      event.CDEF,
			},
		}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		idx := make([]uint8, length)
		for i := range idx {
			idx[i] = uint8(i % 2)
		}
		cdefMap, err := batch.BindCDEFIndexMap(idx, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range cdefMap.Read {
			cdefMap.Read[i] = true
		}
		scratch, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, dg, vg, input, unitDst := testFrameWorkCDEFScratchStorage(scratch)
		req, err := scratch.BindRequest(cdefMap, samples, dst, dg, vg, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return output, ctx, req
	}

	serialPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer serialPool.Close()
	serialOut, serialCtx, serialReq := build(serialPool)
	if _, err := serialCtx.applyCDEFPostFilterMaybePooled(serialReq); err != nil {
		t.Fatal(err)
	}

	for _, workers := range []int{2, 3, 4, 8} {
		pool, err := threading.NewPool(workers)
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		pooledOut, pooledCtx, pooledReq := build(pool)
		if _, err := pooledCtx.applyCDEFPostFilterMaybePooled(pooledReq); err != nil {
			pool.Close()
			t.Fatalf("workers=%d apply: %v", workers, err)
		}
		for _, planes := range [][2]frame.Plane{{serialOut.Y, pooledOut.Y}, {serialOut.U, pooledOut.U}, {serialOut.V, pooledOut.V}} {
			s, p := planes[0], planes[1]
			for y := 0; y < s.Height; y++ {
				for x := 0; x < s.Width; x++ {
					if s.Pix[y*s.Stride+x] != p.Pix[y*p.Stride+x] {
						pool.Close()
						t.Fatalf("workers=%d pooled differs at (%d,%d): serial=%d pooled=%d", workers, x, y, s.Pix[y*s.Stride+x], p.Pix[y*p.Stride+x])
					}
				}
			}
		}
		pool.Close()
	}
}

// Differential proof for the in-place 8-bit CDEF frame walk
// (postfilter_cdef_u8.go): on identical planes, index maps, skip maps, and
// strengths, frameWorkApplyCDEFPlaneRowsU8 must leave every frame byte
// identical to the uint16 snapshot walk (frameWorkApplyCDEFPlaneRows over a
// LoadSamplePlaneFull snapshot plus storeCDEFUnit), with identical
// direction/variance grids and unit/block counts, across full/partial units,
// skip patterns, direction-only luma passes, and chroma geometries.

type cdefU8WalkCase struct {
	name        string
	lumaW       int // MI-aligned luma extent (multiple of 8)
	lumaH       int
	xDecC       int // chroma decimation (420: 1,1; 422: 1,0; 444: 0,0)
	yDecC       int
	mono        bool
	params      parser.CDEFParams
	readProb    uint32   // percent of units with Read=true
	withSkip    bool     // attach a skip map with random SkipTransform blocks
	noPrimary   bool     // all entries use secondary-only strengths; direction grids must stay untouched
	unitIndex   []uint8  // optional exact per-unit strength indexes, row-major
	unitRead    []bool   // optional exact per-unit read mask, row-major
	activeMasks []uint64 // optional explicit active 8x8 masks, row-major
	poisonInput bool     // poison the shared tap scratch before each U8 walk
	lowRange    bool     // use low-range samples so secondary taps produce visible deltas
}

func cdefU8WalkCases() []cdefU8WalkCase {
	mk := func(damping uint8, strengths ...[2]uint8) parser.CDEFParams {
		var p parser.CDEFParams
		p.Damping = damping
		p.StrengthCount = uint8(len(strengths))
		for i, s := range strengths {
			p.YStrength[i] = s[0]
			p.UVStrength[i] = s[1]
		}
		return p
	}
	return []cdefU8WalkCase{
		{name: "multi_unit_420", lumaW: 192, lumaH: 136, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{9<<2 | 2, 5<<2 | 1}, [2]uint8{0, 0}, [2]uint8{15<<2 | 3, 4<<2 | 2}, [2]uint8{1 << 2, 2 << 2}), readProb: 100},
		{name: "partial_units_420", lumaW: 136, lumaH: 72, xDecC: 1, yDecC: 1,
			params: mk(5, [2]uint8{7<<2 | 1, 3<<2 | 2}), readProb: 100},
		{name: "sparse_read_420", lumaW: 256, lumaH: 192, xDecC: 1, yDecC: 1,
			params: mk(3, [2]uint8{12 << 2, 6<<2 | 3}, [2]uint8{2<<2 | 2, 0}), readProb: 45, withSkip: true},
		{name: "direction_only_luma", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{0, 5<<2 | 2}, [2]uint8{0, 3 << 2}), readProb: 100},
		{name: "secondary_only_no_direction_420", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{2, 2}, [2]uint8{0, 0}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_420", lumaW: 192, lumaH: 192, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{2, 2}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_420_partial_crop_skip", lumaW: 200, lumaH: 200, xDecC: 1, yDecC: 1,
			params: mk(5, [2]uint8{1, 2}), readProb: 100, withSkip: true, noPrimary: true},
		{name: "secondary_only_byte_interior_422", lumaW: 192, lumaH: 192, xDecC: 1, yDecC: 0,
			params: mk(4, [2]uint8{2, 2}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_444", lumaW: 192, lumaH: 192, xDecC: 0, yDecC: 0,
			params: mk(4, [2]uint8{2, 2}), readProb: 100, noPrimary: true},
		{name: "secondary_only_byte_interior_mono", lumaW: 192, lumaH: 192, mono: true,
			params: mk(4, [2]uint8{2, 0}), readProb: 100, noPrimary: true},
		{name: "secondary_byte_mixed_skip_transitions", lumaW: 384, lumaH: 256, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{2, 2}, [2]uint8{5<<2 | 2, 5<<2 | 2}, [2]uint8{4, 4}), readProb: 100, withSkip: true, poisonInput: true, lowRange: true,
			unitIndex: []uint8{
				0, 0, 0, 0, 0, 0,
				0, 1, 0, 2, 0, 1,
				0, 0, 1, 0, 2, 0,
				0, 0, 0, 0, 0, 0,
			},
			unitRead: []bool{
				true, true, true, true, true, true,
				true, true, true, false, true, true,
				true, true, true, true, false, true,
				true, true, true, true, true, true,
			}},
		{name: "secondary_only_no_luma_direction_422", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 0,
			params: mk(4, [2]uint8{0, 2}, [2]uint8{0, 2}), readProb: 100, noPrimary: true},
		{name: "luma_only", lumaW: 128, lumaH: 128, xDecC: 1, yDecC: 1,
			params: mk(6, [2]uint8{9<<2 | 3, 0}), readProb: 80, withSkip: true},
		{name: "chroma_422", lumaW: 192, lumaH: 128, xDecC: 1, yDecC: 0,
			params: mk(4, [2]uint8{8<<2 | 2, 6<<2 | 1}), readProb: 90},
		{name: "chroma_444", lumaW: 128, lumaH: 72, xDecC: 0, yDecC: 0,
			params: mk(5, [2]uint8{10<<2 | 1, 7<<2 | 2}), readProb: 100, withSkip: true},
		{name: "single_unit", lumaW: 40, lumaH: 24, xDecC: 1, yDecC: 1,
			params: mk(4, [2]uint8{13<<2 | 2, 9<<2 | 3}), readProb: 100},
		{name: "single_unit_row", lumaW: 320, lumaH: 56, xDecC: 1, yDecC: 1,
			params: mk(3, [2]uint8{6<<2 | 1, 2<<2 | 2}, [2]uint8{0, 4 << 2}), readProb: 100, withSkip: true},
		{name: "monochrome", lumaW: 192, lumaH: 136, mono: true,
			params: mk(4, [2]uint8{11<<2 | 2, 0}), readProb: 100},
	}
}

// cdefU8TestRand is a tiny deterministic LCG (libaom test-style).
type cdefU8TestRand struct{ state uint32 }

func (r *cdefU8TestRand) next(n uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % n
}

func TestCDEFPlaneWalkU8MatchesSnapshot(t *testing.T) {
	for _, tc := range cdefU8WalkCases() {
		t.Run(tc.name, func(t *testing.T) {
			runCDEFU8WalkDifferential(t, tc)
		})
	}
}

func runCDEFU8WalkDifferential(t *testing.T, tc cdefU8WalkCase) {
	t.Helper()
	rnd := &cdefU8TestRand{state: 0x5538 ^ uint32(tc.lumaW*31+tc.lumaH)}

	cols := (tc.lumaW + cdef.BlockSize - 1) / cdef.BlockSize
	rows := (tc.lumaH + cdef.BlockSize - 1) / cdef.BlockSize
	unitCount := cols * rows

	indexMap := threading.FrameWorkCDEFIndexMap{
		Stride: uint16(cols),
		Rows:   uint16(rows),
		Index:  make([]uint8, unitCount),
		Read:   make([]bool, unitCount),
	}
	if len(tc.unitIndex) != 0 && len(tc.unitIndex) != unitCount {
		t.Fatalf("exact unit index map length=%d want=%d", len(tc.unitIndex), unitCount)
	}
	if len(tc.unitRead) != 0 && len(tc.unitRead) != unitCount {
		t.Fatalf("exact unit read map length=%d want=%d", len(tc.unitRead), unitCount)
	}
	for i := range unitCount {
		if len(tc.unitIndex) != 0 {
			indexMap.Index[i] = tc.unitIndex[i]
		} else {
			indexMap.Index[i] = uint8(rnd.next(uint32(tc.params.StrengthCount)))
		}
		if len(tc.unitRead) != 0 {
			indexMap.Read[i] = tc.unitRead[i]
		} else {
			indexMap.Read[i] = rnd.next(100) < tc.readProb
		}
	}

	var skipMap *FrameWorkLoopFilterMap
	if tc.withSkip {
		miCols := tc.lumaW / 4
		miRows := tc.lumaH / 4
		m := threading.FrameWorkLoopFilterMap{
			Stride:  uint16(miCols),
			Rows:    uint16(miRows),
			Records: make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows),
		}
		for i := range m.Records {
			m.Records[i].Valid = true
			m.Records[i].SkipTransform = rnd.next(3) == 0
		}
		if len(tc.activeMasks) != 0 {
			if len(tc.activeMasks) != unitCount {
				t.Fatal("active mask count does not match unit grid")
			}
			for y := 0; y < miRows; y++ {
				for x := 0; x < miCols; x++ {
					unit := (y/16)*cols + x/16
					bit := ((y%16)/2)*8 + (x%16)/2
					m.Records[y*miCols+x].SkipTransform = tc.activeMasks[unit]&(uint64(1)<<uint(bit)) == 0
				}
			}
		}
		skipMap = &m
	}

	planes := 1
	if !tc.mono {
		planes = 3
	}
	chromaFiltering := !tc.mono && frameWorkCDEFChromaHasFiltering(tc.params)

	// Shared direction/variance grids, like the production walk.
	wantDirGrid := make([]cdef.DirectionGrid, unitCount)
	wantVarGrid := make([]cdef.VarianceGrid, unitCount)
	gotDirGrid := make([]cdef.DirectionGrid, unitCount)
	gotVarGrid := make([]cdef.VarianceGrid, unitCount)
	var wantDirs, gotDirs cdef.DirectionGrid
	var wantVars, gotVars cdef.VarianceGrid
	var blockStorage [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
	input := make([]uint16, cdef.InputBufferSize)
	unitDst := make([]uint16, cdef.InputBufferSize)
	for by := range cdef.NBlocks {
		for bx := range cdef.NBlocks {
			wantDirs[by][bx] = 5
			gotDirs[by][bx] = 5
			wantVars[by][bx] = 0x12345
			gotVars[by][bx] = 0x12345
		}
	}
	for i := range unitCount {
		for by := range cdef.NBlocks {
			for bx := range cdef.NBlocks {
				wantDirGrid[i][by][bx] = 5
				gotDirGrid[i][by][bx] = 5
				wantVarGrid[i][by][bx] = 0x12345
				gotVarGrid[i][by][bx] = 0x12345
			}
		}
	}

	var wantUnits, gotUnits uint32
	var wantBlocks, gotBlocks uint32
	for plane := 0; plane < planes; plane++ {
		xDec, yDec := 0, 0
		if plane != 0 {
			xDec, yDec = tc.xDecC, tc.yDecC
		}
		width := tc.lumaW >> xDec
		height := tc.lumaH >> yDec
		stride := width + 16 // stride padding with garbage to pin clamping
		pix := make([]byte, stride*height)
		pixelRange := uint32(256)
		if tc.lowRange {
			pixelRange = 16
		}
		for i := range pix {
			pix[i] = byte(rnd.next(pixelRange))
		}
		processPlane := frameWorkCDEFPlaneHasFiltering(tc.params, plane)
		if plane == 0 && !processPlane {
			processPlane = chromaFiltering
		}
		if !processPlane {
			continue
		}

		wantPlane := frame.Plane{Pix: append([]byte(nil), pix...), Stride: stride, Width: width, Height: height}
		scratch := make([]uint16, stride*height)
		src, _, err := frame.LoadSamplePlaneFull(scratch, wantPlane, 1)
		if err != nil {
			t.Fatalf("snapshot load: %v", err)
		}
		wu, wb, err := frameWorkApplyCDEFPlaneRows(tc.params, indexMap, skipMap, cols, rows, 0, rows, src, wantPlane, 1, input, unitDst, blockStorage[:], &wantDirs, &wantVars, wantDirGrid, wantVarGrid, plane, xDec, yDec, 0, chromaFiltering, nil)
		if err != nil {
			t.Fatalf("u16 walk: %v", err)
		}
		wantUnits += wu
		wantBlocks += wb

		gotPlane := frame.Plane{Pix: append([]byte(nil), pix...), Stride: stride, Width: width, Height: height}
		byteScratch := make([]byte, CDEFU8LineScratchLen(width))
		if tc.poisonInput {
			for i := range input {
				input[i] = 0xdeed
			}
		}
		gu, gb, err := frameWorkApplyCDEFPlaneRowsU8(tc.params, indexMap, skipMap, cols, rows, 0, rows, gotPlane, byteScratch, FrameWorkCDEFPostFilterU8BandBoundary{}, input, blockStorage[:], &gotDirs, &gotVars, gotDirGrid, gotVarGrid, plane, xDec, yDec, chromaFiltering, nil)
		if err != nil {
			t.Fatalf("u8 walk: %v", err)
		}
		gotUnits += gu
		gotBlocks += gb

		for i := range wantPlane.Pix {
			if gotPlane.Pix[i] != wantPlane.Pix[i] {
				t.Fatalf("plane %d byte mismatch at row=%d col=%d: got=%d want=%d",
					plane, i/stride, i%stride, gotPlane.Pix[i], wantPlane.Pix[i])
			}
		}
	}
	if gotUnits != wantUnits || gotBlocks != wantBlocks {
		t.Fatalf("count mismatch: got units=%d blocks=%d want units=%d blocks=%d", gotUnits, gotBlocks, wantUnits, wantBlocks)
	}
	for i := range wantDirGrid {
		if gotDirGrid[i] != wantDirGrid[i] {
			t.Fatalf("direction grid %d differs", i)
		}
		if gotVarGrid[i] != wantVarGrid[i] {
			t.Fatalf("variance grid %d differs", i)
		}
	}
	if tc.noPrimary {
		for by := range cdef.NBlocks {
			for bx := range cdef.NBlocks {
				if gotDirs[by][bx] != 5 || gotVars[by][bx] != 0x12345 {
					t.Fatalf("primary-zero luma changed standalone grids at (%d,%d): dir=%d variance=%d", bx, by, gotDirs[by][bx], gotVars[by][bx])
				}
			}
		}
		for i := range unitCount {
			for by := range cdef.NBlocks {
				for bx := range cdef.NBlocks {
					if gotDirGrid[i][by][bx] != 5 || gotVarGrid[i][by][bx] != 0x12345 {
						t.Fatalf("primary-zero unit %d changed grids at (%d,%d): dir=%d variance=%d", i, bx, by, gotDirGrid[i][by][bx], gotVarGrid[i][by][bx])
					}
				}
			}
		}
	}
}

// TestCDEFPlaneWalkU8IsZeroAlloc protects the in-place walk's steady-state
// allocation budget.
func TestCDEFPlaneWalkU8IsZeroAlloc(t *testing.T) {
	tc := cdefU8WalkCases()[0]
	if allocs := cdefPlaneWalkU8Allocs(t, tc); allocs != 0 {
		t.Fatalf("u8 CDEF walk allocates: %v allocs/op", allocs)
	}
}

func TestCDEFPlaneWalkU8SecondaryByteIsZeroAlloc(t *testing.T) {
	if !cdef.U8SecondaryOnlyByteInteriorAvailable() {
		t.Skip("secondary-only byte kernel is unavailable in this build")
	}
	var tc cdefU8WalkCase
	for _, candidate := range cdefU8WalkCases() {
		if candidate.name == "secondary_only_byte_interior_420" {
			tc = candidate
			break
		}
	}
	if tc.name == "" {
		t.Fatal("secondary-only byte walk case is missing")
	}
	if allocs := cdefPlaneWalkU8Allocs(t, tc); allocs != 0 {
		t.Fatalf("secondary-only byte u8 CDEF walk allocates: %v allocs/op", allocs)
	}
}

func cdefPlaneWalkU8Allocs(t *testing.T, tc cdefU8WalkCase) float64 {
	t.Helper()
	rnd := &cdefU8TestRand{state: 0xa110c}
	cols := (tc.lumaW + cdef.BlockSize - 1) / cdef.BlockSize
	rows := (tc.lumaH + cdef.BlockSize - 1) / cdef.BlockSize
	unitCount := cols * rows
	indexMap := threading.FrameWorkCDEFIndexMap{
		Stride: uint16(cols),
		Rows:   uint16(rows),
		Index:  make([]uint8, unitCount),
		Read:   make([]bool, unitCount),
	}
	for i := range unitCount {
		indexMap.Read[i] = true
	}
	width := tc.lumaW
	height := tc.lumaH
	stride := width + 8
	plane := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	for i := range plane.Pix {
		plane.Pix[i] = byte(rnd.next(256))
	}
	byteScratch := make([]byte, CDEFU8LineScratchLen(width))
	input := make([]uint16, cdef.InputBufferSize)
	dirGrid := make([]cdef.DirectionGrid, unitCount)
	varGrid := make([]cdef.VarianceGrid, unitCount)
	var dirs cdef.DirectionGrid
	var vars cdef.VarianceGrid
	var blockStorage [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
	var skipMap *FrameWorkLoopFilterMap
	if len(tc.activeMasks) != 0 {
		miCols, miRows := width/4, height/4
		records := make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows)
		for y := 0; y < miRows; y++ {
			for x := 0; x < miCols; x++ {
				unit, bit := (y/16)*cols+x/16, ((y%16)/2)*8+(x%16)/2
				records[y*miCols+x].Valid = true
				records[y*miCols+x].SkipTransform = tc.activeMasks[unit]&(uint64(1)<<uint(bit)) == 0
			}
		}
		skipMap = &FrameWorkLoopFilterMap{Records: records, Stride: uint16(miCols), Rows: uint16(miRows)}
	}
	return testing.AllocsPerRun(8, func() {
		if _, _, err := frameWorkApplyCDEFPlaneRowsU8(tc.params, indexMap, skipMap, cols, rows, 0, rows, plane, byteScratch, FrameWorkCDEFPostFilterU8BandBoundary{}, input, blockStorage[:], &dirs, &vars, dirGrid, varGrid, 0, 0, 0, true, nil); err != nil {
			t.Fatal(err)
		}
	})
}

// TestCDEFUnitFilterMaskMatchesPerBlock checks the cached per-unit mask path
// against frameWorkCDEFBlockPositionsFiltered for every unit of random skip
// maps, including units overhanging the map and partial (edge) unit extents.
func TestCDEFUnitFilterMaskMatchesPerBlock(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for iter := 0; iter < 200; iter++ {
		miCols := 1 + rng.Intn(70)
		miRows := 1 + rng.Intn(70)
		recs := make([]threading.FrameWorkLoopFilterBlockRecord, miCols*miRows)
		density := rng.Intn(4)
		for i := range recs {
			recs[i].Valid = rng.Intn(20) != 0
			recs[i].SkipTransform = rng.Intn(4) < density
		}
		skipMap := &FrameWorkLoopFilterMap{Records: recs, Stride: uint16(miCols), Rows: uint16(miRows)}
		cols := (miCols*4 + 63) / 64
		rows := (miRows*4 + 63) / 64
		var cache frameWorkCDEFUnitMaskCache
		if !cache.resetWithCompact(skipMap, nil, nil, 0, rows, cols, false) {
			t.Fatal("cache rejected a small band")
		}
		var a, b [cdef.NBlocks * cdef.NBlocks]cdef.BlockPosition
		for ur := 0; ur < rows; ur++ {
			for uc := 0; uc < cols; uc++ {
				for _, dims := range [][4]int{{64, 64, 8, 8}, {32, 32, 4, 4}, {64, 32, 8, 4}, {1 + rng.Intn(64), 1 + rng.Intn(64), 8, 8}, {1 + rng.Intn(32), 1 + rng.Intn(32), 4, 4}} {
					want := frameWorkCDEFBlockPositionsFiltered(a[:], dims[0], dims[1], dims[2], dims[3], skipMap, ur, uc)
					got := frameWorkCDEFBlockPositionsFromMask(b[:], cache.mask(ur, uc), dims[0], dims[1], dims[2], dims[3])
					if len(got) != len(want) {
						t.Fatalf("iter %d unit (%d,%d) dims %v: %d blocks, want %d", iter, ur, uc, dims, len(got), len(want))
					}
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("iter %d unit (%d,%d) dims %v: block %d = %+v, want %+v", iter, ur, uc, dims, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestPostFilterPrivateQueriesPreserveValueSnapshots(t *testing.T) {
	const all = FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF |
		FrameWorkPostFilterSuperRes | FrameWorkPostFilterLoopRestoration | FrameWorkPostFilterFilmGrain
	masks := []FrameWorkPostFilterStage{0, FrameWorkPostFilterLoopFilter, FrameWorkPostFilterCDEF, all, 0x80, 0xff}
	for active := FrameWorkPostFilterStage(0); active <= all; active++ {
		var event Event
		if active.Has(FrameWorkPostFilterLoopFilter) {
			event.LoopFilter.LevelY[1] = 1
		}
		if active.Has(FrameWorkPostFilterCDEF) {
			event.CDEF.Bits = 1
		}
		event.FrameSize.SuperResEnabled = active.Has(FrameWorkPostFilterSuperRes)
		if active.Has(FrameWorkPostFilterLoopRestoration) {
			event.Restoration.Type[2] = parser.RestorationSGRProj
		}
		event.FilmGrain.Apply = active.Has(FrameWorkPostFilterFilmGrain)
		for _, completed := range masks {
			ctx := FrameWorkPostFilterContext{Event: event, ReferenceCount: 7,
				ExecutedTileWork: true, completedPostFilters: completed, detachedPostFilterOutput: true}
			before := ctx
			if got := ctx.ActivePostFilters(); got != active {
				t.Fatalf("active=%#x want=%#x", got, active)
			}
			if got := frameWorkActivePostFilters(&ctx); got != active {
				t.Fatalf("private active=%#x want=%#x", got, active)
			}
			wantRemaining := active &^ completed
			if ctx.RemainingPostFilters() != wantRemaining || frameWorkRemainingPostFilters(&ctx) != wantRemaining {
				t.Fatalf("remaining differs: active=%#x completed=%#x", active, completed)
			}
			for _, mark := range masks {
				got := ctx.WithCompletedPostFilters(mark)
				private := ctx
				frameWorkMarkCompletedPostFilters(&private, mark)
				want := before
				want.completedPostFilters |= mark & active
				if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(private, want) {
					t.Fatalf("completion changed fields: active=%#x completed=%#x mark=%#x", active, completed, mark)
				}
			}
			if !reflect.DeepEqual(ctx, before) {
				t.Fatal("value methods or read-only private queries mutated the original context")
			}
		}
	}
}

func TestPostFilterPrivateQueriesObserveEventChanges(t *testing.T) {
	ctx := FrameWorkPostFilterContext{Event: Event{CDEF: parser.CDEFParams{Bits: 1}}}
	snapshot := ctx.WithCompletedPostFilters(FrameWorkPostFilterCDEF)
	ctx.Event.CDEF.Bits = 0
	ctx.Event.FrameSize.SuperResEnabled = true
	if frameWorkActivePostFilters(&ctx) != FrameWorkPostFilterSuperRes ||
		frameWorkRemainingPostFilters(&ctx) != FrameWorkPostFilterSuperRes {
		t.Fatal("private query reused a stale event mask")
	}
	if snapshot.ActivePostFilters() != FrameWorkPostFilterCDEF || snapshot.RemainingPostFilters() != 0 {
		t.Fatal("editing the original event changed the returned value snapshot")
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				ARCoeffShift:  6,
				ARCoeffsCb:    [parser.MaxFilmGrainUVCoeffs]int8{0: 64},
			},
		},
		Output: output,
	}
	luma := testFrameWorkFilmGrainLumaTemplate(32)
	result, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), luma, filmgrain.ChromaPlaneCb)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Plane != filmgrain.ChromaPlaneCb ||
		result.Width != filmgrain.ChromaSubsampledGrainWidth ||
		result.Height != filmgrain.ChromaSubsampledGrainHeight ||
		result.Stride != filmgrain.ChromaGrainWidth {
		t.Fatalf("result=%+v", result)
	}
	if result.Grain[0] != 12 || result.Grain[15] != 13 || result.Grain[3*result.Stride+3] != 46 {
		t.Fatalf("cb grain samples=%d %d %d", result.Grain[0], result.Grain[15], result.Grain[3*result.Stride+3])
	}

	cr, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), luma, filmgrain.ChromaPlaneCr)
	if err != nil {
		t.Fatal(err)
	}
	if !cr.Active || cr.Plane != filmgrain.ChromaPlaneCr || cr.Grain[0] != 5 || cr.Grain[15] != 3 {
		t.Fatalf("cr result=%+v samples=%d %d", cr, cr.Grain[0], cr.Grain[15])
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainSkipsInactive(t *testing.T) {
	result, err := (&FrameWorkPostFilterContext{}).GenerateFilmGrainChromaGrain(nil, nil, filmgrain.ChromaPlaneCb)
	if err != nil {
		t.Fatal(err)
	}
	if result.Active || result.Plane != 0 || result.Grain != nil || result.Width != 0 || result.Height != 0 || result.Stride != 0 {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainRejectsInvalidPlane(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ARCoeffShift: 6,
		}},
		Output: output,
	}
	if _, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), nil, 0); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("GenerateFilmGrainChromaGrain err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumYPoints:   1,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ARCoeffShift: 6,
		}},
		Output: output,
	}
	luma := testFrameWorkFilmGrainLumaTemplate(32)
	if _, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples-1), luma, filmgrain.ChromaPlaneCb); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("short chroma scratch err=%v want %v", err, frame.ErrShortBuffer)
	}
	if _, err := ctx.GenerateFilmGrainChromaGrain(make([]int16, filmgrain.ChromaGrainSamples), luma[:filmgrain.LumaGrainSamples-1], filmgrain.ChromaPlaneCb); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("short luma scratch err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainChromaGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				ARCoeffLag:    1,
				ARCoeffShift:  6,
				ARCoeffsCb:    [parser.MaxFilmGrainUVCoeffs]int8{3: 64, 4: 32},
			},
		},
		Output: output,
	}
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	luma := testFrameWorkFilmGrainLumaTemplate(32)
	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.GenerateFilmGrainChromaGrain(grain, luma, filmgrain.ChromaPlaneCb)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active || len(result.Grain) != filmgrain.ChromaGrainSamples {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateFilmGrainChromaGrain allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainChromaRowBuffers(int(plan.Planes[filmgrain.ChromaPlaneCb].Width), filmgrain.LumaBlockSize>>1, int(plan.Planes[filmgrain.ChromaPlaneCb].Stride), 100)
	lumaStride := frameWorkFilmGrainSampleStride(output.Layout.YStride, output.Layout.BytesPerSample)
	luma := testFrameWorkFilmGrainSampleBuffer(output.Y.Width, output.Y.Height, lumaStride, 96)
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	setTestFrameWorkFilmGrainChromaRowGrain(t, grain, 0xd9, true, true, 0, 0, 0, 0, 20)
	lut := testFrameWorkFilmGrainScaling(64)

	result, err := ctx.ApplyFilmGrainChromaRow(dst, src, luma, grain, lut[:], filmgrain.ChromaPlaneCb, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active ||
		result.Plane != filmgrain.ChromaPlaneCb ||
		result.Row != 0 ||
		result.Width != plan.Planes[filmgrain.ChromaPlaneCb].Width ||
		result.Height != filmgrain.LumaBlockSize>>1 ||
		result.Stride != plan.Planes[filmgrain.ChromaPlaneCb].Stride ||
		result.LumaStride != uint32(lumaStride) {
		t.Fatalf("result=%+v", result)
	}
	if got := dst[0]; got != 105 {
		t.Fatalf("chroma row sample=%d want 105", got)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowSkipsInactive(t *testing.T) {
	result, err := (&FrameWorkPostFilterContext{}).ApplyFilmGrainChromaRow(nil, nil, nil, nil, nil, filmgrain.ChromaPlaneCb, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkFilmGrainPostFilterChromaRow{}) {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowRejectsInvalidPlane(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ScalingShift: 8,
		}},
		Output: output,
	}
	if _, err := ctx.ApplyFilmGrainChromaRow(nil, nil, nil, nil, nil, 0, 0); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("ApplyFilmGrainChromaRow err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowRejectsInvalidRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ScalingShift: 8,
		}},
		Output: output,
	}
	if _, err := ctx.ApplyFilmGrainChromaRow(nil, nil, nil, nil, nil, filmgrain.ChromaPlaneCb, 1); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("ApplyFilmGrainChromaRow err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{FilmGrain: parser.FilmGrainParams{
			Apply:        true,
			BitDepth:     8,
			NumCbPoints:  1,
			NumCrPoints:  1,
			ScalingShift: 8,
			CbMult:       64,
			CrMult:       64,
		}},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	height := filmgrain.LumaBlockSize >> 1
	dst, src := testFrameWorkFilmGrainChromaRowBuffers(int(plan.Planes[filmgrain.ChromaPlaneCb].Width), height, int(plan.Planes[filmgrain.ChromaPlaneCb].Stride), 100)
	lumaStride := frameWorkFilmGrainSampleStride(output.Layout.YStride, output.Layout.BytesPerSample)
	luma := testFrameWorkFilmGrainSampleBuffer(output.Y.Width, output.Y.Height, lumaStride, 96)
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	lut := testFrameWorkFilmGrainScaling(64)
	need := (height-1)*int(plan.Planes[filmgrain.ChromaPlaneCb].Stride) + int(plan.Planes[filmgrain.ChromaPlaneCb].Width)
	if _, err := ctx.ApplyFilmGrainChromaRow(dst[:need-1], src, luma, grain, lut[:], filmgrain.ChromaPlaneCb, 0); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyFilmGrainChromaRow err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainChromaRowAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainChromaRowBuffers(int(plan.Planes[filmgrain.ChromaPlaneCb].Width), filmgrain.LumaBlockSize>>1, int(plan.Planes[filmgrain.ChromaPlaneCb].Stride), 100)
	lumaStride := frameWorkFilmGrainSampleStride(output.Layout.YStride, output.Layout.BytesPerSample)
	luma := testFrameWorkFilmGrainSampleBuffer(output.Y.Width, output.Y.Height, lumaStride, 96)
	grain := make([]int16, filmgrain.ChromaGrainSamples)
	setTestFrameWorkFilmGrainChromaRowGrain(t, grain, 0xd9, true, true, 0, 0, 0, 0, 20)
	lut := testFrameWorkFilmGrainScaling(64)

	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.ApplyFilmGrainChromaRow(dst, src, luma, grain, lut[:], filmgrain.ChromaPlaneCb, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyFilmGrainChromaRow allocated: %f", allocs)
	}
}

func testFrameWorkFilmGrainLumaTemplate(value int16) []int16 {
	grain := make([]int16, filmgrain.LumaGrainSamples)
	for i := range grain {
		grain[i] = value
	}
	return grain
}

func testFrameWorkFilmGrainChromaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func testFrameWorkFilmGrainSampleBuffer(width int, height int, stride int, value uint16) []uint16 {
	samples := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			samples[y*stride+x] = value
		}
	}
	return samples
}

func testFrameWorkFilmGrainScaling(value uint8) [filmgrain.ScalingLUTSize]uint8 {
	var lut [filmgrain.ScalingLUTSize]uint8
	for i := range lut {
		lut[i] = value
	}
	return lut
}

func setTestFrameWorkFilmGrainChromaRowGrain(t *testing.T, grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	shiftX := frameWorkFilmGrainSubsamplingShift(subsamplingX)
	shiftY := frameWorkFilmGrainSubsamplingShift(subsamplingY)
	col := 3 + (filmgrain.LumaOverlapSamples>>shiftX)*(3+int(offset>>4)) + x + (filmgrain.LumaBlockSize>>shiftX)*blockCol
	row := 3 + (filmgrain.LumaOverlapSamples>>shiftY)*(3+int(offset&0x0f)) + y + (filmgrain.LumaBlockSize>>shiftY)*blockRow
	width := filmgrain.ChromaGrainWidth
	if subsamplingX {
		width = filmgrain.ChromaSubsampledGrainWidth
	}
	height := filmgrain.ChromaGrainHeight
	if subsamplingY {
		height = filmgrain.ChromaSubsampledGrainHeight
	}
	if col < 0 || col >= width || row < 0 || row >= height {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*filmgrain.ChromaGrainWidth+col] = value
}

func TestFilmGrainDirectPublicationMatchesStaged(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		sx, sy, mono, fromLuma bool
		active                 uint8
	}{
		{"444-all", false, false, false, false, 7},
		{"444-from-luma", false, false, false, true, 7},
		{"422-all", true, false, false, false, 7},
		{"420-from-luma", true, true, false, true, 7},
		{"mono", true, true, true, false, 1},
		{"444-luma-only", false, false, false, false, 1},
		{"444-cb-only", false, false, false, false, 2},
	} {
		for _, bd := range []uint8{8, 10, 12} {
			// Workers 3 runs the grain stripes across a pool (dav1d's
			// delayed film grain task); stripe order must not matter.
			for _, workers := range []int{1, 3} {
				t.Run(fmt.Sprintf("%s-%dbit-w%d", tc.name, bd, workers), func(t *testing.T) {
					ctx := testFilmGrainDirectContext(t, frame.Format{Width: 98, Height: 65, BitDepth: bd, SubsamplingX: tc.sx, SubsamplingY: tc.sy, MonoChrome: tc.mono, Align: 64}, tc.active, tc.fromLuma)
					ctx.Output.ExtendBorders()
					before := [3][]byte{bytes.Clone(ctx.Output.Y.Pix), bytes.Clone(ctx.Output.U.Pix), bytes.Clone(ctx.Output.V.Pix)}
					size, err := ctx.FilmGrainPostFilterScratchLen()
					if err != nil {
						t.Fatal(err)
					}
					legacyReq := testFrameWorkFilmGrainPostFilterRequest(size)
					var legacyView frame.Frame
					legacyReq.OutputView = &legacyView
					legacy, err := ctx.BindFilmGrainPostFilterOutput(legacyReq)
					if err != nil {
						t.Fatal(err)
					}
					legacyCtx := ctx
					legacyCtx.Output = legacy
					legacyCtx.detachedPostFilterOutput = true
					wantResult, err := legacyCtx.ApplyFilmGrainPostFilter(legacyReq)
					if err != nil {
						t.Fatal(err)
					}

					req := testFrameWorkFilmGrainPostFilterRequest(size)
					var view frame.Frame
					req.OutputView = &view
					for _, samples := range [][]uint16{req.LumaSamples, req.ChromaSamples[0], req.ChromaSamples[1]} {
						for i := range samples {
							samples[i] = 0xabad
						}
					}
					par := &FrameWorkPostFilterParallel{Workers: workers}
					defer par.Close()
					ctx.Parallel = par
					next, got, result, err := ctx.ApplySupportedPostFiltersForPublication(FrameWorkPostFilterRequest{FilmGrain: req})
					if err != nil {
						t.Fatal(err)
					}
					if next.Output != ctx.Output || got != &view || result.FilmGrain.Output != got || next.RemainingPostFilters() != 0 {
						t.Fatal("publication ownership/result changed")
					}
					if result.FilmGrain.Plan != wantResult.Plan || result.FilmGrain.LumaRows != wantResult.LumaRows || result.FilmGrain.ChromaRows != wantResult.ChromaRows || result.FilmGrain.OutputSize != wantResult.OutputSize {
						t.Fatal("result geometry changed")
					}
					for i, p := range []frame.Plane{got.Y, got.U, got.V} {
						want := []frame.Plane{legacy.Y, legacy.U, legacy.V}[i]
						if !bytes.Equal(p.Pix, want.Pix) {
							t.Fatalf("plane %d full bytes/padding differ", i)
						}
						original := []frame.Plane{ctx.Output.Y, ctx.Output.U, ctx.Output.V}[i]
						if !bytes.Equal(original.Pix, before[i]) {
							t.Fatalf("reference plane %d modified", i)
						}
					}
					// 10/12-bit planes have no direct []uint16 view and stage.
					for _, samples := range [][]uint16{req.LumaSamples, req.ChromaSamples[0], req.ChromaSamples[1]} {
						for _, v := range samples {
							if bd == 8 && v != 0xabad {
								t.Fatal("direct path used staging scratch")
							}
						}
					}
					if tc.name == "444-all" {
						if allocs := testing.AllocsPerRun(20, func() {
							_, _, _, err := ctx.ApplySupportedPostFiltersForPublication(FrameWorkPostFilterRequest{FilmGrain: req})
							if err != nil {
								panic(err)
							}
						}); allocs != 0 {
							t.Fatalf("steady publication allocs=%v", allocs)
						}
					}
				})
			}
		}
	}
}

func TestFilmGrainDirectPublicationFallbacks(t *testing.T) {
	for _, name := range []string{"alias", "odd-subsampled-width", "allocated-view", "noncanonical-plane"} {
		t.Run(name, func(t *testing.T) {
			format := frame.Format{Width: 34, Height: 33, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
			if name == "odd-subsampled-width" {
				format.Width = 33
			}
			ctx := testFilmGrainDirectContext(t, format, 7, true)
			if name == "noncanonical-plane" {
				ctx.Output.Y.Width--
			}
			size, err := ctx.FilmGrainPostFilterScratchLen()
			if err != nil {
				t.Fatal(err)
			}
			req := testFrameWorkFilmGrainPostFilterRequest(size)
			var view frame.Frame
			req.OutputView = &view
			if name == "allocated-view" {
				req.OutputView = nil
			}
			if name == "alias" {
				req.OutputFrame = ctx.Output.Y.Pix[:ctx.Output.Layout.Size]
			}
			for i := range req.LumaSamples {
				req.LumaSamples[i] = 0xabad
			}
			_, _, err = frameWorkApplyFilmGrainForPublication(&ctx, req)
			if err != nil {
				t.Fatal(err)
			}
			if req.LumaSamples[0] == 0xabad {
				t.Fatal("legacy fallback did not run")
			}
		})
	}
}

func TestFilmGrainDirectPublicationErrorOrdering(t *testing.T) {
	ctx := testFilmGrainDirectContext(t, frame.Format{Width: 34, Height: 33, BitDepth: 8, Align: 32}, 7, false)
	size, err := ctx.FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size)
	var view frame.Frame
	req.OutputView = &view
	req.LumaSamples = req.LumaSamples[:len(req.LumaSamples)-1]
	before := bytes.Clone(ctx.Output.Y.Pix)
	_, _, _, err = ctx.ApplySupportedPostFiltersForPublication(FrameWorkPostFilterRequest{FilmGrain: req})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("err=%v", err)
	}
	if !bytes.Equal(before, ctx.Output.Y.Pix) {
		t.Fatal("short scratch mutated reference")
	}
}

func testFilmGrainDirectContext(t *testing.T, format frame.Format, active uint8, fromLuma bool) FrameWorkPostFilterContext {
	t.Helper()
	output := testFrameWorkCDEFFrame(t, format)
	for plane, p := range []frame.Plane{output.Y, output.U, output.V} {
		for i := range p.Pix {
			p.Pix[i] = byte(i*37 + plane*19)
		}
		if format.BitDepth > 8 {
			for y := 0; y < p.Height; y++ {
				for x := 0; x < p.Width; x++ {
					setTestFrameWorkFilmGrainPlaneSample(p, output.Layout.BytesPerSample, x, y, uint16((x*19+y*31+plane*7)&1023))
				}
			}
		}
	}
	fg := parser.FilmGrainParams{ParamsPresent: true, Apply: true, Seed: 0x4362, BitDepth: format.BitDepth, ScalingShift: 8, ARCoeffShift: 6, Overlap: true, ClipToRestrictedRange: true,
		ChromaScalingFromLuma: fromLuma, CbMult: 140, CrMult: 118, CbLumaMult: 146, CrLumaMult: 126, CbOffset: 260, CrOffset: 250}
	if active&1 != 0 {
		fg.NumYPoints = 2
		fg.YPoints[0] = [2]uint8{0, 255}
		fg.YPoints[1] = [2]uint8{255, 32}
	}
	if active&2 != 0 && !fromLuma {
		fg.NumCbPoints = 2
		fg.CbPoints[0] = [2]uint8{0, 64}
		fg.CbPoints[1] = [2]uint8{255, 250}
	}
	if active&4 != 0 && !fromLuma {
		fg.NumCrPoints = 2
		fg.CrPoints[0] = [2]uint8{0, 250}
		fg.CrPoints[1] = [2]uint8{255, 64}
	}
	ctx := FrameWorkPostFilterContext{Event: Event{FilmGrain: fg}, Output: output}
	if _, err := ctx.FilmGrainPostFilterPlan(); err != nil {
		t.Fatal(fmt.Errorf("fixture: %w", err))
	}
	return ctx
}

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				ARCoeffShift:  6,
			},
		},
		Output: output,
	}
	result, err := ctx.GenerateFilmGrainLumaGrain(make([]int16, filmgrain.LumaGrainSamples))
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Width != filmgrain.LumaGrainWidth ||
		result.Height != filmgrain.LumaGrainHeight ||
		result.Stride != filmgrain.LumaGrainWidth {
		t.Fatalf("result=%+v", result)
	}
	if result.Grain[0] != -29 || result.Grain[15] != 26 || result.Grain[3*result.Stride+3] != -14 {
		t.Fatalf("grain samples=%d %d %d", result.Grain[0], result.Grain[15], result.Grain[3*result.Stride+3])
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrainSkipsInactive(t *testing.T) {
	result, err := (&FrameWorkPostFilterContext{}).GenerateFilmGrainLumaGrain(nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Active || result.Grain != nil || result.Width != 0 || result.Height != 0 || result.Stride != 0 {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrainRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				Apply:        true,
				BitDepth:     8,
				NumYPoints:   1,
				ARCoeffShift: 6,
			},
		},
		Output: output,
	}
	_, err := ctx.GenerateFilmGrainLumaGrain(make([]int16, filmgrain.LumaGrainSamples-1))
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("GenerateFilmGrainLumaGrain err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextGenerateFilmGrainLumaGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				ARCoeffLag:    1,
				ARCoeffShift:  6,
				ARCoeffsY:     [parser.MaxFilmGrainYCoeffs]int8{3: 64},
			},
		},
		Output: output,
	}
	grain := make([]int16, filmgrain.LumaGrainSamples)
	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.GenerateFilmGrainLumaGrain(grain)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active || len(result.Grain) != filmgrain.LumaGrainSamples {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("GenerateFilmGrainLumaGrain allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainLumaRowBuffers(int(plan.Planes[0].Width), filmgrain.LumaBlockSize, int(plan.Planes[0].Stride), 100)
	grain := make([]int16, filmgrain.LumaGrainSamples)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xec, 0, 0, 0, 0, 20)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xd9, 1, 0, 0, 0, 100)

	result, err := ctx.ApplyFilmGrainLumaRow(dst, src, grain, luts.LUTs[0][:], 0)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Row != 0 || result.Width != 34 || result.Height != filmgrain.LumaBlockSize || result.Stride != plan.Planes[0].Stride {
		t.Fatalf("result=%+v", result)
	}
	if got := dst[32]; got != 124 {
		t.Fatalf("luma row sample=%d want 124", got)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowSkipsInactive(t *testing.T) {
	result, err := (&FrameWorkPostFilterContext{}).ApplyFilmGrainLumaRow(nil, nil, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkFilmGrainPostFilterLumaRow{}) {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowRejectsInvalidRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				Apply:        true,
				BitDepth:     8,
				NumYPoints:   1,
				ScalingShift: 8,
			},
		},
		Output: output,
	}
	if _, err := ctx.ApplyFilmGrainLumaRow(nil, nil, nil, nil, 1); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("ApplyFilmGrainLumaRow err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowRejectsShortScratch(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				Apply:        true,
				BitDepth:     8,
				NumYPoints:   1,
				ScalingShift: 8,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainLumaRowBuffers(int(plan.Planes[0].Width), filmgrain.LumaBlockSize, int(plan.Planes[0].Stride), 100)
	grain := make([]int16, filmgrain.LumaGrainSamples)
	var lut [filmgrain.ScalingLUTSize]uint8
	need := (filmgrain.LumaBlockSize-1)*int(plan.Planes[0].Stride) + int(plan.Planes[0].Width)
	if _, err := ctx.ApplyFilmGrainLumaRow(dst[:need-1], src, grain, lut[:], 0); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyFilmGrainLumaRow err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextApplyFilmGrainLumaRowAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	dst, src := testFrameWorkFilmGrainLumaRowBuffers(int(plan.Planes[0].Width), filmgrain.LumaBlockSize, int(plan.Planes[0].Stride), 100)
	grain := make([]int16, filmgrain.LumaGrainSamples)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xec, 0, 0, 0, 0, 20)
	setTestFrameWorkFilmGrainLumaRowGrain(t, grain, 0xd9, 1, 0, 0, 0, 100)

	allocs := testing.AllocsPerRun(1000, func() {
		result, err := ctx.ApplyFilmGrainLumaRow(dst, src, grain, luts.LUTs[0][:], 0)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Active {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyFilmGrainLumaRow allocated: %f", allocs)
	}
}

func testFrameWorkFilmGrainLumaRowBuffers(width int, height int, stride int, value uint16) ([]uint16, []uint16) {
	dst := make([]uint16, stride*height)
	src := make([]uint16, stride*height)
	for y := range height {
		for x := range width {
			src[y*stride+x] = value
		}
	}
	return dst, src
}

func setTestFrameWorkFilmGrainLumaRowGrain(t *testing.T, grain []int16, offset uint8, blockCol int, blockRow int, x int, y int, value int16) {
	t.Helper()
	col := 3 + filmgrain.LumaOverlapSamples*(3+int(offset>>4)) + x + filmgrain.LumaBlockSize*blockCol
	row := 3 + filmgrain.LumaOverlapSamples*(3+int(offset&0x0f)) + y + filmgrain.LumaBlockSize*blockRow
	if col < 0 || col >= filmgrain.LumaGrainWidth || row < 0 || row >= filmgrain.LumaGrainHeight {
		t.Fatalf("grain coordinate out of range: col=%d row=%d", col, row)
	}
	grain[row*filmgrain.LumaGrainWidth+col] = value
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanSkipsInactive(t *testing.T) {
	plan, err := (&FrameWorkPostFilterContext{}).FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if plan != (FrameWorkFilmGrainPostFilterPlan{}) {
		t.Fatalf("plan=%+v want zero", plan)
	}
	size, err := (&FrameWorkPostFilterContext{}).FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v want zero", size)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanReportsLumaInputs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	event := Event{
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent:         true,
			Apply:                 true,
			Update:                true,
			Seed:                  0x1234,
			BitDepth:              8,
			NumYPoints:            1,
			YPoints:               [parser.MaxFilmGrainYPoints][2]uint8{{16, 32}},
			ScalingShift:          9,
			ARCoeffLag:            1,
			ARCoeffShift:          8,
			GrainScaleShift:       1,
			Overlap:               true,
			ClipToRestrictedRange: true,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Seed != 0x1234 || plan.BitDepth != 8 || !plan.Overlap || !plan.ClipToRestrictedRange {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.Planes[0] != (FrameWorkFilmGrainPostFilterPlanePlan{Active: true, ScalingPoints: 1, ARCoeffs: 4, Width: 64, Height: 32, Stride: 64}) {
		t.Fatalf("luma plane=%+v", plan.Planes[0])
	}
	if plan.Planes[1].Active || plan.Planes[2].Active {
		t.Fatalf("chroma planes=%+v %+v", plan.Planes[1], plan.Planes[2])
	}
	size, err := ctx.FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size.ScalingPoints != [3]int{1, 0, 0} || size.ARCoeffs != [3]int{4, 0, 0} {
		t.Fatalf("scratch=%+v", size)
	}
	if size.LumaGrain != filmgrain.LumaGrainSamples ||
		size.LumaSamples != int(plan.Planes[0].Stride)*int(plan.Planes[0].Height) ||
		size.LumaLine != filmgrain.LumaOverlapSamples*int(plan.Planes[0].Stride) ||
		size.LumaColumn != filmgrain.LumaColumnScratchRows*filmgrain.LumaOverlapSamples {
		t.Fatalf("luma scratch=%+v", size)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanReportsChromaScalingFromLuma(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	event := Event{
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent:         true,
			Apply:                 true,
			Seed:                  0x4433,
			BitDepth:              8,
			ChromaScalingFromLuma: true,
			NumYPoints:            1,
			YPoints:               [parser.MaxFilmGrainYPoints][2]uint8{{4, 12}},
			ARCoeffLag:            1,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.ChromaScalingFromLuma || !plan.Planes[1].Active || !plan.Planes[2].Active {
		t.Fatalf("plan=%+v", plan)
	}
	if int(plan.Planes[1].Width) != 32 || int(plan.Planes[1].Height) != 16 || plan.Planes[1].ARCoeffs != 5 {
		t.Fatalf("cb plane=%+v", plan.Planes[1])
	}
	if int(plan.Planes[2].Width) != 32 || int(plan.Planes[2].Height) != 16 || plan.Planes[2].ARCoeffs != 5 {
		t.Fatalf("cr plane=%+v", plan.Planes[2])
	}
	size, err := ctx.FilmGrainPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size.ScalingPoints != [3]int{1, 0, 0} || size.ARCoeffs != [3]int{4, 5, 5} {
		t.Fatalf("scratch=%+v", size)
	}
	if size.ChromaGrain != [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples} {
		t.Fatalf("chroma grain scratch=%+v", size.ChromaGrain)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanReportsSampleStride(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 48, Height: 16, BitDepth: 10, Align: 64})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      10,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
			},
		},
		Output: output,
	}
	plan, err := ctx.FilmGrainPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if int(plan.Planes[0].Width) != 48 || int(plan.Planes[0].Stride) != 64 {
		t.Fatalf("luma plane=%+v", plan.Planes[0])
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterScalingLUTs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
				NumYPoints:    2,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{10, 20}, {20, 40}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 7}},
				NumCrPoints:   2,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 100}, {10, 0}},
			},
		},
		Output: output,
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	if !luts.Active {
		t.Fatalf("luts=%+v", luts)
	}
	if luts.LUTs[0][0] != 20 || luts.LUTs[0][15] != 30 || luts.LUTs[0][255] != 40 {
		t.Fatalf("y lut samples=%d %d %d", luts.LUTs[0][0], luts.LUTs[0][15], luts.LUTs[0][255])
	}
	if luts.LUTs[1][0] != 7 || luts.LUTs[1][255] != 7 {
		t.Fatalf("cb lut samples=%d %d", luts.LUTs[1][0], luts.LUTs[1][255])
	}
	if luts.LUTs[2][5] != 50 || luts.LUTs[2][255] != 0 {
		t.Fatalf("cr lut samples=%d %d", luts.LUTs[2][5], luts.LUTs[2][255])
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterScalingLUTsCopiesLumaForChroma(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent:         true,
				Apply:                 true,
				BitDepth:              8,
				ChromaScalingFromLuma: true,
				NumYPoints:            2,
				YPoints:               [parser.MaxFilmGrainYPoints][2]uint8{{10, 20}, {20, 40}},
			},
		},
		Output: output,
	}
	luts, err := ctx.FilmGrainPostFilterScalingLUTs()
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range []int{0, 15, 255} {
		if luts.LUTs[1][x] != luts.LUTs[0][x] || luts.LUTs[2][x] != luts.LUTs[0][x] {
			t.Fatalf("lut[%d] y=%d cb=%d cr=%d", x, luts.LUTs[0][x], luts.LUTs[1][x], luts.LUTs[2][x])
		}
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterScalingLUTsRejectsInvalidPoints(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
				NumYPoints:    2,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{10, 20}, {10, 40}},
			},
		},
		Output: output,
	}
	if _, err := ctx.FilmGrainPostFilterScalingLUTs(); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("FilmGrainPostFilterScalingLUTs err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersCompletesNoOpFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	output.Y.Pix[0] = 0x5a
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v", size.FilmGrain)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain || !result.FilmGrain.NoOp || !result.FilmGrain.Plan.Active {
		t.Fatalf("result=%+v", result)
	}
	if got := next.RemainingPostFilters(); got != 0 {
		t.Fatalf("remaining=%b want 0", got)
	}
	if output.Y.Pix[0] != 0x5a {
		t.Fatalf("output sample=%d want 0x5a", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsChromaFilmGrainShortScratchBeforeMutation(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	output.Y.Pix[0] = 0x44
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              32,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{
				Damping:       5,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			},
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 16}},
				ScalingShift:  8,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.FilmGrain.LumaSamples == 0 ||
		size.FilmGrain.ChromaGrain != [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples} ||
		size.FilmGrain.ChromaSamples[0] == 0 ||
		size.FilmGrain.ChromaSamples[1] == 0 {
		t.Fatalf("film grain scratch=%+v", size.FilmGrain)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if next.RemainingPostFilters() != ctx.RemainingPostFilters() || result != (FrameWorkPostFilterResult{}) {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if output.Y.Pix[0] != 0x44 {
		t.Fatalf("output sample=%d want 0x44", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesLumaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}
	before := testCopyFrameWorkCDEFPlane(output.Y)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.FilmGrain.LumaGrain != filmgrain.LumaGrainSamples || size.FilmGrain.LumaSamples == 0 {
		t.Fatalf("film grain scratch=%+v", size.FilmGrain)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	next, result, err := ctx.ApplySupportedPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.NoOp ||
		result.FilmGrain.LumaRows != 1 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("film grain did not change luma output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesChromaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 96
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			output.U.Pix[y*output.U.Stride+x] = 100
			output.V.Pix[y*output.V.Stride+x] = 110
		}
	}
	beforeY := testCopyFrameWorkCDEFPlane(output.Y)
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 1 ||
		result.FilmGrain.ChromaRows != [2]uint16{1, 1} ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, beforeY) {
		t.Fatal("film grain did not change luma output")
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("film grain did not change cb output")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("film grain did not change cr output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesHighBitDepthChromaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 64})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			setTestFrameWorkFilmGrainPlaneSample(output.Y, output.Layout.BytesPerSample, x, y, 384)
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			setTestFrameWorkFilmGrainPlaneSample(output.U, output.Layout.BytesPerSample, x, y, 400)
			setTestFrameWorkFilmGrainPlaneSample(output.V, output.Layout.BytesPerSample, x, y, 420)
		}
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      10,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 1 ||
		result.FilmGrain.ChromaRows != [2]uint16{1, 1} ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkFilmGrainPlaneChangedFromValue(output.U, output.Layout.BytesPerSample, 400) {
		t.Fatal("film grain did not change high-bit-depth cb output")
	}
	if !testFrameWorkFilmGrainPlaneChangedFromValue(output.V, output.Layout.BytesPerSample, 420) {
		t.Fatal("film grain did not change high-bit-depth cr output")
	}
	for _, sample := range []uint16{
		getTestFrameWorkFilmGrainPlaneSample(output.U, output.Layout.BytesPerSample, 0, 0),
		getTestFrameWorkFilmGrainPlaneSample(output.V, output.Layout.BytesPerSample, 0, 0),
	} {
		if sample > 1023 {
			t.Fatalf("high-bit-depth chroma sample=%d outside 10-bit range", sample)
		}
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersChromaFilmGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 96
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			output.U.Pix[y*output.U.Stride+x] = 100
			output.V.Pix[y*output.V.Stride+x] = 110
		}
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	allocs := testing.AllocsPerRun(1000, func() {
		next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
		if err != nil {
			t.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterFilmGrain ||
			result.FilmGrain.LumaRows != 1 ||
			result.FilmGrain.ChromaRows != [2]uint16{1, 1} ||
			next.RemainingPostFilters() != 0 {
			t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	})
	if allocs != 0 {
		t.Fatalf("chroma film grain postfilter pipeline allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesPartialFinalChromaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 33, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 96
		}
	}
	for y := 0; y < output.U.Height; y++ {
		for x := 0; x < output.U.Width; x++ {
			output.U.Pix[y*output.U.Stride+x] = 100
			output.V.Pix[y*output.V.Stride+x] = 110
		}
	}
	beforeU := testCopyFrameWorkCDEFPlane(output.U)
	beforeV := testCopyFrameWorkCDEFPlane(output.V)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				NumCbPoints:   1,
				CbPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				NumCrPoints:   1,
				CrPoints:      [parser.MaxFilmGrainUVPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				ARCoeffShift:  6,
				CbMult:        64,
				CrMult:        64,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{FilmGrain: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 2 ||
		result.FilmGrain.ChromaRows != [2]uint16{2, 2} ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.U, beforeU) {
		t.Fatal("film grain did not change partial-row cb output")
	}
	if !testFrameWorkCDEFPlaneChanged(output.V, beforeV) {
		t.Fatal("film grain did not change partial-row cr output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesHighBitDepthLumaFilmGrain(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 10, Align: 64})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			setTestFrameWorkFilmGrainPlaneSample(output.Y, output.Layout.BytesPerSample, x, y, 400)
		}
	}
	before := testCopyFrameWorkCDEFPlane(output.Y)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      10,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	next, result, err := ctx.ApplySupportedPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 1 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("film grain did not change high-bit-depth luma output")
	}
	if got := getTestFrameWorkFilmGrainPlaneSample(output.Y, output.Layout.BytesPerSample, 0, 0); got > 1023 {
		t.Fatalf("high-bit-depth sample=%d outside 10-bit range", got)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersAppliesPartialFinalLumaRow(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 33, BitDepth: 8, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}
	before := testCopyFrameWorkCDEFPlane(output.Y)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	next, result, err := ctx.ApplySupportedPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterFilmGrain ||
		result.FilmGrain.LumaRows != 2 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("film grain did not change partial-row luma output")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsLumaFilmGrainShortScratchBeforeMutation(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	output.Y.Pix[0] = 0x44
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              32,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{
				Damping:       5,
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{63},
			},
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				ScalingShift:  8,
			},
		},
		Output: output,
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if next.RemainingPostFilters() != ctx.RemainingPostFilters() || result != (FrameWorkPostFilterResult{}) {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if output.Y.Pix[0] != 0x44 {
		t.Fatalf("output sample=%d want 0x44", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersLumaFilmGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 34, Height: 32, BitDepth: 8, Align: 32})
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
				NumYPoints:    1,
				YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			},
		},
		Output: output,
	}
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkPostFilterRequest{
		FilmGrain: FrameWorkFilmGrainPostFilterRequest{
			LumaGrain:   make([]int16, size.FilmGrain.LumaGrain),
			LumaSamples: make([]uint16, size.FilmGrain.LumaSamples),
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		next, result, err := ctx.ApplySupportedPostFilters(req)
		if err != nil {
			t.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterFilmGrain || result.FilmGrain.LumaRows != 1 || next.RemainingPostFilters() != 0 {
			t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	})
	if allocs != 0 {
		t.Fatalf("luma film grain postfilter pipeline allocated: %f", allocs)
	}
}

func TestFrameWorkFilmGrainPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		ScalingPoints: [3]int{1, 2, 3},
		ARCoeffs:      [3]int{4, 5, 6},
		LumaGrain:     8,
		ChromaGrain:   [2]int{4, 2},
		LumaSamples:   16,
		ChromaSamples: [2]int{9, 7},
		LumaLine:      3,
		LumaColumn:    2,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	req, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.LumaGrain) != size.LumaGrain || len(req.LumaSamples) != size.LumaSamples ||
		len(req.OutputFrame) != size.OutputFrame ||
		len(req.ChromaGrain[0]) != size.ChromaGrain[0] || len(req.ChromaGrain[1]) != size.ChromaGrain[1] ||
		len(req.ChromaSamples[0]) != size.ChromaSamples[0] || len(req.ChromaSamples[1]) != size.ChromaSamples[1] {
		t.Fatalf("request lumaGrain=%d lumaSamples=%d chromaGrain=%v chromaSamples=%v",
			len(req.LumaGrain), len(req.LumaSamples),
			[2]int{len(req.ChromaGrain[0]), len(req.ChromaGrain[1])},
			[2]int{len(req.ChromaSamples[0]), len(req.ChromaSamples[1])})
	}
}

func TestFrameWorkFilmGrainPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		ScalingPoints: [3]int{1},
		ARCoeffs:      [3]int{4},
		LumaGrain:     8,
		ChromaGrain:   [2]int{4, 2},
		LumaSamples:   16,
		ChromaSamples: [2]int{9, 7},
		LumaLine:      3,
		LumaColumn:    2,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	tests := []struct {
		name          string
		size          FrameWorkFilmGrainPostFilterScratchSize
		outputFrame   []byte
		lumaGrain     []int16
		chromaGrain   [2][]int16
		lumaSamples   []uint16
		chromaSamples [2][]uint16
	}{
		{name: "output-frame", size: FrameWorkFilmGrainPostFilterScratchSize{OutputFrame: 1}, outputFrame: nil, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "luma-grain", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain[:7], chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "luma-samples", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples[:15], chromaSamples: chromaSamples},
		{name: "chroma-grain", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: [2][]int16{chromaGrain[0][:3], chromaGrain[1]}, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "chroma-samples", size: size, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: [2][]uint16{chromaSamples[0], chromaSamples[1][:6]}},
		{name: "negative-requested", size: FrameWorkFilmGrainPostFilterScratchSize{LumaGrain: -1}, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
		{name: "negative-plan", size: FrameWorkFilmGrainPostFilterScratchSize{ScalingPoints: [3]int{-1}}, outputFrame: outputFrame, lumaGrain: lumaGrain, chromaGrain: chromaGrain, lumaSamples: lumaSamples, chromaSamples: chromaSamples},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.size.BindRequest(tt.outputFrame, nil, tt.lumaGrain, tt.chromaGrain, tt.lumaSamples, tt.chromaSamples); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("BindRequest err=%v want %v", err, frame.ErrShortBuffer)
			}
		})
	}
}

func TestFrameWorkFilmGrainPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		LumaGrain:     filmgrain.LumaGrainSamples,
		ChromaGrain:   [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples},
		LumaSamples:   1920 * 1080,
		ChromaSamples: [2]int{960 * 540, 960 * 540},
		LumaLine:      filmgrain.LumaOverlapSamples * 1920,
		LumaColumn:    filmgrain.LumaColumnScratchRows * filmgrain.LumaOverlapSamples,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkFilmGrainPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkFilmGrainPostFilterScratchSize{
		LumaGrain:     filmgrain.LumaGrainSamples,
		ChromaGrain:   [2]int{filmgrain.ChromaGrainSamples, filmgrain.ChromaGrainSamples},
		LumaSamples:   1920 * 1080,
		ChromaSamples: [2]int{960 * 540, 960 * 540},
		LumaLine:      filmgrain.LumaOverlapSamples * 1920,
		LumaColumn:    filmgrain.LumaColumnScratchRows * filmgrain.LumaOverlapSamples,
	}
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkFilmGrainPostFilterRequest(size FrameWorkFilmGrainPostFilterScratchSize) FrameWorkFilmGrainPostFilterRequest {
	outputFrame, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size)
	req, err := size.BindRequest(outputFrame, nil, lumaGrain, chromaGrain, lumaSamples, chromaSamples)
	if err != nil {
		panic(err)
	}
	return req
}

func testFrameWorkFilmGrainScratchStorage(size FrameWorkFilmGrainPostFilterScratchSize) ([]byte, []int16, [2][]int16, []uint16, [2][]uint16) {
	var chromaGrain [2][]int16
	var chromaSamples [2][]uint16
	for plane := range len(chromaGrain) {
		if size.ChromaGrain[plane] > 0 {
			chromaGrain[plane] = make([]int16, size.ChromaGrain[plane])
		}
		if size.ChromaSamples[plane] > 0 {
			chromaSamples[plane] = make([]uint16, size.ChromaSamples[plane])
		}
	}
	return make([]byte, maxInt(size.OutputFrame, 0)),
		make([]int16, maxInt(size.LumaGrain, 0)), chromaGrain,
		make([]uint16, maxInt(size.LumaSamples, 0)), chromaSamples
}

func getTestFrameWorkFilmGrainPlaneSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func setTestFrameWorkFilmGrainPlaneSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func testFrameWorkFilmGrainPlaneChangedFromValue(plane frame.Plane, bytesPerSample int, value uint16) bool {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			if getTestFrameWorkFilmGrainPlaneSample(plane, bytesPerSample, x, y) != value {
				return true
			}
		}
	}
	return false
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRunsCDEFThenNoOpFilmGrain(t *testing.T) {
	const width = 64
	const height = 64

	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x4321,
			BitDepth:      8,
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkCDEFPlane(output.Y)
	before := testCopyFrameWorkCDEFPlane(output.Y)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	size, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if size.CDEF.Input == 0 || size.CDEF.UnitDst == 0 || size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v", size)
	}
	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{CDEF: req})
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterCDEF|FrameWorkPostFilterFilmGrain ||
		result.CDEF.Planes != 1 ||
		!result.FilmGrain.NoOp ||
		result.FilmGrain.Plan.Seed != 0x4321 {
		t.Fatalf("result=%+v", result)
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if !testFrameWorkCDEFPlaneChanged(output.Y, before) {
		t.Fatal("CDEF stage did not run before film grain")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersNoOpFilmGrainAllocs(t *testing.T) {
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          0x1234,
				BitDepth:      8,
			},
		},
		Output: output,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{}); err != nil {
			t.Fatal(err)
		}
		next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{})
		if err != nil {
			t.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterFilmGrain || !result.FilmGrain.NoOp || next.RemainingPostFilters() != 0 {
			t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	})
	if allocs != 0 {
		t.Fatalf("no-op film grain postfilter pipeline allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanRejectsNilOutput(t *testing.T) {
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FilmGrain: parser.FilmGrainParams{Apply: true, BitDepth: 8},
		},
	}
	if _, err := ctx.FilmGrainPostFilterPlan(); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("FilmGrainPostFilterPlan err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkPostFilterContextFilmGrainPostFilterPlanRejectsInvalidActiveState(t *testing.T) {
	tests := []struct {
		name   string
		format frame.Format
		grain  parser.FilmGrainParams
	}{
		{
			name:   "bitdepth",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 10},
		},
		{
			name:   "monochrome-chroma",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, MonoChrome: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 8, ChromaScalingFromLuma: true},
		},
		{
			name:   "lag",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 8, ARCoeffLag: 4},
		},
		{
			name:   "420-one-sided-chroma",
			format: frame.Format{Width: 64, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32},
			grain:  parser.FilmGrainParams{Apply: true, BitDepth: 8, NumCbPoints: 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testFrameWorkCDEFFrame(t, tt.format)
			ctx := FrameWorkPostFilterContext{Event: Event{FilmGrain: tt.grain}, Output: output}
			if _, err := ctx.FilmGrainPostFilterPlan(); !errors.Is(err, frame.ErrInvalidFormat) {
				t.Fatalf("FilmGrainPostFilterPlan err=%v want %v", err, frame.ErrInvalidFormat)
			}
		})
	}
}

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

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSkipsInactive(t *testing.T) {
	plan, err := (&FrameWorkPostFilterContext{}).LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if plan != (FrameWorkLoopFilterPostFilterPlan{}) {
		t.Fatalf("plan=%+v want zero", plan)
	}
	size, err := (&FrameWorkPostFilterContext{}).LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size != (FrameWorkLoopFilterPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v want zero", size)
	}
	upper, err := (&FrameWorkPostFilterContext{}).LoopFilterPostFilterScratchUpperBound()
	if err != nil {
		t.Fatal(err)
	}
	if upper != (FrameWorkLoopFilterPostFilterScratchSize{}) {
		t.Fatalf("upper=%+v want zero", upper)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterScratchUpperBound(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	upper, err := ctx.LoopFilterPostFilterScratchUpperBound()
	if err != nil {
		t.Fatal(err)
	}
	if upper.Edges != cols*rows*8 || upper.Schedule != cols*rows*8 {
		t.Fatalf("upper=%+v want %d edges/schedule", upper, cols*rows*8)
	}
}

func TestFrameWorkLoopFilterWidthTableMatchesTransformDimensions(t *testing.T) {
	for tx := tile.TransformSize(0); tx.Valid(); tx++ {
		dims, ok := tx.Dimensions()
		if !ok {
			t.Fatalf("TransformSize(%d).Dimensions invalid", tx)
		}
		for _, edge := range []loopfilter.Edge{loopfilter.EdgeVertical, loopfilter.EdgeHorizontal} {
			span4 := dims.W4
			if edge == loopfilter.EdgeHorizontal {
				span4 = dims.H4
			}
			wantLuma := uint8(14)
			switch {
			case span4 <= 1:
				wantLuma = 4
			case span4 == 2:
				wantLuma = 8
			}
			gotLuma, err := frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
			if err != nil || gotLuma != wantLuma {
				t.Fatalf("luma width tx=%d edge=%d got=%d err=%v want=%d", tx, edge, gotLuma, err, wantLuma)
			}
			wantChroma := uint8(6)
			if span4 <= 1 {
				wantChroma = 4
			}
			gotChroma, err := frameWorkLoopFilterWidth(loopfilter.PlaneU, edge, tx)
			if err != nil || gotChroma != wantChroma {
				t.Fatalf("chroma width tx=%d edge=%d got=%d err=%v want=%d", tx, edge, gotChroma, err, wantChroma)
			}
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanDefaultsMapAndResolvesLevels(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	record := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	record.DeltaLF = [tile.FrameLoopFilterCount]int8{-2, 3, 4, -1}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, record)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Delta: parser.DeltaParams{
				DeltaLFPresent: true,
				DeltaLFMulti:   true,
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{16, 20},
				LevelU: 8,
				LevelV: 4,
			},
		},
		LoopFilterMap: &filterMap,
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || int(plan.MICols) != cols || int(plan.MIRows) != rows || int(plan.Cells) != cols*rows || plan.Blocks != 1 || plan.Missing != 0 {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.TransformReadyBlocks != 1 || plan.SkipTransformBlocks != 0 || plan.LumaTXBs != 1 {
		t.Fatalf("transform stats=%+v", plan)
	}
	wantYV := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 14}
	if got := plan.Levels[loopfilter.PlaneY][loopfilter.EdgeVertical]; got != wantYV {
		t.Fatalf("Y vertical=%+v want %+v", got, wantYV)
	}
	wantYH := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 23}
	if got := plan.Levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal]; got != wantYH {
		t.Fatalf("Y horizontal=%+v want %+v", got, wantYH)
	}
	wantU := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 12}
	if got := plan.Levels[loopfilter.PlaneU][loopfilter.EdgeHorizontal]; got != wantU {
		t.Fatalf("U horizontal=%+v want %+v", got, wantU)
	}
	wantV := FrameWorkLoopFilterPostFilterLevelStats{Blocks: 1, NonZero: 1, MaxLevel: 3}
	if got := plan.Levels[loopfilter.PlaneV][loopfilter.EdgeVertical]; got != wantV {
		t.Fatalf("V vertical=%+v want %+v", got, wantV)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanIgnoresPaddedMapCells(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	record := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filterMap := FrameWorkLoopFilterMap{
		Records: make([]threading.FrameWorkLoopFilterBlockRecord, (cols+2)*(rows+1)),
		Stride:  uint16(cols + 2),
		Rows:    uint16(rows + 1),
	}
	stride := int(filterMap.Stride)
	for miRow := record.Block.MIRow; miRow < record.Block.MIRowEnd; miRow++ {
		row := int(miRow) * stride
		for miCol := record.Block.MICol; miCol < record.Block.MIColEnd; miCol++ {
			filterMap.Records[row+int(miCol)] = record
		}
	}
	filterMap.Records[cols] = threading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: threading.FrameWorkLoopFilterBlock{MICol: uint16(cols), MIRow: 0, MIColEnd: uint16(cols + 1), MIRowEnd: 1},
	}
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
		LoopFilterMap: &filterMap,
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if int(plan.Cells) != cols*rows || plan.Blocks != 1 || plan.Missing != 0 {
		t.Fatalf("plan=%+v want cells=%d blocks=1 missing=0", plan, cols*rows)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanCountsTransformReplay(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Blocks != 2 || plan.TransformReadyBlocks != 2 || plan.SkipTransformBlocks != 1 || plan.LumaTXBs != 4 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanStoresLumaEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	scratchSize, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.Edges != len(edges) || scratchSize.Schedule != len(edges) {
		t.Fatalf("scratch=%+v want %d edges/schedule", scratchSize, len(edges))
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 4 || plan.StoredEdges != uint32(len(edges)) || plan.DroppedEdges != 0 {
		t.Fatalf("plan=%+v", plan)
	}
	want := []FrameWorkLoopFilterPostFilterEdge{
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 0, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 6, Y4: 0, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 2, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 6, Y4: 2, Length4: 2, Level: 8, Transform: tile.TransformSize8x8, Width: 8, BlockMICol: 4, BlockMIRow: 0},
	}
	for i := range want {
		if edges[i] != want[i] {
			t.Fatalf("edge[%d]=%+v want %+v", i, edges[i], want[i])
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanTrustedMatchesValidated(t *testing.T) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(t)
	validated, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	trustedEdges := make([]FrameWorkLoopFilterPostFilterEdge, len(edges))
	trusted, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:             filterMap,
		Edges:           trustedEdges,
		TrustedCoverage: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if trusted != validated {
		t.Fatalf("trusted plan=%+v want validated %+v", trusted, validated)
	}
	for i := range edges {
		if trustedEdges[i] != edges[i] {
			t.Fatalf("edge[%d]=%+v want %+v", i, trustedEdges[i], edges[i])
		}
	}
}

func TestFrameWorkLoopFilterPostFilterScratchSizeBindEdges(t *testing.T) {
	size := FrameWorkLoopFilterPostFilterScratchSize{Edges: 3}
	storage := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	got, err := size.BindEdges(storage)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || cap(got) != cap(storage) {
		t.Fatalf("bound len/cap=%d/%d want len 3 cap %d", len(got), cap(got), cap(storage))
	}
	if _, err := size.BindEdges(storage[:2]); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("short bind err=%v want %v", err, frame.ErrShortBuffer)
	}
	if _, err := (FrameWorkLoopFilterPostFilterScratchSize{Edges: -1}).BindEdges(storage); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("negative bind err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkLoopFilterPostFilterScratchSizeBindSchedule(t *testing.T) {
	size := FrameWorkLoopFilterPostFilterScratchSize{Schedule: 3}
	storage := make([]uint32, 4)
	got, err := size.BindSchedule(storage)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || cap(got) != cap(storage) {
		t.Fatalf("bound len/cap=%d/%d want len 3 cap %d", len(got), cap(got), cap(storage))
	}
	if _, err := size.BindSchedule(storage[:2]); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("short bind err=%v want %v", err, frame.ErrShortBuffer)
	}
	if _, err := (FrameWorkLoopFilterPostFilterScratchSize{Schedule: -1}).BindSchedule(storage); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("negative bind err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanStoresChromaEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{8},
				LevelU: 7,
				LevelV: 9,
			},
		},
	}
	scratchSize, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.Edges != 3 || scratchSize.Schedule != 3 {
		t.Fatalf("scratch=%+v want 3 edges/schedule", scratchSize)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.Edges)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 3 || plan.StoredEdges != 3 || plan.DroppedEdges != 0 {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.PlaneEdgeCandidates != [3]uint32{1, 1, 1} {
		t.Fatalf("plane edge candidates=%v want [1 1 1]", plan.PlaneEdgeCandidates)
	}
	if plan.ChromaTXBs != 2 {
		t.Fatalf("chroma txbs=%d want 2", plan.ChromaTXBs)
	}
	want := []FrameWorkLoopFilterPostFilterEdge{
		{Plane: loopfilter.PlaneY, Edge: loopfilter.EdgeVertical, X4: 4, Y4: 0, Length4: 4, Level: 8, Transform: tile.TransformSize16x16, Width: 14, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneU, Edge: loopfilter.EdgeVertical, X4: 2, Y4: 0, Length4: 2, Level: 7, Transform: tile.TransformSize8x8, Width: 6, BlockMICol: 4, BlockMIRow: 0},
		{Plane: loopfilter.PlaneV, Edge: loopfilter.EdgeVertical, X4: 2, Y4: 0, Length4: 2, Level: 9, Transform: tile.TransformSize8x8, Width: 6, BlockMICol: 4, BlockMIRow: 0},
	}
	for i := range want {
		if edges[i] != want[i] {
			t.Fatalf("edge[%d]=%+v want %+v", i, edges[i], want[i])
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanChromaSubsamplingGeometries(t *testing.T) {
	tests := []struct {
		name    string
		ssX     bool
		ssY     bool
		wantX4  int
		wantY4  int
		wantLen int
	}{
		{name: "444", wantX4: 4, wantY4: 0, wantLen: 4},
		{name: "422", ssX: true, wantX4: 2, wantY4: 0, wantLen: 4},
		{name: "420", ssX: true, ssY: true, wantX4: 2, wantY4: 0, wantLen: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			size := parser.FrameSize{
				CodedWidth:          32,
				UpscaledWidth:       32,
				Height:              16,
				SuperResDenominator: 8,
			}
			seq := testSequence()
			seq.ColorConfig.SubsamplingX = tt.ssX
			seq.ColorConfig.SubsamplingY = tt.ssY
			first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
			first.SkipTransform = true
			second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
			second.SkipTransform = true
			filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
			ctx := FrameWorkPostFilterContext{
				Event: Event{
					SequenceHeader: seq,
					FrameSize:      size,
					LoopFilter: parser.LoopFilterParams{
						LevelY: [2]uint8{8},
						LevelU: 7,
					},
				},
			}
			edges := make([]FrameWorkLoopFilterPostFilterEdge, 2)

			plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
				Map:   filterMap,
				Edges: edges,
			})
			if err != nil {
				t.Fatal(err)
			}
			if plan.EdgeCandidates != 2 || plan.StoredEdges != 2 || plan.PlaneEdgeCandidates != [3]uint32{1, 1, 0} {
				t.Fatalf("plan=%+v", plan)
			}
			wantU := FrameWorkLoopFilterPostFilterEdge{
				Plane:      loopfilter.PlaneU,
				Edge:       loopfilter.EdgeVertical,
				X4:         uint16(tt.wantX4),
				Y4:         uint16(tt.wantY4),
				Length4:    uint16(tt.wantLen),
				Level:      7,
				Transform:  tile.TransformSize8x8,
				Width:      6,
				BlockMICol: 4,
				BlockMIRow: 0,
			}
			if edges[1] != wantU {
				t.Fatalf("U edge=%+v want %+v", edges[1], wantU)
			}
		})
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSchedulesLumaEdgeWidths(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize32x16
	filler.SkipTransform = true
	tx4 := testFrameWorkLoopFilterPostFilterRecordAt(1, 0, 2, 1)
	tx4.SkipTransform = true
	tx4.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4}
	tx8 := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 2)
	tx8.SkipTransform = true
	tx8.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	tx16 := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	tx16.SkipTransform = true
	tx16.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, tx4, tx8, tx16)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != 3 {
		t.Fatalf("plan=%+v edges=%+v", plan, edges)
	}
	for i, want := range []int{4, 4, 8} {
		if int(edges[i].Width) != want {
			t.Fatalf("edge[%d]=%+v want width %d", i, edges[i], want)
		}
	}
}

// TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSplitsLumaEdgeByPerCellWidth
// covers the 10-bit quantizer 32 LF regression: when a single luma transform
// edge spans multiple MI cells, each MI cell's filter width must come from the
// (current, previous) transform pair at THAT cell, matching libaom's
// set_one_param_for_line_luma. The previous implementation collapsed the edge
// down to min(previousWidth) across the whole length, applying the narrower
// filter at cells that should have used the wider one. The bottom 8x8 TX
// block here straddles two top-side TXs (TX_8X8 vs TX_4X4) and must emit two
// per-MI-cell width-8 / width-4 horizontal edges instead of one width-4 edge.
func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanSplitsLumaEdgeByPerCellWidth(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize16x16
	filler.SkipTransform = true
	// Top row: two side-by-side 1-MI-tall TX blocks with different sizes.
	// Top-left has TX_8X8 (filter width=8 for horizontal); top-right has
	// TX_4X4 (filter width=4). They share the y=4 horizontal boundary with
	// the bottom block below.
	topLeft := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 2, 1)
	topLeft.SkipTransform = true
	topLeft.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	topRight := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 1)
	topRight.SkipTransform = true
	topRight.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4}
	// Bottom block spans cols 0..3, rows 1..3 with TX_8X8 (width=8 horiz).
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 1, 4, 4)
	bottom.SkipTransform = true
	bottom.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, topLeft, topRight, bottom)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 16)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Find plane-Y horizontal edges at y=4 (Y4=1) emitted by `bottom`. The
	// fixed code splits the horizontal edge between cols 0..3 into two
	// segments: cols 0..1 (prev TX_8X8 -> width 8) and cols 2..3 (prev
	// TX_4X4 -> width 4). The old behaviour emitted a single length=4
	// width=4 edge.
	var horizY []FrameWorkLoopFilterPostFilterEdge
	for i := 0; i < int(plan.StoredEdges); i++ {
		e := edges[i]
		if e.Plane == loopfilter.PlaneY && e.Edge == loopfilter.EdgeHorizontal && e.Y4 == 1 {
			horizY = append(horizY, e)
		}
	}
	if len(horizY) != 2 {
		t.Fatalf("expected 2 per-MI-cell width segments, got %d: %+v plan=%+v", len(horizY), horizY, plan)
	}
	if horizY[0].X4 != 0 || horizY[0].Length4 != 2 || horizY[0].Width != 8 {
		t.Fatalf("edge[0]=%+v want X4=0 Length4=2 Width=8", horizY[0])
	}
	if horizY[1].X4 != 2 || horizY[1].Length4 != 2 || horizY[1].Width != 4 {
		t.Fatalf("edge[1]=%+v want X4=2 Length4=2 Width=4", horizY[1])
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanLimitsHorizontalLumaWidthByPreviousTransform(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              32,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize16x32
	filler.SkipTransform = true
	tx4 := testFrameWorkLoopFilterPostFilterRecordAt(0, 1, 1, 2)
	tx4.SkipTransform = true
	tx4.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4}
	tx8 := testFrameWorkLoopFilterPostFilterRecordAt(0, 2, 2, 4)
	tx8.SkipTransform = true
	tx8.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8}
	tx16 := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	tx16.SkipTransform = true
	tx16.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, tx4, tx8, tx16)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{0, 8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != 3 {
		t.Fatalf("plan=%+v edges=%+v", plan, edges)
	}
	for i, want := range []int{4, 4, 8} {
		if int(edges[i].Width) != want {
			t.Fatalf("edge[%d]=%+v want width %d", i, edges[i], want)
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanLimitsChromaWidthByPreviousTransform(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	seq := testSequence()
	seq.ColorConfig.SubsamplingX = false
	seq.ColorConfig.SubsamplingY = false
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 2, 4)
	left.SkipTransform = true
	left.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize4x4, HasUV: true}
	right := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 4)
	right.SkipTransform = true
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, left, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: seq,
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						0: {DeltaLFYV: -1},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{1},
				LevelU: 7,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PlaneEdgeCandidates != [3]uint32{0, 1, 0} {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:      loopfilter.PlaneU,
		Edge:       loopfilter.EdgeVertical,
		X4:         2,
		Y4:         0,
		Length4:    4,
		Level:      7,
		Transform:  tile.TransformSize8x8,
		Width:      4,
		BlockMICol: 2,
		BlockMIRow: 0,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

// TestFrameWorkPostFilterContextLoopFilterPostFilterPlanChromaPreviousLookupHonoursORAnchor
// covers the 10-bit quantizer 32 chroma deblock regression: when chroma is
// subsampled (4:2:0) the previous chroma cell's TX size must be read from the
// libaom OR-anchor (mi_row|=scale, mi_col|=scale) — i.e. the bottom-right luma
// MI of the previous chroma cluster — not from the cluster's top-left luma
// MI. The two MIs disagree whenever the previous chroma cluster contains a
// sub8x8 partition: only the bottom-right MI carries the chroma transform
// info, the other three luma MIs report HasChromaBlock=false. With the
// pre-fix top-left lookup the chroma edge was misclassified as a "no-chroma
// gap" and skipped, leaving the 10-bit quantizer 32 vector ~2000 chroma
// samples off after the loop filter. The fixed lookup lands on the
// bottom-right MI and recovers the chroma TX 4x4 -> 4-tap filter expected by
// libaom's set_one_param_for_line_chroma (av1/common/av1_loopfilter.c).
func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanChromaPreviousLookupHonoursORAnchor(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	// Filler block covers the rest of the frame so every MI cell carries a
	// valid record. The vertical chroma edge we exercise sits between the
	// sub8x8 cluster at luma MI (0..1, 0..1) and an 8x8 block at luma MI
	// (2..3, 0..1).
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize16x16
	filler.SkipTransform = true
	// Sub8x8 cluster: four 4x4 luma blocks at (0,0), (1,0), (0,1), (1,1).
	// Only (1,1) — the bottom-right anchor — carries chroma per libaom's
	// HasChroma condition. We give it a small chroma TX (4x4) so the edge
	// filter width depends on the bottom-right MI lookup.
	subTL := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 1, 1)
	subTL.SkipTransform = true
	subTL.Block.Size = tile.BlockSize4x4
	subTL.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, HasUV: false}
	subTR := testFrameWorkLoopFilterPostFilterRecordAt(1, 0, 2, 1)
	subTR.SkipTransform = true
	subTR.Block.Size = tile.BlockSize4x4
	subTR.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, HasUV: false}
	subBL := testFrameWorkLoopFilterPostFilterRecordAt(0, 1, 1, 2)
	subBL.SkipTransform = true
	subBL.Block.Size = tile.BlockSize4x4
	subBL.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, HasUV: false}
	// Bottom-right anchor block: carries the chroma TX (4x4) that the right
	// neighbour's previous-side lookup must read via the OR-anchor mapping.
	subBR := testFrameWorkLoopFilterPostFilterRecordAt(1, 1, 2, 2)
	subBR.SkipTransform = true
	subBR.Block.Size = tile.BlockSize4x4
	subBR.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize4x4, UV: tile.TransformSize4x4, HasUV: true}
	// Right neighbour: 8x8 luma at MI (2..3, 0..1), chroma TX 4x4. Its
	// left chroma edge sits at chroma_4x4=1; the previous chroma cell is
	// the cluster (0..1, 0..1).
	right := testFrameWorkLoopFilterPostFilterRecordAt(2, 0, 4, 2)
	right.SkipTransform = true
	right.Block.Size = tile.BlockSize8x8
	right.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize8x8, UV: tile.TransformSize4x4, HasUV: true}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, subTL, subTR, subBL, subBR, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{8, 8},
				LevelU: 7,
				LevelV: 5,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 64)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Locate the U/V vertical edges emitted at chroma_4x4=1 (= luma 2),
	// the boundary between the sub8x8 cluster and the 8x8 right block.
	var chromaVerts []FrameWorkLoopFilterPostFilterEdge
	for i := 0; i < int(plan.StoredEdges); i++ {
		e := edges[i]
		if (e.Plane == loopfilter.PlaneU || e.Plane == loopfilter.PlaneV) &&
			e.Edge == loopfilter.EdgeVertical && e.X4 == 1 && e.BlockMICol == 2 {
			chromaVerts = append(chromaVerts, e)
		}
	}
	if len(chromaVerts) != 2 {
		t.Fatalf("expected 1 U + 1 V vertical chroma edge at the cluster boundary, got %d: %+v", len(chromaVerts), chromaVerts)
	}
	for _, e := range chromaVerts {
		// With the OR-anchor lookup the previous chroma TX is TX_4x4, the
		// current TX is also TX_4x4, so libaom selects the 4-tap chroma
		// filter (filter_length = 4) for both U and V.
		if e.Width != 4 {
			t.Fatalf("chroma edge=%+v want Width=4 (OR-anchor lookup should read TX_4x4 from sub-block (1,1))", e)
		}
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanDowngradesBoundaryWidths(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.SkipTransform = true
	nearLeft := testFrameWorkLoopFilterPostFilterRecordAt(1, 0, 4, 4)
	nearLeft.SkipTransform = true
	nearLeft.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize16x16}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, nearLeft)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != 1 || edges[0].Width != 8 {
		t.Fatalf("plan=%+v edge=%+v", plan, edges[0])
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanCountsDroppedEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              32,
		SuperResDenominator: 8,
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size,
		testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 4, 8, 8),
	)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 9}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: make([]FrameWorkLoopFilterPostFilterEdge, 1),
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 4 || plan.StoredEdges != 1 || plan.DroppedEdges != 3 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousLumaLevelVertical(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SkipTransform = true
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.SkipTransform = true
	right.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, left, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						1: {DeltaLFYV: -8},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneY,
		Edge:              loopfilter.EdgeVertical,
		X4:                4,
		Y4:                0,
		Length4:           4,
		Level:             8,
		Transform:         tile.TransformSize16x16,
		Width:             14,
		LevelFromPrevious: true,
		BlockMICol:        4,
		BlockMIRow:        0,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousLumaLevelHorizontal(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              32,
		SuperResDenominator: 8,
	}
	top := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	top.SkipTransform = true
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	bottom.SkipTransform = true
	bottom.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, top, bottom)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						1: {DeltaLFYH: -9},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{0, 9}},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneY,
		Edge:              loopfilter.EdgeHorizontal,
		X4:                0,
		Y4:                4,
		Length4:           4,
		Level:             9,
		Transform:         tile.TransformSize16x16,
		Width:             14,
		LevelFromPrevious: true,
		BlockMICol:        0,
		BlockMIRow:        4,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousChromaLevelVertical(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	left := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	left.SkipTransform = true
	right := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	right.SkipTransform = true
	right.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, left, right)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						0: {DeltaLFYV: -1},
						1: {DeltaLFYV: -1, DeltaLFU: -7},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{1},
				LevelU: 7,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 || plan.PlaneEdgeCandidates != [3]uint32{0, 1, 0} {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneU,
		Edge:              loopfilter.EdgeVertical,
		X4:                2,
		Y4:                0,
		Length4:           2,
		Level:             7,
		Transform:         tile.TransformSize8x8,
		Width:             6,
		LevelFromPrevious: true,
		BlockMICol:        4,
		BlockMIRow:        0,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanUsesPreviousChromaLevelHorizontal(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              32,
		SuperResDenominator: 8,
	}
	top := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	top.SkipTransform = true
	bottom := testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8)
	bottom.SkipTransform = true
	bottom.SegmentID = 1
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, top, bottom)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			Segmentation: parser.SegmentationParams{
				Enabled: true,
				Data: parser.SegmentationData{
					Segments: [parser.MaxSegments]parser.SegmentData{
						0: {DeltaLFYH: -1},
						1: {DeltaLFYH: -1, DeltaLFV: -9},
					},
				},
			},
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{0, 1},
				LevelV: 9,
			},
		},
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 1)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EdgeCandidates != 1 || plan.StoredEdges != 1 || plan.PreviousLevelEdges != 1 || plan.PlaneEdgeCandidates != [3]uint32{0, 0, 1} {
		t.Fatalf("plan=%+v", plan)
	}
	want := FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneV,
		Edge:              loopfilter.EdgeHorizontal,
		X4:                0,
		Y4:                2,
		Length4:           2,
		Level:             9,
		Transform:         tile.TransformSize8x8,
		Width:             6,
		LevelFromPrevious: true,
		BlockMICol:        0,
		BlockMIRow:        4,
	}
	if edges[0] != want {
		t.Fatalf("edge=%+v want %+v", edges[0], want)
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdges(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{63}, Sharpness: 1},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	wantY := append([]byte(nil), output.Y.Pix...)
	wantPlane := output.Y
	wantPlane.Pix = wantY
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range edges[:int(plan.StoredEdges)] {
		thresholds, err := loopfilter.ThresholdsForLevel(edge.Level, ctx.Event.LoopFilter.Sharpness)
		if err != nil {
			t.Fatal(err)
		}
		if err := loopfilter.FilterEdgeByWidth(edge.Width, wantPlane, output.Layout.BytesPerSample, output.Format.BitDepth, edge.Edge, int32(edge.X4)*4, int32(edge.Y4)*4, int32(edge.Length4)*4, thresholds); err != nil {
			t.Fatal(err)
		}
	}

	result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Edges != 4 || result.Applied != 4 || result.MaxLevel != 63 || result.Plan.StoredEdges != 4 {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("loop filter luma edge bridge did not change output")
	}
	if !bytes.Equal(output.Y.Pix, wantY) {
		t.Fatal("loop filter luma edge bridge output did not match direct edge application")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesLumaAndChroma(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	testFillFrameWorkLoopFilterPattern(output.U)
	testFillFrameWorkLoopFilterPattern(output.V)
	beforeY := append([]byte(nil), output.Y.Pix...)
	beforeU := append([]byte(nil), output.U.Pix...)
	beforeV := append([]byte(nil), output.V.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	wantY := append([]byte(nil), output.Y.Pix...)
	wantU := append([]byte(nil), output.U.Pix...)
	wantV := append([]byte(nil), output.V.Pix...)
	wantFrame := *output
	wantFrame.Y.Pix = wantY
	wantFrame.U.Pix = wantU
	wantFrame.V.Pix = wantV
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:int(plan.StoredEdges)])

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active ||
		result.Edges != 3 ||
		result.Applied != 3 ||
		result.MaxLevel != 63 ||
		result.PlaneEdges != [3]uint32{1, 1, 1} ||
		result.PlaneApplied != [3]uint32{1, 1, 1} ||
		result.PlaneMaxLevel != [3]uint8{63, 63, 63} ||
		result.Plan.StoredEdges != 3 {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.Y.Pix, beforeY) || bytes.Equal(output.U.Pix, beforeU) || bytes.Equal(output.V.Pix, beforeV) {
		t.Fatal("loop filter all-plane bridge did not change every active plane")
	}
	if !bytes.Equal(output.Y.Pix, wantY) || !bytes.Equal(output.U.Pix, wantU) || !bytes.Equal(output.V.Pix, wantV) {
		t.Fatal("loop filter all-plane bridge output did not match direct edge application")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesHighBitDepthChroma(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	seq := testSequence()
	seq.ColorConfig.BitDepth = 10
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 64})
	testFillFrameWorkLoopFilterSamplePattern(output.Y, output.Layout.BytesPerSample)
	testFillFrameWorkLoopFilterSamplePattern(output.U, output.Layout.BytesPerSample)
	testFillFrameWorkLoopFilterSamplePattern(output.V, output.Layout.BytesPerSample)
	beforeU := append([]byte(nil), output.U.Pix...)
	beforeV := append([]byte(nil), output.V.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: seq,
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	wantFrame := testCloneFrameWorkLoopFilterFrame(output)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:int(plan.StoredEdges)])

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.PlaneApplied != [3]uint32{1, 1, 1} || result.MaxLevel != 63 {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.U.Pix, beforeU) || bytes.Equal(output.V.Pix, beforeV) {
		t.Fatal("high-bit-depth chroma loop filter did not change U and V")
	}
	if !bytes.Equal(output.Y.Pix, wantFrame.Y.Pix) ||
		!bytes.Equal(output.U.Pix, wantFrame.U.Pix) ||
		!bytes.Equal(output.V.Pix, wantFrame.V.Pix) {
		t.Fatal("high-bit-depth loop filter output did not match direct edge application")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesUsesPlanePassOrder(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              32,
		SuperResDenominator: 8,
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size,
		testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 4, 8, 8),
	)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	rng := rand.New(rand.NewSource(1))
	testFillFrameWorkLoopFilterOrderPattern(output.Y, rng)
	testFillFrameWorkLoopFilterOrderPattern(output.U, rng)
	testFillFrameWorkLoopFilterOrderPattern(output.V, rng)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{63, 63},
				LevelU: 63,
				LevelV: 63,
			},
		},
		Output: output,
	}

	edges := make([]FrameWorkLoopFilterPostFilterEdge, 12)
	wantFrame := testCloneFrameWorkLoopFilterFrame(output)
	storedOrderFrame := testCloneFrameWorkLoopFilterFrame(output)
	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	if plan.StoredEdges != uint32(len(edges)) || plan.PlaneEdgeCandidates != [3]uint32{4, 4, 4} {
		t.Fatalf("plan=%+v", plan)
	}
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:int(plan.StoredEdges)])
	testApplyFrameWorkLoopFilterEdgesStoredOrder(t, storedOrderFrame, ctx.Event.LoopFilter.Sharpness, edges[:int(plan.StoredEdges)])
	if bytes.Equal(storedOrderFrame.Y.Pix, wantFrame.Y.Pix) &&
		bytes.Equal(storedOrderFrame.U.Pix, wantFrame.U.Pix) &&
		bytes.Equal(storedOrderFrame.V.Pix, wantFrame.V.Pix) {
		t.Fatal("test fixture did not distinguish stored order from plane/pass order")
	}

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Edges != 12 ||
		result.PlaneEdges != [3]uint32{4, 4, 4} ||
		result.PlaneApplied != [3]uint32{4, 4, 4} {
		t.Fatalf("result=%+v", result)
	}
	if !bytes.Equal(output.Y.Pix, wantFrame.Y.Pix) ||
		!bytes.Equal(output.U.Pix, wantFrame.U.Pix) ||
		!bytes.Equal(output.V.Pix, wantFrame.V.Pix) {
		t.Fatal("loop filter all-plane bridge did not use plane/pass order")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesLargeScanMatchesScheduledOrder(t *testing.T) {
	ctx, edges := testLargeFrameWorkLoopFilterEdgeApplyFixture(t, frameWorkLoopFilterApplyScheduleCap+1)
	scanFrame := testCloneFrameWorkLoopFilterFrame(ctx.Output)
	scheduledFrame := testCloneFrameWorkLoopFilterFrame(ctx.Output)
	scanCtx := ctx
	scanCtx.Output = &scanFrame
	scheduledCtx := ctx
	scheduledCtx.Output = &scheduledFrame
	expected, ok := frameWorkLoopFilterCounter(len(edges))
	if !ok {
		t.Fatal("edge count overflow")
	}

	var scanResult FrameWorkLoopFilterPostFilterApplyResult
	if err := scanCtx.applyLoopFilterEdgesInPlanePassOrderScan(&scanResult, edges, loopfilter.PlaneY, loopfilter.PlaneV, 0, expected); err != nil {
		t.Fatal(err)
	}
	schedule := make([]uint32, len(edges))
	var scheduledResult FrameWorkLoopFilterPostFilterApplyResult
	if err := scheduledCtx.applyLoopFilterEdgesInPlanePassOrderSchedule(&scheduledResult, edges, schedule, loopfilter.PlaneY, loopfilter.PlaneV, 0, expected); err != nil {
		t.Fatal(err)
	}
	if scheduledResult != scanResult {
		t.Fatalf("scheduled result=%+v scan result=%+v", scheduledResult, scanResult)
	}
	if !bytes.Equal(scheduledFrame.Y.Pix, scanFrame.Y.Pix) ||
		!bytes.Equal(scheduledFrame.U.Pix, scanFrame.U.Pix) ||
		!bytes.Equal(scheduledFrame.V.Pix, scanFrame.V.Pix) {
		t.Fatal("scheduled large-edge loop filter output did not match scan fallback")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterEdgesAllocs(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output: output,
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 3)
	allocs := testing.AllocsPerRun(1000, func() {
		testFillFrameWorkLoopFilterPattern(output.Y)
		testFillFrameWorkLoopFilterPattern(output.U)
		testFillFrameWorkLoopFilterPattern(output.V)
		result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Applied != 3 || result.PlaneApplied != [3]uint32{1, 1, 1} {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyLoopFilterEdges allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRunsLoopFilterThenCDEF(t *testing.T) {
	const width = 32
	const height = 16

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.LoopFilter.Edges < 1 || scratchSize.CDEF.Input != cdef.InputBufferSize {
		t.Fatalf("scratch=%+v", scratchSize)
	}

	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF
	if result.Completed != wantCompleted ||
		result.LoopFilter.Applied != 1 ||
		result.CDEF.Units != 1 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("supported postfilter pipeline did not change luma samples")
	}
}

func TestFrameWorkPostFilterContextApplyPreSuperResPostFiltersRunsLoopFilterThenCDEF(t *testing.T) {
	const width = 32
	const height = 16
	const upscaledWidth = 48

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       upscaledWidth,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.PreSuperResPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	if scratchSize.LoopFilter.Edges < 1 || scratchSize.CDEF.Input != cdef.InputBufferSize || scratchSize.Restoration != (FrameWorkRestorationPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v", scratchSize)
	}

	next, result, err := ctx.ApplyPreSuperResPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF
	if result.Completed != wantCompleted ||
		result.LoopFilter.Applied != 1 ||
		result.CDEF.Units != 1 ||
		next.RemainingPostFilters() != FrameWorkPostFilterSuperRes {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("pre-superres postfilter prefix did not change luma samples")
	}

	superResReq := testFrameWorkSuperResPostFilterRequest(t, next)
	next, superResResult, err := next.ApplySuperResPostFilterToContext(superResReq)
	if err != nil {
		t.Fatal(err)
	}
	if superResResult.Output.Format.Width != upscaledWidth || next.RemainingPostFilters() != 0 {
		t.Fatalf("superres output=%+v remaining=%b", superResResult.Output.Format, next.RemainingPostFilters())
	}
}

func TestFrameWorkPostFilterContextApplyCallerPostFiltersRunsPreSuperResAndSuperRes(t *testing.T) {
	const width = 32
	const height = 16
	const upscaledWidth = 48

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       upscaledWidth,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	req := FrameWorkPostFilterRequest{CDEF: cdefReq}
	sizeReq, err := ctx.CallerPostFilterScratchLen(req)
	if err != nil {
		t.Fatal(err)
	}
	if sizeReq.LoopFilter.Edges < 1 || sizeReq.CDEF.Input != cdef.InputBufferSize || sizeReq.SuperRes.OutputFrame == 0 {
		t.Fatalf("scratch=%+v", sizeReq)
	}
	req.LoopFilter.Edges = make([]FrameWorkLoopFilterPostFilterEdge, sizeReq.LoopFilter.Edges)
	req.SuperRes = testFrameWorkSuperResPostFilterRequest(t, ctx)

	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes
	if result.Completed != wantCompleted ||
		result.LoopFilter.Applied != 1 ||
		result.CDEF.Units != 1 ||
		result.SuperRes.Output.Format.Width != upscaledWidth ||
		next.RemainingPostFilters() != 0 ||
		!next.DetachedPostFilterOutput() {
		t.Fatalf("next remaining=%b detached=%v result=%+v", next.RemainingPostFilters(), next.DetachedPostFilterOutput(), result)
	}
	if bytes.Equal(output.Y.Pix, before) {
		t.Fatal("caller postfilter pipeline did not change coded luma samples before superres")
	}
	if err := next.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplyPreSuperResPostFiltersRejectsShortCDEFScratchBeforeLoopFilterMutation(t *testing.T) {
	const width = 32
	const height = 16

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       48,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.PreSuperResPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	cdefReq.InputScratch = cdefReq.InputScratch[:len(cdefReq.InputScratch)-1]

	next, result, err := ctx.ApplyPreSuperResPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyPreSuperResPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if result != (FrameWorkPostFilterResult{}) || next.RemainingPostFilters() != ctx.RemainingPostFilters() {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("short CDEF scratch mutated output through earlier loop filter")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsShortCDEFScratchBeforeLoopFilterMutation(t *testing.T) {
	const width = 32
	const height = 16

	seq := testSequence()
	seq.EnableCDEF = true
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		CDEF: parser.CDEFParams{
			Damping:       5,
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	cdefReq := testFrameWorkCDEFPostFilterRequest(t, ctx, event)
	scratchSize, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{CDEF: cdefReq})
	if err != nil {
		t.Fatal(err)
	}
	cdefReq.InputScratch = cdefReq.InputScratch[:len(cdefReq.InputScratch)-1]

	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, scratchSize.LoopFilter.Edges),
		},
		CDEF: cdefReq,
	})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, frame.ErrShortBuffer)
	}
	if result != (FrameWorkPostFilterResult{}) || next.RemainingPostFilters() != ctx.RemainingPostFilters() {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("short CDEF scratch mutated output through earlier loop filter")
	}
}

func TestFrameWorkPostFilterContextApplySupportedPostFiltersRejectsShortRestorationScratchBeforeLoopFilterMutation(t *testing.T) {
	const width = 64
	const height = 64

	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filler := testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	filler.Block.Size = tile.BlockSize64x64
	filler.SkipTransform = true
	filler.TransformTree = tile.TransformTreeResult{Y: tile.TransformSize64x64, UV: tile.TransformSize32x32, HasUV: true}
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.SkipTransform = true
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, filler, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	event := Event{
		SequenceHeader: testSequence(),
		FrameSize:      size,
		LoopFilter: parser.LoopFilterParams{
			LevelY:    [2]uint8{63},
			Sharpness: 1,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, LoopFilterMap: &filterMap}
	loopScratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, make([]tile.RestorationUnitRecord, plan.UnitRecordLen()))
	if err != nil {
		t.Fatal(err)
	}
	if err := tile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
		t.Fatal(err)
	}
	restorationScratch, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
	if err != nil {
		t.Fatal(err)
	}
	if restorationScratch.Samples.DataLen == 0 {
		t.Fatalf("restoration scratch=%+v", restorationScratch)
	}

	next, result, err := ctx.ApplySupportedPostFilters(FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Edges: make([]FrameWorkLoopFilterPostFilterEdge, loopScratch.Edges),
		},
		Restoration: FrameWorkRestorationPostFilterRequest{
			Records:     records,
			DataScratch: make([]uint16, restorationScratch.Samples.DataLen-1),
			DstScratch:  make([]uint16, restorationScratch.Samples.DstLen),
			Scratch:     testFrameWorkRestorationPostFilterScratch(restorationScratch.Apply),
		},
	})
	if !errors.Is(err, tile.ErrJobBufferTooSmall) {
		t.Fatalf("ApplySupportedPostFilters err=%v want %v", err, tile.ErrJobBufferTooSmall)
	}
	if result != (FrameWorkPostFilterResult{}) || next.RemainingPostFilters() != ctx.RemainingPostFilters() {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("short restoration scratch mutated output through earlier loop filter")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesSkipsInactive(t *testing.T) {
	result, err := (&FrameWorkPostFilterContext{}).ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkLoopFilterPostFilterApplyResult{}) {
		t.Fatalf("result=%+v want zero", result)
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesRejectsShortEdgeScratchBeforeMutation(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              32,
		SuperResDenominator: 8,
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size,
		testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4),
		testFrameWorkLoopFilterPostFilterRecordAt(0, 4, 4, 8),
		testFrameWorkLoopFilterPostFilterRecordAt(4, 4, 8, 8),
	)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	before := append([]byte(nil), output.Y.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 9}},
		},
		Output: output,
	}

	result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: make([]FrameWorkLoopFilterPostFilterEdge, 1),
	})
	if !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplyLoopFilterLumaEdges err=%v want %v", err, frame.ErrShortBuffer)
	}
	if !result.Active || result.Plan.DroppedEdges == 0 {
		t.Fatalf("result=%+v", result)
	}
	if !bytes.Equal(output.Y.Pix, before) {
		t.Fatal("short edge scratch mutated output")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesRejectsChromaCandidatesBeforeMutation(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	testFillFrameWorkLoopFilterPattern(output.Y)
	beforeY := append([]byte(nil), output.Y.Pix...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY: [2]uint8{8},
				LevelU: 7,
				LevelV: 9,
			},
		},
		Output: output,
	}

	result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: make([]FrameWorkLoopFilterPostFilterEdge, 3),
	})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplyLoopFilterLumaEdges err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if !result.Active || result.Plan.PlaneEdgeCandidates != [3]uint32{1, 1, 1} {
		t.Fatalf("result=%+v", result)
	}
	if !bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("chroma candidate rejection mutated output")
	}
}

func TestFrameWorkPostFilterContextApplyLoopFilterLumaEdgesAllocs(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          32,
		UpscaledWidth:       32,
		Height:              16,
		SuperResDenominator: 8,
	}
	first := testFrameWorkLoopFilterPostFilterRecordAt(0, 0, 4, 4)
	first.SkipTransform = true
	second := testFrameWorkLoopFilterPostFilterRecordAt(4, 0, 8, 4)
	second.TransformTree = tile.TransformTreeResult{
		Y:        tile.TransformSize16x16,
		Variable: true,
		Split:    [2]uint16{1, 0},
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, first, second)
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
		Output: output,
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, 4)
	allocs := testing.AllocsPerRun(1000, func() {
		testFillFrameWorkLoopFilterPattern(output.Y)
		result, err := ctx.ApplyLoopFilterLumaEdges(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			t.Fatal(err)
		}
		if result.Applied != 4 {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyLoopFilterLumaEdges allocated: %f", allocs)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanRejectsMissingActiveMap(t *testing.T) {
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          16,
				UpscaledWidth:       16,
				Height:              16,
				SuperResDenominator: 8,
			},
			LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{})
	if !errors.Is(err, threading.ErrInvalidBatch) {
		t.Fatalf("LoopFilterPostFilterPlan err=%v want %v", err, threading.ErrInvalidBatch)
	}
	if _, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{}); !errors.Is(err, threading.ErrInvalidBatch) {
		t.Fatalf("LoopFilterPostFilterScratchLen err=%v want %v", err, threading.ErrInvalidBatch)
	}
	if !plan.Active || plan.MICols != 4 || plan.MIRows != 4 {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkPostFilterContextLoopFilterPostFilterPlanRejectsIncompleteMap(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:          16,
		UpscaledWidth:       16,
		Height:              16,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filterMap := FrameWorkLoopFilterMap{
		Records: make([]threading.FrameWorkLoopFilterBlockRecord, cols*rows),
		Stride:  uint16(cols),
		Rows:    uint16(rows),
	}
	filterMap.Records[0] = testFrameWorkLoopFilterPostFilterRecord(cols, rows)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8}},
		},
	}

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if !errors.Is(err, threading.ErrInvalidBatch) {
		t.Fatalf("LoopFilterPostFilterPlan err=%v want %v", err, threading.ErrInvalidBatch)
	}
	if plan.Blocks != 1 || int(plan.Missing) != cols*rows-1 {
		t.Fatalf("plan=%+v", plan)
	}
}

func BenchmarkLoopFilterPostFilterPlan(b *testing.B) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			b.Fatal(err)
		}
		if plan.StoredEdges != 72 {
			b.Fatalf("stored edges=%d want 72", plan.StoredEdges)
		}
	}
}

func BenchmarkLoopFilterPostFilterPlanTrusted(b *testing.B) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
			Map:             filterMap,
			Edges:           edges,
			TrustedCoverage: true,
		})
		if err != nil {
			b.Fatal(err)
		}
		if plan.StoredEdges != 72 {
			b.Fatalf("stored edges=%d want 72", plan.StoredEdges)
		}
	}
}

func BenchmarkApplyLoopFilterEdges(b *testing.B) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkFillLoopFilterFrame(ctx.Output)
		result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		})
		if err != nil {
			b.Fatal(err)
		}
		if result.Applied != 72 {
			b.Fatalf("applied=%d want 72", result.Applied)
		}
	}
}

func BenchmarkApplyLoopFilterEdgesLargeScanFallback(b *testing.B) {
	ctx, edges := testLargeFrameWorkLoopFilterEdgeApplyFixture(b, frameWorkLoopFilterApplyScheduleCap+1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkFillLoopFilterFrame(ctx.Output)
		var result FrameWorkLoopFilterPostFilterApplyResult
		if err := ctx.applyLoopFilterEdgesInPlanePassOrder(&result, edges, nil, loopfilter.PlaneV); err != nil {
			b.Fatal(err)
		}
		if result.Applied != uint32(len(edges)) {
			b.Fatalf("applied=%d want %d", result.Applied, len(edges))
		}
	}
}

func BenchmarkApplyLoopFilterEdgesLargeScheduleScratch(b *testing.B) {
	ctx, edges := testLargeFrameWorkLoopFilterEdgeApplyFixture(b, frameWorkLoopFilterApplyScheduleCap+1024)
	schedule := make([]uint32, len(edges))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkFillLoopFilterFrame(ctx.Output)
		var result FrameWorkLoopFilterPostFilterApplyResult
		if err := ctx.applyLoopFilterEdgesInPlanePassOrder(&result, edges, schedule, loopfilter.PlaneV); err != nil {
			b.Fatal(err)
		}
		if result.Applied != uint32(len(edges)) {
			b.Fatalf("applied=%d want %d", result.Applied, len(edges))
		}
	}
}

func BenchmarkApplySupportedPostFiltersLoopFilterCDEF(b *testing.B) {
	ctx, filterMap, edges := benchmarkLoopFilterPostFilterFixture(b)
	ctx.Event.SequenceHeader.EnableCDEF = true
	ctx.Event.CDEF = parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{63},
	}
	cdefReq := testFrameWorkCDEFPostFilterRequest(b, ctx, ctx.Event)
	req := FrameWorkPostFilterRequest{
		LoopFilter: FrameWorkLoopFilterPostFilterRequest{
			Map:   filterMap,
			Edges: edges,
		},
		CDEF: cdefReq,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkFillLoopFilterFrame(ctx.Output)
		next, result, err := ctx.ApplySupportedPostFilters(req)
		if err != nil {
			b.Fatal(err)
		}
		if result.Completed != FrameWorkPostFilterLoopFilter|FrameWorkPostFilterCDEF ||
			result.LoopFilter.Applied != 72 ||
			result.CDEF.Units == 0 ||
			next.RemainingPostFilters() != 0 {
			b.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
		}
	}
}

func testFillFrameWorkLoopFilterPattern(plane frame.Plane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := byte(90)
			if (x/8+y/8)&1 != 0 {
				value = 100
			}
			plane.Pix[y*plane.Stride+x] = value
		}
	}
}

func testFillFrameWorkLoopFilterSamplePattern(plane frame.Plane, bytesPerSample int) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := uint16(360)
			if (x/8+y/8)&1 != 0 {
				value = 400
			}
			testSetFrameWorkLoopFilterSample(plane, bytesPerSample, x, y, value)
		}
	}
}

func testSetFrameWorkLoopFilterSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	binary.LittleEndian.PutUint16(plane.Pix[offset:], value)
}

func testFillFrameWorkLoopFilterOrderPattern(plane frame.Plane, rng *rand.Rand) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := 88 + rng.Intn(25)
			if x >= plane.Width/2 {
				value += rng.Intn(8)
			}
			if y >= plane.Height/2 {
				value += rng.Intn(8)
			}
			plane.Pix[y*plane.Stride+x] = byte(value)
		}
	}
}

func testCloneFrameWorkLoopFilterFrame(src *frame.Frame) frame.Frame {
	clone := *src
	clone.Y.Pix = append([]byte(nil), src.Y.Pix...)
	if !src.Format.MonoChrome {
		clone.U.Pix = append([]byte(nil), src.U.Pix...)
		clone.V.Pix = append([]byte(nil), src.V.Pix...)
	}
	return clone
}

func testApplyFrameWorkLoopFilterEdgesDirect(t *testing.T, output frame.Frame, sharpness uint8, edges []FrameWorkLoopFilterPostFilterEdge) {
	t.Helper()
	for plane := loopfilter.PlaneY; plane <= loopfilter.PlaneV; plane++ {
		for edgeKind := loopfilter.EdgeVertical; edgeKind <= loopfilter.EdgeHorizontal; edgeKind++ {
			for _, edge := range edges {
				if edge.Plane != plane || edge.Edge != edgeKind {
					continue
				}
				testApplyFrameWorkLoopFilterEdgeDirect(t, output, sharpness, edge)
			}
		}
	}
}

func testApplyFrameWorkLoopFilterEdgesStoredOrder(t *testing.T, output frame.Frame, sharpness uint8, edges []FrameWorkLoopFilterPostFilterEdge) {
	t.Helper()
	for _, edge := range edges {
		testApplyFrameWorkLoopFilterEdgeDirect(t, output, sharpness, edge)
	}
}

func testApplyFrameWorkLoopFilterEdgeDirect(t *testing.T, output frame.Frame, sharpness uint8, edge FrameWorkLoopFilterPostFilterEdge) {
	t.Helper()
	thresholds, err := loopfilter.ThresholdsForLevel(edge.Level, sharpness)
	if err != nil {
		t.Fatal(err)
	}
	plane, ok := frameWorkLoopFilterOutputPlane(output, edge.Plane)
	if !ok {
		t.Fatalf("missing output plane for edge=%+v", edge)
	}
	if err := loopfilter.FilterEdgeByWidth(edge.Width, plane, output.Layout.BytesPerSample, output.Format.BitDepth, edge.Edge, int32(edge.X4)*4, int32(edge.Y4)*4, int32(edge.Length4)*4, thresholds); err != nil {
		t.Fatal(err)
	}
}

func testFrameWorkLoopFilterPostFilterRecord(cols int, rows int) threading.FrameWorkLoopFilterBlockRecord {
	return testFrameWorkLoopFilterPostFilterRecordAt(0, 0, cols, rows)
}

func testFrameWorkLoopFilterPostFilterRecordAt(col0 int, row0 int, col1 int, row1 int) threading.FrameWorkLoopFilterBlockRecord {
	visibleW4 := uint8(col1 - col0)
	visibleH4 := uint8(row1 - row0)
	return threading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: threading.FrameWorkLoopFilterBlock{
			MICol:     uint16(col0),
			MIRow:     uint16(row0),
			MIColEnd:  uint16(col1),
			MIRowEnd:  uint16(row1),
			X4:        uint8(col0),
			Y4:        uint8(row0),
			Size:      tile.BlockSize16x16,
			VisibleW4: visibleW4,
			VisibleH4: visibleH4,
		},
		TransformTree: tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true},
		SegmentID:     0,
		RefFrame:      0,
		Mode:          loopfilter.ModeDeltaClassZero,
	}
}

func testFrameWorkLoopFilterPostFilterMap(t testing.TB, size parser.FrameSize, records ...threading.FrameWorkLoopFilterBlockRecord) FrameWorkLoopFilterMap {
	t.Helper()
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	filterMap := FrameWorkLoopFilterMap{
		Records: make([]threading.FrameWorkLoopFilterBlockRecord, cols*rows),
		Stride:  uint16(cols),
		Rows:    uint16(rows),
	}
	stride := int(filterMap.Stride)
	for _, record := range records {
		for miRow := record.Block.MIRow; miRow < record.Block.MIRowEnd; miRow++ {
			row := int(miRow) * stride
			for miCol := record.Block.MICol; miCol < record.Block.MIColEnd; miCol++ {
				filterMap.Records[row+int(miCol)] = record
			}
		}
	}
	return filterMap
}

func benchmarkLoopFilterPostFilterFixture(b testing.TB) (FrameWorkPostFilterContext, FrameWorkLoopFilterMap, []FrameWorkLoopFilterPostFilterEdge) {
	b.Helper()
	const width = 64
	const height = 64
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	records := make([]threading.FrameWorkLoopFilterBlockRecord, 0, 16)
	for row := 0; row < 16; row += 4 {
		for col := 0; col < 16; col += 4 {
			records = append(records, testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4))
		}
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(b, size, records...)
	output := testFrameWorkCDEFFrame(b, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{63, 63},
				LevelU:    63,
				LevelV:    63,
				Sharpness: 1,
			},
		},
		Output:        output,
		LoopFilterMap: &filterMap,
	}
	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		b.Fatal(err)
	}
	edges, err := scratch.BindEdges(make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges))
	if err != nil {
		b.Fatal(err)
	}
	if scratch.Edges != 72 {
		b.Fatalf("edge scratch=%d want 72", scratch.Edges)
	}
	return ctx, filterMap, edges
}

func testLargeFrameWorkLoopFilterEdgeApplyFixture(t testing.TB, count int) (FrameWorkPostFilterContext, []FrameWorkLoopFilterPostFilterEdge) {
	t.Helper()
	const width = 512
	const height = 512
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	benchmarkFillLoopFilterFrame(output)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{32, 32},
				LevelU:    32,
				LevelV:    32,
				Sharpness: 1,
			},
		},
		Output: output,
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, count)
	for i := range edges {
		plane := loopfilter.Plane(i % 3)
		edgeKind := loopfilter.Edge((i / 3) & 1)
		units := 128
		if plane != loopfilter.PlaneY {
			units = 64
		}
		x4 := uint16(i % (units - 1))
		y4 := uint16(1 + ((i / units) % (units - 2)))
		if edgeKind == loopfilter.EdgeVertical {
			x4 = uint16(1 + (i % (units - 2)))
			y4 = uint16((i / units) % units)
		}
		edges[i] = FrameWorkLoopFilterPostFilterEdge{
			X4:         x4,
			Y4:         y4,
			Length4:    1,
			BlockMICol: x4,
			BlockMIRow: y4,
			Plane:      plane,
			Edge:       edgeKind,
			Level:      32,
			Width:      4,
			Transform:  tile.TransformSize4x4,
		}
	}
	return ctx, edges
}

func benchmarkFillLoopFilterFrame(output *frame.Frame) {
	testFillFrameWorkLoopFilterPattern(output.Y)
	testFillFrameWorkLoopFilterPattern(output.U)
	testFillFrameWorkLoopFilterPattern(output.V)
}

// TestFrameWorkPostFilterContextApplyLoopFilterEdgesEnhancementLayerResolution
// pins loop-filter dispatch to the per-event frame size by exercising a
// 1280x720 frame end-to-end. SVC enhancement layers decode at their own
// resolution, so the loop filter must not assume a global frame size: plane
// sizes, the MI grid, edge fit checks, and chroma subsampling must all derive
// from the current event's FrameSize/SequenceHeader, not any stream-wide
// maximum.
func TestFrameWorkPostFilterContextApplyLoopFilterEdgesEnhancementLayerResolution(t *testing.T) {
	const width = 1280
	const height = 720
	const miBlock = 16 // BlockSize64x64 covers a 16x16 MI superblock.

	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 320 || rows != 180 {
		t.Fatalf("frameWorkLoopFilterMapGrid(1280x720)=(%d,%d) want (320,180)", cols, rows)
	}

	records := make([]threading.FrameWorkLoopFilterBlockRecord, 0, 256)
	for row := 0; row < rows; row += miBlock {
		blockRows := miBlock
		if row+blockRows > rows {
			blockRows = rows - row
		}
		for col := 0; col < cols; col += miBlock {
			blockCols := miBlock
			if col+blockCols > cols {
				blockCols = cols - col
			}
			records = append(records, testFrameWorkLoopFilterEnhancementLayerRecord(col, row, blockCols, blockRows))
		}
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)

	output := testFrameWorkCDEFFrame(t, frame.Format{
		Width:        width,
		Height:       height,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	})
	testFillFrameWorkLoopFilterPattern(output.Y)
	testFillFrameWorkLoopFilterPattern(output.U)
	testFillFrameWorkLoopFilterPattern(output.V)
	beforeY := append([]byte(nil), output.Y.Pix...)
	beforeU := append([]byte(nil), output.U.Pix...)
	beforeV := append([]byte(nil), output.V.Pix...)

	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter: parser.LoopFilterParams{
				LevelY:    [2]uint8{32, 32},
				LevelU:    20,
				LevelV:    20,
				Sharpness: 1,
			},
		},
		Output:        output,
		LoopFilterMap: &filterMap,
	}

	scratch, err := ctx.LoopFilterPostFilterScratchLen(FrameWorkLoopFilterPostFilterRequest{Map: filterMap})
	if err != nil {
		t.Fatal(err)
	}
	if scratch.Edges == 0 {
		t.Fatalf("scratch=%+v want nonzero", scratch)
	}
	edges := make([]FrameWorkLoopFilterPostFilterEdge, scratch.Edges)

	plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{Map: filterMap, Edges: edges})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || int(plan.MICols) != cols || int(plan.MIRows) != rows {
		t.Fatalf("plan=%+v want active grid %dx%d", plan, cols, rows)
	}
	if int(plan.Cells) != cols*rows {
		t.Fatalf("plan.Cells=%d want %d", plan.Cells, cols*rows)
	}
	if int(plan.Blocks) != len(records) {
		t.Fatalf("plan.Blocks=%d want %d", plan.Blocks, len(records))
	}
	if plan.Missing != 0 {
		t.Fatalf("plan.Missing=%d want 0", plan.Missing)
	}
	if plan.DroppedEdges != 0 || plan.StoredEdges != plan.EdgeCandidates {
		t.Fatalf("plan storage=%+v want all stored", plan)
	}
	if plan.PlaneEdgeCandidates[loopfilter.PlaneY] == 0 ||
		plan.PlaneEdgeCandidates[loopfilter.PlaneU] == 0 ||
		plan.PlaneEdgeCandidates[loopfilter.PlaneV] == 0 {
		t.Fatalf("plan PlaneEdgeCandidates=%v want all planes nonzero", plan.PlaneEdgeCandidates)
	}

	// Verify the bottom-right edge candidates use the per-event frame size:
	// at least one stored vertical luma edge sits beyond MICol=128 (X=512
	// luma) and at least one horizontal edge sits beyond MIRow=128 (Y=512
	// luma). Without per-event sizing, edges in the >512 region could not
	// resolve a valid width and would be skipped or rejected.
	farLumaVertical := false
	farLumaHorizontal := false
	farChromaVertical := false
	for _, e := range edges[:int(plan.StoredEdges)] {
		if e.Plane == loopfilter.PlaneY && e.Edge == loopfilter.EdgeVertical && e.X4 >= 128 {
			farLumaVertical = true
		}
		if e.Plane == loopfilter.PlaneY && e.Edge == loopfilter.EdgeHorizontal && e.Y4 >= 128 {
			farLumaHorizontal = true
		}
		if (e.Plane == loopfilter.PlaneU || e.Plane == loopfilter.PlaneV) &&
			e.Edge == loopfilter.EdgeVertical && e.X4 >= 64 {
			farChromaVertical = true
		}
	}
	if !farLumaVertical {
		t.Fatal("no luma vertical edge stored past MICol=128 (X=512 luma)")
	}
	if !farLumaHorizontal {
		t.Fatal("no luma horizontal edge stored past MIRow=128 (Y=512 luma)")
	}
	if !farChromaVertical {
		t.Fatal("no chroma vertical edge stored past chroma X4=64")
	}

	wantFrame := testCloneFrameWorkLoopFilterFrame(output)
	testApplyFrameWorkLoopFilterEdgesDirect(t, wantFrame, ctx.Event.LoopFilter.Sharpness, edges[:int(plan.StoredEdges)])

	result, err := ctx.ApplyLoopFilterEdges(FrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Active || result.Applied == 0 || result.Plan.StoredEdges != plan.StoredEdges {
		t.Fatalf("result=%+v", result)
	}
	if bytes.Equal(output.Y.Pix, beforeY) {
		t.Fatal("1280x720 loop filter did not change luma samples")
	}
	if bytes.Equal(output.U.Pix, beforeU) || bytes.Equal(output.V.Pix, beforeV) {
		t.Fatal("1280x720 loop filter did not change chroma samples")
	}
	if !bytes.Equal(output.Y.Pix, wantFrame.Y.Pix) ||
		!bytes.Equal(output.U.Pix, wantFrame.U.Pix) ||
		!bytes.Equal(output.V.Pix, wantFrame.V.Pix) {
		t.Fatal("1280x720 loop filter bridge output did not match direct edge application")
	}
}

// testFrameWorkLoopFilterEnhancementLayerRecord builds a loop-filter block
// record at a frame-relative MI position. Unlike the small-fixture helpers, X4
// and Y4 are pinned to the block's local superblock origin (0,0) so the
// transform-tree validator's MaxBlockModeSlots check passes regardless of how
// far into the frame the block sits. Each block is one SkipTransform
// superblock so the test exercises whole-block luma and chroma edges without
// recursing into variable transform splits.
func testFrameWorkLoopFilterEnhancementLayerRecord(miCol int, miRow int, miCols int, miRows int) threading.FrameWorkLoopFilterBlockRecord {
	return threading.FrameWorkLoopFilterBlockRecord{
		Valid: true,
		Block: threading.FrameWorkLoopFilterBlock{
			MICol:     uint16(miCol),
			MIRow:     uint16(miRow),
			MIColEnd:  uint16(miCol + miCols),
			MIRowEnd:  uint16(miRow + miRows),
			X4:        0,
			Y4:        0,
			Size:      tile.BlockSize64x64,
			VisibleW4: uint8(miCols),
			VisibleH4: uint8(miRows),
		},
		TransformTree: tile.TransformTreeResult{
			Y:     tile.TransformSize32x32,
			UV:    tile.TransformSize16x16,
			HasUV: true,
		},
		SkipTransform: true,
		SegmentID:     0,
		RefFrame:      0,
		Mode:          loopfilter.ModeDeltaClassZero,
	}
}

// TestLoopFilterPostFilterPlanBandedMatchesWhole proves the banded trusted
// sweep partitions the whole-frame plan exactly: bands over block-origin rows
// emit the same edges in the same order once concatenated band by band.
func TestLoopFilterPostFilterPlanBandedMatchesWhole(t *testing.T) {
	const width, height = 128, 128
	size := parser.FrameSize{
		CodedWidth:          width,
		UpscaledWidth:       width,
		Height:              height,
		SuperResDenominator: 8,
	}
	var records []threading.FrameWorkLoopFilterBlockRecord
	// Mixed geometry: 16x16 blocks, one skipped, one var-tx split, plus a
	// run of 8x8s, exercising tx-size variety across band boundaries.
	for row := 0; row < 32; row += 4 {
		for col := 0; col < 32; col += 4 {
			rec := testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4)
			switch (row/4 + col/4) % 3 {
			case 1:
				rec.SkipTransform = true
			case 2:
				rec.TransformTree = tile.TransformTreeResult{
					Y:        tile.TransformSize16x16,
					UV:       tile.TransformSize8x8,
					HasUV:    true,
					Variable: true,
					Split:    [2]uint16{1, 0},
				}
			}
			records = append(records, rec)
		}
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, records...)
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			SequenceHeader: testSequence(),
			FrameSize:      size,
			LoopFilter:     parser.LoopFilterParams{LevelY: [2]uint8{8, 8}, LevelU: 8, LevelV: 8},
		},
	}
	upper, err := ctx.LoopFilterPostFilterScratchUpperBound()
	if err != nil {
		t.Fatal(err)
	}
	whole := make([]FrameWorkLoopFilterPostFilterEdge, upper.Edges)
	wholePlan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
		Map: filterMap, Edges: whole, TrustedCoverage: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	banded := make([]FrameWorkLoopFilterPostFilterEdge, 0, upper.Edges)
	bandScratch := make([]FrameWorkLoopFilterPostFilterEdge, upper.Edges)
	const bandRows = 12 // deliberately unaligned to block rows
	total := uint32(0)
	for r0 := 0; r0 < int(wholePlan.MIRows); r0 += bandRows {
		r1 := min(r0+bandRows, int(wholePlan.MIRows))
		plan, err := ctx.LoopFilterPostFilterPlan(FrameWorkLoopFilterPostFilterRequest{
			Map: filterMap, Edges: bandScratch, TrustedCoverage: true,
			MIRowStart: r0, MIRowEnd: r1,
		})
		if err != nil {
			t.Fatalf("band %d-%d: %v", r0, r1, err)
		}
		if plan.DroppedEdges != 0 {
			t.Fatalf("band %d-%d dropped %d", r0, r1, plan.DroppedEdges)
		}
		banded = append(banded, bandScratch[:plan.StoredEdges]...)
		total += plan.StoredEdges
	}
	if total != wholePlan.StoredEdges {
		t.Fatalf("banded stored %d, whole %d", total, wholePlan.StoredEdges)
	}
	for i := range banded {
		if banded[i] != whole[i] {
			t.Fatalf("edge %d differs: banded %+v whole %+v", i, banded[i], whole[i])
		}
	}
}

func TestPostFilterParallelRunnerCDEFNoPoolAllocsAndParity(t *testing.T) {
	const width, height = 256, 512 // eight CDEF unit rows; two worker ranges are active
	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth: width, UpscaledWidth: width, Height: height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping: 5, StrengthCount: 2,
			YStrength:  [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength: [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	format := frame.Format{
		Width: width, Height: height, BitDepth: 8,
		SubsamplingX: true, SubsamplingY: true, Align: 64,
	}
	build := func(parallel *FrameWorkPostFilterParallel) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		out := testFrameWorkCDEFFrame(t, format)
		testFillFrameWorkCDEFPlane(out.Y)
		testFillFrameWorkCDEFPlane(out.U)
		testFillFrameWorkCDEFPlane(out.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: out, Parallel: parallel}
		ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{
			FrameWorkFrameContext: threading.FrameWorkFrameContext{
				Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
				FrameSize: event.FrameSize,
				CDEF:      event.CDEF,
			},
		}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		indices := make([]uint8, length)
		for i := range indices {
			indices[i] = uint8(i % 2)
		}
		indexMap, err := batch.BindCDEFIndexMap(indices, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range indexMap.Read {
			indexMap.Read[i] = true
		}
		scratchSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directions, variances, input, unitDst := testFrameWorkCDEFScratchStorage(scratchSize)
		req, err := scratchSize.BindRequest(indexMap, samples, dst, directions, variances, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return out, ctx, req
	}

	serialOut, serialCtx, serialReq := build(nil)
	serialResult, err := serialCtx.ApplyCDEFPostFilter(serialReq)
	if err != nil {
		t.Fatal(err)
	}
	if serialResult.Units == 0 || serialResult.Blocks == 0 {
		t.Fatalf("serial CDEF did no filtering: %+v", serialResult)
	}

	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	defer parallel.Close()
	parallelOut, parallelCtx, parallelReq := build(parallel)
	initialY := append([]byte(nil), parallelOut.Y.Pix...)
	initialU := append([]byte(nil), parallelOut.U.Pix...)
	initialV := append([]byte(nil), parallelOut.V.Pix...)
	apply := func() {
		copy(parallelOut.Y.Pix, initialY)
		copy(parallelOut.U.Pix, initialU)
		copy(parallelOut.V.Pix, initialV)
		result, handled, err := parallelCtx.applyCDEFPostFilterParallel(parallelReq, 1)
		if err != nil {
			t.Fatalf("parallel CDEF: %v", err)
		}
		if !handled || result.Units == 0 || result.Blocks == 0 {
			t.Fatalf("parallel CDEF did not process the active bands: handled=%v result=%+v", handled, result)
		}
	}
	apply() // allocate runner pool and scratch once before measuring
	if parallel.ownedPool == nil || parallel.ownedWorkers != 2 {
		t.Fatalf("no-pool apply did not retain a two-worker owned pool: pool=%p workers=%d", parallel.ownedPool, parallel.ownedWorkers)
	}
	allocs := testing.AllocsPerRun(5, apply)
	if allocs != 0 {
		t.Fatalf("warmed two-worker CDEF runner allocated %g times per apply", allocs)
	}
	if len(parallel.cdef) < 2 || parallel.cdef[0].result.Blocks == 0 || parallel.cdef[1].result.Blocks == 0 {
		t.Fatalf("8-bit CDEF did not use both worker ranges: %+v", parallel.cdef)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, parallelOut)

	oldPool := parallel.ownedPool
	parallel.Workers = 3
	apply()
	if parallel.ownedPool == oldPool || parallel.ownedWorkers != 3 {
		t.Fatalf("changing worker count did not rebuild the owned pool: old=%p new=%p workers=%d", oldPool, parallel.ownedPool, parallel.ownedWorkers)
	}
	if err := oldPool.RunRangesRunner(1, 1, postFilterNoopRangeRunner{}); err != threading.ErrPoolClosed {
		t.Fatalf("replaced pool RunRangesRunner error=%v, want ErrPoolClosed", err)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, parallelOut)

	parallel.Close()
	parallel.Close()
	if parallel.ownedPool != nil {
		t.Fatal("Close retained the owned pool")
	}
}

func TestPostFilterParallelRunnerBorrowsContextPool(t *testing.T) {
	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	ctx, req, serialOut, gotOut := buildPostFilterParallelCDEFPair(t, parallel, 8)
	borrowed, err := threading.NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer borrowed.Close()
	ctx.pool = borrowed
	if _, handled, err := ctx.applyCDEFPostFilterParallel(req, 1); err != nil || !handled {
		t.Fatalf("borrowed-pool CDEF handled=%v err=%v", handled, err)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, gotOut)
	parallel.Close()
	if parallel.ownedPool != nil {
		t.Fatalf("borrowed context pool was captured as owned: %p", parallel.ownedPool)
	}
	if err := borrowed.RunRangesRunner(1, 1, postFilterNoopRangeRunner{}); err != nil {
		t.Fatalf("parallel Close closed borrowed pool: %v", err)
	}
}

func TestPostFilterParallelRunnerCDEFSnapshotMatchesSerial(t *testing.T) {
	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	defer parallel.Close()
	ctx, req, serialOut, gotOut := buildPostFilterParallelCDEFPair(t, parallel, 10)
	result, handled, err := ctx.applyCDEFPostFilterParallel(req, 1)
	if err != nil || !handled {
		t.Fatalf("HBD snapshot parallel CDEF handled=%v err=%v", handled, err)
	}
	if result.Units == 0 || result.Blocks == 0 || len(parallel.cdef) < 2 ||
		parallel.cdef[0].result.Blocks == 0 || parallel.cdef[1].result.Blocks == 0 {
		t.Fatalf("HBD CDEF did not use two active worker ranges: result=%+v workers=%+v", result, parallel.cdef)
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, gotOut)
}

func buildPostFilterParallelCDEFPair(t *testing.T, parallel *FrameWorkPostFilterParallel, bitDepth int) (FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest, *frame.Frame, *frame.Frame) {
	t.Helper()
	const width, height = 256, 512
	seq := testSequence()
	seq.EnableCDEF = true
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth: width, UpscaledWidth: width, Height: height,
			SuperResDenominator: 8,
		},
		CDEF: parser.CDEFParams{
			Damping: 5, StrengthCount: 2,
			YStrength:  [parser.MaxCDEFStrengths]uint8{9, 31},
			UVStrength: [parser.MaxCDEFStrengths]uint8{5, 17},
		},
	}
	format := frame.Format{Width: width, Height: height, BitDepth: uint8(bitDepth), SubsamplingX: true, SubsamplingY: true, Align: 64}
	makeContext := func(par *FrameWorkPostFilterParallel) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkCDEFPostFilterRequest) {
		out := testFrameWorkCDEFFrame(t, format)
		fillPostFilterParallelCDEFPlane(out.Y, bitDepth)
		fillPostFilterParallelCDEFPlane(out.U, bitDepth)
		fillPostFilterParallelCDEFPlane(out.V, bitDepth)
		ctx := (FrameWorkPostFilterContext{Event: event, Output: out, Parallel: par}).WithCompletedPostFilters(FrameWorkPostFilterLoopFilter)
		batch := threading.FrameWorkBatch{FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:  threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
			FrameSize: event.FrameSize, CDEF: event.CDEF,
		}}
		_, _, length, err := batch.CDEFIndexMapShape()
		if err != nil {
			t.Fatal(err)
		}
		indices := make([]uint8, length)
		for i := range indices {
			indices[i] = uint8(i % 2)
		}
		indexMap, err := batch.BindCDEFIndexMap(indices, make([]bool, length))
		if err != nil {
			t.Fatal(err)
		}
		for i := range indexMap.Read {
			indexMap.Read[i] = true
		}
		scratchSize, err := ctx.CDEFPostFilterScratchLen()
		if err != nil {
			t.Fatal(err)
		}
		samples, dst, directions, variances, input, unitDst := testFrameWorkCDEFScratchStorage(scratchSize)
		req, err := scratchSize.BindRequest(indexMap, samples, dst, directions, variances, input, unitDst)
		if err != nil {
			t.Fatal(err)
		}
		return out, ctx, req
	}
	serialOut, serialCtx, serialReq := makeContext(nil)
	if _, err := serialCtx.ApplyCDEFPostFilter(serialReq); err != nil {
		t.Fatal(err)
	}
	gotOut, gotCtx, gotReq := makeContext(parallel)
	return gotCtx, gotReq, serialOut, gotOut
}

func fillPostFilterParallelCDEFPlane(plane frame.Plane, bitDepth int) {
	bytesPerSample := 1
	if bitDepth > 8 {
		bytesPerSample = 2
	}
	maxSample := (1 << bitDepth) - 1
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			value := (x*37 + y*53 + (x^y)*17) & maxSample
			offset := y*plane.Stride + x*bytesPerSample
			plane.Pix[offset] = byte(value)
			if bytesPerSample == 2 {
				plane.Pix[offset+1] = byte(value >> 8)
			}
		}
	}
}

func TestPostFilterParallelRunnerLoopFilterNoPoolAllocsAndParity(t *testing.T) {
	const width, height = 256, 512
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	seq := lfMaskApply420Sequence()
	lf := parser.LoopFilterParams{LevelY: [2]uint8{20, 20}, LevelU: 16, LevelV: 16, Sharpness: 1}
	event := lfMaskApplyEvent(seq, size, lf)
	format := frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}
	var records []lfMaskApplyRecord
	for row := 0; row < height/4; row += 4 {
		for col := 0; col < width/4; col += 4 {
			records = append(records, lfMaskApplyRecord{rec: testFrameWorkLoopFilterPostFilterRecordAt(col, row, col+4, row+4)})
		}
	}
	plainRecords := make([]threading.FrameWorkLoopFilterBlockRecord, len(records))
	for i := range records {
		plainRecords[i] = records[i].rec
	}
	filterMap := testFrameWorkLoopFilterPostFilterMap(t, size, plainRecords...)
	build := func(parallel *FrameWorkPostFilterParallel) (*frame.Frame, FrameWorkPostFilterContext, *threading.FrameWorkLoopFilterMasks) {
		out := testFrameWorkCDEFFrame(t, format)
		lfMaskApplyFillFrame(out)
		masks := buildLoopFilterMasksFromRecords(t, event, size, records, 4)
		ctx := FrameWorkPostFilterContext{Event: event, Output: out, LoopFilterMap: &filterMap, LoopFilterMasks: masks, Parallel: parallel}
		return out, ctx, masks
	}
	serialOut, serialCtx, serialMasks := build(nil)
	if _, err := serialCtx.ApplyLoopFilterEdgesFromMasks(serialMasks, filterMap); err != nil {
		t.Fatal(err)
	}

	parallel := &FrameWorkPostFilterParallel{Workers: 2}
	defer parallel.Close()
	parallelOut, parallelCtx, _ := build(parallel)
	initialY := append([]byte(nil), parallelOut.Y.Pix...)
	initialU := append([]byte(nil), parallelOut.U.Pix...)
	initialV := append([]byte(nil), parallelOut.V.Pix...)
	apply := func() {
		copy(parallelOut.Y.Pix, initialY)
		copy(parallelOut.U.Pix, initialU)
		copy(parallelOut.V.Pix, initialV)
		result, handled, err := parallelCtx.applyLoopFilterMaskBandsParallel(filterMap)
		if err != nil {
			t.Fatalf("parallel loop filter: %v", err)
		}
		if !handled || !result.Active {
			t.Fatalf("parallel loop filter did no work: handled=%v result=%+v", handled, result)
		}
	}
	apply()
	if parallel.ownedPool == nil || parallel.ownedWorkers != 2 {
		t.Fatalf("loop-filter path did not retain an owned two-worker pool: pool=%p workers=%d", parallel.ownedPool, parallel.ownedWorkers)
	}
	allocs := testing.AllocsPerRun(5, apply)
	if allocs != 0 {
		t.Fatalf("warmed two-worker loop-filter runner allocated %g times per apply", allocs)
	}
	if bytes.Equal(initialY, parallelOut.Y.Pix) {
		t.Fatal("parallel loop filter left the active luma plane unchanged")
	}
	lfMaskApplyAssertFramesEqual(t, serialOut, parallelOut)
}

type postFilterNoopRangeRunner struct{}

func (postFilterNoopRangeRunner) RunRange(int, int, int) error { return nil }

// TestLoopRestorationPooledMatchesSerial proves the pooled 8-bit in-place
// restoration apply (applyLoopRestorationPostFilterMaybePooled with a multi-lane
// pool) is byte-identical to the serial whole-plane apply, for several worker
// counts, under -race. The frame is tall enough (8 luma RU rows) that the RU-row
// band split engages, and every plane carries a Wiener filter so the kernels run.
// Identical pixels across worker counts is the worker-invariance proof (mirrors
// TestCDEFPostFilterPooledMatchesSerial).
func TestLoopRestorationPooledMatchesSerial(t *testing.T) {
	const width, height = 256, 512
	seq := testSequence()
	seq.EnableRestoration = true
	seq.ColorConfig.BitDepth = 8
	seq.ColorConfig.SubsamplingX = true
	seq.ColorConfig.SubsamplingY = true
	size := parser.FrameSize{CodedWidth: width, UpscaledWidth: width, Height: height, SuperResDenominator: 8}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		Restoration: parser.RestorationParams{
			Type:       [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationWiener, parser.RestorationWiener},
			UnitSizeY:  64,
			UnitSizeUV: 64,
		},
	}
	format := frame.Format{Width: width, Height: height, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64}

	// Records (Wiener on all planes) and zero boundary buffers shared read-only.
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:    threading.FrameWorkSequenceContextFromHeader(seq),
			FrameSize:   size,
			Restoration: event.Restoration,
		},
	}
	plan, err := batch.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	buffers, err := batch.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, plan.UnitRecordLen()),
		make([]uint16, plan.BoundaryBufferLen()),
		make([]uint16, plan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	defaultWiener := av1restoration.DefaultWienerInfo()
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type != parser.RestorationWiener {
			continue
		}
		for i := range buffers.Records[plane] {
			buffers.Records[plane][i].Unit.Type = parser.RestorationWiener
			buffers.Records[plane][i].Unit.Wiener = defaultWiener
		}
	}

	build := func(pool *threading.Pool) (*frame.Frame, FrameWorkPostFilterContext, FrameWorkRestorationPostFilterRequest) {
		output := testFrameWorkCDEFFrame(t, format)
		testFillFrameWorkCDEFPlane(output.Y)
		testFillFrameWorkCDEFPlane(output.U)
		testFillFrameWorkCDEFPlane(output.V)
		ctx := FrameWorkPostFilterContext{Event: event, Output: output, pool: pool}
		scratchSize, err := ctx.LoopRestorationPostFilterScratchLen(buffers.Records, false)
		if err != nil {
			t.Fatal(err)
		}
		req := FrameWorkRestorationPostFilterRequest{
			Records:     buffers.Records,
			Boundaries:  buffers.Boundaries,
			DataScratch: make([]uint16, scratchSize.Samples.DataLen),
			DstScratch:  make([]uint16, scratchSize.Samples.DstLen),
			Scratch:     testFrameWorkRestorationPostFilterScratch(scratchSize.Apply),
			Pool: FrameWorkRestorationPoolScratch{
				Bands:      scratchSize.Pool.Bands,
				BandData:   make([]uint16, scratchSize.Pool.TotalData()),
				BandWiener: make([]uint16, scratchSize.Pool.TotalWiener()),
				BandSGR:    make([]int32, scratchSize.Pool.TotalSGR()),
			},
		}
		return output, ctx, req
	}

	serialOut, serialCtx, serialReq := build(nil)
	if _, err := serialCtx.ApplyLoopRestorationPostFilter(serialReq); err != nil {
		t.Fatal(err)
	}

	for _, workers := range []int{2, 3, 4, 8} {
		pool, err := threading.NewPool(workers)
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		pooledOut, pooledCtx, pooledReq := build(pool)
		if pooledReq.Pool.Bands < 2 {
			pool.Close()
			t.Fatalf("workers=%d: pool scratch not sized (Bands=%d); pooled path would not engage", workers, pooledReq.Pool.Bands)
		}
		res, err := pooledCtx.applyLoopRestorationPostFilterMaybePooled(pooledReq)
		if err != nil {
			pool.Close()
			t.Fatalf("workers=%d apply: %v", workers, err)
		}
		if res.FilteredRecords == 0 {
			pool.Close()
			t.Fatalf("workers=%d degenerate: pooled apply filtered no records", workers)
		}
		for _, planes := range [][2]frame.Plane{{serialOut.Y, pooledOut.Y}, {serialOut.U, pooledOut.U}, {serialOut.V, pooledOut.V}} {
			s, p := planes[0], planes[1]
			for y := 0; y < s.Height; y++ {
				for x := 0; x < s.Width; x++ {
					if s.Pix[y*s.Stride+x] != p.Pix[y*p.Stride+x] {
						pool.Close()
						t.Fatalf("workers=%d pooled differs at (%d,%d): serial=%d pooled=%d", workers, x, y, s.Pix[y*s.Stride+x], p.Pix[y*p.Stride+x])
					}
				}
			}
		}
		pool.Close()
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterAllNoneRecords(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x5a

	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if got := ctx.ActivePostFilters(); got != FrameWorkPostFilterLoopRestoration {
		t.Fatalf("ActivePostFilters=%b want %b", got, FrameWorkPostFilterLoopRestoration)
	}

	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.UnitRecordLen() != 1 {
		t.Fatalf("plan active=%v records=%d", plan.Active, plan.UnitRecordLen())
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, make([]tile.RestorationUnitRecord, plan.UnitRecordLen()))
	if err != nil {
		t.Fatal(err)
	}
	for plane := 0; plane < int(plan.Planes); plane++ {
		if plan.Grids[plane].Type == parser.RestorationNone {
			continue
		}
		if err := tile.ResetRestorationPlaneRecords(plan.Grids[plane], records[plane]); err != nil {
			t.Fatal(err)
		}
	}

	size, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{
		Records:     records,
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Records != 1 || result.FilteredRecords != 0 {
		t.Fatalf("result=%+v", result)
	}
	if output.Y.Pix[0] != 0x5a {
		t.Fatalf("output sample=%d want 0x5a", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterDefaultsBuffersFromContext(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x6b

	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:    threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader),
			FrameSize:   event.FrameSize,
			Restoration: event.Restoration,
		},
	}
	plan, err := batch.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	buffers, err := batch.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, plan.UnitRecordLen()),
		make([]uint16, plan.BoundaryBufferLen()),
		make([]uint16, plan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, RestorationFrameBuffers: &buffers}
	pipelineSize, err := ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if pipelineSize.Restoration.Samples.DataLen == 0 || pipelineSize.Restoration.Samples.DstLen != 0 {
		t.Fatalf("pipeline scratch=%+v", pipelineSize.Restoration)
	}
	size, err := ctx.LoopRestorationPostFilterScratchLen(buffers.Records, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Records != 1 || result.FilteredRecords != 0 {
		t.Fatalf("result=%+v", result)
	}
	if output.Y.Pix[0] != 0x6b {
		t.Fatalf("output sample=%d want 0x6b", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextSaveRestorationBoundariesSkipsOptimized(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := FrameWorkRestorationPostFilterRequest{
		Boundaries: [3]tile.RestorationStripeBoundaries{
			{Above: []uint16{1}},
		},
	}
	if err := ctx.saveRestorationBoundariesForRequest(req, false); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("saveRestorationBoundariesForRequest err=%v want %v", err, tile.ErrInvalidPlan)
	}
	req.Optimized = true
	if err := ctx.saveRestorationBoundariesForRequest(req, false); err != nil {
		t.Fatalf("optimized saveRestorationBoundariesForRequest err=%v", err)
	}
	if err := ctx.saveRestorationBoundariesForRequest(req, true); err != nil {
		t.Fatalf("optimized after-CDEF saveRestorationBoundariesForRequest err=%v", err)
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterRejectsEarlierActiveStages(t *testing.T) {
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			FrameSize: parser.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				SuperResDenominator: 8,
			},
			CDEF: parser.CDEFParams{
				StrengthCount: 1,
				YStrength:     [parser.MaxCDEFStrengths]uint8{1},
			},
			Restoration: parser.RestorationParams{
				Type:      [3]parser.RestorationType{parser.RestorationWiener},
				UnitSizeY: 64,
			},
		},
	}
	_, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplyLoopRestorationPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplyLoopRestorationPostFilterAllowsCompletedEarlierStages(t *testing.T) {
	const width = 64
	const height = 64

	pool := testFramePoolForSize(t, width, height, 1)
	_, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x33

	event := Event{
		SequenceHeader: testSequence(),
		FrameSize: parser.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResEnabled:     true,
			SuperResDenominator: 16,
		},
		LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1}},
		CDEF: parser.CDEFParams{
			StrengthCount: 1,
			YStrength:     [parser.MaxCDEFStrengths]uint8{1},
		},
		Restoration: parser.RestorationParams{
			Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
			UnitSizeY: 64,
		},
		FilmGrain: parser.FilmGrainParams{Apply: true},
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}.WithCompletedPostFilters(
		FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes,
	)
	if got := ctx.RemainingPostFilters(); got != FrameWorkPostFilterLoopRestoration|FrameWorkPostFilterFilmGrain {
		t.Fatalf("remaining=%b", got)
	}
	plan, err := ctx.LoopRestorationPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	records, err := tile.BindRestorationFrameRecordBuffers(plan, make([]tile.RestorationUnitRecord, plan.UnitRecordLen()))
	if err != nil {
		t.Fatal(err)
	}
	if err := tile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
		t.Fatal(err)
	}
	size, err := ctx.LoopRestorationPostFilterScratchLen(records, false)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ctx.ApplyLoopRestorationPostFilter(FrameWorkRestorationPostFilterRequest{
		Records:     records,
		DataScratch: make([]uint16, size.Samples.DataLen),
		DstScratch:  make([]uint16, size.Samples.DstLen),
		Scratch:     testFrameWorkRestorationPostFilterScratch(size.Apply),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 1 || result.Records != 1 || result.FilteredRecords != 0 {
		t.Fatalf("result=%+v", result)
	}
	if output.Y.Pix[0] != 0x33 {
		t.Fatalf("output sample=%d want 0x33", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplyCallerPostFiltersSuperResRestorationAllocs(t *testing.T) {
	const codedWidth = 32
	const upscaledWidth = 48
	const height = 64

	seq := testSequence()
	size := parser.FrameSize{
		CodedWidth:          codedWidth,
		UpscaledWidth:       upscaledWidth,
		Height:              height,
		SuperResEnabled:     true,
		SuperResDenominator: 16,
	}
	restoration := parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener, parser.RestorationNone, parser.RestorationNone},
		UnitSizeY: 64,
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{
		Width:        codedWidth,
		Height:       height,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	for i := range output.Y.Pix {
		output.Y.Pix[i] = byte(32 + i%96)
	}
	event := Event{
		SequenceHeader: seq,
		FrameSize:      size,
		Restoration:    restoration,
	}
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:    threading.FrameWorkSequenceContextFromHeader(seq),
			FrameSize:   size,
			Restoration: restoration,
		},
	}
	plan, err := batch.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	buffers, err := batch.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, plan.UnitRecordLen()),
		make([]uint16, plan.BoundaryBufferLen()),
		make([]uint16, plan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := buffers.ResetRecords(); err != nil {
		t.Fatal(err)
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output, RestorationFrameBuffers: &buffers}
	var outputView frame.Frame
	options := FrameWorkPostFilterBindOptions{
		SuperResOutputView:    &outputView,
		RestorationRecords:    buffers.Records,
		RestorationBoundaries: buffers.Boundaries,
	}
	first, err := ctx.CallerPostFilterScratchLen(FrameWorkPostFilterRequest{
		Restoration: FrameWorkRestorationPostFilterRequest{Records: buffers.Records},
	})
	if err != nil {
		t.Fatal(err)
	}
	if first.SuperRes.OutputFrame == 0 || first.Restoration.Samples.DataLen != 0 {
		t.Fatalf("bootstrap scratch=%+v", first)
	}
	firstScratch := testFrameWorkPostFilterScratchStorage(first)
	probe, err := first.BindRequest(options, firstScratch)
	if err != nil {
		t.Fatal(err)
	}
	full, err := ctx.CallerPostFilterScratchLen(probe)
	if err != nil {
		t.Fatal(err)
	}
	if full.Restoration.Samples.DataLen < output.Y.Width {
		t.Fatalf("restoration scratch too small for superres boundary row: %+v", full.Restoration)
	}
	scratch := testFrameWorkPostFilterScratchStorage(full)
	req, err := full.BindRequest(options, scratch)
	if err != nil {
		t.Fatal(err)
	}

	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterSuperRes | FrameWorkPostFilterLoopRestoration
	if result.Completed != wantCompleted ||
		result.SuperRes.Output.Format.Width != upscaledWidth ||
		result.Restoration.Records != 1 ||
		next.RemainingPostFilters() != 0 ||
		!next.DetachedPostFilterOutput() {
		t.Fatalf("next remaining=%b detached=%v result=%+v", next.RemainingPostFilters(), next.DetachedPostFilterOutput(), result)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := ctx.ApplyCallerPostFilters(req); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("superres restoration caller postfilter allocated: %f", allocs)
	}
}

func TestFrameWorkRestorationPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 16,
			DstLen:  12,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 8, SGRProj: 6},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 4, Below: 2},
		},
	}
	records := [3][]tile.RestorationUnitRecord{{{Index: 1}}}
	boundaries := [3]tile.RestorationStripeBoundaries{{Stride: 16}}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	req, err := size.BindRequest(records, boundaries, data, dst, wiener, sgr, above, below, nil, nil, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.DataScratch) != size.Samples.DataLen || len(req.DstScratch) != size.Samples.DstLen ||
		len(req.Scratch.Unit.Wiener) != size.Apply.Unit.Wiener ||
		len(req.Scratch.Unit.SGRProj) != size.Apply.Unit.SGRProj ||
		len(req.Scratch.Boundary.Above) != size.Apply.Boundary.Above ||
		len(req.Scratch.Boundary.Below) != size.Apply.Boundary.Below ||
		!req.Optimized || len(req.Records[0]) != 1 || req.Boundaries[0].Stride != boundaries[0].Stride {
		t.Fatalf("request data=%d dst=%d wiener=%d sgr=%d above=%d below=%d optimized=%v records=%d boundary=%d",
			len(req.DataScratch), len(req.DstScratch), len(req.Scratch.Unit.Wiener), len(req.Scratch.Unit.SGRProj),
			len(req.Scratch.Boundary.Above), len(req.Scratch.Boundary.Below), req.Optimized, len(req.Records[0]), req.Boundaries[0].Stride)
	}
}

func TestFrameWorkRestorationPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 16,
			DstLen:  12,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 8, SGRProj: 6},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 4, Below: 2},
		},
	}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	tests := []struct {
		name   string
		size   FrameWorkRestorationPostFilterScratchSize
		data   []uint16
		dst    []uint16
		wiener []uint16
		sgr    []int32
		above  []uint16
		below  []uint16
		want   error
	}{
		{name: "data", size: size, data: data[:15], dst: dst, wiener: wiener, sgr: sgr, above: above, below: below, want: tile.ErrJobBufferTooSmall},
		{name: "dst", size: size, data: data, dst: dst[:11], wiener: wiener, sgr: sgr, above: above, below: below, want: tile.ErrJobBufferTooSmall},
		{name: "wiener", size: size, data: data, dst: dst, wiener: wiener[:7], sgr: sgr, above: above, below: below, want: tile.ErrInvalidPlan},
		{name: "sgr", size: size, data: data, dst: dst, wiener: wiener, sgr: sgr[:5], above: above, below: below, want: tile.ErrInvalidPlan},
		{name: "above", size: size, data: data, dst: dst, wiener: wiener, sgr: sgr, above: above[:3], below: below, want: tile.ErrInvalidPlan},
		{name: "below", size: size, data: data, dst: dst, wiener: wiener, sgr: sgr, above: above, below: below[:1], want: tile.ErrInvalidPlan},
		{name: "negative", size: FrameWorkRestorationPostFilterScratchSize{Samples: tile.RestorationFrameSampleScratchSize{DataLen: -1}}, data: data, dst: dst, wiener: wiener, sgr: sgr, above: above, below: below, want: tile.ErrInvalidPlan},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.size.BindRequest([3][]tile.RestorationUnitRecord{}, [3]tile.RestorationStripeBoundaries{}, tt.data, tt.dst, tt.wiener, tt.sgr, tt.above, tt.below, nil, nil, nil, false)
			if !errors.Is(err, tt.want) {
				t.Fatalf("BindRequest err=%v want %v", err, tt.want)
			}
		})
	}
}

func TestFrameWorkRestorationPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 1920 * 1080,
			DstLen:  1920 * 1080,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 4096, SGRProj: 4096},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 1920, Below: 1920},
		},
	}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest([3][]tile.RestorationUnitRecord{}, [3]tile.RestorationStripeBoundaries{}, data, dst, wiener, sgr, above, below, nil, nil, nil, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkRestorationPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkRestorationPostFilterScratchSize{
		Samples: tile.RestorationFrameSampleScratchSize{
			DataLen: 1920 * 1080,
			DstLen:  1920 * 1080,
		},
		Apply: tile.RestorationUnitRecordBoundaryScratchSize{
			Unit:     tile.RestorationUnitScratchSize{Wiener: 4096, SGRProj: 4096},
			Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 1920, Below: 1920},
		},
	}
	data, dst, wiener, sgr, above, below := testFrameWorkRestorationPostFilterScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest([3][]tile.RestorationUnitRecord{}, [3]tile.RestorationStripeBoundaries{}, data, dst, wiener, sgr, above, below, nil, nil, nil, false); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkRestorationPostFilterScratch(size tile.RestorationUnitRecordBoundaryScratchSize) tile.RestorationUnitRecordBoundaryScratch {
	return tile.RestorationUnitRecordBoundaryScratch{
		Unit: tile.RestorationUnitScratch{
			Wiener:  make([]uint16, size.Unit.Wiener),
			SGRProj: make([]int32, size.Unit.SGRProj),
		},
		Boundary: tile.RestorationStripeBoundaryScratch{
			Above: make([]uint16, size.Boundary.Above),
			Below: make([]uint16, size.Boundary.Below),
		},
	}
}

func testFrameWorkRestorationPostFilterScratchStorage(size FrameWorkRestorationPostFilterScratchSize) ([]uint16, []uint16, []uint16, []int32, []uint16, []uint16) {
	return make([]uint16, maxInt(size.Samples.DataLen, 0)),
		make([]uint16, maxInt(size.Samples.DstLen, 0)),
		make([]uint16, maxInt(size.Apply.Unit.Wiener, 0)),
		make([]int32, maxInt(size.Apply.Unit.SGRProj, 0)),
		make([]uint16, maxInt(size.Apply.Boundary.Above, 0)),
		make([]uint16, maxInt(size.Apply.Boundary.Below, 0))
}

func TestFrameWorkPostFilterContextSuperResPostFilterPlanSkipsInactive(t *testing.T) {
	plan, err := (&FrameWorkPostFilterContext{}).SuperResPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if plan != (FrameWorkSuperResPostFilterPlan{}) {
		t.Fatalf("plan=%+v want zero", plan)
	}
	size, err := (&FrameWorkPostFilterContext{}).SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if size != (FrameWorkSuperResPostFilterScratchSize{}) {
		t.Fatalf("scratch=%+v want zero", size)
	}
}

func TestFrameWorkPostFilterContextSuperResPostFilterPlanReportsOutputGeometry(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          16,
			UpscaledWidth:       32,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 16,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	plan, err := ctx.SuperResPostFilterPlan()
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Denominator != 16 || plan.CodedFormat.Width != 16 || plan.OutputFormat.Width != 32 {
		t.Fatalf("plan=%+v", plan)
	}
	// The coded/output frames are allocated at the superblock-aligned extent
	// (16/32 -> 64), so OutputStride is align(64, Align=32) = 64.
	if plan.Planes[0] != (FrameWorkSuperResPostFilterPlanePlan{CodedWidth: 16, OutputWidth: 32, Height: 16, WidthDelta: 16, BytesPerRow: 32, OutputStride: 64}) {
		t.Fatalf("luma plane=%+v", plan.Planes[0])
	}
	if plan.Planes[1].CodedWidth != 8 || plan.Planes[1].OutputWidth != 16 || plan.Planes[1].Height != 8 || plan.Planes[1].WidthDelta != 8 {
		t.Fatalf("chroma plane=%+v", plan.Planes[1])
	}
	outputLayout, err := frame.RequiredSize(plan.OutputFormat)
	if err != nil {
		t.Fatal(err)
	}
	if plan.OutputSize != outputLayout.Size {
		t.Fatalf("output size=%d want %d", plan.OutputSize, outputLayout.Size)
	}
	scratch, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	if scratch.OutputFrame != outputLayout.Size {
		t.Fatalf("scratch=%+v want output frame %d", scratch, outputLayout.Size)
	}
	if scratch.CodedSamples[0] == 0 || scratch.OutputSamples[0] == 0 {
		t.Fatalf("scratch samples=%+v", scratch)
	}
}

func TestFrameWorkPostFilterContextSuperResPostFilterPlanRejectsInvalidActiveState(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          16,
			UpscaledWidth:       32,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 8,
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if _, err := ctx.SuperResPostFilterPlan(); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("SuperResPostFilterPlan err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextSuperResPostFilterPlanRejectsOutputMismatch(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          16,
			UpscaledWidth:       32,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 16,
		},
	}
	output := testFrameWorkCDEFFrame(t, frame.Format{Width: 32, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32})
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if _, err := ctx.SuperResPostFilterPlan(); !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("SuperResPostFilterPlan err=%v want %v", err, frame.ErrInvalidFormat)
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterWritesOutputScratch(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
		output.Y.Pix[x] = value
		output.Y.Pix[output.Y.Stride+x] = 100
	}
	for x, value := range []byte{0, 64, 128, 192} {
		output.U.Pix[x] = value
		output.V.Pix[x] = 55
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkSuperResPostFilterRequest(t, ctx)
	result, err := ctx.ApplySuperResPostFilter(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 3 || result.Output.Format.Width != 13 || result.OutputSize != len(req.OutputFrame) {
		t.Fatalf("result=%+v", result)
	}
	for x, want := range []byte{0, 11, 33, 54, 73, 92, 112, 132, 151, 171, 191, 214, 226} {
		if got := result.Output.Y.Pix[x]; got != want {
			t.Fatalf("Y[%d]=%d want %d", x, got, want)
		}
	}
	for x := range 13 {
		if got := result.Output.Y.Pix[result.Output.Y.Stride+x]; got != 100 {
			t.Fatalf("Y row1[%d]=%d want 100", x, got)
		}
	}
	for x, want := range []byte{0, 19, 59, 96, 134, 174, 197} {
		if got := result.Output.U.Pix[x]; got != want {
			t.Fatalf("U[%d]=%d want %d", x, got, want)
		}
		if got := result.Output.V.Pix[x]; got != 55 {
			t.Fatalf("V[%d]=%d want 55", x, got)
		}
	}
	if output.Y.Pix[1] != 32 {
		t.Fatalf("coded output mutated: %d", output.Y.Pix[1])
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterToContextCompletesStage(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
		output.Y.Pix[x] = value
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkSuperResPostFilterRequest(t, ctx)
	next, result, err := ctx.ApplySuperResPostFilterToContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Planes != 3 || next.RemainingPostFilters() != 0 {
		t.Fatalf("result=%+v remaining=%b", result, next.RemainingPostFilters())
	}
	if !next.DetachedPostFilterOutput() {
		t.Fatal("superres context did not report detached output")
	}
	if next.Output == nil {
		t.Fatal("next output is nil")
	}
	if next.Output == output || next.Output.Format.Width != 13 {
		t.Fatalf("next output=%p format=%+v coded=%p", next.Output, next.Output.Format, output)
	}
	if got := next.Output.Y.Pix[1]; got != 11 {
		t.Fatalf("upscaled Y[1]=%d want 11", got)
	}
	if output.Y.Pix[1] != 32 {
		t.Fatalf("coded output mutated: %d", output.Y.Pix[1])
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if err := next.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterToContextFeedsNoOpFilmGrain(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			BitDepth:      8,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
		output.Y.Pix[x] = value
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkSuperResPostFilterRequest(t, ctx)
	next, result, err := ctx.ApplySuperResPostFilterToContext(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Output.Format.Width != 13 || next.RemainingPostFilters() != FrameWorkPostFilterFilmGrain {
		t.Fatalf("superres result=%+v remaining=%b", result, next.RemainingPostFilters())
	}
	filmGrainResult, err := next.ApplyFilmGrainPostFilter(FrameWorkFilmGrainPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !filmGrainResult.NoOp || filmGrainResult.Plan.Format.Width != 13 {
		t.Fatalf("film grain result=%+v", filmGrainResult)
	}
	next = next.WithCompletedPostFilters(FrameWorkPostFilterFilmGrain)
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
}

func TestFrameWorkPostFilterContextApplyCallerPostFiltersRunsSuperResThenFilmGrain(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              32,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	size, err := ctx.CallerPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if size.SuperRes.OutputFrame == 0 || size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("first scratch=%+v", size)
	}
	req := FrameWorkPostFilterRequest{
		SuperRes: testFrameWorkSuperResPostFilterRequest(t, ctx),
	}
	size, err = ctx.CallerPostFilterScratchLen(req)
	if err != nil {
		t.Fatal(err)
	}
	if size.SuperRes.OutputFrame == 0 || size.FilmGrain.LumaGrain == 0 || size.FilmGrain.LumaSamples == 0 {
		t.Fatalf("second scratch=%+v", size)
	}
	req.FilmGrain = testFrameWorkFilmGrainPostFilterRequest(size.FilmGrain)

	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterSuperRes | FrameWorkPostFilterFilmGrain
	if result.Completed != wantCompleted ||
		result.SuperRes.Output.Format.Width != 13 ||
		result.FilmGrain.NoOp ||
		result.FilmGrain.LumaRows != 1 ||
		next.RemainingPostFilters() != 0 ||
		!next.DetachedPostFilterOutput() {
		t.Fatalf("next remaining=%b detached=%v result=%+v", next.RemainingPostFilters(), next.DetachedPostFilterOutput(), result)
	}
	if !testFrameWorkFilmGrainPlaneChangedFromValue(next.Output.Y, next.Output.Layout.BytesPerSample, 100) {
		t.Fatal("film grain did not change detached superres output")
	}
	if output.Y.Pix[0] != 100 {
		t.Fatalf("coded output mutated after detached stages: %d", output.Y.Pix[0])
	}
	if err := next.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextApplyCallerPostFiltersForPublicationKeepsFilmGrainDisplayOnly(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              32,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
		FilmGrain: parser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = 100
		}
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	var superResView frame.Frame
	var filmGrainView frame.Frame
	options := FrameWorkPostFilterBindOptions{
		SuperResOutputView:  &superResView,
		FilmGrainOutputView: &filmGrainView,
	}
	first, err := ctx.CallerPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		t.Fatal(err)
	}
	firstScratch := testFrameWorkPostFilterScratchStorage(first)
	probe, err := first.BindRequest(options, firstScratch)
	if err != nil {
		t.Fatal(err)
	}
	full, err := ctx.CallerPostFilterScratchLen(probe)
	if err != nil {
		t.Fatal(err)
	}
	scratch := testFrameWorkPostFilterScratchStorage(full)
	req, err := full.BindRequest(options, scratch)
	if err != nil {
		t.Fatal(err)
	}

	next, display, result, err := ctx.ApplyCallerPostFiltersForPublication(req)
	if err != nil {
		t.Fatal(err)
	}
	wantCompleted := FrameWorkPostFilterSuperRes | FrameWorkPostFilterFilmGrain
	if result.Completed != wantCompleted ||
		result.SuperRes.Output.Format.Width != 13 ||
		result.FilmGrain.NoOp ||
		result.FilmGrain.LumaRows != 1 ||
		next.RemainingPostFilters() != 0 {
		t.Fatalf("next remaining=%b result=%+v", next.RemainingPostFilters(), result)
	}
	if next.Output != &superResView || display != &filmGrainView || display == next.Output {
		t.Fatalf("publish=%p display=%p superresView=%p filmGrainView=%p", next.Output, display, &superResView, &filmGrainView)
	}
	if testFrameWorkFilmGrainPlaneChangedFromValue(next.Output.Y, next.Output.Layout.BytesPerSample, 100) {
		t.Fatal("publication output includes display film grain")
	}
	if !testFrameWorkFilmGrainPlaneChangedFromValue(display.Y, display.Layout.BytesPerSample, 100) {
		t.Fatal("display output did not receive film grain")
	}
	if output.Y.Pix[0] != 100 {
		t.Fatalf("coded output mutated: %d", output.Y.Pix[0])
	}
}

func TestFrameWorkCallerPostFilterRunnerApplyRunsSuperRes(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              16,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	for y := 0; y < output.Y.Height; y++ {
		for x := 0; x < output.Y.Width; x++ {
			output.Y.Pix[y*output.Y.Stride+x] = byte(64 + x)
		}
	}

	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := FrameWorkPostFilterRequest{
		SuperRes: testFrameWorkSuperResPostFilterRequest(t, ctx),
	}
	size, err := ctx.CallerPostFilterScratchLen(req)
	if err != nil {
		t.Fatal(err)
	}
	if size.SuperRes.OutputFrame == 0 || size.CDEF != (FrameWorkCDEFPostFilterScratchSize{}) ||
		size.Restoration != (FrameWorkRestorationPostFilterScratchSize{}) ||
		size.FilmGrain != (FrameWorkFilmGrainPostFilterScratchSize{}) {
		t.Fatalf("caller scratch=%+v", size)
	}
	next, result, err := ctx.ApplyCallerPostFilters(req)
	if err != nil {
		t.Fatal(err)
	}
	if result.Completed != FrameWorkPostFilterSuperRes ||
		result.SuperRes.Output.Format.Width != 13 ||
		next.RemainingPostFilters() != 0 ||
		!next.DetachedPostFilterOutput() ||
		next.Output.Format.Width != 13 {
		t.Fatalf("caller context detached=%v remaining=%b result=%+v", next.DetachedPostFilterOutput(), next.RemainingPostFilters(), result)
	}
	if err := next.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if err := next.RequirePublishablePostFilterOutput(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequirePublishablePostFilterOutput err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if output.Format.Width != 8 || output.Y.Pix[0] != 64 {
		t.Fatalf("coded output mutated: width=%d first=%d", output.Format.Width, output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterRejectsShortOutputScratch(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	size, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	req := FrameWorkSuperResPostFilterRequest{OutputFrame: make([]byte, size.OutputFrame-1)}
	if _, err := ctx.ApplySuperResPostFilter(req); !errors.Is(err, frame.ErrShortBuffer) {
		t.Fatalf("ApplySuperResPostFilter err=%v want %v", err, frame.ErrShortBuffer)
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterRejectsShortPlaneScratchBeforeOutputMutation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*FrameWorkSuperResPostFilterRequest)
	}{
		{
			name: "coded-chroma",
			mutate: func(req *FrameWorkSuperResPostFilterRequest) {
				req.CodedScratch[1] = req.CodedScratch[1][:len(req.CodedScratch[1])-1]
			},
		},
		{
			name: "output-chroma",
			mutate: func(req *FrameWorkSuperResPostFilterRequest) {
				req.OutputScratch[1] = req.OutputScratch[1][:len(req.OutputScratch[1])-1]
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			seq := testSequence()
			event := Event{
				SequenceHeader: seq,
				FrameSize: parser.FrameSize{
					CodedWidth:          8,
					UpscaledWidth:       13,
					Height:              2,
					SuperResEnabled:     true,
					SuperResDenominator: 13,
				},
			}
			codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
			if err != nil {
				t.Fatal(err)
			}
			output := testFrameWorkCDEFFrame(t, codedFormat)
			for x, value := range []byte{0, 32, 64, 96, 128, 160, 192, 224} {
				output.Y.Pix[x] = value
			}
			for x, value := range []byte{0, 64, 128, 192} {
				output.U.Pix[x] = value
				output.V.Pix[x] = value
			}

			ctx := FrameWorkPostFilterContext{Event: event, Output: output}
			req := testFrameWorkSuperResPostFilterRequest(t, ctx)
			for i := range req.OutputFrame {
				req.OutputFrame[i] = 0xaa
			}
			tt.mutate(&req)

			if _, err := ctx.ApplySuperResPostFilter(req); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("ApplySuperResPostFilter err=%v want %v", err, frame.ErrShortBuffer)
			}
			for i, got := range req.OutputFrame {
				if got != 0xaa {
					t.Fatalf("output frame byte %d mutated to %#x", i, got)
				}
			}
		})
	}
}

func TestFrameWorkPostFilterContextApplySuperResPostFilterRejectsEarlierStages(t *testing.T) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          8,
			UpscaledWidth:       13,
			Height:              2,
			SuperResEnabled:     true,
			SuperResDenominator: 13,
		},
		CDEF: parser.CDEFParams{StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{4}},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 32)
	if err != nil {
		t.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(t, codedFormat)
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	if _, err := ctx.ApplySuperResPostFilter(FrameWorkSuperResPostFilterRequest{}); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("ApplySuperResPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkSuperResPostFilterScratchSizeBindRequest(t *testing.T) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   16,
		CodedSamples:  [3]int{8, 4, 2},
		OutputSamples: [3]int{9, 5, 3},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	var outputView frame.Frame
	req, err := size.BindRequest(outputFrame, coded, output, &outputView)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.OutputFrame) != size.OutputFrame || req.OutputView != &outputView ||
		len(req.CodedScratch[0]) != size.CodedSamples[0] || len(req.CodedScratch[1]) != size.CodedSamples[1] ||
		len(req.OutputScratch[1]) != size.OutputSamples[1] || len(req.OutputScratch[2]) != size.OutputSamples[2] {
		t.Fatalf("request outputFrame=%d outputView=%p coded=%v output=%v",
			len(req.OutputFrame), req.OutputView,
			[3]int{len(req.CodedScratch[0]), len(req.CodedScratch[1]), len(req.CodedScratch[2])},
			[3]int{len(req.OutputScratch[0]), len(req.OutputScratch[1]), len(req.OutputScratch[2])})
	}
}

func TestFrameWorkSuperResPostFilterScratchSizeBindRequestRejectsShortBuffers(t *testing.T) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   16,
		CodedSamples:  [3]int{8, 4, 2},
		OutputSamples: [3]int{9, 5, 3},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	tests := []struct {
		name        string
		size        FrameWorkSuperResPostFilterScratchSize
		outputFrame []byte
		coded       [3][]uint16
		output      [3][]uint16
	}{
		{name: "output-frame", size: size, outputFrame: outputFrame[:15], coded: coded, output: output},
		{name: "coded", size: size, outputFrame: outputFrame, coded: [3][]uint16{coded[0][:7], coded[1], coded[2]}, output: output},
		{name: "output", size: size, outputFrame: outputFrame, coded: coded, output: [3][]uint16{output[0], output[1][:4], output[2]}},
		{name: "negative", size: FrameWorkSuperResPostFilterScratchSize{OutputFrame: -1}, outputFrame: outputFrame, coded: coded, output: output},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.size.BindRequest(tt.outputFrame, tt.coded, tt.output, nil); !errors.Is(err, frame.ErrShortBuffer) {
				t.Fatalf("BindRequest err=%v want %v", err, frame.ErrShortBuffer)
			}
		})
	}
}

func TestFrameWorkSuperResPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   1920 * 1080 * 3 / 2,
		CodedSamples:  [3]int{1280 * 720, 640 * 360, 640 * 360},
		OutputSamples: [3]int{1920 * 1080, 960 * 540, 960 * 540},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(outputFrame, coded, output, nil); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkSuperResPostFilterScratchSizeBindRequest(b *testing.B) {
	size := FrameWorkSuperResPostFilterScratchSize{
		OutputFrame:   1920 * 1080 * 3 / 2,
		CodedSamples:  [3]int{1280 * 720, 640 * 360, 640 * 360},
		OutputSamples: [3]int{1920 * 1080, 960 * 540, 960 * 540},
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(outputFrame, coded, output, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplySuperResPostFilter(b *testing.B) {
	seq := testSequence()
	event := Event{
		SequenceHeader: seq,
		FrameSize: parser.FrameSize{
			CodedWidth:          960,
			UpscaledWidth:       1280,
			Height:              720,
			SuperResEnabled:     true,
			SuperResDenominator: 12,
		},
	}
	codedFormat, err := codedFrameFormatFromHeaders(seq, event.FrameSize, 64)
	if err != nil {
		b.Fatal(err)
	}
	output := testFrameWorkCDEFFrame(b, codedFormat)
	for y := 0; y < output.Y.Height; y++ {
		row := output.Y.Pix[y*output.Y.Stride:]
		for x := 0; x < output.Y.Width; x++ {
			row[x] = byte((x + y) & 0xff)
		}
	}
	ctx := FrameWorkPostFilterContext{Event: event, Output: output}
	req := testFrameWorkSuperResPostFilterRequest(b, ctx)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := ctx.ApplySuperResPostFilter(req)
		if err != nil {
			b.Fatal(err)
		}
		if result.Planes != 3 {
			b.Fatalf("planes=%d want 3", result.Planes)
		}
	}
}

func testFrameWorkSuperResPostFilterRequest(t testing.TB, ctx FrameWorkPostFilterContext) FrameWorkSuperResPostFilterRequest {
	t.Helper()
	size, err := ctx.SuperResPostFilterScratchLen()
	if err != nil {
		t.Fatal(err)
	}
	outputFrame, coded, output := testFrameWorkSuperResScratchStorage(size)
	req, err := size.BindRequest(outputFrame, coded, output, nil)
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func testFrameWorkSuperResScratchStorage(size FrameWorkSuperResPostFilterScratchSize) ([]byte, [3][]uint16, [3][]uint16) {
	var coded [3][]uint16
	var output [3][]uint16
	for plane := range 3 {
		if size.CodedSamples[plane] > 0 {
			coded[plane] = make([]uint16, size.CodedSamples[plane])
		}
		if size.OutputSamples[plane] > 0 {
			output[plane] = make([]uint16, size.OutputSamples[plane])
		}
	}
	return make([]byte, maxInt(size.OutputFrame, 0)), coded, output
}

func TestFrameWorkPostFilterContextRequireNoActivePostFiltersAllowsInactiveSyntax(t *testing.T) {
	tests := []struct {
		name  string
		event Event
	}{
		{
			name: "loopfilter-tuning-without-levels",
			event: Event{
				LoopFilter: parser.LoopFilterParams{
					Sharpness:           7,
					ModeRefDeltaEnabled: true,
					ModeRefDeltaUpdate:  true,
					Deltas: parser.LoopFilterDeltas{
						Ref:  [parser.RefFrames]int8{1},
						Mode: [parser.LoopFilterModeDeltas]int8{1},
					},
				},
			},
		},
		{
			name: "cdef-selection-with-zero-strengths",
			event: Event{
				CDEF: parser.CDEFParams{
					Damping:       3,
					StrengthCount: 1,
				},
			},
		},
		{
			name: "superres-denominator-without-superres",
			event: Event{
				FrameSize: parser.FrameSize{SuperResDenominator: 8},
			},
		},
		{
			name: "restoration-units-without-restoration-type",
			event: Event{
				Restoration: parser.RestorationParams{
					UnitSizeYLog2:  6,
					UnitSizeUVLog2: 6,
					UnitSizeY:      64,
					UnitSizeUV:     64,
				},
			},
		},
		{
			name: "film-grain-params-not-applied",
			event: Event{
				FilmGrain: parser.FilmGrainParams{
					ParamsPresent: true,
					Update:        true,
					Seed:          1,
					NumYPoints:    1,
					YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{0, 16}},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := (&FrameWorkPostFilterContext{Event: tt.event}).RequireNoActivePostFilters(); err != nil {
				t.Fatalf("RequireNoActivePostFilters err=%v", err)
			}
		})
	}
}

func TestFrameWorkPostFilterContextRequireNoActivePostFiltersRejectsActiveComponents(t *testing.T) {
	tests := []struct {
		name  string
		event Event
	}{
		{
			name: "loopfilter-y0",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1, 0}},
			},
		},
		{
			name: "loopfilter-y1",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{0, 1}},
			},
		},
		{
			name: "loopfilter-u",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelU: 1},
			},
		},
		{
			name: "loopfilter-v",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelV: 1},
			},
		},
		{
			name: "cdef-index-bits",
			event: Event{
				CDEF: parser.CDEFParams{Bits: 1},
			},
		},
		{
			name: "cdef-y-strength",
			event: Event{
				CDEF: parser.CDEFParams{StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{1}},
			},
		},
		{
			name: "cdef-uv-strength",
			event: Event{
				CDEF: parser.CDEFParams{StrengthCount: 1, UVStrength: [parser.MaxCDEFStrengths]uint8{1}},
			},
		},
		{
			name: "superres",
			event: Event{
				FrameSize: parser.FrameSize{SuperResEnabled: true},
			},
		},
		{
			name: "restoration-y",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}},
			},
		},
		{
			name: "restoration-u",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{1: parser.RestorationWiener}},
			},
		},
		{
			name: "restoration-v",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{2: parser.RestorationWiener}},
			},
		},
		{
			name: "film-grain-apply",
			event: Event{
				FilmGrain: parser.FilmGrainParams{Apply: true},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := (&FrameWorkPostFilterContext{Event: tt.event}).RequireNoActivePostFilters()
			if !errors.Is(err, ErrUnsupportedPostFilter) {
				t.Fatalf("RequireNoActivePostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
			}
		})
	}
}

func TestFrameWorkPostFilterContextRemainingPostFilters(t *testing.T) {
	active := FrameWorkPostFilterLoopFilter |
		FrameWorkPostFilterCDEF |
		FrameWorkPostFilterSuperRes |
		FrameWorkPostFilterLoopRestoration |
		FrameWorkPostFilterFilmGrain
	ctx := FrameWorkPostFilterContext{
		Event: Event{
			LoopFilter:  parser.LoopFilterParams{LevelY: [2]uint8{1}},
			CDEF:        parser.CDEFParams{Bits: 1},
			FrameSize:   parser.FrameSize{SuperResEnabled: true},
			Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}},
			FilmGrain:   parser.FilmGrainParams{Apply: true},
		},
	}
	if got := ctx.ActivePostFilters(); got != active {
		t.Fatalf("active=%b want %b", got, active)
	}
	if got := ctx.RemainingPostFilters(); got != active {
		t.Fatalf("remaining=%b want %b", got, active)
	}

	ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopFilter | FrameWorkPostFilterCDEF | FrameWorkPostFilterSuperRes | FrameWorkPostFilterStage(0x80))
	want := FrameWorkPostFilterLoopRestoration | FrameWorkPostFilterFilmGrain
	if got := ctx.RemainingPostFilters(); got != want {
		t.Fatalf("remaining after pre-restoration stages=%b want %b", got, want)
	}
	if err := ctx.RequireNoActivePostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireNoActivePostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if err := ctx.RequireNoRemainingPostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireNoRemainingPostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}

	ctx = ctx.WithCompletedPostFilters(FrameWorkPostFilterLoopRestoration | FrameWorkPostFilterFilmGrain)
	if got := ctx.RemainingPostFilters(); got != 0 {
		t.Fatalf("remaining after all stages=%b want 0", got)
	}
	if err := ctx.RequireNoRemainingPostFilters(); err != nil {
		t.Fatalf("RequireNoRemainingPostFilters err=%v", err)
	}
	if err := ctx.RequireNoActivePostFilters(); !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RequireNoActivePostFilters err=%v want %v", err, ErrUnsupportedPostFilter)
	}
}

func TestFrameWorkPostFilterContextShouldSaveRestorationDeblockBoundaries(t *testing.T) {
	restoration := parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}}
	tests := []struct {
		name      string
		ctx       FrameWorkPostFilterContext
		wantSave  bool
		remaining FrameWorkPostFilterStage
	}{
		{
			name: "restoration-without-cdef",
			ctx: FrameWorkPostFilterContext{
				Event: Event{Restoration: restoration},
			},
			wantSave:  true,
			remaining: FrameWorkPostFilterLoopRestoration,
		},
		{
			name: "restoration-before-cdef",
			ctx: FrameWorkPostFilterContext{
				Event: Event{CDEF: parser.CDEFParams{Bits: 1}, Restoration: restoration},
			},
			wantSave:  true,
			remaining: FrameWorkPostFilterCDEF | FrameWorkPostFilterLoopRestoration,
		},
		{
			name: "restoration-after-cdef",
			ctx: FrameWorkPostFilterContext{
				Event: Event{CDEF: parser.CDEFParams{Bits: 1}, Restoration: restoration},
			}.WithCompletedPostFilters(FrameWorkPostFilterCDEF),
			wantSave:  false,
			remaining: FrameWorkPostFilterLoopRestoration,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.ctx.RemainingPostFilters(); got != tt.remaining {
				t.Fatalf("remaining=%b want %b", got, tt.remaining)
			}
			if got := tt.ctx.shouldSaveRestorationDeblockBoundaries(); got != tt.wantSave {
				t.Fatalf("shouldSaveRestorationDeblockBoundaries=%v want %v", got, tt.wantSave)
			}
		})
	}
}

func TestFrameWorkPostFilterScratchSizeMax(t *testing.T) {
	a := FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 1, Schedule: 2},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{1, 8, 3},
			Dst:           [3]int{4, 2, 6},
			DirectionGrid: 5,
			VarianceGrid:  9,
			Input:         7,
			UnitDst:       11,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   32,
			CodedSamples:  [3]int{6, 2, 8},
			OutputSamples: [3]int{3, 10, 4},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				Data:    [3]frame.BorderedSamplePlaneLayout{{Stride: 4, Origin: 2, Rows: 3, Len: 12}},
				Dst:     [3]frame.BorderedSamplePlaneLayout{{Len: 1}, {Stride: 2, Origin: 3, Rows: 4, Len: 5}},
				DataLen: 20,
				DstLen:  7,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 14, SGRProj: 2},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 6, Below: 1},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			ScalingPoints: [3]int{1, 7, 3},
			ARCoeffs:      [3]int{5, 2, 9},
			LumaGrain:     12,
			ChromaGrain:   [2]int{6, 4},
			LumaSamples:   10,
			ChromaSamples: [2]int{8, 2},
			LumaLine:      3,
			LumaColumn:    11,
		},
	}
	b := FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 3, Schedule: 1},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{2, 4, 9},
			Dst:           [3]int{5, 7, 1},
			DirectionGrid: 4,
			VarianceGrid:  10,
			Input:         12,
			UnitDst:       8,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   64,
			CodedSamples:  [3]int{7, 1, 9},
			OutputSamples: [3]int{8, 6, 5},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				Data:    [3]frame.BorderedSamplePlaneLayout{{Stride: 5, Origin: 1, Rows: 6, Len: 9}, {Len: 3}},
				Dst:     [3]frame.BorderedSamplePlaneLayout{{Len: 2}, {Stride: 1, Origin: 4, Rows: 3, Len: 8}},
				DataLen: 18,
				DstLen:  9,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 9, SGRProj: 5},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 3, Below: 8},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			ScalingPoints: [3]int{2, 5, 4},
			ARCoeffs:      [3]int{3, 8, 7},
			LumaGrain:     9,
			ChromaGrain:   [2]int{7, 3},
			LumaSamples:   11,
			ChromaSamples: [2]int{5, 6},
			LumaLine:      13,
			LumaColumn:    10,
		},
	}
	want := FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 3, Schedule: 2},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{2, 8, 9},
			Dst:           [3]int{5, 7, 6},
			DirectionGrid: 5,
			VarianceGrid:  10,
			Input:         12,
			UnitDst:       11,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   64,
			CodedSamples:  [3]int{7, 2, 9},
			OutputSamples: [3]int{8, 10, 5},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				Data:    [3]frame.BorderedSamplePlaneLayout{{Stride: 5, Origin: 2, Rows: 6, Len: 12}, {Len: 3}},
				Dst:     [3]frame.BorderedSamplePlaneLayout{{Len: 2}, {Stride: 2, Origin: 4, Rows: 4, Len: 8}},
				DataLen: 20,
				DstLen:  9,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 14, SGRProj: 5},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 6, Below: 8},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			ScalingPoints: [3]int{2, 7, 4},
			ARCoeffs:      [3]int{5, 8, 9},
			LumaGrain:     12,
			ChromaGrain:   [2]int{7, 4},
			LumaSamples:   11,
			ChromaSamples: [2]int{8, 6},
			LumaLine:      13,
			LumaColumn:    11,
		},
	}
	if got := a.Max(b); got != want {
		t.Fatalf("a.Max(b)=%+v want %+v", got, want)
	}
	if got := b.Max(a); got != want {
		t.Fatalf("b.Max(a)=%+v want %+v", got, want)
	}
}

func TestFrameWorkPostFilterScratchSizeBindRequest(t *testing.T) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	var outputView frame.Frame
	options := FrameWorkPostFilterBindOptions{
		CDEFIndexMap: FrameWorkCDEFIndexMap{
			Index:  make([]uint8, 4),
			Read:   make([]bool, 4),
			Stride: 2,
			Rows:   2,
		},
		SuperResOutputView: &outputView,
		RestorationRecords: [3][]tile.RestorationUnitRecord{
			{{Index: 3}},
		},
		RestorationBoundaries: [3]tile.RestorationStripeBoundaries{
			{Stride: 16},
		},
		RestorationOptimized: true,
	}
	req, err := size.BindRequest(options, scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(req.LoopFilter.Edges) != size.LoopFilter.Edges ||
		len(req.LoopFilter.Schedule) != size.LoopFilter.Schedule ||
		len(req.CDEF.SampleScratch[0]) != size.CDEF.Samples[0] ||
		len(req.SuperRes.OutputFrame) != size.SuperRes.OutputFrame ||
		req.SuperRes.OutputView != &outputView ||
		len(req.Restoration.DataScratch) != size.Restoration.Samples.DataLen ||
		len(req.Restoration.Scratch.Unit.Wiener) != size.Restoration.Apply.Unit.Wiener ||
		!req.Restoration.Optimized ||
		len(req.Restoration.Records[0]) != 1 ||
		req.Restoration.Boundaries[0].Stride != 16 ||
		len(req.FilmGrain.LumaGrain) != size.FilmGrain.LumaGrain ||
		len(req.FilmGrain.ChromaSamples[1]) != size.FilmGrain.ChromaSamples[1] {
		t.Fatalf("request=%+v", req)
	}
}

func TestFrameWorkPostFilterScratchSizeBindRequestRejectsShortStageScratch(t *testing.T) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	tests := []struct {
		name   string
		mutate func(*FrameWorkPostFilterScratch)
		want   error
	}{
		{name: "loop-filter-edges", mutate: func(s *FrameWorkPostFilterScratch) { s.LoopFilterEdges = s.LoopFilterEdges[:len(s.LoopFilterEdges)-1] }, want: frame.ErrShortBuffer},
		{name: "loop-filter-schedule", mutate: func(s *FrameWorkPostFilterScratch) {
			s.LoopFilterSchedule = s.LoopFilterSchedule[:len(s.LoopFilterSchedule)-1]
		}, want: frame.ErrShortBuffer},
		{name: "cdef", mutate: func(s *FrameWorkPostFilterScratch) { s.CDEFInput = s.CDEFInput[:len(s.CDEFInput)-1] }, want: frame.ErrShortBuffer},
		{name: "superres", mutate: func(s *FrameWorkPostFilterScratch) {
			s.SuperResOutputFrame = s.SuperResOutputFrame[:len(s.SuperResOutputFrame)-1]
		}, want: frame.ErrShortBuffer},
		{name: "restoration", mutate: func(s *FrameWorkPostFilterScratch) { s.RestorationData = s.RestorationData[:len(s.RestorationData)-1] }, want: tile.ErrJobBufferTooSmall},
		{name: "film-grain", mutate: func(s *FrameWorkPostFilterScratch) {
			s.FilmGrainLumaSamples = s.FilmGrainLumaSamples[:len(s.FilmGrainLumaSamples)-1]
		}, want: frame.ErrShortBuffer},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			short := scratch
			tt.mutate(&short)
			if _, err := size.BindRequest(FrameWorkPostFilterBindOptions{}, short); !errors.Is(err, tt.want) {
				t.Fatalf("BindRequest err=%v want %v", err, tt.want)
			}
		})
	}
}

func TestFrameWorkPostFilterScratchSizeBindRequestAllocs(t *testing.T) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := size.BindRequest(FrameWorkPostFilterBindOptions{}, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BindRequest allocated: %f", allocs)
	}
}

func BenchmarkFrameWorkPostFilterScratchSizeBindRequest(b *testing.B) {
	size := testFrameWorkPostFilterScratchSize()
	scratch := testFrameWorkPostFilterScratchStorage(size)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := size.BindRequest(FrameWorkPostFilterBindOptions{}, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func testFrameWorkPostFilterScratchSize() FrameWorkPostFilterScratchSize {
	return FrameWorkPostFilterScratchSize{
		LoopFilter: FrameWorkLoopFilterPostFilterScratchSize{Edges: 8, Schedule: 8},
		CDEF: FrameWorkCDEFPostFilterScratchSize{
			Samples:       [3]int{64, 32, 32},
			Dst:           [3]int{64, 32, 32},
			DirectionGrid: 4,
			VarianceGrid:  4,
			Input:         16,
			UnitDst:       16,
		},
		SuperRes: FrameWorkSuperResPostFilterScratchSize{
			OutputFrame:   128,
			CodedSamples:  [3]int{64, 32, 32},
			OutputSamples: [3]int{96, 48, 48},
		},
		Restoration: FrameWorkRestorationPostFilterScratchSize{
			Samples: tile.RestorationFrameSampleScratchSize{
				DataLen: 256,
				DstLen:  256,
			},
			Apply: tile.RestorationUnitRecordBoundaryScratchSize{
				Unit:     tile.RestorationUnitScratchSize{Wiener: 32, SGRProj: 24},
				Boundary: tile.RestorationStripeBoundaryScratchSize{Above: 16, Below: 16},
			},
		},
		FilmGrain: FrameWorkFilmGrainPostFilterScratchSize{
			LumaGrain:     64,
			ChromaGrain:   [2]int{32, 32},
			LumaSamples:   128,
			ChromaSamples: [2]int{64, 64},
		},
	}
}

func testFrameWorkPostFilterScratchStorage(size FrameWorkPostFilterScratchSize) FrameWorkPostFilterScratch {
	cdefSamples, cdefDst, directionGrid, varianceGrid, input, unitDst := testFrameWorkCDEFScratchStorage(size.CDEF)
	superResOutputFrame, superResCoded, superResOutput := testFrameWorkSuperResScratchStorage(size.SuperRes)
	restorationData, restorationDst, restorationWiener, restorationSGR, restorationAbove, restorationBelow := testFrameWorkRestorationPostFilterScratchStorage(size.Restoration)
	filmGrainOutput, lumaGrain, chromaGrain, lumaSamples, chromaSamples := testFrameWorkFilmGrainScratchStorage(size.FilmGrain)
	return FrameWorkPostFilterScratch{
		LoopFilterEdges:    make([]FrameWorkLoopFilterPostFilterEdge, maxInt(size.LoopFilter.Edges, 0)),
		LoopFilterSchedule: make([]uint32, maxInt(size.LoopFilter.Schedule, 0)),

		CDEFSamples:       cdefSamples,
		CDEFDst:           cdefDst,
		CDEFDirectionGrid: directionGrid,
		CDEFVarianceGrid:  varianceGrid,
		CDEFInput:         input,
		CDEFUnitDst:       unitDst,

		SuperResOutputFrame: superResOutputFrame,
		SuperResCoded:       superResCoded,
		SuperResOutput:      superResOutput,

		RestorationData:   restorationData,
		RestorationDst:    restorationDst,
		RestorationWiener: restorationWiener,
		RestorationSGR:    restorationSGR,
		RestorationAbove:  restorationAbove,
		RestorationBelow:  restorationBelow,

		FilmGrainOutputFrame:   filmGrainOutput,
		FilmGrainLumaGrain:     lumaGrain,
		FilmGrainChromaGrain:   chromaGrain,
		FilmGrainLumaSamples:   lumaSamples,
		FilmGrainChromaSamples: chromaSamples,
	}
}

func TestFrameWorkStateRunStepWithPostFilterGateRejectsBeforePublish(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	output.Y.Pix[0] = 0x55

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	state := FrameWorkState{Surface: surface, active: true}
	event := finalFrameEvent(0xff)
	event.LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{1, 0}}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}

	result, err := state.RunStepWithPostFilter(&refs, &pool, &event, step, nil, nil, nil, releases[:], nil, func(ctx FrameWorkPostFilterContext) error {
		return ctx.RequireNoActivePostFilters()
	})
	if !errors.Is(err, ErrUnsupportedPostFilter) {
		t.Fatalf("RunStepWithPostFilter err=%v want %v", err, ErrUnsupportedPostFilter)
	}
	if result != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	for i := range parser.RefFrames {
		if slot, ok := refs.ReferenceSlot(i); ok || slot != -1 {
			t.Fatalf("reference[%d] slot=%d ok=%v want unpublished", i, slot, ok)
		}
	}
	got, err := pool.Frame(surface)
	if err != nil {
		t.Fatal(err)
	}
	if got != output || got.Y.Pix[0] != 0x55 {
		t.Fatalf("active output=%p sample=%d want %p sample=85", got, got.Y.Pix[0], output)
	}
}

func TestFrameWorkStateRunStepWithPostFilterSkipsHookUntilFinalTileGroup(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	state := FrameWorkState{Surface: surface, active: true}
	event := Event{
		Kind:      EventTileGroup,
		FrameSize: parser.FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: parser.TileGroup{Final: false},
	}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}

	called := false
	result, err := state.RunStepWithPostFilter(&refs, &pool, &event, step, nil, nil, nil, releases[:], nil, func(FrameWorkPostFilterContext) error {
		called = true
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("postfilter hook ran before the final tile group")
	}
	if result != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	for i := range parser.RefFrames {
		if slot, ok := refs.ReferenceSlot(i); ok || slot != -1 {
			t.Fatalf("reference[%d] slot=%d ok=%v want unpublished", i, slot, ok)
		}
	}
}

func segMapTestEvent(w, h uint32, primary uint8, refSlot uint8, refresh uint8, seg parser.SegmentationParams) *Event {
	var ev Event
	ev.FrameSize.CodedWidth = w
	ev.FrameSize.Height = h
	ev.FrameSize.RefreshFrameFlags = refresh
	ev.FrameHeader.PrimaryRefFrame = primary
	for i := range ev.FrameSize.RefFrameIdx {
		ev.FrameSize.RefFrameIdx[i] = refSlot
	}
	ev.Segmentation = seg
	return &ev
}

// TestSegmentMapLifecycle checks the dav1d segmap rules: a fresh map under
// update_map, the primary reference's map shared without it, the previous map
// dropped on a dimension change, show-existing key frame propagation, and the
// sequence reset.
func TestSegmentMapLifecycle(t *testing.T) {
	var store SegmentMapStore
	var f segMapFrame

	// Key frame (primary none): update_map, writes a fresh map to every slot.
	key := segMapTestEvent(64, 64, parser.PrimaryRefNone, 0, 0xff, parser.SegmentationParams{Enabled: true, UpdateMap: true})
	if err := f.prepare(&store, key); err != nil {
		t.Fatal(err)
	}
	cur, prev, stride := f.fields(&store)
	if len(cur) != 16*16 || prev != nil || stride != 16 {
		t.Fatalf("key frame maps cur=%d prev=%v stride=%d", len(cur), prev != nil, stride)
	}
	for i := range cur {
		cur[i] = 3
	}
	keyMap := f.cur
	store.publish(key.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)
	if store.refs[keyMap-1] != parser.RefFrames {
		t.Fatalf("key map refs=%d want %d", store.refs[keyMap-1], parser.RefFrames)
	}

	// update_map=0 inherits the primary reference's map: read-only, shared.
	inherit := segMapTestEvent(64, 64, 0, 2, 0x01, parser.SegmentationParams{Enabled: true})
	if err := f.prepare(&store, inherit); err != nil {
		t.Fatal(err)
	}
	cur, prev, _ = f.fields(&store)
	if cur != nil || len(prev) != 16*16 || prev[0] != 3 || f.cur != keyMap {
		t.Fatalf("inherit cur=%v prev=%d cur handle=%d", cur != nil, len(prev), f.cur)
	}
	store.publish(inherit.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)

	// temporal_update reads the previous map and writes a fresh one.
	temporal := segMapTestEvent(64, 64, 0, 0, 0x02, parser.SegmentationParams{Enabled: true, UpdateMap: true, TemporalUpdate: true})
	if err := f.prepare(&store, temporal); err != nil {
		t.Fatal(err)
	}
	cur, prev, _ = f.fields(&store)
	if len(cur) != 16*16 || len(prev) != 16*16 || f.cur == keyMap || f.prev != keyMap {
		t.Fatalf("temporal cur=%d prev=%d", len(cur), len(prev))
	}
	f.release(&store)

	// A frame-size change drops the previous map (dav1d ref_w/ref_h check).
	resized := segMapTestEvent(32, 64, 0, 0, 0x04, parser.SegmentationParams{Enabled: true, UpdateMap: true, TemporalUpdate: true})
	if err := f.prepare(&store, resized); err != nil {
		t.Fatal(err)
	}
	if _, prev, stride = f.fields(&store); prev != nil || stride != 8 {
		t.Fatalf("resized prev=%v stride=%d", prev != nil, stride)
	}
	store.publish(resized.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)
	resizedMap := store.slot[2]

	// Segmentation disabled: the refreshed slots lose their map.
	off := segMapTestEvent(64, 64, 0, 0, 0x08, parser.SegmentationParams{})
	if err := f.prepare(&store, off); err != nil {
		t.Fatal(err)
	}
	store.publish(off.FrameSize.RefreshFrameFlags, f.cur)
	f.release(&store)
	if store.slot[3] != 0 {
		t.Fatalf("disabled segmentation left slot map %d", store.slot[3])
	}

	// Show-existing key frame: every slot takes the shown slot's map.
	store.showExistingKey(2)
	for i, h := range store.slot {
		if h != resizedMap {
			t.Fatalf("slot %d map %d want %d", i, h, resizedMap)
		}
	}
	if store.refs[resizedMap-1] != parser.RefFrames {
		t.Fatalf("shown map refs=%d", store.refs[resizedMap-1])
	}
	if store.refs[keyMap-1] != 0 {
		t.Fatalf("key map still held: refs=%d", store.refs[keyMap-1])
	}

	store.resetSlots()
	for i := range store.refs {
		if store.refs[i] != 0 {
			t.Fatalf("map %d still held after reset", i+1)
		}
	}
}

// TestSegmentMapSteadyStateNoAllocs checks that reusing the store across
// frames does not allocate once its maps are backed.
func TestSegmentMapSteadyStateNoAllocs(t *testing.T) {
	var store SegmentMapStore
	var f segMapFrame
	ev := segMapTestEvent(64, 64, 0, 0, 0x01, parser.SegmentationParams{Enabled: true, UpdateMap: true, TemporalUpdate: true})
	step := func() {
		if err := f.prepare(&store, ev); err != nil {
			t.Fatal(err)
		}
		store.publish(ev.FrameSize.RefreshFrameFlags, f.cur)
		f.release(&store)
	}
	for range 4 {
		step()
	}
	if n := testing.AllocsPerRun(50, step); n != 0 {
		t.Fatalf("allocs per frame = %v", n)
	}
}

// showExistingKeyTestState binds slot r to a key frame (CDF entry 0, MV grid
// 1, intra metadata) and every other slot to an older inter frame (CDF entry
// 1, MV grid 2).
func showExistingKeyTestState(r int) *FrameWorkState {
	s := &FrameWorkState{ownedTemporalMotion: true}
	for i := range parser.RefFrames {
		cdf, grid := int8(1), uint8(2)
		meta := mvFrameMeta{valid: true, orderHint: 3}
		if i == r {
			cdf, grid = 0, 1
			meta = mvFrameMeta{valid: true, orderHint: 9, intraOnly: true}
		}
		s.tileResidualSlotCDF[i] = cdf
		s.tileResidualFrameContextValid[i] = true
		s.tileResidualCDFPoolRefs[cdf]++
		s.mvSlotGrid[i] = grid
		s.mvGridRefs[grid-1]++
		s.mvFrameStore[i] = tile.ReferenceMVFrame{Rows: 1, Cols: 1, Stride: 1}
		s.mvFrameStoreMeta[i] = meta
	}
	return s
}

func checkShowExistingKeySlots(t *testing.T, s *FrameWorkState, r int) {
	t.Helper()
	for i := range parser.RefFrames {
		if s.tileResidualSlotCDF[i] != 0 || !s.tileResidualFrameContextValid[i] {
			t.Fatalf("slot %d CDF entry %d valid=%v, want the shown key frame's entry 0", i, s.tileResidualSlotCDF[i], s.tileResidualFrameContextValid[i])
		}
		if s.mvFrameStoreMeta[i] != s.mvFrameStoreMeta[r] {
			t.Fatalf("slot %d MV metadata %+v, want %+v", i, s.mvFrameStoreMeta[i], s.mvFrameStoreMeta[r])
		}
		if i != r && (s.mvSlotGrid[i] != 0 || s.mvFrameStore[i].Rows != 0) {
			t.Fatalf("slot %d kept MV grid %d", i, s.mvSlotGrid[i])
		}
	}
	if got := s.tileResidualCDFPoolRefs[0]; got != parser.RefFrames {
		t.Fatalf("key CDF entry refs=%d want %d", got, parser.RefFrames)
	}
	if got := s.tileResidualCDFPoolRefs[1]; got != 0 {
		t.Fatalf("stale CDF entry still held: refs=%d", got)
	}
	if s.mvGridRefs[0] != 1 || s.mvGridRefs[1] != 0 {
		t.Fatalf("MV grid refs=%v want [1 0 ...]", s.mvGridRefs[:2])
	}
}

// TestShowExistingKeyResetsSlotState: a key frame shown again saves its
// state into every reference slot (AV1 7.21 with refresh_frame_flags =
// allFrames; dav1d obu.c copies cdf[r] and segmap[r] to every slot and drops
// the other slots' refmvs). A later frame whose primary_ref_frame names
// another slot must load the key frame's CDFs, not that slot's old ones.
func TestShowExistingKeyResetsSlotState(t *testing.T) {
	const r = 2
	pool := testFramePool(t, parser.RefFrames)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	for i := range parser.RefFrames {
		surface, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := refs.Refresh(1<<uint(i), surface, releases[:]); err != nil {
			t.Fatal(err)
		}
	}
	s := showExistingKeyTestState(r)
	plan, err := s.ShowExisting(&refs, &pool, showExistingWorkEventPtr(r, parser.FrameTypeKey), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.ReleaseCount != parser.RefFrames-1 || pool.Available() != parser.RefFrames-1 {
		t.Fatalf("plan=%+v available=%d", plan, pool.Available())
	}
	checkShowExistingKeySlots(t, s, r)
}

// TestShowExistingFrameThreadedDefersReleases: queued in the frame-thread
// ring, a key frame shown again applies the same slot reset at once but hands
// the surfaces it drops to the ring slot instead of the pool, since frames in
// flight may still read them.
func TestShowExistingFrameThreadedDefersReleases(t *testing.T) {
	const r = 5
	pool := testFramePool(t, parser.RefFrames)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	var surfaces [parser.RefFrames]int
	for i := range parser.RefFrames {
		surface, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		surfaces[i] = surface
		if _, err := refs.Refresh(1<<uint(i), surface, releases[:]); err != nil {
			t.Fatal(err)
		}
	}
	s := showExistingKeyTestState(r)
	var ft FrameThreadFrame
	plan, err := s.ShowExistingFrameThreaded(&refs, showExistingWorkEventPtr(r, parser.FrameTypeKey), &ft)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Surface != surfaces[r] || ft.ReleaseCount != parser.RefFrames-1 || pool.Available() != 0 {
		t.Fatalf("plan=%+v releases=%d available=%d", plan, ft.ReleaseCount, pool.Available())
	}
	for i := range parser.RefFrames {
		if got, _ := refs.ReferenceSlot(i); got != surfaces[r] {
			t.Fatalf("slot %d surface %d want %d", i, got, surfaces[r])
		}
	}
	checkShowExistingKeySlots(t, s, r)

	// A non-key frame shown again changes nothing.
	ft = FrameThreadFrame{}
	if _, err := s.ShowExistingFrameThreaded(&refs, showExistingWorkEventPtr(0, parser.FrameTypeInter), &ft); err != nil || ft.ReleaseCount != 0 {
		t.Fatalf("inter show-existing: releases=%d err=%v", ft.ReleaseCount, err)
	}
}

// fuzzMaxUnits bounds the number of OBU/IVF units pulled from a single fuzz
// input so a hostile stream can never make the harness loop forever. Every
// well-formed stream the conformance suite decodes stays far below this.
const fuzzMaxUnits = 1 << 16

// fuzzStreamSeeds are small, self-contained byte streams that exercise the
// untrusted-input parsing surface. They mirror the synthetic vectors used by
// the conformance core suite so the corpus is meaningful even in checkouts
// where the downloaded libaom .ivf vectors are absent.
var fuzzStreamSeeds = [][]byte{
	// IVF container wrapping a single low-overhead temporal-delimiter OBU.
	{
		'D', 'K', 'I', 'F',
		0x00, 0x00,
		0x20, 0x00,
		'A', 'V', '0', '1',
		0x10, 0x00,
		0x10, 0x00,
		0x1e, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,
		0x01, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x02, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x12, 0x00,
	},
	// Bare low-overhead temporal-delimiter OBU.
	{0x12, 0x00},
	// Full sequence header (low-overhead).
	{0x0a, 0x0b, 0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20},
	// Reduced still-picture sequence header (low-overhead).
	{0x0a, 0x07, 0x18, 0x22, 0x2b, 0xf1, 0xfe, 0xc0, 0x20},
	// Annex B temporal unit with two frame units.
	{0x0a, 0x05, 0x01, 0x10, 0x02, 0x08, 0xaa, 0x03, 0x02, 0x18, 0xbb},
	// Sequence header followed by a temporal delimiter (low-overhead).
	{0x0a, 0x0b, 0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20, 0x12, 0x00},
}

// addIVFSeedFiles best-effort seeds the corpus from any .ivf vectors present in
// the repository testdata tree. The libaom vectors are gitignored and only
// exist in checkouts that have downloaded them, so their absence is not fatal:
// the synthetic seeds above keep the corpus non-empty everywhere.
func addIVFSeedFiles(f *testing.F) {
	for _, root := range []string{
		filepath.Join("..", "testdata"),
		filepath.Join("..", "testdata", "libaom"),
	} {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() || filepath.Ext(e.Name()) != ".ivf" {
				continue
			}
			data, err := os.ReadFile(filepath.Join(root, e.Name()))
			if err != nil {
				continue
			}
			f.Add(data)
		}
	}
}

// FuzzStreamPush feeds arbitrary bytes through every untrusted-input entry
// point the decoder exposes for a byte stream: the IVF demuxer, the
// low-overhead / Annex B / temporal-unit OBU iterators, the single-OBU parser,
// and decoder.Stream's OBU dispatch (which transitively drives every
// parser.ParseXxx frame-header routine). It asserts the decoder never panics
// and always terminates on corrupt, truncated, or hostile input.
func FuzzStreamPush(f *testing.F) {
	for _, seed := range fuzzStreamSeeds {
		f.Add(seed)
	}
	addIVFSeedFiles(f)

	f.Fuzz(func(t *testing.T, data []byte) {
		// A panic anywhere in the parse path is a hardening defect: malformed
		// input must surface as an error, never a crash.
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic on malformed input (len=%d): %v", len(data), r)
			}
		}()

		// Path 1: treat the input as an IVF container and push each demuxed
		// temporal-unit payload through the stream as low-overhead OBUs.
		fuzzDriveIVF(data)

		// Path 2..n: treat the whole input as a raw OBU byte stream in each of
		// the supported framings.
		fuzzDriveLowOverhead(data)
		fuzzDriveSingleElement(data)
	})
}

func fuzzDriveIVF(data []byte) {
	it, err := ivf.NewIterator(data)
	if err != nil {
		return
	}
	var stream Stream
	var events [256]Event
	frames := 0
	for {
		frame, ok, err := it.Next()
		if err != nil || !ok {
			return
		}
		frames++
		if frames > fuzzMaxUnits {
			return
		}
		if _, err := stream.PushLowOverhead(frame.Payload, events[:]); err != nil {
			// A bad payload aborts the stream; this mirrors how a real caller
			// would stop decoding on the first error.
			return
		}
	}
}

func fuzzDriveLowOverhead(data []byte) {
	it := obu.NewLowOverheadIterator(data)
	var stream Stream
	var event Event
	units := 0
	for {
		unit, ok, err := it.Next()
		if err != nil || !ok {
			return
		}
		units++
		if units > fuzzMaxUnits {
			return
		}
		if err := stream.PushUnitInto(&event, unit, units == 1); err != nil {
			return
		}
	}
}

func fuzzDriveSingleElement(data []byte) {
	unit, err := obu.ParseElement(data)
	if err != nil {
		return
	}
	var stream Stream
	var event Event
	_ = stream.PushUnitInto(&event, unit, true)
}

// appendLowOverheadOBUExt is the extension-header counterpart of
// appendLowOverheadOBU: it sets obu_extension_flag and writes the
// (temporal_id, spatial_id) byte. Used by the SVC propagation tests below.
func appendLowOverheadOBUExt(dst []byte, typ obu.Type, temporalID uint8, spatialID uint8, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{
		Type:         typ,
		HasSizeField: true,
		Extension:    true,
		TemporalID:   temporalID,
		SpatialID:    spatialID,
	})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	var size [bitstream.MaxLEB128Bytes]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		panic(err)
	}
	dst = append(dst, size[:n]...)
	dst = append(dst, payload...)
	return dst
}

// TestStreamPropagatesExtensionLayerIDs verifies that an extension-tagged
// OBU group (the AV1 SVC routing path) lands its (temporal_id, spatial_id)
// onto every Event produced from the group: sequence header, frame OBU,
// and tile-group continuations.
func TestStreamPropagatesExtensionLayerIDs(t *testing.T) {
	cases := []struct {
		name           string
		temporal, spat uint8
	}{
		{"base", 0, 0},
		{"T1S0", 1, 0},
		{"T0S1", 0, 1},
		{"T2S3", 2, 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			frame := append([]byte{}, reducedStillFrameHeaderPayload()...)
			frame = append(frame, 0xaa)

			var stream []byte
			stream = appendLowOverheadOBUExt(stream, obu.TypeSequenceHeader, tc.temporal, tc.spat, testSequenceHeaderPayload(16))
			stream = appendLowOverheadOBUExt(stream, obu.TypeFrame, tc.temporal, tc.spat, frame)

			var dec Stream
			var events [2]Event
			count, err := dec.PushLowOverhead(stream, events[:])
			if err != nil {
				t.Fatal(err)
			}
			if count != 2 {
				t.Fatalf("count=%d", count)
			}
			for i, ev := range events[:count] {
				if ev.TemporalID != tc.temporal || ev.SpatialID != tc.spat {
					t.Fatalf("event[%d] T%d S%d want T%d S%d", i, ev.TemporalID, ev.SpatialID, tc.temporal, tc.spat)
				}
			}
		})
	}
}

// TestStreamRejectsExtensionReservedBits verifies that a malformed extension
// byte (with one of the reserved bits set) is rejected at the Stream entry
// point rather than propagated as a "successful" layer-tagged event.
func TestStreamRejectsExtensionReservedBits(t *testing.T) {
	// Manually build a malformed low-overhead unit: extension byte sets the
	// lowest reserved bit. The header parser must reject this before any
	// payload decode.
	raw := []byte{
		byte(obu.TypeSequenceHeader)<<3 | 0x04 /*extension*/ | 0x02, /*has_size_field*/
		0x01, // extension byte: reserved bit set
		0x00, // obu_size = 0
	}
	var dec Stream
	if _, err := pushOBU(&dec, raw, false); err == nil {
		t.Fatal("expected reserved-bit error")
	}
}

func pushOBU(s *Stream, raw []byte, newCodedVideoSequence bool) (Event, error) {
	var event Event
	if err := s.PushOBUInto(&event, raw, newCodedVideoSequence); err != nil {
		return Event{}, err
	}
	return event, nil
}

type testBitWriter struct {
	buf [128]byte
	bit int
}

func (w *testBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		if (value>>uint(i))&1 != 0 {
			w.buf[w.bit>>3] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *testBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *testBitWriter) bytes() []byte {
	return w.buf[:(w.bit+7)>>3]
}

func (w *testBitWriter) trailingBits() []byte {
	w.writeBits(1, 1)
	for w.bit&7 != 0 {
		w.writeBits(0, 1)
	}
	return w.buf[:w.bit>>3]
}

func testSequenceHeaderPayload(width uint64) []byte {
	var w testBitWriter
	w.writeBits(0, 3)       // seq_profile
	w.writeBool(true)       // still_picture
	w.writeBool(true)       // reduced_still_picture_header
	w.writeBits(5, 5)       // seq_level_idx[0]
	w.writeBits(7, 4)       // frame_width_bits_minus_1
	w.writeBits(3, 4)       // frame_height_bits_minus_1
	w.writeBits(width-1, 8) // max_frame_width_minus_1
	w.writeBits(8, 4)       // max_frame_height_minus_1
	w.writeBool(false)      // use_128x128_superblock
	w.writeBool(true)       // enable_filter_intra
	w.writeBool(true)       // enable_intra_edge_filter
	w.writeBool(false)      // enable_superres
	w.writeBool(true)       // enable_cdef
	w.writeBool(false)      // enable_restoration
	w.writeBool(false)      // high_bitdepth
	w.writeBool(false)      // mono_chrome
	w.writeBool(false)      // color_description_present_flag
	w.writeBool(false)      // color_range
	w.writeBits(0, 2)       // chroma_sample_position
	w.writeBool(true)       // separate_uv_delta_q
	w.writeBool(false)      // film_grain_params_present
	return w.trailingBits()
}

func testRealtimeNoOrderSequenceHeaderPayload() []byte {
	var w testBitWriter
	w.writeBits(0, 3)  // seq_profile
	w.writeBool(false) // still_picture
	w.writeBool(false) // reduced_still_picture_header
	w.writeBool(false) // timing_info_present_flag
	w.writeBool(false) // initial_display_delay_present_flag
	w.writeBits(0, 5)  // operating_points_cnt_minus_1
	w.writeBits(0, 12) // operating_point_idc[0]
	w.writeBits(5, 5)  // seq_level_idx[0]
	w.writeBits(3, 4)  // frame_width_bits_minus_1
	w.writeBits(3, 4)  // frame_height_bits_minus_1
	w.writeBits(15, 4) // max_frame_width_minus_1
	w.writeBits(8, 4)  // max_frame_height_minus_1
	w.writeBool(false) // frame_id_numbers_present_flag
	w.writeBool(false) // use_128x128_superblock
	w.writeBool(true)  // enable_filter_intra
	w.writeBool(true)  // enable_intra_edge_filter
	w.writeBool(true)  // enable_interintra_compound
	w.writeBool(true)  // enable_masked_compound
	w.writeBool(false) // enable_warped_motion
	w.writeBool(true)  // enable_dual_filter
	w.writeBool(false) // enable_order_hint
	w.writeBool(false) // seq_choose_screen_content_tools
	w.writeBits(0, 1)  // seq_force_screen_content_tools
	w.writeBool(false) // enable_superres
	w.writeBool(true)  // enable_cdef
	w.writeBool(false) // enable_restoration
	w.writeBool(false) // high_bitdepth
	w.writeBool(false) // mono_chrome
	w.writeBool(false) // color_description_present_flag
	w.writeBool(false) // color_range
	w.writeBits(0, 2)  // chroma_sample_position
	w.writeBool(true)  // separate_uv_delta_q
	w.writeBool(false) // film_grain_params_present
	return w.trailingBits()
}

func shownKeyFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(false)                          // show_existing_frame
	w.writeBits(uint64(parser.FrameTypeKey), 2) // frame_type
	w.writeBool(true)                           // show_frame
	w.writeBool(false)                          // disable_cdf_update
	w.writeBool(false)                          // frame_size_override_flag
	w.writeBool(false)                          // render_and_frame_size_different
	w.writeBool(false)                          // disable_frame_end_update_cdf
	w.writeBool(false)                          // uniform_tile_spacing_flag
	writeZeroQuantParams(&w)
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

func reducedStillFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(true)  // disable_cdf_update
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // uniform_tile_spacing_flag
	writeZeroQuantParams(&w)
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

func interFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(false)                            // show_existing_frame
	w.writeBits(uint64(parser.FrameTypeInter), 2) // frame_type
	w.writeBool(true)                             // show_frame
	w.writeBool(false)                            // error_resilient_mode
	w.writeBool(false)                            // disable_cdf_update
	w.writeBool(false)                            // frame_size_override_flag
	w.writeBits(0, 3)                             // primary_ref_frame
	w.writeBits(0x01, 8)                          // refresh_frame_flags
	for range parser.InterRefsPerFrame {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(0, 2)  // interpolation_filter = EIGHTTAP
	w.writeBool(false) // is_motion_mode_switchable
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	writeZeroQuantParams(&w)
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reference_select
	w.writeBool(false) // reduced_tx_set
	for range parser.InterRefsPerFrame {
		w.writeBool(false) // global_motion_is_global
	}
	return w.bytes()
}

func interFrameHeaderPayloadWithReferenceState(update bool) []byte {
	var w testBitWriter
	w.writeBool(false)                            // show_existing_frame
	w.writeBits(uint64(parser.FrameTypeInter), 2) // frame_type
	w.writeBool(true)                             // show_frame
	w.writeBool(false)                            // error_resilient_mode
	w.writeBool(false)                            // disable_cdf_update
	w.writeBool(false)                            // frame_size_override_flag
	w.writeBits(0, 3)                             // primary_ref_frame
	w.writeBits(0x01, 8)                          // refresh_frame_flags
	for range parser.InterRefsPerFrame {
		w.writeBits(0, 3) // ref_frame_idx[i]
	}
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // allow_high_precision_mv
	w.writeBool(false) // interpolation_filter is fixed
	w.writeBits(0, 2)  // interpolation_filter = EIGHTTAP
	w.writeBool(false) // is_motion_mode_switchable
	w.writeBool(false) // disable_frame_end_update_cdf
	w.writeBool(false) // uniform_tile_spacing_flag
	writeQuantParams(&w, 40)
	if update {
		writeSegmentationUpdateData(&w)
	} else {
		writeSegmentationCopyData(&w)
	}
	w.writeBool(false) // delta_q_present
	if update {
		writeLoopFilterDeltaUpdate(&w)
	} else {
		writeLoopFilterDeltaCopy(&w)
	}
	writeMinimalCDEF(&w)
	w.writeBool(false) // transform_mode_select
	w.writeBool(false) // reference_select
	w.writeBool(false) // reduced_tx_set
	for range parser.InterRefsPerFrame {
		w.writeBool(false) // global_motion_is_global
	}
	return w.bytes()
}

func showExistingFrameHeaderPayload(index uint8) []byte {
	var w testBitWriter
	w.writeBool(true) // show_existing_frame
	w.writeBits(uint64(index&7), 3)
	return w.bytes()
}

// switchFrameHeaderPayload encodes a complete shown S_FRAME frame header that
// the Stream parser pipeline can fully consume. The encoded bitstream matches
// AV1 §5.9.1 / §5.9.5 / §5.9.7 for a switch frame in a sequence with
// enable_order_hint = 0 and frame_id_numbers_present_flag = 0: the syntax
// skips error_resilient_mode (implied 1), frame_size_override_flag (implied
// 1), primary_ref_frame (error_resilient skips), refresh_frame_flags (S_FRAME
// implies 0xff), and frame_refs_short_signaling (no order hint).
func switchFrameHeaderPayload() []byte {
	var w testBitWriter
	w.writeBool(false)
	w.writeBits(uint64(parser.FrameTypeSwitch), 2)
	w.writeBool(true)
	w.writeBool(false)
	for range parser.InterRefsPerFrame {
		w.writeBits(0, 3)
	}
	w.writeBits(15, 4)
	w.writeBits(8, 4)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBits(0, 2)
	w.writeBool(false)
	w.writeBool(false)
	w.writeBool(false)
	writeZeroQuantParams(&w)
	writeZeroSegmentationParams(&w)
	w.writeBool(false)
	w.writeBool(false)
	for range parser.InterRefsPerFrame {
		w.writeBool(false)
	}
	return w.bytes()
}

func writeZeroQuantParams(w *testBitWriter) {
	writeQuantParams(w, 0)
}

func writeQuantParams(w *testBitWriter, baseQIdx uint8) {
	w.writeBits(uint64(baseQIdx), 8) // base_q_idx
	w.writeBool(false)               // y_dc_delta_q
	w.writeBool(false)               // diff_uv_delta
	w.writeBool(false)               // u_dc_delta_q
	w.writeBool(false)               // u_ac_delta_q
	w.writeBool(false)               // using_qmatrix
}

func writeSegmentationUpdateData(w *testBitWriter) {
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(true)  // update_data
	w.writeBool(true)  // segment 0 delta_q enabled
	writeSignedBits(w, 5, 9)
	w.writeBool(false) // delta_lf_y_v
	w.writeBool(false) // delta_lf_y_h
	w.writeBool(false) // delta_lf_u
	w.writeBool(false) // delta_lf_v
	w.writeBool(false) // ref_frame
	w.writeBool(false) // skip
	w.writeBool(false) // globalmv
	for i := 1; i < parser.MaxSegments; i++ {
		writeEmptySegmentData(w)
	}
}

func writeSegmentationCopyData(w *testBitWriter) {
	w.writeBool(true)  // segmentation_enabled
	w.writeBool(false) // update_map
	w.writeBool(false) // update_data
}

func writeEmptySegmentData(w *testBitWriter) {
	w.writeBool(false) // delta_q
	w.writeBool(false) // delta_lf_y_v
	w.writeBool(false) // delta_lf_y_h
	w.writeBool(false) // delta_lf_u
	w.writeBool(false) // delta_lf_v
	w.writeBool(false) // ref_frame
	w.writeBool(false) // skip
	w.writeBool(false) // globalmv
}

func writeLoopFilterDeltaUpdate(w *testBitWriter) {
	w.writeBits(10, 6) // loop_filter_level[0]
	w.writeBits(0, 6)  // loop_filter_level[1]
	w.writeBits(5, 6)  // loop_filter_level_u
	w.writeBits(6, 6)  // loop_filter_level_v
	w.writeBits(0, 3)  // loop_filter_sharpness
	w.writeBool(true)  // mode_ref_delta_enabled
	w.writeBool(true)  // mode_ref_delta_update
	for i := range parser.RefFrames {
		update := i == 2
		w.writeBool(update)
		if update {
			writeSignedBits(w, 4, 7)
		}
	}
	for range parser.LoopFilterModeDeltas {
		w.writeBool(false)
	}
}

func writeLoopFilterDeltaCopy(w *testBitWriter) {
	w.writeBits(11, 6) // loop_filter_level[0]
	w.writeBits(0, 6)  // loop_filter_level[1]
	w.writeBits(5, 6)  // loop_filter_level_u
	w.writeBits(6, 6)  // loop_filter_level_v
	w.writeBits(0, 3)  // loop_filter_sharpness
	w.writeBool(true)  // mode_ref_delta_enabled
	w.writeBool(false) // mode_ref_delta_update
}

func writeMinimalCDEF(w *testBitWriter) {
	w.writeBits(0, 2) // cdef_damping_minus_3
	w.writeBits(0, 2) // cdef_bits
	w.writeBits(0, 6) // cdef_y_pri_strength + cdef_y_sec_strength
	w.writeBits(0, 6) // cdef_uv_pri_strength + cdef_uv_sec_strength
}

func writeSignedBits(w *testBitWriter, value int16, n uint8) {
	mask := (uint64(1) << n) - 1
	w.writeBits(uint64(uint16(value))&mask, n)
}

func writeZeroSegmentationParams(w *testBitWriter) {
	w.writeBool(false) // segmentation_enabled
}

func appendLowOverheadOBU(dst []byte, typ obu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{Type: typ, HasSizeField: true})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	var size [bitstream.MaxLEB128Bytes]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		panic(err)
	}
	dst = append(dst, size[:n]...)
	dst = append(dst, payload...)
	return dst
}

func appendRTPElement(dst []byte, typ obu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{Type: typ})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	dst = append(dst, payload...)
	return dst
}

func TestStreamLowOverheadState(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [4]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}
	if events[0].Kind != EventSequenceHeader || !events[0].NewCodedVideoSequence {
		t.Fatalf("sequence event=%+v", events[0])
	}
	if events[1].Kind != EventFrameHeader {
		t.Fatalf("frame header event=%+v", events[1])
	}
	if events[1].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("frame sequence=%+v want %+v", events[1].SequenceHeader, events[0].SequenceHeader)
	}
	if events[1].FrameHeader.FrameType != parser.FrameTypeKey || !events[1].FrameHeader.ShowFrame {
		t.Fatalf("frame header parse=%+v", events[1].FrameHeader)
	}
	if events[1].FrameSize.CodedWidth != 16 || events[1].FrameSize.Height != 9 {
		t.Fatalf("frame size=%+v", events[1].FrameSize)
	}
	if events[1].TileInfo.Cols != 1 || events[1].TileInfo.Rows != 1 {
		t.Fatalf("tile info=%+v", events[1].TileInfo)
	}
	if events[1].Quantization.BaseQIdx != 0 {
		t.Fatalf("quantization=%+v", events[1].Quantization)
	}
	if events[1].Segmentation.Enabled || !events[1].Segmentation.AllLossless {
		t.Fatalf("segmentation=%+v", events[1].Segmentation)
	}
	if events[1].Delta.DeltaQPresent || events[1].Delta.BitsRead != events[1].Segmentation.BitsRead {
		t.Fatalf("delta=%+v", events[1].Delta)
	}
	if !events[1].LoopFilter.ModeRefDeltaEnabled || !events[1].LoopFilter.ModeRefDeltaUpdate {
		t.Fatalf("loopfilter=%+v", events[1].LoopFilter)
	}
	if events[1].CDEF.BitsRead != events[1].LoopFilter.BitsRead {
		t.Fatalf("cdef=%+v", events[1].CDEF)
	}
	if events[1].Restoration.BitsRead != events[1].CDEF.BitsRead {
		t.Fatalf("restoration=%+v", events[1].Restoration)
	}
	if events[1].TransformRef.TransformMode != parser.TransformMode4x4Only ||
		events[1].TransformRef.ReferenceMode != parser.ReferenceModeSingle ||
		events[1].TransformRef.BitsRead != events[1].Restoration.BitsRead {
		t.Fatalf("transform/reference=%+v", events[1].TransformRef)
	}
	if events[1].SkipMode.Allowed || events[1].SkipMode.Enabled ||
		events[1].SkipMode.BitsRead != events[1].TransformRef.BitsRead {
		t.Fatalf("skip mode=%+v", events[1].SkipMode)
	}
	if events[1].FrameMode.AllowWarpedMotion || events[1].FrameMode.ReducedTxSet ||
		events[1].FrameMode.BitsRead != events[1].SkipMode.BitsRead+1 {
		t.Fatalf("frame mode=%+v", events[1].FrameMode)
	}
	if events[1].GlobalMotion.BitsRead != events[1].FrameMode.BitsRead {
		t.Fatalf("global motion=%+v", events[1].GlobalMotion)
	}
	if events[1].FilmGrain.Apply || events[1].FilmGrain.BitsRead != events[1].GlobalMotion.BitsRead {
		t.Fatalf("film grain=%+v", events[1].FilmGrain)
	}
	if events[2].Kind != EventTileGroup {
		t.Fatalf("tile event=%+v", events[2])
	}
	if events[2].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("tile sequence=%+v want %+v", events[2].SequenceHeader, events[0].SequenceHeader)
	}
	if events[2].FrameHeader != events[1].FrameHeader || events[2].FrameSize != events[1].FrameSize ||
		events[2].Quantization != events[1].Quantization || events[2].Segmentation != events[1].Segmentation ||
		events[2].LoopFilter != events[1].LoopFilter || events[2].FilmGrain != events[1].FilmGrain {
		t.Fatalf("tile frame state=%+v header event=%+v", events[2], events[1])
	}
	if events[2].TileGroup.StartTile != 0 || events[2].TileGroup.EndTile != 0 ||
		events[2].TileGroup.DataOffset != 0 || events[2].TileGroup.DataSize != 1 || !events[2].TileGroup.Final {
		t.Fatalf("tile group=%+v", events[2].TileGroup)
	}
}

func TestStreamFrameOBUParsesImplicitTileGroup(t *testing.T) {
	frame := append([]byte{}, reducedStillFrameHeaderPayload()...)
	frame = append(frame, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, frame)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	if events[1].Kind != EventFrame {
		t.Fatalf("frame event=%+v", events[1])
	}
	if events[1].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("frame sequence=%+v want %+v", events[1].SequenceHeader, events[0].SequenceHeader)
	}
	if events[1].TileGroup.StartTile != 0 || events[1].TileGroup.EndTile != 0 ||
		int(events[1].TileGroup.DataOffset) != len(reducedStillFrameHeaderPayload()) ||
		events[1].TileGroup.DataSize != 1 || !events[1].TileGroup.Final {
		t.Fatalf("frame tile group=%+v", events[1].TileGroup)
	}
}

func TestStreamMetadataOBUParsesHDRCLL(t *testing.T) {
	// metadata_type=1 (HDR-CLL), max_cll=1000, max_fall=400, trailing 0x80.
	payload := []byte{0x01, 0x03, 0xE8, 0x01, 0x90, 0x80}

	var dec Stream
	event, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeMetadata, payload), false)
	if err != nil {
		t.Fatal(err)
	}
	if event.Kind != EventMetadata {
		t.Fatalf("kind=%v want EventMetadata", event.Kind)
	}
	if event.MetadataErr != nil {
		t.Fatalf("MetadataErr=%v", event.MetadataErr)
	}
	if event.Metadata.Type != obu.MetadataTypeHDRCLL {
		t.Fatalf("metadata type=%v", event.Metadata.Type)
	}
	if event.Metadata.HDRCLL.MaxCLL != 1000 || event.Metadata.HDRCLL.MaxFALL != 400 {
		t.Fatalf("cll=%#v", event.Metadata.HDRCLL)
	}
}

func TestStreamTileListOBUParsesAndRequiresFrameContext(t *testing.T) {
	// One output tile referencing anchor row/col 1 from a larger external
	// reference tile grid. The output mosaic size is not the anchor bound.
	payload := []byte{
		0x00,       // output_frame_width_in_tiles_minus_1 = 0
		0x00,       // output_frame_height_in_tiles_minus_1 = 0
		0x00, 0x00, // tile_count_minus_1 = 0 (1 tile)
		0x00, 0x01, 0x01, 0x00, 0x02, // entry header
		0xaa, 0xbb, 0xcc, // tile data
	}

	var dec Stream
	event, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeTileList, payload), false)
	if err != nil {
		t.Fatalf("PushOBUInto err=%v", err)
	}
	if event.Kind != EventTileList {
		t.Fatalf("kind=%v want EventTileList", event.Kind)
	}
	if !errors.Is(event.TileListErr, ErrTileListMissingFrameContext) {
		t.Fatalf("TileListErr=%v want %v", event.TileListErr, ErrTileListMissingFrameContext)
	}
	if event.TileList.TileCount() != 1 || event.TileList.OutputFrameWidthInTiles() != 1 ||
		event.TileList.OutputFrameHeightInTiles() != 1 {
		t.Fatalf("tile list header=%+v", event.TileList)
	}
	if len(event.TileList.Entries) != 1 {
		t.Fatalf("entries=%d want 1", len(event.TileList.Entries))
	}
	e := event.TileList.Entries[0]
	if e.AnchorFrameIdx != 0 || e.AnchorTileRow != 1 || e.AnchorTileCol != 1 ||
		e.TileDataSize() != 3 || string(e.TileData) != "\xaa\xbb\xcc" {
		t.Fatalf("entry=%+v", e)
	}
}

func TestStreamTileListOBUCapturesParseError(t *testing.T) {
	// Truncated tile list payload (header only, claims 1 tile but no entry).
	payload := []byte{0x00, 0x00, 0x00, 0x00}

	var dec Stream
	event, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeTileList, payload), false)
	if err != nil {
		t.Fatalf("PushOBUInto should not propagate tile list parse error: %v", err)
	}
	if event.Kind != EventTileList {
		t.Fatalf("kind=%v want EventTileList", event.Kind)
	}
	if !errors.Is(event.TileListErr, parser.ErrTileListShortEntry) {
		t.Fatalf("TileListErr=%v want ErrTileListShortEntry", event.TileListErr)
	}
}

func TestStreamMetadataOBUCapturesParseError(t *testing.T) {
	// metadata_type=4 (ITUT_T35) with no country code byte.
	payload := []byte{0x04}

	var dec Stream
	event, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeMetadata, payload), false)
	if err != nil {
		t.Fatalf("PushOBUInto should not propagate metadata parse error: %v", err)
	}
	if event.Kind != EventMetadata {
		t.Fatalf("kind=%v", event.Kind)
	}
	if !errors.Is(event.MetadataErr, obu.ErrMetadataMissingCountry) {
		t.Fatalf("MetadataErr=%v want ErrMetadataMissingCountry", event.MetadataErr)
	}
}

// TestStreamRejectsMetadataDav1dRejects: an HDR CLL metadata OBU without the
// bit after its two fields fails the OBU, as in dav1d (src/obu.c).
func TestStreamRejectsMetadataDav1dRejects(t *testing.T) {
	var dec Stream
	_, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeMetadata, []byte{0x01, 0, 0, 0, 0}), false)
	if !errors.Is(err, obu.ErrMetadataShortPayload) {
		t.Fatalf("PushOBUInto err=%v want %v", err, obu.ErrMetadataShortPayload)
	}
}

func TestStreamRejectsFrameBeforeSequenceHeader(t *testing.T) {
	var dec Stream
	_, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrameHeader, []byte{0x80}), false)
	if !errors.Is(err, ErrMissingSequenceHeader) {
		t.Fatalf("PushOBUInto err=%v want %v", err, ErrMissingSequenceHeader)
	}
}

func TestStreamRejectsTileBeforeFrameHeader(t *testing.T) {
	var dec Stream
	_, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16)), false)
	if err != nil {
		t.Fatal(err)
	}
	_, err = pushOBU(&dec, appendRTPElement(nil, obu.TypeTileGroup, []byte{0x80}), false)
	if !errors.Is(err, ErrMissingFrameHeader) {
		t.Fatalf("PushOBUInto err=%v want %v", err, ErrMissingFrameHeader)
	}
}

func TestStreamRejectsTileAfterFinalTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [4]Event
	_, err := dec.PushLowOverhead(stream, events[:])
	if !errors.Is(err, ErrMissingFrameHeader) {
		t.Fatalf("PushLowOverhead err=%v want %v", err, ErrMissingFrameHeader)
	}
}

func TestStreamSequenceChange(t *testing.T) {
	var dec Stream
	first, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16)), false)
	if err != nil {
		t.Fatal(err)
	}
	second, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16)), false)
	if err != nil {
		t.Fatal(err)
	}
	third, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(32)), false)
	if err != nil {
		t.Fatal(err)
	}

	if !first.NewCodedVideoSequence || second.NewCodedVideoSequence || !third.NewCodedVideoSequence {
		t.Fatalf("new sequence flags: first=%v second=%v third=%v", first.NewCodedVideoSequence, second.NewCodedVideoSequence, third.NewCodedVideoSequence)
	}
}

func TestStreamNewCodedVideoSequenceFlagResetsSameSequence(t *testing.T) {
	var dec Stream
	sequence := appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	if _, err := pushOBU(&dec, sequence, false); err != nil {
		t.Fatal(err)
	}
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}
	if !dec.haveFrameHeader || dec.references.Frames[0].Valid {
		t.Fatalf("frame header: pending=%v ref0=%+v, want pending and no reference update before the last tile", dec.haveFrameHeader, dec.references.Frames[0])
	}
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeTileGroup, []byte{0}), false); err != nil {
		t.Fatal(err)
	}
	if dec.haveFrameHeader || !dec.references.Frames[0].Valid {
		t.Fatalf("last tile did not complete the frame: pending=%v ref0=%+v", dec.haveFrameHeader, dec.references.Frames[0])
	}
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}

	event, err := pushOBU(&dec, sequence, true)
	if err != nil {
		t.Fatal(err)
	}
	if !event.NewCodedVideoSequence || dec.haveFrameHeader || dec.references.Frames[0].Valid {
		t.Fatalf("event=%+v pending=%v ref0=%+v", event, dec.haveFrameHeader, dec.references.Frames[0])
	}
}

func TestStreamRTPPayload(t *testing.T) {
	elements := []rtp.Element{
		{Data: appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))},
		{Data: appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())},
	}
	var payload [128]byte
	n, err := rtp.PutPayload(payload[:], rtp.AggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var dec Stream
	var out [128]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event
	used, count, err := dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used == 0 || count != 2 {
		t.Fatalf("used=%d count=%d", used, count)
	}
	if events[0].Kind != EventSequenceHeader || !events[0].NewCodedVideoSequence {
		t.Fatalf("events[0]=%+v", events[0])
	}
	if events[1].Kind != EventFrameHeader {
		t.Fatalf("events[1]=%+v", events[1])
	}
	if events[1].SequenceHeader != events[0].SequenceHeader {
		t.Fatalf("events[1] sequence=%+v want %+v", events[1].SequenceHeader, events[0].SequenceHeader)
	}
	if events[1].FrameHeader.FrameType != parser.FrameTypeKey || !events[1].FrameHeader.DisableCDFUpdate {
		t.Fatalf("events[1] frame header=%+v", events[1].FrameHeader)
	}
	if events[1].FrameSize.CodedWidth != 16 || events[1].FrameSize.RenderHeight != 9 {
		t.Fatalf("events[1] frame size=%+v", events[1].FrameSize)
	}
	if events[1].TileInfo.Cols != 1 || events[1].TileInfo.Rows != 1 {
		t.Fatalf("events[1] tile info=%+v", events[1].TileInfo)
	}
	if events[1].Quantization.BaseQIdx != 0 {
		t.Fatalf("events[1] quantization=%+v", events[1].Quantization)
	}
	if events[1].Segmentation.Enabled || !events[1].Segmentation.AllLossless {
		t.Fatalf("events[1] segmentation=%+v", events[1].Segmentation)
	}
	if events[1].Delta.DeltaQPresent || events[1].Delta.BitsRead != events[1].Segmentation.BitsRead {
		t.Fatalf("events[1] delta=%+v", events[1].Delta)
	}
	if !events[1].LoopFilter.ModeRefDeltaEnabled || !events[1].LoopFilter.ModeRefDeltaUpdate {
		t.Fatalf("events[1] loopfilter=%+v", events[1].LoopFilter)
	}
	if events[1].CDEF.BitsRead != events[1].LoopFilter.BitsRead {
		t.Fatalf("events[1] cdef=%+v", events[1].CDEF)
	}
	if events[1].Restoration.BitsRead != events[1].CDEF.BitsRead {
		t.Fatalf("events[1] restoration=%+v", events[1].Restoration)
	}
	if events[1].TransformRef.TransformMode != parser.TransformMode4x4Only ||
		events[1].TransformRef.ReferenceMode != parser.ReferenceModeSingle ||
		events[1].TransformRef.BitsRead != events[1].Restoration.BitsRead {
		t.Fatalf("events[1] transform/reference=%+v", events[1].TransformRef)
	}
	if events[1].SkipMode.Allowed || events[1].SkipMode.Enabled ||
		events[1].SkipMode.BitsRead != events[1].TransformRef.BitsRead {
		t.Fatalf("events[1] skip mode=%+v", events[1].SkipMode)
	}
	if events[1].FrameMode.AllowWarpedMotion || events[1].FrameMode.ReducedTxSet ||
		events[1].FrameMode.BitsRead != events[1].SkipMode.BitsRead+1 {
		t.Fatalf("events[1] frame mode=%+v", events[1].FrameMode)
	}
	if events[1].GlobalMotion.BitsRead != events[1].FrameMode.BitsRead {
		t.Fatalf("events[1] global motion=%+v", events[1].GlobalMotion)
	}
	if events[1].FilmGrain.Apply || events[1].FilmGrain.BitsRead != events[1].GlobalMotion.BitsRead {
		t.Fatalf("events[1] film grain=%+v", events[1].FilmGrain)
	}
}

func TestStreamRTPPayloadSizeMatchesPush(t *testing.T) {
	elements := []rtp.Element{
		{Data: appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))},
		{Data: appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())},
	}
	var payload [128]byte
	n, err := rtp.PutPayload(payload[:], rtp.AggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var dec Stream
	plannedUsed, plannedEvents, err := dec.PushRTPPayloadSize(0, payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if plannedUsed == 0 || plannedEvents != 2 {
		t.Fatalf("planned used=%d events=%d", plannedUsed, plannedEvents)
	}
	if dec.HasSequenceHeader() || dec.InRTPFragment() {
		t.Fatal("PushRTPPayloadSize mutated stream state")
	}

	var out [128]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event
	used, count, err := dec.PushRTPPayload(out[:plannedUsed], 0, spans[:plannedEvents], events[:plannedEvents], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != plannedUsed || count != plannedEvents {
		t.Fatalf("push used=%d count=%d planned %d,%d", used, count, plannedUsed, plannedEvents)
	}
	if events[0].Kind != EventSequenceHeader || events[1].Kind != EventFrameHeader {
		t.Fatalf("events=%+v,%+v", events[0], events[1])
	}
}

func TestStreamRTPPayloadFragmentedFrameHeader(t *testing.T) {
	var dec Stream
	var out [256]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event

	sequence := []rtp.Element{{Data: appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))}}
	var payload [128]byte
	n, err := rtp.PutPayload(payload[:], rtp.AggregationHeader{
		ElementCount:                1,
		StartsNewCodedVideoSequence: true,
	}, sequence)
	if err != nil {
		t.Fatal(err)
	}
	used, count, err := dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used == 0 || count != 1 || events[0].Kind != EventSequenceHeader {
		t.Fatalf("sequence used=%d count=%d event=%+v", used, count, events[0])
	}

	frameHeader := appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	var packet [8]byte
	n, next, more, err := rtp.PutFragment(packet[:], frameHeader, 0, 4, false)
	if err != nil {
		t.Fatal(err)
	}
	if !more {
		t.Fatal("expected frame header to need a second fragment")
	}
	used = 0
	plannedUsed, plannedEvents, err := dec.PushRTPPayloadSize(used, packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if plannedUsed == 0 || plannedEvents != 0 || dec.InRTPFragment() {
		t.Fatalf("first fragment plan used=%d events=%d inFragment=%v", plannedUsed, plannedEvents, dec.InRTPFragment())
	}
	used, count, err = dec.PushRTPPayload(out[:], used, spans[:], events[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || !dec.InRTPFragment() {
		t.Fatalf("first fragment count=%d inFragment=%v", count, dec.InRTPFragment())
	}

	n, _, more, err = rtp.PutFragment(packet[:], frameHeader, next, len(packet), false)
	if err != nil {
		t.Fatal(err)
	}
	if more {
		t.Fatal("unexpected third fragment")
	}
	plannedUsed, plannedEvents, err = dec.PushRTPPayloadSize(used, packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if plannedUsed != len(frameHeader) || plannedEvents != 1 || !dec.InRTPFragment() {
		t.Fatalf("final fragment plan used=%d events=%d inFragment=%v", plannedUsed, plannedEvents, dec.InRTPFragment())
	}
	used, count, err = dec.PushRTPPayload(out[:], used, spans[:], events[:], packet[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != len(frameHeader) || count != 1 || dec.InRTPFragment() {
		t.Fatalf("final fragment used=%d count=%d inFragment=%v", used, count, dec.InRTPFragment())
	}
	if events[0].Kind != EventFrameHeader || events[0].FrameHeader.FrameType != parser.FrameTypeKey {
		t.Fatalf("frame header event=%+v", events[0])
	}
}

func TestStreamRTPPayloadAllocs(t *testing.T) {
	elements := []rtp.Element{
		{Data: appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))},
		{Data: appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())},
	}
	var payload [128]byte
	n, err := rtp.PutPayload(payload[:], rtp.AggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var out [128]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event
	allocs := testing.AllocsPerRun(1000, func() {
		var dec Stream
		plannedUsed, plannedEvents, err := dec.PushRTPPayloadSize(0, payload[:n])
		if err != nil {
			t.Fatal(err)
		}
		if plannedUsed == 0 || plannedEvents != 2 {
			t.Fatalf("planned used=%d events=%d", plannedUsed, plannedEvents)
		}
		used, count, err := dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
		if err != nil {
			t.Fatal(err)
		}
		if used == 0 || count != 2 {
			t.Fatalf("used=%d count=%d", used, count)
		}
	})
	if allocs != 0 {
		t.Fatalf("Stream.PushRTPPayload allocated: %f", allocs)
	}
}

func BenchmarkStreamRTPPayload(b *testing.B) {
	elements := []rtp.Element{
		{Data: appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))},
		{Data: appendRTPElement(nil, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())},
	}
	var payload [128]byte
	n, err := rtp.PutPayload(payload[:], rtp.AggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		b.Fatal(err)
	}

	var out [128]byte
	var spans [4]rtp.OBUSpan
	var events [4]Event
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var dec Stream
		_, _, _ = dec.PushRTPPayload(out[:], 0, spans[:], events[:], payload[:n])
	}
}

func TestStreamInterFrameUsesReferenceState(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	// Both headers end on a byte boundary, so each frame_header_obu's
	// trailing one bit is a whole 0x80 byte.
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, append(shownKeyFrameHeaderPayload(), 0x80))
	// The key frame's single tile completes it; only a completed frame
	// refreshes the reference slots (dav1d submits it on its last tile).
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0})
	stream = appendLowOverheadOBU(stream, obu.TypeTemporalDelimiter, nil)
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, append(interFrameHeaderPayload(), 0x80))

	var dec Stream
	var events [5]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 5 {
		t.Fatalf("count=%d", count)
	}
	if events[1].FrameSize.RefreshFrameFlags != 0xff {
		t.Fatalf("key frame size=%+v", events[1].FrameSize)
	}
	if events[4].FrameHeader.FrameType != parser.FrameTypeInter {
		t.Fatalf("inter header=%+v", events[4].FrameHeader)
	}
	if events[4].FrameSize.RefreshFrameFlags != 0x01 {
		t.Fatalf("inter frame size=%+v", events[4].FrameSize)
	}
	if events[4].TileInfo.Cols != 1 || events[4].TileInfo.Rows != 1 {
		t.Fatalf("inter tile info=%+v", events[4].TileInfo)
	}
	if events[4].Quantization.BaseQIdx != 0 {
		t.Fatalf("inter quantization=%+v", events[4].Quantization)
	}
	if events[4].Segmentation.Enabled || !events[4].Segmentation.AllLossless {
		t.Fatalf("inter segmentation=%+v", events[4].Segmentation)
	}
	if events[4].Delta.DeltaQPresent || events[4].Delta.BitsRead != events[4].Segmentation.BitsRead {
		t.Fatalf("inter delta=%+v", events[4].Delta)
	}
	if !events[4].LoopFilter.ModeRefDeltaEnabled || !events[4].LoopFilter.ModeRefDeltaUpdate {
		t.Fatalf("inter loopfilter=%+v", events[4].LoopFilter)
	}
	if events[4].CDEF.BitsRead != events[4].LoopFilter.BitsRead {
		t.Fatalf("inter cdef=%+v", events[4].CDEF)
	}
	if events[4].Restoration.BitsRead != events[4].CDEF.BitsRead {
		t.Fatalf("inter restoration=%+v", events[4].Restoration)
	}
	if events[4].TransformRef.TransformMode != parser.TransformMode4x4Only ||
		events[4].TransformRef.ReferenceMode != parser.ReferenceModeSingle ||
		events[4].TransformRef.BitsRead != events[4].Restoration.BitsRead+1 {
		t.Fatalf("inter transform/reference=%+v", events[4].TransformRef)
	}
	if events[4].SkipMode.Allowed || events[4].SkipMode.Enabled ||
		events[4].SkipMode.BitsRead != events[4].TransformRef.BitsRead {
		t.Fatalf("inter skip mode=%+v", events[4].SkipMode)
	}
	if events[4].FrameMode.AllowWarpedMotion || events[4].FrameMode.ReducedTxSet ||
		events[4].FrameMode.BitsRead != events[4].SkipMode.BitsRead+1 {
		t.Fatalf("inter frame mode=%+v", events[4].FrameMode)
	}
	if events[4].GlobalMotion.BitsRead != events[4].FrameMode.BitsRead+parser.InterRefsPerFrame {
		t.Fatalf("inter global motion=%+v", events[4].GlobalMotion)
	}
	if events[4].FilmGrain.Apply || events[4].FilmGrain.BitsRead != events[4].GlobalMotion.BitsRead {
		t.Fatalf("inter film grain=%+v", events[4].FilmGrain)
	}
	for i := range parser.InterRefsPerFrame {
		if events[4].GlobalMotion.Ref[i].Type != parser.GlobalMotionIdentity {
			t.Fatalf("inter global motion ref[%d]=%+v", i, events[4].GlobalMotion.Ref[i])
		}
	}
	for i := range parser.InterRefsPerFrame {
		if events[4].FrameSize.RefFrameIdx[i] != 0 {
			t.Fatalf("inter RefFrameIdx[%d]=%d", i, events[4].FrameSize.RefFrameIdx[i])
		}
	}
}

func TestReferenceOrderHintsFromReferenceState(t *testing.T) {
	refs := parser.ReferenceState{}
	refs.Frames[2] = parser.ReferenceFrame{Valid: true, OrderHint: 11}
	refs.Frames[5] = parser.ReferenceFrame{Valid: true, OrderHint: 21}
	size := parser.FrameSize{
		RefFrameIdx: [parser.InterRefsPerFrame]uint8{2, 5, 2, 5, 2, 5, 2},
	}
	got := referenceOrderHints(size, &refs)
	want := [parser.InterRefsPerFrame]uint8{11, 21, 11, 21, 11, 21, 11}
	if got != want {
		t.Fatalf("reference order hints=%v want %v", got, want)
	}
}

func TestStreamCarriesReferenceFrameHeaderState(t *testing.T) {
	var dec Stream
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}

	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, keyFrame), false); err != nil {
		t.Fatal(err)
	}

	updateFrame := append([]byte{}, interFrameHeaderPayloadWithReferenceState(true)...)
	updateFrame = append(updateFrame, 0xbb)
	updateEvent, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, updateFrame), false)
	if err != nil {
		t.Fatal(err)
	}
	if !updateEvent.Segmentation.Enabled || !updateEvent.Segmentation.UpdateData ||
		updateEvent.Segmentation.Data.Segments[0].DeltaQ != 5 || updateEvent.Segmentation.QIndex[0] != 45 {
		t.Fatalf("updated segmentation=%+v", updateEvent.Segmentation)
	}
	if !updateEvent.LoopFilter.ModeRefDeltaUpdate || updateEvent.LoopFilter.Deltas.Ref[2] != 4 {
		t.Fatalf("updated loopfilter=%+v", updateEvent.LoopFilter)
	}
	if dec.references.Frames[0].Segmentation.Segments[0].DeltaQ != 5 ||
		dec.references.Frames[0].LoopFilterDeltas.Ref[2] != 4 {
		t.Fatalf("stored reference=%+v", dec.references.Frames[0])
	}

	copyFrame := append([]byte{}, interFrameHeaderPayloadWithReferenceState(false)...)
	copyFrame = append(copyFrame, 0xcc)
	copyEvent, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, copyFrame), false)
	if err != nil {
		t.Fatal(err)
	}
	if !copyEvent.Segmentation.Enabled || copyEvent.Segmentation.UpdateData ||
		copyEvent.Segmentation.Data.Segments[0].DeltaQ != 5 || copyEvent.Segmentation.QIndex[0] != 45 {
		t.Fatalf("copied segmentation=%+v", copyEvent.Segmentation)
	}
	if copyEvent.LoopFilter.ModeRefDeltaUpdate || copyEvent.LoopFilter.Deltas.Ref[2] != 4 {
		t.Fatalf("copied loopfilter=%+v", copyEvent.LoopFilter)
	}
}

func TestStreamSwitchFrameRefreshesAllReferenceSlots(t *testing.T) {
	// AV1 §5.9.1: an S_FRAME implies error_resilient_mode = 1 and (via
	// frame_size syntax) refresh_frame_flags = 0xff. A decoder joining
	// the stream at this frame must therefore resume without prior
	// state: every reference slot is rewritten to the decoded switch
	// surface.
	var dec Stream
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}

	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	keyEvent, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, keyFrame), false)
	if err != nil {
		t.Fatal(err)
	}
	if keyEvent.FrameHeader.FrameType != parser.FrameTypeKey {
		t.Fatalf("key event=%+v", keyEvent)
	}
	if keyEvent.FrameSize.RefreshFrameFlags != 0xff {
		t.Fatalf("key RefreshFrameFlags=%02x want ff", keyEvent.FrameSize.RefreshFrameFlags)
	}
	for i := range parser.RefFrames {
		if !dec.references.Frames[i].Valid {
			t.Fatalf("references[%d] should be valid after key", i)
		}
	}

	switchFrame := append([]byte{}, switchFrameHeaderPayload()...)
	switchFrame = append(switchFrame, 0xbb)
	event, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, switchFrame), false)
	if err != nil {
		t.Fatalf("S_FRAME PushOBUInto err=%v", err)
	}
	if event.Kind != EventFrame {
		t.Fatalf("S_FRAME event kind=%v", event.Kind)
	}
	if event.FrameHeader.FrameType != parser.FrameTypeSwitch {
		t.Fatalf("S_FRAME FrameType=%v", event.FrameHeader.FrameType)
	}
	if !event.FrameHeader.ErrorResilientMode || !event.FrameHeader.FrameSizeOverride {
		t.Fatalf("S_FRAME header=%+v", event.FrameHeader)
	}
	if event.FrameSize.RefreshFrameFlags != 0xff {
		t.Fatalf("S_FRAME RefreshFrameFlags=%02x want ff", event.FrameSize.RefreshFrameFlags)
	}
	if event.FrameHeader.PrimaryRefFrame != parser.PrimaryRefNone {
		t.Fatalf("S_FRAME PrimaryRefFrame=%d want %d", event.FrameHeader.PrimaryRefFrame, parser.PrimaryRefNone)
	}
	if !event.TileGroup.Final {
		t.Fatalf("S_FRAME tile group not final: %+v", event.TileGroup)
	}
	for i := range parser.RefFrames {
		ref := dec.references.Frames[i]
		if !ref.Valid || ref.FrameType != parser.FrameTypeSwitch {
			t.Fatalf("references[%d]=%+v want valid FrameTypeSwitch", i, ref)
		}
	}
}

func TestStreamSwitchFrameSurfaceReferencesReset(t *testing.T) {
	// SurfaceReferences book-keeping: an S_FRAME with
	// refresh_frame_flags = 0xff rewrites every surface slot and
	// releases all previously held distinct surfaces.
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x0f, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	event := Event{
		Kind: EventFrame,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:          parser.FrameTypeSwitch,
			ShowFrame:          true,
			ErrorResilientMode: true,
			FrameSizeOverride:  true,
		},
		FrameSize: parser.FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: parser.TileGroup{Final: true},
	}
	count, err := refs.FinishFrame(&event, 7, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("release count=%d want 2", count)
	}
	released5, released6 := false, false
	for i := range count {
		switch releases[i] {
		case 5:
			released5 = true
		case 6:
			released6 = true
		}
	}
	if !released5 || !released6 {
		t.Fatalf("releases=%v want both 5 and 6", releases[:count])
	}
	for i := range parser.RefFrames {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != 7 {
			t.Fatalf("after S_FRAME, ref[%d]=%d ok=%v want 7", i, slot, ok)
		}
	}
}

func TestStreamSwitchFrameProvidesReferencesForFollowingInter(t *testing.T) {
	// Surface routing treats S_FRAME like inter for reference
	// resolution; the seven inter slots all resolve to the switch
	// surface until the next refresh.
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, 9, releases[:]); err != nil {
		t.Fatal(err)
	}
	event := Event{
		Kind: EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:          parser.FrameTypeSwitch,
			ErrorResilientMode: true,
			FrameSizeOverride:  true,
		},
		FrameSize: parser.FrameSize{
			RefFrameIdx: [parser.InterRefsPerFrame]uint8{0, 1, 2, 3, 4, 5, 6},
		},
	}
	var surfaces [parser.InterRefsPerFrame]int
	count, err := refs.FrameReferences(&event, surfaces[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != parser.InterRefsPerFrame {
		t.Fatalf("count=%d", count)
	}
	for i := range parser.InterRefsPerFrame {
		if surfaces[i] != 9 {
			t.Fatalf("surfaces[%d]=%d want 9", i, surfaces[i])
		}
	}
}

func TestStreamShowExistingFrame(t *testing.T) {
	var dec Stream
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}

	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, keyFrame), false); err != nil {
		t.Fatal(err)
	}

	interFrame := append([]byte{}, interFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)
	interEvent, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, interFrame), false)
	if err != nil {
		t.Fatal(err)
	}
	if interEvent.Kind != EventFrame || interEvent.FrameHeader.FrameType != parser.FrameTypeInter {
		t.Fatalf("inter event=%+v", interEvent)
	}

	event, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrameHeader, showExistingFrameHeaderPayload(0)), false)
	if err != nil {
		t.Fatal(err)
	}
	if event.Kind != EventExistingFrame || !event.FrameHeader.ShowExistingFrame || event.FrameHeader.ExistingFrameIdx != 0 {
		t.Fatalf("show existing event=%+v", event)
	}
	if !event.ExistingFrame.Valid || event.ExistingFrame.FrameType != parser.FrameTypeInter {
		t.Fatalf("existing frame=%+v", event.ExistingFrame)
	}
	if event.FrameSize != interEvent.FrameSize || event.FilmGrain != interEvent.FilmGrain {
		t.Fatalf("show existing state=%+v inter=%+v", event, interEvent)
	}

	_, err = pushOBU(&dec, appendRTPElement(nil, obu.TypeTileGroup, []byte{0x80}), false)
	if !errors.Is(err, ErrMissingFrameHeader) {
		t.Fatalf("PushOBUInto err=%v want %v", err, ErrMissingFrameHeader)
	}
}

func TestStreamRejectsShowExistingBeforeReference(t *testing.T) {
	var dec Stream
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}
	_, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrameHeader, showExistingFrameHeaderPayload(0)), false)
	if !errors.Is(err, parser.ErrReferenceFrameNeeded) {
		t.Fatalf("PushOBUInto err=%v want %v", err, parser.ErrReferenceFrameNeeded)
	}
}

func TestStreamRejectsUnshowableExistingFrame(t *testing.T) {
	var dec Stream
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}
	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, keyFrame), false); err != nil {
		t.Fatal(err)
	}

	_, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrameHeader, showExistingFrameHeaderPayload(0)), false)
	if !errors.Is(err, parser.ErrInvalidFrameHeader) {
		t.Fatalf("PushOBUInto err=%v want %v", err, parser.ErrInvalidFrameHeader)
	}
}

func TestStreamRejectsFrameOBUShowExistingFrame(t *testing.T) {
	var dec Stream
	if _, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload()), false); err != nil {
		t.Fatal(err)
	}
	_, err := pushOBU(&dec, appendRTPElement(nil, obu.TypeFrame, showExistingFrameHeaderPayload(0)), false)
	if !errors.Is(err, parser.ErrInvalidFrameHeader) {
		t.Fatalf("PushOBUInto err=%v want %v", err, parser.ErrInvalidFrameHeader)
	}
}

func TestStreamAllocs(t *testing.T) {
	raw := appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	allocs := testing.AllocsPerRun(1000, func() {
		var dec Stream
		var event Event
		if err := dec.PushOBUInto(&event, raw, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Stream allocated: %f", allocs)
	}
}

func BenchmarkStreamPushOBUInto(b *testing.B) {
	raw := appendRTPElement(nil, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var dec Stream
		var event Event
		_ = dec.PushOBUInto(&event, raw, false)
	}
}

// TestStreamFrameHeaderOBUNeedsTrailingBit ports dav1d's non-strict
// check_trailing_bits for OBU_FRAME_HDR and OBU_REDUNDANT_FRAME_HDR
// (src/obu.c): a frame_header_obu whose payload ends exactly where
// uncompressed_header() ends has no trailing one bit and is rejected. The
// bit's value and the padding are not checked.
func TestStreamFrameHeaderOBUNeedsTrailingBit(t *testing.T) {
	seq := appendLowOverheadOBU(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	key := shownKeyFrameHeaderPayload() // ends on a byte boundary
	for _, tc := range []struct {
		name string
		typ  obu.Type
		tail []byte
		ok   bool
	}{
		{"frame header without trailing bit", obu.TypeFrameHeader, nil, false},
		{"frame header with trailing byte", obu.TypeFrameHeader, []byte{0x80}, true},
		{"frame header with non-strict trailing bytes", obu.TypeFrameHeader, []byte{0x00, 0x01}, true},
		{"redundant frame header without trailing bit", obu.TypeRedundantFrameHeader, nil, false},
		{"redundant frame header with trailing byte", obu.TypeRedundantFrameHeader, []byte{0x80}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := append(append([]byte{}, key...), tc.tail...)
			stream := appendLowOverheadOBU(append([]byte{}, seq...), tc.typ, payload)
			var dec Stream
			var events [2]Event
			_, err := dec.PushLowOverhead(stream, events[:])
			if tc.ok && err != nil {
				t.Fatalf("PushLowOverhead: %v", err)
			}
			if !tc.ok && !errors.Is(err, bitstream.ErrNotEnoughBits) {
				t.Fatalf("PushLowOverhead err=%v want %v", err, bitstream.ErrNotEnoughBits)
			}
		})
	}
}

// dav1d submits a frame, and updates the reference slots, only when its tile
// groups have carried every tile. The events of a frame that is replaced by
// a later frame header or left incomplete at the end of the temporal unit are
// returned as EventIgnored and leave the references untouched.
func TestStreamIncompleteFrameIsIgnored(t *testing.T) {
	key := append(shownKeyFrameHeaderPayload(), 0x80)
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, key)
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, key)
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0})
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, key)

	var dec Stream
	var events [5]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 5 {
		t.Fatalf("count=%d", count)
	}
	want := []struct{ kind, dropped EventKind }{
		{EventSequenceHeader, EventIgnored},
		{EventIgnored, EventFrameHeader}, // replaced by the next frame header
		{EventFrameHeader, EventIgnored},
		{EventTileGroup, EventIgnored},
		{EventIgnored, EventFrameHeader}, // no tile group before the unit ends
	}
	for i, w := range want {
		if events[i].Kind != w.kind || events[i].DroppedKind != w.dropped {
			t.Fatalf("event %d kind=%d dropped=%d, want %d/%d", i, events[i].Kind, events[i].DroppedKind, w.kind, w.dropped)
		}
	}
	if !dec.references.Frames[0].Valid {
		t.Fatal("the completed key frame did not refresh the references")
	}

	var fresh Stream
	stream = appendLowOverheadOBU(nil, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, key)
	count, err = fresh.PushLowOverhead(stream, events[:])
	if err != nil || count != 2 || events[1].Kind != EventIgnored {
		t.Fatalf("count=%d err=%v kind=%d", count, err, events[1].Kind)
	}
	if fresh.references.Frames[0].Valid {
		t.Fatal("an incomplete frame refreshed the references")
	}
}

func TestFinishFrameSurfaceReleasesPool(t *testing.T) {
	pool := testFramePool(t, 3)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := FinishFrameSurface(&refs, &pool, finalFrameEventPtr(0xff), index1, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != index0 {
		t.Fatalf("count=%d releases[0]=%d want %d", count, releases[0], index0)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	if _, err := pool.Frame(index0); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("released frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index1 {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, index1)
	}
}

func TestResetFrameSurfaceReferencesReleasesPool(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x03, index0, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x04, index1, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := ResetFrameSurfaceReferences(&refs, &pool, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 || pool.Available() != 2 || refs.Holds(index0) || refs.Holds(index1) {
		t.Fatalf("count=%d available=%d holds0=%v holds1=%v releases=%v", count, pool.Available(), refs.Holds(index0), refs.Holds(index1), releases[:count])
	}
}

func TestResetFrameSurfaceReferencesKeepsRefsOnPoolError(t *testing.T) {
	pool := testFramePool(t, 1)
	index, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x01, index, releases[:]); err != nil {
		t.Fatal(err)
	}

	_, err = ResetFrameSurfaceReferences(&refs, nil, releases[:])
	if !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("ResetFrameSurfaceReferences err=%v want %v", err, frame.ErrInvalidPool)
	}
	if !refs.Holds(index) || pool.Available() != 0 {
		t.Fatalf("refs/pool changed on error: holds=%v available=%d", refs.Holds(index), pool.Available())
	}
}

func TestAcquireFrameSurface(t *testing.T) {
	pool := testFramePool(t, 1)
	index, acquired, err := AcquireFrameSurface(&pool, testSequence(), testFrameSize(16, 16), 32)
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || acquired == nil {
		t.Fatalf("index=%d frame=%p", index, acquired)
	}
	if pool.Available() != 0 {
		t.Fatalf("available=%d want 0", pool.Available())
	}
}

func TestFrameFormatHelpersKeepCodedSurfaceSeparateFromOutput(t *testing.T) {
	size := parser.FrameSize{
		CodedWidth:    16,
		UpscaledWidth: 32,
		Height:        16,
	}
	coded, err := codedFrameFormatFromHeaders(testSequence(), size, 32)
	if err != nil {
		t.Fatal(err)
	}
	if coded.Width != 16 {
		t.Fatalf("coded=%+v", coded)
	}
	output, err := outputFrameFormatFromHeaders(testSequence(), size, 32)
	if err != nil {
		t.Fatal(err)
	}
	if output.Width != 32 || output.Height != 16 || output.BitDepth != testSequence().ColorConfig.BitDepth {
		t.Fatalf("output=%+v", output)
	}

	pool := testFramePool(t, 1)
	index, acquired, err := AcquireFrameSurface(&pool, testSequence(), size, 32)
	if err != nil {
		t.Fatal(err)
	}
	if index != 0 || acquired == nil || acquired.Format.Width != 16 {
		t.Fatalf("index=%d acquired=%+v", index, acquired)
	}
}

func TestAcquireFrameSurfaceRejectsFormatMismatch(t *testing.T) {
	pool := testFramePool(t, 1)
	_, _, err := AcquireFrameSurface(&pool, testSequence(), testFrameSize(80, 16), 32)
	if !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("AcquireFrameSurface err=%v want %v", err, frame.ErrInvalidFormat)
	}
	if pool.Available() != 1 {
		t.Fatalf("mismatch consumed a slot, available=%d", pool.Available())
	}
}

func TestBeginFrameSurfaceIntra(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, refCount, err := BeginFrameSurface(nil, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil)
	if err != nil {
		t.Fatal(err)
	}
	if surface != 0 || output == nil || refCount != 0 {
		t.Fatalf("surface=%d output=%p refCount=%d", surface, output, refCount)
	}
	if pool.Available() != 0 {
		t.Fatalf("available=%d want 0", pool.Available())
	}
}

func TestBeginFrameSurfaceInter(t *testing.T) {
	pool := testFramePool(t, parser.RefFrames+1)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	for i := range parser.RefFrames {
		index, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if index != i {
			t.Fatalf("reference acquire index=%d want %d", index, i)
		}
		if _, err := refs.Refresh(1<<uint(i), index, releases[:]); err != nil {
			t.Fatal(err)
		}
	}

	var references [parser.InterRefsPerFrame]int
	surface, output, refCount, err := BeginFrameSurface(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
		FrameSize: parser.FrameSize{
			CodedWidth:  16,
			Height:      16,
			RefFrameIdx: [parser.InterRefsPerFrame]uint8{0, 1, 2, 3, 4, 5, 6},
		},
	}, 32, references[:])
	if err != nil {
		t.Fatal(err)
	}
	if surface != parser.RefFrames || output == nil || refCount != parser.InterRefsPerFrame {
		t.Fatalf("surface=%d output=%p refCount=%d", surface, output, refCount)
	}
	for i := range parser.InterRefsPerFrame {
		if references[i] != i {
			t.Fatalf("references[%d]=%d want %d", i, references[i], i)
		}
	}
	if pool.Available() != 0 {
		t.Fatalf("available=%d want 0", pool.Available())
	}
}

func TestBeginFrameSurfaceRejectsReferencesBeforeAcquire(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	_, _, _, err := BeginFrameSurface(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil)
	if !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("BeginFrameSurface err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
	if pool.Available() != 1 {
		t.Fatalf("reference failure consumed a slot, available=%d", pool.Available())
	}
}

func TestBeginFrameSurfaceRejectsFormatBeforeAcquire(t *testing.T) {
	pool := testFramePool(t, 1)
	_, _, _, err := BeginFrameSurface(nil, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(80, 16),
	}, 32, nil)
	if !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("BeginFrameSurface err=%v want %v", err, frame.ErrInvalidFormat)
	}
	if pool.Available() != 1 {
		t.Fatalf("format failure consumed a slot, available=%d", pool.Available())
	}
}

func TestBeginFrameSurfaceRejectsTileGroup(t *testing.T) {
	pool := testFramePool(t, 1)
	_, _, _, err := BeginFrameSurface(nil, &pool, testSequence(), &Event{
		Kind:      EventTileGroup,
		FrameSize: testFrameSize(16, 16),
	}, 32, nil)
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("BeginFrameSurface err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	if pool.Available() != 1 {
		t.Fatalf("invalid event consumed a slot, available=%d", pool.Available())
	}
}

func TestResolveFrameReferences(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, frame0, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, frame1, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	surfaces := [parser.InterRefsPerFrame]int{index0, index1}
	var refs [parser.InterRefsPerFrame]*frame.Frame
	count, err := ResolveFrameReferences(&pool, surfaces[:2], refs[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 || refs[0] != frame0 || refs[1] != frame1 {
		t.Fatalf("count=%d refs=%p,%p want %p,%p", count, refs[0], refs[1], frame0, frame1)
	}
}

func TestResolveFrameReferencesRejectsShortBuffer(t *testing.T) {
	pool := testFramePool(t, 1)
	index, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs [1]*frame.Frame
	_, err = ResolveFrameReferences(&pool, []int{index}, refs[:0])
	if !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("ResolveFrameReferences err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
	if refs[0] != nil {
		t.Fatalf("short buffer wrote ref=%p", refs[0])
	}
}

func TestResolveFrameReferencesRejectsInvalidAtomically(t *testing.T) {
	pool := testFramePool(t, 1)
	index, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var sentinel frame.Frame
	refs := [2]*frame.Frame{&sentinel, &sentinel}
	_, err = ResolveFrameReferences(&pool, []int{index, index + 1}, refs[:])
	if !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("ResolveFrameReferences err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if refs != ([2]*frame.Frame{&sentinel, &sentinel}) {
		t.Fatalf("invalid surface wrote refs=%p,%p", refs[0], refs[1])
	}
}

func TestResolveFrameReferencesRejectsInvalidSurface(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs [1]*frame.Frame
	_, err := ResolveFrameReferences(&pool, []int{-1}, refs[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("negative surface err=%v want %v", err, ErrInvalidSurfaceReference)
	}

	var tooMany [parser.InterRefsPerFrame + 1]int
	_, err = ResolveFrameReferences(&pool, tooMany[:], nil)
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("too many surfaces err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

func TestFinishFrameSurfaceRejectsPoolReleaseAtomically(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}

	_, err = FinishFrameSurface(&refs, &pool, finalFrameEventPtr(0xff), index1, releases[:])
	if !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("FinishFrameSurface err=%v want %v", err, frame.ErrInvalidSlot)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index0 {
		t.Fatalf("slot=%d ok=%v want unchanged %d", slot, ok, index0)
	}
	if pool.Available() != 1 {
		t.Fatalf("available=%d want 1", pool.Available())
	}
}

func TestSurfacePoolHelpersAllocs(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	event := finalFrameEvent(0xff)
	sequence := testSequence()
	size := testFrameSize(16, 16)
	beginEvent := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   size,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		index0, _, _, err := BeginFrameSurface(&refs, &pool, sequence, &beginEvent, 32, nil)
		if err != nil {
			t.Fatal(err)
		}
		index1, _, _, err := BeginFrameSurface(&refs, &pool, sequence, &beginEvent, 32, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
			t.Fatal(err)
		}
		referenceSurfaces[0] = index0
		if _, err := ResolveFrameReferences(&pool, referenceSurfaces[:1], referenceFrames[:]); err != nil {
			t.Fatal(err)
		}
		if referenceFrames[0] == nil {
			t.Fatal("nil reference frame")
		}
		if _, err := FinishFrameSurface(&refs, &pool, &event, index1, releases[:]); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("surface pool helpers allocated: %f", allocs)
	}
}

func BenchmarkAcquireFrameSurface(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	sequence := testSequence()
	size := testFrameSize(16, 16)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		index, _, err := AcquireFrameSurface(&pool, sequence, size, 32)
		if err != nil {
			b.Fatal(err)
		}
		if err := pool.Release(index); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBeginFrameSurface(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	sequence := testSequence()
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		surface, _, _, err := BeginFrameSurface(nil, &pool, sequence, &event, 32, nil)
		if err != nil {
			b.Fatal(err)
		}
		if err := pool.Release(surface); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFinishFrameSurface(b *testing.B) {
	pool := benchmarkFramePool(b, 2)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	event := finalFrameEvent(0xff)
	sequence := testSequence()
	size := testFrameSize(16, 16)
	beginEvent := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   size,
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		index0, _, _, err := BeginFrameSurface(&refs, &pool, sequence, &beginEvent, 32, nil)
		if err != nil {
			b.Fatal(err)
		}
		index1, _, _, err := BeginFrameSurface(&refs, &pool, sequence, &beginEvent, 32, nil)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
			b.Fatal(err)
		}
		if _, err := FinishFrameSurface(&refs, &pool, &event, index1, releases[:]); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkResolveFrameReferences(b *testing.B) {
	pool := benchmarkFramePool(b, parser.InterRefsPerFrame)
	var surfaces [parser.InterRefsPerFrame]int
	var refs [parser.InterRefsPerFrame]*frame.Frame
	for i := range parser.InterRefsPerFrame {
		index, _, err := pool.Acquire()
		if err != nil {
			b.Fatal(err)
		}
		surfaces[i] = index
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		count, err := ResolveFrameReferences(&pool, surfaces[:], refs[:])
		if err != nil {
			b.Fatal(err)
		}
		if count != parser.InterRefsPerFrame || refs[0] == nil {
			b.Fatalf("count=%d refs[0]=%p", count, refs[0])
		}
	}
}

func testSequence() parser.SequenceHeader {
	return parser.SequenceHeader{ColorConfig: parser.ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
}

func testFrameSize(width uint32, height uint32) parser.FrameSize {
	return parser.FrameSize{CodedWidth: width, Height: height}
}

func finalFrameEvent(refresh uint8) Event {
	return Event{
		Kind: EventTileGroup,
		FrameSize: parser.FrameSize{
			RefreshFrameFlags: refresh,
		},
		TileGroup: parser.TileGroup{Final: true},
	}
}

func testFramePool(t *testing.T, count int) frame.Pool {
	t.Helper()
	pool, err := makeFramePool(count)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func benchmarkFramePool(b *testing.B, count int) frame.Pool {
	b.Helper()
	pool, err := makeFramePool(count)
	if err != nil {
		b.Fatal(err)
	}
	return pool
}

func makeFramePool(count int) (frame.Pool, error) {
	format := frame.Format{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := frame.RequiredSize(format)
	if err != nil {
		return frame.Pool{}, err
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	return frame.BindPool(backing, format, frames, free, used)
}

func finalFrameEventPtr(refresh uint8) *Event {
	event := finalFrameEvent(refresh)
	return &event
}

func showExistingWorkEventPtr(index uint8, frameType parser.FrameType) *Event {
	event := showExistingWorkEvent(index, frameType)
	return &event
}

func TestSurfaceReferencesRefresh(t *testing.T) {
	var refs SurfaceReferences
	if _, ok := refs.ReferenceSlot(0); ok {
		t.Fatal("zero-value references should start empty")
	}

	var releases [parser.RefFrames]int
	count, err := refs.Refresh(0x03, 5, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("release count=%d", count)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
	if slot, ok := refs.ReferenceSlot(1); !ok || slot != 5 {
		t.Fatalf("ref[1]=%d ok=%v", slot, ok)
	}

	count, err = refs.Refresh(0x01, 6, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 || refs.Holds(5) != true || refs.Holds(6) != true {
		t.Fatalf("count=%d holds5=%v holds6=%v", count, refs.Holds(5), refs.Holds(6))
	}

	count, err = refs.Refresh(0x02, 7, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != 5 || refs.Holds(5) || !refs.Holds(6) || !refs.Holds(7) {
		t.Fatalf("count=%d releases=%v holds5=%v holds6=%v holds7=%v", count, releases[:count], refs.Holds(5), refs.Holds(6), refs.Holds(7))
	}
}

func TestSurfaceReferencesReleaseUniqueOnce(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x03, 5, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.Refresh(0x03, 6, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != 5 {
		t.Fatalf("count=%d releases=%v", count, releases[:count])
	}
}

func TestSurfaceReferencesReleaseAll(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.ReleaseAll(releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 || releases[0] != 5 || releases[1] != 6 || refs.Holds(5) || refs.Holds(6) {
		t.Fatalf("count=%d releases=%v holds5=%v holds6=%v", count, releases[:count], refs.Holds(5), refs.Holds(6))
	}
	if _, ok := refs.ReferenceSlot(0); ok {
		t.Fatal("reference slots were not reset")
	}
}

func TestSurfaceReferencesReleaseAllRejectsShortReleaseBuffer(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	_, err := refs.ReleaseAll(releases[:1])
	if !errors.Is(err, ErrSurfaceReleaseBufferTooSmall) {
		t.Fatalf("ReleaseAll err=%v want %v", err, ErrSurfaceReleaseBufferTooSmall)
	}
	if !refs.Holds(5) || !refs.Holds(6) {
		t.Fatalf("references changed on error: holds5=%v holds6=%v", refs.Holds(5), refs.Holds(6))
	}
}

func TestSurfaceReferencesFrameReferences(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	for i := range parser.RefFrames {
		if _, err := refs.Refresh(1<<uint(i), 10+i, releases[:]); err != nil {
			t.Fatal(err)
		}
	}

	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
		FrameSize: parser.FrameSize{
			RefFrameIdx: [parser.InterRefsPerFrame]uint8{0, 1, 2, 3, 4, 5, 6},
		},
	}
	var surfaces [parser.InterRefsPerFrame]int
	count, err := refs.FrameReferences(&event, surfaces[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != parser.InterRefsPerFrame {
		t.Fatalf("count=%d", count)
	}
	for i := range parser.InterRefsPerFrame {
		if surfaces[i] != 10+i {
			t.Fatalf("surface[%d]=%d want %d", i, surfaces[i], 10+i)
		}
	}
}

func TestSurfaceReferencesFrameReferencesIntra(t *testing.T) {
	var refs SurfaceReferences
	count, err := refs.FrameReferences(&Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("count=%d", count)
	}
}

func TestSurfaceReferencesFrameReferencesRejectsInvalidInputs(t *testing.T) {
	var refs SurfaceReferences
	var surfaces [parser.InterRefsPerFrame]int
	for i := range len(surfaces) {
		surfaces[i] = -99
	}

	_, err := refs.FrameReferences(&Event{Kind: EventExistingFrame}, surfaces[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FrameReferences event err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	_, err = refs.FrameReferences(&Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{ShowExistingFrame: true, FrameType: parser.FrameTypeInter},
	}, surfaces[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FrameReferences show-existing err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	_, err = refs.FrameReferences(&Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
	}, surfaces[:parser.InterRefsPerFrame-1])
	if !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("FrameReferences short buffer err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
	_, err = refs.FrameReferences(&Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
	}, surfaces[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("FrameReferences missing surface err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	for i := range len(surfaces) {
		if surfaces[i] != -99 {
			t.Fatalf("surface[%d]=%d changed on error", i, surfaces[i])
		}
	}
}

func TestSurfaceReferencesFinishFrame(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	event := Event{
		Kind: EventTileGroup,
		FrameSize: parser.FrameSize{
			RefreshFrameFlags: 0x03,
		},
		TileGroup: parser.TileGroup{Final: true},
	}

	count, err := refs.FinishFrame(&event, 5, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("count=%d", count)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
	if slot, ok := refs.ReferenceSlot(1); !ok || slot != 5 {
		t.Fatalf("ref[1]=%d ok=%v", slot, ok)
	}
}

func TestSurfaceReferencesFinishFrameRejectsIncompleteEvent(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	_, err := refs.FinishFrame(&Event{Kind: EventTileGroup}, 5, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FinishFrame non-final err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	_, err = refs.FinishFrame(&Event{Kind: EventFrameHeader, TileGroup: parser.TileGroup{Final: true}}, 5, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("FinishFrame wrong kind err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
}

func TestSurfaceReferencesRefreshRejectsShortReleaseBuffer(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x01, 5, releases[:]); err != nil {
		t.Fatal(err)
	}

	_, err := refs.Refresh(0x01, 6, nil)
	if !errors.Is(err, ErrSurfaceReleaseBufferTooSmall) {
		t.Fatalf("Refresh err=%v want %v", err, ErrSurfaceReleaseBufferTooSmall)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
}

func TestSurfaceReferencesShowExistingFrameEvent(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	surface, count, err := refs.ShowExistingFrame(&Event{
		Kind:        EventExistingFrame,
		FrameHeader: parser.FrameHeaderPrefix{ShowExistingFrame: true, ExistingFrameIdx: 1},
		ExistingFrame: parser.ReferenceFrame{
			FrameType: parser.FrameTypeKey,
		},
	}, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if surface != 6 || count != 1 || releases[0] != 5 {
		t.Fatalf("surface=%d count=%d releases=%v", surface, count, releases[:count])
	}
	for i := range parser.RefFrames {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != 6 {
			t.Fatalf("ref[%d]=%d ok=%v", i, slot, ok)
		}
	}
}

func TestSurfaceReferencesShowExistingFrameEventRejectsMissingSurface(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	_, _, err := refs.ShowExistingFrame(&Event{
		Kind:        EventExistingFrame,
		FrameHeader: parser.FrameHeaderPrefix{ShowExistingFrame: true, ExistingFrameIdx: 1},
	}, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("ShowExistingFrame err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	_, _, err = refs.ShowExistingFrame(&Event{Kind: EventFrameHeader}, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("ShowExistingFrame wrong kind err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
}

func TestSurfaceReferencesShowExistingKeyResetsReferences(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x05, 5, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(0x02, 6, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.ShowExisting(6, parser.FrameTypeKey, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != 5 {
		t.Fatalf("count=%d releases=%v", count, releases[:count])
	}
	for i := range parser.RefFrames {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != 6 {
			t.Fatalf("ref[%d]=%d ok=%v", i, slot, ok)
		}
	}
}

func TestSurfaceReferencesShowExistingInterLeavesReferences(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0x01, 5, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := refs.ShowExisting(6, parser.FrameTypeInter, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("count=%d", count)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != 5 {
		t.Fatalf("ref[0]=%d ok=%v", slot, ok)
	}
	if refs.Holds(6) {
		t.Fatal("inter show-existing should not retain new surface")
	}
}

func TestSurfaceReferencesRejectsInvalidSurface(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	_, err := refs.Refresh(0x01, -1, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("Refresh err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	_, err = refs.ShowExisting(-1, parser.FrameTypeKey, releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("ShowExisting err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

func TestSurfaceReferencesAllocs(t *testing.T) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	allocs := testing.AllocsPerRun(1000, func() {
		_, err := refs.Refresh(0xff, 5, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		_, err = refs.ShowExisting(6, parser.FrameTypeKey, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		_, err = refs.FinishFrame(&Event{
			Kind:      EventFrame,
			FrameSize: parser.FrameSize{RefreshFrameFlags: 0xff},
			TileGroup: parser.TileGroup{Final: true},
		}, 5, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		var surfaces [parser.InterRefsPerFrame]int
		_, err = refs.FrameReferences(&Event{
			Kind:        EventFrameHeader,
			FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter},
			FrameSize: parser.FrameSize{
				RefFrameIdx: [parser.InterRefsPerFrame]uint8{0, 1, 2, 3, 4, 5, 6},
			},
		}, surfaces[:])
		if err != nil {
			t.Fatal(err)
		}
		_, err = refs.ReleaseAll(releases[:])
		if err != nil {
			t.Fatal(err)
		}
		refs.Reset()
	})
	if allocs != 0 {
		t.Fatalf("SurfaceReferences allocated: %f", allocs)
	}
}

func BenchmarkSurfaceReferencesRefresh(b *testing.B) {
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = refs.Refresh(0xff, i&7, releases[:])
	}
}

func TestResolveFrameReferencesWithProviderFanOut(t *testing.T) {
	poolA := testFramePool(t, 1)
	indexA, _, err := poolA.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	poolB := testFramePool(t, 1)
	indexB, _, err := poolB.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	// Caller-owned namespace: layer 0 = poolA, layer 1 = poolB.
	provider := testFrameSurfaceProvider(func(id int) (*frame.Frame, error) {
		switch id {
		case indexA:
			return poolA.Frame(indexA)
		case 256 + indexB:
			return poolB.Frame(indexB)
		}
		return nil, ErrInvalidSurfaceReference
	})
	surfaces := []int{indexA, 256 + indexB}
	dst := make([]*frame.Frame, 2)
	count, err := ResolveFrameReferencesWithProvider(provider, surfaces, dst)
	if err != nil {
		t.Fatalf("ResolveFrameReferencesWithProvider: %v", err)
	}
	if count != 2 || dst[0] == nil || dst[1] == nil {
		t.Fatalf("count=%d dst=%v", count, dst)
	}
}

func TestResolveFrameReferencesWithProviderRejectsInvalid(t *testing.T) {
	provider := testFrameSurfaceProvider(func(id int) (*frame.Frame, error) {
		return nil, ErrInvalidSurfaceReference
	})
	dst := make([]*frame.Frame, 1)
	if _, err := ResolveFrameReferencesWithProvider(provider, []int{0}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if _, err := ResolveFrameReferencesWithProvider(provider, []int{-1}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if _, err := ResolveFrameReferencesWithProvider(nil, []int{0}, dst); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-provider err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

type testFrameSurfaceProvider func(int) (*frame.Frame, error)

func (fn testFrameSurfaceProvider) FrameSurface(id int) (*frame.Frame, error) { return fn(id) }

type testFrameSurfaceReleaser func([]int) error

func (fn testFrameSurfaceReleaser) ReleaseFrameSurfaces(ids []int) error { return fn(ids) }

func bindEmptyReferenceMVFrame(f *frame.Frame, err error) (tile.ReferenceMVFrame, error) {
	if err != nil {
		return tile.ReferenceMVFrame{}, err
	}
	miCols := uint32(f.Format.Width / 4)
	miRows := uint32(f.Format.Height / 4)
	mvf := tile.ReferenceMVFrame{}
	if err := mvf.Init(miRows, miCols, make([]tile.ReferenceMVEntry, miRows*miCols)); err != nil {
		return tile.ReferenceMVFrame{}, err
	}
	return mvf, nil
}

// TestRunEventWithContextAndExternalReferencesSurfaceFanOut exercises the SVC
// multi-pool entry point with a synthetic scenario: a key frame is decoded into
// poolA, refreshes all 8 reference slots with the caller-owned global ID, and
// then an inter frame for "spatial layer 1" picks the same slot but acquires
// its output from poolB. The decoder must not validate cross-pool surface
// indices against poolB; it must route through the provider.
//
// This is a minimal smoke test: we only assert that planning + reference
// resolution succeed at the decoder API boundary. Actual tile decoding is
// orthogonal and exercised by the testvector harness.
func TestRunEventWithContextAndExternalReferencesNilProvider(t *testing.T) {
	var state FrameWorkState
	var refs SurfaceReferences
	var event Event
	if _, err := state.RunEventWithContextAndExternalReferences(&refs, nil, parser.SequenceHeader{}, &event, 32, nil, nil, 1, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-provider err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

func TestFrameWorkStateFinishExternalPublishesPostFilterGlobalSurface(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	const publishedGlobalSurface = 256
	var released []int
	releaser := testFrameSurfaceReleaser(func(ids []int) error {
		released = append(released, ids...)
		for _, id := range ids {
			if id == begin.Surface {
				if err := pool.Release(id); err != nil {
					return err
				}
			}
		}
		return nil
	})
	var frameContexts SharedFrameContextStore
	state.sharedFrameContexts = &frameContexts
	state.sharedFrameContextGlobalSurface = func(local int) int { return local }

	var releases [parser.RefFrames]int
	completed, releaseCount, err := state.finishIfEventCompletesFrameWorkExternal(&refs, &pool, finalFrameEventPtr(0xff), releases[:], func(local int) int { return local }, releaser, publishedGlobalSurface, true)
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 1 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
	if len(released) != 1 || released[0] != begin.Surface {
		t.Fatalf("released=%v want coded surface %d", released, begin.Surface)
	}
	if _, err := pool.Frame(begin.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("coded surface err=%v want %v", err, frame.ErrInvalidSlot)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != publishedGlobalSurface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, publishedGlobalSurface)
	}
	if _, ok := frameContexts.load(publishedGlobalSurface); !ok {
		t.Fatalf("missing frame context for published global surface %d", publishedGlobalSurface)
	}
	if _, ok := frameContexts.load(begin.Surface); ok {
		t.Fatalf("stored frame context under discarded coded surface %d", begin.Surface)
	}
}

func TestFrameWorkStateFinishExternalPublishesTemporalMotionFrame(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, output, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey, OrderHint: 7},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	mvFrame, err := bindEmptyReferenceMVFrame(output, nil)
	if err != nil {
		t.Fatal(err)
	}
	state.currentMVFrame = mvFrame
	state.currentMVFrameValid = true

	var releases [parser.RefFrames]int
	event := finalFrameEvent(0x03)
	event.FrameHeader.FrameType = parser.FrameTypeInter
	event.FrameHeader.OrderHint = 7
	completed, releaseCount, err := state.finishIfEventCompletesFrameWorkExternal(&refs, &pool, &event, releases[:], func(local int) int { return local }, nil, -1, false)
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 0 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
	for slot := range 2 {
		ref, ok := refs.ReferenceSlot(slot)
		if !ok || ref != begin.Surface {
			t.Fatalf("slot %d ref=%d ok=%v want %d", slot, ref, ok, begin.Surface)
		}
		if !state.mvFrameStoreMeta[slot].valid {
			t.Fatalf("slot %d missing temporal motion metadata", slot)
		}
		if state.mvFrameStoreMeta[slot].orderHint != event.FrameHeader.OrderHint {
			t.Fatalf("slot %d order_hint=%d want %d", slot, state.mvFrameStoreMeta[slot].orderHint, event.FrameHeader.OrderHint)
		}
		if err := state.mvFrameStore[slot].Validate(); err != nil {
			t.Fatalf("slot %d temporal motion frame: %v", slot, err)
		}
	}
	for slot := 2; slot < parser.RefFrames; slot++ {
		if state.mvFrameStoreMeta[slot].valid {
			t.Fatalf("slot %d unexpectedly has temporal motion metadata", slot)
		}
	}
}

// newTestLayerPool wires a tiny LayerPool with two sub-pool slots and a
// factory that synthesizes per-format backing storage on demand. It is the
// production-side replacement for the testvector harness per-spatial-layer
// glue.
func newTestLayerPool(t *testing.T, capacity int) (*frame.LayerPool, *[][]byte) {
	t.Helper()
	subPools := make([]frame.Pool, capacity)
	subFormats := make([]frame.Format, capacity)
	subBound := make([]bool, capacity)
	keepAlive := &[][]byte{}
	factory := frame.LayerFactoryFunc(func(format frame.Format) (frame.Pool, error) {
		layout, err := frame.RequiredSize(format)
		if err != nil {
			return frame.Pool{}, err
		}
		const slots = 4
		backing := make([]byte, layout.Size*slots)
		frames := make([]frame.Frame, slots)
		free := make([]int, slots)
		used := make([]bool, slots)
		pool, err := frame.BindPool(backing, format, frames, free, used)
		if err != nil {
			return frame.Pool{}, err
		}
		*keepAlive = append(*keepAlive, backing)
		return pool, nil
	})
	pool, err := frame.BindLayerPool(subPools, subFormats, subBound, 256, factory)
	if err != nil {
		t.Fatal(err)
	}
	return &pool, keepAlive
}

func TestFrameLayerPoolFrameSurfaceProvider(t *testing.T) {
	pool, _ := newTestLayerPool(t, 2)
	layer0 := frame.Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := frame.Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	id0, want0, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	id1, want1, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}

	provider := NewFrameLayerPool(pool)
	got0, err := provider.FrameSurface(id0)
	if err != nil || got0 != want0 {
		t.Fatalf("FrameSurface(layer-0)=%p err=%v want %p", got0, err, want0)
	}
	got1, err := provider.FrameSurface(id1)
	if err != nil || got1 != want1 {
		t.Fatalf("FrameSurface(layer-1)=%p err=%v want %p", got1, err, want1)
	}

	// ResolveFrameReferencesWithProvider must route both IDs through the
	// LayerPool transparently.
	surfaces := []int{id0, id1}
	dst := make([]*frame.Frame, 2)
	count, err := ResolveFrameReferencesWithProvider(provider, surfaces, dst)
	if err != nil {
		t.Fatalf("ResolveFrameReferencesWithProvider: %v", err)
	}
	if count != 2 || dst[0] != want0 || dst[1] != want1 {
		t.Fatalf("count=%d dst=%v want %v,%v", count, dst, want0, want1)
	}
}

func TestFrameLayerPoolFrameSurfaceReleaser(t *testing.T) {
	pool, _ := newTestLayerPool(t, 2)
	layer0 := frame.Format{Width: 640, Height: 360, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layer1 := frame.Format{Width: 1280, Height: 720, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}

	id0, _, err := pool.Acquire(layer0)
	if err != nil {
		t.Fatal(err)
	}
	id1, _, err := pool.Acquire(layer1)
	if err != nil {
		t.Fatal(err)
	}
	releaser := NewFrameLayerPool(pool)
	if err := releaser.ReleaseFrameSurfaces([]int{id0, id1}); err != nil {
		t.Fatalf("ReleaseFrameSurfaces: %v", err)
	}
	if _, err := pool.Frame(id0); err == nil {
		t.Fatalf("layer-0 surface not released")
	}
	if _, err := pool.Frame(id1); err == nil {
		t.Fatalf("layer-1 surface not released")
	}
	// Empty/nil batches are no-ops.
	if err := releaser.ReleaseFrameSurfaces(nil); err != nil {
		t.Fatalf("nil release: %v", err)
	}
}

func TestFrameLayerPoolNilPoolRejected(t *testing.T) {
	provider := NewFrameLayerPool(nil)
	if _, err := provider.FrameSurface(0); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-pool FrameSurface err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if err := provider.ReleaseFrameSurfaces([]int{0}); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("nil-pool ReleaseFrameSurfaces err=%v want %v", err, ErrInvalidSurfaceReference)
	}
}

// copyMVOracle freezes the main-10f4a6ae prepare/publish implementation. It
// retains independent backing per slot and never calls the new pool helpers.
type copyMVOracle struct {
	Sequence            threading.FrameWorkSequenceContext
	sharedFrameContexts *SharedFrameContextStore
	mvFrameStore        [parser.RefFrames]tile.ReferenceMVFrame
	mvFrameStoreBacking [parser.RefFrames][]tile.ReferenceMVEntry
	mvFrameStoreMeta    [parser.RefFrames]mvFrameMeta
	currentMVFrame      tile.ReferenceMVFrame
	currentMVFrameBack  []tile.ReferenceMVEntry
	currentMVFrameValid bool
	temporalMVs         tile.TemporalMotionField
	temporalMVsBack     []tile.TemporalMotionEntry
	temporalMVsValid    bool
	mvReferences        [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
}

func (s *copyMVOracle) prepareTemporalMotionField(event Event) error {
	s.currentMVFrameValid = false
	s.temporalMVsValid = false
	if s.sharedFrameContexts != nil {
		// The shared/SVC path wires per-surface MV frames through the caller's
		// store and batch callback; leave the batch pointers nil here.
		return nil
	}
	if event.FrameHeader.ShowExistingFrame {
		return nil
	}

	miCols := mvFrameMIExtent(event.FrameSize.CodedWidth)
	miRows := mvFrameMIExtent(event.FrameSize.Height)
	need, err := tile.ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		// A zero-dimension frame carries no MV grid; nothing to project.
		return nil
	}
	if cap(s.currentMVFrameBack) < need {
		s.currentMVFrameBack = make([]tile.ReferenceMVEntry, need)
	}
	if err := s.currentMVFrame.Init(miRows, miCols, s.currentMVFrameBack[:need]); err != nil {
		return err
	}
	s.currentMVFrameValid = true

	if !event.TileInfo.UseRefFrameMVS {
		return nil
	}

	// Resolve each inter reference's saved MV_REF side data + order-hint
	// metadata from the per-slot store, keyed by RefFrameIdx exactly as libaom
	// indexes ref_frame_map. A reference whose slot was never published (no
	// prior decoded frame) leaves the temporal projection without that source.
	var refs [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame
	for ref := 0; ref < parser.InterRefsPerFrame; ref++ {
		slot := int(event.FrameSize.RefFrameIdx[ref])
		if slot < 0 || slot >= parser.RefFrames {
			return ErrInvalidSurfaceReference
		}
		meta := s.mvFrameStoreMeta[slot]
		if !meta.valid {
			// A reference slot never written by a prior frame has no MV grid to
			// project. Leave its entry's Frame nil; Setup/projectSetupReference
			// skip nil references per-reference, matching libaom's
			// av1_setup_motion_field guarding on ref_buf[ref] != NULL.
			continue
		}
		refs[ref] = tile.TemporalMotionReferenceFrame{
			Frame:         &s.mvFrameStore[slot],
			OrderHint:     meta.orderHint,
			RefOrderHints: meta.refOrderHints,
			IntraOnly:     meta.intraOnly,
		}
	}
	s.mvReferences = refs

	if cap(s.temporalMVsBack) < need {
		s.temporalMVsBack = make([]tile.TemporalMotionEntry, need)
	}
	if err := s.temporalMVs.Init(miRows, miCols, s.temporalMVsBack[:need]); err != nil {
		return err
	}
	if _, err := s.temporalMVs.Setup(tile.TemporalMotionSetupRequest{
		EnableOrderHint:  s.Sequence.EnableOrderHint,
		OrderHintBits:    s.Sequence.OrderHintBits,
		CurrentOrderHint: event.FrameHeader.OrderHint,
		References:       s.mvReferences,
	}); err != nil {
		return err
	}
	s.temporalMVsValid = true
	return nil
}

func (s *copyMVOracle) publishTemporalMotionFrame(event Event) {
	if s == nil || s.sharedFrameContexts != nil || !s.currentMVFrameValid {
		return
	}
	if err := s.currentMVFrame.Validate(); err != nil {
		return
	}
	meta := mvFrameMeta{
		valid:         true,
		orderHint:     event.FrameHeader.OrderHint,
		refOrderHints: event.ReferenceOrderHints,
		intraOnly: event.FrameHeader.FrameType == parser.FrameTypeKey ||
			event.FrameHeader.FrameType == parser.FrameTypeIntraOnly,
	}
	for i := range parser.RefFrames {
		if (event.FrameSize.RefreshFrameFlags & (1 << uint(i))) == 0 {
			continue
		}
		need := len(s.currentMVFrame.Entries)
		if cap(s.mvFrameStoreBacking[i]) < need {
			s.mvFrameStoreBacking[i] = make([]tile.ReferenceMVEntry, need)
		}
		dst := s.mvFrameStoreBacking[i][:need]
		copy(dst, s.currentMVFrame.Entries)
		s.mvFrameStore[i] = tile.ReferenceMVFrame{
			Rows:    s.currentMVFrame.Rows,
			Cols:    s.currentMVFrame.Cols,
			Stride:  s.currentMVFrame.Stride,
			Entries: dst,
		}
		s.mvFrameStoreMeta[i] = meta
	}
}

func mvPoolEvent(n int) Event {
	e := Event{
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeInter, OrderHint: uint8(8 + n*3)},
		FrameSize:   parser.FrameSize{CodedWidth: 32, Height: 24, RefreshFrameFlags: uint8(1 << uint(n%8))},
	}
	if n%11 == 0 {
		e.FrameHeader.FrameType = parser.FrameTypeKey
		e.FrameSize.RefreshFrameFlags = 255
	}
	if n%11 == 5 {
		e.FrameHeader.FrameType = parser.FrameTypeIntraOnly
		e.FrameSize.AllowIntrabc = n%2 == 0
	}
	if n%7 == 6 {
		e.FrameSize.RefreshFrameFlags = 0
	}
	if n%7 == 3 {
		e.FrameSize.RefreshFrameFlags = 0x95
	}
	if n%13 == 12 {
		e.FrameSize.CodedWidth = 16
		e.FrameSize.Height = 40
	}
	e.TileInfo.UseRefFrameMVS = e.FrameHeader.FrameType == parser.FrameTypeInter
	for i := range e.FrameSize.RefFrameIdx {
		e.FrameSize.RefFrameIdx[i] = uint8((n + i) % 8)
		e.ReferenceOrderHints[i] = e.FrameHeader.OrderHint - uint8(i+1)
	}
	return e
}

func fillMVPoolFrame(f *tile.ReferenceMVFrame, n int) {
	for i := range f.Entries {
		f.Entries[i] = tile.ReferenceMVEntry{Ref: tile.ReferenceFrameLast, MV: motion.Vector{Row: int16((i+n)%17 - 8), Col: int16((i*3+n)%19 - 9)}, Valid: i%3 != 0}
	}
}

func checkMVPoolRefs(t *testing.T, s *FrameWorkState) {
	t.Helper()
	var want [frameWorkMVGrids]uint8
	for _, handle := range s.mvSlotGrid {
		if handle != 0 {
			want[handle-1]++
		}
	}
	if s.currentMVGrid != 0 {
		want[s.currentMVGrid-1]++
	}
	if s.mvGridRefs != want {
		t.Fatalf("pool refs=%v want %v", s.mvGridRefs, want)
	}
}

func TestOwnedMVPoolMatchesFrozenCopy(t *testing.T) {
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	sequence := threading.FrameWorkSequenceContext{EnableOrderHint: true, EnableRefFrameMVS: true, OrderHintBits: 8}
	var old copyMVOracle
	for n := 0; n < 48; n++ {
		s.Sequence = sequence
		old.Sequence = sequence
		event := mvPoolEvent(n)
		// Preparing a new writable grid must not clear any completed slot.
		var saved [parser.RefFrames][]tile.ReferenceMVEntry
		for i := range saved {
			saved[i] = append([]tile.ReferenceMVEntry(nil), s.mvFrameStore[i].Entries...)
		}
		err := s.prepareTemporalMotionField(&event)
		wantErr := old.prepareTemporalMotionField(event)
		if err != wantErr {
			t.Fatalf("frame %d prepare=%v old=%v", n, err, wantErr)
		}
		for i := range saved {
			if !reflect.DeepEqual(saved[i], s.mvFrameStore[i].Entries) {
				t.Fatalf("frame %d mutated saved slot %d during prepare", n, i)
			}
		}
		// Production defers the projection to the tile loops (load_tmvs);
		// the whole-frame field is only set up by the refmvs oracle build.
		prepared := s.temporalMVsValid || s.temporalMVsDeferred
		if prepared != old.temporalMVsValid || (s.temporalMVsValid && !reflect.DeepEqual(s.temporalMVs, old.temporalMVs)) {
			t.Fatalf("frame %d temporal projection mismatch", n)
		}
		if event.TileInfo.UseRefFrameMVS {
			for i := range s.mvReferences {
				got, want := s.mvReferences[i], old.mvReferences[i]
				if (got.Frame == nil) != (want.Frame == nil) || got.OrderHint != want.OrderHint || got.RefOrderHints != want.RefOrderHints || got.IntraOnly != want.IntraOnly {
					t.Fatalf("frame %d reference %d metadata mismatch", n, i)
				}
			}
			if !s.temporalMVsValid {
				miCols := mvFrameMIExtent(event.FrameSize.CodedWidth)
				miRows := mvFrameMIExtent(event.FrameSize.Height)
				need, err := tile.ReferenceMVFrameEntries(miRows, miCols)
				if err != nil {
					t.Fatal(err)
				}
				if err := s.temporalMVs.Init(miRows, miCols, make([]tile.TemporalMotionEntry, need)); err != nil {
					t.Fatal(err)
				}
			}
			req := tile.TemporalMotionSetupRequest{EnableOrderHint: true, OrderHintBits: 8, CurrentOrderHint: event.FrameHeader.OrderHint, References: s.mvReferences}
			gotStats, gotErr := s.temporalMVs.Setup(req)
			req.References = old.mvReferences
			wantStats, wantErr := old.temporalMVs.Setup(req)
			if gotStats != wantStats || gotErr != wantErr {
				t.Fatalf("frame %d setup counters/errors differ: %v/%v versus %v/%v", n, gotStats, gotErr, wantStats, wantErr)
			}
		}
		if event.FrameHeader.FrameType == parser.FrameTypeInter {
			fillMVPoolFrame(&s.currentMVFrame, n)
			fillMVPoolFrame(&old.currentMVFrame, n)
		}
		s.publishTemporalMotionFrame(&event, -1)
		old.publishTemporalMotionFrame(event)
		if s.mvFrameStoreMeta != old.mvFrameStoreMeta {
			t.Fatalf("frame %d saved metadata differs", n)
		}
		for i, meta := range s.mvFrameStoreMeta {
			if !meta.valid || meta.intraOnly {
				continue
			}
			if !reflect.DeepEqual(s.mvFrameStore[i], old.mvFrameStore[i]) {
				t.Fatalf("frame %d slot %d differs from frozen copy", n, i)
			}
		}
		checkMVPoolRefs(t, &s)
		s.resetActive()
		// Match resetActive's legacy transient state without invoking candidate code.
		old.currentMVFrame = tile.ReferenceMVFrame{}
		old.currentMVFrameValid = false
		old.temporalMVs = tile.TemporalMotionField{}
		old.temporalMVsValid = false
		old.mvReferences = [parser.InterRefsPerFrame]tile.TemporalMotionReferenceFrame{}
		checkMVPoolRefs(t, &s)
	}
}

func TestOwnedMVPoolSharingAndGenericIsolation(t *testing.T) {
	for _, owned := range []bool{false, true} {
		var s FrameWorkState
		if owned {
			EnableOwnedTemporalMotionStorage(&s)
		}
		s.Sequence.EnableRefFrameMVS = true
		e := mvPoolEvent(1)
		e.TileInfo.UseRefFrameMVS = false
		e.FrameSize.RefreshFrameFlags = 255
		if err := s.prepareTemporalMotionField(&e); err != nil {
			t.Fatal(err)
		}
		fillMVPoolFrame(&s.currentMVFrame, 1)
		s.publishTemporalMotionFrame(&e, -1)
		current := &s.currentMVFrame.Entries[0]
		for i := range s.mvFrameStore {
			shared := &s.mvFrameStore[i].Entries[0] == current
			if shared != owned {
				t.Fatalf("owned=%v slot %d shared=%v", owned, i, shared)
			}
		}
		if !owned {
			saved := s.mvFrameStore[1].Entries[0]
			s.currentMVFrame.Entries[0].MV.Row++
			s.mvFrameStore[0].Entries[0].MV.Col++
			if s.mvFrameStore[1].Entries[0] != saved {
				t.Fatal("generic mutable slots alias")
			}
		}
		s.resetActive()
		if owned {
			for _, refs := range s.mvGridRefs {
				if refs != 0 && refs != 8 {
					t.Fatalf("shared completed refs=%d", refs)
				}
			}
		}
		e.FrameSize.RefreshFrameFlags = 0
		s.Sequence.EnableRefFrameMVS = true // resetActive cleared the sequence
		if err := s.prepareTemporalMotionField(&e); err != nil {
			t.Fatal(err)
		}
		if owned && &s.currentMVFrame.Entries[0] == current {
			t.Fatal("reused a referenced grid")
		}
		s.resetActive()
		s.resetReferenceState()
		checkMVPoolRefs(t, &s)
		for _, refs := range s.mvGridRefs {
			if refs != 0 {
				t.Fatal("sequence reset retained a slot")
			}
		}
	}
}

func TestOwnedMVPoolIntraMetadataAndFallback(t *testing.T) {
	for _, owned := range []bool{false, true} {
		for _, intrabc := range []bool{false, true} {
			var s FrameWorkState
			if owned {
				EnableOwnedTemporalMotionStorage(&s)
			}
			s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, EnableRefFrameMVS: true, OrderHintBits: 8}
			e := mvPoolEvent(0)
			e.FrameSize.AllowIntrabc = intrabc
			if err := s.prepareTemporalMotionField(&e); err != nil {
				t.Fatal(err)
			}
			omit := owned && !intrabc
			if s.currentMVFrameValid == omit || s.currentMVIntra != omit {
				t.Fatalf("owned=%v intrabc=%v grid=%v intra=%v", owned, intrabc, s.currentMVFrameValid, s.currentMVIntra)
			}
			if (s.motionFields().current == nil) != omit {
				t.Fatal("block-loop current-grid binding differs")
			}
			s.publishTemporalMotionFrame(&e, -1)
			s.resetActive()
			next := mvPoolEvent(1)
			s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, EnableRefFrameMVS: true, OrderHintBits: 8}
			if err := s.prepareTemporalMotionField(&next); err != nil {
				t.Fatal(err)
			}
			for _, ref := range s.mvReferences {
				if ref.Frame == nil || !ref.IntraOnly {
					t.Fatal("intra reference presence lost")
				}
			}
			s.resetActive()
			s.sharedFrameContexts = &SharedFrameContextStore{}
			if err := s.prepareTemporalMotionField(&next); err != nil {
				t.Fatal(err)
			}
			if s.currentMVFrameValid || s.currentMVGrid != 0 || s.currentMVIntra {
				t.Fatal("shared-store path acquired a private grid")
			}
			s.sharedFrameContexts = nil
			next.FrameHeader.ShowExistingFrame = true
			if err := s.prepareTemporalMotionField(&next); err != nil {
				t.Fatal(err)
			}
			if s.currentMVFrameValid || s.currentMVGrid != 0 || s.currentMVIntra {
				t.Fatal("show-existing acquired a grid")
			}
		}
	}
}

func TestOwnedMVPoolFailureAndReuse(t *testing.T) {
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	e := mvPoolEvent(1)
	e.FrameSize.RefFrameIdx[3] = parser.RefFrames
	if err := s.prepareTemporalMotionField(&e); !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("invalid ref error=%v", err)
	}
	if s.currentMVGrid != 0 || s.currentMVFrameValid {
		t.Fatal("failed prepare retained current grid")
	}
	checkMVPoolRefs(t, &s)
	e = mvPoolEvent(1)
	s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, EnableRefFrameMVS: true, OrderHintBits: 0}
	if err := s.prepareTemporalMotionField(&e); err == nil {
		t.Fatal("invalid order bits accepted")
	}
	if s.currentMVGrid != 0 || s.currentMVFrameValid {
		t.Fatal("failed projection retained current grid")
	}
	s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, EnableRefFrameMVS: true, OrderHintBits: 8}
	if err := s.prepareTemporalMotionField(&e); err != nil {
		t.Fatal(err)
	}
	checkMVPoolRefs(t, &s)
	s.resetActive()
	// All eight different slot grids can remain live while the ninth is written.
	e.TileInfo.UseRefFrameMVS = false
	for n := 0; n < 24; n++ {
		e.FrameSize.RefreshFrameFlags = 1 << uint(n%8)
		s.Sequence.EnableRefFrameMVS = true // resetActive cleared the sequence
		if err := s.prepareTemporalMotionField(&e); err != nil {
			t.Fatal(err)
		}
		s.publishTemporalMotionFrame(&e, -1)
		s.resetActive()
		checkMVPoolRefs(t, &s)
	}
	s.Reset()
	if !s.ownedTemporalMotion {
		t.Fatal("Reset lost private ownership mode")
	}
	checkMVPoolRefs(t, &s)
	for _, meta := range s.mvFrameStoreMeta {
		if meta.valid {
			t.Fatal("Reset kept saved metadata")
		}
	}
}

func TestOwnedMVPoolFinishAbortFailureLifetimes(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	event := Event{Kind: EventFrameHeader, FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey}, FrameSize: testFrameSize(16, 16)}
	event.FrameSize.AllowIntrabc = true // retain a writable grid for the lifecycle checks
	seq := testSequence()
	seq.EnableOrderHint, seq.EnableRefFrameMVS, seq.OrderHintBits = true, true, 8
	plan, _, err := s.Begin(&refs, &pool, seq, &event, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	handle := s.currentMVGrid
	if handle == 0 {
		t.Fatal("Begin did not acquire current grid")
	}
	if _, err = s.Finish(&refs, &pool, &Event{}, nil); err == nil {
		t.Fatal("invalid finish accepted")
	}
	if s.currentMVGrid != handle || !s.Active() {
		t.Fatal("failed finish released current grid")
	}
	if err = s.Abort(&pool); err != nil {
		t.Fatal(err)
	}
	if s.currentMVGrid != 0 || s.Active() {
		t.Fatal("Abort kept current grid")
	}
	plan, _, err = s.Begin(&refs, &pool, seq, &event, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	handle = s.currentMVGrid
	if err = pool.Release(plan.Surface); err != nil {
		t.Fatal(err)
	}
	if err = s.Abort(&pool); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("abort failure=%v", err)
	}
	if s.currentMVGrid != handle || !s.Active() {
		t.Fatal("failed abort released current grid")
	}
	s.Reset()
	checkMVPoolRefs(t, &s)
}

// TestOwnedMVPoolSequenceWithoutRefFrameMVs checks that a sequence without
// enable_ref_frame_mvs builds no current-frame grid for any frame type: no
// frame of it can project saved motion, so the block loop skips save_tmvs.
func TestOwnedMVPoolSequenceWithoutRefFrameMVs(t *testing.T) {
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	s.Sequence = threading.FrameWorkSequenceContext{EnableOrderHint: true, OrderHintBits: 8}
	for n := range 4 {
		e := mvPoolEvent(n)
		e.TileInfo.UseRefFrameMVS = false
		e.FrameSize.AllowIntrabc = n == 0
		e.FrameSize.RefreshFrameFlags = 1 << n
		if err := s.prepareTemporalMotionField(&e); err != nil {
			t.Fatal(err)
		}
		if s.currentMVFrameValid || s.currentMVGrid != 0 || s.motionFields().current != nil {
			t.Fatalf("frame %d (%v) bound an MV grid", n, e.FrameHeader.FrameType)
		}
		s.publishTemporalMotionFrame(&e, -1)
		if !s.mvFrameStoreMeta[n].valid || s.mvFrameStore[n].Entries != nil {
			t.Fatalf("frame %d: slot meta=%+v entries=%d", n, s.mvFrameStoreMeta[n], len(s.mvFrameStore[n].Entries))
		}
		s.resetActive()
	}
	checkMVPoolRefs(t, &s)
}

func TestOwnedMVPoolWarmAllocs(t *testing.T) {
	var s FrameWorkState
	EnableOwnedTemporalMotionStorage(&s)
	e := mvPoolEvent(1)
	e.TileInfo.UseRefFrameMVS = false
	e.FrameSize.RefreshFrameFlags = 255
	if err := s.PreallocTemporalMotionScratch(e.FrameSize); err != nil {
		t.Fatal(err)
	}
	var err error
	allocs := testing.AllocsPerRun(20, func() {
		for i := 0; i < 12; i++ {
			s.Sequence.EnableRefFrameMVS = true // resetActive clears the sequence
			err = s.prepareTemporalMotionField(&e)
			if err != nil {
				return
			}
			s.publishTemporalMotionFrame(&e, -1)
			s.resetActive()
		}
		s.Reset()
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("warm allocations=%v", allocs)
	}
}

func TestPlanTileWorkTileGroupEvent(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, err := PlanTileWork(&events[2], 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
	if spans[0] != (parser.TileSpan{Tile: 0, Row: 0, Col: 0, Offset: 0, Size: 1}) {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0].Tile != 0 || jobs[0].SBCols != 1 || jobs[0].SBRows != 1 || jobs[0].Offset != 0 || jobs[0].Size != 1 {
		t.Fatalf("job=%+v", jobs[0])
	}
	if batches[0] != (threading.Batch{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}) {
		t.Fatalf("batch=%+v", batches[0])
	}
}

func TestBeginFrameWorkFrameHeader(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	plan, output, err := BeginFrameWork(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || plan != (FrameWorkPlan{Surface: 0}) {
		t.Fatalf("plan=%+v output=%p", plan, output)
	}
	if pool.Available() != 0 {
		t.Fatalf("available=%d want 0", pool.Available())
	}
}

func TestBeginFrameWorkFrameEventPlansTiles(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, output, err := BeginFrameWork(&refs, &pool, testSequence(), &events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || plan.Surface != 0 || plan.ReferenceCount != 0 || plan.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v output=%p", plan, output)
	}
	if int(spans[0].Offset) != len(reducedStillFrameHeaderPayload()) || spans[0].Size != 1 {
		t.Fatalf("span=%+v", spans[0])
	}
	if jobs[0].Tile != 0 || batches[0].Count != 1 {
		t.Fatalf("job=%+v batch=%+v", jobs[0], batches[0])
	}
}

func TestBeginFrameWorkRejectsTilePlanBeforeAcquire(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, _, err := BeginFrameWork(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrame,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, parser.ErrInvalidTileGroup) {
		t.Fatalf("BeginFrameWork err=%v want %v", err, parser.ErrInvalidTileGroup)
	}
	if pool.Available() != 1 {
		t.Fatalf("tile failure consumed a slot, available=%d", pool.Available())
	}
}

func TestBeginFrameWorkRejectsFormatAfterTilePlanBeforeAcquire(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	events[1].FrameSize.CodedWidth = 80

	var refs SurfaceReferences
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, _, err = BeginFrameWork(&refs, &pool, testSequence(), &events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, frame.ErrInvalidFormat) {
		t.Fatalf("BeginFrameWork err=%v want %v", err, frame.ErrInvalidFormat)
	}
	if pool.Available() != 1 {
		t.Fatalf("format failure consumed a slot, available=%d", pool.Available())
	}
	if spans[0].Size != 1 || jobs[0].Size != 1 || batches[0].Count != 1 {
		t.Fatalf("tile plan was not written before acquire failure: span=%+v job=%+v batch=%+v", spans[0], jobs[0], batches[0])
	}
}

func TestBeginFrameWorkRejectsNonFrameEvent(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	_, _, err := BeginFrameWork(&refs, &pool, testSequence(), &Event{Kind: EventTileGroup}, 32, nil, 1, nil, nil, nil)
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("BeginFrameWork err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	if pool.Available() != 1 {
		t.Fatalf("invalid event consumed a slot, available=%d", pool.Available())
	}
}

func TestPlanFrameTileWorkTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, err := PlanFrameTileWork(&events[2], 5, 0, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.Surface != 5 || plan.ReferenceCount != 0 || plan.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
	if jobs[0].Tile != 0 || batches[0].Count != 1 {
		t.Fatalf("job=%+v batch=%+v", jobs[0], batches[0])
	}
}

func TestPlanFrameTileWorkRejectsInvalidState(t *testing.T) {
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, err := PlanFrameTileWork(&Event{Kind: EventFrameHeader}, 0, 0, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("wrong event err=%v want %v", err, ErrInvalidTileWork)
	}
	_, err = PlanFrameTileWork(&Event{Kind: EventTileGroup}, -1, 0, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("bad surface err=%v want %v", err, ErrInvalidTileWork)
	}
	_, err = PlanFrameTileWork(&Event{Kind: EventTileGroup}, 0, parser.InterRefsPerFrame+1, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("bad references err=%v want %v", err, ErrInvalidTileWork)
	}
}

func TestEventDropsFrameWork(t *testing.T) {
	tests := []struct {
		name  string
		event Event
		want  bool
	}{
		{
			name:  "new sequence",
			event: Event{Kind: EventSequenceHeader, NewCodedVideoSequence: true},
			want:  true,
		},
		{
			name:  "temporal delimiter",
			event: Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true},
			want:  true,
		},
		{
			name:  "show existing",
			event: Event{Kind: EventExistingFrame},
			want:  true,
		},
		{
			name:  "operating parameters",
			event: Event{Kind: EventSequenceHeader, OperatingParametersChanged: true},
			want:  false,
		},
		{
			name:  "tile group",
			event: Event{Kind: EventTileGroup},
			want:  false,
		},
	}
	for _, tt := range tests {
		if got := EventDropsFrameWork(&tt.event); got != tt.want {
			t.Fatalf("%s: EventDropsFrameWork=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestEventCompletesFrameWork(t *testing.T) {
	tests := []struct {
		name  string
		event Event
		want  bool
	}{
		{
			name:  "final frame",
			event: Event{Kind: EventFrame, TileGroup: parser.TileGroup{Final: true}},
			want:  true,
		},
		{
			name:  "final tile group",
			event: Event{Kind: EventTileGroup, TileGroup: parser.TileGroup{Final: true}},
			want:  true,
		},
		{
			name:  "non-final tile group",
			event: Event{Kind: EventTileGroup},
			want:  false,
		},
		{
			name:  "frame header",
			event: Event{Kind: EventFrameHeader, TileGroup: parser.TileGroup{Final: true}},
			want:  false,
		},
	}
	for _, tt := range tests {
		if got := EventCompletesFrameWork(&tt.event); got != tt.want {
			t.Fatalf("%s: EventCompletesFrameWork=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestEventOutputsFrame(t *testing.T) {
	tests := []struct {
		name  string
		event Event
		want  bool
	}{
		{
			name:  "final frame shown",
			event: Event{Kind: EventFrame, FrameHeader: parser.FrameHeaderPrefix{ShowFrame: true}, TileGroup: parser.TileGroup{Final: true}},
			want:  true,
		},
		{
			name:  "final frame not shown",
			event: Event{Kind: EventFrame, TileGroup: parser.TileGroup{Final: true}},
			want:  false,
		},
		{
			name:  "show existing",
			event: Event{Kind: EventExistingFrame},
			want:  true,
		},
		{
			name:  "frame header",
			event: Event{Kind: EventFrameHeader},
			want:  false,
		},
		{
			name:  "non-final tile group",
			event: Event{Kind: EventTileGroup},
			want:  false,
		},
	}
	for _, tt := range tests {
		if got := EventOutputsFrame(&tt.event); got != tt.want {
			t.Fatalf("%s: EventOutputsFrame=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestFrameWorkStateLifecycle(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || !state.Active() || state.Surface != plan.Surface || state.ReferenceCount != plan.ReferenceCount {
		t.Fatalf("state=%+v active=%v plan=%+v output=%p", state, state.Active(), plan, output)
	}
	wantSequence := threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader)
	if state.Sequence != wantSequence {
		t.Fatalf("state sequence=%+v want %+v", state.Sequence, wantSequence)
	}
	if pool.Available() != 0 {
		t.Fatalf("available after begin=%d want 0", pool.Available())
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	tilePlan, err := state.PlanTile(&events[2], 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if tilePlan.Surface != plan.Surface || tilePlan.ReferenceCount != plan.ReferenceCount ||
		tilePlan.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("tile plan=%+v begin plan=%+v", tilePlan, plan)
	}
	if pool.Available() != 0 {
		t.Fatalf("continuation acquired a surface, available=%d", pool.Available())
	}

	var releases [parser.RefFrames]int
	releaseCount, err := state.Finish(&refs, &pool, &events[2], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if releaseCount != 0 || state.Active() {
		t.Fatalf("releaseCount=%d active=%v", releaseCount, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != plan.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, plan.Surface)
	}
}

func TestFrameWorkStatePlanEventLifecycle(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || !state.Active() || step.Kind != FrameWorkStepBegin ||
		step.Begin.Surface != state.Surface || step.Begin.ReferenceCount != state.ReferenceCount {
		t.Fatalf("step=%+v output=%p state=%+v active=%v", step, output, state, state.Active())
	}

	step, output, err = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output != nil || step.Kind != FrameWorkStepTile ||
		step.Tile.Surface != state.Surface ||
		step.Tile.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) {
		t.Fatalf("step=%+v output=%p state=%+v", step, output, state)
	}

	var releases [parser.RefFrames]int
	completed, releaseCount, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, &events[2], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 0 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
}

func TestFrameWorkStatePlanEventFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin ||
		step.Begin.Tile != (TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}) ||
		!state.Active() {
		t.Fatalf("step=%+v output=%p active=%v", step, output, state.Active())
	}
	if int(spans[0].Offset) != len(reducedStillFrameHeaderPayload()) || jobs[0].Size != 1 || batches[0].Count != 1 {
		t.Fatalf("span=%+v job=%+v batch=%+v", spans[0], jobs[0], batches[0])
	}
}

func TestFrameWorkStatePlanEventDropsBoundary(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), &Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true}, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output != nil || step.Kind != FrameWorkStepDropped || !step.DroppedFrameWork || state.Active() || pool.Available() != 1 {
		t.Fatalf("step=%+v output=%p active=%v available=%d", step, output, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(begin.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("dropped frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStatePlanEventDropsBeforeBoundaryBegin(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), &Event{
		Kind:                  EventFrameHeader,
		NewCodedVideoSequence: true,
		FrameHeader:           parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:             testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin || !step.DroppedFrameWork || !state.Active() || pool.Available() != 1 {
		t.Fatalf("step=%+v output=%p active=%v available=%d", step, output, state.Active(), pool.Available())
	}
}

func TestFrameWorkStatePlanEventNewCodedVideoSequenceReleasesReferencesBeforeBegin(t *testing.T) {
	pool := testFramePool(t, 1)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, reference, releases[:]); err != nil {
		t.Fatal(err)
	}
	var state FrameWorkState

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), &Event{
		Kind:                  EventFrameHeader,
		NewCodedVideoSequence: true,
		FrameHeader:           parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:             testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if output == nil ||
		step.Kind != FrameWorkStepBegin ||
		step.ReleaseCount != 1 ||
		!state.Active() ||
		pool.Available() != 0 ||
		refs.Holds(reference) {
		t.Fatalf("step=%+v output=%p active=%v available=%d holdsReference=%v", step, output, state.Active(), pool.Available(), refs.Holds(reference))
	}
}

func TestFrameWorkStatePlanEventNewCodedVideoSequenceReleasesReferencesOnIgnoredEvent(t *testing.T) {
	pool := testFramePool(t, 1)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, reference, releases[:]); err != nil {
		t.Fatal(err)
	}
	var state FrameWorkState
	event := Event{Kind: EventSequenceHeader, NewCodedVideoSequence: true}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if output != nil ||
		step.Kind != FrameWorkStepIgnored ||
		step.ReleaseCount != 1 ||
		state.Active() ||
		pool.Available() != 1 ||
		refs.Holds(reference) {
		t.Fatalf("step=%+v output=%p active=%v available=%d holdsReference=%v", step, output, state.Active(), pool.Available(), refs.Holds(reference))
	}
	run, err := state.RunStep(&refs, &pool, &event, step, nil, nil, nil, releases[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if run != (FrameWorkStepResult{ReleaseCount: 1}) {
		t.Fatalf("run=%+v", run)
	}
}

func TestFrameWorkStatePlanEventShowExisting(t *testing.T) {
	pool := testFramePool(t, 2)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}
	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), showExistingWorkEventPtr(0, parser.FrameTypeInter), 32, nil, 1, nil, nil, nil, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if output != nil || step.Kind != FrameWorkStepShowExisting || !step.DroppedFrameWork ||
		step.ShowExisting != (ShowExistingFrameWorkPlan{Surface: reference, DroppedFrameWork: true}) ||
		state.Active() {
		t.Fatalf("step=%+v output=%p active=%v", step, output, state.Active())
	}
}

func TestFrameWorkStatePlanEventIgnoresNonWorkEvent(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	step, output, err := state.PlanEvent(&refs, &pool, testSequence(), &Event{Kind: EventMetadata}, 32, nil, 1, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if step.Kind != FrameWorkStepIgnored || output != nil || state.Active() || pool.Available() != 1 {
		t.Fatalf("step=%+v output=%p active=%v available=%d", step, output, state.Active(), pool.Available())
	}
}

func TestFrameWorkStatePlanEventRejectsActiveBegin(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = state.PlanEvent(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil, nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("PlanEvent err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 1 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestExecuteTileWorkUsesPlanRanges(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs := [2]tile.Job{
		{Tile: 3, SBCols: 1, SBRows: 1},
		{Tile: 99, SBCols: 1, SBRows: 1},
	}
	batches := [2]threading.Batch{
		{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 3, LastTile: 3, Units: 1},
		{Worker: 0, FirstJob: 1, Count: 1, FirstTile: 99, LastTile: 99, Units: 1},
	}
	plan := TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}
	var seen [2]uint16

	err = ExecuteTileWork(plan, workerPool, jobs[:], batches[:], func(batch threading.Batch, batchJobs []tile.Job) error {
		for i := range batchJobs {
			seen[int(batch.FirstJob)+i] = batchJobs[i].Tile
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if seen != ([2]uint16{3, 0}) {
		t.Fatalf("seen=%v", seen)
	}
}

func TestExecuteFrameWorkStepExecutesBeginAndTile(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind:  FrameWorkStepBegin,
		Begin: FrameWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	var seen uint16
	executed, err := executeTestFrameWorkStep(step, workerPool, jobs[:], batches[:], func(_ threading.Batch, batchJobs []tile.Job) error {
		for i := range batchJobs {
			seen += batchJobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || seen != 3 {
		t.Fatalf("begin executed=%v seen=%d", executed, seen)
	}

	step = FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	seen = 0
	executed, err = executeTestFrameWorkStep(step, workerPool, jobs[:], batches[:], func(_ threading.Batch, batchJobs []tile.Job) error {
		for i := range batchJobs {
			seen += batchJobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || seen != 3 {
		t.Fatalf("tile executed=%v seen=%d", executed, seen)
	}
}

func TestExecuteFrameWorkStepNoopSteps(t *testing.T) {
	for _, step := range []FrameWorkStep{
		{Kind: FrameWorkStepIgnored},
		{Kind: FrameWorkStepDropped},
		{Kind: FrameWorkStepShowExisting},
		{Kind: FrameWorkStepBegin},
	} {
		executed, err := executeTestFrameWorkStep(step, nil, nil, nil, nil)
		if err != nil || executed {
			t.Fatalf("step=%+v executed=%v err=%v", step, executed, err)
		}
	}
}

func TestExecuteFrameWorkStepWithContext(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	output := &frame.Frame{}
	reference := &frame.Frame{}
	references := [parser.InterRefsPerFrame]*frame.Frame{reference, &frame.Frame{}}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			Surface:        3,
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}

	calls := 0
	executed, err := executeTestFrameWorkStepWithContext(step, workerPool, output, references[:], jobs[:], batches[:], func(ctx FrameWorkBatch) error {
		calls++
		if ctx.Step != step {
			t.Fatalf("step=%+v want %+v", ctx.Step, step)
		}
		if ctx.Output != output {
			t.Fatalf("output=%p want %p", ctx.Output, output)
		}
		if ctx.Payload != nil {
			t.Fatalf("payload=%v want nil", ctx.Payload)
		}
		if len(ctx.References) != 1 || ctx.References[0] != reference {
			t.Fatalf("references=%v want %p", ctx.References, reference)
		}
		if ctx.Batch != batches[0] {
			t.Fatalf("batch=%+v want %+v", ctx.Batch, batches[0])
		}
		if len(ctx.Jobs) != 2 || ctx.Jobs[0].Tile != 0 || ctx.Jobs[1].Tile != 1 {
			t.Fatalf("jobs=%+v", ctx.Jobs)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed || calls != 1 {
		t.Fatalf("executed=%v calls=%d", executed, calls)
	}
}

func TestExecuteFrameWorkStepWithPayload(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	output := &frame.Frame{}
	payload := []byte{0xaa, 0xbb, 0xcc}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	executed, err := executeTestFrameWorkStepWithPayload(step, workerPool, output, nil, payload, jobs[:], batches[:], func(ctx FrameWorkBatch) error {
		if ctx.Output != output {
			t.Fatalf("output=%p want %p", ctx.Output, output)
		}
		if len(ctx.Payload) != len(payload) || ctx.Payload[0] != payload[0] || ctx.Payload[2] != payload[2] {
			t.Fatalf("payload=%v want %v", ctx.Payload, payload)
		}
		if len(ctx.Jobs) != 2 {
			t.Fatalf("jobs=%+v", ctx.Jobs)
		}
		data, err := ctx.JobPayload(1)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 2 || data[0] != 0xbb || data[1] != 0xcc {
			t.Fatalf("job payload=%v", data)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !executed {
		t.Fatal("not executed")
	}
}

func TestFrameWorkStateRunStepWithPayloadContextCarriesCDFUpdateMode(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	globalMotion := testFrameWorkGlobalMotion()
	filmGrain := testFrameWorkFilmGrain()
	event := Event{
		Kind:           EventTileGroup,
		SequenceHeader: testFrameWorkSequenceHeader(),
		FrameHeader: parser.FrameHeaderPrefix{
			DisableCDFUpdate: true,
			FrameType:        parser.FrameTypeKey,
		},
		FrameSize:    testFrameSize(128, 64),
		TileInfo:     testFrameWorkTileInfo(),
		Quantization: parser.QuantizationParams{BaseQIdx: 73},
		Segmentation: parser.SegmentationParams{
			QIndex:      [parser.MaxSegments]uint8{73},
			AllLossless: false,
		},
		Delta: parser.DeltaParams{
			DeltaQPresent:  true,
			DeltaQResLog2:  1,
			DeltaLFPresent: true,
			DeltaLFResLog2: 1,
		},
		LoopFilter:  parser.LoopFilterParams{LevelY: [2]uint8{4, 5}, LevelU: 6, LevelV: 7},
		CDEF:        parser.CDEFParams{Damping: 3, StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{8}},
		Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}, UnitSizeY: 64},
		TransformRef: parser.TransformReferenceParams{
			TransformMode: parser.TransformModeSwitchable,
			ReferenceMode: parser.ReferenceModeSelect,
		},
		SkipMode:            parser.SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{1, 2}},
		FrameMode:           parser.FrameModeParams{AllowWarpedMotion: true, ReducedTxSet: true},
		GlobalMotion:        globalMotion,
		FilmGrain:           filmGrain,
		ReferenceOrderHints: [parser.InterRefsPerFrame]uint8{4, 5, 6, 7, 8, 9, 10},
	}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	payload := []byte{0x00, 0x00, 0x00}
	state := FrameWorkState{active: true}
	cdefContext := FrameWorkBatch{FrameWorkFrameContext: frameWorkFrameContext(&event, threading.FrameWorkSequenceContextFromHeader(event.SequenceHeader))}
	_, _, cdefMapLen, err := cdefContext.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := cdefContext.BindCDEFIndexMap(make([]uint8, cdefMapLen), make([]bool, cdefMapLen))
	if err != nil {
		t.Fatal(err)
	}
	cdefMap.Read[0] = true
	if err := state.SetCDEFIndexMap(cdefMap); err != nil {
		t.Fatal(err)
	}
	_, _, lfMapLen, err := cdefContext.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	lfMap, err := cdefContext.BindLoopFilterMap(make([]threading.FrameWorkLoopFilterBlockRecord, lfMapLen))
	if err != nil {
		t.Fatal(err)
	}
	lfMap.Records[0].Valid = true
	if err := state.SetLoopFilterMap(lfMap); err != nil {
		t.Fatal(err)
	}

	result, err := state.RunStepWithPayloadContext(nil, nil, &event, step, workerPool, nil, nil, payload, jobs[:], batches[:], nil, func(ctx FrameWorkBatch) error {
		if !ctx.DisableCDFUpdate {
			t.Fatal("DisableCDFUpdate not propagated")
		}
		if ctx.CDEFIndexMap == nil {
			t.Fatal("CDEFIndexMap not propagated")
		}
		if ctx.CDEFIndexMap.Stride != cdefMap.Stride || ctx.CDEFIndexMap.Rows != cdefMap.Rows ||
			len(ctx.CDEFIndexMap.Index) != cdefMapLen || len(ctx.CDEFIndexMap.Read) != cdefMapLen {
			t.Fatalf("CDEFIndexMap=%+v len=%d/%d want stride=%d rows=%d len=%d", ctx.CDEFIndexMap, len(ctx.CDEFIndexMap.Index), len(ctx.CDEFIndexMap.Read), cdefMap.Stride, cdefMap.Rows, cdefMapLen)
		}
		if ctx.CDEFIndexMap.Read[0] || ctx.CDEFIndexMap.Index[0] != 0 {
			t.Fatalf("CDEFIndexMap was not reset before propagation: read=%v index=%d", ctx.CDEFIndexMap.Read[0], ctx.CDEFIndexMap.Index[0])
		}
		ctx.CDEFIndexMap.Read[0] = true
		if ctx.LoopFilterMap == nil {
			t.Fatal("LoopFilterMap not propagated")
		}
		if ctx.LoopFilterMap.Stride != lfMap.Stride || ctx.LoopFilterMap.Rows != lfMap.Rows ||
			len(ctx.LoopFilterMap.Records) != lfMapLen {
			t.Fatalf("LoopFilterMap=%+v len=%d want stride=%d rows=%d len=%d", ctx.LoopFilterMap, len(ctx.LoopFilterMap.Records), lfMap.Stride, lfMap.Rows, lfMapLen)
		}
		if ctx.LoopFilterMap.Records[0].Valid {
			t.Fatal("LoopFilterMap was not reset before propagation")
		}
		ctx.LoopFilterMap.Records[0].Valid = true
		if ctx.Quantization.BaseQIdx != 73 {
			t.Fatalf("BaseQIdx=%d want 73", ctx.Quantization.BaseQIdx)
		}
		if ctx.FrameHeader != event.FrameHeader {
			t.Fatalf("FrameHeader=%+v want %+v", ctx.FrameHeader, event.FrameHeader)
		}
		if ctx.Sequence != testFrameWorkSequenceContext() {
			t.Fatalf("Sequence=%+v want %+v", ctx.Sequence, testFrameWorkSequenceContext())
		}
		if ctx.FrameSize != event.FrameSize {
			t.Fatalf("FrameSize=%+v want %+v", ctx.FrameSize, event.FrameSize)
		}
		if ctx.TileInfo != event.TileInfo {
			t.Fatalf("TileInfo=%+v want %+v", ctx.TileInfo, event.TileInfo)
		}
		if ctx.Delta != event.Delta {
			t.Fatalf("Delta=%+v want %+v", ctx.Delta, event.Delta)
		}
		if ctx.Segmentation != event.Segmentation {
			t.Fatalf("Segmentation=%+v want %+v", ctx.Segmentation, event.Segmentation)
		}
		if ctx.LoopFilter != event.LoopFilter {
			t.Fatalf("LoopFilter=%+v want %+v", ctx.LoopFilter, event.LoopFilter)
		}
		if ctx.CDEF != event.CDEF {
			t.Fatalf("CDEF=%+v want %+v", ctx.CDEF, event.CDEF)
		}
		if ctx.Restoration != event.Restoration {
			t.Fatalf("Restoration=%+v want %+v", ctx.Restoration, event.Restoration)
		}
		if ctx.TransformRef != event.TransformRef {
			t.Fatalf("TransformRef=%+v want %+v", ctx.TransformRef, event.TransformRef)
		}
		if ctx.SkipMode != event.SkipMode {
			t.Fatalf("SkipMode=%+v want %+v", ctx.SkipMode, event.SkipMode)
		}
		if ctx.FrameMode != event.FrameMode {
			t.Fatalf("FrameMode=%+v want %+v", ctx.FrameMode, event.FrameMode)
		}
		if ctx.GlobalMotion != event.GlobalMotion {
			t.Fatalf("GlobalMotion=%+v want %+v", ctx.GlobalMotion, event.GlobalMotion)
		}
		if ctx.FilmGrain != event.FilmGrain {
			t.Fatalf("FilmGrain=%+v want %+v", ctx.FilmGrain, event.FilmGrain)
		}
		if ctx.ReferenceOrderHints != event.ReferenceOrderHints {
			t.Fatalf("ReferenceOrderHints=%v want %v", ctx.ReferenceOrderHints, event.ReferenceOrderHints)
		}
		r, err := ctx.JobEntropyReader(1)
		if err != nil {
			t.Fatal(err)
		}
		if r.AllowCDFUpdate() {
			t.Fatal("CDF update enabled")
		}
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatal(err)
		}
		if bit != 0 {
			t.Fatalf("bit=%d want 0", bit)
		}
		var state tile.DecodeState
		if err := ctx.JobDecodeState(1, &state); err != nil {
			t.Fatal(err)
		}
		if state.CurrentBaseQIdx != 73 {
			t.Fatalf("CurrentBaseQIdx=%d want 73", state.CurrentBaseQIdx)
		}
		if state.Reader.AllowCDFUpdate() {
			t.Fatal("decode state CDF update enabled")
		}
		var qCDF entropy.CDF
		if err := qCDF.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		var lfCDF entropy.CDF
		if err := lfCDF.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		err = state.ReadBlockDeltas(ctx.Delta, tile.BlockDeltaContext{SBSizeMIB: 16}, tile.DeltaCDFs{Q: &qCDF, LF: &lfCDF})
		if err != nil {
			t.Fatal(err)
		}
		if state.CurrentBaseQIdx != 73 || state.DeltaLFFromBase != 0 {
			t.Fatalf("delta state q=%d lf=%d", state.CurrentBaseQIdx, state.DeltaLFFromBase)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.ExecutedTileWork {
		t.Fatal("not executed")
	}
	if !state.cdefIndexMap.Read[0] {
		t.Fatal("CDEFIndexMap update did not alias frame state")
	}
	if !state.loopFilterMap.Records[0].Valid {
		t.Fatal("LoopFilterMap update did not alias frame state")
	}
}

func TestFrameWorkStateCarriesTileResidualCDFsThroughReferenceSlots(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs := []tile.Job{{Tile: 0, Offset: 0, Size: 1, UpdatesFrameContext: true}}
	batches := []threading.Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0, Units: 1}}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 1, JobCount: 1, BatchCount: 1}},
	}

	key := Event{
		Kind: EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:       parser.FrameTypeKey,
			PrimaryRefFrame: parser.PrimaryRefNone,
		},
		FrameSize:    testFrameSize(16, 16),
		TileInfo:     parser.TileInfo{RefreshContext: true, ContextUpdateTileID: 0},
		Quantization: parser.QuantizationParams{BaseQIdx: 64},
	}
	plan, _, err := state.Begin(&refs, &pool, testSequence(), &key, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	step.Tile.Surface = plan.Surface

	var retainedDeltaQ []uint16
	finalKey := key
	finalKey.Kind = EventTileGroup
	finalKey.Unit.Payload = []byte{0x00}
	finalKey.FrameSize.RefreshFrameFlags = 0xff
	finalKey.TileGroup = parser.TileGroup{Final: true}
	_, err = state.RunStepWithPayloadContext(&refs, &pool, &finalKey, step, workerPool, nil, nil, finalKey.Unit.Payload, jobs, batches, releases[:], func(ctx FrameWorkBatch) error {
		var storage threading.FrameWorkTileResidualCDFStorage
		if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
			return err
		}
		if err := storage.DeltaQ.Update(2); err != nil {
			return err
		}
		retainedDeltaQ = append(retainedDeltaQ[:0], storage.DeltaQ.Values()...)
		retainedDeltaQ[len(retainedDeltaQ)-1] = 0
		var decodeState tile.DecodeState
		if err := ctx.JobDecodeState(0, &decodeState); err != nil {
			return err
		}
		return ctx.RetainTileResidualCDFStorage(0, &decodeState, &storage)
	})
	if err != nil {
		t.Fatal(err)
	}

	inter := Event{
		Kind: EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{
			FrameType:       parser.FrameTypeInter,
			PrimaryRefFrame: 0,
		},
		FrameSize:    testFrameSize(16, 16),
		TileInfo:     parser.TileInfo{RefreshContext: true, ContextUpdateTileID: 0},
		Quantization: parser.QuantizationParams{BaseQIdx: 64},
	}
	for i := range parser.InterRefsPerFrame {
		inter.FrameSize.RefFrameIdx[i] = 0
	}
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	plan, _, err = state.Begin(&refs, &pool, testSequence(), &inter, 32, referenceSurfaces[:], 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan.ReferenceCount != 0 {
		referenceCount := int(plan.ReferenceCount)
		count, err := ResolveFrameReferences(&pool, referenceSurfaces[:referenceCount], referenceFrames[:])
		if err != nil {
			t.Fatal(err)
		}
		if count != referenceCount {
			t.Fatalf("reference count=%d want %d", count, plan.ReferenceCount)
		}
	}
	step.Tile.Surface = plan.Surface
	step.Tile.ReferenceCount = plan.ReferenceCount
	finalInter := inter
	finalInter.Kind = EventTileGroup
	finalInter.Unit.Payload = []byte{0x00}
	finalInter.FrameSize.RefreshFrameFlags = 0x01
	finalInter.TileGroup = parser.TileGroup{Final: true}
	_, err = state.RunStepWithPayloadContext(&refs, &pool, &finalInter, step, workerPool, nil, referenceFrames[:int(plan.ReferenceCount)], finalInter.Unit.Payload, jobs, batches, releases[:], func(ctx FrameWorkBatch) error {
		var storage threading.FrameWorkTileResidualCDFStorage
		if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
			return err
		}
		if !testUint16sEqual(storage.DeltaQ.Values(), retainedDeltaQ) {
			return fmt.Errorf("delta q=%v want retained %v", storage.DeltaQ.Values(), retainedDeltaQ)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestExecuteFrameWorkStepWithPayloadRejectsInvalidPayloadRange(t *testing.T) {
	jobs, batches, batchCount := testExecutionWork(t)
	jobs[1].Offset = 2
	jobs[1].Size = 2
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	_, err := executeTestFrameWorkStepWithPayload(step, nil, nil, nil, []byte{0xaa, 0xbb, 0xcc}, jobs[:], batches[:], func(FrameWorkBatch) error {
		t.Fatal("callback should not run")
		return nil
	})
	if !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("ExecuteFrameWorkStepWithPayload err=%v want %v", err, tile.ErrInvalidPlan)
	}
}

func TestExecuteFrameWorkStepWithContextNoopSteps(t *testing.T) {
	for _, step := range []FrameWorkStep{
		{Kind: FrameWorkStepIgnored},
		{Kind: FrameWorkStepDropped},
		{Kind: FrameWorkStepShowExisting},
		{Kind: FrameWorkStepBegin},
	} {
		executed, err := executeTestFrameWorkStepWithContext(step, nil, nil, nil, nil, nil, nil)
		if err != nil || executed {
			t.Fatalf("step=%+v executed=%v err=%v", step, executed, err)
		}
	}
}

func TestExecuteFrameWorkStepWithContextRejectsNilCallback(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	_, err = executeTestFrameWorkStepWithContext(step, workerPool, nil, nil, jobs[:], batches[:], nil)
	if !errors.Is(err, threading.ErrInvalidCallback) {
		t.Fatalf("ExecuteFrameWorkStepWithContext err=%v want %v", err, threading.ErrInvalidCallback)
	}
}

func TestExecuteFrameWorkStepWithContextRejectsShortReferences(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}
	_, err = executeTestFrameWorkStepWithContext(step, workerPool, nil, nil, jobs[:], batches[:], func(FrameWorkBatch) error {
		return nil
	})
	if !errors.Is(err, ErrSurfaceReferenceBufferTooSmall) {
		t.Fatalf("ExecuteFrameWorkStepWithContext err=%v want %v", err, ErrSurfaceReferenceBufferTooSmall)
	}
}

func TestExecuteTileWorkRejectsInvalidPlan(t *testing.T) {
	jobs, batches, batchCount := testExecutionWork(t)
	tests := []TileWorkPlan{
		{SpanCount: -1},
		{SpanCount: 1, JobCount: 0},
		{SpanCount: 1, JobCount: 1, BatchCount: 0},
		{SpanCount: 1, JobCount: 1, BatchCount: 2},
		{SpanCount: 3, JobCount: 3, BatchCount: batchCount},
		{SpanCount: 2, JobCount: 2, BatchCount: batchCount + 1},
	}
	for _, plan := range tests {
		err := ExecuteTileWork(plan, nil, jobs[:], batches[:batchCount], nil)
		if !errors.Is(err, ErrInvalidTileWork) {
			t.Fatalf("plan=%+v err=%v want %v", plan, err, ErrInvalidTileWork)
		}
	}
}

func TestExecuteFrameWorkStepRejectsInvalidKind(t *testing.T) {
	_, err := executeTestFrameWorkStep(FrameWorkStep{Kind: FrameWorkStepKind(99)}, nil, nil, nil, nil)
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("ExecuteFrameWorkStep err=%v want %v", err, ErrInvalidTileWork)
	}
}

func TestExecuteTileWorkPropagatesCallbackError(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	want := errors.New("tile callback")
	err = ExecuteTileWork(TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}, workerPool, jobs[:], batches[:], func(threading.Batch, []tile.Job) error {
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("ExecuteTileWork err=%v want %v", err, want)
	}
}

func TestFrameWorkStateRunStepFrameOBUExecutesThenFinishes(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin {
		t.Fatalf("step=%+v output=%p", step, output)
	}

	var ranWhileActive bool
	var releases [parser.RefFrames]int
	result, err := state.RunStep(&refs, &pool, &events[1], step, workerPool, jobs[:], batches[:], releases[:], func(batch threading.Batch, batchJobs []tile.Job) error {
		ranWhileActive = state.Active() && batch.Count == 1 && len(batchJobs) == 1
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || !ranWhileActive || state.Active() {
		t.Fatalf("result=%+v ranWhileActive=%v active=%v", result, ranWhileActive, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, step.Begin.Surface)
	}
}

func TestFrameWorkStateRunStepWithContextFrameOBUExecutesThenFinishes(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || step.Kind != FrameWorkStepBegin {
		t.Fatalf("step=%+v output=%p", step, output)
	}

	var ranWhileActive bool
	var releases [parser.RefFrames]int
	result, err := state.RunStepWithContext(&refs, &pool, &events[1], step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(ctx FrameWorkBatch) error {
		ranWhileActive = state.Active() &&
			ctx.Step == step &&
			ctx.Output == output &&
			len(ctx.References) == 0 &&
			ctx.Batch.Count == 1 &&
			len(ctx.Jobs) == 1
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || !ranWhileActive || state.Active() {
		t.Fatalf("result=%+v ranWhileActive=%v active=%v", result, ranWhileActive, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, step.Begin.Surface)
	}
}

func TestFrameWorkStateRunStepTileGroupExecutesThenFinishes(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	begin, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	result, err := state.RunStep(&refs, &pool, &events[1], begin, workerPool, jobs[:], batches[:], nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("begin result=%+v active=%v", result, state.Active())
	}

	step, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	var releases [parser.RefFrames]int
	result, err = state.RunStep(&refs, &pool, &events[2], step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("tile result=%+v active=%v", result, state.Active())
	}
}

func TestFrameWorkStateRunStepKeepsActiveOnCallbackError(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	begin, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	step, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}

	want := errors.New("tile decode")
	var releases [parser.RefFrames]int
	_, err = state.RunStep(&refs, &pool, &events[2], step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("RunStep err=%v want %v", err, want)
	}
	if !state.Active() || state.Surface != begin.Begin.Surface {
		t.Fatalf("state=%+v active=%v begin=%+v", state, state.Active(), begin)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
}

func TestFrameWorkStateRunStepWithContextKeepsActiveOnCallbackError(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	begin, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}
	step, _, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[2], 32, nil, 1, spans[:], jobs[:], batches[:], nil)
	if err != nil {
		t.Fatal(err)
	}

	want := errors.New("tile decode")
	var releases [parser.RefFrames]int
	_, err = state.RunStepWithContext(&refs, &pool, &events[2], step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(ctx FrameWorkBatch) error {
		if ctx.Output != output {
			t.Fatalf("output=%p want %p", ctx.Output, output)
		}
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("RunStepWithContext err=%v want %v", err, want)
	}
	if !state.Active() || state.Surface != begin.Begin.Surface {
		t.Fatalf("state=%+v active=%v begin=%+v", state, state.Active(), begin)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
}

func TestFrameWorkStateRunEventWithContextFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	wantSequence := threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader)
	var seenOutput *frame.Frame
	result, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		seenOutput = ctx.Output
		if ctx.Step.Kind != FrameWorkStepBegin ||
			len(ctx.References) != 0 ||
			len(ctx.Jobs) != 1 ||
			len(ctx.Payload) != len(events[1].Unit.Payload) ||
			ctx.Payload[int(ctx.Jobs[0].Offset)] != 0xaa ||
			ctx.Sequence != wantSequence ||
			ctx.FrameHeader != events[1].FrameHeader ||
			ctx.FrameSize != events[1].FrameSize ||
			ctx.TileInfo != events[1].TileInfo ||
			ctx.Segmentation != events[1].Segmentation ||
			ctx.LoopFilter != events[1].LoopFilter ||
			ctx.CDEF != events[1].CDEF ||
			ctx.Restoration != events[1].Restoration ||
			ctx.TransformRef != events[1].TransformRef ||
			ctx.SkipMode != events[1].SkipMode ||
			ctx.FrameMode != events[1].FrameMode ||
			ctx.GlobalMotion != events[1].GlobalMotion ||
			ctx.FilmGrain != events[1].FilmGrain {
			t.Fatalf("ctx=%+v", ctx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Output != seenOutput ||
		result.ReferenceCount != 0 ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v seen=%p active=%v", result, seenOutput, state.Active())
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkStateRunEventWithContextRunnerFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	runner := frameWorkContextRunner{
		wantKind:     FrameWorkStepBegin,
		wantSequence: threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader),
		wantEvent:    events[1],
		wantPayload:  0xaa,
	}
	result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
	if err != nil {
		t.Fatal(err)
	}
	if runner.err != nil {
		t.Fatal(runner.err)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Output != runner.output ||
		result.ReferenceCount != 0 ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v seen=%p active=%v", result, runner.output, state.Active())
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}

	result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	if stats := runner.Workers[0].Stats; stats.Loop.Blocks == 0 || stats.CoefficientBlocks == 0 || stats.TXBs == 0 {
		t.Fatalf("residual runner stats=%+v", stats)
	}
	if slot, ok := refs.ReferenceSlot(0); !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerSideDataPostFilter(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	events[1].LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{4}}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}
	side := testLoopFilterSideDataRunner{
		records:    make([]threading.FrameWorkLoopFilterBlockRecord, 256),
		masks:      make([]lfmask.FilterMask, 16),
		levelCache: make([]uint8, 4*(256)),
	}
	post := testSupportedPostFilterRunner{
		scratch: FrameWorkPostFilterScratch{
			LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, 256),
		},
	}

	result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
	if err != nil {
		t.Fatal(err)
	}
	if side.loopFilterMap.Stride == 0 {
		t.Fatalf("side data runner did not bind loop-filter map: %+v", side)
	}
	if result.Step.Kind != FrameWorkStepBegin ||
		result.Output == nil ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	if stats := runner.Workers[0].Stats; stats.Loop.Blocks == 0 || stats.CoefficientBlocks == 0 || stats.TXBs == 0 {
		t.Fatalf("residual runner stats=%+v", stats)
	}
	coverage, err := side.loopFilterMap.CoverageStats(int(side.loopFilterMap.Stride), int(side.loopFilterMap.Rows))
	if err != nil {
		t.Fatal(err)
	}
	if coverage.Blocks == 0 || coverage.Missing != 0 {
		t.Fatalf("loop-filter coverage=%+v", coverage)
	}
	if post.result.Completed != FrameWorkPostFilterLoopFilter ||
		!post.result.LoopFilter.Active ||
		post.result.LoopFilter.Plan.Blocks == 0 ||
		post.ctx.RemainingPostFilters() != 0 {
		t.Fatalf("postfilter result=%+v size=%+v remaining=%v", post.result, post.size, post.ctx.RemainingPostFilters())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
}

func TestFrameWorkStateRunEventWithContextTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	begin, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, nil)
	if err != nil {
		t.Fatal(err)
	}
	if begin.Step.Kind != FrameWorkStepBegin || begin.Output == nil || begin.Run != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("begin=%+v active=%v", begin, state.Active())
	}

	wantSequence := threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader)
	events[2].SequenceHeader = parser.SequenceHeader{}
	var seenOutput *frame.Frame
	tileResult, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[2], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		seenOutput = ctx.Output
		if ctx.Step.Kind != FrameWorkStepTile ||
			len(ctx.References) != 0 ||
			len(ctx.Jobs) != 1 ||
			len(ctx.Payload) != len(events[2].Unit.Payload) ||
			ctx.Sequence != wantSequence ||
			ctx.Payload[int(ctx.Jobs[0].Offset)] != 0x80 {
			t.Fatalf("ctx=%+v", ctx)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if tileResult.Step.Kind != FrameWorkStepTile ||
		tileResult.Output != begin.Output ||
		seenOutput != begin.Output ||
		tileResult.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("tile=%+v seen=%p begin=%p active=%v", tileResult, seenOutput, begin.Output, state.Active())
	}
}

func TestFrameWorkStateRunEventWithContextRunnerTileGroup(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	begin, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, nil)
	if err != nil {
		t.Fatal(err)
	}
	if begin.Step.Kind != FrameWorkStepBegin || begin.Output == nil || begin.Run != (FrameWorkStepResult{}) || !state.Active() {
		t.Fatalf("begin=%+v active=%v", begin, state.Active())
	}

	events[2].SequenceHeader = parser.SequenceHeader{}
	runner := frameWorkContextRunner{
		wantKind:     FrameWorkStepTile,
		wantSequence: threading.FrameWorkSequenceContextFromHeader(events[0].SequenceHeader),
		wantEvent:    events[2],
		wantPayload:  0x80,
	}
	tileResult, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, &events[2], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
	if err != nil {
		t.Fatal(err)
	}
	if runner.err != nil {
		t.Fatal(runner.err)
	}
	if tileResult.Step.Kind != FrameWorkStepTile ||
		tileResult.Output != begin.Output ||
		runner.output != begin.Output ||
		tileResult.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		state.Active() {
		t.Fatalf("tile=%+v seen=%p begin=%p active=%v", tileResult, runner.output, begin.Output, state.Active())
	}
}

func TestFrameWorkStateRunEventWithContextPostFilterBeforeFinish(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	var order [2]string
	orderIndex := 0
	result, err := state.RunEventWithContextAndPostFilter(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		order[orderIndex] = "tile"
		orderIndex++
		if !state.Active() {
			t.Fatal("state inactive during tile callback")
		}
		ctx.Output.Y.Pix[0] = 0x11
		return nil
	}, func(ctx FrameWorkPostFilterContext) error {
		order[orderIndex] = "post"
		orderIndex++
		if !state.Active() {
			t.Fatal("state inactive during postfilter")
		}
		if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
			t.Fatalf("published before postfilter slot=%d ok=%v", slot, ok)
		}
		if ctx.Output.Y.Pix[0] != 0x11 {
			t.Fatalf("postfilter saw sample=%d want 0x11", ctx.Output.Y.Pix[0])
		}
		if ctx.Event.Kind != EventFrame || ctx.Step.Kind != FrameWorkStepBegin || !ctx.ExecutedTileWork || ctx.ReferenceCount != 0 {
			t.Fatalf("postfilter context=%+v", ctx)
		}
		ctx.Output.Y.Pix[0] = 0x33
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if orderIndex != 2 || order != ([2]string{"tile", "post"}) {
		t.Fatalf("order=%v", order)
	}
	if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
	published, err := pool.Frame(slot)
	if err != nil {
		t.Fatal(err)
	}
	if published.Y.Pix[0] != 0x33 {
		t.Fatalf("published sample=%d want 0x33", published.Y.Pix[0])
	}
}

func TestFrameWorkStateRunEventWithContextRunnersPostFilterBeforeFinish(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	var order [2]string
	runner := frameWorkWritingRunner{
		order: &order,
		state: &state,
		value: 0x11,
	}
	post := frameWorkCheckingPostRunner{
		order: &order,
		refs:  &refs,
		state: &state,
		value: 0x33,
	}
	result, err := state.RunEventWithContextAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner, &post)
	if err != nil {
		t.Fatal(err)
	}
	if runner.err != nil {
		t.Fatal(runner.err)
	}
	if post.err != nil {
		t.Fatal(post.err)
	}
	if order != ([2]string{"tile", "post"}) {
		t.Fatalf("order=%v", order)
	}
	if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) || state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
	published, err := pool.Frame(slot)
	if err != nil {
		t.Fatal(err)
	}
	if published.Y.Pix[0] != 0x33 {
		t.Fatalf("published sample=%d want 0x33", published.Y.Pix[0])
	}
}

func TestFrameWorkStateRunStepWithPostFilterCarriesSideMaps(t *testing.T) {
	pool := testFramePoolForSize(t, 64, 64, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int

	seq := testSequence()
	begin := Event{
		Kind:           EventFrameHeader,
		SequenceHeader: seq,
		FrameHeader:    parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:      testFrameSize(64, 64),
	}
	plan, _, err := state.Begin(&refs, &pool, seq, &begin, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	final := begin
	final.Kind = EventTileGroup
	final.TileGroup.Final = true
	final.FrameSize.RefreshFrameFlags = 1
	final.FrameSize.UpscaledWidth = 64
	final.FrameSize.SuperResDenominator = 8
	final.CDEF = parser.CDEFParams{Damping: 5, StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{8}}
	final.LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{1}}
	final.Restoration = parser.RestorationParams{
		Type:      [3]parser.RestorationType{parser.RestorationWiener},
		UnitSizeY: 64,
	}
	ctx := FrameWorkBatch{FrameWorkFrameContext: frameWorkFrameContext(&final, threading.FrameWorkSequenceContextFromHeader(final.SequenceHeader))}
	_, _, cdefLen, err := ctx.CDEFIndexMapShape()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := ctx.BindCDEFIndexMap(make([]uint8, cdefLen), make([]bool, cdefLen))
	if err != nil {
		t.Fatal(err)
	}
	if err := state.SetCDEFIndexMap(cdefMap); err != nil {
		t.Fatal(err)
	}
	_, _, lfLen, err := ctx.LoopFilterMapShape()
	if err != nil {
		t.Fatal(err)
	}
	lfMap, err := ctx.BindLoopFilterMap(make([]threading.FrameWorkLoopFilterBlockRecord, lfLen))
	if err != nil {
		t.Fatal(err)
	}
	if err := state.SetLoopFilterMap(lfMap); err != nil {
		t.Fatal(err)
	}
	restorationPlan, err := ctx.RestorationFramePlan()
	if err != nil {
		t.Fatal(err)
	}
	restorationBuffers, err := ctx.BindRestorationFrameBuffers(
		make([]tile.RestorationUnitRecord, restorationPlan.UnitRecordLen()),
		make([]uint16, restorationPlan.BoundaryBufferLen()),
		make([]uint16, restorationPlan.BoundaryBufferLen()),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.SetRestorationFrameBuffers(restorationBuffers); err != nil {
		t.Fatal(err)
	}

	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: plan.Surface},
	}
	var postRan bool
	result, err := state.RunStepWithPostFilter(&refs, &pool, &final, step, nil, nil, nil, releases[:], nil, func(post FrameWorkPostFilterContext) error {
		postRan = true
		if post.CDEFIndexMap == nil || post.CDEFIndexMap.Stride != cdefMap.Stride || post.CDEFIndexMap.Rows != cdefMap.Rows {
			t.Fatalf("CDEFIndexMap=%+v want stride=%d rows=%d", post.CDEFIndexMap, cdefMap.Stride, cdefMap.Rows)
		}
		if post.LoopFilterMap == nil || post.LoopFilterMap.Stride != lfMap.Stride || post.LoopFilterMap.Rows != lfMap.Rows {
			t.Fatalf("LoopFilterMap=%+v want stride=%d rows=%d", post.LoopFilterMap, lfMap.Stride, lfMap.Rows)
		}
		if post.RestorationFrameBuffers == nil || len(post.RestorationFrameBuffers.Records[0]) != int(restorationPlan.UnitRecords[0]) {
			t.Fatalf("RestorationFrameBuffers=%+v want records=%d", post.RestorationFrameBuffers, restorationPlan.UnitRecords[0])
		}
		post.CDEFIndexMap.Read[0] = true
		post.LoopFilterMap.Records[0].Valid = true
		post.RestorationFrameBuffers.Records[0][0].Unit.Type = parser.RestorationWiener
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !postRan || !result.CompletedFrame || state.Active() {
		t.Fatalf("postRan=%v result=%+v active=%v", postRan, result, state.Active())
	}
	if !cdefMap.Read[0] {
		t.Fatal("postfilter CDEFIndexMap did not alias caller storage")
	}
	if !lfMap.Records[0].Valid {
		t.Fatal("postfilter LoopFilterMap did not alias caller storage")
	}
	if restorationBuffers.Records[0][0].Unit.Type != parser.RestorationWiener {
		t.Fatal("postfilter RestorationFrameBuffers did not alias caller storage")
	}
}

func TestFrameWorkStatePostFilterContextPreview(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	cdefMap := threading.FrameWorkCDEFIndexMap{Index: make([]uint8, 1), Read: make([]bool, 1), Stride: 1, Rows: 1}
	lfMap := threading.FrameWorkLoopFilterMap{Records: make([]threading.FrameWorkLoopFilterBlockRecord, 1), Stride: 1, Rows: 1}
	restorationBuffers := threading.FrameWorkRestorationFrameBuffers{Plan: tile.RestorationFramePlan{Planes: 1}}
	state := FrameWorkState{
		Surface:                      surface,
		ReferenceCount:               2,
		cdefIndexMap:                 cdefMap,
		cdefIndexMapValid:            true,
		loopFilterMap:                lfMap,
		loopFilterMapValid:           true,
		restorationFrameBuffers:      restorationBuffers,
		restorationFrameBuffersValid: true,
		active:                       true,
	}
	event := finalFrameEvent(0)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			Surface:        surface,
			ReferenceCount: 2,
		},
	}

	ctx, err := state.PostFilterContext(&pool, &event, step, true)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Event.Kind != event.Kind ||
		!ctx.Event.TileGroup.Final ||
		ctx.Step != step ||
		ctx.Output != output ||
		ctx.ReferenceCount != 2 ||
		!ctx.ExecutedTileWork ||
		ctx.CDEFIndexMap != &state.cdefIndexMap ||
		ctx.LoopFilterMap != &state.loopFilterMap ||
		ctx.RestorationFrameBuffers != &state.restorationFrameBuffers {
		t.Fatalf("ctx=%+v output=%p state=%+v", ctx, output, state)
	}
	ctx.CDEFIndexMap.Read[0] = true
	ctx.LoopFilterMap.Records[0].Valid = true
	if !cdefMap.Read[0] || !lfMap.Records[0].Valid {
		t.Fatal("preview context did not alias caller-owned side maps")
	}
	if !state.Active() {
		t.Fatal("preview context mutated active state")
	}

	nonFinal := event
	nonFinal.TileGroup.Final = false
	noop, err := state.PostFilterContext(&pool, &nonFinal, step, true)
	if err != nil {
		t.Fatal(err)
	}
	if noop.Output != nil || noop.Event.Kind != EventIgnored || noop.Step.Kind != FrameWorkStepIgnored {
		t.Fatalf("non-final context=%+v want zero", noop)
	}
	if _, err := state.PostFilterContext(nil, &event, step, true); !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("nil pool err=%v want %v", err, frame.ErrInvalidPool)
	}
	if _, err := state.PostFilterContext(&pool, &event, FrameWorkStep{}, true); !errors.Is(err, ErrInvalidFrameWorkStep) {
		t.Fatalf("bad step err=%v want %v", err, ErrInvalidFrameWorkStep)
	}
	state.active = false
	if _, err := state.PostFilterContext(&pool, &event, step, true); !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive err=%v want %v", err, ErrInvalidFrameWorkState)
	}
}

func TestFrameWorkStateRunEventWithContextPostFilterErrorKeepsActive(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	want := errors.New("postfilter")
	var tileRan bool
	_, err = state.RunEventWithContextAndPostFilter(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		tileRan = true
		ctx.Output.Y.Pix[0] = 0x22
		return nil
	}, func(ctx FrameWorkPostFilterContext) error {
		if ctx.Output.Y.Pix[0] != 0x22 {
			t.Fatalf("postfilter sample=%d want 0x22", ctx.Output.Y.Pix[0])
		}
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("RunEventWithContextAndPostFilter err=%v want %v", err, want)
	}
	if !tileRan || !state.Active() {
		t.Fatalf("tileRan=%v active=%v", tileRan, state.Active())
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
	output, err := pool.Frame(state.Surface)
	if err != nil {
		t.Fatal(err)
	}
	if output.Y.Pix[0] != 0x22 {
		t.Fatalf("active output sample=%d want 0x22", output.Y.Pix[0])
	}
}

func TestFrameWorkPostFilterContextRequireNoActivePostFilters(t *testing.T) {
	if err := (&FrameWorkPostFilterContext{}).RequireNoActivePostFilters(); err != nil {
		t.Fatalf("inactive postfilter err=%v", err)
	}

	tests := []struct {
		name  string
		event Event
	}{
		{
			name: "loopfilter",
			event: Event{
				LoopFilter: parser.LoopFilterParams{LevelY: [2]uint8{1, 0}},
			},
		},
		{
			name: "cdef-indexes",
			event: Event{
				CDEF: parser.CDEFParams{Bits: 1, StrengthCount: 2},
			},
		},
		{
			name: "cdef-strength",
			event: Event{
				CDEF: parser.CDEFParams{StrengthCount: 1, YStrength: [parser.MaxCDEFStrengths]uint8{4}},
			},
		},
		{
			name: "superres",
			event: Event{
				FrameSize: parser.FrameSize{SuperResEnabled: true},
			},
		},
		{
			name: "restoration",
			event: Event{
				Restoration: parser.RestorationParams{Type: [3]parser.RestorationType{parser.RestorationWiener}},
			},
		},
		{
			name: "film-grain",
			event: Event{
				FilmGrain: parser.FilmGrainParams{Apply: true},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := (&FrameWorkPostFilterContext{Event: tt.event}).RequireNoActivePostFilters()
			if !errors.Is(err, ErrUnsupportedPostFilter) {
				t.Fatalf("err=%v want %v", err, ErrUnsupportedPostFilter)
			}
		})
	}
}

func TestFrameWorkStateRunStepWithPostFilterAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	event := finalFrameEvent(0)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}
	post := func(ctx FrameWorkPostFilterContext) error {
		if ctx.Output != output || ctx.ExecutedTileWork {
			t.Fatalf("ctx=%+v output=%p", ctx, output)
		}
		ctx.Output.Y.Pix[0] ^= 1
		return nil
	}
	var state FrameWorkState
	allocs := testing.AllocsPerRun(1000, func() {
		var refs SurfaceReferences
		state = FrameWorkState{Surface: surface, active: true}
		if _, err := state.RunStepWithPostFilter(&refs, &pool, &event, step, nil, nil, nil, nil, nil, post); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("RunStepWithPostFilter allocated: %f", allocs)
	}
}

func TestFrameWorkStatePostFilterContextAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	surface, output, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	event := finalFrameEvent(0)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Surface: surface},
	}
	state := FrameWorkState{Surface: surface, active: true}

	allocs := testing.AllocsPerRun(1000, func() {
		ctx, err := state.PostFilterContext(&pool, &event, step, false)
		if err != nil {
			t.Fatal(err)
		}
		if ctx.Output != output || ctx.ExecutedTileWork {
			t.Fatalf("ctx=%+v output=%p", ctx, output)
		}
	})
	if allocs != 0 {
		t.Fatalf("PostFilterContext allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithContextInterFrameReferences(t *testing.T) {
	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	interFrame := append([]byte{}, interFrameHeaderPayload()...)
	interFrame = append(interFrame, 0xbb)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, keyFrame)
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, interFrame)

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[2].FrameSize.CodedWidth, events[2].FrameSize.Height, 2)
	referenceSurface, referenceFrame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSurface, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	result, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[2], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		if ctx.Output == nil || ctx.Output == referenceFrame {
			t.Fatalf("output=%p reference=%p", ctx.Output, referenceFrame)
		}
		if len(ctx.Payload) != len(events[2].Unit.Payload) || ctx.Payload[int(ctx.Jobs[0].Offset)] != 0xbb {
			t.Fatalf("payload=%v", ctx.Payload)
		}
		if len(ctx.References) != parser.InterRefsPerFrame {
			t.Fatalf("references=%d", len(ctx.References))
		}
		for i := 0; i < len(ctx.References); i++ {
			if ctx.References[i] != referenceFrame {
				t.Fatalf("reference[%d]=%p want %p", i, ctx.References[i], referenceFrame)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ReferenceCount != parser.InterRefsPerFrame ||
		result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true, ReleaseCount: 1}) ||
		state.Active() {
		t.Fatalf("result=%+v active=%v", result, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != result.Step.Begin.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, result.Step.Begin.Surface)
	}
	if _, err := pool.Frame(referenceSurface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("reference surface err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStateRunEventWithContextSwitchFrameRefreshesAllReferences(t *testing.T) {
	keyFrame := append([]byte{}, shownKeyFrameHeaderPayload()...)
	keyFrame = append(keyFrame, 0xaa)
	switchFrame := append([]byte{}, switchFrameHeaderPayload()...)
	switchFrame = append(switchFrame, 0xbb)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testRealtimeNoOrderSequenceHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, keyFrame)
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, switchFrame)

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}
	if events[2].FrameHeader.FrameType != parser.FrameTypeSwitch ||
		events[2].FrameSize.RefreshFrameFlags != 0xff ||
		!events[2].FrameHeader.ErrorResilientMode ||
		!events[2].FrameHeader.FrameSizeOverride {
		t.Fatalf("switch event=%+v", events[2])
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[2].FrameSize.CodedWidth, events[2].FrameSize.Height, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	var keyOutput *frame.Frame
	keyResult, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		keyOutput = ctx.Output
		if ctx.Output == nil || len(ctx.References) != 0 {
			t.Fatalf("key ctx output=%p references=%d", ctx.Output, len(ctx.References))
		}
		if len(ctx.Payload) != len(events[1].Unit.Payload) || ctx.Payload[int(ctx.Jobs[0].Offset)] != 0xaa {
			t.Fatalf("key payload=%v", ctx.Payload)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if keyResult.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		keyResult.ReferenceCount != 0 ||
		keyOutput == nil ||
		state.Active() {
		t.Fatalf("key result=%+v output=%p active=%v", keyResult, keyOutput, state.Active())
	}
	for i := range parser.RefFrames {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != keyResult.Step.Begin.Surface {
			t.Fatalf("after key ref[%d]=%d ok=%v want %d", i, slot, ok, keyResult.Step.Begin.Surface)
		}
	}

	var switchOutput *frame.Frame
	switchResult, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[2], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx FrameWorkBatch) error {
		switchOutput = ctx.Output
		if ctx.Output == nil || ctx.Output == keyOutput {
			t.Fatalf("switch output=%p key=%p", ctx.Output, keyOutput)
		}
		if len(ctx.Payload) != len(events[2].Unit.Payload) || ctx.Payload[int(ctx.Jobs[0].Offset)] != 0xbb {
			t.Fatalf("switch payload=%v", ctx.Payload)
		}
		if len(ctx.References) != parser.InterRefsPerFrame {
			t.Fatalf("switch references=%d", len(ctx.References))
		}
		for i, ref := range ctx.References {
			if ref != keyOutput {
				t.Fatalf("switch reference[%d]=%p want key %p", i, ref, keyOutput)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if switchResult.ReferenceCount != parser.InterRefsPerFrame ||
		switchResult.Output != switchOutput ||
		switchResult.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true, ReleaseCount: 1}) ||
		state.Active() {
		t.Fatalf("switch result=%+v output=%p active=%v", switchResult, switchOutput, state.Active())
	}
	for i := range parser.RefFrames {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != switchResult.Step.Begin.Surface {
			t.Fatalf("after switch ref[%d]=%d ok=%v want %d", i, slot, ok, switchResult.Step.Begin.Surface)
		}
	}
	if _, err := pool.Frame(keyResult.Step.Begin.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("key surface err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if _, err := pool.Frame(switchResult.Step.Begin.Surface); err != nil {
		t.Fatalf("switch surface err=%v", err)
	}
}

func TestFrameWorkStateRunEventWithContextShowExisting(t *testing.T) {
	pool := testFramePool(t, 1)
	referenceSurface, referenceFrame, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, referenceSurface, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	result, err := state.RunEventWithContext(&refs, &pool, testSequence(), showExistingWorkEventPtr(0, parser.FrameTypeInter), 32, nil, nil, 1, nil, nil, nil, releases[:], nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Step.Kind != FrameWorkStepShowExisting ||
		result.Output != referenceFrame ||
		result.Run != (FrameWorkStepResult{}) ||
		state.Active() {
		t.Fatalf("result=%+v output=%p active=%v", result, referenceFrame, state.Active())
	}
}

func TestFrameWorkStateRunStepRejectsMismatchedFinalStep(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	var releases [parser.RefFrames]int
	_, err = state.RunStep(&refs, &pool, finalFrameEventPtr(0xff), FrameWorkStep{Kind: FrameWorkStepIgnored}, nil, nil, nil, releases[:], nil)
	if !errors.Is(err, ErrInvalidFrameWorkStep) {
		t.Fatalf("RunStep err=%v want %v", err, ErrInvalidFrameWorkStep)
	}
	if !state.Active() || state.Surface != begin.Surface {
		t.Fatalf("state=%+v active=%v begin=%+v", state, state.Active(), begin)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("slot=%d ok=%v want no publication", slot, ok)
	}
}

func TestFrameWorkStateRunStepNoop(t *testing.T) {
	var state FrameWorkState
	result, err := state.RunStep(nil, nil, &Event{Kind: EventMetadata}, FrameWorkStep{Kind: FrameWorkStepIgnored}, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result != (FrameWorkStepResult{}) {
		t.Fatalf("result=%+v", result)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWork(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}

	var releases [parser.RefFrames]int
	completed, releaseCount, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, &events[2], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 0 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != plan.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, plan.Surface)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkFrameOBU(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}

	var releases [parser.RefFrames]int
	completed, releaseCount, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, &events[1], releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if !completed || releaseCount != 0 || state.Active() {
		t.Fatalf("completed=%v releaseCount=%d active=%v", completed, releaseCount, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != plan.Surface {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, plan.Surface)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkNoop(t *testing.T) {
	var nilState *FrameWorkState
	completed, count, err := nilState.FinishIfEventCompletesFrameWork(nil, nil, &Event{Kind: EventTileGroup}, nil)
	if err != nil || completed || count != 0 {
		t.Fatalf("nil state completed=%v count=%d err=%v", completed, count, err)
	}

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	completed, count, err = state.FinishIfEventCompletesFrameWork(&refs, &pool, &Event{Kind: EventTileGroup}, nil)
	if err != nil || completed || count != 0 || !state.Active() {
		t.Fatalf("completed=%v count=%d err=%v active=%v", completed, count, err, state.Active())
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkRejectsInactive(t *testing.T) {
	var state FrameWorkState
	var releases [parser.RefFrames]int
	completed, count, err := state.FinishIfEventCompletesFrameWork(nil, nil, finalFrameEventPtr(0xff), releases[:])
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("FinishIfEventCompletesFrameWork err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	if completed || count != 0 {
		t.Fatalf("completed=%v count=%d", completed, count)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkKeepsActiveOnError(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}

	completed, count, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, finalFrameEventPtr(0xff), releases[:])
	if !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("FinishIfEventCompletesFrameWork err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if completed || count != 0 || !state.Active() || state.Surface != plan.Surface {
		t.Fatalf("completed=%v count=%d state=%+v active=%v", completed, count, state, state.Active())
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index0 {
		t.Fatalf("slot=%d ok=%v want unchanged %d", slot, ok, index0)
	}
}

func TestFrameWorkStateShowExistingDropsActiveWork(t *testing.T) {
	pool := testFramePool(t, 2)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	plan, err := state.ShowExisting(&refs, &pool, showExistingWorkEventPtr(0, parser.FrameTypeInter), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (ShowExistingFrameWorkPlan{Surface: reference, DroppedFrameWork: true}) {
		t.Fatalf("plan=%+v", plan)
	}
	if state.Active() || pool.Available() != 1 {
		t.Fatalf("active=%v available=%d", state.Active(), pool.Available())
	}
	if _, err := pool.Frame(begin.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("dropped frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if _, err := pool.Frame(reference); err != nil {
		t.Fatalf("reference frame err=%v", err)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != reference {
		t.Fatalf("slot=%d ok=%v want %d", slot, ok, reference)
	}
}

func TestFrameWorkStateShowExistingKeyResetsReferences(t *testing.T) {
	pool := testFramePool(t, 3)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, index0, releases[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(1<<1, index1, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	if _, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}

	plan, err := state.ShowExisting(&refs, &pool, showExistingWorkEventPtr(1, parser.FrameTypeKey), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (ShowExistingFrameWorkPlan{Surface: index1, ReleaseCount: 1, DroppedFrameWork: true}) || releases[0] != index0 {
		t.Fatalf("plan=%+v release=%d want surface=%d release=%d", plan, releases[0], index1, index0)
	}
	if pool.Available() != 2 {
		t.Fatalf("available=%d want 2", pool.Available())
	}
	for i := range parser.RefFrames {
		slot, ok := refs.ReferenceSlot(i)
		if !ok || slot != index1 {
			t.Fatalf("slot[%d]=%d ok=%v want %d", i, slot, ok, index1)
		}
	}
}

func TestFrameWorkStateShowExistingInactive(t *testing.T) {
	pool := testFramePool(t, 1)
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	plan, err := state.ShowExisting(&refs, &pool, showExistingWorkEventPtr(0, parser.FrameTypeInter), releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (ShowExistingFrameWorkPlan{Surface: reference}) {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestFrameWorkStateShowExistingRejectsInvalidEventBeforeAbort(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = state.ShowExisting(&refs, &pool, &Event{Kind: EventFrameHeader}, nil)
	if !errors.Is(err, ErrInvalidSurfaceEvent) {
		t.Fatalf("ShowExisting err=%v want %v", err, ErrInvalidSurfaceEvent)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 0 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateShowExistingRejectsMissingReferenceBeforeAbort(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = state.ShowExisting(&refs, &pool, showExistingWorkEventPtr(0, parser.FrameTypeInter), releases[:])
	if !errors.Is(err, ErrInvalidSurfaceReference) {
		t.Fatalf("ShowExisting err=%v want %v", err, ErrInvalidSurfaceReference)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 0 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateShowExistingKeepsActiveOnAbortError(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	reference, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	begin, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = state.ShowExisting(&refs, nil, showExistingWorkEventPtr(0, parser.FrameTypeInter), nil)
	if !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("ShowExisting err=%v want %v", err, frame.ErrInvalidPool)
	}
	if !state.Active() || state.Surface != begin.Surface || pool.Available() != 0 {
		t.Fatalf("state=%+v active=%v available=%d", state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWork(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	aborted, err := state.AbortIfEventDropsFrameWork(&pool, &Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true})
	if err != nil {
		t.Fatal(err)
	}
	if !aborted || state.Active() || pool.Available() != 1 {
		t.Fatalf("aborted=%v state=%+v active=%v available=%d", aborted, state, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStateAbortIfShowExistingFrameDropsFrameWork(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	aborted, err := state.AbortIfEventDropsFrameWork(&pool, &Event{Kind: EventExistingFrame})
	if err != nil {
		t.Fatal(err)
	}
	if !aborted || state.Active() || pool.Available() != 1 {
		t.Fatalf("aborted=%v active=%v available=%d", aborted, state.Active(), pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWorkNoop(t *testing.T) {
	var nilState *FrameWorkState
	aborted, err := nilState.AbortIfEventDropsFrameWork(nil, &Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true})
	if err != nil || aborted {
		t.Fatalf("nil state aborted=%v err=%v", aborted, err)
	}

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	if aborted, err = state.AbortIfEventDropsFrameWork(&pool, &Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true}); err != nil || aborted {
		t.Fatalf("inactive state aborted=%v err=%v", aborted, err)
	}

	if _, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	aborted, err = state.AbortIfEventDropsFrameWork(&pool, &Event{Kind: EventSequenceHeader, OperatingParametersChanged: true})
	if err != nil || aborted || !state.Active() || pool.Available() != 0 {
		t.Fatalf("non-drop event aborted=%v err=%v active=%v available=%d", aborted, err, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWorkKeepsActiveOnError(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	aborted, err := state.AbortIfEventDropsFrameWork(nil, &Event{Kind: EventSequenceHeader, NewCodedVideoSequence: true})
	if !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("AbortIfEventDropsFrameWork err=%v want %v", err, frame.ErrInvalidPool)
	}
	if aborted || !state.Active() || state.Surface != plan.Surface || pool.Available() != 0 {
		t.Fatalf("aborted=%v state=%+v active=%v available=%d", aborted, state, state.Active(), pool.Available())
	}
}

func TestFrameWorkStateRejectsInvalidState(t *testing.T) {
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, err := state.PlanTile(&Event{Kind: EventTileGroup}, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive PlanTile err=%v want %v", err, ErrInvalidFrameWorkState)
	}

	var releases [parser.RefFrames]int
	_, err = state.Finish(nil, nil, finalFrameEventPtr(0xff), releases[:])
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive Finish err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	err = state.Abort(nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("inactive Abort err=%v want %v", err, ErrInvalidFrameWorkState)
	}

	var nilState *FrameWorkState
	err = nilState.Abort(nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("nil Abort err=%v want %v", err, ErrInvalidFrameWorkState)
	}

	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	if _, _, err := state.Begin(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil); err != nil {
		t.Fatal(err)
	}
	_, _, err = state.Begin(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil)
	if !errors.Is(err, ErrInvalidFrameWorkState) {
		t.Fatalf("active Begin err=%v want %v", err, ErrInvalidFrameWorkState)
	}
	if pool.Available() != 1 {
		t.Fatalf("active begin consumed a surface, available=%d", pool.Available())
	}
}

func TestFrameWorkStateAbortReleasesSurface(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, output, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if output == nil || pool.Available() != 0 {
		t.Fatalf("output=%p available=%d", output, pool.Available())
	}

	if err := state.Abort(&pool); err != nil {
		t.Fatal(err)
	}
	if state.Active() {
		t.Fatalf("state still active after abort: %+v", state)
	}
	if pool.Available() != 1 {
		t.Fatalf("available=%d want 1", pool.Available())
	}
	if _, err := pool.Frame(plan.Surface); !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("aborted frame err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if slot, ok := refs.ReferenceSlot(0); ok || slot != -1 {
		t.Fatalf("abort published reference slot=%d ok=%v", slot, ok)
	}
}

func TestFrameWorkStateKeepsActiveOnAbortError(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = state.Abort(nil)
	if !errors.Is(err, frame.ErrInvalidPool) {
		t.Fatalf("Abort err=%v want %v", err, frame.ErrInvalidPool)
	}
	if !state.Active() || state.Surface != plan.Surface {
		t.Fatalf("state after abort error=%+v active=%v plan=%+v", state, state.Active(), plan)
	}
	if pool.Available() != 0 {
		t.Fatalf("failed abort released surface, available=%d", pool.Available())
	}
}

func TestFrameWorkStateKeepsActiveOnFinishError(t *testing.T) {
	pool := testFramePool(t, 2)
	index0, _, err := pool.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	var refs SurfaceReferences
	var releases [parser.RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}

	var state FrameWorkState
	plan, _, err := state.Begin(&refs, &pool, testSequence(), &Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}, 32, nil, 1, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := pool.Release(index0); err != nil {
		t.Fatal(err)
	}

	_, err = state.Finish(&refs, &pool, finalFrameEventPtr(0xff), releases[:])
	if !errors.Is(err, frame.ErrInvalidSlot) {
		t.Fatalf("Finish err=%v want %v", err, frame.ErrInvalidSlot)
	}
	if !state.Active() || state.Surface != plan.Surface {
		t.Fatalf("state after error=%+v active=%v plan=%+v", state, state.Active(), plan)
	}
	slot, ok := refs.ReferenceSlot(0)
	if !ok || slot != index0 {
		t.Fatalf("slot=%d ok=%v want unchanged %d", slot, ok, index0)
	}
}

func TestPlanTileWorkFrameEvent(t *testing.T) {
	frame := append([]byte{}, reducedStillFrameHeaderPayload()...)
	frame = append(frame, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, frame)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	plan, err := PlanTileWork(&events[1], 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan.SpanCount != 1 || plan.JobCount != 1 || plan.BatchCount != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	if int(spans[0].Offset) != len(reducedStillFrameHeaderPayload()) || spans[0].Size != 1 {
		t.Fatalf("span=%+v", spans[0])
	}
}

func TestPlanTileWorkPropagatesContextUpdateTile(t *testing.T) {
	event := Event{
		Kind: EventTileGroup,
		Unit: obu.Unit{Payload: []byte{
			0x00, // tile 0 size minus one.
			0xaa,
			0xbb,
		}},
		TileInfo: parser.TileInfo{
			RefreshContext:      true,
			ContextUpdateTileID: 1,
			TileSizeBytes:       1,
			SBCols:              2,
			SBRows:              1,
			Cols:                2,
			Rows:                1,
			ColStartSB:          [parser.MaxTileCols + 1]uint16{0, 1, 2},
			RowStartSB:          [parser.MaxTileRows + 1]uint16{0, 1},
		},
		TileGroup: parser.TileGroup{
			StartTile: 0,
			EndTile:   1,
			TileCount: 2,
			Final:     true,
		},
	}
	var spans [2]parser.TileSpan
	var jobs [2]tile.Job
	var batches [2]threading.Batch

	plan, err := PlanTileWork(&event, 1, spans[:], jobs[:], batches[:])
	if err != nil {
		t.Fatal(err)
	}
	if plan != (TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: 1}) {
		t.Fatalf("plan=%+v", plan)
	}
	if jobs[0].UpdatesFrameContext {
		t.Fatalf("job[0]=%+v should not update frame context", jobs[0])
	}
	if !jobs[1].UpdatesFrameContext {
		t.Fatalf("job[1]=%+v should update frame context", jobs[1])
	}
}

func TestPlanTileWorkRejectsNonTileEvent(t *testing.T) {
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	_, err := PlanTileWork(&Event{Kind: EventSequenceHeader}, 1, spans[:], jobs[:], batches[:])
	if !errors.Is(err, ErrInvalidTileWork) {
		t.Fatalf("PlanTileWork err=%v want %v", err, ErrInvalidTileWork)
	}
}

func TestPlanTileWorkAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := PlanTileWork(&events[2], 1, spans[:], jobs[:], batches[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PlanTileWork allocated: %f", allocs)
	}
}

func TestBeginFrameWorkAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		plan, _, err := BeginFrameWork(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if plan.Surface != 0 {
			t.Fatalf("surface=%d", plan.Surface)
		}
	})
	if allocs != 0 {
		t.Fatalf("BeginFrameWork allocated: %f", allocs)
	}
}

func TestPlanFrameTileWorkAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := PlanFrameTileWork(&events[2], 0, 0, 1, spans[:], jobs[:], batches[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PlanFrameTileWork allocated: %f", allocs)
	}
}

func TestFrameWorkStateAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = state.PlanTile(&events[2], 1, spans[:], jobs[:], batches[:])
		if err != nil {
			t.Fatal(err)
		}
		if _, err = state.Finish(&refs, &pool, &events[2], releases[:]); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState allocated: %f", allocs)
	}
}

func TestFrameWorkStateAbortAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		if err := state.Abort(&pool); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState Abort allocated: %f", allocs)
	}
}

func TestFrameWorkStateAbortIfEventDropsFrameWorkAllocs(t *testing.T) {
	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	drop := Event{Kind: EventExistingFrame}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		aborted, err := state.AbortIfEventDropsFrameWork(&pool, &drop)
		if err != nil {
			t.Fatal(err)
		}
		if !aborted {
			t.Fatal("not aborted")
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState AbortIfEventDropsFrameWork allocated: %f", allocs)
	}
}

func TestFrameWorkStateFinishIfEventCompletesFrameWorkAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		completed, _, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, &events[2], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if !completed {
			t.Fatal("not completed")
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState FinishIfEventCompletesFrameWork allocated: %f", allocs)
	}
}

func TestFrameWorkStateShowExistingAllocs(t *testing.T) {
	pool := testFramePool(t, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	show := showExistingWorkEvent(0, parser.FrameTypeInter)

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		reference, _, err := pool.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := refs.Refresh(1<<0, reference, releases[:]); err != nil {
			t.Fatal(err)
		}
		if _, _, err := state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		plan, err := state.ShowExisting(&refs, &pool, &show, releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if plan.Surface != reference || !plan.DroppedFrameWork {
			t.Fatalf("plan=%+v reference=%d", plan, reference)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState ShowExisting allocated: %f", allocs)
	}
}

func TestFrameWorkStatePlanEventAllocs(t *testing.T) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("count=%d", count)
	}

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		step, output, err := state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if step.Kind != FrameWorkStepBegin || output == nil {
			t.Fatalf("begin step=%+v output=%p", step, output)
		}
		step, output, err = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[2], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if step.Kind != FrameWorkStepTile || output != nil {
			t.Fatalf("tile step=%+v output=%p", step, output)
		}
		completed, _, err := state.FinishIfEventCompletesFrameWork(&refs, &pool, &events[2], releases[:])
		if err != nil {
			t.Fatal(err)
		}
		if !completed {
			t.Fatal("not completed")
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState PlanEvent allocated: %f", allocs)
	}
}

func TestExecuteFrameWorkStepAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		executed, err := executeTestFrameWorkStep(step, workerPool, jobs[:], batches[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !executed {
			t.Fatal("not executed")
		}
	})
	if allocs != 0 {
		t.Fatalf("ExecuteFrameWorkStep allocated: %f", allocs)
	}
}

func TestExecuteFrameWorkStepWithContextAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	var output frame.Frame
	var reference frame.Frame
	references := [parser.InterRefsPerFrame]*frame.Frame{&reference}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		executed, err := executeTestFrameWorkStepWithContext(step, workerPool, &output, references[:], jobs[:], batches[:], noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
		if !executed {
			t.Fatal("not executed")
		}
	})
	if allocs != 0 {
		t.Fatalf("ExecuteFrameWorkStepWithContext allocated: %f", allocs)
	}
}

func TestExecuteFrameWorkStepWithPayloadAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := testExecutionWork(t)
	payload := []byte{0xaa, 0xbb, 0xcc}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		executed, err := executeTestFrameWorkStepWithPayload(step, workerPool, nil, nil, payload, jobs[:], batches[:], noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
		if !executed {
			t.Fatal("not executed")
		}
	})
	if allocs != 0 {
		t.Fatalf("ExecuteFrameWorkStepWithPayload allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunStepAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := testExecutionWork(t)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		if _, _, err := state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil); err != nil {
			t.Fatal(err)
		}
		result, err := state.RunStep(&refs, &pool, &final, step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunStep allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunStepWithContextAllocs(t *testing.T) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePool(t, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := testExecutionWork(t)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, output, err := state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		result, err := state.RunStepWithContext(&refs, &pool, &final, step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(FrameWorkBatch) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if result != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunStepWithContext allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithContextAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWithContext allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithContextRunnerAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	runner := noopFrameWorkRunner{}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, runner)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWithContextRunner allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &runner)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			t.Fatalf("result=%+v", result)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWith residual runner allocated: %f", allocs)
	}
}

func TestFrameWorkStateRunEventWithResidualRunnerSideDataPostFilterAllocs(t *testing.T) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d", count)
	}
	events[1].LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{4}}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	pool := testFramePoolForSize(t, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}
	side := testLoopFilterSideDataRunner{
		records:    make([]threading.FrameWorkLoopFilterBlockRecord, 256),
		masks:      make([]lfmask.FilterMask, 16),
		levelCache: make([]uint8, 4*(256)),
	}
	post := testSupportedPostFilterRunner{
		scratch: FrameWorkPostFilterScratch{
			LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, 256),
		},
	}

	allocs := testing.AllocsPerRun(1000, func() {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
		if err != nil {
			t.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
			post.result.Completed != FrameWorkPostFilterLoopFilter ||
			post.ctx.RemainingPostFilters() != 0 {
			t.Fatalf("result=%+v post=%+v remaining=%v", result, post.result, post.ctx.RemainingPostFilters())
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkState RunEventWith residual side-data postfilter allocated: %f", allocs)
	}
}

func BenchmarkPlanTileWork(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = PlanTileWork(&events[2], 1, spans[:], jobs[:], batches[:])
	}
}

func BenchmarkPlanFrameTileWork(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = PlanFrameTileWork(&events[2], 0, 0, 1, spans[:], jobs[:], batches[:])
	}
}

func BenchmarkFrameWorkState(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, nil, nil, nil)
		_, _ = state.PlanTile(&events[2], 1, spans[:], jobs[:], batches[:])
		_, _ = state.Finish(&refs, &pool, &events[2], releases[:])
	}
}

func BenchmarkFrameWorkStateAbort(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil)
		_ = state.Abort(&pool)
	}
}

func BenchmarkFrameWorkStateAbortIfEventDropsFrameWork(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	drop := Event{Kind: EventTemporalDelimiter, NewTemporalUnit: true}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.AbortIfEventDropsFrameWork(&pool, &drop)
	}
}

func BenchmarkFrameWorkStateFinishIfEventCompletesFrameWork(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, nil, nil, nil)
		_, _, _ = state.FinishIfEventCompletesFrameWork(&refs, &pool, &events[2], releases[:])
	}
}

func BenchmarkFrameWorkStateShowExisting(b *testing.B) {
	pool := benchmarkFramePool(b, 2)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	show := showExistingWorkEvent(0, parser.FrameTypeInter)

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		reference, _, _ := pool.Acquire()
		_, _ = refs.Refresh(1<<0, reference, releases[:])
		_, _, _ = state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.ShowExisting(&refs, &pool, &show, releases[:])
	}
}

func BenchmarkFrameWorkStatePlanEvent(b *testing.B) {
	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrameHeader, reducedStillFrameHeaderPayload())
	stream = appendLowOverheadOBU(stream, obu.TypeTileGroup, []byte{0x80})

	var dec Stream
	var events [3]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 3 {
		b.Fatalf("count=%d", count)
	}

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[1], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		_, _, _ = state.PlanEvent(&refs, &pool, events[0].SequenceHeader, &events[2], 32, nil, 1, spans[:], jobs[:], batches[:], releases[:])
		_, _, _ = state.FinishIfEventCompletesFrameWork(&refs, &pool, &events[2], releases[:])
	}
}

func BenchmarkExecuteFrameWorkStep(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := benchmarkExecutionWork(b)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = executeTestFrameWorkStep(step, workerPool, jobs[:], batches[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
	}
}

func BenchmarkExecuteFrameWorkStepWithContext(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := benchmarkExecutionWork(b)
	var output frame.Frame
	var reference frame.Frame
	references := [parser.InterRefsPerFrame]*frame.Frame{&reference}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{
			ReferenceCount: 1,
			Tile:           TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount},
		},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = executeTestFrameWorkStepWithContext(step, workerPool, &output, references[:], jobs[:], batches[:], func(FrameWorkBatch) error {
			return nil
		})
	}
}

func BenchmarkExecuteFrameWorkStepWithPayload(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	jobs, batches, batchCount := benchmarkExecutionWork(b)
	payload := []byte{0xaa, 0xbb}
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = executeTestFrameWorkStepWithPayload(step, workerPool, nil, nil, payload, jobs[:], batches[:], noopFrameWorkBatch)
	}
}

func BenchmarkFrameWorkStateRunStep(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := benchmarkExecutionWork(b)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _, _ = state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.RunStep(&refs, &pool, &final, step, workerPool, jobs[:], batches[:], releases[:], func(threading.Batch, []tile.Job) error {
			return nil
		})
	}
}

func BenchmarkFrameWorkStateRunStepWithContext(b *testing.B) {
	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var releases [parser.RefFrames]int
	jobs, batches, batchCount := benchmarkExecutionWork(b)
	begin := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}
	final := finalFrameEvent(0xff)
	step := FrameWorkStep{
		Kind: FrameWorkStepTile,
		Tile: FrameTileWorkPlan{Tile: TileWorkPlan{SpanCount: 2, JobCount: 2, BatchCount: batchCount}},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, output, _ := state.Begin(&refs, &pool, testSequence(), &begin, 32, nil, 1, nil, nil, nil)
		_, _ = state.RunStepWithContext(&refs, &pool, &final, step, workerPool, output, nil, jobs[:], batches[:], releases[:], func(FrameWorkBatch) error {
			return nil
		})
	}
}

func BenchmarkFrameWorkStateRunEventWithContext(b *testing.B) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 2 {
		b.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _ = state.RunEventWithContext(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, noopFrameWorkBatch)
	}
}

func BenchmarkFrameWorkStateRunEventWithContextRunner(b *testing.B) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayload()...)
	framePayload = append(framePayload, 0xaa)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testSequenceHeaderPayload(16))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 2 {
		b.Fatalf("count=%d", count)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	runner := noopFrameWorkRunner{}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		_, _ = state.RunEventWithContextRunner(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, runner)
	}
}

func BenchmarkFrameWorkStateRunEventWithResidualSideDataPostFilter(b *testing.B) {
	framePayload := append([]byte{}, reducedStillFrameHeaderPayloadQ(64)...)
	framePayload = append(framePayload, make([]byte, 256)...)

	var stream []byte
	stream = appendLowOverheadOBU(stream, obu.TypeSequenceHeader, testStillSequenceHeaderPayload(64, 64))
	stream = appendLowOverheadOBU(stream, obu.TypeFrame, framePayload)

	var dec Stream
	var events [2]Event
	count, err := dec.PushLowOverhead(stream, events[:])
	if err != nil {
		b.Fatal(err)
	}
	if count != 2 {
		b.Fatalf("count=%d", count)
	}
	events[1].LoopFilter = parser.LoopFilterParams{LevelY: [2]uint8{4}}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	pool := benchmarkFramePoolForSize(b, events[1].FrameSize.CodedWidth, events[1].FrameSize.Height, 1)
	var refs SurfaceReferences
	var state FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [1]parser.TileSpan
	var jobs [1]tile.Job
	var batches [1]threading.Batch
	var releases [parser.RefFrames]int
	var runner threading.FrameWorkTileResidualRunner
	runner.Workers = []threading.FrameWorkTileResidualRunnerWorker{
		{
			Int32Scratch:    make([]int32, 32768),
			ResidualScratch: make([]int16, 4096),
		},
	}
	side := testLoopFilterSideDataRunner{
		records:    make([]threading.FrameWorkLoopFilterBlockRecord, 256),
		masks:      make([]lfmask.FilterMask, 16),
		levelCache: make([]uint8, 4*(256)),
	}
	post := testSupportedPostFilterRunner{
		scratch: FrameWorkPostFilterScratch{
			LoopFilterEdges: make([]FrameWorkLoopFilterPostFilterEdge, 256),
		},
	}
	pool.Reset()
	refs.Reset()
	state.Reset()
	result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
	if err != nil {
		b.Fatal(err)
	}
	if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
		post.result.Completed != FrameWorkPostFilterLoopFilter ||
		post.ctx.RemainingPostFilters() != 0 {
		b.Fatalf("result=%+v post=%+v remaining=%v", result, post.result, post.ctx.RemainingPostFilters())
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, events[0].SequenceHeader, &events[1], 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, &side, &runner, &post)
		if err != nil {
			b.Fatal(err)
		}
		if result.Run != (FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
			post.result.Completed != FrameWorkPostFilterLoopFilter ||
			post.ctx.RemainingPostFilters() != 0 {
			b.Fatalf("result=%+v post=%+v remaining=%v", result, post.result, post.ctx.RemainingPostFilters())
		}
	}
}

func BenchmarkBeginFrameWork(b *testing.B) {
	pool := benchmarkFramePool(b, 1)
	var refs SurfaceReferences
	event := Event{
		Kind:        EventFrameHeader,
		FrameHeader: parser.FrameHeaderPrefix{FrameType: parser.FrameTypeKey},
		FrameSize:   testFrameSize(16, 16),
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		_, _, _ = BeginFrameWork(&refs, &pool, testSequence(), &event, 32, nil, 1, nil, nil, nil)
	}
}

func testFramePoolForSize(t *testing.T, width uint32, height uint32, count int) frame.Pool {
	t.Helper()
	pool, err := makeFramePoolForSize(width, height, count)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func benchmarkFramePoolForSize(b *testing.B, width uint32, height uint32, count int) frame.Pool {
	b.Helper()
	pool, err := makeFramePoolForSize(width, height, count)
	if err != nil {
		b.Fatal(err)
	}
	return pool
}

func testExecutionWork(t *testing.T) ([2]tile.Job, [2]threading.Batch, int) {
	t.Helper()
	jobs := [2]tile.Job{
		{Tile: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 1},
		{Tile: 1, SBCols: 2, SBRows: 1, Offset: 1, Size: 2},
	}
	var batches [2]threading.Batch
	n, err := threading.BuildBatches(batches[:], jobs[:], 1)
	if err != nil {
		t.Fatal(err)
	}
	return jobs, batches, n
}

func benchmarkExecutionWork(b *testing.B) ([2]tile.Job, [2]threading.Batch, int) {
	b.Helper()
	jobs := [2]tile.Job{
		{Tile: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 1},
		{Tile: 1, SBCols: 2, SBRows: 1, Offset: 1, Size: 1},
	}
	var batches [2]threading.Batch
	n, err := threading.BuildBatches(batches[:], jobs[:], 1)
	if err != nil {
		b.Fatal(err)
	}
	return jobs, batches, n
}

func testFrameWorkTileInfo() parser.TileInfo {
	tiles := parser.TileInfo{
		SBCols: 2,
		SBRows: 1,
		Cols:   2,
		Rows:   1,
	}
	tiles.ColStartSB[1] = 1
	tiles.ColStartSB[2] = 2
	tiles.RowStartSB[1] = 1
	return tiles
}

func testFrameWorkSequenceHeader() parser.SequenceHeader {
	return parser.SequenceHeader{
		SeqProfile:           1,
		Use128x128Superblock: true,
		EnableFilterIntra:    true,
		EnableOrderHint:      true,
		OrderHintBits:        5,
		EnableCDEF:           true,
		EnableRestoration:    true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     10,
			SubsamplingX: true,
			SubsamplingY: true,
		},
		FilmGrainParamsPresent: true,
	}
}

func testFrameWorkSequenceContext() threading.FrameWorkSequenceContext {
	return threading.FrameWorkSequenceContextFromHeader(testFrameWorkSequenceHeader())
}

func testFrameWorkGlobalMotion() parser.GlobalMotionParams {
	motion := parser.DefaultGlobalMotionParams()
	motion.Ref[0].Type = parser.GlobalMotionTranslation
	motion.Ref[0].Matrix[0] = 17
	motion.Ref[0].Matrix[1] = -9
	return motion
}

func testFrameWorkFilmGrain() parser.FilmGrainParams {
	return parser.FilmGrainParams{
		ParamsPresent: true,
		Apply:         true,
		Seed:          99,
		BitDepth:      8,
		NumYPoints:    1,
		YPoints:       [parser.MaxFilmGrainYPoints][2]uint8{{16, 32}},
	}
}

func noopFrameWorkBatch(FrameWorkBatch) error {
	return nil
}

func testStillSequenceHeaderPayload(width uint64, height uint64) []byte {
	var w testBitWriter
	w.writeBits(0, 3)        // seq_profile
	w.writeBool(true)        // still_picture
	w.writeBool(true)        // reduced_still_picture_header
	w.writeBits(5, 5)        // seq_level_idx[0]
	w.writeBits(7, 4)        // frame_width_bits_minus_1
	w.writeBits(7, 4)        // frame_height_bits_minus_1
	w.writeBits(width-1, 8)  // max_frame_width_minus_1
	w.writeBits(height-1, 8) // max_frame_height_minus_1
	w.writeBool(false)       // use_128x128_superblock
	w.writeBool(true)        // enable_filter_intra
	w.writeBool(true)        // enable_intra_edge_filter
	w.writeBool(false)       // enable_superres
	w.writeBool(true)        // enable_cdef
	w.writeBool(false)       // enable_restoration
	w.writeBool(false)       // high_bitdepth
	w.writeBool(false)       // mono_chrome
	w.writeBool(false)       // color_description_present_flag
	w.writeBool(false)       // color_range
	w.writeBits(0, 2)        // chroma_sample_position
	w.writeBool(true)        // separate_uv_delta_q
	w.writeBool(false)       // film_grain_params_present
	return w.trailingBits()
}

func reducedStillFrameHeaderPayloadQ(baseQIdx uint8) []byte {
	var w testBitWriter
	w.writeBool(true)  // disable_cdf_update
	w.writeBool(false) // render_and_frame_size_different
	w.writeBool(false) // uniform_tile_spacing_flag
	writeQuantParams(&w, baseQIdx)
	writeZeroSegmentationParams(&w)
	w.writeBool(false) // reduced_tx_set
	return w.bytes()
}

type noopFrameWorkRunner struct{}

func (noopFrameWorkRunner) Run(FrameWorkBatch) error {
	return nil
}

type frameWorkContextRunner struct {
	wantKind     FrameWorkStepKind
	wantSequence threading.FrameWorkSequenceContext
	wantEvent    Event
	wantPayload  byte
	output       *frame.Frame
	err          error
}

func (r *frameWorkContextRunner) Run(ctx FrameWorkBatch) error {
	r.output = ctx.Output
	if ctx.Step.Kind != r.wantKind ||
		len(ctx.References) != 0 ||
		len(ctx.Jobs) != 1 ||
		len(ctx.Payload) != len(r.wantEvent.Unit.Payload) ||
		int(ctx.Jobs[0].Offset) >= len(ctx.Payload) ||
		ctx.Payload[int(ctx.Jobs[0].Offset)] != r.wantPayload ||
		ctx.Sequence != r.wantSequence ||
		ctx.FrameHeader != r.wantEvent.FrameHeader ||
		ctx.FrameSize != r.wantEvent.FrameSize ||
		ctx.TileInfo != r.wantEvent.TileInfo ||
		ctx.Segmentation != r.wantEvent.Segmentation ||
		ctx.LoopFilter != r.wantEvent.LoopFilter ||
		ctx.CDEF != r.wantEvent.CDEF ||
		ctx.Restoration != r.wantEvent.Restoration ||
		ctx.TransformRef != r.wantEvent.TransformRef ||
		ctx.SkipMode != r.wantEvent.SkipMode ||
		ctx.FrameMode != r.wantEvent.FrameMode ||
		ctx.GlobalMotion != r.wantEvent.GlobalMotion ||
		ctx.FilmGrain != r.wantEvent.FilmGrain {
		r.err = errors.New("unexpected frame-work runner context")
		return r.err
	}
	return nil
}

type frameWorkWritingRunner struct {
	order *[2]string
	state *FrameWorkState
	value byte
	err   error
}

func (r *frameWorkWritingRunner) Run(ctx FrameWorkBatch) error {
	if !r.state.Active() || ctx.Output == nil {
		r.err = errors.New("invalid tile runner state")
		return r.err
	}
	r.order[0] = "tile"
	ctx.Output.Y.Pix[0] = r.value
	return nil
}

type frameWorkCheckingPostRunner struct {
	order *[2]string
	refs  *SurfaceReferences
	state *FrameWorkState
	value byte
	err   error
}

func (r *frameWorkCheckingPostRunner) Apply(ctx FrameWorkPostFilterContext) error {
	if !r.state.Active() {
		r.err = errors.New("state inactive during postfilter runner")
		return r.err
	}
	if slot, ok := r.refs.ReferenceSlot(0); ok || slot != -1 {
		r.err = errors.New("published before postfilter runner")
		return r.err
	}
	if r.order[0] != "tile" || ctx.Output == nil || ctx.Output.Y.Pix[0] != 0x11 {
		r.err = errors.New("postfilter runner observed invalid tile output")
		return r.err
	}
	if ctx.Event.Kind != EventFrame || ctx.Step.Kind != FrameWorkStepBegin || !ctx.ExecutedTileWork || ctx.ReferenceCount != 0 {
		r.err = errors.New("invalid postfilter runner context")
		return r.err
	}
	r.order[1] = "post"
	ctx.Output.Y.Pix[0] = r.value
	return nil
}

func testUint16sEqual(a []uint16, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func showExistingWorkEvent(index uint8, frameType parser.FrameType) Event {
	return Event{
		Kind: EventExistingFrame,
		FrameHeader: parser.FrameHeaderPrefix{
			ShowExistingFrame: true,
			ExistingFrameIdx:  index,
		},
		ExistingFrame: parser.ReferenceFrame{FrameType: frameType},
	}
}

func makeFramePoolForSize(width uint32, height uint32, count int) (frame.Pool, error) {
	format := frame.Format{
		Width:        int(width),
		Height:       int(height),
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := frame.RequiredSize(format)
	if err != nil {
		return frame.Pool{}, err
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		return frame.Pool{}, err
	}
	return pool, nil
}

func executeTestFrameWorkStep(step FrameWorkStep, pool *threading.Pool, jobs []tile.Job, batches []threading.Batch, fn threading.BatchFunc) (bool, error) {
	plan, _, hasTile, err := frameWorkStepTilePlan(step)
	if err != nil || !hasTile {
		return false, err
	}
	if err := ExecuteTileWork(plan, pool, jobs, batches, fn); err != nil {
		return false, err
	}
	return plan.JobCount != 0, nil
}

func executeTestFrameWorkStepWithContext(step FrameWorkStep, pool *threading.Pool, output *frame.Frame, references []*frame.Frame, jobs []tile.Job, batches []threading.Batch, fn FrameWorkBatchFunc) (bool, error) {
	return executeFrameWorkStepWithPayload(step, pool, output, references, nil, false, FrameWorkFrameContext{}, false, nil, nil, nil, frameWorkMotionFields{}, nil, nil, nil, nil, jobs, batches, fn)
}

func executeTestFrameWorkStepWithPayload(step FrameWorkStep, pool *threading.Pool, output *frame.Frame, references []*frame.Frame, payload []byte, jobs []tile.Job, batches []threading.Batch, fn FrameWorkBatchFunc) (bool, error) {
	return executeFrameWorkStepWithPayload(step, pool, output, references, payload, true, FrameWorkFrameContext{}, false, nil, nil, nil, frameWorkMotionFields{}, nil, nil, nil, nil, jobs, batches, fn)
}

// testLoopFilterSideDataRunner binds caller-owned loop-filter map and edge-mask
// storage for frames whose only active post-filter is the loop filter.
type testLoopFilterSideDataRunner struct {
	records       []threading.FrameWorkLoopFilterBlockRecord
	masks         []lfmask.FilterMask
	levelCache    []uint8
	loopFilterMap threading.FrameWorkLoopFilterMap
}

func (r *testLoopFilterSideDataRunner) BindFrameWorkSideData(s *FrameWorkState, b FrameWorkBatch) error {
	if !frameWorkLoopFilterActive(b.LoopFilter) {
		return nil
	}
	cols, rows, length, err := b.LoopFilterMapShape()
	if err != nil {
		return err
	}
	sb128w, sb128h, maskCount := threading.FrameWorkLoopFilterMaskShape(cols, rows)
	if len(r.records) < length || len(r.masks) < maskCount || len(r.levelCache) < length {
		return threading.ErrInvalidBatch
	}
	r.loopFilterMap = threading.FrameWorkLoopFilterMap{Records: r.records[:length], Stride: uint16(cols), Rows: uint16(rows)}
	if err := s.SetLoopFilterMap(r.loopFilterMap); err != nil {
		return err
	}
	masks := r.masks[:maskCount]
	levelCache := threading.FrameWorkLoopFilterLevelCacheBind(r.levelCache, cols, rows)
	clear(masks)
	clear(levelCache)
	color := b.Sequence.ColorConfig
	layout := lfmask.Layout{Mono: color.MonoChrome}
	if color.SubsamplingX {
		layout.SSHor = 1
	}
	if color.SubsamplingY {
		layout.SSVer = 1
	}
	return s.SetLoopFilterMasks(threading.FrameWorkLoopFilterMasks{
		Masks:      masks,
		LevelCache: levelCache,
		Cols:       cols,
		Rows:       rows,
		SB128W:     sb128w,
		SB128H:     sb128h,
		Layout:     layout,
		HasChroma:  !color.MonoChrome,
	})
}

// testSupportedPostFilterRunner binds the supported post-filter request from
// its scratch and the side data bound in ctx, then applies it in place for
// publication.
type testSupportedPostFilterRunner struct {
	scratch FrameWorkPostFilterScratch
	size    FrameWorkPostFilterScratchSize
	ctx     FrameWorkPostFilterContext
	result  FrameWorkPostFilterResult
}

func (r *testSupportedPostFilterRunner) Apply(ctx FrameWorkPostFilterContext) error {
	r.ctx = ctx
	var options FrameWorkPostFilterBindOptions
	if ctx.LoopFilterMap != nil {
		options.LoopFilterMap = *ctx.LoopFilterMap
		options.LoopFilterTrustedCoverage = true
	}
	size, err := r.ctx.SupportedPostFilterScratchLen(FrameWorkPostFilterRequest{})
	if err != nil {
		return err
	}
	if ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopFilter) {
		// The loop filter applies through the bound edge masks; the edge-list
		// fallback reports a short buffer before mutating the output.
		size.LoopFilter = FrameWorkLoopFilterPostFilterScratchSize{Edges: len(r.scratch.LoopFilterEdges), Schedule: len(r.scratch.LoopFilterSchedule)}
	}
	req, err := size.BindRequest(options, r.scratch)
	if err != nil {
		return err
	}
	_, result, err := ApplySupportedPostFiltersForPublicationInPlace(&r.ctx, req)
	if err != nil {
		return err
	}
	r.size = size
	r.result = result
	return r.ctx.RequirePublishablePostFilterOutput()
}
