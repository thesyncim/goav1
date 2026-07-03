package decoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

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
	layout := frameWorkLoopFilterMaskLayout(color)
	hasChroma := !color.MonoChrome
	sb128w, sb128h, maskCount := threading.FrameWorkLoopFilterMaskShape(cols, rows)
	handle := &threading.FrameWorkLoopFilterMasks{
		Masks:      make([]lfmask.FilterMask, maskCount),
		LevelCache: make([][4]uint8, cols*rows),
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
			auv = aUV[bx>>layout.SSHor:]
			luv = lUV[by4>>layout.SSVer:]
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
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			v := (x*37 + y*53 + (x^y)*17 + salt*101) & 0xffff
			v &= maxv
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
	resB, err := ctxB.ApplyLoopFilterEdgesFromMasks(masks, filterMap)
	if err != nil {
		t.Fatalf("mask apply: %v", err)
	}
	if resA.Applied == 0 {
		t.Fatalf("degenerate: edge-list apply filtered no edges")
	}
	// resA counts run-merged edges while resB counts individual 4x4 cells, so
	// the counters differ by construction; the byte-identical planes are the
	// real gate. A non-zero mask count guards against a silently empty apply.
	if resB.Applied == 0 {
		t.Fatalf("degenerate: mask apply filtered no edges")
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

var _ = loopfilter.PlaneY
