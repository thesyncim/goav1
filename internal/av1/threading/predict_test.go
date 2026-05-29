package threading

import (
	"errors"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkBatchPredictBlockLumaIntraDC(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	ctx := testIntraPredictionBatch(output)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 50)
	}

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != 30 {
				t.Fatalf("sample(%d,%d)=%d want 30", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaFilterIntra(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	ctx := testIntraPredictionBatch(output)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(20+x))
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16(40+y))
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 33)

	edges := testFrameWorkIntraEdges(output, output.Y, 16, 16, 16, 16)
	wantPix := make([]byte, 16*16)
	want := frame.Plane{Pix: wantPix, Stride: 16, Width: 16, Height: 16}
	if err := prediction.PredictFilterIntraPlaneBlock(want, 1, 8, 0, 0, 16, 16, prediction.FilterIntraModePaeth, edges); err != nil {
		t.Fatal(err)
	}

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction.FilterIntraValid = true
	visit.Prediction.FilterIntraMode = tile.FilterIntraModePaeth
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+x, 16+y)
			if want := uint16(wantPix[y*16+x]); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", 16+x, 16+y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraStaticModes(t *testing.T) {
	for _, tt := range []struct {
		name string
		mode tile.IntraMode
		want func(x int, y int) uint16
	}{
		{
			name: "vertical",
			mode: tile.IntraModeVertical,
			want: func(x int, _ int) uint16 { return uint16(200 + x - 16) },
		},
		{
			name: "horizontal",
			mode: tile.IntraModeHorizontal,
			want: func(_ int, y int) uint16 { return uint16(300 + y - 16) },
		},
		{
			name: "paeth",
			mode: tile.IntraModePaeth,
			want: func(_ int, y int) uint16 { return uint16(300 + y - 16) },
		},
		{
			name: "smooth",
			mode: tile.IntraModeSmooth,
			want: nil,
		},
		{
			name: "smooth-vertical",
			mode: tile.IntraModeSmoothVertical,
			want: nil,
		},
		{
			name: "smooth-horizontal",
			mode: tile.IntraModeSmoothHorizontal,
			want: nil,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
			ctx := testIntraPredictionBatch(output)
			for x := 16; x < 32; x++ {
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(200+x-16))
			}
			for y := 16; y < 32; y++ {
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16(300+y-16))
			}
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 180)

			var scratch FrameWorkIntraPredictionScratch
			visit := testIntraPredictionVisit(tt.mode)
			if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
				t.Fatal(err)
			}
			for y := 16; y < 32; y++ {
				for x := 16; x < 32; x++ {
					got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
					if got > 1023 {
						t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
					}
					if tt.want != nil {
						if want := tt.want(x, y); got != want {
							t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
						}
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraClipsFrameEdge(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 320, Height: 180, BitDepth: 8, MonoChrome: true, Align: 64})
	testFillFrame(output, 7)
	ctx := testIntraPredictionBatch(output)
	ctx.Jobs = []tile.Job{{SBCols: 5, SBRows: 3}}
	for x := 112; x < 128; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 175, uint16(30+x-112))
	}

	visit := testIntraPredictionVisit(tile.IntraModeVertical)
	visit.Block = tile.BlockVisit{
		MICol: 28, MIRow: 44, MIColEnd: 32, MIRowEnd: 46,
		X4: 12, Y4: 12, Size: tile.BlockSize16x8, VisibleW4: 4, VisibleH4: 2,
		HaveTop: true, HaveLeft: true,
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 176; y < 180; y++ {
		for x := 112; x < 128; x++ {
			want := uint16(30 + x - 112)
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterClipsFrameEdge(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 320, Height: 180, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(output, 7)
	testFillFrame(reference, 0x55)
	ctx := testInterPredictionBatch(output, reference)
	ctx.Jobs = []tile.Job{{SBCols: 5, SBRows: 3}}

	visit := testInterPredictionVisit(motion.Vector{})
	visit.Block = tile.BlockVisit{
		MICol: 0, MIRow: 44, MIColEnd: 16, MIRowEnd: 46,
		X4: 0, Y4: 12, Size: tile.BlockSize64x16, VisibleW4: 16, VisibleH4: 2,
		HaveTop: true,
	}
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 176; y < 180; y++ {
		for x := 0; x < 64; x++ {
			if got := output.Y.Pix[y*output.Y.Stride+x]; got != 0x55 {
				t.Fatalf("sample(%d,%d)=%d want 0x55", x, y, got)
			}
		}
	}
}

// TestFrameWorkClipVisiblePixelsToWindow covers the boundary cases that show
// up in the 34x34 libaom-extended fast-suite vector: MI-aligned blocks at the
// right/bottom edge whose nominal extent (e.g. 4x4 or 8x8 transforms aligned
// to MI col 8 in a 34-pixel-wide luma plane) overshoots the coded frame and
// must clip down to the writable extent.
//
// When ClipWidth/ClipHeight are zero, the clamp falls back to Width/Height
// (visible coded-frame edge) — this is the older single-extent behaviour and
// is what the test harness in TestFrameWorkBatchPredictBlockLumaIntraDC and
// friends still relies on.
//
// When ClipWidth/ClipHeight are set, the clamp clamps to the MI-aligned
// writable extent (xd->mi_params.mi_cols * MI_SIZE in libaom; region.MIColEnd
// * 4 in goav1). This is the regression behaviour pinned by the structural
// past-visible-write fix: blocks that straddle the visible boundary keep
// writing into the past-visible padding so later blocks see the same
// neighbors libaom does. Frame-edge MD5 / output is still computed over the
// visible Width/Height range; the past-visible writes land in the stride
// padding that the MD5 harness ignores.
func TestFrameWorkClipVisiblePixelsToWindow(t *testing.T) {
	luma := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 34, Height: 34}
	chroma := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 17, Height: 17}
	// MI-aligned luma extent for a 34x34 frame: mi_cols = ((34+7)>>3)<<1 = 10,
	// luma write extent = 10*4 = 40. Chroma 4:2:0 = 20.
	lumaAligned := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 34, Height: 34, ClipWidth: 40, ClipHeight: 40}
	chromaAligned := FrameWorkPlaneRegion{X: 0, Y: 0, Width: 17, Height: 17, ClipWidth: 20, ClipHeight: 20}
	tests := []struct {
		name         string
		window       FrameWorkPlaneRegion
		x, y, w, h   int
		wantW, wantH int
		wantOK       bool
	}{
		// Visible-extent fallback (legacy semantics).
		{"luma corner 8x8 visible 2x2", luma, 32, 32, 8, 8, 2, 2, true},
		{"luma right 4x8 visible 2x8", luma, 32, 8, 4, 8, 2, 8, true},
		{"luma right 4x4 visible 2x4", luma, 32, 16, 4, 4, 2, 4, true},
		{"luma bottom 8x4 visible 8x2", luma, 16, 32, 8, 4, 8, 2, true},
		{"luma bottom 16x8 visible 16x2", luma, 0, 32, 16, 8, 16, 2, true},
		{"luma right-most pixel 1x1 visible 1x1", luma, 33, 33, 1, 1, 1, 1, true},
		{"luma 4x4 at MI col 9 entirely past visible", luma, 36, 8, 4, 4, 0, 0, false},
		{"luma 8x8 at MI col 9 entirely past visible", luma, 36, 0, 8, 8, 0, 0, false},
		{"luma block at exact right boundary", luma, 34, 0, 4, 4, 0, 0, false},
		{"chroma corner 4x4 visible 1x1", chroma, 16, 16, 4, 4, 1, 1, true},
		{"chroma right 2x4 visible 1x4", chroma, 16, 8, 2, 4, 1, 4, true},
		{"chroma bottom 4x2 visible 4x1", chroma, 8, 16, 4, 2, 4, 1, true},
		{"chroma 2x4 entirely past visible", chroma, 18, 0, 2, 4, 0, 0, false},
		{"zero width rejects", luma, 0, 0, 0, 4, 0, 0, false},
		{"zero height rejects", luma, 0, 0, 4, 0, 0, 0, false},
		{"negative origin rejects", luma, -1, 0, 4, 4, 0, 0, false},
		// Aligned-extent semantics: ClipWidth/ClipHeight > Width/Height.
		// A block whose nominal extent fits inside the MI-aligned write
		// extent must return its full size, not the smaller visible clip.
		{"aligned luma corner 8x8 full 8x8", lumaAligned, 32, 32, 8, 8, 8, 8, true},
		{"aligned luma right 4x8 full 4x8", lumaAligned, 32, 8, 4, 8, 4, 8, true},
		{"aligned luma right 4x4 full 4x4", lumaAligned, 32, 16, 4, 4, 4, 4, true},
		{"aligned luma bottom 16x8 full 16x8", lumaAligned, 0, 32, 16, 8, 16, 8, true},
		{"aligned luma MI col 9 4x4 full 4x4", lumaAligned, 36, 8, 4, 4, 4, 4, true},
		{"aligned luma MI col 9 8x8 full 4x8", lumaAligned, 36, 0, 8, 8, 4, 8, true},
		{"aligned luma at exact aligned boundary", lumaAligned, 40, 0, 4, 4, 0, 0, false},
		{"aligned chroma corner 4x4 full 4x4", chromaAligned, 16, 16, 4, 4, 4, 4, true},
		{"aligned chroma right 2x4 full 2x4", chromaAligned, 16, 8, 2, 4, 2, 4, true},
		{"aligned chroma bottom 4x2 full 4x2", chromaAligned, 8, 16, 4, 2, 4, 2, true},
		{"aligned chroma 2x4 at MI col 9 full 2x4", chromaAligned, 18, 0, 2, 4, 2, 4, true},
		{"aligned chroma block at exact aligned boundary", chromaAligned, 20, 0, 2, 4, 0, 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotW, gotH, ok := frameWorkClipVisiblePixelsToWindow(tc.window, tc.x, tc.y, tc.w, tc.h)
			if ok != tc.wantOK || gotW != tc.wantW || gotH != tc.wantH {
				t.Fatalf("clip(x=%d,y=%d,w=%d,h=%d) = (%d,%d,%v) want (%d,%d,%v)",
					tc.x, tc.y, tc.w, tc.h, gotW, gotH, ok, tc.wantW, tc.wantH, tc.wantOK)
			}
		})
	}
}

// TestFrameWorkJobOutputPlaneClipExtentMatchesMIAlignment pins the MI-aligned
// writable extent that JobOutputPlane records on a 34x34 frame's output
// region. The visible Width/Height stay at the coded-frame edge (34 / 17 for
// chroma 4:2:0) but ClipWidth/ClipHeight extend to mi_cols * MI_SIZE
// (40 luma, 20 chroma). The window's Pix slice extends through
// (ClipHeight-1)*Stride + ClipRowBytes so AV1 prediction and residual writes
// past the visible edge stay within the underlying plane buffer.
//
// This is the regression behaviour locked in for the structural
// past-visible-write fix: predictors that previously zero-clipped at col 34
// (luma) / 17 (chroma) now write up to col 39 / 19, exactly mirroring
// libaom's xd->mi_params.mi_cols * MI_SIZE write boundary.
func TestFrameWorkJobOutputPlaneClipExtentMatchesMIAlignment(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        34,
		Height:       34,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	ctx := FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
			}),
			FrameSize: parser.FrameSize{CodedWidth: 34, Height: 34},
		},
		Jobs: []tile.Job{
			{Tile: 0, Row: 0, Col: 0, SBX: 0, SBY: 0, SBCols: 1, SBRows: 1},
		},
	}
	y, err := ctx.JobOutputPlane(0, FrameWorkPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if y.X != 0 || y.Y != 0 || y.Width != 34 || y.Height != 34 ||
		y.ClipWidth != 40 || y.ClipHeight != 34 || y.ClipRowBytes != 40 {
		t.Fatalf("Y plane region=%+v", y)
	}
	// Luma plane.Height = 34 caps ClipHeight at 34 (mi_aligned 40 > 34, so
	// the buffer's visible Height bounds win). Past-visible writes land in
	// the stride padding (Stride bytes per row).
	if len(y.Pix) != (y.ClipHeight-1)*y.Stride+y.ClipRowBytes {
		t.Fatalf("Y len=%d region=%+v", len(y.Pix), y)
	}
	if y.Stride < y.ClipRowBytes {
		t.Fatalf("Y Stride=%d ClipRowBytes=%d", y.Stride, y.ClipRowBytes)
	}

	u, err := ctx.JobOutputPlane(0, FrameWorkPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	if u.X != 0 || u.Y != 0 || u.Width != 17 || u.Height != 17 ||
		u.ClipWidth != 20 || u.ClipHeight != 17 || u.ClipRowBytes != 20 {
		t.Fatalf("U plane region=%+v", u)
	}
	if len(u.Pix) != (u.ClipHeight-1)*u.Stride+u.ClipRowBytes {
		t.Fatalf("U len=%d region=%+v", len(u.Pix), u)
	}
	v, err := ctx.JobOutputPlane(0, FrameWorkPlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if v.X != 0 || v.Y != 0 || v.Width != 17 || v.Height != 17 ||
		v.ClipWidth != 20 || v.ClipHeight != 17 || v.ClipRowBytes != 20 {
		t.Fatalf("V plane region=%+v", v)
	}
	// Verify the clip helper accepts past-visible writes: a 4x4 luma block at
	// MI col 8 (luma pixel 32) of a 34-wide frame returns the full 4x4
	// (writable past visible) instead of clipping to (2,4).
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(y, 32, 0, 4, 4); !ok || w != 4 || h != 4 {
		t.Fatalf("luma MI col 8 4x4 clip=(%d,%d,%v) want (4,4,true)", w, h, ok)
	}
	// And a 4x4 luma block at MI col 9 (luma pixel 36) returns (4,4) instead
	// of failing (its origin is within the aligned write extent of 40).
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(y, 36, 0, 4, 4); !ok || w != 4 || h != 4 {
		t.Fatalf("luma MI col 9 4x4 clip=(%d,%d,%v) want (4,4,true)", w, h, ok)
	}
	// A block past the aligned extent still fails.
	if _, _, ok := frameWorkClipVisiblePixelsToWindow(y, 40, 0, 4, 4); ok {
		t.Fatalf("luma at aligned boundary should reject")
	}
	// Chroma equivalents: 2x2 block at chroma MI col 8 (chroma pixel 16) and
	// MI col 9 (chroma pixel 18) both stay inside the aligned chroma extent
	// of 20.
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(u, 16, 0, 4, 4); !ok || w != 4 || h != 4 {
		t.Fatalf("chroma MI col 8 4x4 clip=(%d,%d,%v) want (4,4,true)", w, h, ok)
	}
	if w, h, ok := frameWorkClipVisiblePixelsToWindow(u, 18, 0, 2, 4); !ok || w != 2 || h != 4 {
		t.Fatalf("chroma MI col 9 2x4 clip=(%d,%d,%v) want (2,4,true)", w, h, ok)
	}
}

// TestFrameWorkBatchPredictBlockLumaIntraWritesPastVisibleRightEdge pins the
// MI-aligned past-visible write behaviour locked in by a86e729 on a 34x34
// frame. An 8x8 DC intra block whose MI footprint covers luma columns 32..39
// (block at MI col 8) has only the leftmost two columns (32 and 33) inside
// the coded-frame edge; the remaining six columns (34..39) are past visible
// but inside the MI-aligned writable extent (mi_cols*MI_SIZE = 40).
//
// libaom writes the entire 8x8 transform regardless of where the visible
// boundary lands; later blocks read the past-visible samples as predictor
// neighbors. goav1 mirrors that by extending JobOutputPlane's window through
// the stride padding so the prediction kernel can write all 8 columns. This
// regression test confirms that the prediction call:
//
//  1. Succeeds at the right-edge MI position (cols 32..39 are inside ClipWidth=40).
//  2. Writes a uniform DC value across columns 32..39 of each row, including
//     the past-visible columns 34..39 (which previously stayed zero-clipped
//     under the visible-only write path and caused downstream blocks to see
//     uninitialised stride padding when fetching above neighbors at row 7+).
//
// The frame is allocated with Align=32 (matching the libaom oracle harness's
// frameFormatFromEvent stride alignment), so Y stride = 64 bytes, leaving 30
// bytes of past-visible stride padding per row that the write fills.
func TestFrameWorkBatchPredictBlockLumaIntraWritesPastVisibleRightEdge(t *testing.T) {
	output := testBatchFrame(t, frame.Format{
		Width:        34,
		Height:       34,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	})
	testFillFrame(output, 0)

	// Seed the left-neighbor column at x=31 with a fixed value so the
	// 8x8 DC block at (col 32, row 0) has predictable Above (missing,
	// filled from dst[31, 0]) and Left (col 31 rows 0..7) neighbors.
	// DC = (8*seed + 8*seed) / 16 = seed.
	const seed uint16 = 96
	for y := 0; y < 8; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 31, y, seed)
	}

	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			MonoChrome:   false,
			SubsamplingX: output.Format.SubsamplingX,
			SubsamplingY: output.Format.SubsamplingY,
		},
	})
	ctx.FrameSize = parser.FrameSize{CodedWidth: 34, Height: 34}
	ctx.Jobs = []tile.Job{{SBCols: 1, SBRows: 1}}

	visit := tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 8, MIRow: 0, MIColEnd: 10, MIRowEnd: 2,
			X4: 0, Y4: 0, Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2,
			HaveTop: false, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:    true,
			Intra:    true,
			LumaMode: tile.IntraModeDC,
		},
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
		t.Fatalf("PredictBlockLumaIntra at right-edge MI col 8: %v", err)
	}

	// All 8 rows x 8 cols of the block (cols 32..39, rows 0..7) must hold
	// the DC value, including past-visible cols 34..39. A regression that
	// re-clips writes to the visible 2-col strip would leave cols 34..39
	// at their initial 0, breaking downstream blocks that read row 7
	// cols 32..39 as above neighbors.
	//
	// Read raw bytes by stride (not via frameWorkLoadSample which bounds-
	// checks against plane.Width=34) so we can verify past-visible writes
	// in the stride padding.
	for y := 0; y < 8; y++ {
		row := y * output.Y.Stride
		for x := 32; x < 40; x++ {
			got := uint16(output.Y.Pix[row+x])
			if got != seed {
				t.Fatalf("Y(%d,%d)=%d want %d (past-visible write missing or wrong)", x, y, got, seed)
			}
		}
	}
}

// TestFrameWorkPlaneBlockStartsBeyondOutput covers the rounded-up MI grid
// short-circuit: a 34x34 frame allocates MI cols 0..9 (rounded to a multiple
// of 8), so partition walks can address blocks starting at MI col 9 (luma
// pixel 36) which has zero visible samples. The clip path returns ok=false
// for those blocks; this helper distinguishes that genuinely-out-of-bounds
// case from negative/origin-prefix coordinates so callers can skip silently
// instead of failing the batch.
func TestFrameWorkPlaneBlockStartsBeyondOutput(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 34, Height: 34, BitDepth: 8, Align: 64, SubsamplingX: true, SubsamplingY: true})
	tests := []struct {
		name  string
		plane FrameWorkPlane
		x, y  int
		want  bool
	}{
		{"luma origin", FrameWorkPlaneY, 0, 0, false},
		{"luma last visible (33, 33)", FrameWorkPlaneY, 33, 33, false},
		{"luma at right edge (34, 0)", FrameWorkPlaneY, 34, 0, true},
		{"luma at bottom edge (0, 34)", FrameWorkPlaneY, 0, 34, true},
		{"luma at MI col 9 (36, 8)", FrameWorkPlaneY, 36, 8, true},
		{"luma at MI row 9 (8, 36)", FrameWorkPlaneY, 8, 36, true},
		{"luma negative x rejected (not beyond)", FrameWorkPlaneY, -1, 0, false},
		{"luma negative y rejected (not beyond)", FrameWorkPlaneY, 0, -1, false},
		{"chroma U origin", FrameWorkPlaneU, 0, 0, false},
		{"chroma U last visible (16, 16)", FrameWorkPlaneU, 16, 16, false},
		{"chroma U at right edge (17, 0)", FrameWorkPlaneU, 17, 0, true},
		{"chroma U at MI col 9 (18, 4)", FrameWorkPlaneU, 18, 4, true},
		{"chroma V last visible (16, 16)", FrameWorkPlaneV, 16, 16, false},
		{"chroma V at bottom edge (8, 17)", FrameWorkPlaneV, 8, 17, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := frameWorkPlaneBlockStartsBeyondOutput(output, tc.plane, tc.x, tc.y); got != tc.want {
				t.Fatalf("plane=%d (%d,%d) = %v want %v", tc.plane, tc.x, tc.y, got, tc.want)
			}
		})
	}
	if got := frameWorkPlaneBlockStartsBeyondOutput(nil, FrameWorkPlaneY, 0, 0); got != false {
		t.Fatalf("nil output: got=%v want false", got)
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalKnownVectors(t *testing.T) {
	tests := []struct {
		name string
		mode tile.IntraMode
		seed func(*frame.Frame)
		want []uint16
	}{
		{
			name: "d45",
			mode: tile.IntraModeD45,
			seed: func(output *frame.Frame) {
				for i, sample := range []uint16{10, 20, 30, 40, 50, 60, 70, 80} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+i, 15, sample)
				}
			},
			want: []uint16{
				20, 30, 40, 50,
				30, 40, 50, 60,
				40, 50, 60, 70,
				50, 60, 70, 80,
			},
		},
		{
			name: "d135",
			mode: tile.IntraModeD135,
			seed: func(output *frame.Frame) {
				for i, sample := range []uint16{21, 31, 41, 51} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+i, 15, sample)
				}
				for i, sample := range []uint16{101, 111, 121, 131} {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 16+i, sample)
				}
				setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 11)
			},
			want: []uint16{
				11, 21, 31, 41,
				101, 11, 21, 31,
				111, 101, 11, 21,
				121, 111, 101, 11,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
			ctx := testIntraPredictionBatch(output)
			tt.seed(output)

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, testIntraPrediction4x4Visit(tt.mode), &scratch); err != nil {
				t.Fatal(err)
			}
			got := make([]uint16, 0, 16)
			for y := 16; y < 20; y++ {
				for x := 16; x < 20; x++ {
					got = append(got, frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y))
				}
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("prediction=%v want %v", got, tt.want)
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalModes(t *testing.T) {
	for _, tt := range []struct {
		name string
		mode tile.IntraMode
	}{
		{name: "d45", mode: tile.IntraModeD45},
		{name: "d67", mode: tile.IntraModeD67},
		{name: "d113", mode: tile.IntraModeD113},
		{name: "d135", mode: tile.IntraModeD135},
		{name: "d157", mode: tile.IntraModeD157},
		{name: "d203", mode: tile.IntraModeD203},
	} {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 96, Height: 96, BitDepth: 10, Align: 128})
			ctx := testIntraPredictionBatch(output)
			seedFrameWorkDirectionalEdges(output, 0x3ff, 19, 37, 101)

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, testIntraPredictionVisit(tt.mode), &scratch); err != nil {
				t.Fatal(err)
			}
			for y := 16; y < 32; y++ {
				for x := 16; x < 32; x++ {
					if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got > 0x3ff {
						t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaDirectionalIntraEdgeUpsample(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence.EnableIntraEdgeFilter = true
	leftSamples := []uint16{97, 100, 103, 107, 109, 111, 112, 113}
	for y, sample := range leftSamples {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, sample)
	}
	visit := testIntraPrediction4x4Visit(tile.IntraModeD203)
	visit.Block.MIRow = 0
	visit.Block.MIRowEnd = 1
	visit.Block.Y4 = 0
	visit.Block.HaveTop = false
	visit.Block.HaveLeft = true

	left := make([]uint16, frameWorkDirectionalEdgeSamples)
	origin := frameWorkDirectionalEdgeOrigin
	left[origin-1] = leftSamples[0]
	copy(left[origin:], leftSamples)
	scratch := make([]uint16, frameWorkIntraEdgeScratchSamples)
	if err := prediction.UpsampleIntraEdge(left, origin, 8, scratch, 8); err != nil {
		t.Fatal(err)
	}
	wantPix := make([]byte, 16)
	want := frame.Plane{Pix: wantPix, Stride: 4, Width: 4, Height: 4}
	edges := prediction.DirectionalEdges{
		Left:         left,
		LeftOrigin:   origin,
		UpsampleLeft: true,
	}
	if err := prediction.PredictDirectionalIntraPlaneBlock(want, 1, 8, 0, 0, 4, 4, 203, edges); err != nil {
		t.Fatal(err)
	}

	var predScratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockLumaIntra(0, visit, &predScratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16+x, y)
			if want := uint16(wantPix[y*4+x]); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", 16+x, y, got, want)
			}
		}
	}
}

func TestFrameWorkLumaTransformDirectionalExtendedEdgesMatchesLibaomCases(t *testing.T) {
	tests := []struct {
		name           string
		block          tile.BlockVisit
		miColEnd       uint32
		miRowEnd       uint32
		absX           int
		absY           int
		width          int
		height         int
		wantTopRight   bool
		wantBottomLeft bool
	}{
		{
			name: "d203 right-half 8x4 cannot use bottom-left",
			block: tile.BlockVisit{
				MICol: 56, MIRow: 0, MIColEnd: 58, MIRowEnd: 1,
				Size: tile.BlockSize8x4, VisibleW4: 2, VisibleH4: 1, HaveLeft: true,
			},
			absX: 228, absY: 0, width: 4, height: 4,
		},
		{
			name: "d203 left-half 8x4 table allows bottom-left",
			block: tile.BlockVisit{
				MICol: 56, MIRow: 0, MIColEnd: 58, MIRowEnd: 1,
				Size: tile.BlockSize8x4, VisibleW4: 2, VisibleH4: 1, HaveLeft: true,
			},
			absX: 224, absY: 0, width: 4, height: 4, wantBottomLeft: true,
		},
		{
			name: "d203 4x8 table denies bottom-left",
			block: tile.BlockVisit{
				MICol: 66, MIRow: 0, MIColEnd: 67, MIRowEnd: 2,
				Size: tile.BlockSize4x8, VisibleW4: 1, VisibleH4: 2, HaveLeft: true,
			},
			absX: 264, absY: 4, width: 4, height: 4,
		},
		{
			name: "d45 right-half 8x8 cannot use top-right",
			block: tile.BlockVisit{
				MICol: 74, MIRow: 2, MIColEnd: 76, MIRowEnd: 4,
				Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2, HaveTop: true, HaveLeft: true,
			},
			absX: 300, absY: 8, width: 4, height: 4,
		},
		{
			name: "d45 left-half 8x8 has internal top-right",
			block: tile.BlockVisit{
				MICol: 74, MIRow: 2, MIColEnd: 76, MIRowEnd: 4,
				Size: tile.BlockSize8x8, VisibleW4: 2, VisibleH4: 2, HaveTop: true, HaveLeft: true,
			},
			absX: 296, absY: 8, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "quantizer00 d67 4x4 table allows top-right",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			miColEnd: 88, miRowEnd: 72,
			absX: 32, absY: 8, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "quantizer00 d67 4x4 denies top-right at tile edge",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			miColEnd: 9, miRowEnd: 72,
			absX: 32, absY: 8, width: 4, height: 4, wantBottomLeft: true,
		},
		{
			name: "quantizer00 internal transform top edge allows top-right",
			block: func() tile.BlockVisit {
				parent := tile.BlockVisit{
					MICol: 24, MIRow: 0, MIColEnd: 28, MIRowEnd: 2,
					X4: 24, Y4: 0, Partition: tile.PartitionTBottomSplit, Size: tile.BlockSize16x8,
					VisibleW4: 4, VisibleH4: 2, HaveLeft: true,
				}
				return frameWorkPredictionTransformEdgeBlock(parent, parent.X4, parent.Y4, 24, 1)
			}(),
			miColEnd: 88, miRowEnd: 72,
			absX: 96, absY: 4, width: 4, height: 4, wantTopRight: true, wantBottomLeft: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			miColEnd := tt.miColEnd
			if miColEnd == 0 {
				miColEnd = 88
			}
			miRowEnd := tt.miRowEnd
			if miRowEnd == 0 {
				miRowEnd = 72
			}
			gotTopRight, gotBottomLeft := frameWorkLumaDirectionalExtendedEdges(tt.block, 32, miColEnd, miRowEnd, tt.absX, tt.absY, tt.width, tt.height)
			if gotTopRight != tt.wantTopRight || gotBottomLeft != tt.wantBottomLeft {
				t.Fatalf("topRight=%v bottomLeft=%v want %v/%v", gotTopRight, gotBottomLeft, tt.wantTopRight, tt.wantBottomLeft)
			}
		})
	}
}

func TestFrameWorkDirectionalAvailabilityTablesCoverBlockSizes(t *testing.T) {
	sizes := []tile.BlockSize{
		tile.BlockSize4x4, tile.BlockSize4x8, tile.BlockSize8x4, tile.BlockSize8x8,
		tile.BlockSize8x16, tile.BlockSize16x8, tile.BlockSize16x16,
		tile.BlockSize16x32, tile.BlockSize32x16, tile.BlockSize32x32,
		tile.BlockSize32x64, tile.BlockSize64x32, tile.BlockSize64x64,
		tile.BlockSize64x128, tile.BlockSize128x64, tile.BlockSize128x128,
		tile.BlockSize4x16, tile.BlockSize16x4, tile.BlockSize8x32,
		tile.BlockSize32x8, tile.BlockSize16x64, tile.BlockSize64x16,
	}
	for _, size := range sizes {
		if table := frameWorkTopRightAvailabilityTable(tile.PartitionNone, size); len(table) == 0 {
			t.Fatalf("missing top-right table for size %d", size)
		}
		if table := frameWorkBottomLeftAvailabilityTable(tile.PartitionNone, size); len(table) == 0 {
			t.Fatalf("missing bottom-left table for size %d", size)
		}
	}
	if table := frameWorkTopRightAvailabilityTable(tile.PartitionTLeftSplit, tile.BlockSize8x8); len(table) == 0 || table[2] != 0 {
		t.Fatalf("mixed-vertical top-right table not selected: %v", table)
	}
	if table := frameWorkBottomLeftAvailabilityTable(tile.PartitionTRightSplit, tile.BlockSize8x8); len(table) == 0 || table[0] != 254 {
		t.Fatalf("mixed-vertical bottom-left table not selected: %v", table)
	}
}

func TestFrameWorkFillDirectionalAboveCapsTopRightToPrimaryWidth(t *testing.T) {
	// libaom's intra prediction loads at most primaryWidth additional top-right
	// samples (n_topright_px = min(txwpx, xr)) and extends the rest from the last
	// real sample. The fill helper must mirror that even when allow_top_right is
	// true so directional predictors see the same edge buffer libaom does.
	const W, H = 64, 64
	format := frame.Format{Width: W, Height: H, BitDepth: 8, Align: 64}
	output := testBatchFrame(t, format)
	for x := 0; x < W; x++ {
		// Distinct per-column markers above the block being predicted.
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16(x+1))
	}
	dst := frame.Plane{Pix: output.Y.Pix, Stride: output.Y.Stride, Width: W, Height: H}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		Size:    tile.BlockSize4x8,
		HaveTop: true,
	}
	const primaryWidth = 4
	if err := frameWorkFillDirectionalAbove(dst, 1, 8, 16, 16, 0, primaryWidth+8-1, primaryWidth, true, block, &scratch, 0); err != nil {
		t.Fatalf("fill: %v", err)
	}
	// Indices 0..3 read columns 16..19 verbatim.
	want := []uint16{17, 18, 19, 20}
	for i, v := range want {
		if got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]; got != v {
			t.Fatalf("primary above[%d]=%d want %d", i, got, v)
		}
	}
	// Indices 4..7 are the top-right extension (n_topright_px = primaryWidth).
	for i := 4; i < 8; i++ {
		if got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]; got != uint16(16+i+1) {
			t.Fatalf("topright above[%d]=%d want %d", i, got, 16+i+1)
		}
	}
	// Indices 8..10 must extend from the last real sample at column 16+7=23, not
	// pull additional decoded samples from columns 24+.
	last := uint16(16 + 7 + 1)
	for i := 8; i < primaryWidth+8-1; i++ {
		if got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]; got != last {
			t.Fatalf("extension above[%d]=%d want %d", i, got, last)
		}
	}
}

// TestFrameWorkFillDirectionalAboveTruncatesToVisibleRightEdge pins the
// behavior of frameWorkFillDirectionalAbove when a block straddles the
// visible right edge. libaom's intra predictor caps the real-sample range
// to n_top_px = min(txwpx, xr + txwpx); past-visible columns are replicated
// from the last visible neighbor sample, never read from MI-padding bytes
// (which carry past-visible writes from earlier blocks that libaom never
// sees). The 34x34 libaom-extended vector exposed this divergence: an 8x8
// transform at x=32 in a 34-wide plane had n_top_px=2 in libaom but goav1
// previously loaded all 8 cols from MI-padding, yielding a different above
// row and a downstream MD5 mismatch.
func TestFrameWorkFillDirectionalAboveTruncatesToVisibleRightEdge(t *testing.T) {
	// Allocate a 40-wide buffer so we can seed both the visible cols (0..33)
	// and the MI-padding cols (34..39) with distinct values, then verify the
	// fill ignores the MI-padding cols when visibleW=34.
	const Stride, H = 40, 16
	const visibleW = 34
	pix := make([]byte, Stride*H)
	dst := frame.Plane{Pix: pix, Stride: Stride, Width: Stride, Height: H}
	// Above row (y=7): distinct per-column markers so the fill output records
	// exactly which cols were loaded. Marker(c) = c + 1.
	for c := 0; c < Stride; c++ {
		pix[7*Stride+c] = byte(c + 1)
	}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		Size:    tile.BlockSize8x8,
		HaveTop: true,
	}
	const x = 32
	const primaryWidth = 8
	// Range covers primary (0..primaryWidth-1) and top-right extension
	// (primaryWidth..primaryWidth+primaryWidth-1) per libaom's directional
	// above buffer layout.
	maxIndex := primaryWidth + primaryWidth - 1
	if err := frameWorkFillDirectionalAbove(dst, 1, 8, x, 8, 0, maxIndex, primaryWidth, true, block, &scratch, visibleW); err != nil {
		t.Fatalf("fill: %v", err)
	}
	// libaom: n_top_px = min(8, (34 - 32 - 8) + 8) = min(8, 2) = 2.
	//         n_topright_px = min(8, 34 - 32 - 8) = min(8, -6) -> 0.
	// Real samples land at cols 32, 33 (markers 33, 34); the remaining
	// primary slots (i=2..7) and the entire top-right extension
	// (i=8..15) must replicate col 33's marker (=34).
	want := []uint16{33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34}
	for i, w := range want {
		got := scratch.Above[frameWorkDirectionalEdgeOrigin+i]
		if got != w {
			t.Fatalf("above[%d]=%d want %d (visibleW=%d should cap real loads to cols 32..33)", i, got, w, visibleW)
		}
	}
}

// TestFrameWorkIntraPredictionEdgesWithExtentTruncatesToVisibleRightEdge
// pins the equivalent libaom contract for the non-directional edge builder:
// the above buffer copies n_top_px real samples and replicates the last
// real one for the remaining edgeWidth-n_top_px slots. The 34x34 right-edge
// 8x8 DC / Smooth / Paeth block must see Above[0..1] = real cols and
// Above[2..7] = replicated last visible neighbor.
func TestFrameWorkIntraPredictionEdgesWithExtentTruncatesToVisibleRightEdge(t *testing.T) {
	const Stride, H = 40, 16
	const visibleW = 34
	pix := make([]byte, Stride*H)
	dst := frame.Plane{Pix: pix, Stride: Stride, Width: Stride, Height: H}
	for c := 0; c < Stride; c++ {
		// Distinct marker per column on the above row (y=7) and on the
		// left column (x=31) so we can assert which neighbors were read.
		pix[7*Stride+c] = byte(c + 1)
	}
	for r := 0; r < H; r++ {
		pix[r*Stride+31] = byte(r + 100)
	}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		Size:     tile.BlockSize8x8,
		HaveTop:  true,
		HaveLeft: true,
	}
	const x, y = 32, 8
	edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, 1, 8, x, y, 8, 8, 8, 8, visibleW, 0, block, &scratch, true)
	if err != nil {
		t.Fatalf("edges: %v", err)
	}
	if !edges.AboveAvailable || len(edges.Above) < 8 {
		t.Fatalf("expected AboveAvailable with len>=8, got %v len=%d", edges.AboveAvailable, len(edges.Above))
	}
	// n_top_px = min(8, (34-32-8)+8) = 2. Above[0] = col 32, Above[1] = col 33,
	// Above[2..7] = replicated col 33 (the last visible).
	wantAbove := []uint16{33, 34, 34, 34, 34, 34, 34, 34}
	for i, w := range wantAbove {
		if edges.Above[i] != w {
			t.Fatalf("Above[%d]=%d want %d", i, edges.Above[i], w)
		}
	}
	// Left is unrestricted by the right-edge cap: 8 real samples from col 31.
	if !edges.LeftAvailable || len(edges.Left) < 8 {
		t.Fatalf("expected LeftAvailable with len>=8, got %v len=%d", edges.LeftAvailable, len(edges.Left))
	}
	for i := 0; i < 8; i++ {
		want := uint16(y + i + 100)
		if edges.Left[i] != want {
			t.Fatalf("Left[%d]=%d want %d", i, edges.Left[i], want)
		}
	}
	// AboveLeft = load(x-1, y-1) since n_top_px>0 and n_left_px>0.
	wantAL := uint16((y-1)*1 + 100)
	if edges.AboveLeft != wantAL {
		t.Fatalf("AboveLeft=%d want %d", edges.AboveLeft, wantAL)
	}
}

func TestFrameWorkChromaDirectionalExtendedEdgesMatchesLibaomCases(t *testing.T) {
	tests := []struct {
		name           string
		block          tile.BlockVisit
		originX        int
		originY        int
		absX           int
		absY           int
		width          int
		height         int
		wantTopRight   bool
		wantBottomLeft bool
	}{
		{
			name: "quantizer00 horizontal right-half denies bottom-left",
			block: func() tile.BlockVisit {
				parent := tile.BlockVisit{
					MICol: 4, MIRow: 0, MIColEnd: 8, MIRowEnd: 2,
					X4: 4, Y4: 0, Partition: tile.PartitionH, Size: tile.BlockSize16x8,
					VisibleW4: 4, VisibleH4: 2, HaveLeft: true,
				}
				return frameWorkPredictionTransformEdgeBlock(parent, parent.X4>>1, parent.Y4>>1, 3, 0)
			}(),
			originX: 8, originY: 0,
			absX: 12, absY: 0, width: 4, height: 4,
		},
		{
			name: "quantizer00 dc next block has left-edge bottom-left",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 1, MIColEnd: 10, MIRowEnd: 2,
				X4: 8, Y4: 1, Partition: tile.PartitionH, Size: tile.BlockSize8x4,
				VisibleW4: 2, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 0,
			absX: 16, absY: 0, width: 4, height: 4,
			wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "small 4x4 420 scales to 8x8 for table lookup",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 2, MIColEnd: 9, MIRowEnd: 3,
				Partition: tile.PartitionSplit, Size: tile.BlockSize4x4,
				VisibleW4: 1, VisibleH4: 1, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 4,
			absX: 16, absY: 4, width: 4, height: 4,
			wantTopRight: true, wantBottomLeft: true,
		},
		{
			name: "rightmost 420 chroma tx denies top-right at tile edge",
			block: tile.BlockVisit{
				MICol: 84, MIRow: 8, MIColEnd: 88, MIRowEnd: 12,
				Partition: tile.PartitionNone, Size: tile.BlockSize16x16,
				VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true,
			},
			originX: 168, originY: 16,
			absX: 172, absY: 16, width: 4, height: 4,
		},
		{
			name: "bottom 420 chroma tx denies bottom-left at tile edge",
			block: tile.BlockVisit{
				MICol: 8, MIRow: 68, MIColEnd: 12, MIRowEnd: 72,
				Partition: tile.PartitionNone, Size: tile.BlockSize16x16,
				VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true,
			},
			originX: 16, originY: 136,
			absX: 16, absY: 140, width: 4, height: 4,
			wantTopRight: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotTopRight, gotBottomLeft := frameWorkChromaDirectionalExtendedEdges(tt.block, 32, 88, 72, tt.originX, tt.originY, tt.absX, tt.absY, tt.width, tt.height, true, true)
			if gotTopRight != tt.wantTopRight || gotBottomLeft != tt.wantBottomLeft {
				t.Fatalf("topRight=%v bottomLeft=%v want %v/%v", gotTopRight, gotBottomLeft, tt.wantTopRight, tt.wantBottomLeft)
			}
		})
	}
}

func TestFrameWorkChromaAvailabilityBlockSizeMatchesLibaom(t *testing.T) {
	tests := []struct {
		name         string
		size         tile.BlockSize
		subsamplingX bool
		subsamplingY bool
		want         tile.BlockSize
	}{
		{name: "4x4 420", size: tile.BlockSize4x4, subsamplingX: true, subsamplingY: true, want: tile.BlockSize8x8},
		{name: "4x4 422", size: tile.BlockSize4x4, subsamplingX: true, want: tile.BlockSize8x4},
		{name: "4x4 440", size: tile.BlockSize4x4, subsamplingY: true, want: tile.BlockSize4x8},
		{name: "4x8 422", size: tile.BlockSize4x8, subsamplingX: true, want: tile.BlockSize8x8},
		{name: "8x4 440", size: tile.BlockSize8x4, subsamplingY: true, want: tile.BlockSize8x8},
		{name: "4x16 420", size: tile.BlockSize4x16, subsamplingX: true, subsamplingY: true, want: tile.BlockSize8x16},
		{name: "16x4 420", size: tile.BlockSize16x4, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x8},
		{name: "16x16 420 unchanged", size: tile.BlockSize16x16, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x16},
		{name: "16x8 420 unchanged", size: tile.BlockSize16x8, subsamplingX: true, subsamplingY: true, want: tile.BlockSize16x8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := frameWorkChromaAvailabilityBlockSize(tt.size, tt.subsamplingX, tt.subsamplingY); got != tt.want {
				t.Fatalf("size=%d want %d", got, tt.want)
			}
		})
	}
	got := frameWorkChromaAvailabilityBlockSize(tile.BlockSize16x16, true, true)
	plane, err := tile.PlaneBlockSize(tile.BlockSize16x16, parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got == plane {
		t.Fatalf("availability size used residual plane size %d", plane)
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraBoundaryEdges(t *testing.T) {
	tests := []struct {
		name  string
		mode  tile.IntraMode
		seed  func(*frame.Frame)
		visit func() tile.BlockLoopVisit
		want  uint16
	}{
		{
			name: "vertical-missing-top-uses-left",
			mode: tile.IntraModeVertical,
			seed: func(output *frame.Frame) {
				for y := 0; y < 16; y++ {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 70)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeVertical)
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				return visit
			},
			want: 70,
		},
		{
			name: "directional-missing-top-uses-left",
			mode: tile.IntraModeD45,
			seed: func(output *frame.Frame) {
				for y := 0; y < 16; y++ {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 81)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeD45)
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				return visit
			},
			want: 81,
		},
		{
			name: "horizontal-missing-left-uses-top",
			mode: tile.IntraModeHorizontal,
			seed: func(output *frame.Frame) {
				for x := 0; x < 16; x++ {
					setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 93)
				}
			},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeHorizontal)
				visit.Block.MICol = 0
				visit.Block.MIColEnd = 4
				visit.Block.X4 = 0
				visit.Block.HaveLeft = false
				return visit
			},
			want: 93,
		},
		{
			name: "vertical-missing-both-uses-mid-minus-one",
			mode: tile.IntraModeVertical,
			seed: func(*frame.Frame) {},
			visit: func() tile.BlockLoopVisit {
				visit := testIntraPredictionVisit(tile.IntraModeVertical)
				visit.Block.MICol = 0
				visit.Block.MIColEnd = 4
				visit.Block.MIRow = 0
				visit.Block.MIRowEnd = 4
				visit.Block.X4 = 0
				visit.Block.Y4 = 0
				visit.Block.HaveTop = false
				visit.Block.HaveLeft = false
				return visit
			},
			want: 127,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
			ctx := testIntraPredictionBatch(output)
			tt.seed(output)
			visit := tt.visit()

			var scratch FrameWorkIntraPredictionScratch
			if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
				t.Fatal(err)
			}
			x, y, err := frameWorkBlockLumaPosition(visit.Block)
			if err != nil {
				t.Fatal(err)
			}
			for row := 0; row < 16; row++ {
				for col := 0; col < 16; col++ {
					if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x+col, y+row); got != tt.want {
						t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, got, tt.want)
					}
				}
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaInterFullpel(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Col: 8, Row: -8})
	if err := ctx.PredictBlockLumaInter(0, visit); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
			want := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x+1, y-1)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterFullpelExtendsReferenceEdges(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	visit := testInterPredictionVisit(motion.Vector{Col: -16, Row: 16})
	visit.Block = tile.BlockVisit{
		MICol: 0, MIRow: 0, MIColEnd: 4, MIRowEnd: 4,
		X4: 0, Y4: 0, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
	}
	if err := ctx.PredictBlockInter(0, visit, nil); err != nil {
		t.Fatal(err)
	}

	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y)
			want := frameWorkTestSampleClamped(reference.Y, output.Layout.BytesPerSample, x-2, y+2)
			if got != want {
				t.Fatalf("y sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			wantU := frameWorkTestSampleClamped(reference.U, output.Layout.BytesPerSample, x-1, y+1)
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != wantU {
				t.Fatalf("u sample(%d,%d)=%d want %d", x, y, got, wantU)
			}
			wantV := frameWorkTestSampleClamped(reference.V, output.Layout.BytesPerSample, x-1, y+1)
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != wantV {
				t.Fatalf("v sample(%d,%d)=%d want %d", x, y, got, wantV)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockLumaInterFractionalMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: 5}
	filters := motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockWithFilterBitDepth(want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterInvalidWarpFallsBackToTranslation(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: 5}
	filters := motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	visit.Prediction.WarpedMotionInvalid = true

	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockWithFilterBitDepth(want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterWarpIdentityConstant(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 0xab)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	visit.Prediction.WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true

	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := output.Y.Pix[y*output.Y.Stride+x]; got != 0xab {
				t.Fatalf("sample(%d,%d)=%d want 0xab", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterWarpSmallChromaFallsBackToRegular(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapRegular}
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Block.MIColEnd = 6
	visit.Block.MIRowEnd = 6
	visit.Block.Size = tile.BlockSize8x8
	visit.Block.VisibleW4 = 2
	visit.Block.VisibleH4 = 2
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	params.Matrix[0] = 1 << 16
	visit.Prediction.WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true

	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, filters); err != nil {
		t.Fatal(err)
	}

	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 4, 4, motion.Vector{}, true, true, filters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 4, 4, motion.Vector{}, true, true, filters)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 4, 4)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 4, 4)
}

func TestFrameWorkBatchPredictBlockLumaInterHighBitDepthFractionalMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	want := testBatchFrame(t, output.Format)
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 4, Row: 6}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockWithFilterBitDepth(want.Y, reference.Y, want.Layout.BytesPerSample, want.Format.BitDepth, 16, 16, 16, 16, mv, filters); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockLumaInterOBMCLeftMatchesLibaomMask(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(reference, 0xff, 17)
	ctx := testInterPredictionBatch(output, reference)

	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	visit.Prediction.OverlappableNeighbors.LeftCount = 1
	visit.Prediction.OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{{Col: 8}},
		},
		InterpFilters:      motion.RegularFilters,
		InterpFiltersValid: true,
	}

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}

	mask, ok := frameWorkOBMCMask(8)
	if !ok {
		t.Fatal("missing 8-wide obmc mask")
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			base := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y)
			want := base
			if x < 24 {
				neighbor := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x+1, y)
				want = frameWorkBlendA64(uint16(mask[x-16]), base, neighbor)
			}
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterOBMCChromaSubsampledMatchesLibaomMasks(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	visit := testOBMCInterPredictionVisit(motion.Vector{}, motion.Vector{Row: 16}, motion.Vector{Col: 16})
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}

	mask4, ok := frameWorkOBMCMask(4)
	if !ok {
		t.Fatal("missing 4-wide obmc mask")
	}
	for _, tt := range []struct {
		plane frame.Plane
		ref   frame.Plane
	}{
		{plane: output.U, ref: reference.U},
		{plane: output.V, ref: reference.V},
	} {
		for y := 8; y < 16; y++ {
			for x := 8; x < 16; x++ {
				want := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x, y)
				if y < 12 {
					neighbor := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x, y+1)
					want = frameWorkBlendA64(uint16(mask4[y-8]), want, neighbor)
				}
				if x < 12 {
					neighbor := frameWorkTestSample(tt.ref, reference.Layout.BytesPerSample, x+1, y)
					want = frameWorkBlendA64(uint16(mask4[x-8]), want, neighbor)
				}
				if got := frameWorkTestSample(tt.plane, output.Layout.BytesPerSample, x, y); got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
				}
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterChromaSubsampledMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, SubsamplingX: true, SubsamplingY: true, Align: 128})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: -5}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, filters); err != nil {
		t.Fatal(err)
	}

	wantY := testFrameWorkMotionPredictionPlane(t, reference.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv, filters)
	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, filters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, filters)
	assertFrameWorkPlaneBlockEqualAt(t, output.Y, 16, 16, wantY, 0, 0, output.Layout.BytesPerSample, 16, 16)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 8, 8)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 8, 8)
}

func TestFrameWorkBatchPredictBlockInterIntraDCBlendsPredictors(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 16)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 80)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 80)
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 80)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeDC, false)
	var scratch FrameWorkInterPredictionScratch

	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != 48 {
				t.Fatalf("sample(%d,%d)=%d want 48", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterIntraWedgeBlendsPredictors(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 20)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 84)
	}
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeVertical, true)
	visit.Prediction.InterIntra.WedgeIndex = 0
	var scratch FrameWorkInterPredictionScratch

	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	var mask [16 * 16]byte
	if err := frameWorkBuildWedgeMask(mask[:], 16, tile.BlockSize16x16, 0, false); err != nil {
		t.Fatal(err)
	}
	for _, sample := range [][2]int{{0, 0}, {7, 8}, {15, 15}} {
		x := 16 + sample[0]
		y := 16 + sample[1]
		m := uint16(mask[sample[1]*16+sample[0]])
		want := frameWorkBlendA64(m, 84, 20)
		if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
			t.Fatalf("sample(%d,%d)=%d want %d mask=%d", x, y, got, want, m)
		}
	}
}

func TestFrameWorkBatchPredictBlockInterIntraAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	testFillFrame(reference, 16)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeSmooth, false)
	var scratch FrameWorkInterPredictionScratch

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter inter-intra allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageMatchesMotion(t *testing.T) {
	tests := []struct {
		name    string
		format  frame.Format
		max     uint16
		mv0     motion.Vector
		mv1     motion.Vector
		filters motion.InterpFilters
	}{
		{
			name:    "lowbd",
			format:  frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64},
			max:     0xff,
			mv0:     motion.Vector{Col: 3, Row: 5},
			mv1:     motion.Vector{Col: -1, Row: 6},
			filters: motion.InterpFilters{X: motion.InterpMultiTapSharp, Y: motion.InterpEightTapSmooth},
		},
		{
			name:    "highbd-dist-wtd",
			format:  frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128},
			max:     0x3ff,
			mv0:     motion.Vector{Col: 4, Row: 6},
			mv1:     motion.Vector{Col: -1, Row: 3},
			filters: motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output := testBatchFrame(t, tt.format)
			last := testBatchFrame(t, tt.format)
			bwd := testBatchFrame(t, tt.format)
			fillFrameWorkInterReference(last, tt.max)
			fillFrameWorkInterReferenceVariant(bwd, tt.max, 37)
			ctx := testCompoundInterPredictionBatch(output, last, bwd)
			compoundType := tile.CompoundTypeAverage
			if tt.name == "highbd-dist-wtd" {
				compoundType = tile.CompoundTypeDistWtd
			}
			visit := testCompoundInterPredictionVisit(tt.mv0, tt.mv1, compoundType)

			var scratch FrameWorkInterPredictionScratch
			if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, tt.filters); err != nil {
				t.Fatal(err)
			}

			first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv0, false, false, tt.filters)
			second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv1, false, false, tt.filters)
			fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
			if err != nil {
				t.Fatal(err)
			}
			assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, fwdOffset, bckOffset)
		})
	}
}

// TestFrameWorkBatchPredictBlockLumaInterCompoundAverageZeroMV pins the
// integer-pel compound-average path used by NEAREST_NEAREST blocks with
// MV=(0,0)/(0,0). With sub-pel offsets of zero the convolve degenerates to
// a copy, so the per-pixel blend is exactly (a+b+1)>>1. Pinning specific
// arithmetic samples here protects the libaom-equivalent rounding constant
// (1<<(DistPrecisionBits-1) = 8) at offsets 8/8 from silently regressing
// to a truncating average.
func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageZeroMV(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	// Set up a constant-coloured 16x16 luma block on each reference so the
	// average degenerates to a deterministic per-pixel formula. Position
	// (16,16) matches testCompoundInterPredictionVisit's destination block.
	const blockX, blockY, blockW, blockH = 16, 16, 16, 16
	for row := 0; row < blockH; row++ {
		for col := 0; col < blockW; col++ {
			setFrameWorkTestSample(last.Y, last.Layout.BytesPerSample, blockX+col, blockY+row, 115)
			setFrameWorkTestSample(bwd.Y, bwd.Layout.BytesPerSample, blockX+col, blockY+row, 117)
		}
	}
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < blockH; row++ {
		for col := 0; col < blockW; col++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, blockX+col, blockY+row)
			// (115 + 117 + 1) >> 1 = 116
			if got != 116 {
				t.Fatalf("compound avg(115,117) at (%d,%d) = %d, want 116", blockX+col, blockY+row, got)
			}
		}
	}
	// Also pin the asymmetric case where the average rounds up.
	for row := 0; row < blockH; row++ {
		for col := 0; col < blockW; col++ {
			setFrameWorkTestSample(last.Y, last.Layout.BytesPerSample, blockX+col, blockY+row, 164)
			setFrameWorkTestSample(bwd.Y, bwd.Layout.BytesPerSample, blockX+col, blockY+row, 165)
		}
	}
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < blockH; row++ {
		for col := 0; col < blockW; col++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, blockX+col, blockY+row)
			// (164 + 165 + 1) >> 1 = 165
			if got != 165 {
				t.Fatalf("compound avg(164,165) at (%d,%d) = %d, want 165", blockX+col, blockY+row, got)
			}
		}
	}
	// Same-value inputs must blend back to themselves (no off-by-one drift).
	for row := 0; row < blockH; row++ {
		for col := 0; col < blockW; col++ {
			setFrameWorkTestSample(last.Y, last.Layout.BytesPerSample, blockX+col, blockY+row, 164)
			setFrameWorkTestSample(bwd.Y, bwd.Layout.BytesPerSample, blockX+col, blockY+row, 164)
		}
	}
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < blockH; row++ {
		for col := 0; col < blockW; col++ {
			got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, blockX+col, blockY+row)
			if got != 164 {
				t.Fatalf("compound avg(164,164) at (%d,%d) = %d, want 164", blockX+col, blockY+row, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundChromaSubsampledMatchesMotion(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 19)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 73)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDistWtd)

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstU := testFrameWorkCompoundConvBuf(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkCompoundConvBuf(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkCompoundConvBuf(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkCompoundConvBuf(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
	if err != nil {
		t.Fatal(err)
	}
	assertFrameWorkCompoundBlendEqual(t, output.U, firstU, secondU, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, fwdOffset, bckOffset)
	assertFrameWorkCompoundBlendEqual(t, output.V, firstV, secondV, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, fwdOffset, bckOffset)
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundDiffWtdMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0x3ff, 29)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 113)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapSmooth, Y: motion.InterpMultiTapSharp}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38Inv

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := testFrameWorkDiffWtdMask(t, first, second, output.Format.BitDepth, 16, 16, tile.DiffWtdMaskType38Inv)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, first, second, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockInterCompoundDiffWtdChromaSubsampledMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 17)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 191)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 5, Row: -3}
	mv1 := motion.Vector{Col: -1, Row: 7}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstY := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	secondY := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := testFrameWorkDiffWtdMask(t, firstY, secondY, output.Format.BitDepth, 16, 16, tile.DiffWtdMaskType38)
	firstU := testFrameWorkCompoundConvBuf(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkCompoundConvBuf(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkCompoundConvBuf(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkCompoundConvBuf(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, firstY, secondY, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
	assertFrameWorkMaskedCompoundEqual(t, output.U, firstU, secondU, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
	assertFrameWorkMaskedCompoundEqual(t, output.V, firstV, secondV, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
}

func TestFrameWorkBuildWedgeMaskMatchesLibaomSamples(t *testing.T) {
	tests := []struct {
		name       string
		size       tile.BlockSize
		index      uint8
		sign       bool
		width      int
		height     int
		samples    [][2]int
		wantSample []byte
	}{
		{
			name:       "8x8 oblique27 neg",
			size:       tile.BlockSize8x8,
			index:      0,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {7, 7}},
			wantSample: []byte{64, 0},
		},
		{
			name:       "8x8 oblique27 positive",
			size:       tile.BlockSize8x8,
			index:      0,
			sign:       true,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {7, 7}},
			wantSample: []byte{0, 64},
		},
		{
			name:       "8x8 vertical neg",
			size:       tile.BlockSize8x8,
			index:      6,
			width:      8,
			height:     8,
			samples:    [][2]int{{0, 0}, {1, 0}, {2, 0}, {7, 0}},
			wantSample: []byte{57, 43, 21, 0},
		},
		{
			name:       "32x8 oblique63 positive",
			size:       tile.BlockSize32x8,
			index:      12,
			width:      32,
			height:     8,
			samples:    [][2]int{{0, 0}, {31, 7}},
			wantSample: []byte{0, 64},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mask := make([]byte, tt.width*tt.height)
			if err := frameWorkBuildWedgeMask(mask, tt.width, tt.size, tt.index, tt.sign); err != nil {
				t.Fatal(err)
			}
			for i, sample := range tt.samples {
				got := mask[sample[1]*tt.width+sample[0]]
				if got != tt.wantSample[i] {
					t.Fatalf("mask(%d,%d)=%d want %d", sample[0], sample[1], got, tt.wantSample[i])
				}
			}
		})
	}
}

func TestFrameWorkBuildWedgeMaskSignComplementsLibaom(t *testing.T) {
	sizes := []tile.BlockSize{
		tile.BlockSize8x8,
		tile.BlockSize8x16,
		tile.BlockSize16x8,
		tile.BlockSize16x16,
		tile.BlockSize8x32,
		tile.BlockSize32x8,
		tile.BlockSize16x32,
		tile.BlockSize32x16,
		tile.BlockSize32x32,
	}
	var mask [32 * 32]byte
	var complement [32 * 32]byte
	for _, size := range sizes {
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatalf("missing dimensions for %v", size)
		}
		width := int(dims.W4) * 4
		height := int(dims.H4) * 4
		for index := uint8(0); index < tile.MaxWedgeTypes; index++ {
			if err := frameWorkBuildWedgeMask(mask[:], width, size, index, false); err != nil {
				t.Fatalf("size=%v index=%d neg err=%v", size, index, err)
			}
			if err := frameWorkBuildWedgeMask(complement[:], width, size, index, true); err != nil {
				t.Fatalf("size=%v index=%d pos err=%v", size, index, err)
			}
			for row := 0; row < height; row++ {
				for col := 0; col < width; col++ {
					got := int(mask[row*width+col]) + int(complement[row*width+col])
					if got != frameWorkWedgeMaxAlpha {
						t.Fatalf("size=%v index=%d mask(%d,%d) sum=%d want %d", size, index, col, row, got, frameWorkWedgeMaxAlpha)
					}
				}
			}
		}
	}
	if err := frameWorkBuildWedgeMask(mask[:], 4, tile.BlockSize4x4, 0, false); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("unsupported wedge err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBuildWedgeMaskAllocs(t *testing.T) {
	var mask [16 * 16]byte
	allocs := testing.AllocsPerRun(1000, func() {
		if err := frameWorkBuildWedgeMask(mask[:], 16, tile.BlockSize16x16, 0, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("frameWorkBuildWedgeMask allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundWedgeMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0x3ff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 149)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 3, Row: 5}
	mv1 := motion.Vector{Col: -5, Row: 1}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 0
	visit.Prediction.CompoundBlend.WedgeSign = false

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := make([]byte, 16*16)
	if err := frameWorkBuildWedgeMask(mask, 16, visit.Block.Size, visit.Prediction.CompoundBlend.WedgeIndex, visit.Prediction.CompoundBlend.WedgeSign); err != nil {
		t.Fatal(err)
	}
	assertFrameWorkMaskedCompoundEqual(t, output.Y, first, second, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
}

func TestFrameWorkBatchPredictBlockInterCompoundWedgeChromaSubsampledMatchesLibaom(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 43)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 211)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	mv0 := motion.Vector{Col: 5, Row: -3}
	mv1 := motion.Vector{Col: -1, Row: 7}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	visit := testCompoundInterPredictionVisit(mv0, mv1, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 6
	visit.Prediction.CompoundBlend.WedgeSign = false

	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatal(err)
	}

	firstY := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv0, false, false, filters)
	secondY := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv1, false, false, filters)
	mask := make([]byte, 16*16)
	if err := frameWorkBuildWedgeMask(mask, 16, visit.Block.Size, visit.Prediction.CompoundBlend.WedgeIndex, visit.Prediction.CompoundBlend.WedgeSign); err != nil {
		t.Fatal(err)
	}
	firstU := testFrameWorkCompoundConvBuf(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkCompoundConvBuf(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkCompoundConvBuf(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkCompoundConvBuf(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	assertFrameWorkMaskedCompoundEqual(t, output.Y, firstY, secondY, mask, 16, false, false, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16)
	assertFrameWorkMaskedCompoundEqual(t, output.U, firstU, secondU, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
	assertFrameWorkMaskedCompoundEqual(t, output.V, firstV, secondV, mask, 16, true, true, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8)
}

func TestFrameWorkCompoundDistanceWeightedOffsetsMatchLibaom(t *testing.T) {
	tests := []struct {
		name string
		cur  uint32
		ref0 uint32
		ref1 uint32
		want [2]int
	}{
		{name: "equal distance", cur: 8, ref0: 4, ref1: 12, want: [2]int{7, 9}},
		{name: "forward nearer", cur: 8, ref0: 2, ref1: 10, want: [2]int{4, 12}},
		{name: "backward nearer", cur: 8, ref0: 6, ref1: 15, want: [2]int{13, 3}},
		{name: "zero distance", cur: 8, ref0: 8, ref1: 12, want: [2]int{13, 3}},
		{name: "wraparound", cur: 1, ref0: 14, ref1: 4, want: [2]int{7, 9}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fwd, bck, err := frameWorkDistanceWeightedCompoundOffsets(4, tt.cur, tt.ref0, tt.ref1)
			if err != nil {
				t.Fatal(err)
			}
			if got := [2]int{fwd, bck}; got != tt.want {
				t.Fatalf("offsets=%v want %v", got, tt.want)
			}
		})
	}
	if _, _, err := frameWorkDistanceWeightedCompoundOffsets(0, 0, 0, 0); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad bits err=%v want %v", err, ErrInvalidBatch)
	}
	if _, _, err := frameWorkDistanceWeightedCompoundOffsets(4, 16, 0, 1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("out of range hint err=%v want %v", err, ErrInvalidBatch)
	}
}

func FuzzFrameWorkDistanceWeightedCompoundOffsets(f *testing.F) {
	f.Add(uint8(4), uint32(8), uint32(4), uint32(12))
	f.Add(uint8(4), uint32(8), uint32(2), uint32(10))
	f.Add(uint8(4), uint32(1), uint32(14), uint32(4))
	f.Add(uint8(5), uint32(16), uint32(3), uint32(28))

	f.Fuzz(func(t *testing.T, bits uint8, cur uint32, ref0 uint32, ref1 uint32) {
		bits = bits%8 + 1
		limit := uint32(1) << bits
		cur %= limit
		ref0 %= limit
		ref1 %= limit
		fwd, bck, err := frameWorkDistanceWeightedCompoundOffsets(bits, cur, ref0, ref1)
		if err != nil {
			t.Fatalf("frameWorkDistanceWeightedCompoundOffsets err=%v", err)
		}
		if fwd < 0 || bck < 0 || fwd+bck != 16 {
			t.Fatalf("offsets=%d,%d", fwd, bck)
		}
	})
}

func TestFrameWorkBatchPredictBlockLumaDispatchesCompoundAverage(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(last, 0xff)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 91)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 8, Row: 0}, motion.Vector{Col: -8, Row: 0}, tile.CompoundTypeAverage)

	var interScratch FrameWorkInterPredictionScratch
	scratch := FrameWorkPredictionScratch{Inter: &interScratch}
	if err := ctx.PredictBlockLuma(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	first := testFrameWorkCompoundConvBuf(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[0], false, false, motion.RegularFilters)
	second := testFrameWorkCompoundConvBuf(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[1], false, false, motion.RegularFilters)
	assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, 8, 8)
}

func TestFrameWorkBatchPredictBlockDispatchesCompletePlanes(t *testing.T) {
	mono := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	monoCtx := testIntraPredictionBatch(mono)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, 15, y, 50)
	}
	var predictionScratch FrameWorkPredictionScratch
	if err := monoCtx.PredictBlock(0, testIntraPredictionVisit(tile.IntraModeDC), &predictionScratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(mono.Y, mono.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("mono intra sample=%d want 30", got)
	}

	colorIntra := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	colorIntraCtx := testIntraPredictionBatch(colorIntra)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, 15, y, 50)
	}
	for x := 8; x < 16; x++ {
		setFrameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, x, 7, 20)
		setFrameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, x, 7, 30)
	}
	for y := 8; y < 16; y++ {
		setFrameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, 7, y, 60)
		setFrameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, 7, y, 70)
	}
	colorVisit := testIntraPredictionVisit(tile.IntraModeDC)
	colorVisit.Prediction.ChromaMode = tile.ChromaIntraModeDC
	colorVisit.Prediction.ChromaModeValid = true
	if err := colorIntraCtx.PredictBlock(0, colorVisit, &predictionScratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(colorIntra.Y, colorIntra.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("color intra y sample=%d want 30", got)
	}
	if got := frameWorkTestSample(colorIntra.U, colorIntra.Layout.BytesPerSample, 8, 8); got != 40 {
		t.Fatalf("color intra u sample=%d want 40", got)
	}
	if got := frameWorkTestSample(colorIntra.V, colorIntra.Layout.BytesPerSample, 8, 8); got != 50 {
		t.Fatalf("color intra v sample=%d want 50", got)
	}

	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	mv := motion.Vector{Col: 3, Row: -5}
	visit := testInterPredictionVisit(mv)
	if err := ctx.PredictBlock(0, visit, nil); err != nil {
		t.Fatal(err)
	}

	wantY := testFrameWorkMotionPredictionPlane(t, reference.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, mv, motion.RegularFilters)
	wantU := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, motion.RegularFilters)
	wantV := testFrameWorkMotionPredictionPlaneSubsampled(t, reference.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv, true, true, motion.RegularFilters)
	assertFrameWorkPlaneBlockEqualAt(t, output.Y, 16, 16, wantY, 0, 0, output.Layout.BytesPerSample, 16, 16)
	assertFrameWorkPlaneBlockEqualAt(t, output.U, 8, 8, wantU, 0, 0, output.Layout.BytesPerSample, 8, 8)
	assertFrameWorkPlaneBlockEqualAt(t, output.V, 8, 8, wantV, 0, 0, output.Layout.BytesPerSample, 8, 8)
}

func TestFrameWorkBatchPredictBlockIntraSubsampledChromaUsesPlaneEdges(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	for y := 0; y < 16; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 3, y, 64)
	}

	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 1, MIRow: 0, MIColEnd: 2, MIRowEnd: 4,
		X4: 1, Y4: 0, Size: tile.BlockSize4x16, VisibleW4: 1, VisibleH4: 4,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeVertical
	visit.Prediction.ChromaModeValid = true

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 4; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 127 {
				t.Fatalf("u sample(%d,%d)=%d want 127", x, y, got)
			}
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 127 {
				t.Fatalf("v sample(%d,%d)=%d want 127", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockIntraCoeffClippedChromaSmooth(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 352, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		Use128x128Superblock: true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	})
	ctx.Jobs = []tile.Job{{SBCols: 3, SBRows: 2}}
	for y := 0; y < 32; y++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 159, y, 77)
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 159, y, 91)
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 64, MIRow: 0, MIColEnd: 88, MIRowEnd: 32,
		X4: 0, Y4: 0, Size: tile.BlockSize128x128, VisibleW4: 24, VisibleH4: 32,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeSmooth
	visit.Prediction.ChromaModeValid = true
	block := tile.BlockCoeffBlock{
		Plane: 1,
		Block: tile.TransformBlock{X4: 8, Y4: 0, Size: tile.TransformSize32x32, VisibleW4: 4, VisibleH4: 8},
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 32; y++ {
		for x := 160; x < 176; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 77 {
				t.Fatalf("u sample(%d,%d)=%d want 77", x, y, got)
			}
		}
	}
	block.Plane = 2
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 32; y++ {
		for x := 160; x < 176; x++ {
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 91 {
				t.Fatalf("v sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockIntraCoeffLumaPaddingNoOp(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 320, Height: 180, BitDepth: 8, MonoChrome: true, Align: 64})
	testFillFrame(output, 19)
	before := slices.Clone(output.Y.Pix)
	ctx := testIntraPredictionBatch(output)
	ctx.Jobs = []tile.Job{{SBCols: 5, SBRows: 3}}

	visit := testIntraPredictionVisit(tile.IntraModeSmooth)
	visit.Block = tile.BlockVisit{
		MICol: 4, MIRow: 44, MIColEnd: 5, MIRowEnd: 46,
		X4: 4, Y4: 12, Size: tile.BlockSize4x8, VisibleW4: 1, VisibleH4: 2,
		HaveTop: true, HaveLeft: true,
	}
	block := tile.BlockCoeffBlock{
		Plane: 0,
		Block: tile.TransformBlock{X4: 4, Y4: 13, Size: tile.TransformSize4x4, VisibleW4: 1, VisibleH4: 1},
	}
	var scratch FrameWorkIntraPredictionScratch
	if err := ctx.PredictBlockIntraCoeff(0, visit, block, &scratch); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(output.Y.Pix, before) {
		t.Fatalf("padding-only luma TXB prediction modified output")
	}
}

func TestFrameWorkBatchPredictBlockIntraClippedChromaSmooth(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 352, Height: 256, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	ctx.Sequence = FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
		Use128x128Superblock: true,
		ColorConfig: parser.ColorConfig{
			BitDepth:     output.Format.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
		},
	})
	ctx.Jobs = []tile.Job{{SBCols: 3, SBRows: 2}}
	for y := 0; y < 64; y++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 127, y, 77)
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 127, y, 91)
	}
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Block = tile.BlockVisit{
		MICol: 64, MIRow: 0, MIColEnd: 88, MIRowEnd: 32,
		X4: 0, Y4: 0, Size: tile.BlockSize128x128, VisibleW4: 24, VisibleH4: 32,
		HaveTop: false, HaveLeft: true,
	}
	visit.Prediction.ChromaMode = tile.ChromaIntraModeSmooth
	visit.Prediction.ChromaModeValid = true

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < 64; y++ {
		for x := 128; x < 176; x++ {
			if got := frameWorkTestSample(output.U, output.Layout.BytesPerSample, x, y); got != 77 {
				t.Fatalf("u sample(%d,%d)=%d want 77", x, y, got)
			}
			if got := frameWorkTestSample(output.V, output.Layout.BytesPerSample, x, y); got != 91 {
				t.Fatalf("v sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockRejectsIncompletePlaneSupport(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlock(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing chroma mode err=%v want %v", err, ErrInvalidBatch)
	}
	cfl := testIntraPredictionVisit(tile.IntraModeDC)
	cfl.Prediction.ChromaMode = tile.ChromaIntraModeCFL
	cfl.Prediction.ChromaModeValid = true
	cfl.Prediction.CFLAlphaValid = true
	if err := ctx.PredictBlock(0, cfl, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("cfl intra err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlock(0, tile.BlockLoopVisit{}, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing prediction err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLMatchesPrimitives(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	want := testBatchFrame(t, output.Format)
	visit := testCFLPredictionVisit()
	seedFrameWorkCFLPredictionFrame(output)
	seedFrameWorkCFLPredictionFrame(want)

	ctx := testIntraPredictionBatch(output)
	var scratch FrameWorkCFLPredictionScratch
	if err := ctx.PredictBlockChromaCFL(0, visit, &scratch); err != nil {
		t.Fatal(err)
	}

	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneU)
	testPredictFrameWorkCFLWant(t, want, visit, FrameWorkPlaneV)
	assertFrameWorkPlaneBlockEqual(t, output.U, want.U, output.Layout.BytesPerSample, 8, 8, 8, 8)
	assertFrameWorkPlaneBlockEqual(t, output.V, want.V, output.Layout.BytesPerSample, 8, 8, 8, 8)
}

func TestFrameWorkSubsampleLumaCFLQ3MatchesPrimitives(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		subX     bool
		subY     bool
	}{
		{name: "lowbd-420", bitDepth: 8, subX: true, subY: true},
		{name: "highbd-444", bitDepth: 10},
		{name: "highbd-422", bitDepth: 12, subX: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			format := frame.Format{Width: 64, Height: 64, BitDepth: tt.bitDepth, SubsamplingX: tt.subX, SubsamplingY: tt.subY, Align: 128}
			output := testBatchFrame(t, format)
			bytesPerSample := output.Layout.BytesPerSample
			const x = 16
			const y = 16
			const width = 16
			const height = 16
			stride := width + 5
			max := int((1 << tt.bitDepth) - 1)
			var got [prediction.CFLBufSquare]uint16
			var want [prediction.CFLBufSquare]uint16
			if bytesPerSample == 1 {
				src := make([]uint8, stride*height)
				for row := 0; row < height; row++ {
					for col := 0; col < width; col++ {
						value := uint8((17 + row*13 + col*19) & max)
						src[row*stride+col] = value
						setFrameWorkTestSample(output.Y, bytesPerSample, x+col, y+row, uint16(value))
					}
				}
				if err := prediction.SubsampleLuma8ToQ3(want[:], src, stride, width, height, tt.subX, tt.subY); err != nil {
					t.Fatal(err)
				}
			} else {
				src := make([]uint16, stride*height)
				for row := 0; row < height; row++ {
					for col := 0; col < width; col++ {
						value := uint16((257 + row*83 + col*41) & max)
						src[row*stride+col] = value
						setFrameWorkTestSample(output.Y, bytesPerSample, x+col, y+row, value)
					}
				}
				if err := prediction.SubsampleLuma16ToQ3(want[:], src, stride, width, height, tt.subX, tt.subY, tt.bitDepth); err != nil {
					t.Fatal(err)
				}
			}
			if err := frameWorkSubsampleLumaCFLQ3(got[:], output.Y, bytesPerSample, tt.bitDepth, x, y, width, height, tt.subX, tt.subY); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got[:], want[:]) {
				t.Fatalf("subsample mismatch")
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	ctx := testIntraPredictionBatch(output)
	valid := testCFLPredictionVisit()
	var scratch FrameWorkCFLPredictionScratch
	if err := ctx.PredictBlockChromaCFL(0, valid, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil scratch err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockChromaCFL(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("non-cfl err=%v want %v", err, ErrInvalidBatch)
	}
	noAlpha := valid
	noAlpha.Prediction.CFLAlphaValid = false
	if err := ctx.PredictBlockChromaCFL(0, noAlpha, &scratch); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing alpha err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockChromaCFLAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	seedFrameWorkCFLPredictionFrame(output)
	ctx := testIntraPredictionBatch(output)
	visit := testCFLPredictionVisit()
	var scratch FrameWorkCFLPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockChromaCFL(0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockChromaCFL allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	var scratch FrameWorkPredictionScratch
	intra := testIntraPredictionVisit(tile.IntraModeDC)
	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlock(0, intra, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlock(0, inter, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlock allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	valid := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch

	tests := []struct {
		name    string
		ctx     FrameWorkBatch
		visit   tile.BlockLoopVisit
		scratch *FrameWorkInterPredictionScratch
	}{
		{name: "nil scratch", ctx: ctx, visit: valid},
		{name: "missing blend", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.CompoundBlendValid = false
			return visit
		}(), scratch: &scratch},
		{name: "wedge", ctx: ctx, visit: testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeWedge), scratch: &scratch},
		{name: "dist wtd without order hint", ctx: func() FrameWorkBatch {
			next := ctx
			next.Sequence.EnableOrderHint = false
			return next
		}(), visit: testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeDistWtd), scratch: &scratch},
		{name: "single ref", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
			visit.Prediction.InterReferences = refs
			visit.Prediction.InterMotion.References = refs
			return visit
		}(), scratch: &scratch},
		{name: "inter intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterIntraValid = true
			visit.Prediction.InterIntra.Enabled = true
			return visit
		}(), scratch: &scratch},
		{name: "intrabc", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intrabc = true
			visit.Prediction.IntrabcValid = true
			return visit
		}(), scratch: &scratch},
		{name: "non translation motion mode", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.MotionModeValid = true
			visit.Prediction.MotionMode = tile.MotionModeOBMC
			return visit
		}(), scratch: &scratch},
		{name: "missing second reference frame", ctx: func() FrameWorkBatch {
			next := ctx
			next.References = next.References[:int(FrameWorkReferenceBwd)]
			return next
		}(), visit: valid, scratch: &scratch},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaInterCompoundAverage(0, tt.visit, tt.scratch); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundRejectsInvalidDiffWtdMask(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 11)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 97)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType(99)
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterCompoundAverageAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(last, 0x3ff)
	fillFrameWorkInterReferenceVariant(bwd, 0x3ff, 17)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 4, Row: 6}, motion.Vector{Col: -1, Row: 3}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterCompoundAverageWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaInterCompoundAverage allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundDiffWtdAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 157)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 3, Row: 5}, motion.Vector{Col: -1, Row: 7}, tile.CompoundTypeDiffWtd)
	visit.Prediction.CompoundBlend.MaskType = tile.DiffWtdMaskType38
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter diff-wtd allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundWedgeAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceVariant(last, 0xff, 31)
	fillFrameWorkInterReferenceVariant(bwd, 0xff, 157)
	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{Col: 3, Row: 5}, motion.Vector{Col: -1, Row: 7}, tile.CompoundTypeWedge)
	visit.Prediction.CompoundBlend.WedgeIndex = 0
	var scratch FrameWorkInterPredictionScratch
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter wedge allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterOBMCAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReferenceAllPlanes(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testOBMCInterPredictionVisit(motion.Vector{}, motion.Vector{Row: 16}, motion.Vector{Col: 16})
	var scratch FrameWorkInterPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterOBMCWithFilters(0, visit, &scratch, motion.RegularFilters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockInter OBMC allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaInterRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	ctx := testInterPredictionBatch(output, reference)
	valid := testInterPredictionVisit(motion.Vector{})

	tests := []struct {
		name  string
		ctx   FrameWorkBatch
		visit tile.BlockLoopVisit
	}{
		{name: "missing prediction", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Valid = false
			return visit
		}()},
		{name: "intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intra = true
			return visit
		}()},
		{name: "missing motion", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterMotionValid = false
			return visit
		}()},
		{name: "compound", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterMotion.References.Compound = true
			visit.Prediction.InterMotion.References.Ref[1] = tile.ReferenceFrameBWD
			return visit
		}()},
		{name: "non translation motion mode", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.MotionModeValid = true
			visit.Prediction.MotionMode = tile.MotionModeWarp
			return visit
		}()},
		{name: "inter intra", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.InterIntraValid = true
			visit.Prediction.InterIntra.Enabled = true
			return visit
		}()},
		{name: "intrabc", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intrabc = true
			visit.Prediction.IntrabcValid = true
			return visit
		}()},
		{name: "missing reference frame", ctx: func() FrameWorkBatch {
			next := ctx
			next.References = nil
			return next
		}(), visit: valid},
		{name: "switchable filter not decoded", ctx: func() FrameWorkBatch {
			next := ctx
			next.TileInfo.InterpolationFilter = parser.InterpolationSwitchable
			return next
		}(), visit: valid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaInter(0, tt.visit); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
	// A block whose plane origin lands entirely beyond the output frame (for
	// example because the bitstream rounded the MI grid up past the visible
	// area) is silently skipped to match libaom's clip-to-visible behavior.
	t.Run("outside job", func(t *testing.T) {
		visit := valid
		visit.Block.MICol = 16
		visit.Block.MIColEnd = 20
		if err := ctx.PredictBlockLumaInter(0, visit); err != nil {
			t.Fatalf("outside job err=%v want nil", err)
		}
	})
	if err := ctx.PredictBlockLumaInterWithFilters(0, valid, motion.InterpFilters{X: motion.InterpFilter(99)}); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad filters err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockInterRejectsInterIntraBeforeMutation(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	output.Y.Pix[0] = 0x44
	output.U.Pix[0] = 0x55
	output.V.Pix[0] = 0x66

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.InterIntraValid = true
	visit.Prediction.InterIntra.Enabled = true
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("PredictBlockInterWithFilters err=%v want %v", err, ErrInvalidBatch)
	}
	if output.Y.Pix[0] != 0x44 || output.U.Pix[0] != 0x55 || output.V.Pix[0] != 0x66 {
		t.Fatalf("output mutated y=%#x u=%#x v=%#x", output.Y.Pix[0], output.U.Pix[0], output.V.Pix[0])
	}
}

func TestFrameWorkBatchPredictBlockInterIntrabcCopiesOutput(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	testFillFrame(output, 3)
	ctx := testInterPredictionBatch(output, output)
	for y := 0; y < 16; y++ {
		for x := 16; x < 32; x++ {
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y, uint16(40+y+x-16))
		}
	}

	visit := testInterPredictionVisit(motion.Vector{Row: -16 * 8})
	visit.Prediction.Intrabc = true
	visit.Prediction.IntrabcValid = true
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			want := uint16(40 + y - 16 + x - 16)
			if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestFrameWorkBatchPredictBlockInterRejectsScaledReferenceBeforeMutation(t *testing.T) {
	if frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch enabled; same-size-only rejection is bypassed")
	}
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	output.Y.Pix[16*output.Y.Stride+16] = 0x44
	output.U.Pix[8*output.U.Stride+8] = 0x55
	output.V.Pix[8*output.V.Stride+8] = 0x66

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("PredictBlockInterWithFilters err=%v want %v", err, ErrInvalidBatch)
	}
	if output.Y.Pix[16*output.Y.Stride+16] != 0x44 ||
		output.U.Pix[8*output.U.Stride+8] != 0x55 ||
		output.V.Pix[8*output.V.Stride+8] != 0x66 {
		t.Fatalf("output mutated y=%#x u=%#x v=%#x", output.Y.Pix[16*output.Y.Stride+16], output.U.Pix[8*output.U.Stride+8], output.V.Pix[8*output.V.Stride+8])
	}
}

func TestFrameWorkBatchPredictBlockInterRoutesScaledReferenceWhenEnabled(t *testing.T) {
	if !frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch disabled; set GOAV1_SCALED_PRED=1 or build with goav1_scaled_pred to exercise")
	}
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 64})
	// Fill the reference with a uniform value: 8-tap kernels sum to 128 (1.0
	// in Q7), so a uniform reference must produce that same value at every
	// output sample regardless of scale factors or sub-pel phase.
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, 0xa5)
		}
	}
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, 0x5a)
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, 0x33)
		}
	}
	// Seed a sentinel byte that the scaled convolver must overwrite.
	output.Y.Pix[16*output.Y.Stride+16] = 0x00

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	if err := ctx.PredictBlockInterWithFilters(0, visit, nil, motion.RegularFilters); err != nil {
		t.Fatalf("PredictBlockInterWithFilters err=%v", err)
	}
	if got := output.Y.Pix[16*output.Y.Stride+16]; got != 0xa5 {
		t.Fatalf("scaled prediction did not write uniform ref value at (16,16): got=%#x want=%#x", got, 0xa5)
	}
}

// TestFrameWorkBatchPredictBlockInterWarpRejectsScaledReferenceBeforeMutation
// pins the default-build behavior: warp+scaled is still a hard reject when the
// scaled-prediction gate is off, matching the same-size invariant that the
// 7-vector fast suite relies on.
func TestFrameWorkBatchPredictBlockInterWarpRejectsScaledReferenceBeforeMutation(t *testing.T) {
	if frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch enabled; same-size-only rejection is bypassed")
	}
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	fillFrameWorkInterReference(reference, 0xff)
	output.Y.Pix[16*output.Y.Stride+16] = 0x44

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	visit.Prediction.WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true

	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("PredictBlockLumaInterWithFilters err=%v want %v", err, ErrInvalidBatch)
	}
	if output.Y.Pix[16*output.Y.Stride+16] != 0x44 {
		t.Fatalf("output mutated y=%#x", output.Y.Pix[16*output.Y.Stride+16])
	}
}

// TestFrameWorkBatchPredictBlockInterWarpFallsBackToScaledTranslation anchors
// the libaom behavior described in allow_warp() (av1/common/reconinter.c):
// when av1_is_scaled(sf) is true, allow_warp() returns 0 and the predictor
// mode stays TRANSLATION_PRED, so av1_make_inter_predictor() runs the scaled
// 8-tap convolver on the block-level MV instead of the warp matrix. The
// warp-dispatch output must therefore match the dedicated scaled translational
// helper byte-for-byte.
func TestFrameWorkBatchPredictBlockInterWarpFallsBackToScaledTranslation(t *testing.T) {
	if !frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch disabled; set GOAV1_SCALED_PRED=1 or build with goav1_scaled_pred to exercise")
	}
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	// Deterministic pseudo-random reference so the scaled convolver does
	// real work (no uniform shortcut); the equality check below catches
	// any drift between the warp-dispatch fallback and the libaom
	// reference (translational scaled convolver on the block-level MV).
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*31+x*17+5)&0xff))
		}
	}

	mv := motion.Vector{Row: 5, Col: -3}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	// Drive the warp dispatch with a non-identity affine wm: per libaom,
	// the scaled-ref gate must downgrade this to translation and ignore
	// the warp matrix entirely.
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(mv)
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	// Perturb the affine matrix so a real warp would diverge from
	// translation; the equality check below proves the matrix is unused.
	params.Matrix[2] = 1<<16 + 123
	params.Matrix[5] = 1<<16 - 123
	visit.Prediction.WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatalf("PredictBlockLumaInterWithFilters err=%v", err)
	}

	// Build the libaom reference: translational scaled convolver on the
	// block-level MV with the same filters, written at the same plane
	// position.
	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlane(geom.Output, refPlane, geom.BytesPerSample, want.Format.BitDepth,
		geom.X, geom.Y, geom.X, geom.Y, geom.Width, geom.Height, mv, geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("frameWorkPredictScaledReferencePlane err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.Width, geom.Height)
}

// TestFrameWorkBatchPredictBlockInterGlobalWarpFallsBackToScaledTranslation
// is the GLOBALMV analogue of the warp test above. allow_warp() applies the
// same is_scaled gate to frame-level global motion, so a non-translational
// global warp model paired with a scaled reference must also collapse to the
// translational scaled convolver on the block-level MV.
func TestFrameWorkBatchPredictBlockInterGlobalWarpFallsBackToScaledTranslation(t *testing.T) {
	if !frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch disabled; set GOAV1_SCALED_PRED=1 or build with goav1_scaled_pred to exercise")
	}
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*23+x*13+7)&0xff))
		}
	}

	mv := motion.Vector{Row: -4, Col: 6}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapRegular}

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(mv)
	// MotionMode stays SIMPLE_TRANSLATION so dispatch falls through to
	// the GlobalMV branch; the scaled-ref gate must override the global
	// warp model just like it does for block-level warp.
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeTranslation
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	params.Matrix[2] = 1<<16 + 77
	params.Matrix[5] = 1<<16 - 77
	visit.Prediction.GlobalWarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.GlobalWarpedMotionValid = true
	if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
		t.Fatalf("PredictBlockLumaInterWithFilters err=%v", err)
	}

	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlane(geom.Output, refPlane, geom.BytesPerSample, want.Format.BitDepth,
		geom.X, geom.Y, geom.X, geom.Y, geom.Width, geom.Height, mv, geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("frameWorkPredictScaledReferencePlane err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.Width, geom.Height)
}

// TestFrameWorkBatchPredictBlockInterOBMCScaledRoutesNeighborThroughScaledConvolver
// pins the libaom behaviour described in dec_calc_subpel_params /
// av1_make_inter_predictor (av1/common/reconinter.c) for OBMC neighbor
// predictions against scaled references. The main block prediction already
// dispatches through the scaled convolver, but each OBMC overlappable
// neighbor prediction also has to go through the same scaled path —
// otherwise the neighbor area is filled from the wrong source-plane
// coordinates and the OBMC blend writes garbage into the leftmost / topmost
// overlap columns of the block.
//
// Concretely: with a 32x32 reference scaled into a 64x64 output, OBMC LEFT
// for the central 16x16 block at (16, 16) must produce the same leftmost
// 8 columns as a plain scaled translational predictor on the LEFT
// neighbor's MV, then blended into the base prediction via the 8-wide
// OBMC mask. Before the fix predictInterReferenceAreaToScratch dropped
// to the same-size convolver and corrupted those 8 columns; the SVC L2T1
// spatial=1 enhancement layer flagged it at dst (96..103, 0) of frame 0.
func TestFrameWorkBatchPredictBlockInterOBMCScaledRoutesNeighborThroughScaledConvolver(t *testing.T) {
	if !frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch disabled; set GOAV1_SCALED_PRED=1 or build with goav1_scaled_pred to exercise")
	}
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, MonoChrome: true, Align: 64})
	// Deterministic pseudo-random reference so the scaled convolver does
	// real work and an off-by-scale neighbor read would diverge sample-
	// for-sample from the libaom reference.
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*29+x*19+11)&0xff))
		}
	}

	baseMV := motion.Vector{Row: 4, Col: -6}
	leftMV := motion.Vector{Row: -3, Col: 11}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	ctx := testInterPredictionBatch(output, reference)
	// Single LEFT neighbor with a distinct MV so the OBMC blend reads
	// from a different source-plane area than the base prediction.
	visit := testInterPredictionVisit(baseMV)
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	visit.Prediction.OverlappableNeighbors.LeftCount = 1
	visit.Prediction.OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{leftMV},
		},
		InterpFilters:      filters,
		InterpFiltersValid: true,
	}
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterOBMCWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatalf("PredictBlockLumaInterOBMCWithFilters err=%v", err)
	}

	// libaom reference: scaled translational predictor for the base block,
	// then a scaled neighbor prediction blended into the leftmost overlap
	// columns via the OBMC mask.
	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlane(geom.Output, refPlane, geom.BytesPerSample, want.Format.BitDepth,
		geom.X, geom.Y, geom.X, geom.Y, geom.Width, geom.Height, baseMV, geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("base scaled-ref err=%v", err)
	}
	overlap, err := frameWorkOBMCLeftWidth(visit.Block.Size, geom)
	if err != nil {
		t.Fatalf("OBMC left width err=%v", err)
	}
	if overlap <= 0 || overlap > geom.Width {
		t.Fatalf("unexpected overlap=%d width=%d", overlap, geom.Width)
	}
	height, ok := frameWorkOBMCPlaneSpan(visit.Prediction.OverlappableNeighbors.Left[0].Span4, geom.SubsamplingY)
	if !ok {
		t.Fatalf("OBMC plane span invalid")
	}
	if height > geom.Height {
		height = geom.Height
	}
	var neighborScratch FrameWorkInterPredictionScratch
	neighbor, err := frameWorkInterScratchPlane(neighborScratch.First[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		t.Fatalf("neighbor scratch err=%v", err)
	}
	// Neighbor prediction must go through the scaled convolver anchored
	// to the same geom.Output dimensions as the base prediction; this is
	// what predictInterReferenceAreaToScratch must do under the fix.
	if err := frameWorkPredictScaledReferencePlaneWithDims(neighbor, refPlane, geom.Output.Width, geom.Output.Height,
		geom.BytesPerSample, want.Format.BitDepth, 0, 0, geom.X, geom.Y, overlap, height, leftMV,
		geom.SubsamplingX, geom.SubsamplingY, filters); err != nil {
		t.Fatalf("neighbor scaled-ref err=%v", err)
	}
	mask, ok := frameWorkOBMCMask(overlap)
	if !ok {
		t.Fatalf("missing OBMC mask for overlap=%d", overlap)
	}
	if err := frameWorkBlendOBMCH(want.Y, neighbor, geom.BytesPerSample, geom.X, geom.Y, 0, 0, overlap, height, mask); err != nil {
		t.Fatalf("blend OBMC H err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.Width, geom.Height)
}

// TestFrameWorkBatchPredictBlockInterWarpScaledAllocs locks the steady-state
// allocation profile for the warp+scaled fallback path so the SVC vectors
// stay zero-alloc per visit.
func TestFrameWorkBatchPredictBlockInterWarpScaledAllocs(t *testing.T) {
	if !frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch disabled; set GOAV1_SCALED_PRED=1 or build with goav1_scaled_pred to exercise")
	}
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 64, BitDepth: 8, Align: 64})
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Row: 3, Col: -5})
	visit.Prediction.MotionModeValid = true
	visit.Prediction.MotionMode = tile.MotionModeWarp
	params := parser.DefaultWarpedMotionParams()
	params.Type = parser.GlobalMotionAffine
	visit.Prediction.WarpedMotion = tile.WarpedMotionModel{Params: params}
	visit.Prediction.WarpedMotionValid = true
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warp+scaled fallback allocated: %f", allocs)
	}
}

// TestFrameWorkBatchPredictBlockInterIntraScaledMatchesScaledTranslation
// anchors the libaom behavior described in av1_combine_interintra() and
// av1_make_inter_predictor() (av1/common/reconinter.c): inter-intra blocks
// stage the inter side into a block-sized scratch buffer, and when the
// reference is scaled the scaled 8-tap convolver runs there. The Q14
// reference scale factors come from xd->block_ref_scale_factors, which are
// derived per-frame from the *output frame size*, not from the staging
// buffer. The blend mask and intra side are independent of scaling.
//
// Concretely: the inter-intra luma output for a scaled-reference block must
// equal the same per-sample mask blend of (a) the scaled translational
// predictor on the block MV anchored to the output frame, and (b) the intra
// predictor for the configured mode. This test pins that equivalence
// sample-for-sample on a deterministic 32x32 -> 64x64 reference. If the
// staging path were to derive scale factors from the scratch buffer (16x16)
// instead of the output frame (64x64) the resulting samples would diverge
// from the libaom reference.
func TestFrameWorkBatchPredictBlockInterIntraScaledMatchesScaledTranslation(t *testing.T) {
	if !frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch disabled; set GOAV1_SCALED_PRED=1 or build with goav1_scaled_pred to exercise")
	}
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64}
	output := testBatchFrame(t, format)
	want := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, MonoChrome: true, Align: 64})
	// Deterministic pseudo-random reference so the scaled convolver does
	// real work and any drift between the geom-anchored and (buggy)
	// scratch-anchored scale factors surfaces sample-for-sample.
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((y*29+x*19+11)&0xff))
		}
	}
	// Seed the intra-side neighbors at the block boundary so the smooth
	// intra predictor produces a non-trivial signal that the mask must
	// actually blend (otherwise inter and intra branches could coincide
	// and hide the scaled-vs-scratch dimension bug).
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 200)
		setFrameWorkTestSample(want.Y, want.Layout.BytesPerSample, x, 15, 200)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 40)
		setFrameWorkTestSample(want.Y, want.Layout.BytesPerSample, 15, y, 40)
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 120)
	setFrameWorkTestSample(want.Y, want.Layout.BytesPerSample, 15, 15, 120)

	mv := motion.Vector{Row: 4, Col: -6}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}

	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeSmooth, false)
	visit.Prediction.InterMotion.MV[0] = mv
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
		t.Fatalf("PredictBlockInterWithFilters err=%v", err)
	}

	// Build the libaom reference: scaled translational predictor anchored
	// to the output frame size, blended with the configured intra mode via
	// the same per-sample mask. This drives the same internal helpers as
	// the inter-intra dispatch except for the (geom-anchored) scale factor
	// derivation, which is the new path under test.
	wantCtx := testInterPredictionBatch(want, reference)
	geom, ok, err := wantCtx.blockPredictionPlaneGeometry(0, visit.Block, FrameWorkPlaneY)
	if err != nil || !ok {
		t.Fatalf("plane geometry err=%v ok=%v", err, ok)
	}
	var interBuf, intraBuf FrameWorkInterPredictionScratch
	inter, err := frameWorkInterScratchPlane(interBuf.First[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		t.Fatalf("inter scratch err=%v", err)
	}
	intra, err := frameWorkInterScratchPlane(intraBuf.Second[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		t.Fatalf("intra scratch err=%v", err)
	}
	refPlane := frame.Plane{Pix: reference.Y.Pix, Stride: reference.Y.Stride, Width: reference.Y.Width, Height: reference.Y.Height}
	if err := frameWorkPredictScaledReferencePlaneToBuffer(inter, refPlane, geom, want.Format.BitDepth,
		0, 0, geom.X, geom.Y, mv, filters); err != nil {
		t.Fatalf("scaled-ref to buffer err=%v", err)
	}
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	var intraScratch FrameWorkIntraPredictionScratch
	edges, err := frameWorkIntraPredictionEdges(geom.Output, geom.BytesPerSample, want.Format.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, edgeBlock, &intraScratch, true)
	if err != nil {
		t.Fatalf("intra edges err=%v", err)
	}
	if err := prediction.PredictIntraPlaneBlock(intra, geom.BytesPerSample, want.Format.BitDepth, 0, 0, geom.Width, geom.Height, prediction.IntraModeSmooth, edges); err != nil {
		t.Fatalf("intra predict err=%v", err)
	}
	var mask [frameWorkInterPredictionMaxMaskSamples]byte
	if err := frameWorkBuildInterIntraMask(mask[:], geom.Width, geom.Width, geom.Height, tile.InterIntraModeSmooth); err != nil {
		t.Fatalf("build mask err=%v", err)
	}
	if err := frameWorkBlendInterIntraBlock(geom.Output, inter, intra, geom.BytesPerSample, want.Format.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, mask[:], geom.Width, false, false); err != nil {
		t.Fatalf("blend err=%v", err)
	}

	assertFrameWorkPlaneBlockEqual(t, output.Y, want.Y, output.Layout.BytesPerSample, geom.X, geom.Y, geom.Width, geom.Height)
}

// TestFrameWorkBatchPredictBlockInterIntraScaledAllocs locks the steady-state
// allocation profile for the inter-intra+scaled fallback path so the SVC
// vectors stay zero-alloc per visit (the inter-intra branch stages into the
// caller-owned scratch and the scaled convolver writes directly into it).
func TestFrameWorkBatchPredictBlockInterIntraScaledAllocs(t *testing.T) {
	if !frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch disabled; set GOAV1_SCALED_PRED=1 or build with goav1_scaled_pred to exercise")
	}
	format := frame.Format{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64}
	output := testBatchFrame(t, format)
	reference := testBatchFrame(t, frame.Format{Width: 32, Height: 32, BitDepth: 8, MonoChrome: true, Align: 64})
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterIntraPredictionVisit(tile.InterIntraModeSmooth, false)
	visit.Prediction.InterMotion.MV[0] = motion.Vector{Row: 3, Col: -2}
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	var scratch FrameWorkInterPredictionScratch
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockInterWithFilters(0, visit, &scratch, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("inter-intra+scaled fallback allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockInterCompoundRejectsScaledReferenceBeforeMutation(t *testing.T) {
	if frameWorkScaledRefEnabled() {
		t.Skip("scaled-reference dispatch enabled; same-size-only rejection is bypassed")
	}
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	last := testBatchFrame(t, output.Format)
	bwd := testBatchFrame(t, frame.Format{Width: 64, Height: 32, BitDepth: 8, Align: 64})
	output.Y.Pix[16*output.Y.Stride+16] = 0x77
	fillFrameWorkInterReference(last, 0xff)
	fillFrameWorkInterReference(bwd, 0xff)

	ctx := testCompoundInterPredictionBatch(output, last, bwd)
	visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
	var scratch FrameWorkInterPredictionScratch
	if err := ctx.PredictBlockLumaInterCompoundWithFilters(0, visit, &scratch, motion.RegularFilters); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("PredictBlockLumaInterCompoundWithFilters err=%v want %v", err, ErrInvalidBatch)
	}
	if output.Y.Pix[16*output.Y.Stride+16] != 0x77 {
		t.Fatalf("output mutated y=%#x", output.Y.Pix[16*output.Y.Stride+16])
	}
}

func TestFrameWorkBatchPredictBlockLumaInterAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 10, Align: 128})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0x3ff)
	ctx := testInterPredictionBatch(output, reference)
	visit := testInterPredictionVisit(motion.Vector{Col: 4, Row: 6})
	filters := motion.InterpFilters{X: motion.InterpEightTapRegular, Y: motion.InterpEightTapSmooth}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaInterWithFilters(0, visit, filters); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaInter allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaDispatchesIntraAndInter(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)

	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 10)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 50)
	}

	var scratch FrameWorkPredictionScratch
	if err := ctx.PredictBlockLuma(0, testIntraPredictionVisit(tile.IntraModeDC), &scratch); err != nil {
		t.Fatal(err)
	}
	if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("intra dispatch sample=%d want 30", got)
	}

	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	if err := ctx.PredictBlockLuma(0, inter, nil); err != nil {
		t.Fatal(err)
	}
	got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, 16, 16)
	want := frameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, 17, 16)
	if got != want {
		t.Fatalf("inter dispatch sample=%d want %d", got, want)
	}
}

func TestFrameWorkBatchPredictBlockLumaRejectsInvalidDispatch(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	ctx := testInterPredictionBatch(output, reference)
	if err := ctx.PredictBlockLuma(0, tile.BlockLoopVisit{}, nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("missing prediction err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockLuma(0, testIntraPredictionVisit(tile.IntraModeDC), nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil intra scratch err=%v want %v", err, ErrInvalidBatch)
	}
	if err := ctx.PredictBlockLuma(0, func() tile.BlockLoopVisit {
		visit := testCompoundInterPredictionVisit(motion.Vector{}, motion.Vector{}, tile.CompoundTypeAverage)
		return visit
	}(), nil); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("nil compound scratch err=%v want %v", err, ErrInvalidBatch)
	}
}

func TestFrameWorkBatchPredictBlockLumaAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	reference := testBatchFrame(t, output.Format)
	fillFrameWorkInterReference(reference, 0xff)
	ctx := testInterPredictionBatch(output, reference)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	var scratch FrameWorkPredictionScratch
	intra := testIntraPredictionVisit(tile.IntraModeDC)
	inter := testInterPredictionVisit(motion.Vector{Col: 8, Row: 0})
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLuma(0, intra, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlockLuma(0, inter, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLuma allocated: %f", allocs)
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraRejectsInvalidInputs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	valid := testIntraPredictionVisit(tile.IntraModeDC)
	var scratch FrameWorkIntraPredictionScratch

	tests := []struct {
		name    string
		ctx     FrameWorkBatch
		visit   tile.BlockLoopVisit
		scratch *FrameWorkIntraPredictionScratch
	}{
		{name: "nil scratch", ctx: ctx, visit: valid},
		{name: "missing prediction", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Valid = false
			return visit
		}(), scratch: &scratch},
		{name: "inter", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Prediction.Intra = false
			return visit
		}(), scratch: &scratch},
		{name: "outside job", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Block.MICol = 16
			visit.Block.MIColEnd = 20
			return visit
		}(), scratch: &scratch},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaIntra(0, tt.visit, tt.scratch); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
}

func TestFrameWorkBatchPredictBlockLumaIntraAllocs(t *testing.T) {
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	ctx := testIntraPredictionBatch(output)
	for x := 16; x < 32; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, 90)
	}
	for y := 16; y < 32; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, 92)
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, 91)

	var scratch FrameWorkIntraPredictionScratch
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	directional := testIntraPredictionVisit(tile.IntraModeD135)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
		if err := ctx.PredictBlockLumaIntra(0, directional, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PredictBlockLumaIntra allocated: %f", allocs)
	}
}

func FuzzFrameWorkBatchPredictBlockLumaIntra(f *testing.F) {
	f.Add(uint8(0), uint16(90), uint16(92), uint16(88))
	f.Add(uint8(1), uint16(12), uint16(200), uint16(40))
	f.Add(uint8(6), uint16(511), uint16(700), uint16(512))

	modes := [...]tile.IntraMode{
		tile.IntraModeDC,
		tile.IntraModeVertical,
		tile.IntraModeHorizontal,
		tile.IntraModeSmooth,
		tile.IntraModeSmoothVertical,
		tile.IntraModeSmoothHorizontal,
		tile.IntraModePaeth,
		tile.IntraModeD45,
		tile.IntraModeD67,
		tile.IntraModeD113,
		tile.IntraModeD135,
		tile.IntraModeD157,
		tile.IntraModeD203,
	}
	f.Fuzz(func(t *testing.T, rawMode uint8, above uint16, left uint16, aboveLeft uint16) {
		output := testBatchFrame(t, frame.Format{Width: 96, Height: 96, BitDepth: 10, Align: 128})
		ctx := testIntraPredictionBatch(output)
		above &= 0x3ff
		left &= 0x3ff
		aboveLeft &= 0x3ff
		seedFrameWorkDirectionalEdges(output, 0x3ff, above, left, aboveLeft)

		var scratch FrameWorkIntraPredictionScratch
		visit := testIntraPredictionVisit(modes[int(rawMode)%len(modes)])
		if err := ctx.PredictBlockLumaIntra(0, visit, &scratch); err != nil {
			t.Fatalf("PredictBlockLumaIntra err=%v", err)
		}
		for y := 16; y < 32; y++ {
			for x := 16; x < 32; x++ {
				if got := frameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y); got > 0x3ff {
					t.Fatalf("sample(%d,%d)=%d exceeds 10-bit max", x, y, got)
				}
			}
		}
	})
}

func testCFLPredictionVisit() tile.BlockLoopVisit {
	visit := testIntraPredictionVisit(tile.IntraModeDC)
	visit.Prediction.ChromaMode = tile.ChromaIntraModeCFL
	visit.Prediction.ChromaModeValid = true
	visit.Prediction.CFLAlpha = tile.CFLAlphaResult{Index: 0x21, JointSign: 7}
	visit.Prediction.CFLAlphaValid = true
	return visit
}

func seedFrameWorkCFLPredictionFrame(output *frame.Frame) {
	for y := 16; y < 32; y++ {
		for x := 16; x < 32; x++ {
			value := uint16(35 + ((x-16)*7+(y-16)*11)%180)
			setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, y, value)
		}
	}
	for x := 8; x < 16; x++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, x, 7, uint16(50+x))
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, x, 7, uint16(70+x))
	}
	for y := 8; y < 16; y++ {
		setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 7, y, uint16(90+y))
		setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 7, y, uint16(110+y))
	}
	setFrameWorkTestSample(output.U, output.Layout.BytesPerSample, 7, 7, 80)
	setFrameWorkTestSample(output.V, output.Layout.BytesPerSample, 7, 7, 100)
}

func testPredictFrameWorkCFLWant(t *testing.T, output *frame.Frame, visit tile.BlockLoopVisit, plane FrameWorkPlane) {
	t.Helper()
	var dst frame.Plane
	var predType prediction.CFLPredType
	switch plane {
	case FrameWorkPlaneU:
		dst = output.U
		predType = prediction.CFLPredU
	case FrameWorkPlaneV:
		dst = output.V
		predType = prediction.CFLPredV
	default:
		t.Fatalf("bad plane=%d", plane)
	}
	recon := make([]uint16, prediction.CFLBufSquare)
	ac := make([]int16, prediction.CFLBufSquare)
	offset := 16*output.Y.Stride + 16
	if err := prediction.SubsampleLuma8ToQ3(recon, output.Y.Pix[offset:], output.Y.Stride, 16, 16, true, true); err != nil {
		t.Fatal(err)
	}
	if err := prediction.SubtractCFLAverage(recon, ac, 8, 8); err != nil {
		t.Fatal(err)
	}
	edges := testFrameWorkIntraEdges(output, dst, 8, 8, 8, 8)
	if err := prediction.PredictIntraPlaneBlock(dst, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, prediction.IntraModeDC, edges); err != nil {
		t.Fatal(err)
	}
	alphaQ3, err := prediction.CFLAlphaQ3(visit.Prediction.CFLAlpha.Index, visit.Prediction.CFLAlpha.JointSign, predType)
	if err != nil {
		t.Fatal(err)
	}
	if err := prediction.PredictCFLPlaneBlock(dst, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, ac, alphaQ3); err != nil {
		t.Fatal(err)
	}
}

func testFrameWorkIntraEdges(output *frame.Frame, plane frame.Plane, x int, y int, width int, height int) prediction.IntraEdges {
	above := make([]uint16, width)
	left := make([]uint16, height)
	for col := 0; col < width; col++ {
		above[col] = frameWorkTestSample(plane, output.Layout.BytesPerSample, x+col, y-1)
	}
	for row := 0; row < height; row++ {
		left[row] = frameWorkTestSample(plane, output.Layout.BytesPerSample, x-1, y+row)
	}
	return prediction.IntraEdges{
		Above:              above,
		Left:               left,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeft:          frameWorkTestSample(plane, output.Layout.BytesPerSample, x-1, y-1),
		AboveLeftAvailable: true,
	}
}

func testIntraPredictionBatch(output *frame.Frame) FrameWorkBatch {
	return FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
}

func testInterPredictionBatch(output *frame.Frame, reference *frame.Frame) FrameWorkBatch {
	return FrameWorkBatch{
		Output:     output,
		References: []*frame.Frame{reference},
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				ColorConfig: parser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
			TileInfo:  parser.TileInfo{InterpolationFilter: parser.InterpolationEightTap},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
}

func testCompoundInterPredictionBatch(output *frame.Frame, last *frame.Frame, bwd *frame.Frame) FrameWorkBatch {
	ctx := testInterPredictionBatch(output, last)
	refs := make([]*frame.Frame, int(FrameWorkReferenceBwd)+1)
	refs[int(FrameWorkReferenceLast)] = last
	refs[int(FrameWorkReferenceBwd)] = bwd
	ctx.References = refs
	ctx.Sequence.EnableOrderHint = true
	ctx.Sequence.OrderHintBits = 4
	ctx.FrameHeader.OrderHint = 8
	ctx.ReferenceOrderHints[tile.ReferenceFrameLast] = 4
	ctx.ReferenceOrderHints[tile.ReferenceFrameBWD] = 12
	return ctx
}

func testIntraPredictionVisit(mode tile.IntraMode) tile.BlockLoopVisit {
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:    true,
			Intra:    true,
			LumaMode: mode,
		},
	}
}

func testInterPredictionVisit(mv motion.Vector) tile.BlockLoopVisit {
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameNone}}
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			Intra:                false,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMotion: tile.InterMotionResult{
				References: refs,
				MV:         [2]motion.Vector{mv},
			},
			InterMotionValid: true,
		},
	}
}

func testInterIntraPredictionVisit(mode tile.InterIntraMode, wedge bool) tile.BlockLoopVisit {
	visit := testInterPredictionVisit(motion.Vector{})
	visit.Prediction.InterIntra = tile.InterIntraResult{
		Enabled:  true,
		Mode:     mode,
		UseWedge: wedge,
	}
	visit.Prediction.InterIntraValid = true
	visit.Prediction.InterMotion.InterIntra = true
	visit.Prediction.MotionMode = tile.MotionModeTranslation
	visit.Prediction.MotionModeValid = true
	return visit
}

func testOBMCInterPredictionVisit(baseMV motion.Vector, aboveMV motion.Vector, leftMV motion.Vector) tile.BlockLoopVisit {
	visit := testInterPredictionVisit(baseMV)
	visit.Prediction.MotionMode = tile.MotionModeOBMC
	visit.Prediction.MotionModeValid = true
	visit.Prediction.OverlappableNeighborsValid = true
	visit.Prediction.OverlappableNeighbors.AboveCount = 1
	visit.Prediction.OverlappableNeighbors.Above[0] = tile.OverlappableNeighbor{
		RelX4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{aboveMV},
		},
		InterpFilters:      motion.RegularFilters,
		InterpFiltersValid: true,
	}
	visit.Prediction.OverlappableNeighbors.LeftCount = 1
	visit.Prediction.OverlappableNeighbors.Left[0] = tile.OverlappableNeighbor{
		RelY4: 0,
		Span4: 4,
		Size:  tile.BlockSize16x16,
		Motion: tile.InterMotionResult{
			References: visit.Prediction.InterMotion.References,
			MV:         [2]motion.Vector{leftMV},
		},
		InterpFilters:      motion.RegularFilters,
		InterpFiltersValid: true,
	}
	return visit
}

func testCompoundInterPredictionVisit(mv0 motion.Vector, mv1 motion.Vector, compoundType tile.CompoundType) tile.BlockLoopVisit {
	refs := tile.InterReferencesResult{Ref: [2]tile.ReferenceFrame{tile.ReferenceFrameLast, tile.ReferenceFrameBWD}, Compound: true}
	compoundIndex := uint8(0)
	if compoundType == tile.CompoundTypeAverage {
		compoundIndex = 1
	}
	return tile.BlockLoopVisit{
		Block: tile.BlockVisit{
			MICol: 4, MIRow: 4, MIColEnd: 8, MIRowEnd: 8,
			X4: 4, Y4: 4, Size: tile.BlockSize16x16, VisibleW4: 4, VisibleH4: 4,
			HaveTop: true, HaveLeft: true,
		},
		Prediction: tile.BlockPredictionModeResult{
			Valid:                true,
			Intra:                false,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMode: tile.InterModeResult{
				Compound:     true,
				CompoundMode: tile.CompoundInterModeNearestNearest,
			},
			InterModeValid: true,
			InterMotion: tile.InterMotionResult{
				References: refs,
				Mode: tile.InterModeResult{
					Compound:     true,
					CompoundMode: tile.CompoundInterModeNearestNearest,
				},
				MV: [2]motion.Vector{mv0, mv1},
			},
			InterMotionValid: true,
			MotionMode:       tile.MotionModeTranslation,
			MotionModeValid:  true,
			CompoundBlend: tile.CompoundBlendResult{
				Type:          compoundType,
				CompoundIndex: compoundIndex,
			},
			CompoundBlendValid: true,
		},
	}
}

func testIntraPrediction4x4Visit(mode tile.IntraMode) tile.BlockLoopVisit {
	visit := testIntraPredictionVisit(mode)
	visit.Block.MIColEnd = 5
	visit.Block.MIRowEnd = 5
	visit.Block.Size = tile.BlockSize4x4
	visit.Block.VisibleW4 = 1
	visit.Block.VisibleH4 = 1
	return visit
}

func frameWorkTestSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	sample, ok := frameWorkLoadSample(plane, bytesPerSample, x, y)
	if !ok {
		panic("bad test sample")
	}
	return sample
}

func frameWorkTestSampleClamped(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	if x < 0 {
		x = 0
	} else if x >= plane.Width {
		x = plane.Width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= plane.Height {
		y = plane.Height - 1
	}
	return frameWorkTestSample(plane, bytesPerSample, x, y)
}

func assertFrameWorkPlaneBlockEqual(t *testing.T, got frame.Plane, want frame.Plane, bytesPerSample int, x int, y int, width int, height int) {
	t.Helper()
	assertFrameWorkPlaneBlockEqualAt(t, got, x, y, want, x, y, bytesPerSample, width, height)
}

func assertFrameWorkPlaneBlockEqualAt(t *testing.T, got frame.Plane, gotX int, gotY int, want frame.Plane, wantX int, wantY int, bytesPerSample int, width int, height int) {
	t.Helper()
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			g := frameWorkTestSample(got, bytesPerSample, gotX+col, gotY+row)
			w := frameWorkTestSample(want, bytesPerSample, wantX+col, wantY+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", gotX+col, gotY+row, g, w)
			}
		}
	}
}

// testFrameWorkCompoundConvBuf produces the libaom-faithful un-rounded 16-bit
// CONV_BUF predictor (av1_dist_wtd_convolve_*) for one reference, matching the
// production compound prediction path. The reference helpers below blend two of
// these buffers and round once, exactly like av1_dist_wtd_convolve_* /
// aom_*_blend_a64_d16_mask_c.
func testFrameWorkCompoundConvBuf(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) *motion.CompoundConvBuf {
	t.Helper()
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		t.Fatal(err)
	}
	buf := &motion.CompoundConvBuf{}
	if err := motion.PredictInterCompoundRefToConvBuf(buf, reference, bytesPerSample, bitDepth, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
	return buf
}

func assertFrameWorkCompoundBlendEqual(t *testing.T, got frame.Plane, buf0 *motion.CompoundConvBuf, buf1 *motion.CompoundConvBuf, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, fwdOffset int, bckOffset int) {
	t.Helper()
	stride := width * bytesPerSample
	want := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	if err := motion.BlendCompoundAvg(want, buf0, buf1, bytesPerSample, bitDepth, 0, 0, width, height, fwdOffset, bckOffset); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			w := frameWorkTestSample(want, bytesPerSample, col, row)
			g := frameWorkTestSample(got, bytesPerSample, x+col, y+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, w)
			}
		}
	}
}

func assertFrameWorkMaskedCompoundEqual(t *testing.T, got frame.Plane, buf0 *motion.CompoundConvBuf, buf1 *motion.CompoundConvBuf, mask []byte, maskStride int, subX bool, subY bool, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int) {
	t.Helper()
	stride := width * bytesPerSample
	want := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	if err := motion.BlendCompoundMaskD16(want, buf0, buf1, bytesPerSample, bitDepth, 0, 0, width, height, mask, maskStride, subX, subY); err != nil {
		t.Fatal(err)
	}
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			w := frameWorkTestSample(want, bytesPerSample, col, row)
			g := frameWorkTestSample(got, bytesPerSample, x+col, y+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, w)
			}
		}
	}
}

func setFrameWorkTestSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func testFrameWorkMotionPredictionPlane(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, filters motion.InterpFilters) frame.Plane {
	t.Helper()
	stride := width * bytesPerSample
	dst := frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
	refX, refY, subX, subY, err := motion.ReferenceOrigin(dstX, dstY, mv)
	if err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, reference, bytesPerSample, bitDepth, 0, 0, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
	return dst
}

func testFrameWorkMotionPredictionPlaneSubsampled(t *testing.T, reference frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv motion.Vector, subsamplingX bool, subsamplingY bool, filters motion.InterpFilters) frame.Plane {
	t.Helper()
	stride := width * bytesPerSample
	dst := frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(dstX, dstY, mv, subsamplingX, subsamplingY)
	if err != nil {
		t.Fatal(err)
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, reference, bytesPerSample, bitDepth, 0, 0, refX, refY, width, height, subX, subY, filters); err != nil {
		t.Fatal(err)
	}
	return dst
}

func testFrameWorkDiffWtdMask(t *testing.T, buf0 *motion.CompoundConvBuf, buf1 *motion.CompoundConvBuf, bitDepth uint8, width int, height int, maskType tile.DiffWtdMaskType) []byte {
	t.Helper()
	mask := make([]byte, width*height)
	if err := motion.BuildDiffWtdMaskD16(mask, width, buf0, buf1, bitDepth, width, height, maskType == tile.DiffWtdMaskType38Inv); err != nil {
		t.Fatal(err)
	}
	return mask
}

func testFrameWorkBlendMaskSample(t *testing.T, mask []byte, stride int, row int, col int, subX bool, subY bool) uint8 {
	t.Helper()
	var m int
	switch {
	case !subX && !subY:
		m = int(mask[row*stride+col])
	case subX && subY:
		m = (int(mask[(2*row)*stride+2*col]) +
			int(mask[(2*row+1)*stride+2*col]) +
			int(mask[(2*row)*stride+2*col+1]) +
			int(mask[(2*row+1)*stride+2*col+1]) + 2) >> 2
	case subX:
		m = (int(mask[row*stride+2*col]) + int(mask[row*stride+2*col+1]) + 1) >> 1
	default:
		m = (int(mask[(2*row)*stride+col]) + int(mask[(2*row+1)*stride+col]) + 1) >> 1
	}
	if m < 0 || m > 64 {
		t.Fatalf("mask sample=%d outside A64 range", m)
	}
	return uint8(m)
}

func fillFrameWorkInterReference(reference *frame.Frame, max uint16) {
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((x*x+3*y*y+17*x+11*y)&int(max)))
		}
	}
}

func fillFrameWorkInterReferenceAllPlanes(reference *frame.Frame, max uint16) {
	fillFrameWorkInterReference(reference, max)
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, uint16((5*x*x+7*y*y+13*x+19*y+23)&int(max)))
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, uint16((11*x*x+3*y*y+29*x+31*y+37)&int(max)))
		}
	}
}

func fillFrameWorkInterReferenceVariant(reference *frame.Frame, max uint16, seed uint16) {
	for y := 0; y < reference.Y.Height; y++ {
		for x := 0; x < reference.Y.Width; x++ {
			setFrameWorkTestSample(reference.Y, reference.Layout.BytesPerSample, x, y, uint16((29*x+31*y+int(seed))&int(max)))
		}
	}
	for y := 0; y < reference.U.Height; y++ {
		for x := 0; x < reference.U.Width; x++ {
			setFrameWorkTestSample(reference.U, reference.Layout.BytesPerSample, x, y, uint16((17*x+23*y+int(seed)*3)&int(max)))
			setFrameWorkTestSample(reference.V, reference.Layout.BytesPerSample, x, y, uint16((41*x+43*y+int(seed)*5)&int(max)))
		}
	}
}

func seedFrameWorkDirectionalEdges(output *frame.Frame, max uint16, above uint16, left uint16, aboveLeft uint16) {
	for x := 0; x < output.Y.Width; x++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, x, 15, uint16((int(above)+x)&int(max)))
	}
	for y := 0; y < output.Y.Height; y++ {
		setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, y, uint16((int(left)+y)&int(max)))
	}
	setFrameWorkTestSample(output.Y, output.Layout.BytesPerSample, 15, 15, aboveLeft&max)
}

// TestFrameWorkApplyDirectionalIntraEdgeFilterCornerBlock32x32D157 pins the
// directional intra-edge-filter dispatch for the 32x32 D157 luma block at the
// frame's top-left corner. The 34x34 fast vector exercises this exact
// configuration (mi=0,0, EnableIntraEdgeFilter=true, smoothNeighbor=false,
// HaveTop=false, HaveLeft=false, angle=151 from luma angle delta -2) as its
// first decoded block, so any future change to the dispatch in
// frameWorkApplyDirectionalIntraEdgeFilter that drops the corner-filter call,
// flips the strength derivation, or re-enables the (top, left) edge filters
// when the corresponding neighbour counts are zero would silently regress the
// vector's reconstruction. The fixture mirrors libaom's
// build_directional_and_filter_intra_predictors fallback when both
// n_top_px=0 and n_left_px=0: above defaults to 127, left defaults to 129,
// and the above-left default 128 is preserved by the corner filter
// (128*5 + 127*6 + 129*5 = 2048; (2048+8)>>4 = 128).
func TestFrameWorkApplyDirectionalIntraEdgeFilterCornerBlock32x32D157(t *testing.T) {
	const (
		width  = 32
		height = 32
		angle  = 151 // libaom p_angle for D157 with luma_angle_delta=-2
	)
	var scratch FrameWorkIntraPredictionScratch
	origin := frameWorkDirectionalEdgeOrigin
	for i := range scratch.Above {
		scratch.Above[i] = 127
	}
	for i := range scratch.Left {
		scratch.Left[i] = 129
	}
	scratch.Above[origin-1] = 128
	scratch.Left[origin-1] = 128
	block := tile.BlockVisit{
		MICol: 0, MIRow: 0, MIColEnd: 8, MIRowEnd: 8,
		Size: tile.BlockSize32x32, VisibleW4: 8, VisibleH4: 8,
		HaveTop: false, HaveLeft: false,
	}
	edges := prediction.DirectionalEdges{
		Above:       scratch.Above[:],
		Left:        scratch.Left[:],
		AboveOrigin: origin,
		LeftOrigin:  origin,
	}
	if err := frameWorkApplyDirectionalIntraEdgeFilter(8, width, height, angle, block, false, &edges, &scratch); err != nil {
		t.Fatalf("apply edge filter: %v", err)
	}
	// The corner filter rewrites above[-1] and left[-1] to
	// (left[0]*5 + above[-1]*6 + above[0]*5 + 8) >> 4. For the all-default
	// fallback (127/128/129) the result is again 128.
	if got := scratch.Above[origin-1]; got != 128 {
		t.Fatalf("above[-1]=%d want 128 (corner filter no-op)", got)
	}
	if got := scratch.Left[origin-1]; got != 128 {
		t.Fatalf("left[-1]=%d want 128 (corner filter no-op)", got)
	}
	// The above and left edge filters must be skipped because nTop=0 and
	// nLeft=0 — same as libaom's `if (n_top_px > 0)` guard. Confirm the
	// default above samples remain 127 and the default left samples remain
	// 129 (i.e. FilterIntraEdge was not run on them).
	for i := 0; i < width+height; i++ {
		if got := scratch.Above[origin+i]; got != 127 {
			t.Fatalf("above[%d]=%d want 127 (filter must be skipped)", i, got)
		}
		if got := scratch.Left[origin+i]; got != 129 {
			t.Fatalf("left[%d]=%d want 129 (filter must be skipped)", i, got)
		}
	}
	// UseIntraEdgeUpsample returns false for blockWH=64 with d=61>=40, so
	// the upsample step must NOT be marked active.
	if edges.UpsampleAbove {
		t.Fatalf("UpsampleAbove=true want false (d=61>=40 disables upsample for blockWH=64)")
	}
	if edges.UpsampleLeft {
		t.Fatalf("UpsampleLeft=true want false (d=-29, |d|>=8 + blockWH=64 disables upsample)")
	}
}

// TestFrameWorkDirectionalPredictionEdgesCornerBlock32x32D157 drives the
// full edge-fill + edge-filter dispatch via frameWorkDirectionalPredictionEdges
// for the 34x34 vector's first decoded block. The test seeds an empty plane,
// requests the corner-block edges, and asserts the returned DirectionalEdges
// match libaom's fallback for n_top_px=n_left_px=0 — bit-exactly equivalent
// to what TestPredictDirectionalIntraPlaneBlockCornerFrame32x32D157 pins at
// the prediction-API level.
func TestFrameWorkDirectionalPredictionEdgesCornerBlock32x32D157(t *testing.T) {
	const (
		width  = 32
		height = 32
		angle  = 151
	)
	output := testBatchFrame(t, frame.Format{Width: 64, Height: 64, BitDepth: 8, Align: 64})
	testFillFrame(output, 0)
	dst := frame.Plane{
		Pix:    output.Y.Pix,
		Stride: output.Y.Stride,
		Width:  output.Y.Width,
		Height: output.Y.Height,
	}
	var scratch FrameWorkIntraPredictionScratch
	block := tile.BlockVisit{
		MICol: 0, MIRow: 0, MIColEnd: 8, MIRowEnd: 8,
		Size: tile.BlockSize32x32, VisibleW4: 8, VisibleH4: 8,
		HaveTop: false, HaveLeft: false,
	}
	edges, err := frameWorkDirectionalPredictionEdges(dst, 1, 8, 0, 0, width, height, angle, block, &scratch, true, false, false, false, 0, 0)
	if err != nil {
		t.Fatalf("directional edges: %v", err)
	}
	origin := frameWorkDirectionalEdgeOrigin
	if edges.AboveOrigin != origin || edges.LeftOrigin != origin {
		t.Fatalf("edge origins: got above=%d left=%d want %d", edges.AboveOrigin, edges.LeftOrigin, origin)
	}
	// Corner-filter output preserves 128 for the all-default fallback.
	if got := edges.Above[origin-1]; got != 128 {
		t.Fatalf("above[-1]=%d want 128", got)
	}
	if got := edges.Left[origin-1]; got != 128 {
		t.Fatalf("left[-1]=%d want 128", got)
	}
	// Above samples remain at the missing-above default (127). Left samples
	// remain at the missing-left default (129). The edge filter is skipped
	// for both edges when no real neighbours exist.
	for i := 0; i < width+height-1; i++ {
		if got := edges.Above[origin+i]; got != 127 {
			t.Fatalf("above[%d]=%d want 127", i, got)
		}
		if got := edges.Left[origin+i]; got != 129 {
			t.Fatalf("left[%d]=%d want 129", i, got)
		}
	}
	// Upsample is disabled for blockWH=64 with d=|angle-90|=61>=40 (above)
	// and d=|angle-180|=29 in zone 2 (left).
	if edges.UpsampleAbove || edges.UpsampleLeft {
		t.Fatalf("upsample flags: above=%v left=%v want both false", edges.UpsampleAbove, edges.UpsampleLeft)
	}
}
