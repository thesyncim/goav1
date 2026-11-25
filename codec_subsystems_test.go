package goav1

import (
	"errors"
	"fmt"
	internaldsp "github.com/thesyncim/goav1/internal/av1/dsp"
	internalmotion "github.com/thesyncim/goav1/internal/av1/motion"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalprediction "github.com/thesyncim/goav1/internal/av1/prediction"
	internalquantize "github.com/thesyncim/goav1/internal/av1/quantize"
	internalreconstruct "github.com/thesyncim/goav1/internal/av1/reconstruct"
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"
	"os"
	"path/filepath"
	"testing"
)

func TestPublicTileExtendedTransformSets(t *testing.T) {
	square, err := internaltile.TransformSizeSquare(internaltile.TransformSize16x8)
	if err != nil {
		t.Fatal(err)
	}
	if square != internaltile.TransformSize8x8 {
		t.Fatalf("square=%d want 8x8", square)
	}
	up, err := internaltile.TransformSizeSquareUp(internaltile.TransformSize16x8)
	if err != nil {
		t.Fatal(err)
	}
	if up != internaltile.TransformSize16x16 {
		t.Fatalf("square up=%d want 16x16", up)
	}

	set, err := internaltile.ExtTXSetTypeFor(internaltile.TransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if set != internaltile.ExtTXSetDTT9IDTX1DDCT {
		t.Fatalf("set=%d want DTT9/IDTX/1D-DCT", set)
	}
	index, err := internaltile.ExtTXSetIndex(internaltile.TransformSize16x16, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if index != 2 {
		t.Fatalf("index=%d want 2", index)
	}
	count, err := internaltile.ExtTXTypeCount(set)
	if err != nil {
		t.Fatal(err)
	}
	if count != 12 {
		t.Fatalf("count=%d want 12", count)
	}
	symbol, err := internaltile.ExtTXSymbolForType(set, internaltransform.TypeDCTDCT)
	if err != nil {
		t.Fatal(err)
	}
	if symbol != 3 {
		t.Fatalf("symbol=%d want 3", symbol)
	}
	allowed, err := internaltile.ExtTXTypeAllowed(set, internaltransform.TypeVADST)
	if err != nil {
		t.Fatal(err)
	}
	if allowed {
		t.Fatal("VADST should be disallowed for the DTT9/IDTX/1D-DCT set")
	}
}

// truncateLastTileOBU returns tu with its last frame or tile-group OBU cut to
// half its payload, its size field rewritten so the OBU framing stays valid.
// The tile data then ends mid-symbol, which libaom and dav1d reject as a
// symbol-decoder overread.
func truncateLastTileOBU(t *testing.T, tu []byte) []byte {
	t.Helper()
	var units []internalobu.Unit
	it := internalobu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse OBU: %v", err)
		}
		if !ok {
			break
		}
		units = append(units, unit)
	}
	last := -1
	for i, unit := range units {
		if unit.Header.Type == internalobu.TypeFrame || unit.Header.Type == internalobu.TypeTileGroup {
			last = i
		}
	}
	if last < 0 {
		t.Fatal("temporal unit has no frame or tile-group OBU")
	}
	var out []byte
	for i, unit := range units {
		if i != last {
			out = append(out, unit.Raw...)
			continue
		}
		header := unit.Header
		header.HasSizeField = true
		var headerBytes [2]byte
		n, err := internalobu.PutHeader(headerBytes[:], header)
		if err != nil {
			t.Fatalf("put OBU header: %v", err)
		}
		payload := unit.Payload[:len(unit.Payload)/2]
		out = append(out, headerBytes[:n]...)
		out = appendPublicLEB128(out, uint32(len(payload)))
		out = append(out, payload...)
	}
	return out
}

// TestDecoderRejectsTileDataOverread damages one temporal unit's tile data
// and checks the decoders fail it with ErrTileDataOverread (dav1d's
// msac.cnt <= -15 check in dav1d_decode_tile_sbrow) instead of decoding
// padding into the frame and every frame that references it.
func TestDecoderRejectsTileDataOverread(t *testing.T) {
	ivf, err := os.ReadFile(filepath.Join(featureVectorDir, "rt-cbr.ivf"))
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}
	payloads := featureVectorPayloads(t, ivf)
	const damaged = 3
	if len(payloads) <= damaged+1 {
		t.Fatalf("clip has %d temporal units", len(payloads))
	}
	payloads[damaged] = truncateLastTileOBU(t, payloads[damaged])

	newDecoders := []struct {
		name string
		open func(workers int) (frameSource, error)
	}{
		{"Decoder", func(workers int) (frameSource, error) { return NewDecoder(payloads, WithWorkers(workers)) }},
		{"LayeredDecoder", func(workers int) (frameSource, error) { return NewLayeredDecoder(payloads, WithWorkers(workers)) }},
	}
	for _, nd := range newDecoders {
		for _, workers := range []int{1, 4} {
			t.Run(fmt.Sprintf("%s/workers=%d", nd.name, workers), func(t *testing.T) {
				dec, err := nd.open(workers)
				if err != nil {
					t.Fatalf("open: %v", err)
				}
				defer dec.Close()
				// Frame threading may surface the error up to the frame delay
				// (at most the worker count) calls after the damaged unit.
				outputs := 0
				for call := 0; call <= damaged+workers; call++ {
					frames, ok, err := dec.DecodeNext()
					if err != nil {
						if !errors.Is(err, internaltile.ErrTileDataOverread) {
							t.Fatalf("DecodeNext call %d: err=%v want %v", call, err, internaltile.ErrTileDataOverread)
						}
						if outputs > damaged {
							t.Fatalf("output %d frames before the error, want at most %d", outputs, damaged)
						}
						return
					}
					if !ok {
						t.Fatalf("stream ended after %d calls without an error", call+1)
					}
					outputs += len(frames)
				}
				t.Fatalf("no error within %d DecodeNext calls (%d frames output)", damaged+workers+1, outputs)
			})
		}
	}
}

func TestPublicTileRestorationFramePlanAndBuffers(t *testing.T) {
	params := internalparser.RestorationParams{
		Type:       [3]internalparser.RestorationType{internalparser.RestorationWiener, internalparser.RestorationSGRProj, internalparser.RestorationNone},
		UnitSizeY:  128,
		UnitSizeUV: 64,
	}
	size := FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	color := internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true}

	y, err := internaltile.BuildRestorationPlaneGrid(params, size, color, 0)
	if err != nil {
		t.Fatal(err)
	}
	if y.Type != internalparser.RestorationWiener || y.UnitSize != 128 || y.HorzUnits != 2 || y.VertUnits != 2 {
		t.Fatalf("y grid=%+v", y)
	}
	u, err := internaltile.BuildRestorationPlaneGrid(params, size, color, 1)
	if err != nil {
		t.Fatal(err)
	}
	if u.Type != internalparser.RestorationSGRProj || u.UnitSize != 64 || !u.SubsamplingX || !u.SubsamplingY ||
		u.HorzUnits != 2 || u.VertUnits != 2 {
		t.Fatalf("u grid=%+v", u)
	}

	plan, err := internaltile.BuildRestorationFramePlan(params, size, color)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Active || plan.Planes != 3 || plan.UnitRecords != ([3]uint32{4, 4, 0}) || plan.UnitRecordLen() != 8 {
		t.Fatalf("plan=%+v records=%d", plan, plan.UnitRecordLen())
	}
	if plan.Boundaries[0] != (internaltile.RestorationStripeBoundaryBufferSize{Stride: 320, Rows: 10, Len: 3200}) ||
		plan.Boundaries[1] != (internaltile.RestorationStripeBoundaryBufferSize{Stride: 160, Rows: 10, Len: 1600}) ||
		plan.BoundaryBufferLen() != 4800 {
		t.Fatalf("boundaries=%+v total=%d", plan.Boundaries, plan.BoundaryBufferLen())
	}

	recordBacking := make([]internaltile.RestorationUnitRecord, plan.UnitRecordLen())
	records, err := internaltile.BindRestorationFrameRecordBuffers(plan, recordBacking)
	if err != nil {
		t.Fatal(err)
	}
	if len(records[0]) != 4 || len(records[1]) != 4 || records[2] != nil {
		t.Fatalf("record lens=%d,%d,%d", len(records[0]), len(records[1]), len(records[2]))
	}
	if err := internaltile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
		t.Fatal(err)
	}
	if records[0][0].Unit.Type != internalparser.RestorationNone || records[0][0].StripeCount == 0 {
		t.Fatalf("record[0]=%+v", records[0][0])
	}
	updated := records[0][0]
	updated.Unit = internaltile.RestorationUnit{Type: internalparser.RestorationWiener, Wiener: internalrestoration.DefaultWienerInfo()}
	records[0][updated.Index] = updated
	if records[0][0].Unit.Type != internalparser.RestorationWiener {
		t.Fatalf("stored record=%+v", records[0][0])
	}

	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	boundaries, err := internaltile.BindRestorationFrameBoundaryBuffers(plan, above, below)
	if err != nil {
		t.Fatal(err)
	}
	if boundaries[0].Stride != 320 || len(boundaries[0].Above) != 3200 ||
		boundaries[1].Stride != 160 || len(boundaries[1].Below) != 1600 {
		t.Fatalf("boundaries=%+v", boundaries)
	}
}

func TestPublicTileRestorationGeometryBoundaryAndApply(t *testing.T) {
	grid, err := internaltile.BuildRestorationPlaneGrid(
		internalparser.RestorationParams{Type: [3]internalparser.RestorationType{internalparser.RestorationWiener}, UnitSizeY: 128, UnitSizeUV: 64},
		FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8},
		internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	rect, err := grid.UnitRect(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if rect.Width() != 128 || rect.Height() != 120 {
		t.Fatalf("rect=%+v width=%d height=%d", rect, rect.Width(), rect.Height())
	}
	stripe, ok, err := grid.ProcessingStripe(rect, 0)
	if err != nil || !ok {
		t.Fatalf("stripe=%+v ok=%v err=%v", stripe, ok, err)
	}
	if stripe.ProcUnitWidth != internalrestoration.ProcUnitSize || stripe.TileStripe != 0 {
		t.Fatalf("stripe=%+v", stripe)
	}
	if scratch, err := internaltile.RestorationStripeBoundaryScratchLen(stripe, false); err != nil || scratch.Above != 0 || scratch.Below == 0 {
		t.Fatalf("boundary scratch=%+v err=%v", scratch, err)
	}

	size, err := internaltile.RestorationUnitScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantWiener, err := internalrestoration.WienerScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantSGR, err := internalrestoration.SelfguidedScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	if size != (internaltile.RestorationUnitScratchSize{Wiener: wantWiener, SGRProj: wantSGR}) {
		t.Fatalf("unit scratch=%+v want %d/%d", size, wantWiener, wantSGR)
	}

	const width, height = 5, 4
	const srcStride, dstStride = 11, 8
	const origin = 13
	src := make([]uint16, origin+(height-1)*srcStride+width)
	dst := make([]uint16, dstStride*height)
	for i := range dst {
		dst[i] = 0xffff
	}
	for row := range height {
		for col := range width {
			src[origin+row*srcStride+col] = uint16(20 + row*7 + col)
		}
	}
	result, err := internaltile.ApplyRestorationUnit(src, srcStride, origin, dst, dstStride, width, height, internaltile.RestorationUnit{Type: internalparser.RestorationNone}, 8, internaltile.RestorationUnitScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (internaltile.RestorationUnitApplyResult{Type: internalparser.RestorationNone}) {
		t.Fatalf("result=%+v", result)
	}
	for row := range height {
		for col := range width {
			if got, want := dst[row*dstStride+col], src[origin+row*srcStride+col]; got != want {
				t.Fatalf("dst[%d,%d]=%d want %d", row, col, got, want)
			}
		}
	}

	bordered, stride, borderOrigin := publicTileRestorationBordered(3, 2, 2, 1)
	bordered[borderOrigin+0*stride+0] = 10
	bordered[borderOrigin+0*stride+1] = 20
	bordered[borderOrigin+0*stride+2] = 30
	bordered[borderOrigin+1*stride+0] = 40
	bordered[borderOrigin+1*stride+1] = 50
	bordered[borderOrigin+1*stride+2] = 60
	if err := internaltile.ExtendRestorationFrame(bordered, stride, borderOrigin, 3, 2, 2, 1); err != nil {
		t.Fatal(err)
	}
	if bordered[borderOrigin-2] != 10 || bordered[borderOrigin+3] != 30 ||
		bordered[borderOrigin-stride] != 10 || bordered[borderOrigin+stride+4] != 60 {
		t.Fatalf("extended frame border not replicated")
	}
}

func TestPublicTileRestorationRejectsInvalid(t *testing.T) {
	params := internalparser.RestorationParams{Type: [3]internalparser.RestorationType{internalparser.RestorationWiener}, UnitSizeY: 128, UnitSizeUV: 64}
	if _, err := internaltile.BuildRestorationPlaneGrid(params, FrameSize{}, internalparser.ColorConfig{}, 0); !errors.Is(err, internaltile.ErrInvalidPlan) {
		t.Fatalf("BuildTileRestorationPlaneGrid err=%v want %v", err, internaltile.ErrInvalidPlan)
	}
	plan, err := internaltile.BuildRestorationFramePlan(params, FrameSize{UpscaledWidth: 128, Height: 64, SuperResDenominator: 8}, internalparser.ColorConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := internaltile.BindRestorationFrameRecordBuffers(plan, make([]internaltile.RestorationUnitRecord, plan.UnitRecordLen()-1)); !errors.Is(err, internaltile.ErrJobBufferTooSmall) {
		t.Fatalf("BindTileRestorationFrameRecordBuffers err=%v want %v", err, internaltile.ErrJobBufferTooSmall)
	}
	if _, err := internaltile.RestorationUnitScratchLen(0, 8); !errors.Is(err, internalrestoration.ErrInvalidRestoration) {
		t.Fatalf("TileRestorationUnitScratchLen err=%v want %v", err, internalrestoration.ErrInvalidRestoration)
	}
	if _, err := internaltile.ApplyRestorationUnit(nil, 0, 0, nil, 0, 1, 1, internaltile.RestorationUnit{Type: internalparser.RestorationType(99)}, 8, internaltile.RestorationUnitScratch{}); !errors.Is(err, internaltile.ErrInvalidPlan) {
		t.Fatalf("ApplyTileRestorationUnit err=%v want %v", err, internaltile.ErrInvalidPlan)
	}
}

func TestPublicTileRestorationAllocs(t *testing.T) {
	params := internalparser.RestorationParams{Type: [3]internalparser.RestorationType{internalparser.RestorationWiener, internalparser.RestorationSGRProj, internalparser.RestorationNone}, UnitSizeY: 128, UnitSizeUV: 64}
	size := FrameSize{UpscaledWidth: 300, Height: 260, SuperResDenominator: 8}
	color := internalparser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	plan, err := internaltile.BuildRestorationFramePlan(params, size, color)
	if err != nil {
		t.Fatal(err)
	}
	recordsBacking := make([]internaltile.RestorationUnitRecord, plan.UnitRecordLen())
	above := make([]uint16, plan.BoundaryBufferLen())
	below := make([]uint16, plan.BoundaryBufferLen())
	src := make([]uint16, 13+3*11+5)
	dst := make([]uint16, 8*4)
	allocs := testing.AllocsPerRun(1000, func() {
		records, err := internaltile.BindRestorationFrameRecordBuffers(plan, recordsBacking)
		if err != nil {
			t.Fatalf("BindTileRestorationFrameRecordBuffers err=%v", err)
		}
		if err := internaltile.ResetRestorationPlaneRecords(plan.Grids[0], records[0]); err != nil {
			t.Fatalf("ResetTileRestorationPlaneRecords err=%v", err)
		}
		if _, err := internaltile.BindRestorationFrameBoundaryBuffers(plan, above, below); err != nil {
			t.Fatalf("BindTileRestorationFrameBoundaryBuffers err=%v", err)
		}
		if _, err := internaltile.RestorationUnitScratchLen(64, 32); err != nil {
			t.Fatalf("TileRestorationUnitScratchLen err=%v", err)
		}
		if _, err := internaltile.ApplyRestorationUnit(src, 11, 13, dst, 8, 5, 4, internaltile.RestorationUnit{Type: internalparser.RestorationNone}, 8, internaltile.RestorationUnitScratch{}); err != nil {
			t.Fatalf("ApplyTileRestorationUnit err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicTileRestorationBordered(width int, height int, borderHorz int, borderVert int) ([]uint16, int, int) {
	stride := width + 2*borderHorz
	origin := borderVert*stride + borderHorz
	return make([]uint16, stride*(height+2*borderVert)), stride, origin
}

func TestPublicDSPPlaneBlockOps(t *testing.T) {
	src := publicPredictionPlane(6, 5, 2, 16)
	dst := publicPredictionPlane(7, 4, 2, 16)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setPublicFrameSample(src, 2, x, y, uint16(1000+y*10+x))
		}
	}
	if err := internaldsp.CopyPlaneBlock(dst, src, 2, 2, 1, 1, 2, 3, 2); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getPublicFrameSample(dst, 2, 2+x, 1+y)
			want := uint16(1000 + (2+y)*10 + 1 + x)
			if got != want {
				t.Fatalf("copy sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}

	residualPlane := publicPredictionPlane(4, 2, 1, 6)
	for y := range 2 {
		for x := range 4 {
			setPublicFrameSample(residualPlane, 1, x, y, 100)
		}
	}
	residual := []int16{
		-200, -1, 0, 200,
		20, -20, 155, 156,
	}
	internaldsp.AddResidualPlaneBlockTrusted(residualPlane.Pix, residualPlane.Stride, 1, 0xff, 4, 2, residual, 4)
	want := []uint16{0, 99, 100, 255, 120, 80, 255, 255}
	for y := range 2 {
		for x := range 4 {
			if got := getPublicFrameSample(residualPlane, 1, x, y); got != want[y*4+x] {
				t.Fatalf("residual sample(%d,%d)=%d want %d", x, y, got, want[y*4+x])
			}
		}
	}
}

func TestPublicDSPRejectsInvalid(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := internaldsp.CopyPlaneBlock(plane, plane, 1, 3, 0, 0, 0, 2, 1); !errors.Is(err, internaldsp.ErrInvalidBlock) {
		t.Fatalf("CopyPlaneBlock err=%v want %v", err, internaldsp.ErrInvalidBlock)
	}
}

func TestPublicDSPAllocs(t *testing.T) {
	src := publicPredictionPlane(16, 16, 1, 32)
	dst := publicPredictionPlane(16, 16, 1, 32)
	residual := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := internaldsp.CopyPlaneBlock(dst, src, 1, 0, 0, 0, 0, 16, 16); err != nil {
			t.Fatalf("CopyPlaneBlock err=%v", err)
		}
		internaldsp.AddResidualPlaneBlockTrusted(dst.Pix, dst.Stride, 1, 0xff, 16, 16, residual, 16)
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func TestPublicPredictInterPlaneBlock8Bit(t *testing.T) {
	src := publicPredictionPlane(7, 6, 1, 9)
	dst := publicPredictionPlane(7, 6, 1, 9)
	for i := range dst.Pix {
		dst.Pix[i] = 0xee
	}
	fillPublicMotionPlane(src, 1)
	if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 2, 1, 1, 2, 3, 2, 0, 0, internalmotion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < dst.Height; y++ {
		for x := 0; x < dst.Width; x++ {
			got := getPublicFrameSample(dst, 1, x, y)
			want := uint16(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = uint16(10*(y+1) + (x - 1))
			}
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicPredictInterPlaneBlockHighBitDepth(t *testing.T) {
	src := publicPredictionPlane(6, 5, 2, 7)
	dst := publicPredictionPlane(6, 5, 2, 7)
	fillPublicMotionPlane(src, 2)
	if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 2, 12, 1, 2, 2, 1, 3, 2, 0, 0, internalmotion.RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getPublicFrameSample(dst, 2, 1+x, 2+y)
			want := uint16(1000 + (1+y)*10 + 2 + x)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicPredictInterPlaneBlockFractionalBilinear(t *testing.T) {
	src := publicPredictionPlane(16, 16, 1, 16)
	dst := publicPredictionPlane(4, 4, 1, 4)
	fillPublicMotionPlane(src, 1)
	filters := internalmotion.InterpFilters{X: internalmotion.InterpBilinear, Y: internalmotion.InterpBilinear}
	if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, 4, 4, 3, 2, 8, 0, filters); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getPublicFrameSample(dst, 1, x, y)
			want := uint16(10*(4+y) + 5 + x)
			if got != want {
				t.Fatalf("fractional sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicMotionRejectsInvalid(t *testing.T) {
	if _, _, err := (internalmotion.Vector{Col: 4}).FullpelOffset(); !errors.Is(err, internalmotion.ErrInvalidMotion) {
		t.Fatalf("fractional offset err=%v want %v", err, internalmotion.ErrInvalidMotion)
	}
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(plane, plane, 1, 8, 0, 0, 0, 0, 1, 1, 16, 0, internalmotion.InterpFilters{X: internalmotion.InterpBilinear, Y: internalmotion.InterpBilinear}); !errors.Is(err, internalmotion.ErrInvalidMotion) {
		t.Fatalf("invalid subpel err=%v want %v", err, internalmotion.ErrInvalidMotion)
	}
}

func TestPublicMotionAllocs(t *testing.T) {
	src := publicPredictionPlane(16, 16, 1, 16)
	dst := publicPredictionPlane(16, 16, 1, 16)
	fillPublicMotionPlane(src, 1)
	filters := internalmotion.InterpFilters{X: internalmotion.InterpBilinear, Y: internalmotion.InterpBilinear}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, 1, 1, 8, 8, 0, 0, internalmotion.RegularFilters); err != nil {
			t.Fatalf("fullpel err=%v", err)
		}
		if err := internalmotion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, 4, 4, 8, 8, 8, 0, filters); err != nil {
			t.Fatalf("fractional err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func fillPublicMotionPlane(plane FramePlane, bytesPerSample int) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, bytesPerSample, x, y, uint16(1000*(bytesPerSample-1)+10*y+x))
		}
	}
}

func TestPublicPredictIntraPlaneBlock(t *testing.T) {
	plane := publicPredictionPlane(6, 5, 1, 8)
	for i := range plane.Pix {
		plane.Pix[i] = 0xee
	}
	edges := internalprediction.IntraEdges{
		Above:          []uint16{10, 20, 30},
		Left:           []uint16{50, 70},
		AboveAvailable: true,
		LeftAvailable:  true,
	}
	if err := internalprediction.PredictIntraPlaneBlock(plane, 1, 8, 2, 1, 3, 2, internalprediction.IntraModeDC, edges); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = 36
			}
			if got := getPublicFrameSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicPredictIntraPlaneBlockHighBitDepth(t *testing.T) {
	plane := publicPredictionPlane(4, 3, 2, 4)
	edges := internalprediction.IntraEdges{
		Above:          []uint16{100, 200, 300, 400},
		AboveAvailable: true,
	}
	if err := internalprediction.PredictIntraPlaneBlock(plane, 2, 12, 0, 0, 4, 3, internalprediction.IntraModeVertical, edges); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 4 {
			want := uint16((x + 1) * 100)
			if got := getPublicFrameSample(plane, 2, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicPredictDirectionalIntraPlaneBlock(t *testing.T) {
	plane := publicPredictionPlane(4, 3, 1, 4)
	above := []uint16{7, 11, 13, 17}
	if err := internalprediction.PredictDirectionalIntraPlaneBlock(plane, 1, 8, 0, 0, 4, 3, 90, internalprediction.DirectionalEdges{
		Above:       above,
		AboveOrigin: 0,
	}); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 4 {
			if got := getPublicFrameSample(plane, 1, x, y); got != above[x] {
				t.Fatalf("vertical sample(%d,%d)=%d want %d", x, y, got, above[x])
			}
		}
	}
	if dx, dy := internalprediction.DirectionalDX(45), internalprediction.DirectionalDY(225); dx <= 1 || dy <= 1 {
		t.Fatalf("directional derivatives dx=%d dy=%d", dx, dy)
	}
}

func TestPublicPredictFilterIntraAndEdges(t *testing.T) {
	edge := []uint16{10, 20, 30, 40}
	scratch := make([]uint16, len(edge))
	if err := internalprediction.FilterIntraEdgeBounded(edge, scratch, 1, 8, false); err != nil {
		t.Fatal(err)
	}
	wantEdge := []uint16{10, 20, 30, 38}
	for i := range wantEdge {
		if edge[i] != wantEdge[i] {
			t.Fatalf("edge[%d]=%d want %d", i, edge[i], wantEdge[i])
		}
	}

	above := []uint16{77, 10, 20, 30, 40}
	left := []uint16{99, 50, 60, 70, 80}
	if err := internalprediction.FilterIntraEdgeCorner(above, 1, left, 1, 8); err != nil {
		t.Fatal(err)
	}
	if above[0] != left[0] || above[0] != 48 {
		t.Fatalf("corner above=%d left=%d want 48", above[0], left[0])
	}

	plane := publicPredictionPlane(4, 4, 1, 4)
	edges := internalprediction.IntraEdges{
		Above:              []uint16{10, 20, 30, 40},
		Left:               []uint16{50, 60, 70, 80},
		AboveLeft:          31,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	if err := internalprediction.PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, 4, 4, 4, 4, internalprediction.FilterIntraModeDC, edges, false); err != nil {
		t.Fatal(err)
	}
	if got := getPublicFrameSample(plane, 1, 0, 0); got == 0 {
		t.Fatalf("filter intra left top sample was not written")
	}
	if strength := internalprediction.IntraEdgeFilterStrength(8, 8, 40, false); strength != 1 {
		t.Fatalf("strength=%d want 1", strength)
	}
	if !internalprediction.UseIntraEdgeUpsample(8, 8, 20, false) {
		t.Fatalf("expected intra edge upsample for small directional block")
	}
}

func TestPublicCFLPredictionHelpers(t *testing.T) {
	luma := []uint8{
		10, 20, 30, 40,
		50, 60, 70, 80,
		90, 100, 110, 120,
		130, 140, 150, 160,
	}
	recon := make([]uint16, internalprediction.CFLBufSquare)
	if err := internalprediction.SubsampleLuma8ToQ3(recon, luma, 4, 4, 4, true, true); err != nil {
		t.Fatal(err)
	}
	if recon[0] != 280 || recon[1] != 440 || recon[internalprediction.CFLBufLine] != 920 || recon[internalprediction.CFLBufLine+1] != 1080 {
		t.Fatalf("unexpected cfl recon values %d %d %d %d", recon[0], recon[1], recon[internalprediction.CFLBufLine], recon[internalprediction.CFLBufLine+1])
	}
	ac := make([]int16, internalprediction.CFLBufSquare)
	if err := internalprediction.SubtractCFLAverage(recon, ac, 4, 4); err != nil {
		t.Fatal(err)
	}
	alpha, err := internalprediction.CFLAlphaQ3(0x21, 6, internalprediction.CFLPredU)
	if err != nil {
		t.Fatal(err)
	}
	if alpha != 3 {
		t.Fatalf("alpha=%d want 3", alpha)
	}

	plane := publicPredictionPlane(4, 4, 1, 4)
	fillPublicReconstructPlane(plane, 1, 100)
	if err := internalprediction.PredictCFLPlaneBlock(plane, 1, 8, 0, 0, 4, 4, ac, alpha); err != nil {
		t.Fatal(err)
	}
	if got := getPublicFrameSample(plane, 1, 0, 0); got == 100 {
		t.Fatalf("cfl prediction did not adjust chroma sample")
	}
}

func TestPublicPredictionRejectsInvalid(t *testing.T) {
	plane := publicPredictionPlane(4, 4, 1, 4)
	if err := internalprediction.PredictIntraPlaneBlock(plane, 1, 10, 0, 0, 1, 1, internalprediction.IntraModeDC, internalprediction.IntraEdges{}); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("bitdepth mismatch err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
	if err := internalprediction.PredictDirectionalIntraPlaneBlock(plane, 1, 8, 0, 0, 1, 1, 0, internalprediction.DirectionalEdges{}); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("directional angle err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
	if err := internalprediction.FilterIntraEdgeBounded([]uint16{1, 2}, nil, 1, 8, false); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("filter edge scratch err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
	if _, err := internalprediction.CFLAlphaQ3(0, 99, internalprediction.CFLPredU); !errors.Is(err, internalprediction.ErrInvalidPrediction) {
		t.Fatalf("cfl alpha err=%v want %v", err, internalprediction.ErrInvalidPrediction)
	}
}

func TestPublicPredictionAllocs(t *testing.T) {
	plane := publicPredictionPlane(16, 16, 1, 16)
	edges := internalprediction.IntraEdges{
		Above:              make([]uint16, 16),
		Left:               make([]uint16, 16),
		AboveLeft:          91,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	for i := range 16 {
		edges.Above[i] = 90
		edges.Left[i] = 92
	}
	edge := make([]uint16, 16)
	filterScratch := make([]uint16, 16)
	recon := make([]uint16, internalprediction.CFLBufSquare)
	ac := make([]int16, internalprediction.CFLBufSquare)

	allocs := testing.AllocsPerRun(1000, func() {
		if err := internalprediction.PredictIntraPlaneBlock(plane, 1, 8, 0, 0, 16, 16, internalprediction.IntraModeDC, edges); err != nil {
			t.Fatalf("PredictIntraPlaneBlock err=%v", err)
		}
		if err := internalprediction.PredictFilterIntraPlaneBlockWithExtentBounded(plane, 1, 8, 0, 0, 16, 16, 16, 16, internalprediction.FilterIntraModeDC, edges, false); err != nil {
			t.Fatalf("PredictFilterIntraPlaneBlock err=%v", err)
		}
		copy(edge, edges.Above)
		if err := internalprediction.FilterIntraEdgeBounded(edge, filterScratch, 1, 8, false); err != nil {
			t.Fatalf("FilterIntraEdge err=%v", err)
		}
		if err := internalprediction.SubtractCFLAverage(recon, ac, 4, 4); err != nil {
			t.Fatalf("SubtractCFLAverage err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicPredictionPlane(width int, height int, bytesPerSample int, strideSamples int) FramePlane {
	return FramePlane{
		Pix:    make([]byte, strideSamples*height*bytesPerSample),
		Stride: strideSamples * bytesPerSample,
		Width:  width,
		Height: height,
	}
}

func TestPublicTransformScanAndScratchHelpers(t *testing.T) {
	size, err := internaltransform.SizeFromDimensions(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !size.Valid() {
		t.Fatalf("size %+v is not valid", size)
	}
	class, err := internaltransform.TypeDCTDCT.Class()
	if err != nil {
		t.Fatal(err)
	}
	if class != internaltransform.Class2D {
		t.Fatalf("class=%v want %v", class, internaltransform.Class2D)
	}
	mode, err := internaltransform.DefaultScanMode(size, class)
	if err != nil {
		t.Fatal(err)
	}
	if mode != internaltransform.ScanModeZigZag {
		t.Fatalf("mode=%v want %v", mode, internaltransform.ScanModeZigZag)
	}

	scanSize, err := internaltransform.ScanSize(internaltransform.Size{Width: 64, Height: 16})
	if err != nil {
		t.Fatal(err)
	}
	if scanSize != (internaltransform.Size{Width: 32, Height: 16}) {
		t.Fatalf("scanSize=%+v want 32x16", scanSize)
	}

	scan := make([]int16, size.Width*size.Height)
	inverse := make([]int16, len(scan))
	if err := internaltransform.FillDefaultScan(scan, inverse, size, class); err != nil {
		t.Fatal(err)
	}
	for i, coeff := range scan {
		if coeff < 0 || int(coeff) >= len(scan) {
			t.Fatalf("scan[%d]=%d outside range", i, coeff)
		}
		if inverse[coeff] != int16(i) {
			t.Fatalf("inverse[%d]=%d want %d", coeff, inverse[coeff], i)
		}
	}
	scratch, err := internaltransform.ScratchLenForType(internaltransform.TypeDCTDCT, size)
	if err != nil {
		t.Fatal(err)
	}
	if scratch != 16 {
		t.Fatalf("scratch=%d want 16", scratch)
	}
}

func TestPublicPlaneQuantizer(t *testing.T) {
	params := internalparser.QuantizationParams{
		BaseQIdx: 96,
		YDCDelta: -2,
		UDCDelta: 3,
		UACDelta: -1,
		VDCDelta: 4,
		VACDelta: 5,
	}
	y, err := internalquantize.PlaneQuantizer(params, params.BaseQIdx, 8, internalquantize.PlaneY)
	if err != nil {
		t.Fatal(err)
	}
	u, err := internalquantize.PlaneQuantizer(params, params.BaseQIdx, 8, internalquantize.PlaneU)
	if err != nil {
		t.Fatal(err)
	}
	v, err := internalquantize.PlaneQuantizer(params, params.BaseQIdx, 8, internalquantize.PlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if y != (internalquantize.Quantizer{DC: 85, AC: 104}) ||
		u != (internalquantize.Quantizer{DC: 92, AC: 102}) ||
		v != (internalquantize.Quantizer{DC: 93, AC: 114}) {
		t.Fatalf("quantizers y=%+v u=%+v v=%+v", y, u, v)
	}

	scale, err := internalquantize.TransformScale(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	if scale != 2 {
		t.Fatalf("scale=%d want 2", scale)
	}
	if got := internalquantize.ClampQIndex(2, -9); got != 0 {
		t.Fatalf("clamped low=%d want 0", got)
	}
	if got := internalquantize.ClampQIndex(internalquantize.MaxQIndex, 9); got != internalquantize.MaxQIndex {
		t.Fatalf("clamped high=%d want %d", got, internalquantize.MaxQIndex)
	}

}

func TestPublicReconstructBlockMaxScratchLen(t *testing.T) {
	int32Len, int16Len, err := reconstructBlockMaxScratchLen(false)
	if err != nil {
		t.Fatal(err)
	}
	want32, want16, err := internalreconstruct.ScratchLen(internalreconstruct.Block{
		Size:      internaltransform.Size{Width: 64, Height: 64},
		Transform: internaltransform.TypeDCTDCT,
	})
	if err != nil {
		t.Fatal(err)
	}
	if int32Len != want32 || int16Len != want16 {
		t.Fatalf("max scratch=%d/%d want %d/%d", int32Len, int16Len, want32, want16)
	}

	lossless32, lossless16, err := reconstructBlockMaxScratchLen(true)
	if err != nil {
		t.Fatal(err)
	}
	wantLossless32, wantLossless16, err := internalreconstruct.ScratchLen(internalreconstruct.Block{
		Size:      internaltransform.Size{Width: 4, Height: 4},
		Transform: internaltransform.TypeDCTDCT,
		Lossless:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if lossless32 != wantLossless32 || lossless16 != wantLossless16 {
		t.Fatalf("lossless max scratch=%d/%d want %d/%d", lossless32, lossless16, wantLossless32, wantLossless16)
	}
}

func fillPublicReconstructPlane(plane FramePlane, bytesPerSample int, sample uint16) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, bytesPerSample, x, y, sample)
		}
	}
}
