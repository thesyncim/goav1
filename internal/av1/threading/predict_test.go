package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
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

			first := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv0, tt.filters)
			second := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, tt.mv1, tt.filters)
			fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
			if err != nil {
				t.Fatal(err)
			}
			assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, 16, 16, 16, 16, fwdOffset, bckOffset)
		})
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

	firstU := testFrameWorkMotionPredictionPlaneSubsampled(t, last.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondU := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.U, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	firstV := testFrameWorkMotionPredictionPlaneSubsampled(t, last.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv0, true, true, filters)
	secondV := testFrameWorkMotionPredictionPlaneSubsampled(t, bwd.V, output.Layout.BytesPerSample, output.Format.BitDepth, 8, 8, 8, 8, mv1, true, true, filters)
	fwdOffset, bckOffset, err := ctx.frameWorkCompoundOffsets(visit.Prediction.InterMotion.References, visit.Prediction.CompoundBlend)
	if err != nil {
		t.Fatal(err)
	}
	assertFrameWorkCompoundBlendEqual(t, output.U, firstU, secondU, output.Layout.BytesPerSample, 8, 8, 8, 8, fwdOffset, bckOffset)
	assertFrameWorkCompoundBlendEqual(t, output.V, firstV, secondV, output.Layout.BytesPerSample, 8, 8, 8, 8, fwdOffset, bckOffset)
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
	first := testFrameWorkMotionPredictionPlane(t, last.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[0], motion.RegularFilters)
	second := testFrameWorkMotionPredictionPlane(t, bwd.Y, output.Layout.BytesPerSample, output.Format.BitDepth, 16, 16, 16, 16, visit.Prediction.InterMotion.MV[1], motion.RegularFilters)
	assertFrameWorkCompoundBlendEqual(t, output.Y, first, second, output.Layout.BytesPerSample, 16, 16, 16, 16, 8, 8)
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
		{name: "outside job", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := valid
			visit.Block.MICol = 16
			visit.Block.MIColEnd = 20
			return visit
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.ctx.PredictBlockLumaInter(0, tt.visit); !errors.Is(err, ErrInvalidBatch) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
			}
		})
	}
	if err := ctx.PredictBlockLumaInterWithFilters(0, valid, motion.InterpFilters{X: motion.InterpFilter(99)}); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("bad filters err=%v want %v", err, ErrInvalidBatch)
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
		{name: "directional missing top", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := testIntraPredictionVisit(tile.IntraModeD45)
			visit.Block.HaveTop = false
			return visit
		}(), scratch: &scratch},
		{name: "directional missing left", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := testIntraPredictionVisit(tile.IntraModeD203)
			visit.Block.HaveLeft = false
			return visit
		}(), scratch: &scratch},
		{name: "vertical missing top", ctx: ctx, visit: func() tile.BlockLoopVisit {
			visit := testIntraPredictionVisit(tile.IntraModeVertical)
			visit.Block.HaveTop = false
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

func assertFrameWorkCompoundBlendEqual(t *testing.T, got frame.Plane, first frame.Plane, second frame.Plane, bytesPerSample int, x int, y int, width int, height int, fwdOffset int, bckOffset int) {
	t.Helper()
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			a := frameWorkTestSample(first, bytesPerSample, col, row)
			b := frameWorkTestSample(second, bytesPerSample, col, row)
			want := uint16((uint32(a)*uint32(fwdOffset) + uint32(b)*uint32(bckOffset) + 8) >> 4)
			g := frameWorkTestSample(got, bytesPerSample, x+col, y+row)
			if g != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, want)
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
