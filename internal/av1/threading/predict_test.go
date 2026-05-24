package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
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
				ColorConfig: parser.ColorConfig{BitDepth: output.Format.BitDepth},
			}),
			FrameSize: parser.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height)},
		},
		Jobs: []tile.Job{{SBCols: 1, SBRows: 1}},
	}
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

func setFrameWorkTestSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
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
