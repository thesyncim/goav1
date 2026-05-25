package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicPredictDecoderFrameWorkBlockIntra(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	publicSeedDecoderPredictionIntraEdges(output, 10, 50)

	var scratch av1.DecoderFrameWorkPredictionScratch
	if err := av1.PredictDecoderFrameWorkBlock(batch, 0, publicDecoderPredictionIntraVisit(av1.TileIntraModeDC), &scratch); err != nil {
		t.Fatal(err)
	}
	if got := getPublicFrameSample(output.Y, output.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("predicted sample=%d want 30", got)
	}
}

func TestPublicPredictDecoderFrameWorkBlockLumaInter(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := publicDecoderPostFilterFrame(t, output.Format)
	publicFillDecoderPostFilterPlane(reference.Y)
	batch := publicDecoderPredictionInterBatch(output, reference)

	if err := av1.PredictDecoderFrameWorkBlockLumaInter(batch, 0, publicDecoderPredictionInterVisit(av1.MotionVector{})); err != nil {
		t.Fatal(err)
	}
	got := getPublicFrameSample(output.Y, output.Layout.BytesPerSample, 16, 16)
	want := getPublicFrameSample(reference.Y, reference.Layout.BytesPerSample, 16, 16)
	if got != want {
		t.Fatalf("inter predicted sample=%d want %d", got, want)
	}
}

func TestPublicPredictDecoderFrameWorkBlockRejectsInvalidInputs(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	visit := publicDecoderPredictionIntraVisit(av1.TileIntraModeDC)

	if err := av1.PredictDecoderFrameWorkBlock(batch, 0, av1.TileBlockLoopVisit{}, &av1.DecoderFrameWorkPredictionScratch{}); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("invalid visit err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.PredictDecoderFrameWorkBlockLuma(batch, 0, visit, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil luma scratch err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.PredictDecoderFrameWorkBlockChromaCFL(batch, 0, visit, &av1.DecoderFrameWorkCFLPredictionScratch{}); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("non-cfl chroma err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicPredictDecoderFrameWorkBlockAllocs(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	visit := publicDecoderPredictionIntraVisit(av1.TileIntraModeDC)
	var scratch av1.DecoderFrameWorkIntraPredictionScratch

	allocs := testing.AllocsPerRun(1000, func() {
		publicSeedDecoderPredictionIntraEdges(output, 10, 50)
		if err := av1.PredictDecoderFrameWorkBlockLumaIntra(batch, 0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicDecoderPredictionBatch(output *av1.Frame) av1.DecoderFrameWorkBatch {
	return av1.DecoderFrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				ColorConfig: av1.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: av1.FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height), SuperResDenominator: 8},
		},
		Jobs: []av1.TileJob{{SBCols: 1, SBRows: 1}},
	}
}

func publicDecoderPredictionInterBatch(output *av1.Frame, reference *av1.Frame) av1.DecoderFrameWorkBatch {
	batch := publicDecoderPredictionBatch(output)
	batch.References = []*av1.Frame{reference}
	batch.TileInfo = av1.TileInfo{InterpolationFilter: av1.InterpolationEightTap}
	return batch
}

func publicDecoderPredictionIntraVisit(mode av1.TileIntraMode) av1.TileBlockLoopVisit {
	return av1.TileBlockLoopVisit{
		Block: publicDecoderPredictionBlockVisit(),
		Prediction: av1.TileBlockPredictionModeResult{
			Valid:    true,
			Intra:    true,
			LumaMode: mode,
		},
	}
}

func publicDecoderPredictionInterVisit(mv av1.MotionVector) av1.TileBlockLoopVisit {
	refs := av1.TileInterReferencesResult{Ref: [2]av1.TileReferenceFrame{av1.TileReferenceFrameLast, av1.TileReferenceFrameNone}}
	return av1.TileBlockLoopVisit{
		Block: publicDecoderPredictionBlockVisit(),
		Prediction: av1.TileBlockPredictionModeResult{
			Valid:                true,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMotion: av1.TileInterMotionResult{
				References: refs,
				MV:         [2]av1.MotionVector{mv},
			},
			InterMotionValid: true,
		},
	}
}

func publicDecoderPredictionBlockVisit() av1.TileBlockVisit {
	return av1.TileBlockVisit{
		MICol:     4,
		MIRow:     4,
		MIColEnd:  8,
		MIRowEnd:  8,
		X4:        4,
		Y4:        4,
		Size:      av1.TileBlockSize16x16,
		VisibleW4: 4,
		VisibleH4: 4,
		HaveTop:   true,
		HaveLeft:  true,
	}
}

func publicSeedDecoderPredictionIntraEdges(output *av1.Frame, above uint16, left uint16) {
	for x := 16; x < 32; x++ {
		setPublicFrameSample(output.Y, output.Layout.BytesPerSample, x, 15, above)
	}
	for y := 16; y < 32; y++ {
		setPublicFrameSample(output.Y, output.Layout.BytesPerSample, 15, y, left)
	}
}
