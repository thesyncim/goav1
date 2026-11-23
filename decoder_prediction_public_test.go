package goav1

import (
	"errors"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalmotion "github.com/thesyncim/goav1/internal/av1/motion"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicPredictDecoderFrameWorkBlockIntra(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	publicSeedDecoderPredictionIntraEdges(output, 10, 50)

	var scratch internalthreading.FrameWorkPredictionScratch
	if err := predictDecoderFrameWorkBlock(batch, 0, publicDecoderPredictionIntraVisit(internaltile.IntraModeDC), &scratch); err != nil {
		t.Fatal(err)
	}
	if got := getPublicFrameSample(output.Y, output.Layout.BytesPerSample, 16, 16); got != 30 {
		t.Fatalf("predicted sample=%d want 30", got)
	}
}

func TestPublicPredictDecoderFrameWorkBlockLumaInter(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	reference := publicDecoderPostFilterFrame(t, output.Format)
	publicFillDecoderPostFilterPlane(reference.Y)
	batch := publicDecoderPredictionInterBatch(output, reference)

	if err := predictDecoderFrameWorkBlockLumaInter(batch, 0, publicDecoderPredictionInterVisit(internalmotion.Vector{})); err != nil {
		t.Fatal(err)
	}
	got := getPublicFrameSample(output.Y, output.Layout.BytesPerSample, 16, 16)
	want := getPublicFrameSample(reference.Y, reference.Layout.BytesPerSample, 16, 16)
	if got != want {
		t.Fatalf("inter predicted sample=%d want %d", got, want)
	}
}

func TestPublicPredictDecoderFrameWorkBlockRejectsInvalidInputs(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	visit := publicDecoderPredictionIntraVisit(internaltile.IntraModeDC)

	if err := predictDecoderFrameWorkBlock(batch, 0, internaltile.BlockLoopVisit{}, &internalthreading.FrameWorkPredictionScratch{}); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("invalid visit err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if err := predictDecoderFrameWorkBlockLuma(batch, 0, visit, nil); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("nil luma scratch err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
	if err := predictDecoderFrameWorkBlockChromaCFL(batch, 0, visit, &internalthreading.FrameWorkCFLPredictionScratch{}); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("non-cfl chroma err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicPredictDecoderFrameWorkBlockAllocs(t *testing.T) {
	output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	visit := publicDecoderPredictionIntraVisit(internaltile.IntraModeDC)
	var scratch internalthreading.FrameWorkIntraPredictionScratch

	allocs := testing.AllocsPerRun(1000, func() {
		publicSeedDecoderPredictionIntraEdges(output, 10, 50)
		if err := predictDecoderFrameWorkBlockLumaIntra(batch, 0, visit, &scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicDecoderPredictionBatch(output *Frame) internaldecoder.FrameWorkBatch {
	return internaldecoder.FrameWorkBatch{
		Output: output,
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				ColorConfig: internalparser.ColorConfig{
					BitDepth:     output.Format.BitDepth,
					MonoChrome:   output.Format.MonoChrome,
					SubsamplingX: output.Format.SubsamplingX,
					SubsamplingY: output.Format.SubsamplingY,
				},
			}),
			FrameSize: FrameSize{CodedWidth: uint32(output.Format.Width), Height: uint32(output.Format.Height), SuperResDenominator: 8},
		},
		Jobs: []internaltile.Job{{SBCols: 1, SBRows: 1}},
	}
}

func publicDecoderPredictionInterBatch(output *Frame, reference *Frame) internaldecoder.FrameWorkBatch {
	batch := publicDecoderPredictionBatch(output)
	batch.References = []*Frame{reference}
	batch.TileInfo = internalparser.TileInfo{InterpolationFilter: internalparser.InterpolationEightTap}
	return batch
}

func publicDecoderPredictionIntraVisit(mode internaltile.IntraMode) internaltile.BlockLoopVisit {
	return internaltile.BlockLoopVisit{
		Block: publicDecoderPredictionBlockVisit(),
		Prediction: internaltile.BlockPredictionModeResult{
			Valid:    true,
			Intra:    true,
			LumaMode: mode,
		},
	}
}

func publicDecoderPredictionInterVisit(mv internalmotion.Vector) internaltile.BlockLoopVisit {
	refs := internaltile.InterReferencesResult{Ref0: internaltile.ReferenceFrameLast, Ref1: internaltile.ReferenceFrameNone}
	return internaltile.BlockLoopVisit{
		Block: publicDecoderPredictionBlockVisit(),
		Prediction: internaltile.BlockPredictionModeResult{
			Valid:                true,
			InterReferences:      refs,
			InterReferencesValid: true,
			InterMotion: internaltile.InterMotionResult{
				References: refs,
				MV:         [2]internalmotion.Vector{mv},
			},
			InterMotionValid: true,
		},
	}
}

func publicDecoderPredictionBlockVisit() internaltile.BlockVisit {
	return internaltile.BlockVisit{
		MICol:     4,
		MIRow:     4,
		MIColEnd:  8,
		MIRowEnd:  8,
		X4:        4,
		Y4:        4,
		Size:      internaltile.BlockSize16x16,
		VisibleW4: 4,
		VisibleH4: 4,
		HaveTop:   true,
		HaveLeft:  true,
	}
}

func publicSeedDecoderPredictionIntraEdges(output *Frame, above uint16, left uint16) {
	for x := 16; x < 32; x++ {
		setPublicFrameSample(output.Y, output.Layout.BytesPerSample, x, 15, above)
	}
	for y := 16; y < 32; y++ {
		setPublicFrameSample(output.Y, output.Layout.BytesPerSample, 15, y, left)
	}
}
