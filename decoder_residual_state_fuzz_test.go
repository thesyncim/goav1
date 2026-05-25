package goav1_test

import (
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func FuzzPublicDecodeAndReconstructDecoderFrameWorkJobResiduals(f *testing.F) {
	f.Add([]byte{0x00, 0x00, 0x00, 0x00}, uint8(64), false)
	f.Add([]byte{0xff, 0x80, 0x00, 0x7f}, uint8(3), true)
	f.Add([]byte{0xaa, 0x55, 0x11, 0xee, 0x00}, uint8(127), false)

	f.Fuzz(func(t *testing.T, payload []byte, rawQ uint8, sideMaps bool) {
		if len(payload) == 0 || len(payload) > 128 {
			return
		}
		qIndex := rawQ | 1
		output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
		publicFillDecoderPostFilterPlane(output.Y)
		batch := av1.DecoderFrameWorkBatch{
			Output:  output,
			Payload: payload,
			FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
				Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
					ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
				}),
				FrameSize:    av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8},
				Quantization: av1.QuantizationParams{BaseQIdx: qIndex},
				TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
			},
			Jobs: []av1.TileJob{{SBCols: 1, SBRows: 1, Offset: 0, Size: len(payload)}},
		}
		if sideMaps {
			publicBindResidualFuzzSideMaps(t, &batch)
		}

		var state av1.TileDecodeState
		if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
			t.Fatal(err)
		}
		var storage av1.DecoderFrameWorkTileResidualCDFStorage
		if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, qIndex); err != nil {
			t.Fatal(err)
		}
		loopReq, err := av1.DecoderFrameWorkJobBlockLoopRequest(batch, 0, nil, nil, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		int32Len, int16Len, err := av1.DecoderFrameWorkResidualScratchLen(batch, qIndex, 0, av1.DecoderFrameWorkPlaneY, av1.TransformSize{Width: 64, Height: 64}, av1.TransformTypeDCTDCT)
		if err != nil {
			t.Fatal(err)
		}

		var scratch av1.DecoderFrameWorkTileResidualScratch
		req := av1.DecoderFrameWorkTileResidualRequest{
			Loop:          loopReq,
			TransformMode: batch.TransformRef.TransformMode,
			Transforms: func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
				return av1.ReadDecoderFrameWorkInterBlockTransforms(batch, &state, visit)
			},
			Int32Scratch:    make([]int32, int32Len),
			ResidualScratch: make([]int16, int16Len),
		}
		stats, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, &state, av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage), &scratch, req)
		if err != nil {
			return
		}
		if stats.TXBs != stats.NonZero+stats.AllZero {
			t.Fatalf("residual stats=%+v inconsistent txb counts", stats)
		}
		if stats.Loop.CoefficientTXBs != stats.TXBs {
			t.Fatalf("loop stats=%+v residual stats=%+v", stats.Loop, stats)
		}
	})
}

func publicBindResidualFuzzSideMaps(t *testing.T, batch *av1.DecoderFrameWorkBatch) {
	t.Helper()
	batch.CDEF = av1.CDEFParams{Bits: 0, StrengthCount: 1}
	sequence := av1.SequenceHeader{
		Use128x128Superblock: batch.Sequence.Use128x128Superblock,
		ColorConfig:          batch.Sequence.ColorConfig,
	}
	_, _, cdefLength, err := av1.DecoderFrameWorkCDEFIndexMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, batch.FrameSize, batch.CDEF, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}
	_, _, loopLength, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	loopMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, batch.FrameSize, make([]av1.DecoderFrameWorkLoopFilterBlockRecord, loopLength))
	if err != nil {
		t.Fatal(err)
	}
	batch.CDEFIndexMap = &cdefMap
	batch.LoopFilterMap = &loopMap
}
