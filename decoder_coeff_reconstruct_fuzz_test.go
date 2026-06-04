package goav1_test

import (
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func FuzzPublicDecodeTileBlockCoefficients(f *testing.F) {
	f.Add([]byte{0x00}, uint8(0), false, uint8(0), uint8(0), uint8(0))
	f.Add([]byte{0x00, 0x00}, uint8(12), true, uint8(1), uint8(1), uint8(0))
	f.Add([]byte{0xff, 0x80, 0x00}, uint8(20), true, uint8(0), uint8(1), uint8(9))

	f.Fuzz(func(t *testing.T, payload []byte, rawBlock uint8, chroma bool, rawSSX uint8, rawSSY uint8, rawType uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		block := publicFuzzTileBlockSize(rawBlock)
		dims, ok := av1.TileBlockDimensionsOf(block)
		if !ok {
			t.Fatal("invalid normalized block")
		}
		typ := publicFuzzTransformType(rawType)
		var transformCDFs av1.TileTransformCDFs
		if err := av1.InitTileTransformCDFsDefault(&transformCDFs); err != nil {
			t.Fatal(err)
		}
		var coeffCDFs av1.TileCoeffCDFs
		if err := av1.InitTileCoeffCDFsDefault(&coeffCDFs, 0); err != nil {
			t.Fatal(err)
		}
		var state av1.TileDecodeState
		if err := av1.ResetTileDecodeState(&state, payload, av1.TileJob{Offset: 0, Size: uint32(len(payload))}, av1.TileDecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		var transformCtx av1.TileTransformContext
		var coeffCtx av1.TileCoeffEntropyContext
		var scratch av1.TileBlockCoeffScratch
		result, err := av1.DecodeTileBlockCoefficients(&state, av1.TileBlockCoeffCDFs{
			Transform: &transformCDFs,
			Coeff:     &coeffCDFs,
		}, &transformCtx, &coeffCtx, &scratch, av1.TileBlockCoeffRequest{
			Transform: av1.TileTransformTreeRequest{
				Size:          block,
				VisibleW4:     dims.W4,
				VisibleH4:     dims.H4,
				Color:         av1.ColorConfig{MonoChrome: !chroma, SubsamplingX: rawSSX&1 != 0, SubsamplingY: rawSSY&1 != 0},
				TransformMode: av1.TransformModeLargest,
			},
			LumaType:   typ,
			ChromaType: [2]av1.TransformType{typ, typ},
		}, func(block av1.TileBlockCoeffBlock) error {
			publicAssertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
			return nil
		})
		if err != nil {
			return
		}
		total := result.TotalStats()
		if total.TXBs != total.NonZero+total.AllZero {
			t.Fatalf("stats=%+v inconsistent counts", total)
		}
	})
}

func FuzzPublicDecodeAndReconstructDecoderFrameWorkBlockCoefficients(f *testing.F) {
	f.Add([]byte{0x00}, uint8(32))
	f.Add([]byte{0xff, 0x80}, uint8(64))
	f.Add([]byte{0xaa, 0x55, 0x11}, uint8(127))

	f.Fuzz(func(t *testing.T, payload []byte, rawQ uint8) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		output := publicDecoderBlockCoeffFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
		fillPublicReconstructPlane(output.Y, output.Layout.BytesPerSample, 128)
		batch := publicDecoderBlockCoeffSimpleBatch(output)
		qIndex := rawQ | 1
		batch.Quantization.BaseQIdx = qIndex
		req := publicDecoderBlockCoeffDecodeRequest(t, batch, false)
		req.Reconstruction.CurrentQIndex = qIndex

		var transformCDFs av1.TileTransformCDFs
		if err := av1.InitTileTransformCDFsDefault(&transformCDFs); err != nil {
			t.Fatal(err)
		}
		var coeffCDFs av1.TileCoeffCDFs
		if err := av1.InitTileCoeffCDFsDefault(&coeffCDFs, qIndex); err != nil {
			t.Fatal(err)
		}
		var state av1.TileDecodeState
		if err := av1.ResetTileDecodeState(&state, payload, av1.TileJob{Offset: 0, Size: uint32(len(payload))}, av1.TileDecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		var transformCtx av1.TileTransformContext
		var coeffCtx av1.TileCoeffEntropyContext
		var scratch av1.TileBlockCoeffScratch
		result, err := av1.DecodeAndReconstructDecoderFrameWorkBlockCoefficients(batch, 0, &state, av1.TileBlockCoeffCDFs{
			Transform: &transformCDFs,
			Coeff:     &coeffCDFs,
		}, &transformCtx, &coeffCtx, &scratch, req, func(block av1.TileBlockCoeffBlock) error {
			publicAssertTXBDecodeInvariants(t, block.Result, block.Coeffs, block.Scan)
			return nil
		})
		if err != nil {
			return
		}
		total := result.TotalStats()
		if total.TXBs != total.NonZero+total.AllZero {
			t.Fatalf("stats=%+v inconsistent counts", total)
		}
	})
}

func publicFuzzTileBlockSize(raw uint8) av1.TileBlockSize {
	sizes := [...]av1.TileBlockSize{
		av1.TileBlockSize128x128,
		av1.TileBlockSize128x64,
		av1.TileBlockSize64x128,
		av1.TileBlockSize64x64,
		av1.TileBlockSize64x32,
		av1.TileBlockSize64x16,
		av1.TileBlockSize32x64,
		av1.TileBlockSize32x32,
		av1.TileBlockSize32x16,
		av1.TileBlockSize32x8,
		av1.TileBlockSize16x64,
		av1.TileBlockSize16x32,
		av1.TileBlockSize16x16,
		av1.TileBlockSize16x8,
		av1.TileBlockSize16x4,
		av1.TileBlockSize8x32,
		av1.TileBlockSize8x16,
		av1.TileBlockSize8x8,
		av1.TileBlockSize8x4,
		av1.TileBlockSize4x16,
		av1.TileBlockSize4x8,
		av1.TileBlockSize4x4,
	}
	return sizes[int(raw)%len(sizes)]
}

func publicFuzzTransformType(raw uint8) av1.TransformType {
	types := [...]av1.TransformType{
		av1.TransformTypeDCTDCT,
		av1.TransformTypeADSTDCT,
		av1.TransformTypeDCTADST,
		av1.TransformTypeADSTADST,
		av1.TransformTypeFlipADSTDCT,
		av1.TransformTypeDCTFlipADST,
		av1.TransformTypeFlipADSTFlipADST,
		av1.TransformTypeADSTFlipADST,
		av1.TransformTypeFlipADSTADST,
		av1.TransformTypeIDTX,
		av1.TransformTypeVDCT,
		av1.TransformTypeHDCT,
		av1.TransformTypeVADST,
		av1.TransformTypeHADST,
		av1.TransformTypeVFlipADST,
		av1.TransformTypeHFlipADST,
	}
	return types[int(raw)%len(types)]
}

func publicAssertTXBDecodeInvariants(t *testing.T, result av1.TileTXBDecodeResult, coeffs []int16, scan []int16) {
	t.Helper()
	if int(result.EOB) > len(coeffs) || int(result.EOB) > len(scan) {
		t.Fatalf("invalid eob=%d coeffs=%d scan=%d", result.EOB, len(coeffs), len(scan))
	}
	if result.AllZero && result.EOB != 0 {
		t.Fatalf("all-zero txb has eob=%d", result.EOB)
	}
}
