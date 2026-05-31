package goav1_test

import (
	"os"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

const fuzzSimpleDecoderMaxIVFBytes = 8 << 10
const fuzzSimpleDecoderMaxFrames = 64

func FuzzPublicSimpleDecoderIVF(f *testing.F) {
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendPublicLowOverheadOBU(nil, av1.OBUTemporalDelimiter, nil)},
	}))

	validTilePayload := av1.AppendTileListOBU(nil, av1.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []av1.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      0,
			AnchorTileCol:      0,
			TileDataSizeMinus1: 2,
			TileData:           []byte{0xaa, 0xbb, 0xcc},
		}},
	})
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendPublicLowOverheadOBU(nil, av1.OBUTileList, validTilePayload)},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendPublicLowOverheadOBU(nil, av1.OBUTileList, []byte{0x00, 0x00, 0x00, 0x00})},
	}))

	if ivf, err := os.ReadFile(profileClipPath(profileClips[0].file)); err == nil {
		f.Add(ivf)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > fuzzSimpleDecoderMaxIVFBytes {
			data = data[:fuzzSimpleDecoderMaxIVFBytes]
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic in simple decoder IVF path (len=%d): %v", len(data), r)
			}
		}()

		dec, err := av1.NewDecoderFromIVF(data)
		if err != nil {
			return
		}
		defer dec.Close()

		for range fuzzSimpleDecoderMaxFrames {
			_, ok, err := dec.DecodeNext()
			if err != nil || !ok {
				return
			}
		}
	})
}
