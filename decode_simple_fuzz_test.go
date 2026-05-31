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

	switchOnly := appendPublicLowOverheadOBU(nil, av1.OBUSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	switchFrame := append([]byte{}, publicSimpleDecoderSwitchFrameHeaderPayload()...)
	switchFrame = append(switchFrame, 0xbb)
	switchOnly = appendPublicLowOverheadOBU(switchOnly, av1.OBUFrame, switchFrame)
	f.Add(appendPublicIVF(nil, 16, 9, 30, 1, []publicIVFFrame{
		{payload: switchOnly},
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
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendPublicLowOverheadOBU(nil, av1.OBUTileList, []byte{0xff, 0xff, 0xff, 0xff})},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendPublicLowOverheadOBU(nil, av1.OBUTileList, []byte{0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0x00, 0x04, 0xaa})},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendPublicLowOverheadOBU(nil, av1.OBUTileList, append(append([]byte{}, validTilePayload...), 0xff))},
	}))
	truncatedIVF := appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: []byte{0xaa, 0xbb}},
	})
	f.Add(truncatedIVF[:len(truncatedIVF)-1])
	shortFrameHeaderIVF := appendPublicIVF(nil, 16, 16, 30, 1, nil)
	shortFrameHeaderIVF = append(shortFrameHeaderIVF, 0x01)
	f.Add(shortFrameHeaderIVF)
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: []byte{0x10}},
	}))
	shortOBU := []byte{0x12}
	shortOBU = appendPublicLEB128(shortOBU, 4)
	shortOBU = append(shortOBU, 0xaa)
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: shortOBU},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: []byte{0x80}},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: []byte{0x13, 0x00}},
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
