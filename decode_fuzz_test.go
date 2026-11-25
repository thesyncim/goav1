package goav1

import (
	"encoding/binary"
	"errors"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
	"os"
	"testing"
)

// The decoder fuzz targets feed hostile payloads through the public decoders.
// Seeds are real encoder output so mutations mostly land in tile data and reach
// entropy decoding, reconstruction, and the post-filters. Each input is a list
// of payloads framed by big-endian uint16 lengths.

const (
	publicFuzzMaxPayloads   = 64
	publicFuzzFrameSizeMax  = 256 * 256
	publicFuzzSeedWidth     = 64
	publicFuzzSeedHeight    = 64
	publicFuzzRTPPayloadLen = 400
)

func splitPublicFuzzPayloads(data []byte) [][]byte {
	var payloads [][]byte
	for len(data) >= 2 && len(payloads) < publicFuzzMaxPayloads {
		n := int(binary.BigEndian.Uint16(data))
		data = data[2:]
		n = min(n, len(data))
		payloads = append(payloads, data[:n])
		data = data[n:]
	}
	return payloads
}

func joinPublicFuzzPayloads(payloads [][]byte) []byte {
	var out []byte
	for _, p := range payloads {
		out = binary.BigEndian.AppendUint16(out, uint16(len(p)))
		out = append(out, p...)
	}
	return out
}

// publicFuzzSeeds encodes a short clip with mode and returns the coded frames
// in decode order, framed both as low-overhead temporal units and as RTP
// payload bodies.
func publicFuzzSeeds(f *testing.F, mode EncoderScalabilityMode, width, height, pictures int) (lowOverhead []byte, rtp []byte) {
	f.Helper()
	cfg := publicRTCMatrixConfig(width, height, mode)
	enc, err := NewRTCEncoderWithConfig(cfg)
	if err != nil {
		f.Fatalf("NewRTCEncoderWithConfig(%s): %v", mode, err)
	}
	defer enc.Close()
	var lowOverheads, rtpPayloads [][]byte
	for i := range pictures {
		picture, err := enc.EncodePicture(publicRTCMatrixFrame(width, height, i), false)
		if err != nil {
			f.Fatalf("EncodePicture(%s, %d): %v", mode, i, err)
		}
		for _, frame := range picture.Frames[:picture.FrameNum] {
			lowOverheads = append(lowOverheads, append([]byte(nil), frame.Data...))
			rtpPayloads = append(rtpPayloads, publicDecoderRTPPayloadsForFrameWithLimits(f, frame, RTPPayloadSizeLimits{MaxPayloadLen: publicFuzzRTPPayloadLen})...)
		}
	}
	return joinPublicFuzzPayloads(lowOverheads), joinPublicFuzzPayloads(rtpPayloads)
}

// publicFuzzSharedSVCMode returns the first WebRTC mode whose spatial layers
// share reference slots, the stream shape only LayeredDecoder accepts.
func publicFuzzSharedSVCMode(f *testing.F) EncoderScalabilityMode {
	f.Helper()
	for _, mode := range EncoderWebRTCScalabilityModes() {
		if mode.UsesSharedReferenceSlots() {
			return mode
		}
	}
	f.Fatal("no shared-reference WebRTC scalability mode")
	return 0
}

// drainPublicFuzzDecoder decodes until the payloads are exhausted or the
// decoder reports an error. Errors are expected on hostile input; a panic or a
// hang is the failure.
func drainPublicFuzzDecoder(next func() (bool, error), payloads int) {
	for range payloads + 8 {
		ok, err := next()
		if err != nil || !ok {
			return
		}
	}
}

func FuzzPublicDecoderLowOverhead(f *testing.F) {
	seed, _ := publicFuzzSeeds(f, EncoderScalabilityModeL1T2, publicFuzzSeedWidth, publicFuzzSeedHeight, 3)
	key, _ := publicFuzzSeeds(f, EncoderScalabilityModeL1T1, publicFuzzSeedWidth, publicFuzzSeedHeight, 1)
	f.Add(seed, uint8(0))
	f.Add(seed, uint8(3))
	f.Add(key, uint8(1))

	f.Fuzz(func(t *testing.T, data []byte, rawWorkers uint8) {
		payloads := splitPublicFuzzPayloads(data)
		if len(payloads) == 0 {
			return
		}
		dec, err := NewDecoder(payloads, WithWorkers(1+int(rawWorkers%4)), WithFrameSizeLimit(publicFuzzFrameSizeMax))
		if err != nil {
			return
		}
		defer dec.Close()
		drainPublicFuzzDecoder(func() (bool, error) {
			_, ok, err := dec.DecodeNext()
			return ok, err
		}, len(payloads))
	})
}

func FuzzPublicDecoderRTPPayloads(f *testing.F) {
	_, seed := publicFuzzSeeds(f, EncoderScalabilityModeL1T2, publicFuzzSeedWidth, publicFuzzSeedHeight, 3)
	f.Add(seed, false)
	f.Add(seed, true)

	f.Fuzz(func(t *testing.T, data []byte, afterLoss bool) {
		payloads := splitPublicFuzzPayloads(data)
		if len(payloads) == 0 {
			return
		}
		dec, err := NewDecoderFromRTPPayloads(payloads, WithFrameSizeLimit(publicFuzzFrameSizeMax))
		if err != nil {
			return
		}
		defer dec.Close()
		if !afterLoss {
			drainPublicFuzzDecoder(func() (bool, error) {
				_, ok, err := dec.DecodeNext()
				return ok, err
			}, len(payloads))
			return
		}
		// Live receive with a loss before every other payload exercises the
		// retained-fragment reset paths on hostile input.
		for i, payload := range payloads {
			decode := dec.DecodeRTPPayload
			if i%2 == 1 {
				decode = dec.DecodeRTPPayloadAfterLoss
			}
			if _, err := decode(payload); err != nil {
				return
			}
		}
	})
}

func FuzzPublicLayeredDecoder(f *testing.F) {
	lowOverhead, rtp := publicFuzzSeeds(f, publicFuzzSharedSVCMode(f), 2*publicFuzzSeedWidth, 2*publicFuzzSeedHeight, 2)
	f.Add(lowOverhead, false)
	f.Add(rtp, true)

	f.Fuzz(func(t *testing.T, data []byte, rtp bool) {
		payloads := splitPublicFuzzPayloads(data)
		if len(payloads) == 0 {
			return
		}
		newDecoder := NewLayeredDecoder
		if rtp {
			newDecoder = NewLayeredDecoderFromRTPPayloads
		}
		dec, err := newDecoder(payloads, WithFrameSizeLimit(publicFuzzFrameSizeMax))
		if err != nil {
			return
		}
		defer dec.Close()
		drainPublicFuzzDecoder(func() (bool, error) {
			_, ok, err := dec.DecodeNext()
			return ok, err
		}, len(payloads))
	})
}

const fuzzSimpleDecoderMaxIVFBytes = 8 << 10
const fuzzSimpleDecoderMaxFrames = 64

func FuzzPublicSimpleDecoderIVF(f *testing.F) {
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendOBUForTest(nil, internalobu.TypeTemporalDelimiter, nil)},
	}))

	switchOnly := appendOBUForTest(nil, internalobu.TypeSequenceHeader, publicDecoderResidualRealtimeSequenceHeaderPayload())
	switchFrame := append([]byte{}, publicSimpleDecoderSwitchFrameHeaderPayload()...)
	switchFrame = append(switchFrame, 0xbb)
	switchOnly = appendOBUForTest(switchOnly, internalobu.TypeFrame, switchFrame)
	f.Add(appendPublicIVF(nil, 16, 9, 30, 1, []publicIVFFrame{
		{payload: switchOnly},
	}))

	validTilePayload := appendTileListOBU(nil, internalparser.TileList{
		OutputFrameWidthInTilesMinus1:  0,
		OutputFrameHeightInTilesMinus1: 0,
		TileCountMinus1:                0,
		Entries: []internalparser.TileListEntry{{
			AnchorFrameIdx:     0,
			AnchorTileRow:      0,
			AnchorTileCol:      0,
			TileDataSizeMinus1: 2,
			TileData:           []byte{0xaa, 0xbb, 0xcc},
		}},
	})
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendOBUForTest(nil, internalobu.TypeTileList, validTilePayload)},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendOBUForTest(nil, internalobu.TypeTileList, []byte{0x00, 0x00, 0x00, 0x00})},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendOBUForTest(nil, internalobu.TypeTileList, []byte{0xff, 0xff, 0xff, 0xff})},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendOBUForTest(nil, internalobu.TypeTileList, []byte{0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0x00, 0x04, 0xaa})},
	}))
	f.Add(appendPublicIVF(nil, 16, 16, 30, 1, []publicIVFFrame{
		{payload: appendOBUForTest(nil, internalobu.TypeTileList, append(append([]byte{}, validTilePayload...), 0xff))},
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

		dec, err := NewDecoderFromIVF(data)
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

func FuzzPublicDecoderFrameWorkBatchResidualRunnerSideData(f *testing.F) {
	f.Add(uint8(1), uint8(0), false, false, false)
	f.Add(uint8(2), uint8(1), true, false, false)
	f.Add(uint8(4), uint8(2), true, true, false)
	f.Add(uint8(3), uint8(3), false, false, true)

	f.Fuzz(func(t *testing.T, rawWorkers uint8, rawCDEFBits uint8, activeRestoration bool, omitRestorationScratch bool, corruptLoopMap bool) {
		workers := int(rawWorkers%4) + 1
		sequence := SequenceHeader{
			EnableCDEF:        true,
			EnableRestoration: true,
			ColorConfig: internalparser.ColorConfig{
				BitDepth:   8,
				MonoChrome: true,
			},
		}
		size := FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8}
		cdefBits := rawCDEFBits & 3
		cdef := internalparser.CDEFParams{
			Bits:          cdefBits,
			StrengthCount: uint8(1) << cdefBits,
		}
		restoration := internalparser.RestorationParams{}
		if activeRestoration {
			restoration = internalparser.RestorationParams{
				Type:      [3]internalparser.RestorationType{internalparser.RestorationWiener},
				UnitSizeY: 64,
			}
		}

		batch := internaldecoder.FrameWorkBatch{
			FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
				Sequence:     internalthreading.FrameWorkSequenceContextFromHeader(sequence),
				FrameSize:    size,
				CDEF:         cdef,
				Restoration:  restoration,
				Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
				TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
			},
			Jobs: []internaltile.Job{
				{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 1},
				{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 1, Size: 1},
			},
		}
		runnerSize, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, workers)
		if err != nil {
			t.Fatal(err)
		}
		runnerScratch := publicDecoderBatchResidualRunnerScratch(runnerSize)
		if omitRestorationScratch {
			runnerScratch.RestorationRequests = nil
		}
		runner, err := bindDecoderFrameWorkBatchResidualRunner(runnerSize, runnerScratch)
		if err != nil {
			t.Fatal(err)
		}

		sideSize, err := decoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
		if err != nil {
			t.Fatal(err)
		}
		side, err := bindDecoderFrameWorkSideData(sequence, size, cdef, restoration, decoderFrameWorkSideDataScratch{
			CDEFIndexMap:             make([]uint8, sideSize.CDEFIndexMap),
			CDEFReadMap:              make([]bool, sideSize.CDEFReadMap),
			LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, sideSize.LoopFilterMap),
			RestorationRecords:       make([]internaltile.RestorationUnitRecord, sideSize.RestorationRecords),
			RestorationBoundaryAbove: make([]uint16, sideSize.RestorationBoundaryAbove),
			RestorationBoundaryBelow: make([]uint16, sideSize.RestorationBoundaryBelow),
		})
		if err != nil {
			t.Fatal(err)
		}
		side.CDEFIndexMap.Read[0] = true
		side.LoopFilterMap.Records[0].Valid = true
		if len(side.RestorationFrameBuffers.Records[0]) != 0 {
			side.RestorationFrameBuffers.Records[0][0].Index = 99
		}
		if corruptLoopMap {
			side.LoopFilterMap = internaldecoder.FrameWorkLoopFilterMap{Stride: 1, Rows: 1}
		}

		err = setDecoderFrameWorkBatchResidualRunnerSideData(&runner, side)
		if activeRestoration && omitRestorationScratch {
			if !errors.Is(err, internalframe.ErrShortBuffer) {
				t.Fatalf("missing restoration requests err=%v want %v", err, internalframe.ErrShortBuffer)
			}
			return
		}
		if corruptLoopMap {
			if !errors.Is(err, internalthreading.ErrInvalidBatch) {
				t.Fatalf("corrupt loop map err=%v want %v", err, internalthreading.ErrInvalidBatch)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if side.CDEFIndexMap.Read[0] || side.LoopFilterMap.Records[0].Valid {
			t.Fatalf("side data was not reset: cdef=%v loop=%+v", side.CDEFIndexMap.Read[0], side.LoopFilterMap.Records[0])
		}
		if runner.CDEFIndexMap.Stride != side.CDEFIndexMap.Stride ||
			runner.LoopFilterMap.Stride != side.LoopFilterMap.Stride {
			t.Fatalf("runner side maps cdef=%+v loop=%+v side=%+v", runner.CDEFIndexMap, runner.LoopFilterMap, side)
		}
		if activeRestoration {
			if len(runner.RestorationRequests) < workers ||
				!runner.RestorationRequests[0].Buffers.Plan.Active {
				t.Fatalf("runner restoration requests=%+v workers=%d", runner.RestorationRequests, workers)
			}
		}
	})
}
