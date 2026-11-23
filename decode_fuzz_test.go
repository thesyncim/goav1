package goav1

import (
	"encoding/binary"
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
