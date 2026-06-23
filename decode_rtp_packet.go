package goav1

import "fmt"

// NewDecoderFromRTPPackets builds a Decoder from ordered complete RTP packets.
// RTP fixed headers, CSRCs, extension envelopes, and padding are parsed and
// stripped before the AV1 RTP payload bodies enter the same receive path used by
// NewDecoderFromRTPPayloads.
//
// The packet payload slices alias packets; callers must keep the packet backing
// bytes alive and unmodified for the Decoder's lifetime.
func NewDecoderFromRTPPackets(packets [][]byte, opts ...Option) (*Decoder, error) {
	payloads, err := rtpPayloadsFromPackets(packets)
	if err != nil {
		return nil, err
	}
	return NewDecoderFromRTPPayloads(payloads, opts...)
}

// NewLayeredDecoderFromRTPPackets builds a LayeredDecoder from ordered complete
// RTP packets. It is the full-packet counterpart to
// NewLayeredDecoderFromRTPPayloads for shared-reference WebRTC SVC receive
// loops.
//
// The packet payload slices alias packets; callers must keep the packet backing
// bytes alive and unmodified for the LayeredDecoder's lifetime.
func NewLayeredDecoderFromRTPPackets(packets [][]byte, opts ...Option) (*LayeredDecoder, error) {
	payloads, err := rtpPayloadsFromPackets(packets)
	if err != nil {
		return nil, err
	}
	return NewLayeredDecoderFromRTPPayloads(payloads, opts...)
}

func rtpPayloadsFromPackets(packets [][]byte) ([][]byte, error) {
	payloads := make([][]byte, len(packets))
	for i := range packets {
		packet, err := ParseRTPPacket(packets[i])
		if err != nil {
			return nil, fmt.Errorf("goav1: rtp packet %d: %w", i, err)
		}
		payloads[i] = packet.Payload
	}
	return payloads, nil
}
