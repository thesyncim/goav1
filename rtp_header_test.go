package goav1

import (
	"bytes"
	"errors"
	"testing"
)

func TestPublicRTPPacketHeaderOneByteExtensionRoundTrip(t *testing.T) {
	var mid [16]byte
	midN, err := PutRTPMIDHeaderExtension(mid[:], "v0")
	if err != nil {
		t.Fatalf("PutRTPMIDHeaderExtension: %v", err)
	}
	var twcc [RTPTransportWideCCHeaderExtensionSize]byte
	twccN, err := PutRTPTransportWideCCHeaderExtension(twcc[:], 0x3456)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCCHeaderExtension: %v", err)
	}
	elements := [...]RTPHeaderExtensionElement{
		{ID: 1, Payload: mid[:midN]},
		{ID: 3, Payload: twcc[:twccN]},
	}
	var ext [32]byte
	extN, err := PutRTPHeaderExtensionElements(ext[:], RTPExtensionProfileOneByte, elements[:])
	if err != nil {
		t.Fatalf("PutRTPHeaderExtensionElements: %v", err)
	}
	if extN != 6 {
		t.Fatalf("extension bytes=%d want 6", extN)
	}

	header := RTPHeader{
		Marker:           true,
		PayloadType:      96,
		SequenceNumber:   0x1234,
		Timestamp:        0x01020304,
		SSRC:             0xaabbccdd,
		CSRCCount:        2,
		CSRC:             [RTPMaxCSRC]uint32{0x11111111, 0x22222222},
		ExtensionProfile: RTPExtensionProfileOneByte,
		ExtensionPayload: ext[:extN],
	}
	payload := []byte{0x12, 0x34, 0x56}
	packetSize, err := RTPPacketSize(header, payload, 4)
	if err != nil {
		t.Fatalf("RTPPacketSize: %v", err)
	}
	var packetBuf [128]byte
	n, err := PutRTPPacket(packetBuf[:], header, payload, 4)
	if err != nil {
		t.Fatalf("PutRTPPacket: %v", err)
	}
	if n != packetSize {
		t.Fatalf("PutRTPPacket wrote %d want size %d", n, packetSize)
	}

	headerOnly, payloadOffset, err := ParseRTPHeader(packetBuf[:n])
	if err != nil {
		t.Fatalf("ParseRTPHeader: %v", err)
	}
	if payloadOffset != n-len(payload)-4 || !headerOnly.Padding || headerOnly.PaddingSize != 0 {
		t.Fatalf("header payloadOffset=%d padding=%v/%d n=%d", payloadOffset, headerOnly.Padding, headerOnly.PaddingSize, n)
	}

	packet, err := ParseRTPPacket(packetBuf[:n])
	if err != nil {
		t.Fatalf("ParseRTPPacket: %v", err)
	}
	if !packet.Header.Marker ||
		packet.Header.PayloadType != 96 ||
		packet.Header.SequenceNumber != 0x1234 ||
		packet.Header.Timestamp != 0x01020304 ||
		packet.Header.SSRC != 0xaabbccdd ||
		packet.Header.CSRCCount != 2 ||
		packet.Header.CSRC[0] != 0x11111111 ||
		packet.Header.CSRC[1] != 0x22222222 ||
		packet.Header.ExtensionProfile != RTPExtensionProfileOneByte ||
		packet.Header.PaddingSize != 4 {
		t.Fatalf("parsed header=%+v", packet.Header)
	}
	if !bytes.Equal(packet.Payload, payload) {
		t.Fatalf("payload=%x want %x", packet.Payload, payload)
	}
	if len(packet.Header.ExtensionPayload) != 8 ||
		!bytes.Equal(packet.Header.ExtensionPayload[:extN], ext[:extN]) ||
		!bytes.Equal(packet.Header.ExtensionPayload[extN:], []byte{0, 0}) {
		t.Fatalf("extension payload=%x", packet.Header.ExtensionPayload)
	}

	var parsedExt [4]RTPHeaderExtensionElement
	count, err := ParseRTPHeaderExtensionElements(packet.Header.ExtensionProfile, packet.Header.ExtensionPayload, parsedExt[:])
	if err != nil {
		t.Fatalf("ParseRTPHeaderExtensionElements: %v", err)
	}
	if count != 2 {
		t.Fatalf("extension count=%d want 2", count)
	}
	gotMID, err := ParseRTPMIDHeaderExtension(parsedExt[0].Payload)
	if err != nil || gotMID != "v0" {
		t.Fatalf("MID=%q err=%v", gotMID, err)
	}
	gotTWCC, err := ParseRTPTransportWideCCHeaderExtension(parsedExt[1].Payload)
	if err != nil || gotTWCC != 0x3456 {
		t.Fatalf("TWCC=%04x err=%v", gotTWCC, err)
	}
}

func TestPublicRTPTwoByteHeaderExtensionElements(t *testing.T) {
	profile := RTPExtensionProfileTwoByte | 0x000b
	long := bytes.Repeat([]byte{0xab}, 32)
	elements := [...]RTPHeaderExtensionElement{
		{ID: 33, Payload: long},
		{ID: 9, Payload: nil},
	}
	size, err := RTPHeaderExtensionElementsSize(profile, elements[:])
	if err != nil {
		t.Fatalf("RTPHeaderExtensionElementsSize: %v", err)
	}
	if size != 36 {
		t.Fatalf("size=%d want 36", size)
	}
	buf := make([]byte, size+3)
	n, err := PutRTPHeaderExtensionElements(buf, profile, elements[:])
	if err != nil {
		t.Fatalf("PutRTPHeaderExtensionElements: %v", err)
	}
	if n != size {
		t.Fatalf("wrote=%d want %d", n, size)
	}
	payload := append(buf[:n], 0, 0, 0)

	var parsed [2]RTPHeaderExtensionElement
	count, err := ParseRTPHeaderExtensionElements(profile, payload, parsed[:])
	if err != nil {
		t.Fatalf("ParseRTPHeaderExtensionElements: %v", err)
	}
	if count != 2 ||
		parsed[0].ID != 33 ||
		!bytes.Equal(parsed[0].Payload, long) ||
		parsed[1].ID != 9 ||
		len(parsed[1].Payload) != 0 {
		t.Fatalf("parsed count=%d elements=%+v", count, parsed)
	}
}

func TestPublicRTPHeaderExtensionElementFind(t *testing.T) {
	oneBytePayload := []byte{0x10, 0xaa, 0x32, 0xde, 0xad, 0xbe, 0, 0}
	element, ok, err := FindRTPHeaderExtensionElement(RTPExtensionProfileOneByte, oneBytePayload, 3)
	if err != nil || !ok || element.ID != 3 || !bytes.Equal(element.Payload, []byte{0xde, 0xad, 0xbe}) {
		t.Fatalf("one-byte element=%+v ok=%v err=%v", element, ok, err)
	}
	if element, ok, err := FindRTPHeaderExtensionElement(RTPExtensionProfileOneByte, oneBytePayload, 4); err != nil || ok || element.ID != 0 {
		t.Fatalf("missing one-byte element=%+v ok=%v err=%v", element, ok, err)
	}
	if element, ok, err := FindRTPHeaderExtensionElement(RTPExtensionProfileOneByte, []byte{0xf0, 0x10, 0xaa}, 1); err != nil || ok || element.ID != 0 {
		t.Fatalf("terminated one-byte element=%+v ok=%v err=%v", element, ok, err)
	}
	if _, _, err := FindRTPHeaderExtensionElement(RTPExtensionProfileOneByte, oneBytePayload, 0); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("one-byte id zero err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	if _, _, err := FindRTPHeaderExtensionElement(RTPExtensionProfileOneByte, oneBytePayload, 15); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("one-byte id reserved err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	if _, _, err := FindRTPHeaderExtensionElement(RTPExtensionProfileOneByte, []byte{0x31, 0xaa}, 3); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short one-byte payload err=%v want %v", err, ErrRTPShortPayload)
	}

	twoByteProfile := RTPExtensionProfileTwoByte | 0x000f
	twoBytePayload := []byte{0, 33, 2, 0xca, 0xfe, 9, 0, 0}
	element, ok, err = FindRTPHeaderExtensionElement(twoByteProfile, twoBytePayload, 33)
	if err != nil || !ok || element.ID != 33 || !bytes.Equal(element.Payload, []byte{0xca, 0xfe}) {
		t.Fatalf("two-byte element=%+v ok=%v err=%v", element, ok, err)
	}
	element, ok, err = FindRTPHeaderExtensionElement(twoByteProfile, twoBytePayload, 9)
	if err != nil || !ok || element.ID != 9 || len(element.Payload) != 0 {
		t.Fatalf("empty two-byte element=%+v ok=%v err=%v", element, ok, err)
	}
	if element, ok, err := FindRTPHeaderExtensionElement(twoByteProfile, twoBytePayload, 34); err != nil || ok || element.ID != 0 {
		t.Fatalf("missing two-byte element=%+v ok=%v err=%v", element, ok, err)
	}
	if _, _, err := FindRTPHeaderExtensionElement(twoByteProfile, []byte{33}, 33); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short two-byte length err=%v want %v", err, ErrRTPShortPayload)
	}
	if _, _, err := FindRTPHeaderExtensionElement(twoByteProfile, []byte{33, 3, 0xca}, 33); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short two-byte payload err=%v want %v", err, ErrRTPShortPayload)
	}
}

func TestPublicRTPHeaderRejectsInvalid(t *testing.T) {
	if _, _, err := ParseRTPHeader([]byte{0x80}); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short header err=%v want %v", err, ErrRTPShortPayload)
	}
	if _, _, err := ParseRTPHeader(append([]byte{0x40}, make([]byte, RTPHeaderMinSize-1)...)); !errors.Is(err, ErrRTPInvalidHeader) {
		t.Fatalf("bad version err=%v want %v", err, ErrRTPInvalidHeader)
	}
	if _, _, err := ParseRTPHeader([]byte{0x81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short csrc err=%v want %v", err, ErrRTPShortPayload)
	}
	if _, _, err := ParseRTPHeader([]byte{0x90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short extension err=%v want %v", err, ErrRTPShortPayload)
	}
	if _, err := ParseRTPPacket([]byte{0xa0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}); !errors.Is(err, ErrRTPInvalidHeader) {
		t.Fatalf("padding without payload err=%v want %v", err, ErrRTPInvalidHeader)
	}
	if _, err := RTPPacketSize(RTPHeader{PayloadType: 128}, nil, 0); !errors.Is(err, ErrRTPInvalidHeader) {
		t.Fatalf("bad payload type err=%v want %v", err, ErrRTPInvalidHeader)
	}
	if _, err := PutRTPPacket(make([]byte, RTPHeaderMinSize-1), RTPHeader{}, nil, 0); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPPacket err=%v want %v", err, ErrRTPShortBuffer)
	}
}

func TestPublicRTPHeaderExtensionElementsRejectInvalid(t *testing.T) {
	tests := []struct {
		name     string
		profile  uint16
		elements []RTPHeaderExtensionElement
		want     error
	}{
		{
			name:     "one-byte-id-zero",
			profile:  RTPExtensionProfileOneByte,
			elements: []RTPHeaderExtensionElement{{ID: 0, Payload: []byte{1}}},
			want:     ErrRTPInvalidHeaderExtension,
		},
		{
			name:     "one-byte-id-reserved",
			profile:  RTPExtensionProfileOneByte,
			elements: []RTPHeaderExtensionElement{{ID: 15, Payload: []byte{1}}},
			want:     ErrRTPInvalidHeaderExtension,
		},
		{
			name:     "one-byte-empty",
			profile:  RTPExtensionProfileOneByte,
			elements: []RTPHeaderExtensionElement{{ID: 1}},
			want:     ErrRTPInvalidHeaderExtension,
		},
		{
			name:     "one-byte-too-long",
			profile:  RTPExtensionProfileOneByte,
			elements: []RTPHeaderExtensionElement{{ID: 1, Payload: make([]byte, 17)}},
			want:     ErrRTPInvalidHeaderExtension,
		},
		{
			name:     "two-byte-id-zero",
			profile:  RTPExtensionProfileTwoByte,
			elements: []RTPHeaderExtensionElement{{ID: 0, Payload: []byte{1}}},
			want:     ErrRTPInvalidHeaderExtension,
		},
		{
			name:     "two-byte-too-long",
			profile:  RTPExtensionProfileTwoByte,
			elements: []RTPHeaderExtensionElement{{ID: 1, Payload: make([]byte, 256)}},
			want:     ErrRTPInvalidHeaderExtension,
		},
		{
			name:     "unknown-profile",
			profile:  0x1234,
			elements: []RTPHeaderExtensionElement{{ID: 1, Payload: []byte{1}}},
			want:     ErrRTPInvalidHeaderExtension,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := RTPHeaderExtensionElementsSize(tc.profile, tc.elements); !errors.Is(err, tc.want) {
				t.Fatalf("RTPHeaderExtensionElementsSize err=%v want %v", err, tc.want)
			}
			if _, err := PutRTPHeaderExtensionElements(make([]byte, 512), tc.profile, tc.elements); !errors.Is(err, tc.want) {
				t.Fatalf("PutRTPHeaderExtensionElements err=%v want %v", err, tc.want)
			}
		})
	}

	if _, err := PutRTPHeaderExtensionElements(nil, RTPExtensionProfileOneByte, []RTPHeaderExtensionElement{{ID: 1, Payload: []byte{1}}}); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPHeaderExtensionElements err=%v want %v", err, ErrRTPShortBuffer)
	}
	var out [1]RTPHeaderExtensionElement
	if _, err := ParseRTPHeaderExtensionElements(RTPExtensionProfileOneByte, []byte{0x11, 0xaa}, out[:]); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short one-byte payload err=%v want %v", err, ErrRTPShortPayload)
	}
	if count, err := ParseRTPHeaderExtensionElements(RTPExtensionProfileOneByte, []byte{0x10, 0xaa, 0xf0, 0x20}, out[:]); err != nil || count != 1 || out[0].ID != 1 || !bytes.Equal(out[0].Payload, []byte{0xaa}) {
		t.Fatalf("reserved one-byte terminator count=%d out=%+v err=%v", count, out, err)
	}
	if _, err := ParseRTPHeaderExtensionElements(RTPExtensionProfileOneByte, []byte{0x00, 0x10, 0xaa}, nil); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short one-byte output err=%v want %v", err, ErrRTPShortBuffer)
	}
	if _, err := ParseRTPHeaderExtensionElements(RTPExtensionProfileTwoByte, []byte{0x01}, out[:]); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short two-byte length err=%v want %v", err, ErrRTPShortPayload)
	}
	if _, err := ParseRTPHeaderExtensionElements(RTPExtensionProfileTwoByte, []byte{0x01, 0x02, 0xaa}, out[:]); !errors.Is(err, ErrRTPShortPayload) {
		t.Fatalf("short two-byte payload err=%v want %v", err, ErrRTPShortPayload)
	}
	if _, err := ParseRTPHeaderExtensionElements(0x1234, nil, out[:]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("unknown profile parse err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
	if _, err := rtpPaddedExtensionPayloadLen(int(^uint(0) >> 1)); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("overflow extension payload length err=%v want %v", err, ErrRTPInvalidHeaderExtension)
	}
}
