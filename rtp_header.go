package goav1

import (
	"encoding/binary"
	"errors"
)

const (
	// RTPVersion is the RTP version supported by WebRTC media streams.
	RTPVersion = 2
	// RTPHeaderMinSize is the fixed RTP header size without CSRC or extension
	// words.
	RTPHeaderMinSize = 12
	// RTPMaxCSRC is the maximum CSRC count encodable in the RTP fixed header.
	RTPMaxCSRC = 15

	// RTPExtensionProfileOneByte is the RFC 8285 one-byte RTP header-extension
	// profile identifier.
	RTPExtensionProfileOneByte uint16 = 0xBEDE
	// RTPExtensionProfileTwoByte is the RFC 8285 two-byte RTP header-extension
	// profile identifier with zero appbits. Two-byte profiles with non-zero
	// appbits are accepted when the high 12 bits match this value.
	RTPExtensionProfileTwoByte uint16 = 0x1000
)

// ErrRTPInvalidHeader is returned when an RTP fixed header, CSRC list, extension
// envelope, or packet padding value is malformed.
var ErrRTPInvalidHeader = errors.New("goav1: invalid RTP header")

// ErrRTPHeaderExtensionNotFound is returned when a negotiated RTP header
// extension ID is absent from an otherwise valid RTP header-extension block.
var ErrRTPHeaderExtensionNotFound = errors.New("goav1: rtp header extension not found")

// RTPHeader is the parsed RTP packet header. ExtensionPayload aliases the packet
// input returned by ParseRTPHeader/ParseRTPPacket and includes RFC 8285 padding
// bytes up to the 32-bit extension-word boundary.
type RTPHeader struct {
	Padding bool
	Marker  bool

	PayloadType    uint8
	SequenceNumber uint16
	Timestamp      uint32
	SSRC           uint32

	CSRC      [RTPMaxCSRC]uint32
	CSRCCount uint8

	ExtensionProfile uint16
	ExtensionPayload []byte

	// PaddingSize is populated by ParseRTPPacket and used by RTPPacketSize /
	// PutRTPPacket. ParseRTPHeader only reports whether the padding bit is set.
	PaddingSize uint8
}

// RTPPacket is one RTP packet with payload aliasing the input passed to
// ParseRTPPacket. Padding bytes are excluded from Payload.
type RTPPacket struct {
	Header  RTPHeader
	Payload []byte
}

// RTPHeaderExtensionElement is one RFC 8285 RTP header-extension element. Payload
// aliases the extension payload passed to ParseRTPHeaderExtensionElements.
type RTPHeaderExtensionElement struct {
	ID      uint8
	Payload []byte
}

// RTPHeaderExtensionElementsSize reports the bytes required for the RFC 8285
// extension element payload before RTP's 32-bit extension-word padding.
func RTPHeaderExtensionElementsSize(profile uint16, elements []RTPHeaderExtensionElement) (int, error) {
	kind, err := rtpHeaderExtensionProfileKind(profile)
	if err != nil {
		return 0, ErrRTPInvalidHeaderExtension
	}
	size := 0
	for i := range elements {
		elementSize, err := rtpHeaderExtensionElementSize(kind, elements[i].ID, len(elements[i].Payload))
		if err != nil {
			return 0, err
		}
		size += elementSize
	}
	return size, nil
}

// PutRTPHeaderExtensionElements writes RFC 8285 extension elements into dst and
// returns the unpadded payload length. RTPHeader/PutRTPPacket add the required
// 32-bit RTP extension padding.
func PutRTPHeaderExtensionElements(dst []byte, profile uint16, elements []RTPHeaderExtensionElement) (int, error) {
	kind, err := rtpHeaderExtensionProfileKind(profile)
	if err != nil {
		return 0, err
	}
	size, err := RTPHeaderExtensionElementsSize(profile, elements)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTPShortBuffer
	}
	off := 0
	for i := range elements {
		payload := elements[i].Payload
		switch kind {
		case rtpHeaderExtensionProfileOneByte:
			dst[off] = elements[i].ID<<4 | uint8(len(payload)-1)
			off++
		case rtpHeaderExtensionProfileTwoByte:
			dst[off] = elements[i].ID
			dst[off+1] = uint8(len(payload))
			off += 2
		}
		off += copy(dst[off:], payload)
	}
	return off, nil
}

// ParseRTPHeaderExtensionElements parses RFC 8285 one-byte or two-byte extension
// elements from payload into out and returns the number of elements written.
// Padding bytes are ignored.
func ParseRTPHeaderExtensionElements(profile uint16, payload []byte, out []RTPHeaderExtensionElement) (int, error) {
	kind, err := rtpHeaderExtensionProfileKind(profile)
	if err != nil {
		return 0, ErrRTPInvalidHeaderExtension
	}
	count := 0
	off := 0
	for off < len(payload) {
		switch kind {
		case rtpHeaderExtensionProfileOneByte:
			header := payload[off]
			off++
			if header == 0 {
				continue
			}
			id := header >> 4
			if id == 15 {
				return count, nil
			}
			n := int(header&0x0f) + 1
			if off+n > len(payload) {
				return count, ErrRTPShortPayload
			}
			if count >= len(out) {
				return count, ErrRTPShortBuffer
			}
			out[count] = RTPHeaderExtensionElement{ID: id, Payload: payload[off : off+n]}
			count++
			off += n
		case rtpHeaderExtensionProfileTwoByte:
			id := payload[off]
			off++
			if id == 0 {
				continue
			}
			if off >= len(payload) {
				return count, ErrRTPShortPayload
			}
			n := int(payload[off])
			off++
			if off+n > len(payload) {
				return count, ErrRTPShortPayload
			}
			if count >= len(out) {
				return count, ErrRTPShortBuffer
			}
			out[count] = RTPHeaderExtensionElement{ID: id, Payload: payload[off : off+n]}
			count++
			off += n
		}
	}
	return count, nil
}

// FindRTPHeaderExtensionElement scans an RFC 8285 one-byte or two-byte
// extension payload for id without allocating. Padding bytes are ignored, and
// the returned Payload aliases payload.
func FindRTPHeaderExtensionElement(profile uint16, payload []byte, id uint8) (RTPHeaderExtensionElement, bool, error) {
	if id == 0 {
		return RTPHeaderExtensionElement{}, false, ErrRTPInvalidHeaderExtension
	}
	kind, err := rtpHeaderExtensionProfileKind(profile)
	if err != nil {
		return RTPHeaderExtensionElement{}, false, ErrRTPInvalidHeaderExtension
	}
	if kind == rtpHeaderExtensionProfileOneByte && id >= 15 {
		return RTPHeaderExtensionElement{}, false, ErrRTPInvalidHeaderExtension
	}
	off := 0
	for off < len(payload) {
		switch kind {
		case rtpHeaderExtensionProfileOneByte:
			header := payload[off]
			off++
			if header == 0 {
				continue
			}
			elementID := header >> 4
			if elementID == 15 {
				return RTPHeaderExtensionElement{}, false, nil
			}
			n := int(header&0x0f) + 1
			if off+n > len(payload) {
				return RTPHeaderExtensionElement{}, false, ErrRTPShortPayload
			}
			if elementID == id {
				return RTPHeaderExtensionElement{ID: elementID, Payload: payload[off : off+n]}, true, nil
			}
			off += n
		case rtpHeaderExtensionProfileTwoByte:
			elementID := payload[off]
			off++
			if elementID == 0 {
				continue
			}
			if off >= len(payload) {
				return RTPHeaderExtensionElement{}, false, ErrRTPShortPayload
			}
			n := int(payload[off])
			off++
			if off+n > len(payload) {
				return RTPHeaderExtensionElement{}, false, ErrRTPShortPayload
			}
			if elementID == id {
				return RTPHeaderExtensionElement{ID: elementID, Payload: payload[off : off+n]}, true, nil
			}
			off += n
		}
	}
	return RTPHeaderExtensionElement{}, false, nil
}

// RTPHeaderSize reports the bytes required to write header, including CSRCs and
// the extension envelope if ExtensionProfile is set.
func RTPHeaderSize(header RTPHeader) (int, error) {
	if header.PayloadType > 127 || header.CSRCCount > RTPMaxCSRC {
		return 0, ErrRTPInvalidHeader
	}
	size := RTPHeaderMinSize + int(header.CSRCCount)*4
	if header.ExtensionProfile != 0 || len(header.ExtensionPayload) != 0 {
		if header.ExtensionProfile == 0 {
			return 0, ErrRTPInvalidHeader
		}
		extLen, err := rtpPaddedExtensionPayloadLen(len(header.ExtensionPayload))
		if err != nil {
			return 0, err
		}
		size += 4 + extLen
	}
	return size, nil
}

// RTPPacketSize reports the bytes required to write header, payload, and packet
// padding. If header.Padding is true, paddingSize must be non-zero.
func RTPPacketSize(header RTPHeader, payload []byte, paddingSize uint8) (int, error) {
	if header.Padding && paddingSize == 0 {
		return 0, ErrRTPInvalidHeader
	}
	header.Padding = paddingSize != 0
	header.PaddingSize = paddingSize
	size, err := RTPHeaderSize(header)
	if err != nil {
		return 0, err
	}
	return size + len(payload) + int(paddingSize), nil
}

// PutRTPHeader writes header into dst and returns the bytes written. Packet
// padding bytes are written by PutRTPPacket.
func PutRTPHeader(dst []byte, header RTPHeader) (int, error) {
	size, err := RTPHeaderSize(header)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTPShortBuffer
	}

	hasExtension := header.ExtensionProfile != 0 || len(header.ExtensionPayload) != 0
	dst[0] = byte(RTPVersion << 6)
	if header.Padding || header.PaddingSize != 0 {
		dst[0] |= 0x20
	}
	if hasExtension {
		dst[0] |= 0x10
	}
	dst[0] |= header.CSRCCount & 0x0f
	dst[1] = header.PayloadType & 0x7f
	if header.Marker {
		dst[1] |= 0x80
	}
	binary.BigEndian.PutUint16(dst[2:4], header.SequenceNumber)
	binary.BigEndian.PutUint32(dst[4:8], header.Timestamp)
	binary.BigEndian.PutUint32(dst[8:12], header.SSRC)

	off := RTPHeaderMinSize
	for i := uint8(0); i < header.CSRCCount; i++ {
		binary.BigEndian.PutUint32(dst[off:off+4], header.CSRC[i])
		off += 4
	}

	if hasExtension {
		extLen, err := rtpPaddedExtensionPayloadLen(len(header.ExtensionPayload))
		if err != nil {
			return 0, err
		}
		binary.BigEndian.PutUint16(dst[off:off+2], header.ExtensionProfile)
		binary.BigEndian.PutUint16(dst[off+2:off+4], uint16(extLen/4))
		off += 4
		off += copy(dst[off:off+extLen], header.ExtensionPayload)
		clear(dst[off : off+extLen-len(header.ExtensionPayload)])
		off += extLen - len(header.ExtensionPayload)
	}
	return off, nil
}

// PutRTPPacket writes a complete RTP packet into dst and returns the bytes
// written. paddingSize, when non-zero, writes RTP packet padding and sets the
// fixed-header padding bit.
func PutRTPPacket(dst []byte, header RTPHeader, payload []byte, paddingSize uint8) (int, error) {
	size, err := RTPPacketSize(header, payload, paddingSize)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTPShortBuffer
	}
	header.Padding = paddingSize != 0
	header.PaddingSize = paddingSize
	n, err := PutRTPHeader(dst, header)
	if err != nil {
		return 0, err
	}
	n += copy(dst[n:], payload)
	if paddingSize != 0 {
		clear(dst[n : n+int(paddingSize)-1])
		dst[n+int(paddingSize)-1] = paddingSize
		n += int(paddingSize)
	}
	return n, nil
}

// ParseRTPHeader parses the RTP header at the front of src. The returned byte
// count is the payload offset. Use ParseRTPPacket when RTP padding must be
// removed and validated.
func ParseRTPHeader(src []byte) (RTPHeader, int, error) {
	if len(src) < RTPHeaderMinSize {
		return RTPHeader{}, 0, ErrRTPShortPayload
	}
	if src[0]>>6 != RTPVersion {
		return RTPHeader{}, 0, ErrRTPInvalidHeader
	}
	header := RTPHeader{
		Padding:        src[0]&0x20 != 0,
		Marker:         src[1]&0x80 != 0,
		PayloadType:    src[1] & 0x7f,
		SequenceNumber: binary.BigEndian.Uint16(src[2:4]),
		Timestamp:      binary.BigEndian.Uint32(src[4:8]),
		SSRC:           binary.BigEndian.Uint32(src[8:12]),
		CSRCCount:      src[0] & 0x0f,
	}
	off := RTPHeaderMinSize
	if len(src) < off+int(header.CSRCCount)*4 {
		return RTPHeader{}, 0, ErrRTPShortPayload
	}
	for i := uint8(0); i < header.CSRCCount; i++ {
		header.CSRC[i] = binary.BigEndian.Uint32(src[off : off+4])
		off += 4
	}
	if src[0]&0x10 != 0 {
		if len(src) < off+4 {
			return RTPHeader{}, 0, ErrRTPShortPayload
		}
		header.ExtensionProfile = binary.BigEndian.Uint16(src[off : off+2])
		words := binary.BigEndian.Uint16(src[off+2 : off+4])
		off += 4
		extLen := int(words) * 4
		if len(src) < off+extLen {
			return RTPHeader{}, 0, ErrRTPShortPayload
		}
		header.ExtensionPayload = src[off : off+extLen]
		off += extLen
	}
	return header, off, nil
}

// ParseRTPPacket parses one RTP packet. The payload aliases src and excludes RTP
// padding bytes.
func ParseRTPPacket(src []byte) (RTPPacket, error) {
	header, off, err := ParseRTPHeader(src)
	if err != nil {
		return RTPPacket{}, err
	}
	payloadEnd := len(src)
	if header.Padding {
		if off >= len(src) {
			return RTPPacket{}, ErrRTPInvalidHeader
		}
		paddingSize := int(src[len(src)-1])
		if paddingSize == 0 || paddingSize > len(src)-off {
			return RTPPacket{}, ErrRTPInvalidHeader
		}
		header.PaddingSize = uint8(paddingSize)
		payloadEnd -= paddingSize
	}
	return RTPPacket{Header: header, Payload: src[off:payloadEnd]}, nil
}

func rtpPaddedExtensionPayloadLen(n int) (int, error) {
	if n < 0 || n > int(^uint(0)>>1)-3 {
		return 0, ErrRTPInvalidHeaderExtension
	}
	padded := (n + 3) &^ 3
	if padded/4 > 0xffff {
		return 0, ErrRTPInvalidHeaderExtension
	}
	return padded, nil
}

const (
	rtpHeaderExtensionProfileOneByte = 1
	rtpHeaderExtensionProfileTwoByte = 2
)

func rtpHeaderExtensionProfileKind(profile uint16) (int, error) {
	switch {
	case profile == RTPExtensionProfileOneByte:
		return rtpHeaderExtensionProfileOneByte, nil
	case profile&0xfff0 == RTPExtensionProfileTwoByte:
		return rtpHeaderExtensionProfileTwoByte, nil
	default:
		return 0, ErrRTPInvalidHeaderExtension
	}
}

func rtpHeaderExtensionElementSize(kind int, id uint8, payloadLen int) (int, error) {
	if payloadLen < 0 {
		return 0, ErrRTPInvalidHeaderExtension
	}
	switch kind {
	case rtpHeaderExtensionProfileOneByte:
		if id == 0 || id == 15 || payloadLen == 0 || payloadLen > 16 {
			return 0, ErrRTPInvalidHeaderExtension
		}
		return 1 + payloadLen, nil
	case rtpHeaderExtensionProfileTwoByte:
		if id == 0 || payloadLen > 255 {
			return 0, ErrRTPInvalidHeaderExtension
		}
		return 2 + payloadLen, nil
	default:
		return 0, ErrRTPInvalidHeaderExtension
	}
}
