package goav1

import (
	"encoding/binary"
	"errors"
)

const (
	// RTPTransportWideCCHeaderExtensionSize is the payload size, in bytes, of
	// WebRTC's transport-wide congestion-control RTP header extension.
	RTPTransportWideCCHeaderExtensionSize = 2
	// RTPAbsoluteSendTimeHeaderExtensionSize is the payload size, in bytes, of
	// WebRTC's absolute-send-time RTP header extension.
	RTPAbsoluteSendTimeHeaderExtensionSize = 3
	// RTPAbsoluteSendTimeMaxValue is the largest raw 24-bit absolute-send-time
	// value accepted by PutRTPAbsoluteSendTimeHeaderExtension.
	RTPAbsoluteSendTimeMaxValue = 1<<24 - 1
)

// ErrRTPInvalidHeaderExtension is returned when a fixed-size RTP header
// extension payload has an invalid length or raw value.
var ErrRTPInvalidHeaderExtension = errors.New("goav1: invalid RTP header extension")

// ParseRTPTransportWideCCHeaderExtension parses a transport-wide congestion-
// control RTP header-extension element payload. The RTP extension element
// header is not part of src.
func ParseRTPTransportWideCCHeaderExtension(src []byte) (uint16, error) {
	if len(src) < RTPTransportWideCCHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	if len(src) != RTPTransportWideCCHeaderExtensionSize {
		return 0, ErrRTPInvalidHeaderExtension
	}
	return binary.BigEndian.Uint16(src), nil
}

// PutRTPTransportWideCCHeaderExtension writes a transport-wide congestion-
// control RTP header-extension element payload. The RTP extension element
// header is not written.
func PutRTPTransportWideCCHeaderExtension(dst []byte, sequenceNumber uint16) (int, error) {
	if len(dst) < RTPTransportWideCCHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	binary.BigEndian.PutUint16(dst[:RTPTransportWideCCHeaderExtensionSize], sequenceNumber)
	return RTPTransportWideCCHeaderExtensionSize, nil
}

// ParseRTPAbsoluteSendTimeHeaderExtension parses a raw 24-bit absolute-send-
// time RTP header-extension element payload. The RTP extension element header
// is not part of src.
func ParseRTPAbsoluteSendTimeHeaderExtension(src []byte) (uint32, error) {
	if len(src) < RTPAbsoluteSendTimeHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	if len(src) != RTPAbsoluteSendTimeHeaderExtensionSize {
		return 0, ErrRTPInvalidHeaderExtension
	}
	return uint32(src[0])<<16 | uint32(src[1])<<8 | uint32(src[2]), nil
}

// PutRTPAbsoluteSendTimeHeaderExtension writes a raw 24-bit absolute-send-time
// RTP header-extension element payload. The RTP extension element header is
// not written.
func PutRTPAbsoluteSendTimeHeaderExtension(dst []byte, timestamp uint32) (int, error) {
	if timestamp > RTPAbsoluteSendTimeMaxValue {
		return 0, ErrRTPInvalidHeaderExtension
	}
	if len(dst) < RTPAbsoluteSendTimeHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	dst[0] = byte(timestamp >> 16)
	dst[1] = byte(timestamp >> 8)
	dst[2] = byte(timestamp)
	return RTPAbsoluteSendTimeHeaderExtensionSize, nil
}
