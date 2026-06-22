package goav1

import (
	"encoding/binary"
	"errors"
)

const (
	// RTPCoordinationOfVideoOrientationHeaderExtensionSize is the payload size,
	// in bytes, of the WebRTC CVO RTP header extension.
	RTPCoordinationOfVideoOrientationHeaderExtensionSize = 1
	// RTPPlayoutDelayHeaderExtensionSize is the payload size, in bytes, of the
	// WebRTC playout-delay RTP header extension.
	RTPPlayoutDelayHeaderExtensionSize = 3
	// RTPTransportWideCCHeaderExtensionSize is the payload size, in bytes, of
	// WebRTC's transport-wide congestion-control RTP header extension.
	RTPTransportWideCCHeaderExtensionSize = 2
	// RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest is the
	// payload size, in bytes, of WebRTC's transport-wide-cc-02 RTP header
	// extension when immediate feedback is not requested.
	RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest = 2
	// RTPTransportWideCC02HeaderExtensionSize is the payload size, in bytes, of
	// WebRTC's transport-wide-cc-02 RTP header extension when immediate
	// feedback is requested.
	RTPTransportWideCC02HeaderExtensionSize = 4
	// RTPAbsoluteSendTimeHeaderExtensionSize is the payload size, in bytes, of
	// WebRTC's absolute-send-time RTP header extension.
	RTPAbsoluteSendTimeHeaderExtensionSize = 3
	// RTPVideoContentTypeHeaderExtensionSize is the payload size, in bytes, of
	// the WebRTC video-content-type RTP header extension.
	RTPVideoContentTypeHeaderExtensionSize = 1
	// RTPVideoTimingHeaderExtensionSize is the payload size, in bytes, of the
	// WebRTC video-timing RTP header extension.
	RTPVideoTimingHeaderExtensionSize = 13
	// RTPPlayoutDelayMaxMilliseconds is the largest playout delay that fits the
	// WebRTC playout-delay RTP header extension.
	RTPPlayoutDelayMaxMilliseconds = 40950
	// RTPTransportWideCC02MaxFeedbackSequenceCount is the largest feedback
	// packet-count request that fits transport-wide-cc-02.
	RTPTransportWideCC02MaxFeedbackSequenceCount = 1<<15 - 1
	// RTPAbsoluteSendTimeMaxValue is the largest raw 24-bit absolute-send-time
	// value accepted by PutRTPAbsoluteSendTimeHeaderExtension.
	RTPAbsoluteSendTimeMaxValue = 1<<24 - 1

	RTPVideoContentTypeUnspecified RTPVideoContentType = 0
	RTPVideoContentTypeScreenshare RTPVideoContentType = 1

	RTPVideoTimingFlagTriggeredByTimer     uint8 = 1 << 0
	RTPVideoTimingFlagFrameLargerThanKnown uint8 = 1 << 1
)

// ErrRTPInvalidHeaderExtension is returned when a fixed-size RTP header
// extension payload has an invalid length or raw value.
var ErrRTPInvalidHeaderExtension = errors.New("goav1: invalid RTP header extension")

// RTPCoordinationOfVideoOrientation is the WebRTC CVO RTP header-extension
// payload interpreted as camera/front-facing state, horizontal flip state, and
// clockwise rotation.
type RTPCoordinationOfVideoOrientation struct {
	Camera   bool
	Flip     bool
	Rotation uint16
}

// RTPPlayoutDelay is the WebRTC playout-delay RTP header-extension payload in
// milliseconds. Values must be multiples of 10 ms on write.
type RTPPlayoutDelay struct {
	MinDelayMs int
	MaxDelayMs int
}

// RTPTransportWideCC02 is WebRTC's transport-wide-cc-02 RTP header-extension
// payload. FeedbackRequest selects the optional 4-byte form that asks the
// receiver to send immediate transport feedback covering FeedbackSequenceCount
// packets including the current packet.
type RTPTransportWideCC02 struct {
	SequenceNumber        uint16
	FeedbackRequest       bool
	IncludeTimestamps     bool
	FeedbackSequenceCount uint16
}

// RTPVideoContentType is the WebRTC video-content-type RTP header-extension
// payload value.
type RTPVideoContentType uint8

// RTPVideoTiming is the WebRTC video-timing RTP header-extension payload. All
// deltas are milliseconds relative to the RTP timestamp of the packet carrying
// the extension.
type RTPVideoTiming struct {
	Flags                        uint8
	EncodeStartDeltaMs           uint16
	EncodeFinishDeltaMs          uint16
	PacketizationCompleteDeltaMs uint16
	PacerExitDeltaMs             uint16
	NetworkTimestampDeltaMs      uint16
	NetworkTimestamp2DeltaMs     uint16
}

// ParseRTPCoordinationOfVideoOrientationHeaderExtension parses a WebRTC CVO
// RTP header-extension element payload. The RTP extension element header is
// not part of src.
func ParseRTPCoordinationOfVideoOrientationHeaderExtension(src []byte) (RTPCoordinationOfVideoOrientation, error) {
	if len(src) < RTPCoordinationOfVideoOrientationHeaderExtensionSize {
		return RTPCoordinationOfVideoOrientation{}, ErrRTPShortBuffer
	}
	if len(src) != RTPCoordinationOfVideoOrientationHeaderExtensionSize {
		return RTPCoordinationOfVideoOrientation{}, ErrRTPInvalidHeaderExtension
	}
	return RTPCoordinationOfVideoOrientation{
		Camera:   src[0]&0x08 != 0,
		Flip:     src[0]&0x04 != 0,
		Rotation: uint16(src[0]&0x03) * 90,
	}, nil
}

// PutRTPCoordinationOfVideoOrientationHeaderExtension writes a WebRTC CVO RTP
// header-extension element payload. The RTP extension element header is not
// written.
func PutRTPCoordinationOfVideoOrientationHeaderExtension(dst []byte, orientation RTPCoordinationOfVideoOrientation) (int, error) {
	if err := ValidateRTPCoordinationOfVideoOrientation(orientation); err != nil {
		return 0, err
	}
	if len(dst) < RTPCoordinationOfVideoOrientationHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	value := byte(orientation.Rotation / 90)
	if orientation.Camera {
		value |= 0x08
	}
	if orientation.Flip {
		value |= 0x04
	}
	dst[0] = value
	return RTPCoordinationOfVideoOrientationHeaderExtensionSize, nil
}

func ValidateRTPCoordinationOfVideoOrientation(orientation RTPCoordinationOfVideoOrientation) error {
	switch orientation.Rotation {
	case 0, 90, 180, 270:
		return nil
	default:
		return ErrRTPInvalidHeaderExtension
	}
}

// ParseRTPPlayoutDelayHeaderExtension parses a WebRTC playout-delay RTP
// header-extension element payload. The RTP extension element header is not
// part of src.
func ParseRTPPlayoutDelayHeaderExtension(src []byte) (RTPPlayoutDelay, error) {
	if len(src) < RTPPlayoutDelayHeaderExtensionSize {
		return RTPPlayoutDelay{}, ErrRTPShortBuffer
	}
	if len(src) != RTPPlayoutDelayHeaderExtensionSize {
		return RTPPlayoutDelay{}, ErrRTPInvalidHeaderExtension
	}
	minUnits := uint16(src[0])<<4 | uint16(src[1]>>4)
	maxUnits := uint16(src[1]&0x0f)<<8 | uint16(src[2])
	delay := RTPPlayoutDelay{
		MinDelayMs: int(minUnits) * 10,
		MaxDelayMs: int(maxUnits) * 10,
	}
	if err := ValidateRTPPlayoutDelay(delay); err != nil {
		return RTPPlayoutDelay{}, err
	}
	return delay, nil
}

// PutRTPPlayoutDelayHeaderExtension writes a WebRTC playout-delay RTP header-
// extension element payload. The RTP extension element header is not written.
func PutRTPPlayoutDelayHeaderExtension(dst []byte, delay RTPPlayoutDelay) (int, error) {
	if err := ValidateRTPPlayoutDelay(delay); err != nil {
		return 0, err
	}
	if len(dst) < RTPPlayoutDelayHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	minUnits := uint16(delay.MinDelayMs / 10)
	maxUnits := uint16(delay.MaxDelayMs / 10)
	dst[0] = byte(minUnits >> 4)
	dst[1] = byte(minUnits<<4) | byte(maxUnits>>8)
	dst[2] = byte(maxUnits)
	return RTPPlayoutDelayHeaderExtensionSize, nil
}

func ValidateRTPPlayoutDelay(delay RTPPlayoutDelay) error {
	if delay.MinDelayMs < 0 || delay.MaxDelayMs < 0 ||
		delay.MinDelayMs > delay.MaxDelayMs ||
		delay.MaxDelayMs > RTPPlayoutDelayMaxMilliseconds ||
		delay.MinDelayMs%10 != 0 || delay.MaxDelayMs%10 != 0 {
		return ErrRTPInvalidHeaderExtension
	}
	return nil
}

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

// RTPTransportWideCC02Size returns the payload size needed to write cc. The
// RTP extension element header is not counted.
func RTPTransportWideCC02Size(cc RTPTransportWideCC02) (int, error) {
	if err := ValidateRTPTransportWideCC02(cc); err != nil {
		return 0, err
	}
	if cc.FeedbackRequest {
		return RTPTransportWideCC02HeaderExtensionSize, nil
	}
	return RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest, nil
}

// ParseRTPTransportWideCC02HeaderExtension parses WebRTC's transport-wide-cc-02
// RTP header-extension element payload. The RTP extension element header is not
// part of src.
func ParseRTPTransportWideCC02HeaderExtension(src []byte) (RTPTransportWideCC02, error) {
	if len(src) < RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest {
		return RTPTransportWideCC02{}, ErrRTPShortBuffer
	}
	if len(src) != RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest &&
		len(src) != RTPTransportWideCC02HeaderExtensionSize {
		return RTPTransportWideCC02{}, ErrRTPInvalidHeaderExtension
	}
	cc := RTPTransportWideCC02{
		SequenceNumber: binary.BigEndian.Uint16(src[:2]),
	}
	if len(src) == RTPTransportWideCC02HeaderExtensionSize {
		raw := binary.BigEndian.Uint16(src[2:4])
		sequenceCount := raw & RTPTransportWideCC02MaxFeedbackSequenceCount
		if sequenceCount != 0 {
			cc.FeedbackRequest = true
			cc.IncludeTimestamps = raw&(1<<15) != 0
			cc.FeedbackSequenceCount = sequenceCount
		}
	}
	return cc, nil
}

// PutRTPTransportWideCC02HeaderExtension writes WebRTC's transport-wide-cc-02
// RTP header-extension element payload. The RTP extension element header is
// not written.
func PutRTPTransportWideCC02HeaderExtension(dst []byte, cc RTPTransportWideCC02) (int, error) {
	size, err := RTPTransportWideCC02Size(cc)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTPShortBuffer
	}
	binary.BigEndian.PutUint16(dst[:2], cc.SequenceNumber)
	if cc.FeedbackRequest {
		raw := cc.FeedbackSequenceCount
		if cc.IncludeTimestamps {
			raw |= 1 << 15
		}
		binary.BigEndian.PutUint16(dst[2:4], raw)
	}
	return size, nil
}

func ValidateRTPTransportWideCC02(cc RTPTransportWideCC02) error {
	if cc.FeedbackRequest {
		if cc.FeedbackSequenceCount == 0 ||
			cc.FeedbackSequenceCount > RTPTransportWideCC02MaxFeedbackSequenceCount {
			return ErrRTPInvalidHeaderExtension
		}
		return nil
	}
	if cc.IncludeTimestamps || cc.FeedbackSequenceCount != 0 {
		return ErrRTPInvalidHeaderExtension
	}
	return nil
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

// ParseRTPVideoContentTypeHeaderExtension parses a WebRTC video-content-type
// RTP header-extension element payload. The RTP extension element header is not
// part of src.
func ParseRTPVideoContentTypeHeaderExtension(src []byte) (RTPVideoContentType, error) {
	if len(src) < RTPVideoContentTypeHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	if len(src) != RTPVideoContentTypeHeaderExtensionSize {
		return 0, ErrRTPInvalidHeaderExtension
	}
	content := RTPVideoContentType(src[0])
	if err := ValidateRTPVideoContentType(content); err != nil {
		return 0, err
	}
	return content, nil
}

// PutRTPVideoContentTypeHeaderExtension writes a WebRTC video-content-type RTP
// header-extension element payload. The RTP extension element header is not
// written.
func PutRTPVideoContentTypeHeaderExtension(dst []byte, content RTPVideoContentType) (int, error) {
	if err := ValidateRTPVideoContentType(content); err != nil {
		return 0, err
	}
	if len(dst) < RTPVideoContentTypeHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	dst[0] = byte(content)
	return RTPVideoContentTypeHeaderExtensionSize, nil
}

func ValidateRTPVideoContentType(content RTPVideoContentType) error {
	switch content {
	case RTPVideoContentTypeUnspecified, RTPVideoContentTypeScreenshare:
		return nil
	default:
		return ErrRTPInvalidHeaderExtension
	}
}

// ParseRTPVideoTimingHeaderExtension parses a WebRTC video-timing RTP header-
// extension element payload. The RTP extension element header is not part of
// src.
func ParseRTPVideoTimingHeaderExtension(src []byte) (RTPVideoTiming, error) {
	if len(src) < RTPVideoTimingHeaderExtensionSize {
		return RTPVideoTiming{}, ErrRTPShortBuffer
	}
	if len(src) != RTPVideoTimingHeaderExtensionSize {
		return RTPVideoTiming{}, ErrRTPInvalidHeaderExtension
	}
	return RTPVideoTiming{
		Flags:                        src[0] & (RTPVideoTimingFlagTriggeredByTimer | RTPVideoTimingFlagFrameLargerThanKnown),
		EncodeStartDeltaMs:           binary.BigEndian.Uint16(src[1:3]),
		EncodeFinishDeltaMs:          binary.BigEndian.Uint16(src[3:5]),
		PacketizationCompleteDeltaMs: binary.BigEndian.Uint16(src[5:7]),
		PacerExitDeltaMs:             binary.BigEndian.Uint16(src[7:9]),
		NetworkTimestampDeltaMs:      binary.BigEndian.Uint16(src[9:11]),
		NetworkTimestamp2DeltaMs:     binary.BigEndian.Uint16(src[11:13]),
	}, nil
}

// PutRTPVideoTimingHeaderExtension writes a WebRTC video-timing RTP header-
// extension element payload. The RTP extension element header is not written.
func PutRTPVideoTimingHeaderExtension(dst []byte, timing RTPVideoTiming) (int, error) {
	if err := ValidateRTPVideoTiming(timing); err != nil {
		return 0, err
	}
	if len(dst) < RTPVideoTimingHeaderExtensionSize {
		return 0, ErrRTPShortBuffer
	}
	dst[0] = timing.Flags
	binary.BigEndian.PutUint16(dst[1:3], timing.EncodeStartDeltaMs)
	binary.BigEndian.PutUint16(dst[3:5], timing.EncodeFinishDeltaMs)
	binary.BigEndian.PutUint16(dst[5:7], timing.PacketizationCompleteDeltaMs)
	binary.BigEndian.PutUint16(dst[7:9], timing.PacerExitDeltaMs)
	binary.BigEndian.PutUint16(dst[9:11], timing.NetworkTimestampDeltaMs)
	binary.BigEndian.PutUint16(dst[11:13], timing.NetworkTimestamp2DeltaMs)
	return RTPVideoTimingHeaderExtensionSize, nil
}

func ValidateRTPVideoTiming(timing RTPVideoTiming) error {
	if timing.Flags&^(RTPVideoTimingFlagTriggeredByTimer|RTPVideoTimingFlagFrameLargerThanKnown) != 0 {
		return ErrRTPInvalidHeaderExtension
	}
	return nil
}
