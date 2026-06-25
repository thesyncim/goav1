package goav1

import (
	"encoding/binary"
	"errors"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
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
	// RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset is the
	// payload size, in bytes, of WebRTC's absolute-capture-time RTP header
	// extension without the optional estimated capture clock offset.
	RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset = 8
	// RTPAbsoluteCaptureTimeHeaderExtensionSize is the payload size, in bytes,
	// of WebRTC's absolute-capture-time RTP header extension with the optional
	// estimated capture clock offset.
	RTPAbsoluteCaptureTimeHeaderExtensionSize = 16
	// RTPVideoContentTypeHeaderExtensionSize is the payload size, in bytes, of
	// the WebRTC video-content-type RTP header extension.
	RTPVideoContentTypeHeaderExtensionSize = 1
	// RTPVideoTimingHeaderExtensionSize is the payload size, in bytes, of the
	// WebRTC video-timing RTP header extension.
	RTPVideoTimingHeaderExtensionSize = 13
	// RTPVideoTimingHeaderExtensionSizeWithoutFlags is the legacy payload size,
	// in bytes, of WebRTC's video-timing RTP header extension without the flags
	// byte. Writes always use RTPVideoTimingHeaderExtensionSize.
	RTPVideoTimingHeaderExtensionSizeWithoutFlags = 12
	// RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata is the payload size,
	// in bytes, of WebRTC's color-space RTP header extension without HDR
	// metadata.
	RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata = 4
	// RTPColorSpaceHeaderExtensionSize is the payload size, in bytes, of
	// WebRTC's color-space RTP header extension with HDR metadata.
	RTPColorSpaceHeaderExtensionSize = 28
	// RTPVideoLayersAllocationMaxHeaderExtensionSize is the largest payload
	// size, in bytes, accepted by PutRTPVideoLayersAllocationHeaderExtension.
	RTPVideoLayersAllocationMaxHeaderExtensionSize = 407
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

const (
	RTPColorSpacePrimaryBT709       RTPColorSpacePrimary = 1
	RTPColorSpacePrimaryUnspecified RTPColorSpacePrimary = 2
	RTPColorSpacePrimaryBT470M      RTPColorSpacePrimary = 4
	RTPColorSpacePrimaryBT470BG     RTPColorSpacePrimary = 5
	RTPColorSpacePrimarySMPTE170M   RTPColorSpacePrimary = 6
	RTPColorSpacePrimarySMPTE240M   RTPColorSpacePrimary = 7
	RTPColorSpacePrimaryFilm        RTPColorSpacePrimary = 8
	RTPColorSpacePrimaryBT2020      RTPColorSpacePrimary = 9
	RTPColorSpacePrimarySMPTEST428  RTPColorSpacePrimary = 10
	RTPColorSpacePrimarySMPTEST431  RTPColorSpacePrimary = 11
	RTPColorSpacePrimarySMPTEST432  RTPColorSpacePrimary = 12
	RTPColorSpacePrimaryJEDECP22    RTPColorSpacePrimary = 22

	RTPColorSpaceTransferBT709       RTPColorSpaceTransfer = 1
	RTPColorSpaceTransferUnspecified RTPColorSpaceTransfer = 2
	RTPColorSpaceTransferGamma22     RTPColorSpaceTransfer = 4
	RTPColorSpaceTransferGamma28     RTPColorSpaceTransfer = 5
	RTPColorSpaceTransferSMPTE170M   RTPColorSpaceTransfer = 6
	RTPColorSpaceTransferSMPTE240M   RTPColorSpaceTransfer = 7
	RTPColorSpaceTransferLinear      RTPColorSpaceTransfer = 8
	RTPColorSpaceTransferLog         RTPColorSpaceTransfer = 9
	RTPColorSpaceTransferLogSqrt     RTPColorSpaceTransfer = 10
	RTPColorSpaceTransferIEC6196624  RTPColorSpaceTransfer = 11
	RTPColorSpaceTransferBT1361ECG   RTPColorSpaceTransfer = 12
	RTPColorSpaceTransferIEC6196621  RTPColorSpaceTransfer = 13
	RTPColorSpaceTransferBT202010    RTPColorSpaceTransfer = 14
	RTPColorSpaceTransferBT202012    RTPColorSpaceTransfer = 15
	RTPColorSpaceTransferSMPTEST2084 RTPColorSpaceTransfer = 16
	RTPColorSpaceTransferSMPTEST428  RTPColorSpaceTransfer = 17
	RTPColorSpaceTransferARIBSTDB67  RTPColorSpaceTransfer = 18

	RTPColorSpaceMatrixRGB         RTPColorSpaceMatrix = 0
	RTPColorSpaceMatrixBT709       RTPColorSpaceMatrix = 1
	RTPColorSpaceMatrixUnspecified RTPColorSpaceMatrix = 2
	RTPColorSpaceMatrixFCC         RTPColorSpaceMatrix = 4
	RTPColorSpaceMatrixBT470BG     RTPColorSpaceMatrix = 5
	RTPColorSpaceMatrixSMPTE170M   RTPColorSpaceMatrix = 6
	RTPColorSpaceMatrixSMPTE240M   RTPColorSpaceMatrix = 7
	RTPColorSpaceMatrixYCgCo       RTPColorSpaceMatrix = 8
	RTPColorSpaceMatrixBT2020NCL   RTPColorSpaceMatrix = 9
	RTPColorSpaceMatrixBT2020CL    RTPColorSpaceMatrix = 10
	RTPColorSpaceMatrixSMPTE2085   RTPColorSpaceMatrix = 11
	RTPColorSpaceMatrixCDNCLS      RTPColorSpaceMatrix = 12
	RTPColorSpaceMatrixCDCLS       RTPColorSpaceMatrix = 13
	RTPColorSpaceMatrixBT2100ICTCP RTPColorSpaceMatrix = 14

	RTPColorSpaceRangeInvalid RTPColorSpaceRange = 0
	RTPColorSpaceRangeLimited RTPColorSpaceRange = 1
	RTPColorSpaceRangeFull    RTPColorSpaceRange = 2
	RTPColorSpaceRangeDerived RTPColorSpaceRange = 3

	RTPColorSpaceChromaSitingUnspecified RTPColorSpaceChromaSiting = 0
	RTPColorSpaceChromaSitingCollocated  RTPColorSpaceChromaSiting = 1
	RTPColorSpaceChromaSitingHalf        RTPColorSpaceChromaSiting = 2
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

// RTPAbsoluteCaptureTime is WebRTC's absolute-capture-time RTP header-
// extension payload. AbsoluteCaptureTimestamp uses the NTP timestamp format.
// The optional EstimatedCaptureClockOffset is encoded when
// EstimatedCaptureClockOffsetPresent is true.
type RTPAbsoluteCaptureTime struct {
	AbsoluteCaptureTimestamp           uint64
	EstimatedCaptureClockOffsetPresent bool
	EstimatedCaptureClockOffset        int64
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

// RTPColorSpace enum values follow WebRTC's T-REC H.273 color-space RTP
// header-extension representation.
type RTPColorSpacePrimary uint8
type RTPColorSpaceTransfer uint8
type RTPColorSpaceMatrix uint8
type RTPColorSpaceRange uint8
type RTPColorSpaceChromaSiting uint8

// RTPColorSpaceChromaticity stores x/y chromaticity coordinates in WebRTC's
// raw RTP color-space scale, where one unit is 1/50000.
type RTPColorSpaceChromaticity struct {
	X uint16
	Y uint16
}

// RTPColorSpaceHDRMetadata stores WebRTC color-space HDR metadata in raw RTP
// extension units. LuminanceMax is cd/m2, LuminanceMin is cd/m2 times 10000,
// chromaticities are times 50000, and light levels are nits.
type RTPColorSpaceHDRMetadata struct {
	LuminanceMax              uint16
	LuminanceMin              uint16
	PrimaryR                  RTPColorSpaceChromaticity
	PrimaryG                  RTPColorSpaceChromaticity
	PrimaryB                  RTPColorSpaceChromaticity
	WhitePoint                RTPColorSpaceChromaticity
	MaxContentLightLevel      uint16
	MaxFrameAverageLightLevel uint16
}

// RTPColorSpace is WebRTC's color-space RTP header-extension payload. Set
// HDRMetadataPresent to write or require the 28-byte form with HDR metadata;
// otherwise the 4-byte color-space-only form is used.
type RTPColorSpace struct {
	Primaries              RTPColorSpacePrimary
	Transfer               RTPColorSpaceTransfer
	Matrix                 RTPColorSpaceMatrix
	Range                  RTPColorSpaceRange
	ChromaSitingHorizontal RTPColorSpaceChromaSiting
	ChromaSitingVertical   RTPColorSpaceChromaSiting
	HDRMetadataPresent     bool
	HDRMetadata            RTPColorSpaceHDRMetadata
}

// RTPVideoLayersAllocationLayer describes one active spatial layer in
// WebRTC's video-layers-allocation RTP header-extension payload. Target
// bitrates are cumulative required bitrates in kbps for temporal IDs in
// ascending order.
type RTPVideoLayersAllocationLayer struct {
	RTPStreamID        int
	SpatialID          int
	TargetBitratesKbps []uint32
	Width              int
	Height             int
	Framerate          int
}

// RTPVideoLayersAllocation is WebRTC's video-layers-allocation RTP header-
// extension payload. ActiveSpatialLayers are written and parsed in
// RTPStreamID, SpatialID ascending order. When HasResolutionAndFramerate is
// true, each active layer also carries width, height, and max framerate.
type RTPVideoLayersAllocation struct {
	RTPStreamID               int
	RTPStreamCount            int
	ActiveSpatialLayers       []RTPVideoLayersAllocationLayer
	HasResolutionAndFramerate bool
}

type rtpVideoLayersAllocationLayout struct {
	spatialMasks [4]uint8
	layerIndex   [4][4]int
	commonMask   uint8
	layerCount   int
	size         int
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

// RTPAbsoluteCaptureTimeSize returns the payload size needed to write capture.
// The RTP extension element header is not counted.
func RTPAbsoluteCaptureTimeSize(capture RTPAbsoluteCaptureTime) (int, error) {
	if err := ValidateRTPAbsoluteCaptureTime(capture); err != nil {
		return 0, err
	}
	if capture.EstimatedCaptureClockOffsetPresent {
		return RTPAbsoluteCaptureTimeHeaderExtensionSize, nil
	}
	return RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset, nil
}

// ParseRTPAbsoluteCaptureTimeHeaderExtension parses WebRTC's absolute-capture-
// time RTP header-extension element payload. The RTP extension element header
// is not part of src.
func ParseRTPAbsoluteCaptureTimeHeaderExtension(src []byte) (RTPAbsoluteCaptureTime, error) {
	if len(src) < RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset {
		return RTPAbsoluteCaptureTime{}, ErrRTPShortBuffer
	}
	if len(src) != RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset &&
		len(src) != RTPAbsoluteCaptureTimeHeaderExtensionSize {
		return RTPAbsoluteCaptureTime{}, ErrRTPInvalidHeaderExtension
	}
	capture := RTPAbsoluteCaptureTime{
		AbsoluteCaptureTimestamp: binary.BigEndian.Uint64(src[:8]),
	}
	if len(src) == RTPAbsoluteCaptureTimeHeaderExtensionSize {
		capture.EstimatedCaptureClockOffsetPresent = true
		capture.EstimatedCaptureClockOffset = int64(binary.BigEndian.Uint64(src[8:16]))
	}
	return capture, nil
}

// PutRTPAbsoluteCaptureTimeHeaderExtension writes WebRTC's absolute-capture-
// time RTP header-extension element payload. The RTP extension element header
// is not written.
func PutRTPAbsoluteCaptureTimeHeaderExtension(dst []byte, capture RTPAbsoluteCaptureTime) (int, error) {
	size, err := RTPAbsoluteCaptureTimeSize(capture)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTPShortBuffer
	}
	binary.BigEndian.PutUint64(dst[:8], capture.AbsoluteCaptureTimestamp)
	if capture.EstimatedCaptureClockOffsetPresent {
		binary.BigEndian.PutUint64(dst[8:16], uint64(capture.EstimatedCaptureClockOffset))
	}
	return size, nil
}

func ValidateRTPAbsoluteCaptureTime(capture RTPAbsoluteCaptureTime) error {
	if !capture.EstimatedCaptureClockOffsetPresent && capture.EstimatedCaptureClockOffset != 0 {
		return ErrRTPInvalidHeaderExtension
	}
	return nil
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
	if len(src) < RTPVideoTimingHeaderExtensionSizeWithoutFlags {
		return RTPVideoTiming{}, ErrRTPShortBuffer
	}
	flags := uint8(0)
	offset := 0
	switch len(src) {
	case RTPVideoTimingHeaderExtensionSizeWithoutFlags:
	case RTPVideoTimingHeaderExtensionSize:
		flags = src[0] & (RTPVideoTimingFlagTriggeredByTimer | RTPVideoTimingFlagFrameLargerThanKnown)
		offset = 1
	default:
		return RTPVideoTiming{}, ErrRTPInvalidHeaderExtension
	}
	return RTPVideoTiming{
		Flags:                        flags,
		EncodeStartDeltaMs:           binary.BigEndian.Uint16(src[offset : offset+2]),
		EncodeFinishDeltaMs:          binary.BigEndian.Uint16(src[offset+2 : offset+4]),
		PacketizationCompleteDeltaMs: binary.BigEndian.Uint16(src[offset+4 : offset+6]),
		PacerExitDeltaMs:             binary.BigEndian.Uint16(src[offset+6 : offset+8]),
		NetworkTimestampDeltaMs:      binary.BigEndian.Uint16(src[offset+8 : offset+10]),
		NetworkTimestamp2DeltaMs:     binary.BigEndian.Uint16(src[offset+10 : offset+12]),
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

// RTPVideoLayersAllocationSize returns the payload size needed to write
// allocation. The RTP extension element header is not counted.
func RTPVideoLayersAllocationSize(allocation RTPVideoLayersAllocation) (int, error) {
	var layout rtpVideoLayersAllocationLayout
	if err := analyzeRTPVideoLayersAllocation(allocation, &layout); err != nil {
		return 0, err
	}
	return layout.size, nil
}

// ParseRTPVideoLayersAllocationHeaderExtension parses WebRTC's
// video-layers-allocation RTP header-extension element payload. The RTP
// extension element header is not part of src.
func ParseRTPVideoLayersAllocationHeaderExtension(src []byte) (RTPVideoLayersAllocation, error) {
	if len(src) == 0 {
		return RTPVideoLayersAllocation{}, ErrRTPShortBuffer
	}
	if len(src) == 1 && src[0] == 0 {
		return RTPVideoLayersAllocation{RTPStreamCount: 1}, nil
	}

	allocation := RTPVideoLayersAllocation{
		RTPStreamID:    int(src[0] >> 6),
		RTPStreamCount: int((src[0]>>4)&0x03) + 1,
	}
	if allocation.RTPStreamID >= allocation.RTPStreamCount {
		return RTPVideoLayersAllocation{}, ErrRTPInvalidHeaderExtension
	}

	var masks [4]uint8
	offset := 1
	commonMask := src[0] & 0x0f
	if commonMask != 0 {
		for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
			masks[streamID] = commonMask
		}
	} else {
		maskBytes := rtpVideoLayersAllocationSpatialMaskBytes(allocation.RTPStreamCount)
		if len(src)-offset < maskBytes {
			return RTPVideoLayersAllocation{}, ErrRTPShortBuffer
		}
		for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
			if streamID%2 == 0 {
				masks[streamID] = (src[offset+streamID/2] >> 4) & 0x0f
			} else {
				masks[streamID] = src[offset+streamID/2] & 0x0f
			}
		}
		offset += maskBytes
	}

	layerCount := 0
	for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
		for spatialID := 0; spatialID < 4; spatialID++ {
			if masks[streamID]&(1<<spatialID) != 0 {
				layerCount++
			}
		}
	}
	if layerCount == 0 {
		return RTPVideoLayersAllocation{}, ErrRTPInvalidHeaderExtension
	}

	tlBytes := rtpVideoLayersAllocationTemporalLayerBytes(layerCount)
	if len(src)-offset < tlBytes {
		return RTPVideoLayersAllocation{}, ErrRTPShortBuffer
	}
	allocation.ActiveSpatialLayers = make([]RTPVideoLayersAllocationLayer, 0, layerCount)
	layerOrdinal := 0
	for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
		for spatialID := 0; spatialID < 4; spatialID++ {
			if masks[streamID]&(1<<spatialID) == 0 {
				continue
			}
			tlByte := src[offset+layerOrdinal/4]
			shift := uint(2 * (3 - layerOrdinal%4))
			temporalLayers := int((tlByte>>shift)&0x03) + 1
			allocation.ActiveSpatialLayers = append(allocation.ActiveSpatialLayers, RTPVideoLayersAllocationLayer{
				RTPStreamID:        streamID,
				SpatialID:          spatialID,
				TargetBitratesKbps: make([]uint32, temporalLayers),
			})
			layerOrdinal++
		}
	}
	offset += tlBytes

	for i := range allocation.ActiveSpatialLayers {
		for j := range allocation.ActiveSpatialLayers[i].TargetBitratesKbps {
			value, n, err := bitstream.ReadLEB128(src[offset:])
			if err != nil {
				if errors.Is(err, bitstream.ErrShortLEB128) {
					return RTPVideoLayersAllocation{}, ErrRTPShortBuffer
				}
				return RTPVideoLayersAllocation{}, ErrRTPInvalidHeaderExtension
			}
			allocation.ActiveSpatialLayers[i].TargetBitratesKbps[j] = value
			offset += n
		}
	}

	remaining := len(src) - offset
	if remaining == 0 {
		return allocation, nil
	}
	resolutionBytes := layerCount * 5
	if remaining < resolutionBytes {
		return RTPVideoLayersAllocation{}, ErrRTPShortBuffer
	}
	if remaining != resolutionBytes {
		return RTPVideoLayersAllocation{}, ErrRTPInvalidHeaderExtension
	}
	allocation.HasResolutionAndFramerate = true
	for i := range allocation.ActiveSpatialLayers {
		allocation.ActiveSpatialLayers[i].Width = int(binary.BigEndian.Uint16(src[offset:offset+2])) + 1
		allocation.ActiveSpatialLayers[i].Height = int(binary.BigEndian.Uint16(src[offset+2:offset+4])) + 1
		allocation.ActiveSpatialLayers[i].Framerate = int(src[offset+4])
		offset += 5
	}
	return allocation, nil
}

// PutRTPVideoLayersAllocationHeaderExtension writes WebRTC's
// video-layers-allocation RTP header-extension element payload. The RTP
// extension element header is not written.
func PutRTPVideoLayersAllocationHeaderExtension(dst []byte, allocation RTPVideoLayersAllocation) (int, error) {
	var layout rtpVideoLayersAllocationLayout
	if err := analyzeRTPVideoLayersAllocation(allocation, &layout); err != nil {
		return 0, err
	}
	if len(dst) < layout.size {
		return 0, ErrRTPShortBuffer
	}
	if layout.layerCount == 0 {
		dst[0] = 0
		return 1, nil
	}

	offset := 0
	dst[offset] = byte(allocation.RTPStreamID<<6) | byte((allocation.RTPStreamCount-1)<<4) | layout.commonMask
	offset++
	if layout.commonMask == 0 {
		maskBytes := rtpVideoLayersAllocationSpatialMaskBytes(allocation.RTPStreamCount)
		for i := 0; i < maskBytes; i++ {
			dst[offset+i] = 0
		}
		for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
			if streamID%2 == 0 {
				dst[offset+streamID/2] |= layout.spatialMasks[streamID] << 4
			} else {
				dst[offset+streamID/2] |= layout.spatialMasks[streamID]
			}
		}
		offset += maskBytes
	}

	tlBytes := rtpVideoLayersAllocationTemporalLayerBytes(layout.layerCount)
	for i := 0; i < tlBytes; i++ {
		dst[offset+i] = 0
	}
	layerOrdinal := 0
	for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
		for spatialID := 0; spatialID < 4; spatialID++ {
			idx := layout.layerIndex[streamID][spatialID]
			if idx < 0 {
				continue
			}
			tlCount := len(allocation.ActiveSpatialLayers[idx].TargetBitratesKbps)
			shift := uint(2 * (3 - layerOrdinal%4))
			dst[offset+layerOrdinal/4] |= byte(tlCount-1) << shift
			layerOrdinal++
		}
	}
	offset += tlBytes

	for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
		for spatialID := 0; spatialID < 4; spatialID++ {
			idx := layout.layerIndex[streamID][spatialID]
			if idx < 0 {
				continue
			}
			layer := allocation.ActiveSpatialLayers[idx]
			for _, bitrate := range layer.TargetBitratesKbps {
				n, err := bitstream.PutLEB128(dst[offset:], bitrate)
				if err != nil {
					return 0, ErrRTPShortBuffer
				}
				offset += n
			}
		}
	}

	if allocation.HasResolutionAndFramerate {
		for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
			for spatialID := 0; spatialID < 4; spatialID++ {
				idx := layout.layerIndex[streamID][spatialID]
				if idx < 0 {
					continue
				}
				layer := allocation.ActiveSpatialLayers[idx]
				binary.BigEndian.PutUint16(dst[offset:offset+2], uint16(layer.Width-1))
				binary.BigEndian.PutUint16(dst[offset+2:offset+4], uint16(layer.Height-1))
				dst[offset+4] = byte(layer.Framerate)
				offset += 5
			}
		}
	}
	return layout.size, nil
}

func ValidateRTPVideoLayersAllocation(allocation RTPVideoLayersAllocation) error {
	var layout rtpVideoLayersAllocationLayout
	return analyzeRTPVideoLayersAllocation(allocation, &layout)
}

func analyzeRTPVideoLayersAllocation(allocation RTPVideoLayersAllocation, layout *rtpVideoLayersAllocationLayout) error {
	*layout = rtpVideoLayersAllocationLayout{}
	for streamID := range layout.layerIndex {
		for spatialID := range layout.layerIndex[streamID] {
			layout.layerIndex[streamID][spatialID] = -1
		}
	}

	if len(allocation.ActiveSpatialLayers) == 0 {
		if allocation.RTPStreamID != 0 ||
			allocation.RTPStreamCount < 0 ||
			allocation.RTPStreamCount > 1 ||
			allocation.HasResolutionAndFramerate {
			return ErrRTPInvalidHeaderExtension
		}
		layout.size = 1
		return nil
	}
	if allocation.RTPStreamCount <= 0 || allocation.RTPStreamCount > 4 ||
		allocation.RTPStreamID < 0 || allocation.RTPStreamID >= allocation.RTPStreamCount {
		return ErrRTPInvalidHeaderExtension
	}

	for i, layer := range allocation.ActiveSpatialLayers {
		if layer.RTPStreamID < 0 || layer.RTPStreamID >= allocation.RTPStreamCount ||
			layer.SpatialID < 0 || layer.SpatialID >= 4 {
			return ErrRTPInvalidHeaderExtension
		}
		if layout.layerIndex[layer.RTPStreamID][layer.SpatialID] >= 0 {
			return ErrRTPInvalidHeaderExtension
		}
		if len(layer.TargetBitratesKbps) == 0 || len(layer.TargetBitratesKbps) > 4 {
			return ErrRTPInvalidHeaderExtension
		}
		if allocation.HasResolutionAndFramerate {
			if layer.Width <= 0 || layer.Width > 1<<16 ||
				layer.Height <= 0 || layer.Height > 1<<16 ||
				layer.Framerate < 0 || layer.Framerate > 0xff {
				return ErrRTPInvalidHeaderExtension
			}
		} else if layer.Width != 0 || layer.Height != 0 || layer.Framerate != 0 {
			return ErrRTPInvalidHeaderExtension
		}
		layout.layerIndex[layer.RTPStreamID][layer.SpatialID] = i
		layout.spatialMasks[layer.RTPStreamID] |= 1 << layer.SpatialID
		layout.layerCount++
	}

	layout.commonMask = rtpVideoLayersAllocationCommonSpatialMask(layout.spatialMasks, allocation.RTPStreamCount)
	layout.size = 1
	if layout.commonMask == 0 {
		layout.size += rtpVideoLayersAllocationSpatialMaskBytes(allocation.RTPStreamCount)
	}
	layout.size += rtpVideoLayersAllocationTemporalLayerBytes(layout.layerCount)
	for streamID := 0; streamID < allocation.RTPStreamCount; streamID++ {
		for spatialID := 0; spatialID < 4; spatialID++ {
			idx := layout.layerIndex[streamID][spatialID]
			if idx < 0 {
				continue
			}
			for _, bitrate := range allocation.ActiveSpatialLayers[idx].TargetBitratesKbps {
				layout.size += bitstream.LEB128Len(bitrate)
			}
		}
	}
	if allocation.HasResolutionAndFramerate {
		layout.size += layout.layerCount * 5
	}
	return nil
}

func rtpVideoLayersAllocationCommonSpatialMask(masks [4]uint8, streamCount int) uint8 {
	common := masks[0]
	if common == 0 {
		return 0
	}
	for streamID := 1; streamID < streamCount; streamID++ {
		if masks[streamID] != common {
			return 0
		}
	}
	return common
}

func rtpVideoLayersAllocationSpatialMaskBytes(streamCount int) int {
	return (streamCount-1)/2 + 1
}

func rtpVideoLayersAllocationTemporalLayerBytes(layerCount int) int {
	return (layerCount + 3) / 4
}

// RTPColorSpaceSize returns the payload size needed to write colorSpace. The
// RTP extension element header is not counted.
func RTPColorSpaceSize(colorSpace RTPColorSpace) (int, error) {
	if err := ValidateRTPColorSpace(colorSpace); err != nil {
		return 0, err
	}
	if colorSpace.HDRMetadataPresent {
		return RTPColorSpaceHeaderExtensionSize, nil
	}
	return RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata, nil
}

// ParseRTPColorSpaceHeaderExtension parses WebRTC's color-space RTP header-
// extension element payload. The RTP extension element header is not part of
// src.
func ParseRTPColorSpaceHeaderExtension(src []byte) (RTPColorSpace, error) {
	if len(src) < RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata {
		return RTPColorSpace{}, ErrRTPShortBuffer
	}
	if len(src) != RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata &&
		len(src) != RTPColorSpaceHeaderExtensionSize {
		return RTPColorSpace{}, ErrRTPInvalidHeaderExtension
	}
	rangeAndChromaSiting := src[3]
	colorSpace := RTPColorSpace{
		Primaries:              RTPColorSpacePrimary(src[0]),
		Transfer:               RTPColorSpaceTransfer(src[1]),
		Matrix:                 RTPColorSpaceMatrix(src[2]),
		Range:                  RTPColorSpaceRange(rangeAndChromaSiting >> 4),
		ChromaSitingHorizontal: RTPColorSpaceChromaSiting((rangeAndChromaSiting >> 2) & 0x03),
		ChromaSitingVertical:   RTPColorSpaceChromaSiting(rangeAndChromaSiting & 0x03),
	}
	if len(src) == RTPColorSpaceHeaderExtensionSize {
		colorSpace.HDRMetadataPresent = true
		colorSpace.HDRMetadata = RTPColorSpaceHDRMetadata{
			LuminanceMax: binary.BigEndian.Uint16(src[4:6]),
			LuminanceMin: binary.BigEndian.Uint16(src[6:8]),
			PrimaryR: RTPColorSpaceChromaticity{
				X: binary.BigEndian.Uint16(src[8:10]),
				Y: binary.BigEndian.Uint16(src[10:12]),
			},
			PrimaryG: RTPColorSpaceChromaticity{
				X: binary.BigEndian.Uint16(src[12:14]),
				Y: binary.BigEndian.Uint16(src[14:16]),
			},
			PrimaryB: RTPColorSpaceChromaticity{
				X: binary.BigEndian.Uint16(src[16:18]),
				Y: binary.BigEndian.Uint16(src[18:20]),
			},
			WhitePoint: RTPColorSpaceChromaticity{
				X: binary.BigEndian.Uint16(src[20:22]),
				Y: binary.BigEndian.Uint16(src[22:24]),
			},
			MaxContentLightLevel:      binary.BigEndian.Uint16(src[24:26]),
			MaxFrameAverageLightLevel: binary.BigEndian.Uint16(src[26:28]),
		}
	}
	if err := ValidateRTPColorSpace(colorSpace); err != nil {
		return RTPColorSpace{}, err
	}
	return colorSpace, nil
}

// PutRTPColorSpaceHeaderExtension writes WebRTC's color-space RTP header-
// extension element payload. The RTP extension element header is not written.
func PutRTPColorSpaceHeaderExtension(dst []byte, colorSpace RTPColorSpace) (int, error) {
	size, err := RTPColorSpaceSize(colorSpace)
	if err != nil {
		return 0, err
	}
	if len(dst) < size {
		return 0, ErrRTPShortBuffer
	}
	dst[0] = byte(colorSpace.Primaries)
	dst[1] = byte(colorSpace.Transfer)
	dst[2] = byte(colorSpace.Matrix)
	dst[3] = byte(colorSpace.Range<<4) |
		byte(colorSpace.ChromaSitingHorizontal<<2) |
		byte(colorSpace.ChromaSitingVertical)
	if colorSpace.HDRMetadataPresent {
		hdr := colorSpace.HDRMetadata
		binary.BigEndian.PutUint16(dst[4:6], hdr.LuminanceMax)
		binary.BigEndian.PutUint16(dst[6:8], hdr.LuminanceMin)
		binary.BigEndian.PutUint16(dst[8:10], hdr.PrimaryR.X)
		binary.BigEndian.PutUint16(dst[10:12], hdr.PrimaryR.Y)
		binary.BigEndian.PutUint16(dst[12:14], hdr.PrimaryG.X)
		binary.BigEndian.PutUint16(dst[14:16], hdr.PrimaryG.Y)
		binary.BigEndian.PutUint16(dst[16:18], hdr.PrimaryB.X)
		binary.BigEndian.PutUint16(dst[18:20], hdr.PrimaryB.Y)
		binary.BigEndian.PutUint16(dst[20:22], hdr.WhitePoint.X)
		binary.BigEndian.PutUint16(dst[22:24], hdr.WhitePoint.Y)
		binary.BigEndian.PutUint16(dst[24:26], hdr.MaxContentLightLevel)
		binary.BigEndian.PutUint16(dst[26:28], hdr.MaxFrameAverageLightLevel)
	}
	return size, nil
}

func ValidateRTPColorSpace(colorSpace RTPColorSpace) error {
	if !validRTPColorSpacePrimary(colorSpace.Primaries) ||
		!validRTPColorSpaceTransfer(colorSpace.Transfer) ||
		!validRTPColorSpaceMatrix(colorSpace.Matrix) ||
		!validRTPColorSpaceRange(colorSpace.Range) ||
		!validRTPColorSpaceChromaSiting(colorSpace.ChromaSitingHorizontal) ||
		!validRTPColorSpaceChromaSiting(colorSpace.ChromaSitingVertical) {
		return ErrRTPInvalidHeaderExtension
	}
	if colorSpace.HDRMetadataPresent {
		return ValidateRTPColorSpaceHDRMetadata(colorSpace.HDRMetadata)
	}
	if colorSpace.HDRMetadata != (RTPColorSpaceHDRMetadata{}) {
		return ErrRTPInvalidHeaderExtension
	}
	return nil
}

func ValidateRTPColorSpaceHDRMetadata(hdr RTPColorSpaceHDRMetadata) error {
	if hdr.LuminanceMax > 20000 ||
		hdr.LuminanceMin > 50000 ||
		hdr.MaxContentLightLevel > 20000 ||
		hdr.MaxFrameAverageLightLevel > 20000 ||
		!validRTPColorSpaceChromaticity(hdr.PrimaryR) ||
		!validRTPColorSpaceChromaticity(hdr.PrimaryG) ||
		!validRTPColorSpaceChromaticity(hdr.PrimaryB) ||
		!validRTPColorSpaceChromaticity(hdr.WhitePoint) {
		return ErrRTPInvalidHeaderExtension
	}
	return nil
}

func validRTPColorSpaceChromaticity(chromaticity RTPColorSpaceChromaticity) bool {
	return chromaticity.X <= 50000 && chromaticity.Y <= 50000
}

func validRTPColorSpacePrimary(primary RTPColorSpacePrimary) bool {
	switch primary {
	case RTPColorSpacePrimaryBT709,
		RTPColorSpacePrimaryUnspecified,
		RTPColorSpacePrimaryBT470M,
		RTPColorSpacePrimaryBT470BG,
		RTPColorSpacePrimarySMPTE170M,
		RTPColorSpacePrimarySMPTE240M,
		RTPColorSpacePrimaryFilm,
		RTPColorSpacePrimaryBT2020,
		RTPColorSpacePrimarySMPTEST428,
		RTPColorSpacePrimarySMPTEST431,
		RTPColorSpacePrimarySMPTEST432,
		RTPColorSpacePrimaryJEDECP22:
		return true
	default:
		return false
	}
}

func validRTPColorSpaceTransfer(transfer RTPColorSpaceTransfer) bool {
	switch transfer {
	case RTPColorSpaceTransferBT709,
		RTPColorSpaceTransferUnspecified,
		RTPColorSpaceTransferGamma22,
		RTPColorSpaceTransferGamma28,
		RTPColorSpaceTransferSMPTE170M,
		RTPColorSpaceTransferSMPTE240M,
		RTPColorSpaceTransferLinear,
		RTPColorSpaceTransferLog,
		RTPColorSpaceTransferLogSqrt,
		RTPColorSpaceTransferIEC6196624,
		RTPColorSpaceTransferBT1361ECG,
		RTPColorSpaceTransferIEC6196621,
		RTPColorSpaceTransferBT202010,
		RTPColorSpaceTransferBT202012,
		RTPColorSpaceTransferSMPTEST2084,
		RTPColorSpaceTransferSMPTEST428,
		RTPColorSpaceTransferARIBSTDB67:
		return true
	default:
		return false
	}
}

func validRTPColorSpaceMatrix(matrix RTPColorSpaceMatrix) bool {
	switch matrix {
	case RTPColorSpaceMatrixRGB,
		RTPColorSpaceMatrixBT709,
		RTPColorSpaceMatrixUnspecified,
		RTPColorSpaceMatrixFCC,
		RTPColorSpaceMatrixBT470BG,
		RTPColorSpaceMatrixSMPTE170M,
		RTPColorSpaceMatrixSMPTE240M,
		RTPColorSpaceMatrixYCgCo,
		RTPColorSpaceMatrixBT2020NCL,
		RTPColorSpaceMatrixBT2020CL,
		RTPColorSpaceMatrixSMPTE2085,
		RTPColorSpaceMatrixCDNCLS,
		RTPColorSpaceMatrixCDCLS,
		RTPColorSpaceMatrixBT2100ICTCP:
		return true
	default:
		return false
	}
}

func validRTPColorSpaceRange(rng RTPColorSpaceRange) bool {
	switch rng {
	case RTPColorSpaceRangeInvalid,
		RTPColorSpaceRangeLimited,
		RTPColorSpaceRangeFull,
		RTPColorSpaceRangeDerived:
		return true
	default:
		return false
	}
}

func validRTPColorSpaceChromaSiting(siting RTPColorSpaceChromaSiting) bool {
	switch siting {
	case RTPColorSpaceChromaSitingUnspecified,
		RTPColorSpaceChromaSitingCollocated,
		RTPColorSpaceChromaSitingHalf:
		return true
	default:
		return false
	}
}
