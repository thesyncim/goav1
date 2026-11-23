package goav1

import (
	"encoding/hex"
	"errors"
	"testing"
)

func TestPublicRTPWebRTCHeaderExtensionConstants(t *testing.T) {
	if RTPCoordinationOfVideoOrientationHeaderExtensionSize != 1 {
		t.Fatalf("RTPCoordinationOfVideoOrientationHeaderExtensionSize = %d, want 1", RTPCoordinationOfVideoOrientationHeaderExtensionSize)
	}
	if RTPPlayoutDelayHeaderExtensionSize != 3 {
		t.Fatalf("RTPPlayoutDelayHeaderExtensionSize = %d, want 3", RTPPlayoutDelayHeaderExtensionSize)
	}
	if RTPTransportWideCCHeaderExtensionSize != 2 {
		t.Fatalf("RTPTransportWideCCHeaderExtensionSize = %d, want 2", RTPTransportWideCCHeaderExtensionSize)
	}
	if RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest != 2 {
		t.Fatalf("RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest = %d, want 2", RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest)
	}
	if RTPTransportWideCC02HeaderExtensionSize != 4 {
		t.Fatalf("RTPTransportWideCC02HeaderExtensionSize = %d, want 4", RTPTransportWideCC02HeaderExtensionSize)
	}
	if RTPAbsoluteSendTimeHeaderExtensionSize != 3 {
		t.Fatalf("RTPAbsoluteSendTimeHeaderExtensionSize = %d, want 3", RTPAbsoluteSendTimeHeaderExtensionSize)
	}
	if RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset != 8 {
		t.Fatalf("RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset = %d, want 8", RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset)
	}
	if RTPAbsoluteCaptureTimeHeaderExtensionSize != 16 {
		t.Fatalf("RTPAbsoluteCaptureTimeHeaderExtensionSize = %d, want 16", RTPAbsoluteCaptureTimeHeaderExtensionSize)
	}
	if RTPVideoContentTypeHeaderExtensionSize != 1 {
		t.Fatalf("RTPVideoContentTypeHeaderExtensionSize = %d, want 1", RTPVideoContentTypeHeaderExtensionSize)
	}
	if RTPVideoTimingHeaderExtensionSize != 13 {
		t.Fatalf("RTPVideoTimingHeaderExtensionSize = %d, want 13", RTPVideoTimingHeaderExtensionSize)
	}
	if RTPVideoTimingHeaderExtensionSizeWithoutFlags != 12 {
		t.Fatalf("RTPVideoTimingHeaderExtensionSizeWithoutFlags = %d, want 12", RTPVideoTimingHeaderExtensionSizeWithoutFlags)
	}
	if RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata != 4 {
		t.Fatalf("RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata = %d, want 4", RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata)
	}
	if RTPColorSpaceHeaderExtensionSize != 28 {
		t.Fatalf("RTPColorSpaceHeaderExtensionSize = %d, want 28", RTPColorSpaceHeaderExtensionSize)
	}
	if RTPVideoLayersAllocationMaxHeaderExtensionSize != 407 {
		t.Fatalf("RTPVideoLayersAllocationMaxHeaderExtensionSize = %d, want 407", RTPVideoLayersAllocationMaxHeaderExtensionSize)
	}
	if RTPPlayoutDelayMaxMilliseconds != 40950 {
		t.Fatalf("RTPPlayoutDelayMaxMilliseconds = %d, want 40950", RTPPlayoutDelayMaxMilliseconds)
	}
	if RTPTransportWideCC02MaxFeedbackSequenceCount != 0x7fff {
		t.Fatalf("RTPTransportWideCC02MaxFeedbackSequenceCount = %#x, want 0x7fff", RTPTransportWideCC02MaxFeedbackSequenceCount)
	}
	if RTPAbsoluteSendTimeMaxValue != 0x00ffffff {
		t.Fatalf("RTPAbsoluteSendTimeMaxValue = %#x, want 0x00ffffff", RTPAbsoluteSendTimeMaxValue)
	}
	if RTPVideoContentTypeUnspecified != 0 ||
		RTPVideoContentTypeScreenshare != 1 {
		t.Fatalf("unexpected RTP video content type constants")
	}
	if RTPVideoTimingFlagTriggeredByTimer != 0x01 ||
		RTPVideoTimingFlagFrameLargerThanKnown != 0x02 {
		t.Fatalf("unexpected RTP video timing flags")
	}
	if RTPColorSpacePrimaryBT2020 != 9 ||
		RTPColorSpaceTransferBT202010 != 14 ||
		RTPColorSpaceMatrixBT2020NCL != 9 ||
		RTPColorSpaceRangeFull != 2 ||
		RTPColorSpaceChromaSitingHalf != 2 {
		t.Fatalf("unexpected RTP color-space constants")
	}
}

func TestPublicRTPCoordinationOfVideoOrientationHeaderExtension(t *testing.T) {
	orientation := RTPCoordinationOfVideoOrientation{
		Camera:   true,
		Flip:     true,
		Rotation: 270,
	}
	var buf [2]byte
	n, err := PutRTPCoordinationOfVideoOrientationHeaderExtension(buf[:], orientation)
	if err != nil {
		t.Fatalf("PutRTPCoordinationOfVideoOrientationHeaderExtension returned error: %v", err)
	}
	if n != RTPCoordinationOfVideoOrientationHeaderExtensionSize || buf[0] != 0x0f {
		t.Fatalf("encoded CVO n=%d buf=%#v", n, buf)
	}
	got, err := ParseRTPCoordinationOfVideoOrientationHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPCoordinationOfVideoOrientationHeaderExtension returned error: %v", err)
	}
	if got != orientation {
		t.Fatalf("ParseRTPCoordinationOfVideoOrientationHeaderExtension = %+v, want %+v", got, orientation)
	}
	got, err = ParseRTPCoordinationOfVideoOrientationHeaderExtension([]byte{0xf1})
	if err != nil {
		t.Fatalf("ParseRTPCoordinationOfVideoOrientationHeaderExtension ignored reserved bits: %v", err)
	}
	if got != (RTPCoordinationOfVideoOrientation{Rotation: 90}) {
		t.Fatalf("reserved-bit CVO parse = %+v", got)
	}
	if err := ValidateRTPCoordinationOfVideoOrientation(RTPCoordinationOfVideoOrientation{Rotation: 45}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("invalid CVO rotation error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPCoordinationOfVideoOrientationHeaderExtension(nil); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPCoordinationOfVideoOrientationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPCoordinationOfVideoOrientationHeaderExtension(buf[:2]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPCoordinationOfVideoOrientationHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPCoordinationOfVideoOrientationHeaderExtension(nil, orientation); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPCoordinationOfVideoOrientationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPPlayoutDelayHeaderExtension(t *testing.T) {
	delay := RTPPlayoutDelay{MinDelayMs: 120, MaxDelayMs: 3450}
	var buf [4]byte
	n, err := PutRTPPlayoutDelayHeaderExtension(buf[:], delay)
	if err != nil {
		t.Fatalf("PutRTPPlayoutDelayHeaderExtension returned error: %v", err)
	}
	if n != RTPPlayoutDelayHeaderExtensionSize ||
		buf[0] != 0x00 || buf[1] != 0xc1 || buf[2] != 0x59 {
		t.Fatalf("encoded playout-delay n=%d buf=%#v", n, buf)
	}
	got, err := ParseRTPPlayoutDelayHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPPlayoutDelayHeaderExtension returned error: %v", err)
	}
	if got != delay {
		t.Fatalf("ParseRTPPlayoutDelayHeaderExtension = %+v, want %+v", got, delay)
	}
	for _, invalid := range []RTPPlayoutDelay{
		{MinDelayMs: -10, MaxDelayMs: 0},
		{MinDelayMs: 100, MaxDelayMs: 90},
		{MinDelayMs: 0, MaxDelayMs: RTPPlayoutDelayMaxMilliseconds + 10},
		{MinDelayMs: 5, MaxDelayMs: 10},
	} {
		if err := ValidateRTPPlayoutDelay(invalid); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
			t.Fatalf("ValidateRTPPlayoutDelay(%+v) error = %v, want ErrRTPInvalidHeaderExtension", invalid, err)
		}
	}
	if _, err := ParseRTPPlayoutDelayHeaderExtension(buf[:2]); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPPlayoutDelayHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPPlayoutDelayHeaderExtension(buf[:4]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPPlayoutDelayHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPPlayoutDelayHeaderExtension(buf[:2], delay); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPPlayoutDelayHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPTransportWideCCHeaderExtension(t *testing.T) {
	var buf [4]byte
	n, err := PutRTPTransportWideCCHeaderExtension(buf[:], 0x1234)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCCHeaderExtension returned error: %v", err)
	}
	if n != RTPTransportWideCCHeaderExtensionSize || buf[0] != 0x12 || buf[1] != 0x34 {
		t.Fatalf("encoded transport-wide cc n=%d buf=%#v", n, buf)
	}
	got, err := ParseRTPTransportWideCCHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCCHeaderExtension returned error: %v", err)
	}
	if got != 0x1234 {
		t.Fatalf("ParseRTPTransportWideCCHeaderExtension = %#x, want 0x1234", got)
	}
	if _, err := ParseRTPTransportWideCCHeaderExtension(buf[:1]); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPTransportWideCCHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPTransportWideCCHeaderExtension(buf[:3]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPTransportWideCCHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPTransportWideCCHeaderExtension(buf[:1], 1); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPTransportWideCCHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPTransportWideCC02HeaderExtension(t *testing.T) {
	var buf [4]byte
	plain := RTPTransportWideCC02{SequenceNumber: 0x1234}
	size, err := RTPTransportWideCC02Size(plain)
	if err != nil {
		t.Fatalf("RTPTransportWideCC02Size plain returned error: %v", err)
	}
	if size != RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest {
		t.Fatalf("plain TransportWideCC02 size=%d want %d", size, RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest)
	}
	n, err := PutRTPTransportWideCC02HeaderExtension(buf[:], plain)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCC02HeaderExtension plain returned error: %v", err)
	}
	if n != size || buf[0] != 0x12 || buf[1] != 0x34 {
		t.Fatalf("encoded plain transport-wide-cc-02 n=%d buf=%#v", n, buf)
	}
	got, err := ParseRTPTransportWideCC02HeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension plain returned error: %v", err)
	}
	if got != plain {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension plain = %+v, want %+v", got, plain)
	}

	request := RTPTransportWideCC02{
		SequenceNumber:        0xabcd,
		FeedbackRequest:       true,
		IncludeTimestamps:     true,
		FeedbackSequenceCount: 0x1234,
	}
	size, err = RTPTransportWideCC02Size(request)
	if err != nil {
		t.Fatalf("RTPTransportWideCC02Size request returned error: %v", err)
	}
	if size != RTPTransportWideCC02HeaderExtensionSize {
		t.Fatalf("request TransportWideCC02 size=%d want %d", size, RTPTransportWideCC02HeaderExtensionSize)
	}
	n, err = PutRTPTransportWideCC02HeaderExtension(buf[:], request)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCC02HeaderExtension request returned error: %v", err)
	}
	want := []byte{0xab, 0xcd, 0x92, 0x34}
	if n != size || string(buf[:n]) != string(want) {
		t.Fatalf("encoded feedback transport-wide-cc-02 n=%d buf=%#v want %#v", n, buf[:n], want)
	}
	got, err = ParseRTPTransportWideCC02HeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension request returned error: %v", err)
	}
	if got != request {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension request = %+v, want %+v", got, request)
	}
	got, err = ParseRTPTransportWideCC02HeaderExtension([]byte{0xab, 0xcd, 0x80, 0x00})
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension zero count returned error: %v", err)
	}
	if got != (RTPTransportWideCC02{SequenceNumber: 0xabcd}) {
		t.Fatalf("zero-count transport-wide-cc-02 parse = %+v", got)
	}

	for _, invalid := range []RTPTransportWideCC02{
		{SequenceNumber: 1, FeedbackRequest: true},
		{SequenceNumber: 1, IncludeTimestamps: true},
		{SequenceNumber: 1, FeedbackSequenceCount: 1},
	} {
		if err := ValidateRTPTransportWideCC02(invalid); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
			t.Fatalf("ValidateRTPTransportWideCC02(%+v) error = %v, want ErrRTPInvalidHeaderExtension", invalid, err)
		}
	}
	if _, err := ParseRTPTransportWideCC02HeaderExtension(buf[:1]); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPTransportWideCC02HeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPTransportWideCC02HeaderExtension(buf[:3]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("bad-size ParseRTPTransportWideCC02HeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPTransportWideCC02HeaderExtension(buf[:1], plain); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPTransportWideCC02HeaderExtension plain error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := PutRTPTransportWideCC02HeaderExtension(buf[:3], request); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPTransportWideCC02HeaderExtension request error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPAbsoluteSendTimeHeaderExtension(t *testing.T) {
	var buf [4]byte
	n, err := PutRTPAbsoluteSendTimeHeaderExtension(buf[:], 0xabcdef)
	if err != nil {
		t.Fatalf("PutRTPAbsoluteSendTimeHeaderExtension returned error: %v", err)
	}
	if n != RTPAbsoluteSendTimeHeaderExtensionSize ||
		buf[0] != 0xab || buf[1] != 0xcd || buf[2] != 0xef {
		t.Fatalf("encoded abs-send-time n=%d buf=%#v", n, buf)
	}
	got, err := ParseRTPAbsoluteSendTimeHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPAbsoluteSendTimeHeaderExtension returned error: %v", err)
	}
	if got != 0xabcdef {
		t.Fatalf("ParseRTPAbsoluteSendTimeHeaderExtension = %#x, want 0xabcdef", got)
	}
	if _, err := ParseRTPAbsoluteSendTimeHeaderExtension(buf[:2]); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPAbsoluteSendTimeHeaderExtension(buf[:4]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPAbsoluteSendTimeHeaderExtension(buf[:2], 1); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := PutRTPAbsoluteSendTimeHeaderExtension(buf[:], 0x01000000); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("large PutRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
}

func TestPublicRTPAbsoluteCaptureTimeHeaderExtension(t *testing.T) {
	var buf [16]byte
	plain := RTPAbsoluteCaptureTime{AbsoluteCaptureTimestamp: 0x0102030405060708}
	size, err := RTPAbsoluteCaptureTimeSize(plain)
	if err != nil {
		t.Fatalf("RTPAbsoluteCaptureTimeSize plain returned error: %v", err)
	}
	if size != RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset {
		t.Fatalf("plain abs-capture-time size=%d want %d", size, RTPAbsoluteCaptureTimeHeaderExtensionSizeWithoutClockOffset)
	}
	n, err := PutRTPAbsoluteCaptureTimeHeaderExtension(buf[:], plain)
	if err != nil {
		t.Fatalf("PutRTPAbsoluteCaptureTimeHeaderExtension plain returned error: %v", err)
	}
	wantPlain := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	if n != size || string(buf[:n]) != string(wantPlain) {
		t.Fatalf("encoded plain abs-capture-time n=%d buf=%#v want %#v", n, buf[:n], wantPlain)
	}
	got, err := ParseRTPAbsoluteCaptureTimeHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPAbsoluteCaptureTimeHeaderExtension plain returned error: %v", err)
	}
	if got != plain {
		t.Fatalf("ParseRTPAbsoluteCaptureTimeHeaderExtension plain = %+v, want %+v", got, plain)
	}

	withOffset := RTPAbsoluteCaptureTime{
		AbsoluteCaptureTimestamp:           0x1112131415161718,
		EstimatedCaptureClockOffsetPresent: true,
		EstimatedCaptureClockOffset:        -2,
	}
	size, err = RTPAbsoluteCaptureTimeSize(withOffset)
	if err != nil {
		t.Fatalf("RTPAbsoluteCaptureTimeSize offset returned error: %v", err)
	}
	if size != RTPAbsoluteCaptureTimeHeaderExtensionSize {
		t.Fatalf("offset abs-capture-time size=%d want %d", size, RTPAbsoluteCaptureTimeHeaderExtensionSize)
	}
	n, err = PutRTPAbsoluteCaptureTimeHeaderExtension(buf[:], withOffset)
	if err != nil {
		t.Fatalf("PutRTPAbsoluteCaptureTimeHeaderExtension offset returned error: %v", err)
	}
	wantOffset := []byte{0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe}
	if n != size || string(buf[:n]) != string(wantOffset) {
		t.Fatalf("encoded offset abs-capture-time n=%d buf=%#v want %#v", n, buf[:n], wantOffset)
	}
	got, err = ParseRTPAbsoluteCaptureTimeHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPAbsoluteCaptureTimeHeaderExtension offset returned error: %v", err)
	}
	if got != withOffset {
		t.Fatalf("ParseRTPAbsoluteCaptureTimeHeaderExtension offset = %+v, want %+v", got, withOffset)
	}
	invalid := RTPAbsoluteCaptureTime{AbsoluteCaptureTimestamp: 1, EstimatedCaptureClockOffset: 1}
	if err := ValidateRTPAbsoluteCaptureTime(invalid); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPAbsoluteCaptureTime invalid error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPAbsoluteCaptureTimeHeaderExtension(buf[:7]); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPAbsoluteCaptureTimeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPAbsoluteCaptureTimeHeaderExtension(buf[:9]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("bad-size ParseRTPAbsoluteCaptureTimeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPAbsoluteCaptureTimeHeaderExtension(buf[:7], plain); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPAbsoluteCaptureTimeHeaderExtension plain error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := PutRTPAbsoluteCaptureTimeHeaderExtension(buf[:15], withOffset); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPAbsoluteCaptureTimeHeaderExtension offset error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPVideoContentTypeHeaderExtension(t *testing.T) {
	var buf [2]byte
	n, err := PutRTPVideoContentTypeHeaderExtension(buf[:], RTPVideoContentTypeScreenshare)
	if err != nil {
		t.Fatalf("PutRTPVideoContentTypeHeaderExtension returned error: %v", err)
	}
	if n != RTPVideoContentTypeHeaderExtensionSize || buf[0] != 0x01 {
		t.Fatalf("encoded video-content-type n=%d buf=%#v", n, buf)
	}
	got, err := ParseRTPVideoContentTypeHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPVideoContentTypeHeaderExtension returned error: %v", err)
	}
	if got != RTPVideoContentTypeScreenshare {
		t.Fatalf("ParseRTPVideoContentTypeHeaderExtension = %d, want screenshare", got)
	}
	if err := ValidateRTPVideoContentType(RTPVideoContentType(2)); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("invalid ValidateRTPVideoContentType error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPVideoContentTypeHeaderExtension(nil); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPVideoContentTypeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPVideoContentTypeHeaderExtension(buf[:2]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPVideoContentTypeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPVideoContentTypeHeaderExtension(nil, RTPVideoContentTypeScreenshare); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPVideoContentTypeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPVideoTimingHeaderExtension(t *testing.T) {
	timing := RTPVideoTiming{
		Flags:                        RTPVideoTimingFlagTriggeredByTimer | RTPVideoTimingFlagFrameLargerThanKnown,
		EncodeStartDeltaMs:           1,
		EncodeFinishDeltaMs:          2,
		PacketizationCompleteDeltaMs: 3,
		PacerExitDeltaMs:             4,
		NetworkTimestampDeltaMs:      5,
		NetworkTimestamp2DeltaMs:     6,
	}
	var buf [14]byte
	n, err := PutRTPVideoTimingHeaderExtension(buf[:], timing)
	if err != nil {
		t.Fatalf("PutRTPVideoTimingHeaderExtension returned error: %v", err)
	}
	want := []byte{0x03, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6}
	if n != RTPVideoTimingHeaderExtensionSize || string(buf[:n]) != string(want) {
		t.Fatalf("encoded video-timing n=%d buf=%#v want %#v", n, buf[:n], want)
	}
	got, err := ParseRTPVideoTimingHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPVideoTimingHeaderExtension returned error: %v", err)
	}
	if got != timing {
		t.Fatalf("ParseRTPVideoTimingHeaderExtension = %+v, want %+v", got, timing)
	}
	reserved, err := ParseRTPVideoTimingHeaderExtension(append([]byte{0xf3}, want[1:]...))
	if err != nil {
		t.Fatalf("ParseRTPVideoTimingHeaderExtension with reserved flags returned error: %v", err)
	}
	if reserved != timing {
		t.Fatalf("reserved-flag video timing parse = %+v, want %+v", reserved, timing)
	}
	legacy, err := ParseRTPVideoTimingHeaderExtension(want[1:])
	if err != nil {
		t.Fatalf("ParseRTPVideoTimingHeaderExtension legacy returned error: %v", err)
	}
	legacyWant := timing
	legacyWant.Flags = 0
	if legacy != legacyWant {
		t.Fatalf("legacy video timing parse = %+v, want %+v", legacy, legacyWant)
	}
	bad := timing
	bad.Flags = 0x80
	if err := ValidateRTPVideoTiming(bad); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("invalid ValidateRTPVideoTiming error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPVideoTimingHeaderExtension(buf[:11]); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPVideoTimingHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPVideoTimingHeaderExtension(buf[:14]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPVideoTimingHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPVideoTimingHeaderExtension(buf[:12], timing); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPVideoTimingHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPColorSpaceHeaderExtension(t *testing.T) {
	colorSpace := RTPColorSpace{
		Primaries:              RTPColorSpacePrimaryBT709,
		Transfer:               RTPColorSpaceTransferBT202010,
		Matrix:                 RTPColorSpaceMatrixBT2020NCL,
		Range:                  RTPColorSpaceRangeFull,
		ChromaSitingHorizontal: RTPColorSpaceChromaSitingHalf,
		ChromaSitingVertical:   RTPColorSpaceChromaSitingCollocated,
	}
	var buf [28]byte
	size, err := RTPColorSpaceSize(colorSpace)
	if err != nil {
		t.Fatalf("RTPColorSpaceSize plain returned error: %v", err)
	}
	if size != RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata {
		t.Fatalf("plain color-space size=%d want %d", size, RTPColorSpaceHeaderExtensionSizeWithoutHDRMetadata)
	}
	n, err := PutRTPColorSpaceHeaderExtension(buf[:], colorSpace)
	if err != nil {
		t.Fatalf("PutRTPColorSpaceHeaderExtension plain returned error: %v", err)
	}
	wantPlain := []byte{0x01, 0x0e, 0x09, 0x29}
	if n != size || string(buf[:n]) != string(wantPlain) {
		t.Fatalf("encoded plain color-space n=%d buf=%#v want %#v", n, buf[:n], wantPlain)
	}
	got, err := ParseRTPColorSpaceHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPColorSpaceHeaderExtension plain returned error: %v", err)
	}
	if got != colorSpace {
		t.Fatalf("ParseRTPColorSpaceHeaderExtension plain = %+v, want %+v", got, colorSpace)
	}

	withHDR := colorSpace
	withHDR.HDRMetadataPresent = true
	withHDR.HDRMetadata = RTPColorSpaceHDRMetadata{
		LuminanceMax:              1000,
		LuminanceMin:              50,
		PrimaryR:                  RTPColorSpaceChromaticity{X: 34000, Y: 16000},
		PrimaryG:                  RTPColorSpaceChromaticity{X: 13250, Y: 34500},
		PrimaryB:                  RTPColorSpaceChromaticity{X: 7500, Y: 3000},
		WhitePoint:                RTPColorSpaceChromaticity{X: 15635, Y: 16450},
		MaxContentLightLevel:      1000,
		MaxFrameAverageLightLevel: 400,
	}
	size, err = RTPColorSpaceSize(withHDR)
	if err != nil {
		t.Fatalf("RTPColorSpaceSize HDR returned error: %v", err)
	}
	if size != RTPColorSpaceHeaderExtensionSize {
		t.Fatalf("HDR color-space size=%d want %d", size, RTPColorSpaceHeaderExtensionSize)
	}
	n, err = PutRTPColorSpaceHeaderExtension(buf[:], withHDR)
	if err != nil {
		t.Fatalf("PutRTPColorSpaceHeaderExtension HDR returned error: %v", err)
	}
	wantHDR := []byte{
		0x01, 0x0e, 0x09, 0x29,
		0x03, 0xe8, 0x00, 0x32,
		0x84, 0xd0, 0x3e, 0x80,
		0x33, 0xc2, 0x86, 0xc4,
		0x1d, 0x4c, 0x0b, 0xb8,
		0x3d, 0x13, 0x40, 0x42,
		0x03, 0xe8, 0x01, 0x90,
	}
	if n != size || string(buf[:n]) != string(wantHDR) {
		t.Fatalf("encoded HDR color-space n=%d buf=%#v want %#v", n, buf[:n], wantHDR)
	}
	got, err = ParseRTPColorSpaceHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPColorSpaceHeaderExtension HDR returned error: %v", err)
	}
	if got != withHDR {
		t.Fatalf("ParseRTPColorSpaceHeaderExtension HDR = %+v, want %+v", got, withHDR)
	}

	invalid := colorSpace
	invalid.Primaries = RTPColorSpacePrimary(3)
	if err := ValidateRTPColorSpace(invalid); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPColorSpace invalid primary error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	invalid = colorSpace
	invalid.HDRMetadata.LuminanceMax = 1
	if err := ValidateRTPColorSpace(invalid); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPColorSpace dropped HDR error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	badHDR := withHDR.HDRMetadata
	badHDR.LuminanceMax = 20001
	if err := ValidateRTPColorSpaceHDRMetadata(badHDR); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPColorSpaceHDRMetadata invalid error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPColorSpaceHeaderExtension(buf[:3]); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPColorSpaceHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPColorSpaceHeaderExtension(buf[:5]); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("bad-size ParseRTPColorSpaceHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPColorSpaceHeaderExtension([]byte{0x03, 0x0e, 0x09, 0x29}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("invalid-value ParseRTPColorSpaceHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := PutRTPColorSpaceHeaderExtension(buf[:3], colorSpace); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPColorSpaceHeaderExtension plain error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := PutRTPColorSpaceHeaderExtension(buf[:27], withHDR); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPColorSpaceHeaderExtension HDR error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPVideoLayersAllocationHeaderExtension(t *testing.T) {
	cases := []struct {
		name       string
		allocation RTPVideoLayersAllocation
		hex        string
	}{
		{
			name: "three streams",
			allocation: RTPVideoLayersAllocation{
				RTPStreamID:    0,
				RTPStreamCount: 3,
				ActiveSpatialLayers: []RTPVideoLayersAllocationLayer{
					{RTPStreamID: 0, SpatialID: 0, TargetBitratesKbps: []uint32{150}},
					{RTPStreamID: 1, SpatialID: 0, TargetBitratesKbps: []uint32{240, 400}},
					{RTPStreamID: 2, SpatialID: 0, TargetBitratesKbps: []uint32{720, 1200}},
				},
			},
			hex: "21149601f0019003d005b009",
		},
		{
			name: "three streams resolution",
			allocation: RTPVideoLayersAllocation{
				RTPStreamID:    2,
				RTPStreamCount: 3,
				ActiveSpatialLayers: []RTPVideoLayersAllocationLayer{
					{RTPStreamID: 0, SpatialID: 0, TargetBitratesKbps: []uint32{150}, Width: 320, Height: 180, Framerate: 30},
					{RTPStreamID: 1, SpatialID: 0, TargetBitratesKbps: []uint32{240, 400}, Width: 640, Height: 360, Framerate: 30},
					{RTPStreamID: 2, SpatialID: 0, TargetBitratesKbps: []uint32{720, 1200}, Width: 1280, Height: 720, Framerate: 30},
				},
				HasResolutionAndFramerate: true,
			},
			hex: "a1149601f0019003d005b009013f00b31e027f01671e04ff02cf1e",
		},
		{
			name: "paused middle stream",
			allocation: RTPVideoLayersAllocation{
				RTPStreamID:    1,
				RTPStreamCount: 3,
				ActiveSpatialLayers: []RTPVideoLayersAllocationLayer{
					{RTPStreamID: 0, SpatialID: 0, TargetBitratesKbps: []uint32{150}, Width: 320, Height: 180, Framerate: 30},
					{RTPStreamID: 2, SpatialID: 0, TargetBitratesKbps: []uint32{720, 1200}, Width: 1280, Height: 720, Framerate: 30},
				},
				HasResolutionAndFramerate: true,
			},
			hex: "601010109601d005b009013f00b31e04ff02cf1e",
		},
		{
			name: "multiple spatial layers",
			allocation: RTPVideoLayersAllocation{
				RTPStreamID:    0,
				RTPStreamCount: 1,
				ActiveSpatialLayers: []RTPVideoLayersAllocationLayer{
					{RTPStreamID: 0, SpatialID: 0, TargetBitratesKbps: []uint32{90}},
					{RTPStreamID: 0, SpatialID: 1, TargetBitratesKbps: []uint32{180, 260}},
					{RTPStreamID: 0, SpatialID: 2, TargetBitratesKbps: []uint32{360, 520, 700}},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			size, err := RTPVideoLayersAllocationSize(tc.allocation)
			if err != nil {
				t.Fatalf("RTPVideoLayersAllocationSize returned error: %v", err)
			}
			buf := make([]byte, size)
			n, err := PutRTPVideoLayersAllocationHeaderExtension(buf, tc.allocation)
			if err != nil {
				t.Fatalf("PutRTPVideoLayersAllocationHeaderExtension returned error: %v", err)
			}
			if n != size {
				t.Fatalf("PutRTPVideoLayersAllocationHeaderExtension n=%d want %d", n, size)
			}
			if tc.hex != "" {
				want, err := hex.DecodeString(tc.hex)
				if err != nil {
					t.Fatalf("DecodeString(%q): %v", tc.hex, err)
				}
				if string(buf) != string(want) {
					t.Fatalf("encoded allocation = %x, want %x", buf, want)
				}
			}
			got, err := ParseRTPVideoLayersAllocationHeaderExtension(buf)
			if err != nil {
				t.Fatalf("ParseRTPVideoLayersAllocationHeaderExtension returned error: %v", err)
			}
			if !sameRTPVideoLayersAllocation(got, tc.allocation) {
				t.Fatalf("ParseRTPVideoLayersAllocationHeaderExtension = %+v, want %+v", got, tc.allocation)
			}
		})
	}
}

func TestPublicRTPVideoLayersAllocationHeaderExtensionErrors(t *testing.T) {
	empty := RTPVideoLayersAllocation{}
	size, err := RTPVideoLayersAllocationSize(empty)
	if err != nil {
		t.Fatalf("RTPVideoLayersAllocationSize(empty) returned error: %v", err)
	}
	if size != 1 {
		t.Fatalf("RTPVideoLayersAllocationSize(empty)=%d want 1", size)
	}
	var one [1]byte
	n, err := PutRTPVideoLayersAllocationHeaderExtension(one[:], empty)
	if err != nil {
		t.Fatalf("PutRTPVideoLayersAllocationHeaderExtension(empty) returned error: %v", err)
	}
	if n != 1 || one[0] != 0 {
		t.Fatalf("empty allocation n=%d byte=%#x, want 1/0", n, one[0])
	}
	got, err := ParseRTPVideoLayersAllocationHeaderExtension(one[:])
	if err != nil {
		t.Fatalf("ParseRTPVideoLayersAllocationHeaderExtension(empty) returned error: %v", err)
	}
	if got.RTPStreamID != 0 || got.RTPStreamCount != 1 || len(got.ActiveSpatialLayers) != 0 {
		t.Fatalf("empty allocation parse = %+v", got)
	}
	negativeEmpty := RTPVideoLayersAllocation{RTPStreamCount: -1}
	if err := ValidateRTPVideoLayersAllocation(negativeEmpty); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPVideoLayersAllocation negative empty error = %v, want ErrRTPInvalidHeaderExtension", err)
	}

	allocation := RTPVideoLayersAllocation{
		RTPStreamID:    0,
		RTPStreamCount: 1,
		ActiveSpatialLayers: []RTPVideoLayersAllocationLayer{
			{RTPStreamID: 0, SpatialID: 0, TargetBitratesKbps: []uint32{150}},
		},
	}
	if _, err := PutRTPVideoLayersAllocationHeaderExtension(nil, allocation); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPVideoLayersAllocationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPVideoLayersAllocationHeaderExtension(nil); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPVideoLayersAllocationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPVideoLayersAllocationHeaderExtension([]byte{0x40}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("bad RID ParseRTPVideoLayersAllocationHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPVideoLayersAllocationHeaderExtension([]byte{0x00, 0x00}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("no-layer ParseRTPVideoLayersAllocationHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := ParseRTPVideoLayersAllocationHeaderExtension([]byte{0x01, 0x00, 0x80}); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("truncated LEB ParseRTPVideoLayersAllocationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPVideoLayersAllocationHeaderExtension([]byte{0x01, 0x00, 0x96, 0x01, 0x00}); !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short resolution ParseRTPVideoLayersAllocationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := ParseRTPVideoLayersAllocationHeaderExtension([]byte{0x01, 0x00, 0x96, 0x01, 0, 0, 0, 0, 0, 0}); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("extra resolution ParseRTPVideoLayersAllocationHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}

	duplicate := allocation
	duplicate.ActiveSpatialLayers = append(duplicate.ActiveSpatialLayers, duplicate.ActiveSpatialLayers[0])
	if err := ValidateRTPVideoLayersAllocation(duplicate); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPVideoLayersAllocation duplicate error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	staleResolution := allocation
	staleResolution.ActiveSpatialLayers[0].Width = 320
	if err := ValidateRTPVideoLayersAllocation(staleResolution); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPVideoLayersAllocation stale resolution error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	badResolution := allocation
	badResolution.HasResolutionAndFramerate = true
	badResolution.ActiveSpatialLayers[0].Width = 1 << 16
	badResolution.ActiveSpatialLayers[0].Height = 720
	badResolution.ActiveSpatialLayers[0].Framerate = 30
	if err := ValidateRTPVideoLayersAllocation(badResolution); err != nil {
		t.Fatalf("ValidateRTPVideoLayersAllocation max resolution returned error: %v", err)
	}
	badResolution.ActiveSpatialLayers[0].Width = 1<<16 + 1
	if err := ValidateRTPVideoLayersAllocation(badResolution); !errors.Is(err, ErrRTPInvalidHeaderExtension) {
		t.Fatalf("ValidateRTPVideoLayersAllocation invalid resolution error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
}

func sameRTPVideoLayersAllocation(a, b RTPVideoLayersAllocation) bool {
	if a.RTPStreamID != b.RTPStreamID ||
		a.RTPStreamCount != b.RTPStreamCount ||
		a.HasResolutionAndFramerate != b.HasResolutionAndFramerate ||
		len(a.ActiveSpatialLayers) != len(b.ActiveSpatialLayers) {
		return false
	}
	for i := range a.ActiveSpatialLayers {
		la := a.ActiveSpatialLayers[i]
		lb := b.ActiveSpatialLayers[i]
		if la.RTPStreamID != lb.RTPStreamID ||
			la.SpatialID != lb.SpatialID ||
			la.Width != lb.Width ||
			la.Height != lb.Height ||
			la.Framerate != lb.Framerate ||
			len(la.TargetBitratesKbps) != len(lb.TargetBitratesKbps) {
			return false
		}
		for j := range la.TargetBitratesKbps {
			if la.TargetBitratesKbps[j] != lb.TargetBitratesKbps[j] {
				return false
			}
		}
	}
	return true
}
