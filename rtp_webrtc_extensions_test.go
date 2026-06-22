package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicRTPWebRTCHeaderExtensionConstants(t *testing.T) {
	if av1.RTPCoordinationOfVideoOrientationHeaderExtensionSize != 1 {
		t.Fatalf("RTPCoordinationOfVideoOrientationHeaderExtensionSize = %d, want 1", av1.RTPCoordinationOfVideoOrientationHeaderExtensionSize)
	}
	if av1.RTPPlayoutDelayHeaderExtensionSize != 3 {
		t.Fatalf("RTPPlayoutDelayHeaderExtensionSize = %d, want 3", av1.RTPPlayoutDelayHeaderExtensionSize)
	}
	if av1.RTPTransportWideCCHeaderExtensionSize != 2 {
		t.Fatalf("RTPTransportWideCCHeaderExtensionSize = %d, want 2", av1.RTPTransportWideCCHeaderExtensionSize)
	}
	if av1.RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest != 2 {
		t.Fatalf("RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest = %d, want 2", av1.RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest)
	}
	if av1.RTPTransportWideCC02HeaderExtensionSize != 4 {
		t.Fatalf("RTPTransportWideCC02HeaderExtensionSize = %d, want 4", av1.RTPTransportWideCC02HeaderExtensionSize)
	}
	if av1.RTPAbsoluteSendTimeHeaderExtensionSize != 3 {
		t.Fatalf("RTPAbsoluteSendTimeHeaderExtensionSize = %d, want 3", av1.RTPAbsoluteSendTimeHeaderExtensionSize)
	}
	if av1.RTPVideoContentTypeHeaderExtensionSize != 1 {
		t.Fatalf("RTPVideoContentTypeHeaderExtensionSize = %d, want 1", av1.RTPVideoContentTypeHeaderExtensionSize)
	}
	if av1.RTPVideoTimingHeaderExtensionSize != 13 {
		t.Fatalf("RTPVideoTimingHeaderExtensionSize = %d, want 13", av1.RTPVideoTimingHeaderExtensionSize)
	}
	if av1.RTPPlayoutDelayMaxMilliseconds != 40950 {
		t.Fatalf("RTPPlayoutDelayMaxMilliseconds = %d, want 40950", av1.RTPPlayoutDelayMaxMilliseconds)
	}
	if av1.RTPTransportWideCC02MaxFeedbackSequenceCount != 0x7fff {
		t.Fatalf("RTPTransportWideCC02MaxFeedbackSequenceCount = %#x, want 0x7fff", av1.RTPTransportWideCC02MaxFeedbackSequenceCount)
	}
	if av1.RTPAbsoluteSendTimeMaxValue != 0x00ffffff {
		t.Fatalf("RTPAbsoluteSendTimeMaxValue = %#x, want 0x00ffffff", av1.RTPAbsoluteSendTimeMaxValue)
	}
	if av1.RTPVideoContentTypeUnspecified != 0 ||
		av1.RTPVideoContentTypeScreenshare != 1 {
		t.Fatalf("unexpected RTP video content type constants")
	}
	if av1.RTPVideoTimingFlagTriggeredByTimer != 0x01 ||
		av1.RTPVideoTimingFlagFrameLargerThanKnown != 0x02 {
		t.Fatalf("unexpected RTP video timing flags")
	}
}

func TestPublicRTPCoordinationOfVideoOrientationHeaderExtension(t *testing.T) {
	orientation := av1.RTPCoordinationOfVideoOrientation{
		Camera:   true,
		Flip:     true,
		Rotation: 270,
	}
	var buf [2]byte
	n, err := av1.PutRTPCoordinationOfVideoOrientationHeaderExtension(buf[:], orientation)
	if err != nil {
		t.Fatalf("PutRTPCoordinationOfVideoOrientationHeaderExtension returned error: %v", err)
	}
	if n != av1.RTPCoordinationOfVideoOrientationHeaderExtensionSize || buf[0] != 0x0f {
		t.Fatalf("encoded CVO n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPCoordinationOfVideoOrientationHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPCoordinationOfVideoOrientationHeaderExtension returned error: %v", err)
	}
	if got != orientation {
		t.Fatalf("ParseRTPCoordinationOfVideoOrientationHeaderExtension = %+v, want %+v", got, orientation)
	}
	got, err = av1.ParseRTPCoordinationOfVideoOrientationHeaderExtension([]byte{0xf1})
	if err != nil {
		t.Fatalf("ParseRTPCoordinationOfVideoOrientationHeaderExtension ignored reserved bits: %v", err)
	}
	if got != (av1.RTPCoordinationOfVideoOrientation{Rotation: 90}) {
		t.Fatalf("reserved-bit CVO parse = %+v", got)
	}
	if err := av1.ValidateRTPCoordinationOfVideoOrientation(av1.RTPCoordinationOfVideoOrientation{Rotation: 45}); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("invalid CVO rotation error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.ParseRTPCoordinationOfVideoOrientationHeaderExtension(nil); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPCoordinationOfVideoOrientationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPCoordinationOfVideoOrientationHeaderExtension(buf[:2]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPCoordinationOfVideoOrientationHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPCoordinationOfVideoOrientationHeaderExtension(nil, orientation); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPCoordinationOfVideoOrientationHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPPlayoutDelayHeaderExtension(t *testing.T) {
	delay := av1.RTPPlayoutDelay{MinDelayMs: 120, MaxDelayMs: 3450}
	var buf [4]byte
	n, err := av1.PutRTPPlayoutDelayHeaderExtension(buf[:], delay)
	if err != nil {
		t.Fatalf("PutRTPPlayoutDelayHeaderExtension returned error: %v", err)
	}
	if n != av1.RTPPlayoutDelayHeaderExtensionSize ||
		buf[0] != 0x00 || buf[1] != 0xc1 || buf[2] != 0x59 {
		t.Fatalf("encoded playout-delay n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPPlayoutDelayHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPPlayoutDelayHeaderExtension returned error: %v", err)
	}
	if got != delay {
		t.Fatalf("ParseRTPPlayoutDelayHeaderExtension = %+v, want %+v", got, delay)
	}
	for _, invalid := range []av1.RTPPlayoutDelay{
		{MinDelayMs: -10, MaxDelayMs: 0},
		{MinDelayMs: 100, MaxDelayMs: 90},
		{MinDelayMs: 0, MaxDelayMs: av1.RTPPlayoutDelayMaxMilliseconds + 10},
		{MinDelayMs: 5, MaxDelayMs: 10},
	} {
		if err := av1.ValidateRTPPlayoutDelay(invalid); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
			t.Fatalf("ValidateRTPPlayoutDelay(%+v) error = %v, want ErrRTPInvalidHeaderExtension", invalid, err)
		}
	}
	if _, err := av1.ParseRTPPlayoutDelayHeaderExtension(buf[:2]); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPPlayoutDelayHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPPlayoutDelayHeaderExtension(buf[:4]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPPlayoutDelayHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPPlayoutDelayHeaderExtension(buf[:2], delay); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPPlayoutDelayHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPTransportWideCCHeaderExtension(t *testing.T) {
	var buf [4]byte
	n, err := av1.PutRTPTransportWideCCHeaderExtension(buf[:], 0x1234)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCCHeaderExtension returned error: %v", err)
	}
	if n != av1.RTPTransportWideCCHeaderExtensionSize || buf[0] != 0x12 || buf[1] != 0x34 {
		t.Fatalf("encoded transport-wide cc n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPTransportWideCCHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCCHeaderExtension returned error: %v", err)
	}
	if got != 0x1234 {
		t.Fatalf("ParseRTPTransportWideCCHeaderExtension = %#x, want 0x1234", got)
	}
	if _, err := av1.ParseRTPTransportWideCCHeaderExtension(buf[:1]); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPTransportWideCCHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPTransportWideCCHeaderExtension(buf[:3]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPTransportWideCCHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPTransportWideCCHeaderExtension(buf[:1], 1); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPTransportWideCCHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPTransportWideCC02HeaderExtension(t *testing.T) {
	var buf [4]byte
	plain := av1.RTPTransportWideCC02{SequenceNumber: 0x1234}
	size, err := av1.RTPTransportWideCC02Size(plain)
	if err != nil {
		t.Fatalf("RTPTransportWideCC02Size plain returned error: %v", err)
	}
	if size != av1.RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest {
		t.Fatalf("plain TransportWideCC02 size=%d want %d", size, av1.RTPTransportWideCC02HeaderExtensionSizeWithoutFeedbackRequest)
	}
	n, err := av1.PutRTPTransportWideCC02HeaderExtension(buf[:], plain)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCC02HeaderExtension plain returned error: %v", err)
	}
	if n != size || buf[0] != 0x12 || buf[1] != 0x34 {
		t.Fatalf("encoded plain transport-wide-cc-02 n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPTransportWideCC02HeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension plain returned error: %v", err)
	}
	if got != plain {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension plain = %+v, want %+v", got, plain)
	}

	request := av1.RTPTransportWideCC02{
		SequenceNumber:        0xabcd,
		FeedbackRequest:       true,
		IncludeTimestamps:     true,
		FeedbackSequenceCount: 0x1234,
	}
	size, err = av1.RTPTransportWideCC02Size(request)
	if err != nil {
		t.Fatalf("RTPTransportWideCC02Size request returned error: %v", err)
	}
	if size != av1.RTPTransportWideCC02HeaderExtensionSize {
		t.Fatalf("request TransportWideCC02 size=%d want %d", size, av1.RTPTransportWideCC02HeaderExtensionSize)
	}
	n, err = av1.PutRTPTransportWideCC02HeaderExtension(buf[:], request)
	if err != nil {
		t.Fatalf("PutRTPTransportWideCC02HeaderExtension request returned error: %v", err)
	}
	want := []byte{0xab, 0xcd, 0x92, 0x34}
	if n != size || string(buf[:n]) != string(want) {
		t.Fatalf("encoded feedback transport-wide-cc-02 n=%d buf=%#v want %#v", n, buf[:n], want)
	}
	got, err = av1.ParseRTPTransportWideCC02HeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension request returned error: %v", err)
	}
	if got != request {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension request = %+v, want %+v", got, request)
	}
	got, err = av1.ParseRTPTransportWideCC02HeaderExtension([]byte{0xab, 0xcd, 0x80, 0x00})
	if err != nil {
		t.Fatalf("ParseRTPTransportWideCC02HeaderExtension zero count returned error: %v", err)
	}
	if got != (av1.RTPTransportWideCC02{SequenceNumber: 0xabcd}) {
		t.Fatalf("zero-count transport-wide-cc-02 parse = %+v", got)
	}

	for _, invalid := range []av1.RTPTransportWideCC02{
		{SequenceNumber: 1, FeedbackRequest: true},
		{SequenceNumber: 1, IncludeTimestamps: true},
		{SequenceNumber: 1, FeedbackSequenceCount: 1},
	} {
		if err := av1.ValidateRTPTransportWideCC02(invalid); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
			t.Fatalf("ValidateRTPTransportWideCC02(%+v) error = %v, want ErrRTPInvalidHeaderExtension", invalid, err)
		}
	}
	if _, err := av1.ParseRTPTransportWideCC02HeaderExtension(buf[:1]); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPTransportWideCC02HeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPTransportWideCC02HeaderExtension(buf[:3]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("bad-size ParseRTPTransportWideCC02HeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPTransportWideCC02HeaderExtension(buf[:1], plain); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPTransportWideCC02HeaderExtension plain error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.PutRTPTransportWideCC02HeaderExtension(buf[:3], request); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPTransportWideCC02HeaderExtension request error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPAbsoluteSendTimeHeaderExtension(t *testing.T) {
	var buf [4]byte
	n, err := av1.PutRTPAbsoluteSendTimeHeaderExtension(buf[:], 0xabcdef)
	if err != nil {
		t.Fatalf("PutRTPAbsoluteSendTimeHeaderExtension returned error: %v", err)
	}
	if n != av1.RTPAbsoluteSendTimeHeaderExtensionSize ||
		buf[0] != 0xab || buf[1] != 0xcd || buf[2] != 0xef {
		t.Fatalf("encoded abs-send-time n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPAbsoluteSendTimeHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPAbsoluteSendTimeHeaderExtension returned error: %v", err)
	}
	if got != 0xabcdef {
		t.Fatalf("ParseRTPAbsoluteSendTimeHeaderExtension = %#x, want 0xabcdef", got)
	}
	if _, err := av1.ParseRTPAbsoluteSendTimeHeaderExtension(buf[:2]); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPAbsoluteSendTimeHeaderExtension(buf[:4]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPAbsoluteSendTimeHeaderExtension(buf[:2], 1); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.PutRTPAbsoluteSendTimeHeaderExtension(buf[:], 0x01000000); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("large PutRTPAbsoluteSendTimeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
}

func TestPublicRTPVideoContentTypeHeaderExtension(t *testing.T) {
	var buf [2]byte
	n, err := av1.PutRTPVideoContentTypeHeaderExtension(buf[:], av1.RTPVideoContentTypeScreenshare)
	if err != nil {
		t.Fatalf("PutRTPVideoContentTypeHeaderExtension returned error: %v", err)
	}
	if n != av1.RTPVideoContentTypeHeaderExtensionSize || buf[0] != 0x01 {
		t.Fatalf("encoded video-content-type n=%d buf=%#v", n, buf)
	}
	got, err := av1.ParseRTPVideoContentTypeHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPVideoContentTypeHeaderExtension returned error: %v", err)
	}
	if got != av1.RTPVideoContentTypeScreenshare {
		t.Fatalf("ParseRTPVideoContentTypeHeaderExtension = %d, want screenshare", got)
	}
	if err := av1.ValidateRTPVideoContentType(av1.RTPVideoContentType(2)); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("invalid ValidateRTPVideoContentType error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.ParseRTPVideoContentTypeHeaderExtension(nil); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPVideoContentTypeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPVideoContentTypeHeaderExtension(buf[:2]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPVideoContentTypeHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPVideoContentTypeHeaderExtension(nil, av1.RTPVideoContentTypeScreenshare); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPVideoContentTypeHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}

func TestPublicRTPVideoTimingHeaderExtension(t *testing.T) {
	timing := av1.RTPVideoTiming{
		Flags:                        av1.RTPVideoTimingFlagTriggeredByTimer | av1.RTPVideoTimingFlagFrameLargerThanKnown,
		EncodeStartDeltaMs:           1,
		EncodeFinishDeltaMs:          2,
		PacketizationCompleteDeltaMs: 3,
		PacerExitDeltaMs:             4,
		NetworkTimestampDeltaMs:      5,
		NetworkTimestamp2DeltaMs:     6,
	}
	var buf [14]byte
	n, err := av1.PutRTPVideoTimingHeaderExtension(buf[:], timing)
	if err != nil {
		t.Fatalf("PutRTPVideoTimingHeaderExtension returned error: %v", err)
	}
	want := []byte{0x03, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6}
	if n != av1.RTPVideoTimingHeaderExtensionSize || string(buf[:n]) != string(want) {
		t.Fatalf("encoded video-timing n=%d buf=%#v want %#v", n, buf[:n], want)
	}
	got, err := av1.ParseRTPVideoTimingHeaderExtension(buf[:n])
	if err != nil {
		t.Fatalf("ParseRTPVideoTimingHeaderExtension returned error: %v", err)
	}
	if got != timing {
		t.Fatalf("ParseRTPVideoTimingHeaderExtension = %+v, want %+v", got, timing)
	}
	reserved, err := av1.ParseRTPVideoTimingHeaderExtension(append([]byte{0xf3}, want[1:]...))
	if err != nil {
		t.Fatalf("ParseRTPVideoTimingHeaderExtension with reserved flags returned error: %v", err)
	}
	if reserved != timing {
		t.Fatalf("reserved-flag video timing parse = %+v, want %+v", reserved, timing)
	}
	bad := timing
	bad.Flags = 0x80
	if err := av1.ValidateRTPVideoTiming(bad); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("invalid ValidateRTPVideoTiming error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.ParseRTPVideoTimingHeaderExtension(buf[:12]); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short ParseRTPVideoTimingHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
	if _, err := av1.ParseRTPVideoTimingHeaderExtension(buf[:14]); !errors.Is(err, av1.ErrRTPInvalidHeaderExtension) {
		t.Fatalf("long ParseRTPVideoTimingHeaderExtension error = %v, want ErrRTPInvalidHeaderExtension", err)
	}
	if _, err := av1.PutRTPVideoTimingHeaderExtension(buf[:12], timing); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short PutRTPVideoTimingHeaderExtension error = %v, want ErrRTPShortBuffer", err)
	}
}
