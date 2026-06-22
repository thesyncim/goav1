package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestAV1RTCPFeedbackConstants(t *testing.T) {
	if av1.RTCPRTPFBGenericNACKFMT != 1 {
		t.Fatalf("RTCPRTPFBGenericNACKFMT = %d, want 1", av1.RTCPRTPFBGenericNACKFMT)
	}
	if av1.RTCPRTPFBTransportFeedbackFMT != 15 {
		t.Fatalf("RTCPRTPFBTransportFeedbackFMT = %d, want 15", av1.RTCPRTPFBTransportFeedbackFMT)
	}
	if av1.RTCPPSFBPictureLossIndicationFMT != 1 {
		t.Fatalf("RTCPPSFBPictureLossIndicationFMT = %d, want 1", av1.RTCPPSFBPictureLossIndicationFMT)
	}
	if av1.RTCPPSFBFullIntraRequestFMT != 4 {
		t.Fatalf("RTCPPSFBFullIntraRequestFMT = %d, want 4", av1.RTCPPSFBFullIntraRequestFMT)
	}
	if av1.RTCPPSFBLayerRefreshRequestFMT != 10 {
		t.Fatalf("RTCPPSFBLayerRefreshRequestFMT = %d, want 10", av1.RTCPPSFBLayerRefreshRequestFMT)
	}
	if av1.RTCPPSFBApplicationLayerFeedbackFMT != 15 {
		t.Fatalf("RTCPPSFBApplicationLayerFeedbackFMT = %d, want 15", av1.RTCPPSFBApplicationLayerFeedbackFMT)
	}
	if av1.RTCPGenericNACKPairSize != 4 {
		t.Fatalf("RTCPGenericNACKPairSize = %d, want 4", av1.RTCPGenericNACKPairSize)
	}
	if av1.RTCPTransportFeedbackFCIHeaderSize != 8 ||
		av1.RTCPTransportFeedbackChunkSize != 2 ||
		av1.RTCPTransportFeedbackFCIMinSize != 10 ||
		av1.RTCPTransportFeedbackDeltaTickMicros != 250 ||
		av1.RTCPTransportFeedbackBaseTimeTickMicros != 64000 ||
		av1.RTCPTransportFeedbackMaxPackets != 0xffff ||
		av1.RTCPTransportFeedbackMaxReferenceTimeTicks != 0xffffff {
		t.Fatalf("unexpected RTCP transport feedback constants")
	}
	if av1.RTCPPictureLossIndicationFCISize != 0 {
		t.Fatalf("RTCPPictureLossIndicationFCISize = %d, want 0", av1.RTCPPictureLossIndicationFCISize)
	}
	if av1.RTCPFullIntraRequestEntrySize != 8 {
		t.Fatalf("RTCPFullIntraRequestEntrySize = %d, want 8", av1.RTCPFullIntraRequestEntrySize)
	}
	if av1.RTCPReceiverEstimatedMaximumBitrateUniqueIdentifier != 0x52454D42 ||
		av1.RTCPReceiverEstimatedMaximumBitrateFCIMinSize != 8 ||
		av1.RTCPReceiverEstimatedMaximumBitrateSSRCSize != 4 ||
		av1.RTCPReceiverEstimatedMaximumBitrateMaxSSRCs != 0xff {
		t.Fatalf("unexpected RTCP REMB constants")
	}
	if av1.RTCPTransportFeedbackStatusNotReceived != 0 ||
		av1.RTCPTransportFeedbackStatusSmallDelta != 1 ||
		av1.RTCPTransportFeedbackStatusLargeOrNegativeDelta != 2 {
		t.Fatalf("unexpected RTCP transport feedback status constants")
	}
	if av1.AV1RTCPLayerRefreshLayerIndexSize != 2 {
		t.Fatalf("AV1RTCPLayerRefreshLayerIndexSize = %d, want 2", av1.AV1RTCPLayerRefreshLayerIndexSize)
	}
	if av1.AV1RTCPLayerRefreshRequestEntrySize != 12 {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntrySize = %d, want 12", av1.AV1RTCPLayerRefreshRequestEntrySize)
	}
	if av1.AV1RTCPLayerRefreshMaxTemporalID != 7 ||
		av1.AV1RTCPLayerRefreshMaxSpatialID != 3 {
		t.Fatalf("unexpected AV1 LRR layer index limits")
	}
	if av1.AV1SDPRTCPFeedbackNACK != "nack" ||
		av1.AV1SDPRTCPFeedbackPLI != "nack pli" ||
		av1.AV1SDPRTCPFeedbackFIR != "ccm fir" ||
		av1.AV1SDPRTCPFeedbackLRR != "ccm lrr" ||
		av1.AV1SDPRTCPFeedbackTransportCC != "transport-cc" ||
		av1.AV1SDPRTCPFeedbackREMB != "goog-remb" {
		t.Fatalf("unexpected AV1 rtcp-fb constants")
	}
}

func TestRTCPPictureLossIndicationFCIRoundTrip(t *testing.T) {
	var buf [4]byte
	n, err := av1.PutRTCPPictureLossIndicationFCI(buf[:])
	if err != nil {
		t.Fatalf("PutRTCPPictureLossIndicationFCI: %v", err)
	}
	if n != 0 || buf != [4]byte{} {
		t.Fatalf("encoded PLI FCI n=%d buf=%#v, want empty no-op", n, buf)
	}
	prefix := []byte{0xaa}
	appended, err := av1.AppendRTCPPictureLossIndicationFCI(prefix)
	if err != nil {
		t.Fatalf("AppendRTCPPictureLossIndicationFCI: %v", err)
	}
	if len(appended) != len(prefix) || appended[0] != prefix[0] {
		t.Fatalf("appended PLI FCI=%#v want unchanged prefix %#v", appended, prefix)
	}
	if err := av1.ParseRTCPPictureLossIndicationFCI(nil); err != nil {
		t.Fatalf("ParseRTCPPictureLossIndicationFCI empty: %v", err)
	}
}

func TestRTCPPictureLossIndicationFCIRejectsInvalid(t *testing.T) {
	if err := av1.ParseRTCPPictureLossIndicationFCI([]byte{0x00}); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("ParseRTCPPictureLossIndicationFCI non-empty err=%v want %v", err, av1.ErrRTCPInvalidFeedback)
	}
}

func TestRTCPReceiverEstimatedMaximumBitrateFCIRoundTrip(t *testing.T) {
	const bitrateBps = 0x3fb93 * 2
	ssrcs := []uint32{0x23456789, 0x2345678a, 0x2345678b}
	size, err := av1.RTCPReceiverEstimatedMaximumBitrateFCISize(ssrcs)
	if err != nil {
		t.Fatalf("RTCPReceiverEstimatedMaximumBitrateFCISize: %v", err)
	}
	if size != av1.RTCPReceiverEstimatedMaximumBitrateFCIMinSize+
		len(ssrcs)*av1.RTCPReceiverEstimatedMaximumBitrateSSRCSize {
		t.Fatalf("REMB FCI size=%d", size)
	}

	buf := make([]byte, size)
	n, err := av1.PutRTCPReceiverEstimatedMaximumBitrateFCI(buf, bitrateBps, ssrcs)
	if err != nil {
		t.Fatalf("PutRTCPReceiverEstimatedMaximumBitrateFCI: %v", err)
	}
	if n != size {
		t.Fatalf("PutRTCPReceiverEstimatedMaximumBitrateFCI n=%d want %d", n, size)
	}
	want := []byte{
		'R', 'E', 'M', 'B',
		0x03, 0x07, 0xfb, 0x93,
		0x23, 0x45, 0x67, 0x89,
		0x23, 0x45, 0x67, 0x8a,
		0x23, 0x45, 0x67, 0x8b,
	}
	if string(buf) != string(want) {
		t.Fatalf("encoded REMB FCI bytes=%#v want %#v", buf, want)
	}

	base := make([]uint32, 1, 1+len(ssrcs))
	base[0] = 0xdeadbeef
	parsed, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(buf, base[:1:cap(base)])
	if err != nil {
		t.Fatalf("ParseRTCPReceiverEstimatedMaximumBitrateFCI: %v", err)
	}
	if parsed.BitrateBps != bitrateBps {
		t.Fatalf("parsed REMB bitrate=%d want %d", parsed.BitrateBps, bitrateBps)
	}
	if base[0] != 0xdeadbeef {
		t.Fatalf("ParseRTCPReceiverEstimatedMaximumBitrateFCI clobbered dst prefix")
	}
	if len(parsed.SSRCs) != len(ssrcs) {
		t.Fatalf("parsed REMB SSRC len=%d want %d", len(parsed.SSRCs), len(ssrcs))
	}
	for i := range ssrcs {
		if parsed.SSRCs[i] != ssrcs[i] {
			t.Fatalf("parsed REMB SSRC[%d]=%#x want %#x", i, parsed.SSRCs[i], ssrcs[i])
		}
	}

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xaa
	appended, err := av1.AppendRTCPReceiverEstimatedMaximumBitrateFCI(prefix, bitrateBps, ssrcs)
	if err != nil {
		t.Fatalf("AppendRTCPReceiverEstimatedMaximumBitrateFCI: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended REMB FCI bytes=%#v", appended)
	}

	emptyBuf := make([]byte, av1.RTCPReceiverEstimatedMaximumBitrateFCIMinSize)
	if _, err := av1.PutRTCPReceiverEstimatedMaximumBitrateFCI(emptyBuf, 123, nil); err != nil {
		t.Fatalf("PutRTCPReceiverEstimatedMaximumBitrateFCI no SSRCs: %v", err)
	}
	empty, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(emptyBuf, nil)
	if err != nil {
		t.Fatalf("ParseRTCPReceiverEstimatedMaximumBitrateFCI no SSRCs: %v", err)
	}
	if empty.BitrateBps != 123 || len(empty.SSRCs) != 0 {
		t.Fatalf("parsed empty REMB=%+v", empty)
	}
}

func TestRTCPReceiverEstimatedMaximumBitrateFCIRejectsInvalid(t *testing.T) {
	valid := []byte{
		'R', 'E', 'M', 'B',
		0x03, 0x07, 0xfb, 0x93,
		0x23, 0x45, 0x67, 0x89,
		0x23, 0x45, 0x67, 0x8a,
		0x23, 0x45, 0x67, 0x8b,
	}
	if _, err := av1.PutRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 7), 1, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPReceiverEstimatedMaximumBitrateFCI err=%v", err)
	}
	if _, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(valid[:7], nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPReceiverEstimatedMaximumBitrateFCI err=%v", err)
	}
	if _, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(valid, make([]uint32, 0, 2)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPReceiverEstimatedMaximumBitrateFCI dst err=%v", err)
	}
	if out, err := av1.AppendRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 0, 7), 1, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short AppendRTCPReceiverEstimatedMaximumBitrateFCI out=%d err=%v", len(out), err)
	}

	tooMany := make([]uint32, av1.RTCPReceiverEstimatedMaximumBitrateMaxSSRCs+1)
	if _, err := av1.RTCPReceiverEstimatedMaximumBitrateFCISize(tooMany); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("too many REMB SSRCs size err=%v", err)
	}
	if _, err := av1.PutRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 4096), 1, tooMany); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("too many REMB SSRCs put err=%v", err)
	}
	if _, err := av1.PutRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, av1.RTCPReceiverEstimatedMaximumBitrateFCIMinSize), uint64(1)<<63, nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("too high REMB bitrate err=%v", err)
	}

	invalidID := append([]byte(nil), valid...)
	invalidID[0] = 'N'
	if _, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(invalidID, nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid REMB unique ID err=%v", err)
	}
	countMismatch := append([]byte(nil), valid...)
	countMismatch[4]++
	if _, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(countMismatch, make([]uint32, 0, 4)); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid REMB SSRC count err=%v", err)
	}
	shiftOverflow := []byte{'R', 'E', 'M', 'B', 0, 63 << 2, 0, 2}
	if _, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(shiftOverflow, nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("overflow REMB bitrate err=%v", err)
	}
	int64Overflow := []byte{'R', 'E', 'M', 'B', 0, 56 << 2, 0, 200}
	if _, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(int64Overflow, nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("int64 overflow REMB bitrate err=%v", err)
	}
}

func TestRTCPGenericNACKPairsRoundTrip(t *testing.T) {
	pairs := []av1.RTCPGenericNACKPair{
		{PacketID: 0x1234, LostPacketBitmask: 0x8001},
		{PacketID: 0xfffe, LostPacketBitmask: 0x0003},
	}
	size, err := av1.RTCPGenericNACKPairsSize(pairs)
	if err != nil {
		t.Fatalf("RTCPGenericNACKPairsSize: %v", err)
	}
	if size != len(pairs)*av1.RTCPGenericNACKPairSize {
		t.Fatalf("NACK size=%d", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPGenericNACKPairs(buf, pairs)
	if err != nil {
		t.Fatalf("PutRTCPGenericNACKPairs: %v", err)
	}
	if n != size {
		t.Fatalf("PutRTCPGenericNACKPairs n=%d want %d", n, size)
	}
	want := []byte{0x12, 0x34, 0x80, 0x01, 0xff, 0xfe, 0x00, 0x03}
	if string(buf) != string(want) {
		t.Fatalf("encoded NACK bytes=%#v want %#v", buf, want)
	}
	parsed, err := av1.ParseRTCPGenericNACKPairs(buf, make([]av1.RTCPGenericNACKPair, 0, len(pairs)))
	if err != nil {
		t.Fatalf("ParseRTCPGenericNACKPairs: %v", err)
	}
	if len(parsed) != len(pairs) {
		t.Fatalf("parsed NACK pairs len=%d want %d", len(parsed), len(pairs))
	}
	for i := range pairs {
		if parsed[i] != pairs[i] {
			t.Fatalf("parsed[%d]=%+v want %+v", i, parsed[i], pairs[i])
		}
	}
	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xaa
	appended, err := av1.AppendRTCPGenericNACKPairs(prefix, pairs)
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairs: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended NACK bytes=%#v", appended)
	}
}

func TestRTCPGenericNACKPairsRejectsInvalid(t *testing.T) {
	pair := av1.RTCPGenericNACKPair{PacketID: 1, LostPacketBitmask: 2}
	var buf [av1.RTCPGenericNACKPairSize]byte
	if _, err := av1.PutRTCPGenericNACKPair(buf[:1], pair); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPGenericNACKPair err=%v", err)
	}
	if _, _, err := av1.ParseRTCPGenericNACKPair(buf[:1]); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPGenericNACKPair err=%v", err)
	}
	pairs := []av1.RTCPGenericNACKPair{pair, pair}
	size, err := av1.RTCPGenericNACKPairsSize(pairs)
	if err != nil {
		t.Fatalf("RTCPGenericNACKPairsSize valid: %v", err)
	}
	if _, err := av1.PutRTCPGenericNACKPairs(make([]byte, size-1), pairs); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPGenericNACKPairs err=%v", err)
	}
	if out, err := av1.AppendRTCPGenericNACKPairs(make([]byte, 0, size-1), pairs); !errors.Is(err, av1.ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short AppendRTCPGenericNACKPairs out=%d err=%v", len(out), err)
	}
	if _, err := av1.ParseRTCPGenericNACKPairs(make([]byte, size), make([]av1.RTCPGenericNACKPair, 0, 1)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPGenericNACKPairs dst err=%v", err)
	}
	if _, err := av1.ParseRTCPGenericNACKPairs(make([]byte, size-1), nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("ragged ParseRTCPGenericNACKPairs err=%v", err)
	}
}

func TestRTCPTransportFeedbackFCIOneBitRoundTrip(t *testing.T) {
	feedback := av1.RTCPTransportFeedback{
		BaseSequenceNumber:  0x0001,
		ReferenceTimeTicks:  0x010203,
		FeedbackPacketCount: 0x7a,
		DeltasPresent:       true,
		Packets: []av1.RTCPTransportFeedbackPacket{
			{SequenceNumber: 0x0001, Received: true, DeltaTicks: 4},
			{SequenceNumber: 0x0002, Received: true, DeltaTicks: 4},
			{SequenceNumber: 0x0003},
			{SequenceNumber: 0x0004},
			{SequenceNumber: 0x0005, Received: true, DeltaTicks: 4},
		},
	}
	size, err := av1.RTCPTransportFeedbackFCISize(feedback)
	if err != nil {
		t.Fatalf("RTCPTransportFeedbackFCISize: %v", err)
	}
	if size != 13 {
		t.Fatalf("transport feedback size=%d want 13", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPTransportFeedbackFCI(buf, feedback)
	if err != nil {
		t.Fatalf("PutRTCPTransportFeedbackFCI: %v", err)
	}
	if n != size {
		t.Fatalf("PutRTCPTransportFeedbackFCI n=%d want %d", n, size)
	}
	want := []byte{0x00, 0x01, 0x00, 0x05, 0x01, 0x02, 0x03, 0x7a, 0xb2, 0x00, 0x04, 0x04, 0x04}
	if string(buf) != string(want) {
		t.Fatalf("transport feedback bytes=%#v want %#v", buf, want)
	}

	base := make([]av1.RTCPTransportFeedbackPacket, 1, 1+len(feedback.Packets))
	base[0].SequenceNumber = 0xdead
	parsed, err := av1.ParseRTCPTransportFeedbackFCI(buf, base[:1:cap(base)])
	if err != nil {
		t.Fatalf("ParseRTCPTransportFeedbackFCI: %v", err)
	}
	if parsed.BaseSequenceNumber != feedback.BaseSequenceNumber ||
		parsed.ReferenceTimeTicks != feedback.ReferenceTimeTicks ||
		parsed.FeedbackPacketCount != feedback.FeedbackPacketCount ||
		!parsed.DeltasPresent {
		t.Fatalf("parsed transport feedback header=%+v", parsed)
	}
	if base[0].SequenceNumber != 0xdead {
		t.Fatalf("ParseRTCPTransportFeedbackFCI clobbered dst prefix")
	}
	assertRTCPTransportFeedbackPackets(t, parsed.Packets, feedback.Packets)

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xaa
	appended, err := av1.AppendRTCPTransportFeedbackFCI(prefix, feedback)
	if err != nil {
		t.Fatalf("AppendRTCPTransportFeedbackFCI: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended transport feedback bytes=%#v", appended)
	}
}

func TestRTCPTransportFeedbackFCITwoBitRoundTrip(t *testing.T) {
	feedback := av1.RTCPTransportFeedback{
		BaseSequenceNumber:  0xfffe,
		ReferenceTimeTicks:  0x00ffff,
		FeedbackPacketCount: 0x02,
		DeltasPresent:       true,
		Packets: []av1.RTCPTransportFeedbackPacket{
			{SequenceNumber: 0xfffe, Received: true, DeltaTicks: -4},
			{SequenceNumber: 0xffff},
			{SequenceNumber: 0x0000, Received: true, DeltaTicks: 300},
			{SequenceNumber: 0x0001, Received: true, DeltaTicks: 0},
		},
	}
	buf := make([]byte, 15)
	n, err := av1.PutRTCPTransportFeedbackFCI(buf, feedback)
	if err != nil {
		t.Fatalf("PutRTCPTransportFeedbackFCI: %v", err)
	}
	want := []byte{
		0xff, 0xfe, 0x00, 0x04, 0x00, 0xff, 0xff, 0x02,
		0xe2, 0x40,
		0xff, 0xfc,
		0x01, 0x2c,
		0x00,
	}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("transport feedback two-bit n=%d bytes=%#v want %#v", n, buf, want)
	}
	parsed, err := av1.ParseRTCPTransportFeedbackFCI(buf, make([]av1.RTCPTransportFeedbackPacket, 0, len(feedback.Packets)))
	if err != nil {
		t.Fatalf("ParseRTCPTransportFeedbackFCI: %v", err)
	}
	assertRTCPTransportFeedbackPackets(t, parsed.Packets, feedback.Packets)
}

func TestRTCPTransportFeedbackFCIRunLengthAndNoTimestamp(t *testing.T) {
	missing := av1.RTCPTransportFeedback{
		BaseSequenceNumber:  100,
		ReferenceTimeTicks:  0,
		FeedbackPacketCount: 1,
		DeltasPresent:       true,
		Packets:             make([]av1.RTCPTransportFeedbackPacket, 20),
	}
	for i := range missing.Packets {
		missing.Packets[i].SequenceNumber = missing.BaseSequenceNumber + uint16(i)
	}
	buf := make([]byte, 10)
	if _, err := av1.PutRTCPTransportFeedbackFCI(buf, missing); err != nil {
		t.Fatalf("Put run-length transport feedback: %v", err)
	}
	want := []byte{0x00, 0x64, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x14}
	if string(buf) != string(want) {
		t.Fatalf("run-length transport feedback=%#v want %#v", buf, want)
	}

	noTimestamp := av1.RTCPTransportFeedback{
		BaseSequenceNumber:  10,
		ReferenceTimeTicks:  7,
		FeedbackPacketCount: 3,
		DeltasPresent:       false,
		Packets: []av1.RTCPTransportFeedbackPacket{
			{SequenceNumber: 10, Received: true},
			{SequenceNumber: 11},
			{SequenceNumber: 12, Received: true},
		},
	}
	noTimestampBuf := make([]byte, 10)
	if _, err := av1.PutRTCPTransportFeedbackFCI(noTimestampBuf, noTimestamp); err != nil {
		t.Fatalf("Put no-timestamp transport feedback: %v", err)
	}
	wantNoTimestamp := []byte{0x00, 0x0a, 0x00, 0x03, 0x00, 0x00, 0x07, 0x03, 0xa8, 0x00}
	if string(noTimestampBuf) != string(wantNoTimestamp) {
		t.Fatalf("no-timestamp transport feedback=%#v want %#v", noTimestampBuf, wantNoTimestamp)
	}
	parsed, err := av1.ParseRTCPTransportFeedbackFCI(noTimestampBuf, make([]av1.RTCPTransportFeedbackPacket, 0, 3))
	if err != nil {
		t.Fatalf("Parse no-timestamp transport feedback: %v", err)
	}
	if parsed.DeltasPresent {
		t.Fatalf("parsed no-timestamp feedback reported deltas present")
	}
	assertRTCPTransportFeedbackPackets(t, parsed.Packets, noTimestamp.Packets)
}

func TestRTCPTransportFeedbackFCIRejectsInvalid(t *testing.T) {
	valid := av1.RTCPTransportFeedback{
		BaseSequenceNumber:  1,
		ReferenceTimeTicks:  2,
		FeedbackPacketCount: 3,
		DeltasPresent:       true,
		Packets: []av1.RTCPTransportFeedbackPacket{
			{SequenceNumber: 1, Received: true, DeltaTicks: 4},
		},
	}
	if _, err := av1.PutRTCPTransportFeedbackFCI(make([]byte, 9), valid); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0}, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPTransportFeedbackFCI header err=%v", err)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0}, make([]av1.RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPTransportFeedbackFCI chunk err=%v", err)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 0, 0, 0, 0, 0}, nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("zero status-count ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0x60, 0x01}, make([]av1.RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("reserved run status ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0xf0, 0x00}, make([]av1.RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("reserved vector status ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0xc0, 0x00, 0xff}, make([]av1.RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("truncated nonzero delta ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0x20, 0x01}, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short dst ParseRTCPTransportFeedbackFCI err=%v", err)
	}

	badSequence := valid
	badSequence.Packets = append([]av1.RTCPTransportFeedbackPacket(nil), valid.Packets...)
	badSequence.Packets[0].SequenceNumber = 2
	if _, err := av1.RTCPTransportFeedbackFCISize(badSequence); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad sequence transport feedback err=%v", err)
	}
	badReference := valid
	badReference.ReferenceTimeTicks = av1.RTCPTransportFeedbackMaxReferenceTimeTicks + 1
	if _, err := av1.RTCPTransportFeedbackFCISize(badReference); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad reference time transport feedback err=%v", err)
	}
	badNoTimestamp := valid
	badNoTimestamp.DeltasPresent = false
	if _, err := av1.RTCPTransportFeedbackFCISize(badNoTimestamp); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad no-timestamp transport feedback err=%v", err)
	}
	badMissingDelta := valid
	badMissingDelta.Packets = []av1.RTCPTransportFeedbackPacket{{SequenceNumber: 1, DeltaTicks: 1}}
	if _, err := av1.RTCPTransportFeedbackFCISize(badMissingDelta); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad missing packet delta transport feedback err=%v", err)
	}
	empty := valid
	empty.Packets = nil
	if _, err := av1.RTCPTransportFeedbackFCISize(empty); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("empty transport feedback err=%v", err)
	}
}

func TestRTCPFullIntraRequestEntriesRoundTrip(t *testing.T) {
	entries := []av1.RTCPFullIntraRequestEntry{
		{SSRC: 0x11223344, SequenceNumber: 7},
		{SSRC: 0xaabbccdd, SequenceNumber: 250},
	}
	size, err := av1.RTCPFullIntraRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("RTCPFullIntraRequestEntriesSize: %v", err)
	}
	if size != len(entries)*av1.RTCPFullIntraRequestEntrySize {
		t.Fatalf("FIR size=%d", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPFullIntraRequestEntries(buf, entries)
	if err != nil {
		t.Fatalf("PutRTCPFullIntraRequestEntries: %v", err)
	}
	if n != size {
		t.Fatalf("PutRTCPFullIntraRequestEntries n=%d want %d", n, size)
	}
	want := []byte{0x11, 0x22, 0x33, 0x44, 0x07, 0, 0, 0, 0xaa, 0xbb, 0xcc, 0xdd, 0xfa, 0, 0, 0}
	if string(buf) != string(want) {
		t.Fatalf("encoded FIR bytes=%#v want %#v", buf, want)
	}
	buf[5], buf[6], buf[7] = 0xde, 0xad, 0xbe
	parsed, err := av1.ParseRTCPFullIntraRequestEntries(buf, make([]av1.RTCPFullIntraRequestEntry, 0, len(entries)))
	if err != nil {
		t.Fatalf("ParseRTCPFullIntraRequestEntries: %v", err)
	}
	for i := range entries {
		if parsed[i] != entries[i] {
			t.Fatalf("parsed[%d]=%+v want %+v", i, parsed[i], entries[i])
		}
	}
	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xaa
	appended, err := av1.AppendRTCPFullIntraRequestEntries(prefix, entries)
	if err != nil {
		t.Fatalf("AppendRTCPFullIntraRequestEntries: %v", err)
	}
	clean := make([]byte, size)
	if _, err := av1.PutRTCPFullIntraRequestEntries(clean, entries); err != nil {
		t.Fatalf("Put clean FIR: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(clean) {
		t.Fatalf("appended FIR bytes=%#v", appended)
	}
}

func TestRTCPFullIntraRequestEntriesRejectsInvalid(t *testing.T) {
	entry := av1.RTCPFullIntraRequestEntry{SSRC: 1, SequenceNumber: 2}
	var buf [av1.RTCPFullIntraRequestEntrySize]byte
	if _, err := av1.PutRTCPFullIntraRequestEntry(buf[:1], entry); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPFullIntraRequestEntry err=%v", err)
	}
	if _, _, err := av1.ParseRTCPFullIntraRequestEntry(buf[:1]); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPFullIntraRequestEntry err=%v", err)
	}
	entries := []av1.RTCPFullIntraRequestEntry{entry, entry}
	size, err := av1.RTCPFullIntraRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("RTCPFullIntraRequestEntriesSize valid: %v", err)
	}
	if _, err := av1.PutRTCPFullIntraRequestEntries(make([]byte, size-1), entries); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPFullIntraRequestEntries err=%v", err)
	}
	if out, err := av1.AppendRTCPFullIntraRequestEntries(make([]byte, 0, size-1), entries); !errors.Is(err, av1.ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short AppendRTCPFullIntraRequestEntries out=%d err=%v", len(out), err)
	}
	if _, err := av1.ParseRTCPFullIntraRequestEntries(make([]byte, size), make([]av1.RTCPFullIntraRequestEntry, 0, 1)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPFullIntraRequestEntries dst err=%v", err)
	}
	if _, err := av1.ParseRTCPFullIntraRequestEntries(make([]byte, size-1), nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("ragged ParseRTCPFullIntraRequestEntries err=%v", err)
	}
}

func TestAV1RTCPLayerRefreshLayerIndexRoundTrip(t *testing.T) {
	var buf [2]byte
	index := av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 7, SpatialID: 3}
	n, err := av1.PutAV1RTCPLayerRefreshLayerIndex(buf[:], index)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshLayerIndex: %v", err)
	}
	if n != 2 || buf != [2]byte{0x07, 0x03} {
		t.Fatalf("encoded layer index n=%d bytes=%#v", n, buf)
	}
	got, n, err := av1.ParseAV1RTCPLayerRefreshLayerIndex(buf[:])
	if err != nil {
		t.Fatalf("ParseAV1RTCPLayerRefreshLayerIndex: %v", err)
	}
	if n != 2 || got != index {
		t.Fatalf("parsed layer index n=%d got=%+v want=%+v", n, got, index)
	}

	got, _, err = av1.ParseAV1RTCPLayerRefreshLayerIndex([]byte{0xfa, 0xf1})
	if err != nil {
		t.Fatalf("Parse with reserved bits returned error: %v", err)
	}
	if got != (av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1}) {
		t.Fatalf("reserved-bit parse = %+v", got)
	}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshLayerIndex([]byte{0x00, 0x04}); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("Parse high SID bit error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if _, err := av1.PutAV1RTCPLayerRefreshLayerIndex(buf[:1], index); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Put error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshLayerIndex(buf[:1]); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Parse error = %v, want ErrRTCPShortBuffer", err)
	}
}

func TestAV1RTCPLayerRefreshRequestEntryRoundTrip(t *testing.T) {
	entry := av1.AV1RTCPLayerRefreshRequestEntry{
		SSRC:           0x11223344,
		SequenceNumber: 250,
		PayloadType:    98,
		Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1},
		CurrentPresent: true,
		Current:        av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	var buf [12]byte
	n, err := av1.PutAV1RTCPLayerRefreshRequestEntry(buf[:], entry)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshRequestEntry: %v", err)
	}
	wantBytes := [12]byte{0x11, 0x22, 0x33, 0x44, 0xfa, 0xe2, 0x00, 0x00, 0x02, 0x01, 0x00, 0x00}
	if n != 12 || buf != wantBytes {
		t.Fatalf("encoded LRR entry n=%d bytes=%#v want=%#v", n, buf, wantBytes)
	}
	got, n, err := av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:])
	if err != nil {
		t.Fatalf("ParseAV1RTCPLayerRefreshRequestEntry: %v", err)
	}
	if n != 12 || got != entry {
		t.Fatalf("parsed LRR entry n=%d got=%+v want=%+v", n, got, entry)
	}

	noCurrent := entry
	noCurrent.CurrentPresent = false
	noCurrent.Current = av1.AV1RTCPLayerRefreshLayerIndex{}
	n, err = av1.PutAV1RTCPLayerRefreshRequestEntry(buf[:], noCurrent)
	if err != nil {
		t.Fatalf("Put no-current LRR entry: %v", err)
	}
	if buf[5] != 98 || buf[10] != 0 || buf[11] != 0 {
		t.Fatalf("no-current encoded bytes=%#v", buf)
	}
	buf[10], buf[11] = 0x07, 0x03
	got, _, err = av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:])
	if err != nil {
		t.Fatalf("Parse no-current with ignored current field: %v", err)
	}
	if got.CurrentPresent || got.Current != (av1.AV1RTCPLayerRefreshLayerIndex{}) {
		t.Fatalf("no-current parse got=%+v", got)
	}
}

func TestAV1RTCPLayerRefreshRequestEntriesRoundTrip(t *testing.T) {
	entries := []av1.AV1RTCPLayerRefreshRequestEntry{
		{
			SSRC:           0x11223344,
			SequenceNumber: 250,
			PayloadType:    98,
			Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1},
			CurrentPresent: true,
			Current:        av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
		},
		{
			SSRC:           0x55667788,
			SequenceNumber: 251,
			PayloadType:    98,
			Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 0},
		},
	}
	size, err := av1.AV1RTCPLayerRefreshRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntriesSize: %v", err)
	}
	if size != 2*av1.AV1RTCPLayerRefreshRequestEntrySize {
		t.Fatalf("LRR entries size = %d", size)
	}

	buf := make([]byte, size)
	n, err := av1.PutAV1RTCPLayerRefreshRequestEntries(buf, entries)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshRequestEntries: %v", err)
	}
	if n != size {
		t.Fatalf("PutAV1RTCPLayerRefreshRequestEntries n=%d want %d", n, size)
	}

	parsed, err := av1.ParseAV1RTCPLayerRefreshRequestEntries(buf, make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 2))
	if err != nil {
		t.Fatalf("ParseAV1RTCPLayerRefreshRequestEntries: %v", err)
	}
	if len(parsed) != len(entries) {
		t.Fatalf("parsed entries len=%d want %d", len(parsed), len(entries))
	}
	for i := range entries {
		if parsed[i] != entries[i] {
			t.Fatalf("parsed[%d]=%+v want %+v", i, parsed[i], entries[i])
		}
	}

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xaa
	appended, err := av1.AppendAV1RTCPLayerRefreshRequestEntries(prefix, entries)
	if err != nil {
		t.Fatalf("AppendAV1RTCPLayerRefreshRequestEntries: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa {
		t.Fatalf("appended len/first=%d/%#x", len(appended), appended[0])
	}
	for i := range buf {
		if appended[1+i] != buf[i] {
			t.Fatalf("appended byte %d=%#x want %#x", 1+i, appended[1+i], buf[i])
		}
	}

	empty, err := av1.ParseAV1RTCPLayerRefreshRequestEntries(nil, parsed[:0])
	if err != nil {
		t.Fatalf("Parse empty entries: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("empty parse len=%d", len(empty))
	}
}

func TestAV1RTCPLayerRefreshRequestRejectsInvalid(t *testing.T) {
	valid := av1.AV1RTCPLayerRefreshRequestEntry{
		PayloadType:    98,
		Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 1},
		CurrentPresent: true,
		Current:        av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	for _, entry := range []av1.AV1RTCPLayerRefreshRequestEntry{
		{PayloadType: 128, Target: valid.Target},
		{PayloadType: 98, Target: av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 8}},
		{PayloadType: 98, Target: av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 4}},
		{PayloadType: 98, Target: valid.Target, Current: av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 1}},
		{PayloadType: 98, Target: valid.Current, CurrentPresent: true, Current: valid.Target},
		{PayloadType: 98, Target: valid.Target, CurrentPresent: true, Current: valid.Target},
	} {
		if err := entry.Validate(); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
			t.Fatalf("Validate(%+v) error = %v, want ErrRTCPInvalidLayerRefreshRequest", entry, err)
		}
	}

	var buf [12]byte
	if _, err := av1.PutAV1RTCPLayerRefreshRequestEntry(buf[:1], valid); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Put entry error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:1]); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Parse entry error = %v, want ErrRTCPShortBuffer", err)
	}
	buf = [12]byte{0, 0, 0, 1, 1, 0x80 | 98, 0, 0, 0, 0, 0, 0}
	if _, _, err := av1.ParseAV1RTCPLayerRefreshRequestEntry(buf[:]); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("Parse no-upgrade entry error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}

	validList := []av1.AV1RTCPLayerRefreshRequestEntry{valid, valid}
	validList[1].SSRC = 2
	size, err := av1.AV1RTCPLayerRefreshRequestEntriesSize(validList)
	if err != nil {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntriesSize valid list: %v", err)
	}
	if _, err := av1.PutAV1RTCPLayerRefreshRequestEntries(make([]byte, size-1), validList); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Put entries error = %v, want ErrRTCPShortBuffer", err)
	}
	if out, err := av1.AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, size-1), validList); !errors.Is(err, av1.ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short Append entries out=%d err=%v want ErrRTCPShortBuffer", len(out), err)
	}
	if _, err := av1.ParseAV1RTCPLayerRefreshRequestEntries(make([]byte, size), make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short Parse entries dst error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, err := av1.ParseAV1RTCPLayerRefreshRequestEntries(make([]byte, size-1), nil); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("ragged Parse entries error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	invalidList := []av1.AV1RTCPLayerRefreshRequestEntry{valid}
	invalidList[0].PayloadType = 128
	if _, err := av1.AV1RTCPLayerRefreshRequestEntriesSize(invalidList); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("invalid EntriesSize error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}

	encoded := make([]byte, size)
	if _, err := av1.PutAV1RTCPLayerRefreshRequestEntries(encoded, validList); err != nil {
		t.Fatalf("Put valid list for invalid parse: %v", err)
	}
	copy(encoded[av1.AV1RTCPLayerRefreshRequestEntrySize:], buf[:])
	base := make([]av1.AV1RTCPLayerRefreshRequestEntry, 1, 3)
	base[0].SSRC = 0xdeadbeef
	out, err := av1.ParseAV1RTCPLayerRefreshRequestEntries(encoded, base[:1:cap(base)])
	if !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("invalid Parse entries error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if len(out) != 1 || out[0].SSRC != 0xdeadbeef {
		t.Fatalf("invalid Parse entries returned partial output: %+v", out)
	}
}

func TestEncoderWebRTCValidateLayerRefreshRequest(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(av1.EncoderScalabilityModeL2T2)
	valid := av1.AV1RTCPLayerRefreshRequestEntry{
		PayloadType:    98,
		Target:         av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 1},
		CurrentPresent: true,
		Current:        av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(cfg, valid); err != nil {
		t.Fatalf("EncoderWebRTCValidateLayerRefreshRequest valid: %v", err)
	}

	badTemporal := valid
	badTemporal.Target.TemporalID = 2
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(cfg, badTemporal); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad temporal error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	badSpatial := valid
	badSpatial.Target.SpatialID = 2
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(cfg, badSpatial); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad spatial error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if err := av1.EncoderWebRTCValidateLayerRefreshRequest(av1.EncoderConfig{}, valid); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
		t.Fatalf("invalid config error = %v, want ErrEncoderInvalidConfig", err)
	}

	validList := []av1.AV1RTCPLayerRefreshRequestEntry{valid}
	validList = append(validList, valid)
	validList[1].SSRC = 2
	validList[1].Target = av1.AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 1}
	if err := av1.EncoderWebRTCValidateLayerRefreshRequests(cfg, validList); err != nil {
		t.Fatalf("EncoderWebRTCValidateLayerRefreshRequests valid: %v", err)
	}
	badCurrent := validList
	badCurrent[0].Current.SpatialID = 2
	if err := av1.EncoderWebRTCValidateLayerRefreshRequests(cfg, badCurrent); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad current list error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
}

func assertRTCPTransportFeedbackPackets(
	t *testing.T, got []av1.RTCPTransportFeedbackPacket, want []av1.RTCPTransportFeedbackPacket,
) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("transport feedback packet len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("transport feedback packet[%d]=%+v want %+v", i, got[i], want[i])
		}
	}
}

func testAV1RTCPEncoderConfig(mode av1.EncoderScalabilityMode) av1.EncoderConfig {
	return av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		Scalability:       mode,
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
		RateControl:       av1.EncoderRateControlCBR,
	}
}
