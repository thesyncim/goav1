package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestAV1RTCPFeedbackConstants(t *testing.T) {
	if av1.RTCPVersion != 2 ||
		av1.RTCPHeaderSize != 4 ||
		av1.RTCPMaxPacketSize != 262144 ||
		av1.RTCPPacketCountMax != 31 ||
		av1.RTCPPacketPayloadMaxSize != 262140 ||
		av1.RTCPSenderReportPacketType != 200 ||
		av1.RTCPReceiverReportPacketType != 201 ||
		av1.RTCPSDESPacketType != 202 ||
		av1.RTCPByePacketType != 203 ||
		av1.RTCPReportMaxBlocks != 31 ||
		av1.RTCPByeMaxSources != 31 ||
		av1.RTCPByeReasonMaxTextLen != 0xff ||
		av1.RTCPReportBlockSize != 24 ||
		av1.RTCPReportCumulativeLostMin != -0x800000 ||
		av1.RTCPReportCumulativeLostMax != 0x7fffff ||
		av1.RTCPSenderReportSenderInfoSize != 24 ||
		av1.RTCPSenderReportPacketMinSize != 28 ||
		av1.RTCPReceiverReportPacketMinSize != 8 ||
		av1.RTCPSDESMaxChunks != 31 ||
		av1.RTCPSDESItemEnd != 0 ||
		av1.RTCPSDESItemCNAME != 1 ||
		av1.RTCPSDESItemTool != 6 ||
		av1.RTCPSDESItemPrivate != 8 ||
		av1.RTCPSDESItemMaxTextLen != 0xff ||
		av1.RTCPFeedbackCommonSize != 8 ||
		av1.RTCPFeedbackPacketHeaderSize != 12 ||
		av1.RTCPFeedbackMaxFCISize != 262132 ||
		av1.RTCPRTPFBPacketType != 205 ||
		av1.RTCPPSFBPacketType != 206 {
		t.Fatalf("unexpected RTCP common constants")
	}
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

func TestRTCPPacketRoundTrip(t *testing.T) {
	packet := av1.RTCPPacket{
		PacketType: 204,
		Count:      7,
		Payload:    []byte{1, 2, 3, 4, 5},
	}
	size, err := av1.RTCPPacketSize(len(packet.Payload))
	if err != nil {
		t.Fatalf("RTCPPacketSize: %v", err)
	}
	if size != 12 {
		t.Fatalf("RTCP packet size=%d want 12", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPPacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPPacket: %v", err)
	}
	want := []byte{0xa7, 204, 0x00, 0x02, 1, 2, 3, 4, 5, 0, 0, 3}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("RTCP packet n=%d bytes=%#v want %#v", n, buf, want)
	}
	parsed, consumed, err := av1.ParseRTCPPacket(buf)
	if err != nil {
		t.Fatalf("ParseRTCPPacket: %v", err)
	}
	if consumed != len(buf) ||
		parsed.PacketType != packet.PacketType ||
		parsed.Count != packet.Count ||
		string(parsed.Payload) != string(packet.Payload) ||
		parsed.Padding != 3 {
		t.Fatalf("parsed RTCP packet consumed=%d packet=%+v", consumed, parsed)
	}

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xaa
	appended, err := av1.AppendRTCPPacket(prefix, packet)
	if err != nil {
		t.Fatalf("AppendRTCPPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended RTCP packet=%#v", appended)
	}
}

func TestRTCPCompoundPackets(t *testing.T) {
	sr := make([]byte, av1.RTCPSenderReportPacketMinSize)
	if _, err := av1.PutRTCPSenderReportPacket(sr, av1.RTCPSenderReport{SenderSSRC: 0x11111111}); err != nil {
		t.Fatalf("PutRTCPSenderReportPacket: %v", err)
	}
	sdesPacket := av1.RTCPSDESPacket{Chunks: []av1.RTCPSDESChunk{{
		Source: 0x22222222,
		Items:  []av1.RTCPSDESItem{{Type: av1.RTCPSDESItemCNAME, Text: []byte("x")}},
	}}}
	sdes := make([]byte, 12)
	if _, err := av1.PutRTCPSDESPacket(sdes, sdesPacket); err != nil {
		t.Fatalf("PutRTCPSDESPacket: %v", err)
	}
	pli := make([]byte, av1.RTCPFeedbackPacketHeaderSize)
	if _, err := av1.PutRTCPFeedbackPacket(pli, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x33333333,
		MediaSSRC:  0x44444444,
	}); err != nil {
		t.Fatalf("PutRTCPFeedbackPacket: %v", err)
	}
	bye := make([]byte, 8)
	if _, err := av1.PutRTCPByePacket(bye, av1.RTCPByePacket{Sources: []uint32{0x55555555}}); err != nil {
		t.Fatalf("PutRTCPByePacket: %v", err)
	}
	unknown := make([]byte, 12)
	if _, err := av1.PutRTCPPacket(unknown, av1.RTCPPacket{
		PacketType: 204,
		Count:      7,
		Payload:    []byte{9, 8, 7, 6, 5},
	}); err != nil {
		t.Fatalf("PutRTCPPacket unknown: %v", err)
	}
	compound := append(append(append(append(append([]byte{}, sr...), sdes...), pli...), bye...), unknown...)
	prefix := make([]av1.RTCPPacket, 1, 6)
	prefix[0].PacketType = 255
	packets, err := av1.ParseRTCPCompoundPackets(compound, prefix[:1:6])
	if err != nil {
		t.Fatalf("ParseRTCPCompoundPackets: %v", err)
	}
	if prefix[0].PacketType != 255 {
		t.Fatalf("ParseRTCPCompoundPackets clobbered prefix: %+v", prefix[0])
	}
	if len(packets) != 5 {
		t.Fatalf("compound packet count=%d want 5", len(packets))
	}
	wantTypes := []uint8{
		av1.RTCPSenderReportPacketType,
		av1.RTCPSDESPacketType,
		av1.RTCPPSFBPacketType,
		av1.RTCPByePacketType,
		204,
	}
	wantCounts := []uint8{0, 1, av1.RTCPPSFBPictureLossIndicationFMT, 1, 7}
	wantPayloadLens := []int{24, 8, 8, 4, 5}
	wantPadding := []int{0, 0, 0, 0, 3}
	for i := range packets {
		if packets[i].PacketType != wantTypes[i] ||
			packets[i].Count != wantCounts[i] ||
			len(packets[i].Payload) != wantPayloadLens[i] ||
			packets[i].Padding != wantPadding[i] {
			t.Fatalf("packet %d=%+v want type=%d count=%d payload=%d padding=%d",
				i, packets[i], wantTypes[i], wantCounts[i], wantPayloadLens[i], wantPadding[i])
		}
	}
}

func TestRTCPPacketRejectsInvalid(t *testing.T) {
	if _, err := av1.RTCPPacketSize(-1); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("negative RTCPPacketSize err=%v", err)
	}
	if _, err := av1.RTCPPacketSize(av1.RTCPPacketPayloadMaxSize + 1); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("oversized RTCPPacketSize err=%v", err)
	}
	badCount := av1.RTCPPacket{Count: av1.RTCPPacketCountMax + 1}
	if _, err := av1.PutRTCPPacket(make([]byte, av1.RTCPHeaderSize), badCount); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad count PutRTCPPacket err=%v", err)
	}
	if out, err := av1.AppendRTCPPacket(make([]byte, 0, av1.RTCPHeaderSize), badCount); !errors.Is(err, av1.ErrRTCPInvalidPacket) || len(out) != 0 {
		t.Fatalf("bad count AppendRTCPPacket out=%d err=%v", len(out), err)
	}
	if _, err := av1.PutRTCPPacket(make([]byte, av1.RTCPHeaderSize-1), av1.RTCPPacket{}); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPPacket err=%v", err)
	}
	if _, _, err := av1.ParseRTCPPacket(nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPPacket err=%v", err)
	}
	badVersion := []byte{0x40, 204, 0, 0}
	if _, _, err := av1.ParseRTCPPacket(badVersion); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad version ParseRTCPPacket err=%v", err)
	}
	truncated := []byte{0x80, 204, 0, 1}
	if _, _, err := av1.ParseRTCPPacket(truncated); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("truncated ParseRTCPPacket err=%v", err)
	}
	zeroPadding := []byte{0xa0, 204, 0, 0}
	if _, _, err := av1.ParseRTCPPacket(zeroPadding); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("zero padding ParseRTCPPacket err=%v", err)
	}
	tooMuchPadding := []byte{0xa0, 204, 0, 1, 1, 2, 3, 8}
	if _, _, err := av1.ParseRTCPPacket(tooMuchPadding); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("too much padding ParseRTCPPacket err=%v", err)
	}
	if _, err := av1.ParseRTCPCompoundPackets(nil, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("empty ParseRTCPCompoundPackets err=%v", err)
	}
	valid := []byte{0x80, 204, 0, 0}
	if _, err := av1.ParseRTCPCompoundPackets(valid, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short dst ParseRTCPCompoundPackets err=%v", err)
	}
	compoundTruncated := append(append([]byte{}, valid...), truncated...)
	if _, err := av1.ParseRTCPCompoundPackets(compoundTruncated, make([]av1.RTCPPacket, 0, 2)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("truncated compound ParseRTCPCompoundPackets err=%v", err)
	}
}

func TestRTCPByePacketRoundTrip(t *testing.T) {
	packet := av1.RTCPByePacket{
		Sources: []uint32{0x11111111, 0x22222222},
		Reason:  []byte("done"),
	}
	size, err := av1.RTCPByePacketSize(packet.Sources, packet.Reason)
	if err != nil {
		t.Fatalf("RTCPByePacketSize: %v", err)
	}
	if size != 20 {
		t.Fatalf("BYE packet size=%d want 20", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPByePacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPByePacket: %v", err)
	}
	want := []byte{
		0x82, 203, 0x00, 0x04,
		0x11, 0x11, 0x11, 0x11,
		0x22, 0x22, 0x22, 0x22,
		0x04, 'd', 'o', 'n', 'e', 0x00, 0x00, 0x00,
	}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("BYE packet n=%d bytes=%#v want %#v", n, buf, want)
	}
	dst := make([]uint32, 1, 3)
	dst[0] = 0x99999999
	parsed, consumed, err := av1.ParseRTCPByePacket(buf, dst[:1:3])
	if err != nil {
		t.Fatalf("ParseRTCPByePacket: %v", err)
	}
	if consumed != len(buf) || dst[0] != 0x99999999 ||
		len(parsed.Sources) != 2 || parsed.Sources[0] != packet.Sources[0] ||
		parsed.Sources[1] != packet.Sources[1] || string(parsed.Reason) != string(packet.Reason) {
		t.Fatalf("parsed BYE consumed=%d packet=%+v dst0=%#x", consumed, parsed, dst[0])
	}
	storage := dst[:cap(dst)]
	if len(parsed.Sources) == 0 || &parsed.Sources[0] != &storage[1] {
		t.Fatalf("parsed BYE sources did not alias caller buffer")
	}

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xbe
	appended, err := av1.AppendRTCPByePacket(prefix, packet)
	if err != nil {
		t.Fatalf("AppendRTCPByePacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xbe || string(appended[1:]) != string(buf) {
		t.Fatalf("appended BYE packet=%#v", appended)
	}
}

func TestRTCPByePacketWithoutReason(t *testing.T) {
	packet := av1.RTCPByePacket{Sources: []uint32{0x01020304}}
	size, err := av1.RTCPByePacketSize(packet.Sources, nil)
	if err != nil {
		t.Fatalf("RTCPByePacketSize no reason: %v", err)
	}
	if size != 8 {
		t.Fatalf("BYE no reason size=%d want 8", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPByePacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPByePacket no reason: %v", err)
	}
	want := []byte{0x81, 203, 0x00, 0x01, 0x01, 0x02, 0x03, 0x04}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("BYE no reason n=%d bytes=%#v want %#v", n, buf, want)
	}
	parsed, consumed, err := av1.ParseRTCPByePacket(buf, make([]uint32, 0, 1))
	if err != nil {
		t.Fatalf("ParseRTCPByePacket no reason: %v", err)
	}
	if consumed != len(buf) || len(parsed.Sources) != 1 || parsed.Sources[0] != 0x01020304 || len(parsed.Reason) != 0 {
		t.Fatalf("parsed BYE no reason consumed=%d packet=%+v", consumed, parsed)
	}
}

func TestRTCPByePacketRejectsInvalid(t *testing.T) {
	tooMany := make([]uint32, av1.RTCPByeMaxSources+1)
	if _, err := av1.RTCPByePacketSize(tooMany, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("too many BYE sources size err=%v", err)
	}
	if _, err := av1.PutRTCPByePacket(make([]byte, 4096), av1.RTCPByePacket{Reason: make([]byte, av1.RTCPByeReasonMaxTextLen+1)}); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("too long BYE reason put err=%v", err)
	}
	if _, err := av1.PutRTCPByePacket(make([]byte, av1.RTCPHeaderSize-1), av1.RTCPByePacket{}); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short BYE put err=%v", err)
	}
	if out, err := av1.AppendRTCPByePacket(make([]byte, 0, av1.RTCPHeaderSize-1), av1.RTCPByePacket{}); !errors.Is(err, av1.ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short BYE append out=%d err=%v", len(out), err)
	}
	if _, _, err := av1.ParseRTCPByePacket(nil, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short BYE parse err=%v", err)
	}
	badVersion := []byte{0x40, 203, 0, 0}
	if _, _, err := av1.ParseRTCPByePacket(badVersion, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad version BYE parse err=%v", err)
	}
	badType := []byte{0x80, 202, 0, 0}
	if _, _, err := av1.ParseRTCPByePacket(badType, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad type BYE parse err=%v", err)
	}
	truncated := []byte{0x80, 203, 0, 1}
	if _, _, err := av1.ParseRTCPByePacket(truncated, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("truncated BYE parse err=%v", err)
	}
	missingSource := []byte{0x81, 203, 0, 0}
	if _, _, err := av1.ParseRTCPByePacket(missingSource, make([]uint32, 0, 1)); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("missing BYE source parse err=%v", err)
	}
	validOneSource := []byte{0x81, 203, 0, 1, 0, 0, 0, 1}
	if _, _, err := av1.ParseRTCPByePacket(validOneSource, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short BYE source dst parse err=%v", err)
	}
	badReasonLength := []byte{0x80, 203, 0, 1, 4, 'x', 0, 0}
	if _, _, err := av1.ParseRTCPByePacket(badReasonLength, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad BYE reason length parse err=%v", err)
	}
	badReasonPadding := []byte{0x80, 203, 0, 1, 1, 'x', 1, 0}
	if _, _, err := av1.ParseRTCPByePacket(badReasonPadding, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad BYE reason padding parse err=%v", err)
	}
	validPacketPadding := []byte{0xa0, 203, 0, 1, 0, 0, 0, 4}
	parsed, consumed, err := av1.ParseRTCPByePacket(validPacketPadding, nil)
	if err != nil {
		t.Fatalf("valid padded BYE parse: %v", err)
	}
	if consumed != len(validPacketPadding) || len(parsed.Sources) != 0 || len(parsed.Reason) != 0 {
		t.Fatalf("valid padded BYE consumed=%d packet=%+v", consumed, parsed)
	}
	zeroPacketPadding := []byte{0xa0, 203, 0, 1, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPByePacket(zeroPacketPadding, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("zero BYE packet padding err=%v", err)
	}
}

func TestRTCPSDESPacketRoundTrip(t *testing.T) {
	packet := av1.RTCPSDESPacket{Chunks: []av1.RTCPSDESChunk{
		{
			Source: 0x11111111,
			Items: []av1.RTCPSDESItem{
				{Type: av1.RTCPSDESItemCNAME, Text: []byte("alice")},
				{Type: av1.RTCPSDESItemTool, Text: []byte("goav1")},
			},
		},
		{
			Source: 0x22222222,
			Items: []av1.RTCPSDESItem{
				{Type: av1.RTCPSDESItemCNAME, Text: []byte("b")},
			},
		},
	}}
	size, err := av1.RTCPSDESPacketSize(packet.Chunks)
	if err != nil {
		t.Fatalf("RTCPSDESPacketSize: %v", err)
	}
	if size != 32 {
		t.Fatalf("SDES packet size=%d want 32", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPSDESPacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPSDESPacket: %v", err)
	}
	want := []byte{
		0x82, 202, 0x00, 0x07,
		0x11, 0x11, 0x11, 0x11,
		0x01, 0x05, 'a', 'l', 'i', 'c', 'e',
		0x06, 0x05, 'g', 'o', 'a', 'v', '1',
		0x00, 0x00,
		0x22, 0x22, 0x22, 0x22,
		0x01, 0x01, 'b',
		0x00,
	}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("SDES packet n=%d bytes=%#v want %#v", n, buf, want)
	}
	chunkDst := make([]av1.RTCPSDESChunk, 1, 3)
	chunkDst[0].Source = 0x99999999
	itemDst := make([]av1.RTCPSDESItem, 1, 4)
	itemDst[0] = av1.RTCPSDESItem{Type: av1.RTCPSDESItemName, Text: []byte("prefix")}
	parsed, consumed, err := av1.ParseRTCPSDESPacket(buf, chunkDst[:1:3], itemDst[:1:4])
	if err != nil {
		t.Fatalf("ParseRTCPSDESPacket: %v", err)
	}
	if consumed != len(buf) || len(parsed.Chunks) != len(packet.Chunks) {
		t.Fatalf("parsed SDES consumed=%d packet=%+v", consumed, parsed)
	}
	if chunkDst[0].Source != 0x99999999 || itemDst[0].Type != av1.RTCPSDESItemName {
		t.Fatalf("ParseRTCPSDESPacket clobbered prefixes chunks=%+v items=%+v", chunkDst[0], itemDst[0])
	}
	assertRTCPSDESPacketEqual(t, parsed, packet)
	itemStorage := itemDst[:cap(itemDst)]
	if len(parsed.Chunks[0].Items) == 0 || &parsed.Chunks[0].Items[0] != &itemStorage[1] {
		t.Fatalf("parsed SDES items did not alias caller item buffer")
	}

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xef
	appended, err := av1.AppendRTCPSDESPacket(prefix, packet)
	if err != nil {
		t.Fatalf("AppendRTCPSDESPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xef || string(appended[1:]) != string(buf) {
		t.Fatalf("appended SDES packet=%#v", appended)
	}
}

func TestRTCPSDESPacketRejectsInvalid(t *testing.T) {
	tooMany := av1.RTCPSDESPacket{Chunks: make([]av1.RTCPSDESChunk, av1.RTCPSDESMaxChunks+1)}
	if _, err := av1.RTCPSDESPacketSize(tooMany.Chunks); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("too many SDES chunks size err=%v", err)
	}
	badEndItem := av1.RTCPSDESPacket{Chunks: []av1.RTCPSDESChunk{{Items: []av1.RTCPSDESItem{{Type: av1.RTCPSDESItemEnd}}}}}
	if _, err := av1.RTCPSDESPacketSize(badEndItem.Chunks); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("SDES end item size err=%v", err)
	}
	tooLong := av1.RTCPSDESPacket{Chunks: []av1.RTCPSDESChunk{{Items: []av1.RTCPSDESItem{{Type: av1.RTCPSDESItemCNAME, Text: make([]byte, av1.RTCPSDESItemMaxTextLen+1)}}}}}
	if _, err := av1.PutRTCPSDESPacket(make([]byte, 4096), tooLong); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("too long SDES item put err=%v", err)
	}
	if _, err := av1.PutRTCPSDESPacket(make([]byte, av1.RTCPHeaderSize-1), av1.RTCPSDESPacket{}); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short SDES put err=%v", err)
	}
	if out, err := av1.AppendRTCPSDESPacket(make([]byte, 0, av1.RTCPHeaderSize-1), av1.RTCPSDESPacket{}); !errors.Is(err, av1.ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short SDES append out=%d err=%v", len(out), err)
	}
	if _, _, err := av1.ParseRTCPSDESPacket(nil, nil, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short SDES parse err=%v", err)
	}
	badVersion := []byte{0x41, 202, 0, 0}
	if _, _, err := av1.ParseRTCPSDESPacket(badVersion, nil, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad version SDES parse err=%v", err)
	}
	badType := []byte{0x80, 201, 0, 0}
	if _, _, err := av1.ParseRTCPSDESPacket(badType, nil, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad type SDES parse err=%v", err)
	}
	truncated := []byte{0x80, 202, 0, 1}
	if _, _, err := av1.ParseRTCPSDESPacket(truncated, nil, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("truncated SDES parse err=%v", err)
	}
	missingChunk := []byte{0x81, 202, 0, 0}
	if _, _, err := av1.ParseRTCPSDESPacket(missingChunk, make([]av1.RTCPSDESChunk, 0, 1), nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("missing SDES chunk parse err=%v", err)
	}
	validOneChunk := []byte{0x81, 202, 0, 1, 0, 0, 0, 1}
	if _, _, err := av1.ParseRTCPSDESPacket(validOneChunk, nil, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short SDES chunk dst parse err=%v", err)
	}
	validOneItem := []byte{0x81, 202, 0, 2, 0, 0, 0, 1, av1.RTCPSDESItemCNAME, 1, 'x', 0}
	if _, _, err := av1.ParseRTCPSDESPacket(validOneItem, make([]av1.RTCPSDESChunk, 0, 1), nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short SDES item dst parse err=%v", err)
	}
	noEnd := []byte{0x81, 202, 0, 1, 0, 0, 0, 1}
	if _, _, err := av1.ParseRTCPSDESPacket(noEnd, make([]av1.RTCPSDESChunk, 0, 1), nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("missing SDES end parse err=%v", err)
	}
	badItemLength := []byte{0x81, 202, 0, 2, 0, 0, 0, 1, av1.RTCPSDESItemCNAME, 4, 'x', 0}
	if _, _, err := av1.ParseRTCPSDESPacket(badItemLength, make([]av1.RTCPSDESChunk, 0, 1), make([]av1.RTCPSDESItem, 0, 1)); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad SDES item length parse err=%v", err)
	}
	badChunkPadding := []byte{0x81, 202, 0, 2, 0, 0, 0, 1, 0, 1, 0, 0}
	if _, _, err := av1.ParseRTCPSDESPacket(badChunkPadding, make([]av1.RTCPSDESChunk, 0, 1), nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad SDES chunk padding parse err=%v", err)
	}
	validPadded := []byte{0xa0, 202, 0, 1, 0, 0, 0, 4}
	parsed, consumed, err := av1.ParseRTCPSDESPacket(validPadded, nil, nil)
	if err != nil {
		t.Fatalf("valid padded SDES parse: %v", err)
	}
	if consumed != len(validPadded) || len(parsed.Chunks) != 0 {
		t.Fatalf("valid padded SDES consumed=%d packet=%+v", consumed, parsed)
	}
	zeroPadding := []byte{0xa0, 202, 0, 1, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPSDESPacket(zeroPadding, nil, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("zero SDES packet padding err=%v", err)
	}
}

func assertRTCPSDESPacketEqual(t *testing.T, got av1.RTCPSDESPacket, want av1.RTCPSDESPacket) {
	t.Helper()
	if len(got.Chunks) != len(want.Chunks) {
		t.Fatalf("SDES chunks=%d want %d", len(got.Chunks), len(want.Chunks))
	}
	for i := range want.Chunks {
		gotChunk := got.Chunks[i]
		wantChunk := want.Chunks[i]
		if gotChunk.Source != wantChunk.Source || len(gotChunk.Items) != len(wantChunk.Items) {
			t.Fatalf("SDES chunk %d=%+v want %+v", i, gotChunk, wantChunk)
		}
		for j := range wantChunk.Items {
			gotItem := gotChunk.Items[j]
			wantItem := wantChunk.Items[j]
			if gotItem.Type != wantItem.Type || string(gotItem.Text) != string(wantItem.Text) {
				t.Fatalf("SDES chunk %d item %d=%+v want %+v", i, j, gotItem, wantItem)
			}
		}
	}
}

func TestRTCPSenderReportPacketRoundTrip(t *testing.T) {
	report := av1.RTCPSenderReport{
		SenderSSRC:        0x11223344,
		NTPSeconds:        0x01020304,
		NTPFraction:       0x05060708,
		RTPTimestamp:      0x090a0b0c,
		SenderPacketCount: 0x0d0e0f10,
		SenderOctetCount:  0x11121314,
		Reports: []av1.RTCPReportBlock{
			{
				SSRC:                          0x22334455,
				FractionLost:                  7,
				CumulativePacketsLost:         -3,
				ExtendedHighestSequenceNumber: 0x01020304,
				InterarrivalJitter:            0x05060708,
				LastSenderReport:              0x090a0b0c,
				DelaySinceLastSenderReport:    0x0d0e0f10,
			},
			{
				SSRC:                          0xaabbccdd,
				FractionLost:                  0xee,
				CumulativePacketsLost:         0x123456,
				ExtendedHighestSequenceNumber: 0x01000002,
				InterarrivalJitter:            0x03000004,
				LastSenderReport:              0x05000006,
				DelaySinceLastSenderReport:    0x07000008,
			},
		},
	}
	size, err := av1.RTCPSenderReportPacketSize(len(report.Reports))
	if err != nil {
		t.Fatalf("RTCPSenderReportPacketSize: %v", err)
	}
	if size != 76 {
		t.Fatalf("sender report size=%d want 76", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPSenderReportPacket(buf, report)
	if err != nil {
		t.Fatalf("PutRTCPSenderReportPacket: %v", err)
	}
	want := []byte{
		0x82, 200, 0x00, 0x12,
		0x11, 0x22, 0x33, 0x44,
		0x01, 0x02, 0x03, 0x04,
		0x05, 0x06, 0x07, 0x08,
		0x09, 0x0a, 0x0b, 0x0c,
		0x0d, 0x0e, 0x0f, 0x10,
		0x11, 0x12, 0x13, 0x14,
		0x22, 0x33, 0x44, 0x55,
		0x07, 0xff, 0xff, 0xfd,
		0x01, 0x02, 0x03, 0x04,
		0x05, 0x06, 0x07, 0x08,
		0x09, 0x0a, 0x0b, 0x0c,
		0x0d, 0x0e, 0x0f, 0x10,
		0xaa, 0xbb, 0xcc, 0xdd,
		0xee, 0x12, 0x34, 0x56,
		0x01, 0x00, 0x00, 0x02,
		0x03, 0x00, 0x00, 0x04,
		0x05, 0x00, 0x00, 0x06,
		0x07, 0x00, 0x00, 0x08,
	}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("sender report n=%d bytes=%#v want %#v", n, buf, want)
	}

	dst := make([]av1.RTCPReportBlock, 1, 3)
	dst[0].SSRC = 0x99999999
	parsed, consumed, err := av1.ParseRTCPSenderReportPacket(buf, dst[:1:3])
	if err != nil {
		t.Fatalf("ParseRTCPSenderReportPacket: %v", err)
	}
	if consumed != len(buf) ||
		parsed.SenderSSRC != report.SenderSSRC ||
		parsed.NTPSeconds != report.NTPSeconds ||
		parsed.NTPFraction != report.NTPFraction ||
		parsed.RTPTimestamp != report.RTPTimestamp ||
		parsed.SenderPacketCount != report.SenderPacketCount ||
		parsed.SenderOctetCount != report.SenderOctetCount {
		t.Fatalf("parsed sender report consumed=%d report=%+v", consumed, parsed)
	}
	if dst[0].SSRC != 0x99999999 {
		t.Fatalf("ParseRTCPSenderReportPacket clobbered dst prefix")
	}
	assertRTCPReportBlocksEqual(t, parsed.Reports, report.Reports)

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xab
	appended, err := av1.AppendRTCPSenderReportPacket(prefix, report)
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xab || string(appended[1:]) != string(buf) {
		t.Fatalf("appended sender report=%#v", appended)
	}
}

func TestRTCPReceiverReportPacketRoundTrip(t *testing.T) {
	report := av1.RTCPReceiverReport{
		SenderSSRC: 0x10203040,
		Reports: []av1.RTCPReportBlock{{
			SSRC:                          0x01020304,
			FractionLost:                  0x20,
			CumulativePacketsLost:         av1.RTCPReportCumulativeLostMax,
			ExtendedHighestSequenceNumber: 0x11121314,
			InterarrivalJitter:            0x21222324,
			LastSenderReport:              0x31323334,
			DelaySinceLastSenderReport:    0x41424344,
		}},
	}
	size, err := av1.RTCPReceiverReportPacketSize(len(report.Reports))
	if err != nil {
		t.Fatalf("RTCPReceiverReportPacketSize: %v", err)
	}
	if size != 32 {
		t.Fatalf("receiver report size=%d want 32", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPReceiverReportPacket(buf, report)
	if err != nil {
		t.Fatalf("PutRTCPReceiverReportPacket: %v", err)
	}
	want := []byte{
		0x81, 201, 0x00, 0x07,
		0x10, 0x20, 0x30, 0x40,
		0x01, 0x02, 0x03, 0x04,
		0x20, 0x7f, 0xff, 0xff,
		0x11, 0x12, 0x13, 0x14,
		0x21, 0x22, 0x23, 0x24,
		0x31, 0x32, 0x33, 0x34,
		0x41, 0x42, 0x43, 0x44,
	}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("receiver report n=%d bytes=%#v want %#v", n, buf, want)
	}
	parsed, consumed, err := av1.ParseRTCPReceiverReportPacket(buf, make([]av1.RTCPReportBlock, 0, 1))
	if err != nil {
		t.Fatalf("ParseRTCPReceiverReportPacket: %v", err)
	}
	if consumed != len(buf) || parsed.SenderSSRC != report.SenderSSRC {
		t.Fatalf("parsed receiver report consumed=%d report=%+v", consumed, parsed)
	}
	assertRTCPReportBlocksEqual(t, parsed.Reports, report.Reports)

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xcd
	appended, err := av1.AppendRTCPReceiverReportPacket(prefix, report)
	if err != nil {
		t.Fatalf("AppendRTCPReceiverReportPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xcd || string(appended[1:]) != string(buf) {
		t.Fatalf("appended receiver report=%#v", appended)
	}
}

func TestRTCPReportPacketRejectsInvalid(t *testing.T) {
	if _, err := av1.RTCPSenderReportPacketSize(-1); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("negative sender report size err=%v", err)
	}
	if _, err := av1.RTCPReceiverReportPacketSize(av1.RTCPReportMaxBlocks + 1); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("too many receiver report blocks size err=%v", err)
	}
	report := av1.RTCPReceiverReport{Reports: []av1.RTCPReportBlock{{CumulativePacketsLost: av1.RTCPReportCumulativeLostMax + 1}}}
	if _, err := av1.PutRTCPReceiverReportPacket(make([]byte, av1.RTCPReceiverReportPacketMinSize+av1.RTCPReportBlockSize), report); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("invalid cumulative lost receiver report err=%v", err)
	}
	report.Reports[0].CumulativePacketsLost = av1.RTCPReportCumulativeLostMin - 1
	if _, err := av1.PutRTCPReceiverReportPacket(make([]byte, av1.RTCPReceiverReportPacketMinSize+av1.RTCPReportBlockSize), report); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("invalid negative cumulative lost receiver report err=%v", err)
	}
	if _, err := av1.PutRTCPReceiverReportPacket(make([]byte, av1.RTCPReceiverReportPacketMinSize-1), av1.RTCPReceiverReport{}); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short receiver report put err=%v", err)
	}
	if out, err := av1.AppendRTCPReceiverReportPacket(make([]byte, 0, av1.RTCPReceiverReportPacketMinSize-1), av1.RTCPReceiverReport{}); !errors.Is(err, av1.ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short receiver report append out=%d err=%v", len(out), err)
	}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(nil, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short receiver report parse err=%v", err)
	}
	if _, _, err := av1.ParseRTCPSenderReportPacket(make([]byte, av1.RTCPSenderReportPacketMinSize-1), nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short sender report parse err=%v", err)
	}
	badVersion := []byte{0x40, 201, 0, 1, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(badVersion, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad version receiver report parse err=%v", err)
	}
	badType := []byte{0x80, 200, 0, 1, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(badType, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("bad type receiver report parse err=%v", err)
	}
	lengthTooShort := []byte{0x80, 201, 0, 0, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(lengthTooShort, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("short length receiver report parse err=%v", err)
	}
	truncated := []byte{0x80, 201, 0, 7, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(truncated, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("truncated receiver report parse err=%v", err)
	}
	extraWithoutPadding := []byte{0x80, 201, 0, 2, 0, 0, 0, 0, 1, 2, 3, 4}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(extraWithoutPadding, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("extra receiver report parse err=%v", err)
	}
	validPadded := []byte{0xa0, 201, 0, 2, 0, 0, 0, 1, 0, 0, 0, 4}
	parsed, consumed, err := av1.ParseRTCPReceiverReportPacket(validPadded, nil)
	if err != nil {
		t.Fatalf("valid padded receiver report parse: %v", err)
	}
	if consumed != len(validPadded) || parsed.SenderSSRC != 1 || len(parsed.Reports) != 0 {
		t.Fatalf("valid padded receiver report consumed=%d report=%+v", consumed, parsed)
	}
	zeroPadding := []byte{0xa0, 201, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(zeroPadding, nil); !errors.Is(err, av1.ErrRTCPInvalidPacket) {
		t.Fatalf("zero padding receiver report parse err=%v", err)
	}
	if _, _, err := av1.ParseRTCPReceiverReportPacket(bufWithOneReport(), nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short report dst parse err=%v", err)
	}
}

func bufWithOneReport() []byte {
	buf := make([]byte, av1.RTCPReceiverReportPacketMinSize+av1.RTCPReportBlockSize)
	buf[0] = 0x81
	buf[1] = av1.RTCPReceiverReportPacketType
	buf[3] = 0x07
	return buf
}

func assertRTCPReportBlocksEqual(t *testing.T, got []av1.RTCPReportBlock, want []av1.RTCPReportBlock) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("report block len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("report block %d=%+v want %+v", i, got[i], want[i])
		}
	}
}

func TestRTCPFeedbackPacketRoundTrip(t *testing.T) {
	pli := av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x11223344,
		MediaSSRC:  0x55667788,
	}
	size, err := av1.RTCPFeedbackPacketSize(len(pli.FCI))
	if err != nil {
		t.Fatalf("RTCPFeedbackPacketSize PLI: %v", err)
	}
	if size != av1.RTCPFeedbackPacketHeaderSize {
		t.Fatalf("PLI packet size=%d want %d", size, av1.RTCPFeedbackPacketHeaderSize)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPFeedbackPacket(buf, pli)
	if err != nil {
		t.Fatalf("PutRTCPFeedbackPacket PLI: %v", err)
	}
	wantPLI := []byte{0x81, 206, 0x00, 0x02, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}
	if n != len(wantPLI) || string(buf) != string(wantPLI) {
		t.Fatalf("PLI packet n=%d bytes=%#v want %#v", n, buf, wantPLI)
	}
	parsed, consumed, err := av1.ParseRTCPFeedbackPacket(buf)
	if err != nil {
		t.Fatalf("ParseRTCPFeedbackPacket PLI: %v", err)
	}
	if consumed != len(buf) || parsed.PacketType != pli.PacketType || parsed.FMT != pli.FMT ||
		parsed.SenderSSRC != pli.SenderSSRC || parsed.MediaSSRC != pli.MediaSSRC || len(parsed.FCI) != 0 {
		t.Fatalf("parsed PLI consumed=%d packet=%+v", consumed, parsed)
	}

	prefix := make([]byte, 1, 1+len(buf))
	prefix[0] = 0xaa
	appended, err := av1.AppendRTCPFeedbackPacket(prefix, pli)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket PLI: %v", err)
	}
	if len(appended) != 1+len(buf) || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended PLI packet=%#v", appended)
	}
}

func TestRTCPFeedbackPacketWrapsREMBVector(t *testing.T) {
	rembFCI := []byte{
		'R', 'E', 'M', 'B',
		0x03, 0x07, 0xfb, 0x93,
		0x23, 0x45, 0x67, 0x89,
		0x23, 0x45, 0x67, 0x8a,
		0x23, 0x45, 0x67, 0x8b,
	}
	packet := av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBApplicationLayerFeedbackFMT,
		SenderSSRC: 0x12345678,
		MediaSSRC:  0,
		FCI:        rembFCI,
	}
	buf := make([]byte, 32)
	n, err := av1.PutRTCPFeedbackPacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPFeedbackPacket REMB: %v", err)
	}
	want := []byte{0x8f, 206, 0x00, 0x07, 0x12, 0x34, 0x56, 0x78,
		0x00, 0x00, 0x00, 0x00, 'R', 'E', 'M', 'B',
		0x03, 0x07, 0xfb, 0x93, 0x23, 0x45, 0x67, 0x89,
		0x23, 0x45, 0x67, 0x8a, 0x23, 0x45, 0x67, 0x8b}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("REMB packet n=%d bytes=%#v want %#v", n, buf, want)
	}
	parsed, consumed, err := av1.ParseRTCPFeedbackPacket(buf)
	if err != nil {
		t.Fatalf("ParseRTCPFeedbackPacket REMB: %v", err)
	}
	if consumed != len(buf) || parsed.PacketType != packet.PacketType || parsed.FMT != packet.FMT ||
		parsed.SenderSSRC != packet.SenderSSRC || parsed.MediaSSRC != packet.MediaSSRC ||
		string(parsed.FCI) != string(rembFCI) {
		t.Fatalf("parsed REMB consumed=%d packet=%+v", consumed, parsed)
	}
	if _, err := av1.ParseRTCPReceiverEstimatedMaximumBitrateFCI(parsed.FCI, make([]uint32, 0, 3)); err != nil {
		t.Fatalf("ParseRTCPReceiverEstimatedMaximumBitrateFCI from packet: %v", err)
	}
}

func TestRTCPFeedbackPacketWrapsPaddedTransportFeedback(t *testing.T) {
	transportFCI := []byte{0x00, 0x01, 0x00, 0x05, 0x01, 0x02, 0x03, 0x7a, 0xb2, 0x00, 0x04, 0x04, 0x04}
	packet := av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPRTPFBPacketType,
		FMT:        av1.RTCPRTPFBTransportFeedbackFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        transportFCI,
	}
	size, err := av1.RTCPFeedbackPacketSize(len(transportFCI))
	if err != nil {
		t.Fatalf("RTCPFeedbackPacketSize transport feedback: %v", err)
	}
	if size != 28 {
		t.Fatalf("transport feedback packet size=%d want 28", size)
	}
	buf := make([]byte, size)
	n, err := av1.PutRTCPFeedbackPacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPFeedbackPacket transport feedback: %v", err)
	}
	wantPrefix := []byte{0xaf, 205, 0x00, 0x06, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
	if n != size || string(buf[:len(wantPrefix)]) != string(wantPrefix) ||
		string(buf[12:25]) != string(transportFCI) ||
		buf[25] != 0 || buf[26] != 0 || buf[27] != 3 {
		t.Fatalf("transport feedback packet n=%d bytes=%#v", n, buf)
	}
	parsed, consumed, err := av1.ParseRTCPFeedbackPacket(buf)
	if err != nil {
		t.Fatalf("ParseRTCPFeedbackPacket transport feedback: %v", err)
	}
	if consumed != len(buf) || parsed.PacketType != packet.PacketType || parsed.FMT != packet.FMT ||
		parsed.SenderSSRC != packet.SenderSSRC || parsed.MediaSSRC != packet.MediaSSRC ||
		string(parsed.FCI) != string(transportFCI) {
		t.Fatalf("parsed transport feedback consumed=%d packet=%+v", consumed, parsed)
	}
	if _, err := av1.ParseRTCPTransportFeedbackFCI(parsed.FCI, make([]av1.RTCPTransportFeedbackPacket, 0, 5)); err != nil {
		t.Fatalf("ParseRTCPTransportFeedbackFCI from packet: %v", err)
	}
}

func TestRTCPFeedbackPacketRejectsInvalid(t *testing.T) {
	pli := av1.RTCPFeedbackPacket{PacketType: av1.RTCPPSFBPacketType, FMT: av1.RTCPPSFBPictureLossIndicationFMT}
	if _, err := av1.PutRTCPFeedbackPacket(make([]byte, av1.RTCPFeedbackPacketHeaderSize-1), pli); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPFeedbackPacket err=%v", err)
	}
	if _, err := av1.RTCPFeedbackPacketSize(-1); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("negative RTCPFeedbackPacketSize err=%v", err)
	}
	maxSize, err := av1.RTCPFeedbackPacketSize(av1.RTCPFeedbackMaxFCISize)
	if err != nil || maxSize != av1.RTCPMaxPacketSize {
		t.Fatalf("max RTCPFeedbackPacketSize size=%d err=%v", maxSize, err)
	}
	if _, err := av1.RTCPFeedbackPacketSize(av1.RTCPFeedbackMaxFCISize + 1); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("oversized RTCPFeedbackPacketSize err=%v", err)
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := av1.RTCPFeedbackPacketSize(maxInt); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("overflow RTCPFeedbackPacketSize err=%v", err)
	}
	badFMT := pli
	badFMT.FMT = 32
	if _, err := av1.PutRTCPFeedbackPacket(make([]byte, av1.RTCPFeedbackPacketHeaderSize), badFMT); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad FMT PutRTCPFeedbackPacket err=%v", err)
	}
	badType := pli
	badType.PacketType = 200
	if _, err := av1.PutRTCPFeedbackPacket(make([]byte, av1.RTCPFeedbackPacketHeaderSize), badType); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad type PutRTCPFeedbackPacket err=%v", err)
	}

	if _, _, err := av1.ParseRTCPFeedbackPacket(make([]byte, av1.RTCPFeedbackPacketHeaderSize-1)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPFeedbackPacket err=%v", err)
	}
	badVersion := []byte{0x41, 206, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPFeedbackPacket(badVersion); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad version ParseRTCPFeedbackPacket err=%v", err)
	}
	badPacketType := []byte{0x81, 200, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPFeedbackPacket(badPacketType); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("bad packet type ParseRTCPFeedbackPacket err=%v", err)
	}
	shortLength := []byte{0x81, 206, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPFeedbackPacket(shortLength); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("short length ParseRTCPFeedbackPacket err=%v", err)
	}
	truncated := []byte{0x81, 206, 0, 2, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPFeedbackPacket(truncated); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("truncated ParseRTCPFeedbackPacket err=%v", err)
	}
	zeroPadding := []byte{0xa1, 206, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := av1.ParseRTCPFeedbackPacket(zeroPadding); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("zero padding ParseRTCPFeedbackPacket err=%v", err)
	}
	tooMuchPadding := []byte{0xa1, 206, 0, 2, 0, 0, 0, 0, 0, 0, 0, 9}
	if _, _, err := av1.ParseRTCPFeedbackPacket(tooMuchPadding); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("too much padding ParseRTCPFeedbackPacket err=%v", err)
	}
}

func TestEncoderWebRTCRTCPFeedbackRequiresKeyFrame(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:   av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:  av1.EncoderScalabilityModeL2T2,
		MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
	}

	firFCI, err := av1.AppendRTCPFullIntraRequestEntries(make([]byte, 0, av1.RTCPFullIntraRequestEntrySize), []av1.RTCPFullIntraRequestEntry{{
		SSRC:           0x11112222,
		SequenceNumber: 7,
	}})
	if err != nil {
		t.Fatalf("AppendRTCPFullIntraRequestEntries: %v", err)
	}
	lrrFCI, err := av1.AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, av1.AV1RTCPLayerRefreshRequestEntrySize), []av1.AV1RTCPLayerRefreshRequestEntry{{
		SSRC:           0x11112222,
		SequenceNumber: 8,
		PayloadType:    96,
		Target:         av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 1, TemporalID: 1},
		CurrentPresent: true,
		Current:        av1.AV1RTCPLayerRefreshLayerIndex{},
	}})
	if err != nil {
		t.Fatalf("AppendAV1RTCPLayerRefreshRequestEntries: %v", err)
	}

	tests := []struct {
		name   string
		packet av1.RTCPFeedbackPacket
		want   bool
	}{
		{
			name: "pli",
			packet: av1.RTCPFeedbackPacket{
				PacketType: av1.RTCPPSFBPacketType,
				FMT:        av1.RTCPPSFBPictureLossIndicationFMT,
			},
			want: true,
		},
		{
			name: "fir",
			packet: av1.RTCPFeedbackPacket{
				PacketType: av1.RTCPPSFBPacketType,
				FMT:        av1.RTCPPSFBFullIntraRequestFMT,
				FCI:        firFCI,
			},
			want: true,
		},
		{
			name: "lrr",
			packet: av1.RTCPFeedbackPacket{
				PacketType: av1.RTCPPSFBPacketType,
				FMT:        av1.RTCPPSFBLayerRefreshRequestFMT,
				FCI:        lrrFCI,
			},
			want: true,
		},
		{
			name: "transport-feedback",
			packet: av1.RTCPFeedbackPacket{
				PacketType: av1.RTCPRTPFBPacketType,
				FMT:        av1.RTCPRTPFBTransportFeedbackFMT,
				FCI:        []byte{0, 1, 0, 0},
			},
		},
		{
			name: "remb",
			packet: av1.RTCPFeedbackPacket{
				PacketType: av1.RTCPPSFBPacketType,
				FMT:        av1.RTCPPSFBApplicationLayerFeedbackFMT,
				FCI:        []byte{'R', 'E', 'M', 'B', 0, 0, 0, 0},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := av1.EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
				cfg,
				tc.packet,
				make([]av1.RTCPFullIntraRequestEntry, 0, 1),
				make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
			)
			if err != nil {
				t.Fatalf("EncoderWebRTCRTCPFeedbackRequiresKeyFrame: %v", err)
			}
			if got != tc.want {
				t.Fatalf("requires key=%v want %v", got, tc.want)
			}
		})
	}
}

func TestEncoderWebRTCRTCPFeedbackRequiresKeyFrameRejectsInvalid(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:   av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:  av1.EncoderScalabilityModeL2T2,
		MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
	}
	invalidLRR, err := av1.AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, av1.AV1RTCPLayerRefreshRequestEntrySize), []av1.AV1RTCPLayerRefreshRequestEntry{{
		SSRC:        0x11112222,
		PayloadType: 96,
		Target:      av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 1},
	}})
	if err != nil {
		t.Fatalf("AppendAV1RTCPLayerRefreshRequestEntries invalid grid fixture: %v", err)
	}

	if _, err := av1.EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		av1.RTCPFeedbackPacket{PacketType: av1.RTCPPSFBPacketType, FMT: av1.RTCPPSFBPictureLossIndicationFMT, FCI: []byte{0}},
		nil,
		nil,
	); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("PLI invalid FCI err=%v want %v", err, av1.ErrRTCPInvalidFeedback)
	}
	if _, err := av1.EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		av1.RTCPFeedbackPacket{PacketType: av1.RTCPPSFBPacketType, FMT: av1.RTCPPSFBFullIntraRequestFMT, FCI: make([]byte, av1.RTCPFullIntraRequestEntrySize)},
		nil,
		nil,
	); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("FIR short scratch err=%v want %v", err, av1.ErrRTCPShortBuffer)
	}
	if _, err := av1.EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		av1.RTCPFeedbackPacket{PacketType: av1.RTCPPSFBPacketType, FMT: av1.RTCPPSFBLayerRefreshRequestFMT, FCI: invalidLRR},
		nil,
		make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
	); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("LRR invalid grid err=%v want %v", err, av1.ErrRTCPInvalidLayerRefreshRequest)
	}
	if _, err := av1.EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		av1.RTCPFeedbackPacket{PacketType: av1.RTCPSenderReportPacketType},
		nil,
		nil,
	); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid packet type err=%v want %v", err, av1.ErrRTCPInvalidFeedback)
	}
}

func TestEncoderWebRTCRTCPPacketsRequireKeyFrame(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:   av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:  av1.EncoderScalabilityModeL2T2,
		MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
	}

	var compound []byte
	var err error
	compound, err = av1.AppendRTCPSenderReportPacket(make([]byte, 0, 128), av1.RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPRTPFBPacketType,
		FMT:        av1.RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket PLI: %v", err)
	}

	packets, err := av1.ParseRTCPCompoundPackets(compound, make([]av1.RTCPPacket, 0, 3))
	if err != nil {
		t.Fatalf("ParseRTCPCompoundPackets: %v", err)
	}
	force, err := av1.EncoderWebRTCRTCPPacketsRequireKeyFrame(
		cfg,
		packets,
		make([]av1.RTCPFullIntraRequestEntry, 0, 1),
		make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPPacketsRequireKeyFrame: %v", err)
	}
	if !force {
		t.Fatal("compound feedback did not require key frame")
	}

	transportOnly := packets[:2]
	force, err = av1.EncoderWebRTCRTCPPacketsRequireKeyFrame(cfg, transportOnly, nil, nil)
	if err != nil {
		t.Fatalf("transport-only compound decision: %v", err)
	}
	if force {
		t.Fatal("transport-only compound required key frame")
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(av1.EncoderScalabilityModeL2T2)

	compound, err := av1.AppendRTCPSenderReportPacket(make([]byte, 0, 128), av1.RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPRTPFBPacketType,
		FMT:        av1.RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	transportOnlyLen := len(compound)
	compound, err = av1.AppendRTCPFeedbackPacket(compound, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket PLI: %v", err)
	}

	packetScratch := make([]av1.RTCPPacket, 1, 4)
	packetScratch[0] = av1.RTCPPacket{
		PacketType: av1.RTCPPSFBPacketType,
		Count:      av1.RTCPPSFBPictureLossIndicationFMT,
		Payload:    []byte{0},
	}
	force, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		cfg,
		compound,
		packetScratch[:1:4],
		make([]av1.RTCPFullIntraRequestEntry, 0, 1),
		make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame: %v", err)
	}
	if !force {
		t.Fatal("compound feedback did not require key frame")
	}
	if len(packets) != 3 {
		t.Fatalf("parsed packet len=%d want 3", len(packets))
	}
	if packetScratch[0].PacketType != av1.RTCPPSFBPacketType || len(packetScratch[0].Payload) != 1 {
		t.Fatalf("scratch prefix clobbered: %+v", packetScratch[0])
	}

	force, packets, err = av1.EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		cfg,
		compound[:transportOnlyLen],
		make([]av1.RTCPPacket, 0, 2),
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("transport-only compound decision: %v", err)
	}
	if force {
		t.Fatal("transport-only compound required key frame")
	}
	if len(packets) != 2 {
		t.Fatalf("transport-only parsed packet len=%d want 2", len(packets))
	}

	_, _, err = av1.EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		cfg,
		compound[:len(compound)-1],
		make([]av1.RTCPPacket, 0, 3),
		nil,
		nil,
	)
	if !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("truncated compound err=%v want %v", err, av1.ErrRTCPShortBuffer)
	}
}

func TestEncoderWebRTCRTCPLayerRefreshTarget(t *testing.T) {
	mode := av1.EncoderScalabilityModeL3T3
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	entries := []av1.AV1RTCPLayerRefreshRequestEntry{
		{
			SSRC:           0x11112222,
			SequenceNumber: 1,
			PayloadType:    96,
			Target:         av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 0, TemporalID: 2},
			CurrentPresent: true,
			Current:        av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 0, TemporalID: 1},
		},
		{
			SSRC:           0x33334444,
			SequenceNumber: 2,
			PayloadType:    96,
			Target:         av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 1},
			CurrentPresent: true,
			Current:        av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 1, TemporalID: 1},
		},
	}
	target, ok, err := av1.EncoderWebRTCLayerRefreshRequestTarget(cfg, entries)
	if err != nil {
		t.Fatalf("EncoderWebRTCLayerRefreshRequestTarget: %v", err)
	}
	if !ok || target != (av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 2}) {
		t.Fatalf("target=%+v ok=%v want S2/T2,true", target, ok)
	}
	if target, ok, err := av1.EncoderWebRTCLayerRefreshRequestTarget(cfg, nil); err != nil || ok || target != (av1.AV1RTCPLayerRefreshLayerIndex{}) {
		t.Fatalf("empty target=%+v ok=%v err=%v", target, ok, err)
	}

	feedback := av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        testAV1RTCPLayerRefreshFCI(t, entries),
	}
	scratch := make([]av1.AV1RTCPLayerRefreshRequestEntry, 1, 3)
	scratch[0].SSRC = 0xdeadbeef
	target, ok, err = av1.EncoderWebRTCRTCPFeedbackLayerRefreshTarget(cfg, feedback, scratch[:1:3])
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPFeedbackLayerRefreshTarget: %v", err)
	}
	if !ok || target != (av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 2}) ||
		scratch[0].SSRC != 0xdeadbeef {
		t.Fatalf("feedback target=%+v ok=%v scratch0=%+v", target, ok, scratch[0])
	}
	target, ok, err = av1.EncoderWebRTCRTCPFeedbackLayerRefreshTarget(cfg, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBPictureLossIndicationFMT,
	}, nil)
	if err != nil || ok || target != (av1.AV1RTCPLayerRefreshLayerIndex{}) {
		t.Fatalf("non-LRR target=%+v ok=%v err=%v", target, ok, err)
	}

	compound, err := av1.AppendRTCPSenderReportPacket(make([]byte, 0, 160), av1.RTCPSenderReport{SenderSSRC: 0x01020304})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPRTPFBPacketType,
		FMT:        av1.RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, feedback)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket LRR: %v", err)
	}
	target, ok, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
		cfg,
		compound,
		make([]av1.RTCPPacket, 0, 3),
		make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 2),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget: %v", err)
	}
	if !ok || target != (av1.AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 2}) || len(packets) != 3 {
		t.Fatalf("compound target=%+v ok=%v packets=%d", target, ok, len(packets))
	}

	bad := entries[:1]
	bad[0].Target.SpatialID = 3
	badFCI := testAV1RTCPLayerRefreshFCI(t, bad)
	if _, _, err := av1.EncoderWebRTCRTCPFeedbackLayerRefreshTarget(cfg, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBLayerRefreshRequestFMT,
		FCI:        badFCI,
	}, make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1)); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("invalid LRR target err=%v want %v", err, av1.ErrRTCPInvalidLayerRefreshRequest)
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsLayerRefreshTargetAllocs(t *testing.T) {
	mode := av1.EncoderScalabilityModeL2T2
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	valid := testAV1RTCPValidLayerRefreshEntry(t, mode)
	compound := testAV1RTCPFeedbackCompound(t, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        testAV1RTCPLayerRefreshFCI(t, []av1.AV1RTCPLayerRefreshRequestEntry{valid}),
	})
	packetScratch := make([]av1.RTCPPacket, 0, 1)
	lrrScratch := make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		target, ok, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
			cfg,
			compound,
			packetScratch[:0],
			lrrScratch[:0],
		)
		if err != nil {
			t.Fatalf("EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget: %v", err)
		}
		if !ok || target != valid.Target || len(packets) != 1 {
			t.Fatalf("target=%+v ok=%v packet len=%d want %+v,true,1", target, ok, len(packets), valid.Target)
		}
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget allocs/run=%f want 0", allocs)
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsRequireKeyFrameAllocs(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(av1.EncoderScalabilityModeL1T1)
	compound := testAV1RTCPFeedbackCompound(t, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	packetScratch := make([]av1.RTCPPacket, 0, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		force, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
			cfg,
			compound,
			packetScratch[:0],
			nil,
			nil,
		)
		if err != nil {
			t.Fatalf("EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame: %v", err)
		}
		if !force || len(packets) != 1 {
			t.Fatalf("force=%v packet len=%d want true,1", force, len(packets))
		}
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame allocations=%v want 0", allocs)
	}
}

func TestEncoderWebRTCRTCPPacketsRequireKeyFrameRejectsInvalid(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:   av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:  av1.EncoderScalabilityModeL2T2,
		MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
	}
	if _, err := av1.EncoderWebRTCRTCPPacketsRequireKeyFrame(
		cfg,
		[]av1.RTCPPacket{{
			PacketType: av1.RTCPPSFBPacketType,
			Count:      av1.RTCPPSFBPictureLossIndicationFMT,
			Payload:    []byte{0, 1, 2, 3},
		}},
		nil,
		nil,
	); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("short feedback payload err=%v want %v", err, av1.ErrRTCPInvalidFeedback)
	}
}

func TestEncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(t *testing.T) {
	fci := testAV1RTCPReceiverEstimatedMaximumBitrateFCI(t, 640_000, []uint32{0x11112222, 0x33334444})
	scratch := make([]uint32, 1, 3)
	scratch[0] = 0xdeadbeef
	remb, ok, err := av1.EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBApplicationLayerFeedbackFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0,
		FCI:        fci,
	}, scratch[:1:3])
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate: %v", err)
	}
	if !ok || remb.BitrateBps != 640_000 || len(remb.SSRCs) != 2 ||
		remb.SSRCs[0] != 0x11112222 || remb.SSRCs[1] != 0x33334444 || scratch[0] != 0xdeadbeef {
		t.Fatalf("REMB=%+v ok=%v scratch0=%#x", remb, ok, scratch[0])
	}

	for _, packet := range []av1.RTCPFeedbackPacket{
		{PacketType: av1.RTCPRTPFBPacketType, FMT: av1.RTCPRTPFBGenericNACKFMT},
		{PacketType: av1.RTCPPSFBPacketType, FMT: av1.RTCPPSFBPictureLossIndicationFMT},
		{PacketType: av1.RTCPPSFBPacketType, FMT: av1.RTCPPSFBApplicationLayerFeedbackFMT, FCI: []byte("TEST")},
	} {
		remb, ok, err = av1.EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(packet, nil)
		if err != nil || ok || remb.BitrateBps != 0 || len(remb.SSRCs) != 0 {
			t.Fatalf("non-REMB packet %+v returned REMB=%+v ok=%v err=%v", packet, remb, ok, err)
		}
	}

	if _, _, err := av1.EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPSenderReportPacketType,
		FMT:        av1.RTCPPSFBApplicationLayerFeedbackFMT,
		FCI:        fci,
	}, nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid packet type err=%v want %v", err, av1.ErrRTCPInvalidFeedback)
	}
	if _, _, err := av1.EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBApplicationLayerFeedbackFMT,
		FCI:        []byte{'R', 'E', 'M', 'B'},
	}, nil); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("malformed REMB err=%v want %v", err, av1.ErrRTCPShortBuffer)
	}
	if _, _, err := av1.EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBApplicationLayerFeedbackFMT,
		FCI:        fci,
	}, make([]uint32, 0, 1)); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short REMB SSRC scratch err=%v want %v", err, av1.ErrRTCPShortBuffer)
	}
}

func TestEncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate(t *testing.T) {
	first := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 600_000, []uint32{0x11112222})
	second := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 900_000, []uint32{0x33334444})
	compound, err := av1.AppendRTCPSenderReportPacket(make([]byte, 0, 256), av1.RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, first)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket first REMB: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPRTPFBPacketType,
		FMT:        av1.RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, second)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket second REMB: %v", err)
	}

	packetScratch := make([]av1.RTCPPacket, 1, 5)
	packetScratch[0] = av1.RTCPPacket{PacketType: av1.RTCPByePacketType, Payload: []byte{0xaa}}
	ssrcScratch := make([]uint32, 1, 2)
	ssrcScratch[0] = 0xdeadbeef
	remb, ok, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrate(
		compound,
		packetScratch[:1:5],
		ssrcScratch[:1:2],
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrate: %v", err)
	}
	if !ok || remb.BitrateBps != 900_000 || len(remb.SSRCs) != 1 || remb.SSRCs[0] != 0x33334444 || len(packets) != 4 {
		t.Fatalf("compound REMB=%+v ok=%v packets=%d", remb, ok, len(packets))
	}
	if packetScratch[0].PacketType != av1.RTCPByePacketType || len(packetScratch[0].Payload) != 1 ||
		ssrcScratch[0] != 0xdeadbeef {
		t.Fatalf("scratch prefix clobbered packet=%+v ssrc0=%#x", packetScratch[0], ssrcScratch[0])
	}

	remb, ok, err = av1.EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate(
		packets[:2],
		make([]uint32, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate first: %v", err)
	}
	if !ok || remb.BitrateBps != 600_000 || len(remb.SSRCs) != 1 || remb.SSRCs[0] != 0x11112222 {
		t.Fatalf("first REMB=%+v ok=%v", remb, ok)
	}

	remb, ok, err = av1.EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate(packets[:1], nil)
	if err != nil || ok || remb.BitrateBps != 0 || len(remb.SSRCs) != 0 {
		t.Fatalf("no REMB=%+v ok=%v err=%v", remb, ok, err)
	}
	if _, _, err := av1.EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate([]av1.RTCPPacket{{
		PacketType: av1.RTCPPSFBPacketType,
		Count:      av1.RTCPPSFBApplicationLayerFeedbackFMT,
		Payload:    []byte{0, 1, 2, 3},
	}}, nil); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
		t.Fatalf("short feedback payload err=%v want %v", err, av1.ErrRTCPInvalidFeedback)
	}
}

func TestEncoderWebRTCApplyReceiverEstimatedMaximumBitrate(t *testing.T) {
	mode := av1.EncoderScalabilityModeL3T3
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	cfg.MinBitrateKbps = 100
	cfg.MaxBitrateKbps = 1200
	cfg.TargetBitrateKbps = 1000
	cfg.SpatialLayers[0].MinBitrateKbps = 50
	cfg.SpatialLayers[0].MaxBitrateKbps = 300
	cfg.SpatialLayers[0].TargetBitrateKbps = 100
	cfg.SpatialLayers[1].MinBitrateKbps = 100
	cfg.SpatialLayers[1].MaxBitrateKbps = 600
	cfg.SpatialLayers[1].TargetBitrateKbps = 300
	cfg.SpatialLayers[2].MinBitrateKbps = 200
	cfg.SpatialLayers[2].MaxBitrateKbps = 1200
	cfg.SpatialLayers[2].TargetBitrateKbps = 600

	updated, err := av1.EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		cfg,
		av1.RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 600_000},
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCApplyReceiverEstimatedMaximumBitrate: %v", err)
	}
	if updated.TargetBitrateKbps != 600 ||
		updated.SpatialLayers[0].TargetBitrateKbps != 60 ||
		updated.SpatialLayers[1].TargetBitrateKbps != 180 ||
		updated.SpatialLayers[2].TargetBitrateKbps != 360 {
		t.Fatalf("updated bitrates target=%d layers=%d/%d/%d",
			updated.TargetBitrateKbps,
			updated.SpatialLayers[0].TargetBitrateKbps,
			updated.SpatialLayers[1].TargetBitrateKbps,
			updated.SpatialLayers[2].TargetBitrateKbps,
		)
	}
	if updated.Scalability != cfg.Scalability || updated.MaxFramerate != cfg.MaxFramerate || updated.RateControl != cfg.RateControl {
		t.Fatalf("apply changed non-bitrate controls: before=%+v after=%+v", cfg, updated)
	}

	low, err := av1.EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		cfg,
		av1.RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 40_000},
	)
	if err != nil {
		t.Fatalf("low REMB apply: %v", err)
	}
	if low.TargetBitrateKbps != 100 ||
		low.SpatialLayers[0].TargetBitrateKbps != 50 ||
		low.SpatialLayers[1].TargetBitrateKbps != 100 ||
		low.SpatialLayers[2].TargetBitrateKbps != 200 {
		t.Fatalf("low REMB target=%d layers=%d/%d/%d",
			low.TargetBitrateKbps,
			low.SpatialLayers[0].TargetBitrateKbps,
			low.SpatialLayers[1].TargetBitrateKbps,
			low.SpatialLayers[2].TargetBitrateKbps,
		)
	}

	high, err := av1.EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		cfg,
		av1.RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 5_000_000},
	)
	if err != nil {
		t.Fatalf("high REMB apply: %v", err)
	}
	if high.TargetBitrateKbps != 1200 ||
		high.SpatialLayers[0].TargetBitrateKbps != 120 ||
		high.SpatialLayers[1].TargetBitrateKbps != 360 ||
		high.SpatialLayers[2].TargetBitrateKbps != 720 {
		t.Fatalf("high REMB target=%d layers=%d/%d/%d",
			high.TargetBitrateKbps,
			high.SpatialLayers[0].TargetBitrateKbps,
			high.SpatialLayers[1].TargetBitrateKbps,
			high.SpatialLayers[2].TargetBitrateKbps,
		)
	}

	unsetRange := av1.EncoderConfig{
		Resolution:   av1.EncoderResolution{Width: 640, Height: 360},
		MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
		RateControl:  av1.EncoderRateControlCBR,
	}
	unsetRange, err = av1.EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		unsetRange,
		av1.RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 123_001},
	)
	if err != nil {
		t.Fatalf("unset range REMB apply: %v", err)
	}
	if unsetRange.TargetBitrateKbps != 124 || unsetRange.MaxBitrateKbps != 124 ||
		unsetRange.SpatialLayers[0].TargetBitrateKbps != 124 ||
		unsetRange.SpatialLayers[0].MaxBitrateKbps != 124 {
		t.Fatalf("unset range target=%d max=%d layer=%+v", unsetRange.TargetBitrateKbps, unsetRange.MaxBitrateKbps, unsetRange.SpatialLayers[0])
	}

	if _, err := av1.EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		av1.EncoderConfig{},
		av1.RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 100_000},
	); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
		t.Fatalf("invalid config err=%v want %v", err, av1.ErrEncoderInvalidConfig)
	}
}

func TestEncoderWebRTCRTCPApplyReceiverEstimatedMaximumBitrate(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(av1.EncoderScalabilityModeL1T1)
	first := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 350_000, []uint32{0x11112222})
	second := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 450_000, []uint32{0x33334444})
	compound, err := av1.AppendRTCPFeedbackPacket(make([]byte, 0, 128), first)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket first: %v", err)
	}
	compound, err = av1.AppendRTCPFeedbackPacket(compound, second)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket second: %v", err)
	}

	updated, ok, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsApplyReceiverEstimatedMaximumBitrate(
		cfg,
		compound,
		make([]av1.RTCPPacket, 0, 2),
		make([]uint32, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsApplyReceiverEstimatedMaximumBitrate: %v", err)
	}
	if !ok || len(packets) != 2 || updated.TargetBitrateKbps != 450 || updated.SpatialLayers[0].TargetBitrateKbps != 450 {
		t.Fatalf("updated=%+v ok=%v packets=%d", updated, ok, len(packets))
	}

	unchanged, ok, err := av1.EncoderWebRTCRTCPFeedbackApplyReceiverEstimatedMaximumBitrate(
		cfg,
		av1.RTCPFeedbackPacket{PacketType: av1.RTCPPSFBPacketType, FMT: av1.RTCPPSFBPictureLossIndicationFMT},
		nil,
	)
	if err != nil || ok || unchanged != cfg {
		t.Fatalf("non-REMB apply config=%+v ok=%v err=%v", unchanged, ok, err)
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrateAllocs(t *testing.T) {
	packet := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 500_000, []uint32{0x11112222})
	compound := testAV1RTCPFeedbackCompound(t, packet)
	packetScratch := make([]av1.RTCPPacket, 0, 1)
	ssrcScratch := make([]uint32, 0, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		remb, ok, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrate(
			compound,
			packetScratch[:0],
			ssrcScratch[:0],
		)
		if err != nil {
			t.Fatalf("EncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrate: %v", err)
		}
		if !ok || remb.BitrateBps != 500_000 || len(remb.SSRCs) != 1 || len(packets) != 1 {
			t.Fatalf("REMB=%+v ok=%v packets=%d", remb, ok, len(packets))
		}
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrate allocs/run=%f want 0", allocs)
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

func TestRTCPGenericNACKPairsForLostSequenceNumbers(t *testing.T) {
	pairs, err := av1.AppendRTCPGenericNACKPairsForLostSequenceNumbers(
		make([]av1.RTCPGenericNACKPair, 0, 2),
		[]uint16{100, 101, 104, 116, 117, 130},
	)
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers: %v", err)
	}
	want := []av1.RTCPGenericNACKPair{
		{PacketID: 100, LostPacketBitmask: 0x8009},
		{PacketID: 117, LostPacketBitmask: 0x1000},
	}
	if len(pairs) != len(want) {
		t.Fatalf("pairs len=%d want %d", len(pairs), len(want))
	}
	for i := range want {
		if pairs[i] != want[i] {
			t.Fatalf("pair[%d]=%+v want %+v", i, pairs[i], want[i])
		}
	}

	wrapped, err := av1.AppendRTCPGenericNACKPairsForLostSequenceNumbers(
		make([]av1.RTCPGenericNACKPair, 0, 2),
		[]uint16{0xfffe, 0xffff, 0x0000, 0x000f},
	)
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers wrap: %v", err)
	}
	wantWrapped := []av1.RTCPGenericNACKPair{
		{PacketID: 0xfffe, LostPacketBitmask: 0x0003},
		{PacketID: 0x000f},
	}
	if len(wrapped) != len(wantWrapped) {
		t.Fatalf("wrapped len=%d want %d", len(wrapped), len(wantWrapped))
	}
	for i := range wantWrapped {
		if wrapped[i] != wantWrapped[i] {
			t.Fatalf("wrapped[%d]=%+v want %+v", i, wrapped[i], wantWrapped[i])
		}
	}

	prefix := make([]av1.RTCPGenericNACKPair, 1, 3)
	prefix[0] = av1.RTCPGenericNACKPair{PacketID: 7}
	prefixed, err := av1.AppendRTCPGenericNACKPairsForLostSequenceNumbers(prefix[:1:3], []uint16{20})
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers prefixed: %v", err)
	}
	if len(prefixed) != 2 || prefixed[0] != prefix[0] || prefixed[1] != (av1.RTCPGenericNACKPair{PacketID: 20}) {
		t.Fatalf("prefixed=%+v", prefixed)
	}

	if out, err := av1.AppendRTCPGenericNACKPairsForLostSequenceNumbers(nil, nil); err != nil || len(out) != 0 {
		t.Fatalf("empty out=%d err=%v", len(out), err)
	}
	if _, err := av1.AppendRTCPGenericNACKPairsForLostSequenceNumbers(make([]av1.RTCPGenericNACKPair, 0, 1), []uint16{10, 27}); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short NACK pair dst err=%v want %v", err, av1.ErrRTCPShortBuffer)
	}
	for _, lost := range [][]uint16{
		{10, 10},
		{10, 9},
	} {
		if _, err := av1.AppendRTCPGenericNACKPairsForLostSequenceNumbers(make([]av1.RTCPGenericNACKPair, 0, 2), lost); !errors.Is(err, av1.ErrRTCPInvalidFeedback) {
			t.Fatalf("lost=%v err=%v want %v", lost, err, av1.ErrRTCPInvalidFeedback)
		}
	}
}

func TestRTCPGenericNACKPairsForLostSequenceNumbersAllocs(t *testing.T) {
	lost := []uint16{0xfffc, 0xfffd, 0x0001, 0x0003, 0x0015}
	pairs := make([]av1.RTCPGenericNACKPair, 0, 3)
	allocs := testing.AllocsPerRun(1000, func() {
		var err error
		pairs, err = av1.AppendRTCPGenericNACKPairsForLostSequenceNumbers(pairs[:0], lost)
		if err != nil {
			t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers allocs/run=%f want 0", allocs)
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

func TestEncoderWebRTCValidateLayerRefreshRequestModeMatrix(t *testing.T) {
	for _, mode := range av1.EncoderWebRTCScalabilityModes() {
		t.Run(mode.String(), func(t *testing.T) {
			spatialLayers, temporalLayers, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid scalability mode %s", mode)
			}
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			valid := testAV1RTCPValidLayerRefreshEntry(t, mode)
			compound := testAV1RTCPFeedbackCompound(t, av1.RTCPFeedbackPacket{
				PacketType: av1.RTCPPSFBPacketType,
				FMT:        av1.RTCPPSFBLayerRefreshRequestFMT,
				SenderSSRC: 0x01020304,
				MediaSSRC:  0x05060708,
				FCI:        testAV1RTCPLayerRefreshFCI(t, []av1.AV1RTCPLayerRefreshRequestEntry{valid}),
			})
			force, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
				cfg,
				compound,
				make([]av1.RTCPPacket, 0, 1),
				nil,
				make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
			)
			if err != nil {
				t.Fatalf("valid LRR compound decision: %v", err)
			}
			if !force || len(packets) != 1 {
				t.Fatalf("valid LRR force=%v packet len=%d want true,1", force, len(packets))
			}
			target, ok, packets, err := av1.EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
				cfg,
				compound,
				make([]av1.RTCPPacket, 0, 1),
				make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
			)
			if err != nil {
				t.Fatalf("valid LRR compound target: %v", err)
			}
			if !ok || len(packets) != 1 || target != valid.Target {
				t.Fatalf("valid LRR target=%+v ok=%v packet len=%d want %+v,true,1", target, ok, len(packets), valid.Target)
			}

			badTemporal := valid
			badTemporal.Target.TemporalID = temporalLayers
			if err := av1.EncoderWebRTCValidateLayerRefreshRequest(cfg, badTemporal); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
				t.Fatalf("bad temporal target error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
			}

			badSpatial := valid
			badSpatial.Target.SpatialID = spatialLayers
			badCompound := testAV1RTCPFeedbackCompound(t, av1.RTCPFeedbackPacket{
				PacketType: av1.RTCPPSFBPacketType,
				FMT:        av1.RTCPPSFBLayerRefreshRequestFMT,
				SenderSSRC: 0x01020304,
				MediaSSRC:  0x05060708,
				FCI:        testAV1RTCPLayerRefreshFCI(t, []av1.AV1RTCPLayerRefreshRequestEntry{badSpatial}),
			})
			if _, _, err := av1.EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
				cfg,
				badCompound,
				make([]av1.RTCPPacket, 0, 1),
				nil,
				make([]av1.AV1RTCPLayerRefreshRequestEntry, 0, 1),
			); !errors.Is(err, av1.ErrRTCPInvalidLayerRefreshRequest) {
				t.Fatalf("bad spatial target compound error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
			}
		})
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

func testAV1RTCPFeedbackCompound(t *testing.T, packet av1.RTCPFeedbackPacket) []byte {
	t.Helper()
	compound, err := av1.AppendRTCPFeedbackPacket(
		make([]byte, 0, av1.RTCPFeedbackPacketHeaderSize+len(packet.FCI)),
		packet,
	)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket: %v", err)
	}
	return compound
}

func testAV1RTCPReceiverEstimatedMaximumBitratePacket(t *testing.T, bitrateBps uint64, ssrcs []uint32) av1.RTCPFeedbackPacket {
	t.Helper()
	return av1.RTCPFeedbackPacket{
		PacketType: av1.RTCPPSFBPacketType,
		FMT:        av1.RTCPPSFBApplicationLayerFeedbackFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0,
		FCI:        testAV1RTCPReceiverEstimatedMaximumBitrateFCI(t, bitrateBps, ssrcs),
	}
}

func testAV1RTCPReceiverEstimatedMaximumBitrateFCI(t *testing.T, bitrateBps uint64, ssrcs []uint32) []byte {
	t.Helper()
	size, err := av1.RTCPReceiverEstimatedMaximumBitrateFCISize(ssrcs)
	if err != nil {
		t.Fatalf("RTCPReceiverEstimatedMaximumBitrateFCISize: %v", err)
	}
	fci, err := av1.AppendRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 0, size), bitrateBps, ssrcs)
	if err != nil {
		t.Fatalf("AppendRTCPReceiverEstimatedMaximumBitrateFCI: %v", err)
	}
	return fci
}

func testAV1RTCPLayerRefreshFCI(t *testing.T, entries []av1.AV1RTCPLayerRefreshRequestEntry) []byte {
	t.Helper()
	size, err := av1.AV1RTCPLayerRefreshRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntriesSize: %v", err)
	}
	fci, err := av1.AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, size), entries)
	if err != nil {
		t.Fatalf("AppendAV1RTCPLayerRefreshRequestEntries: %v", err)
	}
	return fci
}

func testAV1RTCPValidLayerRefreshEntry(t *testing.T, mode av1.EncoderScalabilityMode) av1.AV1RTCPLayerRefreshRequestEntry {
	t.Helper()
	spatialLayers, temporalLayers, _, ok := mode.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", mode)
	}
	target := av1.AV1RTCPLayerRefreshLayerIndex{
		TemporalID: temporalLayers - 1,
		SpatialID:  spatialLayers - 1,
	}
	entry := av1.AV1RTCPLayerRefreshRequestEntry{
		SSRC:           0x05060708,
		SequenceNumber: 17,
		PayloadType:    96,
		Target:         target,
	}
	if target != (av1.AV1RTCPLayerRefreshLayerIndex{}) {
		entry.CurrentPresent = true
	}
	return entry
}
