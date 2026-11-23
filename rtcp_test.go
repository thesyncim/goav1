package goav1

import (
	"errors"
	"testing"

	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
)

func TestAV1RTCPFeedbackConstants(t *testing.T) {
	if RTCPVersion != 2 ||
		RTCPHeaderSize != 4 ||
		RTCPMaxPacketSize != 262144 ||
		RTCPPacketCountMax != 31 ||
		RTCPPacketPayloadMaxSize != 262140 ||
		RTCPSenderReportPacketType != 200 ||
		RTCPReceiverReportPacketType != 201 ||
		RTCPSDESPacketType != 202 ||
		RTCPByePacketType != 203 ||
		RTCPReportMaxBlocks != 31 ||
		RTCPByeMaxSources != 31 ||
		RTCPByeReasonMaxTextLen != 0xff ||
		RTCPReportBlockSize != 24 ||
		RTCPReportCumulativeLostMin != -0x800000 ||
		RTCPReportCumulativeLostMax != 0x7fffff ||
		RTCPSenderReportSenderInfoSize != 24 ||
		RTCPSenderReportPacketMinSize != 28 ||
		RTCPReceiverReportPacketMinSize != 8 ||
		RTCPSDESMaxChunks != 31 ||
		RTCPSDESItemEnd != 0 ||
		RTCPSDESItemCNAME != 1 ||
		RTCPSDESItemTool != 6 ||
		RTCPSDESItemPrivate != 8 ||
		RTCPSDESItemMaxTextLen != 0xff ||
		RTCPFeedbackCommonSize != 8 ||
		RTCPFeedbackPacketHeaderSize != 12 ||
		RTCPFeedbackMaxFCISize != 262132 ||
		RTCPRTPFBPacketType != 205 ||
		RTCPPSFBPacketType != 206 {
		t.Fatalf("unexpected RTCP common constants")
	}
	if RTCPRTPFBGenericNACKFMT != 1 {
		t.Fatalf("RTCPRTPFBGenericNACKFMT = %d, want 1", RTCPRTPFBGenericNACKFMT)
	}
	if RTCPRTPFBTransportFeedbackFMT != 15 {
		t.Fatalf("RTCPRTPFBTransportFeedbackFMT = %d, want 15", RTCPRTPFBTransportFeedbackFMT)
	}
	if RTCPPSFBPictureLossIndicationFMT != 1 {
		t.Fatalf("RTCPPSFBPictureLossIndicationFMT = %d, want 1", RTCPPSFBPictureLossIndicationFMT)
	}
	if RTCPPSFBFullIntraRequestFMT != 4 {
		t.Fatalf("RTCPPSFBFullIntraRequestFMT = %d, want 4", RTCPPSFBFullIntraRequestFMT)
	}
	if RTCPPSFBLayerRefreshRequestFMT != 10 {
		t.Fatalf("RTCPPSFBLayerRefreshRequestFMT = %d, want 10", RTCPPSFBLayerRefreshRequestFMT)
	}
	if RTCPPSFBApplicationLayerFeedbackFMT != 15 {
		t.Fatalf("RTCPPSFBApplicationLayerFeedbackFMT = %d, want 15", RTCPPSFBApplicationLayerFeedbackFMT)
	}
	if RTCPGenericNACKPairSize != 4 {
		t.Fatalf("RTCPGenericNACKPairSize = %d, want 4", RTCPGenericNACKPairSize)
	}
	if RTCPTransportFeedbackFCIHeaderSize != 8 ||
		RTCPTransportFeedbackChunkSize != 2 ||
		RTCPTransportFeedbackFCIMinSize != 10 ||
		RTCPTransportFeedbackDeltaTickMicros != 250 ||
		RTCPTransportFeedbackBaseTimeTickMicros != 64000 ||
		RTCPTransportFeedbackMaxPackets != 0xffff ||
		RTCPTransportFeedbackMaxReferenceTimeTicks != 0xffffff {
		t.Fatalf("unexpected RTCP transport feedback constants")
	}
	if RTCPPictureLossIndicationFCISize != 0 {
		t.Fatalf("RTCPPictureLossIndicationFCISize = %d, want 0", RTCPPictureLossIndicationFCISize)
	}
	if RTCPFullIntraRequestEntrySize != 8 {
		t.Fatalf("RTCPFullIntraRequestEntrySize = %d, want 8", RTCPFullIntraRequestEntrySize)
	}
	if RTCPReceiverEstimatedMaximumBitrateUniqueIdentifier != 0x52454D42 ||
		RTCPReceiverEstimatedMaximumBitrateFCIMinSize != 8 ||
		RTCPReceiverEstimatedMaximumBitrateSSRCSize != 4 ||
		RTCPReceiverEstimatedMaximumBitrateMaxSSRCs != 0xff {
		t.Fatalf("unexpected RTCP REMB constants")
	}
	if RTCPTransportFeedbackStatusNotReceived != 0 ||
		RTCPTransportFeedbackStatusSmallDelta != 1 ||
		RTCPTransportFeedbackStatusLargeOrNegativeDelta != 2 {
		t.Fatalf("unexpected RTCP transport feedback status constants")
	}
	if AV1RTCPLayerRefreshLayerIndexSize != 2 {
		t.Fatalf("AV1RTCPLayerRefreshLayerIndexSize = %d, want 2", AV1RTCPLayerRefreshLayerIndexSize)
	}
	if AV1RTCPLayerRefreshRequestEntrySize != 12 {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntrySize = %d, want 12", AV1RTCPLayerRefreshRequestEntrySize)
	}
	if AV1RTCPLayerRefreshMaxTemporalID != 7 ||
		AV1RTCPLayerRefreshMaxSpatialID != 3 {
		t.Fatalf("unexpected AV1 LRR layer index limits")
	}
	if AV1SDPRTCPFeedbackNACK != "nack" ||
		AV1SDPRTCPFeedbackPLI != "nack pli" ||
		AV1SDPRTCPFeedbackFIR != "ccm fir" ||
		AV1SDPRTCPFeedbackLRR != "ccm lrr" ||
		AV1SDPRTCPFeedbackTransportCC != "transport-cc" ||
		AV1SDPRTCPFeedbackREMB != "goog-remb" {
		t.Fatalf("unexpected AV1 rtcp-fb constants")
	}
}

func TestRTCPPacketRoundTrip(t *testing.T) {
	packet := RTCPPacket{
		PacketType: 204,
		Count:      7,
		Payload:    []byte{1, 2, 3, 4, 5},
	}
	size, err := RTCPPacketSize(len(packet.Payload))
	if err != nil {
		t.Fatalf("RTCPPacketSize: %v", err)
	}
	if size != 12 {
		t.Fatalf("RTCP packet size=%d want 12", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPPacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPPacket: %v", err)
	}
	want := []byte{0xa7, 204, 0x00, 0x02, 1, 2, 3, 4, 5, 0, 0, 3}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("RTCP packet n=%d bytes=%#v want %#v", n, buf, want)
	}
	parsed, consumed, err := ParseRTCPPacket(buf)
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
	appended, err := AppendRTCPPacket(prefix, packet)
	if err != nil {
		t.Fatalf("AppendRTCPPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended RTCP packet=%#v", appended)
	}
}

func TestRTCPCompoundPackets(t *testing.T) {
	sr := make([]byte, RTCPSenderReportPacketMinSize)
	if _, err := PutRTCPSenderReportPacket(sr, RTCPSenderReport{SenderSSRC: 0x11111111}); err != nil {
		t.Fatalf("PutRTCPSenderReportPacket: %v", err)
	}
	sdesPacket := RTCPSDESPacket{Chunks: []RTCPSDESChunk{{
		Source: 0x22222222,
		Items:  []RTCPSDESItem{{Type: RTCPSDESItemCNAME, Text: []byte("x")}},
	}}}
	sdes := make([]byte, 12)
	if _, err := PutRTCPSDESPacket(sdes, sdesPacket); err != nil {
		t.Fatalf("PutRTCPSDESPacket: %v", err)
	}
	pli := make([]byte, RTCPFeedbackPacketHeaderSize)
	if _, err := PutRTCPFeedbackPacket(pli, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x33333333,
		MediaSSRC:  0x44444444,
	}); err != nil {
		t.Fatalf("PutRTCPFeedbackPacket: %v", err)
	}
	bye := make([]byte, 8)
	if _, err := PutRTCPByePacket(bye, RTCPByePacket{Sources: []uint32{0x55555555}}); err != nil {
		t.Fatalf("PutRTCPByePacket: %v", err)
	}
	unknown := make([]byte, 12)
	if _, err := PutRTCPPacket(unknown, RTCPPacket{
		PacketType: 204,
		Count:      7,
		Payload:    []byte{9, 8, 7, 6, 5},
	}); err != nil {
		t.Fatalf("PutRTCPPacket unknown: %v", err)
	}
	compound := append(append(append(append(append([]byte{}, sr...), sdes...), pli...), bye...), unknown...)
	prefix := make([]RTCPPacket, 1, 6)
	prefix[0].PacketType = 255
	packets, err := ParseRTCPCompoundPackets(compound, prefix[:1:6])
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
		RTCPSenderReportPacketType,
		RTCPSDESPacketType,
		RTCPPSFBPacketType,
		RTCPByePacketType,
		204,
	}
	wantCounts := []uint8{0, 1, RTCPPSFBPictureLossIndicationFMT, 1, 7}
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
	if _, err := RTCPPacketSize(-1); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("negative RTCPPacketSize err=%v", err)
	}
	if _, err := RTCPPacketSize(RTCPPacketPayloadMaxSize + 1); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("oversized RTCPPacketSize err=%v", err)
	}
	badCount := RTCPPacket{Count: RTCPPacketCountMax + 1}
	if _, err := PutRTCPPacket(make([]byte, RTCPHeaderSize), badCount); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad count PutRTCPPacket err=%v", err)
	}
	if out, err := AppendRTCPPacket(make([]byte, 0, RTCPHeaderSize), badCount); !errors.Is(err, ErrRTCPInvalidPacket) || len(out) != 0 {
		t.Fatalf("bad count AppendRTCPPacket out=%d err=%v", len(out), err)
	}
	if _, err := PutRTCPPacket(make([]byte, RTCPHeaderSize-1), RTCPPacket{}); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPPacket err=%v", err)
	}
	if _, _, err := ParseRTCPPacket(nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPPacket err=%v", err)
	}
	badVersion := []byte{0x40, 204, 0, 0}
	if _, _, err := ParseRTCPPacket(badVersion); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad version ParseRTCPPacket err=%v", err)
	}
	truncated := []byte{0x80, 204, 0, 1}
	if _, _, err := ParseRTCPPacket(truncated); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("truncated ParseRTCPPacket err=%v", err)
	}
	zeroPadding := []byte{0xa0, 204, 0, 0}
	if _, _, err := ParseRTCPPacket(zeroPadding); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("zero padding ParseRTCPPacket err=%v", err)
	}
	tooMuchPadding := []byte{0xa0, 204, 0, 1, 1, 2, 3, 8}
	if _, _, err := ParseRTCPPacket(tooMuchPadding); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("too much padding ParseRTCPPacket err=%v", err)
	}
	if _, err := ParseRTCPCompoundPackets(nil, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("empty ParseRTCPCompoundPackets err=%v", err)
	}
	valid := []byte{0x80, 204, 0, 0}
	if _, err := ParseRTCPCompoundPackets(valid, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short dst ParseRTCPCompoundPackets err=%v", err)
	}
	compoundTruncated := append(append([]byte{}, valid...), truncated...)
	if _, err := ParseRTCPCompoundPackets(compoundTruncated, make([]RTCPPacket, 0, 2)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("truncated compound ParseRTCPCompoundPackets err=%v", err)
	}
}

func TestRTCPByePacketRoundTrip(t *testing.T) {
	packet := RTCPByePacket{
		Sources: []uint32{0x11111111, 0x22222222},
		Reason:  []byte("done"),
	}
	size, err := RTCPByePacketSize(packet.Sources, packet.Reason)
	if err != nil {
		t.Fatalf("RTCPByePacketSize: %v", err)
	}
	if size != 20 {
		t.Fatalf("BYE packet size=%d want 20", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPByePacket(buf, packet)
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
	parsed, consumed, err := ParseRTCPByePacket(buf, dst[:1:3])
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
	appended, err := AppendRTCPByePacket(prefix, packet)
	if err != nil {
		t.Fatalf("AppendRTCPByePacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xbe || string(appended[1:]) != string(buf) {
		t.Fatalf("appended BYE packet=%#v", appended)
	}
}

func TestRTCPByePacketWithoutReason(t *testing.T) {
	packet := RTCPByePacket{Sources: []uint32{0x01020304}}
	size, err := RTCPByePacketSize(packet.Sources, nil)
	if err != nil {
		t.Fatalf("RTCPByePacketSize no reason: %v", err)
	}
	if size != 8 {
		t.Fatalf("BYE no reason size=%d want 8", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPByePacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPByePacket no reason: %v", err)
	}
	want := []byte{0x81, 203, 0x00, 0x01, 0x01, 0x02, 0x03, 0x04}
	if n != len(want) || string(buf) != string(want) {
		t.Fatalf("BYE no reason n=%d bytes=%#v want %#v", n, buf, want)
	}
	parsed, consumed, err := ParseRTCPByePacket(buf, make([]uint32, 0, 1))
	if err != nil {
		t.Fatalf("ParseRTCPByePacket no reason: %v", err)
	}
	if consumed != len(buf) || len(parsed.Sources) != 1 || parsed.Sources[0] != 0x01020304 || len(parsed.Reason) != 0 {
		t.Fatalf("parsed BYE no reason consumed=%d packet=%+v", consumed, parsed)
	}
}

func TestRTCPByePacketRejectsInvalid(t *testing.T) {
	tooMany := make([]uint32, RTCPByeMaxSources+1)
	if _, err := RTCPByePacketSize(tooMany, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("too many BYE sources size err=%v", err)
	}
	if _, err := PutRTCPByePacket(make([]byte, 4096), RTCPByePacket{Reason: make([]byte, RTCPByeReasonMaxTextLen+1)}); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("too long BYE reason put err=%v", err)
	}
	if _, err := PutRTCPByePacket(make([]byte, RTCPHeaderSize-1), RTCPByePacket{}); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short BYE put err=%v", err)
	}
	if out, err := AppendRTCPByePacket(make([]byte, 0, RTCPHeaderSize-1), RTCPByePacket{}); !errors.Is(err, ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short BYE append out=%d err=%v", len(out), err)
	}
	if _, _, err := ParseRTCPByePacket(nil, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short BYE parse err=%v", err)
	}
	badVersion := []byte{0x40, 203, 0, 0}
	if _, _, err := ParseRTCPByePacket(badVersion, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad version BYE parse err=%v", err)
	}
	badType := []byte{0x80, 202, 0, 0}
	if _, _, err := ParseRTCPByePacket(badType, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad type BYE parse err=%v", err)
	}
	truncated := []byte{0x80, 203, 0, 1}
	if _, _, err := ParseRTCPByePacket(truncated, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("truncated BYE parse err=%v", err)
	}
	missingSource := []byte{0x81, 203, 0, 0}
	if _, _, err := ParseRTCPByePacket(missingSource, make([]uint32, 0, 1)); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("missing BYE source parse err=%v", err)
	}
	validOneSource := []byte{0x81, 203, 0, 1, 0, 0, 0, 1}
	if _, _, err := ParseRTCPByePacket(validOneSource, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short BYE source dst parse err=%v", err)
	}
	badReasonLength := []byte{0x80, 203, 0, 1, 4, 'x', 0, 0}
	if _, _, err := ParseRTCPByePacket(badReasonLength, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad BYE reason length parse err=%v", err)
	}
	badReasonPadding := []byte{0x80, 203, 0, 1, 1, 'x', 1, 0}
	if _, _, err := ParseRTCPByePacket(badReasonPadding, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad BYE reason padding parse err=%v", err)
	}
	validPacketPadding := []byte{0xa0, 203, 0, 1, 0, 0, 0, 4}
	parsed, consumed, err := ParseRTCPByePacket(validPacketPadding, nil)
	if err != nil {
		t.Fatalf("valid padded BYE parse: %v", err)
	}
	if consumed != len(validPacketPadding) || len(parsed.Sources) != 0 || len(parsed.Reason) != 0 {
		t.Fatalf("valid padded BYE consumed=%d packet=%+v", consumed, parsed)
	}
	zeroPacketPadding := []byte{0xa0, 203, 0, 1, 0, 0, 0, 0}
	if _, _, err := ParseRTCPByePacket(zeroPacketPadding, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("zero BYE packet padding err=%v", err)
	}
}

func TestRTCPSDESPacketRoundTrip(t *testing.T) {
	packet := RTCPSDESPacket{Chunks: []RTCPSDESChunk{
		{
			Source: 0x11111111,
			Items: []RTCPSDESItem{
				{Type: RTCPSDESItemCNAME, Text: []byte("alice")},
				{Type: RTCPSDESItemTool, Text: []byte("goav1")},
			},
		},
		{
			Source: 0x22222222,
			Items: []RTCPSDESItem{
				{Type: RTCPSDESItemCNAME, Text: []byte("b")},
			},
		},
	}}
	size, err := RTCPSDESPacketSize(packet.Chunks)
	if err != nil {
		t.Fatalf("RTCPSDESPacketSize: %v", err)
	}
	if size != 32 {
		t.Fatalf("SDES packet size=%d want 32", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPSDESPacket(buf, packet)
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
	chunkDst := make([]RTCPSDESChunk, 1, 3)
	chunkDst[0].Source = 0x99999999
	itemDst := make([]RTCPSDESItem, 1, 4)
	itemDst[0] = RTCPSDESItem{Type: RTCPSDESItemName, Text: []byte("prefix")}
	parsed, consumed, err := ParseRTCPSDESPacket(buf, chunkDst[:1:3], itemDst[:1:4])
	if err != nil {
		t.Fatalf("ParseRTCPSDESPacket: %v", err)
	}
	if consumed != len(buf) || len(parsed.Chunks) != len(packet.Chunks) {
		t.Fatalf("parsed SDES consumed=%d packet=%+v", consumed, parsed)
	}
	if chunkDst[0].Source != 0x99999999 || itemDst[0].Type != RTCPSDESItemName {
		t.Fatalf("ParseRTCPSDESPacket clobbered prefixes chunks=%+v items=%+v", chunkDst[0], itemDst[0])
	}
	assertRTCPSDESPacketEqual(t, parsed, packet)
	itemStorage := itemDst[:cap(itemDst)]
	if len(parsed.Chunks[0].Items) == 0 || &parsed.Chunks[0].Items[0] != &itemStorage[1] {
		t.Fatalf("parsed SDES items did not alias caller item buffer")
	}

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xef
	appended, err := AppendRTCPSDESPacket(prefix, packet)
	if err != nil {
		t.Fatalf("AppendRTCPSDESPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xef || string(appended[1:]) != string(buf) {
		t.Fatalf("appended SDES packet=%#v", appended)
	}
}

func TestRTCPSDESPacketRejectsInvalid(t *testing.T) {
	tooMany := RTCPSDESPacket{Chunks: make([]RTCPSDESChunk, RTCPSDESMaxChunks+1)}
	if _, err := RTCPSDESPacketSize(tooMany.Chunks); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("too many SDES chunks size err=%v", err)
	}
	badEndItem := RTCPSDESPacket{Chunks: []RTCPSDESChunk{{Items: []RTCPSDESItem{{Type: RTCPSDESItemEnd}}}}}
	if _, err := RTCPSDESPacketSize(badEndItem.Chunks); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("SDES end item size err=%v", err)
	}
	tooLong := RTCPSDESPacket{Chunks: []RTCPSDESChunk{{Items: []RTCPSDESItem{{Type: RTCPSDESItemCNAME, Text: make([]byte, RTCPSDESItemMaxTextLen+1)}}}}}
	if _, err := PutRTCPSDESPacket(make([]byte, 4096), tooLong); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("too long SDES item put err=%v", err)
	}
	if _, err := PutRTCPSDESPacket(make([]byte, RTCPHeaderSize-1), RTCPSDESPacket{}); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short SDES put err=%v", err)
	}
	if out, err := AppendRTCPSDESPacket(make([]byte, 0, RTCPHeaderSize-1), RTCPSDESPacket{}); !errors.Is(err, ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short SDES append out=%d err=%v", len(out), err)
	}
	if _, _, err := ParseRTCPSDESPacket(nil, nil, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short SDES parse err=%v", err)
	}
	badVersion := []byte{0x41, 202, 0, 0}
	if _, _, err := ParseRTCPSDESPacket(badVersion, nil, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad version SDES parse err=%v", err)
	}
	badType := []byte{0x80, 201, 0, 0}
	if _, _, err := ParseRTCPSDESPacket(badType, nil, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad type SDES parse err=%v", err)
	}
	truncated := []byte{0x80, 202, 0, 1}
	if _, _, err := ParseRTCPSDESPacket(truncated, nil, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("truncated SDES parse err=%v", err)
	}
	missingChunk := []byte{0x81, 202, 0, 0}
	if _, _, err := ParseRTCPSDESPacket(missingChunk, make([]RTCPSDESChunk, 0, 1), nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("missing SDES chunk parse err=%v", err)
	}
	validOneChunk := []byte{0x81, 202, 0, 1, 0, 0, 0, 1}
	if _, _, err := ParseRTCPSDESPacket(validOneChunk, nil, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short SDES chunk dst parse err=%v", err)
	}
	validOneItem := []byte{0x81, 202, 0, 2, 0, 0, 0, 1, RTCPSDESItemCNAME, 1, 'x', 0}
	if _, _, err := ParseRTCPSDESPacket(validOneItem, make([]RTCPSDESChunk, 0, 1), nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short SDES item dst parse err=%v", err)
	}
	noEnd := []byte{0x81, 202, 0, 1, 0, 0, 0, 1}
	if _, _, err := ParseRTCPSDESPacket(noEnd, make([]RTCPSDESChunk, 0, 1), nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("missing SDES end parse err=%v", err)
	}
	badItemLength := []byte{0x81, 202, 0, 2, 0, 0, 0, 1, RTCPSDESItemCNAME, 4, 'x', 0}
	if _, _, err := ParseRTCPSDESPacket(badItemLength, make([]RTCPSDESChunk, 0, 1), make([]RTCPSDESItem, 0, 1)); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad SDES item length parse err=%v", err)
	}
	badChunkPadding := []byte{0x81, 202, 0, 2, 0, 0, 0, 1, 0, 1, 0, 0}
	if _, _, err := ParseRTCPSDESPacket(badChunkPadding, make([]RTCPSDESChunk, 0, 1), nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad SDES chunk padding parse err=%v", err)
	}
	validPadded := []byte{0xa0, 202, 0, 1, 0, 0, 0, 4}
	parsed, consumed, err := ParseRTCPSDESPacket(validPadded, nil, nil)
	if err != nil {
		t.Fatalf("valid padded SDES parse: %v", err)
	}
	if consumed != len(validPadded) || len(parsed.Chunks) != 0 {
		t.Fatalf("valid padded SDES consumed=%d packet=%+v", consumed, parsed)
	}
	zeroPadding := []byte{0xa0, 202, 0, 1, 0, 0, 0, 0}
	if _, _, err := ParseRTCPSDESPacket(zeroPadding, nil, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("zero SDES packet padding err=%v", err)
	}
}

func assertRTCPSDESPacketEqual(t *testing.T, got RTCPSDESPacket, want RTCPSDESPacket) {
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
	report := RTCPSenderReport{
		SenderSSRC:        0x11223344,
		NTPSeconds:        0x01020304,
		NTPFraction:       0x05060708,
		RTPTimestamp:      0x090a0b0c,
		SenderPacketCount: 0x0d0e0f10,
		SenderOctetCount:  0x11121314,
		Reports: []RTCPReportBlock{
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
	size, err := RTCPSenderReportPacketSize(len(report.Reports))
	if err != nil {
		t.Fatalf("RTCPSenderReportPacketSize: %v", err)
	}
	if size != 76 {
		t.Fatalf("sender report size=%d want 76", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPSenderReportPacket(buf, report)
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

	dst := make([]RTCPReportBlock, 1, 3)
	dst[0].SSRC = 0x99999999
	parsed, consumed, err := ParseRTCPSenderReportPacket(buf, dst[:1:3])
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
	appended, err := AppendRTCPSenderReportPacket(prefix, report)
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xab || string(appended[1:]) != string(buf) {
		t.Fatalf("appended sender report=%#v", appended)
	}
}

func TestRTCPReceiverReportPacketRoundTrip(t *testing.T) {
	report := RTCPReceiverReport{
		SenderSSRC: 0x10203040,
		Reports: []RTCPReportBlock{{
			SSRC:                          0x01020304,
			FractionLost:                  0x20,
			CumulativePacketsLost:         RTCPReportCumulativeLostMax,
			ExtendedHighestSequenceNumber: 0x11121314,
			InterarrivalJitter:            0x21222324,
			LastSenderReport:              0x31323334,
			DelaySinceLastSenderReport:    0x41424344,
		}},
	}
	size, err := RTCPReceiverReportPacketSize(len(report.Reports))
	if err != nil {
		t.Fatalf("RTCPReceiverReportPacketSize: %v", err)
	}
	if size != 32 {
		t.Fatalf("receiver report size=%d want 32", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPReceiverReportPacket(buf, report)
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
	parsed, consumed, err := ParseRTCPReceiverReportPacket(buf, make([]RTCPReportBlock, 0, 1))
	if err != nil {
		t.Fatalf("ParseRTCPReceiverReportPacket: %v", err)
	}
	if consumed != len(buf) || parsed.SenderSSRC != report.SenderSSRC {
		t.Fatalf("parsed receiver report consumed=%d report=%+v", consumed, parsed)
	}
	assertRTCPReportBlocksEqual(t, parsed.Reports, report.Reports)

	prefix := make([]byte, 1, 1+size)
	prefix[0] = 0xcd
	appended, err := AppendRTCPReceiverReportPacket(prefix, report)
	if err != nil {
		t.Fatalf("AppendRTCPReceiverReportPacket: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xcd || string(appended[1:]) != string(buf) {
		t.Fatalf("appended receiver report=%#v", appended)
	}
}

func TestRTCPReportPacketRejectsInvalid(t *testing.T) {
	if _, err := RTCPSenderReportPacketSize(-1); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("negative sender report size err=%v", err)
	}
	if _, err := RTCPReceiverReportPacketSize(RTCPReportMaxBlocks + 1); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("too many receiver report blocks size err=%v", err)
	}
	report := RTCPReceiverReport{Reports: []RTCPReportBlock{{CumulativePacketsLost: RTCPReportCumulativeLostMax + 1}}}
	if _, err := PutRTCPReceiverReportPacket(make([]byte, RTCPReceiverReportPacketMinSize+RTCPReportBlockSize), report); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("invalid cumulative lost receiver report err=%v", err)
	}
	report.Reports[0].CumulativePacketsLost = RTCPReportCumulativeLostMin - 1
	if _, err := PutRTCPReceiverReportPacket(make([]byte, RTCPReceiverReportPacketMinSize+RTCPReportBlockSize), report); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("invalid negative cumulative lost receiver report err=%v", err)
	}
	if _, err := PutRTCPReceiverReportPacket(make([]byte, RTCPReceiverReportPacketMinSize-1), RTCPReceiverReport{}); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short receiver report put err=%v", err)
	}
	if out, err := AppendRTCPReceiverReportPacket(make([]byte, 0, RTCPReceiverReportPacketMinSize-1), RTCPReceiverReport{}); !errors.Is(err, ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short receiver report append out=%d err=%v", len(out), err)
	}
	if _, _, err := ParseRTCPReceiverReportPacket(nil, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short receiver report parse err=%v", err)
	}
	if _, _, err := ParseRTCPSenderReportPacket(make([]byte, RTCPSenderReportPacketMinSize-1), nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short sender report parse err=%v", err)
	}
	badVersion := []byte{0x40, 201, 0, 1, 0, 0, 0, 0}
	if _, _, err := ParseRTCPReceiverReportPacket(badVersion, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad version receiver report parse err=%v", err)
	}
	badType := []byte{0x80, 200, 0, 1, 0, 0, 0, 0}
	if _, _, err := ParseRTCPReceiverReportPacket(badType, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("bad type receiver report parse err=%v", err)
	}
	lengthTooShort := []byte{0x80, 201, 0, 0, 0, 0, 0, 0}
	if _, _, err := ParseRTCPReceiverReportPacket(lengthTooShort, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("short length receiver report parse err=%v", err)
	}
	truncated := []byte{0x80, 201, 0, 7, 0, 0, 0, 0}
	if _, _, err := ParseRTCPReceiverReportPacket(truncated, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("truncated receiver report parse err=%v", err)
	}
	extraWithoutPadding := []byte{0x80, 201, 0, 2, 0, 0, 0, 0, 1, 2, 3, 4}
	if _, _, err := ParseRTCPReceiverReportPacket(extraWithoutPadding, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("extra receiver report parse err=%v", err)
	}
	validPadded := []byte{0xa0, 201, 0, 2, 0, 0, 0, 1, 0, 0, 0, 4}
	parsed, consumed, err := ParseRTCPReceiverReportPacket(validPadded, nil)
	if err != nil {
		t.Fatalf("valid padded receiver report parse: %v", err)
	}
	if consumed != len(validPadded) || parsed.SenderSSRC != 1 || len(parsed.Reports) != 0 {
		t.Fatalf("valid padded receiver report consumed=%d report=%+v", consumed, parsed)
	}
	zeroPadding := []byte{0xa0, 201, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0}
	if _, _, err := ParseRTCPReceiverReportPacket(zeroPadding, nil); !errors.Is(err, ErrRTCPInvalidPacket) {
		t.Fatalf("zero padding receiver report parse err=%v", err)
	}
	if _, _, err := ParseRTCPReceiverReportPacket(bufWithOneReport(), nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short report dst parse err=%v", err)
	}
}

func bufWithOneReport() []byte {
	buf := make([]byte, RTCPReceiverReportPacketMinSize+RTCPReportBlockSize)
	buf[0] = 0x81
	buf[1] = RTCPReceiverReportPacketType
	buf[3] = 0x07
	return buf
}

func assertRTCPReportBlocksEqual(t *testing.T, got []RTCPReportBlock, want []RTCPReportBlock) {
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
	pli := RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x11223344,
		MediaSSRC:  0x55667788,
	}
	size, err := RTCPFeedbackPacketSize(len(pli.FCI))
	if err != nil {
		t.Fatalf("RTCPFeedbackPacketSize PLI: %v", err)
	}
	if size != RTCPFeedbackPacketHeaderSize {
		t.Fatalf("PLI packet size=%d want %d", size, RTCPFeedbackPacketHeaderSize)
	}
	buf := make([]byte, size)
	n, err := PutRTCPFeedbackPacket(buf, pli)
	if err != nil {
		t.Fatalf("PutRTCPFeedbackPacket PLI: %v", err)
	}
	wantPLI := []byte{0x81, 206, 0x00, 0x02, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88}
	if n != len(wantPLI) || string(buf) != string(wantPLI) {
		t.Fatalf("PLI packet n=%d bytes=%#v want %#v", n, buf, wantPLI)
	}
	parsed, consumed, err := ParseRTCPFeedbackPacket(buf)
	if err != nil {
		t.Fatalf("ParseRTCPFeedbackPacket PLI: %v", err)
	}
	if consumed != len(buf) || parsed.PacketType != pli.PacketType || parsed.FMT != pli.FMT ||
		parsed.SenderSSRC != pli.SenderSSRC || parsed.MediaSSRC != pli.MediaSSRC || len(parsed.FCI) != 0 {
		t.Fatalf("parsed PLI consumed=%d packet=%+v", consumed, parsed)
	}

	prefix := make([]byte, 1, 1+len(buf))
	prefix[0] = 0xaa
	appended, err := AppendRTCPFeedbackPacket(prefix, pli)
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
	packet := RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
		SenderSSRC: 0x12345678,
		MediaSSRC:  0,
		FCI:        rembFCI,
	}
	buf := make([]byte, 32)
	n, err := PutRTCPFeedbackPacket(buf, packet)
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
	parsed, consumed, err := ParseRTCPFeedbackPacket(buf)
	if err != nil {
		t.Fatalf("ParseRTCPFeedbackPacket REMB: %v", err)
	}
	if consumed != len(buf) || parsed.PacketType != packet.PacketType || parsed.FMT != packet.FMT ||
		parsed.SenderSSRC != packet.SenderSSRC || parsed.MediaSSRC != packet.MediaSSRC ||
		string(parsed.FCI) != string(rembFCI) {
		t.Fatalf("parsed REMB consumed=%d packet=%+v", consumed, parsed)
	}
	if _, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(parsed.FCI, make([]uint32, 0, 3)); err != nil {
		t.Fatalf("ParseRTCPReceiverEstimatedMaximumBitrateFCI from packet: %v", err)
	}
}

func TestRTCPFeedbackPacketWrapsPaddedTransportFeedback(t *testing.T) {
	transportFCI := []byte{0x00, 0x01, 0x00, 0x05, 0x01, 0x02, 0x03, 0x7a, 0xb2, 0x00, 0x04, 0x04, 0x04}
	packet := RTCPFeedbackPacket{
		PacketType: RTCPRTPFBPacketType,
		FMT:        RTCPRTPFBTransportFeedbackFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        transportFCI,
	}
	size, err := RTCPFeedbackPacketSize(len(transportFCI))
	if err != nil {
		t.Fatalf("RTCPFeedbackPacketSize transport feedback: %v", err)
	}
	if size != 28 {
		t.Fatalf("transport feedback packet size=%d want 28", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPFeedbackPacket(buf, packet)
	if err != nil {
		t.Fatalf("PutRTCPFeedbackPacket transport feedback: %v", err)
	}
	wantPrefix := []byte{0xaf, 205, 0x00, 0x06, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
	if n != size || string(buf[:len(wantPrefix)]) != string(wantPrefix) ||
		string(buf[12:25]) != string(transportFCI) ||
		buf[25] != 0 || buf[26] != 0 || buf[27] != 3 {
		t.Fatalf("transport feedback packet n=%d bytes=%#v", n, buf)
	}
	parsed, consumed, err := ParseRTCPFeedbackPacket(buf)
	if err != nil {
		t.Fatalf("ParseRTCPFeedbackPacket transport feedback: %v", err)
	}
	if consumed != len(buf) || parsed.PacketType != packet.PacketType || parsed.FMT != packet.FMT ||
		parsed.SenderSSRC != packet.SenderSSRC || parsed.MediaSSRC != packet.MediaSSRC ||
		string(parsed.FCI) != string(transportFCI) {
		t.Fatalf("parsed transport feedback consumed=%d packet=%+v", consumed, parsed)
	}
	if _, err := ParseRTCPTransportFeedbackFCI(parsed.FCI, make([]RTCPTransportFeedbackPacket, 0, 5)); err != nil {
		t.Fatalf("ParseRTCPTransportFeedbackFCI from packet: %v", err)
	}
}

func TestRTCPFeedbackPacketRejectsInvalid(t *testing.T) {
	pli := RTCPFeedbackPacket{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBPictureLossIndicationFMT}
	if _, err := PutRTCPFeedbackPacket(make([]byte, RTCPFeedbackPacketHeaderSize-1), pli); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPFeedbackPacket err=%v", err)
	}
	if _, err := RTCPFeedbackPacketSize(-1); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("negative RTCPFeedbackPacketSize err=%v", err)
	}
	maxSize, err := RTCPFeedbackPacketSize(RTCPFeedbackMaxFCISize)
	if err != nil || maxSize != RTCPMaxPacketSize {
		t.Fatalf("max RTCPFeedbackPacketSize size=%d err=%v", maxSize, err)
	}
	if _, err := RTCPFeedbackPacketSize(RTCPFeedbackMaxFCISize + 1); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("oversized RTCPFeedbackPacketSize err=%v", err)
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := RTCPFeedbackPacketSize(maxInt); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("overflow RTCPFeedbackPacketSize err=%v", err)
	}
	badFMT := pli
	badFMT.FMT = 32
	if _, err := PutRTCPFeedbackPacket(make([]byte, RTCPFeedbackPacketHeaderSize), badFMT); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad FMT PutRTCPFeedbackPacket err=%v", err)
	}
	badType := pli
	badType.PacketType = 200
	if _, err := PutRTCPFeedbackPacket(make([]byte, RTCPFeedbackPacketHeaderSize), badType); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad type PutRTCPFeedbackPacket err=%v", err)
	}

	if _, _, err := ParseRTCPFeedbackPacket(make([]byte, RTCPFeedbackPacketHeaderSize-1)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPFeedbackPacket err=%v", err)
	}
	badVersion := []byte{0x41, 206, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := ParseRTCPFeedbackPacket(badVersion); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad version ParseRTCPFeedbackPacket err=%v", err)
	}
	badPacketType := []byte{0x81, 200, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := ParseRTCPFeedbackPacket(badPacketType); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad packet type ParseRTCPFeedbackPacket err=%v", err)
	}
	shortLength := []byte{0x81, 206, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := ParseRTCPFeedbackPacket(shortLength); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("short length ParseRTCPFeedbackPacket err=%v", err)
	}
	truncated := []byte{0x81, 206, 0, 2, 0, 0, 0, 0}
	if _, _, err := ParseRTCPFeedbackPacket(truncated); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("truncated ParseRTCPFeedbackPacket err=%v", err)
	}
	zeroPadding := []byte{0xa1, 206, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0}
	if _, _, err := ParseRTCPFeedbackPacket(zeroPadding); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("zero padding ParseRTCPFeedbackPacket err=%v", err)
	}
	tooMuchPadding := []byte{0xa1, 206, 0, 2, 0, 0, 0, 0, 0, 0, 0, 9}
	if _, _, err := ParseRTCPFeedbackPacket(tooMuchPadding); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("too much padding ParseRTCPFeedbackPacket err=%v", err)
	}
}

func TestEncoderWebRTCRTCPFeedbackRequiresKeyFrame(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:   EncoderResolution{Width: 640, Height: 360},
		Scalability:  internalencoder.ScalabilityModeL2T2,
		MaxFramerate: EncoderRational{Num: 30, Den: 1},
	}

	firFCI, err := AppendRTCPFullIntraRequestEntries(make([]byte, 0, RTCPFullIntraRequestEntrySize), []RTCPFullIntraRequestEntry{{
		SSRC:           0x11112222,
		SequenceNumber: 7,
	}})
	if err != nil {
		t.Fatalf("AppendRTCPFullIntraRequestEntries: %v", err)
	}
	lrrFCI, err := AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, AV1RTCPLayerRefreshRequestEntrySize), []AV1RTCPLayerRefreshRequestEntry{{
		SSRC:           0x11112222,
		SequenceNumber: 8,
		PayloadType:    96,
		Target:         AV1RTCPLayerRefreshLayerIndex{SpatialID: 1, TemporalID: 1},
		CurrentPresent: true,
		Current:        AV1RTCPLayerRefreshLayerIndex{},
	}})
	if err != nil {
		t.Fatalf("AppendAV1RTCPLayerRefreshRequestEntries: %v", err)
	}

	tests := []struct {
		name   string
		packet RTCPFeedbackPacket
		want   bool
	}{
		{
			name: "pli",
			packet: RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBPictureLossIndicationFMT,
			},
			want: true,
		},
		{
			name: "fir",
			packet: RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBFullIntraRequestFMT,
				FCI:        firFCI,
			},
			want: true,
		},
		{
			name: "lrr",
			packet: RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBLayerRefreshRequestFMT,
				FCI:        lrrFCI,
			},
			want: true,
		},
		{
			name: "transport-feedback",
			packet: RTCPFeedbackPacket{
				PacketType: RTCPRTPFBPacketType,
				FMT:        RTCPRTPFBTransportFeedbackFMT,
				FCI:        []byte{0, 1, 0, 0},
			},
		},
		{
			name: "remb",
			packet: RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
				FCI:        []byte{'R', 'E', 'M', 'B', 0, 0, 0, 0},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
				cfg,
				tc.packet,
				make([]RTCPFullIntraRequestEntry, 0, 1),
				make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
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
	cfg := EncoderConfig{
		Resolution:   EncoderResolution{Width: 640, Height: 360},
		Scalability:  internalencoder.ScalabilityModeL2T2,
		MaxFramerate: EncoderRational{Num: 30, Den: 1},
	}
	invalidLRR, err := AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, AV1RTCPLayerRefreshRequestEntrySize), []AV1RTCPLayerRefreshRequestEntry{{
		SSRC:        0x11112222,
		PayloadType: 96,
		Target:      AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 1},
	}})
	if err != nil {
		t.Fatalf("AppendAV1RTCPLayerRefreshRequestEntries invalid grid fixture: %v", err)
	}

	if _, err := EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		RTCPFeedbackPacket{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBPictureLossIndicationFMT, FCI: []byte{0}},
		nil,
		nil,
	); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("PLI invalid FCI err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
	if _, err := EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		RTCPFeedbackPacket{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBFullIntraRequestFMT, FCI: make([]byte, RTCPFullIntraRequestEntrySize)},
		nil,
		nil,
	); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("FIR short scratch err=%v want %v", err, ErrRTCPShortBuffer)
	}
	if _, err := EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		RTCPFeedbackPacket{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBLayerRefreshRequestFMT, FCI: invalidLRR},
		nil,
		make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
	); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("LRR invalid grid err=%v want %v", err, ErrRTCPInvalidLayerRefreshRequest)
	}
	if _, err := EncoderWebRTCRTCPFeedbackRequiresKeyFrame(
		cfg,
		RTCPFeedbackPacket{PacketType: RTCPSenderReportPacketType},
		nil,
		nil,
	); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid packet type err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
}

func TestEncoderWebRTCRTCPPacketsRequireKeyFrame(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:   EncoderResolution{Width: 640, Height: 360},
		Scalability:  internalencoder.ScalabilityModeL2T2,
		MaxFramerate: EncoderRational{Num: 30, Den: 1},
	}

	var compound []byte
	var err error
	compound, err = AppendRTCPSenderReportPacket(make([]byte, 0, 128), RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, RTCPFeedbackPacket{
		PacketType: RTCPRTPFBPacketType,
		FMT:        RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket PLI: %v", err)
	}

	packets, err := ParseRTCPCompoundPackets(compound, make([]RTCPPacket, 0, 3))
	if err != nil {
		t.Fatalf("ParseRTCPCompoundPackets: %v", err)
	}
	force, err := EncoderWebRTCRTCPPacketsRequireKeyFrame(
		cfg,
		packets,
		make([]RTCPFullIntraRequestEntry, 0, 1),
		make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPPacketsRequireKeyFrame: %v", err)
	}
	if !force {
		t.Fatal("compound feedback did not require key frame")
	}

	transportOnly := packets[:2]
	force, err = EncoderWebRTCRTCPPacketsRequireKeyFrame(cfg, transportOnly, nil, nil)
	if err != nil {
		t.Fatalf("transport-only compound decision: %v", err)
	}
	if force {
		t.Fatal("transport-only compound required key frame")
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(internalencoder.ScalabilityModeL2T2)

	compound, err := AppendRTCPSenderReportPacket(make([]byte, 0, 128), RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, RTCPFeedbackPacket{
		PacketType: RTCPRTPFBPacketType,
		FMT:        RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	transportOnlyLen := len(compound)
	compound, err = AppendRTCPFeedbackPacket(compound, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket PLI: %v", err)
	}

	packetScratch := make([]RTCPPacket, 1, 4)
	packetScratch[0] = RTCPPacket{
		PacketType: RTCPPSFBPacketType,
		Count:      RTCPPSFBPictureLossIndicationFMT,
		Payload:    []byte{0},
	}
	force, packets, err := EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		cfg,
		compound,
		packetScratch[:1:4],
		make([]RTCPFullIntraRequestEntry, 0, 1),
		make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
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
	if packetScratch[0].PacketType != RTCPPSFBPacketType || len(packetScratch[0].Payload) != 1 {
		t.Fatalf("scratch prefix clobbered: %+v", packetScratch[0])
	}

	force, packets, err = EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		cfg,
		compound[:transportOnlyLen],
		make([]RTCPPacket, 0, 2),
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

	_, _, err = EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		cfg,
		compound[:len(compound)-1],
		make([]RTCPPacket, 0, 3),
		nil,
		nil,
	)
	if !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("truncated compound err=%v want %v", err, ErrRTCPShortBuffer)
	}
}

func TestEncoderWebRTCRTCPLayerRefreshTarget(t *testing.T) {
	mode := internalencoder.ScalabilityModeL3T3
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	entries := []AV1RTCPLayerRefreshRequestEntry{
		{
			SSRC:           0x11112222,
			SequenceNumber: 1,
			PayloadType:    96,
			Target:         AV1RTCPLayerRefreshLayerIndex{SpatialID: 0, TemporalID: 2},
			CurrentPresent: true,
			Current:        AV1RTCPLayerRefreshLayerIndex{SpatialID: 0, TemporalID: 1},
		},
		{
			SSRC:           0x33334444,
			SequenceNumber: 2,
			PayloadType:    96,
			Target:         AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 1},
			CurrentPresent: true,
			Current:        AV1RTCPLayerRefreshLayerIndex{SpatialID: 1, TemporalID: 1},
		},
	}
	target, ok, err := EncoderWebRTCLayerRefreshRequestTarget(cfg, entries)
	if err != nil {
		t.Fatalf("EncoderWebRTCLayerRefreshRequestTarget: %v", err)
	}
	if !ok || target != (AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 2}) {
		t.Fatalf("target=%+v ok=%v want S2/T2,true", target, ok)
	}
	if target, ok, err := EncoderWebRTCLayerRefreshRequestTarget(cfg, nil); err != nil || ok || target != (AV1RTCPLayerRefreshLayerIndex{}) {
		t.Fatalf("empty target=%+v ok=%v err=%v", target, ok, err)
	}

	feedback := RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        testAV1RTCPLayerRefreshFCI(t, entries),
	}
	scratch := make([]AV1RTCPLayerRefreshRequestEntry, 1, 3)
	scratch[0].SSRC = 0xdeadbeef
	target, ok, err = EncoderWebRTCRTCPFeedbackLayerRefreshTarget(cfg, feedback, scratch[:1:3])
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPFeedbackLayerRefreshTarget: %v", err)
	}
	if !ok || target != (AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 2}) ||
		scratch[0].SSRC != 0xdeadbeef {
		t.Fatalf("feedback target=%+v ok=%v scratch0=%+v", target, ok, scratch[0])
	}
	target, ok, err = EncoderWebRTCRTCPFeedbackLayerRefreshTarget(cfg, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
	}, nil)
	if err != nil || ok || target != (AV1RTCPLayerRefreshLayerIndex{}) {
		t.Fatalf("non-LRR target=%+v ok=%v err=%v", target, ok, err)
	}

	compound, err := AppendRTCPSenderReportPacket(make([]byte, 0, 160), RTCPSenderReport{SenderSSRC: 0x01020304})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, RTCPFeedbackPacket{
		PacketType: RTCPRTPFBPacketType,
		FMT:        RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, feedback)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket LRR: %v", err)
	}
	target, ok, packets, err := EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
		cfg,
		compound,
		make([]RTCPPacket, 0, 3),
		make([]AV1RTCPLayerRefreshRequestEntry, 0, 2),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget: %v", err)
	}
	if !ok || target != (AV1RTCPLayerRefreshLayerIndex{SpatialID: 2, TemporalID: 2}) || len(packets) != 3 {
		t.Fatalf("compound target=%+v ok=%v packets=%d", target, ok, len(packets))
	}

	bad := entries[:1]
	bad[0].Target.SpatialID = 3
	badFCI := testAV1RTCPLayerRefreshFCI(t, bad)
	if _, _, err := EncoderWebRTCRTCPFeedbackLayerRefreshTarget(cfg, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBLayerRefreshRequestFMT,
		FCI:        badFCI,
	}, make([]AV1RTCPLayerRefreshRequestEntry, 0, 1)); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("invalid LRR target err=%v want %v", err, ErrRTCPInvalidLayerRefreshRequest)
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsLayerRefreshTargetAllocs(t *testing.T) {
	mode := internalencoder.ScalabilityModeL2T2
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	valid := testAV1RTCPValidLayerRefreshEntry(t, mode)
	compound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        testAV1RTCPLayerRefreshFCI(t, []AV1RTCPLayerRefreshRequestEntry{valid}),
	})
	packetScratch := make([]RTCPPacket, 0, 1)
	lrrScratch := make([]AV1RTCPLayerRefreshRequestEntry, 0, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		target, ok, packets, err := EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
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
	cfg := testAV1RTCPEncoderConfig(EncoderScalabilityModeL1T1)
	compound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	packetScratch := make([]RTCPPacket, 0, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		force, packets, err := EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
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
	cfg := EncoderConfig{
		Resolution:   EncoderResolution{Width: 640, Height: 360},
		Scalability:  internalencoder.ScalabilityModeL2T2,
		MaxFramerate: EncoderRational{Num: 30, Den: 1},
	}
	if _, err := EncoderWebRTCRTCPPacketsRequireKeyFrame(
		cfg,
		[]RTCPPacket{{
			PacketType: RTCPPSFBPacketType,
			Count:      RTCPPSFBPictureLossIndicationFMT,
			Payload:    []byte{0, 1, 2, 3},
		}},
		nil,
		nil,
	); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("short feedback payload err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
}

func TestEncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(t *testing.T) {
	fci := testAV1RTCPReceiverEstimatedMaximumBitrateFCI(t, 640_000, []uint32{0x11112222, 0x33334444})
	scratch := make([]uint32, 1, 3)
	scratch[0] = 0xdeadbeef
	remb, ok, err := EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
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

	for _, packet := range []RTCPFeedbackPacket{
		{PacketType: RTCPRTPFBPacketType, FMT: RTCPRTPFBGenericNACKFMT},
		{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBPictureLossIndicationFMT},
		{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBApplicationLayerFeedbackFMT, FCI: []byte("TEST")},
	} {
		remb, ok, err = EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(packet, nil)
		if err != nil || ok || remb.BitrateBps != 0 || len(remb.SSRCs) != 0 {
			t.Fatalf("non-REMB packet %+v returned REMB=%+v ok=%v err=%v", packet, remb, ok, err)
		}
	}

	if _, _, err := EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(RTCPFeedbackPacket{
		PacketType: RTCPSenderReportPacketType,
		FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
		FCI:        fci,
	}, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid packet type err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
	if _, _, err := EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
		FCI:        []byte{'R', 'E', 'M', 'B'},
	}, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("malformed REMB err=%v want %v", err, ErrRTCPShortBuffer)
	}
	if _, _, err := EncoderWebRTCRTCPFeedbackReceiverEstimatedMaximumBitrate(RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
		FCI:        fci,
	}, make([]uint32, 0, 1)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short REMB SSRC scratch err=%v want %v", err, ErrRTCPShortBuffer)
	}
}

func TestEncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate(t *testing.T) {
	first := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 600_000, []uint32{0x11112222})
	second := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 900_000, []uint32{0x33334444})
	compound, err := AppendRTCPSenderReportPacket(make([]byte, 0, 256), RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, first)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket first REMB: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, RTCPFeedbackPacket{
		PacketType: RTCPRTPFBPacketType,
		FMT:        RTCPRTPFBGenericNACKFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket NACK: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, second)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket second REMB: %v", err)
	}

	packetScratch := make([]RTCPPacket, 1, 5)
	packetScratch[0] = RTCPPacket{PacketType: RTCPByePacketType, Payload: []byte{0xaa}}
	ssrcScratch := make([]uint32, 1, 2)
	ssrcScratch[0] = 0xdeadbeef
	remb, ok, packets, err := EncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrate(
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
	if packetScratch[0].PacketType != RTCPByePacketType || len(packetScratch[0].Payload) != 1 ||
		ssrcScratch[0] != 0xdeadbeef {
		t.Fatalf("scratch prefix clobbered packet=%+v ssrc0=%#x", packetScratch[0], ssrcScratch[0])
	}

	remb, ok, err = EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate(
		packets[:2],
		make([]uint32, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate first: %v", err)
	}
	if !ok || remb.BitrateBps != 600_000 || len(remb.SSRCs) != 1 || remb.SSRCs[0] != 0x11112222 {
		t.Fatalf("first REMB=%+v ok=%v", remb, ok)
	}

	remb, ok, err = EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate(packets[:1], nil)
	if err != nil || ok || remb.BitrateBps != 0 || len(remb.SSRCs) != 0 {
		t.Fatalf("no REMB=%+v ok=%v err=%v", remb, ok, err)
	}
	if _, _, err := EncoderWebRTCRTCPPacketsReceiverEstimatedMaximumBitrate([]RTCPPacket{{
		PacketType: RTCPPSFBPacketType,
		Count:      RTCPPSFBApplicationLayerFeedbackFMT,
		Payload:    []byte{0, 1, 2, 3},
	}}, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("short feedback payload err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
}

func TestEncoderWebRTCApplyReceiverEstimatedMaximumBitrate(t *testing.T) {
	mode := internalencoder.ScalabilityModeL3T3
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

	updated, err := EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		cfg,
		RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 600_000},
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

	low, err := EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		cfg,
		RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 40_000},
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

	high, err := EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		cfg,
		RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 5_000_000},
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

	unsetRange := EncoderConfig{
		Resolution:   EncoderResolution{Width: 640, Height: 360},
		MaxFramerate: EncoderRational{Num: 30, Den: 1},
		RateControl:  EncoderRateControlCBR,
	}
	unsetRange, err = EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		unsetRange,
		RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 123_001},
	)
	if err != nil {
		t.Fatalf("unset range REMB apply: %v", err)
	}
	if unsetRange.TargetBitrateKbps != 124 || unsetRange.MaxBitrateKbps != 124 ||
		unsetRange.SpatialLayers[0].TargetBitrateKbps != 124 ||
		unsetRange.SpatialLayers[0].MaxBitrateKbps != 124 {
		t.Fatalf("unset range target=%d max=%d layer=%+v", unsetRange.TargetBitrateKbps, unsetRange.MaxBitrateKbps, unsetRange.SpatialLayers[0])
	}

	if _, err := EncoderWebRTCApplyReceiverEstimatedMaximumBitrate(
		EncoderConfig{},
		RTCPReceiverEstimatedMaximumBitrate{BitrateBps: 100_000},
	); !errors.Is(err, internalencoder.ErrInvalidConfig) {
		t.Fatalf("invalid config err=%v want %v", err, internalencoder.ErrInvalidConfig)
	}
}

func TestEncoderWebRTCRTCPApplyReceiverEstimatedMaximumBitrate(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(EncoderScalabilityModeL1T1)
	first := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 350_000, []uint32{0x11112222})
	second := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 450_000, []uint32{0x33334444})
	compound, err := AppendRTCPFeedbackPacket(make([]byte, 0, 128), first)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket first: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, second)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket second: %v", err)
	}

	updated, ok, packets, err := EncoderWebRTCRTCPCompoundPacketsApplyReceiverEstimatedMaximumBitrate(
		cfg,
		compound,
		make([]RTCPPacket, 0, 2),
		make([]uint32, 0, 1),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsApplyReceiverEstimatedMaximumBitrate: %v", err)
	}
	if !ok || len(packets) != 2 || updated.TargetBitrateKbps != 450 || updated.SpatialLayers[0].TargetBitrateKbps != 450 {
		t.Fatalf("updated=%+v ok=%v packets=%d", updated, ok, len(packets))
	}

	unchanged, ok, err := EncoderWebRTCRTCPFeedbackApplyReceiverEstimatedMaximumBitrate(
		cfg,
		RTCPFeedbackPacket{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBPictureLossIndicationFMT},
		nil,
	)
	if err != nil || ok || unchanged != cfg {
		t.Fatalf("non-REMB apply config=%+v ok=%v err=%v", unchanged, ok, err)
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrateAllocs(t *testing.T) {
	packet := testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 500_000, []uint32{0x11112222})
	compound := testAV1RTCPFeedbackCompound(t, packet)
	packetScratch := make([]RTCPPacket, 0, 1)
	ssrcScratch := make([]uint32, 0, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		remb, ok, packets, err := EncoderWebRTCRTCPCompoundPacketsReceiverEstimatedMaximumBitrate(
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
	n, err := PutRTCPPictureLossIndicationFCI(buf[:])
	if err != nil {
		t.Fatalf("PutRTCPPictureLossIndicationFCI: %v", err)
	}
	if n != 0 || buf != [4]byte{} {
		t.Fatalf("encoded PLI FCI n=%d buf=%#v, want empty no-op", n, buf)
	}
	prefix := []byte{0xaa}
	appended, err := AppendRTCPPictureLossIndicationFCI(prefix)
	if err != nil {
		t.Fatalf("AppendRTCPPictureLossIndicationFCI: %v", err)
	}
	if len(appended) != len(prefix) || appended[0] != prefix[0] {
		t.Fatalf("appended PLI FCI=%#v want unchanged prefix %#v", appended, prefix)
	}
	if err := ParseRTCPPictureLossIndicationFCI(nil); err != nil {
		t.Fatalf("ParseRTCPPictureLossIndicationFCI empty: %v", err)
	}
}

func TestRTCPPictureLossIndicationFCIRejectsInvalid(t *testing.T) {
	if err := ParseRTCPPictureLossIndicationFCI([]byte{0x00}); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("ParseRTCPPictureLossIndicationFCI non-empty err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
}

func TestRTCPReceiverEstimatedMaximumBitrateFCIRoundTrip(t *testing.T) {
	const bitrateBps = 0x3fb93 * 2
	ssrcs := []uint32{0x23456789, 0x2345678a, 0x2345678b}
	size, err := RTCPReceiverEstimatedMaximumBitrateFCISize(ssrcs)
	if err != nil {
		t.Fatalf("RTCPReceiverEstimatedMaximumBitrateFCISize: %v", err)
	}
	if size != RTCPReceiverEstimatedMaximumBitrateFCIMinSize+
		len(ssrcs)*RTCPReceiverEstimatedMaximumBitrateSSRCSize {
		t.Fatalf("REMB FCI size=%d", size)
	}

	buf := make([]byte, size)
	n, err := PutRTCPReceiverEstimatedMaximumBitrateFCI(buf, bitrateBps, ssrcs)
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
	parsed, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(buf, base[:1:cap(base)])
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
	appended, err := AppendRTCPReceiverEstimatedMaximumBitrateFCI(prefix, bitrateBps, ssrcs)
	if err != nil {
		t.Fatalf("AppendRTCPReceiverEstimatedMaximumBitrateFCI: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended REMB FCI bytes=%#v", appended)
	}

	emptyBuf := make([]byte, RTCPReceiverEstimatedMaximumBitrateFCIMinSize)
	if _, err := PutRTCPReceiverEstimatedMaximumBitrateFCI(emptyBuf, 123, nil); err != nil {
		t.Fatalf("PutRTCPReceiverEstimatedMaximumBitrateFCI no SSRCs: %v", err)
	}
	empty, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(emptyBuf, nil)
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
	if _, err := PutRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 7), 1, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPReceiverEstimatedMaximumBitrateFCI err=%v", err)
	}
	if _, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(valid[:7], nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPReceiverEstimatedMaximumBitrateFCI err=%v", err)
	}
	if _, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(valid, make([]uint32, 0, 2)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPReceiverEstimatedMaximumBitrateFCI dst err=%v", err)
	}
	if out, err := AppendRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 0, 7), 1, nil); !errors.Is(err, ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short AppendRTCPReceiverEstimatedMaximumBitrateFCI out=%d err=%v", len(out), err)
	}

	tooMany := make([]uint32, RTCPReceiverEstimatedMaximumBitrateMaxSSRCs+1)
	if _, err := RTCPReceiverEstimatedMaximumBitrateFCISize(tooMany); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("too many REMB SSRCs size err=%v", err)
	}
	if _, err := PutRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 4096), 1, tooMany); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("too many REMB SSRCs put err=%v", err)
	}
	if _, err := PutRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, RTCPReceiverEstimatedMaximumBitrateFCIMinSize), uint64(1)<<63, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("too high REMB bitrate err=%v", err)
	}

	invalidID := append([]byte(nil), valid...)
	invalidID[0] = 'N'
	if _, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(invalidID, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid REMB unique ID err=%v", err)
	}
	countMismatch := append([]byte(nil), valid...)
	countMismatch[4]++
	if _, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(countMismatch, make([]uint32, 0, 4)); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid REMB SSRC count err=%v", err)
	}
	shiftOverflow := []byte{'R', 'E', 'M', 'B', 0, 63 << 2, 0, 2}
	if _, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(shiftOverflow, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("overflow REMB bitrate err=%v", err)
	}
	int64Overflow := []byte{'R', 'E', 'M', 'B', 0, 56 << 2, 0, 200}
	if _, err := ParseRTCPReceiverEstimatedMaximumBitrateFCI(int64Overflow, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("int64 overflow REMB bitrate err=%v", err)
	}
}

func TestRTCPGenericNACKPairsRoundTrip(t *testing.T) {
	pairs := []RTCPGenericNACKPair{
		{PacketID: 0x1234, LostPacketBitmask: 0x8001},
		{PacketID: 0xfffe, LostPacketBitmask: 0x0003},
	}
	size, err := RTCPGenericNACKPairsSize(pairs)
	if err != nil {
		t.Fatalf("RTCPGenericNACKPairsSize: %v", err)
	}
	if size != len(pairs)*RTCPGenericNACKPairSize {
		t.Fatalf("NACK size=%d", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPGenericNACKPairs(buf, pairs)
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
	parsed, err := ParseRTCPGenericNACKPairs(buf, make([]RTCPGenericNACKPair, 0, len(pairs)))
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
	appended, err := AppendRTCPGenericNACKPairs(prefix, pairs)
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairs: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended NACK bytes=%#v", appended)
	}
}

func TestRTCPGenericNACKPairsForLostSequenceNumbers(t *testing.T) {
	pairs, err := AppendRTCPGenericNACKPairsForLostSequenceNumbers(
		make([]RTCPGenericNACKPair, 0, 2),
		[]uint16{100, 101, 104, 116, 117, 130},
	)
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers: %v", err)
	}
	want := []RTCPGenericNACKPair{
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

	wrapped, err := AppendRTCPGenericNACKPairsForLostSequenceNumbers(
		make([]RTCPGenericNACKPair, 0, 2),
		[]uint16{0xfffe, 0xffff, 0x0000, 0x000f},
	)
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers wrap: %v", err)
	}
	wantWrapped := []RTCPGenericNACKPair{
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

	prefix := make([]RTCPGenericNACKPair, 1, 3)
	prefix[0] = RTCPGenericNACKPair{PacketID: 7}
	prefixed, err := AppendRTCPGenericNACKPairsForLostSequenceNumbers(prefix[:1:3], []uint16{20})
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers prefixed: %v", err)
	}
	if len(prefixed) != 2 || prefixed[0] != prefix[0] || prefixed[1] != (RTCPGenericNACKPair{PacketID: 20}) {
		t.Fatalf("prefixed=%+v", prefixed)
	}

	if out, err := AppendRTCPGenericNACKPairsForLostSequenceNumbers(nil, nil); err != nil || len(out) != 0 {
		t.Fatalf("empty out=%d err=%v", len(out), err)
	}
	if _, err := AppendRTCPGenericNACKPairsForLostSequenceNumbers(make([]RTCPGenericNACKPair, 0, 1), []uint16{10, 27}); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short NACK pair dst err=%v want %v", err, ErrRTCPShortBuffer)
	}
	for _, lost := range [][]uint16{
		{10, 10},
		{10, 9},
	} {
		if _, err := AppendRTCPGenericNACKPairsForLostSequenceNumbers(make([]RTCPGenericNACKPair, 0, 2), lost); !errors.Is(err, ErrRTCPInvalidFeedback) {
			t.Fatalf("lost=%v err=%v want %v", lost, err, ErrRTCPInvalidFeedback)
		}
	}
}

func TestRTCPGenericNACKPairsForLostSequenceNumbersAllocs(t *testing.T) {
	lost := []uint16{0xfffc, 0xfffd, 0x0001, 0x0003, 0x0015}
	pairs := make([]RTCPGenericNACKPair, 0, 3)
	allocs := testing.AllocsPerRun(1000, func() {
		var err error
		pairs, err = AppendRTCPGenericNACKPairsForLostSequenceNumbers(pairs[:0], lost)
		if err != nil {
			t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("AppendRTCPGenericNACKPairsForLostSequenceNumbers allocs/run=%f want 0", allocs)
	}
}

func TestRTCPGenericNACKPairsRejectsInvalid(t *testing.T) {
	pair := RTCPGenericNACKPair{PacketID: 1, LostPacketBitmask: 2}
	var buf [RTCPGenericNACKPairSize]byte
	if _, err := PutRTCPGenericNACKPair(buf[:1], pair); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPGenericNACKPair err=%v", err)
	}
	if _, _, err := ParseRTCPGenericNACKPair(buf[:1]); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPGenericNACKPair err=%v", err)
	}
	pairs := []RTCPGenericNACKPair{pair, pair}
	size, err := RTCPGenericNACKPairsSize(pairs)
	if err != nil {
		t.Fatalf("RTCPGenericNACKPairsSize valid: %v", err)
	}
	if _, err := PutRTCPGenericNACKPairs(make([]byte, size-1), pairs); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPGenericNACKPairs err=%v", err)
	}
	if out, err := AppendRTCPGenericNACKPairs(make([]byte, 0, size-1), pairs); !errors.Is(err, ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short AppendRTCPGenericNACKPairs out=%d err=%v", len(out), err)
	}
	if _, err := ParseRTCPGenericNACKPairs(make([]byte, size), make([]RTCPGenericNACKPair, 0, 1)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPGenericNACKPairs dst err=%v", err)
	}
	if _, err := ParseRTCPGenericNACKPairs(make([]byte, size-1), nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("ragged ParseRTCPGenericNACKPairs err=%v", err)
	}
}

func TestRTCPTransportFeedbackFCIOneBitRoundTrip(t *testing.T) {
	feedback := RTCPTransportFeedback{
		BaseSequenceNumber:  0x0001,
		ReferenceTimeTicks:  0x010203,
		FeedbackPacketCount: 0x7a,
		DeltasPresent:       true,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 0x0001, Received: true, DeltaTicks: 4},
			{SequenceNumber: 0x0002, Received: true, DeltaTicks: 4},
			{SequenceNumber: 0x0003},
			{SequenceNumber: 0x0004},
			{SequenceNumber: 0x0005, Received: true, DeltaTicks: 4},
		},
	}
	size, err := RTCPTransportFeedbackFCISize(feedback)
	if err != nil {
		t.Fatalf("RTCPTransportFeedbackFCISize: %v", err)
	}
	if size != 13 {
		t.Fatalf("transport feedback size=%d want 13", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPTransportFeedbackFCI(buf, feedback)
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

	base := make([]RTCPTransportFeedbackPacket, 1, 1+len(feedback.Packets))
	base[0].SequenceNumber = 0xdead
	parsed, err := ParseRTCPTransportFeedbackFCI(buf, base[:1:cap(base)])
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
	appended, err := AppendRTCPTransportFeedbackFCI(prefix, feedback)
	if err != nil {
		t.Fatalf("AppendRTCPTransportFeedbackFCI: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(buf) {
		t.Fatalf("appended transport feedback bytes=%#v", appended)
	}
}

func TestRTCPTransportFeedbackFCITwoBitRoundTrip(t *testing.T) {
	feedback := RTCPTransportFeedback{
		BaseSequenceNumber:  0xfffe,
		ReferenceTimeTicks:  0x00ffff,
		FeedbackPacketCount: 0x02,
		DeltasPresent:       true,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 0xfffe, Received: true, DeltaTicks: -4},
			{SequenceNumber: 0xffff},
			{SequenceNumber: 0x0000, Received: true, DeltaTicks: 300},
			{SequenceNumber: 0x0001, Received: true, DeltaTicks: 0},
		},
	}
	buf := make([]byte, 15)
	n, err := PutRTCPTransportFeedbackFCI(buf, feedback)
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
	parsed, err := ParseRTCPTransportFeedbackFCI(buf, make([]RTCPTransportFeedbackPacket, 0, len(feedback.Packets)))
	if err != nil {
		t.Fatalf("ParseRTCPTransportFeedbackFCI: %v", err)
	}
	assertRTCPTransportFeedbackPackets(t, parsed.Packets, feedback.Packets)
}

func TestRTCPTransportFeedbackFCIRunLengthAndNoTimestamp(t *testing.T) {
	missing := RTCPTransportFeedback{
		BaseSequenceNumber:  100,
		ReferenceTimeTicks:  0,
		FeedbackPacketCount: 1,
		DeltasPresent:       true,
		Packets:             make([]RTCPTransportFeedbackPacket, 20),
	}
	for i := range missing.Packets {
		missing.Packets[i].SequenceNumber = missing.BaseSequenceNumber + uint16(i)
	}
	buf := make([]byte, 10)
	if _, err := PutRTCPTransportFeedbackFCI(buf, missing); err != nil {
		t.Fatalf("Put run-length transport feedback: %v", err)
	}
	want := []byte{0x00, 0x64, 0x00, 0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x14}
	if string(buf) != string(want) {
		t.Fatalf("run-length transport feedback=%#v want %#v", buf, want)
	}

	noTimestamp := RTCPTransportFeedback{
		BaseSequenceNumber:  10,
		ReferenceTimeTicks:  7,
		FeedbackPacketCount: 3,
		DeltasPresent:       false,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 10, Received: true},
			{SequenceNumber: 11},
			{SequenceNumber: 12, Received: true},
		},
	}
	noTimestampBuf := make([]byte, 10)
	if _, err := PutRTCPTransportFeedbackFCI(noTimestampBuf, noTimestamp); err != nil {
		t.Fatalf("Put no-timestamp transport feedback: %v", err)
	}
	wantNoTimestamp := []byte{0x00, 0x0a, 0x00, 0x03, 0x00, 0x00, 0x07, 0x03, 0xa8, 0x00}
	if string(noTimestampBuf) != string(wantNoTimestamp) {
		t.Fatalf("no-timestamp transport feedback=%#v want %#v", noTimestampBuf, wantNoTimestamp)
	}
	parsed, err := ParseRTCPTransportFeedbackFCI(noTimestampBuf, make([]RTCPTransportFeedbackPacket, 0, 3))
	if err != nil {
		t.Fatalf("Parse no-timestamp transport feedback: %v", err)
	}
	if parsed.DeltasPresent {
		t.Fatalf("parsed no-timestamp feedback reported deltas present")
	}
	assertRTCPTransportFeedbackPackets(t, parsed.Packets, noTimestamp.Packets)
}

func TestRTCPTransportFeedbackFCIRejectsInvalid(t *testing.T) {
	valid := RTCPTransportFeedback{
		BaseSequenceNumber:  1,
		ReferenceTimeTicks:  2,
		FeedbackPacketCount: 3,
		DeltasPresent:       true,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 1, Received: true, DeltaTicks: 4},
		},
	}
	if _, err := PutRTCPTransportFeedbackFCI(make([]byte, 9), valid); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0}, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPTransportFeedbackFCI header err=%v", err)
	}
	if _, err := ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0}, make([]RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPTransportFeedbackFCI chunk err=%v", err)
	}
	if _, err := ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 0, 0, 0, 0, 0}, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("zero status-count ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0x60, 0x01}, make([]RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("reserved run status ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0xf0, 0x00}, make([]RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("reserved vector status ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0xc0, 0x00, 0xff}, make([]RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("truncated nonzero delta ParseRTCPTransportFeedbackFCI err=%v", err)
	}
	if _, err := ParseRTCPTransportFeedbackFCI([]byte{0, 1, 0, 1, 0, 0, 0, 0, 0x20, 0x01}, nil); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short dst ParseRTCPTransportFeedbackFCI err=%v", err)
	}

	badSequence := valid
	badSequence.Packets = append([]RTCPTransportFeedbackPacket(nil), valid.Packets...)
	badSequence.Packets[0].SequenceNumber = 2
	if _, err := RTCPTransportFeedbackFCISize(badSequence); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad sequence transport feedback err=%v", err)
	}
	badReference := valid
	badReference.ReferenceTimeTicks = RTCPTransportFeedbackMaxReferenceTimeTicks + 1
	if _, err := RTCPTransportFeedbackFCISize(badReference); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad reference time transport feedback err=%v", err)
	}
	badNoTimestamp := valid
	badNoTimestamp.DeltasPresent = false
	if _, err := RTCPTransportFeedbackFCISize(badNoTimestamp); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad no-timestamp transport feedback err=%v", err)
	}
	badMissingDelta := valid
	badMissingDelta.Packets = []RTCPTransportFeedbackPacket{{SequenceNumber: 1, DeltaTicks: 1}}
	if _, err := RTCPTransportFeedbackFCISize(badMissingDelta); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad missing packet delta transport feedback err=%v", err)
	}
	empty := valid
	empty.Packets = nil
	if _, err := RTCPTransportFeedbackFCISize(empty); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("empty transport feedback err=%v", err)
	}
}

func TestRTCPTransportFeedbackPacketReceptionTimeline(t *testing.T) {
	feedback := RTCPTransportFeedback{
		BaseSequenceNumber:  0xfffe,
		ReferenceTimeTicks:  2,
		FeedbackPacketCount: 9,
		DeltasPresent:       true,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 0xfffe, Received: true, DeltaTicks: 4},
			{SequenceNumber: 0xffff},
			{SequenceNumber: 0x0000, Received: true, DeltaTicks: -2},
			{SequenceNumber: 0x0001, Received: true, DeltaTicks: 8},
		},
	}

	if got := RTCPTransportFeedbackReferenceTimeMicros(feedback); got != 128_000 {
		t.Fatalf("reference time micros=%d want 128000", got)
	}
	summary, err := SummarizeRTCPTransportFeedback(feedback)
	if err != nil {
		t.Fatalf("SummarizeRTCPTransportFeedback: %v", err)
	}
	if summary.BaseSequenceNumber != 0xfffe ||
		summary.LastSequenceNumber != 0x0001 ||
		summary.FeedbackPacketCount != 9 ||
		summary.PacketCount != 4 ||
		summary.ReceivedCount != 3 ||
		summary.NotReceivedCount != 1 ||
		summary.FirstReceivedSequenceNumber != 0xfffe ||
		summary.LastReceivedSequenceNumber != 0x0001 ||
		!summary.DeltasPresent ||
		!summary.ReceiveTimePresent ||
		summary.ReferenceTimeMicros != 128_000 ||
		summary.FirstReceiveTimeMicros != 129_000 ||
		summary.LastReceiveTimeMicros != 130_500 ||
		summary.MinReceiveDeltaMicros != -500 ||
		summary.MaxReceiveDeltaMicros != 2_000 {
		t.Fatalf("summary=%+v", summary)
	}

	prefix := make([]RTCPTransportFeedbackPacketReception, 1, 1+len(feedback.Packets))
	prefix[0] = RTCPTransportFeedbackPacketReception{
		SequenceNumber:     0xdddd,
		Received:           true,
		ReceiveTimePresent: true,
		ReceiveTimeMicros:  77,
	}
	receptions, err := AppendRTCPTransportFeedbackPacketReceptions(prefix[:1:cap(prefix)], feedback)
	if err != nil {
		t.Fatalf("AppendRTCPTransportFeedbackPacketReceptions: %v", err)
	}
	if receptions[0] != prefix[0] {
		t.Fatalf("reception prefix clobbered: %+v", receptions[0])
	}
	want := []RTCPTransportFeedbackPacketReception{
		{SequenceNumber: 0xfffe, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 129_000},
		{SequenceNumber: 0xffff},
		{SequenceNumber: 0x0000, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 128_500},
		{SequenceNumber: 0x0001, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 130_500},
	}
	got := receptions[1:]
	if len(got) != len(want) {
		t.Fatalf("receptions len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("reception[%d]=%+v want %+v", i, got[i], want[i])
		}
	}

	noTimestamp := RTCPTransportFeedback{
		BaseSequenceNumber:  10,
		ReferenceTimeTicks:  3,
		FeedbackPacketCount: 10,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 10, Received: true},
			{SequenceNumber: 11},
			{SequenceNumber: 12, Received: true},
		},
	}
	summary, err = SummarizeRTCPTransportFeedback(noTimestamp)
	if err != nil {
		t.Fatalf("SummarizeRTCPTransportFeedback no timestamp: %v", err)
	}
	if summary.ReceivedCount != 2 || summary.NotReceivedCount != 1 ||
		summary.ReceiveTimePresent || summary.ReferenceTimeMicros != 192_000 {
		t.Fatalf("no-timestamp summary=%+v", summary)
	}
	receptions, err = AppendRTCPTransportFeedbackPacketReceptions(make([]RTCPTransportFeedbackPacketReception, 0, 3), noTimestamp)
	if err != nil {
		t.Fatalf("Append no-timestamp receptions: %v", err)
	}
	if !receptions[0].Received || receptions[0].ReceiveTimePresent ||
		receptions[1].Received || !receptions[2].Received || receptions[2].ReceiveTimePresent {
		t.Fatalf("no-timestamp receptions=%+v", receptions)
	}

	if _, err := AppendRTCPTransportFeedbackPacketReceptions(make([]RTCPTransportFeedbackPacketReception, 0, 1), feedback); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short reception dst err=%v want %v", err, ErrRTCPShortBuffer)
	}
	bad := feedback
	bad.Packets = append([]RTCPTransportFeedbackPacket(nil), feedback.Packets...)
	bad.Packets[0].SequenceNumber = 7
	if _, err := SummarizeRTCPTransportFeedback(bad); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("bad summary err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
}

func TestRTCPTransportFeedbackForReceivedPackets(t *testing.T) {
	received := []RTCPTransportFeedbackReceivedPacket{
		{SequenceNumber: 0xfffe, ReceiveTimeMicros: 129_000},
		{SequenceNumber: 0x0000, ReceiveTimeMicros: 128_500},
		{SequenceNumber: 0x0002, ReceiveTimeMicros: 130_500},
	}
	packetScratch := make([]RTCPTransportFeedbackPacket, 1, 6)
	packetScratch[0] = RTCPTransportFeedbackPacket{
		SequenceNumber: 0xdddd,
		Received:       true,
		DeltaTicks:     77,
	}
	feedback, err := RTCPTransportFeedbackForReceivedPackets(
		0xfffe,
		0x0002,
		2,
		7,
		received,
		packetScratch[:1:cap(packetScratch)],
	)
	if err != nil {
		t.Fatalf("RTCPTransportFeedbackForReceivedPackets: %v", err)
	}
	if packetScratch[0].SequenceNumber != 0xdddd || !packetScratch[0].Received || packetScratch[0].DeltaTicks != 77 {
		t.Fatalf("packet scratch prefix clobbered: %+v", packetScratch[0])
	}
	if feedback.BaseSequenceNumber != 0xfffe ||
		feedback.ReferenceTimeTicks != 2 ||
		feedback.FeedbackPacketCount != 7 ||
		!feedback.DeltasPresent {
		t.Fatalf("feedback header=%+v", feedback)
	}
	wantPackets := []RTCPTransportFeedbackPacket{
		{SequenceNumber: 0xfffe, Received: true, DeltaTicks: 4},
		{SequenceNumber: 0xffff},
		{SequenceNumber: 0x0000, Received: true, DeltaTicks: -2},
		{SequenceNumber: 0x0001},
		{SequenceNumber: 0x0002, Received: true, DeltaTicks: 8},
	}
	assertRTCPTransportFeedbackPackets(t, feedback.Packets, wantPackets)

	fci, err := AppendRTCPTransportFeedbackFCI(make([]byte, 0, 32), feedback)
	if err != nil {
		t.Fatalf("AppendRTCPTransportFeedbackFCI: %v", err)
	}
	parsed, err := ParseRTCPTransportFeedbackFCI(fci, make([]RTCPTransportFeedbackPacket, 0, len(wantPackets)))
	if err != nil {
		t.Fatalf("ParseRTCPTransportFeedbackFCI: %v", err)
	}
	assertRTCPTransportFeedbackPackets(t, parsed.Packets, wantPackets)

	summary, err := SummarizeRTCPTransportFeedback(parsed)
	if err != nil {
		t.Fatalf("SummarizeRTCPTransportFeedback: %v", err)
	}
	if summary.ReceivedCount != 3 ||
		summary.NotReceivedCount != 2 ||
		summary.FirstReceiveTimeMicros != 129_000 ||
		summary.LastReceiveTimeMicros != 130_500 ||
		summary.MinReceiveDeltaMicros != -500 ||
		summary.MaxReceiveDeltaMicros != 2_000 {
		t.Fatalf("summary=%+v", summary)
	}
	receptions, err := AppendRTCPTransportFeedbackPacketReceptions(
		make([]RTCPTransportFeedbackPacketReception, 0, len(wantPackets)),
		parsed,
	)
	if err != nil {
		t.Fatalf("AppendRTCPTransportFeedbackPacketReceptions: %v", err)
	}
	wantReceptions := []RTCPTransportFeedbackPacketReception{
		{SequenceNumber: 0xfffe, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 129_000},
		{SequenceNumber: 0xffff},
		{SequenceNumber: 0x0000, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 128_500},
		{SequenceNumber: 0x0001},
		{SequenceNumber: 0x0002, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 130_500},
	}
	if len(receptions) != len(wantReceptions) {
		t.Fatalf("receptions len=%d want %d", len(receptions), len(wantReceptions))
	}
	for i := range wantReceptions {
		if receptions[i] != wantReceptions[i] {
			t.Fatalf("reception[%d]=%+v want %+v", i, receptions[i], wantReceptions[i])
		}
	}
}

func TestRTCPTransportFeedbackForReceivedPacketsRoundsDeltas(t *testing.T) {
	received := []RTCPTransportFeedbackReceivedPacket{
		{SequenceNumber: 10, ReceiveTimeMicros: 126},
		{SequenceNumber: 11, ReceiveTimeMicros: 624},
	}
	feedback, err := RTCPTransportFeedbackForReceivedPackets(
		10,
		11,
		0,
		1,
		received,
		make([]RTCPTransportFeedbackPacket, 0, 2),
	)
	if err != nil {
		t.Fatalf("RTCPTransportFeedbackForReceivedPackets: %v", err)
	}
	wantPackets := []RTCPTransportFeedbackPacket{
		{SequenceNumber: 10, Received: true, DeltaTicks: 1},
		{SequenceNumber: 11, Received: true, DeltaTicks: 1},
	}
	assertRTCPTransportFeedbackPackets(t, feedback.Packets, wantPackets)
	receptions, err := AppendRTCPTransportFeedbackPacketReceptions(
		make([]RTCPTransportFeedbackPacketReception, 0, 2),
		feedback,
	)
	if err != nil {
		t.Fatalf("AppendRTCPTransportFeedbackPacketReceptions: %v", err)
	}
	wantReceptions := []RTCPTransportFeedbackPacketReception{
		{SequenceNumber: 10, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 250},
		{SequenceNumber: 11, Received: true, ReceiveTimePresent: true, ReceiveTimeMicros: 500},
	}
	for i := range wantReceptions {
		if receptions[i] != wantReceptions[i] {
			t.Fatalf("reception[%d]=%+v want %+v", i, receptions[i], wantReceptions[i])
		}
	}
}

func TestAppendRTCPTransportFeedbackPacketsForReceivedPacketsRejectsInvalid(t *testing.T) {
	if _, err := AppendRTCPTransportFeedbackPacketsForReceivedPackets(
		make([]RTCPTransportFeedbackPacket, 0, 1),
		1,
		2,
		0,
		nil,
	); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short packet scratch err=%v want %v", err, ErrRTCPShortBuffer)
	}
	if _, err := AppendRTCPTransportFeedbackPacketsForReceivedPackets(
		nil,
		0,
		0xffff,
		0,
		nil,
	); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("full 16-bit packet window err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
	if _, err := AppendRTCPTransportFeedbackPacketsForReceivedPackets(
		nil,
		10,
		10,
		RTCPTransportFeedbackMaxReferenceTimeTicks+1,
		nil,
	); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("reference overflow err=%v want %v", err, ErrRTCPInvalidFeedback)
	}

	tests := []struct {
		name     string
		received []RTCPTransportFeedbackReceivedPacket
	}{
		{
			name:     "outside-window",
			received: []RTCPTransportFeedbackReceivedPacket{{SequenceNumber: 13, ReceiveTimeMicros: 0}},
		},
		{
			name: "duplicate",
			received: []RTCPTransportFeedbackReceivedPacket{
				{SequenceNumber: 10, ReceiveTimeMicros: 0},
				{SequenceNumber: 10, ReceiveTimeMicros: 250},
			},
		},
		{
			name: "unsorted",
			received: []RTCPTransportFeedbackReceivedPacket{
				{SequenceNumber: 11, ReceiveTimeMicros: 0},
				{SequenceNumber: 10, ReceiveTimeMicros: 250},
			},
		},
		{
			name:     "positive-delta-overflow",
			received: []RTCPTransportFeedbackReceivedPacket{{SequenceNumber: 10, ReceiveTimeMicros: 8_192_000}},
		},
		{
			name:     "negative-delta-overflow",
			received: []RTCPTransportFeedbackReceivedPacket{{SequenceNumber: 10, ReceiveTimeMicros: -8_192_250}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prefix := make([]RTCPTransportFeedbackPacket, 1, 4)
			prefix[0].SequenceNumber = 0xeeee
			out, err := AppendRTCPTransportFeedbackPacketsForReceivedPackets(
				prefix[:1:4],
				10,
				12,
				0,
				tc.received,
			)
			if !errors.Is(err, ErrRTCPInvalidFeedback) {
				t.Fatalf("err=%v want %v", err, ErrRTCPInvalidFeedback)
			}
			if len(out) != 1 || out[0].SequenceNumber != 0xeeee {
				t.Fatalf("invalid append changed returned prefix: %+v", out)
			}
		})
	}

	packets, err := AppendRTCPTransportFeedbackPacketsForReceivedPackets(
		make([]RTCPTransportFeedbackPacket, 0, 3),
		100,
		102,
		0,
		nil,
	)
	if err != nil {
		t.Fatalf("all-missing packet window: %v", err)
	}
	want := []RTCPTransportFeedbackPacket{
		{SequenceNumber: 100},
		{SequenceNumber: 101},
		{SequenceNumber: 102},
	}
	assertRTCPTransportFeedbackPackets(t, packets, want)
}

func TestRTCPTransportFeedbackForReceivedPacketsAllocs(t *testing.T) {
	received := []RTCPTransportFeedbackReceivedPacket{
		{SequenceNumber: 100, ReceiveTimeMicros: 64_000},
		{SequenceNumber: 102, ReceiveTimeMicros: 65_000},
		{SequenceNumber: 103, ReceiveTimeMicros: 64_750},
	}
	packetScratch := make([]RTCPTransportFeedbackPacket, 0, 4)
	feedback, err := RTCPTransportFeedbackForReceivedPackets(100, 103, 1, 3, received, packetScratch[:0])
	if err != nil {
		t.Fatalf("RTCPTransportFeedbackForReceivedPackets: %v", err)
	}
	fciSize, err := RTCPTransportFeedbackFCISize(feedback)
	if err != nil {
		t.Fatalf("RTCPTransportFeedbackFCISize: %v", err)
	}
	fciScratch := make([]byte, 0, fciSize)
	parseScratch := make([]RTCPTransportFeedbackPacket, 0, 4)
	receptionScratch := make([]RTCPTransportFeedbackPacketReception, 0, 4)
	allocs := testing.AllocsPerRun(1000, func() {
		feedback, err := RTCPTransportFeedbackForReceivedPackets(100, 103, 1, 3, received, packetScratch[:0])
		if err != nil {
			t.Fatalf("RTCPTransportFeedbackForReceivedPackets: %v", err)
		}
		fci, err := AppendRTCPTransportFeedbackFCI(fciScratch[:0], feedback)
		if err != nil {
			t.Fatalf("AppendRTCPTransportFeedbackFCI: %v", err)
		}
		parsed, err := ParseRTCPTransportFeedbackFCI(fci, parseScratch[:0])
		if err != nil {
			t.Fatalf("ParseRTCPTransportFeedbackFCI: %v", err)
		}
		receptions, err := AppendRTCPTransportFeedbackPacketReceptions(receptionScratch[:0], parsed)
		if err != nil {
			t.Fatalf("AppendRTCPTransportFeedbackPacketReceptions: %v", err)
		}
		if len(receptions) != 4 {
			t.Fatalf("receptions=%d want 4", len(receptions))
		}
	})
	if allocs != 0 {
		t.Fatalf("Transport-CC received-packet builder allocs/run=%f want 0", allocs)
	}
}

func TestEncoderWebRTCRTCPTransportFeedback(t *testing.T) {
	first := RTCPTransportFeedback{
		BaseSequenceNumber:  100,
		ReferenceTimeTicks:  1,
		FeedbackPacketCount: 1,
		DeltasPresent:       true,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 100, Received: true, DeltaTicks: 4},
			{SequenceNumber: 101},
		},
	}
	second := RTCPTransportFeedback{
		BaseSequenceNumber:  200,
		ReferenceTimeTicks:  2,
		FeedbackPacketCount: 2,
		DeltasPresent:       true,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 200, Received: true, DeltaTicks: 8},
		},
	}
	packet := testAV1RTCPTransportFeedbackPacket(t, first)
	parsed, ok, err := EncoderWebRTCRTCPFeedbackTransportFeedback(
		packet,
		make([]RTCPTransportFeedbackPacket, 0, len(first.Packets)),
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPFeedbackTransportFeedback: %v", err)
	}
	if !ok || parsed.BaseSequenceNumber != first.BaseSequenceNumber || len(parsed.Packets) != len(first.Packets) {
		t.Fatalf("parsed=%+v ok=%v", parsed, ok)
	}

	for _, packet := range []RTCPFeedbackPacket{
		{PacketType: RTCPRTPFBPacketType, FMT: RTCPRTPFBGenericNACKFMT},
		{PacketType: RTCPPSFBPacketType, FMT: RTCPPSFBApplicationLayerFeedbackFMT},
	} {
		parsed, ok, err = EncoderWebRTCRTCPFeedbackTransportFeedback(packet, nil)
		if err != nil || ok || len(parsed.Packets) != 0 {
			t.Fatalf("non-TWCC packet %+v returned %+v ok=%v err=%v", packet, parsed, ok, err)
		}
	}
	if _, _, err := EncoderWebRTCRTCPFeedbackTransportFeedback(RTCPFeedbackPacket{
		PacketType: RTCPSenderReportPacketType,
		FMT:        RTCPRTPFBTransportFeedbackFMT,
	}, nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("invalid packet type err=%v want %v", err, ErrRTCPInvalidFeedback)
	}
	if _, _, err := EncoderWebRTCRTCPFeedbackTransportFeedback(packet, make([]RTCPTransportFeedbackPacket, 0, 1)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short transport scratch err=%v want %v", err, ErrRTCPShortBuffer)
	}

	compound, err := AppendRTCPSenderReportPacket(make([]byte, 0, 256), RTCPSenderReport{
		SenderSSRC: 0x01020304,
	})
	if err != nil {
		t.Fatalf("AppendRTCPSenderReportPacket: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, testAV1RTCPTransportFeedbackPacket(t, first))
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket first TWCC: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, testAV1RTCPReceiverEstimatedMaximumBitratePacket(t, 500_000, []uint32{0x11112222}))
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket REMB: %v", err)
	}
	compound, err = AppendRTCPFeedbackPacket(compound, testAV1RTCPTransportFeedbackPacket(t, second))
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket second TWCC: %v", err)
	}

	packetScratch := make([]RTCPPacket, 1, 5)
	packetScratch[0] = RTCPPacket{PacketType: RTCPByePacketType, Payload: []byte{0xaa}}
	feedbackScratch := make([]RTCPTransportFeedback, 1, 3)
	feedbackScratch[0].BaseSequenceNumber = 0xdddd
	transportScratch := make([]RTCPTransportFeedbackPacket, 1, 1+len(first.Packets)+len(second.Packets))
	transportScratch[0].SequenceNumber = 0xeeee
	feedbacks, packets, err := EncoderWebRTCRTCPCompoundPacketsTransportFeedback(
		compound,
		packetScratch[:1:5],
		feedbackScratch[:1:3],
		transportScratch[:1:cap(transportScratch)],
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTCPCompoundPacketsTransportFeedback: %v", err)
	}
	if len(packets) != 4 || len(feedbacks) != 2 ||
		feedbacks[0].BaseSequenceNumber != first.BaseSequenceNumber ||
		feedbacks[1].BaseSequenceNumber != second.BaseSequenceNumber ||
		len(feedbacks[0].Packets) != len(first.Packets) ||
		len(feedbacks[1].Packets) != len(second.Packets) {
		t.Fatalf("feedbacks=%+v packets=%d", feedbacks, len(packets))
	}
	if packetScratch[0].PacketType != RTCPByePacketType ||
		feedbackScratch[0].BaseSequenceNumber != 0xdddd ||
		transportScratch[0].SequenceNumber != 0xeeee {
		t.Fatalf("scratch prefix clobbered packet=%+v feedback=%+v transport=%+v", packetScratch[0], feedbackScratch[0], transportScratch[0])
	}

	if _, err := EncoderWebRTCRTCPPacketsTransportFeedback(
		packets,
		make([]RTCPTransportFeedback, 0, 1),
		make([]RTCPTransportFeedbackPacket, 0, len(first.Packets)+len(second.Packets)),
	); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short feedback scratch err=%v want %v", err, ErrRTCPShortBuffer)
	}
}

func TestEncoderWebRTCRTCPCompoundPacketsTransportFeedbackAllocs(t *testing.T) {
	feedback := RTCPTransportFeedback{
		BaseSequenceNumber:  100,
		ReferenceTimeTicks:  1,
		FeedbackPacketCount: 1,
		DeltasPresent:       true,
		Packets: []RTCPTransportFeedbackPacket{
			{SequenceNumber: 100, Received: true, DeltaTicks: 4},
			{SequenceNumber: 101},
			{SequenceNumber: 102, Received: true, DeltaTicks: 8},
		},
	}
	compound := testAV1RTCPFeedbackCompound(t, testAV1RTCPTransportFeedbackPacket(t, feedback))
	packetScratch := make([]RTCPPacket, 0, 1)
	feedbackScratch := make([]RTCPTransportFeedback, 0, 1)
	transportScratch := make([]RTCPTransportFeedbackPacket, 0, len(feedback.Packets))
	receptionScratch := make([]RTCPTransportFeedbackPacketReception, 0, len(feedback.Packets))
	allocs := testing.AllocsPerRun(1000, func() {
		feedbacks, packets, err := EncoderWebRTCRTCPCompoundPacketsTransportFeedback(
			compound,
			packetScratch[:0],
			feedbackScratch[:0],
			transportScratch[:0],
		)
		if err != nil {
			t.Fatalf("EncoderWebRTCRTCPCompoundPacketsTransportFeedback: %v", err)
		}
		if len(feedbacks) != 1 || len(packets) != 1 {
			t.Fatalf("feedback len=%d packet len=%d", len(feedbacks), len(packets))
		}
		summary, err := SummarizeRTCPTransportFeedback(feedbacks[0])
		if err != nil {
			t.Fatalf("SummarizeRTCPTransportFeedback: %v", err)
		}
		if summary.ReceivedCount != 2 || summary.NotReceivedCount != 1 {
			t.Fatalf("summary=%+v", summary)
		}
		receptions, err := AppendRTCPTransportFeedbackPacketReceptions(receptionScratch[:0], feedbacks[0])
		if err != nil {
			t.Fatalf("AppendRTCPTransportFeedbackPacketReceptions: %v", err)
		}
		if len(receptions) != len(feedback.Packets) {
			t.Fatalf("receptions=%d want %d", len(receptions), len(feedback.Packets))
		}
	})
	if allocs != 0 {
		t.Fatalf("Transport-CC feedback helper allocs/run=%f want 0", allocs)
	}
}

func TestRTCPFullIntraRequestEntriesRoundTrip(t *testing.T) {
	entries := []RTCPFullIntraRequestEntry{
		{SSRC: 0x11223344, SequenceNumber: 7},
		{SSRC: 0xaabbccdd, SequenceNumber: 250},
	}
	size, err := RTCPFullIntraRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("RTCPFullIntraRequestEntriesSize: %v", err)
	}
	if size != len(entries)*RTCPFullIntraRequestEntrySize {
		t.Fatalf("FIR size=%d", size)
	}
	buf := make([]byte, size)
	n, err := PutRTCPFullIntraRequestEntries(buf, entries)
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
	parsed, err := ParseRTCPFullIntraRequestEntries(buf, make([]RTCPFullIntraRequestEntry, 0, len(entries)))
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
	appended, err := AppendRTCPFullIntraRequestEntries(prefix, entries)
	if err != nil {
		t.Fatalf("AppendRTCPFullIntraRequestEntries: %v", err)
	}
	clean := make([]byte, size)
	if _, err := PutRTCPFullIntraRequestEntries(clean, entries); err != nil {
		t.Fatalf("Put clean FIR: %v", err)
	}
	if len(appended) != 1+size || appended[0] != 0xaa || string(appended[1:]) != string(clean) {
		t.Fatalf("appended FIR bytes=%#v", appended)
	}
}

func TestRTCPFullIntraRequestEntriesRejectsInvalid(t *testing.T) {
	entry := RTCPFullIntraRequestEntry{SSRC: 1, SequenceNumber: 2}
	var buf [RTCPFullIntraRequestEntrySize]byte
	if _, err := PutRTCPFullIntraRequestEntry(buf[:1], entry); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPFullIntraRequestEntry err=%v", err)
	}
	if _, _, err := ParseRTCPFullIntraRequestEntry(buf[:1]); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPFullIntraRequestEntry err=%v", err)
	}
	entries := []RTCPFullIntraRequestEntry{entry, entry}
	size, err := RTCPFullIntraRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("RTCPFullIntraRequestEntriesSize valid: %v", err)
	}
	if _, err := PutRTCPFullIntraRequestEntries(make([]byte, size-1), entries); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short PutRTCPFullIntraRequestEntries err=%v", err)
	}
	if out, err := AppendRTCPFullIntraRequestEntries(make([]byte, 0, size-1), entries); !errors.Is(err, ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short AppendRTCPFullIntraRequestEntries out=%d err=%v", len(out), err)
	}
	if _, err := ParseRTCPFullIntraRequestEntries(make([]byte, size), make([]RTCPFullIntraRequestEntry, 0, 1)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short ParseRTCPFullIntraRequestEntries dst err=%v", err)
	}
	if _, err := ParseRTCPFullIntraRequestEntries(make([]byte, size-1), nil); !errors.Is(err, ErrRTCPInvalidFeedback) {
		t.Fatalf("ragged ParseRTCPFullIntraRequestEntries err=%v", err)
	}
}

func TestAV1RTCPLayerRefreshLayerIndexRoundTrip(t *testing.T) {
	var buf [2]byte
	index := AV1RTCPLayerRefreshLayerIndex{TemporalID: 7, SpatialID: 3}
	n, err := PutAV1RTCPLayerRefreshLayerIndex(buf[:], index)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshLayerIndex: %v", err)
	}
	if n != 2 || buf != [2]byte{0x07, 0x03} {
		t.Fatalf("encoded layer index n=%d bytes=%#v", n, buf)
	}
	got, n, err := ParseAV1RTCPLayerRefreshLayerIndex(buf[:])
	if err != nil {
		t.Fatalf("ParseAV1RTCPLayerRefreshLayerIndex: %v", err)
	}
	if n != 2 || got != index {
		t.Fatalf("parsed layer index n=%d got=%+v want=%+v", n, got, index)
	}

	got, _, err = ParseAV1RTCPLayerRefreshLayerIndex([]byte{0xfa, 0xf1})
	if err != nil {
		t.Fatalf("Parse with reserved bits returned error: %v", err)
	}
	if got != (AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1}) {
		t.Fatalf("reserved-bit parse = %+v", got)
	}
	if _, _, err := ParseAV1RTCPLayerRefreshLayerIndex([]byte{0x00, 0x04}); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("Parse high SID bit error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if _, err := PutAV1RTCPLayerRefreshLayerIndex(buf[:1], index); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short Put error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, _, err := ParseAV1RTCPLayerRefreshLayerIndex(buf[:1]); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short Parse error = %v, want ErrRTCPShortBuffer", err)
	}
}

func TestAV1RTCPLayerRefreshRequestEntryRoundTrip(t *testing.T) {
	entry := AV1RTCPLayerRefreshRequestEntry{
		SSRC:           0x11223344,
		SequenceNumber: 250,
		PayloadType:    98,
		Target:         AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1},
		CurrentPresent: true,
		Current:        AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	var buf [12]byte
	n, err := PutAV1RTCPLayerRefreshRequestEntry(buf[:], entry)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshRequestEntry: %v", err)
	}
	wantBytes := [12]byte{0x11, 0x22, 0x33, 0x44, 0xfa, 0xe2, 0x00, 0x00, 0x02, 0x01, 0x00, 0x00}
	if n != 12 || buf != wantBytes {
		t.Fatalf("encoded LRR entry n=%d bytes=%#v want=%#v", n, buf, wantBytes)
	}
	got, n, err := ParseAV1RTCPLayerRefreshRequestEntry(buf[:])
	if err != nil {
		t.Fatalf("ParseAV1RTCPLayerRefreshRequestEntry: %v", err)
	}
	if n != 12 || got != entry {
		t.Fatalf("parsed LRR entry n=%d got=%+v want=%+v", n, got, entry)
	}

	noCurrent := entry
	noCurrent.CurrentPresent = false
	noCurrent.Current = AV1RTCPLayerRefreshLayerIndex{}
	n, err = PutAV1RTCPLayerRefreshRequestEntry(buf[:], noCurrent)
	if err != nil {
		t.Fatalf("Put no-current LRR entry: %v", err)
	}
	if buf[5] != 98 || buf[10] != 0 || buf[11] != 0 {
		t.Fatalf("no-current encoded bytes=%#v", buf)
	}
	buf[10], buf[11] = 0x07, 0x03
	got, _, err = ParseAV1RTCPLayerRefreshRequestEntry(buf[:])
	if err != nil {
		t.Fatalf("Parse no-current with ignored current field: %v", err)
	}
	if got.CurrentPresent || got.Current != (AV1RTCPLayerRefreshLayerIndex{}) {
		t.Fatalf("no-current parse got=%+v", got)
	}
}

func TestAV1RTCPLayerRefreshRequestEntriesRoundTrip(t *testing.T) {
	entries := []AV1RTCPLayerRefreshRequestEntry{
		{
			SSRC:           0x11223344,
			SequenceNumber: 250,
			PayloadType:    98,
			Target:         AV1RTCPLayerRefreshLayerIndex{TemporalID: 2, SpatialID: 1},
			CurrentPresent: true,
			Current:        AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
		},
		{
			SSRC:           0x55667788,
			SequenceNumber: 251,
			PayloadType:    98,
			Target:         AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 0},
		},
	}
	size, err := AV1RTCPLayerRefreshRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntriesSize: %v", err)
	}
	if size != 2*AV1RTCPLayerRefreshRequestEntrySize {
		t.Fatalf("LRR entries size = %d", size)
	}

	buf := make([]byte, size)
	n, err := PutAV1RTCPLayerRefreshRequestEntries(buf, entries)
	if err != nil {
		t.Fatalf("PutAV1RTCPLayerRefreshRequestEntries: %v", err)
	}
	if n != size {
		t.Fatalf("PutAV1RTCPLayerRefreshRequestEntries n=%d want %d", n, size)
	}

	parsed, err := ParseAV1RTCPLayerRefreshRequestEntries(buf, make([]AV1RTCPLayerRefreshRequestEntry, 0, 2))
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
	appended, err := AppendAV1RTCPLayerRefreshRequestEntries(prefix, entries)
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

	empty, err := ParseAV1RTCPLayerRefreshRequestEntries(nil, parsed[:0])
	if err != nil {
		t.Fatalf("Parse empty entries: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("empty parse len=%d", len(empty))
	}
}

func TestAV1RTCPLayerRefreshRequestRejectsInvalid(t *testing.T) {
	valid := AV1RTCPLayerRefreshRequestEntry{
		PayloadType:    98,
		Target:         AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 1},
		CurrentPresent: true,
		Current:        AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	for _, entry := range []AV1RTCPLayerRefreshRequestEntry{
		{PayloadType: 128, Target: valid.Target},
		{PayloadType: 98, Target: AV1RTCPLayerRefreshLayerIndex{TemporalID: 8}},
		{PayloadType: 98, Target: AV1RTCPLayerRefreshLayerIndex{SpatialID: 4}},
		{PayloadType: 98, Target: valid.Target, Current: AV1RTCPLayerRefreshLayerIndex{TemporalID: 1}},
		{PayloadType: 98, Target: valid.Current, CurrentPresent: true, Current: valid.Target},
		{PayloadType: 98, Target: valid.Target, CurrentPresent: true, Current: valid.Target},
	} {
		if err := entry.Validate(); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
			t.Fatalf("Validate(%+v) error = %v, want ErrRTCPInvalidLayerRefreshRequest", entry, err)
		}
	}

	var buf [12]byte
	if _, err := PutAV1RTCPLayerRefreshRequestEntry(buf[:1], valid); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short Put entry error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, _, err := ParseAV1RTCPLayerRefreshRequestEntry(buf[:1]); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short Parse entry error = %v, want ErrRTCPShortBuffer", err)
	}
	buf = [12]byte{0, 0, 0, 1, 1, 0x80 | 98, 0, 0, 0, 0, 0, 0}
	if _, _, err := ParseAV1RTCPLayerRefreshRequestEntry(buf[:]); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("Parse no-upgrade entry error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}

	validList := []AV1RTCPLayerRefreshRequestEntry{valid, valid}
	validList[1].SSRC = 2
	size, err := AV1RTCPLayerRefreshRequestEntriesSize(validList)
	if err != nil {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntriesSize valid list: %v", err)
	}
	if _, err := PutAV1RTCPLayerRefreshRequestEntries(make([]byte, size-1), validList); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short Put entries error = %v, want ErrRTCPShortBuffer", err)
	}
	if out, err := AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, size-1), validList); !errors.Is(err, ErrRTCPShortBuffer) || len(out) != 0 {
		t.Fatalf("short Append entries out=%d err=%v want ErrRTCPShortBuffer", len(out), err)
	}
	if _, err := ParseAV1RTCPLayerRefreshRequestEntries(make([]byte, size), make([]AV1RTCPLayerRefreshRequestEntry, 0, 1)); !errors.Is(err, ErrRTCPShortBuffer) {
		t.Fatalf("short Parse entries dst error = %v, want ErrRTCPShortBuffer", err)
	}
	if _, err := ParseAV1RTCPLayerRefreshRequestEntries(make([]byte, size-1), nil); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("ragged Parse entries error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	invalidList := []AV1RTCPLayerRefreshRequestEntry{valid}
	invalidList[0].PayloadType = 128
	if _, err := AV1RTCPLayerRefreshRequestEntriesSize(invalidList); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("invalid EntriesSize error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}

	encoded := make([]byte, size)
	if _, err := PutAV1RTCPLayerRefreshRequestEntries(encoded, validList); err != nil {
		t.Fatalf("Put valid list for invalid parse: %v", err)
	}
	copy(encoded[AV1RTCPLayerRefreshRequestEntrySize:], buf[:])
	base := make([]AV1RTCPLayerRefreshRequestEntry, 1, 3)
	base[0].SSRC = 0xdeadbeef
	out, err := ParseAV1RTCPLayerRefreshRequestEntries(encoded, base[:1:cap(base)])
	if !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("invalid Parse entries error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if len(out) != 1 || out[0].SSRC != 0xdeadbeef {
		t.Fatalf("invalid Parse entries returned partial output: %+v", out)
	}
}

func TestEncoderWebRTCValidateLayerRefreshRequest(t *testing.T) {
	cfg := testAV1RTCPEncoderConfig(internalencoder.ScalabilityModeL2T2)
	valid := AV1RTCPLayerRefreshRequestEntry{
		PayloadType:    98,
		Target:         AV1RTCPLayerRefreshLayerIndex{TemporalID: 1, SpatialID: 1},
		CurrentPresent: true,
		Current:        AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 0},
	}
	if err := EncoderWebRTCValidateLayerRefreshRequest(cfg, valid); err != nil {
		t.Fatalf("EncoderWebRTCValidateLayerRefreshRequest valid: %v", err)
	}

	badTemporal := valid
	badTemporal.Target.TemporalID = 2
	if err := EncoderWebRTCValidateLayerRefreshRequest(cfg, badTemporal); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad temporal error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	badSpatial := valid
	badSpatial.Target.SpatialID = 2
	if err := EncoderWebRTCValidateLayerRefreshRequest(cfg, badSpatial); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad spatial error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
	if err := EncoderWebRTCValidateLayerRefreshRequest(EncoderConfig{}, valid); !errors.Is(err, internalencoder.ErrInvalidConfig) {
		t.Fatalf("invalid config error = %v, want ErrEncoderInvalidConfig", err)
	}

	validList := []AV1RTCPLayerRefreshRequestEntry{valid}
	validList = append(validList, valid)
	validList[1].SSRC = 2
	validList[1].Target = AV1RTCPLayerRefreshLayerIndex{TemporalID: 0, SpatialID: 1}
	if err := EncoderWebRTCValidateLayerRefreshRequests(cfg, validList); err != nil {
		t.Fatalf("EncoderWebRTCValidateLayerRefreshRequests valid: %v", err)
	}
	badCurrent := validList
	badCurrent[0].Current.SpatialID = 2
	if err := EncoderWebRTCValidateLayerRefreshRequests(cfg, badCurrent); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
		t.Fatalf("bad current list error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
	}
}

func TestEncoderWebRTCValidateLayerRefreshRequestModeMatrix(t *testing.T) {
	for _, mode := range encoderWebRTCSVCScalabilityModes() {
		t.Run(mode.String(), func(t *testing.T) {
			spatialLayers, temporalLayers, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid scalability mode %s", mode)
			}
			width, height := publicRTCMatrixGeometry(t, mode)
			cfg := publicRTCMatrixConfig(width, height, mode)
			valid := testAV1RTCPValidLayerRefreshEntry(t, mode)
			compound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBLayerRefreshRequestFMT,
				SenderSSRC: 0x01020304,
				MediaSSRC:  0x05060708,
				FCI:        testAV1RTCPLayerRefreshFCI(t, []AV1RTCPLayerRefreshRequestEntry{valid}),
			})
			force, packets, err := EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
				cfg,
				compound,
				make([]RTCPPacket, 0, 1),
				nil,
				make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
			)
			if err != nil {
				t.Fatalf("valid LRR compound decision: %v", err)
			}
			if !force || len(packets) != 1 {
				t.Fatalf("valid LRR force=%v packet len=%d want true,1", force, len(packets))
			}
			target, ok, packets, err := EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
				cfg,
				compound,
				make([]RTCPPacket, 0, 1),
				make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
			)
			if err != nil {
				t.Fatalf("valid LRR compound target: %v", err)
			}
			if !ok || len(packets) != 1 || target != valid.Target {
				t.Fatalf("valid LRR target=%+v ok=%v packet len=%d want %+v,true,1", target, ok, len(packets), valid.Target)
			}

			badTemporal := valid
			badTemporal.Target.TemporalID = temporalLayers
			if err := EncoderWebRTCValidateLayerRefreshRequest(cfg, badTemporal); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
				t.Fatalf("bad temporal target error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
			}

			badSpatial := valid
			badSpatial.Target.SpatialID = spatialLayers
			badCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
				PacketType: RTCPPSFBPacketType,
				FMT:        RTCPPSFBLayerRefreshRequestFMT,
				SenderSSRC: 0x01020304,
				MediaSSRC:  0x05060708,
				FCI:        testAV1RTCPLayerRefreshFCI(t, []AV1RTCPLayerRefreshRequestEntry{badSpatial}),
			})
			if _, _, err := EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
				cfg,
				badCompound,
				make([]RTCPPacket, 0, 1),
				nil,
				make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
			); !errors.Is(err, ErrRTCPInvalidLayerRefreshRequest) {
				t.Fatalf("bad spatial target compound error = %v, want ErrRTCPInvalidLayerRefreshRequest", err)
			}
		})
	}
}

func assertRTCPTransportFeedbackPackets(
	t *testing.T, got []RTCPTransportFeedbackPacket, want []RTCPTransportFeedbackPacket,
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

func testAV1RTCPEncoderConfig(mode EncoderScalabilityMode) EncoderConfig {
	return EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		Scalability:       mode,
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
		RateControl:       EncoderRateControlCBR,
	}
}

func testAV1RTCPFeedbackCompound(t *testing.T, packet RTCPFeedbackPacket) []byte {
	t.Helper()
	compound, err := AppendRTCPFeedbackPacket(
		make([]byte, 0, RTCPFeedbackPacketHeaderSize+len(packet.FCI)),
		packet,
	)
	if err != nil {
		t.Fatalf("AppendRTCPFeedbackPacket: %v", err)
	}
	return compound
}

func testAV1RTCPReceiverEstimatedMaximumBitratePacket(t *testing.T, bitrateBps uint64, ssrcs []uint32) RTCPFeedbackPacket {
	t.Helper()
	return RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBApplicationLayerFeedbackFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0,
		FCI:        testAV1RTCPReceiverEstimatedMaximumBitrateFCI(t, bitrateBps, ssrcs),
	}
}

func testAV1RTCPReceiverEstimatedMaximumBitrateFCI(t *testing.T, bitrateBps uint64, ssrcs []uint32) []byte {
	t.Helper()
	size, err := RTCPReceiverEstimatedMaximumBitrateFCISize(ssrcs)
	if err != nil {
		t.Fatalf("RTCPReceiverEstimatedMaximumBitrateFCISize: %v", err)
	}
	fci, err := AppendRTCPReceiverEstimatedMaximumBitrateFCI(make([]byte, 0, size), bitrateBps, ssrcs)
	if err != nil {
		t.Fatalf("AppendRTCPReceiverEstimatedMaximumBitrateFCI: %v", err)
	}
	return fci
}

func testAV1RTCPTransportFeedbackPacket(t *testing.T, feedback RTCPTransportFeedback) RTCPFeedbackPacket {
	t.Helper()
	return RTCPFeedbackPacket{
		PacketType: RTCPRTPFBPacketType,
		FMT:        RTCPRTPFBTransportFeedbackFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
		FCI:        testAV1RTCPTransportFeedbackFCI(t, feedback),
	}
}

func testAV1RTCPTransportFeedbackFCI(t *testing.T, feedback RTCPTransportFeedback) []byte {
	t.Helper()
	size, err := RTCPTransportFeedbackFCISize(feedback)
	if err != nil {
		t.Fatalf("RTCPTransportFeedbackFCISize: %v", err)
	}
	fci, err := AppendRTCPTransportFeedbackFCI(make([]byte, 0, size), feedback)
	if err != nil {
		t.Fatalf("AppendRTCPTransportFeedbackFCI: %v", err)
	}
	return fci
}

func testAV1RTCPLayerRefreshFCI(t *testing.T, entries []AV1RTCPLayerRefreshRequestEntry) []byte {
	t.Helper()
	size, err := AV1RTCPLayerRefreshRequestEntriesSize(entries)
	if err != nil {
		t.Fatalf("AV1RTCPLayerRefreshRequestEntriesSize: %v", err)
	}
	fci, err := AppendAV1RTCPLayerRefreshRequestEntries(make([]byte, 0, size), entries)
	if err != nil {
		t.Fatalf("AppendAV1RTCPLayerRefreshRequestEntries: %v", err)
	}
	return fci
}

func testAV1RTCPValidLayerRefreshEntry(t *testing.T, mode EncoderScalabilityMode) AV1RTCPLayerRefreshRequestEntry {
	t.Helper()
	spatialLayers, temporalLayers, _, ok := mode.Layers()
	if !ok {
		t.Fatalf("invalid scalability mode %s", mode)
	}
	target := AV1RTCPLayerRefreshLayerIndex{
		TemporalID: temporalLayers - 1,
		SpatialID:  spatialLayers - 1,
	}
	entry := AV1RTCPLayerRefreshRequestEntry{
		SSRC:           0x05060708,
		SequenceNumber: 17,
		PayloadType:    96,
		Target:         target,
	}
	if target != (AV1RTCPLayerRefreshLayerIndex{}) {
		entry.CurrentPresent = true
	}
	return entry
}
