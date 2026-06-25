package goav1_test

import (
	"bytes"
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicRTCPGenericNACKPairSequenceNumbers(t *testing.T) {
	pairs := []av1.RTCPGenericNACKPair{
		{PacketID: 100, LostPacketBitmask: 0x0005},
		{PacketID: 0xffff, LostPacketBitmask: 0x0003},
	}
	got, err := av1.AppendRTCPGenericNACKPairSequenceNumbers(make([]uint16, 0, 6), pairs)
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairSequenceNumbers: %v", err)
	}
	want := []uint16{100, 101, 103, 0xffff, 0, 1}
	if !equalUint16s(got, want) {
		t.Fatalf("NACK sequence numbers=%v want %v", got, want)
	}
	prefix := make([]uint16, 1, 4)
	prefix[0] = 7
	prefixed, err := av1.AppendRTCPGenericNACKPairSequenceNumbers(prefix[:1:4], pairs[:1])
	if err != nil {
		t.Fatalf("AppendRTCPGenericNACKPairSequenceNumbers prefixed: %v", err)
	}
	if !equalUint16s(prefixed, []uint16{7, 100, 101, 103}) {
		t.Fatalf("prefixed NACK sequence numbers=%v", prefixed)
	}
	if _, err := av1.AppendRTCPGenericNACKPairSequenceNumbers(make([]uint16, 0, 2), pairs[:1]); !errors.Is(err, av1.ErrRTCPShortBuffer) {
		t.Fatalf("short NACK sequence dst err=%v want %v", err, av1.ErrRTCPShortBuffer)
	}
	if out, err := av1.AppendRTCPGenericNACKPairSequenceNumbers(nil, nil); err != nil || len(out) != 0 {
		t.Fatalf("empty NACK sequence out=%v err=%v", out, err)
	}
}

func TestPublicRTPRetransmissionCacheStoresAndAppendsNACKPackets(t *testing.T) {
	if _, err := av1.BindRTPRetransmissionCache(nil); !errors.Is(err, av1.ErrRTPInvalidRetransmissionCache) {
		t.Fatalf("BindRTPRetransmissionCache nil err=%v want %v", err, av1.ErrRTPInvalidRetransmissionCache)
	}
	cache := bindTestRetransmissionCache(t, 8, 64)
	packets := [][]byte{
		testRetransmissionRTPPacket(t, 0xfffe, []byte("before-wrap")),
		testRetransmissionRTPPacket(t, 0xffff, []byte("wrap-a")),
		testRetransmissionRTPPacket(t, 0, []byte("wrap-b")),
		testRetransmissionRTPPacket(t, 1, []byte("wrap-c")),
	}
	for _, packet := range packets {
		if err := cache.Store(packet); err != nil {
			t.Fatalf("Store seq %d: %v", packetSequence(t, packet), err)
		}
	}
	packets[1][len(packets[1])-1] ^= 0xff
	if !cache.Contains(0xffff) || cache.Contains(2) {
		t.Fatalf("cache contains wrap=%v missing=%v", cache.Contains(0xffff), cache.Contains(2))
	}
	pairs := []av1.RTCPGenericNACKPair{{PacketID: 0xffff, LostPacketBitmask: 0x0003}}
	spans := make([]av1.RTPRetransmissionPacketSpan, 3)
	out, count, err := cache.AppendPacketsForRTCPGenericNACKPairs(make([]byte, 0, 256), spans, pairs)
	if err != nil {
		t.Fatalf("AppendPacketsForRTCPGenericNACKPairs: %v", err)
	}
	if count != 3 {
		t.Fatalf("retransmission count=%d want 3", count)
	}
	wantSeqs := []uint16{0xffff, 0, 1}
	for i := 0; i < count; i++ {
		span := spans[i]
		if span.SequenceNumber != wantSeqs[i] {
			t.Fatalf("span %d seq=%d want %d", i, span.SequenceNumber, wantSeqs[i])
		}
		got := out[span.Offset : span.Offset+span.Length]
		if gotSeq := packetSequence(t, got); gotSeq != wantSeqs[i] {
			t.Fatalf("span %d packet seq=%d want %d", i, gotSeq, wantSeqs[i])
		}
	}
	if bytes.Equal(out[spans[0].Offset:spans[0].Offset+spans[0].Length], packets[1]) {
		t.Fatal("cached packet aliases caller-owned input after Store")
	}

	missingOnly := []av1.RTCPGenericNACKPair{{PacketID: 22}}
	out, count, err = cache.AppendPacketsForRTCPGenericNACKPairs(out[:0], spans, missingOnly)
	if err != nil || count != 0 || len(out) != 0 {
		t.Fatalf("missing-only NACK out=%d count=%d err=%v", len(out), count, err)
	}
}

func TestPublicRTPRetransmissionCacheEvictionResetAndShortBuffers(t *testing.T) {
	cache := bindTestRetransmissionCache(t, 2, 40)
	first := testRetransmissionRTPPacket(t, 1, []byte("one"))
	replacement := testRetransmissionRTPPacket(t, 3, []byte("three"))
	if err := cache.Store(first); err != nil {
		t.Fatalf("Store first: %v", err)
	}
	if err := cache.Store(replacement); err != nil {
		t.Fatalf("Store replacement: %v", err)
	}
	if cache.Contains(1) || !cache.Contains(3) {
		t.Fatalf("eviction contains seq1=%v seq3=%v", cache.Contains(1), cache.Contains(3))
	}
	out, found, err := cache.AppendPacket(make([]byte, 0, len(replacement)), 3)
	if err != nil || !found || !bytes.Equal(out, replacement) {
		t.Fatalf("AppendPacket out=% x found=%v err=%v", out, found, err)
	}
	if _, found, err := cache.AppendPacket(out[:0], 1); err != nil || found {
		t.Fatalf("AppendPacket evicted found=%v err=%v", found, err)
	}
	if _, found, err := cache.AppendPacket(make([]byte, 0, len(replacement)-1), 3); !found || !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short AppendPacket found=%v err=%v want %v", found, err, av1.ErrRTPShortBuffer)
	}
	spans := make([]av1.RTPRetransmissionPacketSpan, 0)
	if _, _, err := cache.AppendPacketsForRTCPGenericNACKPairs(make([]byte, 0, len(replacement)), spans, []av1.RTCPGenericNACKPair{{PacketID: 3}}); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short retransmission spans err=%v want %v", err, av1.ErrRTPShortBuffer)
	}
	cache.Reset()
	if cache.Contains(3) {
		t.Fatal("Reset left cached sequence present")
	}

	shortSlots := []av1.RTPRetransmissionCacheSlot{{Packet: make([]byte, 0, 8)}}
	shortCache, err := av1.BindRTPRetransmissionCache(shortSlots)
	if err != nil {
		t.Fatalf("Bind short slot cache: %v", err)
	}
	if err := shortCache.Store(first); !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short Store err=%v want %v", err, av1.ErrRTPShortBuffer)
	}
	var zero av1.RTPRetransmissionCache
	if err := zero.Store(first); !errors.Is(err, av1.ErrRTPInvalidRetransmissionCache) {
		t.Fatalf("zero Store err=%v want %v", err, av1.ErrRTPInvalidRetransmissionCache)
	}
}

func TestPublicRTPRetransmissionCacheAllocs(t *testing.T) {
	cache := bindTestRetransmissionCache(t, 8, 64)
	for seq := uint16(10); seq < 14; seq++ {
		if err := cache.Store(testRetransmissionRTPPacket(t, seq, []byte{byte(seq)})); err != nil {
			t.Fatalf("Store seq %d: %v", seq, err)
		}
	}
	pairs := []av1.RTCPGenericNACKPair{{PacketID: 10, LostPacketBitmask: 0x0007}}
	dst := make([]byte, 0, 256)
	spans := make([]av1.RTPRetransmissionPacketSpan, 4)
	allocs := testing.AllocsPerRun(1000, func() {
		var err error
		dst, _, err = cache.AppendPacketsForRTCPGenericNACKPairs(dst[:0], spans, pairs)
		if err != nil {
			t.Fatalf("AppendPacketsForRTCPGenericNACKPairs: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("RTPRetransmissionCache append allocs/run=%f want 0", allocs)
	}
}

func bindTestRetransmissionCache(t *testing.T, slots int, packetCap int) av1.RTPRetransmissionCache {
	t.Helper()
	storage := make([]av1.RTPRetransmissionCacheSlot, slots)
	for i := range storage {
		storage[i].Packet = make([]byte, 0, packetCap)
	}
	cache, err := av1.BindRTPRetransmissionCache(storage)
	if err != nil {
		t.Fatalf("BindRTPRetransmissionCache: %v", err)
	}
	return cache
}

func testRetransmissionRTPPacket(t *testing.T, sequence uint16, payload []byte) []byte {
	t.Helper()
	header := av1.RTPHeader{
		Marker:         true,
		PayloadType:    96,
		SequenceNumber: sequence,
		Timestamp:      90000 + uint32(sequence),
		SSRC:           0xdec0de,
	}
	size, err := av1.RTPPacketSize(header, payload, 0)
	if err != nil {
		t.Fatalf("RTPPacketSize: %v", err)
	}
	packet := make([]byte, size)
	n, err := av1.PutRTPPacket(packet, header, payload, 0)
	if err != nil {
		t.Fatalf("PutRTPPacket: %v", err)
	}
	return packet[:n]
}

func packetSequence(t *testing.T, packet []byte) uint16 {
	t.Helper()
	parsed, err := av1.ParseRTPPacket(packet)
	if err != nil {
		t.Fatalf("ParseRTPPacket: %v", err)
	}
	return parsed.Header.SequenceNumber
}

func equalUint16s(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
