package rtp

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

func appendPacketizerOBU(dst []byte, typ obu.Type, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{Type: typ, HasSizeField: true})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	var size [bitstream.MaxLEB128Bytes]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		panic(err)
	}
	dst = append(dst, size[:n]...)
	dst = append(dst, payload...)
	return dst
}

func appendPacketizerOBUExt(dst []byte, typ obu.Type, temporalID uint8, spatialID uint8, payload []byte) []byte {
	var header [2]byte
	n, err := obu.PutHeader(header[:], obu.Header{
		Type:         typ,
		Extension:    true,
		HasSizeField: true,
		TemporalID:   temporalID,
		SpatialID:    spatialID,
	})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	var size [bitstream.MaxLEB128Bytes]byte
	n, err = bitstream.PutLEB128(size[:], uint32(len(payload)))
	if err != nil {
		panic(err)
	}
	dst = append(dst, size[:n]...)
	dst = append(dst, payload...)
	return dst
}

func TestPacketizerFiltersAndStripsSizeFields(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeTemporalDelimiter, nil)
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb, 0xcc})
	frame = appendPacketizerOBU(frame, obu.TypePadding, []byte{0x00})

	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != 1 {
		t.Fatalf("NumPackets=%d", packetizer.NumPackets())
	}

	var payload [32]byte
	n, marker, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok || !marker {
		t.Fatalf("ok=%v marker=%v", ok, marker)
	}

	want := []byte{
		0x28, // W=2, N=1.
		0x02, byte(obu.TypeSequenceHeader) << 3, 0xaa,
		byte(obu.TypeFrameHeader) << 3, 0xbb, 0xcc,
	}
	if string(payload[:n]) != string(want) {
		t.Fatalf("payload=%x want=%x", payload[:n], want)
	}
}

func TestCountPacketizerOBUsMatchesParse(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeTemporalDelimiter, nil)
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBUExt(frame, obu.TypeFrameHeader, 1, 2, []byte{0xbb})
	frame = appendPacketizerOBU(frame, obu.TypePadding, []byte{0x00})

	count, err := CountPacketizerOBUs(frame)
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("count=%d want 2", count)
	}

	var obus [2]PacketizerOBU
	parsed, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if parsed != count {
		t.Fatalf("parsed=%d count=%d", parsed, count)
	}
	if obus[0].Type != obu.TypeSequenceHeader ||
		obus[0].Header.Type != obu.TypeSequenceHeader ||
		obus[0].Header.HasSizeField ||
		obus[0].HeaderSize != 1 ||
		obus[0].PayloadSize != 1 ||
		obus[0].Size != 2 {
		t.Fatalf("obu0=%+v", obus[0])
	}
	if obus[1].Type != obu.TypeFrameHeader ||
		obus[1].Header.Type != obu.TypeFrameHeader ||
		!obus[1].Header.Extension ||
		obus[1].Header.HasSizeField ||
		obus[1].Header.TemporalID != 1 ||
		obus[1].Header.SpatialID != 2 ||
		obus[1].HeaderSize != 2 ||
		obus[1].PayloadSize != 1 ||
		obus[1].Size != 3 ||
		!obus[1].hasExtension {
		t.Fatalf("obus=%+v", obus)
	}
}

func TestCountPacketizerOBUsRejectsMalformedOBU(t *testing.T) {
	payload := appendPacketizerOBU(nil, obu.TypeFrame, []byte{0xaa, 0xbb})
	payload[1] = 3
	if _, err := CountPacketizerOBUs(payload); !errors.Is(err, obu.ErrShortPayload) {
		t.Fatalf("CountPacketizerOBUs err=%v want %v", err, obu.ErrShortPayload)
	}
}

func TestPacketizerWritesLengthForMoreThanThreeOBUs(t *testing.T) {
	var frame []byte
	for i := range 4 {
		frame = appendPacketizerOBU(frame, obu.TypeMetadata, []byte{byte(i)})
	}

	var obus [4]PacketizerOBU
	var packets [2]PacketPlan
	var work [2]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	var payload [32]byte
	n, _, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("no packet")
	}
	want := []byte{
		0x00,
		0x02, byte(obu.TypeMetadata) << 3, 0x00,
		0x02, byte(obu.TypeMetadata) << 3, 0x01,
		0x02, byte(obu.TypeMetadata) << 3, 0x02,
		0x02, byte(obu.TypeMetadata) << 3, 0x03,
	}
	if string(payload[:n]) != string(want) {
		t.Fatalf("payload=%x want=%x", payload[:n], want)
	}
}

func TestPacketizerNextPacketSize(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	frame := appendPacketizerOBU(nil, obu.TypeFrame, payloadBytes)

	var obus [2]PacketizerOBU
	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 6}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	size, ok := packetizer.NextPacketSize()
	if !ok || size != 6 {
		t.Fatalf("first size=%d ok=%v want 6,true", size, ok)
	}
	plan, ok := packetizer.NextPacketPlan()
	if !ok || plan.PacketSize+1 != size || plan.FirstOBU != 0 || plan.NumOBUElements != 1 || plan.FirstOBUOffset != 0 || plan.LastOBUSize != 5 {
		t.Fatalf("first plan=%+v ok=%v size=%d", plan, ok, size)
	}
	var short [5]byte
	if _, _, _, err := packetizer.NextPacket(short[:]); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short NextPacket err=%v want %v", err, ErrShortBuffer)
	}
	if retryPlan, ok := packetizer.NextPacketPlan(); !ok || retryPlan != plan {
		t.Fatalf("retry plan=%+v ok=%v want %+v,true", retryPlan, ok, plan)
	}
	if retrySize, ok := packetizer.NextPacketSize(); !ok || retrySize != size || packetizer.NumPackets() != 3 {
		t.Fatalf("retry size=%d ok=%v remaining=%d want %d,true,3", retrySize, ok, packetizer.NumPackets(), size)
	}

	var payload [16]byte
	for i := range 3 {
		size, ok := packetizer.NextPacketSize()
		if !ok {
			t.Fatalf("missing size for packet %d", i)
		}
		n, _, ok, err := packetizer.NextPacket(payload[:size])
		if err != nil {
			t.Fatal(err)
		}
		if !ok || n != size {
			t.Fatalf("packet %d n=%d ok=%v want %d,true", i, n, ok, size)
		}
	}
	if size, ok := packetizer.NextPacketSize(); ok || size != 0 {
		t.Fatalf("done size=%d ok=%v want 0,false", size, ok)
	}
	if plan, ok := packetizer.NextPacketPlan(); ok || plan != (PacketPlan{}) {
		t.Fatalf("done plan=%+v ok=%v want zero,false", plan, ok)
	}
}

func TestPacketizerFragmentsOBU(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	frame := appendPacketizerOBU(nil, obu.TypeFrame, payloadBytes)

	var obus [2]PacketizerOBU
	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 6}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != 3 {
		t.Fatalf("NumPackets=%d", packetizer.NumPackets())
	}

	wantHeaders := []byte{0x50, 0xd0, 0x90}
	var out [16]byte
	var assembled [32]byte
	var spans [4]OBUSpan
	var dep Depacketizer
	used := 0
	for i := range 3 {
		n, marker, ok, err := packetizer.NextPacket(out[:])
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			t.Fatalf("missing packet %d", i)
		}
		if out[0] != wantHeaders[i] {
			t.Fatalf("packet %d header=%02x want %02x", i, out[0], wantHeaders[i])
		}
		if marker != (i == 2) {
			t.Fatalf("packet %d marker=%v", i, marker)
		}
		var count int
		used, count, _, err = dep.Push(assembled[:], used, spans[:], out[:n])
		if err != nil {
			t.Fatal(err)
		}
		if i < 2 && count != 0 {
			t.Fatalf("packet %d completed early", i)
		}
		if i == 2 && count != 1 {
			t.Fatalf("final count=%d", count)
		}
	}

	wantOBU := append([]byte{byte(obu.TypeFrame) << 3}, payloadBytes...)
	if string(assembled[:used]) != string(wantOBU) {
		t.Fatalf("assembled=%x want=%x", assembled[:used], wantOBU)
	}
}

func TestPacketizeOBUCountMatchesPacketizeOBUs(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	frame := appendPacketizerOBU(nil, obu.TypeFrame, payloadBytes)

	obuCount, err := CountPacketizerOBUs(frame)
	if err != nil {
		t.Fatal(err)
	}
	var obus [2]PacketizerOBU
	parsed, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if parsed != obuCount {
		t.Fatalf("parsed=%d count=%d", parsed, obuCount)
	}

	limits := PayloadSizeLimits{MaxPayloadLen: 6}
	planCount, err := PacketizeOBUCount(obus[:parsed], limits)
	if err != nil {
		t.Fatal(err)
	}
	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetized, err := PacketizeOBUs(obus[:parsed], limits, packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}
	if planCount != packetized || planCount != 3 {
		t.Fatalf("planCount=%d packetized=%d want 3", planCount, packetized)
	}
	for i := range planCount {
		if packets[i].PacketSize == 0 || packets[i].NumOBUElements == 0 {
			t.Fatalf("packet[%d]=%+v", i, packets[i])
		}
	}
}

func TestPacketizeOBUCountRejectsInvalidLimits(t *testing.T) {
	obus := []PacketizerOBU{{Type: obu.TypeFrame, Size: 3}}
	if _, err := PacketizeOBUCount(obus, PayloadSizeLimits{MaxPayloadLen: 2}); !errors.Is(err, ErrInvalidPayloadLimits) {
		t.Fatalf("PacketizeOBUCount err=%v want %v", err, ErrInvalidPayloadLimits)
	}
}

func TestPacketizerScratchLenTwoPass(t *testing.T) {
	payloadBytes := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeTemporalDelimiter, nil)
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrame, payloadBytes)
	limits := PayloadSizeLimits{MaxPayloadLen: 6}

	size, err := PacketizerScratchLen(frame, limits, nil)
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("first pass size=%+v want OBUs=2 only", size)
	}

	var obus [4]PacketizerOBU
	size, err = PacketizerScratchLen(frame, limits, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets <= 1 || size.Work != size.Packets {
		t.Fatalf("second pass size=%+v", size)
	}

	var packets [8]PacketPlan
	var work [8]PacketPlan
	packetizer, err := NewPacketizer(frame, limits, true, true, obus[:size.OBUs], packets[:size.Packets], work[:size.Work])
	if err != nil {
		t.Fatal(err)
	}
	if packetizer.NumPackets() != size.Packets {
		t.Fatalf("NumPackets=%d scratch packets=%d", packetizer.NumPackets(), size.Packets)
	}
}

func TestPacketizerScratchLenWaitsForOBUScratchBeforePacketSizing(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	limits := PayloadSizeLimits{MaxPayloadLen: 2}

	var short [1]PacketizerOBU
	size, err := PacketizerScratchLen(frame, limits, short[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("short scratch size=%+v want OBUs=2 only", size)
	}

	var obus [2]PacketizerOBU
	size, err = PacketizerScratchLen(frame, limits, obus[:])
	if !errors.Is(err, ErrInvalidPayloadLimits) {
		t.Fatalf("PacketizerScratchLen err=%v want %v", err, ErrInvalidPayloadLimits)
	}
	if size.OBUs != 2 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("invalid-limit size=%+v want OBUs=2 only", size)
	}
}

func TestPacketizerScratchLenEmptyPayloadIgnoresLimits(t *testing.T) {
	size, err := PacketizerScratchLen(nil, PayloadSizeLimits{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if size != (PacketizerScratchSize{}) {
		t.Fatalf("size=%+v want zero", size)
	}
}

func TestPacketizerPreservesExtensionAndClearsSize(t *testing.T) {
	frame := appendPacketizerOBUExt(nil, obu.TypeTileGroup, 1, 2, []byte{0x99})
	var obus [1]PacketizerOBU
	var packets [1]PacketPlan
	var work [1]PacketPlan
	packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, obus[:], packets[:], work[:])
	if err != nil {
		t.Fatal(err)
	}

	var payload [16]byte
	n, _, ok, err := packetizer.NextPacket(payload[:])
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("no packet")
	}
	want := []byte{
		0x10,
		byte(obu.TypeTileGroup)<<3 | 0x04,
		(1 << 5) | (2 << 3),
		0x99,
	}
	if string(payload[:n]) != string(want) {
		t.Fatalf("payload=%x want=%x", payload[:n], want)
	}
}

func TestPacketizerScratchTooSmall(t *testing.T) {
	frame := appendPacketizerOBU(nil, obu.TypeFrameHeader, []byte{0xaa})
	_, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, false, true, nil, nil, nil)
	if !errors.Is(err, ErrOBUBufferTooSmall) {
		t.Fatalf("NewPacketizer err=%v want %v", err, ErrOBUBufferTooSmall)
	}

	var obus [1]PacketizerOBU
	count, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PacketizeOBUs(obus[:count], PayloadSizeLimits{MaxPayloadLen: 1200}, nil, nil); !errors.Is(err, ErrPacketPlanTooSmall) {
		t.Fatalf("PacketizeOBUs err=%v want %v", err, ErrPacketPlanTooSmall)
	}
}

func TestPacketizerAllocs(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	var payload [32]byte

	allocs := testing.AllocsPerRun(1000, func() {
		packetizer, err := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
		if err != nil {
			t.Fatal(err)
		}
		_, _, _, err = packetizer.NextPacket(payload[:])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Packetizer allocated: %f", allocs)
	}
}

func TestPacketizerSizingAllocs(t *testing.T) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	var obus [4]PacketizerOBU
	count, err := ParsePacketizerOBUs(frame, obus[:])
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		obuCount, err := CountPacketizerOBUs(frame)
		if err != nil {
			t.Fatal(err)
		}
		if obuCount != count {
			t.Fatalf("obuCount=%d want %d", obuCount, count)
		}
		packetCount, err := PacketizeOBUCount(obus[:count], PayloadSizeLimits{MaxPayloadLen: 1200})
		if err != nil {
			t.Fatal(err)
		}
		if packetCount != 1 {
			t.Fatalf("packetCount=%d want 1", packetCount)
		}
		size, err := PacketizerScratchLen(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if size.OBUs != count || size.Packets != 0 || size.Work != 0 {
			t.Fatalf("first pass size=%+v", size)
		}
		size, err = PacketizerScratchLen(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, obus[:])
		if err != nil {
			t.Fatal(err)
		}
		if size.OBUs != count || size.Packets != packetCount || size.Work != 0 {
			t.Fatalf("second pass size=%+v packetCount=%d", size, packetCount)
		}
	})
	if allocs != 0 {
		t.Fatalf("packetizer sizing allocated: %f", allocs)
	}
}

func BenchmarkPacketizer(b *testing.B) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	var payload [32]byte

	b.ReportAllocs()
	for b.Loop() {
		packetizer, _ := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
		_, _, _, _ = packetizer.NextPacket(payload[:])
	}
}
