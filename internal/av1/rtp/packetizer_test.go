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

func TestPacketizerWritesLengthForMoreThanThreeOBUs(t *testing.T) {
	var frame []byte
	for i := 0; i < 4; i++ {
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
	for i := 0; i < 3; i++ {
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

func BenchmarkPacketizer(b *testing.B) {
	var frame []byte
	frame = appendPacketizerOBU(frame, obu.TypeSequenceHeader, []byte{0xaa})
	frame = appendPacketizerOBU(frame, obu.TypeFrameHeader, []byte{0xbb})
	var obus [4]PacketizerOBU
	var packets [4]PacketPlan
	var work [4]PacketPlan
	var payload [32]byte

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		packetizer, _ := NewPacketizer(frame, PayloadSizeLimits{MaxPayloadLen: 1200}, true, true, obus[:], packets[:], work[:])
		_, _, _, _ = packetizer.NextPacket(payload[:])
	}
}
