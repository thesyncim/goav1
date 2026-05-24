package rtp

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/obu"
)

func FuzzPayloadIterator(f *testing.F) {
	for _, seed := range [][]byte{
		{0x00, 0x01, 0xaa},
		{0x10, 0xaa},
		{0x20, 0x01, 0xaa, 0xbb},
		{0x88, 0xaa},
		{0x01},
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, payload []byte) {
		it, err := NewIterator(payload)
		if err != nil {
			return
		}
		lastIndex := -1
		for {
			elem, ok, err := it.Next()
			if err != nil {
				return
			}
			if !ok {
				return
			}
			if len(elem.Data) == 0 && !elem.fragmented() {
				t.Fatal("empty element returned without error")
			}
			if elem.Index < lastIndex {
				t.Fatalf("index regressed from %d to %d", lastIndex, elem.Index)
			}
			lastIndex = elem.Index
		}
	})
}

func FuzzPacketizer(f *testing.F) {
	f.Add(appendPacketizerOBU(nil, obu.TypeFrameHeader, []byte{0x80}), uint8(6))
	f.Add(appendPacketizerOBU(nil, obu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6}), uint8(8))

	f.Fuzz(func(t *testing.T, payload []byte, maxPayloadLen uint8) {
		limit := int(maxPayloadLen%64) + 3
		var obus [16]PacketizerOBU
		var packets [128]PacketPlan
		var work [128]PacketPlan

		obuCount, err := CountPacketizerOBUs(payload)
		if err != nil {
			return
		}
		if obuCount > len(obus) {
			return
		}
		parsed, err := ParsePacketizerOBUs(payload, obus[:])
		if err != nil {
			t.Fatalf("ParsePacketizerOBUs after count: %v", err)
		}
		if parsed != obuCount {
			t.Fatalf("parsed=%d count=%d", parsed, obuCount)
		}
		packetCount, err := PacketizeOBUCount(obus[:parsed], PayloadSizeLimits{MaxPayloadLen: limit})
		if err != nil {
			t.Fatalf("PacketizeOBUCount: %v", err)
		}
		if packetCount > len(packets) {
			return
		}

		packetizer, err := NewPacketizer(payload, PayloadSizeLimits{MaxPayloadLen: limit}, true, true, obus[:], packets[:], work[:])
		if err != nil {
			t.Fatalf("NewPacketizer after sizing: %v", err)
		}
		if packetizer.NumPackets() != packetCount {
			t.Fatalf("NumPackets=%d count=%d", packetizer.NumPackets(), packetCount)
		}

		var out [128]byte
		for {
			n, _, ok, err := packetizer.NextPacket(out[:])
			if err != nil {
				t.Fatalf("NextPacket: %v", err)
			}
			if !ok {
				return
			}
			if n > limit {
				t.Fatalf("packet size=%d limit=%d", n, limit)
			}
			it, err := NewIterator(out[:n])
			if err != nil {
				t.Fatalf("NewIterator: %v", err)
			}
			for {
				_, ok, err := it.Next()
				if err != nil {
					t.Fatalf("Iterator.Next: %v", err)
				}
				if !ok {
					break
				}
			}
		}
	})
}

func FuzzAssembleFrame(f *testing.F) {
	f.Add([]byte{0x10, byte(obu.TypeFrame) << 3, 0xaa})
	f.Add([]byte{0x20, 0x02, byte(obu.TypeSequenceHeader) << 3, 0xaa, byte(obu.TypeFrameHeader) << 3, 0xbb})
	f.Add([]byte{0x50, byte(obu.TypeFrame) << 3, 0xaa})

	f.Fuzz(func(t *testing.T, payload []byte) {
		var out [512]byte
		var obus [16]FrameOBU
		size, count, err := AssembleFrameSize([][]byte{payload})
		if err != nil {
			return
		}
		if size > len(out) || count > len(obus) {
			return
		}
		wrote, assembledCount, err := AssembleFrame(out[:], [][]byte{payload}, obus[:])
		if err != nil {
			t.Fatalf("AssembleFrame after sizing: %v", err)
		}
		if wrote != size || assembledCount != count {
			t.Fatalf("assembled size=%d count=%d want %d,%d", wrote, assembledCount, size, count)
		}

		it := obu.NewLowOverheadIterator(out[:wrote])
		for {
			_, ok, err := it.Next()
			if err != nil {
				t.Fatalf("assembled low-overhead parse failed: %v", err)
			}
			if !ok {
				return
			}
		}
	})
}
