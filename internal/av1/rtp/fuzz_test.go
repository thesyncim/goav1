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
			if len(elem.Data) == 0 {
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
		packetizer, err := NewPacketizer(payload, PayloadSizeLimits{MaxPayloadLen: limit}, true, true, obus[:], packets[:], work[:])
		if err != nil {
			return
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
