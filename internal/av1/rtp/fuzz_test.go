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
		var dep Depacketizer
		plannedUsed, plannedSpans, plannedHeader, err := dep.PushSize(0, payload)
		if err == nil && plannedUsed <= 512 && plannedSpans <= 16 {
			var out [512]byte
			var spans [16]OBUSpan
			used, count, header, err := dep.Push(out[:plannedUsed], 0, spans[:plannedSpans], payload)
			if err != nil {
				t.Fatalf("Push after PushSize: %v", err)
			}
			if used != plannedUsed || count != plannedSpans || header != plannedHeader {
				t.Fatalf("Push used=%d count=%d header=%+v planned %d,%d,%+v", used, count, header, plannedUsed, plannedSpans, plannedHeader)
			}
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
		scratch, err := PacketizerScratchLen(payload, PayloadSizeLimits{MaxPayloadLen: limit}, nil)
		if err != nil {
			t.Fatalf("PacketizerScratchLen first pass after count: %v", err)
		}
		if scratch.OBUs != obuCount || scratch.Packets != 0 || scratch.Work != 0 {
			t.Fatalf("first pass scratch=%+v count=%d", scratch, obuCount)
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
		for i, info := range obus[:parsed] {
			if info.Header.Type != info.Type ||
				info.Header.HasSizeField ||
				info.HeaderSize != info.Header.HeaderLen() ||
				info.Size != info.HeaderSize+info.PayloadSize {
				t.Fatalf("obu[%d]=%+v", i, info)
			}
		}
		packetCount, err := PacketizeOBUCount(obus[:parsed], PayloadSizeLimits{MaxPayloadLen: limit})
		if err != nil {
			t.Fatalf("PacketizeOBUCount: %v", err)
		}
		scratch, err = PacketizerScratchLen(payload, PayloadSizeLimits{MaxPayloadLen: limit}, obus[:])
		if err != nil {
			t.Fatalf("PacketizerScratchLen second pass: %v", err)
		}
		if scratch.OBUs != obuCount || scratch.Packets != packetCount {
			t.Fatalf("second pass scratch=%+v count=%d packets=%d", scratch, obuCount, packetCount)
		}
		if packetCount <= 1 {
			if scratch.Work != 0 {
				t.Fatalf("scratch work=%d want 0 for packetCount=%d", scratch.Work, packetCount)
			}
		} else if scratch.Work != packetCount {
			t.Fatalf("scratch work=%d packetCount=%d", scratch.Work, packetCount)
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
			nextSize, sizeOK := packetizer.NextPacketSize()
			nextPlan, planOK := packetizer.NextPacketPlan()
			if sizeOK != planOK {
				t.Fatalf("NextPacketSize ok=%v NextPacketPlan ok=%v", sizeOK, planOK)
			}
			if sizeOK && nextSize != nextPlan.PacketSize+1 {
				t.Fatalf("next size=%d plan=%+v", nextSize, nextPlan)
			}
			n, _, ok, err := packetizer.NextPacket(out[:])
			if err != nil {
				t.Fatalf("NextPacket: %v", err)
			}
			if !ok {
				if sizeOK {
					t.Fatalf("NextPacket done after size=%d plan=%+v", nextSize, nextPlan)
				}
				return
			}
			if !sizeOK || n != nextSize {
				t.Fatalf("packet n=%d size=%d ok=%v", n, nextSize, sizeOK)
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
		offset := 0
		index := 0
		for {
			unit, ok, err := it.Next()
			if err != nil {
				t.Fatalf("assembled low-overhead parse failed: %v", err)
			}
			if !ok {
				if index != assembledCount || offset != wrote {
					t.Fatalf("parsed index=%d offset=%d want count=%d wrote=%d", index, offset, assembledCount, wrote)
				}
				return
			}
			if index >= assembledCount {
				t.Fatalf("parsed more OBUs than metadata count=%d", assembledCount)
			}
			info := obus[index]
			if info.Header != unit.Header ||
				info.Offset != offset ||
				info.Length != len(unit.Raw) ||
				info.PrefixSize != int(unit.HeaderLen)+int(unit.SizeLen) ||
				info.PayloadSize != len(unit.Payload) ||
				info.PayloadOffset() != offset+int(unit.HeaderLen)+int(unit.SizeLen) ||
				info.PayloadEnd() != offset+len(unit.Raw) {
				t.Fatalf("metadata[%d]=%+v unit=%+v offset=%d", index, info, unit, offset)
			}
			offset += len(unit.Raw)
			index++
		}
	})
}
