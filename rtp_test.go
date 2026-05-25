package goav1

import "testing"

func appendPublicLowOverheadOBU(dst []byte, typ OBUType, payload []byte) []byte {
	if len(payload) > 0x7f {
		panic("test payload too large for one-byte LEB128")
	}
	var header [2]byte
	n, err := PutOBUHeader(header[:], OBUHeader{Type: typ, HasSizeField: true})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	dst = append(dst, byte(len(payload)))
	dst = append(dst, payload...)
	return dst
}

func TestPublicRTPPacketizerScratchAndAssembleSizing(t *testing.T) {
	var frame []byte
	frame = appendPublicLowOverheadOBU(frame, OBUSequenceHeader, []byte{0xaa})
	frame = appendPublicLowOverheadOBU(frame, OBUFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}

	size, err := RTPPacketizerScratchLen(frame, limits, nil)
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("first pass size=%+v want OBUs=2 only", size)
	}

	var packetizerOBUs [4]RTPPacketizerOBU
	size, err = RTPPacketizerScratchLen(frame, limits, packetizerOBUs[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 2 || size.Packets <= 1 || size.Work != size.Packets {
		t.Fatalf("second pass size=%+v", size)
	}

	var packets [8]RTPPacketPlan
	var work [8]RTPPacketPlan
	packetizer, err := NewRTPPacketizer(frame, limits, true, true, packetizerOBUs[:size.OBUs], packets[:size.Packets], work[:size.Work])
	if err != nil {
		t.Fatal(err)
	}

	var packetBytes [8][32]byte
	var payloads [8][]byte
	payloadCount := 0
	for {
		if payloadCount >= len(payloads) {
			t.Fatal("too many packets")
		}
		size, ok := packetizer.NextPacketSize()
		if !ok {
			break
		}
		n, marker, ok, err := packetizer.NextPacket(packetBytes[payloadCount][:size])
		if err != nil {
			t.Fatal(err)
		}
		if !ok || n != size {
			t.Fatalf("packet %d n=%d ok=%v want %d,true", payloadCount, n, ok, size)
		}
		if marker != (packetizer.NumPackets() == 0) {
			t.Fatalf("packet %d marker=%v remaining=%d", payloadCount, marker, packetizer.NumPackets())
		}
		payloads[payloadCount] = packetBytes[payloadCount][:n]
		payloadCount++
	}
	if payloadCount != size.Packets {
		t.Fatalf("payloadCount=%d scratch packets=%d", payloadCount, size.Packets)
	}
	if next, ok := packetizer.NextPacketSize(); ok || next != 0 {
		t.Fatalf("done packet size=%d ok=%v want 0,false", next, ok)
	}

	assembledLen, obuCount, err := AssembleRTPFrameSize(payloads[:payloadCount])
	if err != nil {
		t.Fatal(err)
	}
	var assembled [64]byte
	var assembledOBUs [4]RTPFrameOBU
	wrote, assembledCount, err := AssembleRTPFrame(assembled[:assembledLen], payloads[:payloadCount], assembledOBUs[:obuCount])
	if err != nil {
		t.Fatal(err)
	}
	if wrote != assembledLen || assembledCount != obuCount {
		t.Fatalf("assembled=%d,%d want %d,%d", wrote, assembledCount, assembledLen, obuCount)
	}
	if string(assembled[:wrote]) != string(frame) {
		t.Fatalf("assembled=%x want %x", assembled[:wrote], frame)
	}
}

func TestPublicRTPDepacketizerPushSize(t *testing.T) {
	elements := []RTPElement{
		{Data: []byte{byte(OBUSequenceHeader) << 3, 0xaa}},
		{Data: []byte{byte(OBUFrameHeader) << 3, 0xbb}},
	}
	var payload [32]byte
	n, err := PutRTPPayload(payload[:], RTPAggregationHeader{
		ElementCount:                2,
		StartsNewCodedVideoSequence: true,
	}, elements)
	if err != nil {
		t.Fatal(err)
	}

	var dep RTPDepacketizer
	plannedUsed, plannedSpans, header, err := dep.PushSize(0, payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if plannedUsed != 4 || plannedSpans != 2 || !header.StartsNewCodedVideoSequence {
		t.Fatalf("planned used=%d spans=%d header=%+v", plannedUsed, plannedSpans, header)
	}

	var out [16]byte
	var spans [2]RTPObuSpan
	used, count, pushedHeader, err := dep.Push(out[:plannedUsed], 0, spans[:plannedSpans], payload[:n])
	if err != nil {
		t.Fatal(err)
	}
	if used != plannedUsed || count != plannedSpans || pushedHeader != header {
		t.Fatalf("push used=%d count=%d header=%+v planned %d,%d,%+v", used, count, pushedHeader, plannedUsed, plannedSpans, header)
	}
	if spans[0] != (RTPObuSpan{Offset: 0, Length: 2, NewSequence: true}) {
		t.Fatalf("span0=%+v", spans[0])
	}
	if spans[1] != (RTPObuSpan{Offset: 2, Length: 2}) {
		t.Fatalf("span1=%+v", spans[1])
	}
}

func TestPublicRTPSizingAllocs(t *testing.T) {
	var frame []byte
	frame = appendPublicLowOverheadOBU(frame, OBUSequenceHeader, []byte{0xaa})
	frame = appendPublicLowOverheadOBU(frame, OBUFrameHeader, []byte{0xbb})

	var packetizerOBUs [4]RTPPacketizerOBU
	var packets [4]RTPPacketPlan
	var work [4]RTPPacketPlan
	var packetBytes [4][32]byte
	var payloads [4][]byte
	var frameOBUs [4]RTPFrameOBU
	var assembled [64]byte
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 1200}

	allocs := testing.AllocsPerRun(1000, func() {
		size, err := RTPPacketizerScratchLen(frame, limits, nil)
		if err != nil {
			t.Fatal(err)
		}
		if size.OBUs != 2 {
			t.Fatalf("first pass size=%+v", size)
		}
		size, err = RTPPacketizerScratchLen(frame, limits, packetizerOBUs[:])
		if err != nil {
			t.Fatal(err)
		}
		packetizer, err := NewRTPPacketizer(frame, limits, true, true, packetizerOBUs[:size.OBUs], packets[:size.Packets], work[:size.Work])
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for {
			next, ok := packetizer.NextPacketSize()
			if !ok {
				break
			}
			n, _, ok, err := packetizer.NextPacket(packetBytes[count][:next])
			if err != nil {
				t.Fatal(err)
			}
			if !ok || n != next {
				t.Fatalf("packet n=%d ok=%v want %d,true", n, ok, next)
			}
			payloads[count] = packetBytes[count][:n]
			count++
		}
		assembledLen, obuCount, err := AssembleRTPFrameSize(payloads[:count])
		if err != nil {
			t.Fatal(err)
		}
		wrote, assembledCount, err := AssembleRTPFrame(assembled[:assembledLen], payloads[:count], frameOBUs[:obuCount])
		if err != nil {
			t.Fatal(err)
		}
		if wrote != len(frame) || assembledCount != 2 {
			t.Fatalf("assembled=%d,%d", wrote, assembledCount)
		}
	})
	if allocs != 0 {
		t.Fatalf("public RTP sizing allocated: %f", allocs)
	}
}
