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
	if packetizerOBUs[0].Header.Type != OBUSequenceHeader ||
		packetizerOBUs[0].Header.HasSizeField ||
		packetizerOBUs[0].HeaderSize != 1 ||
		packetizerOBUs[0].PayloadSize != 1 ||
		packetizerOBUs[0].Size != 2 {
		t.Fatalf("packetizer OBU0=%+v", packetizerOBUs[0])
	}
	if packetizerOBUs[1].Header.Type != OBUFrame ||
		packetizerOBUs[1].Header.HasSizeField ||
		packetizerOBUs[1].HeaderSize != 1 ||
		packetizerOBUs[1].PayloadSize != 10 ||
		packetizerOBUs[1].Size != 11 {
		t.Fatalf("packetizer OBU1=%+v", packetizerOBUs[1])
	}

	var packetBytes [8][32]byte
	var descriptors [8][16]byte
	var descriptorLens [8]int
	var payloads [8][]byte
	payloadCount := 0
	control, err := EncoderWebRTCTemporalUnitControlForFrames(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	}, []EncoderFrameEncodeSettings{{
		Type:            EncoderFrameTypeKey,
		Resolution:      EncoderResolution{Width: 640, Height: 360},
		UpdateBuffer:    0,
		UpdateBufferSet: true,
		Output:          true,
	}}, EncoderReferenceBufferState{}, EncoderFrameIDBufferState{}, 77)
	if err != nil {
		t.Fatal(err)
	}
	frameControl := control.Frames[0]
	for {
		if payloadCount >= len(payloads) {
			t.Fatal("too many packets")
		}
		size, ok := packetizer.NextPacketSize()
		if !ok {
			break
		}
		plan, ok := packetizer.NextPacketPlan()
		if !ok || plan.PacketSize+1 != size {
			t.Fatalf("packet %d plan=%+v ok=%v size=%d", payloadCount, plan, ok, size)
		}
		payloadSize, descriptorSize, ok, err := EncoderWebRTCFrameControlRTPPacketSize(&packetizer, frameControl, control.DependencyStructure)
		if err != nil {
			t.Fatal(err)
		}
		if !ok || payloadSize != size {
			t.Fatalf("packet %d payloadSize=%d size=%d ok=%v", payloadCount, payloadSize, size, ok)
		}
		payload, descriptor, marker, ok, err := AppendEncoderWebRTCFrameControlRTPPacket(packetBytes[payloadCount][:0], descriptors[payloadCount][:0], &packetizer, frameControl, control.DependencyStructure)
		if err != nil {
			t.Fatal(err)
		}
		if !ok || len(payload) != size || len(descriptor) != descriptorSize {
			t.Fatalf("packet %d payload=%d/%d descriptor=%d/%d ok=%v", payloadCount, len(payload), size, len(descriptor), descriptorSize, ok)
		}
		if marker != (packetizer.NumPackets() == 0) {
			t.Fatalf("packet %d marker=%v remaining=%d", payloadCount, marker, packetizer.NumPackets())
		}
		descriptorLens[payloadCount] = len(descriptor)
		payloads[payloadCount] = payload
		payloadCount++
	}
	if payloadCount != size.Packets {
		t.Fatalf("payloadCount=%d scratch packets=%d", payloadCount, size.Packets)
	}
	if descriptors[0][0]&0xc0 != 0x80 || descriptors[payloadCount-1][0]&0xc0 != 0x40 {
		t.Fatalf("descriptor flags first=%02x last=%02x", descriptors[0][0], descriptors[payloadCount-1][0])
	}
	if descriptorLens[0] == RTPDependencyDescriptorMandatorySize {
		t.Fatal("first packet descriptor did not attach dependency structure")
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
	if assembledOBUs[0].Header.Type != OBUSequenceHeader ||
		!assembledOBUs[0].Header.HasSizeField ||
		assembledOBUs[0].Offset != 0 ||
		assembledOBUs[0].PrefixSize != 2 ||
		assembledOBUs[0].PayloadSize != 1 ||
		assembledOBUs[0].PayloadOffset() != 2 ||
		assembledOBUs[0].PayloadEnd() != 3 {
		t.Fatalf("assembled OBU0=%+v", assembledOBUs[0])
	}
	if assembledOBUs[1].Header.Type != OBUFrame ||
		!assembledOBUs[1].Header.HasSizeField ||
		assembledOBUs[1].Offset != 3 ||
		assembledOBUs[1].PrefixSize != 2 ||
		assembledOBUs[1].PayloadSize != 10 ||
		assembledOBUs[1].PayloadOffset() != 5 ||
		assembledOBUs[1].PayloadEnd() != wrote {
		t.Fatalf("assembled OBU1=%+v wrote=%d", assembledOBUs[1], wrote)
	}
}

func TestPublicRTPPacketDependencyDescriptorAllocs(t *testing.T) {
	frame := appendPublicLowOverheadOBU(nil, OBUFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}
	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [8]RTPPacketPlan
	var workScratch [8]RTPPacketPlan
	packetizer, err := NewRTPPacketizer(frame, limits, false, true, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatal(err)
	}
	control, err := EncoderWebRTCTemporalUnitControlForFrames(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	}, []EncoderFrameEncodeSettings{{
		Type:            EncoderFrameTypeKey,
		Resolution:      EncoderResolution{Width: 640, Height: 360},
		UpdateBuffer:    0,
		UpdateBufferSet: true,
		Output:          true,
	}}, EncoderReferenceBufferState{}, EncoderFrameIDBufferState{}, 77)
	if err != nil {
		t.Fatal(err)
	}
	frameControl := control.Frames[0]
	payloadSize, descriptorSize, ok, err := EncoderWebRTCFrameControlRTPPacketSize(&packetizer, frameControl, control.DependencyStructure)
	if err != nil || !ok || payloadSize == 0 || descriptorSize == 0 {
		t.Fatalf("preflight size payload=%d descriptor=%d ok=%v err=%v", payloadSize, descriptorSize, ok, err)
	}
	packetizer, err = NewRTPPacketizer(frame, limits, false, true, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatal(err)
	}
	var payloadBuf [16]byte
	var buf [32]byte
	allocs := testing.AllocsPerRun(1000, func() {
		p := packetizer
		_, _, _, _ = EncoderWebRTCFrameControlRTPPacketSize(&p, frameControl, control.DependencyStructure)
		_, _, _, _, _ = AppendEncoderWebRTCFrameControlRTPPacket(payloadBuf[:0], buf[:0], &p, frameControl, control.DependencyStructure)
	})
	if allocs != 0 {
		t.Fatalf("packet descriptor allocs=%f want 0", allocs)
	}
}

func TestPublicRTPAssembleFrameOBUMetadata(t *testing.T) {
	payloads := [][]byte{{
		0x10,
		byte(OBUTileGroup)<<3 | 0x04,
		(1 << 5) | (2 << 3),
		0xaa,
	}}

	var assembled [16]byte
	var obus [1]RTPFrameOBU
	wrote, count, err := AssembleRTPFrame(assembled[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{byte(OBUTileGroup)<<3 | 0x06, (1 << 5) | (2 << 3), 0x01, 0xaa}
	if count != 1 || string(assembled[:wrote]) != string(want) {
		t.Fatalf("count=%d assembled=%x want=%x", count, assembled[:wrote], want)
	}
	if obus[0].Header.Type != OBUTileGroup ||
		!obus[0].Header.Extension ||
		!obus[0].Header.HasSizeField ||
		obus[0].Header.TemporalID != 1 ||
		obus[0].Header.SpatialID != 2 ||
		obus[0].Offset != 0 ||
		obus[0].Length != wrote ||
		obus[0].PrefixSize != 3 ||
		obus[0].PayloadSize != 1 ||
		obus[0].PayloadOffset() != 3 ||
		obus[0].PayloadEnd() != 4 ||
		obus[0].End() != 4 {
		t.Fatalf("obu=%+v wrote=%d", obus[0], wrote)
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
