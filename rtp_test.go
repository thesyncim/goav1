package goav1

import (
	"errors"
	"testing"

	internalbitstream "github.com/thesyncim/goav1/internal/av1/bitstream"
	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
)

func appendTestLowOverheadOBU(dst []byte, typ internalobu.Type, payload []byte) []byte {
	if len(payload) > 0x7f {
		panic("test payload too large for one-byte LEB128")
	}
	var header [2]byte
	n, err := internalobu.PutHeader(header[:], internalobu.Header{Type: typ, HasSizeField: true})
	if err != nil {
		panic(err)
	}
	dst = append(dst, header[:n]...)
	dst = append(dst, byte(len(payload)))
	dst = append(dst, payload...)
	return dst
}

func appendTestLowOverheadOBUExt(dst []byte, typ internalobu.Type, temporalID uint8, spatialID uint8, payload []byte) []byte {
	if len(payload) > 0x7f {
		panic("test payload too large for one-byte LEB128")
	}
	var header [2]byte
	n, err := internalobu.PutHeader(header[:], internalobu.Header{
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
	dst = append(dst, byte(len(payload)))
	dst = append(dst, payload...)
	return dst
}

func TestPublicRTPPacketizerScratchAndAssembleSizing(t *testing.T) {
	var frame []byte
	frame = appendTestLowOverheadOBU(frame, internalobu.TypeSequenceHeader, []byte{0xaa})
	frame = appendTestLowOverheadOBU(frame, internalobu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
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
	if packetizerOBUs[0].Header.Type != internalobu.TypeSequenceHeader ||
		packetizerOBUs[0].Header.HasSizeField ||
		packetizerOBUs[0].HeaderSize != 1 ||
		packetizerOBUs[0].PayloadSize != 1 ||
		packetizerOBUs[0].Size != 2 {
		t.Fatalf("packetizer OBU0=%+v", packetizerOBUs[0])
	}
	if packetizerOBUs[1].Header.Type != internalobu.TypeFrame ||
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
	control, err := internalencoder.WebRTCTemporalUnitControlForFrames(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	}, []internalencoder.FrameEncodeSettings{{
		Type:            internalencoder.FrameTypeKey,
		Resolution:      EncoderResolution{Width: 640, Height: 360},
		UpdateBuffer:    0,
		UpdateBufferSet: true,
		Output:          true,
	}}, internalencoder.ReferenceBufferState{}, internalencoder.FrameIDBufferState{}, 77)
	if err != nil {
		t.Fatal(err)
	}
	frameControl := control.Frames[0]
	options := EncoderWebRTCRTPPacketDependencyDescriptorOptions{AttachStructureOnFirstPacket: frameControl.AttachDependencyStructure}
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
		descriptorSize, ok, err := publicRTPPacketDescriptorSize(&packetizer, control.DependencyStructure, frameControl.GenericFrameInfo, options)
		if err != nil || !ok {
			t.Fatalf("packet %d descriptor size ok=%v err=%v", payloadCount, ok, err)
		}
		payload, descriptor, marker, ok, err := appendEncoderWebRTCFrameControlRTPPacketWithOptions(packetBytes[payloadCount][:0], descriptors[payloadCount][:0], &packetizer, frameControl, control.DependencyStructure, options)
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
	if assembledOBUs[0].Header.Type != internalobu.TypeSequenceHeader ||
		!assembledOBUs[0].Header.HasSizeField ||
		assembledOBUs[0].Offset != 0 ||
		assembledOBUs[0].PrefixSize != 2 ||
		assembledOBUs[0].PayloadSize != 1 ||
		assembledOBUs[0].PayloadOffset() != 2 ||
		assembledOBUs[0].PayloadEnd() != 3 {
		t.Fatalf("assembled OBU0=%+v", assembledOBUs[0])
	}
	if assembledOBUs[1].Header.Type != internalobu.TypeFrame ||
		!assembledOBUs[1].Header.HasSizeField ||
		assembledOBUs[1].Offset != 3 ||
		assembledOBUs[1].PrefixSize != 2 ||
		assembledOBUs[1].PayloadSize != 10 ||
		assembledOBUs[1].PayloadOffset() != 5 ||
		assembledOBUs[1].PayloadEnd() != wrote {
		t.Fatalf("assembled OBU1=%+v wrote=%d", assembledOBUs[1], wrote)
	}
}

func TestPublicRTPPacketizerSplitsLayerTaggedOBUs(t *testing.T) {
	var frame []byte
	frame = appendTestLowOverheadOBU(frame, internalobu.TypeSequenceHeader, []byte{0xaa})
	frame = appendTestLowOverheadOBUExt(frame, internalobu.TypeFrameHeader, 0, 0, []byte{0x10})
	frame = appendTestLowOverheadOBU(frame, internalobu.TypeSequenceHeader, []byte{0xbb})
	frame = appendTestLowOverheadOBUExt(frame, internalobu.TypeTileGroup, 0, 1, []byte{0x11})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 1200}

	size, err := RTPPacketizerScratchLen(frame, limits, nil)
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 4 || size.Packets != 0 || size.Work != 0 {
		t.Fatalf("first pass size=%+v", size)
	}
	var packetizerOBUs [4]RTPPacketizerOBU
	size, err = RTPPacketizerScratchLen(frame, limits, packetizerOBUs[:])
	if err != nil {
		t.Fatal(err)
	}
	if size.OBUs != 4 || size.Packets != 2 || size.Work != 2 {
		t.Fatalf("second pass size=%+v want OBUs=4 Packets=2 Work=2", size)
	}

	var packets [2]RTPPacketPlan
	var work [2]RTPPacketPlan
	packetizer, err := NewRTPPacketizer(frame, limits, true, true, packetizerOBUs[:size.OBUs], packets[:size.Packets], work[:size.Work])
	if err != nil {
		t.Fatal(err)
	}
	var packetBytes [2][64]byte
	var payloads [2][]byte
	for i := range payloads {
		n, marker, ok, err := packetizer.NextPacket(packetBytes[i][:])
		if err != nil {
			t.Fatal(err)
		}
		if !ok || marker != (i == len(payloads)-1) {
			t.Fatalf("packet %d ok=%v marker=%v", i, ok, marker)
		}
		payloads[i] = packetBytes[i][:n]
	}
	if packetizer.NumPackets() != 0 {
		t.Fatalf("remaining packets=%d", packetizer.NumPackets())
	}

	firstHeaders := publicRTPPacketOBUHeaders(t, payloads[0])
	if len(firstHeaders) != 2 ||
		firstHeaders[0].Type != internalobu.TypeSequenceHeader || firstHeaders[0].Extension ||
		firstHeaders[1].Type != internalobu.TypeFrameHeader || !firstHeaders[1].Extension ||
		firstHeaders[1].TemporalID != 0 || firstHeaders[1].SpatialID != 0 {
		t.Fatalf("first packet headers=%+v", firstHeaders)
	}
	firstIterator, err := NewRTPPayloadIterator(payloads[0])
	if err != nil {
		t.Fatal(err)
	}
	if header := firstIterator.Header(); !header.StartsNewCodedVideoSequence || header.ElementCount != 2 {
		t.Fatalf("first aggregation header=%+v", header)
	}
	secondHeaders := publicRTPPacketOBUHeaders(t, payloads[1])
	if len(secondHeaders) != 2 ||
		secondHeaders[0].Type != internalobu.TypeSequenceHeader || secondHeaders[0].Extension ||
		secondHeaders[1].Type != internalobu.TypeTileGroup || !secondHeaders[1].Extension ||
		secondHeaders[1].TemporalID != 0 || secondHeaders[1].SpatialID != 1 {
		t.Fatalf("second packet headers=%+v", secondHeaders)
	}

	assembledLen, obuCount, err := AssembleRTPFrameSize(payloads[:])
	if err != nil {
		t.Fatal(err)
	}
	var assembled [32]byte
	var assembledOBUs [4]RTPFrameOBU
	wrote, count, err := AssembleRTPFrame(assembled[:assembledLen], payloads[:], assembledOBUs[:obuCount])
	if err != nil {
		t.Fatal(err)
	}
	if wrote != len(frame) || count != 4 || string(assembled[:wrote]) != string(frame) {
		t.Fatalf("assembled wrote=%d count=%d bytes=%x want=%x", wrote, count, assembled[:wrote], frame)
	}
}

func publicRTPPacketOBUHeaders(t *testing.T, payload []byte) []internalobu.Header {
	t.Helper()
	it, err := NewRTPPayloadIterator(payload)
	if err != nil {
		t.Fatal(err)
	}
	var headers []internalobu.Header
	for {
		element, ok, err := it.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			return headers
		}
		header, _, err := internalobu.ParseHeader(element.Data)
		if err != nil {
			t.Fatal(err)
		}
		headers = append(headers, header)
	}
}

func TestPublicRTPPacketDependencyDescriptorAllocs(t *testing.T) {
	frame := appendTestLowOverheadOBU(nil, internalobu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}
	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [8]RTPPacketPlan
	var workScratch [8]RTPPacketPlan
	packetizer, err := NewRTPPacketizer(frame, limits, false, true, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatal(err)
	}
	control, err := internalencoder.WebRTCTemporalUnitControlForFrames(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	}, []internalencoder.FrameEncodeSettings{{
		Type:            internalencoder.FrameTypeKey,
		Resolution:      EncoderResolution{Width: 640, Height: 360},
		UpdateBuffer:    0,
		UpdateBufferSet: true,
		Output:          true,
	}}, internalencoder.ReferenceBufferState{}, internalencoder.FrameIDBufferState{}, 77)
	if err != nil {
		t.Fatal(err)
	}
	frameControl := control.Frames[0]
	options := EncoderWebRTCRTPPacketDependencyDescriptorOptions{AttachStructureOnFirstPacket: frameControl.AttachDependencyStructure}
	payloadSize, _ := packetizer.NextPacketSize()
	descriptorSize, ok, err := publicRTPPacketDescriptorSize(&packetizer, control.DependencyStructure, frameControl.GenericFrameInfo, options)
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
		_, _, _ = publicRTPPacketDescriptorSize(&p, control.DependencyStructure, frameControl.GenericFrameInfo, options)
		_, _, _, _, _ = appendEncoderWebRTCFrameControlRTPPacketWithOptions(payloadBuf[:0], buf[:0], &p, frameControl, control.DependencyStructure, options)
	})
	if allocs != 0 {
		t.Fatalf("packet descriptor allocs=%f want 0", allocs)
	}
}

func TestPublicRTPScheduledPictureDependencyDescriptor(t *testing.T) {
	frame := appendTestLowOverheadOBU(nil, internalobu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	var keyFrame []byte
	keyFrame = appendTestLowOverheadOBU(keyFrame, internalobu.TypeSequenceHeader, []byte{0xaa})
	keyFrame = appendTestLowOverheadOBU(keyFrame, internalobu.TypeFrame, []byte{0, 1, 2, 3})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 64}
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}

	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 100}, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState key: %v", err)
	}
	if encoderWebRTCPictureTemporalUnitFrameNum(unit) != 2 {
		t.Fatalf("scheduled key unit=%+v", unit)
	}
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, 0)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFrameControl key: %v", err)
	}
	if !control.AttachDependencyStructure || structure.TemplateNum == 0 {
		t.Fatalf("key control=%+v structure=%+v", control, structure)
	}
	if _, _, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, 2); err != ErrEncoderInvalidFrame {
		t.Fatalf("bad key frame index err=%v want %v", err, ErrEncoderInvalidFrame)
	}
	firstScratch, err := encoderWebRTCPictureTemporalUnitRTPScratchLen(frame, limits, unit, state, 0, nil)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitRTPScratchLen key first: %v", err)
	}
	if firstScratch.Packetizer.OBUs != 1 || firstScratch.Packetizer.Packets != 0 ||
		firstScratch.MaxPayloadBytes != limits.MaxPayloadLen || firstScratch.MaxDescriptorBytes == 0 {
		t.Fatalf("key first scratch=%+v", firstScratch)
	}
	keyDescriptorSize, err := encoderWebRTCPictureTemporalUnitDependencyDescriptorSize(unit, state, 0, true)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitDependencyDescriptorSize key: %v", err)
	}
	maxDescriptorSize, err := encoderWebRTCPictureTemporalUnitMaxDependencyDescriptorSize(unit, state, 0)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitMaxDependencyDescriptorSize key: %v", err)
	}
	if keyDescriptorSize != maxDescriptorSize || keyDescriptorSize <= RTPDependencyDescriptorMandatorySize {
		t.Fatalf("key descriptor size=%d max=%d", keyDescriptorSize, maxDescriptorSize)
	}
	var directDescriptor [64]byte
	direct, err := appendEncoderWebRTCPictureTemporalUnitDependencyDescriptor(directDescriptor[:0], unit, state, 0, true, false, true)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCPictureTemporalUnitDependencyDescriptor key: %v", err)
	}
	if len(direct) != keyDescriptorSize {
		t.Fatalf("direct key descriptor=%d want=%d", len(direct), keyDescriptorSize)
	}
	if out, err := appendEncoderWebRTCPictureTemporalUnitDependencyDescriptor(directDescriptor[:0:1], unit, state, 0, true, false, true); err == nil || len(out) != 0 {
		t.Fatalf("short key descriptor out=% x err=%v", out, err)
	}

	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [4]RTPPacketPlan
	var workScratch [4]RTPPacketPlan
	fullScratch, err := encoderWebRTCPictureTemporalUnitRTPScratchLen(frame, limits, unit, state, 0, obuScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitRTPScratchLen key full: %v", err)
	}
	if fullScratch.Packetizer.OBUs != 1 || fullScratch.Packetizer.Packets != 1 ||
		fullScratch.Packetizer.Work != 0 || fullScratch.MaxDescriptorBytes != keyDescriptorSize {
		t.Fatalf("key full scratch=%+v descriptor=%d", fullScratch, keyDescriptorSize)
	}
	packetizer, frameControl, frameStructure, err := newEncoderWebRTCPictureTemporalUnitRTPPacketizer(keyFrame, limits, unit, state, 0, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("NewEncoderWebRTCPictureTemporalUnitRTPPacketizer key: %v", err)
	}
	if frameControl != control || frameStructure != structure {
		t.Fatalf("key packetizer control=%+v structure=%+v want control=%+v structure=%+v", frameControl, frameStructure, control, structure)
	}
	payloadSize, descriptorSize, ok, err := encoderWebRTCPictureTemporalUnitRTPPacketSize(&packetizer, unit, state, 0)
	if err != nil || !ok || payloadSize == 0 || descriptorSize <= RTPDependencyDescriptorMandatorySize {
		t.Fatalf("key packet size payload=%d descriptor=%d ok=%v err=%v", payloadSize, descriptorSize, ok, err)
	}
	if descriptorSize != keyDescriptorSize {
		t.Fatalf("key packet descriptor size=%d direct=%d", descriptorSize, keyDescriptorSize)
	}
	packetizer, _, _, err = newEncoderWebRTCPictureTemporalUnitRTPPacketizer(keyFrame, limits, unit, state, 0, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("NewEncoderWebRTCPictureTemporalUnitRTPPacketizer key append: %v", err)
	}
	var payloadBuf [64]byte
	var descriptorBuf [64]byte
	payload, descriptor, marker, ok, err := appendEncoderWebRTCPictureTemporalUnitRTPPacket(payloadBuf[:0], descriptorBuf[:0], &packetizer, unit, state, 0)
	if err != nil || !ok || marker || len(payload) != payloadSize || len(descriptor) != descriptorSize {
		t.Fatalf("key packet payload=%d/%d descriptor=%d/%d marker=%v ok=%v err=%v", len(payload), payloadSize, len(descriptor), descriptorSize, marker, ok, err)
	}
	header, _, err := ParseRTPAggregationHeader(payload)
	if err != nil {
		t.Fatalf("ParseRTPAggregationHeader key: %v", err)
	}
	if !header.StartsNewCodedVideoSequence {
		t.Fatalf("key aggregation header=%+v", header)
	}
	directPayload, directPacketDescriptor, directMarker, directOK, directControl, directStructure, err := appendEncoderWebRTCPictureTemporalUnitFirstRTPPacket(
		payloadBuf[:0],
		descriptorBuf[:0],
		keyFrame,
		limits,
		unit,
		state,
		0,
		obuScratch[:],
		packetScratch[:],
		workScratch[:],
	)
	if err != nil || !directOK || directMarker || len(directPayload) != len(payload) || len(directPacketDescriptor) != len(descriptor) ||
		directControl != control || directStructure != structure {
		t.Fatalf("direct key payload=%d/%d descriptor=%d/%d marker=%v ok=%v control=%+v structure=%+v err=%v", len(directPayload), len(payload), len(directPacketDescriptor), len(descriptor), directMarker, directOK, directControl, directStructure, err)
	}

	unit, state, err = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta: %v", err)
	}
	fullScratch, err = encoderWebRTCPictureTemporalUnitRTPScratchLen(frame, limits, unit, state, 1, obuScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitRTPScratchLen delta full: %v", err)
	}
	if fullScratch.Packetizer.Packets != 1 || fullScratch.MaxPayloadBytes != limits.MaxPayloadLen {
		t.Fatalf("delta full scratch=%+v", fullScratch)
	}
	packetizer, frameControl, _, err = newEncoderWebRTCPictureTemporalUnitRTPPacketizer(frame, limits, unit, state, 1, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("NewEncoderWebRTCPictureTemporalUnitRTPPacketizer delta: %v", err)
	}
	if frameControl.Settings.Type != internalencoder.FrameTypeDelta {
		t.Fatalf("delta frame control=%+v", frameControl)
	}
	payloadSize, descriptorSize, ok, err = encoderWebRTCPictureTemporalUnitRTPPacketSize(&packetizer, unit, state, 1)
	if err != nil || !ok || payloadSize == 0 || descriptorSize == 0 {
		t.Fatalf("delta packet size payload=%d descriptor=%d ok=%v err=%v", payloadSize, descriptorSize, ok, err)
	}
	deltaDescriptorSize, err := encoderWebRTCPictureTemporalUnitDependencyDescriptorSize(unit, state, 1, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitDependencyDescriptorSize delta: %v", err)
	}
	if descriptorSize != deltaDescriptorSize {
		t.Fatalf("delta packet descriptor size=%d direct=%d", descriptorSize, deltaDescriptorSize)
	}
	if fullScratch.MaxDescriptorBytes != deltaDescriptorSize {
		t.Fatalf("delta max descriptor=%d direct=%d scratch=%+v", fullScratch.MaxDescriptorBytes, deltaDescriptorSize, fullScratch)
	}
	packetizer, _, _, err = newEncoderWebRTCPictureTemporalUnitRTPPacketizer(frame, limits, unit, state, 1, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("NewEncoderWebRTCPictureTemporalUnitRTPPacketizer delta append: %v", err)
	}
	payload, descriptor, marker, ok, err = appendEncoderWebRTCPictureTemporalUnitRTPPacket(payloadBuf[:0], descriptorBuf[:0], &packetizer, unit, state, 1)
	if err != nil || !ok || !marker || len(payload) != payloadSize || len(descriptor) != descriptorSize {
		t.Fatalf("delta packet payload=%d/%d descriptor=%d/%d marker=%v ok=%v err=%v", len(payload), payloadSize, len(descriptor), descriptorSize, marker, ok, err)
	}
	header, _, err = ParseRTPAggregationHeader(payload)
	if err != nil {
		t.Fatalf("ParseRTPAggregationHeader delta: %v", err)
	}
	if header.StartsNewCodedVideoSequence {
		t.Fatalf("delta aggregation header=%+v", header)
	}
	directPayload, directPacketDescriptor, directMarker, directOK, directControl, _, err = appendEncoderWebRTCPictureTemporalUnitFirstRTPPacket(
		payloadBuf[:0],
		descriptorBuf[:0],
		frame,
		limits,
		unit,
		state,
		1,
		obuScratch[:],
		packetScratch[:],
		workScratch[:],
	)
	if err != nil || !directOK || !directMarker || len(directPayload) != len(payload) || len(directPacketDescriptor) != len(descriptor) ||
		directControl.Settings.Type != internalencoder.FrameTypeDelta {
		t.Fatalf("direct delta payload=%d/%d descriptor=%d/%d marker=%v ok=%v control=%+v err=%v", len(directPayload), len(payload), len(directPacketDescriptor), len(descriptor), directMarker, directOK, directControl, err)
	}
	if _, _, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, internalencoder.WebRTCEncoderState{}, 1); err != ErrEncoderInvalidFrame {
		t.Fatalf("missing delta structure err=%v want %v", err, ErrEncoderInvalidFrame)
	}
}

func TestPublicRTPDependencyDescriptorParseEncoderRoundTrip(t *testing.T) {
	structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(EncoderConfig{
		Resolution:  EncoderResolution{Width: 640, Height: 360},
		Scalability: internalencoder.ScalabilityModeL2T2_KEY_SHIFT,
	})
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := internalencoder.WebRTCGenericFrameInfo{
		FrameID:    100,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     structure.NumDecodeTargets,
	}
	info.DTIs = structure.Templates[0].DTIs

	var buf [128]byte
	descriptorBytes, err := internalencoder.AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, false, true)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor key: %v", err)
	}
	var descriptorState RTPDependencyDescriptorState
	parsed, consumed, err := descriptorState.Parse(descriptorBytes)
	if err != nil {
		t.Fatalf("state Parse key: %v", err)
	}
	if consumed != len(descriptorBytes) ||
		parsed.Mandatory != (RTPDependencyDescriptorMandatory{FirstPacketInFrame: true, TemplateID: 0, FrameNumber: 100}) ||
		!parsed.HasAttachedStructure || !parsed.HasActiveDecodeTargets ||
		parsed.ActiveDecodeTargetsMask != 0x0f ||
		!descriptorState.Valid {
		t.Fatalf("parsed key consumed=%d/%d descriptor=%+v state=%+v", consumed, len(descriptorBytes), parsed, descriptorState)
	}
	assertPublicRTPStructureMatchesEncoder(t, parsed.AttachedStructure, structure)
	if parsed.FrameDependencies.SpatialID != 0 || parsed.FrameDependencies.TemporalID != 0 ||
		!parsed.HasResolution || parsed.Resolution != (RTPDependencyDescriptorResolution{Width: 320, Height: 180}) {
		t.Fatalf("parsed key frame=%+v resolution=%+v has=%v", parsed.FrameDependencies, parsed.Resolution, parsed.HasResolution)
	}

	info = internalencoder.WebRTCGenericFrameInfo{
		FrameID:       106,
		SpatialID:     1,
		TemporalID:    0,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 102
	info.DTIs = structure.Templates[5].DTIs
	descriptorBytes, err = internalencoder.AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor delta: %v", err)
	}
	parsed, consumed, err = descriptorState.Parse(descriptorBytes)
	if err != nil {
		t.Fatalf("state Parse delta: %v", err)
	}
	if consumed != len(descriptorBytes) ||
		parsed.Mandatory.TemplateID != 5 ||
		parsed.FrameDependencies.SpatialID != 1 ||
		parsed.FrameDependencies.TemporalID != 0 ||
		parsed.FrameDependencies.FrameDiffNum != 1 ||
		parsed.FrameDependencies.FrameDiffs[0] != 4 ||
		!parsed.HasResolution ||
		parsed.Resolution != (RTPDependencyDescriptorResolution{Width: 640, Height: 360}) {
		t.Fatalf("parsed delta consumed=%d/%d descriptor=%+v", consumed, len(descriptorBytes), parsed)
	}
}

func TestPublicRTPDependencyDescriptorActiveDecodeTargetsRoundTrip(t *testing.T) {
	structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(EncoderConfig{
		Resolution:  EncoderResolution{Width: 960, Height: 540},
		Scalability: internalencoder.ScalabilityModeL3T2,
	})
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := internalencoder.WebRTCGenericFrameInfo{
		FrameID:    77,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     structure.NumDecodeTargets,
	}
	info.DTIs = structure.Templates[0].DTIs

	options := internalencoder.WebRTCDependencyDescriptorOptions{
		FirstPacketInFrame:         true,
		ActiveDecodeTargetsPresent: true,
		ActiveDecodeTargetsMask:    0x15,
	}
	size, err := internalencoder.WebRTCDependencyDescriptorSizeWithOptions(structure, info, options)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSizeWithOptions: %v", err)
	}
	var buf [256]byte
	descriptorBytes, err := internalencoder.AppendWebRTCDependencyDescriptorWithOptions(buf[:0], structure, info, options)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptorWithOptions: %v", err)
	}
	if len(descriptorBytes) != size {
		t.Fatalf("descriptor len=%d want size %d", len(descriptorBytes), size)
	}

	parsedStructure := convertPublicEncoderRTPDependencyStructure(structure)
	parsed, consumed, err := ParseRTPDependencyDescriptor(descriptorBytes, &parsedStructure)
	if err != nil {
		t.Fatalf("ParseRTPDependencyDescriptor active mask: %v", err)
	}
	if consumed != len(descriptorBytes) ||
		!parsed.HasExtendedFields ||
		parsed.HasAttachedStructure ||
		!parsed.HasActiveDecodeTargets ||
		parsed.ActiveDecodeTargetsMask != options.ActiveDecodeTargetsMask ||
		parsed.Mandatory != (RTPDependencyDescriptorMandatory{FirstPacketInFrame: true, TemplateID: 0, FrameNumber: uint16(info.FrameID)}) {
		t.Fatalf("parsed active mask consumed=%d/%d descriptor=%+v", consumed, len(descriptorBytes), parsed)
	}

	options.AttachStructure = true
	options.LastPacketInFrame = true
	options.ActiveDecodeTargetsMask = 0x03
	descriptorBytes, err = internalencoder.AppendWebRTCDependencyDescriptorWithOptions(buf[:0], structure, info, options)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptorWithOptions attached: %v", err)
	}
	parsed, consumed, err = ParseRTPDependencyDescriptor(descriptorBytes, nil)
	if err != nil {
		t.Fatalf("ParseRTPDependencyDescriptor attached active mask: %v", err)
	}
	if consumed != len(descriptorBytes) ||
		!parsed.HasAttachedStructure ||
		!parsed.HasActiveDecodeTargets ||
		parsed.ActiveDecodeTargetsMask != options.ActiveDecodeTargetsMask ||
		parsed.Mandatory != (RTPDependencyDescriptorMandatory{FirstPacketInFrame: true, LastPacketInFrame: true, TemplateID: 0, FrameNumber: uint16(info.FrameID)}) {
		t.Fatalf("parsed attached active mask consumed=%d/%d descriptor=%+v", consumed, len(descriptorBytes), parsed)
	}

	options.ActiveDecodeTargetsMask = 1 << structure.NumDecodeTargets
	if _, err := internalencoder.WebRTCDependencyDescriptorSizeWithOptions(structure, info, options); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("invalid active mask size err=%v want %v", err, ErrEncoderInvalidFrame)
	}
	if _, err := internalencoder.AppendWebRTCDependencyDescriptorWithOptions(buf[:0], structure, info, options); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("invalid active mask append err=%v want %v", err, ErrEncoderInvalidFrame)
	}
}

// publicRTPPacketDescriptorSize is the dependency-descriptor size the next
// packet from packetizer carries under options, computed independently of the
// descriptor writer.
func publicRTPPacketDescriptorSize(packetizer *RTPPacketizer, structure internalencoder.WebRTCFrameDependencyStructure, info internalencoder.WebRTCGenericFrameInfo, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (int, bool, error) {
	flags, ok := NextRTPPacketDependencyDescriptorFlags(packetizer)
	if !ok {
		return 0, false, nil
	}
	size, err := internalencoder.WebRTCDependencyDescriptorSizeWithOptions(structure, info, encoderWebRTCRTPPacketDependencyDescriptorOptions(flags, options))
	return size, true, err
}

func TestPublicRTPPacketDependencyDescriptorActiveDecodeTargets(t *testing.T) {
	frame := appendTestLowOverheadOBU(nil, internalobu.TypeFrame, []byte{
		0, 1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22, 23,
	})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}
	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [8]RTPPacketPlan
	var workScratch [8]RTPPacketPlan
	packetizer, err := NewRTPPacketizer(frame, limits, false, true, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("NewRTPPacketizer: %v", err)
	}
	if packetizer.NumPackets() < 2 {
		t.Fatalf("packetizer packets=%d want fragmented frame", packetizer.NumPackets())
	}

	structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(EncoderConfig{
		Resolution:  EncoderResolution{Width: 960, Height: 540},
		Scalability: internalencoder.ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := internalencoder.WebRTCGenericFrameInfo{
		FrameID:    200,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     structure.NumDecodeTargets,
	}
	info.DTIs = structure.Templates[0].DTIs
	options := EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket:            true,
		ActiveDecodeTargetsPresentOnFirstPacket: true,
		ActiveDecodeTargetsMask:                 0x03,
	}

	size, ok, err := publicRTPPacketDescriptorSize(&packetizer, structure, info, options)
	if err != nil || !ok {
		t.Fatalf("publicRTPPacketDescriptorSize first size=%d ok=%v err=%v", size, ok, err)
	}
	var descriptorBuf [256]byte
	descriptor, ok, err := appendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(descriptorBuf[:0], &packetizer, structure, info, options)
	if err != nil || !ok || len(descriptor) != size {
		t.Fatalf("AppendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions first len=%d size=%d ok=%v err=%v", len(descriptor), size, ok, err)
	}
	var receiver RTPDependencyDescriptorState
	parsed, consumed, err := receiver.Parse(descriptor)
	if err != nil {
		t.Fatalf("receiver Parse first: %v", err)
	}
	if consumed != len(descriptor) ||
		!parsed.Mandatory.FirstPacketInFrame ||
		parsed.Mandatory.LastPacketInFrame ||
		!parsed.HasAttachedStructure ||
		!parsed.HasActiveDecodeTargets ||
		parsed.ActiveDecodeTargetsMask != options.ActiveDecodeTargetsMask {
		t.Fatalf("first descriptor consumed=%d/%d parsed=%+v", consumed, len(descriptor), parsed)
	}
	if !receiver.ActiveDecodeTargetsValid ||
		receiver.ActiveDecodeTargetsMask != options.ActiveDecodeTargetsMask {
		t.Fatalf("first descriptor active target state=%+v", receiver)
	}
	var payloadBuf [16]byte
	if _, _, ok, err := packetizer.NextPacket(payloadBuf[:]); err != nil || !ok {
		t.Fatalf("NextPacket first ok=%v err=%v", ok, err)
	}

	secondSize, ok, err := publicRTPPacketDescriptorSize(&packetizer, structure, info, options)
	if err != nil || !ok {
		t.Fatalf("publicRTPPacketDescriptorSize second size=%d ok=%v err=%v", secondSize, ok, err)
	}
	secondDescriptor, ok, err := appendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(descriptorBuf[:0], &packetizer, structure, info, options)
	if err != nil || !ok || len(secondDescriptor) != secondSize {
		t.Fatalf("AppendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions second len=%d size=%d ok=%v err=%v", len(secondDescriptor), secondSize, ok, err)
	}
	parsed, consumed, err = receiver.Parse(secondDescriptor)
	if err != nil {
		t.Fatalf("receiver Parse second: %v", err)
	}
	if consumed != len(secondDescriptor) ||
		parsed.Mandatory.FirstPacketInFrame ||
		parsed.HasAttachedStructure ||
		parsed.HasActiveDecodeTargets ||
		secondSize != RTPDependencyDescriptorMandatorySize ||
		!receiver.ActiveDecodeTargetsValid ||
		receiver.ActiveDecodeTargetsMask != options.ActiveDecodeTargetsMask {
		t.Fatalf("second descriptor size=%d consumed=%d/%d parsed=%+v state=%+v", secondSize, consumed, len(secondDescriptor), parsed, receiver)
	}

	packetizer, err = NewRTPPacketizer(frame, limits, false, true, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("NewRTPPacketizer invalid: %v", err)
	}
	options.ActiveDecodeTargetsMask = 1 << structure.NumDecodeTargets
	if _, ok, err := publicRTPPacketDescriptorSize(&packetizer, structure, info, options); !ok || !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("invalid active mask packet size ok=%v err=%v want %v", ok, err, ErrEncoderInvalidFrame)
	}
	if _, ok, err := appendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(descriptorBuf[:0], &packetizer, structure, info, options); !ok || !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("invalid active mask packet append ok=%v err=%v want %v", ok, err, ErrEncoderInvalidFrame)
	}
}

func TestPublicRTPDependencyDescriptorParseAllocs(t *testing.T) {
	structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(EncoderConfig{
		Resolution:  EncoderResolution{Width: 640, Height: 360},
		Scalability: internalencoder.ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := internalencoder.WebRTCGenericFrameInfo{
		FrameID:    100,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     structure.NumDecodeTargets,
	}
	info.DTIs = structure.Templates[0].DTIs
	var buf [128]byte
	descriptorBytes, err := internalencoder.AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, true)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	parsedStructure := convertPublicEncoderRTPDependencyStructure(structure)
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = ParseRTPDependencyDescriptor(descriptorBytes, nil)
		_, _, _ = ParseRTPDependencyDescriptor(descriptorBytes[:RTPDependencyDescriptorMandatorySize], &parsedStructure)
	})
	if allocs != 0 {
		t.Fatalf("dependency descriptor parse allocs=%f want 0", allocs)
	}
}

func assertPublicRTPStructureMatchesEncoder(t *testing.T, got RTPDependencyDescriptorStructure, want internalencoder.WebRTCFrameDependencyStructure) {
	t.Helper()
	if got.StructureID != want.StructureID ||
		got.NumDecodeTargets != want.NumDecodeTargets ||
		got.NumChains != want.NumChains ||
		got.TemplateNum != want.TemplateNum ||
		got.ResolutionNum != want.ResolutionNum {
		t.Fatalf("structure header=%+v want id=%d targets=%d chains=%d templates=%d resolutions=%d", got, want.StructureID, want.NumDecodeTargets, want.NumChains, want.TemplateNum, want.ResolutionNum)
	}
	for i := uint8(0); i < want.NumDecodeTargets; i++ {
		if got.DecodeTargetProtectedByChain[i] != want.DecodeTargetProtectedByChain[i] {
			t.Fatalf("protected chain[%d]=%d want %d", i, got.DecodeTargetProtectedByChain[i], want.DecodeTargetProtectedByChain[i])
		}
	}
	for i := uint8(0); i < want.TemplateNum; i++ {
		gotTemplate := got.Templates[i]
		wantTemplate := want.Templates[i]
		if gotTemplate.SpatialID != wantTemplate.SpatialID ||
			gotTemplate.TemporalID != wantTemplate.TemporalID ||
			gotTemplate.DTINum != wantTemplate.DTINum ||
			gotTemplate.FrameDiffNum != wantTemplate.FrameDiffNum ||
			gotTemplate.ChainDiffNum != wantTemplate.ChainDiffNum {
			t.Fatalf("template[%d]=%+v want %+v", i, gotTemplate, wantTemplate)
		}
		for j := uint8(0); j < wantTemplate.DTINum; j++ {
			if uint8(gotTemplate.DTIs[j]) != uint8(wantTemplate.DTIs[j]) {
				t.Fatalf("template[%d].dti[%d]=%d want %d", i, j, gotTemplate.DTIs[j], wantTemplate.DTIs[j])
			}
		}
		for j := uint8(0); j < wantTemplate.FrameDiffNum; j++ {
			if gotTemplate.FrameDiffs[j] != wantTemplate.FrameDiffs[j] {
				t.Fatalf("template[%d].fdiff[%d]=%d want %d", i, j, gotTemplate.FrameDiffs[j], wantTemplate.FrameDiffs[j])
			}
		}
		for j := uint8(0); j < wantTemplate.ChainDiffNum; j++ {
			if gotTemplate.ChainDiffs[j] != wantTemplate.ChainDiffs[j] {
				t.Fatalf("template[%d].chain[%d]=%d want %d", i, j, gotTemplate.ChainDiffs[j], wantTemplate.ChainDiffs[j])
			}
		}
	}
	for i := uint8(0); i < want.ResolutionNum; i++ {
		if got.Resolutions[i].Width != uint16(want.Resolutions[i].Width) ||
			got.Resolutions[i].Height != uint16(want.Resolutions[i].Height) {
			t.Fatalf("resolution[%d]=%+v want %+v", i, got.Resolutions[i], want.Resolutions[i])
		}
	}
}

func convertPublicEncoderRTPDependencyStructure(src internalencoder.WebRTCFrameDependencyStructure) RTPDependencyDescriptorStructure {
	var out RTPDependencyDescriptorStructure
	out.StructureID = src.StructureID
	out.NumDecodeTargets = src.NumDecodeTargets
	out.NumChains = src.NumChains
	out.TemplateNum = src.TemplateNum
	out.ResolutionNum = src.ResolutionNum
	for i := uint8(0); i < src.NumDecodeTargets; i++ {
		out.DecodeTargetProtectedByChain[i] = src.DecodeTargetProtectedByChain[i]
	}
	for i := uint8(0); i < src.TemplateNum; i++ {
		inTemplate := src.Templates[i]
		outTemplate := &out.Templates[i]
		outTemplate.SpatialID = inTemplate.SpatialID
		outTemplate.TemporalID = inTemplate.TemporalID
		outTemplate.DTINum = inTemplate.DTINum
		outTemplate.FrameDiffNum = inTemplate.FrameDiffNum
		outTemplate.ChainDiffNum = inTemplate.ChainDiffNum
		for j := uint8(0); j < inTemplate.DTINum; j++ {
			outTemplate.DTIs[j] = RTPDependencyDescriptorDecodeTargetIndication(inTemplate.DTIs[j])
		}
		for j := uint8(0); j < inTemplate.FrameDiffNum; j++ {
			outTemplate.FrameDiffs[j] = inTemplate.FrameDiffs[j]
		}
		for j := uint8(0); j < inTemplate.ChainDiffNum; j++ {
			outTemplate.ChainDiffs[j] = inTemplate.ChainDiffs[j]
		}
	}
	for i := uint8(0); i < src.ResolutionNum; i++ {
		out.Resolutions[i] = RTPDependencyDescriptorResolution{
			Width:  uint16(src.Resolutions[i].Width),
			Height: uint16(src.Resolutions[i].Height),
		}
	}
	for target := uint8(0); target < out.NumDecodeTargets; target++ {
		var spatialID uint8
		var temporalID uint8
		for templateIndex := uint8(0); templateIndex < out.TemplateNum; templateIndex++ {
			template := out.Templates[templateIndex]
			if template.DTIs[target] == RTPDependencyDescriptorDecodeTargetNotPresent {
				continue
			}
			if template.SpatialID > spatialID {
				spatialID = template.SpatialID
			}
			if template.TemporalID > temporalID {
				temporalID = template.TemporalID
			}
		}
		out.DecodeTargetSpatialID[target] = spatialID
		out.DecodeTargetTemporalID[target] = temporalID
	}
	return out
}

func TestPublicRTPScheduledPictureDependencyDescriptorAllocs(t *testing.T) {
	frame := appendTestLowOverheadOBU(nil, internalobu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 64}
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 100}, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState key: %v", err)
	}
	unit, state, err = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta: %v", err)
	}
	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [4]RTPPacketPlan
	var workScratch [4]RTPPacketPlan
	packetizer, err := NewRTPPacketizer(frame, limits, false, true, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("NewRTPPacketizer: %v", err)
	}
	var payloadBuf [64]byte
	var descriptorBuf [64]byte
	var frameOBUBuf [32]byte
	var frameRTPBuf [32]byte
	var spans [4]EncoderWebRTCRTPPacketSpan
	var obuSpans [2]encoderWebRTCFrameOBUSpan
	var frameSpans [2]encoderWebRTCFrameRTPPacketSpan
	framePayloads := [...][]byte{frame, frame}
	allocs := testing.AllocsPerRun(1000, func() {
		p := packetizer
		_, _ = encoderWebRTCPictureTemporalUnitRTPScratchLen(frame, limits, unit, state, 1, obuScratch[:])
		_, _, _, _ = newEncoderWebRTCPictureTemporalUnitRTPPacketizer(frame, limits, unit, state, 1, obuScratch[:], packetScratch[:], workScratch[:])
		_, _, _, _ = encoderWebRTCPictureTemporalUnitRTPPacketsSize(frame, limits, unit, state, 1, obuScratch[:], packetScratch[:], workScratch[:])
		_, _ = encoderWebRTCPictureTemporalUnitDependencyDescriptorSize(unit, state, 1, false)
		_, _ = encoderWebRTCPictureTemporalUnitMaxDependencyDescriptorSize(unit, state, 1)
		_, _ = appendEncoderWebRTCPictureTemporalUnitDependencyDescriptor(descriptorBuf[:0], unit, state, 1, true, true, false)
		_, _, _, _ = encoderWebRTCPictureTemporalUnitFrameOBUSize(frame, unit, state, 1)
		_, _, _, _ = appendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUBuf[:0], frame, unit, state, 1)
		_, _ = encoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads[:], unit, state)
		_, _, _ = appendEncoderWebRTCPictureTemporalUnitFramesOBU(frameRTPBuf[:0], obuSpans[:], framePayloads[:], unit, state)
		_, _ = encoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads[:], limits, unit, state, frameRTPBuf[:0], obuScratch[:])
		_, _, _, _ = encoderWebRTCPictureTemporalUnitFrameRTPPacketsSize(frame, limits, unit, state, 1, frameRTPBuf[:0], obuScratch[:], packetScratch[:], workScratch[:])
		_, _, _, _, _, _, _ = appendEncoderWebRTCPictureTemporalUnitFrameRTPPackets(payloadBuf[:0], descriptorBuf[:0], spans[:], frameRTPBuf[:0], frame, limits, unit, state, 1, obuScratch[:], packetScratch[:], workScratch[:])
		rtpScratch, _ := encoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads[:], limits, unit, state, frameRTPBuf[:0], obuScratch[:])
		_, _ = bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(rtpScratch, encoderWebRTCPictureTemporalUnitFramesRTPScratch{FrameOBU: frameRTPBuf[:0], FrameSpans: frameSpans[:], PacketSpans: spans[:], OBUs: obuScratch[:], Packets: packetScratch[:], Work: workScratch[:]})
		_, _ = encoderWebRTCPictureTemporalUnitFramesRTPPacketsSize(framePayloads[:], limits, unit, state, frameRTPBuf[:0], obuScratch[:], packetScratch[:], workScratch[:])
		_, _, _, _, _, _ = appendEncoderWebRTCPictureTemporalUnitFramesRTPPackets(frameRTPBuf[:0], payloadBuf[:0], descriptorBuf[:0], frameSpans[:], spans[:], framePayloads[:], limits, unit, state, obuScratch[:], packetScratch[:], workScratch[:])
		_, _, _, _, _, _ = appendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch(payloadBuf[:0], descriptorBuf[:0], encoderWebRTCPictureTemporalUnitFramesRTPScratch{FrameOBU: frameRTPBuf[:0], FrameSpans: frameSpans[:], PacketSpans: spans[:], OBUs: obuScratch[:], Packets: packetScratch[:], Work: workScratch[:]}, framePayloads[:], limits, unit, state)
		_, _, _, _ = encoderWebRTCNextTemporalUnitFramesRTPScratchLen(framePayloads[:], limits, cfg, internalencoder.WebRTCEncoderState{NextFrameID: 100}, false, frameRTPBuf[:0], obuScratch[:])
		_, _, _, _ = encoderWebRTCNextTemporalUnitFramesRTPPacketsSize(framePayloads[:], limits, cfg, internalencoder.WebRTCEncoderState{NextFrameID: 100}, false, frameRTPBuf[:0], obuScratch[:], packetScratch[:], workScratch[:])
		_, _, _, _, _, _, _, _ = appendEncoderWebRTCNextTemporalUnitFramesRTPPacketsWithScratch(payloadBuf[:0], descriptorBuf[:0], encoderWebRTCPictureTemporalUnitFramesRTPScratch{FrameOBU: frameRTPBuf[:0], FrameSpans: frameSpans[:], PacketSpans: spans[:], OBUs: obuScratch[:], Packets: packetScratch[:], Work: workScratch[:]}, framePayloads[:], limits, cfg, internalencoder.WebRTCEncoderState{NextFrameID: 100}, false)
		_, _, _, _ = encoderWebRTCPictureTemporalUnitRTPPacketSize(&p, unit, state, 1)
		_, _, _, _, _ = appendEncoderWebRTCPictureTemporalUnitRTPPacket(payloadBuf[:0], descriptorBuf[:0], &p, unit, state, 1)
		_, _, _, _, _, _, _ = appendEncoderWebRTCPictureTemporalUnitFirstRTPPacket(payloadBuf[:0], descriptorBuf[:0], frame, limits, unit, state, 1, obuScratch[:], packetScratch[:], workScratch[:])
		_, _, _, _, _, _ = appendEncoderWebRTCPictureTemporalUnitRTPPackets(payloadBuf[:0], descriptorBuf[:0], spans[:], frame, limits, unit, state, 1, obuScratch[:], packetScratch[:], workScratch[:])
	})
	if allocs != 0 {
		t.Fatalf("scheduled picture RTP helper allocated: %f", allocs)
	}
}

func TestPublicRTPScheduledPictureFrameOBU(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 100}, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState key: %v", err)
	}
	payload := []byte{0xaa, 0xbb, 0xcc}
	size, wantControl, wantStructure, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(payload, unit, state, 1)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFrameOBUSize: %v", err)
	}
	if wantControl.Settings.SpatialID != 1 || wantControl.Settings.TemporalID != 0 || wantStructure.TemplateNum == 0 {
		t.Fatalf("control=%+v structure=%+v", wantControl, wantStructure)
	}
	var buf [16]byte
	out, gotControl, gotStructure, err := appendEncoderWebRTCPictureTemporalUnitFrameOBU(buf[:0], payload, unit, state, 1)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCPictureTemporalUnitFrameOBU: %v", err)
	}
	if len(out) != size || gotControl != wantControl || gotStructure != wantStructure {
		t.Fatalf("len=%d want=%d control=%+v/%+v structure=%+v/%+v", len(out), size, gotControl, wantControl, gotStructure, wantStructure)
	}
	parsed, consumed, err := internalobu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || parsed.Header.Type != internalobu.TypeFrame || !parsed.Header.Extension ||
		parsed.Header.TemporalID != wantControl.Settings.TemporalID || parsed.Header.SpatialID != wantControl.Settings.SpatialID ||
		string(parsed.Payload) != string(payload) {
		t.Fatalf("parsed=%+v consumed=%d payload=% x", parsed.Header, consumed, parsed.Payload)
	}
	dst := buf[:1]
	dst[0] = 0xee
	short, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameOBU(dst[:1:1], payload, unit, state, 1)
	if !errors.Is(err, internalbitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want %v", err, internalbitstream.ErrShortBuffer)
	}
	if len(short) != len(dst) || short[0] != 0xee {
		t.Fatalf("short buffer mutated out=% x", short)
	}
	if _, _, _, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(payload, unit, state, 2); err != ErrEncoderInvalidFrame {
		t.Fatalf("bad frame index err=%v want %v", err, ErrEncoderInvalidFrame)
	}
}

func TestPublicRTPScheduledPictureFramesOBU(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 40}, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState key: %v", err)
	}
	framePayloads := [][]byte{
		{0xaa, 0xbb},
		{0xcc, 0xdd, 0xee},
	}
	size, err := encoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads, unit, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFramesOBUSize: %v", err)
	}
	frame0, control0, _, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(framePayloads[0], unit, state, 0)
	if err != nil {
		t.Fatalf("frame0 size: %v", err)
	}
	frame1, control1, _, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(framePayloads[1], unit, state, 1)
	if err != nil {
		t.Fatalf("frame1 size: %v", err)
	}
	if size != frame0+frame1 || control0.Settings.SpatialID != 0 || control1.Settings.SpatialID != 1 {
		t.Fatalf("size=%d frame0=%d frame1=%d control0=%+v control1=%+v", size, frame0, frame1, control0, control1)
	}

	buf := make([]byte, 0, size)
	var spans [2]encoderWebRTCFrameOBUSpan
	out, frameCount, err := appendEncoderWebRTCPictureTemporalUnitFramesOBU(buf, spans[:], framePayloads, unit, state)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCPictureTemporalUnitFramesOBU: %v", err)
	}
	if len(out) != size || frameCount != 2 || spans[0].Offset != 0 || spans[0].Length != frame0 ||
		spans[1].Offset != frame0 || spans[1].Length != frame1 {
		t.Fatalf("len=%d/%d frameCount=%d spans=%+v", len(out), size, frameCount, spans)
	}
	for i, span := range spans {
		parsed, consumed, err := internalobu.ParseLowOverhead(out[span.Offset : span.Offset+span.Length])
		if err != nil {
			t.Fatalf("ParseLowOverheadOBU frame %d: %v", i, err)
		}
		if consumed != span.Length || parsed.Header.Type != internalobu.TypeFrame || parsed.Header.SpatialID != uint8(i) ||
			string(parsed.Payload) != string(framePayloads[i]) {
			t.Fatalf("frame %d parsed=%+v consumed=%d span=%+v payload=% x", i, parsed.Header, consumed, span, parsed.Payload)
		}
	}
	if _, err := encoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads[:1], unit, state); err != ErrEncoderInvalidFrame {
		t.Fatalf("short payload list size err=%v want %v", err, ErrEncoderInvalidFrame)
	}
	if _, _, err := appendEncoderWebRTCPictureTemporalUnitFramesOBU(buf[:0], spans[:1], framePayloads, unit, state); !errors.Is(err, internalbitstream.ErrShortBuffer) {
		t.Fatalf("short spans err=%v want %v", err, internalbitstream.ErrShortBuffer)
	}
	if out, _, err := appendEncoderWebRTCPictureTemporalUnitFramesOBU(buf[:0:1], spans[:], framePayloads, unit, state); !errors.Is(err, internalbitstream.ErrShortBuffer) || len(out) != 0 {
		t.Fatalf("short dst out=% x err=%v want %v", out, err, internalbitstream.ErrShortBuffer)
	}
}

func TestPublicRTPScheduledPictureFrameRTPPackets(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       EncoderScalabilityModeL1T1,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 20}, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState key: %v", err)
	}
	unit, state, err = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta: %v", err)
	}

	framePayload := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8}
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}
	frameOBUSize, _, _, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(framePayload, unit, state, 0)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFrameOBUSize: %v", err)
	}
	frameOBUScratch := make([]byte, 0, frameOBUSize)
	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [8]RTPPacketPlan
	var workScratch [8]RTPPacketPlan
	size, sizeControl, sizeStructure, err := encoderWebRTCPictureTemporalUnitFrameRTPPacketsSize(framePayload, limits, unit, state, 0, frameOBUScratch, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFrameRTPPacketsSize: %v", err)
	}
	if size.FrameOBUBytes != frameOBUSize || size.RTP.PacketCount < 2 || size.RTP.PayloadBytes == 0 || size.RTP.DescriptorBytes == 0 ||
		sizeControl.GenericFrameInfo.FrameID != 21 || sizeStructure.TemplateNum == 0 {
		t.Fatalf("size=%+v frameOBUSize=%d control=%+v structure=%+v", size, frameOBUSize, sizeControl, sizeStructure)
	}

	payloadBuf := make([]byte, 0, size.RTP.PayloadBytes)
	descriptorBuf := make([]byte, 0, size.RTP.DescriptorBytes)
	var spans [8]EncoderWebRTCRTPPacketSpan
	frameOBU, rtpPayloads, descriptors, packetCount, control, structure, err := appendEncoderWebRTCPictureTemporalUnitFrameRTPPackets(
		payloadBuf,
		descriptorBuf,
		spans[:],
		frameOBUScratch,
		framePayload,
		limits,
		unit,
		state,
		0,
		obuScratch[:],
		packetScratch[:],
		workScratch[:],
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCPictureTemporalUnitFrameRTPPackets: %v", err)
	}
	if len(frameOBU) != size.FrameOBUBytes || len(rtpPayloads) != size.RTP.PayloadBytes || len(descriptors) != size.RTP.DescriptorBytes ||
		packetCount != size.RTP.PacketCount || control != sizeControl || structure != sizeStructure {
		t.Fatalf("frameOBU=%d/%d payload=%d/%d descriptor=%d/%d packets=%d/%d", len(frameOBU), size.FrameOBUBytes, len(rtpPayloads), size.RTP.PayloadBytes, len(descriptors), size.RTP.DescriptorBytes, packetCount, size.RTP.PacketCount)
	}
	parsed, consumed, err := internalobu.ParseLowOverhead(frameOBU)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(frameOBU) || parsed.Header.Type != internalobu.TypeFrame || parsed.Header.TemporalID != control.Settings.TemporalID ||
		parsed.Header.SpatialID != control.Settings.SpatialID || string(parsed.Payload) != string(framePayload) {
		t.Fatalf("parsed=%+v consumed=%d payload=% x", parsed.Header, consumed, parsed.Payload)
	}

	rtpSlices := make([][]byte, packetCount)
	for i := 0; i < packetCount; i++ {
		span := spans[i]
		rtpSlices[i] = rtpPayloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
	}
	assembledLen, obuCount, err := AssembleRTPFrameSize(rtpSlices)
	if err != nil {
		t.Fatalf("AssembleRTPFrameSize: %v", err)
	}
	assembled := make([]byte, assembledLen)
	var obus [2]RTPFrameOBU
	wrote, count, err := AssembleRTPFrame(assembled, rtpSlices, obus[:obuCount])
	if err != nil {
		t.Fatalf("AssembleRTPFrame: %v", err)
	}
	if wrote != len(frameOBU) || count != 1 || string(assembled[:wrote]) != string(frameOBU) {
		t.Fatalf("assembled wrote=%d count=%d bytes=% x want=% x", wrote, count, assembled[:wrote], frameOBU)
	}
	if _, _, _, _, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameRTPPackets(payloadBuf[:0], descriptorBuf[:0], spans[:packetCount-1], frameOBUScratch[:0], framePayload, limits, unit, state, 0, obuScratch[:], packetScratch[:], workScratch[:]); err != ErrRTPPacketPlanTooSmall {
		t.Fatalf("short spans err=%v want %v", err, ErrRTPPacketPlanTooSmall)
	}
	if _, _, _, _, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameRTPPackets(payloadBuf[:0], descriptorBuf[:0], spans[:], frameOBUScratch[:0:1], framePayload, limits, unit, state, 0, obuScratch[:], packetScratch[:], workScratch[:]); !errors.Is(err, internalbitstream.ErrShortBuffer) {
		t.Fatalf("short frame scratch err=%v want %v", err, internalbitstream.ErrShortBuffer)
	}
}

func TestPublicRTPScheduledPictureFramesRTPPackets(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 30}, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState key: %v", err)
	}
	framePayloads := [][]byte{
		{0, 1, 2, 3, 4},
		{5, 6, 7, 8, 9, 10},
	}
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}
	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [8]RTPPacketPlan
	var workScratch [8]RTPPacketPlan
	var frameOBUScratch [64]byte
	partialScratch, err := encoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads, limits, unit, state, nil, nil)
	if !errors.Is(err, internalbitstream.ErrShortBuffer) || partialScratch.FrameOBUBytes == 0 || partialScratch.FrameSpans != 2 ||
		partialScratch.Packetizer != (RTPPacketizerScratchSize{}) {
		t.Fatalf("partial scratch=%+v err=%v", partialScratch, err)
	}
	countOnlyScratch, err := encoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads, limits, unit, state, frameOBUScratch[:0], nil)
	if err != nil {
		t.Fatalf("count-only scratch: %v", err)
	}
	if countOnlyScratch.FrameOBUBytes != partialScratch.FrameOBUBytes || countOnlyScratch.FrameSpans != 2 ||
		countOnlyScratch.Packetizer.OBUs != 1 || countOnlyScratch.Packetizer.Packets != 0 || countOnlyScratch.PacketSpans != 0 {
		t.Fatalf("count-only scratch=%+v partial=%+v", countOnlyScratch, partialScratch)
	}
	fullScratch, err := encoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads, limits, unit, state, frameOBUScratch[:0], obuScratch[:])
	if err != nil {
		t.Fatalf("full scratch: %v", err)
	}
	if fullScratch.FrameOBUBytes != partialScratch.FrameOBUBytes || fullScratch.FrameSpans != 2 ||
		fullScratch.Packetizer.OBUs != 1 || fullScratch.Packetizer.Packets == 0 ||
		fullScratch.PacketSpans == 0 || fullScratch.MaxPayloadBytes != limits.MaxPayloadLen || fullScratch.MaxDescriptorBytes == 0 {
		t.Fatalf("full scratch=%+v partial=%+v", fullScratch, partialScratch)
	}
	size, err := encoderWebRTCPictureTemporalUnitFramesRTPPacketsSize(framePayloads, limits, unit, state, frameOBUScratch[:0], obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFramesRTPPacketsSize: %v", err)
	}
	if size.FrameOBUBytes != fullScratch.FrameOBUBytes || size.RTP.PacketCount != fullScratch.PacketSpans ||
		size.RTP.PayloadBytes == 0 || size.RTP.DescriptorBytes == 0 {
		t.Fatalf("size=%+v fullScratch=%+v", size, fullScratch)
	}

	frameOBUBuf := make([]byte, fullScratch.FrameOBUBytes)
	payloadBuf := make([]byte, 0, size.RTP.PayloadBytes)
	descriptorBuf := make([]byte, 0, size.RTP.DescriptorBytes)
	frameSpanScratch := make([]encoderWebRTCFrameRTPPacketSpan, fullScratch.FrameSpans)
	packetSpanScratch := make([]EncoderWebRTCRTPPacketSpan, fullScratch.PacketSpans)
	packetScratchBound := make([]RTPPacketPlan, fullScratch.Packetizer.Packets)
	workScratchBound := make([]RTPPacketPlan, fullScratch.Packetizer.Work)
	bound, err := bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(fullScratch, encoderWebRTCPictureTemporalUnitFramesRTPScratch{
		FrameOBU:    frameOBUBuf,
		FrameSpans:  frameSpanScratch,
		PacketSpans: packetSpanScratch,
		OBUs:        obuScratch[:],
		Packets:     packetScratchBound,
		Work:        workScratchBound,
	})
	if err != nil {
		t.Fatalf("BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch: %v", err)
	}
	if len(bound.FrameOBU) != 0 || cap(bound.FrameOBU) < fullScratch.FrameOBUBytes ||
		len(bound.FrameSpans) != fullScratch.FrameSpans || len(bound.PacketSpans) != fullScratch.PacketSpans ||
		len(bound.OBUs) != fullScratch.Packetizer.OBUs || len(bound.Packets) != fullScratch.Packetizer.Packets ||
		len(bound.Work) != fullScratch.Packetizer.Work {
		t.Fatalf("bound scratch=%+v full=%+v", bound, fullScratch)
	}
	frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, err := appendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch(
		payloadBuf,
		descriptorBuf,
		bound,
		framePayloads,
		limits,
		unit,
		state,
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch: %v", err)
	}
	if frameCount != 2 || packetCount != size.RTP.PacketCount || len(frameOBUs) != size.FrameOBUBytes ||
		len(rtpPayloads) != size.RTP.PayloadBytes || len(descriptors) != size.RTP.DescriptorBytes {
		t.Fatalf("frames=%d packets=%d/%d frameOBUs=%d/%d rtp=%d/%d descriptors=%d/%d", frameCount, packetCount, size.RTP.PacketCount, len(frameOBUs), size.FrameOBUBytes, len(rtpPayloads), size.RTP.PayloadBytes, len(descriptors), size.RTP.DescriptorBytes)
	}
	for i := range framePayloads {
		span := bound.FrameSpans[i]
		if span.FrameOBULength == 0 || span.PacketCount == 0 {
			t.Fatalf("frame span[%d]=%+v", i, span)
		}
		frameOBU := frameOBUs[span.FrameOBUOffset : span.FrameOBUOffset+span.FrameOBULength]
		parsed, consumed, err := internalobu.ParseLowOverhead(frameOBU)
		if err != nil {
			t.Fatalf("ParseLowOverheadOBU frame %d: %v", i, err)
		}
		if consumed != len(frameOBU) || parsed.Header.Type != internalobu.TypeFrame || parsed.Header.SpatialID != uint8(i) ||
			string(parsed.Payload) != string(framePayloads[i]) {
			t.Fatalf("frame %d parsed=%+v consumed=%d payload=% x", i, parsed.Header, consumed, parsed.Payload)
		}

		rtpSlices := make([][]byte, span.PacketCount)
		for j := 0; j < span.PacketCount; j++ {
			packetSpan := bound.PacketSpans[span.PacketOffset+j]
			rtpSlices[j] = rtpPayloads[packetSpan.PayloadOffset : packetSpan.PayloadOffset+packetSpan.PayloadLength]
		}
		assembledLen, obuCount, err := AssembleRTPFrameSize(rtpSlices)
		if err != nil {
			t.Fatalf("AssembleRTPFrameSize frame %d: %v", i, err)
		}
		assembled := make([]byte, assembledLen)
		var obus [2]RTPFrameOBU
		wrote, count, err := AssembleRTPFrame(assembled, rtpSlices, obus[:obuCount])
		if err != nil {
			t.Fatalf("AssembleRTPFrame frame %d: %v", i, err)
		}
		if wrote != len(frameOBU) || count != 1 || string(assembled[:wrote]) != string(frameOBU) {
			t.Fatalf("frame %d assembled wrote=%d count=%d bytes=% x want=% x", i, wrote, count, assembled[:wrote], frameOBU)
		}
	}
	if _, err := encoderWebRTCPictureTemporalUnitFramesRTPPacketsSize(framePayloads[:1], limits, unit, state, frameOBUScratch[:0], obuScratch[:], packetScratch[:], workScratch[:]); err != ErrEncoderInvalidFrame {
		t.Fatalf("short frame payload list size err=%v want %v", err, ErrEncoderInvalidFrame)
	}
	if _, err := bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(fullScratch, encoderWebRTCPictureTemporalUnitFramesRTPScratch{FrameOBU: frameOBUBuf, FrameSpans: frameSpanScratch[:1], PacketSpans: packetSpanScratch, OBUs: obuScratch[:], Packets: packetScratchBound, Work: workScratchBound}); !errors.Is(err, internalbitstream.ErrShortBuffer) {
		t.Fatalf("short bind frame spans err=%v want %v", err, internalbitstream.ErrShortBuffer)
	}
	if _, err := bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(encoderWebRTCPictureTemporalUnitFramesRTPScratchSize{FrameOBUBytes: -1}, bound); !errors.Is(err, internalbitstream.ErrShortBuffer) {
		t.Fatalf("negative bind err=%v want %v", err, internalbitstream.ErrShortBuffer)
	}
	if _, _, _, _, _, err := appendEncoderWebRTCPictureTemporalUnitFramesRTPPackets(frameOBUBuf[:0], payloadBuf[:0], descriptorBuf[:0], bound.FrameSpans[:1], bound.PacketSpans, framePayloads, limits, unit, state, bound.OBUs, bound.Packets, bound.Work); err != ErrRTPPacketPlanTooSmall {
		t.Fatalf("short frame spans err=%v want %v", err, ErrRTPPacketPlanTooSmall)
	}
	shortScratch := bound
	shortScratch.FrameOBU = frameOBUBuf[:0:1]
	if _, _, _, _, _, err := appendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch(payloadBuf[:0], descriptorBuf[:0], shortScratch, framePayloads, limits, unit, state); !errors.Is(err, internalbitstream.ErrShortBuffer) {
		t.Fatalf("short frame OBU dst err=%v want %v", err, internalbitstream.ErrShortBuffer)
	}
}

func TestPublicRTPNextTemporalUnitFramesRTPPackets(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	state := internalencoder.WebRTCEncoderState{NextFrameID: 70}
	framePayloads := [][]byte{
		{0, 1, 2, 3, 4},
		{5, 6, 7, 8, 9, 10},
	}
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}
	var frameOBUScratch [64]byte
	var obuScratch [2]RTPPacketizerOBU
	var packetScratch [8]RTPPacketPlan
	var workScratch [8]RTPPacketPlan

	scratchSize, scheduled, next, err := encoderWebRTCNextTemporalUnitFramesRTPScratchLen(framePayloads, limits, cfg, state, false, frameOBUScratch[:0], obuScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitFramesRTPScratchLen: %v", err)
	}
	if !scheduled.Key || scheduled.Delta || next.NextFrameID != 72 || scratchSize.FrameSpans != 2 ||
		scratchSize.PacketSpans == 0 || scratchSize.MaxPayloadBytes != limits.MaxPayloadLen {
		t.Fatalf("scratch=%+v scheduled=%+v next=%+v", scratchSize, scheduled, next)
	}

	size, sizedUnit, sizedNext, err := encoderWebRTCNextTemporalUnitFramesRTPPacketsSize(framePayloads, limits, cfg, state, false, frameOBUScratch[:0], obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitFramesRTPPacketsSize: %v", err)
	}
	if sizedUnit != scheduled || sizedNext != next || size.FrameOBUBytes != scratchSize.FrameOBUBytes ||
		size.RTP.PacketCount != scratchSize.PacketSpans || size.RTP.PayloadBytes == 0 || size.RTP.DescriptorBytes == 0 {
		t.Fatalf("size=%+v scratch=%+v sizedUnit=%+v scheduled=%+v sizedNext=%+v next=%+v", size, scratchSize, sizedUnit, scheduled, sizedNext, next)
	}

	var frameSpans [2]encoderWebRTCFrameRTPPacketSpan
	var packetSpans [8]EncoderWebRTCRTPPacketSpan
	bound, err := bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(scratchSize, encoderWebRTCPictureTemporalUnitFramesRTPScratch{
		FrameOBU:    frameOBUScratch[:],
		FrameSpans:  frameSpans[:],
		PacketSpans: packetSpans[:],
		OBUs:        obuScratch[:],
		Packets:     packetScratch[:],
		Work:        workScratch[:],
	})
	if err != nil {
		t.Fatalf("BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch: %v", err)
	}
	var payloadBuf [64]byte
	var descriptorBuf [256]byte
	frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, appendedUnit, appendedNext, err := appendEncoderWebRTCNextTemporalUnitFramesRTPPacketsWithScratch(
		payloadBuf[:0],
		descriptorBuf[:0],
		bound,
		framePayloads,
		limits,
		cfg,
		state,
		false,
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCNextTemporalUnitFramesRTPPacketsWithScratch: %v", err)
	}
	if appendedUnit != scheduled || appendedNext != next || frameCount != 2 || packetCount != size.RTP.PacketCount ||
		len(frameOBUs) != size.FrameOBUBytes || len(rtpPayloads) != size.RTP.PayloadBytes ||
		len(descriptors) != size.RTP.DescriptorBytes {
		t.Fatalf("frames=%d packets=%d/%d frameOBUs=%d/%d rtp=%d/%d descriptors=%d/%d unit=%+v next=%+v", frameCount, packetCount, size.RTP.PacketCount, len(frameOBUs), size.FrameOBUBytes, len(rtpPayloads), size.RTP.PayloadBytes, len(descriptors), size.RTP.DescriptorBytes, appendedUnit, appendedNext)
	}
	for i, span := range bound.FrameSpans {
		if span.FrameOBULength == 0 || span.PacketCount == 0 {
			t.Fatalf("frame span[%d]=%+v", i, span)
		}
		parsed, consumed, err := internalobu.ParseLowOverhead(frameOBUs[span.FrameOBUOffset : span.FrameOBUOffset+span.FrameOBULength])
		if err != nil {
			t.Fatalf("ParseLowOverheadOBU frame %d: %v", i, err)
		}
		if consumed != span.FrameOBULength || parsed.Header.Type != internalobu.TypeFrame || parsed.Header.SpatialID != uint8(i) ||
			string(parsed.Payload) != string(framePayloads[i]) {
			t.Fatalf("frame %d parsed=%+v consumed=%d span=%+v payload=% x", i, parsed.Header, consumed, span, parsed.Payload)
		}
	}
}

func TestPublicRTPScheduledPictureAppendPackets(t *testing.T) {
	frame := appendTestLowOverheadOBU(nil, internalobu.TypeFrame, []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 6}
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       EncoderScalabilityModeL1T1,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 10}, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState key: %v", err)
	}
	unit, state, err = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta: %v", err)
	}
	var obuScratch [2]RTPPacketizerOBU
	scratch, err := encoderWebRTCPictureTemporalUnitRTPScratchLen(frame, limits, unit, state, 0, obuScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitRTPScratchLen: %v", err)
	}
	if scratch.Packetizer.Packets < 2 {
		t.Fatalf("scratch=%+v want fragmentation", scratch)
	}
	var packetScratch [8]RTPPacketPlan
	var workScratch [8]RTPPacketPlan
	size, sizeControl, sizeStructure, err := encoderWebRTCPictureTemporalUnitRTPPacketsSize(frame, limits, unit, state, 0, obuScratch[:], packetScratch[:], workScratch[:])
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitRTPPacketsSize: %v", err)
	}
	if size.PacketCount != scratch.Packetizer.Packets || size.PacketCount < 2 || size.PayloadBytes == 0 || size.DescriptorBytes == 0 ||
		sizeControl.GenericFrameInfo.FrameID != 11 || sizeStructure.TemplateNum == 0 {
		t.Fatalf("size=%+v scratch=%+v control=%+v structure=%+v", size, scratch, sizeControl, sizeStructure)
	}
	payloadBuf := make([]byte, 0, size.PayloadBytes)
	descriptorBuf := make([]byte, 0, size.DescriptorBytes)
	var spans [8]EncoderWebRTCRTPPacketSpan
	payloads, descriptors, packetCount, control, _, err := appendEncoderWebRTCPictureTemporalUnitRTPPackets(
		payloadBuf,
		descriptorBuf,
		spans[:],
		frame,
		limits,
		unit,
		state,
		0,
		obuScratch[:],
		packetScratch[:],
		workScratch[:],
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCPictureTemporalUnitRTPPackets: %v", err)
	}
	if packetCount != scratch.Packetizer.Packets || control.GenericFrameInfo.FrameID != 11 {
		t.Fatalf("packetCount=%d scratch=%+v control=%+v", packetCount, scratch, control)
	}
	if packetCount != size.PacketCount || len(payloads) != size.PayloadBytes || len(descriptors) != size.DescriptorBytes {
		t.Fatalf("appended packets=%d/%d payload=%d/%d descriptors=%d/%d", packetCount, size.PacketCount, len(payloads), size.PayloadBytes, len(descriptors), size.DescriptorBytes)
	}
	if len(payloads) == 0 || len(descriptors) == 0 || !spans[packetCount-1].Marker {
		t.Fatalf("payloads=%d descriptors=%d spans=%+v", len(payloads), len(descriptors), spans[:packetCount])
	}
	rtpPayloads := make([][]byte, packetCount)
	for i := 0; i < packetCount; i++ {
		span := spans[i]
		if span.PayloadLength == 0 || span.DescriptorLength == 0 {
			t.Fatalf("span[%d]=%+v", i, span)
		}
		if i != packetCount-1 && span.Marker {
			t.Fatalf("early marker span[%d]=%+v", i, span)
		}
		rtpPayloads[i] = payloads[span.PayloadOffset : span.PayloadOffset+span.PayloadLength]
	}
	assembledLen, obuCount, err := AssembleRTPFrameSize(rtpPayloads)
	if err != nil {
		t.Fatalf("AssembleRTPFrameSize: %v", err)
	}
	var assembled [32]byte
	var obus [2]RTPFrameOBU
	wrote, count, err := AssembleRTPFrame(assembled[:assembledLen], rtpPayloads, obus[:obuCount])
	if err != nil {
		t.Fatalf("AssembleRTPFrame: %v", err)
	}
	if wrote != len(frame) || count != 1 || string(assembled[:wrote]) != string(frame) {
		t.Fatalf("assembled wrote=%d count=%d bytes=% x want=% x", wrote, count, assembled[:wrote], frame)
	}
	if _, _, _, _, _, err := appendEncoderWebRTCPictureTemporalUnitRTPPackets(payloadBuf[:0], descriptorBuf[:0], spans[:packetCount-1], frame, limits, unit, state, 0, obuScratch[:], packetScratch[:], workScratch[:]); err != ErrRTPPacketPlanTooSmall {
		t.Fatalf("short spans err=%v want %v", err, ErrRTPPacketPlanTooSmall)
	}
}

func TestPublicRTPAssembleFrameOBUMetadata(t *testing.T) {
	payloads := [][]byte{{
		0x10,
		byte(internalobu.TypeTileGroup)<<3 | 0x04,
		(1 << 5) | (2 << 3),
		0xaa,
	}}

	var assembled [16]byte
	var obus [1]RTPFrameOBU
	wrote, count, err := AssembleRTPFrame(assembled[:], payloads, obus[:])
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{byte(internalobu.TypeTileGroup)<<3 | 0x06, (1 << 5) | (2 << 3), 0x01, 0xaa}
	if count != 1 || string(assembled[:wrote]) != string(want) {
		t.Fatalf("count=%d assembled=%x want=%x", count, assembled[:wrote], want)
	}
	if obus[0].Header.Type != internalobu.TypeTileGroup ||
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
		{Data: []byte{byte(internalobu.TypeSequenceHeader) << 3, 0xaa}},
		{Data: []byte{byte(internalobu.TypeFrameHeader) << 3, 0xbb}},
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
	frame = appendTestLowOverheadOBU(frame, internalobu.TypeSequenceHeader, []byte{0xaa})
	frame = appendTestLowOverheadOBU(frame, internalobu.TypeFrameHeader, []byte{0xbb})

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
