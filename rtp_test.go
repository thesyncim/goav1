package goav1

import (
	"errors"
	"testing"

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
