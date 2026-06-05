package goav1_test

import (
	"bytes"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicEncoderControlSurface(t *testing.T) {
	mode, ok := av1.ParseEncoderScalabilityMode("L2T2_KEY")
	if !ok || mode != av1.EncoderScalabilityModeL2T2_KEY {
		t.Fatalf("ParseEncoderScalabilityMode = %v,%v", mode, ok)
	}

	cfg, err := av1.SetWebRTCEncoderSVCConfig(av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		Scalability:       mode,
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
	}, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCEncoderSVCConfig: %v", err)
	}
	if cfg.SpatialLayerCount != 2 || cfg.TemporalLayerCount != 2 {
		t.Fatalf("layers = %d,%d; want 2,2", cfg.SpatialLayerCount, cfg.TemporalLayerCount)
	}
	if cfg.SpatialLayers[0].Resolution != (av1.EncoderResolution{Width: 320, Height: 180}) {
		t.Fatalf("base layer resolution = %+v", cfg.SpatialLayers[0].Resolution)
	}

	state := av1.EncoderReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = av1.EncoderResolution{Width: 320, Height: 180}
	frames := [...]av1.EncoderFrameEncodeSettings{
		{
			Type:             av1.EncoderFrameTypeDelta,
			Resolution:       av1.EncoderResolution{Width: 640, Height: 360},
			SpatialID:        1,
			ReferenceBuffers: [av1.EncoderWebRTCMaxReferences]uint8{0},
			ReferenceCount:   1,
			Output:           true,
		},
	}
	if _, err := av1.ValidateEncoderTemporalUnitFrames(frames[:], state, av1.EncoderRateControlCBR); err != nil {
		t.Fatalf("ValidateEncoderTemporalUnitFrames: %v", err)
	}

	refConfig, err := av1.EncoderLibaomSVCRefFrameConfigForFrame(frames[0])
	if err != nil {
		t.Fatalf("EncoderLibaomSVCRefFrameConfigForFrame: %v", err)
	}
	if refConfig.Reference[0] != 1 || refConfig.RefIdx[0] != 0 {
		t.Fatalf("libaom ref config = %+v", refConfig)
	}

	idState := av1.EncoderFrameIDBufferState{}
	idState.Valid[0] = true
	idState.FrameIDs[0] = 7
	info, nextIDState, err := av1.EncoderWebRTCGenericFrameInfoForFrame(frames[0], 8, idState, cfg.SpatialLayerCount, cfg.TemporalLayerCount)
	if err != nil {
		t.Fatalf("EncoderWebRTCGenericFrameInfoForFrame: %v", err)
	}
	if info.DependencyNum != 1 || info.Dependencies[0] != 7 || info.DTIs[1] != av1.EncoderDecodeTargetSwitch {
		t.Fatalf("generic frame info = %+v", info)
	}
	if nextIDState.Valid[0] != idState.Valid[0] || nextIDState.FrameIDs[0] != idState.FrameIDs[0] {
		t.Fatalf("unexpected id state mutation = %+v", nextIDState)
	}

	structure, err := av1.EncoderWebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if structure.NumDecodeTargets != 2 || structure.TemplateNum != 4 {
		t.Fatalf("dependency structure = %+v", structure)
	}

	control, err := av1.EncoderWebRTCTemporalUnitControlForFrames(cfg, frames[:], state, idState, 8)
	if err != nil {
		t.Fatalf("EncoderWebRTCTemporalUnitControlForFrames: %v", err)
	}
	if control.FrameNum != 1 || control.Frames[0].GenericFrameInfo.DependencyNum != 1 ||
		control.Frames[0].LibaomSVCRefFrameConfig.Reference[0] != 1 {
		t.Fatalf("temporal-unit control = %+v", control)
	}

	descriptor, err := av1.EncoderWebRTCRTPDependencyDescriptorMandatoryForFrame(structure, control.Frames[0].GenericFrameInfo, true, true)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTPDependencyDescriptorMandatoryForFrame: %v", err)
	}
	var descriptorBuf [av1.RTPDependencyDescriptorMandatorySize]byte
	n, err := av1.PutRTPDependencyDescriptorMandatory(descriptorBuf[:], descriptor)
	if err != nil {
		t.Fatalf("PutRTPDependencyDescriptorMandatory: %v", err)
	}
	parsed, consumed, err := av1.ParseRTPDependencyDescriptorMandatory(descriptorBuf[:])
	if err != nil {
		t.Fatalf("ParseRTPDependencyDescriptorMandatory: %v", err)
	}
	if n != av1.RTPDependencyDescriptorMandatorySize || consumed != n || parsed != descriptor {
		t.Fatalf("descriptor roundtrip n=%d consumed=%d parsed=%+v want=%+v", n, consumed, parsed, descriptor)
	}

	fullSize, err := av1.EncoderWebRTCDependencyDescriptorSize(structure, control.Frames[0].GenericFrameInfo, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSize: %v", err)
	}
	var fullBuf [16]byte
	full, err := av1.AppendEncoderWebRTCDependencyDescriptor(fullBuf[:0], structure, control.Frames[0].GenericFrameInfo, true, true, false)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	if len(full) != fullSize || full[0] != descriptorBuf[0] || full[1] != descriptorBuf[1] || full[2] != descriptorBuf[2] {
		t.Fatalf("full descriptor=% x size=%d mandatory=% x", full, fullSize, descriptorBuf)
	}
}

func TestPublicEncoderLowOverheadOBU(t *testing.T) {
	unit := av1.EncoderOBU{
		Type:       av1.OBUFrame,
		TemporalID: 1,
		SpatialID:  1,
		Payload:    []byte{0xaa, 0xbb},
	}
	size, err := av1.EncoderLowOverheadOBUSize(unit)
	if err != nil {
		t.Fatalf("EncoderLowOverheadOBUSize: %v", err)
	}
	var buf [8]byte
	out, err := av1.AppendEncoderLowOverheadOBU(buf[:0], unit)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("encoded len=%d want %d", len(out), size)
	}
	parsed, consumed, err := av1.ParseLowOverheadOBU(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || parsed.Header.Type != av1.OBUFrame || parsed.Header.TemporalID != 1 || parsed.Header.SpatialID != 1 {
		t.Fatalf("parsed header=%+v consumed=%d", parsed.Header, consumed)
	}
	if !bytes.Equal(parsed.Payload, unit.Payload) {
		t.Fatalf("payload=% x want % x", parsed.Payload, unit.Payload)
	}
}

func TestPublicEncoderSequenceHeaderForConfig(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution:  av1.EncoderResolution{Width: 640, Height: 360},
		Scalability: av1.EncoderScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	if seq.OperatingPointsCount != 4 || seq.MaxFrameWidth != 640 || seq.MaxFrameHeight != 360 {
		t.Fatalf("sequence header = %+v", seq)
	}
	var buf [160]byte
	out, err := av1.AppendEncoderLowOverheadSequenceHeaderOBU(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadSequenceHeaderOBU: %v", err)
	}
	unit, consumed, err := av1.ParseLowOverheadOBU(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != av1.OBUSequenceHeader {
		t.Fatalf("parsed header=%+v consumed=%d", unit.Header, consumed)
	}
	if _, err := av1.ParseSequenceHeader(unit.Payload); err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
}

func TestPublicEncoderFrameHeaderPrefixPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := av1.EncoderFrameHeaderPrefix{
		FrameType:               av1.EncoderFrameHeaderTypeKey,
		ShowFrame:               true,
		ShowableFrame:           false,
		ErrorResilientMode:      true,
		DisableCDFUpdate:        true,
		AllowScreenContentTools: false,
		ForceIntegerMV:          true,
		OrderHint:               5,
		PrimaryRefFrame:         av1.EncoderPrimaryRefNone,
	}
	size, err := av1.EncoderFrameHeaderPrefixPayloadSize(seq, prefix)
	if err != nil {
		t.Fatalf("EncoderFrameHeaderPrefixPayloadSize: %v", err)
	}
	var buf [8]byte
	out, err := av1.AppendEncoderFrameHeaderPrefixPayload(buf[:0], seq, prefix)
	if err != nil {
		t.Fatalf("AppendEncoderFrameHeaderPrefixPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("frame prefix len=%d want %d", len(out), size)
	}
	var seqBuf [128]byte
	seqPayload, err := av1.AppendEncoderSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := av1.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsed, err := av1.ParseFrameHeaderPrefix(out, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	if parsed.FrameType != av1.FrameTypeKey || !parsed.ShowFrame || parsed.OrderHint != 5 {
		t.Fatalf("parsed frame prefix=%+v", parsed)
	}
}

func TestPublicEncoderFrameHeaderIntraPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := av1.EncoderFrameHeaderPrefix{
		FrameType:          av1.EncoderFrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		OrderHint:          5,
		PrimaryRefFrame:    av1.EncoderPrimaryRefNone,
	}
	size := av1.EncoderIntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	payloadSize, err := av1.EncoderFrameHeaderIntraPayloadSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderFrameHeaderIntraPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := av1.AppendEncoderFrameHeaderIntraPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderFrameHeaderIntraPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	var seqBuf [128]byte
	seqPayload, err := av1.AppendEncoderSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := av1.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsedPrefix, err := av1.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := av1.ParseIntraFrameSize(payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != 640 || parsedSize.Height != 360 {
		t.Fatalf("parsed size=%+v", parsedSize)
	}
}

func TestPublicEncoderLowOverheadTemporalUnit(t *testing.T) {
	units := [...]av1.EncoderOBU{
		{Type: av1.OBUFrame, Payload: []byte{0xaa}},
	}
	size, err := av1.EncoderLowOverheadTemporalUnitSize(units[:])
	if err != nil {
		t.Fatalf("EncoderLowOverheadTemporalUnitSize: %v", err)
	}
	var buf [8]byte
	out, err := av1.AppendEncoderLowOverheadTemporalUnit(buf[:0], units[:])
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadTemporalUnit: %v", err)
	}
	if len(out) != size {
		t.Fatalf("temporal unit len=%d want %d", len(out), size)
	}
	it := av1.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil {
		t.Fatalf("TemporalUnitIterator.Next: %v", err)
	}
	if !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit parsed ok=%v raw=% x", ok, tu.Raw)
	}
}

func TestPublicEncoderLowOverheadIntraHeaderTemporalUnit(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := av1.EncoderFrameHeaderPrefix{
		FrameType:          av1.EncoderFrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		OrderHint:          5,
		PrimaryRefFrame:    av1.EncoderPrimaryRefNone,
	}
	size := av1.EncoderIntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}

	unitSize, err := av1.EncoderLowOverheadIntraHeaderTemporalUnitSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderLowOverheadIntraHeaderTemporalUnitSize: %v", err)
	}
	var buf [192]byte
	out, err := av1.AppendEncoderLowOverheadIntraHeaderTemporalUnit(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnit: %v", err)
	}
	if len(out) != unitSize {
		t.Fatalf("temporal unit len=%d want %d", len(out), unitSize)
	}

	it := av1.NewLowOverheadIterator(out)
	td, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("temporal delimiter ok=%v err=%v", ok, err)
	}
	if td.Header.Type != av1.OBUTemporalDelimiter || len(td.Payload) != 0 {
		t.Fatalf("temporal delimiter=%+v payload=% x", td.Header, td.Payload)
	}
	seqUnit, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("sequence header ok=%v err=%v", ok, err)
	}
	if seqUnit.Header.Type != av1.OBUSequenceHeader {
		t.Fatalf("sequence unit type=%v", seqUnit.Header.Type)
	}
	parsedSeq, err := av1.ParseSequenceHeader(seqUnit.Payload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	frameUnit, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("frame header ok=%v err=%v", ok, err)
	}
	if frameUnit.Header.Type != av1.OBUFrameHeader {
		t.Fatalf("frame unit type=%v", frameUnit.Header.Type)
	}
	parsedPrefix, err := av1.ParseFrameHeaderPrefix(frameUnit.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := av1.ParseIntraFrameSize(frameUnit.Payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != 640 || parsedSize.Height != 360 {
		t.Fatalf("parsed size=%+v", parsedSize)
	}
	if extra, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("extra unit ok=%v err=%v unit=%+v", ok, err, extra.Header)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = av1.AppendEncoderLowOverheadIntraHeaderTemporalUnit(buf[:0], seq, prefix, size)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnit allocated: %f", allocs)
	}
}

func TestPublicEncoderLowOverheadIntraHeaderTemporalUnitForConfig(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unitSize, wantUnit, err := av1.EncoderLowOverheadIntraHeaderTemporalUnitForConfigSize(cfg, 11)
	if err != nil {
		t.Fatalf("EncoderLowOverheadIntraHeaderTemporalUnitForConfigSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, err := av1.AppendEncoderLowOverheadIntraHeaderTemporalUnitForConfig(buf[:0], cfg, 11)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnitForConfig: %v", err)
	}
	if len(out) != unitSize || gotUnit != wantUnit {
		t.Fatalf("len=%d want %d gotUnit=%+v wantUnit=%+v", len(out), unitSize, gotUnit, wantUnit)
	}

	it := av1.NewLowOverheadIterator(out)
	td, ok, err := it.Next()
	if err != nil || !ok || td.Header.Type != av1.OBUTemporalDelimiter || len(td.Payload) != 0 {
		t.Fatalf("temporal delimiter ok=%v err=%v unit=%+v payload=% x", ok, err, td.Header, td.Payload)
	}
	seqUnit, ok, err := it.Next()
	if err != nil || !ok || seqUnit.Header.Type != av1.OBUSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v unit=%+v", ok, err, seqUnit.Header)
	}
	parsedSeq, err := av1.ParseSequenceHeader(seqUnit.Payload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if parsedSeq.OperatingPointsCount != 4 || parsedSeq.MaxFrameWidth != 640 || parsedSeq.MaxFrameHeight != 360 {
		t.Fatalf("parsed sequence=%+v", parsedSeq)
	}
	frameUnit, ok, err := it.Next()
	if err != nil || !ok || frameUnit.Header.Type != av1.OBUFrameHeader {
		t.Fatalf("frame ok=%v err=%v unit=%+v", ok, err, frameUnit.Header)
	}
	parsedPrefix, err := av1.ParseFrameHeaderPrefix(frameUnit.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := av1.ParseIntraFrameSize(frameUnit.Payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedPrefix.FrameType != av1.FrameTypeKey || !parsedPrefix.ShowFrame || parsedPrefix.OrderHint != 11 ||
		parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != 640 || parsedSize.Height != 360 {
		t.Fatalf("parsed prefix=%+v size=%+v", parsedPrefix, parsedSize)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = av1.AppendEncoderLowOverheadIntraHeaderTemporalUnitForConfig(buf[:0], cfg, 11)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestPublicEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unitSize, wantUnit, err := av1.EncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfigSize(cfg, 12, 300)
	if err != nil {
		t.Fatalf("EncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfigSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, err := av1.AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig(buf[:0], cfg, 12, 300)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	if len(out) != unitSize || gotUnit != wantUnit || gotUnit.FrameNum != 2 ||
		gotUnit.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 300 {
		t.Fatalf("len=%d want=%d got=%+v want=%+v", len(out), unitSize, gotUnit, wantUnit)
	}

	it := av1.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil || !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit ok=%v err=%v raw=% x", ok, err, tu.Raw)
	}
	descriptorSize, err := av1.EncoderWebRTCDependencyDescriptorSize(
		gotUnit.Control.DependencyStructure,
		gotUnit.Control.Frames[0].GenericFrameInfo,
		gotUnit.Control.Frames[0].AttachDependencyStructure,
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSize: %v", err)
	}
	var descriptorBuf [64]byte
	descriptor, err := av1.AppendEncoderWebRTCDependencyDescriptor(
		descriptorBuf[:0],
		gotUnit.Control.DependencyStructure,
		gotUnit.Control.Frames[0].GenericFrameInfo,
		true,
		gotUnit.FrameNum == 1,
		gotUnit.Control.Frames[0].AttachDependencyStructure,
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	if len(descriptor) != descriptorSize {
		t.Fatalf("descriptor len=%d want=%d", len(descriptor), descriptorSize)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = av1.AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig(buf[:0], cfg, 12, 300)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCDeltaFrameTemporalUnitForConfig(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := av1.EncoderWebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 400)
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := av1.EncoderWebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 500)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	if delta.FrameNum != 2 || delta.Control.Frames[1].GenericFrameInfo.DependencyNum != 2 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 401 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[1] != 500 {
		t.Fatalf("delta=%+v", delta)
	}

	structure, err := av1.EncoderWebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	size, err := av1.EncoderWebRTCDependencyDescriptorSize(structure, delta.Control.Frames[1].GenericFrameInfo, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSize: %v", err)
	}
	var descriptorBuf [16]byte
	descriptor, err := av1.AppendEncoderWebRTCDependencyDescriptor(
		descriptorBuf[:0],
		structure,
		delta.Control.Frames[1].GenericFrameInfo,
		true,
		true,
		false,
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	if len(descriptor) != size {
		t.Fatalf("descriptor len=%d want %d", len(descriptor), size)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = av1.EncoderWebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 500)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCDependencyStructureStateForTemporalUnit(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := av1.EncoderWebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 700)
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	state, structure, err := av1.EncoderWebRTCDependencyStructureStateForTemporalUnit(key.Control, av1.EncoderWebRTCDependencyStructureState{})
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyStructureStateForTemporalUnit key: %v", err)
	}
	if !state.Valid || structure != key.Control.DependencyStructure {
		t.Fatalf("key state=%+v structure=%+v control=%+v", state, structure, key.Control.DependencyStructure)
	}

	delta, err := av1.EncoderWebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 800)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	next, carried, err := av1.EncoderWebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyStructureStateForTemporalUnit delta: %v", err)
	}
	if !next.Valid || carried != state.Structure {
		t.Fatalf("delta state=%+v carried=%+v previous=%+v", next, carried, state.Structure)
	}

	var descriptorBuf [16]byte
	descriptor, err := av1.AppendEncoderWebRTCDependencyDescriptor(
		descriptorBuf[:0],
		carried,
		delta.Control.Frames[1].GenericFrameInfo,
		true,
		true,
		delta.Control.Frames[1].AttachDependencyStructure,
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	if len(descriptor) == 0 {
		t.Fatal("empty descriptor")
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = av1.EncoderWebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCDependencyStructureStateForTemporalUnit allocated: %f", allocs)
	}
}

func TestPublicEncoderSequenceHeaderOBU(t *testing.T) {
	seq := av1.EncoderSequenceHeader{
		Profile:               av1.EncoderProfile0,
		OperatingPointsCount:  1,
		MaxFrameWidth:         16,
		MaxFrameHeight:        9,
		EnableFilterIntra:     true,
		EnableIntraEdgeFilter: true,
		EnableOrderHint:       true,
		OrderHintBits:         7,
		EnableSuperRes:        true,
		EnableCDEF:            true,
		EnableRestoration:     true,
		ColorConfig: av1.EncoderSequenceColorConfig{
			BitDepth:                8,
			ColorDescriptionPresent: true,
			ColorPrimaries:          av1.EncoderSequenceColorPrimariesBT709,
			TransferCharacteristics: av1.EncoderSequenceTransferCharacteristicsSRGB,
			MatrixCoefficients:      av1.EncoderSequenceMatrixCoefficientsIdentity,
			ColorRange:              true,
		},
	}
	seq.OperatingPoints[0].SeqLevelIdx = 5

	size, err := av1.EncoderLowOverheadSequenceHeaderOBUSize(seq)
	if err != nil {
		t.Fatalf("EncoderLowOverheadSequenceHeaderOBUSize: %v", err)
	}
	var buf [80]byte
	out, err := av1.AppendEncoderLowOverheadSequenceHeaderOBU(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadSequenceHeaderOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("sequence obu len=%d want %d", len(out), size)
	}
	unit, consumed, err := av1.ParseLowOverheadOBU(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != av1.OBUSequenceHeader {
		t.Fatalf("parsed header=%+v consumed=%d", unit.Header, consumed)
	}
	if _, err := av1.ParseSequenceHeader(unit.Payload); err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
}
