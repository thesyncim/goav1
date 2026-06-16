package goav1_test

import (
	"bytes"
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicEncoderControlSurface(t *testing.T) {
	mode, ok := av1.ParseEncoderScalabilityMode("L2T2_KEY")
	if !ok || mode != av1.EncoderScalabilityModeL2T2_KEY {
		t.Fatalf("ParseEncoderScalabilityMode = %v,%v", mode, ok)
	}
	if shifted, ok := av1.ParseEncoderScalabilityMode("L3T3_KEY_SHIFT"); !ok || shifted != av1.EncoderScalabilityModeL3T3_KEY_SHIFT {
		t.Fatalf("ParseEncoderScalabilityMode L3T3_KEY_SHIFT = %v,%v", shifted, ok)
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
	if info.DependencyNum != 1 || info.Dependencies[0] != 7 || info.DTINum != 4 ||
		info.DTIs[2] != av1.EncoderDecodeTargetSwitch || info.DTIs[3] != av1.EncoderDecodeTargetSwitch {
		t.Fatalf("generic frame info = %+v", info)
	}
	if nextIDState.Valid[0] != idState.Valid[0] || nextIDState.FrameIDs[0] != idState.FrameIDs[0] {
		t.Fatalf("unexpected id state mutation = %+v", nextIDState)
	}

	structure, err := av1.EncoderWebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if structure.NumDecodeTargets != 4 || structure.TemplateNum != 4 {
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

func TestPublicEncoderTileGroupPayload(t *testing.T) {
	tiles := av1.EncoderTileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]av1.EncoderTilePayload{
		{Data: []byte{0x80}},
		{Data: []byte{0x81, 0x82}},
	}
	size, err := av1.EncoderTileGroupPayloadSize(tiles, 0, 1, payloads[:])
	if err != nil {
		t.Fatalf("EncoderTileGroupPayloadSize: %v", err)
	}
	var buf [8]byte
	out, err := av1.AppendEncoderTileGroupPayload(buf[:0], tiles, 0, 1, payloads[:])
	if err != nil {
		t.Fatalf("AppendEncoderTileGroupPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("payload len=%d want %d", len(out), size)
	}
}

func TestPublicEncoderFrameHeaderInterPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := av1.EncoderFrameHeaderPrefix{
		FrameType:          av1.EncoderFrameHeaderTypeInter,
		ShowFrame:          true,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		FrameSizeOverride:  true,
		OrderHint:          6,
		PrimaryRefFrame:    av1.EncoderPrimaryRefNone,
	}
	size := av1.EncoderInterFrameSize{
		UpscaledWidth:       320,
		Height:              180,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	payloadSize, err := av1.EncoderFrameHeaderInterPayloadSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderFrameHeaderInterPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := av1.AppendEncoderFrameHeaderInterPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderFrameHeaderInterPayload: %v", err)
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
	refs := publicEncoderReferenceState(seq, size.RefFrameIdx[:])
	parsedSize, err := av1.ParseFrameSize(payload, parsedSeq, parsedPrefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	if parsedSize.RefreshFrameFlags != 0x01 || parsedSize.UpscaledWidth != 320 || parsedSize.Height != 180 {
		t.Fatalf("parsed inter size=%+v", parsedSize)
	}

	obuSize, err := av1.EncoderLowOverheadFrameHeaderInterOBUSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderLowOverheadFrameHeaderInterOBUSize: %v", err)
	}
	var obuBuf [40]byte
	obuOut, err := av1.AppendEncoderLowOverheadFrameHeaderInterOBU(obuBuf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadFrameHeaderInterOBU: %v", err)
	}
	if len(obuOut) != obuSize {
		t.Fatalf("inter frame-header obu len=%d want %d", len(obuOut), obuSize)
	}
}

func TestPublicEncoderQuantizationParamsPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.ColorConfig.SeparateUVDeltaQ = true
	quant := av1.EncoderQuantizationParams{
		BaseQIdx:      37,
		YDCDelta:      -2,
		UDCDelta:      5,
		UACDelta:      -3,
		VDCDelta:      7,
		VACDelta:      -9,
		DiffUVDeltas:  true,
		UsingQMatrix:  true,
		QMatrixLevelY: 2,
		QMatrixLevelU: 3,
		QMatrixLevelV: 4,
	}
	payloadSize, err := av1.EncoderQuantizationParamsPayloadSize(seq, quant)
	if err != nil {
		t.Fatalf("EncoderQuantizationParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := av1.AppendEncoderQuantizationParamsPayload(buf[:0], seq, quant)
	if err != nil {
		t.Fatalf("AppendEncoderQuantizationParamsPayload: %v", err)
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
	parsed, err := av1.ParseQuantizationParams(payload, parsedSeq, av1.TileInfo{})
	if err != nil {
		t.Fatalf("ParseQuantizationParams: %v", err)
	}
	if parsed.BaseQIdx != quant.BaseQIdx || parsed.YDCDelta != quant.YDCDelta ||
		parsed.UDCDelta != quant.UDCDelta || parsed.UACDelta != quant.UACDelta ||
		parsed.VDCDelta != quant.VDCDelta || parsed.VACDelta != quant.VACDelta {
		t.Fatalf("parsed quant=%+v want %+v", parsed, quant)
	}
	if !parsed.UsingQMatrix || parsed.QMatrixLevelY != 2 || parsed.QMatrixLevelU != 3 || parsed.QMatrixLevelV != 4 {
		t.Fatalf("parsed qmatrix=%+v", parsed)
	}
}

func TestPublicEncoderSegmentationParamsPayload(t *testing.T) {
	prefix := av1.EncoderFrameHeaderPrefix{PrimaryRefFrame: av1.EncoderPrimaryRefNone}
	var data av1.EncoderSegmentationData
	for i := 0; i < 8; i++ {
		data.Segments[i].RefFrame = -1
	}
	data.Segments[0].DeltaQ = -8
	data.Segments[2].RefFrame = 4
	seg := av1.EncoderSegmentationParams{
		Enabled:    true,
		UpdateMap:  true,
		UpdateData: true,
		Data:       data,
	}
	payloadSize, err := av1.EncoderSegmentationParamsPayloadSize(prefix, seg)
	if err != nil {
		t.Fatalf("EncoderSegmentationParamsPayloadSize: %v", err)
	}
	var buf [96]byte
	payload, err := av1.AppendEncoderSegmentationParamsPayload(buf[:0], prefix, seg)
	if err != nil {
		t.Fatalf("AppendEncoderSegmentationParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseSegmentationParams(
		payload,
		av1.FrameHeaderPrefix{PrimaryRefFrame: av1.PrimaryRefNone},
		av1.QuantizationParams{BaseQIdx: 20},
		nil,
	)
	if err != nil {
		t.Fatalf("ParseSegmentationParams: %v", err)
	}
	if !parsed.Enabled || !parsed.UpdateMap || !parsed.UpdateData ||
		parsed.Data.Segments[0].DeltaQ != -8 || parsed.QIndex[0] != 12 ||
		parsed.Data.Segments[2].RefFrame != 4 {
		t.Fatalf("parsed segmentation=%+v", parsed)
	}
}

func TestPublicEncoderDeltaParamsPayload(t *testing.T) {
	size := av1.EncoderIntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	quant := av1.EncoderQuantizationParams{BaseQIdx: 37}
	delta := av1.EncoderDeltaParams{
		DeltaQPresent:  true,
		DeltaQResLog2:  1,
		DeltaLFPresent: true,
		DeltaLFResLog2: 2,
		DeltaLFMulti:   true,
	}
	payloadSize, err := av1.EncoderDeltaParamsPayloadSize(size, quant, delta)
	if err != nil {
		t.Fatalf("EncoderDeltaParamsPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := av1.AppendEncoderDeltaParamsPayload(buf[:0], size, quant, delta)
	if err != nil {
		t.Fatalf("AppendEncoderDeltaParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseDeltaParams(
		payload,
		av1.FrameSize{},
		av1.QuantizationParams{BaseQIdx: quant.BaseQIdx},
		av1.SegmentationParams{},
	)
	if err != nil {
		t.Fatalf("ParseDeltaParams: %v", err)
	}
	if !parsed.DeltaQPresent || parsed.DeltaQResLog2 != 1 ||
		!parsed.DeltaLFPresent || parsed.DeltaLFResLog2 != 2 || !parsed.DeltaLFMulti {
		t.Fatalf("parsed delta=%+v", parsed)
	}
}

func TestPublicEncoderLoopFilterParamsPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := av1.EncoderFrameHeaderPrefix{PrimaryRefFrame: av1.EncoderPrimaryRefNone}
	size := av1.EncoderIntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	lf := av1.EncoderLoopFilterParams{
		LevelY:              [2]uint8{10, 0},
		LevelU:              5,
		LevelV:              6,
		Sharpness:           3,
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  true,
		Deltas: av1.EncoderLoopFilterDeltas{
			Ref:  [8]int8{-2, 0, 0, 0, -1, 0, -1, -1},
			Mode: [2]int8{0, 3},
		},
	}
	payloadSize, err := av1.EncoderLoopFilterParamsPayloadSize(seq, prefix, size, false, lf, nil)
	if err != nil {
		t.Fatalf("EncoderLoopFilterParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := av1.AppendEncoderLoopFilterParamsPayload(buf[:0], seq, prefix, size, false, lf, nil)
	if err != nil {
		t.Fatalf("AppendEncoderLoopFilterParamsPayload: %v", err)
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
	parsed, err := av1.ParseLoopFilterParams(
		payload,
		parsedSeq,
		av1.FrameHeaderPrefix{PrimaryRefFrame: av1.PrimaryRefNone},
		av1.FrameSize{},
		av1.SegmentationParams{},
		av1.DeltaParams{},
		nil,
	)
	if err != nil {
		t.Fatalf("ParseLoopFilterParams: %v", err)
	}
	if parsed.LevelY != [2]uint8{10, 0} || parsed.LevelU != 5 || parsed.LevelV != 6 ||
		parsed.Sharpness != 3 || parsed.Deltas.Ref[0] != -2 || parsed.Deltas.Mode[1] != 3 {
		t.Fatalf("parsed loopfilter=%+v", parsed)
	}
}

func TestPublicEncoderCDEFParamsPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	size := av1.EncoderIntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	cdef := av1.EncoderCDEFParams{Damping: 5, Bits: 2}
	for i := uint8(0); i < 4; i++ {
		cdef.YStrength[i] = 10 + i
		cdef.UVStrength[i] = 20 + i
	}
	payloadSize, err := av1.EncoderCDEFParamsPayloadSize(seq, size, false, cdef)
	if err != nil {
		t.Fatalf("EncoderCDEFParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := av1.AppendEncoderCDEFParamsPayload(buf[:0], seq, size, false, cdef)
	if err != nil {
		t.Fatalf("AppendEncoderCDEFParamsPayload: %v", err)
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
	parsed, err := av1.ParseCDEFParams(payload, parsedSeq, av1.FrameSize{}, av1.SegmentationParams{}, av1.LoopFilterParams{})
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	if parsed.Damping != 5 || parsed.Bits != 2 || parsed.StrengthCount != 4 ||
		parsed.YStrength[3] != 13 || parsed.UVStrength[3] != 23 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
}

func TestPublicEncoderRestorationParamsPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.EnableRestoration = true
	size := av1.EncoderIntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	restoration := av1.EncoderRestorationParams{
		Type:           [3]av1.EncoderRestorationType{av1.EncoderRestorationWiener, av1.EncoderRestorationSGRProj, av1.EncoderRestorationNone},
		UnitSizeYLog2:  7,
		UnitSizeUVLog2: 6,
		UnitSizeY:      128,
		UnitSizeUV:     64,
	}
	payloadSize, err := av1.EncoderRestorationParamsPayloadSize(seq, size, false, restoration)
	if err != nil {
		t.Fatalf("EncoderRestorationParamsPayloadSize: %v", err)
	}
	var buf [8]byte
	payload, err := av1.AppendEncoderRestorationParamsPayload(buf[:0], seq, size, false, restoration)
	if err != nil {
		t.Fatalf("AppendEncoderRestorationParamsPayload: %v", err)
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
	parsed, err := av1.ParseRestorationParams(payload, parsedSeq, av1.FrameSize{}, av1.SegmentationParams{}, av1.CDEFParams{})
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	if parsed.Type[0] != av1.RestorationWiener || parsed.Type[1] != av1.RestorationSGRProj ||
		parsed.UnitSizeY != 128 || parsed.UnitSizeUV != 64 {
		t.Fatalf("parsed restoration=%+v", parsed)
	}
}

func TestPublicEncoderTransformReferenceParamsPayload(t *testing.T) {
	prefix := av1.EncoderFrameHeaderPrefix{FrameType: av1.EncoderFrameHeaderTypeInter}
	params := av1.EncoderTransformReferenceParams{
		TransformMode: av1.EncoderTransformModeSwitchable,
		ReferenceMode: av1.EncoderReferenceModeSelect,
	}
	payloadSize, err := av1.EncoderTransformReferenceParamsPayloadSize(prefix, false, params)
	if err != nil {
		t.Fatalf("EncoderTransformReferenceParamsPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := av1.AppendEncoderTransformReferenceParamsPayload(buf[:0], prefix, false, params)
	if err != nil {
		t.Fatalf("AppendEncoderTransformReferenceParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseTransformReferenceParams(
		payload,
		av1.FrameHeaderPrefix{FrameType: av1.FrameTypeInter},
		av1.SegmentationParams{},
		av1.RestorationParams{},
	)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	if parsed.TransformMode != av1.TransformModeSwitchable || parsed.ReferenceMode != av1.ReferenceModeSelect {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestPublicEncoderSkipModeParamsPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.EnableOrderHint = true
	seq.OrderHintBits = 5
	prefix := av1.EncoderFrameHeaderPrefix{FrameType: av1.EncoderFrameHeaderTypeInter, OrderHint: 16}
	var size av1.EncoderInterFrameSize
	var refs av1.ReferenceState
	orderHints := [7]uint8{15, 17, 14, 18, 13, 19, 12}
	for i := uint8(0); i < av1.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = av1.ReferenceFrame{Valid: true, OrderHint: orderHints[i]}
	}
	transformRef := av1.EncoderTransformReferenceParams{ReferenceMode: av1.EncoderReferenceModeSelect}
	params := av1.EncoderSkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{0, 1}}
	payloadSize, err := av1.EncoderSkipModeParamsPayloadSize(seq, prefix, size, &refs, transformRef, params)
	if err != nil {
		t.Fatalf("EncoderSkipModeParamsPayloadSize: %v", err)
	}
	var buf [1]byte
	payload, err := av1.AppendEncoderSkipModeParamsPayload(buf[:0], seq, prefix, size, &refs, transformRef, params)
	if err != nil {
		t.Fatalf("AppendEncoderSkipModeParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseSkipModeParams(
		payload,
		av1.SequenceHeader{EnableOrderHint: true, OrderHintBits: 5},
		av1.FrameHeaderPrefix{FrameType: av1.FrameTypeInter, OrderHint: 16},
		av1.FrameSize{RefFrameIdx: size.RefFrameIdx},
		&refs,
		av1.TransformReferenceParams{ReferenceMode: av1.ReferenceModeSelect},
	)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	if !parsed.Allowed || !parsed.Enabled || parsed.RefFrameIdx != [2]uint8{0, 1} {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestPublicEncoderFrameModeParamsPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.EnableWarpedMotion = true
	prefix := av1.EncoderFrameHeaderPrefix{FrameType: av1.EncoderFrameHeaderTypeInter}
	params := av1.EncoderFrameModeParams{AllowWarpedMotion: true, ReducedTxSet: true}
	payloadSize, err := av1.EncoderFrameModeParamsPayloadSize(seq, prefix, params)
	if err != nil {
		t.Fatalf("EncoderFrameModeParamsPayloadSize: %v", err)
	}
	var buf [1]byte
	payload, err := av1.AppendEncoderFrameModeParamsPayload(buf[:0], seq, prefix, params)
	if err != nil {
		t.Fatalf("AppendEncoderFrameModeParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseFrameModeParams(
		payload,
		av1.SequenceHeader{EnableWarpedMotion: true},
		av1.FrameHeaderPrefix{FrameType: av1.FrameTypeInter},
		av1.SkipModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	if !parsed.AllowWarpedMotion || !parsed.ReducedTxSet {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestPublicEncoderGlobalMotionParamsPayload(t *testing.T) {
	prefix := av1.EncoderFrameHeaderPrefix{FrameType: av1.EncoderFrameHeaderTypeInter, PrimaryRefFrame: av1.EncoderPrimaryRefNone}
	var size av1.EncoderInterFrameSize
	var refs av1.ReferenceState
	for i := uint8(0); i < av1.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = av1.ReferenceFrame{Valid: true}
	}
	params := av1.EncoderDefaultGlobalMotionParams()
	params.Ref[0] = av1.EncoderDefaultWarpedMotionParams()
	params.Ref[0].Type = av1.EncoderGlobalMotionTranslation
	params.Ref[0].Matrix[0] = 2 << 13
	payloadSize, err := av1.EncoderGlobalMotionParamsPayloadSize(prefix, size, av1.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	if err != nil {
		t.Fatalf("EncoderGlobalMotionParamsPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := av1.AppendEncoderGlobalMotionParamsPayload(buf[:0], prefix, size, av1.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	if err != nil {
		t.Fatalf("AppendEncoderGlobalMotionParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseGlobalMotionParams(
		payload,
		av1.FrameHeaderPrefix{FrameType: av1.FrameTypeInter, PrimaryRefFrame: av1.PrimaryRefNone},
		av1.FrameSize{RefFrameIdx: size.RefFrameIdx},
		av1.TileInfo{AllowHighPrecisionMV: true},
		&refs,
		av1.FrameModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	if parsed.Ref[0].Type != av1.GlobalMotionTranslation || parsed.Ref[0].Matrix[0] != params.Ref[0].Matrix[0] {
		t.Fatalf("parsed global motion ref[0]=%+v", parsed.Ref[0])
	}
}

func TestPublicEncoderFilmGrainParamsPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.FilmGrainParamsPresent = true
	prefix := av1.EncoderFrameHeaderPrefix{FrameType: av1.EncoderFrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	params := av1.EncoderFilmGrainParams{
		Apply:           true,
		Update:          true,
		Seed:            0x1234,
		NumYPoints:      1,
		ScalingShift:    9,
		ARCoeffShift:    8,
		GrainScaleShift: 3,
		Overlap:         true,
	}
	params.YPoints[0] = [2]uint8{10, 20}
	payloadSize, err := av1.EncoderFilmGrainParamsPayloadSize(seq, prefix, av1.EncoderInterFrameSize{}, nil, params)
	if err != nil {
		t.Fatalf("EncoderFilmGrainParamsPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := av1.AppendEncoderFilmGrainParamsPayload(buf[:0], seq, prefix, av1.EncoderInterFrameSize{}, nil, params)
	if err != nil {
		t.Fatalf("AppendEncoderFilmGrainParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseFilmGrainParams(
		payload,
		av1.SequenceHeader{
			FilmGrainParamsPresent: true,
			ColorConfig: av1.ColorConfig{
				BitDepth:     8,
				SubsamplingX: true,
				SubsamplingY: true,
			},
		},
		av1.FrameHeaderPrefix{FrameType: av1.FrameTypeKey, ShowFrame: true},
		av1.FrameSize{},
		nil,
		av1.GlobalMotionParams{},
	)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	if !parsed.Apply || !parsed.Update || parsed.Seed != 0x1234 || parsed.YPoints[0] != [2]uint8{10, 20} {
		t.Fatalf("parsed film grain=%+v", parsed)
	}
}

func TestPublicEncoderTileInfoPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := av1.EncoderFrameHeaderPrefix{FrameType: av1.EncoderFrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := av1.EncoderTileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [av1.EncoderMaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [av1.EncoderMaxTileRows + 1]uint16{0, 1},
	}
	payloadSize, err := av1.EncoderTileInfoPayloadSize(seq, prefix, 64, 64, tiles)
	if err != nil {
		t.Fatalf("EncoderTileInfoPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := av1.AppendEncoderTileInfoPayload(buf[:0], seq, prefix, 64, 64, tiles)
	if err != nil {
		t.Fatalf("AppendEncoderTileInfoPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := av1.ParseTileInfo(
		payload,
		av1.SequenceHeader{},
		av1.FrameHeaderPrefix{FrameType: av1.FrameTypeKey, ErrorResilientMode: true},
		av1.FrameSize{CodedWidth: 64, Height: 64},
	)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	if !parsed.RefreshContext || parsed.Cols != 1 || parsed.Rows != 1 {
		t.Fatalf("parsed tiles=%+v", parsed)
	}
}

func TestPublicEncoderIntraFrameHeaderPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.SeqForceScreenContentTools = 1
	seq.SeqForceIntegerMV = 1
	seq.EnableCDEF = true
	seq.EnableRestoration = false
	header := av1.EncoderIntraFrameHeaderParams{
		Prefix: av1.EncoderFrameHeaderPrefix{
			FrameType:               av1.EncoderFrameHeaderTypeKey,
			ShowFrame:               true,
			ErrorResilientMode:      true,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         av1.EncoderPrimaryRefNone,
		},
		Size: av1.EncoderIntraFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		Tile: av1.EncoderTileInfo{
			RefreshContext: true,
			SBCols:         1,
			SBRows:         1,
			Cols:           1,
			Rows:           1,
			ColStartSB:     [av1.EncoderMaxTileCols + 1]uint16{0, 1},
			RowStartSB:     [av1.EncoderMaxTileRows + 1]uint16{0, 1},
		},
		Quantization: av1.EncoderQuantizationParams{BaseQIdx: 50},
		LoopFilter: av1.EncoderLoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			ModeRefDeltaEnabled: false,
			Deltas: av1.EncoderLoopFilterDeltas{
				Ref: [8]int8{1, 0, 0, 0, -1, 0, -1, -1},
			},
		},
		CDEF: av1.EncoderCDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		TransformRef: av1.EncoderTransformReferenceParams{
			TransformMode: av1.EncoderTransformModeLargest,
			ReferenceMode: av1.EncoderReferenceModeSingle,
		},
	}
	payloadSize, err := av1.EncoderIntraFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("EncoderIntraFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := av1.AppendEncoderIntraFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("AppendEncoderIntraFrameHeaderPayload: %v", err)
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
	prefix, err := av1.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := av1.ParseIntraFrameSize(payload, parsedSeq, prefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	tiles, err := av1.ParseTileInfo(payload, parsedSeq, prefix, size)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	if prefix.FrameType != av1.FrameTypeKey || size.CodedWidth != 64 || tiles.Cols != 1 || tiles.Rows != 1 {
		t.Fatalf("parsed header prefix=%+v size=%+v tiles=%+v", prefix, size, tiles)
	}
}

func TestPublicEncoderInterFrameHeaderPayload(t *testing.T) {
	seq, err := av1.EncoderSequenceHeaderForConfig(av1.EncoderConfig{
		Resolution: av1.EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.SeqForceScreenContentTools = 1
	seq.SeqForceIntegerMV = 1
	seq.EnableCDEF = true
	var refs av1.ReferenceState
	for i := uint8(0); i < av1.RefFrames; i++ {
		refs.Frames[i] = av1.ReferenceFrame{
			Valid: true,
			Size: av1.FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				RenderWidth:         64,
				RenderHeight:        64,
				SuperResDenominator: 8,
			},
		}
	}
	header := av1.EncoderInterFrameHeaderParams{
		Prefix: av1.EncoderFrameHeaderPrefix{
			FrameType:               av1.EncoderFrameHeaderTypeInter,
			ShowFrame:               true,
			ShowableFrame:           true,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         av1.EncoderPrimaryRefNone,
		},
		Size: av1.EncoderInterFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0x02,
		},
		Tile: av1.EncoderTileInfo{
			InterpolationFilter: av1.EncoderInterpolationEightTap,
			RefreshContext:      true,
			SBCols:              1,
			SBRows:              1,
			Cols:                1,
			Rows:                1,
			ColStartSB:          [av1.EncoderMaxTileCols + 1]uint16{0, 1},
			RowStartSB:          [av1.EncoderMaxTileRows + 1]uint16{0, 1},
		},
		Quantization: av1.EncoderQuantizationParams{BaseQIdx: 50},
		LoopFilter: av1.EncoderLoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			ModeRefDeltaEnabled: false,
			Deltas: av1.EncoderLoopFilterDeltas{
				Ref: [8]int8{1, 0, 0, 0, -1, 0, -1, -1},
			},
		},
		CDEF: av1.EncoderCDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		TransformRef: av1.EncoderTransformReferenceParams{
			TransformMode: av1.EncoderTransformModeLargest,
			ReferenceMode: av1.EncoderReferenceModeSingle,
		},
		GlobalMotion: av1.EncoderDefaultGlobalMotionParams(),
		References:   &refs,
	}
	payloadSize, err := av1.EncoderInterFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("EncoderInterFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := av1.AppendEncoderInterFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("AppendEncoderInterFrameHeaderPayload: %v", err)
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
	prefix, err := av1.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := av1.ParseFrameSize(payload, parsedSeq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	if prefix.FrameType != av1.FrameTypeInter || size.RefreshFrameFlags != 0x02 || size.RefFrameIdx[0] != 0 {
		t.Fatalf("parsed inter header prefix=%+v size=%+v", prefix, size)
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

func TestPublicEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	state := av1.EncoderWebRTCState{NextOrderHint: 13, NextFrameID: 700}
	unitSize, wantUnit, wantState, err := av1.EncoderLowOverheadWebRTCKeyFrameTemporalUnitForStateSize(cfg, state)
	if err != nil {
		t.Fatalf("EncoderLowOverheadWebRTCKeyFrameTemporalUnitForStateSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, gotState, err := av1.AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState(buf[:0], cfg, state)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if len(out) != unitSize || gotUnit != wantUnit || gotState != wantState ||
		gotUnit.Header.Prefix.OrderHint != 13 || gotState.NextFrameID != 702 ||
		!gotState.DependencyStructureState.Valid {
		t.Fatalf("len=%d want=%d unit=%+v wantUnit=%+v state=%+v wantState=%+v", len(out), unitSize, gotUnit, wantUnit, gotState, wantState)
	}
	it := av1.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil || !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit ok=%v err=%v raw=% x", ok, err, tu.Raw)
	}
	control, structure, err := av1.EncoderWebRTCPictureTemporalUnitFrameControl(
		av1.EncoderWebRTCPictureTemporalUnit{Key: true, KeyUnit: gotUnit},
		gotState,
		0,
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFrameControl: %v", err)
	}
	var descriptorBuf [64]byte
	descriptor, err := av1.AppendEncoderWebRTCDependencyDescriptor(
		descriptorBuf[:0],
		structure,
		control.GenericFrameInfo,
		true,
		false,
		control.AttachDependencyStructure,
	)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	if len(descriptor) <= av1.RTPDependencyDescriptorMandatorySize {
		t.Fatalf("descriptor too small: % x", descriptor)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, _ = av1.AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState(buf[:0], cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState allocated: %f", allocs)
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
	if delta.Headers[1].Prefix.FrameType != av1.EncoderFrameHeaderTypeInter ||
		delta.Headers[1].Size.UpscaledWidth != 640 ||
		delta.Headers[1].Size.Height != 360 ||
		delta.Headers[1].Size.RefreshFrameFlags != 0x00 ||
		delta.Headers[1].Size.RefFrameIdx[0] != 1 ||
		delta.Headers[1].Size.RefFrameIdx[1] != 2 {
		t.Fatalf("delta header=%+v", delta.Headers[1])
	}
	var headerBuf [80]byte
	headerSize, err := av1.EncoderLowOverheadInterHeaderFrameOBUSize(delta.Headers[1])
	if err != nil {
		t.Fatalf("EncoderLowOverheadInterHeaderFrameOBUSize: %v", err)
	}
	headerOBU, err := av1.AppendEncoderLowOverheadInterHeaderFrameOBU(headerBuf[:0], delta.Headers[1])
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadInterHeaderFrameOBU: %v", err)
	}
	unit, consumed, err := av1.ParseLowOverheadOBU(headerOBU)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if len(headerOBU) != headerSize || consumed != len(headerOBU) ||
		unit.Header.Type != av1.OBUFrameHeader ||
		unit.Header.TemporalID != delta.Headers[1].TemporalID ||
		unit.Header.SpatialID != delta.Headers[1].SpatialID {
		t.Fatalf("header obu=%+v consumed=%d len=%d wantSize=%d", unit.Header, consumed, len(headerOBU), headerSize)
	}

	ordered, err := av1.EncoderWebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 500, 13)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	if ordered.Headers[0].Prefix.OrderHint != 13 {
		t.Fatalf("ordered header hint=%d want 13", ordered.Headers[0].Prefix.OrderHint)
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

func TestPublicEncoderWebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
		Scalability:       av1.EncoderScalabilityModeL3T3_KEY_SHIFT,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	key, err := av1.EncoderWebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 10)
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := av1.EncoderWebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 2, 20, 7)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex: %v", err)
	}
	if delta.FrameNum != 3 ||
		delta.Frames[0].TemporalID != 2 ||
		delta.Frames[1].TemporalID != 0 ||
		delta.Frames[2].TemporalID != 1 ||
		delta.Headers[0].Prefix.OrderHint != 7 {
		t.Fatalf("delta=%+v", delta)
	}
}

func TestPublicEncoderWebRTCQuantizerPropagation(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		RateControl:       av1.EncoderRateControlCQP,
		Quantizer:         42,
	}
	key, state, err := av1.EncoderWebRTCKeyFrameTemporalUnitForState(cfg, av1.EncoderWebRTCState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.Frames[0].RateControl != av1.EncoderRateControlCQP || key.Frames[0].Quantizer != 42 {
		t.Fatalf("key frames=%+v", key.Frames)
	}
	delta, _, err := av1.EncoderWebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	if delta.Frames[1].RateControl != av1.EncoderRateControlCQP || delta.Frames[1].Quantizer != 42 {
		t.Fatalf("delta frames=%+v", delta.Frames)
	}
}

func TestPublicEncoderWebRTCScreenContentHeaders(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Content:           av1.EncoderContentScreen,
	}
	key, state, err := av1.EncoderWebRTCKeyFrameTemporalUnitForState(cfg, av1.EncoderWebRTCState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if !key.Header.Prefix.AllowScreenContentTools || !key.Header.Prefix.ForceIntegerMV {
		t.Fatalf("key prefix=%+v", key.Header.Prefix)
	}
	delta, _, err := av1.EncoderWebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	if !delta.Headers[1].Prefix.AllowScreenContentTools || !delta.Headers[1].Prefix.ForceIntegerMV {
		t.Fatalf("delta headers=%+v", delta.Headers)
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

func TestPublicEncoderWebRTCStateTemporalUnits(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 960, Height: 540},
		Scalability:       av1.EncoderScalabilityModeL3T3,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 800,
	}
	state := av1.EncoderWebRTCState{NextOrderHint: 126, NextFrameID: 900}
	key, state, err := av1.EncoderWebRTCKeyFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.FrameNum != 3 || key.Header.Prefix.OrderHint != 126 ||
		state.NextOrderHint != 127 || state.NextFrameID != 903 ||
		state.DeltaPictureIndex != 1 || !state.DependencyStructureState.Valid {
		t.Fatalf("key=%+v state=%+v", key, state)
	}

	wantTemporal := [...]uint8{2, 1, 2, 0}
	for i, want := range wantTemporal {
		delta, next, err := av1.EncoderWebRTCDeltaFrameTemporalUnitForState(cfg, state)
		if err != nil {
			t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState %d: %v", i, err)
		}
		if delta.FrameNum != 3 || delta.Frames[0].TemporalID != want ||
			delta.Control.Frames[0].GenericFrameInfo.FrameID != state.NextFrameID {
			t.Fatalf("delta %d=%+v state=%+v", i, delta, state)
		}
		if i == 0 {
			headerSize, sizedUnit, sizedNext, err := av1.EncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize(cfg, state)
			if err != nil {
				t.Fatalf("EncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize: %v", err)
			}
			var headerBuf [256]byte
			headerTU, appendedUnit, appendedNext, err := av1.AppendEncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForState(headerBuf[:0], cfg, state)
			if err != nil {
				t.Fatalf("AppendEncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForState: %v", err)
			}
			if len(headerTU) != headerSize || appendedUnit != sizedUnit || appendedNext != sizedNext ||
				appendedUnit.Control.Frames[0].GenericFrameInfo.FrameID != delta.Control.Frames[0].GenericFrameInfo.FrameID {
				t.Fatalf("header temporal unit len=%d want=%d unit=%+v sized=%+v next=%+v sizedNext=%+v", len(headerTU), headerSize, appendedUnit, sizedUnit, appendedNext, sizedNext)
			}
		}
		var descriptorBuf [32]byte
		descriptor, err := av1.AppendEncoderWebRTCDependencyDescriptor(
			descriptorBuf[:0],
			next.DependencyStructureState.Structure,
			delta.Control.Frames[2].GenericFrameInfo,
			true,
			true,
			false,
		)
		if err != nil {
			t.Fatalf("AppendEncoderWebRTCDependencyDescriptor %d: %v", i, err)
		}
		if len(descriptor) == 0 || next.NextFrameID != state.NextFrameID+3 {
			t.Fatalf("delta %d descriptor=% x next=%+v state=%+v", i, descriptor, next, state)
		}
		state = next
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = av1.EncoderWebRTCDeltaFrameTemporalUnitForState(cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCNextTemporalUnitForState(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		KeyFrameInterval:  3,
	}
	state := av1.EncoderWebRTCState{NextFrameID: 50}
	unit, state, err := av1.EncoderWebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState initial: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 || state.NextFrameID != 52 {
		t.Fatalf("initial unit=%+v state=%+v", unit, state)
	}
	headerSize, headerUnit, headerNext, err := av1.EncoderLowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderLowOverheadWebRTCPictureHeaderTemporalUnitForStateSize: %v", err)
	}
	var headerBuf [256]byte
	headerTU, appendedHeaderUnit, appendedHeaderNext, err := av1.AppendEncoderLowOverheadWebRTCPictureHeaderTemporalUnitForState(headerBuf[:0], cfg, state, false)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadWebRTCPictureHeaderTemporalUnitForState: %v", err)
	}
	if len(headerTU) != headerSize || appendedHeaderUnit != headerUnit || appendedHeaderNext != headerNext ||
		!appendedHeaderUnit.Delta || appendedHeaderUnit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 52 {
		t.Fatalf("header TU len=%d want=%d unit=%+v sized=%+v next=%+v sizedNext=%+v", len(headerTU), headerSize, appendedHeaderUnit, headerUnit, appendedHeaderNext, headerNext)
	}
	unit, state, err = av1.EncoderWebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta: %v", err)
	}
	if !unit.Delta || unit.DeltaUnit.Frames[0].TemporalID != 1 ||
		unit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 52 {
		t.Fatalf("delta unit=%+v state=%+v", unit, state)
	}
	unit, state, err = av1.EncoderWebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta2: %v", err)
	}
	if !unit.Delta || unit.DeltaUnit.Frames[0].TemporalID != 0 {
		t.Fatalf("delta2 unit=%+v state=%+v", unit, state)
	}
	unit, _, err = av1.EncoderWebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState interval key: %v", err)
	}
	if !unit.Key || !unit.KeyUnit.Control.HasDependencyStructure {
		t.Fatalf("interval key unit=%+v", unit)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = av1.EncoderWebRTCNextTemporalUnitForState(cfg, state, false)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState allocated: %f", allocs)
	}
}

func TestPublicWebRTCEncoderPictureHeaderTemporalUnits(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		KeyFrameInterval:  4,
	}
	enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 90})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	if got := enc.Config(); got.SpatialLayerCount != 2 || got.TemporalLayerCount != 2 {
		t.Fatalf("normalized config=%+v", got)
	}
	initialState := enc.State()
	keySize, sizedKey, err := enc.LowOverheadPictureHeaderTemporalUnitSize(false)
	if err != nil {
		t.Fatalf("LowOverheadPictureHeaderTemporalUnitSize key: %v", err)
	}
	if !sizedKey.Key || sizedKey.KeyUnit.FrameNum != 2 {
		t.Fatalf("sized key=%+v", sizedKey)
	}
	var tiny [1]byte
	short, _, err := enc.AppendLowOverheadPictureHeaderTemporalUnit(tiny[:0:1], false)
	if !errors.Is(err, av1.ErrEncoderShortBuffer) || len(short) != 0 || enc.State() != initialState {
		t.Fatalf("short out=% x err=%v state=%+v want=%+v", short, err, enc.State(), initialState)
	}

	var buf [256]byte
	keyOut, gotKey, err := enc.AppendLowOverheadPictureHeaderTemporalUnit(buf[:0], false)
	if err != nil {
		t.Fatalf("AppendLowOverheadPictureHeaderTemporalUnit key: %v", err)
	}
	keyState := enc.State()
	if len(keyOut) != keySize || gotKey != sizedKey || keyState.NextFrameID != 92 ||
		keyState.DeltaPictureIndex != 1 || !keyState.DependencyStructureState.Valid {
		t.Fatalf("key len=%d want=%d unit=%+v sized=%+v state=%+v", len(keyOut), keySize, gotKey, sizedKey, keyState)
	}
	it := av1.NewTemporalUnitIterator(keyOut)
	tu, ok, err := it.Next()
	if err != nil || !ok || !bytes.Equal(tu.Raw, keyOut) {
		t.Fatalf("key temporal unit ok=%v err=%v raw=% x", ok, err, tu.Raw)
	}

	deltaSize, sizedDelta, err := enc.LowOverheadPictureHeaderTemporalUnitSize(false)
	if err != nil {
		t.Fatalf("LowOverheadPictureHeaderTemporalUnitSize delta: %v", err)
	}
	deltaOut, gotDelta, err := enc.AppendLowOverheadPictureHeaderTemporalUnit(buf[:0], false)
	if err != nil {
		t.Fatalf("AppendLowOverheadPictureHeaderTemporalUnit delta: %v", err)
	}
	if len(deltaOut) != deltaSize || gotDelta != sizedDelta || !gotDelta.Delta ||
		gotDelta.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 92 ||
		enc.State().NextFrameID != 94 || enc.State().DeltaPictureIndex != 2 {
		t.Fatalf("delta len=%d want=%d unit=%+v sized=%+v state=%+v", len(deltaOut), deltaSize, gotDelta, sizedDelta, enc.State())
	}

	if err := enc.ResetState(initialState); err != nil {
		t.Fatalf("ResetState: %v", err)
	}
	unit, err := enc.NextTemporalUnit(true)
	if err != nil {
		t.Fatalf("NextTemporalUnit forced key: %v", err)
	}
	if !unit.Key || enc.State().NextFrameID != 92 {
		t.Fatalf("forced unit=%+v state=%+v", unit, enc.State())
	}

	base, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 90})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder alloc base: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		local := base
		_, _, _ = local.AppendLowOverheadPictureHeaderTemporalUnit(buf[:0], false)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCEncoder AppendLowOverheadPictureHeaderTemporalUnit allocated: %f", allocs)
	}
}

func TestPublicWebRTCEncoderSetConfigReconfigure(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
		Scalability:       av1.EncoderScalabilityModeL1T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 30})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	if unit, err := enc.NextTemporalUnit(false); err != nil || !unit.Key {
		t.Fatalf("initial key unit=%+v err=%v", unit, err)
	}
	if unit, err := enc.NextTemporalUnit(false); err != nil || !unit.Delta || unit.DeltaUnit.FrameNum != 1 {
		t.Fatalf("warm delta unit=%+v err=%v", unit, err)
	}

	controlChange := cfg
	controlChange.MaxFramerate = av1.EncoderRational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1800
	controlChange.TargetBitrateKbps = 1100
	if err := enc.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	before := enc.State()
	unit, err := enc.NextTemporalUnit(false)
	if err != nil {
		t.Fatalf("NextTemporalUnit after control change: %v", err)
	}
	if !unit.Delta || unit.Key || unit.DeltaUnit.FrameNum != 1 ||
		enc.State().NextFrameID != before.NextFrameID+1 || enc.Config().TargetBitrateKbps != 1100 {
		t.Fatalf("control change unit=%+v before=%+v state=%+v config=%+v", unit, before, enc.State(), enc.Config())
	}

	structureChange := controlChange
	structureChange.Scalability = av1.EncoderScalabilityModeS2T2
	if err := enc.SetConfig(structureChange); err != nil {
		t.Fatalf("SetConfig structure change: %v", err)
	}
	before = enc.State()
	unit, err = enc.NextTemporalUnit(false)
	if err != nil {
		t.Fatalf("NextTemporalUnit after structure change: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 ||
		!unit.KeyUnit.Control.HasDependencyStructure ||
		enc.State().NextFrameID != before.NextFrameID+2 ||
		enc.State().DeltaPictureIndex != 1 {
		t.Fatalf("structure change unit=%+v before=%+v state=%+v", unit, before, enc.State())
	}

	unit, err = enc.NextTemporalUnit(false)
	if err != nil {
		t.Fatalf("warm same-shape delta: %v", err)
	}
	if !unit.Delta || unit.Key || unit.DeltaUnit.FrameNum != 2 {
		t.Fatalf("warm same-shape unit=%+v", unit)
	}
	sameShapeChange := enc.Config()
	sameShapeChange.Scalability = av1.EncoderScalabilityModeL2T2
	before = enc.State()
	if before.DeltaPictureIndex == 0 || !before.DependencyStructureState.Valid {
		t.Fatalf("warm same-shape state=%+v", before)
	}
	if err := enc.SetConfig(sameShapeChange); err != nil {
		t.Fatalf("SetConfig same-shape structure change: %v", err)
	}
	afterSet := enc.State()
	if afterSet.NextFrameID != before.NextFrameID ||
		afterSet.NextOrderHint != before.NextOrderHint ||
		afterSet.DeltaPictureIndex != 0 ||
		afterSet.DependencyStructureState.Valid {
		t.Fatalf("same-shape SetConfig state=%+v before=%+v", afterSet, before)
	}
	unit, err = enc.NextTemporalUnit(false)
	if err != nil {
		t.Fatalf("NextTemporalUnit after same-shape structure change: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 ||
		!unit.KeyUnit.Control.HasDependencyStructure ||
		enc.State().NextFrameID != before.NextFrameID+2 ||
		enc.State().DeltaPictureIndex != 1 {
		t.Fatalf("same-shape structure change unit=%+v before=%+v state=%+v", unit, before, enc.State())
	}

	keepConfig := enc.Config()
	keepState := enc.State()
	bad := enc.Config()
	bad.TargetBitrateKbps = bad.MaxBitrateKbps + 1
	if err := enc.SetConfig(bad); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
		t.Fatalf("SetConfig invalid err=%v want %v", err, av1.ErrEncoderInvalidConfig)
	}
	if enc.Config() != keepConfig || enc.State() != keepState {
		t.Fatalf("invalid SetConfig mutated config/state config=%+v want=%+v state=%+v want=%+v", enc.Config(), keepConfig, enc.State(), keepState)
	}
}

func TestPublicWebRTCEncoderPictureHeaderTemporalUnitsForFrames(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 120})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	normalized := enc.Config()
	var frames [2]av1.Frame
	var backing [2][]byte
	for i := range frames {
		layer := normalized.SpatialLayers[i]
		format := av1.FrameFormat{
			Width:        int(layer.Resolution.Width),
			Height:       int(layer.Resolution.Height),
			BitDepth:     normalized.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
			Align:        64,
		}
		layout, err := av1.FrameRequiredSize(format)
		if err != nil {
			t.Fatalf("FrameRequiredSize layer %d: %v", i, err)
		}
		backing[i] = make([]byte, layout.Size)
		frames[i], err = av1.BindFrame(backing[i], format)
		if err != nil {
			t.Fatalf("BindFrame layer %d: %v", i, err)
		}
	}
	size, sizedUnit, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize(frames[:], false)
	if err != nil {
		t.Fatalf("LowOverheadPictureHeaderTemporalUnitForFramesSize key: %v", err)
	}
	if !sizedUnit.Key || sizedUnit.KeyUnit.FrameNum != 2 {
		t.Fatalf("sized unit=%+v", sizedUnit)
	}
	state := enc.State()
	var tiny [1]byte
	short, _, err := enc.AppendLowOverheadPictureHeaderTemporalUnitForFrames(tiny[:0:1], frames[:], false)
	if !errors.Is(err, av1.ErrEncoderShortBuffer) || len(short) != 0 || enc.State() != state {
		t.Fatalf("short out=% x err=%v state=%+v want=%+v", short, err, enc.State(), state)
	}

	var outBuf [256]byte
	out, unit, err := enc.AppendLowOverheadPictureHeaderTemporalUnitForFrames(outBuf[:0], frames[:], false)
	if err != nil {
		t.Fatalf("AppendLowOverheadPictureHeaderTemporalUnitForFrames key: %v", err)
	}
	if len(out) != size || unit != sizedUnit || enc.State().NextFrameID != 122 {
		t.Fatalf("out len=%d want=%d unit=%+v sized=%+v state=%+v", len(out), size, unit, sizedUnit, enc.State())
	}

	bad := frames
	bad[1].Format.Width--
	if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize(bad[:], false); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
		t.Fatalf("bad frame err=%v want %v", err, av1.ErrEncoderInvalidFrame)
	}
	if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize(frames[:1], false); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
		t.Fatalf("short frame list err=%v want %v", err, av1.ErrEncoderInvalidFrame)
	}

	base, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 120})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder alloc base: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		local := base
		_, _, _ = local.AppendLowOverheadPictureHeaderTemporalUnitForFrames(outBuf[:0], frames[:], false)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCEncoder AppendLowOverheadPictureHeaderTemporalUnitForFrames allocated: %f", allocs)
	}
}

func TestPublicWebRTCEncoderPictureSamplePlanes(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 200})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	normalized := enc.Config()
	var frames [2]av1.Frame
	var backing [2][]byte
	for i := range frames {
		layer := normalized.SpatialLayers[i]
		format := av1.FrameFormat{
			Width:        int(layer.Resolution.Width),
			Height:       int(layer.Resolution.Height),
			BitDepth:     normalized.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
			Align:        64,
		}
		layout, err := av1.FrameRequiredSize(format)
		if err != nil {
			t.Fatalf("FrameRequiredSize layer %d: %v", i, err)
		}
		backing[i] = make([]byte, layout.Size)
		frames[i], err = av1.BindFrame(backing[i], format)
		if err != nil {
			t.Fatalf("BindFrame layer %d: %v", i, err)
		}
		frames[i].Y.Pix[2*frames[i].Y.Stride+3] = byte(20 + i)
		frames[i].U.Pix[1*frames[i].U.Stride+2] = byte(40 + i)
		frames[i].V.Pix[1*frames[i].V.Stride+2] = byte(60 + i)
	}
	size, unit, err := enc.PictureSampleScratchSize(frames[:], false)
	if err != nil {
		t.Fatalf("PictureSampleScratchSize: %v", err)
	}
	if size.Samples == 0 || size.FrameNum != 2 || !unit.Key {
		t.Fatalf("scratch size=%+v unit=%+v", size, unit)
	}
	if _, _, err := enc.LoadPictureSamplePlanes(av1.EncoderWebRTCPictureSampleScratch{Samples: make([]uint16, size.Samples-1)}, frames[:], false); !errors.Is(err, av1.ErrFrameShortBuffer) {
		t.Fatalf("short scratch err=%v want %v", err, av1.ErrFrameShortBuffer)
	}
	samples := make([]uint16, size.Samples)
	planes, gotUnit, err := enc.LoadPictureSamplePlanes(av1.EncoderWebRTCPictureSampleScratch{Samples: samples}, frames[:], false)
	if err != nil {
		t.Fatalf("LoadPictureSamplePlanes: %v", err)
	}
	if gotUnit != unit || planes.FrameNum != 2 {
		t.Fatalf("planes=%+v unit=%+v want=%+v", planes, gotUnit, unit)
	}
	if planes.Frames[0].Y.Pix[2*planes.Frames[0].Y.Stride+3] != 20 ||
		planes.Frames[1].Y.Pix[2*planes.Frames[1].Y.Stride+3] != 21 ||
		planes.Frames[0].U.Pix[1*planes.Frames[0].U.Stride+2] != 40 ||
		planes.Frames[1].V.Pix[1*planes.Frames[1].V.Stride+2] != 61 {
		t.Fatalf("loaded samples layer0=%+v layer1=%+v", planes.Frames[0], planes.Frames[1])
	}

	base, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 200})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder alloc base: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		local := base
		_, _, _ = local.LoadPictureSamplePlanes(av1.EncoderWebRTCPictureSampleScratch{Samples: samples}, frames[:], false)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCEncoder LoadPictureSamplePlanes allocated: %f", allocs)
	}
}

func TestPublicWebRTCEncoderPictureFramesRTPPackets(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		Scalability:       av1.EncoderScalabilityModeL2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 300})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	framePayloads := [...][]byte{
		{0x10, 0x11, 0x12, 0x13},
		{0x20, 0x21, 0x22, 0x23},
	}
	limits := av1.RTPPayloadSizeLimits{MaxPayloadLen: 64}

	var frameOBUProbe [64]byte
	firstSize, firstUnit, err := enc.PictureTemporalUnitFramesRTPScratchSize(framePayloads[:], limits, false, frameOBUProbe[:0], nil)
	if err != nil {
		t.Fatalf("PictureTemporalUnitFramesRTPScratchSize first pass: %v", err)
	}
	if !firstUnit.Key || firstSize.FrameOBUBytes == 0 || firstSize.Packetizer.OBUs == 0 || firstSize.PacketSpans != 0 {
		t.Fatalf("first size=%+v unit=%+v", firstSize, firstUnit)
	}

	var obuScratch [4]av1.RTPPacketizerOBU
	size, unit, err := enc.PictureTemporalUnitFramesRTPScratchSize(framePayloads[:], limits, false, frameOBUProbe[:0], obuScratch[:firstSize.Packetizer.OBUs])
	if err != nil {
		t.Fatalf("PictureTemporalUnitFramesRTPScratchSize full pass: %v", err)
	}
	if unit != firstUnit || size.FrameSpans != 2 || size.PacketSpans != 2 ||
		size.Packetizer.Packets == 0 || size.MaxPayloadBytes != limits.MaxPayloadLen ||
		size.MaxDescriptorBytes <= av1.RTPDependencyDescriptorMandatorySize {
		t.Fatalf("size=%+v unit=%+v first=%+v", size, unit, firstUnit)
	}

	var frameOBUs [64]byte
	var frameSpans [2]av1.EncoderWebRTCFrameRTPPacketSpan
	var packetSpans [4]av1.EncoderWebRTCRTPPacketSpan
	var packets [4]av1.RTPPacketPlan
	var work [4]av1.RTPPacketPlan
	scratch, err := av1.BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size, av1.EncoderWebRTCPictureTemporalUnitFramesRTPScratch{
		FrameOBU:    frameOBUs[:0],
		FrameSpans:  frameSpans[:],
		PacketSpans: packetSpans[:],
		OBUs:        obuScratch[:],
		Packets:     packets[:],
		Work:        work[:],
	})
	if err != nil {
		t.Fatalf("BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch: %v", err)
	}
	initialState := enc.State()
	var payloads [128]byte
	var tinyDescriptor [1]byte
	shortFrameOBUs, shortPayloads, shortDescriptors, _, _, _, err := enc.AppendPictureTemporalUnitFramesRTPPacketsWithScratch(payloads[:0], tinyDescriptor[:0:1], scratch, framePayloads[:], limits, false)
	if err == nil || !errors.Is(err, av1.ErrEncoderShortBuffer) && !errors.Is(err, av1.ErrRTPShortBuffer) {
		t.Fatalf("short descriptor err=%v", err)
	}
	if len(shortFrameOBUs) != 0 || len(shortPayloads) != 0 || len(shortDescriptors) != 0 || enc.State() != initialState {
		t.Fatalf("short frameOBUs=% x payloads=% x descriptors=% x state=%+v want=%+v", shortFrameOBUs, shortPayloads, shortDescriptors, enc.State(), initialState)
	}

	scratch, err = av1.BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size, av1.EncoderWebRTCPictureTemporalUnitFramesRTPScratch{
		FrameOBU:    frameOBUs[:0],
		FrameSpans:  frameSpans[:],
		PacketSpans: packetSpans[:],
		OBUs:        obuScratch[:],
		Packets:     packets[:],
		Work:        work[:],
	})
	if err != nil {
		t.Fatalf("BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch retry: %v", err)
	}
	var descriptors [128]byte
	frameOBUOut, rtpPayloads, descriptorOut, frameCount, packetCount, gotUnit, err := enc.AppendPictureTemporalUnitFramesRTPPacketsWithScratch(payloads[:0], descriptors[:0], scratch, framePayloads[:], limits, false)
	if err != nil {
		t.Fatalf("AppendPictureTemporalUnitFramesRTPPacketsWithScratch: %v", err)
	}
	if gotUnit != unit || frameCount != 2 || packetCount != 2 || len(frameOBUOut) != size.FrameOBUBytes ||
		len(rtpPayloads) == 0 || len(descriptorOut) <= av1.RTPDependencyDescriptorMandatorySize ||
		enc.State().NextFrameID != 302 || enc.State().DeltaPictureIndex != 1 {
		t.Fatalf("frameOBUs=%d/%d payloads=%d descriptors=%d frames=%d packets=%d unit=%+v sized=%+v state=%+v", len(frameOBUOut), size.FrameOBUBytes, len(rtpPayloads), len(descriptorOut), frameCount, packetCount, gotUnit, unit, enc.State())
	}
	header, _, err := av1.ParseRTPAggregationHeader(rtpPayloads[packetSpans[0].PayloadOffset:])
	if err != nil {
		t.Fatalf("ParseRTPAggregationHeader first packet: %v", err)
	}
	if header.ElementCount == 0 || !packetSpans[1].Marker {
		t.Fatalf("header=%+v packetSpans=%+v", header, packetSpans[:packetCount])
	}

	base, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 300})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder alloc base: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		local := base
		localScratch, bindErr := av1.BindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size, av1.EncoderWebRTCPictureTemporalUnitFramesRTPScratch{
			FrameOBU:    frameOBUs[:0],
			FrameSpans:  frameSpans[:],
			PacketSpans: packetSpans[:],
			OBUs:        obuScratch[:],
			Packets:     packets[:],
			Work:        work[:],
		})
		if bindErr != nil {
			t.Fatal(bindErr)
		}
		_, _, _, _, _, _, _ = local.AppendPictureTemporalUnitFramesRTPPacketsWithScratch(payloads[:0], descriptors[:0], localScratch, framePayloads[:], limits, false)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCEncoder AppendPictureTemporalUnitFramesRTPPacketsWithScratch allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCTemporalIDForDeltaPicture(t *testing.T) {
	got, err := av1.EncoderWebRTCTemporalIDForDeltaPicture(3, 4)
	if err != nil {
		t.Fatalf("EncoderWebRTCTemporalIDForDeltaPicture: %v", err)
	}
	if got != 0 {
		t.Fatalf("temporal id=%d want 0", got)
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

func publicEncoderReferenceState(seq av1.EncoderSequenceHeader, indices []uint8) av1.ReferenceState {
	var refs av1.ReferenceState
	for _, idx := range indices {
		refs.Frames[idx].Valid = true
		refs.Frames[idx].OrderHint = 1
		refs.Frames[idx].Size.UpscaledWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.CodedWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.Height = seq.MaxFrameHeight
		refs.Frames[idx].Size.RenderWidth = seq.MaxFrameWidth
		refs.Frames[idx].Size.RenderHeight = seq.MaxFrameHeight
		refs.Frames[idx].Size.SuperResDenominator = 8
	}
	return refs
}
