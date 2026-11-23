package goav1

import (
	"bytes"
	"errors"
	"math/bits"
	"testing"

	internalbitstream "github.com/thesyncim/goav1/internal/av1/bitstream"
	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
)

func TestPublicEncoderControlSurface(t *testing.T) {
	mode, ok := internalencoder.ParseScalabilityMode("L2T2_KEY")
	if !ok || mode != internalencoder.ScalabilityModeL2T2_KEY {
		t.Fatalf("ParseEncoderScalabilityMode = %v,%v", mode, ok)
	}
	if shifted, ok := internalencoder.ParseScalabilityMode("L3T3_KEY_SHIFT"); !ok || shifted != EncoderScalabilityModeL3T3_KEY_SHIFT {
		t.Fatalf("ParseEncoderScalabilityMode L3T3_KEY_SHIFT = %v,%v", shifted, ok)
	}

	cfg, err := SetWebRTCEncoderSVCConfig(EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
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
	if cfg.SpatialLayers[0].Resolution != (EncoderResolution{Width: 320, Height: 180}) {
		t.Fatalf("base layer resolution = %+v", cfg.SpatialLayers[0].Resolution)
	}

	state := internalencoder.ReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = EncoderResolution{Width: 320, Height: 180}
	frames := [...]internalencoder.FrameEncodeSettings{
		{
			Type:             internalencoder.FrameTypeDelta,
			Resolution:       EncoderResolution{Width: 640, Height: 360},
			SpatialID:        1,
			ReferenceBuffers: [internalencoder.WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			Output:           true,
		},
	}
	if _, err := internalencoder.ValidateTemporalUnitFrames(frames[:], state, EncoderRateControlCBR); err != nil {
		t.Fatalf("ValidateEncoderTemporalUnitFrames: %v", err)
	}

	refConfig, err := internalencoder.LibaomSVCRefFrameConfigForFrame(frames[0])
	if err != nil {
		t.Fatalf("EncoderLibaomSVCRefFrameConfigForFrame: %v", err)
	}
	if refConfig.Reference[0] != 1 || refConfig.RefIdx[0] != 0 {
		t.Fatalf("libaom ref config = %+v", refConfig)
	}

	idState := internalencoder.FrameIDBufferState{}
	idState.Valid[0] = true
	idState.FrameIDs[0] = 7
	info, nextIDState, err := internalencoder.WebRTCGenericFrameInfoForFrame(frames[0], 8, idState, cfg.SpatialLayerCount, cfg.TemporalLayerCount)
	if err != nil {
		t.Fatalf("EncoderWebRTCGenericFrameInfoForFrame: %v", err)
	}
	if info.DependencyNum != 1 || info.Dependencies[0] != 7 || info.DTINum != 4 ||
		info.DTIs[2] != internalencoder.DecodeTargetSwitch || info.DTIs[3] != internalencoder.DecodeTargetSwitch {
		t.Fatalf("generic frame info = %+v", info)
	}
	if nextIDState.Valid[0] != idState.Valid[0] || nextIDState.FrameIDs[0] != idState.FrameIDs[0] {
		t.Fatalf("unexpected id state mutation = %+v", nextIDState)
	}

	structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if structure.NumDecodeTargets != 4 || structure.TemplateNum != 6 {
		t.Fatalf("dependency structure = %+v", structure)
	}

	control, err := internalencoder.WebRTCTemporalUnitControlForFrames(cfg, frames[:], state, idState, 8)
	if err != nil {
		t.Fatalf("EncoderWebRTCTemporalUnitControlForFrames: %v", err)
	}
	if control.FrameNum != 1 || control.Frames[0].GenericFrameInfo.DependencyNum != 1 ||
		control.Frames[0].LibaomSVCRefFrameConfig.Reference[0] != 1 {
		t.Fatalf("temporal-unit control = %+v", control)
	}

	controlInfo := control.Frames[0].GenericFrameInfo
	templateID, err := internalencoder.WebRTCTemplateIDForFrame(structure, controlInfo)
	if err != nil {
		t.Fatalf("WebRTCTemplateIDForFrame: %v", err)
	}
	descriptor := RTPDependencyDescriptorMandatory{
		FirstPacketInFrame: true,
		LastPacketInFrame:  true,
		TemplateID:         templateID,
		FrameNumber:        uint16(controlInfo.FrameID),
	}
	var descriptorBuf [RTPDependencyDescriptorMandatorySize]byte
	n, err := PutRTPDependencyDescriptorMandatory(descriptorBuf[:], descriptor)
	if err != nil {
		t.Fatalf("PutRTPDependencyDescriptorMandatory: %v", err)
	}
	parsed, consumed, err := ParseRTPDependencyDescriptorMandatory(descriptorBuf[:])
	if err != nil {
		t.Fatalf("ParseRTPDependencyDescriptorMandatory: %v", err)
	}
	if n != RTPDependencyDescriptorMandatorySize || consumed != n || parsed != descriptor {
		t.Fatalf("descriptor roundtrip n=%d consumed=%d parsed=%+v want=%+v", n, consumed, parsed, descriptor)
	}

	fullSize, err := internalencoder.WebRTCDependencyDescriptorSize(structure, control.Frames[0].GenericFrameInfo, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSize: %v", err)
	}
	var fullBuf [16]byte
	full, err := internalencoder.AppendWebRTCDependencyDescriptor(fullBuf[:0], structure, control.Frames[0].GenericFrameInfo, true, true, false)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	if len(full) != fullSize || full[0] != descriptorBuf[0] || full[1] != descriptorBuf[1] || full[2] != descriptorBuf[2] {
		t.Fatalf("full descriptor=% x size=%d mandatory=% x", full, fullSize, descriptorBuf)
	}
}

func TestPublicWebRTCEncoderRTCPFeedbackMethods(t *testing.T) {
	mode := internalencoder.ScalabilityModeL3T2h
	width, height := publicRTCMatrixGeometry(t, mode)
	cfg := publicRTCMatrixConfig(width, height, mode)
	publicRTCApplyControlBitrates(&cfg, 1500)
	state := internalencoder.WebRTCEncoderState{NextOrderHint: 9, NextFrameID: 23}
	enc, err := newWebRTCEncoder(cfg, state)
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}

	pliCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBPictureLossIndicationFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  0x05060708,
	})
	applied, packets, err := enc.ApplyRTCPReceiverEstimatedMaximumBitrate(
		pliCompound,
		make([]RTCPPacket, 0, 1),
		make([]uint32, 0, 1),
	)
	if err != nil {
		t.Fatalf("ApplyRTCPReceiverEstimatedMaximumBitrate non-REMB: %v", err)
	}
	if applied || len(packets) != 1 || enc.Config().TargetBitrateKbps != cfg.TargetBitrateKbps {
		t.Fatalf("non-REMB applied=%v packets=%d target=%d want false,1,%d", applied, len(packets), enc.Config().TargetBitrateKbps, cfg.TargetBitrateKbps)
	}

	rembCompound := publicRTCREMBFeedbackCompound(t, 700_000)
	applied, packets, err = enc.ApplyRTCPReceiverEstimatedMaximumBitrate(
		rembCompound,
		make([]RTCPPacket, 0, 1),
		make([]uint32, 0, 1),
	)
	if err != nil {
		t.Fatalf("ApplyRTCPReceiverEstimatedMaximumBitrate: %v", err)
	}
	if !applied || len(packets) != 1 || enc.Config().TargetBitrateKbps != 700 {
		t.Fatalf("REMB applied=%v packets=%d target=%d want true,1,700", applied, len(packets), enc.Config().TargetBitrateKbps)
	}
	if enc.State() != state {
		t.Fatalf("REMB control change reset state: got %+v want %+v", enc.State(), state)
	}

	forceKey, packets, err := enc.RTCPRequiresKeyFrame(
		pliCompound,
		make([]RTCPPacket, 0, 1),
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("RTCPRequiresKeyFrame PLI: %v", err)
	}
	if !forceKey || len(packets) != 1 {
		t.Fatalf("PLI forceKey=%v packets=%d want true,1", forceKey, len(packets))
	}

	entry := testAV1RTCPValidLayerRefreshEntry(t, mode)
	lrrCompound := testAV1RTCPFeedbackCompound(t, RTCPFeedbackPacket{
		PacketType: RTCPPSFBPacketType,
		FMT:        RTCPPSFBLayerRefreshRequestFMT,
		SenderSSRC: 0x01020304,
		MediaSSRC:  entry.SSRC,
		FCI:        testAV1RTCPLayerRefreshFCI(t, []AV1RTCPLayerRefreshRequestEntry{entry}),
	})
	target, ok, packets, err := enc.RTCPLayerRefreshTarget(
		lrrCompound,
		make([]RTCPPacket, 0, 1),
		make([]AV1RTCPLayerRefreshRequestEntry, 0, 1),
	)
	if err != nil {
		t.Fatalf("RTCPLayerRefreshTarget: %v", err)
	}
	if !ok || len(packets) != 1 || target != entry.Target {
		t.Fatalf("LRR target=%+v ok=%v packets=%d want %+v,true,1", target, ok, len(packets), entry.Target)
	}

	packetScratch := make([]RTCPPacket, 0, 1)
	ssrcScratch := make([]uint32, 0, 1)
	lrrScratch := make([]AV1RTCPLayerRefreshRequestEntry, 0, 1)
	allocs := testing.AllocsPerRun(1000, func() {
		if ok, packets, err := enc.ApplyRTCPReceiverEstimatedMaximumBitrate(rembCompound, packetScratch[:0], ssrcScratch[:0]); err != nil || !ok || len(packets) != 1 {
			t.Fatalf("alloc REMB ok=%v packets=%d err=%v", ok, len(packets), err)
		}
		if force, packets, err := enc.RTCPRequiresKeyFrame(pliCompound, packetScratch[:0], nil, nil); err != nil || !force || len(packets) != 1 {
			t.Fatalf("alloc PLI force=%v packets=%d err=%v", force, len(packets), err)
		}
		if target, ok, packets, err := enc.RTCPLayerRefreshTarget(lrrCompound, packetScratch[:0], lrrScratch[:0]); err != nil || !ok || len(packets) != 1 || target != entry.Target {
			t.Fatalf("alloc LRR target=%+v ok=%v packets=%d err=%v", target, ok, len(packets), err)
		}
	})
	if allocs != 0 {
		t.Fatalf("WebRTCEncoder RTCP feedback method allocs/run=%f want 0", allocs)
	}
}

func TestPublicEncoderWebRTCRTPFrameDuration(t *testing.T) {
	base := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       EncoderScalabilityModeL1T2,
	}
	tests := []struct {
		name     string
		fps      EncoderRational
		timebase EncoderRational
		want     EncoderRational
	}{
		{name: "defaults", want: EncoderRational{Num: 3000, Den: 1}},
		{name: "thirty", fps: EncoderRational{Num: 30, Den: 1}, want: EncoderRational{Num: 3000, Den: 1}},
		{name: "ntsc", fps: EncoderRational{Num: 30000, Den: 1001}, want: EncoderRational{Num: 3003, Den: 1}},
		{name: "custom timebase", fps: EncoderRational{Num: 24, Den: 1}, timebase: EncoderRational{Num: 1, Den: 1000}, want: EncoderRational{Num: 125, Den: 3}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			cfg.MaxFramerate = tc.fps
			cfg.RTPTimebase = tc.timebase
			got, err := EncoderWebRTCRTPFrameDuration(cfg)
			if err != nil {
				t.Fatalf("EncoderWebRTCRTPFrameDuration: %v", err)
			}
			if got != tc.want {
				t.Fatalf("duration=%+v want %+v", got, tc.want)
			}
		})
	}

	invalid := base
	invalid.MaxFramerate = EncoderRational{Num: 1, Den: 2}
	if _, err := EncoderWebRTCRTPFrameDuration(invalid); !errors.Is(err, internalencoder.ErrInvalidConfig) {
		t.Fatalf("invalid framerate err=%v want %v", err, internalencoder.ErrInvalidConfig)
	}
}

func TestPublicEncoderWebRTCScalabilityModes(t *testing.T) {
	wantNames := []string{
		"L1T1", "L1T2", "L1T3",
		"L2T1", "L2T1h", "L2T1_KEY",
		"L2T2", "L2T2h", "L2T2_KEY", "L2T2_KEY_SHIFT",
		"L2T3", "L2T3h", "L2T3_KEY",
		"L3T1", "L3T1h", "L3T1_KEY",
		"L3T2", "L3T2h", "L3T2_KEY",
		"L3T3", "L3T3h", "L3T3_KEY",
		"S2T1", "S2T1h", "S2T2", "S2T2h", "S2T3", "S2T3h",
		"S3T1", "S3T1h", "S3T2", "S3T2h", "S3T3", "S3T3h",
	}
	modes := EncoderWebRTCScalabilityModes()
	if len(modes) != len(wantNames) {
		t.Fatalf("mode count=%d want %d", len(modes), len(wantNames))
	}
	seen := make(map[EncoderScalabilityMode]bool, len(modes))
	for i, mode := range modes {
		if mode.String() != wantNames[i] {
			t.Fatalf("mode[%d]=%s want %s", i, mode, wantNames[i])
		}
		if seen[mode] {
			t.Fatalf("duplicate mode %s", mode)
		}
		seen[mode] = true
		parsed, ok := internalencoder.ParseScalabilityMode(wantNames[i])
		if !ok || parsed != mode {
			t.Fatalf("ParseEncoderScalabilityMode(%q)=%s,%v want %s,true", wantNames[i], parsed, ok, mode)
		}
		normalized, err := SetWebRTCEncoderSVCConfig(EncoderConfig{
			Resolution:        EncoderResolution{Width: 1280, Height: 720},
			MaxFramerate:      EncoderRational{Num: 30, Den: 1},
			MinBitrateKbps:    120,
			MaxBitrateKbps:    1600,
			TargetBitrateKbps: 900,
			Scalability:       mode,
		}, 0, 0)
		if err != nil {
			t.Fatalf("SetWebRTCEncoderSVCConfig(%s): %v", mode, err)
		}
		if normalized.Scalability != mode {
			t.Fatalf("normalized mode=%s want %s", normalized.Scalability, mode)
		}
	}

	modes[0] = internalencoder.ScalabilityModeS3T3h
	if got := EncoderWebRTCScalabilityModes()[0]; got != EncoderScalabilityModeL1T1 {
		t.Fatalf("EncoderWebRTCScalabilityModes aliased caller mutation: %s", got)
	}
	prefix := []EncoderScalabilityMode{internalencoder.ScalabilityModeL3T3}
	appended := internalencoder.AppendWebRTCScalabilityModes(prefix)
	if len(appended) != len(wantNames)+1 || appended[0] != internalencoder.ScalabilityModeL3T3 ||
		appended[1] != EncoderScalabilityModeL1T1 || appended[len(appended)-1] != internalencoder.ScalabilityModeS3T3h {
		t.Fatalf("appended modes=%v", appended)
	}
}

func TestPublicValidateEncoderWebRTCActiveScalabilityModes(t *testing.T) {
	for _, modes := range [][]EncoderScalabilityMode{
		nil,
		{EncoderScalabilityModeL1T1},
		{internalencoder.ScalabilityModeS3T3h},
		{EncoderScalabilityModeL1T3, EncoderScalabilityModeL1T3, EncoderScalabilityModeL1T3},
		{internalencoder.ScalabilityModeL2T3_KEY, internalencoder.ScalabilityModeL3T3_KEY},
		{EncoderScalabilityModeL2T3_KEY_SHIFT},
		{EncoderScalabilityModeL3T2_KEY_SHIFT},
		{EncoderScalabilityModeL3T3_KEY_SHIFT},
	} {
		if err := internalencoder.ValidateWebRTCActiveScalabilityModes(modes); err != nil {
			t.Fatalf("ValidateEncoderWebRTCActiveScalabilityModes(%v): %v", modes, err)
		}
	}

	for _, modes := range [][]EncoderScalabilityMode{
		{EncoderScalabilityMode(255)},
		{EncoderScalabilityModeL1T3, internalencoder.ScalabilityModeS2T1},
		{internalencoder.ScalabilityModeS2T3h, internalencoder.ScalabilityModeS3T3},
	} {
		if err := internalencoder.ValidateWebRTCActiveScalabilityModes(modes); !errors.Is(err, internalencoder.ErrInvalidConfig) {
			t.Fatalf("ValidateEncoderWebRTCActiveScalabilityModes(%v) err=%v want %v", modes, err, internalencoder.ErrInvalidConfig)
		}
	}
}

func TestPublicEncoderWebRTCScalabilityModeIDC(t *testing.T) {
	want := map[string]struct {
		idc uint8
		ok  bool
	}{
		"L1T1":           {},
		"L1T2":           {idc: internalobu.ScalabilityModeL1T2, ok: true},
		"L1T3":           {idc: internalobu.ScalabilityModeL1T3, ok: true},
		"L2T1":           {idc: internalobu.ScalabilityModeL2T1, ok: true},
		"L2T1h":          {idc: internalobu.ScalabilityModeL2T1h, ok: true},
		"L2T1_KEY":       {},
		"L2T2":           {idc: internalobu.ScalabilityModeL2T2, ok: true},
		"L2T2h":          {idc: internalobu.ScalabilityModeL2T2h, ok: true},
		"L2T2_KEY":       {idc: internalobu.ScalabilityModeL3T2_KEY, ok: true},
		"L2T2_KEY_SHIFT": {idc: internalobu.ScalabilityModeL3T2_KEY_SHIFT, ok: true},
		"L2T3":           {idc: internalobu.ScalabilityModeL2T3, ok: true},
		"L2T3h":          {idc: internalobu.ScalabilityModeL2T3h, ok: true},
		"L2T3_KEY":       {idc: internalobu.ScalabilityModeL3T3_KEY, ok: true},
		"L2T3_KEY_SHIFT": {idc: internalobu.ScalabilityModeL3T3_KEY_SHIFT, ok: true},
		"L3T1":           {idc: internalobu.ScalabilityModeL3T1, ok: true},
		"L3T1h":          {},
		"L3T1_KEY":       {},
		"L3T2":           {idc: internalobu.ScalabilityModeL3T2, ok: true},
		"L3T2h":          {},
		"L3T2_KEY":       {idc: internalobu.ScalabilityModeL4T5_KEY, ok: true},
		"L3T2_KEY_SHIFT": {idc: internalobu.ScalabilityModeL4T5_KEY_SHIFT, ok: true},
		"L3T3":           {idc: internalobu.ScalabilityModeL3T3, ok: true},
		"L3T3h":          {},
		"L3T3_KEY":       {idc: internalobu.ScalabilityModeL4T7_KEY, ok: true},
		"L3T3_KEY_SHIFT": {idc: internalobu.ScalabilityModeL4T7_KEY_SHIFT, ok: true},
		"S2T1":           {idc: internalobu.ScalabilityModeS2T1, ok: true},
		"S2T1h":          {idc: internalobu.ScalabilityModeS2T1h, ok: true},
		"S2T2":           {idc: internalobu.ScalabilityModeS2T2, ok: true},
		"S2T2h":          {idc: internalobu.ScalabilityModeS2T2h, ok: true},
		"S2T3":           {idc: internalobu.ScalabilityModeS2T3, ok: true},
		"S2T3h":          {idc: internalobu.ScalabilityModeS2T3h, ok: true},
		"S3T1":           {idc: internalobu.ScalabilityModeS3T1, ok: true},
		"S3T1h":          {},
		"S3T2":           {idc: internalobu.ScalabilityModeS3T2, ok: true},
		"S3T2h":          {},
		"S3T3":           {idc: internalobu.ScalabilityModeS3T3, ok: true},
		"S3T3h":          {},
	}
	for _, mode := range internalencoder.AppendWebRTCSVCScalabilityModes(nil) {
		tc, ok := want[mode.String()]
		if !ok {
			t.Fatalf("missing expected IDC entry for %s", mode)
		}
		got, gotOK := internalencoder.WebRTCScalabilityModeIDC(mode)
		if got != tc.idc || gotOK != tc.ok {
			t.Fatalf("EncoderWebRTCScalabilityModeIDC(%s)=%d,%v want %d,%v", mode, got, gotOK, tc.idc, tc.ok)
		}
		got, gotOK = mode.AV1ScalabilityModeIDC()
		if got != tc.idc || gotOK != tc.ok {
			t.Fatalf("%s.AV1ScalabilityModeIDC()=%d,%v want %d,%v", mode, got, gotOK, tc.idc, tc.ok)
		}
	}
	if got, ok := internalencoder.WebRTCScalabilityModeIDC(EncoderScalabilityMode(255)); got != 0 || ok {
		t.Fatalf("invalid mode IDC=%d,%v want 0,false", got, ok)
	}
}

func TestPublicMetadataScalabilityModeIDCConstants(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  uint8
		want uint8
	}{
		{name: "L1T2", got: internalobu.ScalabilityModeL1T2, want: 0},
		{name: "L1T3", got: internalobu.ScalabilityModeL1T3, want: 1},
		{name: "L2T1", got: internalobu.ScalabilityModeL2T1, want: 2},
		{name: "L2T2", got: internalobu.ScalabilityModeL2T2, want: 3},
		{name: "L2T3", got: internalobu.ScalabilityModeL2T3, want: 4},
		{name: "S2T1", got: internalobu.ScalabilityModeS2T1, want: 5},
		{name: "S2T2", got: internalobu.ScalabilityModeS2T2, want: 6},
		{name: "S2T3", got: internalobu.ScalabilityModeS2T3, want: 7},
		{name: "L2T1h", got: internalobu.ScalabilityModeL2T1h, want: 8},
		{name: "L2T2h", got: internalobu.ScalabilityModeL2T2h, want: 9},
		{name: "L2T3h", got: internalobu.ScalabilityModeL2T3h, want: 10},
		{name: "S2T1h", got: internalobu.ScalabilityModeS2T1h, want: 11},
		{name: "S2T2h", got: internalobu.ScalabilityModeS2T2h, want: 12},
		{name: "S2T3h", got: internalobu.ScalabilityModeS2T3h, want: 13},
		{name: "SS", got: internalobu.ScalabilityModeSS, want: 14},
		{name: "L3T1", got: internalobu.ScalabilityModeL3T1, want: 15},
		{name: "L3T2", got: internalobu.ScalabilityModeL3T2, want: 16},
		{name: "L3T3", got: internalobu.ScalabilityModeL3T3, want: 17},
		{name: "S3T1", got: internalobu.ScalabilityModeS3T1, want: 18},
		{name: "S3T2", got: internalobu.ScalabilityModeS3T2, want: 19},
		{name: "S3T3", got: internalobu.ScalabilityModeS3T3, want: 20},
		{name: "L3T2_KEY", got: internalobu.ScalabilityModeL3T2_KEY, want: 21},
		{name: "L3T3_KEY", got: internalobu.ScalabilityModeL3T3_KEY, want: 22},
		{name: "L4T5_KEY", got: internalobu.ScalabilityModeL4T5_KEY, want: 23},
		{name: "L4T7_KEY", got: internalobu.ScalabilityModeL4T7_KEY, want: 24},
		{name: "L3T2_KEY_SHIFT", got: internalobu.ScalabilityModeL3T2_KEY_SHIFT, want: 25},
		{name: "L3T3_KEY_SHIFT", got: internalobu.ScalabilityModeL3T3_KEY_SHIFT, want: 26},
		{name: "L4T5_KEY_SHIFT", got: internalobu.ScalabilityModeL4T5_KEY_SHIFT, want: 27},
		{name: "L4T7_KEY_SHIFT", got: internalobu.ScalabilityModeL4T7_KEY_SHIFT, want: 28},
	} {
		if tc.got != tc.want {
			t.Fatalf("%s IDC=%d want %d", tc.name, tc.got, tc.want)
		}
	}
}

func TestPublicEncoderWebRTCScalabilityMetadataOBU(t *testing.T) {
	size, ok, err := internalencoder.LowOverheadWebRTCScalabilityMetadataOBUSize(internalencoder.ScalabilityModeL2T2)
	if err != nil || !ok {
		t.Fatalf("EncoderLowOverheadWebRTCScalabilityMetadataOBUSize: size=%d ok=%v err=%v", size, ok, err)
	}
	var buf [8]byte
	out, ok, err := internalencoder.AppendLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], internalencoder.ScalabilityModeL2T2)
	if err != nil || !ok {
		t.Fatalf("AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
	}
	if len(out) != size {
		t.Fatalf("len=%d want %d", len(out), size)
	}
	unit, consumed, err := internalobu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != internalobu.TypeMetadata {
		t.Fatalf("metadata consumed=%d header=%+v", consumed, unit.Header)
	}
	meta, err := internalobu.ParseMetadata(unit.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != internalobu.MetadataTypeScalability || meta.Scalability.ModeIDC != internalobu.ScalabilityModeL2T2 || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v", meta)
	}

	ssSize, ok, err := internalencoder.LowOverheadWebRTCScalabilityMetadataOBUSize(internalencoder.ScalabilityModeS3T3h)
	if err != nil || !ok {
		t.Fatalf("EncoderLowOverheadWebRTCScalabilityMetadataOBUSize S3T3h: size=%d ok=%v err=%v", ssSize, ok, err)
	}
	var ssBuf [64]byte
	ssOut, ok, err := internalencoder.AppendLowOverheadWebRTCScalabilityMetadataOBU(ssBuf[:0], internalencoder.ScalabilityModeS3T3h)
	if err != nil || !ok {
		t.Fatalf("AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU S3T3h: ok=%v err=%v", ok, err)
	}
	if len(ssOut) != ssSize {
		t.Fatalf("S3T3h len=%d want %d", len(ssOut), ssSize)
	}
	ssUnit, _, err := internalobu.ParseLowOverhead(ssOut)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU S3T3h: %v", err)
	}
	ssMeta, err := internalobu.ParseMetadata(ssUnit.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU S3T3h: %v", err)
	}
	if ssMeta.Type != internalobu.MetadataTypeScalability ||
		ssMeta.Scalability.ModeIDC != internalobu.ScalabilityModeSS ||
		!ssMeta.Scalability.HasStructure {
		t.Fatalf("S3T3h metadata=%+v", ssMeta)
	}
	ss := ssMeta.Scalability.Structure
	if ss.SpatialLayersCountMinus1 != 2 ||
		ss.SpatialLayerDimensionsPresent ||
		!ss.SpatialLayerDescriptionPresent ||
		!ss.TemporalGroupDescriptionPresent ||
		ss.SpatialLayerRefID[0] != 0xff ||
		ss.SpatialLayerRefID[1] != 0xff ||
		ss.SpatialLayerRefID[2] != 0xff ||
		ss.TemporalGroupSize != 4 ||
		ss.TemporalGroup[0].TemporalID != 0 ||
		ss.TemporalGroup[0].RefCount != 1 ||
		ss.TemporalGroup[0].RefPicDiff[0] != 4 ||
		ss.TemporalGroup[1].TemporalID != 2 ||
		ss.TemporalGroup[1].RefCount != 1 ||
		ss.TemporalGroup[1].RefPicDiff[0] != 1 ||
		ss.TemporalGroup[2].TemporalID != 1 ||
		ss.TemporalGroup[2].RefCount != 1 ||
		ss.TemporalGroup[2].RefPicDiff[0] != 2 ||
		ss.TemporalGroup[2].TemporalSwitchingUp ||
		ss.TemporalGroup[3].TemporalID != 2 ||
		ss.TemporalGroup[3].RefCount != 1 ||
		ss.TemporalGroup[3].RefPicDiff[0] != 1 {
		t.Fatalf("S3T3h structure=%+v", ss)
	}

	prefix := []byte{0xee}
	noMetadata, ok, err := internalencoder.AppendLowOverheadWebRTCScalabilityMetadataOBU(prefix, EncoderScalabilityModeL1T1)
	if err != nil || ok || !bytes.Equal(noMetadata, prefix) {
		t.Fatalf("L1T1 metadata out=% x ok=%v err=%v", noMetadata, ok, err)
	}
}

func TestPublicEncoderWebRTCActiveDecodeTargetsMask(t *testing.T) {
	for _, mode := range internalencoder.AppendWebRTCSVCScalabilityModes(nil) {
		t.Run(mode.String(), func(t *testing.T) {
			spatialLayers, temporalLayers, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid mode %s", mode)
			}
			normalized, err := SetWebRTCEncoderSVCConfig(EncoderConfig{
				Resolution:        EncoderResolution{Width: 1280, Height: 720},
				MaxFramerate:      EncoderRational{Num: 30, Den: 1},
				MinBitrateKbps:    120,
				MaxBitrateKbps:    1600,
				TargetBitrateKbps: 900,
				Scalability:       mode,
			}, 0, 0)
			if err != nil {
				t.Fatalf("SetWebRTCEncoderSVCConfig(%s): %v", mode, err)
			}
			structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(normalized)
			if err != nil {
				t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", mode, err)
			}
			all, err := internalencoder.WebRTCAllDecodeTargetsMask(structure)
			if err != nil {
				t.Fatalf("EncoderWebRTCAllDecodeTargetsMask(%s): %v", mode, err)
			}
			wantTargets := int(spatialLayers * temporalLayers)
			if bits.OnesCount32(all) != wantTargets {
				t.Fatalf("all mask=%#x targets=%d want %d", all, bits.OnesCount32(all), wantTargets)
			}
			top, err := internalencoder.WebRTCActiveDecodeTargetsMask(structure, spatialLayers-1, temporalLayers-1)
			if err != nil {
				t.Fatalf("EncoderWebRTCActiveDecodeTargetsMask top(%s): %v", mode, err)
			}
			if top != all {
				t.Fatalf("top mask=%#x want all %#x", top, all)
			}
			baseSpatial, err := internalencoder.WebRTCActiveDecodeTargetsMask(structure, 0, temporalLayers-1)
			if err != nil {
				t.Fatalf("EncoderWebRTCActiveDecodeTargetsMask base spatial(%s): %v", mode, err)
			}
			if bits.OnesCount32(baseSpatial) != int(temporalLayers) {
				t.Fatalf("base spatial mask=%#x targets=%d want %d", baseSpatial, bits.OnesCount32(baseSpatial), temporalLayers)
			}
			baseTemporal, err := internalencoder.WebRTCActiveDecodeTargetsMask(structure, spatialLayers-1, 0)
			if err != nil {
				t.Fatalf("EncoderWebRTCActiveDecodeTargetsMask base temporal(%s): %v", mode, err)
			}
			if bits.OnesCount32(baseTemporal) != int(spatialLayers) {
				t.Fatalf("base temporal mask=%#x targets=%d want %d", baseTemporal, bits.OnesCount32(baseTemporal), spatialLayers)
			}
			if _, err := internalencoder.WebRTCActiveDecodeTargetsMask(structure, EncoderWebRTCMaxSpatialLayers, 0); !errors.Is(err, ErrEncoderInvalidFrame) {
				t.Fatalf("invalid spatial mask err=%v want %v", err, ErrEncoderInvalidFrame)
			}
			if _, err := internalencoder.WebRTCActiveDecodeTargetsMask(structure, 0, internalencoder.WebRTCMaxTemporalLayers); !errors.Is(err, ErrEncoderInvalidFrame) {
				t.Fatalf("invalid temporal mask err=%v want %v", err, ErrEncoderInvalidFrame)
			}
		})
	}
}

func TestPublicEncoderWebRTCSpatialDecodeTargetsMask(t *testing.T) {
	for _, mode := range internalencoder.AppendWebRTCSVCScalabilityModes(nil) {
		t.Run(mode.String(), func(t *testing.T) {
			spatialLayers, temporalLayers, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid mode %s", mode)
			}
			normalized, err := SetWebRTCEncoderSVCConfig(EncoderConfig{
				Resolution:        EncoderResolution{Width: 1280, Height: 720},
				MaxFramerate:      EncoderRational{Num: 30, Den: 1},
				MinBitrateKbps:    120,
				MaxBitrateKbps:    1600,
				TargetBitrateKbps: 900,
				Scalability:       mode,
			}, 0, 0)
			if err != nil {
				t.Fatalf("SetWebRTCEncoderSVCConfig(%s): %v", mode, err)
			}
			structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(normalized)
			if err != nil {
				t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", mode, err)
			}
			for spatialID := uint8(0); spatialID < spatialLayers; spatialID++ {
				mask, err := internalencoder.WebRTCSpatialDecodeTargetsMask(structure, spatialID, temporalLayers-1)
				if err != nil {
					t.Fatalf("EncoderWebRTCSpatialDecodeTargetsMask(%s,S%d): %v", mode, spatialID, err)
				}
				if bits.OnesCount32(mask) != int(temporalLayers) {
					t.Fatalf("%s S%d mask=%#x targets=%d want %d",
						mode, spatialID, mask, bits.OnesCount32(mask), temporalLayers)
				}
				for temporalID := uint8(0); temporalID < temporalLayers; temporalID++ {
					target := uint(spatialID*temporalLayers + temporalID)
					if mask&(uint32(1)<<target) == 0 {
						t.Fatalf("%s S%d mask=%#x missing T%d target bit %d", mode, spatialID, mask, temporalID, target)
					}
				}
			}
			if _, err := internalencoder.WebRTCSpatialDecodeTargetsMask(structure, spatialLayers, 0); !errors.Is(err, ErrEncoderInvalidFrame) {
				t.Fatalf("invalid spatial mask err=%v want %v", err, ErrEncoderInvalidFrame)
			}
			if _, err := internalencoder.WebRTCSpatialDecodeTargetsMask(structure, 0, temporalLayers); !errors.Is(err, ErrEncoderInvalidFrame) {
				t.Fatalf("invalid temporal mask err=%v want %v", err, ErrEncoderInvalidFrame)
			}
		})
	}
}

func TestPublicEncoderLowOverheadOBU(t *testing.T) {
	unit := internalencoder.OBU{
		Type:       internalobu.TypeFrame,
		TemporalID: 1,
		SpatialID:  1,
		Payload:    []byte{0xaa, 0xbb},
	}
	size, err := internalencoder.LowOverheadOBUSize(unit)
	if err != nil {
		t.Fatalf("EncoderLowOverheadOBUSize: %v", err)
	}
	var buf [8]byte
	out, err := internalencoder.AppendLowOverheadOBU(buf[:0], unit)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("encoded len=%d want %d", len(out), size)
	}
	parsed, consumed, err := internalobu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || parsed.Header.Type != internalobu.TypeFrame || parsed.Header.TemporalID != 1 || parsed.Header.SpatialID != 1 {
		t.Fatalf("parsed header=%+v consumed=%d", parsed.Header, consumed)
	}
	if !bytes.Equal(parsed.Payload, unit.Payload) {
		t.Fatalf("payload=% x want % x", parsed.Payload, unit.Payload)
	}
}

func TestPublicEncoderSequenceHeaderForConfig(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution:  EncoderResolution{Width: 640, Height: 360},
		Scalability: internalencoder.ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	if seq.OperatingPointsCount != 4 || seq.MaxFrameWidth != 640 || seq.MaxFrameHeight != 360 {
		t.Fatalf("sequence header = %+v", seq)
	}
	var buf [160]byte
	out, err := internalencoder.AppendLowOverheadSequenceHeaderOBU(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadSequenceHeaderOBU: %v", err)
	}
	unit, consumed, err := internalobu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != internalobu.TypeSequenceHeader {
		t.Fatalf("parsed header=%+v consumed=%d", unit.Header, consumed)
	}
	if _, err := internalparser.ParseSequenceHeader(unit.Payload); err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
}

func TestPublicEncoderFrameHeaderPrefixPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := internalencoder.FrameHeaderPrefix{
		FrameType:               internalencoder.FrameHeaderTypeKey,
		ShowFrame:               true,
		ShowableFrame:           false,
		ErrorResilientMode:      true,
		DisableCDFUpdate:        true,
		AllowScreenContentTools: false,
		ForceIntegerMV:          true,
		OrderHint:               5,
		PrimaryRefFrame:         internalencoder.EncoderPrimaryRefNone,
	}
	size, err := internalencoder.FrameHeaderPrefixPayloadSize(seq, prefix)
	if err != nil {
		t.Fatalf("EncoderFrameHeaderPrefixPayloadSize: %v", err)
	}
	var buf [8]byte
	out, err := internalencoder.AppendFrameHeaderPrefixPayload(buf[:0], seq, prefix)
	if err != nil {
		t.Fatalf("AppendEncoderFrameHeaderPrefixPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("frame prefix len=%d want %d", len(out), size)
	}
	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsed, err := internalparser.ParseFrameHeaderPrefix(out, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	if parsed.FrameType != internalparser.FrameTypeKey || !parsed.ShowFrame || parsed.OrderHint != 5 {
		t.Fatalf("parsed frame prefix=%+v", parsed)
	}
}

func TestPublicEncoderFrameHeaderIntraPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := internalencoder.FrameHeaderPrefix{
		FrameType:          internalencoder.FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		OrderHint:          5,
		PrimaryRefFrame:    internalencoder.EncoderPrimaryRefNone,
	}
	size := internalencoder.IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	payloadSize, err := internalencoder.FrameHeaderIntraPayloadSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderFrameHeaderIntraPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := internalencoder.AppendFrameHeaderIntraPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderFrameHeaderIntraPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsedPrefix, err := internalparser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := internalparser.ParseIntraFrameSize(payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != 640 || parsedSize.Height != 360 {
		t.Fatalf("parsed size=%+v", parsedSize)
	}
}

func TestPublicEncoderTileGroupPayload(t *testing.T) {
	tiles := internalencoder.TileInfo{
		Cols:          2,
		Rows:          1,
		Log2Cols:      1,
		TileSizeBytes: 1,
	}
	payloads := [...]internalencoder.TilePayload{
		{Data: []byte{0x80}},
		{Data: []byte{0x81, 0x82}},
	}
	size, err := internalencoder.TileGroupPayloadSize(tiles, 0, 1, payloads[:])
	if err != nil {
		t.Fatalf("EncoderTileGroupPayloadSize: %v", err)
	}
	var buf [8]byte
	out, err := internalencoder.AppendTileGroupPayload(buf[:0], tiles, 0, 1, payloads[:])
	if err != nil {
		t.Fatalf("AppendEncoderTileGroupPayload: %v", err)
	}
	if len(out) != size {
		t.Fatalf("payload len=%d want %d", len(out), size)
	}
}

func TestPublicEncoderFrameHeaderInterPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := internalencoder.FrameHeaderPrefix{
		FrameType:          internalencoder.FrameHeaderTypeInter,
		ShowFrame:          true,
		ShowableFrame:      true,
		ErrorResilientMode: true,
		FrameSizeOverride:  true,
		OrderHint:          6,
		PrimaryRefFrame:    internalencoder.EncoderPrimaryRefNone,
	}
	size := internalencoder.InterFrameSize{
		UpscaledWidth:       320,
		Height:              180,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0x01,
	}
	payloadSize, err := internalencoder.FrameHeaderInterPayloadSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderFrameHeaderInterPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := internalencoder.AppendFrameHeaderInterPayload(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderFrameHeaderInterPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}

	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsedPrefix, err := internalparser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	refs := publicEncoderReferenceState(seq, size.RefFrameIdx[:])
	parsedSize, err := internalparser.ParseFrameSize(payload, parsedSeq, parsedPrefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	if parsedSize.RefreshFrameFlags != 0x01 || parsedSize.UpscaledWidth != 320 || parsedSize.Height != 180 {
		t.Fatalf("parsed inter size=%+v", parsedSize)
	}

	obuSize, err := internalencoder.LowOverheadFrameHeaderInterOBUSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderLowOverheadFrameHeaderInterOBUSize: %v", err)
	}
	var obuBuf [40]byte
	obuOut, err := internalencoder.AppendLowOverheadFrameHeaderInterOBU(obuBuf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadFrameHeaderInterOBU: %v", err)
	}
	if len(obuOut) != obuSize {
		t.Fatalf("inter frame-header obu len=%d want %d", len(obuOut), obuSize)
	}
}

func TestPublicEncoderQuantizationParamsPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.ColorConfig.SeparateUVDeltaQ = true
	quant := internalencoder.QuantizationParams{
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
	payloadSize, err := internalencoder.QuantizationParamsPayloadSize(seq, quant)
	if err != nil {
		t.Fatalf("EncoderQuantizationParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := internalencoder.AppendQuantizationParamsPayload(buf[:0], seq, quant)
	if err != nil {
		t.Fatalf("AppendEncoderQuantizationParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}

	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsed, err := internalparser.ParseQuantizationParams(payload, parsedSeq, internalparser.TileInfo{})
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
	prefix := internalencoder.FrameHeaderPrefix{PrimaryRefFrame: internalencoder.EncoderPrimaryRefNone}
	var data internalencoder.SegmentationData
	for i := 0; i < 8; i++ {
		data.Segments[i].RefFrame = -1
	}
	data.Segments[0].DeltaQ = -8
	data.Segments[2].RefFrame = 4
	seg := internalencoder.SegmentationParams{
		Enabled:    true,
		UpdateMap:  true,
		UpdateData: true,
		Data:       data,
	}
	payloadSize, err := internalencoder.SegmentationParamsPayloadSize(prefix, seg)
	if err != nil {
		t.Fatalf("EncoderSegmentationParamsPayloadSize: %v", err)
	}
	var buf [96]byte
	payload, err := internalencoder.AppendSegmentationParamsPayload(buf[:0], prefix, seg)
	if err != nil {
		t.Fatalf("AppendEncoderSegmentationParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseSegmentationParams(
		payload,
		internalparser.FrameHeaderPrefix{PrimaryRefFrame: internalparser.PrimaryRefNone},
		internalparser.QuantizationParams{BaseQIdx: 20},
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
	size := internalencoder.IntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	quant := internalencoder.QuantizationParams{BaseQIdx: 37}
	delta := internalencoder.DeltaParams{
		DeltaQPresent:  true,
		DeltaQResLog2:  1,
		DeltaLFPresent: true,
		DeltaLFResLog2: 2,
		DeltaLFMulti:   true,
	}
	payloadSize, err := internalencoder.DeltaParamsPayloadSize(size, quant, delta)
	if err != nil {
		t.Fatalf("EncoderDeltaParamsPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := internalencoder.AppendDeltaParamsPayload(buf[:0], size, quant, delta)
	if err != nil {
		t.Fatalf("AppendEncoderDeltaParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseDeltaParams(
		payload,
		FrameSize{},
		internalparser.QuantizationParams{BaseQIdx: quant.BaseQIdx},
		internalparser.SegmentationParams{},
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
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := internalencoder.FrameHeaderPrefix{PrimaryRefFrame: internalencoder.EncoderPrimaryRefNone}
	size := internalencoder.IntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	lf := internalencoder.LoopFilterParams{
		LevelY:              [2]uint8{10, 0},
		LevelU:              5,
		LevelV:              6,
		Sharpness:           3,
		ModeRefDeltaEnabled: true,
		ModeRefDeltaUpdate:  true,
		Deltas: internalencoder.LoopFilterDeltas{
			Ref:  [8]int8{-2, 0, 0, 0, -1, 0, -1, -1},
			Mode: [2]int8{0, 3},
		},
	}
	payloadSize, err := internalencoder.LoopFilterParamsPayloadSize(seq, prefix, size, false, lf, nil)
	if err != nil {
		t.Fatalf("EncoderLoopFilterParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := internalencoder.AppendLoopFilterParamsPayload(buf[:0], seq, prefix, size, false, lf, nil)
	if err != nil {
		t.Fatalf("AppendEncoderLoopFilterParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsed, err := internalparser.ParseLoopFilterParams(
		payload,
		parsedSeq,
		internalparser.FrameHeaderPrefix{PrimaryRefFrame: internalparser.PrimaryRefNone},
		FrameSize{},
		internalparser.SegmentationParams{},
		internalparser.DeltaParams{},
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
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	size := internalencoder.IntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	cdef := internalencoder.CDEFParams{Damping: 5, Bits: 2}
	for i := uint8(0); i < 4; i++ {
		cdef.YStrength[i] = 10 + i
		cdef.UVStrength[i] = 20 + i
	}
	payloadSize, err := internalencoder.CDEFParamsPayloadSize(seq, size, false, cdef)
	if err != nil {
		t.Fatalf("EncoderCDEFParamsPayloadSize: %v", err)
	}
	var buf [16]byte
	payload, err := internalencoder.AppendCDEFParamsPayload(buf[:0], seq, size, false, cdef)
	if err != nil {
		t.Fatalf("AppendEncoderCDEFParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsed, err := internalparser.ParseCDEFParams(payload, parsedSeq, FrameSize{}, internalparser.SegmentationParams{}, internalparser.LoopFilterParams{})
	if err != nil {
		t.Fatalf("ParseCDEFParams: %v", err)
	}
	if parsed.Damping != 5 || parsed.Bits != 2 || parsed.StrengthCount != 4 ||
		parsed.YStrength[3] != 13 || parsed.UVStrength[3] != 23 {
		t.Fatalf("parsed cdef=%+v", parsed)
	}
}

func TestPublicEncoderRestorationParamsPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.EnableRestoration = true
	size := internalencoder.IntraFrameSize{
		UpscaledWidth:       640,
		Height:              360,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}
	restoration := internalencoder.RestorationParams{
		Type:           [3]internalencoder.RestorationType{internalencoder.RestorationWiener, internalencoder.RestorationSGRProj, internalencoder.RestorationNone},
		UnitSizeYLog2:  7,
		UnitSizeUVLog2: 6,
		UnitSizeY:      128,
		UnitSizeUV:     64,
	}
	payloadSize, err := internalencoder.RestorationParamsPayloadSize(seq, size, false, restoration)
	if err != nil {
		t.Fatalf("EncoderRestorationParamsPayloadSize: %v", err)
	}
	var buf [8]byte
	payload, err := internalencoder.AppendRestorationParamsPayload(buf[:0], seq, size, false, restoration)
	if err != nil {
		t.Fatalf("AppendEncoderRestorationParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	parsed, err := internalparser.ParseRestorationParams(payload, parsedSeq, FrameSize{}, internalparser.SegmentationParams{}, internalparser.CDEFParams{})
	if err != nil {
		t.Fatalf("ParseRestorationParams: %v", err)
	}
	if parsed.Type[0] != internalparser.RestorationWiener || parsed.Type[1] != internalparser.RestorationSGRProj ||
		parsed.UnitSizeY != 128 || parsed.UnitSizeUV != 64 {
		t.Fatalf("parsed restoration=%+v", parsed)
	}
}

func TestPublicEncoderTransformReferenceParamsPayload(t *testing.T) {
	prefix := internalencoder.FrameHeaderPrefix{FrameType: internalencoder.FrameHeaderTypeInter}
	params := internalencoder.TransformReferenceParams{
		TransformMode: internalencoder.TransformModeSwitchable,
		ReferenceMode: internalencoder.ReferenceModeSelect,
	}
	payloadSize, err := internalencoder.TransformReferenceParamsPayloadSize(prefix, false, params)
	if err != nil {
		t.Fatalf("EncoderTransformReferenceParamsPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := internalencoder.AppendTransformReferenceParamsPayload(buf[:0], prefix, false, params)
	if err != nil {
		t.Fatalf("AppendEncoderTransformReferenceParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseTransformReferenceParams(
		payload,
		internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeInter},
		internalparser.SegmentationParams{},
		internalparser.RestorationParams{},
	)
	if err != nil {
		t.Fatalf("ParseTransformReferenceParams: %v", err)
	}
	if parsed.TransformMode != internalparser.TransformModeSwitchable || parsed.ReferenceMode != internalparser.ReferenceModeSelect {
		t.Fatalf("parsed transform/reference=%+v", parsed)
	}
}

func TestPublicEncoderSkipModeParamsPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.EnableOrderHint = true
	seq.OrderHintBits = 5
	prefix := internalencoder.FrameHeaderPrefix{FrameType: internalencoder.FrameHeaderTypeInter, OrderHint: 16}
	var size internalencoder.InterFrameSize
	var refs internalparser.ReferenceState
	orderHints := [7]uint8{15, 17, 14, 18, 13, 19, 12}
	for i := uint8(0); i < internalparser.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = internalparser.ReferenceFrame{Valid: true, OrderHint: orderHints[i]}
	}
	transformRef := internalencoder.TransformReferenceParams{ReferenceMode: internalencoder.ReferenceModeSelect}
	params := internalencoder.SkipModeParams{Allowed: true, Enabled: true, RefFrameIdx: [2]uint8{0, 1}}
	payloadSize, err := internalencoder.SkipModeParamsPayloadSize(seq, prefix, size, &refs, transformRef, params)
	if err != nil {
		t.Fatalf("EncoderSkipModeParamsPayloadSize: %v", err)
	}
	var buf [1]byte
	payload, err := internalencoder.AppendSkipModeParamsPayload(buf[:0], seq, prefix, size, &refs, transformRef, params)
	if err != nil {
		t.Fatalf("AppendEncoderSkipModeParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseSkipModeParams(
		payload,
		SequenceHeader{EnableOrderHint: true, OrderHintBits: 5},
		internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeInter, OrderHint: 16},
		FrameSize{RefFrameIdx: size.RefFrameIdx},
		&refs,
		internalparser.TransformReferenceParams{ReferenceMode: internalparser.ReferenceModeSelect},
	)
	if err != nil {
		t.Fatalf("ParseSkipModeParams: %v", err)
	}
	if !parsed.Allowed || !parsed.Enabled || parsed.RefFrameIdx != [2]uint8{0, 1} {
		t.Fatalf("parsed skip mode=%+v", parsed)
	}
}

func TestPublicEncoderFrameModeParamsPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.EnableWarpedMotion = true
	prefix := internalencoder.FrameHeaderPrefix{FrameType: internalencoder.FrameHeaderTypeInter}
	params := internalencoder.FrameModeParams{AllowWarpedMotion: true, ReducedTxSet: true}
	payloadSize, err := internalencoder.FrameModeParamsPayloadSize(seq, prefix, params)
	if err != nil {
		t.Fatalf("EncoderFrameModeParamsPayloadSize: %v", err)
	}
	var buf [1]byte
	payload, err := internalencoder.AppendFrameModeParamsPayload(buf[:0], seq, prefix, params)
	if err != nil {
		t.Fatalf("AppendEncoderFrameModeParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseFrameModeParams(
		payload,
		SequenceHeader{EnableWarpedMotion: true},
		internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeInter},
		internalparser.SkipModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseFrameModeParams: %v", err)
	}
	if !parsed.AllowWarpedMotion || !parsed.ReducedTxSet {
		t.Fatalf("parsed frame mode=%+v", parsed)
	}
}

func TestPublicEncoderGlobalMotionParamsPayload(t *testing.T) {
	prefix := internalencoder.FrameHeaderPrefix{FrameType: internalencoder.FrameHeaderTypeInter, PrimaryRefFrame: internalencoder.EncoderPrimaryRefNone}
	var size internalencoder.InterFrameSize
	var refs internalparser.ReferenceState
	for i := uint8(0); i < internalparser.InterRefsPerFrame; i++ {
		size.RefFrameIdx[i] = i
		refs.Frames[i] = internalparser.ReferenceFrame{Valid: true}
	}
	params := internalencoder.DefaultGlobalMotionParams()
	params.Ref[0] = internalencoder.DefaultWarpedMotionParams()
	params.Ref[0].Type = internalencoder.GlobalMotionTranslation
	params.Ref[0].Matrix[0] = 2 << 13
	payloadSize, err := internalencoder.GlobalMotionParamsPayloadSize(prefix, size, internalparser.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	if err != nil {
		t.Fatalf("EncoderGlobalMotionParamsPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := internalencoder.AppendGlobalMotionParamsPayload(buf[:0], prefix, size, internalparser.TileInfo{AllowHighPrecisionMV: true}, &refs, params)
	if err != nil {
		t.Fatalf("AppendEncoderGlobalMotionParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseGlobalMotionParams(
		payload,
		internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeInter, PrimaryRefFrame: internalparser.PrimaryRefNone},
		FrameSize{RefFrameIdx: size.RefFrameIdx},
		internalparser.TileInfo{AllowHighPrecisionMV: true},
		&refs,
		internalparser.FrameModeParams{},
	)
	if err != nil {
		t.Fatalf("ParseGlobalMotionParams: %v", err)
	}
	if parsed.Ref[0].Type != internalparser.GlobalMotionTranslation || parsed.Ref[0].Matrix[0] != params.Ref[0].Matrix[0] {
		t.Fatalf("parsed global motion ref[0]=%+v", parsed.Ref[0])
	}
}

func TestPublicEncoderFilmGrainParamsPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.FilmGrainParamsPresent = true
	prefix := internalencoder.FrameHeaderPrefix{FrameType: internalencoder.FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	params := internalencoder.FilmGrainParams{
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
	payloadSize, err := internalencoder.FilmGrainParamsPayloadSize(seq, prefix, internalencoder.InterFrameSize{}, nil, params)
	if err != nil {
		t.Fatalf("EncoderFilmGrainParamsPayloadSize: %v", err)
	}
	var buf [32]byte
	payload, err := internalencoder.AppendFilmGrainParamsPayload(buf[:0], seq, prefix, internalencoder.InterFrameSize{}, nil, params)
	if err != nil {
		t.Fatalf("AppendEncoderFilmGrainParamsPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseFilmGrainParams(
		payload,
		SequenceHeader{
			FilmGrainParamsPresent: true,
			ColorConfig: internalparser.ColorConfig{
				BitDepth:     8,
				SubsamplingX: true,
				SubsamplingY: true,
			},
		},
		internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey, ShowFrame: true},
		FrameSize{},
		nil,
		internalparser.GlobalMotionParams{},
	)
	if err != nil {
		t.Fatalf("ParseFilmGrainParams: %v", err)
	}
	if !parsed.Apply || !parsed.Update || parsed.Seed != 0x1234 || parsed.YPoints[0] != [2]uint8{10, 20} {
		t.Fatalf("parsed film grain=%+v", parsed)
	}
}

func TestPublicEncoderTileInfoPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := internalencoder.FrameHeaderPrefix{FrameType: internalencoder.FrameHeaderTypeKey, ShowFrame: true, ErrorResilientMode: true}
	tiles := internalencoder.TileInfo{
		RefreshContext: true,
		SBCols:         1,
		SBRows:         1,
		Cols:           1,
		Rows:           1,
		ColStartSB:     [internalencoder.MaxTileCols + 1]uint16{0, 1},
		RowStartSB:     [internalencoder.MaxTileRows + 1]uint16{0, 1},
	}
	payloadSize, err := internalencoder.TileInfoPayloadSize(seq, prefix, 64, 64, tiles)
	if err != nil {
		t.Fatalf("EncoderTileInfoPayloadSize: %v", err)
	}
	var buf [2]byte
	payload, err := internalencoder.AppendTileInfoPayload(buf[:0], seq, prefix, 64, 64, tiles)
	if err != nil {
		t.Fatalf("AppendEncoderTileInfoPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	parsed, err := internalparser.ParseTileInfo(
		payload,
		SequenceHeader{},
		internalparser.FrameHeaderPrefix{FrameType: internalparser.FrameTypeKey, ErrorResilientMode: true},
		FrameSize{CodedWidth: 64, Height: 64},
	)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	if !parsed.RefreshContext || parsed.Cols != 1 || parsed.Rows != 1 {
		t.Fatalf("parsed tiles=%+v", parsed)
	}
}

func TestPublicEncoderIntraFrameHeaderPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.SeqForceScreenContentTools = 1
	seq.SeqForceIntegerMV = 1
	seq.EnableCDEF = true
	seq.EnableRestoration = false
	header := internalencoder.IntraFrameHeaderParams{
		Prefix: internalencoder.FrameHeaderPrefix{
			FrameType:               internalencoder.FrameHeaderTypeKey,
			ShowFrame:               true,
			ErrorResilientMode:      true,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         internalencoder.EncoderPrimaryRefNone,
		},
		Size: internalencoder.IntraFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0xff,
		},
		Tile: internalencoder.TileInfo{
			RefreshContext: true,
			SBCols:         1,
			SBRows:         1,
			Cols:           1,
			Rows:           1,
			ColStartSB:     [internalencoder.MaxTileCols + 1]uint16{0, 1},
			RowStartSB:     [internalencoder.MaxTileRows + 1]uint16{0, 1},
		},
		Quantization: internalencoder.QuantizationParams{BaseQIdx: 50},
		LoopFilter: internalencoder.LoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			ModeRefDeltaEnabled: false,
			Deltas: internalencoder.LoopFilterDeltas{
				Ref: [8]int8{1, 0, 0, 0, -1, 0, -1, -1},
			},
		},
		CDEF: internalencoder.CDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		TransformRef: internalencoder.TransformReferenceParams{
			TransformMode: internalencoder.TransformModeLargest,
			ReferenceMode: internalencoder.ReferenceModeSingle,
		},
	}
	payloadSize, err := internalencoder.IntraFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("EncoderIntraFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := internalencoder.AppendIntraFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("AppendEncoderIntraFrameHeaderPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	prefix, err := internalparser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := internalparser.ParseIntraFrameSize(payload, parsedSeq, prefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	tiles, err := internalparser.ParseTileInfo(payload, parsedSeq, prefix, size)
	if err != nil {
		t.Fatalf("ParseTileInfo: %v", err)
	}
	if prefix.FrameType != internalparser.FrameTypeKey || size.CodedWidth != 64 || tiles.Cols != 1 || tiles.Rows != 1 {
		t.Fatalf("parsed header prefix=%+v size=%+v tiles=%+v", prefix, size, tiles)
	}
}

func TestPublicEncoderInterFrameHeaderPayload(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 64, Height: 64},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	seq.SeqForceScreenContentTools = 1
	seq.SeqForceIntegerMV = 1
	seq.EnableCDEF = true
	var refs internalparser.ReferenceState
	for i := uint8(0); i < internalparser.RefFrames; i++ {
		refs.Frames[i] = internalparser.ReferenceFrame{
			Valid: true,
			Size: FrameSize{
				CodedWidth:          64,
				UpscaledWidth:       64,
				Height:              64,
				RenderWidth:         64,
				RenderHeight:        64,
				SuperResDenominator: 8,
			},
		}
	}
	header := internalencoder.InterFrameHeaderParams{
		Prefix: internalencoder.FrameHeaderPrefix{
			FrameType:               internalencoder.FrameHeaderTypeInter,
			ShowFrame:               true,
			ShowableFrame:           true,
			AllowScreenContentTools: true,
			ForceIntegerMV:          true,
			PrimaryRefFrame:         internalencoder.EncoderPrimaryRefNone,
		},
		Size: internalencoder.InterFrameSize{
			UpscaledWidth:       64,
			Height:              64,
			RenderWidth:         64,
			RenderHeight:        64,
			SuperResDenominator: 8,
			RefreshFrameFlags:   0x02,
		},
		Tile: internalencoder.TileInfo{
			InterpolationFilter: internalencoder.InterpolationEightTap,
			RefreshContext:      true,
			SBCols:              1,
			SBRows:              1,
			Cols:                1,
			Rows:                1,
			ColStartSB:          [internalencoder.MaxTileCols + 1]uint16{0, 1},
			RowStartSB:          [internalencoder.MaxTileRows + 1]uint16{0, 1},
		},
		Quantization: internalencoder.QuantizationParams{BaseQIdx: 50},
		LoopFilter: internalencoder.LoopFilterParams{
			LevelY:              [2]uint8{4, 4},
			ModeRefDeltaEnabled: false,
			Deltas: internalencoder.LoopFilterDeltas{
				Ref: [8]int8{1, 0, 0, 0, -1, 0, -1, -1},
			},
		},
		CDEF: internalencoder.CDEFParams{
			Damping:    3,
			YStrength:  [8]uint8{1},
			UVStrength: [8]uint8{1},
		},
		TransformRef: internalencoder.TransformReferenceParams{
			TransformMode: internalencoder.TransformModeLargest,
			ReferenceMode: internalencoder.ReferenceModeSingle,
		},
		GlobalMotion: internalencoder.DefaultGlobalMotionParams(),
		References:   &refs,
	}
	payloadSize, err := internalencoder.InterFrameHeaderPayloadSize(seq, header)
	if err != nil {
		t.Fatalf("EncoderInterFrameHeaderPayloadSize: %v", err)
	}
	var buf [256]byte
	payload, err := internalencoder.AppendInterFrameHeaderPayload(buf[:0], seq, header)
	if err != nil {
		t.Fatalf("AppendEncoderInterFrameHeaderPayload: %v", err)
	}
	if len(payload) != payloadSize {
		t.Fatalf("payload len=%d want %d", len(payload), payloadSize)
	}
	var seqBuf [128]byte
	seqPayload, err := internalencoder.AppendSequenceHeaderPayload(seqBuf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderSequenceHeaderPayload: %v", err)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqPayload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	prefix, err := internalparser.ParseFrameHeaderPrefix(payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	size, err := internalparser.ParseFrameSize(payload, parsedSeq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	if prefix.FrameType != internalparser.FrameTypeInter || size.RefreshFrameFlags != 0x02 || size.RefFrameIdx[0] != 0 {
		t.Fatalf("parsed inter header prefix=%+v size=%+v", prefix, size)
	}
}

func TestPublicEncoderLowOverheadTemporalUnit(t *testing.T) {
	units := [...]internalencoder.OBU{
		{Type: internalobu.TypeFrame, Payload: []byte{0xaa}},
	}
	size, err := internalencoder.LowOverheadTemporalUnitSize(units[:])
	if err != nil {
		t.Fatalf("EncoderLowOverheadTemporalUnitSize: %v", err)
	}
	var buf [8]byte
	out, err := internalencoder.AppendLowOverheadTemporalUnit(buf[:0], units[:])
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadTemporalUnit: %v", err)
	}
	if len(out) != size {
		t.Fatalf("temporal unit len=%d want %d", len(out), size)
	}
	it := internalobu.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil {
		t.Fatalf("TemporalUnitIterator.Next: %v", err)
	}
	if !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit parsed ok=%v raw=% x", ok, tu.Raw)
	}
}

func TestPublicEncoderLowOverheadIntraHeaderTemporalUnit(t *testing.T) {
	seq, err := internalencoder.SequenceHeaderForConfig(EncoderConfig{
		Resolution: EncoderResolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	prefix := internalencoder.FrameHeaderPrefix{
		FrameType:          internalencoder.FrameHeaderTypeKey,
		ShowFrame:          true,
		ErrorResilientMode: true,
		ForceIntegerMV:     true,
		OrderHint:          5,
		PrimaryRefFrame:    internalencoder.EncoderPrimaryRefNone,
	}
	size := internalencoder.IntraFrameSize{
		UpscaledWidth:       seq.MaxFrameWidth,
		Height:              seq.MaxFrameHeight,
		SuperResDenominator: 8,
		RefreshFrameFlags:   0xff,
	}

	unitSize, err := internalencoder.LowOverheadIntraHeaderTemporalUnitSize(seq, prefix, size)
	if err != nil {
		t.Fatalf("EncoderLowOverheadIntraHeaderTemporalUnitSize: %v", err)
	}
	var buf [192]byte
	out, err := internalencoder.AppendLowOverheadIntraHeaderTemporalUnit(buf[:0], seq, prefix, size)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnit: %v", err)
	}
	if len(out) != unitSize {
		t.Fatalf("temporal unit len=%d want %d", len(out), unitSize)
	}

	it := internalobu.NewLowOverheadIterator(out)
	td, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("temporal delimiter ok=%v err=%v", ok, err)
	}
	if td.Header.Type != internalobu.TypeTemporalDelimiter || len(td.Payload) != 0 {
		t.Fatalf("temporal delimiter=%+v payload=% x", td.Header, td.Payload)
	}
	seqUnit, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("sequence header ok=%v err=%v", ok, err)
	}
	if seqUnit.Header.Type != internalobu.TypeSequenceHeader {
		t.Fatalf("sequence unit type=%v", seqUnit.Header.Type)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqUnit.Payload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	frameUnit, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("frame header ok=%v err=%v", ok, err)
	}
	if frameUnit.Header.Type != internalobu.TypeFrameHeader {
		t.Fatalf("frame unit type=%v", frameUnit.Header.Type)
	}
	parsedPrefix, err := internalparser.ParseFrameHeaderPrefix(frameUnit.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := internalparser.ParseIntraFrameSize(frameUnit.Payload, parsedSeq, parsedPrefix, 0, 0)
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
		_, _ = internalencoder.AppendLowOverheadIntraHeaderTemporalUnit(buf[:0], seq, prefix, size)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnit allocated: %f", allocs)
	}
}

func TestPublicEncoderLowOverheadIntraHeaderTemporalUnitForConfig(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unitSize, wantUnit, err := internalencoder.LowOverheadIntraHeaderTemporalUnitForConfigSize(cfg, 11)
	if err != nil {
		t.Fatalf("EncoderLowOverheadIntraHeaderTemporalUnitForConfigSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, err := internalencoder.AppendLowOverheadIntraHeaderTemporalUnitForConfig(buf[:0], cfg, 11)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnitForConfig: %v", err)
	}
	if len(out) != unitSize || gotUnit != wantUnit {
		t.Fatalf("len=%d want %d gotUnit=%+v wantUnit=%+v", len(out), unitSize, gotUnit, wantUnit)
	}

	it := internalobu.NewLowOverheadIterator(out)
	td, ok, err := it.Next()
	if err != nil || !ok || td.Header.Type != internalobu.TypeTemporalDelimiter || len(td.Payload) != 0 {
		t.Fatalf("temporal delimiter ok=%v err=%v unit=%+v payload=% x", ok, err, td.Header, td.Payload)
	}
	seqUnit, ok, err := it.Next()
	if err != nil || !ok || seqUnit.Header.Type != internalobu.TypeSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v unit=%+v", ok, err, seqUnit.Header)
	}
	parsedSeq, err := internalparser.ParseSequenceHeader(seqUnit.Payload)
	if err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
	if parsedSeq.OperatingPointsCount != 4 || parsedSeq.MaxFrameWidth != 640 || parsedSeq.MaxFrameHeight != 360 {
		t.Fatalf("parsed sequence=%+v", parsedSeq)
	}
	frameUnit, ok, err := it.Next()
	if err != nil || !ok || frameUnit.Header.Type != internalobu.TypeFrameHeader {
		t.Fatalf("frame ok=%v err=%v unit=%+v", ok, err, frameUnit.Header)
	}
	parsedPrefix, err := internalparser.ParseFrameHeaderPrefix(frameUnit.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	parsedSize, err := internalparser.ParseIntraFrameSize(frameUnit.Payload, parsedSeq, parsedPrefix, 0, 0)
	if err != nil {
		t.Fatalf("ParseIntraFrameSize: %v", err)
	}
	if parsedPrefix.FrameType != internalparser.FrameTypeKey || !parsedPrefix.ShowFrame || parsedPrefix.OrderHint != 11 ||
		parsedSize.RefreshFrameFlags != 0xff || parsedSize.UpscaledWidth != 640 || parsedSize.Height != 360 {
		t.Fatalf("parsed prefix=%+v size=%+v", parsedPrefix, parsedSize)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = internalencoder.AppendLowOverheadIntraHeaderTemporalUnitForConfig(buf[:0], cfg, 11)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadIntraHeaderTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestPublicEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	unitSize, wantUnit, err := internalencoder.LowOverheadWebRTCKeyFrameTemporalUnitForConfigSize(cfg, 12, 300)
	if err != nil {
		t.Fatalf("EncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfigSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, err := internalencoder.AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(buf[:0], cfg, 12, 300)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	if len(out) != unitSize || gotUnit != wantUnit || gotUnit.FrameNum != 2 ||
		gotUnit.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 300 {
		t.Fatalf("len=%d want=%d got=%+v want=%+v", len(out), unitSize, gotUnit, wantUnit)
	}

	it := internalobu.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil || !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit ok=%v err=%v raw=% x", ok, err, tu.Raw)
	}
	descriptorSize, err := internalencoder.WebRTCDependencyDescriptorSize(
		gotUnit.Control.DependencyStructure,
		gotUnit.Control.Frames[0].GenericFrameInfo,
		gotUnit.Control.Frames[0].AttachDependencyStructure,
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSize: %v", err)
	}
	var descriptorBuf [64]byte
	descriptor, err := internalencoder.AppendWebRTCDependencyDescriptor(
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
		_, _, _ = internalencoder.AppendLowOverheadWebRTCKeyFrameTemporalUnitForConfig(buf[:0], cfg, 12, 300)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestPublicEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	state := internalencoder.WebRTCEncoderState{NextOrderHint: 13, NextFrameID: 700}
	unitSize, wantUnit, wantState, err := internalencoder.LowOverheadWebRTCKeyFrameTemporalUnitForStateSize(cfg, state)
	if err != nil {
		t.Fatalf("EncoderLowOverheadWebRTCKeyFrameTemporalUnitForStateSize: %v", err)
	}
	var buf [192]byte
	out, gotUnit, gotState, err := internalencoder.AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(buf[:0], cfg, state)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if len(out) != unitSize || gotUnit != wantUnit || gotState != wantState ||
		gotUnit.Header.Prefix.OrderHint != 13 || gotState.NextFrameID != 702 ||
		!gotState.DependencyStructureState.Valid {
		t.Fatalf("len=%d want=%d unit=%+v wantUnit=%+v state=%+v wantState=%+v", len(out), unitSize, gotUnit, wantUnit, gotState, wantState)
	}
	it := internalobu.NewTemporalUnitIterator(out)
	tu, ok, err := it.Next()
	if err != nil || !ok || !bytes.Equal(tu.Raw, out) {
		t.Fatalf("temporal unit ok=%v err=%v raw=% x", ok, err, tu.Raw)
	}
	control, structure, err := publicRTCUnitFrameControl(
		internalencoder.WebRTCPictureTemporalUnit{Key: true, KeyUnit: gotUnit},
		gotState,
		0,
	)
	if err != nil {
		t.Fatalf("EncoderWebRTCPictureTemporalUnitFrameControl: %v", err)
	}
	var descriptorBuf [64]byte
	descriptor, err := internalencoder.AppendWebRTCDependencyDescriptor(
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
	if len(descriptor) <= RTPDependencyDescriptorMandatorySize {
		t.Fatalf("descriptor too small: % x", descriptor)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, _ = internalencoder.AppendLowOverheadWebRTCKeyFrameTemporalUnitForState(buf[:0], cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("AppendEncoderLowOverheadWebRTCKeyFrameTemporalUnitForState allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCDeltaFrameTemporalUnitForConfig(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := internalencoder.WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 400)
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := internalencoder.WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 500)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	if delta.FrameNum != 2 || delta.Control.Frames[1].GenericFrameInfo.DependencyNum != 2 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[0] != 401 ||
		delta.Control.Frames[1].GenericFrameInfo.Dependencies[1] != 500 {
		t.Fatalf("delta=%+v", delta)
	}
	if delta.Headers[1].Prefix.FrameType != internalencoder.FrameHeaderTypeInter ||
		delta.Headers[1].Size.UpscaledWidth != 640 ||
		delta.Headers[1].Size.Height != 360 ||
		delta.Headers[1].Size.RefreshFrameFlags != 0x00 ||
		delta.Headers[1].Size.RefFrameIdx[0] != 1 ||
		delta.Headers[1].Size.RefFrameIdx[1] != 2 {
		t.Fatalf("delta header=%+v", delta.Headers[1])
	}
	var headerBuf [80]byte
	headerSize, err := internalencoder.LowOverheadInterHeaderFrameOBUSize(delta.Headers[1])
	if err != nil {
		t.Fatalf("EncoderLowOverheadInterHeaderFrameOBUSize: %v", err)
	}
	headerOBU, err := internalencoder.AppendLowOverheadInterHeaderFrameOBU(headerBuf[:0], delta.Headers[1])
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadInterHeaderFrameOBU: %v", err)
	}
	unit, consumed, err := internalobu.ParseLowOverhead(headerOBU)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if len(headerOBU) != headerSize || consumed != len(headerOBU) ||
		unit.Header.Type != internalobu.TypeFrameHeader ||
		unit.Header.TemporalID != delta.Headers[1].TemporalID ||
		unit.Header.SpatialID != delta.Headers[1].SpatialID {
		t.Fatalf("header obu=%+v consumed=%d len=%d wantSize=%d", unit.Header, consumed, len(headerOBU), headerSize)
	}

	ordered, err := internalencoder.WebRTCDeltaFrameTemporalUnitForConfigWithOrderHint(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 500, 13)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfigWithOrderHint: %v", err)
	}
	if ordered.Headers[0].Prefix.OrderHint != 13 {
		t.Fatalf("ordered header hint=%d want 13", ordered.Headers[0].Prefix.OrderHint)
	}

	structure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig: %v", err)
	}
	size, err := internalencoder.WebRTCDependencyDescriptorSize(structure, delta.Control.Frames[1].GenericFrameInfo, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSize: %v", err)
	}
	var descriptorBuf [16]byte
	descriptor, err := internalencoder.AppendWebRTCDependencyDescriptor(
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
		_, _ = internalencoder.WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 500)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfig allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 1280, Height: 720},
		Scalability:       EncoderScalabilityModeL3T3_KEY_SHIFT,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	key, err := internalencoder.WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 10)
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := internalencoder.WebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 2, 20, 7)
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
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		RateControl:       EncoderRateControlCQP,
		Quantizer:         42,
	}
	key, state, err := internalencoder.WebRTCKeyFrameTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if key.Frames[0].RateControl != EncoderRateControlCQP || key.Frames[0].Quantizer != 42 {
		t.Fatalf("key frames=%+v", key.Frames)
	}
	delta, _, err := internalencoder.WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	if delta.Frames[1].RateControl != EncoderRateControlCQP || delta.Frames[1].Quantizer != 42 {
		t.Fatalf("delta frames=%+v", delta.Frames)
	}
}

func TestPublicEncoderWebRTCScreenContentHeaders(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Content:           EncoderContentScreen,
	}
	key, state, err := internalencoder.WebRTCKeyFrameTemporalUnitForState(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 1})
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForState: %v", err)
	}
	if !key.Header.Prefix.AllowScreenContentTools || !key.Header.Prefix.ForceIntegerMV {
		t.Fatalf("key prefix=%+v", key.Header.Prefix)
	}
	delta, _, err := internalencoder.WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState: %v", err)
	}
	// SVT-AV1 rtc: screen tools on I-slices only, force_integer_mv 0.
	if delta.Headers[1].Prefix.AllowScreenContentTools || delta.Headers[1].Prefix.ForceIntegerMV {
		t.Fatalf("delta headers=%+v", delta.Headers)
	}
}

func TestPublicEncoderWebRTCDependencyStructureStateForTemporalUnit(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	key, err := internalencoder.WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 700)
	if err != nil {
		t.Fatalf("EncoderWebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	state, structure, err := internalencoder.WebRTCDependencyStructureStateForTemporalUnit(key.Control, internalencoder.WebRTCDependencyStructureState{})
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyStructureStateForTemporalUnit key: %v", err)
	}
	if !state.Valid || structure != key.Control.DependencyStructure {
		t.Fatalf("key state=%+v structure=%+v control=%+v", state, structure, key.Control.DependencyStructure)
	}

	delta, err := internalencoder.WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 800)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	next, carried, err := internalencoder.WebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyStructureStateForTemporalUnit delta: %v", err)
	}
	if !next.Valid || carried != state.Structure {
		t.Fatalf("delta state=%+v carried=%+v previous=%+v", next, carried, state.Structure)
	}

	var descriptorBuf [16]byte
	descriptor, err := internalencoder.AppendWebRTCDependencyDescriptor(
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
		_, _, _ = internalencoder.WebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCDependencyStructureStateForTemporalUnit allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCStateTemporalUnits(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 960, Height: 540},
		Scalability:       internalencoder.ScalabilityModeL3T3,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 800,
	}
	state := internalencoder.WebRTCEncoderState{NextOrderHint: 126, NextFrameID: 900}
	key, state, err := internalencoder.WebRTCKeyFrameTemporalUnitForState(cfg, state)
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
		delta, next, err := internalencoder.WebRTCDeltaFrameTemporalUnitForState(cfg, state)
		if err != nil {
			t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState %d: %v", i, err)
		}
		if delta.FrameNum != 3 || delta.Frames[0].TemporalID != want ||
			delta.Control.Frames[0].GenericFrameInfo.FrameID != state.NextFrameID {
			t.Fatalf("delta %d=%+v state=%+v", i, delta, state)
		}
		if i == 0 {
			headerSize, sizedUnit, sizedNext, err := internalencoder.LowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize(cfg, state)
			if err != nil {
				t.Fatalf("EncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForStateSize: %v", err)
			}
			var headerBuf [256]byte
			headerTU, appendedUnit, appendedNext, err := internalencoder.AppendLowOverheadWebRTCDeltaHeaderTemporalUnitForState(headerBuf[:0], cfg, state)
			if err != nil {
				t.Fatalf("AppendEncoderLowOverheadWebRTCDeltaHeaderTemporalUnitForState: %v", err)
			}
			if len(headerTU) != headerSize || appendedUnit != sizedUnit || appendedNext != sizedNext ||
				appendedUnit.Control.Frames[0].GenericFrameInfo.FrameID != delta.Control.Frames[0].GenericFrameInfo.FrameID {
				t.Fatalf("header temporal unit len=%d want=%d unit=%+v sized=%+v next=%+v sizedNext=%+v", len(headerTU), headerSize, appendedUnit, sizedUnit, appendedNext, sizedNext)
			}
		}
		var descriptorBuf [32]byte
		descriptor, err := internalencoder.AppendWebRTCDependencyDescriptor(
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
		_, _, _ = internalencoder.WebRTCDeltaFrameTemporalUnitForState(cfg, state)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForState allocated: %f", allocs)
	}
}

func TestPublicEncoderWebRTCNextTemporalUnitForState(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		KeyFrameInterval:  3,
	}
	state := internalencoder.WebRTCEncoderState{NextFrameID: 50}
	unit, state, err := internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState initial: %v", err)
	}
	if !unit.Key || unit.Delta || unit.KeyUnit.FrameNum != 2 || state.NextFrameID != 52 {
		t.Fatalf("initial unit=%+v state=%+v", unit, state)
	}
	headerSize, headerUnit, headerNext, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderLowOverheadWebRTCPictureHeaderTemporalUnitForStateSize: %v", err)
	}
	var headerBuf [256]byte
	headerTU, appendedHeaderUnit, appendedHeaderNext, err := internalencoder.AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(headerBuf[:0], cfg, state, false)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadWebRTCPictureHeaderTemporalUnitForState: %v", err)
	}
	if len(headerTU) != headerSize || appendedHeaderUnit != headerUnit || appendedHeaderNext != headerNext ||
		!appendedHeaderUnit.Delta || appendedHeaderUnit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 52 {
		t.Fatalf("header TU len=%d want=%d unit=%+v sized=%+v next=%+v sizedNext=%+v", len(headerTU), headerSize, appendedHeaderUnit, headerUnit, appendedHeaderNext, headerNext)
	}
	unit, state, err = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta: %v", err)
	}
	if !unit.Delta || unit.DeltaUnit.Frames[0].TemporalID != 1 ||
		unit.DeltaUnit.Control.Frames[0].GenericFrameInfo.FrameID != 52 {
		t.Fatalf("delta unit=%+v state=%+v", unit, state)
	}
	unit, state, err = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState delta2: %v", err)
	}
	if !unit.Delta || unit.DeltaUnit.Frames[0].TemporalID != 0 {
		t.Fatalf("delta2 unit=%+v state=%+v", unit, state)
	}
	unit, _, err = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState interval key: %v", err)
	}
	if !unit.Key || !unit.KeyUnit.Control.HasDependencyStructure {
		t.Fatalf("interval key unit=%+v", unit)
	}

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = internalencoder.WebRTCNextTemporalUnitForState(cfg, state, false)
	})
	if allocs != 0 {
		t.Fatalf("EncoderWebRTCNextTemporalUnitForState allocated: %f", allocs)
	}
}

func TestPublicWebRTCEncoderPictureHeaderTemporalUnits(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		KeyFrameInterval:  4,
	}
	enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 90})
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
	if !errors.Is(err, internalbitstream.ErrShortBuffer) || len(short) != 0 || enc.State() != initialState {
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
	it := internalobu.NewTemporalUnitIterator(keyOut)
	tu, ok, err := it.Next()
	if err != nil || !ok || !bytes.Equal(tu.Raw, keyOut) {
		t.Fatalf("key temporal unit ok=%v err=%v raw=% x", ok, err, tu.Raw)
	}
	assertPublicWebRTCHeaderScalabilityMetadata(t, keyOut, internalobu.ScalabilityModeL2T2, true)

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

	base, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 90})
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
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 1280, Height: 720},
		Scalability:       EncoderScalabilityModeL1T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 30})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 3000, Den: 1}) {
		t.Fatalf("initial RTPFrameDuration=%+v err=%v", got, err)
	}
	if unit, err := enc.NextTemporalUnit(false); err != nil || !unit.Key {
		t.Fatalf("initial key unit=%+v err=%v", unit, err)
	}
	if unit, err := enc.NextTemporalUnit(false); err != nil || !unit.Delta || unit.DeltaUnit.FrameNum != 1 {
		t.Fatalf("warm delta unit=%+v err=%v", unit, err)
	}

	controlChange := cfg
	controlChange.MaxFramerate = EncoderRational{Num: 60, Den: 1}
	controlChange.MinBitrateKbps = 200
	controlChange.MaxBitrateKbps = 1800
	controlChange.TargetBitrateKbps = 1100
	if err := enc.SetConfig(controlChange); err != nil {
		t.Fatalf("SetConfig control change: %v", err)
	}
	if got, err := enc.RTPFrameDuration(); err != nil || got != (EncoderRational{Num: 1500, Den: 1}) {
		t.Fatalf("control RTPFrameDuration=%+v err=%v", got, err)
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
	structureChange.Scalability = internalencoder.ScalabilityModeS2T2
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
	sameShapeChange.Scalability = internalencoder.ScalabilityModeL2T2
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
	if err := enc.SetConfig(bad); !errors.Is(err, internalencoder.ErrInvalidConfig) {
		t.Fatalf("SetConfig invalid err=%v want %v", err, internalencoder.ErrInvalidConfig)
	}
	if enc.Config() != keepConfig || enc.State() != keepState {
		t.Fatalf("invalid SetConfig mutated config/state config=%+v want=%+v state=%+v want=%+v", enc.Config(), keepConfig, enc.State(), keepState)
	}
}

func TestPublicWebRTCEncoderSetConfigRejectsInvalidControlsWithoutMutation(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 1280, Height: 720},
		Scalability:       internalencoder.ScalabilityModeS2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 900})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	if _, err := enc.NextTemporalUnit(false); err != nil {
		t.Fatalf("warm key: %v", err)
	}
	if _, err := enc.NextTemporalUnit(false); err != nil {
		t.Fatalf("warm delta: %v", err)
	}
	keepConfig := enc.Config()
	keepState := enc.State()
	if keepState.NextFrameID == 0 || !keepState.DependencyStructureState.Valid {
		t.Fatalf("warm state=%+v", keepState)
	}

	tests := []struct {
		name string
		edit func(*EncoderConfig)
	}{
		{name: "zero-fps", edit: func(cfg *EncoderConfig) {
			cfg.MaxFramerate = EncoderRational{Num: 0, Den: 1}
		}},
		{name: "sub-one-fps", edit: func(cfg *EncoderConfig) {
			cfg.MaxFramerate = EncoderRational{Num: 1, Den: 2}
		}},
		{name: "min-above-max-bitrate", edit: func(cfg *EncoderConfig) {
			cfg.MinBitrateKbps = cfg.MaxBitrateKbps + 1
		}},
		{name: "target-below-min-bitrate", edit: func(cfg *EncoderConfig) {
			cfg.TargetBitrateKbps = cfg.MinBitrateKbps - 1
		}},
		{name: "target-above-max-bitrate", edit: func(cfg *EncoderConfig) {
			cfg.TargetBitrateKbps = cfg.MaxBitrateKbps + 1
		}},
		{name: "max-bitrate-limit", edit: func(cfg *EncoderConfig) {
			cfg.MaxBitrateKbps = internalencoder.WebRTCMaxBitrateKbps + 1
			cfg.TargetBitrateKbps = cfg.MinBitrateKbps
		}},
		{name: "cbr-quantizer", edit: func(cfg *EncoderConfig) {
			cfg.RateControl = EncoderRateControlCBR
			cfg.Quantizer = 1
		}},
		{name: "cqp-quantizer-limit", edit: func(cfg *EncoderConfig) {
			cfg.RateControl = EncoderRateControlCQP
			cfg.Quantizer = uint8(internalencoder.WebRTCMaxQuantizer + 1)
		}},
		{name: "spatial-layer-min-above-max", edit: func(cfg *EncoderConfig) {
			cfg.SpatialLayers[0].MinBitrateKbps = cfg.SpatialLayers[0].MaxBitrateKbps + 1
		}},
		{name: "spatial-layer-target-above-max", edit: func(cfg *EncoderConfig) {
			cfg.SpatialLayers[1].TargetBitrateKbps = cfg.SpatialLayers[1].MaxBitrateKbps + 1
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bad := keepConfig
			tc.edit(&bad)
			if err := enc.SetConfig(bad); !errors.Is(err, internalencoder.ErrInvalidConfig) {
				t.Fatalf("SetConfig err=%v want %v", err, internalencoder.ErrInvalidConfig)
			}
			if enc.Config() != keepConfig || enc.State() != keepState {
				t.Fatalf("SetConfig mutated state/config for %s config=%+v want=%+v state=%+v want=%+v",
					tc.name, enc.Config(), keepConfig, enc.State(), keepState)
			}
		})
	}
}

func TestPublicWebRTCEncoderSetConfigScalabilityTransitionMatrix(t *testing.T) {
	modes := internalencoder.AppendWebRTCSVCScalabilityModes(nil)
	for fromIndex, fromMode := range modes {
		fromMode := fromMode
		for toIndex, toMode := range modes {
			toMode := toMode
			t.Run(fromMode.String()+"->"+toMode.String(), func(t *testing.T) {
				const firstFrameID = 7000
				fromCfg := publicWebRTCControllerTransitionConfig(fromMode, fromIndex)
				toCfg := publicWebRTCControllerTransitionConfig(toMode, len(modes)+toIndex)
				enc, err := newWebRTCEncoder(fromCfg, internalencoder.WebRTCEncoderState{NextFrameID: firstFrameID})
				if err != nil {
					t.Fatalf("NewWebRTCEncoder(%s): %v", fromMode, err)
				}
				if got := enc.Config().Scalability; got != fromMode {
					t.Fatalf("initial config mode=%s want %s", got, fromMode)
				}
				assertPublicRTCConfigControls(t, enc.Config(), fromCfg)
				assertPublicWebRTCEncoderDuration(t, &enc)

				var receiver RTPDependencyDescriptorState
				history := make(map[uint64]publicWebRTCControllerLayer, 16)
				nextFrameID := uint64(firstFrameID)

				key, err := enc.NextTemporalUnit(false)
				if err != nil {
					t.Fatalf("initial key NextTemporalUnit: %v", err)
				}
				assertPublicWebRTCControllerUnit(t, &receiver, enc.Config(), enc.State(), key, true, 0, &nextFrameID, history)

				beforeWarm := enc.State()
				warm, err := enc.NextTemporalUnit(false)
				if err != nil {
					t.Fatalf("warm delta NextTemporalUnit: %v", err)
				}
				assertPublicWebRTCControllerUnit(t, &receiver, enc.Config(), enc.State(), warm, false, beforeWarm.DeltaPictureIndex, &nextFrameID, history)

				beforeSet := enc.State()
				if beforeSet.DeltaPictureIndex == 0 || !beforeSet.DependencyStructureState.Valid {
					t.Fatalf("warm state not ready for transition: %+v", beforeSet)
				}
				wantKey := publicWebRTCControllerSetConfigNeedsKey(t, enc.Config(), toCfg)
				if err := enc.SetConfig(toCfg); err != nil {
					t.Fatalf("SetConfig(%s->%s): %v", fromMode, toMode, err)
				}
				afterSet := enc.State()
				if afterSet.NextFrameID != beforeSet.NextFrameID || afterSet.NextOrderHint != beforeSet.NextOrderHint {
					t.Fatalf("SetConfig changed continuity state: before id=%d hint=%d after id=%d hint=%d",
						beforeSet.NextFrameID, beforeSet.NextOrderHint, afterSet.NextFrameID, afterSet.NextOrderHint)
				}
				if wantKey {
					if afterSet.DeltaPictureIndex != 0 || afterSet.DependencyStructureState.Valid {
						t.Fatalf("wire-structure transition did not reset dependency state: before delta=%d dep=%v after delta=%d dep=%v",
							beforeSet.DeltaPictureIndex, beforeSet.DependencyStructureState.Valid,
							afterSet.DeltaPictureIndex, afterSet.DependencyStructureState.Valid)
					}
				} else if afterSet != beforeSet {
					t.Fatalf("wire-compatible control change reset state: before delta=%d dep=%v after delta=%d dep=%v",
						beforeSet.DeltaPictureIndex, beforeSet.DependencyStructureState.Valid,
						afterSet.DeltaPictureIndex, afterSet.DependencyStructureState.Valid)
				}
				normalized := enc.Config()
				assertPublicRTCConfigControls(t, normalized, toCfg)
				if normalized.Scalability != toMode ||
					normalized.MaxFramerate != toCfg.MaxFramerate ||
					normalized.TargetBitrateKbps != toCfg.TargetBitrateKbps ||
					normalized.Content != toCfg.Content {
					t.Fatalf("normalized config=%+v want mode=%s fps=%+v target=%d content=%d",
						normalized, toMode, toCfg.MaxFramerate, toCfg.TargetBitrateKbps, toCfg.Content)
				}
				assertPublicWebRTCEncoderDuration(t, &enc)

				beforeUnit := enc.State()
				unit, err := enc.NextTemporalUnit(false)
				if err != nil {
					t.Fatalf("post-transition NextTemporalUnit: %v", err)
				}
				assertPublicWebRTCControllerUnit(t, &receiver, normalized, enc.State(), unit, wantKey, beforeUnit.DeltaPictureIndex, &nextFrameID, history)

				beforeDelta := enc.State()
				delta, err := enc.NextTemporalUnit(false)
				if err != nil {
					t.Fatalf("post-transition delta NextTemporalUnit: %v", err)
				}
				assertPublicWebRTCControllerUnit(t, &receiver, normalized, enc.State(), delta, false, beforeDelta.DeltaPictureIndex, &nextFrameID, history)
			})
		}
	}
}

func publicWebRTCControllerSetConfigNeedsKey(t *testing.T, from EncoderConfig, to EncoderConfig) bool {
	t.Helper()
	fromSeq, err := internalencoder.SequenceHeaderForConfig(from)
	if err != nil {
		t.Fatalf("from SequenceHeaderForConfig(%s): %v", from.Scalability, err)
	}
	toSeq, err := internalencoder.SequenceHeaderForConfig(to)
	if err != nil {
		t.Fatalf("to SequenceHeaderForConfig(%s): %v", to.Scalability, err)
	}
	fromStructure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(from)
	if err != nil {
		t.Fatalf("from EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", from.Scalability, err)
	}
	toStructure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(to)
	if err != nil {
		t.Fatalf("to EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", to.Scalability, err)
	}
	return fromSeq != toSeq || fromStructure != toStructure
}

func TestPublicWebRTCEncoderControllerSettingsMatrix(t *testing.T) {
	profiles := []struct {
		name        string
		fps         EncoderRational
		rateControl EncoderRateControlMode
		quantizer   uint8
		minKbps     int32
		maxKbps     int32
		targetKbps  int32
		content     EncoderContentHint
	}{
		{
			name:        "camera-cbr-30fps",
			fps:         EncoderRational{Num: 30, Den: 1},
			rateControl: EncoderRateControlCBR,
			minKbps:     120,
			maxKbps:     1800,
			targetKbps:  900,
			content:     EncoderContentCamera,
		},
		{
			name:        "screen-cqp-ntsc",
			fps:         EncoderRational{Num: 30000, Den: 1001},
			rateControl: EncoderRateControlCQP,
			quantizer:   37,
			minKbps:     80,
			maxKbps:     1200,
			targetKbps:  640,
			content:     EncoderContentScreen,
		},
	}

	for _, modeName := range publicWebRTCControllerModeNames() {
		mode, ok := internalencoder.ParseScalabilityMode(modeName)
		if !ok {
			t.Fatalf("ParseEncoderScalabilityMode(%q) failed", modeName)
		}
		for _, profile := range profiles {
			t.Run(modeName+"/"+profile.name, func(t *testing.T) {
				cfg := EncoderConfig{
					Resolution:        EncoderResolution{Width: 1280, Height: 720},
					Scalability:       mode,
					MaxFramerate:      profile.fps,
					RateControl:       profile.rateControl,
					Quantizer:         profile.quantizer,
					MinBitrateKbps:    profile.minKbps,
					MaxBitrateKbps:    profile.maxKbps,
					TargetBitrateKbps: profile.targetKbps,
					Content:           profile.content,
				}
				enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 1000})
				if err != nil {
					t.Fatalf("NewWebRTCEncoder(%s): %v", modeName, err)
				}
				normalized := enc.Config()
				assertPublicRTCConfigControls(t, normalized, cfg)
				spatialLayers, temporalLayers, _, ok := normalized.Scalability.Layers()
				if !ok {
					t.Fatalf("normalized invalid mode=%s", normalized.Scalability)
				}
				if normalized.SpatialLayerCount != spatialLayers || normalized.TemporalLayerCount != temporalLayers {
					t.Fatalf("normalized layers=%d,%d want %d,%d", normalized.SpatialLayerCount, normalized.TemporalLayerCount, spatialLayers, temporalLayers)
				}
				assertPublicWebRTCEncoderDuration(t, &enc)

				var receiver RTPDependencyDescriptorState
				history := make(map[uint64]publicWebRTCControllerLayer, 32)
				nextFrameID := uint64(1000)
				key, err := enc.NextTemporalUnit(false)
				if err != nil {
					t.Fatalf("initial key NextTemporalUnit: %v", err)
				}
				assertPublicWebRTCControllerUnit(t, &receiver, normalized, enc.State(), key, true, 0, &nextFrameID, history)

				controlChange := normalized
				controlChange.MaxFramerate = EncoderRational{Num: 60, Den: 1}
				if controlChange.RateControl == EncoderRateControlCQP {
					controlChange.Quantizer = 29
				} else {
					controlChange.MinBitrateKbps += 25
					controlChange.MaxBitrateKbps += 250
					controlChange.TargetBitrateKbps = (controlChange.MinBitrateKbps + controlChange.MaxBitrateKbps) / 2
				}
				if err := enc.SetConfig(controlChange); err != nil {
					t.Fatalf("SetConfig control change: %v", err)
				}
				normalized = enc.Config()
				assertPublicRTCConfigControls(t, normalized, controlChange)
				assertPublicWebRTCEncoderDuration(t, &enc)
				for step := uint64(0); step < publicWebRTCControllerMatrixSteps(temporalLayers); step++ {
					before := enc.State()
					unit, err := enc.NextTemporalUnit(false)
					if err != nil {
						t.Fatalf("delta %d NextTemporalUnit: %v", step, err)
					}
					assertPublicWebRTCControllerUnit(t, &receiver, normalized, enc.State(), unit, false, before.DeltaPictureIndex, &nextFrameID, history)
				}
			})
		}
	}
}

func TestPublicWebRTCEncoderExplicitKeyShiftControllerModes(t *testing.T) {
	for _, mode := range []EncoderScalabilityMode{
		EncoderScalabilityModeL2T3_KEY_SHIFT,
		EncoderScalabilityModeL3T2_KEY_SHIFT,
		EncoderScalabilityModeL3T3_KEY_SHIFT,
	} {
		t.Run(mode.String(), func(t *testing.T) {
			cfg := EncoderConfig{
				Resolution:        EncoderResolution{Width: 1280, Height: 720},
				Scalability:       mode,
				MaxFramerate:      EncoderRational{Num: 30, Den: 1},
				MinBitrateKbps:    120,
				MaxBitrateKbps:    1800,
				TargetBitrateKbps: 900,
				Content:           EncoderContentCamera,
			}
			enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 4000})
			if err != nil {
				t.Fatalf("NewWebRTCEncoder(%s): %v", mode, err)
			}
			if enc.Config().Scalability != mode {
				t.Fatalf("normalized mode=%s want %s", enc.Config().Scalability, mode)
			}

			var receiver RTPDependencyDescriptorState
			history := make(map[uint64]publicWebRTCControllerLayer, 32)
			nextFrameID := uint64(4000)
			key, err := enc.NextTemporalUnit(false)
			if err != nil {
				t.Fatalf("initial key NextTemporalUnit: %v", err)
			}
			assertPublicWebRTCControllerUnit(t, &receiver, enc.Config(), enc.State(), key, true, 0, &nextFrameID, history)

			_, temporalLayers, _, ok := enc.Config().Scalability.Layers()
			if !ok {
				t.Fatalf("invalid normalized mode=%s", enc.Config().Scalability)
			}
			for step := uint64(0); step < publicWebRTCControllerMatrixSteps(temporalLayers); step++ {
				before := enc.State()
				unit, err := enc.NextTemporalUnit(false)
				if err != nil {
					t.Fatalf("delta %d NextTemporalUnit: %v", step, err)
				}
				assertPublicWebRTCControllerUnit(t, &receiver, enc.Config(), enc.State(), unit, false, before.DeltaPictureIndex, &nextFrameID, history)
			}

			controlChange := enc.Config()
			controlChange.MaxFramerate = EncoderRational{Num: 60, Den: 1}
			controlChange.MinBitrateKbps = 200
			controlChange.MaxBitrateKbps = 2200
			controlChange.TargetBitrateKbps = 1200
			controlChange.Content = EncoderContentScreen
			before := enc.State()
			if err := enc.SetConfig(controlChange); err != nil {
				t.Fatalf("SetConfig control change: %v", err)
			}
			if enc.State() != before {
				t.Fatalf("control change reset state: before=%+v after=%+v", before, enc.State())
			}
			assertPublicRTCConfigControls(t, enc.Config(), controlChange)
			assertPublicWebRTCEncoderDuration(t, &enc)
			unit, err := enc.NextTemporalUnit(false)
			if err != nil {
				t.Fatalf("delta after control change: %v", err)
			}
			assertPublicWebRTCControllerUnit(t, &receiver, enc.Config(), enc.State(), unit, false, before.DeltaPictureIndex, &nextFrameID, history)
		})
	}
}

func assertPublicWebRTCEncoderDuration(t *testing.T, enc *webRTCEncoder) {
	t.Helper()
	cfg := enc.Config()
	want, err := EncoderWebRTCRTPFrameDuration(cfg)
	if err != nil {
		t.Fatalf("EncoderWebRTCRTPFrameDuration(%+v): %v", cfg.MaxFramerate, err)
	}
	got, err := enc.RTPFrameDuration()
	if err != nil || got != want {
		t.Fatalf("RTPFrameDuration=%+v err=%v want %+v for fps=%+v", got, err, want, cfg.MaxFramerate)
	}
}

func assertPublicWebRTCHeaderScalabilityMetadata(t *testing.T, tu []byte, wantIDC uint8, wantPresent bool) {
	t.Helper()
	it := internalobu.NewLowOverheadIterator(tu)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != internalobu.TypeTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := it.Next(); err != nil || !ok || seq.Header.Type != internalobu.TypeSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	next, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("next OBU ok=%v err=%v header=%+v", ok, err, next.Header)
	}
	if !wantPresent {
		if next.Header.Type == internalobu.TypeMetadata {
			t.Fatalf("unexpected metadata payload=% x", next.Payload)
		}
		return
	}
	if next.Header.Type != internalobu.TypeMetadata {
		t.Fatalf("metadata header=%+v", next.Header)
	}
	meta, err := internalobu.ParseMetadata(next.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != internalobu.MetadataTypeScalability || meta.Scalability.ModeIDC != wantIDC || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v want idc=%d", meta, wantIDC)
	}
}

func publicWebRTCControllerTransitionConfig(mode EncoderScalabilityMode, step int) EncoderConfig {
	fps := []EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 15, Den: 1},
	}
	minKbps := int32(120 + (step%5)*20)
	maxKbps := int32(1400 + (step%7)*120)
	targetKbps := minKbps + (maxKbps-minKbps)/2
	content := EncoderContentCamera
	if step%2 == 1 {
		content = EncoderContentScreen
	}
	return EncoderConfig{
		Resolution:        EncoderResolution{Width: 1280, Height: 720},
		Scalability:       mode,
		MaxFramerate:      fps[step%len(fps)],
		MinBitrateKbps:    minKbps,
		MaxBitrateKbps:    maxKbps,
		TargetBitrateKbps: targetKbps,
		Content:           content,
	}
}

type publicWebRTCControllerLayer struct {
	spatial  uint8
	temporal uint8
}

func publicWebRTCControllerModeNames() []string {
	modes := internalencoder.AppendWebRTCSVCScalabilityModes(nil)
	names := make([]string, len(modes))
	for i, mode := range modes {
		names[i] = mode.String()
	}
	return names
}

func publicWebRTCControllerMatrixSteps(temporalLayers uint8) uint64 {
	switch temporalLayers {
	case 1:
		return 3
	case 2:
		return 4
	default:
		return 8
	}
}

func assertPublicWebRTCControllerUnit(t *testing.T, receiver *RTPDependencyDescriptorState, cfg EncoderConfig, descriptorState internalencoder.WebRTCEncoderState, unit internalencoder.WebRTCPictureTemporalUnit, wantKey bool, deltaPictureIndex uint64, nextFrameID *uint64, history map[uint64]publicWebRTCControllerLayer) {
	t.Helper()
	spatialLayers, temporalLayers, _, ok := cfg.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid mode=%s", cfg.Scalability)
	}
	frameNum := encoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum != spatialLayers || unit.Key != wantKey || unit.Delta == wantKey {
		t.Fatalf("unit key=%v delta=%v frames=%d want key=%v frames=%d", unit.Key, unit.Delta, frameNum, wantKey, spatialLayers)
	}
	for i := uint8(0); i < frameNum; i++ {
		control, structure, err := publicRTCUnitFrameControl(unit, descriptorState, i)
		if err != nil {
			t.Fatalf("frame %d control: %v", i, err)
		}
		settings := control.Settings
		info := control.GenericFrameInfo
		wantFrameID := *nextFrameID + uint64(i)
		if settings.SpatialID != i || info.SpatialID != i ||
			settings.TemporalID != info.TemporalID ||
			info.FrameID != wantFrameID ||
			info.TemporalID >= temporalLayers ||
			info.DTINum != spatialLayers*temporalLayers {
			t.Fatalf("frame %d settings=%+v info=%+v want id=%d temporal<%d", i, settings, info, wantFrameID, temporalLayers)
		}
		if !wantKey {
			wantTemporal := publicWebRTCExpectedTemporalID(cfg.Scalability, i, deltaPictureIndex)
			if info.TemporalID != wantTemporal {
				t.Fatalf("frame %d delta index=%d temporal=%d want %d mode=%s", i, deltaPictureIndex, info.TemporalID, wantTemporal, cfg.Scalability)
			}
		}
		if settings.RateControl != cfg.RateControl ||
			(cfg.RateControl == EncoderRateControlCQP && settings.Quantizer != cfg.Quantizer) ||
			settings.EffortLevel != cfg.Speed {
			t.Fatalf("frame %d rate settings=%+v config=%+v", i, settings, cfg)
		}
		if cfg.Content == EncoderContentScreen {
			if wantKey {
				if !unit.KeyUnit.Header.Prefix.AllowScreenContentTools || !unit.KeyUnit.Header.Prefix.ForceIntegerMV {
					t.Fatalf("screen key prefix=%+v", unit.KeyUnit.Header.Prefix)
				}
			} else if unit.DeltaUnit.Headers[i].Prefix.AllowScreenContentTools || unit.DeltaUnit.Headers[i].Prefix.ForceIntegerMV {
				t.Fatalf("screen delta header[%d]=%+v", i, unit.DeltaUnit.Headers[i])
			}
		}
		wantAttach := wantKey && i == 0
		if control.AttachDependencyStructure != wantAttach {
			t.Fatalf("frame %d attach=%v want %v", i, control.AttachDependencyStructure, wantAttach)
		}
		var descriptorBuf [2048]byte
		descriptor, err := appendEncoderWebRTCPictureTemporalUnitDependencyDescriptor(
			descriptorBuf[:0],
			unit,
			descriptorState,
			i,
			true,
			true,
			control.AttachDependencyStructure,
		)
		if err != nil {
			t.Fatalf("frame %d descriptor append: %v", i, err)
		}
		parsed, consumed, err := receiver.Parse(descriptor)
		if err != nil {
			t.Fatalf("frame %d descriptor parse: %v", i, err)
		}
		if consumed != len(descriptor) ||
			parsed.Mandatory.FrameNumber != uint16(info.FrameID) ||
			parsed.HasAttachedStructure != control.AttachDependencyStructure {
			t.Fatalf("frame %d parsed=%+v consumed=%d len=%d info=%+v attach=%v", i, parsed, consumed, len(descriptor), info, control.AttachDependencyStructure)
		}
		if parsed.HasAttachedStructure {
			assertPublicRTCAttachedStructure(t, parsed.AttachedStructure, cfg)
		}
		wantActiveMask, err := internalencoder.WebRTCAllDecodeTargetsMask(structure)
		if err != nil {
			t.Fatalf("frame %d all decode target mask: %v", i, err)
		}
		if control.AttachDependencyStructure {
			if !parsed.HasActiveDecodeTargets || parsed.ActiveDecodeTargetsMask != wantActiveMask {
				t.Fatalf("frame %d active targets=%v/%#x want true/%#x", i, parsed.HasActiveDecodeTargets, parsed.ActiveDecodeTargetsMask, wantActiveMask)
			}
		} else if parsed.HasActiveDecodeTargets {
			t.Fatalf("frame %d repeated active decode targets: %+v", i, parsed)
		}
		if !receiver.ActiveDecodeTargetsValid || receiver.ActiveDecodeTargetsMask != wantActiveMask {
			t.Fatalf("frame %d receiver active targets valid=%v mask=%#x want %#x", i, receiver.ActiveDecodeTargetsValid, receiver.ActiveDecodeTargetsMask, wantActiveMask)
		}
		deps := parsed.FrameDependencies
		if deps.SpatialID != info.SpatialID || deps.TemporalID != info.TemporalID ||
			deps.DTINum != info.DTINum || deps.FrameDiffNum != info.DependencyNum {
			t.Fatalf("frame %d deps=%+v info=%+v", i, deps, info)
		}
		for j := uint8(0); j < info.DependencyNum; j++ {
			dependencyFrameID := info.Dependencies[j]
			if deps.FrameDiffs[j] != uint16(info.FrameID-dependencyFrameID) {
				t.Fatalf("frame %d dep %d diff=%d want %d info=%+v", i, j, deps.FrameDiffs[j], info.FrameID-dependencyFrameID, info)
			}
			dependency, ok := history[dependencyFrameID]
			if !ok {
				t.Fatalf("frame %d dependency %d missing history", i, dependencyFrameID)
			}
			if dependency.temporal > info.TemporalID {
				t.Fatalf("frame %d dependency %d temporal=%d > %d", i, dependencyFrameID, dependency.temporal, info.TemporalID)
			}
			if cfg.Scalability.IsSimulcast() && dependency.spatial != info.SpatialID {
				t.Fatalf("frame %d simulcast dependency spatial=%d want %d", i, dependency.spatial, info.SpatialID)
			}
		}
		history[info.FrameID] = publicWebRTCControllerLayer{spatial: info.SpatialID, temporal: info.TemporalID}
	}
	*nextFrameID += uint64(frameNum)
}

func publicWebRTCExpectedTemporalID(mode EncoderScalabilityMode, spatialID uint8, deltaPictureIndex uint64) uint8 {
	if mode.UsesKeyFrameInterLayerDependencyShift() {
		switch mode {
		case internalencoder.ScalabilityModeL2T2_KEY_SHIFT:
			table := [2][2]uint8{{0, 1}, {1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case EncoderScalabilityModeL2T3_KEY_SHIFT:
			table := [4][2]uint8{{2, 2}, {0, 1}, {2, 2}, {1, 0}}
			return table[(deltaPictureIndex-1)%4][spatialID]
		case EncoderScalabilityModeL3T2_KEY_SHIFT:
			table := [2][3]uint8{{0, 0, 1}, {1, 1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case EncoderScalabilityModeL3T3_KEY_SHIFT:
			table := [4][3]uint8{{0, 2, 2}, {2, 0, 1}, {1, 2, 2}, {2, 1, 0}}
			return table[(deltaPictureIndex-1)%4][spatialID]
		}
	}
	_, temporalLayers, _, ok := mode.Layers()
	if !ok || deltaPictureIndex == 0 {
		return 0
	}
	trailingZeroCount := uint8(0)
	for value := deltaPictureIndex; value&1 == 0 && trailingZeroCount < temporalLayers-1; value >>= 1 {
		trailingZeroCount++
	}
	return temporalLayers - 1 - trailingZeroCount
}

func TestPublicWebRTCEncoderPictureHeaderTemporalUnitsForFrames(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 120})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	normalized := enc.Config()
	var frames [2]Frame
	var backing [2][]byte
	for i := range frames {
		layer := normalized.SpatialLayers[i]
		format := FrameFormat{
			Width:        int(layer.Resolution.Width),
			Height:       int(layer.Resolution.Height),
			BitDepth:     normalized.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
			Align:        64,
		}
		layout, err := internalframe.RequiredSize(format)
		if err != nil {
			t.Fatalf("FrameRequiredSize layer %d: %v", i, err)
		}
		backing[i] = make([]byte, layout.Size)
		frames[i], err = internalframe.Bind(backing[i], format)
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
	if !errors.Is(err, internalbitstream.ErrShortBuffer) || len(short) != 0 || enc.State() != state {
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
	assertPublicWebRTCHeaderScalabilityMetadata(t, out, internalobu.ScalabilityModeL2T2, true)

	bad := frames
	bad[1].Format.Width--
	if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize(bad[:], false); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("bad frame err=%v want %v", err, ErrEncoderInvalidFrame)
	}
	if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize(frames[:1], false); !errors.Is(err, ErrEncoderInvalidFrame) {
		t.Fatalf("short frame list err=%v want %v", err, ErrEncoderInvalidFrame)
	}

	base, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 120})
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
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 200})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	normalized := enc.Config()
	var frames [2]Frame
	var backing [2][]byte
	for i := range frames {
		layer := normalized.SpatialLayers[i]
		format := FrameFormat{
			Width:        int(layer.Resolution.Width),
			Height:       int(layer.Resolution.Height),
			BitDepth:     normalized.BitDepth,
			SubsamplingX: true,
			SubsamplingY: true,
			Align:        64,
		}
		layout, err := internalframe.RequiredSize(format)
		if err != nil {
			t.Fatalf("FrameRequiredSize layer %d: %v", i, err)
		}
		backing[i] = make([]byte, layout.Size)
		frames[i], err = internalframe.Bind(backing[i], format)
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
	if _, _, err := enc.LoadPictureSamplePlanes(encoderWebRTCPictureSampleScratch{Samples: make([]uint16, size.Samples-1)}, frames[:], false); !errors.Is(err, internalframe.ErrShortBuffer) {
		t.Fatalf("short scratch err=%v want %v", err, internalframe.ErrShortBuffer)
	}
	samples := make([]uint16, size.Samples)
	planes, gotUnit, err := enc.LoadPictureSamplePlanes(encoderWebRTCPictureSampleScratch{Samples: samples}, frames[:], false)
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

	base, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 200})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder alloc base: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		local := base
		_, _, _ = local.LoadPictureSamplePlanes(encoderWebRTCPictureSampleScratch{Samples: samples}, frames[:], false)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCEncoder LoadPictureSamplePlanes allocated: %f", allocs)
	}
}

func TestPublicWebRTCEncoderPictureSamplePlanesNonI420Formats(t *testing.T) {
	tests := []struct {
		name         string
		cfg          EncoderConfig
		wantBitDepth uint8
		wantSubX     bool
		wantSubY     bool
	}{
		{
			name: "profile1-444-8bit",
			cfg: EncoderConfig{
				Resolution:   EncoderResolution{Width: 64, Height: 64},
				Profile:      internalencoder.Profile1,
				BitDepth:     8,
				MaxFramerate: EncoderRational{Num: 30, Den: 1},
			},
			wantBitDepth: 8,
		},
		{
			name: "profile0-420-10bit",
			cfg: EncoderConfig{
				Resolution:   EncoderResolution{Width: 64, Height: 64},
				Profile:      EncoderProfile0,
				BitDepth:     10,
				MaxFramerate: EncoderRational{Num: 30, Den: 1},
			},
			wantBitDepth: 10,
			wantSubX:     true,
			wantSubY:     true,
		},
		{
			name: "profile2-422-12bit",
			cfg: EncoderConfig{
				Resolution:   EncoderResolution{Width: 64, Height: 64},
				Profile:      internalencoder.Profile2,
				BitDepth:     12,
				MaxFramerate: EncoderRational{Num: 30, Den: 1},
			},
			wantBitDepth: 12,
			wantSubX:     true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := newWebRTCEncoder(tc.cfg, internalencoder.WebRTCEncoderState{})
			if err != nil {
				t.Fatalf("NewWebRTCEncoder: %v", err)
			}
			normalized := enc.Config()
			seq, err := internalencoder.SequenceHeaderForConfig(normalized)
			if err != nil {
				t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
			}
			if seq.ColorConfig.BitDepth != tc.wantBitDepth ||
				seq.ColorConfig.SubsamplingX != tc.wantSubX ||
				seq.ColorConfig.SubsamplingY != tc.wantSubY {
				t.Fatalf("color config=%+v", seq.ColorConfig)
			}

			frame, _ := publicWebRTCEncoderFrameForConfig(t, normalized)
			bytesPerSample := 1
			if frame.Format.BitDepth > 8 {
				bytesPerSample = 2
			}
			ySample, uSample, vSample := uint16(0x23), uint16(0x34), uint16(0x45)
			if bytesPerSample == 2 {
				ySample, uSample, vSample = 0x0123, 0x0234, 0x0345
			}
			publicSetFramePlaneSample(frame.Y, bytesPerSample, 3, 2, ySample)
			publicSetFramePlaneSample(frame.U, bytesPerSample, 2, 1, uSample)
			publicSetFramePlaneSample(frame.V, bytesPerSample, 1, 1, vSample)

			if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize([]Frame{frame}, false); err != nil {
				t.Fatalf("LowOverheadPictureHeaderTemporalUnitForFramesSize: %v", err)
			}
			size, unit, err := enc.PictureSampleScratchSize([]Frame{frame}, false)
			if err != nil {
				t.Fatalf("PictureSampleScratchSize: %v", err)
			}
			if size.Samples == 0 || size.FrameNum != 1 || !unit.Key {
				t.Fatalf("scratch size=%+v unit=%+v", size, unit)
			}
			samples := make([]uint16, size.Samples)
			planes, gotUnit, err := enc.LoadPictureSamplePlanes(
				encoderWebRTCPictureSampleScratch{Samples: samples},
				[]Frame{frame},
				false,
			)
			if err != nil {
				t.Fatalf("LoadPictureSamplePlanes: %v", err)
			}
			if gotUnit != unit || planes.FrameNum != 1 ||
				planes.Frames[0].Y.Pix[2*planes.Frames[0].Y.Stride+3] != ySample ||
				planes.Frames[0].U.Pix[1*planes.Frames[0].U.Stride+2] != uSample ||
				planes.Frames[0].V.Pix[1*planes.Frames[0].V.Stride+1] != vSample {
				t.Fatalf("loaded planes=%+v unit=%+v want=%+v", planes, gotUnit, unit)
			}

			bad := frame
			bad.Format.BitDepth = 8
			if bad.Format.BitDepth == frame.Format.BitDepth {
				bad.Format.BitDepth = 10
			}
			if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize([]Frame{bad}, false); !errors.Is(err, ErrEncoderInvalidFrame) {
				t.Fatalf("mismatched bitdepth err=%v want %v", err, ErrEncoderInvalidFrame)
			}
			bad = frame
			bad.Format.SubsamplingX = !bad.Format.SubsamplingX
			if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize([]Frame{bad}, false); !errors.Is(err, ErrEncoderInvalidFrame) {
				t.Fatalf("mismatched subsampling err=%v want %v", err, ErrEncoderInvalidFrame)
			}
		})
	}
}

func TestPublicWebRTCEncoderExplicitColorConfigSamplePlanes(t *testing.T) {
	tests := []struct {
		name string
		cfg  EncoderConfig
		want internalencoder.SequenceColorConfig
	}{
		{
			name: "profile2-420-12bit-svc",
			cfg: EncoderConfig{
				Resolution:     EncoderResolution{Width: 640, Height: 360},
				Profile:        internalencoder.Profile2,
				BitDepth:       12,
				Scalability:    internalencoder.ScalabilityModeL2T1,
				MaxFramerate:   EncoderRational{Num: 30, Den: 1},
				ColorConfigSet: true,
				ColorConfig: internalencoder.SequenceColorConfig{
					BitDepth:             12,
					SubsamplingX:         true,
					SubsamplingY:         true,
					ChromaSamplePosition: 1,
				},
			},
			want: internalencoder.SequenceColorConfig{
				BitDepth:             12,
				SubsamplingX:         true,
				SubsamplingY:         true,
				ChromaSamplePosition: 1,
			},
		},
		{
			name: "profile2-444-12bit-color-bitdepth-only-svc",
			cfg: EncoderConfig{
				Resolution:     EncoderResolution{Width: 640, Height: 360},
				Profile:        internalencoder.Profile2,
				Scalability:    internalencoder.ScalabilityModeL2T1,
				MaxFramerate:   EncoderRational{Num: 30, Den: 1},
				ColorConfigSet: true,
				ColorConfig: internalencoder.SequenceColorConfig{
					BitDepth: 12,
				},
			},
			want: internalencoder.SequenceColorConfig{
				BitDepth: 12,
			},
		},
		{
			name: "profile0-mono-10bit",
			cfg: EncoderConfig{
				Resolution:     EncoderResolution{Width: 64, Height: 64},
				Profile:        EncoderProfile0,
				BitDepth:       10,
				MaxFramerate:   EncoderRational{Num: 30, Den: 1},
				ColorConfigSet: true,
				ColorConfig: internalencoder.SequenceColorConfig{
					BitDepth:   10,
					MonoChrome: true,
				},
			},
			want: internalencoder.SequenceColorConfig{
				BitDepth:     10,
				MonoChrome:   true,
				SubsamplingX: true,
				SubsamplingY: true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := newWebRTCEncoder(tc.cfg, internalencoder.WebRTCEncoderState{NextFrameID: 400})
			if err != nil {
				t.Fatalf("NewWebRTCEncoder: %v", err)
			}
			normalized := enc.Config()
			if !normalized.ColorConfigSet || normalized.ColorConfig != tc.want || normalized.BitDepth != tc.want.BitDepth {
				t.Fatalf("normalized color config=%+v set=%v bitdepth=%d want %+v", normalized.ColorConfig, normalized.ColorConfigSet, normalized.BitDepth, tc.want)
			}
			seq, err := internalencoder.SequenceHeaderForConfig(normalized)
			if err != nil {
				t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
			}
			if seq.ColorConfig != tc.want {
				t.Fatalf("sequence color config=%+v want %+v", seq.ColorConfig, tc.want)
			}

			frameNum := int(normalized.SpatialLayerCount)
			frames := make([]Frame, frameNum)
			backing := make([][]byte, frameNum)
			for i := 0; i < frameNum; i++ {
				layer := normalized.SpatialLayers[i]
				format := FrameFormat{
					Width:        int(layer.Resolution.Width),
					Height:       int(layer.Resolution.Height),
					BitDepth:     seq.ColorConfig.BitDepth,
					MonoChrome:   seq.ColorConfig.MonoChrome,
					SubsamplingX: seq.ColorConfig.SubsamplingX,
					SubsamplingY: seq.ColorConfig.SubsamplingY,
					Align:        64,
				}
				layout, err := internalframe.RequiredSize(format)
				if err != nil {
					t.Fatalf("FrameRequiredSize layer %d: %v", i, err)
				}
				backing[i] = make([]byte, layout.Size)
				frames[i], err = internalframe.Bind(backing[i], format)
				if err != nil {
					t.Fatalf("BindFrame layer %d: %v", i, err)
				}
				publicSetFramePlaneSample(frames[i].Y, layout.BytesPerSample, 3, 2, uint16(0x120+uint16(i)))
				if !format.MonoChrome {
					publicSetFramePlaneSample(frames[i].U, layout.BytesPerSample, 1, 1, uint16(0x230+uint16(i)))
					publicSetFramePlaneSample(frames[i].V, layout.BytesPerSample, 1, 1, uint16(0x340+uint16(i)))
				}
			}

			size, unit, err := enc.PictureSampleScratchSize(frames, false)
			if err != nil {
				t.Fatalf("PictureSampleScratchSize: %v", err)
			}
			if !unit.Key || size.FrameNum != frameNum || int(unit.KeyUnit.FrameNum) != frameNum {
				t.Fatalf("scratch size=%+v unit=%+v frameNum=%d", size, unit, frameNum)
			}
			planes, gotUnit, err := enc.LoadPictureSamplePlanes(
				encoderWebRTCPictureSampleScratch{Samples: make([]uint16, size.Samples)},
				frames,
				false,
			)
			if err != nil {
				t.Fatalf("LoadPictureSamplePlanes: %v", err)
			}
			if gotUnit != unit || int(planes.FrameNum) != frameNum {
				t.Fatalf("planes frameNum=%d unit=%+v want=%+v", planes.FrameNum, gotUnit, unit)
			}
			for i := 0; i < frameNum; i++ {
				yWant := uint16(0x120 + uint16(i))
				if got := planes.Frames[i].Y.Pix[2*planes.Frames[i].Y.Stride+3]; got != yWant {
					t.Fatalf("layer %d Y sample=%#x want %#x", i, got, yWant)
				}
				if tc.want.MonoChrome {
					if len(planes.Frames[i].U.Pix) != 0 || len(planes.Frames[i].V.Pix) != 0 {
						t.Fatalf("layer %d monochrome chroma planes not empty: U=%+v V=%+v", i, planes.Frames[i].U, planes.Frames[i].V)
					}
					continue
				}
				uWant := uint16(0x230 + uint16(i))
				vWant := uint16(0x340 + uint16(i))
				if got := planes.Frames[i].U.Pix[planes.Frames[i].U.Stride+1]; got != uWant {
					t.Fatalf("layer %d U sample=%#x want %#x", i, got, uWant)
				}
				if got := planes.Frames[i].V.Pix[planes.Frames[i].V.Stride+1]; got != vWant {
					t.Fatalf("layer %d V sample=%#x want %#x", i, got, vWant)
				}
			}
		})
	}
}

func TestPublicWebRTCEncoderSetConfigColorConfigTransition(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:     EncoderResolution{Width: 640, Height: 360},
		Profile:        internalencoder.Profile2,
		BitDepth:       12,
		Scalability:    internalencoder.ScalabilityModeL2T1,
		MaxFramerate:   EncoderRational{Num: 30, Den: 1},
		ColorConfigSet: true,
		ColorConfig: internalencoder.SequenceColorConfig{
			BitDepth:             12,
			SubsamplingX:         true,
			SubsamplingY:         true,
			ChromaSamplePosition: 1,
		},
	}
	enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 500})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	if unit, err := enc.NextTemporalUnit(false); err != nil || !unit.Key {
		t.Fatalf("initial key unit=%+v err=%v", unit, err)
	}
	if unit, err := enc.NextTemporalUnit(false); err != nil || !unit.Delta {
		t.Fatalf("warm delta unit=%+v err=%v", unit, err)
	}
	before := enc.State()
	if before.DeltaPictureIndex == 0 || !before.DependencyStructureState.Valid {
		t.Fatalf("warm state=%+v", before)
	}

	changed := enc.Config()
	changed.ColorConfig = internalencoder.SequenceColorConfig{BitDepth: 12}
	if err := enc.SetConfig(changed); err != nil {
		t.Fatalf("SetConfig color change: %v", err)
	}
	afterSet := enc.State()
	if afterSet.NextFrameID != before.NextFrameID ||
		afterSet.NextOrderHint != before.NextOrderHint ||
		afterSet.DeltaPictureIndex != 0 ||
		afterSet.DependencyStructureState.Valid {
		t.Fatalf("color change state=%+v before=%+v", afterSet, before)
	}
	if enc.Config().ColorConfig != changed.ColorConfig {
		t.Fatalf("config color=%+v want %+v", enc.Config().ColorConfig, changed.ColorConfig)
	}
	unit, err := enc.NextTemporalUnit(false)
	if err != nil {
		t.Fatalf("NextTemporalUnit after color change: %v", err)
	}
	if !unit.Key || unit.KeyUnit.FrameNum != 2 || enc.State().NextFrameID != before.NextFrameID+2 {
		t.Fatalf("post-color-change unit=%+v state=%+v before=%+v", unit, enc.State(), before)
	}

	keepConfig := enc.Config()
	keepState := enc.State()
	bad := keepConfig
	bad.ColorConfig.BitDepth = 10
	if err := enc.SetConfig(bad); !errors.Is(err, internalencoder.ErrInvalidConfig) {
		t.Fatalf("SetConfig mismatched color bitdepth err=%v want %v", err, internalencoder.ErrInvalidConfig)
	}
	if enc.Config() != keepConfig || enc.State() != keepState {
		t.Fatalf("bad color config mutated config/state config=%+v want=%+v state=%+v want=%+v", enc.Config(), keepConfig, enc.State(), keepState)
	}
}

func publicWebRTCEncoderFrameForConfig(t *testing.T, cfg EncoderConfig) (Frame, []byte) {
	t.Helper()
	seq, err := internalencoder.SequenceHeaderForConfig(cfg)
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	format := FrameFormat{
		Width:        int(cfg.Resolution.Width),
		Height:       int(cfg.Resolution.Height),
		BitDepth:     seq.ColorConfig.BitDepth,
		MonoChrome:   seq.ColorConfig.MonoChrome,
		SubsamplingX: seq.ColorConfig.SubsamplingX,
		SubsamplingY: seq.ColorConfig.SubsamplingY,
		Align:        64,
	}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		t.Fatalf("FrameRequiredSize: %v", err)
	}
	backing := make([]byte, layout.Size)
	frame, err := internalframe.Bind(backing, format)
	if err != nil {
		t.Fatalf("BindFrame: %v", err)
	}
	return frame, backing
}

func publicSetFramePlaneSample(
	plane FramePlane,
	bytesPerSample int,
	x int,
	y int,
	value uint16,
) {
	off := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[off] = byte(value)
		return
	}
	plane.Pix[off] = byte(value)
	plane.Pix[off+1] = byte(value >> 8)
}

func TestPublicWebRTCEncoderPictureFramesRTPPackets(t *testing.T) {
	cfg := EncoderConfig{
		Resolution:        EncoderResolution{Width: 640, Height: 360},
		Scalability:       internalencoder.ScalabilityModeL2T2,
		MaxFramerate:      EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
	}
	enc, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 300})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder: %v", err)
	}
	framePayloads := [...][]byte{
		{0x10, 0x11, 0x12, 0x13},
		{0x20, 0x21, 0x22, 0x23},
	}
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 64}

	var frameOBUProbe [64]byte
	firstSize, firstUnit, err := enc.PictureTemporalUnitFramesRTPScratchSize(framePayloads[:], limits, false, frameOBUProbe[:0], nil)
	if err != nil {
		t.Fatalf("PictureTemporalUnitFramesRTPScratchSize first pass: %v", err)
	}
	if !firstUnit.Key || firstSize.FrameOBUBytes == 0 || firstSize.Packetizer.OBUs == 0 || firstSize.PacketSpans != 0 {
		t.Fatalf("first size=%+v unit=%+v", firstSize, firstUnit)
	}

	var obuScratch [4]RTPPacketizerOBU
	size, unit, err := enc.PictureTemporalUnitFramesRTPScratchSize(framePayloads[:], limits, false, frameOBUProbe[:0], obuScratch[:firstSize.Packetizer.OBUs])
	if err != nil {
		t.Fatalf("PictureTemporalUnitFramesRTPScratchSize full pass: %v", err)
	}
	if unit != firstUnit || size.FrameSpans != 2 || size.PacketSpans != 2 ||
		size.Packetizer.Packets == 0 || size.MaxPayloadBytes != limits.MaxPayloadLen ||
		size.MaxDescriptorBytes <= RTPDependencyDescriptorMandatorySize {
		t.Fatalf("size=%+v unit=%+v first=%+v", size, unit, firstUnit)
	}

	var frameOBUs [64]byte
	var frameSpans [2]encoderWebRTCFrameRTPPacketSpan
	var packetSpans [4]EncoderWebRTCRTPPacketSpan
	var packets [4]RTPPacketPlan
	var work [4]RTPPacketPlan
	scratch, err := bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size, encoderWebRTCPictureTemporalUnitFramesRTPScratch{
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
	if err == nil || !errors.Is(err, internalbitstream.ErrShortBuffer) && !errors.Is(err, ErrRTPShortBuffer) {
		t.Fatalf("short descriptor err=%v", err)
	}
	if len(shortFrameOBUs) != 0 || len(shortPayloads) != 0 || len(shortDescriptors) != 0 || enc.State() != initialState {
		t.Fatalf("short frameOBUs=% x payloads=% x descriptors=% x state=%+v want=%+v", shortFrameOBUs, shortPayloads, shortDescriptors, enc.State(), initialState)
	}

	scratch, err = bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size, encoderWebRTCPictureTemporalUnitFramesRTPScratch{
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
		len(rtpPayloads) == 0 || len(descriptorOut) <= RTPDependencyDescriptorMandatorySize ||
		enc.State().NextFrameID != 302 || enc.State().DeltaPictureIndex != 1 {
		t.Fatalf("frameOBUs=%d/%d payloads=%d descriptors=%d frames=%d packets=%d unit=%+v sized=%+v state=%+v", len(frameOBUOut), size.FrameOBUBytes, len(rtpPayloads), len(descriptorOut), frameCount, packetCount, gotUnit, unit, enc.State())
	}
	header, _, err := ParseRTPAggregationHeader(rtpPayloads[packetSpans[0].PayloadOffset:])
	if err != nil {
		t.Fatalf("ParseRTPAggregationHeader first packet: %v", err)
	}
	if header.ElementCount == 0 || !packetSpans[1].Marker {
		t.Fatalf("header=%+v packetSpans=%+v", header, packetSpans[:packetCount])
	}

	base, err := newWebRTCEncoder(cfg, internalencoder.WebRTCEncoderState{NextFrameID: 300})
	if err != nil {
		t.Fatalf("NewWebRTCEncoder alloc base: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		local := base
		localScratch, bindErr := bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size, encoderWebRTCPictureTemporalUnitFramesRTPScratch{
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
	got, err := internalencoder.WebRTCTemporalIDForDeltaPicture(3, 4)
	if err != nil {
		t.Fatalf("EncoderWebRTCTemporalIDForDeltaPicture: %v", err)
	}
	if got != 0 {
		t.Fatalf("temporal id=%d want 0", got)
	}
}

func TestPublicEncoderSequenceHeaderOBU(t *testing.T) {
	seq := internalencoder.SequenceHeader{
		Profile:               EncoderProfile0,
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
		ColorConfig: internalencoder.SequenceColorConfig{
			BitDepth:                8,
			ColorDescriptionPresent: true,
			ColorPrimaries:          internalencoder.SequenceColorPrimariesBT709,
			TransferCharacteristics: internalencoder.SequenceTransferCharacteristicsSRGB,
			MatrixCoefficients:      internalencoder.SequenceMatrixCoefficientsIdentity,
			ColorRange:              true,
		},
	}
	seq.OperatingPoints[0].SeqLevelIdx = 5

	size, err := internalencoder.LowOverheadSequenceHeaderOBUSize(seq)
	if err != nil {
		t.Fatalf("EncoderLowOverheadSequenceHeaderOBUSize: %v", err)
	}
	var buf [80]byte
	out, err := internalencoder.AppendLowOverheadSequenceHeaderOBU(buf[:0], seq)
	if err != nil {
		t.Fatalf("AppendEncoderLowOverheadSequenceHeaderOBU: %v", err)
	}
	if len(out) != size {
		t.Fatalf("sequence obu len=%d want %d", len(out), size)
	}
	unit, consumed, err := internalobu.ParseLowOverhead(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != internalobu.TypeSequenceHeader {
		t.Fatalf("parsed header=%+v consumed=%d", unit.Header, consumed)
	}
	if _, err := internalparser.ParseSequenceHeader(unit.Payload); err != nil {
		t.Fatalf("ParseSequenceHeader: %v", err)
	}
}

func publicEncoderReferenceState(seq internalencoder.SequenceHeader, indices []uint8) internalparser.ReferenceState {
	var refs internalparser.ReferenceState
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
