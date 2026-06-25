package goav1_test

import (
	"bytes"
	"errors"
	"math/bits"
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

func TestPublicEncoderWebRTCRTPFrameDuration(t *testing.T) {
	base := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 640, Height: 360},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    800,
		TargetBitrateKbps: 500,
		Scalability:       av1.EncoderScalabilityModeL1T2,
	}
	tests := []struct {
		name     string
		fps      av1.EncoderRational
		timebase av1.EncoderRational
		want     av1.EncoderRational
	}{
		{name: "defaults", want: av1.EncoderRational{Num: 3000, Den: 1}},
		{name: "thirty", fps: av1.EncoderRational{Num: 30, Den: 1}, want: av1.EncoderRational{Num: 3000, Den: 1}},
		{name: "ntsc", fps: av1.EncoderRational{Num: 30000, Den: 1001}, want: av1.EncoderRational{Num: 3003, Den: 1}},
		{name: "custom timebase", fps: av1.EncoderRational{Num: 24, Den: 1}, timebase: av1.EncoderRational{Num: 1, Den: 1000}, want: av1.EncoderRational{Num: 125, Den: 3}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			cfg.MaxFramerate = tc.fps
			cfg.RTPTimebase = tc.timebase
			got, err := av1.EncoderWebRTCRTPFrameDuration(cfg)
			if err != nil {
				t.Fatalf("EncoderWebRTCRTPFrameDuration: %v", err)
			}
			if got != tc.want {
				t.Fatalf("duration=%+v want %+v", got, tc.want)
			}
		})
	}

	invalid := base
	invalid.MaxFramerate = av1.EncoderRational{Num: 1, Den: 2}
	if _, err := av1.EncoderWebRTCRTPFrameDuration(invalid); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
		t.Fatalf("invalid framerate err=%v want %v", err, av1.ErrEncoderInvalidConfig)
	}
}

func TestPublicEncoderWebRTCScalabilityModes(t *testing.T) {
	wantNames := []string{
		"L1T1", "L1T2", "L1T3",
		"L2T1", "L2T1h", "L2T1_KEY",
		"L2T2", "L2T2h", "L2T2_KEY", "L2T2_KEY_SHIFT",
		"L2T3", "L2T3h", "L2T3_KEY", "L2T3_KEY_SHIFT",
		"L3T1", "L3T1h", "L3T1_KEY",
		"L3T2", "L3T2h", "L3T2_KEY", "L3T2_KEY_SHIFT",
		"L3T3", "L3T3h", "L3T3_KEY", "L3T3_KEY_SHIFT",
		"S2T1", "S2T1h", "S2T2", "S2T2h", "S2T3", "S2T3h",
		"S3T1", "S3T1h", "S3T2", "S3T2h", "S3T3", "S3T3h",
	}
	modes := av1.EncoderWebRTCScalabilityModes()
	if len(modes) != len(wantNames) {
		t.Fatalf("mode count=%d want %d", len(modes), len(wantNames))
	}
	seen := make(map[av1.EncoderScalabilityMode]bool, len(modes))
	for i, mode := range modes {
		if mode.String() != wantNames[i] {
			t.Fatalf("mode[%d]=%s want %s", i, mode, wantNames[i])
		}
		if seen[mode] {
			t.Fatalf("duplicate mode %s", mode)
		}
		seen[mode] = true
		parsed, ok := av1.ParseEncoderScalabilityMode(wantNames[i])
		if !ok || parsed != mode {
			t.Fatalf("ParseEncoderScalabilityMode(%q)=%s,%v want %s,true", wantNames[i], parsed, ok, mode)
		}
		normalized, err := av1.SetWebRTCEncoderSVCConfig(av1.EncoderConfig{
			Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
			MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
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

	modes[0] = av1.EncoderScalabilityModeS3T3h
	if got := av1.EncoderWebRTCScalabilityModes()[0]; got != av1.EncoderScalabilityModeL1T1 {
		t.Fatalf("EncoderWebRTCScalabilityModes aliased caller mutation: %s", got)
	}
	prefix := []av1.EncoderScalabilityMode{av1.EncoderScalabilityModeL3T3}
	appended := av1.AppendEncoderWebRTCScalabilityModes(prefix)
	if len(appended) != len(wantNames)+1 || appended[0] != av1.EncoderScalabilityModeL3T3 ||
		appended[1] != av1.EncoderScalabilityModeL1T1 || appended[len(appended)-1] != av1.EncoderScalabilityModeS3T3h {
		t.Fatalf("appended modes=%v", appended)
	}
}

func TestPublicValidateEncoderWebRTCActiveScalabilityModes(t *testing.T) {
	for _, modes := range [][]av1.EncoderScalabilityMode{
		nil,
		{av1.EncoderScalabilityModeL1T1},
		{av1.EncoderScalabilityModeS3T3h},
		{av1.EncoderScalabilityModeL1T3, av1.EncoderScalabilityModeL1T3, av1.EncoderScalabilityModeL1T3},
		{av1.EncoderScalabilityModeL2T3_KEY_SHIFT, av1.EncoderScalabilityModeL3T3_KEY},
	} {
		if err := av1.ValidateEncoderWebRTCActiveScalabilityModes(modes...); err != nil {
			t.Fatalf("ValidateEncoderWebRTCActiveScalabilityModes(%v): %v", modes, err)
		}
	}

	for _, modes := range [][]av1.EncoderScalabilityMode{
		{av1.EncoderScalabilityMode(255)},
		{av1.EncoderScalabilityModeL1T3, av1.EncoderScalabilityModeS2T1},
		{av1.EncoderScalabilityModeS2T3h, av1.EncoderScalabilityModeS3T3},
	} {
		if err := av1.ValidateEncoderWebRTCActiveScalabilityModes(modes...); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
			t.Fatalf("ValidateEncoderWebRTCActiveScalabilityModes(%v) err=%v want %v", modes, err, av1.ErrEncoderInvalidConfig)
		}
	}
}

func TestPublicEncoderWebRTCScalabilityModeIDC(t *testing.T) {
	want := map[string]struct {
		idc uint8
		ok  bool
	}{
		"L1T1":           {},
		"L1T2":           {idc: av1.MetadataScalabilityModeL1T2, ok: true},
		"L1T3":           {idc: av1.MetadataScalabilityModeL1T3, ok: true},
		"L2T1":           {idc: av1.MetadataScalabilityModeL2T1, ok: true},
		"L2T1h":          {idc: av1.MetadataScalabilityModeL2T1h, ok: true},
		"L2T1_KEY":       {},
		"L2T2":           {idc: av1.MetadataScalabilityModeL2T2, ok: true},
		"L2T2h":          {idc: av1.MetadataScalabilityModeL2T2h, ok: true},
		"L2T2_KEY":       {idc: av1.MetadataScalabilityModeL3T2_KEY, ok: true},
		"L2T2_KEY_SHIFT": {idc: av1.MetadataScalabilityModeL3T2_KEY_SHIFT, ok: true},
		"L2T3":           {idc: av1.MetadataScalabilityModeL2T3, ok: true},
		"L2T3h":          {idc: av1.MetadataScalabilityModeL2T3h, ok: true},
		"L2T3_KEY":       {idc: av1.MetadataScalabilityModeL3T3_KEY, ok: true},
		"L2T3_KEY_SHIFT": {idc: av1.MetadataScalabilityModeL3T3_KEY_SHIFT, ok: true},
		"L3T1":           {idc: av1.MetadataScalabilityModeL3T1, ok: true},
		"L3T1h":          {},
		"L3T1_KEY":       {},
		"L3T2":           {idc: av1.MetadataScalabilityModeL3T2, ok: true},
		"L3T2h":          {},
		"L3T2_KEY":       {idc: av1.MetadataScalabilityModeL4T5_KEY, ok: true},
		"L3T2_KEY_SHIFT": {idc: av1.MetadataScalabilityModeL4T5_KEY_SHIFT, ok: true},
		"L3T3":           {idc: av1.MetadataScalabilityModeL3T3, ok: true},
		"L3T3h":          {},
		"L3T3_KEY":       {idc: av1.MetadataScalabilityModeL4T7_KEY, ok: true},
		"L3T3_KEY_SHIFT": {idc: av1.MetadataScalabilityModeL4T7_KEY_SHIFT, ok: true},
		"S2T1":           {idc: av1.MetadataScalabilityModeS2T1, ok: true},
		"S2T1h":          {idc: av1.MetadataScalabilityModeS2T1h, ok: true},
		"S2T2":           {idc: av1.MetadataScalabilityModeS2T2, ok: true},
		"S2T2h":          {idc: av1.MetadataScalabilityModeS2T2h, ok: true},
		"S2T3":           {idc: av1.MetadataScalabilityModeS2T3, ok: true},
		"S2T3h":          {idc: av1.MetadataScalabilityModeS2T3h, ok: true},
		"S3T1":           {idc: av1.MetadataScalabilityModeS3T1, ok: true},
		"S3T1h":          {},
		"S3T2":           {idc: av1.MetadataScalabilityModeS3T2, ok: true},
		"S3T2h":          {},
		"S3T3":           {idc: av1.MetadataScalabilityModeS3T3, ok: true},
		"S3T3h":          {},
	}
	for _, mode := range av1.EncoderWebRTCScalabilityModes() {
		tc, ok := want[mode.String()]
		if !ok {
			t.Fatalf("missing expected IDC entry for %s", mode)
		}
		got, gotOK := av1.EncoderWebRTCScalabilityModeIDC(mode)
		if got != tc.idc || gotOK != tc.ok {
			t.Fatalf("EncoderWebRTCScalabilityModeIDC(%s)=%d,%v want %d,%v", mode, got, gotOK, tc.idc, tc.ok)
		}
		got, gotOK = mode.AV1ScalabilityModeIDC()
		if got != tc.idc || gotOK != tc.ok {
			t.Fatalf("%s.AV1ScalabilityModeIDC()=%d,%v want %d,%v", mode, got, gotOK, tc.idc, tc.ok)
		}
	}
	if got, ok := av1.EncoderWebRTCScalabilityModeIDC(av1.EncoderScalabilityMode(255)); got != 0 || ok {
		t.Fatalf("invalid mode IDC=%d,%v want 0,false", got, ok)
	}
}

func TestPublicMetadataScalabilityModeIDCConstants(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  uint8
		want uint8
	}{
		{name: "L1T2", got: av1.MetadataScalabilityModeL1T2, want: 0},
		{name: "L1T3", got: av1.MetadataScalabilityModeL1T3, want: 1},
		{name: "L2T1", got: av1.MetadataScalabilityModeL2T1, want: 2},
		{name: "L2T2", got: av1.MetadataScalabilityModeL2T2, want: 3},
		{name: "L2T3", got: av1.MetadataScalabilityModeL2T3, want: 4},
		{name: "S2T1", got: av1.MetadataScalabilityModeS2T1, want: 5},
		{name: "S2T2", got: av1.MetadataScalabilityModeS2T2, want: 6},
		{name: "S2T3", got: av1.MetadataScalabilityModeS2T3, want: 7},
		{name: "L2T1h", got: av1.MetadataScalabilityModeL2T1h, want: 8},
		{name: "L2T2h", got: av1.MetadataScalabilityModeL2T2h, want: 9},
		{name: "L2T3h", got: av1.MetadataScalabilityModeL2T3h, want: 10},
		{name: "S2T1h", got: av1.MetadataScalabilityModeS2T1h, want: 11},
		{name: "S2T2h", got: av1.MetadataScalabilityModeS2T2h, want: 12},
		{name: "S2T3h", got: av1.MetadataScalabilityModeS2T3h, want: 13},
		{name: "SS", got: av1.MetadataScalabilityModeSS, want: 14},
		{name: "L3T1", got: av1.MetadataScalabilityModeL3T1, want: 15},
		{name: "L3T2", got: av1.MetadataScalabilityModeL3T2, want: 16},
		{name: "L3T3", got: av1.MetadataScalabilityModeL3T3, want: 17},
		{name: "S3T1", got: av1.MetadataScalabilityModeS3T1, want: 18},
		{name: "S3T2", got: av1.MetadataScalabilityModeS3T2, want: 19},
		{name: "S3T3", got: av1.MetadataScalabilityModeS3T3, want: 20},
		{name: "L3T2_KEY", got: av1.MetadataScalabilityModeL3T2_KEY, want: 21},
		{name: "L3T3_KEY", got: av1.MetadataScalabilityModeL3T3_KEY, want: 22},
		{name: "L4T5_KEY", got: av1.MetadataScalabilityModeL4T5_KEY, want: 23},
		{name: "L4T7_KEY", got: av1.MetadataScalabilityModeL4T7_KEY, want: 24},
		{name: "L3T2_KEY_SHIFT", got: av1.MetadataScalabilityModeL3T2_KEY_SHIFT, want: 25},
		{name: "L3T3_KEY_SHIFT", got: av1.MetadataScalabilityModeL3T3_KEY_SHIFT, want: 26},
		{name: "L4T5_KEY_SHIFT", got: av1.MetadataScalabilityModeL4T5_KEY_SHIFT, want: 27},
		{name: "L4T7_KEY_SHIFT", got: av1.MetadataScalabilityModeL4T7_KEY_SHIFT, want: 28},
	} {
		if tc.got != tc.want {
			t.Fatalf("%s IDC=%d want %d", tc.name, tc.got, tc.want)
		}
	}
}

func TestPublicEncoderWebRTCScalabilityMetadataOBU(t *testing.T) {
	size, ok, err := av1.EncoderLowOverheadWebRTCScalabilityMetadataOBUSize(av1.EncoderScalabilityModeL2T2)
	if err != nil || !ok {
		t.Fatalf("EncoderLowOverheadWebRTCScalabilityMetadataOBUSize: size=%d ok=%v err=%v", size, ok, err)
	}
	var buf [8]byte
	out, ok, err := av1.AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU(buf[:0], av1.EncoderScalabilityModeL2T2)
	if err != nil || !ok {
		t.Fatalf("AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU: ok=%v err=%v", ok, err)
	}
	if len(out) != size {
		t.Fatalf("len=%d want %d", len(out), size)
	}
	unit, consumed, err := av1.ParseLowOverheadOBU(out)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU: %v", err)
	}
	if consumed != len(out) || unit.Header.Type != av1.OBUMetadata {
		t.Fatalf("metadata consumed=%d header=%+v", consumed, unit.Header)
	}
	meta, err := av1.ParseMetadataOBU(unit.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != av1.MetadataTypeScalability || meta.Scalability.ModeIDC != av1.MetadataScalabilityModeL2T2 || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v", meta)
	}

	ssSize, ok, err := av1.EncoderLowOverheadWebRTCScalabilityMetadataOBUSize(av1.EncoderScalabilityModeS3T3h)
	if err != nil || !ok {
		t.Fatalf("EncoderLowOverheadWebRTCScalabilityMetadataOBUSize S3T3h: size=%d ok=%v err=%v", ssSize, ok, err)
	}
	var ssBuf [64]byte
	ssOut, ok, err := av1.AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU(ssBuf[:0], av1.EncoderScalabilityModeS3T3h)
	if err != nil || !ok {
		t.Fatalf("AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU S3T3h: ok=%v err=%v", ok, err)
	}
	if len(ssOut) != ssSize {
		t.Fatalf("S3T3h len=%d want %d", len(ssOut), ssSize)
	}
	ssUnit, _, err := av1.ParseLowOverheadOBU(ssOut)
	if err != nil {
		t.Fatalf("ParseLowOverheadOBU S3T3h: %v", err)
	}
	ssMeta, err := av1.ParseMetadataOBU(ssUnit.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU S3T3h: %v", err)
	}
	if ssMeta.Type != av1.MetadataTypeScalability ||
		ssMeta.Scalability.ModeIDC != av1.MetadataScalabilityModeSS ||
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
	noMetadata, ok, err := av1.AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU(prefix, av1.EncoderScalabilityModeL1T1)
	if err != nil || ok || !bytes.Equal(noMetadata, prefix) {
		t.Fatalf("L1T1 metadata out=% x ok=%v err=%v", noMetadata, ok, err)
	}
}

func TestPublicEncoderWebRTCActiveDecodeTargetsMask(t *testing.T) {
	for _, mode := range av1.EncoderWebRTCScalabilityModes() {
		t.Run(mode.String(), func(t *testing.T) {
			spatialLayers, temporalLayers, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid mode %s", mode)
			}
			normalized, err := av1.SetWebRTCEncoderSVCConfig(av1.EncoderConfig{
				Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
				MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
				MinBitrateKbps:    120,
				MaxBitrateKbps:    1600,
				TargetBitrateKbps: 900,
				Scalability:       mode,
			}, 0, 0)
			if err != nil {
				t.Fatalf("SetWebRTCEncoderSVCConfig(%s): %v", mode, err)
			}
			structure, err := av1.EncoderWebRTCFrameDependencyStructureForConfig(normalized)
			if err != nil {
				t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", mode, err)
			}
			all, err := av1.EncoderWebRTCAllDecodeTargetsMask(structure)
			if err != nil {
				t.Fatalf("EncoderWebRTCAllDecodeTargetsMask(%s): %v", mode, err)
			}
			wantTargets := int(spatialLayers * temporalLayers)
			if bits.OnesCount32(all) != wantTargets {
				t.Fatalf("all mask=%#x targets=%d want %d", all, bits.OnesCount32(all), wantTargets)
			}
			top, err := av1.EncoderWebRTCActiveDecodeTargetsMask(structure, spatialLayers-1, temporalLayers-1)
			if err != nil {
				t.Fatalf("EncoderWebRTCActiveDecodeTargetsMask top(%s): %v", mode, err)
			}
			if top != all {
				t.Fatalf("top mask=%#x want all %#x", top, all)
			}
			baseSpatial, err := av1.EncoderWebRTCActiveDecodeTargetsMask(structure, 0, temporalLayers-1)
			if err != nil {
				t.Fatalf("EncoderWebRTCActiveDecodeTargetsMask base spatial(%s): %v", mode, err)
			}
			if bits.OnesCount32(baseSpatial) != int(temporalLayers) {
				t.Fatalf("base spatial mask=%#x targets=%d want %d", baseSpatial, bits.OnesCount32(baseSpatial), temporalLayers)
			}
			baseTemporal, err := av1.EncoderWebRTCActiveDecodeTargetsMask(structure, spatialLayers-1, 0)
			if err != nil {
				t.Fatalf("EncoderWebRTCActiveDecodeTargetsMask base temporal(%s): %v", mode, err)
			}
			if bits.OnesCount32(baseTemporal) != int(spatialLayers) {
				t.Fatalf("base temporal mask=%#x targets=%d want %d", baseTemporal, bits.OnesCount32(baseTemporal), spatialLayers)
			}
			if _, err := av1.EncoderWebRTCActiveDecodeTargetsMask(structure, av1.EncoderWebRTCMaxSpatialLayers, 0); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
				t.Fatalf("invalid spatial mask err=%v want %v", err, av1.ErrEncoderInvalidFrame)
			}
			if _, err := av1.EncoderWebRTCActiveDecodeTargetsMask(structure, 0, av1.EncoderWebRTCMaxTemporalLayers); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
				t.Fatalf("invalid temporal mask err=%v want %v", err, av1.ErrEncoderInvalidFrame)
			}
		})
	}
}

func TestPublicEncoderWebRTCSpatialDecodeTargetsMask(t *testing.T) {
	for _, mode := range av1.EncoderWebRTCScalabilityModes() {
		t.Run(mode.String(), func(t *testing.T) {
			spatialLayers, temporalLayers, _, ok := mode.Layers()
			if !ok {
				t.Fatalf("invalid mode %s", mode)
			}
			normalized, err := av1.SetWebRTCEncoderSVCConfig(av1.EncoderConfig{
				Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
				MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
				MinBitrateKbps:    120,
				MaxBitrateKbps:    1600,
				TargetBitrateKbps: 900,
				Scalability:       mode,
			}, 0, 0)
			if err != nil {
				t.Fatalf("SetWebRTCEncoderSVCConfig(%s): %v", mode, err)
			}
			structure, err := av1.EncoderWebRTCFrameDependencyStructureForConfig(normalized)
			if err != nil {
				t.Fatalf("EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", mode, err)
			}
			for spatialID := uint8(0); spatialID < spatialLayers; spatialID++ {
				mask, err := av1.EncoderWebRTCSpatialDecodeTargetsMask(structure, spatialID, temporalLayers-1)
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
			if _, err := av1.EncoderWebRTCSpatialDecodeTargetsMask(structure, spatialLayers, 0); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
				t.Fatalf("invalid spatial mask err=%v want %v", err, av1.ErrEncoderInvalidFrame)
			}
			if _, err := av1.EncoderWebRTCSpatialDecodeTargetsMask(structure, 0, temporalLayers); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
				t.Fatalf("invalid temporal mask err=%v want %v", err, av1.ErrEncoderInvalidFrame)
			}
		})
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
	assertPublicWebRTCHeaderScalabilityMetadata(t, keyOut, av1.MetadataScalabilityModeL2T2, true)

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
	if got, err := enc.RTPFrameDuration(); err != nil || got != (av1.EncoderRational{Num: 3000, Den: 1}) {
		t.Fatalf("initial RTPFrameDuration=%+v err=%v", got, err)
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
	if got, err := enc.RTPFrameDuration(); err != nil || got != (av1.EncoderRational{Num: 1500, Den: 1}) {
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

func TestPublicWebRTCEncoderSetConfigRejectsInvalidControlsWithoutMutation(t *testing.T) {
	cfg := av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
		Scalability:       av1.EncoderScalabilityModeS2T2,
		MaxFramerate:      av1.EncoderRational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1200,
		TargetBitrateKbps: 700,
	}
	enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 900})
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
		edit func(*av1.EncoderConfig)
	}{
		{name: "zero-fps", edit: func(cfg *av1.EncoderConfig) {
			cfg.MaxFramerate = av1.EncoderRational{Num: 0, Den: 1}
		}},
		{name: "sub-one-fps", edit: func(cfg *av1.EncoderConfig) {
			cfg.MaxFramerate = av1.EncoderRational{Num: 1, Den: 2}
		}},
		{name: "min-above-max-bitrate", edit: func(cfg *av1.EncoderConfig) {
			cfg.MinBitrateKbps = cfg.MaxBitrateKbps + 1
		}},
		{name: "target-below-min-bitrate", edit: func(cfg *av1.EncoderConfig) {
			cfg.TargetBitrateKbps = cfg.MinBitrateKbps - 1
		}},
		{name: "target-above-max-bitrate", edit: func(cfg *av1.EncoderConfig) {
			cfg.TargetBitrateKbps = cfg.MaxBitrateKbps + 1
		}},
		{name: "max-bitrate-limit", edit: func(cfg *av1.EncoderConfig) {
			cfg.MaxBitrateKbps = av1.EncoderWebRTCMaxBitrateKbps + 1
			cfg.TargetBitrateKbps = cfg.MinBitrateKbps
		}},
		{name: "cbr-quantizer", edit: func(cfg *av1.EncoderConfig) {
			cfg.RateControl = av1.EncoderRateControlCBR
			cfg.Quantizer = 1
		}},
		{name: "cqp-quantizer-limit", edit: func(cfg *av1.EncoderConfig) {
			cfg.RateControl = av1.EncoderRateControlCQP
			cfg.Quantizer = uint8(av1.EncoderWebRTCMaxQuantizer + 1)
		}},
		{name: "spatial-layer-min-above-max", edit: func(cfg *av1.EncoderConfig) {
			cfg.SpatialLayers[0].MinBitrateKbps = cfg.SpatialLayers[0].MaxBitrateKbps + 1
		}},
		{name: "spatial-layer-target-above-max", edit: func(cfg *av1.EncoderConfig) {
			cfg.SpatialLayers[1].TargetBitrateKbps = cfg.SpatialLayers[1].MaxBitrateKbps + 1
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bad := keepConfig
			tc.edit(&bad)
			if err := enc.SetConfig(bad); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
				t.Fatalf("SetConfig err=%v want %v", err, av1.ErrEncoderInvalidConfig)
			}
			if enc.Config() != keepConfig || enc.State() != keepState {
				t.Fatalf("SetConfig mutated state/config for %s config=%+v want=%+v state=%+v want=%+v",
					tc.name, enc.Config(), keepConfig, enc.State(), keepState)
			}
		})
	}
}

func TestPublicWebRTCEncoderSetConfigScalabilityTransitionMatrix(t *testing.T) {
	modes := av1.EncoderWebRTCScalabilityModes()
	for fromIndex, fromMode := range modes {
		fromMode := fromMode
		for toIndex, toMode := range modes {
			toMode := toMode
			t.Run(fromMode.String()+"->"+toMode.String(), func(t *testing.T) {
				const firstFrameID = 7000
				fromCfg := publicWebRTCControllerTransitionConfig(fromMode, fromIndex)
				toCfg := publicWebRTCControllerTransitionConfig(toMode, len(modes)+toIndex)
				enc, err := av1.NewWebRTCEncoder(fromCfg, av1.EncoderWebRTCState{NextFrameID: firstFrameID})
				if err != nil {
					t.Fatalf("NewWebRTCEncoder(%s): %v", fromMode, err)
				}
				if got := enc.Config().Scalability; got != fromMode {
					t.Fatalf("initial config mode=%s want %s", got, fromMode)
				}
				assertPublicRTCConfigControls(t, enc.Config(), fromCfg)
				assertPublicWebRTCEncoderDuration(t, &enc)

				var receiver av1.RTPDependencyDescriptorState
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

func publicWebRTCControllerSetConfigNeedsKey(t *testing.T, from av1.EncoderConfig, to av1.EncoderConfig) bool {
	t.Helper()
	fromSeq, err := av1.EncoderSequenceHeaderForConfig(from)
	if err != nil {
		t.Fatalf("from SequenceHeaderForConfig(%s): %v", from.Scalability, err)
	}
	toSeq, err := av1.EncoderSequenceHeaderForConfig(to)
	if err != nil {
		t.Fatalf("to SequenceHeaderForConfig(%s): %v", to.Scalability, err)
	}
	fromStructure, err := av1.EncoderWebRTCFrameDependencyStructureForConfig(from)
	if err != nil {
		t.Fatalf("from EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", from.Scalability, err)
	}
	toStructure, err := av1.EncoderWebRTCFrameDependencyStructureForConfig(to)
	if err != nil {
		t.Fatalf("to EncoderWebRTCFrameDependencyStructureForConfig(%s): %v", to.Scalability, err)
	}
	return fromSeq != toSeq || fromStructure != toStructure
}

func TestPublicWebRTCEncoderControllerSettingsMatrix(t *testing.T) {
	profiles := []struct {
		name        string
		fps         av1.EncoderRational
		rateControl av1.EncoderRateControlMode
		quantizer   uint8
		minKbps     int32
		maxKbps     int32
		targetKbps  int32
		content     av1.EncoderContentHint
	}{
		{
			name:        "camera-cbr-30fps",
			fps:         av1.EncoderRational{Num: 30, Den: 1},
			rateControl: av1.EncoderRateControlCBR,
			minKbps:     120,
			maxKbps:     1800,
			targetKbps:  900,
			content:     av1.EncoderContentCamera,
		},
		{
			name:        "screen-cqp-ntsc",
			fps:         av1.EncoderRational{Num: 30000, Den: 1001},
			rateControl: av1.EncoderRateControlCQP,
			quantizer:   37,
			minKbps:     80,
			maxKbps:     1200,
			targetKbps:  640,
			content:     av1.EncoderContentScreen,
		},
	}

	for _, modeName := range publicWebRTCControllerModeNames() {
		mode, ok := av1.ParseEncoderScalabilityMode(modeName)
		if !ok {
			t.Fatalf("ParseEncoderScalabilityMode(%q) failed", modeName)
		}
		for _, profile := range profiles {
			t.Run(modeName+"/"+profile.name, func(t *testing.T) {
				cfg := av1.EncoderConfig{
					Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
					Scalability:       mode,
					MaxFramerate:      profile.fps,
					RateControl:       profile.rateControl,
					Quantizer:         profile.quantizer,
					MinBitrateKbps:    profile.minKbps,
					MaxBitrateKbps:    profile.maxKbps,
					TargetBitrateKbps: profile.targetKbps,
					Content:           profile.content,
				}
				enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 1000})
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

				var receiver av1.RTPDependencyDescriptorState
				history := make(map[uint64]publicWebRTCControllerLayer, 32)
				nextFrameID := uint64(1000)
				key, err := enc.NextTemporalUnit(false)
				if err != nil {
					t.Fatalf("initial key NextTemporalUnit: %v", err)
				}
				assertPublicWebRTCControllerUnit(t, &receiver, normalized, enc.State(), key, true, 0, &nextFrameID, history)

				controlChange := normalized
				controlChange.MaxFramerate = av1.EncoderRational{Num: 60, Den: 1}
				if controlChange.RateControl == av1.EncoderRateControlCQP {
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

func assertPublicWebRTCEncoderDuration(t *testing.T, enc *av1.WebRTCEncoder) {
	t.Helper()
	cfg := enc.Config()
	want, err := av1.EncoderWebRTCRTPFrameDuration(cfg)
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
	it := av1.NewLowOverheadIterator(tu)
	if td, ok, err := it.Next(); err != nil || !ok || td.Header.Type != av1.OBUTemporalDelimiter {
		t.Fatalf("TD ok=%v err=%v header=%+v", ok, err, td.Header)
	}
	if seq, ok, err := it.Next(); err != nil || !ok || seq.Header.Type != av1.OBUSequenceHeader {
		t.Fatalf("sequence ok=%v err=%v header=%+v", ok, err, seq.Header)
	}
	next, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("next OBU ok=%v err=%v header=%+v", ok, err, next.Header)
	}
	if !wantPresent {
		if next.Header.Type == av1.OBUMetadata {
			t.Fatalf("unexpected metadata payload=% x", next.Payload)
		}
		return
	}
	if next.Header.Type != av1.OBUMetadata {
		t.Fatalf("metadata header=%+v", next.Header)
	}
	meta, err := av1.ParseMetadataOBU(next.Payload)
	if err != nil {
		t.Fatalf("ParseMetadataOBU: %v", err)
	}
	if meta.Type != av1.MetadataTypeScalability || meta.Scalability.ModeIDC != wantIDC || meta.Scalability.HasStructure {
		t.Fatalf("metadata=%+v want idc=%d", meta, wantIDC)
	}
}

func publicWebRTCControllerTransitionConfig(mode av1.EncoderScalabilityMode, step int) av1.EncoderConfig {
	fps := []av1.EncoderRational{
		{Num: 30, Den: 1},
		{Num: 60, Den: 1},
		{Num: 30000, Den: 1001},
		{Num: 24, Den: 1},
		{Num: 15, Den: 1},
	}
	minKbps := int32(120 + (step%5)*20)
	maxKbps := int32(1400 + (step%7)*120)
	targetKbps := minKbps + (maxKbps-minKbps)/2
	content := av1.EncoderContentCamera
	if step%2 == 1 {
		content = av1.EncoderContentScreen
	}
	return av1.EncoderConfig{
		Resolution:        av1.EncoderResolution{Width: 1280, Height: 720},
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
	modes := av1.EncoderWebRTCScalabilityModes()
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

func assertPublicWebRTCControllerUnit(t *testing.T, receiver *av1.RTPDependencyDescriptorState, cfg av1.EncoderConfig, descriptorState av1.EncoderWebRTCState, unit av1.EncoderWebRTCPictureTemporalUnit, wantKey bool, deltaPictureIndex uint64, nextFrameID *uint64, history map[uint64]publicWebRTCControllerLayer) {
	t.Helper()
	spatialLayers, temporalLayers, _, ok := cfg.Scalability.Layers()
	if !ok {
		t.Fatalf("invalid mode=%s", cfg.Scalability)
	}
	frameNum := av1.EncoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum != spatialLayers || unit.Key != wantKey || unit.Delta == wantKey {
		t.Fatalf("unit key=%v delta=%v frames=%d want key=%v frames=%d", unit.Key, unit.Delta, frameNum, wantKey, spatialLayers)
	}
	for i := uint8(0); i < frameNum; i++ {
		control, structure, err := av1.EncoderWebRTCPictureTemporalUnitFrameControl(unit, descriptorState, i)
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
			(cfg.RateControl == av1.EncoderRateControlCQP && settings.Quantizer != cfg.Quantizer) ||
			settings.EffortLevel != cfg.Speed {
			t.Fatalf("frame %d rate settings=%+v config=%+v", i, settings, cfg)
		}
		if cfg.Content == av1.EncoderContentScreen {
			if wantKey {
				if !unit.KeyUnit.Header.Prefix.AllowScreenContentTools || !unit.KeyUnit.Header.Prefix.ForceIntegerMV {
					t.Fatalf("screen key prefix=%+v", unit.KeyUnit.Header.Prefix)
				}
			} else if !unit.DeltaUnit.Headers[i].Prefix.AllowScreenContentTools || !unit.DeltaUnit.Headers[i].Prefix.ForceIntegerMV {
				t.Fatalf("screen delta header[%d]=%+v", i, unit.DeltaUnit.Headers[i])
			}
		}
		wantAttach := wantKey && i == 0
		if control.AttachDependencyStructure != wantAttach {
			t.Fatalf("frame %d attach=%v want %v", i, control.AttachDependencyStructure, wantAttach)
		}
		var descriptorBuf [2048]byte
		descriptor, err := av1.AppendEncoderWebRTCPictureTemporalUnitDependencyDescriptor(
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
		wantActiveMask, err := av1.EncoderWebRTCAllDecodeTargetsMask(structure)
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

func publicWebRTCExpectedTemporalID(mode av1.EncoderScalabilityMode, spatialID uint8, deltaPictureIndex uint64) uint8 {
	if mode.UsesKeyFrameInterLayerDependencyShift() {
		switch mode {
		case av1.EncoderScalabilityModeL2T2_KEY_SHIFT:
			table := [2][2]uint8{{0, 1}, {1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case av1.EncoderScalabilityModeL2T3_KEY_SHIFT:
			table := [4][2]uint8{{2, 2}, {0, 1}, {2, 2}, {1, 0}}
			return table[(deltaPictureIndex-1)%4][spatialID]
		case av1.EncoderScalabilityModeL3T2_KEY_SHIFT:
			table := [2][3]uint8{{0, 0, 1}, {1, 1, 0}}
			return table[(deltaPictureIndex-1)%2][spatialID]
		case av1.EncoderScalabilityModeL3T3_KEY_SHIFT:
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
	assertPublicWebRTCHeaderScalabilityMetadata(t, out, av1.MetadataScalabilityModeL2T2, true)

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

func TestPublicWebRTCEncoderPictureSamplePlanesNonI420Formats(t *testing.T) {
	tests := []struct {
		name         string
		cfg          av1.EncoderConfig
		wantBitDepth uint8
		wantSubX     bool
		wantSubY     bool
	}{
		{
			name: "profile1-444-8bit",
			cfg: av1.EncoderConfig{
				Resolution:   av1.EncoderResolution{Width: 64, Height: 64},
				Profile:      av1.EncoderProfile1,
				BitDepth:     8,
				MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
			},
			wantBitDepth: 8,
		},
		{
			name: "profile0-420-10bit",
			cfg: av1.EncoderConfig{
				Resolution:   av1.EncoderResolution{Width: 64, Height: 64},
				Profile:      av1.EncoderProfile0,
				BitDepth:     10,
				MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
			},
			wantBitDepth: 10,
			wantSubX:     true,
			wantSubY:     true,
		},
		{
			name: "profile2-422-12bit",
			cfg: av1.EncoderConfig{
				Resolution:   av1.EncoderResolution{Width: 64, Height: 64},
				Profile:      av1.EncoderProfile2,
				BitDepth:     12,
				MaxFramerate: av1.EncoderRational{Num: 30, Den: 1},
			},
			wantBitDepth: 12,
			wantSubX:     true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := av1.NewWebRTCEncoder(tc.cfg, av1.EncoderWebRTCState{})
			if err != nil {
				t.Fatalf("NewWebRTCEncoder: %v", err)
			}
			normalized := enc.Config()
			seq, err := av1.EncoderSequenceHeaderForConfig(normalized)
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

			if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize([]av1.Frame{frame}, false); err != nil {
				t.Fatalf("LowOverheadPictureHeaderTemporalUnitForFramesSize: %v", err)
			}
			size, unit, err := enc.PictureSampleScratchSize([]av1.Frame{frame}, false)
			if err != nil {
				t.Fatalf("PictureSampleScratchSize: %v", err)
			}
			if size.Samples == 0 || size.FrameNum != 1 || !unit.Key {
				t.Fatalf("scratch size=%+v unit=%+v", size, unit)
			}
			samples := make([]uint16, size.Samples)
			planes, gotUnit, err := enc.LoadPictureSamplePlanes(
				av1.EncoderWebRTCPictureSampleScratch{Samples: samples},
				[]av1.Frame{frame},
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
			if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize([]av1.Frame{bad}, false); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
				t.Fatalf("mismatched bitdepth err=%v want %v", err, av1.ErrEncoderInvalidFrame)
			}
			bad = frame
			bad.Format.SubsamplingX = !bad.Format.SubsamplingX
			if _, _, err := enc.LowOverheadPictureHeaderTemporalUnitForFramesSize([]av1.Frame{bad}, false); !errors.Is(err, av1.ErrEncoderInvalidFrame) {
				t.Fatalf("mismatched subsampling err=%v want %v", err, av1.ErrEncoderInvalidFrame)
			}
		})
	}
}

func TestPublicWebRTCEncoderExplicitColorConfigSamplePlanes(t *testing.T) {
	tests := []struct {
		name string
		cfg  av1.EncoderConfig
		want av1.EncoderSequenceColorConfig
	}{
		{
			name: "profile2-420-12bit-svc",
			cfg: av1.EncoderConfig{
				Resolution:     av1.EncoderResolution{Width: 640, Height: 360},
				Profile:        av1.EncoderProfile2,
				BitDepth:       12,
				Scalability:    av1.EncoderScalabilityModeL2T1,
				MaxFramerate:   av1.EncoderRational{Num: 30, Den: 1},
				ColorConfigSet: true,
				ColorConfig: av1.EncoderSequenceColorConfig{
					BitDepth:             12,
					SubsamplingX:         true,
					SubsamplingY:         true,
					ChromaSamplePosition: 1,
				},
			},
			want: av1.EncoderSequenceColorConfig{
				BitDepth:             12,
				SubsamplingX:         true,
				SubsamplingY:         true,
				ChromaSamplePosition: 1,
			},
		},
		{
			name: "profile2-444-12bit-color-bitdepth-only-svc",
			cfg: av1.EncoderConfig{
				Resolution:     av1.EncoderResolution{Width: 640, Height: 360},
				Profile:        av1.EncoderProfile2,
				Scalability:    av1.EncoderScalabilityModeL2T1,
				MaxFramerate:   av1.EncoderRational{Num: 30, Den: 1},
				ColorConfigSet: true,
				ColorConfig: av1.EncoderSequenceColorConfig{
					BitDepth: 12,
				},
			},
			want: av1.EncoderSequenceColorConfig{
				BitDepth: 12,
			},
		},
		{
			name: "profile0-mono-10bit",
			cfg: av1.EncoderConfig{
				Resolution:     av1.EncoderResolution{Width: 64, Height: 64},
				Profile:        av1.EncoderProfile0,
				BitDepth:       10,
				MaxFramerate:   av1.EncoderRational{Num: 30, Den: 1},
				ColorConfigSet: true,
				ColorConfig: av1.EncoderSequenceColorConfig{
					BitDepth:   10,
					MonoChrome: true,
				},
			},
			want: av1.EncoderSequenceColorConfig{
				BitDepth:     10,
				MonoChrome:   true,
				SubsamplingX: true,
				SubsamplingY: true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := av1.NewWebRTCEncoder(tc.cfg, av1.EncoderWebRTCState{NextFrameID: 400})
			if err != nil {
				t.Fatalf("NewWebRTCEncoder: %v", err)
			}
			normalized := enc.Config()
			if !normalized.ColorConfigSet || normalized.ColorConfig != tc.want || normalized.BitDepth != tc.want.BitDepth {
				t.Fatalf("normalized color config=%+v set=%v bitdepth=%d want %+v", normalized.ColorConfig, normalized.ColorConfigSet, normalized.BitDepth, tc.want)
			}
			seq, err := av1.EncoderSequenceHeaderForConfig(normalized)
			if err != nil {
				t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
			}
			if seq.ColorConfig != tc.want {
				t.Fatalf("sequence color config=%+v want %+v", seq.ColorConfig, tc.want)
			}

			frameNum := int(normalized.SpatialLayerCount)
			frames := make([]av1.Frame, frameNum)
			backing := make([][]byte, frameNum)
			for i := 0; i < frameNum; i++ {
				layer := normalized.SpatialLayers[i]
				format := av1.FrameFormat{
					Width:        int(layer.Resolution.Width),
					Height:       int(layer.Resolution.Height),
					BitDepth:     seq.ColorConfig.BitDepth,
					MonoChrome:   seq.ColorConfig.MonoChrome,
					SubsamplingX: seq.ColorConfig.SubsamplingX,
					SubsamplingY: seq.ColorConfig.SubsamplingY,
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
				av1.EncoderWebRTCPictureSampleScratch{Samples: make([]uint16, size.Samples)},
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
	cfg := av1.EncoderConfig{
		Resolution:     av1.EncoderResolution{Width: 640, Height: 360},
		Profile:        av1.EncoderProfile2,
		BitDepth:       12,
		Scalability:    av1.EncoderScalabilityModeL2T1,
		MaxFramerate:   av1.EncoderRational{Num: 30, Den: 1},
		ColorConfigSet: true,
		ColorConfig: av1.EncoderSequenceColorConfig{
			BitDepth:             12,
			SubsamplingX:         true,
			SubsamplingY:         true,
			ChromaSamplePosition: 1,
		},
	}
	enc, err := av1.NewWebRTCEncoder(cfg, av1.EncoderWebRTCState{NextFrameID: 500})
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
	changed.ColorConfig = av1.EncoderSequenceColorConfig{BitDepth: 12}
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
	if err := enc.SetConfig(bad); !errors.Is(err, av1.ErrEncoderInvalidConfig) {
		t.Fatalf("SetConfig mismatched color bitdepth err=%v want %v", err, av1.ErrEncoderInvalidConfig)
	}
	if enc.Config() != keepConfig || enc.State() != keepState {
		t.Fatalf("bad color config mutated config/state config=%+v want=%+v state=%+v want=%+v", enc.Config(), keepConfig, enc.State(), keepState)
	}
}

func publicWebRTCEncoderFrameForConfig(t *testing.T, cfg av1.EncoderConfig) (av1.Frame, []byte) {
	t.Helper()
	seq, err := av1.EncoderSequenceHeaderForConfig(cfg)
	if err != nil {
		t.Fatalf("EncoderSequenceHeaderForConfig: %v", err)
	}
	format := av1.FrameFormat{
		Width:        int(cfg.Resolution.Width),
		Height:       int(cfg.Resolution.Height),
		BitDepth:     seq.ColorConfig.BitDepth,
		MonoChrome:   seq.ColorConfig.MonoChrome,
		SubsamplingX: seq.ColorConfig.SubsamplingX,
		SubsamplingY: seq.ColorConfig.SubsamplingY,
		Align:        64,
	}
	layout, err := av1.FrameRequiredSize(format)
	if err != nil {
		t.Fatalf("FrameRequiredSize: %v", err)
	}
	backing := make([]byte, layout.Size)
	frame, err := av1.BindFrame(backing, format)
	if err != nil {
		t.Fatalf("BindFrame: %v", err)
	}
	return frame, backing
}

func publicSetFramePlaneSample(
	plane av1.FramePlane,
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
