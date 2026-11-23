package goav1

import (
	"bytes"
	"errors"
	"math/bits"
	"testing"

	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
)

func TestPublicEncoderControlSurface(t *testing.T) {
	mode := internalencoder.ScalabilityModeL2T2_KEY
	if got := EncoderScalabilityModeL3T3_KEY_SHIFT.String(); got != "L3T3_KEY_SHIFT" {
		t.Fatalf("L3T3_KEY_SHIFT String() = %q", got)
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

	fullSize, err := internalencoder.WebRTCDependencyDescriptorSize(structure, control.Frames[0].GenericFrameInfo, false)
	if err != nil {
		t.Fatalf("EncoderWebRTCDependencyDescriptorSize: %v", err)
	}
	var fullBuf [16]byte
	full, err := internalencoder.AppendWebRTCDependencyDescriptor(fullBuf[:0], structure, control.Frames[0].GenericFrameInfo, true, true, false)
	if err != nil {
		t.Fatalf("AppendEncoderWebRTCDependencyDescriptor: %v", err)
	}
	controlInfo := control.Frames[0].GenericFrameInfo
	fullMandatory, _, err := ParseRTPDependencyDescriptorMandatory(full)
	if err != nil {
		t.Fatalf("ParseRTPDependencyDescriptorMandatory full: %v", err)
	}
	descriptor := RTPDependencyDescriptorMandatory{
		FirstPacketInFrame: true,
		LastPacketInFrame:  true,
		TemplateID:         fullMandatory.TemplateID,
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

	if len(full) != fullSize || full[0] != descriptorBuf[0] || full[1] != descriptorBuf[1] || full[2] != descriptorBuf[2] {
		t.Fatalf("full descriptor=% x size=%d mandatory=% x", full, fullSize, descriptorBuf)
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
	for _, mode := range svcScalabilityModes() {
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

func TestPublicEncoderWebRTCActiveDecodeTargetsMask(t *testing.T) {
	for _, mode := range svcScalabilityModes() {
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
	for _, mode := range svcScalabilityModes() {
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

	delta, err := internalencoder.WebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 800, 0)
	if err != nil {
		t.Fatalf("EncoderWebRTCDeltaFrameTemporalUnitForConfigWithDeltaPictureIndex: %v", err)
	}
	if delta.Control.Frames[1].GenericFrameInfo.TemporalID != 1 {
		t.Fatalf("delta picture 1 temporal id=%d want 1", delta.Control.Frames[1].GenericFrameInfo.TemporalID)
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

// svcScalabilityModes lists every scalability mode the WebRTC encoder accepts,
// including the *_KEY_SHIFT modes libwebrtc does not export.
func svcScalabilityModes() []EncoderScalabilityMode {
	return []EncoderScalabilityMode{
		internalencoder.ScalabilityModeL1T1, internalencoder.ScalabilityModeL1T2,
		internalencoder.ScalabilityModeL1T3, internalencoder.ScalabilityModeL2T1,
		internalencoder.ScalabilityModeL2T1h, internalencoder.ScalabilityModeL2T1_KEY,
		internalencoder.ScalabilityModeL2T2, internalencoder.ScalabilityModeL2T2h,
		internalencoder.ScalabilityModeL2T2_KEY, internalencoder.ScalabilityModeL2T2_KEY_SHIFT,
		internalencoder.ScalabilityModeL2T3, internalencoder.ScalabilityModeL2T3h,
		internalencoder.ScalabilityModeL2T3_KEY, internalencoder.ScalabilityModeL2T3_KEY_SHIFT,
		internalencoder.ScalabilityModeL3T1, internalencoder.ScalabilityModeL3T1h,
		internalencoder.ScalabilityModeL3T1_KEY, internalencoder.ScalabilityModeL3T2,
		internalencoder.ScalabilityModeL3T2h, internalencoder.ScalabilityModeL3T2_KEY,
		internalencoder.ScalabilityModeL3T2_KEY_SHIFT, internalencoder.ScalabilityModeL3T3,
		internalencoder.ScalabilityModeL3T3h, internalencoder.ScalabilityModeL3T3_KEY,
		internalencoder.ScalabilityModeL3T3_KEY_SHIFT, internalencoder.ScalabilityModeS2T1,
		internalencoder.ScalabilityModeS2T1h, internalencoder.ScalabilityModeS2T2,
		internalencoder.ScalabilityModeS2T2h, internalencoder.ScalabilityModeS2T3,
		internalencoder.ScalabilityModeS2T3h, internalencoder.ScalabilityModeS3T1,
		internalencoder.ScalabilityModeS3T1h, internalencoder.ScalabilityModeS3T2,
		internalencoder.ScalabilityModeS3T2h, internalencoder.ScalabilityModeS3T3,
		internalencoder.ScalabilityModeS3T3h,
	}
}
