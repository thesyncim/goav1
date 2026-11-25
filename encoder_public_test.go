package goav1

import (
	"bytes"
	"errors"
	"fmt"
	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	"math/bits"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
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
		Profile:               internalencoder.Profile1, // sRGB is 4:4:4
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

func publicOverflowStride(width int) (height, stride int, ok bool) {
	maxInt := int(^uint(0) >> 1)
	for height = 8; height <= 1<<20; height += 8 {
		rows := height - 1
		stride = (maxInt - (width - 1)) / rows
		if stride > (maxInt-width)/rows {
			return height, stride, true
		}
	}
	return 0, 0, false
}

func TestPublicEncodeRejectsStrideWhosePlaneLengthOverflows(t *testing.T) {
	const width = 8
	height, stride, ok := publicOverflowStride(width)
	if !ok {
		t.Fatal("could not find a portable overflowing stride case")
	}

	t.Run("8-bit monochrome", func(t *testing.T) {
		_, _, err := EncodeI400Keyframe(I400Frame{
			YStride: stride,
			Width:   width,
			Height:  height,
		}, 0)
		if err == nil {
			t.Fatal("EncodeI400Keyframe accepted a stride whose required plane length overflows int")
		}
	})

	t.Run("high-bit-depth monochrome", func(t *testing.T) {
		_, _, err := EncodeI400HighBitDepthLosslessKeyframe(I400HighBitDepthFrame{
			YStride:  stride,
			Width:    width,
			Height:   height,
			BitDepth: 10,
		})
		if err == nil {
			t.Fatal("EncodeI400HighBitDepthLosslessKeyframe accepted a stride whose required plane length overflows int")
		}
	})

	t.Run("high-bit-depth 4:2:0 luma", func(t *testing.T) {
		_, _, err := EncodeI420HighBitDepthLosslessKeyframe(I420HighBitDepthFrame{
			YStride:      stride,
			ChromaStride: width / 2,
			Width:        width,
			Height:       height,
			BitDepth:     10,
		})
		if err == nil {
			t.Fatal("EncodeI420HighBitDepthLosslessKeyframe accepted a stride whose required luma length overflows int")
		}
	})
}

func TestPublicEncodeAcceptsMinimalAndPaddedStrides(t *testing.T) {
	const width, height = 8, 8
	cases := []struct {
		name         string
		yStride      int
		chromaStride int
	}{
		{name: "minimal", yStride: width, chromaStride: width / 2},
		{name: "padded", yStride: width + 2, chromaStride: width/2 + 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ySize := (height-1)*tc.yStride + width
			chromaWidth, chromaHeight := width/2, height/2
			chromaSize := (chromaHeight-1)*tc.chromaStride + chromaWidth

			mono8 := I400Frame{
				Y:       make([]byte, ySize),
				YStride: tc.yStride,
				Width:   width,
				Height:  height,
			}
			if tu, _, err := EncodeI400Keyframe(mono8, 0); err != nil || len(tu) == 0 {
				t.Fatalf("EncodeI400Keyframe: TU length %d, error %v", len(tu), err)
			}

			mono16 := I400HighBitDepthFrame{
				Y:        make([]uint16, ySize),
				YStride:  tc.yStride,
				Width:    width,
				Height:   height,
				BitDepth: 10,
			}
			if tu, _, err := EncodeI400HighBitDepthLosslessKeyframe(mono16); err != nil || len(tu) == 0 {
				t.Fatalf("EncodeI400HighBitDepthLosslessKeyframe: TU length %d, error %v", len(tu), err)
			}

			color16 := I420HighBitDepthFrame{
				Y:            make([]uint16, ySize),
				U:            make([]uint16, chromaSize),
				V:            make([]uint16, chromaSize),
				YStride:      tc.yStride,
				ChromaStride: tc.chromaStride,
				Width:        width,
				Height:       height,
				BitDepth:     10,
			}
			if tu, _, err := EncodeI420HighBitDepthLosslessKeyframe(color16); err != nil || len(tu) == 0 {
				t.Fatalf("EncodeI420HighBitDepthLosslessKeyframe: TU length %d, error %v", len(tu), err)
			}
		})
	}
}

// TestVideoEncoderThreadsSteadyStateZeroAlloc pins the zero-allocation design
// on the threaded encoder: wavefront and tile workers, plus forced keyframes
// in the steady state. Parking a goroutine draws a sudog from the runtime's
// per-P cache, and that cache fills over the first few hundred frames. The
// test therefore pins GOMAXPROCS and requires an allocation-free window once
// warm. A per-frame allocation in goav1 would never produce one.
func TestVideoEncoderThreadsSteadyStateZeroAlloc(t *testing.T) {
	if testing.Short() {
		t.Skip("encodes up to a few thousand frames")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	const w, h, window, maxWindows = 640, 360, 40, 40
	for _, tc := range []struct {
		threads, tiles, keyEvery int
	}{
		{threads: 4},
		{threads: 4, keyEvery: 13},
		{threads: 4, tiles: 2, keyEvery: 13},
	} {
		t.Run(fmt.Sprintf("threads=%d/tiles=%d/key=%d", tc.threads, tc.tiles, tc.keyEvery), func(t *testing.T) {
			enc, err := NewVideoEncoder(VideoEncoderConfig{
				Width: w, Height: h, TargetBitrate: 1_000_000, Framerate: 30, MaxThreads: tc.threads,
			})
			if err != nil {
				t.Fatalf("NewVideoEncoder: %v", err)
			}
			defer enc.Close()
			if tc.tiles > 0 {
				enc.SetTileColumns(tc.tiles)
			}
			frame := I420Frame{
				Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
				YStride: w, ChromaStride: w / 2, Width: w, Height: h,
			}
			n := 0
			encodeWindow := func() {
				for range window {
					fillMovingI420(frame, n)
					forceKey := n == 0 || (tc.keyEvery > 0 && n%tc.keyEvery == 0)
					if _, err := enc.Encode(frame, forceKey); err != nil {
						t.Fatalf("Encode frame %d: %v", n, err)
					}
					n++
				}
			}
			encodeWindow()
			var before, after runtime.MemStats
			var allocs uint64
			for range maxWindows {
				runtime.ReadMemStats(&before)
				encodeWindow()
				runtime.ReadMemStats(&after)
				if allocs = after.Mallocs - before.Mallocs; allocs == 0 {
					return
				}
			}
			t.Fatalf("no allocation-free window of %d frames after %d frames; last window allocated %d", window, n, allocs)
		})
	}
}

// fillMovingI420 draws a moving XOR texture with a bright disc so that inter
// frames carry real motion and residual.
func fillMovingI420(f I420Frame, t int) {
	for y := 0; y < f.Height; y++ {
		for x := 0; x < f.Width; x++ {
			v := (x + 2*t) ^ (y + t)
			dx, dy := x-f.Width/2-3*t%200, y-f.Height/2
			if dx*dx+dy*dy < 60*60 {
				v = 230
			}
			f.Y[y*f.YStride+x] = byte(v)
		}
	}
	for y := 0; y < f.Height/2; y++ {
		for x := 0; x < f.Width/2; x++ {
			f.U[y*f.ChromaStride+x] = byte(128 + (x+t)%32)
			f.V[y*f.ChromaStride+x] = byte(128 - (y+t)%32)
		}
	}
}

// TestEncoderAllocCensus reports heap allocations per encoder lifecycle
// event: construction, the first keyframe, the first inter frames, steady
// state, forced keyframes, scene cuts, threading changes and RTC
// reconfiguration. It is a measurement tool (GOAV1_ALLOC_CENSUS=1); input
// is the raw 1280x720 I420 file GOAV1_CENSUS_YUV, else a synthetic scene.
func TestEncoderAllocCensus(t *testing.T) {
	if os.Getenv("GOAV1_ALLOC_CENSUS") == "" {
		t.Skip("set GOAV1_ALLOC_CENSUS=1")
	}
	const w, h = 1280, 720
	frames := censusFrames(t, w, h, 120)
	scene := censusScene(w, h)

	var ms runtime.MemStats
	// GOAV1_CENSUS_MEMPROFILE=<prefix> (with -memprofilerate=1) writes the
	// cumulative allocs profile after every phase as <prefix>.NN; diff
	// consecutive snapshots (pprof -diff_base) to isolate one phase.
	profOut := os.Getenv("GOAV1_CENSUS_MEMPROFILE")
	phase := 0
	measure := func(name string, n int, f func()) {
		runtime.GC()
		defer func() {
			if profOut == "" {
				return
			}
			// The allocs profile publishes a cycle two GCs after it ends.
			runtime.GC()
			runtime.GC()
			runtime.GC()
			phase++
			if pf, err := os.Create(fmt.Sprintf("%s.%02d", profOut, phase)); err == nil {
				_ = pprof.Lookup("allocs").WriteTo(pf, 0)
				_ = pf.Close()
			}
			fmt.Printf("census phase %02d = %s\n", phase, name)
		}()
		runtime.ReadMemStats(&ms)
		m0, b0 := ms.Mallocs, ms.TotalAlloc
		f()
		runtime.ReadMemStats(&ms)
		dm, db := ms.Mallocs-m0, ms.TotalAlloc-b0
		fmt.Printf("census %-34s allocs=%8d bytes=%11d", name, dm, db)
		if n > 1 {
			fmt.Printf("  per-frame allocs=%.1f bytes=%.0f", float64(dm)/float64(n), float64(db)/float64(n))
		}
		fmt.Println()
	}

	for _, threads := range []int{1, 4} {
		p := fmt.Sprintf("video t%d ", threads)
		var enc *VideoEncoder
		measure(p+"NewVideoEncoder", 1, func() {
			var err error
			enc, err = NewVideoEncoder(VideoEncoderConfig{Width: w, Height: h, TargetBitrate: 1_000_000, Framerate: 30, MaxThreads: threads})
			if err != nil {
				t.Fatal(err)
			}
		})
		i := 0
		enc1 := func(force bool, f I420Frame) {
			if _, err := enc.Encode(f, force); err != nil {
				t.Fatal(err)
			}
			i++
		}
		measure(p+"first keyframe", 1, func() { enc1(false, frames[0]) })
		measure(p+"inter frames 1-15", 15, func() {
			for range 15 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"inter frames 16-63", 48, func() {
			for range 48 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"steady 64-111", 48, func() {
			for range 48 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"forced keyframe", 1, func() { enc1(true, frames[i%len(frames)]) })
		measure(p+"after forced key x16", 16, func() {
			for range 16 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"scene cut + 8", 9, func() {
			enc1(false, scene)
			for range 8 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"forced key on noise", 1, func() { enc1(true, scene) })
		measure(p+"second forced key on noise", 1, func() { enc1(true, scene) })
		measure(p+"second scene cut + 8", 9, func() {
			enc1(false, scene)
			for range 8 {
				enc1(false, frames[i%len(frames)])
			}
		})
		if threads == 1 {
			measure(p+"SetMaxThreads(4) + 16", 16, func() {
				enc.SetMaxThreads(4)
				for range 16 {
					enc1(false, frames[i%len(frames)])
				}
			})
		}
		measure(p+"Close", 1, func() { _ = enc.Close() })
	}

	var rtc *RTCEncoder
	measure("rtc NewRTCEncoder", 1, func() {
		var err error
		rtc, err = NewRTCEncoder(VideoEncoderConfig{Width: w, Height: h, TargetBitrate: 1_000_000, Framerate: 30, MaxThreads: 1})
		if err != nil {
			t.Fatal(err)
		}
	})
	j := 0
	rtc1 := func(force bool) {
		if _, err := rtc.Encode(frames[j%len(frames)], force); err != nil {
			t.Fatal(err)
		}
		j++
	}
	measure("rtc first 64 frames", 64, func() {
		for range 64 {
			rtc1(false)
		}
	})
	measure("rtc steady 48", 48, func() {
		for range 48 {
			rtc1(false)
		}
	})
	cfg := rtc.Config()
	measure("rtc bitrate change + 16", 16, func() {
		c := cfg
		c.TargetBitrateKbps /= 2
		c.MinBitrateKbps /= 2
		c.MaxBitrateKbps /= 2
		for k := range c.SpatialLayers {
			c.SpatialLayers[k].TargetBitrateKbps /= 2
			c.SpatialLayers[k].MinBitrateKbps /= 2
			c.SpatialLayers[k].MaxBitrateKbps /= 2
		}
		if err := rtc.SetConfig(c); err != nil {
			t.Fatalf("%v: %+v", err, c)
		}
		for range 16 {
			rtc1(false)
		}
	})
	measure("rtc forced key + 16", 17, func() {
		rtc1(true)
		for range 16 {
			rtc1(false)
		}
	})
	measure("rtc layers change (L1T2) + 16", 16, func() {
		c := rtc.Config()
		c.Scalability = EncoderScalabilityModeL1T2
		c.TemporalLayerCount = 2
		c.SpatialLayers[0].TemporalLayers = 2
		if err := rtc.SetConfig(c); err != nil {
			t.Logf("layers change: %v", err)
			return
		}
		for range 16 {
			rtc1(false)
		}
	})
	measure("rtc Close", 1, func() { _ = rtc.Close() })
}

func censusFrames(t *testing.T, w, h, n int) []I420Frame {
	size := w * h * 3 / 2
	var raw []byte
	if p := os.Getenv("GOAV1_CENSUS_YUV"); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		raw = b
	}
	out := make([]I420Frame, n)
	for i := range out {
		var buf []byte
		if len(raw) >= (i+1)*size {
			buf = raw[i*size : (i+1)*size]
		} else {
			buf = make([]byte, size)
			for y := range h {
				for x := range w {
					buf[y*w+x] = byte((x + 3*i + (y/8)*5) ^ (y >> 2))
				}
			}
			for k := w * h; k < size; k++ {
				buf[k] = 128
			}
		}
		out[i] = I420Frame{Y: buf[:w*h], U: buf[w*h : w*h+w*h/4], V: buf[w*h+w*h/4:], YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	}
	return out
}

func censusScene(w, h int) I420Frame {
	size := w * h * 3 / 2
	buf := make([]byte, size)
	for i := range buf[:w*h] {
		buf[i] = byte(255 - (i*7)%251)
	}
	for k := w * h; k < size; k++ {
		buf[k] = 60
	}
	return I420Frame{Y: buf[:w*h], U: buf[w*h : w*h+w*h/4], V: buf[w*h+w*h/4:], YStride: w, ChromaStride: w / 2, Width: w, Height: h}
}

func TestEncoderColorAndMetadataRoundTrip(t *testing.T) {
	const w, h = 128, 128
	wantColor := ColorInfo{
		Primaries:               9,  // BT.2020
		TransferCharacteristics: 16, // SMPTE ST 2084 (PQ)
		MatrixCoefficients:      9,  // BT.2020 NCL
		ChromaSamplePosition:    2,  // Colocated
		ColorRange:              true,
	}
	wantMeta := FrameMetadata{
		HasHDRCLL:  true,
		HDRCLL:     MetadataHDRCLL{MaxCLL: 1200, MaxFALL: 450},
		HasHDRMDCV: true,
		HDRMDCV: MetadataHDRMDCV{
			PrimaryChromaticityX:    [3]uint16{35400, 8500, 6550},
			PrimaryChromaticityY:    [3]uint16{14600, 39850, 2300},
			WhitePointChromaticityX: 15635,
			WhitePointChromaticityY: 16450,
			LuminanceMax:            10000000,
			LuminanceMin:            50,
		},
	}

	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width:            w,
		Height:           h,
		QIndex:           90,
		KeyFrameInterval: 3,
		Color:            &wantColor,
		Metadata:         &wantMeta,
	})
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	defer enc.Close()

	frames := censusFrames(t, w, h, 5)
	var tus [][]byte
	for i, f := range frames {
		if i == 1 {
			enc.SetMetadata(&FrameMetadata{
				HasTimecode: true,
				Timecode: MetadataTimecode{
					CountingType:  1,
					FullTimestamp: true,
					NFrames:       24,
					Seconds:       12,
					Minutes:       34,
					Hours:         1,
				},
				ITUTT35: []MetadataITUTT35{{
					CountryCode: 0xb5,
					Payload:     []byte("HDR10PLUS"),
				}},
			})
		}
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("Encode(%d): %v", i, err)
		}
		wantKey := i == 0 || i == 3
		if out.Keyframe != wantKey {
			t.Fatalf("frame %d: Keyframe=%v, want %v", i, out.Keyframe, wantKey)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}

	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	defer dec.Close()

	var decoded []DecodedFrame
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext: %v", err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			decoded = append(decoded, copyDecodedFrame(f))
		}
	}
	if len(decoded) != 5 {
		t.Fatalf("decoded %d frames, want 5", len(decoded))
	}
	for i, f := range decoded {
		if f.Color.Primaries != wantColor.Primaries ||
			f.Color.TransferCharacteristics != wantColor.TransferCharacteristics ||
			f.Color.MatrixCoefficients != wantColor.MatrixCoefficients ||
			f.Color.ChromaSamplePosition != wantColor.ChromaSamplePosition ||
			f.Color.ColorRange != wantColor.ColorRange {
			t.Fatalf("frame %d color = %+v, want %+v", i, f.Color, wantColor)
		}
		if !f.Metadata.HasHDRCLL || f.Metadata.HDRCLL != wantMeta.HDRCLL {
			t.Fatalf("frame %d HDRCLL = %+v (has=%v), want %+v", i, f.Metadata.HDRCLL, f.Metadata.HasHDRCLL, wantMeta.HDRCLL)
		}
		if !f.Metadata.HasHDRMDCV || f.Metadata.HDRMDCV != wantMeta.HDRMDCV {
			t.Fatalf("frame %d HDRMDCV = %+v (has=%v), want %+v", i, f.Metadata.HDRMDCV, f.Metadata.HasHDRMDCV, wantMeta.HDRMDCV)
		}
	}
	// Frame 1 carries TU-scoped timecode and ITU-T T.35.
	if !decoded[1].Metadata.HasTimecode || decoded[1].Metadata.Timecode.Hours != 1 || decoded[1].Metadata.Timecode.Minutes != 34 || decoded[1].Metadata.Timecode.Seconds != 12 || decoded[1].Metadata.Timecode.NFrames != 24 {
		t.Fatalf("frame 1 timecode = %+v (has=%v)", decoded[1].Metadata.Timecode, decoded[1].Metadata.HasTimecode)
	}
	if len(decoded[1].Metadata.ITUTT35) != 1 || string(decoded[1].Metadata.ITUTT35[0].Payload) != "HDR10PLUS" {
		t.Fatalf("frame 1 ITUTT35 = %+v", decoded[1].Metadata.ITUTT35)
	}
	// Frame 2 must not inherit frame 1's TU-scoped metadata.
	if decoded[2].Metadata.HasTimecode || len(decoded[2].Metadata.ITUTT35) != 0 {
		t.Fatalf("frame 2 leaked TU metadata: %+v", decoded[2].Metadata)
	}
}

func TestEncoderI420LosslessKeyframeAndStream(t *testing.T) {
	const w, h = 64, 64
	src := censusFrames(t, w, h, 2)

	tu, recon, err := EncodeI420LosslessKeyframe(src[0])
	if err != nil {
		t.Fatalf("EncodeI420LosslessKeyframe: %v", err)
	}
	if !bytes.Equal(recon.Y, src[0].Y) || !bytes.Equal(recon.U, src[0].U) || !bytes.Equal(recon.V, src[0].V) {
		t.Fatal("lossless recon does not match source")
	}

	enc, err := NewVideoEncoder(VideoEncoderConfig{Width: w, Height: h, Lossless: true})
	if err != nil {
		t.Fatalf("NewVideoEncoder(Lossless): %v", err)
	}
	defer enc.Close()

	var tus [][]byte
	tus = append(tus, tu)
	for _, f := range src {
		out, err := enc.Encode(f, false)
		if err != nil {
			t.Fatalf("lossless Encode: %v", err)
		}
		tus = append(tus, append([]byte(nil), out.Data...))
	}

	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatalf("NewDecoder(lossless): %v", err)
	}
	defer dec.Close()

	want := []I420Frame{src[0], src[0], src[1]}
	for i, wFrame := range want {
		batch, ok, err := dec.DecodeNext()
		if err != nil || !ok || len(batch) != 1 {
			t.Fatalf("frame %d DecodeNext: len=%d ok=%v err=%v", i, len(batch), ok, err)
		}
		got := copyDecodedFrame(batch[0])
		if !bytes.Equal(got.Y, wFrame.Y) || !bytes.Equal(got.U, wFrame.U) || !bytes.Equal(got.V, wFrame.V) {
			t.Fatalf("lossless frame %d pixels differ from source", i)
		}
	}
}

func TestEncoderHighBitDepthAndMonochromeStreams(t *testing.T) {
	const w, h = 64, 64

	// 10-bit 4:2:0 HDR10 streaming encoder.
	hdrEnc, err := NewHighBitDepthVideoEncoder(HighBitDepthVideoEncoderConfig{
		Width:            w,
		Height:           h,
		BitDepth:         10,
		Subsampling:      420,
		QIndex:           90,
		KeyFrameInterval: 2,
		Color:            &ColorInfo{Primaries: 9, TransferCharacteristics: 16, MatrixCoefficients: 9},
		Metadata:         &FrameMetadata{HasHDRCLL: true, HDRCLL: MetadataHDRCLL{MaxCLL: 1000, MaxFALL: 300}},
	})
	if err != nil {
		t.Fatalf("NewHighBitDepthVideoEncoder(10-bit 420): %v", err)
	}
	var hdrTUs [][]byte
	for i := range 3 {
		f := I420HighBitDepthFrame{
			Y: make([]uint16, w*h), U: make([]uint16, (w/2)*(h/2)), V: make([]uint16, (w/2)*(h/2)),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h, BitDepth: 10,
		}
		for j := range f.Y {
			f.Y[j] = uint16((j*7 + i*31) % 1024)
		}
		for j := range f.U {
			f.U[j], f.V[j] = 512, 512
		}
		out, err := hdrEnc.Encode(f, false)
		if err != nil {
			t.Fatalf("10-bit Encode(%d): %v", i, err)
		}
		hdrTUs = append(hdrTUs, append([]byte(nil), out.Data...))
	}
	dec, err := NewDecoder(hdrTUs)
	if err != nil {
		t.Fatalf("NewDecoder(10-bit HDR): %v", err)
	}
	batch, ok, err := dec.DecodeNext()
	if err != nil || !ok || len(batch) != 1 {
		t.Fatalf("10-bit DecodeNext: len=%d ok=%v err=%v", len(batch), ok, err)
	}
	got := copyDecodedFrame(batch[0])
	dec.Close()
	if got.BytesPerSample != 2 || got.Color.Primaries != 9 || !got.Metadata.HasHDRCLL || got.Metadata.HDRCLL.MaxCLL != 1000 {
		t.Fatalf("10-bit HDR decode mismatch: bps=%d color=%+v meta=%+v", got.BytesPerSample, got.Color, got.Metadata)
	}

	// Native 8-bit monochrome streaming encoder.
	monoEnc, err := NewMonochromeVideoEncoder(VideoEncoderConfig{Width: w, Height: h, QIndex: 90})
	if err != nil {
		t.Fatalf("NewMonochromeVideoEncoder: %v", err)
	}
	var monoTUs [][]byte
	for i := range 2 {
		f := I400Frame{Y: make([]byte, w*h), YStride: w, Width: w, Height: h}
		for j := range f.Y {
			f.Y[j] = byte((j*5 + i*19) % 256)
		}
		out, err := monoEnc.Encode(f, false)
		if err != nil {
			t.Fatalf("mono Encode(%d): %v", i, err)
		}
		monoTUs = append(monoTUs, append([]byte(nil), out.Data...))
	}
	monoDec, err := NewDecoder(monoTUs)
	if err != nil {
		t.Fatalf("NewDecoder(mono): %v", err)
	}
	mb, ok, err := monoDec.DecodeNext()
	if err != nil || !ok || len(mb) != 1 {
		t.Fatalf("mono DecodeNext: len=%d ok=%v err=%v", len(mb), ok, err)
	}
	mg := copyDecodedFrame(mb[0])
	monoDec.Close()
	if mg.ChromaWidth != 0 || mg.ChromaHeight != 0 || mg.U != nil || mg.V != nil {
		t.Fatalf("expected monochrome frame, got chroma %dx%d", mg.ChromaWidth, mg.ChromaHeight)
	}

	// Native 10-bit monochrome streaming encoder.
	mono10Enc, err := NewHighBitDepthMonochromeVideoEncoder(HighBitDepthVideoEncoderConfig{
		Width: w, Height: h, BitDepth: 10, QIndex: 90,
	})
	if err != nil {
		t.Fatalf("NewHighBitDepthMonochromeVideoEncoder: %v", err)
	}
	var mono10TUs [][]byte
	for i := range 2 {
		f := I400HighBitDepthFrame{Y: make([]uint16, w*h), YStride: w, Width: w, Height: h, BitDepth: 10}
		for j := range f.Y {
			f.Y[j] = uint16((j*9 + i*29) % 1024)
		}
		out, err := mono10Enc.Encode(f, false)
		if err != nil {
			t.Fatalf("mono10 Encode(%d): %v", i, err)
		}
		mono10TUs = append(mono10TUs, append([]byte(nil), out.Data...))
	}
	mono10Dec, err := NewDecoder(mono10TUs)
	if err != nil {
		t.Fatalf("NewDecoder(mono10): %v", err)
	}
	m10b, ok, err := mono10Dec.DecodeNext()
	if err != nil || !ok || len(m10b) != 1 {
		t.Fatalf("mono10 DecodeNext: len=%d ok=%v err=%v", len(m10b), ok, err)
	}
	m10g := copyDecodedFrame(m10b[0])
	mono10Dec.Close()
	if m10g.BytesPerSample != 2 || m10g.ChromaWidth != 0 {
		t.Fatalf("expected 10-bit monochrome, got bps=%d chromaW=%d", m10g.BytesPerSample, m10g.ChromaWidth)
	}
}

func TestEncoderNative422And444AndRuntimeRateControl(t *testing.T) {
	const w, h = 64, 64

	// Native 8-bit 4:2:2 (Profile 2)
	enc422, err := NewVideoEncoder(VideoEncoderConfig{Width: w, Height: h, QIndex: 90, Subsampling: 422})
	if err != nil {
		t.Fatalf("NewVideoEncoder(422): %v", err)
	}
	defer enc422.Close()
	f422 := I422Frame{
		Y: make([]byte, w*h), U: make([]byte, (w/2)*h), V: make([]byte, (w/2)*h),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h,
	}
	for i := range f422.Y {
		f422.Y[i] = byte((i * 7) % 256)
	}
	for i := range f422.U {
		f422.U[i], f422.V[i] = 110, 145
	}
	out422, err := enc422.EncodeI422(f422, true)
	if err != nil {
		t.Fatalf("EncodeI422: %v", err)
	}
	dec422, err := NewDecoder([][]byte{append([]byte(nil), out422.Data...)})
	if err != nil {
		t.Fatalf("NewDecoder(422): %v", err)
	}
	b422, ok, err := dec422.DecodeNext()
	dec422.Close()
	if err != nil || !ok || len(b422) != 1 || b422[0].U.Width != w/2 || b422[0].U.Height != h {
		t.Fatalf("422 decode mismatch: ok=%v err=%v", ok, err)
	}

	// Native 8-bit 4:4:4 (Profile 1) + runtime SetQIndex / SetBitrate
	enc444, err := NewVideoEncoder(VideoEncoderConfig{Width: w, Height: h, QIndex: 90, Subsampling: 444})
	if err != nil {
		t.Fatalf("NewVideoEncoder(444): %v", err)
	}
	defer enc444.Close()
	f444 := I444Frame{
		Y: make([]byte, w*h), U: make([]byte, w*h), V: make([]byte, w*h),
		YStride: w, UStride: w, VStride: w, Width: w, Height: h,
	}
	for i := range f444.Y {
		f444.Y[i], f444.U[i], f444.V[i] = byte((i*5)%256), 120, 136
	}
	var tus444 [][]byte
	out0, err := enc444.EncodeI444(f444, true)
	if err != nil {
		t.Fatalf("EncodeI444(0): %v", err)
	}
	tus444 = append(tus444, append([]byte(nil), out0.Data...))
	if err := enc444.SetQIndex(120); err != nil {
		t.Fatalf("SetQIndex: %v", err)
	}
	out1, err := enc444.EncodeI444(f444, false)
	if err != nil {
		t.Fatalf("EncodeI444(1): %v", err)
	}
	tus444 = append(tus444, append([]byte(nil), out1.Data...))
	if err := enc444.SetBitrate(500_000, 30); err != nil {
		t.Fatalf("SetBitrate: %v", err)
	}
	out2, err := enc444.EncodeI444(f444, false)
	if err != nil {
		t.Fatalf("EncodeI444(2): %v", err)
	}
	tus444 = append(tus444, append([]byte(nil), out2.Data...))

	dec444, err := NewDecoder(tus444)
	if err != nil {
		t.Fatalf("NewDecoder(444): %v", err)
	}
	defer dec444.Close()
	b444, ok, err := dec444.DecodeNext()
	if err != nil || !ok || len(b444) != 1 || b444[0].U.Width != w || b444[0].U.Height != h {
		t.Fatalf("444 decode mismatch: ok=%v err=%v", ok, err)
	}
}
