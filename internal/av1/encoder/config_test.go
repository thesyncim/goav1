package encoder

import (
	"errors"
	"testing"
)

func TestProfileParseAndString(t *testing.T) {
	for _, tc := range []struct {
		in      string
		profile Profile
	}{
		{in: "0", profile: Profile0},
		{in: "1", profile: Profile1},
		{in: "2", profile: Profile2},
	} {
		got, ok := ParseProfile(tc.in)
		if !ok || got != tc.profile {
			t.Fatalf("ParseProfile(%q) = %v, %v; want %v, true", tc.in, got, ok, tc.profile)
		}
		if got.String() != tc.in {
			t.Fatalf("Profile(%q).String() = %q", tc.in, got.String())
		}
	}
	if _, ok := ParseProfile("3"); ok {
		t.Fatal("ParseProfile accepted invalid profile")
	}
}

func TestScalabilityModeParseAndLayers(t *testing.T) {
	for _, tc := range []struct {
		in       string
		mode     ScalabilityMode
		spatial  uint8
		temporal uint8
		key      bool
		small    bool
		sim      bool
	}{
		{in: "L1T1", mode: ScalabilityModeL1T1, spatial: 1, temporal: 1},
		{in: "L2T2_KEY_SHIFT", mode: ScalabilityModeL2T2_KEY_SHIFT, spatial: 2, temporal: 2, key: true},
		{in: "L2T1h", mode: ScalabilityModeL2T1h, spatial: 2, temporal: 1, small: true},
		{in: "L3T3_KEY", mode: ScalabilityModeL3T3_KEY, spatial: 3, temporal: 3, key: true},
		{in: "S3T2h", mode: ScalabilityModeS3T2h, spatial: 3, temporal: 2, small: true, sim: true},
	} {
		got, ok := ParseScalabilityMode(tc.in)
		if !ok || got != tc.mode {
			t.Fatalf("ParseScalabilityMode(%q) = %v, %v; want %v, true", tc.in, got, ok, tc.mode)
		}
		spatial, temporal, key, ok := got.Layers()
		if !ok || spatial != tc.spatial || temporal != tc.temporal || key != tc.key {
			t.Fatalf("%q Layers() = %d,%d,%v,%v", tc.in, spatial, temporal, key, ok)
		}
		if got.UsesSmallResolutionStep() != tc.small || got.IsSimulcast() != tc.sim {
			t.Fatalf("%q flags small=%v sim=%v", tc.in, got.UsesSmallResolutionStep(), got.IsSimulcast())
		}
		if got.String() != tc.in {
			t.Fatalf("%v.String() = %q; want %q", got, got.String(), tc.in)
		}
	}
	if _, ok := ParseScalabilityMode("L4T4"); ok {
		t.Fatal("ParseScalabilityMode accepted invalid mode")
	}
}

func TestDefaultScalabilityMode(t *testing.T) {
	mode, ok := DefaultScalabilityMode(3, 2)
	if !ok || mode != ScalabilityModeL2T3_KEY {
		t.Fatalf("DefaultScalabilityMode(3,2) = %v,%v; want L2T3_KEY,true", mode, ok)
	}
	mode, ok = DefaultScalabilityMode(1, 1)
	if !ok || mode != ScalabilityModeL1T1 {
		t.Fatalf("DefaultScalabilityMode(1,1) = %v,%v; want L1T1,true", mode, ok)
	}
	if _, ok := DefaultScalabilityMode(4, 2); ok {
		t.Fatal("DefaultScalabilityMode accepted unsupported temporal layer count")
	}
}

func TestLimitedSpatialLayers(t *testing.T) {
	for _, tc := range []struct {
		res  Resolution
		want uint8
	}{
		{res: Resolution{Width: 1280, Height: 720}, want: 3},
		{res: Resolution{Width: 640, Height: 360}, want: 2},
		{res: Resolution{Width: 256, Height: 144}, want: 1},
		{res: Resolution{Width: 720, Height: 1280}, want: 3},
	} {
		if got := LimitedSpatialLayers(tc.res); got != tc.want {
			t.Fatalf("LimitedSpatialLayers(%+v) = %d; want %d", tc.res, got, tc.want)
		}
	}
}

func TestSetWebRTCSVCConfig(t *testing.T) {
	base := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
	}

	got, err := SetWebRTCSVCConfig(base, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L1T1: %v", err)
	}
	if got.Scalability != ScalabilityModeL1T1 || got.SpatialLayerCount != 1 || got.TemporalLayerCount != 1 {
		t.Fatalf("L1T1 config mode=%v spatial=%d temporal=%d", got.Scalability, got.SpatialLayerCount, got.TemporalLayerCount)
	}
	if !got.SpatialLayers[0].Active || got.SpatialLayers[1].Active {
		t.Fatalf("unexpected active layers: %+v", got.SpatialLayers)
	}
	if got.SpatialLayers[0].MinBitrateKbps != 100 || got.SpatialLayers[0].MaxBitrateKbps != 500 || got.SpatialLayers[0].TargetBitrateKbps != 300 {
		t.Fatalf("single-layer bitrate not copied: %+v", got.SpatialLayers[0])
	}

	base.Scalability = ScalabilityModeL2T2
	got, err = SetWebRTCSVCConfig(base, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L2T2: %v", err)
	}
	if got.SpatialLayers[0].Resolution != (Resolution{Width: 320, Height: 180}) || got.SpatialLayers[1].Resolution != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("L2T2 resolutions = %+v %+v", got.SpatialLayers[0].Resolution, got.SpatialLayers[1].Resolution)
	}
	if got.SpatialLayers[0].MinBitrateKbps != 20 || got.SpatialLayers[0].MaxBitrateKbps != 142 {
		t.Fatalf("L2T2 base bitrate = %+v", got.SpatialLayers[0])
	}
	if got.SpatialLayers[1].MinBitrateKbps != 135 || got.SpatialLayers[1].MaxBitrateKbps != 418 {
		t.Fatalf("L2T2 top bitrate = %+v", got.SpatialLayers[1])
	}
}

func TestSetWebRTCSVCConfigSmallStepAndReduction(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 1500, Height: 900},
		Scalability: ScalabilityModeL2T1h,
	}
	got, err := SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L2T1h: %v", err)
	}
	if got.SpatialLayers[0].Resolution != (Resolution{Width: 1000, Height: 600}) {
		t.Fatalf("L2T1h base resolution = %+v", got.SpatialLayers[0].Resolution)
	}

	cfg = Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL3T3,
	}
	got, err = SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L3T3 reduction: %v", err)
	}
	if got.Scalability != ScalabilityModeL2T3 {
		t.Fatalf("small input reduced to %v; want L2T3", got.Scalability)
	}

	cfg = Config{
		Resolution:  Resolution{Width: 256, Height: 144},
		Scalability: ScalabilityModeL3T3_KEY,
	}
	got, err = SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L3T3_KEY single-layer reduction: %v", err)
	}
	if got.Scalability != ScalabilityModeL1T3 {
		t.Fatalf("tiny input reduced to %v; want L1T3", got.Scalability)
	}
}

func TestSetWebRTCSVCConfigRejectsInvalidConfig(t *testing.T) {
	valid := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 300,
	}
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{name: "wide", cfg: withResolution(valid, Resolution{Width: WebRTCMaxDimension + 1, Height: 360})},
		{name: "area", cfg: withResolution(valid, Resolution{Width: 65536, Height: 65536})},
		{name: "subfps", cfg: withFramerate(valid, Rational{Num: 1, Den: 2})},
		{name: "bitrate", cfg: withBitrate(valid, 0, WebRTCMaxBitrateKbps+1, 0)},
		{name: "target-outside", cfg: withBitrate(valid, 100, 500, 600)},
		{name: "cbr-quantizer", cfg: withQuantizer(valid, RateControlCBR, 12)},
		{name: "cqp-quantizer", cfg: withQuantizer(valid, RateControlCQP, WebRTCMaxQuantizer+1)},
	} {
		if _, err := SetWebRTCSVCConfig(tc.cfg, 0, 0); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("%s err = %v; want ErrInvalidConfig", tc.name, err)
		}
	}
}

func withResolution(cfg Config, resolution Resolution) Config {
	cfg.Resolution = resolution
	return cfg
}

func withFramerate(cfg Config, framerate Rational) Config {
	cfg.MaxFramerate = framerate
	return cfg
}

func withBitrate(cfg Config, minBitrateKbps int32, maxBitrateKbps int32, targetBitrateKbps int32) Config {
	cfg.MinBitrateKbps = minBitrateKbps
	cfg.MaxBitrateKbps = maxBitrateKbps
	cfg.TargetBitrateKbps = targetBitrateKbps
	return cfg
}

func withQuantizer(cfg Config, rc RateControlMode, quantizer uint8) Config {
	cfg.RateControl = rc
	cfg.Quantizer = quantizer
	return cfg
}

func TestValidateTemporalUnitFrames(t *testing.T) {
	state := ReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = Resolution{Width: 640, Height: 360}

	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	out, err := ValidateTemporalUnitFrames(frames[:], state, RateControlCBR)
	if err != nil {
		t.Fatalf("ValidateTemporalUnitFrames valid delta: %v", err)
	}
	if !out.Valid[1] || out.Resolutions[1] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("updated buffer state = %+v", out)
	}

	tu := [...]FrameEncodeSettings{
		{
			Type:            FrameTypeKey,
			Resolution:      Resolution{Width: 320, Height: 180},
			SpatialID:       0,
			UpdateBuffer:    0,
			UpdateBufferSet: true,
			Output:          true,
		},
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	out, err = ValidateTemporalUnitFrames(tu[:], ReferenceBufferState{}, RateControlCBR)
	if err != nil {
		t.Fatalf("ValidateTemporalUnitFrames key plus upper layer: %v", err)
	}
	if !out.Valid[0] || !out.Valid[1] {
		t.Fatalf("temporal unit updates missing: %+v", out)
	}
}

func TestValidateTemporalUnitFramesRejectsInvalidRefs(t *testing.T) {
	state := ReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = Resolution{Width: 640, Height: 360}
	state.Valid[1] = true
	state.Resolutions[1] = Resolution{Width: 640, Height: 360}

	duplicate := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0, 0},
			ReferenceCount:   2,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(duplicate[:], state, RateControlCBR); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("duplicate refs err = %v; want ErrInvalidFrame", err)
	}

	badScale := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 1000, Height: 600},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(badScale[:], state, RateControlCBR); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("bad scaling err = %v; want ErrInvalidFrame", err)
	}

	reset := [...]FrameEncodeSettings{
		{
			Type:            FrameTypeKey,
			Resolution:      Resolution{Width: 320, Height: 180},
			UpdateBuffer:    0,
			UpdateBufferSet: true,
			Output:          true,
		},
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{1},
			ReferenceCount:   1,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(reset[:], state, RateControlCBR); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("stale ref after keyframe err = %v; want ErrInvalidFrame", err)
	}
}

func TestSupportedResolutionScaling(t *testing.T) {
	factor, ok := SupportedResolutionScaling(
		Resolution{Width: 320, Height: 180},
		Resolution{Width: 640, Height: 360},
	)
	if !ok || factor != (Rational{Num: 2, Den: 1}) {
		t.Fatalf("SupportedResolutionScaling = %+v,%v; want 2/1,true", factor, ok)
	}
	if _, ok := SupportedResolutionScaling(
		Resolution{Width: 640, Height: 360},
		Resolution{Width: 1000, Height: 600},
	); ok {
		t.Fatal("SupportedResolutionScaling accepted unsupported ratio")
	}
}

func TestValidateTemporalUnitFramesAllocs(t *testing.T) {
	state := ReferenceBufferState{}
	state.Valid[0] = true
	state.Resolutions[0] = Resolution{Width: 640, Height: 360}
	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	if _, err := ValidateTemporalUnitFrames(frames[:], state, RateControlCBR); err != nil {
		t.Fatalf("preflight failed: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = ValidateTemporalUnitFrames(frames[:], state, RateControlCBR)
	})
	if allocs != 0 {
		t.Fatalf("ValidateTemporalUnitFrames allocs = %f; want 0", allocs)
	}
}
