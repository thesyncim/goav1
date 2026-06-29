package encoder

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/obu"
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
		{in: "L2T3_KEY_SHIFT", mode: ScalabilityModeL2T3_KEY_SHIFT, spatial: 2, temporal: 3, key: true},
		{in: "L2T1h", mode: ScalabilityModeL2T1h, spatial: 2, temporal: 1, small: true},
		{in: "L3T2_KEY_SHIFT", mode: ScalabilityModeL3T2_KEY_SHIFT, spatial: 3, temporal: 2, key: true},
		{in: "L3T3_KEY", mode: ScalabilityModeL3T3_KEY, spatial: 3, temporal: 3, key: true},
		{in: "L3T3_KEY_SHIFT", mode: ScalabilityModeL3T3_KEY_SHIFT, spatial: 3, temporal: 3, key: true},
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
		wantShared := tc.spatial > 1 && !tc.sim
		if got.UsesSmallResolutionStep() != tc.small ||
			got.IsSimulcast() != tc.sim ||
			got.UsesSharedReferenceSlots() != wantShared {
			t.Fatalf("%q flags small=%v sim=%v shared=%v", tc.in, got.UsesSmallResolutionStep(), got.IsSimulcast(), got.UsesSharedReferenceSlots())
		}
		if got.UsesKeyFrameInterLayerDependency() != tc.key {
			t.Fatalf("%q key flag=%v", tc.in, got.UsesKeyFrameInterLayerDependency())
		}
		if got.String() != tc.in {
			t.Fatalf("%v.String() = %q; want %q", got, got.String(), tc.in)
		}
	}
	if _, ok := ParseScalabilityMode("L4T4"); ok {
		t.Fatal("ParseScalabilityMode accepted invalid mode")
	}
}

func TestScalabilityModePinnedLibWebRTCSVCModes(t *testing.T) {
	for _, tc := range []struct {
		name      string
		spatial   uint8
		temporal  uint8
		key       bool
		keyShift  bool
		smallStep bool
		simulcast bool
	}{
		{name: "L1T1", spatial: 1, temporal: 1},
		{name: "L1T2", spatial: 1, temporal: 2},
		{name: "L1T3", spatial: 1, temporal: 3},
		{name: "L2T1", spatial: 2, temporal: 1},
		{name: "L2T1h", spatial: 2, temporal: 1, smallStep: true},
		{name: "L2T1_KEY", spatial: 2, temporal: 1, key: true},
		{name: "L2T2", spatial: 2, temporal: 2},
		{name: "L2T2h", spatial: 2, temporal: 2, smallStep: true},
		{name: "L2T2_KEY", spatial: 2, temporal: 2, key: true},
		{name: "L2T2_KEY_SHIFT", spatial: 2, temporal: 2, key: true, keyShift: true},
		{name: "L2T3", spatial: 2, temporal: 3},
		{name: "L2T3h", spatial: 2, temporal: 3, smallStep: true},
		{name: "L2T3_KEY", spatial: 2, temporal: 3, key: true},
		{name: "L3T1", spatial: 3, temporal: 1},
		{name: "L3T1h", spatial: 3, temporal: 1, smallStep: true},
		{name: "L3T1_KEY", spatial: 3, temporal: 1, key: true},
		{name: "L3T2", spatial: 3, temporal: 2},
		{name: "L3T2h", spatial: 3, temporal: 2, smallStep: true},
		{name: "L3T2_KEY", spatial: 3, temporal: 2, key: true},
		{name: "L3T3", spatial: 3, temporal: 3},
		{name: "L3T3h", spatial: 3, temporal: 3, smallStep: true},
		{name: "L3T3_KEY", spatial: 3, temporal: 3, key: true},
		{name: "S2T1", spatial: 2, temporal: 1, simulcast: true},
		{name: "S2T1h", spatial: 2, temporal: 1, smallStep: true, simulcast: true},
		{name: "S2T2", spatial: 2, temporal: 2, simulcast: true},
		{name: "S2T2h", spatial: 2, temporal: 2, smallStep: true, simulcast: true},
		{name: "S2T3", spatial: 2, temporal: 3, simulcast: true},
		{name: "S2T3h", spatial: 2, temporal: 3, smallStep: true, simulcast: true},
		{name: "S3T1", spatial: 3, temporal: 1, simulcast: true},
		{name: "S3T1h", spatial: 3, temporal: 1, smallStep: true, simulcast: true},
		{name: "S3T2", spatial: 3, temporal: 2, simulcast: true},
		{name: "S3T2h", spatial: 3, temporal: 2, smallStep: true, simulcast: true},
		{name: "S3T3", spatial: 3, temporal: 3, simulcast: true},
		{name: "S3T3h", spatial: 3, temporal: 3, smallStep: true, simulcast: true},
	} {
		mode, ok := ParseScalabilityMode(tc.name)
		if !ok {
			t.Fatalf("ParseScalabilityMode(%q) failed", tc.name)
		}
		if got := mode.String(); got != tc.name {
			t.Fatalf("%q String()=%q", tc.name, got)
		}
		spatial, temporal, key, ok := mode.Layers()
		if !ok || spatial != tc.spatial || temporal != tc.temporal || key != tc.key {
			t.Fatalf("%q Layers()=%d,%d,%v,%v", tc.name, spatial, temporal, key, ok)
		}
		if mode.UsesSmallResolutionStep() != tc.smallStep ||
			mode.IsSimulcast() != tc.simulcast ||
			mode.UsesKeyFrameInterLayerDependency() != tc.key ||
			mode.UsesKeyFrameInterLayerDependencyShift() != tc.keyShift {
			t.Fatalf("%q flags small=%v sim=%v key=%v shift=%v", tc.name,
				mode.UsesSmallResolutionStep(),
				mode.IsSimulcast(),
				mode.UsesKeyFrameInterLayerDependency(),
				mode.UsesKeyFrameInterLayerDependencyShift())
		}
		cfg, err := SetWebRTCSVCConfig(Config{
			Resolution:  Resolution{Width: 1280, Height: 720},
			Scalability: mode,
		}, 0, 0)
		if err != nil {
			t.Fatalf("SetWebRTCSVCConfig(%q): %v", tc.name, err)
		}
		if cfg.SpatialLayerCount != tc.spatial || cfg.TemporalLayerCount != tc.temporal {
			t.Fatalf("%q normalized layers=%d,%d", tc.name, cfg.SpatialLayerCount, cfg.TemporalLayerCount)
		}
		structure, err := WebRTCFrameDependencyStructureForConfig(cfg)
		if err != nil {
			t.Fatalf("WebRTCFrameDependencyStructureForConfig(%q): %v", tc.name, err)
		}
		wantTemplates := webRTCTestReferenceTemplateNum(mode, tc.spatial, tc.temporal)
		if structure.NumDecodeTargets != tc.spatial*tc.temporal ||
			structure.NumChains != tc.spatial ||
			structure.TemplateNum != wantTemplates ||
			structure.ResolutionNum != tc.spatial {
			t.Fatalf("%q structure shape=%+v", tc.name, structure)
		}
	}
}

func webRTCTestReferenceTemplateNum(mode ScalabilityMode, spatial uint8, temporal uint8) uint8 {
	switch mode {
	case ScalabilityModeL1T2:
		return 3
	case ScalabilityModeL1T3:
		return 5
	case ScalabilityModeL2T1, ScalabilityModeL2T1h, ScalabilityModeL2T1_KEY,
		ScalabilityModeS2T1, ScalabilityModeS2T1h:
		return 4
	case ScalabilityModeL2T2, ScalabilityModeL2T2h, ScalabilityModeL2T2_KEY,
		ScalabilityModeS2T2, ScalabilityModeS2T2h:
		return 6
	case ScalabilityModeL2T2_KEY_SHIFT:
		return 7
	case ScalabilityModeL2T3, ScalabilityModeL2T3h, ScalabilityModeL2T3_KEY,
		ScalabilityModeS2T3, ScalabilityModeS2T3h:
		return 10
	case ScalabilityModeL3T1, ScalabilityModeL3T1h, ScalabilityModeL3T1_KEY,
		ScalabilityModeS3T1, ScalabilityModeS3T1h:
		return 6
	case ScalabilityModeL3T2, ScalabilityModeL3T2h, ScalabilityModeL3T2_KEY,
		ScalabilityModeS3T2, ScalabilityModeS3T2h:
		return 9
	case ScalabilityModeL3T3, ScalabilityModeL3T3h, ScalabilityModeL3T3_KEY,
		ScalabilityModeS3T3, ScalabilityModeS3T3h:
		return 15
	default:
		return spatial * temporal
	}
}

func TestAppendWebRTCScalabilityModesMatchesPinnedLibWebRTC(t *testing.T) {
	want := pinnedLibWebRTCScalabilityModes(t)
	prefix := []ScalabilityMode{ScalabilityModeL3T3}
	modes := AppendWebRTCScalabilityModes(prefix)
	if WebRTCScalabilityModeCount() != len(want) {
		t.Fatalf("WebRTCScalabilityModeCount=%d want %d", WebRTCScalabilityModeCount(), len(want))
	}
	if len(modes) != len(want)+1 {
		t.Fatalf("len=%d want %d", len(modes), len(want)+1)
	}
	if modes[0] != ScalabilityModeL3T3 {
		t.Fatalf("prefix mutated to %s", modes[0])
	}
	seen := make(map[ScalabilityMode]bool, len(want))
	for i, wantMode := range want {
		mode := modes[i+1]
		if mode != wantMode || !mode.webRTCSupported() {
			t.Fatalf("mode[%d]=%s supported=%v want %s", i+1, mode, mode.webRTCSupported(), wantMode)
		}
		if seen[mode] {
			t.Fatalf("duplicate mode %s", mode)
		}
		seen[mode] = true
	}
	for _, unsupported := range []ScalabilityMode{
		ScalabilityModeL2T3_KEY_SHIFT,
		ScalabilityModeL3T2_KEY_SHIFT,
		ScalabilityModeL3T3_KEY_SHIFT,
	} {
		if unsupported.webRTCSupported() {
			t.Fatalf("%s unexpectedly exported as a pinned libwebrtc WebRTC mode", unsupported)
		}
	}
}

func pinnedLibWebRTCScalabilityModes(t *testing.T) []ScalabilityMode {
	t.Helper()
	root := repoRootFromTestWD(t)
	path := filepath.Join(root, "third_party", "upstream", "webrtc", "api", "video_codecs", "scalability_mode.h")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read pinned libwebrtc scalability mode source: %v", err)
	}

	var modes []ScalabilityMode
	inCatalog := false
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.Contains(line, "kAllScalabilityModes[]") {
			inCatalog = true
			continue
		}
		if !inCatalog {
			continue
		}
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "};") {
			break
		}
		const token = "ScalabilityMode::k"
		idx := strings.Index(line, token)
		if idx < 0 {
			continue
		}
		name := line[idx+len(token):]
		if end := strings.IndexAny(name, ", \t\r/"); end >= 0 {
			name = name[:end]
		}
		mode, ok := ParseScalabilityMode(name)
		if !ok {
			t.Fatalf("pinned libwebrtc scalability mode %q is not implemented", name)
		}
		modes = append(modes, mode)
	}
	if len(modes) == 0 {
		t.Fatalf("no pinned libwebrtc scalability modes found in %s", path)
	}
	return modes
}

func repoRootFromTestWD(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("get test working directory: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("could not find repo root from %s", dir)
		}
		dir = parent
	}
}

func TestValidateWebRTCActiveScalabilityModes(t *testing.T) {
	for _, modes := range [][]ScalabilityMode{
		nil,
		{ScalabilityModeL1T1},
		{ScalabilityModeS3T3h},
		{ScalabilityModeL1T3, ScalabilityModeL1T3, ScalabilityModeL1T3},
		{ScalabilityModeL2T3_KEY, ScalabilityModeL3T3_KEY},
	} {
		if err := ValidateWebRTCActiveScalabilityModes(modes); err != nil {
			t.Fatalf("ValidateWebRTCActiveScalabilityModes(%v): %v", modes, err)
		}
	}

	for _, modes := range [][]ScalabilityMode{
		{ScalabilityMode(scalabilityModeCount)},
		{ScalabilityModeL2T3_KEY_SHIFT},
		{ScalabilityModeL3T2_KEY_SHIFT},
		{ScalabilityModeL3T3_KEY_SHIFT},
		{ScalabilityModeL1T3, ScalabilityModeS2T1},
		{ScalabilityModeS2T3h, ScalabilityModeS3T3},
	} {
		if err := ValidateWebRTCActiveScalabilityModes(modes); !errors.Is(err, ErrInvalidConfig) {
			t.Fatalf("ValidateWebRTCActiveScalabilityModes(%v) err=%v want %v", modes, err, ErrInvalidConfig)
		}
	}
}

func TestWebRTCScalabilityModeIDC(t *testing.T) {
	seen := make(map[ScalabilityMode]bool, scalabilityModeCount)
	for _, tc := range []struct {
		mode ScalabilityMode
		idc  uint8
		ok   bool
	}{
		{mode: ScalabilityModeL1T1},
		{mode: ScalabilityModeL1T2, idc: obu.ScalabilityModeL1T2, ok: true},
		{mode: ScalabilityModeL1T3, idc: obu.ScalabilityModeL1T3, ok: true},
		{mode: ScalabilityModeL2T1, idc: obu.ScalabilityModeL2T1, ok: true},
		{mode: ScalabilityModeL2T1h, idc: obu.ScalabilityModeL2T1h, ok: true},
		{mode: ScalabilityModeL2T1_KEY},
		{mode: ScalabilityModeL2T2, idc: obu.ScalabilityModeL2T2, ok: true},
		{mode: ScalabilityModeL2T2h, idc: obu.ScalabilityModeL2T2h, ok: true},
		{mode: ScalabilityModeL2T2_KEY, idc: obu.ScalabilityModeL3T2_KEY, ok: true},
		{mode: ScalabilityModeL2T2_KEY_SHIFT, idc: obu.ScalabilityModeL3T2_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeL2T3, idc: obu.ScalabilityModeL2T3, ok: true},
		{mode: ScalabilityModeL2T3h, idc: obu.ScalabilityModeL2T3h, ok: true},
		{mode: ScalabilityModeL2T3_KEY, idc: obu.ScalabilityModeL3T3_KEY, ok: true},
		{mode: ScalabilityModeL2T3_KEY_SHIFT, idc: obu.ScalabilityModeL3T3_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeL3T1, idc: obu.ScalabilityModeL3T1, ok: true},
		{mode: ScalabilityModeL3T1h},
		{mode: ScalabilityModeL3T1_KEY},
		{mode: ScalabilityModeL3T2, idc: obu.ScalabilityModeL3T2, ok: true},
		{mode: ScalabilityModeL3T2h},
		{mode: ScalabilityModeL3T2_KEY, idc: obu.ScalabilityModeL4T5_KEY, ok: true},
		{mode: ScalabilityModeL3T2_KEY_SHIFT, idc: obu.ScalabilityModeL4T5_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeL3T3, idc: obu.ScalabilityModeL3T3, ok: true},
		{mode: ScalabilityModeL3T3h},
		{mode: ScalabilityModeL3T3_KEY, idc: obu.ScalabilityModeL4T7_KEY, ok: true},
		{mode: ScalabilityModeL3T3_KEY_SHIFT, idc: obu.ScalabilityModeL4T7_KEY_SHIFT, ok: true},
		{mode: ScalabilityModeS2T1, idc: obu.ScalabilityModeS2T1, ok: true},
		{mode: ScalabilityModeS2T1h, idc: obu.ScalabilityModeS2T1h, ok: true},
		{mode: ScalabilityModeS2T2, idc: obu.ScalabilityModeS2T2, ok: true},
		{mode: ScalabilityModeS2T2h, idc: obu.ScalabilityModeS2T2h, ok: true},
		{mode: ScalabilityModeS2T3, idc: obu.ScalabilityModeS2T3, ok: true},
		{mode: ScalabilityModeS2T3h, idc: obu.ScalabilityModeS2T3h, ok: true},
		{mode: ScalabilityModeS3T1, idc: obu.ScalabilityModeS3T1, ok: true},
		{mode: ScalabilityModeS3T1h},
		{mode: ScalabilityModeS3T2, idc: obu.ScalabilityModeS3T2, ok: true},
		{mode: ScalabilityModeS3T2h},
		{mode: ScalabilityModeS3T3, idc: obu.ScalabilityModeS3T3, ok: true},
		{mode: ScalabilityModeS3T3h},
	} {
		got, ok := WebRTCScalabilityModeIDC(tc.mode)
		if got != tc.idc || ok != tc.ok {
			t.Fatalf("WebRTCScalabilityModeIDC(%s)=%d,%v want %d,%v", tc.mode, got, ok, tc.idc, tc.ok)
		}
		got, ok = tc.mode.AV1ScalabilityModeIDC()
		if got != tc.idc || ok != tc.ok {
			t.Fatalf("%s.AV1ScalabilityModeIDC()=%d,%v want %d,%v", tc.mode, got, ok, tc.idc, tc.ok)
		}
		seen[tc.mode] = true
	}
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		if !seen[mode] {
			t.Fatalf("missing IDC test coverage for %s", mode)
		}
	}
	if got, ok := WebRTCScalabilityModeIDC(ScalabilityMode(scalabilityModeCount)); got != 0 || ok {
		t.Fatalf("invalid mode IDC=%d,%v want 0,false", got, ok)
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
		TargetBitrateKbps: 350,
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
	if got.SpatialLayers[0].MinBitrateKbps != 100 || got.SpatialLayers[0].MaxBitrateKbps != 500 || got.SpatialLayers[0].TargetBitrateKbps != 350 {
		t.Fatalf("single-layer bitrate not copied: %+v", got.SpatialLayers[0])
	}

	midpoint := base
	midpoint.TargetBitrateKbps = 0
	got, err = SetWebRTCSVCConfig(midpoint, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig L1T1 midpoint: %v", err)
	}
	if got.SpatialLayers[0].TargetBitrateKbps != 300 {
		t.Fatalf("single-layer default target=%d want 300", got.SpatialLayers[0].TargetBitrateKbps)
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

func TestSetWebRTCSVCConfigExplicitColorBitDepthOverridesStaleTopLevel(t *testing.T) {
	base := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    500,
		TargetBitrateKbps: 350,
	}
	normalized, err := SetWebRTCSVCConfig(base, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig base: %v", err)
	}
	if normalized.BitDepth != 8 {
		t.Fatalf("base normalized bit depth=%d want 8", normalized.BitDepth)
	}

	reconfigured := normalized
	reconfigured.Profile = Profile0
	reconfigured.ColorConfigSet = true
	reconfigured.ColorConfig = SequenceColorConfig{
		BitDepth:   10,
		MonoChrome: true,
	}
	got, err := SetWebRTCSVCConfig(reconfigured, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig high-bit-depth mono reconfigure: %v", err)
	}
	if got.BitDepth != 10 || got.ColorConfig.BitDepth != 10 || !got.ColorConfig.MonoChrome {
		t.Fatalf("reconfigured color=%+v top bitDepth=%d want 10-bit monochrome", got.ColorConfig, got.BitDepth)
	}
}

func TestSetWebRTCSVCConfigPreservesSpatialLayerBitrates(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 640, Height: 360},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    1000,
		TargetBitrateKbps: 650,
		Scalability:       ScalabilityModeS2T3,
	}
	cfg.SpatialLayers[0].MinBitrateKbps = 120
	cfg.SpatialLayers[0].MaxBitrateKbps = 520
	cfg.SpatialLayers[0].TargetBitrateKbps = 260
	cfg.SpatialLayers[1].MinBitrateKbps = 340
	cfg.SpatialLayers[1].MaxBitrateKbps = 1500
	cfg.SpatialLayers[1].TargetBitrateKbps = 900

	got, err := SetWebRTCSVCConfig(cfg, 0, 0)
	if err != nil {
		t.Fatalf("SetWebRTCSVCConfig explicit layers: %v", err)
	}
	if got.SpatialLayers[0].Resolution != (Resolution{Width: 320, Height: 180}) ||
		got.SpatialLayers[0].MinBitrateKbps != 120 ||
		got.SpatialLayers[0].MaxBitrateKbps != 520 ||
		got.SpatialLayers[0].TargetBitrateKbps != 260 {
		t.Fatalf("base layer = %+v", got.SpatialLayers[0])
	}
	if got.SpatialLayers[1].Resolution != (Resolution{Width: 640, Height: 360}) ||
		got.SpatialLayers[1].MinBitrateKbps != 340 ||
		got.SpatialLayers[1].MaxBitrateKbps != 1500 ||
		got.SpatialLayers[1].TargetBitrateKbps != 900 {
		t.Fatalf("top layer = %+v", got.SpatialLayers[1])
	}

	if _, err := SetWebRTCSVCConfig(got, 0, 0); err != nil {
		t.Fatalf("SetWebRTCSVCConfig normalized explicit layers: %v", err)
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
		{name: "layer-target-outside", cfg: withSpatialLayerBitrate(withScalability(valid, ScalabilityModeS2T2), 1, 400, 800, 900)},
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

func withScalability(cfg Config, mode ScalabilityMode) Config {
	cfg.Scalability = mode
	return cfg
}

func withSpatialLayerBitrate(cfg Config, layer int, minBitrateKbps int32, maxBitrateKbps int32, targetBitrateKbps int32) Config {
	cfg.SpatialLayers[layer].MinBitrateKbps = minBitrateKbps
	cfg.SpatialLayers[layer].MaxBitrateKbps = maxBitrateKbps
	cfg.SpatialLayers[layer].TargetBitrateKbps = targetBitrateKbps
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
	factor, ok = SupportedResolutionScaling(
		Resolution{Width: 960, Height: 540},
		Resolution{Width: 1440, Height: 810},
	)
	if !ok || factor != (Rational{Num: 3, Den: 2}) {
		t.Fatalf("SupportedResolutionScaling 3:2 = %+v,%v; want 3/2,true", factor, ok)
	}
	factor, ok = SupportedResolutionScaling(
		Resolution{Width: 1440, Height: 810},
		Resolution{Width: 960, Height: 540},
	)
	if !ok || factor != (Rational{Num: 2, Den: 3}) {
		t.Fatalf("SupportedResolutionScaling 2:3 = %+v,%v; want 2/3,true", factor, ok)
	}
	factor, ok = SupportedResolutionScaling(
		Resolution{Width: 853, Height: 480},
		Resolution{Width: 1280, Height: 720},
	)
	if !ok || factor != (Rational{Num: 3, Den: 2}) {
		t.Fatalf("SupportedResolutionScaling truncated 3:2 = %+v,%v; want 3/2,true", factor, ok)
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
