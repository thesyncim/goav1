package encoder

import (
	"errors"
	"testing"
)

func TestLibaomSVCRefFrameConfigForFrame(t *testing.T) {
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3, 1},
		ReferenceCount:   2,
		UpdateBuffer:     4,
		UpdateBufferSet:  true,
	}

	got, err := LibaomSVCRefFrameConfigForFrame(settings)
	if err != nil {
		t.Fatalf("LibaomSVCRefFrameConfigForFrame: %v", err)
	}
	if got.Reference[0] != 1 || got.Reference[1] != 1 || got.Reference[2] != 0 {
		t.Fatalf("reference flags = %+v", got.Reference)
	}
	if got.RefIdx[0] != 3 || got.RefIdx[1] != 1 || got.RefIdx[2] != 3 || got.RefIdx[6] != 3 {
		t.Fatalf("ref idx = %+v", got.RefIdx)
	}
	if got.Refresh[4] != 1 || got.Refresh[0] != 0 {
		t.Fatalf("refresh = %+v", got.Refresh)
	}
}

func TestLibaomSVCRefFrameConfigForFrameRejectsDuplicate(t *testing.T) {
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3, 3},
		ReferenceCount:   2,
	}
	if _, err := LibaomSVCRefFrameConfigForFrame(settings); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("duplicate reference err=%v want ErrInvalidFrame", err)
	}
}

func TestWebRTCGenericFrameInfoForFrame(t *testing.T) {
	state := FrameIDBufferState{}
	state.Valid[3] = true
	state.FrameIDs[3] = 100
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		SpatialID:        1,
		TemporalID:       1,
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3},
		ReferenceCount:   1,
		UpdateBuffer:     4,
		UpdateBufferSet:  true,
	}

	info, out, err := WebRTCGenericFrameInfoForFrame(settings, 101, state, 2, 2)
	if err != nil {
		t.Fatalf("WebRTCGenericFrameInfoForFrame: %v", err)
	}
	if info.FrameID != 101 || info.SpatialID != 1 || info.TemporalID != 1 {
		t.Fatalf("info ids = %+v", info)
	}
	if info.DependencyNum != 1 || info.Dependencies[0] != 100 {
		t.Fatalf("dependencies = %+v count=%d", info.Dependencies, info.DependencyNum)
	}
	if info.DTINum != 2 || info.DTIs[0] != DecodeTargetNotPresent || info.DTIs[1] != DecodeTargetDiscardable {
		t.Fatalf("dtis = %+v count=%d", info.DTIs, info.DTINum)
	}
	if !out.Valid[4] || out.FrameIDs[4] != 101 || !out.Valid[3] {
		t.Fatalf("out state = %+v", out)
	}
}

func TestWebRTCGenericFrameInfoForFrameKeyResetsFrameIDs(t *testing.T) {
	state := FrameIDBufferState{}
	state.Valid[7] = true
	state.FrameIDs[7] = 88
	settings := FrameEncodeSettings{
		Type:            FrameTypeKey,
		Resolution:      Resolution{Width: 320, Height: 180},
		SpatialID:       0,
		TemporalID:      0,
		UpdateBuffer:    0,
		UpdateBufferSet: true,
	}

	info, out, err := WebRTCGenericFrameInfoForFrame(settings, 99, state, 2, 2)
	if err != nil {
		t.Fatalf("WebRTCGenericFrameInfoForFrame key: %v", err)
	}
	if info.DependencyNum != 0 || info.DTIs[0] != DecodeTargetSwitch || info.DTIs[1] != DecodeTargetSwitch {
		t.Fatalf("key info = %+v", info)
	}
	if out.Valid[7] || !out.Valid[0] || out.FrameIDs[0] != 99 {
		t.Fatalf("key reset state = %+v", out)
	}
}

func TestWebRTCFrameDependencyStructureForConfig(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	got, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if got.NumDecodeTargets != 2 || got.NumChains != 2 || got.TemplateNum != 4 || got.ResolutionNum != 2 {
		t.Fatalf("shape = %+v", got)
	}
	if got.Resolutions[0] != (Resolution{Width: 320, Height: 180}) || got.Resolutions[1] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("resolutions = %+v", got.Resolutions)
	}
	want := [...]struct {
		spatial uint8
		temp    uint8
		d0      DecodeTargetIndication
		d1      DecodeTargetIndication
	}{
		{spatial: 0, temp: 0, d0: DecodeTargetSwitch, d1: DecodeTargetNotPresent},
		{spatial: 0, temp: 1, d0: DecodeTargetDiscardable, d1: DecodeTargetNotPresent},
		{spatial: 1, temp: 0, d0: DecodeTargetNotPresent, d1: DecodeTargetSwitch},
		{spatial: 1, temp: 1, d0: DecodeTargetNotPresent, d1: DecodeTargetDiscardable},
	}
	for i, want := range want {
		template := got.Templates[i]
		if template.SpatialID != want.spatial || template.TemporalID != want.temp ||
			template.DTIs[0] != want.d0 || template.DTIs[1] != want.d1 {
			t.Fatalf("template[%d] = %+v", i, template)
		}
	}
}

func TestWebRTCDependencyHelpersAllocs(t *testing.T) {
	state := FrameIDBufferState{}
	state.Valid[3] = true
	state.FrameIDs[3] = 100
	settings := FrameEncodeSettings{
		Type:             FrameTypeDelta,
		Resolution:       Resolution{Width: 640, Height: 360},
		SpatialID:        1,
		TemporalID:       1,
		ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{3},
		ReferenceCount:   1,
		UpdateBuffer:     4,
		UpdateBufferSet:  true,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = LibaomSVCRefFrameConfigForFrame(settings)
		_, _, _ = WebRTCGenericFrameInfoForFrame(settings, 101, state, 2, 2)
	})
	if allocs != 0 {
		t.Fatalf("dependency helpers allocs = %f; want 0", allocs)
	}
}
