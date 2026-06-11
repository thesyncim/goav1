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

func TestWebRTCFrameDependencyStructureForConfigL1T3(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL1T3,
	}
	got, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if got.NumDecodeTargets != 1 || got.NumChains != 1 || got.TemplateNum != 3 || got.ResolutionNum != 1 {
		t.Fatalf("shape = %+v", got)
	}
	if got.Resolutions[0] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("resolutions = %+v", got.Resolutions)
	}
	want := [...]DecodeTargetIndication{
		DecodeTargetSwitch,
		DecodeTargetDiscardable,
		DecodeTargetDiscardable,
	}
	for i, wantDTI := range want {
		template := got.Templates[i]
		if template.SpatialID != 0 || template.TemporalID != uint8(i) ||
			template.DTINum != 1 || template.DTIs[0] != wantDTI {
			t.Fatalf("template[%d] = %+v", i, template)
		}
	}
}

func TestWebRTCTemplateIDForFrame(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	structure.StructureID = 62
	id, err := WebRTCTemplateIDForFrame(structure, WebRTCGenericFrameInfo{
		SpatialID:  1,
		TemporalID: 0,
	})
	if err != nil {
		t.Fatalf("WebRTCTemplateIDForFrame: %v", err)
	}
	if id != 0 {
		t.Fatalf("template id=%d want 0", id)
	}
	if _, err := WebRTCTemplateIDForFrame(structure, WebRTCGenericFrameInfo{SpatialID: 3}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing layer err=%v want ErrInvalidFrame", err)
	}
}

func TestWebRTCTemporalUnitControlForFramesKeyUnit(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	frames := [...]FrameEncodeSettings{
		{
			Type:            FrameTypeKey,
			Resolution:      Resolution{Width: 320, Height: 180},
			SpatialID:       0,
			TemporalID:      0,
			UpdateBuffer:    0,
			UpdateBufferSet: true,
			Output:          true,
		},
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			TemporalID:       0,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{0},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}

	got, err := WebRTCTemporalUnitControlForFrames(cfg, frames[:], ReferenceBufferState{}, FrameIDBufferState{}, 10)
	if err != nil {
		t.Fatalf("WebRTCTemporalUnitControlForFrames key unit: %v", err)
	}
	if got.FrameNum != 2 || !got.HasDependencyStructure || !got.Frames[0].AttachDependencyStructure || got.Frames[1].AttachDependencyStructure {
		t.Fatalf("control flags = %+v", got)
	}
	if got.Frames[1].GenericFrameInfo.DependencyNum != 1 || got.Frames[1].GenericFrameInfo.Dependencies[0] != 10 {
		t.Fatalf("upper-layer generic info = %+v", got.Frames[1].GenericFrameInfo)
	}
	if got.Frames[1].LibaomSVCRefFrameConfig.Reference[0] != 1 || got.Frames[1].LibaomSVCRefFrameConfig.RefIdx[0] != 0 {
		t.Fatalf("upper-layer ref config = %+v", got.Frames[1].LibaomSVCRefFrameConfig)
	}
	if !got.ReferenceState.Valid[0] || !got.ReferenceState.Valid[1] ||
		got.FrameIDState.FrameIDs[0] != 10 || got.FrameIDState.FrameIDs[1] != 11 {
		t.Fatalf("states = refs %+v ids %+v", got.ReferenceState, got.FrameIDState)
	}
	if got.DependencyStructure.TemplateNum != 4 || got.DependencyStructure.NumDecodeTargets != 2 {
		t.Fatalf("dependency structure = %+v", got.DependencyStructure)
	}
}

func TestWebRTCDependencyStructureStateForTemporalUnit(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	state, structure, err := WebRTCDependencyStructureStateForTemporalUnit(key.Control, WebRTCDependencyStructureState{})
	if err != nil {
		t.Fatalf("key structure state: %v", err)
	}
	if !state.Valid || structure != key.Control.DependencyStructure || state.Structure != key.Control.DependencyStructure {
		t.Fatalf("key state=%+v structure=%+v control=%+v", state, structure, key.Control.DependencyStructure)
	}

	delta, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 200)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	next, carried, err := WebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	if err != nil {
		t.Fatalf("delta structure state: %v", err)
	}
	if !next.Valid || next.Structure != state.Structure || carried != state.Structure {
		t.Fatalf("delta state=%+v carried=%+v previous=%+v", next, carried, state.Structure)
	}

	var descriptorBuf [16]byte
	descriptor, err := AppendWebRTCDependencyDescriptor(
		descriptorBuf[:0],
		carried,
		delta.Control.Frames[1].GenericFrameInfo,
		true,
		true,
		delta.Control.Frames[1].AttachDependencyStructure,
	)
	if err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor: %v", err)
	}
	if len(descriptor) == 0 {
		t.Fatal("empty dependency descriptor")
	}
}

func TestWebRTCDependencyStructureStateForTemporalUnitRejectsMissingState(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL1T1,
	}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 10)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	delta, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 0, 11)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	if _, _, err := WebRTCDependencyStructureStateForTemporalUnit(delta.Control, WebRTCDependencyStructureState{}); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("missing carried structure err=%v want %v", err, ErrInvalidFrame)
	}
}

func TestWebRTCDependencyStructureStateForTemporalUnitAllocs(t *testing.T) {
	cfg := Config{Resolution: Resolution{Width: 640, Height: 360}, Scalability: ScalabilityModeL2T2}
	key, err := WebRTCKeyFrameTemporalUnitForConfig(cfg, 0, 100)
	if err != nil {
		t.Fatalf("WebRTCKeyFrameTemporalUnitForConfig: %v", err)
	}
	state, _, err := WebRTCDependencyStructureStateForTemporalUnit(key.Control, WebRTCDependencyStructureState{})
	if err != nil {
		t.Fatalf("key structure state: %v", err)
	}
	delta, err := WebRTCDeltaFrameTemporalUnitForConfig(cfg, key.Control.ReferenceState, key.Control.FrameIDState, 1, 200)
	if err != nil {
		t.Fatalf("WebRTCDeltaFrameTemporalUnitForConfig: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _ = WebRTCDependencyStructureStateForTemporalUnit(delta.Control, state)
	})
	if allocs != 0 {
		t.Fatalf("WebRTCDependencyStructureStateForTemporalUnit allocated: %f", allocs)
	}
}

func TestWebRTCTemporalUnitControlForFramesDeltaUnit(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	refState := ReferenceBufferState{}
	refState.Valid[1] = true
	refState.Resolutions[1] = Resolution{Width: 640, Height: 360}
	idState := FrameIDBufferState{}
	idState.Valid[1] = true
	idState.FrameIDs[1] = 22
	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			TemporalID:       1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{1},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}

	got, err := WebRTCTemporalUnitControlForFrames(cfg, frames[:], refState, idState, 23)
	if err != nil {
		t.Fatalf("WebRTCTemporalUnitControlForFrames delta unit: %v", err)
	}
	if got.HasDependencyStructure || got.Frames[0].AttachDependencyStructure {
		t.Fatalf("delta unexpectedly attached structure: %+v", got)
	}
	if got.Frames[0].GenericFrameInfo.DependencyNum != 1 || got.Frames[0].GenericFrameInfo.Dependencies[0] != 22 {
		t.Fatalf("delta generic info = %+v", got.Frames[0].GenericFrameInfo)
	}
	if got.FrameIDState.FrameIDs[1] != 23 {
		t.Fatalf("delta id state = %+v", got.FrameIDState)
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

func TestWebRTCTemporalUnitControlForFramesAllocs(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	refState := ReferenceBufferState{}
	refState.Valid[1] = true
	refState.Resolutions[1] = Resolution{Width: 640, Height: 360}
	idState := FrameIDBufferState{}
	idState.Valid[1] = true
	idState.FrameIDs[1] = 22
	frames := [...]FrameEncodeSettings{
		{
			Type:             FrameTypeDelta,
			Resolution:       Resolution{Width: 640, Height: 360},
			SpatialID:        1,
			TemporalID:       1,
			ReferenceBuffers: [WebRTCMaxFrameReferences]uint8{1},
			ReferenceCount:   1,
			UpdateBuffer:     1,
			UpdateBufferSet:  true,
			Output:           true,
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = WebRTCTemporalUnitControlForFrames(cfg, frames[:], refState, idState, 23)
	})
	if allocs != 0 {
		t.Fatalf("temporal-unit control allocs = %f; want 0", allocs)
	}
}
