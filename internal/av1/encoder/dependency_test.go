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
	if info.DTINum != 4 ||
		info.DTIs[0] != DecodeTargetNotPresent ||
		info.DTIs[1] != DecodeTargetNotPresent ||
		info.DTIs[2] != DecodeTargetNotPresent ||
		info.DTIs[3] != DecodeTargetDiscardable {
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
	if info.DependencyNum != 0 || info.DTINum != 4 ||
		info.DTIs[0] != DecodeTargetSwitch || info.DTIs[1] != DecodeTargetSwitch ||
		info.DTIs[2] != DecodeTargetSwitch || info.DTIs[3] != DecodeTargetSwitch {
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
	if got.NumDecodeTargets != 4 || got.NumChains != 2 || got.TemplateNum != 6 || got.ResolutionNum != 2 {
		t.Fatalf("shape = %+v", got)
	}
	if got.Resolutions[0] != (Resolution{Width: 320, Height: 180}) || got.Resolutions[1] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("resolutions = %+v", got.Resolutions)
	}
	assertWebRTCTemplateSpecs(t, got, []webRTCTemplateSpec{
		{spatial: 0, temporal: 0, dtis: "SSSS", chainDiffs: []uint8{0, 0}},
		{spatial: 0, temporal: 0, dtis: "SSRR", frameDiffs: []uint16{4}, chainDiffs: []uint8{4, 3}},
		{spatial: 0, temporal: 1, dtis: "-D-R", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
		{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
		{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{4, 1}, chainDiffs: []uint8{1, 1}},
		{spatial: 1, temporal: 1, dtis: "---D", frameDiffs: []uint16{2, 1}, chainDiffs: []uint8{3, 2}},
	})
}

func TestWebRTCFrameDependencyStructureForConfigL2T2KeyShift(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2_KEY_SHIFT,
	}
	got, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	if got.NumDecodeTargets != 4 || got.NumChains != 2 || got.TemplateNum != 7 || got.ResolutionNum != 2 {
		t.Fatalf("shape = %+v", got)
	}
	want := [...]struct {
		spatial    uint8
		temporal   uint8
		dtis       [4]DecodeTargetIndication
		frameDiffs [WebRTCMaxFrameReferences]uint16
		frameNum   uint8
		chainDiffs [2]uint8
	}{
		{spatial: 0, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetSwitch}, chainDiffs: [2]uint8{0, 0}},
		{spatial: 0, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetNotPresent, DecodeTargetNotPresent}, frameDiffs: [WebRTCMaxFrameReferences]uint16{2}, frameNum: 1, chainDiffs: [2]uint8{2, 1}},
		{spatial: 0, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetSwitch, DecodeTargetSwitch, DecodeTargetNotPresent, DecodeTargetNotPresent}, frameDiffs: [WebRTCMaxFrameReferences]uint16{4}, frameNum: 1, chainDiffs: [2]uint8{4, 1}},
		{spatial: 0, temporal: 1, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetDiscardable, DecodeTargetNotPresent, DecodeTargetNotPresent}, frameDiffs: [WebRTCMaxFrameReferences]uint16{2}, frameNum: 1, chainDiffs: [2]uint8{2, 3}},
		{spatial: 1, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetSwitch, DecodeTargetSwitch}, frameDiffs: [WebRTCMaxFrameReferences]uint16{1}, frameNum: 1, chainDiffs: [2]uint8{1, 1}},
		{spatial: 1, temporal: 0, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetSwitch, DecodeTargetSwitch}, frameDiffs: [WebRTCMaxFrameReferences]uint16{4}, frameNum: 1, chainDiffs: [2]uint8{3, 4}},
		{spatial: 1, temporal: 1, dtis: [4]DecodeTargetIndication{DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetNotPresent, DecodeTargetDiscardable}, frameDiffs: [WebRTCMaxFrameReferences]uint16{2}, frameNum: 1, chainDiffs: [2]uint8{1, 2}},
	}
	for i, want := range want {
		template := got.Templates[i]
		if template.SpatialID != want.spatial || template.TemporalID != want.temporal ||
			template.DTINum != 4 || template.FrameDiffNum != want.frameNum ||
			template.ChainDiffNum != 2 {
			t.Fatalf("template[%d] shape = %+v", i, template)
		}
		for j := 0; j < 4; j++ {
			if template.DTIs[j] != want.dtis[j] {
				t.Fatalf("template[%d] dti[%d]=%v want %v", i, j, template.DTIs[j], want.dtis[j])
			}
		}
		for j := uint8(0); j < want.frameNum; j++ {
			if template.FrameDiffs[j] != want.frameDiffs[j] {
				t.Fatalf("template[%d] frame diff[%d]=%d want %d", i, j, template.FrameDiffs[j], want.frameDiffs[j])
			}
		}
		if template.ChainDiffs[0] != want.chainDiffs[0] || template.ChainDiffs[1] != want.chainDiffs[1] {
			t.Fatalf("template[%d] chain diffs=%+v want %+v", i, template.ChainDiffs, want.chainDiffs)
		}
	}
}

func TestWebRTCFrameDependencyStructureForConfigKeyReferenceTemplates(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode ScalabilityMode
		want []webRTCTemplateSpec
	}{
		{
			name: "L2T1_KEY",
			mode: ScalabilityModeL2T1_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "S-", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
				{spatial: 0, temporal: 0, dtis: "SS", chainDiffs: []uint8{0, 0}},
				{spatial: 1, temporal: 0, dtis: "-S", frameDiffs: []uint16{2}, chainDiffs: []uint8{1, 2}},
				{spatial: 1, temporal: 0, dtis: "-S", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
			},
		},
		{
			name: "L2T2_KEY",
			mode: ScalabilityModeL2T2_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SSSS", chainDiffs: []uint8{0, 0}},
				{spatial: 0, temporal: 0, dtis: "SS--", frameDiffs: []uint16{4}, chainDiffs: []uint8{4, 3}},
				{spatial: 0, temporal: 1, dtis: "-D--", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
				{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
				{spatial: 1, temporal: 0, dtis: "--SS", frameDiffs: []uint16{4}, chainDiffs: []uint8{1, 4}},
				{spatial: 1, temporal: 1, dtis: "---D", frameDiffs: []uint16{2}, chainDiffs: []uint8{3, 2}},
			},
		},
		{
			name: "L2T3_KEY",
			mode: ScalabilityModeL2T3_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SSSSSS", chainDiffs: []uint8{0, 0}},
				{spatial: 0, temporal: 0, dtis: "SSS---", frameDiffs: []uint16{8}, chainDiffs: []uint8{8, 7}},
				{spatial: 0, temporal: 1, dtis: "-DS---", frameDiffs: []uint16{4}, chainDiffs: []uint8{4, 3}},
				{spatial: 0, temporal: 2, dtis: "--D---", frameDiffs: []uint16{2}, chainDiffs: []uint8{2, 1}},
				{spatial: 0, temporal: 2, dtis: "--D---", frameDiffs: []uint16{2}, chainDiffs: []uint8{6, 5}},
				{spatial: 1, temporal: 0, dtis: "---SSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1}},
				{spatial: 1, temporal: 0, dtis: "---SSS", frameDiffs: []uint16{8}, chainDiffs: []uint8{1, 8}},
				{spatial: 1, temporal: 1, dtis: "----DS", frameDiffs: []uint16{4}, chainDiffs: []uint8{5, 4}},
				{spatial: 1, temporal: 2, dtis: "-----D", frameDiffs: []uint16{2}, chainDiffs: []uint8{3, 2}},
				{spatial: 1, temporal: 2, dtis: "-----D", frameDiffs: []uint16{2}, chainDiffs: []uint8{7, 6}},
			},
		},
		{
			name: "L3T1_KEY",
			mode: ScalabilityModeL3T1_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "S--", frameDiffs: []uint16{3}, chainDiffs: []uint8{3, 2, 1}},
				{spatial: 0, temporal: 0, dtis: "SSS", chainDiffs: []uint8{0, 0, 0}},
				{spatial: 1, temporal: 0, dtis: "-S-", frameDiffs: []uint16{3}, chainDiffs: []uint8{1, 3, 2}},
				{spatial: 1, temporal: 0, dtis: "-SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1, 1}},
				{spatial: 2, temporal: 0, dtis: "--S", frameDiffs: []uint16{3}, chainDiffs: []uint8{2, 1, 3}},
				{spatial: 2, temporal: 0, dtis: "--S", frameDiffs: []uint16{1}, chainDiffs: []uint8{2, 1, 1}},
			},
		},
		{
			name: "L3T2_KEY",
			mode: ScalabilityModeL3T2_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SS----", frameDiffs: []uint16{6}, chainDiffs: []uint8{6, 5, 4}},
				{spatial: 0, temporal: 0, dtis: "SSSSSS", chainDiffs: []uint8{0, 0, 0}},
				{spatial: 0, temporal: 1, dtis: "-D----", frameDiffs: []uint16{3}, chainDiffs: []uint8{3, 2, 1}},
				{spatial: 1, temporal: 0, dtis: "--SS--", frameDiffs: []uint16{6}, chainDiffs: []uint8{1, 6, 5}},
				{spatial: 1, temporal: 0, dtis: "--SSSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1, 1}},
				{spatial: 1, temporal: 1, dtis: "---D--", frameDiffs: []uint16{3}, chainDiffs: []uint8{4, 3, 2}},
				{spatial: 2, temporal: 0, dtis: "----SS", frameDiffs: []uint16{6}, chainDiffs: []uint8{2, 1, 6}},
				{spatial: 2, temporal: 0, dtis: "----SS", frameDiffs: []uint16{1}, chainDiffs: []uint8{2, 1, 1}},
				{spatial: 2, temporal: 1, dtis: "-----D", frameDiffs: []uint16{3}, chainDiffs: []uint8{5, 4, 3}},
			},
		},
		{
			name: "L3T3_KEY",
			mode: ScalabilityModeL3T3_KEY,
			want: []webRTCTemplateSpec{
				{spatial: 0, temporal: 0, dtis: "SSSSSSSSS", chainDiffs: []uint8{0, 0, 0}},
				{spatial: 0, temporal: 0, dtis: "SSS------", frameDiffs: []uint16{12}, chainDiffs: []uint8{12, 11, 10}},
				{spatial: 0, temporal: 1, dtis: "-DS------", frameDiffs: []uint16{6}, chainDiffs: []uint8{6, 5, 4}},
				{spatial: 0, temporal: 2, dtis: "--D------", frameDiffs: []uint16{3}, chainDiffs: []uint8{3, 2, 1}},
				{spatial: 0, temporal: 2, dtis: "--D------", frameDiffs: []uint16{3}, chainDiffs: []uint8{9, 8, 7}},
				{spatial: 1, temporal: 0, dtis: "---SSSSSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{1, 1, 1}},
				{spatial: 1, temporal: 0, dtis: "---SSS---", frameDiffs: []uint16{12}, chainDiffs: []uint8{1, 12, 11}},
				{spatial: 1, temporal: 1, dtis: "----DS---", frameDiffs: []uint16{6}, chainDiffs: []uint8{7, 6, 5}},
				{spatial: 1, temporal: 2, dtis: "-----D---", frameDiffs: []uint16{3}, chainDiffs: []uint8{4, 3, 2}},
				{spatial: 1, temporal: 2, dtis: "-----D---", frameDiffs: []uint16{3}, chainDiffs: []uint8{10, 9, 8}},
				{spatial: 2, temporal: 0, dtis: "------SSS", frameDiffs: []uint16{1}, chainDiffs: []uint8{2, 1, 1}},
				{spatial: 2, temporal: 0, dtis: "------SSS", frameDiffs: []uint16{12}, chainDiffs: []uint8{2, 1, 12}},
				{spatial: 2, temporal: 1, dtis: "-------DS", frameDiffs: []uint16{6}, chainDiffs: []uint8{8, 7, 6}},
				{spatial: 2, temporal: 2, dtis: "--------D", frameDiffs: []uint16{3}, chainDiffs: []uint8{5, 4, 3}},
				{spatial: 2, temporal: 2, dtis: "--------D", frameDiffs: []uint16{3}, chainDiffs: []uint8{11, 10, 9}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := WebRTCFrameDependencyStructureForConfig(Config{
				Resolution:  Resolution{Width: 1280, Height: 720},
				Scalability: tc.mode,
			})
			if err != nil {
				t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
			}
			assertWebRTCTemplateSpecs(t, got, tc.want)
		})
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
	if got.NumDecodeTargets != 3 || got.NumChains != 1 || got.TemplateNum != 5 || got.ResolutionNum != 1 {
		t.Fatalf("shape = %+v", got)
	}
	if got.Resolutions[0] != (Resolution{Width: 640, Height: 360}) {
		t.Fatalf("resolutions = %+v", got.Resolutions)
	}
	assertWebRTCTemplateSpecs(t, got, []webRTCTemplateSpec{
		{spatial: 0, temporal: 0, dtis: "SSS", chainDiffs: []uint8{0}},
		{spatial: 0, temporal: 0, dtis: "SSS", frameDiffs: []uint16{4}, chainDiffs: []uint8{4}},
		{spatial: 0, temporal: 1, dtis: "-DS", frameDiffs: []uint16{2}, chainDiffs: []uint8{2}},
		{spatial: 0, temporal: 2, dtis: "--D", frameDiffs: []uint16{1}, chainDiffs: []uint8{1}},
		{spatial: 0, temporal: 2, dtis: "--D", frameDiffs: []uint16{1}, chainDiffs: []uint8{3}},
	})
}

type webRTCTemplateSpec struct {
	spatial    uint8
	temporal   uint8
	dtis       string
	frameDiffs []uint16
	chainDiffs []uint8
}

func assertWebRTCTemplateSpecs(t *testing.T, structure WebRTCFrameDependencyStructure, want []webRTCTemplateSpec) {
	t.Helper()
	if int(structure.TemplateNum) != len(want) {
		t.Fatalf("template num=%d want %d", structure.TemplateNum, len(want))
	}
	for i, want := range want {
		template := structure.Templates[i]
		if template.SpatialID != want.spatial || template.TemporalID != want.temporal ||
			template.DTINum != uint8(len(want.dtis)) ||
			template.FrameDiffNum != uint8(len(want.frameDiffs)) ||
			template.ChainDiffNum != uint8(len(want.chainDiffs)) {
			t.Fatalf("template[%d] shape=%+v want spatial=%d temporal=%d dtis=%q frameDiffs=%v chainDiffs=%v",
				i, template, want.spatial, want.temporal, want.dtis, want.frameDiffs, want.chainDiffs)
		}
		for j := 0; j < len(want.dtis); j++ {
			if got, want := template.DTIs[j], webRTCTestDTI(want.dtis[j]); got != want {
				t.Fatalf("template[%d] dti[%d]=%v want %v", i, j, got, want)
			}
		}
		for j := 0; j < len(want.frameDiffs); j++ {
			if got := template.FrameDiffs[j]; got != want.frameDiffs[j] {
				t.Fatalf("template[%d] frame diff[%d]=%d want %d", i, j, got, want.frameDiffs[j])
			}
		}
		for j := 0; j < len(want.chainDiffs); j++ {
			if got := template.ChainDiffs[j]; got != want.chainDiffs[j] {
				t.Fatalf("template[%d] chain diff[%d]=%d want %d", i, j, got, want.chainDiffs[j])
			}
		}
	}
}

func webRTCTestDTI(symbol byte) DecodeTargetIndication {
	switch symbol {
	case 'D':
		return DecodeTargetDiscardable
	case 'S':
		return DecodeTargetSwitch
	case 'R':
		return DecodeTargetRequired
	default:
		return DecodeTargetNotPresent
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
	structure.StructureID = 61
	info := WebRTCGenericFrameInfo{
		FrameID:       10,
		SpatialID:     1,
		TemporalID:    0,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 9
	info.DTIs = structure.Templates[3].DTIs
	id, err := WebRTCTemplateIDForFrame(structure, WebRTCGenericFrameInfo{
		FrameID:       info.FrameID,
		SpatialID:     info.SpatialID,
		TemporalID:    info.TemporalID,
		Dependencies:  info.Dependencies,
		DependencyNum: info.DependencyNum,
		DTIs:          info.DTIs,
		DTINum:        info.DTINum,
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
	if got.DependencyStructure.TemplateNum != 6 || got.DependencyStructure.NumDecodeTargets != 4 {
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
