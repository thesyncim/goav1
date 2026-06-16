package encoder

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

func TestAppendWebRTCDependencyDescriptorAttachedStructure(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:    10,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     structure.NumDecodeTargets,
	}
	info.DTIs = structure.Templates[0].DTIs

	size, err := WebRTCDependencyDescriptorSize(structure, info, true)
	if err != nil {
		t.Fatalf("WebRTCDependencyDescriptorSize: %v", err)
	}
	var buf [64]byte
	out, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, true)
	if err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor: %v", err)
	}
	if len(out) != size {
		t.Fatalf("len=%d want %d", len(out), size)
	}
	if out[0] != 0xc0 || out[1] != 0x00 || out[2] != 0x0a || out[3] != 0x80 {
		t.Fatalf("descriptor prefix=% x", out[:4])
	}
}

func TestAppendWebRTCDependencyDescriptorCustomFrameDiff(t *testing.T) {
	cfg := Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(cfg)
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:       23,
		SpatialID:     1,
		TemporalID:    1,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 22
	info.DTIs = structure.Templates[3].DTIs

	var buf [16]byte
	out, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false)
	if err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor: %v", err)
	}
	if len(out) != 5 {
		t.Fatalf("descriptor=% x len=%d want 5", out, len(out))
	}
	if out[0] != 0xc3 || out[1] != 0x00 || out[2] != 0x17 || out[3] != 0x12 {
		t.Fatalf("descriptor prefix=% x", out[:4])
	}
}

func TestAppendWebRTCDependencyDescriptorSingleChainStructure(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution: Resolution{Width: 640, Height: 360},
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:    77,
		SpatialID:  0,
		TemporalID: 0,
		DTINum:     1,
	}
	info.DTIs[0] = DecodeTargetSwitch
	var buf [32]byte
	if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, false, true); err != nil {
		t.Fatalf("AppendWebRTCDependencyDescriptor single-chain: %v", err)
	}
}

func TestAppendWebRTCDependencyDescriptorL1T3TemplateDTIs(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL1T3,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	for temporalID, wantTemplate := range [...]uint8{0, 1, 2} {
		info := WebRTCGenericFrameInfo{
			FrameID:       uint64(100 + temporalID),
			TemporalID:    uint8(temporalID),
			DependencyNum: 1,
			DTINum:        structure.NumDecodeTargets,
		}
		info.DTIs = structure.Templates[temporalID].DTIs
		info.Dependencies[0] = info.FrameID - 1

		match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
		if err != nil {
			t.Fatalf("match temporal %d: %v", temporalID, err)
		}
		if match.templateIndex != wantTemplate || match.needCustomDTIs {
			t.Fatalf("match temporal %d = %+v", temporalID, match)
		}
		var buf [16]byte
		if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false); err != nil {
			t.Fatalf("AppendWebRTCDependencyDescriptor temporal %d: %v", temporalID, err)
		}
	}
}

func TestAppendWebRTCDependencyDescriptorL2T2KeyShiftDuplicateTemplates(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2_KEY_SHIFT,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	for _, tc := range [...]struct {
		name          string
		frameID       uint64
		spatialID     uint8
		temporalID    uint8
		dependency    uint64
		templateIndex uint8
	}{
		{name: "s0t0-diff2", frameID: 102, spatialID: 0, temporalID: 0, dependency: 100, templateIndex: 1},
		{name: "s0t0-diff4", frameID: 106, spatialID: 0, temporalID: 0, dependency: 102, templateIndex: 2},
		{name: "s1t0-diff1", frameID: 101, spatialID: 1, temporalID: 0, dependency: 100, templateIndex: 4},
		{name: "s1t0-diff4", frameID: 105, spatialID: 1, temporalID: 0, dependency: 101, templateIndex: 5},
	} {
		info := WebRTCGenericFrameInfo{
			FrameID:       tc.frameID,
			SpatialID:     tc.spatialID,
			TemporalID:    tc.temporalID,
			DependencyNum: 1,
			DTINum:        structure.NumDecodeTargets,
		}
		info.Dependencies[0] = tc.dependency
		info.DTIs = structure.Templates[tc.templateIndex].DTIs
		match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
		if err != nil {
			t.Fatalf("%s match: %v", tc.name, err)
		}
		if match.templateIndex != tc.templateIndex || match.needCustomDTIs || match.needCustomDiffs {
			t.Fatalf("%s match=%+v", tc.name, match)
		}
		templateID, err := WebRTCTemplateIDForFrame(structure, info)
		if err != nil {
			t.Fatalf("%s template id: %v", tc.name, err)
		}
		if templateID != tc.templateIndex {
			t.Fatalf("%s template id=%d want %d", tc.name, templateID, tc.templateIndex)
		}
		var buf [16]byte
		if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false); err != nil {
			t.Fatalf("%s AppendWebRTCDependencyDescriptor: %v", tc.name, err)
		}
	}
}

func TestAppendWebRTCDependencyDescriptorRejectsInvalid(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:       23,
		SpatialID:     1,
		TemporalID:    1,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 23
	info.DTIs = structure.Templates[3].DTIs
	var buf [16]byte
	if _, err := AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false); !errors.Is(err, ErrInvalidFrame) {
		t.Fatalf("invalid dependency err=%v want ErrInvalidFrame", err)
	}

	info.Dependencies[0] = 22
	if _, err := AppendWebRTCDependencyDescriptor(buf[:0:4], structure, info, true, true, false); !errors.Is(err, bitstream.ErrShortBuffer) {
		t.Fatalf("short buffer err=%v want ErrShortBuffer", err)
	}
}

func TestAppendWebRTCDependencyDescriptorAllocs(t *testing.T) {
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: 640, Height: 360},
		Scalability: ScalabilityModeL2T2,
	})
	if err != nil {
		t.Fatalf("WebRTCFrameDependencyStructureForConfig: %v", err)
	}
	info := WebRTCGenericFrameInfo{
		FrameID:       23,
		SpatialID:     1,
		TemporalID:    1,
		DependencyNum: 1,
		DTINum:        structure.NumDecodeTargets,
	}
	info.Dependencies[0] = 22
	info.DTIs = structure.Templates[3].DTIs
	var buf [16]byte
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = WebRTCDependencyDescriptorSize(structure, info, false)
		_, _ = AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false)
	})
	if allocs != 0 {
		t.Fatalf("dependency descriptor allocs=%f want 0", allocs)
	}
}
