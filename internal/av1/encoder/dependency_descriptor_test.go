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
		DTINum:     2,
	}
	info.DTIs[0] = DecodeTargetSwitch
	info.DTIs[1] = DecodeTargetSwitch

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
	if out[0] != 0xc0 || out[1] != 0x00 || out[2] != 0x0a || out[3] != 0xa0 {
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
		DTINum:        2,
	}
	info.Dependencies[0] = 22
	info.DTIs[0] = DecodeTargetNotPresent
	info.DTIs[1] = DecodeTargetDiscardable

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
		DTINum:        2,
	}
	info.Dependencies[0] = 23
	info.DTIs[0] = DecodeTargetNotPresent
	info.DTIs[1] = DecodeTargetDiscardable
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
		DTINum:        2,
	}
	info.Dependencies[0] = 22
	info.DTIs[0] = DecodeTargetNotPresent
	info.DTIs[1] = DecodeTargetDiscardable
	var buf [16]byte
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = WebRTCDependencyDescriptorSize(structure, info, false)
		_, _ = AppendWebRTCDependencyDescriptor(buf[:0], structure, info, true, true, false)
	})
	if allocs != 0 {
		t.Fatalf("dependency descriptor allocs=%f want 0", allocs)
	}
}
