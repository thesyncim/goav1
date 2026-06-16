package rtp

import (
	"errors"
	"testing"
)

func TestDependencyDescriptorMandatoryRoundTrip(t *testing.T) {
	want := DependencyDescriptorMandatory{
		FirstPacketInFrame: true,
		LastPacketInFrame:  false,
		TemplateID:         17,
		FrameNumber:        0x1234,
	}
	var buf [DependencyDescriptorMandatorySize]byte
	n, err := PutDependencyDescriptorMandatory(buf[:], want)
	if err != nil {
		t.Fatal(err)
	}
	if n != DependencyDescriptorMandatorySize || buf != [3]byte{0x91, 0x12, 0x34} {
		t.Fatalf("encoded n=%d bytes=% x", n, buf)
	}

	got, consumed, err := ParseDependencyDescriptorMandatory(buf[:])
	if err != nil {
		t.Fatal(err)
	}
	if consumed != DependencyDescriptorMandatorySize || got != want {
		t.Fatalf("parsed=%+v consumed=%d want=%+v", got, consumed, want)
	}
}

func TestDependencyDescriptorMandatoryRejectsInvalid(t *testing.T) {
	if _, err := PutDependencyDescriptorMandatory(make([]byte, 2), DependencyDescriptorMandatory{}); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("short write err=%v want ErrShortBuffer", err)
	}
	var buf [3]byte
	if _, err := PutDependencyDescriptorMandatory(buf[:], DependencyDescriptorMandatory{TemplateID: 64}); !errors.Is(err, ErrInvalidDependencyDescriptor) {
		t.Fatalf("invalid template err=%v want ErrInvalidDependencyDescriptor", err)
	}
	if _, _, err := ParseDependencyDescriptorMandatory([]byte{0, 1}); !errors.Is(err, ErrShortPayload) {
		t.Fatalf("short parse err=%v want ErrShortPayload", err)
	}
}

func TestDependencyDescriptorMandatoryAllocs(t *testing.T) {
	descriptor := DependencyDescriptorMandatory{
		FirstPacketInFrame: true,
		LastPacketInFrame:  true,
		TemplateID:         3,
		FrameNumber:        9,
	}
	var buf [DependencyDescriptorMandatorySize]byte
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = PutDependencyDescriptorMandatory(buf[:], descriptor)
		_, _, _ = ParseDependencyDescriptorMandatory(buf[:])
	})
	if allocs != 0 {
		t.Fatalf("mandatory descriptor allocs=%f want 0", allocs)
	}
}

func TestDependencyDescriptorParseActiveTargetsAndCustomChains(t *testing.T) {
	structure := DependencyDescriptorStructure{
		StructureID:      5,
		NumDecodeTargets: 2,
		NumChains:        2,
		TemplateNum:      2,
		ResolutionNum:    2,
	}
	structure.DecodeTargetProtectedByChain[0] = 0
	structure.DecodeTargetProtectedByChain[1] = 1
	structure.DecodeTargetSpatialID[0] = 0
	structure.DecodeTargetSpatialID[1] = 1
	structure.Resolutions[0] = DependencyDescriptorResolution{Width: 320, Height: 180}
	structure.Resolutions[1] = DependencyDescriptorResolution{Width: 640, Height: 360}
	structure.Templates[0] = DependencyDescriptorFrameDependencies{
		SpatialID:    0,
		TemporalID:   0,
		DTINum:       2,
		ChainDiffNum: 2,
	}
	structure.Templates[0].DTIs[0] = DependencyDescriptorDecodeTargetSwitch
	structure.Templates[0].DTIs[1] = DependencyDescriptorDecodeTargetRequired
	structure.Templates[0].ChainDiffs[0] = 1
	structure.Templates[0].ChainDiffs[1] = 2
	structure.Templates[1] = DependencyDescriptorFrameDependencies{
		SpatialID:    1,
		TemporalID:   0,
		DTINum:       2,
		ChainDiffNum: 2,
	}
	structure.Templates[1].DTIs[1] = DependencyDescriptorDecodeTargetSwitch
	structure.Templates[1].FrameDiffNum = 1
	structure.Templates[1].FrameDiffs[0] = 1
	structure.Templates[1].ChainDiffs[0] = 3
	structure.Templates[1].ChainDiffs[1] = 4

	var w dependencyDescriptorTestBitWriter
	w.writeBool(true)
	w.writeBool(true)
	w.writeBits(6, 6)
	w.writeBits(200, 16)
	w.writeBool(false) // no attached structure
	w.writeBool(true)  // active decode targets
	w.writeBool(true)  // custom dtis
	w.writeBool(true)  // custom frame diffs
	w.writeBool(true)  // custom chains
	w.writeBits(0x01, 2)
	w.writeBits(uint64(DependencyDescriptorDecodeTargetRequired), 2)
	w.writeBits(uint64(DependencyDescriptorDecodeTargetNotPresent), 2)
	w.writeBits(2, 2)
	w.writeBits(16, 8)
	w.writeBits(0, 2)
	w.writeBits(9, 8)
	w.writeBits(10, 8)

	descriptorBytes := w.bytes()
	got, consumed, err := ParseDependencyDescriptor(descriptorBytes, &structure)
	if err != nil {
		t.Fatalf("ParseDependencyDescriptor: %v descriptor=% x", err, descriptorBytes)
	}
	if consumed != len(descriptorBytes) {
		t.Fatalf("consumed=%d len=%d", consumed, len(descriptorBytes))
	}
	if !got.HasActiveDecodeTargets || got.ActiveDecodeTargetsMask != 0x01 ||
		!got.HasCustomDTIs || !got.HasCustomFrameDiffs || !got.HasCustomChainDiffs {
		t.Fatalf("flags active=%v mask=%#x dtis=%v fdiffs=%v chains=%v", got.HasActiveDecodeTargets, got.ActiveDecodeTargetsMask, got.HasCustomDTIs, got.HasCustomFrameDiffs, got.HasCustomChainDiffs)
	}
	if got.FrameDependencies.SpatialID != 1 || got.FrameDependencies.TemporalID != 0 ||
		got.FrameDependencies.DTIs[0] != DependencyDescriptorDecodeTargetRequired ||
		got.FrameDependencies.DTIs[1] != DependencyDescriptorDecodeTargetNotPresent ||
		got.FrameDependencies.FrameDiffNum != 1 || got.FrameDependencies.FrameDiffs[0] != 17 ||
		got.FrameDependencies.ChainDiffNum != 2 || got.FrameDependencies.ChainDiffs[0] != 9 || got.FrameDependencies.ChainDiffs[1] != 10 {
		t.Fatalf("frame dependencies=%+v", got.FrameDependencies)
	}
	if !got.HasResolution || got.Resolution != (DependencyDescriptorResolution{Width: 640, Height: 360}) {
		t.Fatalf("resolution=%+v has=%v", got.Resolution, got.HasResolution)
	}
}

func TestDependencyDescriptorRejectsInvalidFull(t *testing.T) {
	if _, _, err := ParseDependencyDescriptor([]byte{0xc0, 0, 1}, nil); !errors.Is(err, ErrInvalidDependencyDescriptor) {
		t.Fatalf("missing structure err=%v want ErrInvalidDependencyDescriptor", err)
	}
	if _, _, err := ParseDependencyDescriptor([]byte{0xc0, 0, 1, 0x80}, nil); !errors.Is(err, ErrShortPayload) {
		t.Fatalf("truncated attached err=%v want ErrShortPayload", err)
	}
	structure := DependencyDescriptorStructure{
		StructureID:      3,
		NumDecodeTargets: 1,
		NumChains:        0,
		TemplateNum:      1,
	}
	structure.DecodeTargetSpatialID[0] = 0
	structure.DecodeTargetTemporalID[0] = 0
	structure.Templates[0] = DependencyDescriptorFrameDependencies{DTINum: 1}
	structure.Templates[0].DTIs[0] = DependencyDescriptorDecodeTargetSwitch
	if _, _, err := ParseDependencyDescriptor([]byte{0xc5, 0, 1}, &structure); !errors.Is(err, ErrInvalidDependencyDescriptor) {
		t.Fatalf("bad template err=%v want ErrInvalidDependencyDescriptor", err)
	}
}

type dependencyDescriptorTestBitWriter struct {
	buf []byte
	bit int
}

func (w *dependencyDescriptorTestBitWriter) writeBool(value bool) {
	if value {
		w.writeBits(1, 1)
		return
	}
	w.writeBits(0, 1)
}

func (w *dependencyDescriptorTestBitWriter) writeBits(value uint64, n uint8) {
	for i := int(n) - 1; i >= 0; i-- {
		byteIndex := w.bit >> 3
		if byteIndex == len(w.buf) {
			w.buf = append(w.buf, 0)
		}
		if value&(1<<uint(i)) != 0 {
			w.buf[byteIndex] |= 1 << uint(7-(w.bit&7))
		}
		w.bit++
	}
}

func (w *dependencyDescriptorTestBitWriter) bytes() []byte {
	return w.buf
}
