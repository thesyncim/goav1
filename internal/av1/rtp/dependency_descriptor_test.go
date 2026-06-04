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
