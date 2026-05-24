package cdef

import (
	"errors"
	"testing"
)

func TestDecodeStrength(t *testing.T) {
	tests := []struct {
		name      string
		packed    uint8
		level     int
		secondary int
		wantErr   error
	}{
		{name: "zero", packed: 0, level: 0, secondary: 0},
		{name: "secondary-one", packed: 1, level: 0, secondary: 1},
		{name: "secondary-two", packed: 2, level: 0, secondary: 2},
		{name: "secondary-three-maps-to-four", packed: 3, level: 0, secondary: 4},
		{name: "primary-and-secondary", packed: 43, level: 10, secondary: 4},
		{name: "max", packed: 63, level: 15, secondary: 4},
		{name: "overflow", packed: 64, wantErr: ErrInvalidCDEF},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			level, secondary, err := DecodeStrength(tt.packed)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("err=%v want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if level != tt.level || secondary != tt.secondary {
				t.Fatalf("level=%d secondary=%d want %d,%d", level, secondary, tt.level, tt.secondary)
			}
		})
	}
}

func TestFrameFilterParamsFromStrength(t *testing.T) {
	params, err := FrameFilterParamsFromStrength(PlaneU, 1, 0, 43, 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := FrameFilterParams{
		XDec:              1,
		YDec:              0,
		Plane:             PlaneU,
		Level:             10,
		SecondaryStrength: 4,
		Damping:           5,
		CoeffShift:        2,
	}
	if params != want {
		t.Fatalf("params=%+v want %+v", params, want)
	}
}

func TestFrameFilterParamsFromStrengthRejectsInvalidInputs(t *testing.T) {
	tests := []struct {
		name       string
		plane      Plane
		xDec       int
		yDec       int
		packed     uint8
		damping    int
		coeffShift int
	}{
		{name: "bad-plane", plane: Plane(3), xDec: 0, yDec: 0, damping: 3},
		{name: "bad-xdec", plane: PlaneY, xDec: 2, yDec: 0, damping: 3},
		{name: "bad-ydec", plane: PlaneY, xDec: 0, yDec: -1, damping: 3},
		{name: "bad-packed", plane: PlaneY, xDec: 0, yDec: 0, packed: 64, damping: 3},
		{name: "bad-damping", plane: PlaneY, xDec: 0, yDec: 0, damping: -1},
		{name: "bad-shift", plane: PlaneY, xDec: 0, yDec: 0, damping: 3, coeffShift: 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := FrameFilterParamsFromStrength(tt.plane, tt.xDec, tt.yDec, tt.packed, tt.damping, tt.coeffShift)
			if !errors.Is(err, ErrInvalidCDEF) {
				t.Fatalf("err=%v want %v", err, ErrInvalidCDEF)
			}
		})
	}
}

func TestCDEFStrengthHelpersAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		if _, _, err := DecodeStrength(63); err != nil {
			t.Fatal(err)
		}
		if _, err := FrameFilterParamsFromStrength(PlaneV, 1, 1, 63, 6, 1); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("CDEF strength helpers allocated: %f", allocs)
	}
}
