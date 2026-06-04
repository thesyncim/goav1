package cdef

import (
	"testing"
	"unsafe"
)

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "BlockFilterParams", size: unsafe.Sizeof(BlockFilterParams{}), max: 8},
		{name: "FrameFilterParams", size: unsafe.Sizeof(FrameFilterParams{}), max: 7},
		{name: "DirectionGrid", size: unsafe.Sizeof(DirectionGrid{}), max: 256},
		{name: "VarianceGrid", size: unsafe.Sizeof(VarianceGrid{}), max: 1024},
		{name: "cdefDirections", size: unsafe.Sizeof(cdefDirections), max: 48},
		{name: "cdefPrimaryTaps", size: unsafe.Sizeof(cdefPrimaryTaps), max: 4},
		{name: "cdefSecondaryTaps", size: unsafe.Sizeof(cdefSecondaryTaps), max: 2},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
