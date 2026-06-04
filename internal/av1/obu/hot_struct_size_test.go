package obu

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
		{name: "Unit", size: unsafe.Sizeof(Unit{}), max: 64},
		{name: "AnnexBIterator", size: unsafe.Sizeof(AnnexBIterator{}), max: 56},
		{name: "TemporalUnitIterator", size: unsafe.Sizeof(TemporalUnitIterator{}), max: 32},
		{name: "LowOverheadIterator", size: unsafe.Sizeof(LowOverheadIterator{}), max: 32},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
