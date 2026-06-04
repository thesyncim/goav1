package motion

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
		{name: "Vector", size: unsafe.Sizeof(Vector{}), max: 4},
		{name: "ScaleFactors", size: unsafe.Sizeof(ScaleFactors{}), max: 10},
		{name: "CompoundConvBuf", size: unsafe.Sizeof(CompoundConvBuf{}), max: compoundMaxConvSamples*2 + 2},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
