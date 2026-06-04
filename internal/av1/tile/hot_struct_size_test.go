package tile

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
		{name: "LumaCoeffStats", size: unsafe.Sizeof(LumaCoeffStats{}), max: 16},
		{name: "LumaCoeffTreeScratch", size: unsafe.Sizeof(LumaCoeffTreeScratch{}), max: 10 * 1024},
		{name: "BlockVisit", size: unsafe.Sizeof(BlockVisit{}), max: 28},
		{name: "TransformTreeRequest", size: unsafe.Sizeof(TransformTreeRequest{}), max: 24},
		{name: "SelectedTransformRequest", size: unsafe.Sizeof(SelectedTransformRequest{}), max: 7},
		{name: "TransformPartitionRequest", size: unsafe.Sizeof(TransformPartitionRequest{}), max: 7},
		{name: "BlockCoeffBlock", size: unsafe.Sizeof(BlockCoeffBlock{}), max: 64},
		{name: "BlockLoopStats", size: unsafe.Sizeof(BlockLoopStats{}), max: 100},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
