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
		{name: "LumaCoeffTreeScratch", size: unsafe.Sizeof(LumaCoeffTreeScratch{}), max: 9492},
		{name: "TXBDecodeRequest", size: unsafe.Sizeof(TXBDecodeRequest{}), max: 56},
		{name: "BlockVisit", size: unsafe.Sizeof(BlockVisit{}), max: 28},
		{name: "TransformTreeRequest", size: unsafe.Sizeof(TransformTreeRequest{}), max: 24},
		{name: "SelectedTransformRequest", size: unsafe.Sizeof(SelectedTransformRequest{}), max: 7},
		{name: "TransformPartitionRequest", size: unsafe.Sizeof(TransformPartitionRequest{}), max: 7},
		{name: "BlockModeRequest", size: unsafe.Sizeof(BlockModeRequest{}), max: 80},
		{name: "IntraFlagRequest", size: unsafe.Sizeof(IntraFlagRequest{}), max: 22},
		{name: "LumaIntraModeRequest", size: unsafe.Sizeof(LumaIntraModeRequest{}), max: 4},
		{name: "InterReferenceRequest", size: unsafe.Sizeof(InterReferenceRequest{}), max: 24},
		{name: "InterpFilterRequest", size: unsafe.Sizeof(InterpFilterRequest{}), max: 18},
		{name: "CompoundBlendRequest", size: unsafe.Sizeof(CompoundBlendRequest{}), max: 24},
		{name: "OverlappableNeighborRequest", size: unsafe.Sizeof(OverlappableNeighborRequest{}), max: 8},
		{name: "PaletteModeRequest", size: unsafe.Sizeof(PaletteModeRequest{}), max: 24},
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
