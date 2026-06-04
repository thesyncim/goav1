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
		{name: "coeffGeometry", size: unsafe.Sizeof(coeffGeometry{}), max: 12},
		{name: "coeffPos", size: unsafe.Sizeof(coeffPos{}), max: 6},
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
		{name: "InterMotionResult", size: unsafe.Sizeof(InterMotionResult{}), max: 16},
		{name: "MVComponentResult", size: unsafe.Sizeof(MVComponentResult{}), max: 10},
		{name: "MVResidualResult", size: unsafe.Sizeof(MVResidualResult{}), max: 28},
		{name: "ReferenceMVCandidate", size: unsafe.Sizeof(ReferenceMVCandidate{}), max: 10},
		{name: "ReferenceMVStack", size: unsafe.Sizeof(ReferenceMVStack{}), max: 92},
		{name: "InterMVReferenceSet", size: unsafe.Sizeof(InterMVReferenceSet{}), max: 24},
		{name: "ReferenceMVEntry", size: unsafe.Sizeof(ReferenceMVEntry{}), max: 8},
		{name: "TemporalMotionEntry", size: unsafe.Sizeof(TemporalMotionEntry{}), max: 6},
		{name: "TemporalMotionSetupStats", size: unsafe.Sizeof(TemporalMotionSetupStats{}), max: 3},
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
