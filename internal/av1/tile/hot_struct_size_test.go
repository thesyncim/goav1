package tile

import (
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "eobGroupStart", size: unsafe.Sizeof(eobGroupStart), max: 24},
		{name: "eobOffsetBits", size: unsafe.Sizeof(eobOffsetBits), max: 12},
		{name: "eobToPosSmall", size: unsafe.Sizeof(eobToPosSmall), max: 33},
		{name: "eobToPosLarge", size: unsafe.Sizeof(eobToPosLarge), max: 17},
		{name: "eobMultiSizeTable", size: unsafe.Sizeof(eobMultiSizeTable), max: 19},
		{name: "LumaCoeffStats", size: unsafe.Sizeof(LumaCoeffStats{}), max: 12},
		{name: "LumaCoeffTreeScratch", size: unsafe.Sizeof(LumaCoeffTreeScratch{}), max: 9492},
		{name: "TXBSkipRequest", size: unsafe.Sizeof(TXBSkipRequest{}), max: 2},
		{name: "CoeffTokenRequest", size: unsafe.Sizeof(CoeffTokenRequest{}), max: 3},
		{name: "TXBDecodeRequest", size: unsafe.Sizeof(TXBDecodeRequest{}), max: 56},
		{name: "BlockDeltaContext", size: unsafe.Sizeof(BlockDeltaContext{}), max: 8},
		{name: "coeffGeometry", size: unsafe.Sizeof(coeffGeometry{}), max: 12},
		{name: "coeffPos", size: unsafe.Sizeof(coeffPos{}), max: 6},
		{name: "coeffUnitWindow", size: unsafe.Sizeof(coeffUnitWindow{}), max: 4},
		{name: "BlockVisit", size: unsafe.Sizeof(BlockVisit{}), max: 18},
		{name: "BlockWalkStats", size: unsafe.Sizeof(BlockWalkStats{}), max: 8},
		{name: "TransformTreeRequest", size: unsafe.Sizeof(TransformTreeRequest{}), max: 24},
		{name: "SelectedTransformRequest", size: unsafe.Sizeof(SelectedTransformRequest{}), max: 7},
		{name: "TransformPartitionRequest", size: unsafe.Sizeof(TransformPartitionRequest{}), max: 7},
		{name: "partitionSymbolCount", size: unsafe.Sizeof(partitionSymbolCount), max: 5},
		{name: "yModeSizeContext", size: unsafe.Sizeof(yModeSizeContext), max: 22},
		{name: "intraModeContext", size: unsafe.Sizeof(intraModeContext), max: 13},
		{name: "compoundModeContextMap", size: unsafe.Sizeof(compoundModeContextMap), max: 15},
		{name: "Job", size: unsafe.Sizeof(Job{}), max: 24},
		{name: "TileSpan", size: unsafe.Sizeof(parser.TileSpan{}), max: 12},
		{name: "BlockLoopRequest", size: unsafe.Sizeof(BlockLoopRequest{}), max: 672},
		{name: "BlockModeRequest", size: unsafe.Sizeof(BlockModeRequest{}), max: 80},
		{name: "BlockPredictionModeResult", size: unsafe.Sizeof(BlockPredictionModeResult{}), max: 784},
		{name: "IntraFlagRequest", size: unsafe.Sizeof(IntraFlagRequest{}), max: 22},
		{name: "LumaIntraModeRequest", size: unsafe.Sizeof(LumaIntraModeRequest{}), max: 4},
		{name: "InterReferenceRequest", size: unsafe.Sizeof(InterReferenceRequest{}), max: 24},
		{name: "InterpFilterRequest", size: unsafe.Sizeof(InterpFilterRequest{}), max: 18},
		{name: "CompoundBlendRequest", size: unsafe.Sizeof(CompoundBlendRequest{}), max: 15},
		{name: "maxOBMCNeighbors", size: unsafe.Sizeof(maxOBMCNeighbors), max: 6},
		{name: "MotionModeRequest", size: unsafe.Sizeof(MotionModeRequest{}), max: 12},
		{name: "OverlappableNeighborRequest", size: unsafe.Sizeof(OverlappableNeighborRequest{}), max: 8},
		{name: "OverlappableNeighbor", size: unsafe.Sizeof(OverlappableNeighbor{}), max: 24},
		{name: "OverlappableNeighborSet", size: unsafe.Sizeof(OverlappableNeighborSet{}), max: 248},
		{name: "PaletteModeRequest", size: unsafe.Sizeof(PaletteModeRequest{}), max: 24},
		{name: "InterMotionResult", size: unsafe.Sizeof(InterMotionResult{}), max: 16},
		{name: "DRLRequest", size: unsafe.Sizeof(DRLRequest{}), max: 7},
		{name: "MVComponentResult", size: unsafe.Sizeof(MVComponentResult{}), max: 10},
		{name: "MVResidualResult", size: unsafe.Sizeof(MVResidualResult{}), max: 28},
		{name: "ReferenceMVCandidate", size: unsafe.Sizeof(ReferenceMVCandidate{}), max: 10},
		{name: "ReferenceMVStack", size: unsafe.Sizeof(ReferenceMVStack{}), max: 92},
		{name: "ReferenceMVStackRequest", size: unsafe.Sizeof(ReferenceMVStackRequest{}), max: 64},
		{name: "ReferenceMVStackResult", size: unsafe.Sizeof(ReferenceMVStackResult{}), max: 98},
		{name: "referenceMVStackSearch", size: unsafe.Sizeof(referenceMVStackSearch{}), max: 96},
		{name: "InterMVReferenceSet", size: unsafe.Sizeof(InterMVReferenceSet{}), max: 24},
		{name: "compoundReferenceLists", size: unsafe.Sizeof(compoundReferenceLists{}), max: 36},
		{name: "ReferenceMVFrame", size: unsafe.Sizeof(ReferenceMVFrame{}), max: 32},
		{name: "ReferenceMVEntry", size: unsafe.Sizeof(ReferenceMVEntry{}), max: 8},
		{name: "TemporalMotionField", size: unsafe.Sizeof(TemporalMotionField{}), max: 32},
		{name: "TemporalMotionEntry", size: unsafe.Sizeof(TemporalMotionEntry{}), max: 6},
		{name: "TemporalMotionSetupStats", size: unsafe.Sizeof(TemporalMotionSetupStats{}), max: 3},
		{name: "SubChromaInterCell", size: unsafe.Sizeof(SubChromaInterCell{}), max: 12},
		{name: "SubChromaInterResult", size: unsafe.Sizeof(SubChromaInterResult{}), max: 50},
		{name: "warpSample", size: unsafe.Sizeof(warpSample{}), max: 12},
		{name: "SGRProjInfo", size: unsafe.Sizeof(SGRProjInfo{}), max: 3},
		{name: "RestorationReferences", size: unsafe.Sizeof(RestorationReferences{}), max: 36},
		{name: "RestorationUnit", size: unsafe.Sizeof(RestorationUnit{}), max: 38},
		{name: "RestorationUnitRecord", size: unsafe.Sizeof(RestorationUnitRecord{}), max: 64},
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

func TestWarpSampleCoordinateStorageBounds(t *testing.T) {
	const (
		minInt16 = -1 << 15
		maxInt16 = 1<<15 - 1
		// recordWarpSample stores SB-relative sample positions in Q3 units. The
		// largest local offset comes from a 128x128 block's top-right sample:
		// col_offset=32 MI plus half a 128-wide neighbor minus one sample.
		maxLocal = (MaxBlockModeSlots*4 + MaxBlockModeSlots*2 - 1) * motion.SubpelScale
		minRef   = -maxLocal + (MVLower + 1)
		maxRef   = maxLocal + (MVUpper - 1)
	)
	if minRef < minInt16 || maxRef > maxInt16 {
		t.Fatalf("warpSample int16 coordinate bounds [%d,%d] do not cover refs [%d,%d]", minInt16, maxInt16, minRef, maxRef)
	}
}
