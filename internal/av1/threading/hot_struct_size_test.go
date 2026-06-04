package threading

import (
	"testing"
	"unsafe"
)

const hotStructKiB = 1024

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Batch", size: unsafe.Sizeof(Batch{}), max: 20},
		{name: "FrameWorkBatch", size: unsafe.Sizeof(FrameWorkBatch{}), max: 1608},
		{name: "FrameWorkFrameContext", size: unsafe.Sizeof(FrameWorkFrameContext{}), max: 1152},
		{name: "FrameWorkPlaneRegion", size: unsafe.Sizeof(FrameWorkPlaneRegion{}), max: 72},
		{name: "FrameWorkTileResidualCDFStorage", size: unsafe.Sizeof(FrameWorkTileResidualCDFStorage{}), max: 56 * hotStructKiB},
		{name: "FrameWorkTileResidualScratch", size: unsafe.Sizeof(FrameWorkTileResidualScratch{}), max: 85120},
		{name: "FrameWorkJobRegion", size: unsafe.Sizeof(FrameWorkJobRegion{}), max: 36},
		{name: "FrameWorkCDEFIndexMap", size: unsafe.Sizeof(FrameWorkCDEFIndexMap{}), max: 56},
		{name: "FrameWorkLoopFilterBlockRecord", size: unsafe.Sizeof(FrameWorkLoopFilterBlockRecord{}), max: 34},
		{name: "FrameWorkLoopFilterMapStats", size: unsafe.Sizeof(FrameWorkLoopFilterMapStats{}), max: 12},
		{name: "FrameWorkTileResidualStats", size: unsafe.Sizeof(FrameWorkTileResidualStats{}), max: 136},
		{name: "poolTask", size: unsafe.Sizeof(poolTask{}), max: 1688},
		{name: "frameWorkReconEvent", size: unsafe.Sizeof(frameWorkReconEvent{}), max: 8},
		{name: "frameWorkReconPaletteBinding", size: unsafe.Sizeof(frameWorkReconPaletteBinding{}), max: 12},
		{name: "frameWorkReconSB", size: unsafe.Sizeof(frameWorkReconSB{}), max: 12},
		{name: "frameWorkBlockCoeffGeometry", size: unsafe.Sizeof(frameWorkBlockCoeffGeometry{}), max: 88},
		{name: "frameWorkJobGeometryCache", size: unsafe.Sizeof(frameWorkJobGeometryCache{}), max: 264},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
