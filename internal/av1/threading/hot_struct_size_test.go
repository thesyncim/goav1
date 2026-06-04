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
		{name: "FrameWorkBatch", size: unsafe.Sizeof(FrameWorkBatch{}), max: 1784},
		{name: "FrameWorkFrameContext", size: unsafe.Sizeof(FrameWorkFrameContext{}), max: 1304},
		{name: "FrameWorkTileResidualCDFStorage", size: unsafe.Sizeof(FrameWorkTileResidualCDFStorage{}), max: 56 * hotStructKiB},
		{name: "FrameWorkTileResidualScratch", size: unsafe.Sizeof(FrameWorkTileResidualScratch{}), max: 86264},
		{name: "FrameWorkJobRegion", size: unsafe.Sizeof(FrameWorkJobRegion{}), max: 36},
		{name: "FrameWorkLoopFilterBlockRecord", size: unsafe.Sizeof(FrameWorkLoopFilterBlockRecord{}), max: 34},
		{name: "FrameWorkLoopFilterMapStats", size: unsafe.Sizeof(FrameWorkLoopFilterMapStats{}), max: 12},
		{name: "FrameWorkTileResidualStats", size: unsafe.Sizeof(FrameWorkTileResidualStats{}), max: 136},
		{name: "frameWorkReconEvent", size: unsafe.Sizeof(frameWorkReconEvent{}), max: 8},
		{name: "frameWorkJobGeometryCache", size: unsafe.Sizeof(frameWorkJobGeometryCache{}), max: 432},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
