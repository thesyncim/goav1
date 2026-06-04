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
		{name: "FrameWorkBatch", size: unsafe.Sizeof(FrameWorkBatch{}), max: 2008},
		{name: "FrameWorkFrameContext", size: unsafe.Sizeof(FrameWorkFrameContext{}), max: 1536},
		{name: "FrameWorkTileResidualCDFStorage", size: unsafe.Sizeof(FrameWorkTileResidualCDFStorage{}), max: 56 * hotStructKiB},
		{name: "FrameWorkTileResidualScratch", size: unsafe.Sizeof(FrameWorkTileResidualScratch{}), max: 86752},
		{name: "FrameWorkLoopFilterBlockRecord", size: unsafe.Sizeof(FrameWorkLoopFilterBlockRecord{}), max: 44},
		{name: "FrameWorkLoopFilterMapStats", size: unsafe.Sizeof(FrameWorkLoopFilterMapStats{}), max: 12},
		{name: "FrameWorkTileResidualStats", size: unsafe.Sizeof(FrameWorkTileResidualStats{}), max: 136},
		{name: "frameWorkReconEvent", size: unsafe.Sizeof(frameWorkReconEvent{}), max: 8},
		{name: "frameWorkJobGeometryCache", size: unsafe.Sizeof(frameWorkJobGeometryCache{}), max: 440},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
