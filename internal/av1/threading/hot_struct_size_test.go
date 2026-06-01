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
		{name: "FrameWorkBatch", size: unsafe.Sizeof(FrameWorkBatch{}), max: 2 * hotStructKiB},
		{name: "FrameWorkFrameContext", size: unsafe.Sizeof(FrameWorkFrameContext{}), max: 1536},
		{name: "FrameWorkTileResidualCDFStorage", size: unsafe.Sizeof(FrameWorkTileResidualCDFStorage{}), max: 56 * hotStructKiB},
		{name: "FrameWorkTileResidualScratch", size: unsafe.Sizeof(FrameWorkTileResidualScratch{}), max: 100 * hotStructKiB},
		{name: "frameWorkReconEvent", size: unsafe.Sizeof(frameWorkReconEvent{}), max: 8},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
