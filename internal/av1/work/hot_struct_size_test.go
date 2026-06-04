package work

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
		{name: "TilePlan", size: unsafe.Sizeof(TilePlan{}), max: 24},
		{name: "FramePlan", size: unsafe.Sizeof(FramePlan{}), max: 40},
		{name: "FrameTilePlan", size: unsafe.Sizeof(FrameTilePlan{}), max: 40},
		{name: "ShowExistingFramePlan", size: unsafe.Sizeof(ShowExistingFramePlan{}), max: 16},
		{name: "FrameStep", size: unsafe.Sizeof(FrameStep{}), max: 104},
		{name: "FrameStepResult", size: unsafe.Sizeof(FrameStepResult{}), max: 3},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
