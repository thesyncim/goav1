package decoder

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
		{name: "Event", size: unsafe.Sizeof(Event{}), max: 5632},
		{name: "FrameWorkState", size: unsafe.Sizeof(FrameWorkState{}), max: 554 * hotStructKiB},
		{name: "FrameWorkLoopFilterPostFilterEdge", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterEdge{}), max: 16},
		{name: "FrameWorkLoopFilterPostFilterLevelStats", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterLevelStats{}), max: 12},
		{name: "FrameWorkLoopFilterPostFilterPlan", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterPlan{}), max: 136},
		{name: "frameWorkLoopFilterPlanningContext", size: unsafe.Sizeof(frameWorkLoopFilterPlanningContext{}), max: 64},
		{name: "FrameWorkSuperResPostFilterPlanePlan", size: unsafe.Sizeof(FrameWorkSuperResPostFilterPlanePlan{}), max: 24},
		{name: "FrameWorkSuperResPostFilterPlan", size: unsafe.Sizeof(FrameWorkSuperResPostFilterPlan{}), max: 160},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
