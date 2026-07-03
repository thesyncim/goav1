package decoder

import (
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/threading"
)

const hotStructKiB = 1024

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Event", size: unsafe.Sizeof(Event{}), max: 5632},
		{name: "FrameWorkState", size: unsafe.Sizeof(FrameWorkState{}), max: 566776},
		{name: "FrameWorkLoopFilterPostFilterEdge", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterEdge{}), max: 16},
		{name: "FrameWorkLoopFilterPostFilterLevelStats", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterLevelStats{}), max: 12},
		{name: "FrameWorkLoopFilterPostFilterPlan", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterPlan{}), max: 136},
		{name: "FrameWorkLoopFilterPostFilterApplyResult", size: unsafe.Sizeof(FrameWorkLoopFilterPostFilterApplyResult{}), max: 176},
		{name: "frameWorkLoopFilterPlanningContext", size: unsafe.Sizeof(frameWorkLoopFilterPlanningContext{}), max: 64},
		{name: "FrameWorkLoopFilterBlockRecord", size: unsafe.Sizeof(threading.FrameWorkLoopFilterBlockRecord{}), max: 34},
		{name: "frameWorkLoopFilterLumaPreviousCache", size: unsafe.Sizeof(frameWorkLoopFilterLumaPreviousCache{}), max: 74},
		{name: "frameWorkLoopFilterChromaPreviousCache", size: unsafe.Sizeof(frameWorkLoopFilterChromaPreviousCache{}), max: 48},
		{name: "FrameWorkCDEFPostFilterResult", size: unsafe.Sizeof(FrameWorkCDEFPostFilterResult{}), max: 12},
		{name: "FrameWorkSuperResPostFilterPlanePlan", size: unsafe.Sizeof(FrameWorkSuperResPostFilterPlanePlan{}), max: 24},
		{name: "FrameWorkSuperResPostFilterPlan", size: unsafe.Sizeof(FrameWorkSuperResPostFilterPlan{}), max: 160},
		{name: "FrameWorkSuperResPostFilterResult", size: unsafe.Sizeof(FrameWorkSuperResPostFilterResult{}), max: 432},
		{name: "FrameWorkFilmGrainPostFilterPlanePlan", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterPlanePlan{}), max: 16},
		{name: "FrameWorkFilmGrainPostFilterPlan", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterPlan{}), max: 264},
		{name: "FrameWorkFilmGrainPostFilterResult", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterResult{}), max: 288},
		{name: "FrameWorkFilmGrainPostFilterLumaGrain", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterLumaGrain{}), max: 32},
		{name: "FrameWorkFilmGrainPostFilterChromaGrain", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterChromaGrain{}), max: 32},
		{name: "FrameWorkFilmGrainPostFilterLumaRow", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterLumaRow{}), max: 16},
		{name: "FrameWorkFilmGrainPostFilterChromaRow", size: unsafe.Sizeof(FrameWorkFilmGrainPostFilterChromaRow{}), max: 20},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
