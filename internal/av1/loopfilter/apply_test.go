package loopfilter

import (
	"strconv"
	"testing"
)

// TestFilterEdgeByWidthAllocs covers the lower-level FilterEdgeByWidth used by
// callers that already hold a resolved level/thresholds pair.
func TestFilterEdgeByWidthAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, uint16((x*7+y*11)&0xff))
		}
	}
	thresholds, err := ThresholdsForLevel(16, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, width := range []uint8{4, 6, 8, 14} {
		t.Run("width="+strconv.Itoa(int(width)), func(t *testing.T) {
			if err := FilterEdgeByWidth(width, plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
				t.Fatalf("warm-up err=%v", err)
			}
			allocs := testing.AllocsPerRun(1000, func() {
				if err := FilterEdgeByWidth(width, plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Fatalf("FilterEdgeByWidth(%d) allocated: %f", width, allocs)
			}
		})
	}
}
