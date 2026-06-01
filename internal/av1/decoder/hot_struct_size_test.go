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
		{name: "Event", size: unsafe.Sizeof(Event{}), max: 6 * hotStructKiB},
		{name: "FrameWorkState", size: unsafe.Sizeof(FrameWorkState{}), max: 576 * hotStructKiB},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
