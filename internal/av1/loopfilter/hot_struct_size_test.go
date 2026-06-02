package loopfilter

import (
	"testing"
	"unsafe"
)

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		got  uintptr
		max  uintptr
	}{
		{name: "LevelRequest", got: unsafe.Sizeof(LevelRequest{}), max: 6},
		{name: "FilterEdgeRequest", got: unsafe.Sizeof(FilterEdgeRequest{}), max: 20},
		{name: "FilterBlockRequest", got: unsafe.Sizeof(FilterBlockRequest{}), max: 24},
	}
	for _, tt := range tests {
		t.Logf("%s size=%d max=%d", tt.name, tt.got, tt.max)
		if tt.got > tt.max {
			t.Fatalf("%s grew to %d bytes; max %d", tt.name, tt.got, tt.max)
		}
	}
}
