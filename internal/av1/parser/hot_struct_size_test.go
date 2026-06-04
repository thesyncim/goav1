package parser

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
		{name: "FrameHeaderPrefix", size: unsafe.Sizeof(FrameHeaderPrefix{}), max: 32},
		{name: "FrameSize", size: unsafe.Sizeof(FrameSize{}), max: 200},
		{name: "ReferenceFrame", size: unsafe.Sizeof(ReferenceFrame{}), max: 720},
		{name: "ReferenceState", size: unsafe.Sizeof(ReferenceState{}), max: 5760},
		{name: "shortRefInfo", size: unsafe.Sizeof(shortRefInfo{}), max: 4},
		{name: "TileGroup", size: unsafe.Sizeof(TileGroup{}), max: 40},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
