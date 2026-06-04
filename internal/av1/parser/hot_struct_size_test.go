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
		{name: "FrameHeaderPrefix", size: unsafe.Sizeof(FrameHeaderPrefix{}), max: 24},
		{name: "FrameSize", size: unsafe.Sizeof(FrameSize{}), max: 192},
		{name: "ReferenceFrame", size: unsafe.Sizeof(ReferenceFrame{}), max: 696},
		{name: "ReferenceState", size: unsafe.Sizeof(ReferenceState{}), max: 5568},
		{name: "shortRefInfo", size: unsafe.Sizeof(shortRefInfo{}), max: 4},
		{name: "TileGroup", size: unsafe.Sizeof(TileGroup{}), max: 20},
		{name: "TileSpan", size: unsafe.Sizeof(TileSpan{}), max: 12},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}
