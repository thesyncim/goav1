package reconstruct

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
		{name: "Block", size: unsafe.Sizeof(Block{}), max: 40},
	}
	for _, tt := range tests {
		t.Logf("%s size=%d max=%d", tt.name, tt.size, tt.max)
		if tt.size > tt.max {
			t.Fatalf("%s grew to %d bytes; max %d", tt.name, tt.size, tt.max)
		}
	}
}
