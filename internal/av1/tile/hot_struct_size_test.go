package tile

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
		{name: "TXBContext", got: unsafe.Sizeof(TXBContext{}), max: 2},
		{name: "TXBDecodeRequest", got: unsafe.Sizeof(TXBDecodeRequest{}), max: 32},
		{name: "TXBDecodeResult", got: unsafe.Sizeof(TXBDecodeResult{}), max: 6},
		{name: "EOBResult", got: unsafe.Sizeof(EOBResult{}), max: 6},
		{name: "LumaCoeffBlock", got: unsafe.Sizeof(LumaCoeffBlock{}), max: 80},
		{name: "ChromaCoeffBlock", got: unsafe.Sizeof(ChromaCoeffBlock{}), max: 88},
		{name: "BlockCoeffBlock", got: unsafe.Sizeof(BlockCoeffBlock{}), max: 88},
		{name: "chromaCoeffPlanePrep", got: unsafe.Sizeof(chromaCoeffPlanePrep{}), max: 16},
	}
	for _, tt := range tests {
		t.Logf("%s size=%d max=%d", tt.name, tt.got, tt.max)
		if tt.got > tt.max {
			t.Fatalf("%s size=%d max=%d", tt.name, tt.got, tt.max)
		}
	}
}
