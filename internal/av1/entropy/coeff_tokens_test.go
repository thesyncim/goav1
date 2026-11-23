package entropy

import (
	"testing"
	"unsafe"
)

func TestCoeffScanContextLayout(t *testing.T) {
	var p CoeffScanContext
	if unsafe.Sizeof(p) != 8 || unsafe.Offsetof(p.Pos) != 0 || unsafe.Offsetof(p.Padded) != 2 || unsafe.Offsetof(p.Lower2DOffset) != 4 || unsafe.Offsetof(p.BR2DOffset) != 5 || unsafe.Offsetof(p.LowerEOBCtx) != 6 || unsafe.Offsetof(p.BREOBCtx) != 7 {
		t.Fatal("coefficient scan descriptor layout changed")
	}
}
