package quantize

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestQMatrixMatchesLibaomTables(t *testing.T) {
	y := QMatrix(0, false, transform.Size{Width: 4, Height: 4})
	wantY := []uint16{32, 43, 73, 97, 43, 67, 94, 110, 73, 94, 137, 150, 97, 110, 150, 200}
	for i, want := range wantY {
		if y[i] != want {
			t.Fatalf("y[%d]=%d want %d", i, y[i], want)
		}
	}
	u := QMatrix(0, true, transform.Size{Width: 4, Height: 4})
	wantU := []uint16{35, 46, 57, 66, 46, 60, 69, 71, 57, 69, 90, 90, 66, 71, 90, 109}
	for i, want := range wantU {
		if u[i] != want {
			t.Fatalf("u[%d]=%d want %d", i, u[i], want)
		}
	}
	v := QMatrix(14, true, transform.Size{Width: 64, Height: 64})
	if len(v) != 32*32 || v[0] != 32 || v[5] != 31 {
		t.Fatalf("adjusted 64x64 matrix len=%d first=%d sixth=%d", len(v), v[0], v[5])
	}
	if m := QMatrix(QMLevels-1, false, transform.Size{Width: 4, Height: 4}); m != nil {
		t.Fatalf("flat level 15 matrix %v", m)
	}
	if m := QMatrix(0, false, transform.Size{Width: 4, Height: 12}); m != nil {
		t.Fatalf("invalid size matrix %v", m)
	}
}
