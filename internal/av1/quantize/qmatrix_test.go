package quantize

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestQMLevelMatchesLibaomRegular(t *testing.T) {
	q1 := QMLevel(1, DefaultQMFirst, DefaultQMLast)
	q60 := QMLevel(60, DefaultQMFirst, DefaultQMLast)
	q120 := QMLevel(120, DefaultQMFirst, DefaultQMLast)
	q180 := QMLevel(180, DefaultQMFirst, DefaultQMLast)
	q255 := QMLevel(255, DefaultQMFirst, DefaultQMLast)

	if q1 != DefaultQMFirst {
		t.Fatalf("q1=%d want %d", q1, DefaultQMFirst)
	}
	if q255 != DefaultQMLast {
		t.Fatalf("q255=%d want %d", q255, DefaultQMLast)
	}
	if !(q255 >= q180 && q180 >= q120 && q120 >= q60 && q60 >= q1 && q255 > q1) {
		t.Fatalf("regular qm levels not monotonic: %d %d %d %d %d", q1, q60, q120, q180, q255)
	}

	belowFirst1 := QMLevel(1, 1, DefaultQMFirst-1)
	belowFirst255 := QMLevel(255, 1, DefaultQMFirst-1)
	if belowFirst1 >= DefaultQMFirst || belowFirst255 >= DefaultQMFirst {
		t.Fatalf("below-first levels=%d/%d first=%d", belowFirst1, belowFirst255, DefaultQMFirst)
	}

	aboveLast1 := QMLevel(1, DefaultQMLast+1, QMLevels-1)
	aboveLast255 := QMLevel(255, DefaultQMLast+1, QMLevels-1)
	if aboveLast1 <= DefaultQMLast || aboveLast255 <= DefaultQMLast {
		t.Fatalf("above-last levels=%d/%d last=%d", aboveLast1, aboveLast255, DefaultQMLast)
	}
}

func TestQMLevelMatchesLibaomAllIntra(t *testing.T) {
	q1 := QMLevelAllIntra(1, DefaultQMFirstAllIntra, DefaultQMLastAllIntra)
	q60 := QMLevelAllIntra(60, DefaultQMFirstAllIntra, DefaultQMLastAllIntra)
	q120 := QMLevelAllIntra(120, DefaultQMFirstAllIntra, DefaultQMLastAllIntra)
	q180 := QMLevelAllIntra(180, DefaultQMFirstAllIntra, DefaultQMLastAllIntra)
	q255 := QMLevelAllIntra(255, DefaultQMFirstAllIntra, DefaultQMLastAllIntra)

	if q1 != DefaultQMLastAllIntra {
		t.Fatalf("q1=%d want %d", q1, DefaultQMLastAllIntra)
	}
	if q255 != DefaultQMFirstAllIntra {
		t.Fatalf("q255=%d want %d", q255, DefaultQMFirstAllIntra)
	}
	if !(q255 <= q180 && q180 <= q120 && q120 <= q60 && q60 <= q1 && q255 < q1) {
		t.Fatalf("all-intra qm levels not monotonic: %d %d %d %d %d", q1, q60, q120, q180, q255)
	}

	belowFirst1 := QMLevelAllIntra(1, 1, DefaultQMFirstAllIntra-1)
	belowFirst255 := QMLevelAllIntra(255, 1, DefaultQMFirstAllIntra-1)
	if belowFirst1 != DefaultQMFirstAllIntra-1 || belowFirst255 != DefaultQMFirstAllIntra-1 {
		t.Fatalf("below-first levels=%d/%d want %d", belowFirst1, belowFirst255, DefaultQMFirstAllIntra-1)
	}

	aboveLast1 := QMLevelAllIntra(1, DefaultQMLastAllIntra+1, QMLevels-1)
	aboveLast255 := QMLevelAllIntra(255, DefaultQMLastAllIntra+1, QMLevels-1)
	if aboveLast1 != DefaultQMLastAllIntra+1 || aboveLast255 != DefaultQMLastAllIntra+1 {
		t.Fatalf("above-last levels=%d/%d want %d", aboveLast1, aboveLast255, DefaultQMLastAllIntra+1)
	}
}

func TestQMLevelAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		if QMLevel(255, DefaultQMFirst, DefaultQMLast) != DefaultQMLast {
			t.Fatal("bad regular level")
		}
		if QMLevelAllIntra(255, DefaultQMFirstAllIntra, DefaultQMLastAllIntra) != DefaultQMFirstAllIntra {
			t.Fatal("bad all-intra level")
		}
	})
	if allocs != 0 {
		t.Fatalf("QM level helpers allocated: %f", allocs)
	}
}

func TestInverseQMatrixAllocs(t *testing.T) {
	params := parser.QuantizationParams{
		UsingQMatrix:  true,
		QMatrixLevelY: 0,
	}
	allocs := testing.AllocsPerRun(1000, func() {
		m, err := InverseQMatrix(params, PlaneY, transform.Size{Width: 32, Height: 32}, transform.TypeDCTDCT, false)
		if err != nil {
			t.Fatal(err)
		}
		if len(m) != 32*32 {
			t.Fatalf("matrix len=%d want %d", len(m), 32*32)
		}
	})
	if allocs != 0 {
		t.Fatalf("inverse qmatrix lookup allocated: %f", allocs)
	}
}

func TestInverseQMatrixMatchesLibaomTables(t *testing.T) {
	params := parser.QuantizationParams{
		UsingQMatrix:  true,
		QMatrixLevelY: 0,
		QMatrixLevelU: 0,
		QMatrixLevelV: 14,
	}
	y, err := InverseQMatrix(params, PlaneY, transform.Size{Width: 4, Height: 4}, transform.TypeDCTDCT, false)
	if err != nil {
		t.Fatal(err)
	}
	wantY := []uint16{32, 43, 73, 97, 43, 67, 94, 110, 73, 94, 137, 150, 97, 110, 150, 200}
	for i, want := range wantY {
		if y[i] != want {
			t.Fatalf("y[%d]=%d want %d", i, y[i], want)
		}
	}
	u, err := InverseQMatrix(params, PlaneU, transform.Size{Width: 4, Height: 4}, transform.TypeDCTDCT, false)
	if err != nil {
		t.Fatal(err)
	}
	wantU := []uint16{35, 46, 57, 66, 46, 60, 69, 71, 57, 69, 90, 90, 66, 71, 90, 109}
	for i, want := range wantU {
		if u[i] != want {
			t.Fatalf("u[%d]=%d want %d", i, u[i], want)
		}
	}
	v, err := InverseQMatrix(params, PlaneV, transform.Size{Width: 64, Height: 64}, transform.TypeDCTDCT, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 32*32 || v[0] != 32 || v[5] != 31 {
		t.Fatalf("adjusted 64x64 matrix len=%d first=%d sixth=%d", len(v), v[0], v[5])
	}
}

func TestInverseQMatrixFlatPaths(t *testing.T) {
	params := parser.QuantizationParams{UsingQMatrix: true, QMatrixLevelY: 0}
	tests := []struct {
		name     string
		params   parser.QuantizationParams
		typ      transform.Type
		lossless bool
	}{
		{name: "disabled", params: parser.QuantizationParams{}, typ: transform.TypeDCTDCT},
		{name: "level15", params: parser.QuantizationParams{UsingQMatrix: true, QMatrixLevelY: QMLevels - 1}, typ: transform.TypeDCTDCT},
		{name: "idtx", params: params, typ: transform.TypeIDTX},
		{name: "one-dimensional", params: params, typ: transform.TypeHDCT},
		{name: "lossless", params: params, typ: transform.TypeDCTDCT, lossless: true},
	}
	for _, tt := range tests {
		got, err := InverseQMatrix(tt.params, PlaneY, transform.Size{Width: 4, Height: 4}, tt.typ, tt.lossless)
		if err != nil {
			t.Fatalf("%s err=%v", tt.name, err)
		}
		if got != nil {
			t.Fatalf("%s matrix=%v want nil", tt.name, got)
		}
	}
	if _, err := InverseQMatrix(params, Plane(99), transform.Size{Width: 4, Height: 4}, transform.TypeDCTDCT, false); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid plane err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := InverseQMatrix(params, PlaneY, transform.Size{Width: 4, Height: 12}, transform.TypeDCTDCT, false); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid size err=%v want %v", err, ErrInvalidQuantizer)
	}
}
