package transform

import (
	"errors"
	"testing"
)

func TestTypeClassMatchesLibaomAndDav1d(t *testing.T) {
	tests := []struct {
		name string
		typ  Type
		want Class
	}{
		{name: "DCT_DCT", typ: TypeDCTDCT, want: Class2D},
		{name: "ADST_DCT", typ: TypeADSTDCT, want: Class2D},
		{name: "DCT_ADST", typ: TypeDCTADST, want: Class2D},
		{name: "ADST_ADST", typ: TypeADSTADST, want: Class2D},
		{name: "FLIPADST_DCT", typ: TypeFlipADSTDCT, want: Class2D},
		{name: "DCT_FLIPADST", typ: TypeDCTFlipADST, want: Class2D},
		{name: "FLIPADST_FLIPADST", typ: TypeFlipADSTFlipADST, want: Class2D},
		{name: "ADST_FLIPADST", typ: TypeADSTFlipADST, want: Class2D},
		{name: "FLIPADST_ADST", typ: TypeFlipADSTADST, want: Class2D},
		{name: "IDTX", typ: TypeIDTX, want: Class2D},
		{name: "V_DCT", typ: TypeVDCT, want: ClassVert},
		{name: "H_DCT", typ: TypeHDCT, want: ClassHoriz},
		{name: "V_ADST", typ: TypeVADST, want: ClassVert},
		{name: "H_ADST", typ: TypeHADST, want: ClassHoriz},
		{name: "V_FLIPADST", typ: TypeVFlipADST, want: ClassVert},
		{name: "H_FLIPADST", typ: TypeHFlipADST, want: ClassHoriz},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.typ.Class()
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("Class(%d)=%d want %d", tt.typ, got, tt.want)
			}
		})
	}
}

func TestTypeClassRejectsInvalid(t *testing.T) {
	if TypeCount.Valid() {
		t.Fatal("TypeCount is valid")
	}
	if _, err := TypeCount.Class(); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("bad type err=%v want %v", err, ErrInvalidTransform)
	}
}

func FuzzTypeClass(f *testing.F) {
	f.Add(uint8(TypeDCTDCT))
	f.Add(uint8(TypeVDCT))
	f.Add(uint8(TypeHFlipADST))
	f.Fuzz(func(t *testing.T, raw uint8) {
		typ := Type(raw % uint8(TypeCount))
		class, err := typ.Class()
		if err != nil {
			t.Fatalf("Class err=%v", err)
		}
		if !class.Valid() {
			t.Fatalf("class=%d invalid", class)
		}
	})
}
