package transform

import (
	"errors"
	"testing"
)

func TestTypeSupportAndScratchLen(t *testing.T) {
	tests := []struct {
		name        string
		typ         Type
		size        Size
		wantSupport bool
		wantScratch int
	}{
		{name: "dct 4x4", typ: TypeDCTDCT, size: Size{Width: 4, Height: 4}, wantSupport: true, wantScratch: 16},
		{name: "dct 8x8", typ: TypeDCTDCT, size: Size{Width: 8, Height: 8}, wantSupport: true, wantScratch: 64},
		{name: "dct unsupported 16x16", typ: TypeDCTDCT, size: Size{Width: 16, Height: 16}},
		{name: "idtx 4x16", typ: TypeIDTX, size: Size{Width: 4, Height: 16}, wantSupport: true},
		{name: "idtx 32x32", typ: TypeIDTX, size: Size{Width: 32, Height: 32}, wantSupport: true},
		{name: "idtx unsupported 64x64", typ: TypeIDTX, size: Size{Width: 64, Height: 64}},
		{name: "invalid type", typ: Type(99), size: Size{Width: 4, Height: 4}},
	}
	for _, tt := range tests {
		if got := tt.typ.Valid(); got != (tt.typ == TypeDCTDCT || tt.typ == TypeIDTX) {
			t.Fatalf("%s Valid=%t", tt.name, got)
		}
		if got := tt.typ.Supported(tt.size); got != tt.wantSupport {
			t.Fatalf("%s Supported=%t want %t", tt.name, got, tt.wantSupport)
		}
		gotScratch, err := ScratchLenForType(tt.typ, tt.size)
		if !tt.wantSupport {
			if !errors.Is(err, ErrInvalidTransform) {
				t.Fatalf("%s ScratchLenForType err=%v want %v", tt.name, err, ErrInvalidTransform)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s ScratchLenForType err=%v", tt.name, err)
		}
		if gotScratch != tt.wantScratch {
			t.Fatalf("%s scratch=%d want %d", tt.name, gotScratch, tt.wantScratch)
		}
	}
}

func TestSizeFromDimensionsMatchesLibaomTable(t *testing.T) {
	sizes := [...]Size{
		{Width: 4, Height: 4},
		{Width: 8, Height: 8},
		{Width: 16, Height: 16},
		{Width: 32, Height: 32},
		{Width: 64, Height: 64},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 8, Height: 16},
		{Width: 16, Height: 8},
		{Width: 16, Height: 32},
		{Width: 32, Height: 16},
		{Width: 32, Height: 64},
		{Width: 64, Height: 32},
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 32},
		{Width: 32, Height: 8},
		{Width: 16, Height: 64},
		{Width: 64, Height: 16},
	}
	for _, size := range sizes {
		got, err := SizeFromDimensions(size.Width, size.Height)
		if err != nil {
			t.Fatalf("SizeFromDimensions(%d,%d): %v", size.Width, size.Height, err)
		}
		if got != size {
			t.Fatalf("SizeFromDimensions(%d,%d)=%+v", size.Width, size.Height, got)
		}
	}
	if _, err := SizeFromDimensions(4, 12); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid dimensions err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestInverseBlockDCTMatchesDirect(t *testing.T) {
	coeff := []int32{
		100, -20, 30, 7,
		5, -9, 13, -17,
		0, 4, -8, 12,
		-16, 20, -24, 28,
	}
	direct := make([]int16, 4*4)
	dispatched := make([]int16, 4*4)
	scratchDirect := make([]int32, 4*4)
	scratchDispatched := make([]int32, 4*4)
	size := Size{Width: 4, Height: 4}
	if err := InverseDCTBlock(direct, 4, coeff, 4, scratchDirect, size); err != nil {
		t.Fatal(err)
	}
	if err := InverseBlock(dispatched, 4, coeff, 4, scratchDispatched, size, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	for i := range direct {
		if dispatched[i] != direct[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dispatched[i], direct[i])
		}
	}
}

func TestInverseBlockIDTXMatchesDirect(t *testing.T) {
	coeff := make([]int32, 4*8)
	coeff[0] = 64
	coeff[7] = -32
	direct := make([]int16, 4*8)
	dispatched := make([]int16, 4*8)
	size := Size{Width: 4, Height: 8}
	if err := InverseIdentityBlock(direct, 4, coeff, 4, size); err != nil {
		t.Fatal(err)
	}
	if err := InverseBlock(dispatched, 4, coeff, 4, nil, size, TypeIDTX); err != nil {
		t.Fatal(err)
	}
	for i := range direct {
		if dispatched[i] != direct[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dispatched[i], direct[i])
		}
	}
}

func TestInverseBlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int16, 4*4)
	coeff := make([]int32, 4*4)
	scratch := make([]int32, 4*4)
	if err := InverseBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4}, Type(99)); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid type err=%v want %v", err, ErrInvalidTransform)
	}
	if err := InverseBlock(dst, 4, coeff, 4, nil, Size{Width: 4, Height: 4}, TypeDCTDCT); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short dct scratch err=%v want %v", err, ErrInvalidTransform)
	}
	if err := InverseBlock(make([]int16, 64*64), 64, make([]int32, 64*64), 64, nil, Size{Width: 64, Height: 64}, TypeIDTX); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("unsupported idtx err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestInverseBlockAllocs(t *testing.T) {
	dctCoeff := make([]int32, 8*8)
	dctDst := make([]int16, 8*8)
	dctScratch := make([]int32, 8*8)
	idtxCoeff := make([]int32, 16*16)
	idtxDst := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseBlock(dctDst, 8, dctCoeff, 8, dctScratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != nil {
			t.Fatal(err)
		}
		if err := InverseBlock(idtxDst, 16, idtxCoeff, 16, nil, Size{Width: 16, Height: 16}, TypeIDTX); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseBlock allocated: %f", allocs)
	}
}

func FuzzInverseBlock(f *testing.F) {
	f.Add(uint8(0), int16(16), int16(0))
	f.Add(uint8(1), int16(64), int16(-7))
	f.Add(uint8(2), int16(-128), int16(13))
	f.Add(uint8(3), int16(255), int16(-31))

	f.Fuzz(func(t *testing.T, rawMode uint8, coeffValue int16, delta int16) {
		typ := TypeDCTDCT
		size := Size{Width: 4, Height: 4}
		if rawMode&1 == 1 {
			size = Size{Width: 8, Height: 8}
		}
		if rawMode&2 != 0 {
			typ = TypeIDTX
			size = Size{Width: 16, Height: 16}
		}
		coeffStride := size.Width + 3
		dstStride := size.Width + 2
		coeff := make([]int32, coeffStride*size.Height)
		dst := make([]int16, dstStride*size.Height)
		scratchLen, err := ScratchLenForType(typ, size)
		if err != nil {
			t.Fatalf("ScratchLenForType err=%v", err)
		}
		scratch := make([]int32, scratchLen+4)
		for row := 0; row < size.Height; row++ {
			for col := 0; col < size.Width; col++ {
				coeff[row*coeffStride+col] = int32(coeffValue) + int32(delta)*int32((row+col)&3)
			}
		}
		if err := InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, typ); err != nil {
			t.Fatalf("InverseBlock err=%v", err)
		}
		for row := 0; row < size.Height; row++ {
			for col := size.Width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != 0 {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			if scratch[i] != 0 {
				t.Fatalf("scratch padding[%d]=%d want 0", i, scratch[i])
			}
		}
	})
}

func BenchmarkInverseBlockDCT8x8(b *testing.B) {
	coeff := make([]int32, 8*8)
	dst := make([]int16, 8*8)
	scratch := make([]int32, 8*8)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = InverseBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT)
	}
}

func BenchmarkInverseBlockIDTX16x16(b *testing.B) {
	coeff := make([]int32, 16*16)
	dst := make([]int16, 16*16)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = InverseBlock(dst, 16, coeff, 16, nil, Size{Width: 16, Height: 16}, TypeIDTX)
	}
}
