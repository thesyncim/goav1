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
		wantValid   bool
		wantSupport bool
		wantScratch int
	}{
		{name: "dct 4x4", typ: TypeDCTDCT, size: Size{Width: 4, Height: 4}, wantValid: true, wantSupport: true, wantScratch: 16},
		{name: "dct 8x8", typ: TypeDCTDCT, size: Size{Width: 8, Height: 8}, wantValid: true, wantSupport: true, wantScratch: 64},
		{name: "dct 16x16", typ: TypeDCTDCT, size: Size{Width: 16, Height: 16}, wantValid: true, wantSupport: true, wantScratch: 256},
		{name: "dct 32x16", typ: TypeDCTDCT, size: Size{Width: 32, Height: 16}, wantValid: true, wantSupport: true, wantScratch: 512},
		{name: "dct 64x64", typ: TypeDCTDCT, size: Size{Width: 64, Height: 64}, wantValid: true, wantSupport: true, wantScratch: 4096},
		{name: "idtx 4x16", typ: TypeIDTX, size: Size{Width: 4, Height: 16}, wantValid: true, wantSupport: true},
		{name: "idtx 32x32", typ: TypeIDTX, size: Size{Width: 32, Height: 32}, wantValid: true, wantSupport: true},
		{name: "idtx unsupported 64x64", typ: TypeIDTX, size: Size{Width: 64, Height: 64}, wantValid: true},
		{name: "adst 4x4", typ: TypeADSTDCT, size: Size{Width: 4, Height: 4}, wantValid: true, wantSupport: true, wantScratch: 16},
		{name: "flipadst 16x8", typ: TypeFlipADSTADST, size: Size{Width: 16, Height: 8}, wantValid: true, wantSupport: true, wantScratch: 128},
		{name: "adst unsupported horizontal 32", typ: TypeDCTADST, size: Size{Width: 32, Height: 16}, wantValid: true},
		{name: "adst unsupported vertical 32", typ: TypeADSTDCT, size: Size{Width: 16, Height: 32}, wantValid: true},
		{name: "invalid type", typ: Type(99), size: Size{Width: 4, Height: 4}},
	}
	for _, tt := range tests {
		if got := tt.typ.Valid(); got != tt.wantValid {
			t.Fatalf("%s Valid=%t want %t", tt.name, got, tt.wantValid)
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
		got, err := SizeFromDimensions(int(size.Width), int(size.Height))
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
	coeff = av1CoeffOrder(coeff, 4, 4)
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

func TestInverseDCTDCOnlyBlockBitDepthMatchesFullInverse(t *testing.T) {
	sizes := []Size{
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
	dcs := []int32{-8192, -257, -1, 0, 1, 255, 8191}
	for _, size := range sizes {
		if !TypeDCTDCT.Supported(size) {
			continue
		}
		for _, bitDepth := range []uint8{8, 10, 12} {
			for _, dc := range dcs {
				width := int(size.Width)
				height := int(size.Height)
				coeff := make([]int32, width*height)
				coeff[0] = dc
				want := make([]int16, width*height)
				got := make([]int16, width*height)
				fullScratch := make([]int32, width*height)
				dcScratch := make([]int32, width+height)
				if err := InverseBlockBitDepth(want, width, coeff, height, fullScratch, size, TypeDCTDCT, bitDepth); err != nil {
					t.Fatalf("full size=%+v bd=%d dc=%d: %v", size, bitDepth, dc, err)
				}
				if err := InverseDCTDCOnlyBlockBitDepth(got, width, dc, dcScratch, size, bitDepth); err != nil {
					t.Fatalf("dc size=%+v bd=%d dc=%d: %v", size, bitDepth, dc, err)
				}
				sample, err := InverseDCTDCOnlySampleBitDepth(dc, dcScratch, size, bitDepth)
				if err != nil {
					t.Fatalf("dc sample size=%+v bd=%d dc=%d: %v", size, bitDepth, dc, err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("size=%+v bd=%d dc=%d dst[%d]=%d want %d", size, bitDepth, dc, i, got[i], want[i])
					}
					if got[i] != sample {
						t.Fatalf("size=%+v bd=%d dc=%d dst[%d]=%d sample %d", size, bitDepth, dc, i, got[i], sample)
					}
				}
			}
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
	if err := InverseIdentityBlock(direct, 4, coeff, 8, size); err != nil {
		t.Fatal(err)
	}
	if err := InverseBlock(dispatched, 4, coeff, 8, nil, size, TypeIDTX); err != nil {
		t.Fatal(err)
	}
	for i := range direct {
		if dispatched[i] != direct[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dispatched[i], direct[i])
		}
	}
}

func TestInverseBlockHybridTransforms(t *testing.T) {
	tests := []struct {
		name string
		typ  Type
		size Size
	}{
		{name: "adst dct", typ: TypeADSTDCT, size: Size{Width: 16, Height: 8}},
		{name: "dct adst", typ: TypeDCTADST, size: Size{Width: 8, Height: 16}},
		{name: "adst adst", typ: TypeADSTADST, size: Size{Width: 16, Height: 16}},
		{name: "flip flip", typ: TypeFlipADSTFlipADST, size: Size{Width: 8, Height: 8}},
		{name: "v adst", typ: TypeVADST, size: Size{Width: 4, Height: 16}},
		{name: "h flip", typ: TypeHFlipADST, size: Size{Width: 16, Height: 4}},
		{name: "v dct", typ: TypeVDCT, size: Size{Width: 32, Height: 16}},
		{name: "h dct", typ: TypeHDCT, size: Size{Width: 16, Height: 32}},
	}
	for _, tt := range tests {
		width := int(tt.size.Width)
		height := int(tt.size.Height)
		coeffStride := height + 1
		dstStride := width + 2
		coeff := make([]int32, coeffStride*width)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[col*coeffStride+row] = int32(((row*11+col*19+7)%43)-21) * 2
			}
		}
		dst := make([]int16, dstStride*height)
		scratchLen, err := ScratchLenForType(tt.typ, tt.size)
		if err != nil {
			t.Fatalf("%s ScratchLenForType err=%v", tt.name, err)
		}
		scratch := make([]int32, scratchLen+3)
		if err := InverseBlock(dst, dstStride, coeff, coeffStride, scratch, tt.size, tt.typ); err != nil {
			t.Fatalf("%s InverseBlock err=%v", tt.name, err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != 0 {
					t.Fatalf("%s dst padding row=%d col=%d overwritten with %d", tt.name, row, col, got)
				}
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			if scratch[i] != 0 {
				t.Fatalf("%s scratch padding[%d]=%d want 0", tt.name, i, scratch[i])
			}
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
	dctCoeff := make([]int32, 32*32)
	dctDst := make([]int16, 32*32)
	dctScratch := make([]int32, 32*32)
	idtxCoeff := make([]int32, 16*16)
	idtxDst := make([]int16, 16*16)
	hybridCoeff := make([]int32, 16*16)
	hybridDst := make([]int16, 16*16)
	hybridScratch := make([]int32, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseBlock(dctDst, 32, dctCoeff, 32, dctScratch, Size{Width: 32, Height: 32}, TypeDCTDCT); err != nil {
			t.Fatal(err)
		}
		if err := InverseBlock(idtxDst, 16, idtxCoeff, 16, nil, Size{Width: 16, Height: 16}, TypeIDTX); err != nil {
			t.Fatal(err)
		}
		if err := InverseBlock(hybridDst, 16, hybridCoeff, 16, hybridScratch, Size{Width: 16, Height: 16}, TypeADSTFlipADST); err != nil {
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
	f.Add(uint8(8), int16(32), int16(5))
	f.Add(uint8(24), int16(-64), int16(11))

	f.Fuzz(func(t *testing.T, rawMode uint8, coeffValue int16, delta int16) {
		typ := TypeDCTDCT
		size := Size{Width: 4, Height: 4}
		if rawMode&8 != 0 {
			hybrids := [...]Type{
				TypeADSTDCT,
				TypeDCTADST,
				TypeADSTADST,
				TypeFlipADSTDCT,
				TypeDCTFlipADST,
				TypeADSTFlipADST,
				TypeFlipADSTADST,
				TypeHFlipADST,
			}
			typ = hybrids[(rawMode>>4)&7]
			switch rawMode & 3 {
			case 1:
				size = Size{Width: 8, Height: 8}
			default:
				size = Size{Width: 16, Height: 16}
			}
		} else if rawMode&4 != 0 {
			typ = TypeIDTX
			size = Size{Width: 16, Height: 16}
		} else {
			switch rawMode & 3 {
			case 1:
				size = Size{Width: 8, Height: 8}
			case 2:
				size = Size{Width: 16, Height: 16}
			case 3:
				size = Size{Width: 32, Height: 32}
			}
		}
		width := int(size.Width)
		height := int(size.Height)
		coeffStride := height + 3
		dstStride := width + 2
		coeff := make([]int32, coeffStride*width)
		dst := make([]int16, dstStride*height)
		scratchLen, err := ScratchLenForType(typ, size)
		if err != nil {
			t.Fatalf("ScratchLenForType err=%v", err)
		}
		scratch := make([]int32, scratchLen+4)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[col*coeffStride+row] = int32(coeffValue) + int32(delta)*int32((row+col)&3)
			}
		}
		if err := InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, typ); err != nil {
			t.Fatalf("InverseBlock err=%v", err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
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
	for b.Loop() {
		_ = InverseBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT)
	}
}

func BenchmarkInverseBlockIDTX16x16(b *testing.B) {
	coeff := make([]int32, 16*16)
	dst := make([]int16, 16*16)
	b.ReportAllocs()
	for b.Loop() {
		_ = InverseBlock(dst, 16, coeff, 16, nil, Size{Width: 16, Height: 16}, TypeIDTX)
	}
}
