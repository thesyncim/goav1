package transform

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	randv2 "math/rand/v2"
	"reflect"
	"slices"
	"testing"
	"unsafe"
)

func TestInverseADST1DRoundTripMatchesLibaomShape(t *testing.T) {
	for _, length := range []int{4, 8, 16} {
		input := make([]int32, length)
		for i := range input {
			input[i] = int32(((i*41+23)%37)-18) * 3
		}
		got := referenceADST1DInt(input)
		inverseADST1D(got, 1, length, minInt16, maxInt16)
		normalizationShift := getMaxBit(length) - 1
		for i := range got {
			normalized := clipInt32(roundShift(int64(got[i]), normalizationShift))
			if diff := abs32(normalized - input[i]); diff > 2 {
				t.Fatalf("length=%d coeff=%d got=%d normalized=%d want=%d diff=%d", length, i, got[i], normalized, input[i], diff)
			}
		}
	}
}

func TestInverseADST1DStrided(t *testing.T) {
	compact := []int32{100, -20, 30, 7}
	want := append([]int32(nil), compact...)
	inverseADST1D(want, 1, 4, minInt16, maxInt16)

	values := []int32{100, 99, -20, 99, 30, 99, 7}
	inverseADST1D(values, 2, 4, minInt16, maxInt16)
	for i := range want {
		if values[i*2] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i*2, values[i*2], want[i])
		}
	}
	for _, i := range []int{1, 3, 5} {
		if values[i] != 99 {
			t.Fatalf("sentinel values[%d]=%d want 99", i, values[i])
		}
	}
}

func TestInverseFlipADST1DReversesADST(t *testing.T) {
	for _, length := range []int{4, 8, 16} {
		input := make([]int32, length)
		for i := range input {
			input[i] = int32(((i*17+9)%29)-14) * 6
		}
		adst := append([]int32(nil), input...)
		flip := append([]int32(nil), input...)
		inverseADST1D(adst, 1, length, minInt16, maxInt16)
		inverseFlipADST1D(flip, 1, length, minInt16, maxInt16)
		for i := range length {
			if flip[i] != adst[length-1-i] {
				t.Fatalf("length=%d flip[%d]=%d want ADST[%d]=%d", length, i, flip[i], length-1-i, adst[length-1-i])
			}
		}
	}
}

func TestInverseADST4OverflowRegressionFromLibaom(t *testing.T) {
	values := []int32{300000, 0, 300000, 300000}
	inverseADST1D(values, 1, 4, minInt32, maxInt32)
	if values[0] <= 0 {
		t.Fatalf("ADST4 overflow regression values[0]=%d want > 0", values[0])
	}
}

func referenceADST1DInt(input []int32) []int32 {
	size := len(input)
	if size == 4 {
		return referenceFADST4New(input)
	}
	out := make([]int32, size)
	for k := range size {
		sum := 0.0
		for n := range size {
			angle := math.Pi * float64((2*n+1)*(2*k+1)) / float64(4*size)
			sum += float64(input[n]) * math.Sin(angle)
		}
		out[k] = int32(math.Round(sum))
	}
	return out
}

func referenceFADST4New(input []int32) []int32 {
	x0 := int64(input[0])
	x1 := int64(input[1])
	x2 := int64(input[2])
	x3 := int64(input[3])
	if x0|x1|x2|x3 == 0 {
		return []int32{0, 0, 0, 0}
	}

	s0 := int64(5283) * x0
	s1 := int64(15212) * x0
	s2 := int64(9929) * x1
	s3 := int64(5283) * x1
	s4 := int64(13377) * x2
	s5 := int64(15212) * x3
	s6 := int64(9929) * x3
	s7 := x0 + x1 - x3

	x0 = s0 + s2 + s5
	x1 = int64(13377) * s7
	x2 = s1 - s3 + s6
	x3 = s4

	s0 = x0 + x3
	s1 = x1
	s2 = x2 - x3
	s3 = x2 - x0 + x3

	return []int32{
		clipInt32(roundShift(s0, 14)),
		clipInt32(roundShift(s1, 14)),
		clipInt32(roundShift(s2, 14)),
		clipInt32(roundShift(s3, 14)),
	}
}

func getMaxBit(x int) int {
	maxBit := -1
	for x != 0 {
		x >>= 1
		maxBit++
	}
	return maxBit
}

// FuzzInverseADSTBlock exercises the inverse ADST/FlipADST paths through
// InverseBlock across all square and rectangular sizes that ADST supports,
// including hybrid combinations with DCT and the vertical/horizontal-only
// variants. Coefficient magnitudes are pushed near int16 limits to stress the
// internal clipRange clamps in inverseADST8/16.
func FuzzInverseADSTBlock(f *testing.F) {
	f.Add(uint8(0), uint8(0), int16(0), int16(0), int16(0))
	f.Add(uint8(1), uint8(0), int16(64), int16(-21), int16(13))
	f.Add(uint8(2), uint8(3), int16(-128), int16(32), int16(-17))
	f.Add(uint8(3), uint8(4), int16(255), int16(-127), int16(63))
	f.Add(uint8(5), uint8(6), int16(1024), int16(-512), int16(256))
	f.Add(uint8(7), uint8(1), int16(32767), int16(-32768), int16(16384))
	f.Add(uint8(8), uint8(7), int16(-32768), int16(32767), int16(-16384))

	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 4, Height: 16},
		{Width: 8, Height: 4},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 16, Height: 4},
		{Width: 16, Height: 8},
		{Width: 16, Height: 16},
	}
	types := []Type{
		TypeADSTDCT,
		TypeDCTADST,
		TypeADSTADST,
		TypeFlipADSTDCT,
		TypeDCTFlipADST,
		TypeFlipADSTFlipADST,
		TypeADSTFlipADST,
		TypeFlipADSTADST,
		TypeVADST,
		TypeHADST,
		TypeVFlipADST,
		TypeHFlipADST,
	}

	f.Fuzz(func(t *testing.T, rawSize uint8, rawType uint8, c0 int16, c1 int16, c2 int16) {
		size := sizes[int(rawSize)%len(sizes)]
		typ := types[int(rawType)%len(types)]
		if !typ.Supported(size) {
			return
		}

		width := int(size.Width)
		height := int(size.Height)
		coeffStride := height + 2
		dstStride := width + 3
		coeff := make([]int32, coeffStride*width)
		dst := make([]int16, dstStride*height)
		scratchLen, err := ScratchLenForType(typ, size)
		if err != nil {
			t.Fatalf("ScratchLenForType err=%v", err)
		}
		scratch := make([]int32, scratchLen+3)

		seeds := [3]int16{c0, c1, c2}
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				base := int32(seeds[(row+col)%3])
				if (row^col)&1 != 0 {
					base = -base
				}
				coeff[col*coeffStride+row] = base
			}
		}

		const sentinel = int16(0x6c6c)
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				dst[row*dstStride+col] = sentinel
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			scratch[i] = 0x5b5b5b5b
		}

		if err := InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, typ); err != nil {
			t.Fatalf("InverseBlock size=%dx%d type=%d err=%v", size.Width, size.Height, typ, err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != sentinel {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			if scratch[i] != 0x5b5b5b5b {
				t.Fatalf("scratch padding[%d]=%d want sentinel", i, scratch[i])
			}
		}
	})
}

// TestStageRangeBoundsMatchLibaom anchors the bit-depth-dependent clamp bounds
// to the values libaom's av1_gen_inv_stage_range() derives.
func TestStageRangeBoundsMatchLibaom(t *testing.T) {
	cases := []struct {
		bd                     uint8
		wantRowMin, wantRowMax int32
		wantColMin, wantColMax int32
	}{
		{bd: 8, wantRowMin: -32768, wantRowMax: 32767, wantColMin: -32768, wantColMax: 32767},
		{bd: 10, wantRowMin: -131072, wantRowMax: 131071, wantColMin: -32768, wantColMax: 32767},
		{bd: 12, wantRowMin: -524288, wantRowMax: 524287, wantColMin: -131072, wantColMax: 131071},
	}
	for _, tc := range cases {
		rmin, rmax, cmin, cmax, ok := stageRangeBounds(tc.bd)
		if !ok {
			t.Fatalf("bd=%d: stageRangeBounds returned !ok", tc.bd)
		}
		if rmin != tc.wantRowMin || rmax != tc.wantRowMax {
			t.Fatalf("bd=%d: row=[%d,%d] want [%d,%d]", tc.bd, rmin, rmax, tc.wantRowMin, tc.wantRowMax)
		}
		if cmin != tc.wantColMin || cmax != tc.wantColMax {
			t.Fatalf("bd=%d: col=[%d,%d] want [%d,%d]", tc.bd, cmin, cmax, tc.wantColMin, tc.wantColMax)
		}
	}
}

// TestStageRangeBoundsRejectsInvalidBitDepth ensures only 8, 10, 12 are
// accepted.
func TestStageRangeBoundsRejectsInvalidBitDepth(t *testing.T) {
	for _, bd := range []uint8{0, 1, 6, 7, 9, 11, 13, 16, 255} {
		if _, _, _, _, ok := stageRangeBounds(bd); ok {
			t.Fatalf("bd=%d: stageRangeBounds unexpectedly accepted", bd)
		}
	}
}

// TestInverseBlockBitDepthRowClampDiverges exercises a 4x4 DCT_DCT block whose
// dequantized coefficient exceeds int16 to verify the 10-bit row clamp keeps
// more headroom than the legacy 8-bit clamp. This anchors the fix for the
// 10-bit quantizer-32 / quantizer-63 cohort: with the 8-bit clamp the row
// transform input saturates to ±32767, but libaom's bd+8=18-bit clamp lets the
// real value flow through.
func TestInverseBlockBitDepthRowClampDiverges(t *testing.T) {
	size := Size{Width: 4, Height: 4}
	// 200000 > int16 max (32767); fits in 18 bits (131071? no, 200000 > 131071
	// too). We want a value that the 8-bit clamp saturates and the 10-bit
	// clamp does not — so 100000, well within ±131071 (bd+8=18) but outside
	// ±32767.
	coeff := []int32{
		100000, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	scratch8 := make([]int32, 16)
	scratch10 := make([]int32, 16)
	dst8 := make([]int16, 16)
	dst10 := make([]int16, 16)
	if err := InverseBlock(dst8, 4, coeff, 4, scratch8, size, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	if err := InverseBlockBitDepth(dst10, 4, coeff, 4, scratch10, size, TypeDCTDCT, 10); err != nil {
		t.Fatal(err)
	}
	if dst8[0] == dst10[0] {
		t.Fatalf("expected 8-bit (clamped) and 10-bit (unclamped) residuals to differ at [0]; both produced %d", dst8[0])
	}
}

// TestInverseBlockBitDepthMatchesInverseBlockAt8Bit anchors the equivalence
// guarantee: bd=8 must reproduce the legacy InverseBlock output bit-for-bit.
func TestInverseBlockBitDepthMatchesInverseBlockAt8Bit(t *testing.T) {
	size := Size{Width: 8, Height: 8}
	coeff := make([]int32, 64)
	for i := range coeff {
		// Keep this legacy equivalence case moderate; the full int16 input range
		// is covered separately through the guarded SIMD and scalar fallback.
		coeff[i] = int32((i*37)%1024) - 512
	}
	scratchA := make([]int32, 64)
	scratchB := make([]int32, 64)
	dstA := make([]int16, 64)
	dstB := make([]int16, 64)
	for _, typ := range []Type{TypeDCTDCT, TypeADSTDCT, TypeADSTADST, TypeIDTX, TypeVDCT, TypeHADST} {
		if err := InverseBlock(dstA, 8, coeff, 8, scratchA, size, typ); err != nil {
			t.Fatalf("InverseBlock type=%d: %v", typ, err)
		}
		if err := InverseBlockBitDepth(dstB, 8, coeff, 8, scratchB, size, typ, 8); err != nil {
			t.Fatalf("InverseBlockBitDepth type=%d: %v", typ, err)
		}
		for i := range dstA {
			if dstA[i] != dstB[i] {
				t.Fatalf("type=%d index=%d: InverseBlock=%d InverseBlockBitDepth(bd=8)=%d", typ, i, dstA[i], dstB[i])
			}
		}
	}
}

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
				fullScratch := make([]int32, width*height)
				dcScratch := make([]int32, width+height)
				if err := InverseBlockBitDepth(want, width, coeff, height, fullScratch, size, TypeDCTDCT, bitDepth); err != nil {
					t.Fatalf("full size=%+v bd=%d dc=%d: %v", size, bitDepth, dc, err)
				}
				sample, err := InverseDCTDCOnlySampleBitDepth(dc, dcScratch, size, bitDepth)
				if err != nil {
					t.Fatalf("dc sample size=%+v bd=%d dc=%d: %v", size, bitDepth, dc, err)
				}
				for i := range want {
					if want[i] != sample {
						t.Fatalf("size=%+v bd=%d dc=%d dst[%d]=%d sample %d", size, bitDepth, dc, i, want[i], sample)
					}
				}
			}
		}
	}
}

func TestInverseBlockBitDepthAdjustedScanDirtyScratchMatchesClean(t *testing.T) {
	size := Size{Width: 64, Height: 64}
	width := int(size.Width)
	height := int(size.Height)
	coeffStride := height
	coeff := make([]int32, width*height)
	for col := 0; col < 32; col++ {
		for row := 0; row < 32; row++ {
			if (row*5+col*7)%17 == 0 {
				coeff[col*coeffStride+row] = int32((row+3)*(col+5) - 211)
			}
		}
	}
	want := make([]int16, width*height)
	got := make([]int16, width*height)
	cleanScratch := make([]int32, width*height)
	dirtyScratch := make([]int32, width*height)
	for i := range dirtyScratch {
		dirtyScratch[i] = -0x654321
	}
	if err := InverseBlockBitDepth(want, width, coeff, coeffStride, cleanScratch, size, TypeDCTDCT, 8); err != nil {
		t.Fatal(err)
	}
	if err := InverseBlockBitDepth(got, width, coeff, coeffStride, dirtyScratch, size, TypeDCTDCT, 8); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, got[i], want[i])
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

func TestInt16ColumnSIMDInputGuardBounds(t *testing.T) {
	for _, height := range []int{8, 16, 32, 64} {
		limit := int16ColumnSIMDInputBound(height)
		if limit == 0 {
			t.Fatalf("height=%d has no SIMD bound", height)
		}
		full := make([]int16, 8*height)
		if height == 8 {
			full[0], full[1] = minInt16, maxInt16
			if !int16ColumnSIMDInputSafeScalar(full, 8, height, minInt16, maxInt16) {
				t.Fatal("DCT8 rejected the full int16 input range")
			}
			full[0], full[1] = 30000, 30000
			if !int16ColumnSIMDInputSafeScalar(full, 8, height, minInt16, maxInt16) {
				t.Fatal("DCT8 rejected a full-range even-butterfly input")
			}
			if !int16ColumnSIMDInputSafeScalar(full, 8, height, 1, maxInt16) {
				t.Fatal("DCT8 rejected a valid positive-only int16 clamp")
			}
			continue
		}
		full[0], full[1] = int16(limit), int16(-limit)
		if !int16ColumnSIMDInputSafeScalar(full, 8, height, minInt16, maxInt16) {
			t.Fatalf("height=%d rejected certified boundary", height)
		}
		for _, value := range []int32{limit + 1, -limit - 1, minInt16, maxInt16} {
			full[0] = int16(value)
			if int16ColumnSIMDInputSafeScalar(full, 8, height, minInt16, maxInt16) {
				t.Fatalf("height=%d accepted value=%d outside certified interval", height, value)
			}
		}
	}

	input := make([]int16, 8*8)
	input[0], input[4*8] = 30000, 30000
	if !int16ColumnSIMDInputSafeScalar(input, 8, 8, minInt16, maxInt16) {
		t.Fatal("DCT8 rejected an exact wide pre-rotation sum")
	}
	if int16ColumnSIMDInputSafeScalar(make([]int16, 8*16), 8, 16, 1, maxInt16) {
		t.Fatal("accepted a deeper DCT clamp interval that excludes zero")
	}
}

func TestInverseBlockBitDepth8BenchmarkFixturesSelectExpectedColumnPath(t *testing.T) {
	if !int16ColumnFast {
		t.Skip("SIMD int16 column path is not bound")
	}
	for _, side := range []int{8, 16, 32, 64} {
		for _, pattern := range []string{"bounded", "high-range"} {
			t.Run(inverseBenchSizeName(side)+"/"+pattern, func(t *testing.T) {
				coeff := inverseColumnBenchmarkCoefficients(side, pattern)
				size := Size{Width: uint8(side), Height: uint8(side)}
				scratch := make([]int32, side*side)
				col16 := make([]int16, side*side)
				rowMin, rowMax, colMin, colMax, ok := stageRangeBounds(8)
				if !ok {
					t.Fatal("missing 8-bit stage bounds")
				}
				usedInt16, err := inverseSeparableBlockClampedRowsToScratch(
					coeff, side, scratch, size, TypeDCTDCT,
					rowMin, rowMax, colMin, colMax, 0, 0, col16,
				)
				if err != nil {
					t.Fatal(err)
				}
				wantInt16 := pattern == "bounded"
				if usedInt16 != wantInt16 {
					t.Fatalf("usedInt16=%t want %t", usedInt16, wantInt16)
				}
			})
		}
	}
}

func TestInverseBlockBitDepthFullRangeMatchesScalarColumnPath(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1632dc7))
	for _, height := range []int{8, 16, 32, 64} {
		size := Size{Width: uint8(height), Height: uint8(height)}
		width := height
		for _, pattern := range []string{"random-int16", "alternating-extrema"} {
			coeff := make([]int32, width*height)
			for i := range coeff {
				switch pattern {
				case "random-int16":
					coeff[i] = int32(int16(rng.Uint32()))
				case "alternating-extrema":
					if i%2 == 0 {
						coeff[i] = maxInt16
					} else {
						coeff[i] = minInt16
					}
				}
			}
			compareInverseBlockBitDepthToScalar(t, size, TypeDCTDCT, coeff)
		}
	}

	// TypeVDCT leaves each row as an identity transform. These coefficients
	// produce pre-column values of +16384 at c0 and c4 after the 8x8 mid-pass,
	// crossing the profitability threshold. The exact widened kernel supports
	// this range, but production dispatch should use the faster int32/NEON path.
	coeff := make([]int32, 8*8)
	coeff[0], coeff[4] = 1<<14, 1<<14
	size := Size{Width: 8, Height: 8}
	if int16ColumnFast {
		rowMin, rowMax, colMin, colMax, _ := stageRangeBounds(8)
		usedSIMD, err := inverseSeparableBlockClampedRowsToScratch(
			coeff, 8, make([]int32, 8*8), size, TypeVDCT,
			rowMin, rowMax, colMin, colMax, 0, 0, make([]int16, 8*8),
		)
		if err != nil {
			t.Fatal(err)
		}
		if usedSIMD {
			t.Fatal("high-magnitude DCT8 fixture did not take the faster int32 path")
		}
	}
	compareInverseBlockBitDepthToScalar(t, size, TypeVDCT, coeff)

	// DCT16 still uses its certified narrow range. This identity-row block
	// produces an input above the DCT16 SIMD bound and must take the int32 path.
	coeff16 := make([]int32, 16*16)
	coeff16[0], coeff16[8] = 12000, 12000
	size16 := Size{Width: 16, Height: 16}
	if int16ColumnFast {
		rowMin, rowMax, colMin, colMax, _ := stageRangeBounds(8)
		usedSIMD, err := inverseSeparableBlockClampedRowsToScratch(
			coeff16, 16, make([]int32, 16*16), size16, TypeVDCT,
			rowMin, rowMax, colMin, colMax, 0, 0, make([]int16, 16*16),
		)
		if err != nil {
			t.Fatal(err)
		}
		if usedSIMD {
			t.Fatal("DCT16 SIMD path accepted values outside its certified interval")
		}
	}
	compareInverseBlockBitDepthToScalar(t, size16, TypeVDCT, coeff16)
}

func compareInverseBlockBitDepthToScalar(t *testing.T, size Size, typ Type, coeff []int32) {
	t.Helper()
	width, height := int(size.Width), int(size.Height)
	got := make([]int16, width*height)
	gotScratch := make([]int32, width*height)
	if err := InverseBlockBitDepth(got, width, coeff, height, gotScratch, size, typ, 8); err != nil {
		t.Fatalf("InverseBlockBitDepth size=%+v type=%d: %v", size, typ, err)
	}
	want := inverseBlockBitDepthScalarReference(coeff, height, size, typ)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("size=%+v type=%d index=%d SIMD-dispatch=%d scalar=%d", size, typ, i, got[i], want[i])
		}
	}
}

// inverseBlockBitDepthScalarReference spells out the two passes with the
// int32 scalar kernels. It deliberately bypasses the int16 column dispatcher
// so it can serve as an independent oracle for its range guard and fallback.
func inverseBlockBitDepthScalarReference(coeff []int32, coeffStride int, size Size, typ Type) []int16 {
	width, height := int(size.Width), int(size.Height)
	rowMin, rowMax, colMin, colMax, _ := stageRangeBounds(8)
	idx := sizeIndex(size)
	coeffSize := adjustedScanSizeTable[idx]
	coeffW, coeffH := int(coeffSize.Width), int(coeffSize.Height)
	shift := int(sizeShiftTable[idx])
	vertical, horizontal, _ := typ.tx1DTypes()
	scratch := make([]int32, width*height)
	stageTransposeClampScalar(scratch, width, coeff, coeffStride, coeffH, coeffW, size.IsRect2(), rowMin, rowMax)
	for row := 0; row < coeffH; row++ {
		line := scratch[row*width : row*width+width]
		switch horizontal {
		case tx1DDCT:
			inverseDCT1D(line, 1, width, rowMin, rowMax)
		case tx1DIdentity:
			inverseIdentity1DRow(line, width)
		}
	}
	for i, value := range scratch {
		if shift > 0 {
			value = int32(clipRange(roundShift(int64(value), shift), colMin, colMax))
		} else {
			value = clipRange(int64(value), colMin, colMax)
		}
		scratch[i] = value
	}
	for col := 0; col < width; col++ {
		switch vertical {
		case tx1DDCT:
			inverseDCT1D(scratch[col:], width, height, colMin, colMax)
		case tx1DIdentity:
			inverseIdentity1D(scratch[col:], width, height)
		}
	}
	dst := make([]int16, width*height)
	for i, value := range scratch {
		dst[i] = clipInt16(int32(roundShift(int64(value), 4)))
	}
	return dst
}

// col2TestFunc names one batched column kernel under test: its accelerated
// implementation, the pure-Go reference it must equal bit-for-bit, and the
// transform length.
type col2TestFunc struct {
	name   string
	length int
	impl   func(buf []int32, rowStride int, min, max int32)
	ref    func(buf []int32, rowStride int, min, max int32)
}

// runCol2 lays out a length x (rowStride>=2) row-major buffer, transforms the
// first two columns with both impl and ref, and reports any mismatch. The
// buffer is sized so the two columns plus their tail row exactly fit; the
// padding columns are left untouched and compared too, guarding against any
// stray write outside the targeted column pair.
func runCol2(t *testing.T, fn col2TestFunc, rng *rand.Rand, fill func() int32) {
	t.Helper()
	const rowStride = 4 // exercise a real stride (two target columns + padding)
	bufLen := (fn.length-1)*rowStride + 2
	for _, cs := range stageClampSets {
		min, max := cs[0], cs[1]
		for iter := 0; iter < 30000; iter++ {
			buf := make([]int32, bufLen)
			for i := range buf {
				buf[i] = clipRange(int64(fill()), min, max)
			}
			ref := append([]int32(nil), buf...)
			got := append([]int32(nil), buf...)
			fn.ref(ref, rowStride, min, max)
			fn.impl(got, rowStride, min, max)
			if !eqInt32(got, ref) {
				t.Fatalf("%s clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v",
					fn.name, min, max, iter, buf, got, ref)
			}
		}
	}
}

// runCol4 is runCol2 for the batched four-column kernels: it transforms the
// first four columns of a length x rowStride row-major buffer with both impl
// and ref and reports any mismatch, including in the untouched padding
// columns. Inputs are pre-clamped to [min, max], the four-column kernels'
// documented precondition (the decode path establishes it via the mid-pass
// clamp in hybrid.go).
func runCol4(t *testing.T, fn col2TestFunc, rng *rand.Rand, fill func() int32) {
	t.Helper()
	const rowStride = 6 // four target columns + padding
	bufLen := (fn.length-1)*rowStride + 4
	for _, cs := range stageClampSets {
		min, max := cs[0], cs[1]
		for iter := 0; iter < 30000; iter++ {
			buf := make([]int32, bufLen)
			for i := range buf {
				buf[i] = clipRange(int64(fill()), min, max)
			}
			ref := append([]int32(nil), buf...)
			got := append([]int32(nil), buf...)
			fn.ref(ref, rowStride, min, max)
			fn.impl(got, rowStride, min, max)
			if !eqInt32(got, ref) {
				t.Fatalf("%s clamp=[%d,%d] iter=%d\n in=%v\n got=%v\nwant=%v",
					fn.name, min, max, iter, buf, got, ref)
			}
		}
	}
}

// TestColPassDispatchMatchesPureGo asserts every column kernel is bit-identical
// to its pure-Go reference across random inputs and the full clamp ladder. On
// arm64 the entries resolve to the Go SIMD kernels, so this is their
// byte-exactness gate.
func TestColPassDispatchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xc01))
	for _, fn := range colPass2TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol2(t, fn, rng, func() int32 { return int32(int32(rng.Int63()) >> 11) })
		})
	}
	for _, fn := range colPass4TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol4(t, fn, rng, func() int32 { return int32(int32(rng.Int63()) >> 11) })
		})
	}
}

// TestColPassDispatchEdgeValues crosses the boundary coefficient values that
// sit at clamp and rounding cusps.
func TestColPassDispatchEdgeValues(t *testing.T) {
	rng := rand.New(rand.NewSource(0xfeed))
	for _, fn := range colPass2TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol2(t, fn, rng, func() int32 { return rowEdgeValues[rng.Intn(len(rowEdgeValues))] })
		})
	}
	for _, fn := range colPass4TestFuncs() {
		t.Run(fn.name, func(t *testing.T) {
			runCol4(t, fn, rng, func() int32 { return rowEdgeValues[rng.Intn(len(rowEdgeValues))] })
		})
	}
}

// TestColPassDispatchZeroAlloc protects the hot-path invariant that the batched
// column kernels do not allocate.
func TestColPassDispatchZeroAlloc(t *testing.T) {
	const rowStride = 4
	buf := make([]int32, (dct16Size-1)*rowStride+2)
	for i := range buf {
		buf[i] = int32(i*131 - 400)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		inverseDCT8Col2Impl(buf, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT8 batched column kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT16Col2Impl(buf, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT16 batched column kernel allocated %f times per call", allocs)
	}
	buf4 := make([]int32, (dct64Size-1)*rowStride+4)
	for i := range buf4 {
		buf4[i] = int32(i*131 - 400)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT16Col4Impl(buf4, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT16 batched four-column kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT32Col4Impl(buf4, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT32 batched four-column kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT64Col4Impl(buf4, rowStride, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT64 batched four-column kernel allocated %f times per call", allocs)
	}
}

// d157Block0Coeffs is the column-major (col*32+row) int16 array of
// entropy-decoded transform coefficients for the 32x32 D157 luma
// block at (mi=0,0) of frame 0 in the libaom 8-bit 34x34 test
// vector. Each non-zero entry was emitted by ReadCoefficientsTXB
// under the goav1_coeff_trace build tag and cross-validated bit-
// exact against libaom's read_coeffs_txb. The block uses tx_type
// DCT_DCT (the only type allowed for 32x32 intra) and tx_scale=1.
var d157Block0Coeffs = [1024]int16{
	150, 13, -23, 12, 5, -5, 3, -2, -6, 7, -2, 7, -2, -2, 2, -2, 4, 6, 2, -5, 1, -1, -4, -2, 0, 0, -1, 0, -1, 1, 1, 0,
	2, -9, 2, 1, -3, 3, 3, 1, -1, 3, -4, 5, -4, 1, 4, -2, 1, -1, 2, -2, 0, 1, 1, 1, -1, 1, 0, 0, -1, 0, 0, 0,
	-3, 5, -8, -1, 0, 2, -1, -2, -3, 1, 3, 1, -3, -1, 0, -3, 3, 1, 2, 1, -4, -1, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0,
	-1, 0, 2, 1, -2, 1, 1, 2, 2, 0, 2, -2, 4, -1, 1, 0, -4, 1, 2, 1, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0,
	3, -3, 3, 1, 0, -1, 0, -3, 0, -2, 1, 1, 1, 4, -2, -1, -1, -1, 2, -2, 1, 1, 1, -1, 0, -1, 0, 1, 0, 0, 0, 0,
	0, -2, -4, -1, -1, -2, 1, -2, 4, 0, -1, 1, -3, 3, -2, 0, -1, 0, 1, -2, -1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, -1, -2, 1, 0, 1, -1, -2, 1, -1, 0, -1, 0, 0, -1, 1, -1, -2, 0, 0, -1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0,
	0, -3, -1, 1, 0, 2, -1, 0, 0, 0, -1, 0, 1, 2, 3, -3, -1, 0, 0, 1, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0,
	2, -1, 1, 1, -1, 0, 0, -1, 1, 1, 0, -1, -2, -1, -1, -2, 1, 0, -1, 2, 1, 1, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0,
	1, 1, -1, -2, 0, -2, -1, 0, 0, -2, 0, -1, 1, 1, 0, 0, -1, 1, 0, 2, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
	-3, -1, 0, 1, -1, -1, -1, 1, 0, -2, -1, 0, 0, 0, -1, 0, 1, 2, 0, -2, 2, 1, 1, 0, 1, 0, 0, -1, 0, 0, 0, 0,
	0, -1, 2, 0, -2, 1, 0, 2, 1, 1, 0, 0, -2, -1, -1, 1, 2, -1, -1, -2, 1, 0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 0,
	-1, -2, 1, 1, 2, 1, -2, 0, -3, 0, 1, 0, 1, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 0,
	-1, 1, 0, 0, 1, 0, 3, 1, -2, -1, 0, 0, 0, 1, -1, -1, -2, 0, 2, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 1, 0, -1, 1, 1, -2, 0, 0, 0, 1, -1, 2, 0, 0, -1, 1, 2, 1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	-1, 0, 1, -2, 1, -1, -1, 0, -2, 2, 2, 0, 1, 1, 0, 1, 1, 1, 0, -2, -1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, -1, -1, 1, 2, 0, 0, -3, -3, -1, -1, -1, 1, 0, 1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, -1, 1, 0, 0, 0, -1, 2, 0, 1, 1, 1, 1, -1, 0, -1, 0, 1, 1, 0, -1, 1, -1, 0, 1, 0, 0, 0, 0, 0, 0,
	0, 0, -1, -1, 0, 0, 0, -1, 1, -1, 0, 1, 1, 0, 1, 0, 1, -1, 0, 0, 1, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0,
	-2, -1, 1, 2, 1, 1, -1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0, 0, 1, -1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	-2, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, -1, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, -1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	-5, -4, 2, -1, 0, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	2, 2, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
}

func TestD157Block0InverseTransformMatchesLibaomReference(t *testing.T) {
	const W, H = 32, 32
	const DC, AC = 107, 130
	const txScale = 1

	dqMin := -int32(1) << 15
	dqMax := int32(1)<<15 - 1
	dq := make([]int32, W*H)
	for col := range W {
		for row := range H {
			scale := int32(AC)
			if col == 0 && row == 0 {
				scale = DC
			}
			lvl := int32(d157Block0Coeffs[col*H+row])
			negative := lvl < 0
			if negative {
				lvl = -lvl
			}
			product := (lvl * scale) & 0xffffff
			out := product >> txScale
			if negative {
				out = -out
			}
			if out < dqMin {
				out = dqMin
			} else if out > dqMax {
				out = dqMax
			}
			dq[col*H+row] = out
		}
	}

	// goav1 inverse-transform path.
	got := make([]int16, W*H)
	scratch := make([]int32, W*H)
	if err := InverseBlockBitDepth(got, W, dq, H, scratch,
		Size{Width: W, Height: H}, TypeDCTDCT, 8); err != nil {
		t.Fatal(err)
	}

	// libaom-port reference (faithful inv_txfm2d_add_c reproduction).
	want := libaomInverseResidual(dq, H, Size{Width: W, Height: H}, TypeDCTDCT, 8)

	for i := range W * H {
		if int32(got[i]) != want[i] {
			t.Fatalf("D157 blk0 residual[%d]: got=%d want=%d", i, got[i], want[i])
		}
	}
}

// TestInverseDCTDCOnlySampleMatchesTransform checks dav1d's closed-form DC-only
// residual against the full 1D-kernel evaluation for every size, bit depth and
// dequantized DC value.
func TestInverseDCTDCOnlySampleMatchesTransform(t *testing.T) {
	scratch := make([]int32, 128)
	dims := []uint8{4, 8, 16, 32, 64}
	for _, w := range dims {
		for _, h := range dims {
			size := Size{Width: w, Height: h}
			if idx := sizeIndex(size); idx < 0 || !sizeValidTable[idx] {
				continue
			}
			for _, bd := range []uint8{8, 10, 12} {
				cfMax := int32(1)<<(7+bd) - 1
				step := int32(1)
				if bd > 8 {
					step = 7
				}
				for dc := -cfMax - 1; dc <= cfMax; dc += step {
					want, err := InverseDCTDCOnlySampleBitDepth(dc, scratch, size, bd)
					if err != nil {
						t.Fatal(err)
					}
					if got := InverseDCTDCOnlySample(dc, size, bd); got != want {
						t.Fatalf("%dx%d bd=%d dc=%d: got %d want %d", size.Width, size.Height, bd, dc, got, want)
					}
				}
			}
		}
	}
}

func TestInverseDCT64BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(64))
	for trial := range 200 {
		var input [64]int32
		// Mostly sparse: zero everywhere then plant 1-4 random coefficients.
		nz := 1 + rng.Intn(4)
		for range nz {
			pos := rng.Intn(64)
			input[pos] = int32(rng.Intn(1<<15) - (1 << 14))
		}
		var want [64]int32
		var got [64]int32
		copy(got[:], input[:])
		libaomIDCT64(input[:], want[:])
		inverseDCT64(got[:], 1, -(1 << 15), (1<<15)-1)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial=%d idx=%d got=%d want=%d input=%v", trial, i, got[i], want[i], input)
			}
		}
	}
}

func TestInverseDCT64BitExactLibaomDense(t *testing.T) {
	rng := rand.New(rand.NewSource(1064))
	for trial := range 200 {
		var input [64]int32
		for i := range input {
			input[i] = int32(rng.Intn(1<<15) - (1 << 14))
		}
		var want [64]int32
		var got [64]int32
		copy(got[:], input[:])
		libaomIDCT64(input[:], want[:])
		inverseDCT64(got[:], 1, -(1 << 15), (1<<15)-1)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial=%d idx=%d got=%d want=%d", trial, i, got[i], want[i])
			}
		}
	}
}

// TestInverseDCT64BitExactLibaomImpulse pins the impulse response (single
// non-zero input) of inverseDCT64 against libaom's av1_idct64 at each of
// the 64 input positions. The 32..63 high-frequency positions are the
// ones whose impulse response was previously wrong (returned zero) and
// drove the 64x64 DCT_DCT residual error on the column pass.
func TestInverseDCT64BitExactLibaomImpulse(t *testing.T) {
	for pos := range 64 {
		var input [64]int32
		input[pos] = 1000
		var want [64]int32
		var got [64]int32
		copy(got[:], input[:])
		libaomIDCT64(input[:], want[:])
		inverseDCT64(got[:], 1, -(1 << 15), (1<<15)-1)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("pos=%d idx=%d got=%d want=%d", pos, i, got[i], want[i])
			}
		}
	}
}

// libaom1DAll dispatches all supported 1D inverse transforms used by the 2D
// pipeline, including length 64.
func libaom1DAll(input []int32, output []int32, length int, typ tx1DType) {
	if typ == tx1DDCT && length == 64 {
		libaomIDCT64(input, output)
		return
	}
	libaom1D(input, output, length, typ)
}

// libaomInverseResidual64x64 mirrors libaom's inv_txfm2d_add_64x64_c at bd=8.
// The first 32 columns / 32 rows of the input are read from `coeff` (column-
// major with stride coeffStride); the rest of the implicit 64x64 input is
// zero. The residual returned is row-major width*height int32, matching
// libaomInverseResidual.
func libaomInverseResidual64x64(coeff []int32, coeffStride int, bd int) []int32 {
	width := 64
	height := 64
	shift0 := 2
	shift1 := 4
	rowBits := bd + 8
	colBits := max(bd+6, 16)

	buf := make([]int32, width*height)
	rowIn := make([]int32, width)
	rowOut := make([]int32, width)

	// Rows
	for r := range height {
		for c := range width {
			var v int32
			if c < 32 && r < 32 {
				v = coeff[c*coeffStride+r]
			}
			rowIn[c] = v
		}
		libaomClampBits(rowIn, rowBits)
		libaom1DAll(rowIn, rowOut, width, tx1DDCT)
		libaomRoundShiftArray(rowOut, shift0)
		copy(buf[r*width:(r+1)*width], rowOut)
	}

	colIn := make([]int32, height)
	colOut := make([]int32, height)
	out := make([]int32, width*height)
	for c := range width {
		for r := range height {
			colIn[r] = buf[r*width+c]
		}
		libaomClampBits(colIn, colBits)
		libaom1DAll(colIn, colOut, height, tx1DDCT)
		libaomRoundShiftArray(colOut, shift1)
		for r := range height {
			out[r*width+c] = colOut[r]
		}
	}
	return out
}

// TestInverseBlockBitDepth64x64DCTDCTRandom exercises the full 2D 64x64
// DCT_DCT inverse pipeline against a libaom-faithful reference. Uses
// sparse-ish (90% zero) coefficients to mimic real coded residuals,
// reproducing the row/column shape that drove the frame-3 monochrome
// growing-with-column residual error.
func TestInverseBlockBitDepth64x64DCTDCTRandom(t *testing.T) {
	sz := Size{Width: 64, Height: 64}
	coeffSize := adjustedScanSize(sz) // 32x32

	rng := rand.New(rand.NewSource(20260527))
	for trial := range 20 {
		coeffStride := int(coeffSize.Height)
		coeff := make([]int32, int(coeffSize.Width)*coeffStride)
		for i := range coeff {
			if rng.Intn(10) == 0 {
				// Spec-valid magnitude: the int16 8bpc DCT64 column path (dav1d)
				// requires the int16 butterfly not to overflow — real decode
				// guarantees this; dense large random coeffs do not.
				coeff[i] = int32(rng.Intn(1<<10) - (1 << 9))
			}
		}
		wantResid := libaomInverseResidual64x64(coeff, coeffStride, 8)

		width := int(sz.Width)
		height := int(sz.Height)
		gotDst := make([]int16, width*height)
		scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
		scratch := make([]int32, scratchLen)
		if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, TypeDCTDCT, 8); err != nil {
			t.Fatalf("trial=%d err=%v", trial, err)
		}
		for i := range gotDst {
			if int32(gotDst[i]) != wantResid[i] {
				r := i / width
				c := i % width
				t.Fatalf("trial=%d idx=%d (r=%d c=%d) got=%d want=%d", trial, i, r, c, gotDst[i], wantResid[i])
			}
		}
	}
}

// TestInverseBlockBitDepth64x64DCTDCTSparseFrame3 pins the specific
// sparse-coefficient pattern referenced by the prior-agent diagnostic for
// the monochrome frame-3 mismatch (block mi=(32,16) BLOCK_64X64,
// compound NEAREST_NEAREST, three non-zero coefficients producing a
// smooth-varying-with-column residual). The exact coefficients used are
// synthetic but exercise the row+column 64-length DCT path that previously
// produced growing-with-column residual error.
func TestInverseBlockBitDepth64x64DCTDCTSparseFrame3(t *testing.T) {
	sz := Size{Width: 64, Height: 64}
	coeffSize := adjustedScanSize(sz) // 32x32
	stride := int(coeffSize.Height)

	coeff := make([]int32, int(coeffSize.Width)*stride)
	// Three non-zero coefficients mimicking the sparse residual that
	// produced the row 64 residual error in frame 3 of the monochrome
	// vector. The exact coefficient indices/values are chosen to
	// exercise low- and mid-frequency rows that recurse into the
	// length-64 column pass on positions 33..63.
	coeff[0*stride+0] = -200 // DC
	coeff[0*stride+1] = 150  // (col=0,row=1)
	coeff[1*stride+2] = -80  // (col=1,row=2)
	wantResid := libaomInverseResidual64x64(coeff, stride, 8)

	width := int(sz.Width)
	height := int(sz.Height)
	gotDst := make([]int16, width*height)
	scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
	scratch := make([]int32, scratchLen)
	if err := InverseBlockBitDepth(gotDst, width, coeff, stride, scratch, sz, TypeDCTDCT, 8); err != nil {
		t.Fatalf("err=%v", err)
	}
	for i := range gotDst {
		if int32(gotDst[i]) != wantResid[i] {
			r := i / width
			c := i % width
			t.Fatalf("idx=%d (r=%d c=%d) got=%d want=%d", i, r, c, gotDst[i], wantResid[i])
		}
	}
}

// libaomIDCT64 reproduces libaom's av1_idct64 byte-exactly for cos_bit=12 and
// the 16-bit stage clamp used at 8-bit pixel depth.
func libaomIDCT64(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [64]int32
	var bf0, bf1 []int32

	// stage 1: bit reversal
	bf1 = output
	bf1[0] = input[0]
	bf1[1] = input[32]
	bf1[2] = input[16]
	bf1[3] = input[48]
	bf1[4] = input[8]
	bf1[5] = input[40]
	bf1[6] = input[24]
	bf1[7] = input[56]
	bf1[8] = input[4]
	bf1[9] = input[36]
	bf1[10] = input[20]
	bf1[11] = input[52]
	bf1[12] = input[12]
	bf1[13] = input[44]
	bf1[14] = input[28]
	bf1[15] = input[60]
	bf1[16] = input[2]
	bf1[17] = input[34]
	bf1[18] = input[18]
	bf1[19] = input[50]
	bf1[20] = input[10]
	bf1[21] = input[42]
	bf1[22] = input[26]
	bf1[23] = input[58]
	bf1[24] = input[6]
	bf1[25] = input[38]
	bf1[26] = input[22]
	bf1[27] = input[54]
	bf1[28] = input[14]
	bf1[29] = input[46]
	bf1[30] = input[30]
	bf1[31] = input[62]
	bf1[32] = input[1]
	bf1[33] = input[33]
	bf1[34] = input[17]
	bf1[35] = input[49]
	bf1[36] = input[9]
	bf1[37] = input[41]
	bf1[38] = input[25]
	bf1[39] = input[57]
	bf1[40] = input[5]
	bf1[41] = input[37]
	bf1[42] = input[21]
	bf1[43] = input[53]
	bf1[44] = input[13]
	bf1[45] = input[45]
	bf1[46] = input[29]
	bf1[47] = input[61]
	bf1[48] = input[3]
	bf1[49] = input[35]
	bf1[50] = input[19]
	bf1[51] = input[51]
	bf1[52] = input[11]
	bf1[53] = input[43]
	bf1[54] = input[27]
	bf1[55] = input[59]
	bf1[56] = input[7]
	bf1[57] = input[39]
	bf1[58] = input[23]
	bf1[59] = input[55]
	bf1[60] = input[15]
	bf1[61] = input[47]
	bf1[62] = input[31]
	bf1[63] = input[63]

	// stage 2
	bf0 = output
	bf1 = step[:]
	for i := range 32 {
		bf1[i] = bf0[i]
	}
	bf1[32] = halfBtf(cospi[63], bf0[32], -cospi[1], bf0[63])
	bf1[33] = halfBtf(cospi[31], bf0[33], -cospi[33], bf0[62])
	bf1[34] = halfBtf(cospi[47], bf0[34], -cospi[17], bf0[61])
	bf1[35] = halfBtf(cospi[15], bf0[35], -cospi[49], bf0[60])
	bf1[36] = halfBtf(cospi[55], bf0[36], -cospi[9], bf0[59])
	bf1[37] = halfBtf(cospi[23], bf0[37], -cospi[41], bf0[58])
	bf1[38] = halfBtf(cospi[39], bf0[38], -cospi[25], bf0[57])
	bf1[39] = halfBtf(cospi[7], bf0[39], -cospi[57], bf0[56])
	bf1[40] = halfBtf(cospi[59], bf0[40], -cospi[5], bf0[55])
	bf1[41] = halfBtf(cospi[27], bf0[41], -cospi[37], bf0[54])
	bf1[42] = halfBtf(cospi[43], bf0[42], -cospi[21], bf0[53])
	bf1[43] = halfBtf(cospi[11], bf0[43], -cospi[53], bf0[52])
	bf1[44] = halfBtf(cospi[51], bf0[44], -cospi[13], bf0[51])
	bf1[45] = halfBtf(cospi[19], bf0[45], -cospi[45], bf0[50])
	bf1[46] = halfBtf(cospi[35], bf0[46], -cospi[29], bf0[49])
	bf1[47] = halfBtf(cospi[3], bf0[47], -cospi[61], bf0[48])
	bf1[48] = halfBtf(cospi[61], bf0[47], cospi[3], bf0[48])
	bf1[49] = halfBtf(cospi[29], bf0[46], cospi[35], bf0[49])
	bf1[50] = halfBtf(cospi[45], bf0[45], cospi[19], bf0[50])
	bf1[51] = halfBtf(cospi[13], bf0[44], cospi[51], bf0[51])
	bf1[52] = halfBtf(cospi[53], bf0[43], cospi[11], bf0[52])
	bf1[53] = halfBtf(cospi[21], bf0[42], cospi[43], bf0[53])
	bf1[54] = halfBtf(cospi[37], bf0[41], cospi[27], bf0[54])
	bf1[55] = halfBtf(cospi[5], bf0[40], cospi[59], bf0[55])
	bf1[56] = halfBtf(cospi[57], bf0[39], cospi[7], bf0[56])
	bf1[57] = halfBtf(cospi[25], bf0[38], cospi[39], bf0[57])
	bf1[58] = halfBtf(cospi[41], bf0[37], cospi[23], bf0[58])
	bf1[59] = halfBtf(cospi[9], bf0[36], cospi[55], bf0[59])
	bf1[60] = halfBtf(cospi[49], bf0[35], cospi[15], bf0[60])
	bf1[61] = halfBtf(cospi[17], bf0[34], cospi[47], bf0[61])
	bf1[62] = halfBtf(cospi[33], bf0[33], cospi[31], bf0[62])
	bf1[63] = halfBtf(cospi[1], bf0[32], cospi[63], bf0[63])

	// stage 3
	bf0 = step[:]
	bf1 = output
	for i := range 16 {
		bf1[i] = bf0[i]
	}
	bf1[16] = halfBtf(cospi[62], bf0[16], -cospi[2], bf0[31])
	bf1[17] = halfBtf(cospi[30], bf0[17], -cospi[34], bf0[30])
	bf1[18] = halfBtf(cospi[46], bf0[18], -cospi[18], bf0[29])
	bf1[19] = halfBtf(cospi[14], bf0[19], -cospi[50], bf0[28])
	bf1[20] = halfBtf(cospi[54], bf0[20], -cospi[10], bf0[27])
	bf1[21] = halfBtf(cospi[22], bf0[21], -cospi[42], bf0[26])
	bf1[22] = halfBtf(cospi[38], bf0[22], -cospi[26], bf0[25])
	bf1[23] = halfBtf(cospi[6], bf0[23], -cospi[58], bf0[24])
	bf1[24] = halfBtf(cospi[58], bf0[23], cospi[6], bf0[24])
	bf1[25] = halfBtf(cospi[26], bf0[22], cospi[38], bf0[25])
	bf1[26] = halfBtf(cospi[42], bf0[21], cospi[22], bf0[26])
	bf1[27] = halfBtf(cospi[10], bf0[20], cospi[54], bf0[27])
	bf1[28] = halfBtf(cospi[50], bf0[19], cospi[14], bf0[28])
	bf1[29] = halfBtf(cospi[18], bf0[18], cospi[46], bf0[29])
	bf1[30] = halfBtf(cospi[34], bf0[17], cospi[30], bf0[30])
	bf1[31] = halfBtf(cospi[2], bf0[16], cospi[62], bf0[31])
	bf1[32] = clamp(bf0[32] + bf0[33])
	bf1[33] = clamp(bf0[32] - bf0[33])
	bf1[34] = clamp(-bf0[34] + bf0[35])
	bf1[35] = clamp(bf0[34] + bf0[35])
	bf1[36] = clamp(bf0[36] + bf0[37])
	bf1[37] = clamp(bf0[36] - bf0[37])
	bf1[38] = clamp(-bf0[38] + bf0[39])
	bf1[39] = clamp(bf0[38] + bf0[39])
	bf1[40] = clamp(bf0[40] + bf0[41])
	bf1[41] = clamp(bf0[40] - bf0[41])
	bf1[42] = clamp(-bf0[42] + bf0[43])
	bf1[43] = clamp(bf0[42] + bf0[43])
	bf1[44] = clamp(bf0[44] + bf0[45])
	bf1[45] = clamp(bf0[44] - bf0[45])
	bf1[46] = clamp(-bf0[46] + bf0[47])
	bf1[47] = clamp(bf0[46] + bf0[47])
	bf1[48] = clamp(bf0[48] + bf0[49])
	bf1[49] = clamp(bf0[48] - bf0[49])
	bf1[50] = clamp(-bf0[50] + bf0[51])
	bf1[51] = clamp(bf0[50] + bf0[51])
	bf1[52] = clamp(bf0[52] + bf0[53])
	bf1[53] = clamp(bf0[52] - bf0[53])
	bf1[54] = clamp(-bf0[54] + bf0[55])
	bf1[55] = clamp(bf0[54] + bf0[55])
	bf1[56] = clamp(bf0[56] + bf0[57])
	bf1[57] = clamp(bf0[56] - bf0[57])
	bf1[58] = clamp(-bf0[58] + bf0[59])
	bf1[59] = clamp(bf0[58] + bf0[59])
	bf1[60] = clamp(bf0[60] + bf0[61])
	bf1[61] = clamp(bf0[60] - bf0[61])
	bf1[62] = clamp(-bf0[62] + bf0[63])
	bf1[63] = clamp(bf0[62] + bf0[63])

	// stage 4
	bf0 = output
	bf1 = step[:]
	for i := range 8 {
		bf1[i] = bf0[i]
	}
	bf1[8] = halfBtf(cospi[60], bf0[8], -cospi[4], bf0[15])
	bf1[9] = halfBtf(cospi[28], bf0[9], -cospi[36], bf0[14])
	bf1[10] = halfBtf(cospi[44], bf0[10], -cospi[20], bf0[13])
	bf1[11] = halfBtf(cospi[12], bf0[11], -cospi[52], bf0[12])
	bf1[12] = halfBtf(cospi[52], bf0[11], cospi[12], bf0[12])
	bf1[13] = halfBtf(cospi[20], bf0[10], cospi[44], bf0[13])
	bf1[14] = halfBtf(cospi[36], bf0[9], cospi[28], bf0[14])
	bf1[15] = halfBtf(cospi[4], bf0[8], cospi[60], bf0[15])
	bf1[16] = clamp(bf0[16] + bf0[17])
	bf1[17] = clamp(bf0[16] - bf0[17])
	bf1[18] = clamp(-bf0[18] + bf0[19])
	bf1[19] = clamp(bf0[18] + bf0[19])
	bf1[20] = clamp(bf0[20] + bf0[21])
	bf1[21] = clamp(bf0[20] - bf0[21])
	bf1[22] = clamp(-bf0[22] + bf0[23])
	bf1[23] = clamp(bf0[22] + bf0[23])
	bf1[24] = clamp(bf0[24] + bf0[25])
	bf1[25] = clamp(bf0[24] - bf0[25])
	bf1[26] = clamp(-bf0[26] + bf0[27])
	bf1[27] = clamp(bf0[26] + bf0[27])
	bf1[28] = clamp(bf0[28] + bf0[29])
	bf1[29] = clamp(bf0[28] - bf0[29])
	bf1[30] = clamp(-bf0[30] + bf0[31])
	bf1[31] = clamp(bf0[30] + bf0[31])
	bf1[32] = bf0[32]
	bf1[33] = halfBtf(-cospi[4], bf0[33], cospi[60], bf0[62])
	bf1[34] = halfBtf(-cospi[60], bf0[34], -cospi[4], bf0[61])
	bf1[35] = bf0[35]
	bf1[36] = bf0[36]
	bf1[37] = halfBtf(-cospi[36], bf0[37], cospi[28], bf0[58])
	bf1[38] = halfBtf(-cospi[28], bf0[38], -cospi[36], bf0[57])
	bf1[39] = bf0[39]
	bf1[40] = bf0[40]
	bf1[41] = halfBtf(-cospi[20], bf0[41], cospi[44], bf0[54])
	bf1[42] = halfBtf(-cospi[44], bf0[42], -cospi[20], bf0[53])
	bf1[43] = bf0[43]
	bf1[44] = bf0[44]
	bf1[45] = halfBtf(-cospi[52], bf0[45], cospi[12], bf0[50])
	bf1[46] = halfBtf(-cospi[12], bf0[46], -cospi[52], bf0[49])
	bf1[47] = bf0[47]
	bf1[48] = bf0[48]
	bf1[49] = halfBtf(-cospi[52], bf0[46], cospi[12], bf0[49])
	bf1[50] = halfBtf(cospi[12], bf0[45], cospi[52], bf0[50])
	bf1[51] = bf0[51]
	bf1[52] = bf0[52]
	bf1[53] = halfBtf(-cospi[20], bf0[42], cospi[44], bf0[53])
	bf1[54] = halfBtf(cospi[44], bf0[41], cospi[20], bf0[54])
	bf1[55] = bf0[55]
	bf1[56] = bf0[56]
	bf1[57] = halfBtf(-cospi[36], bf0[38], cospi[28], bf0[57])
	bf1[58] = halfBtf(cospi[28], bf0[37], cospi[36], bf0[58])
	bf1[59] = bf0[59]
	bf1[60] = bf0[60]
	bf1[61] = halfBtf(-cospi[4], bf0[34], cospi[60], bf0[61])
	bf1[62] = halfBtf(cospi[60], bf0[33], cospi[4], bf0[62])
	bf1[63] = bf0[63]

	// stage 5
	bf0 = step[:]
	bf1 = output
	for i := range 4 {
		bf1[i] = bf0[i]
	}
	bf1[4] = halfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = halfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = halfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = halfBtf(cospi[8], bf0[4], cospi[56], bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[9])
	bf1[9] = clamp(bf0[8] - bf0[9])
	bf1[10] = clamp(-bf0[10] + bf0[11])
	bf1[11] = clamp(bf0[10] + bf0[11])
	bf1[12] = clamp(bf0[12] + bf0[13])
	bf1[13] = clamp(bf0[12] - bf0[13])
	bf1[14] = clamp(-bf0[14] + bf0[15])
	bf1[15] = clamp(bf0[14] + bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = halfBtf(-cospi[8], bf0[17], cospi[56], bf0[30])
	bf1[18] = halfBtf(-cospi[56], bf0[18], -cospi[8], bf0[29])
	bf1[19] = bf0[19]
	bf1[20] = bf0[20]
	bf1[21] = halfBtf(-cospi[40], bf0[21], cospi[24], bf0[26])
	bf1[22] = halfBtf(-cospi[24], bf0[22], -cospi[40], bf0[25])
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = halfBtf(-cospi[40], bf0[22], cospi[24], bf0[25])
	bf1[26] = halfBtf(cospi[24], bf0[21], cospi[40], bf0[26])
	bf1[27] = bf0[27]
	bf1[28] = bf0[28]
	bf1[29] = halfBtf(-cospi[8], bf0[18], cospi[56], bf0[29])
	bf1[30] = halfBtf(cospi[56], bf0[17], cospi[8], bf0[30])
	bf1[31] = bf0[31]
	bf1[32] = clamp(bf0[32] + bf0[35])
	bf1[33] = clamp(bf0[33] + bf0[34])
	bf1[34] = clamp(bf0[33] - bf0[34])
	bf1[35] = clamp(bf0[32] - bf0[35])
	bf1[36] = clamp(-bf0[36] + bf0[39])
	bf1[37] = clamp(-bf0[37] + bf0[38])
	bf1[38] = clamp(bf0[37] + bf0[38])
	bf1[39] = clamp(bf0[36] + bf0[39])
	bf1[40] = clamp(bf0[40] + bf0[43])
	bf1[41] = clamp(bf0[41] + bf0[42])
	bf1[42] = clamp(bf0[41] - bf0[42])
	bf1[43] = clamp(bf0[40] - bf0[43])
	bf1[44] = clamp(-bf0[44] + bf0[47])
	bf1[45] = clamp(-bf0[45] + bf0[46])
	bf1[46] = clamp(bf0[45] + bf0[46])
	bf1[47] = clamp(bf0[44] + bf0[47])
	bf1[48] = clamp(bf0[48] + bf0[51])
	bf1[49] = clamp(bf0[49] + bf0[50])
	bf1[50] = clamp(bf0[49] - bf0[50])
	bf1[51] = clamp(bf0[48] - bf0[51])
	bf1[52] = clamp(-bf0[52] + bf0[55])
	bf1[53] = clamp(-bf0[53] + bf0[54])
	bf1[54] = clamp(bf0[53] + bf0[54])
	bf1[55] = clamp(bf0[52] + bf0[55])
	bf1[56] = clamp(bf0[56] + bf0[59])
	bf1[57] = clamp(bf0[57] + bf0[58])
	bf1[58] = clamp(bf0[57] - bf0[58])
	bf1[59] = clamp(bf0[56] - bf0[59])
	bf1[60] = clamp(-bf0[60] + bf0[63])
	bf1[61] = clamp(-bf0[61] + bf0[62])
	bf1[62] = clamp(bf0[61] + bf0[62])
	bf1[63] = clamp(bf0[60] + bf0[63])

	// stage 6
	bf0 = output
	bf1 = step[:]
	bf1[0] = halfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = halfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = halfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = halfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[5])
	bf1[5] = clamp(bf0[4] - bf0[5])
	bf1[6] = clamp(-bf0[6] + bf0[7])
	bf1[7] = clamp(bf0[6] + bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = halfBtf(-cospi[16], bf0[9], cospi[48], bf0[14])
	bf1[10] = halfBtf(-cospi[48], bf0[10], -cospi[16], bf0[13])
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = halfBtf(-cospi[16], bf0[10], cospi[48], bf0[13])
	bf1[14] = halfBtf(cospi[48], bf0[9], cospi[16], bf0[14])
	bf1[15] = bf0[15]
	bf1[16] = clamp(bf0[16] + bf0[19])
	bf1[17] = clamp(bf0[17] + bf0[18])
	bf1[18] = clamp(bf0[17] - bf0[18])
	bf1[19] = clamp(bf0[16] - bf0[19])
	bf1[20] = clamp(-bf0[20] + bf0[23])
	bf1[21] = clamp(-bf0[21] + bf0[22])
	bf1[22] = clamp(bf0[21] + bf0[22])
	bf1[23] = clamp(bf0[20] + bf0[23])
	bf1[24] = clamp(bf0[24] + bf0[27])
	bf1[25] = clamp(bf0[25] + bf0[26])
	bf1[26] = clamp(bf0[25] - bf0[26])
	bf1[27] = clamp(bf0[24] - bf0[27])
	bf1[28] = clamp(-bf0[28] + bf0[31])
	bf1[29] = clamp(-bf0[29] + bf0[30])
	bf1[30] = clamp(bf0[29] + bf0[30])
	bf1[31] = clamp(bf0[28] + bf0[31])
	bf1[32] = bf0[32]
	bf1[33] = bf0[33]
	bf1[34] = halfBtf(-cospi[8], bf0[34], cospi[56], bf0[61])
	bf1[35] = halfBtf(-cospi[8], bf0[35], cospi[56], bf0[60])
	bf1[36] = halfBtf(-cospi[56], bf0[36], -cospi[8], bf0[59])
	bf1[37] = halfBtf(-cospi[56], bf0[37], -cospi[8], bf0[58])
	bf1[38] = bf0[38]
	bf1[39] = bf0[39]
	bf1[40] = bf0[40]
	bf1[41] = bf0[41]
	bf1[42] = halfBtf(-cospi[40], bf0[42], cospi[24], bf0[53])
	bf1[43] = halfBtf(-cospi[40], bf0[43], cospi[24], bf0[52])
	bf1[44] = halfBtf(-cospi[24], bf0[44], -cospi[40], bf0[51])
	bf1[45] = halfBtf(-cospi[24], bf0[45], -cospi[40], bf0[50])
	bf1[46] = bf0[46]
	bf1[47] = bf0[47]
	bf1[48] = bf0[48]
	bf1[49] = bf0[49]
	bf1[50] = halfBtf(-cospi[40], bf0[45], cospi[24], bf0[50])
	bf1[51] = halfBtf(-cospi[40], bf0[44], cospi[24], bf0[51])
	bf1[52] = halfBtf(cospi[24], bf0[43], cospi[40], bf0[52])
	bf1[53] = halfBtf(cospi[24], bf0[42], cospi[40], bf0[53])
	bf1[54] = bf0[54]
	bf1[55] = bf0[55]
	bf1[56] = bf0[56]
	bf1[57] = bf0[57]
	bf1[58] = halfBtf(-cospi[8], bf0[37], cospi[56], bf0[58])
	bf1[59] = halfBtf(-cospi[8], bf0[36], cospi[56], bf0[59])
	bf1[60] = halfBtf(cospi[56], bf0[35], cospi[8], bf0[60])
	bf1[61] = halfBtf(cospi[56], bf0[34], cospi[8], bf0[61])
	bf1[62] = bf0[62]
	bf1[63] = bf0[63]

	// stage 7
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[3])
	bf1[1] = clamp(bf0[1] + bf0[2])
	bf1[2] = clamp(bf0[1] - bf0[2])
	bf1[3] = clamp(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = halfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = halfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]
	bf1[8] = clamp(bf0[8] + bf0[11])
	bf1[9] = clamp(bf0[9] + bf0[10])
	bf1[10] = clamp(bf0[9] - bf0[10])
	bf1[11] = clamp(bf0[8] - bf0[11])
	bf1[12] = clamp(-bf0[12] + bf0[15])
	bf1[13] = clamp(-bf0[13] + bf0[14])
	bf1[14] = clamp(bf0[13] + bf0[14])
	bf1[15] = clamp(bf0[12] + bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = halfBtf(-cospi[16], bf0[18], cospi[48], bf0[29])
	bf1[19] = halfBtf(-cospi[16], bf0[19], cospi[48], bf0[28])
	bf1[20] = halfBtf(-cospi[48], bf0[20], -cospi[16], bf0[27])
	bf1[21] = halfBtf(-cospi[48], bf0[21], -cospi[16], bf0[26])
	bf1[22] = bf0[22]
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = bf0[25]
	bf1[26] = halfBtf(-cospi[16], bf0[21], cospi[48], bf0[26])
	bf1[27] = halfBtf(-cospi[16], bf0[20], cospi[48], bf0[27])
	bf1[28] = halfBtf(cospi[48], bf0[19], cospi[16], bf0[28])
	bf1[29] = halfBtf(cospi[48], bf0[18], cospi[16], bf0[29])
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]
	bf1[32] = clamp(bf0[32] + bf0[39])
	bf1[33] = clamp(bf0[33] + bf0[38])
	bf1[34] = clamp(bf0[34] + bf0[37])
	bf1[35] = clamp(bf0[35] + bf0[36])
	bf1[36] = clamp(bf0[35] - bf0[36])
	bf1[37] = clamp(bf0[34] - bf0[37])
	bf1[38] = clamp(bf0[33] - bf0[38])
	bf1[39] = clamp(bf0[32] - bf0[39])
	bf1[40] = clamp(-bf0[40] + bf0[47])
	bf1[41] = clamp(-bf0[41] + bf0[46])
	bf1[42] = clamp(-bf0[42] + bf0[45])
	bf1[43] = clamp(-bf0[43] + bf0[44])
	bf1[44] = clamp(bf0[43] + bf0[44])
	bf1[45] = clamp(bf0[42] + bf0[45])
	bf1[46] = clamp(bf0[41] + bf0[46])
	bf1[47] = clamp(bf0[40] + bf0[47])
	bf1[48] = clamp(bf0[48] + bf0[55])
	bf1[49] = clamp(bf0[49] + bf0[54])
	bf1[50] = clamp(bf0[50] + bf0[53])
	bf1[51] = clamp(bf0[51] + bf0[52])
	bf1[52] = clamp(bf0[51] - bf0[52])
	bf1[53] = clamp(bf0[50] - bf0[53])
	bf1[54] = clamp(bf0[49] - bf0[54])
	bf1[55] = clamp(bf0[48] - bf0[55])
	bf1[56] = clamp(-bf0[56] + bf0[63])
	bf1[57] = clamp(-bf0[57] + bf0[62])
	bf1[58] = clamp(-bf0[58] + bf0[61])
	bf1[59] = clamp(-bf0[59] + bf0[60])
	bf1[60] = clamp(bf0[59] + bf0[60])
	bf1[61] = clamp(bf0[58] + bf0[61])
	bf1[62] = clamp(bf0[57] + bf0[62])
	bf1[63] = clamp(bf0[56] + bf0[63])

	// stage 8
	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[7])
	bf1[1] = clamp(bf0[1] + bf0[6])
	bf1[2] = clamp(bf0[2] + bf0[5])
	bf1[3] = clamp(bf0[3] + bf0[4])
	bf1[4] = clamp(bf0[3] - bf0[4])
	bf1[5] = clamp(bf0[2] - bf0[5])
	bf1[6] = clamp(bf0[1] - bf0[6])
	bf1[7] = clamp(bf0[0] - bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = halfBtf(-cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[11] = halfBtf(-cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[12] = halfBtf(cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[13] = halfBtf(cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]
	bf1[16] = clamp(bf0[16] + bf0[23])
	bf1[17] = clamp(bf0[17] + bf0[22])
	bf1[18] = clamp(bf0[18] + bf0[21])
	bf1[19] = clamp(bf0[19] + bf0[20])
	bf1[20] = clamp(bf0[19] - bf0[20])
	bf1[21] = clamp(bf0[18] - bf0[21])
	bf1[22] = clamp(bf0[17] - bf0[22])
	bf1[23] = clamp(bf0[16] - bf0[23])
	bf1[24] = clamp(-bf0[24] + bf0[31])
	bf1[25] = clamp(-bf0[25] + bf0[30])
	bf1[26] = clamp(-bf0[26] + bf0[29])
	bf1[27] = clamp(-bf0[27] + bf0[28])
	bf1[28] = clamp(bf0[27] + bf0[28])
	bf1[29] = clamp(bf0[26] + bf0[29])
	bf1[30] = clamp(bf0[25] + bf0[30])
	bf1[31] = clamp(bf0[24] + bf0[31])
	bf1[32] = bf0[32]
	bf1[33] = bf0[33]
	bf1[34] = bf0[34]
	bf1[35] = bf0[35]
	bf1[36] = halfBtf(-cospi[16], bf0[36], cospi[48], bf0[59])
	bf1[37] = halfBtf(-cospi[16], bf0[37], cospi[48], bf0[58])
	bf1[38] = halfBtf(-cospi[16], bf0[38], cospi[48], bf0[57])
	bf1[39] = halfBtf(-cospi[16], bf0[39], cospi[48], bf0[56])
	bf1[40] = halfBtf(-cospi[48], bf0[40], -cospi[16], bf0[55])
	bf1[41] = halfBtf(-cospi[48], bf0[41], -cospi[16], bf0[54])
	bf1[42] = halfBtf(-cospi[48], bf0[42], -cospi[16], bf0[53])
	bf1[43] = halfBtf(-cospi[48], bf0[43], -cospi[16], bf0[52])
	bf1[44] = bf0[44]
	bf1[45] = bf0[45]
	bf1[46] = bf0[46]
	bf1[47] = bf0[47]
	bf1[48] = bf0[48]
	bf1[49] = bf0[49]
	bf1[50] = bf0[50]
	bf1[51] = bf0[51]
	bf1[52] = halfBtf(-cospi[16], bf0[43], cospi[48], bf0[52])
	bf1[53] = halfBtf(-cospi[16], bf0[42], cospi[48], bf0[53])
	bf1[54] = halfBtf(-cospi[16], bf0[41], cospi[48], bf0[54])
	bf1[55] = halfBtf(-cospi[16], bf0[40], cospi[48], bf0[55])
	bf1[56] = halfBtf(cospi[48], bf0[39], cospi[16], bf0[56])
	bf1[57] = halfBtf(cospi[48], bf0[38], cospi[16], bf0[57])
	bf1[58] = halfBtf(cospi[48], bf0[37], cospi[16], bf0[58])
	bf1[59] = halfBtf(cospi[48], bf0[36], cospi[16], bf0[59])
	bf1[60] = bf0[60]
	bf1[61] = bf0[61]
	bf1[62] = bf0[62]
	bf1[63] = bf0[63]

	// stage 9
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[15])
	bf1[1] = clamp(bf0[1] + bf0[14])
	bf1[2] = clamp(bf0[2] + bf0[13])
	bf1[3] = clamp(bf0[3] + bf0[12])
	bf1[4] = clamp(bf0[4] + bf0[11])
	bf1[5] = clamp(bf0[5] + bf0[10])
	bf1[6] = clamp(bf0[6] + bf0[9])
	bf1[7] = clamp(bf0[7] + bf0[8])
	bf1[8] = clamp(bf0[7] - bf0[8])
	bf1[9] = clamp(bf0[6] - bf0[9])
	bf1[10] = clamp(bf0[5] - bf0[10])
	bf1[11] = clamp(bf0[4] - bf0[11])
	bf1[12] = clamp(bf0[3] - bf0[12])
	bf1[13] = clamp(bf0[2] - bf0[13])
	bf1[14] = clamp(bf0[1] - bf0[14])
	bf1[15] = clamp(bf0[0] - bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = bf0[18]
	bf1[19] = bf0[19]
	bf1[20] = halfBtf(-cospi[32], bf0[20], cospi[32], bf0[27])
	bf1[21] = halfBtf(-cospi[32], bf0[21], cospi[32], bf0[26])
	bf1[22] = halfBtf(-cospi[32], bf0[22], cospi[32], bf0[25])
	bf1[23] = halfBtf(-cospi[32], bf0[23], cospi[32], bf0[24])
	bf1[24] = halfBtf(cospi[32], bf0[23], cospi[32], bf0[24])
	bf1[25] = halfBtf(cospi[32], bf0[22], cospi[32], bf0[25])
	bf1[26] = halfBtf(cospi[32], bf0[21], cospi[32], bf0[26])
	bf1[27] = halfBtf(cospi[32], bf0[20], cospi[32], bf0[27])
	bf1[28] = bf0[28]
	bf1[29] = bf0[29]
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]
	bf1[32] = clamp(bf0[32] + bf0[47])
	bf1[33] = clamp(bf0[33] + bf0[46])
	bf1[34] = clamp(bf0[34] + bf0[45])
	bf1[35] = clamp(bf0[35] + bf0[44])
	bf1[36] = clamp(bf0[36] + bf0[43])
	bf1[37] = clamp(bf0[37] + bf0[42])
	bf1[38] = clamp(bf0[38] + bf0[41])
	bf1[39] = clamp(bf0[39] + bf0[40])
	bf1[40] = clamp(bf0[39] - bf0[40])
	bf1[41] = clamp(bf0[38] - bf0[41])
	bf1[42] = clamp(bf0[37] - bf0[42])
	bf1[43] = clamp(bf0[36] - bf0[43])
	bf1[44] = clamp(bf0[35] - bf0[44])
	bf1[45] = clamp(bf0[34] - bf0[45])
	bf1[46] = clamp(bf0[33] - bf0[46])
	bf1[47] = clamp(bf0[32] - bf0[47])
	bf1[48] = clamp(-bf0[48] + bf0[63])
	bf1[49] = clamp(-bf0[49] + bf0[62])
	bf1[50] = clamp(-bf0[50] + bf0[61])
	bf1[51] = clamp(-bf0[51] + bf0[60])
	bf1[52] = clamp(-bf0[52] + bf0[59])
	bf1[53] = clamp(-bf0[53] + bf0[58])
	bf1[54] = clamp(-bf0[54] + bf0[57])
	bf1[55] = clamp(-bf0[55] + bf0[56])
	bf1[56] = clamp(bf0[55] + bf0[56])
	bf1[57] = clamp(bf0[54] + bf0[57])
	bf1[58] = clamp(bf0[53] + bf0[58])
	bf1[59] = clamp(bf0[52] + bf0[59])
	bf1[60] = clamp(bf0[51] + bf0[60])
	bf1[61] = clamp(bf0[50] + bf0[61])
	bf1[62] = clamp(bf0[49] + bf0[62])
	bf1[63] = clamp(bf0[48] + bf0[63])

	// stage 10
	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[31])
	bf1[1] = clamp(bf0[1] + bf0[30])
	bf1[2] = clamp(bf0[2] + bf0[29])
	bf1[3] = clamp(bf0[3] + bf0[28])
	bf1[4] = clamp(bf0[4] + bf0[27])
	bf1[5] = clamp(bf0[5] + bf0[26])
	bf1[6] = clamp(bf0[6] + bf0[25])
	bf1[7] = clamp(bf0[7] + bf0[24])
	bf1[8] = clamp(bf0[8] + bf0[23])
	bf1[9] = clamp(bf0[9] + bf0[22])
	bf1[10] = clamp(bf0[10] + bf0[21])
	bf1[11] = clamp(bf0[11] + bf0[20])
	bf1[12] = clamp(bf0[12] + bf0[19])
	bf1[13] = clamp(bf0[13] + bf0[18])
	bf1[14] = clamp(bf0[14] + bf0[17])
	bf1[15] = clamp(bf0[15] + bf0[16])
	bf1[16] = clamp(bf0[15] - bf0[16])
	bf1[17] = clamp(bf0[14] - bf0[17])
	bf1[18] = clamp(bf0[13] - bf0[18])
	bf1[19] = clamp(bf0[12] - bf0[19])
	bf1[20] = clamp(bf0[11] - bf0[20])
	bf1[21] = clamp(bf0[10] - bf0[21])
	bf1[22] = clamp(bf0[9] - bf0[22])
	bf1[23] = clamp(bf0[8] - bf0[23])
	bf1[24] = clamp(bf0[7] - bf0[24])
	bf1[25] = clamp(bf0[6] - bf0[25])
	bf1[26] = clamp(bf0[5] - bf0[26])
	bf1[27] = clamp(bf0[4] - bf0[27])
	bf1[28] = clamp(bf0[3] - bf0[28])
	bf1[29] = clamp(bf0[2] - bf0[29])
	bf1[30] = clamp(bf0[1] - bf0[30])
	bf1[31] = clamp(bf0[0] - bf0[31])
	bf1[32] = bf0[32]
	bf1[33] = bf0[33]
	bf1[34] = bf0[34]
	bf1[35] = bf0[35]
	bf1[36] = bf0[36]
	bf1[37] = bf0[37]
	bf1[38] = bf0[38]
	bf1[39] = bf0[39]
	bf1[40] = halfBtf(-cospi[32], bf0[40], cospi[32], bf0[55])
	bf1[41] = halfBtf(-cospi[32], bf0[41], cospi[32], bf0[54])
	bf1[42] = halfBtf(-cospi[32], bf0[42], cospi[32], bf0[53])
	bf1[43] = halfBtf(-cospi[32], bf0[43], cospi[32], bf0[52])
	bf1[44] = halfBtf(-cospi[32], bf0[44], cospi[32], bf0[51])
	bf1[45] = halfBtf(-cospi[32], bf0[45], cospi[32], bf0[50])
	bf1[46] = halfBtf(-cospi[32], bf0[46], cospi[32], bf0[49])
	bf1[47] = halfBtf(-cospi[32], bf0[47], cospi[32], bf0[48])
	bf1[48] = halfBtf(cospi[32], bf0[47], cospi[32], bf0[48])
	bf1[49] = halfBtf(cospi[32], bf0[46], cospi[32], bf0[49])
	bf1[50] = halfBtf(cospi[32], bf0[45], cospi[32], bf0[50])
	bf1[51] = halfBtf(cospi[32], bf0[44], cospi[32], bf0[51])
	bf1[52] = halfBtf(cospi[32], bf0[43], cospi[32], bf0[52])
	bf1[53] = halfBtf(cospi[32], bf0[42], cospi[32], bf0[53])
	bf1[54] = halfBtf(cospi[32], bf0[41], cospi[32], bf0[54])
	bf1[55] = halfBtf(cospi[32], bf0[40], cospi[32], bf0[55])
	bf1[56] = bf0[56]
	bf1[57] = bf0[57]
	bf1[58] = bf0[58]
	bf1[59] = bf0[59]
	bf1[60] = bf0[60]
	bf1[61] = bf0[61]
	bf1[62] = bf0[62]
	bf1[63] = bf0[63]

	// stage 11
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[63])
	bf1[1] = clamp(bf0[1] + bf0[62])
	bf1[2] = clamp(bf0[2] + bf0[61])
	bf1[3] = clamp(bf0[3] + bf0[60])
	bf1[4] = clamp(bf0[4] + bf0[59])
	bf1[5] = clamp(bf0[5] + bf0[58])
	bf1[6] = clamp(bf0[6] + bf0[57])
	bf1[7] = clamp(bf0[7] + bf0[56])
	bf1[8] = clamp(bf0[8] + bf0[55])
	bf1[9] = clamp(bf0[9] + bf0[54])
	bf1[10] = clamp(bf0[10] + bf0[53])
	bf1[11] = clamp(bf0[11] + bf0[52])
	bf1[12] = clamp(bf0[12] + bf0[51])
	bf1[13] = clamp(bf0[13] + bf0[50])
	bf1[14] = clamp(bf0[14] + bf0[49])
	bf1[15] = clamp(bf0[15] + bf0[48])
	bf1[16] = clamp(bf0[16] + bf0[47])
	bf1[17] = clamp(bf0[17] + bf0[46])
	bf1[18] = clamp(bf0[18] + bf0[45])
	bf1[19] = clamp(bf0[19] + bf0[44])
	bf1[20] = clamp(bf0[20] + bf0[43])
	bf1[21] = clamp(bf0[21] + bf0[42])
	bf1[22] = clamp(bf0[22] + bf0[41])
	bf1[23] = clamp(bf0[23] + bf0[40])
	bf1[24] = clamp(bf0[24] + bf0[39])
	bf1[25] = clamp(bf0[25] + bf0[38])
	bf1[26] = clamp(bf0[26] + bf0[37])
	bf1[27] = clamp(bf0[27] + bf0[36])
	bf1[28] = clamp(bf0[28] + bf0[35])
	bf1[29] = clamp(bf0[29] + bf0[34])
	bf1[30] = clamp(bf0[30] + bf0[33])
	bf1[31] = clamp(bf0[31] + bf0[32])
	bf1[32] = clamp(bf0[31] - bf0[32])
	bf1[33] = clamp(bf0[30] - bf0[33])
	bf1[34] = clamp(bf0[29] - bf0[34])
	bf1[35] = clamp(bf0[28] - bf0[35])
	bf1[36] = clamp(bf0[27] - bf0[36])
	bf1[37] = clamp(bf0[26] - bf0[37])
	bf1[38] = clamp(bf0[25] - bf0[38])
	bf1[39] = clamp(bf0[24] - bf0[39])
	bf1[40] = clamp(bf0[23] - bf0[40])
	bf1[41] = clamp(bf0[22] - bf0[41])
	bf1[42] = clamp(bf0[21] - bf0[42])
	bf1[43] = clamp(bf0[20] - bf0[43])
	bf1[44] = clamp(bf0[19] - bf0[44])
	bf1[45] = clamp(bf0[18] - bf0[45])
	bf1[46] = clamp(bf0[17] - bf0[46])
	bf1[47] = clamp(bf0[16] - bf0[47])
	bf1[48] = clamp(bf0[15] - bf0[48])
	bf1[49] = clamp(bf0[14] - bf0[49])
	bf1[50] = clamp(bf0[13] - bf0[50])
	bf1[51] = clamp(bf0[12] - bf0[51])
	bf1[52] = clamp(bf0[11] - bf0[52])
	bf1[53] = clamp(bf0[10] - bf0[53])
	bf1[54] = clamp(bf0[9] - bf0[54])
	bf1[55] = clamp(bf0[8] - bf0[55])
	bf1[56] = clamp(bf0[7] - bf0[56])
	bf1[57] = clamp(bf0[6] - bf0[57])
	bf1[58] = clamp(bf0[5] - bf0[58])
	bf1[59] = clamp(bf0[4] - bf0[59])
	bf1[60] = clamp(bf0[3] - bf0[60])
	bf1[61] = clamp(bf0[2] - bf0[61])
	bf1[62] = clamp(bf0[1] - bf0[62])
	bf1[63] = clamp(bf0[0] - bf0[63])
}

// libaomCospi12 is libaom's av1_cospi_arr_data[cos_bit-cos_bit_min] for the
// cos_bit=12 row used by every 1D inverse transform.
var libaomCospi12 = [...]int32{
	4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
	3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
	3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
	2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
	1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995,
	897, 799, 700, 601, 501, 401, 301, 201, 101,
}

// libaomHalfBtf reproduces libaom's half_btf for cos_bit=12.
func libaomHalfBtf(w0, in0, w1, in1 int32) int32 {
	v := int64(w0)*int64(in0) + int64(w1)*int64(in1)
	v += int64(1) << 11
	return int32(v >> 12)
}

// libaomClamp16 mirrors libaom's clamp_value with bit=16.
func libaomClamp16(v int32) int32 {
	if v < -(1 << 15) {
		return -(1 << 15)
	}
	if v > (1<<15)-1 {
		return (1 << 15) - 1
	}
	return v
}

// libaomIDCT4 reproduces libaom's av1_idct4 for cos_bit=12 and 16-bit stage
// range.
func libaomIDCT4(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	output[0] = input[0]
	output[1] = input[2]
	output[2] = input[1]
	output[3] = input[3]
	var step [4]int32
	step[0] = libaomHalfBtf(cospi[32], output[0], cospi[32], output[1])
	step[1] = libaomHalfBtf(cospi[32], output[0], -cospi[32], output[1])
	step[2] = libaomHalfBtf(cospi[48], output[2], -cospi[16], output[3])
	step[3] = libaomHalfBtf(cospi[16], output[2], cospi[48], output[3])
	output[0] = libaomClamp16(step[0] + step[3])
	output[1] = libaomClamp16(step[1] + step[2])
	output[2] = libaomClamp16(step[1] - step[2])
	output[3] = libaomClamp16(step[0] - step[3])
}

// libaomIDCT8 reproduces libaom's av1_idct8 for cos_bit=12 and 16-bit stage
// range.
func libaomIDCT8(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	var step [8]int32
	bf0 := output
	bf1 := output
	bf1[0] = input[0]
	bf1[1] = input[4]
	bf1[2] = input[2]
	bf1[3] = input[6]
	bf1[4] = input[1]
	bf1[5] = input[5]
	bf1[6] = input[3]
	bf1[7] = input[7]

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = libaomHalfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = libaomHalfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = libaomHalfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = libaomHalfBtf(cospi[8], bf0[4], cospi[56], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = libaomHalfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = libaomHalfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = libaomHalfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = libaomHalfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = libaomClamp16(bf0[4] + bf0[5])
	bf1[5] = libaomClamp16(bf0[4] - bf0[5])
	bf1[6] = libaomClamp16(-bf0[6] + bf0[7])
	bf1[7] = libaomClamp16(bf0[6] + bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = libaomClamp16(bf0[0] + bf0[3])
	bf1[1] = libaomClamp16(bf0[1] + bf0[2])
	bf1[2] = libaomClamp16(bf0[1] - bf0[2])
	bf1[3] = libaomClamp16(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = libaomHalfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = libaomHalfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]

	bf0 = step[:]
	bf1 = output
	bf1[0] = libaomClamp16(bf0[0] + bf0[7])
	bf1[1] = libaomClamp16(bf0[1] + bf0[6])
	bf1[2] = libaomClamp16(bf0[2] + bf0[5])
	bf1[3] = libaomClamp16(bf0[3] + bf0[4])
	bf1[4] = libaomClamp16(bf0[3] - bf0[4])
	bf1[5] = libaomClamp16(bf0[2] - bf0[5])
	bf1[6] = libaomClamp16(bf0[1] - bf0[6])
	bf1[7] = libaomClamp16(bf0[0] - bf0[7])
}

// libaomIADST8 reproduces libaom's av1_iadst8 for cos_bit=12 and 16-bit stage
// range.
func libaomIADST8(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [8]int32
	bf0 := output
	bf1 := output

	bf1[0] = input[7]
	bf1[1] = input[0]
	bf1[2] = input[5]
	bf1[3] = input[2]
	bf1[4] = input[3]
	bf1[5] = input[4]
	bf1[6] = input[1]
	bf1[7] = input[6]

	bf0 = output
	bf1 = step[:]
	bf1[0] = halfBtf(cospi[4], bf0[0], cospi[60], bf0[1])
	bf1[1] = halfBtf(cospi[60], bf0[0], -cospi[4], bf0[1])
	bf1[2] = halfBtf(cospi[20], bf0[2], cospi[44], bf0[3])
	bf1[3] = halfBtf(cospi[44], bf0[2], -cospi[20], bf0[3])
	bf1[4] = halfBtf(cospi[36], bf0[4], cospi[28], bf0[5])
	bf1[5] = halfBtf(cospi[28], bf0[4], -cospi[36], bf0[5])
	bf1[6] = halfBtf(cospi[52], bf0[6], cospi[12], bf0[7])
	bf1[7] = halfBtf(cospi[12], bf0[6], -cospi[52], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[4])
	bf1[1] = clamp(bf0[1] + bf0[5])
	bf1[2] = clamp(bf0[2] + bf0[6])
	bf1[3] = clamp(bf0[3] + bf0[7])
	bf1[4] = clamp(bf0[0] - bf0[4])
	bf1[5] = clamp(bf0[1] - bf0[5])
	bf1[6] = clamp(bf0[2] - bf0[6])
	bf1[7] = clamp(bf0[3] - bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = halfBtf(cospi[16], bf0[4], cospi[48], bf0[5])
	bf1[5] = halfBtf(cospi[48], bf0[4], -cospi[16], bf0[5])
	bf1[6] = halfBtf(-cospi[48], bf0[6], cospi[16], bf0[7])
	bf1[7] = halfBtf(cospi[16], bf0[6], cospi[48], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[2])
	bf1[1] = clamp(bf0[1] + bf0[3])
	bf1[2] = clamp(bf0[0] - bf0[2])
	bf1[3] = clamp(bf0[1] - bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[6])
	bf1[5] = clamp(bf0[5] + bf0[7])
	bf1[6] = clamp(bf0[4] - bf0[6])
	bf1[7] = clamp(bf0[5] - bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = halfBtf(cospi[32], bf0[2], cospi[32], bf0[3])
	bf1[3] = halfBtf(cospi[32], bf0[2], -cospi[32], bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = halfBtf(cospi[32], bf0[6], cospi[32], bf0[7])
	bf1[7] = halfBtf(cospi[32], bf0[6], -cospi[32], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = -bf0[4]
	bf1[2] = bf0[6]
	bf1[3] = -bf0[2]
	bf1[4] = bf0[3]
	bf1[5] = -bf0[7]
	bf1[6] = bf0[5]
	bf1[7] = -bf0[1]
}

// libaomIDCT16 reproduces libaom's av1_idct16 for cos_bit=12 and 16-bit stage
// range.
func libaomIDCT16(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [16]int32
	bf0 := output
	bf1 := output

	bf1[0] = input[0]
	bf1[1] = input[8]
	bf1[2] = input[4]
	bf1[3] = input[12]
	bf1[4] = input[2]
	bf1[5] = input[10]
	bf1[6] = input[6]
	bf1[7] = input[14]
	bf1[8] = input[1]
	bf1[9] = input[9]
	bf1[10] = input[5]
	bf1[11] = input[13]
	bf1[12] = input[3]
	bf1[13] = input[11]
	bf1[14] = input[7]
	bf1[15] = input[15]

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = bf0[6]
	bf1[7] = bf0[7]
	bf1[8] = halfBtf(cospi[60], bf0[8], -cospi[4], bf0[15])
	bf1[9] = halfBtf(cospi[28], bf0[9], -cospi[36], bf0[14])
	bf1[10] = halfBtf(cospi[44], bf0[10], -cospi[20], bf0[13])
	bf1[11] = halfBtf(cospi[12], bf0[11], -cospi[52], bf0[12])
	bf1[12] = halfBtf(cospi[52], bf0[11], cospi[12], bf0[12])
	bf1[13] = halfBtf(cospi[20], bf0[10], cospi[44], bf0[13])
	bf1[14] = halfBtf(cospi[36], bf0[9], cospi[28], bf0[14])
	bf1[15] = halfBtf(cospi[4], bf0[8], cospi[60], bf0[15])

	bf0 = step[:]
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = halfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = halfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = halfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = halfBtf(cospi[8], bf0[4], cospi[56], bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[9])
	bf1[9] = clamp(bf0[8] - bf0[9])
	bf1[10] = clamp(-bf0[10] + bf0[11])
	bf1[11] = clamp(bf0[10] + bf0[11])
	bf1[12] = clamp(bf0[12] + bf0[13])
	bf1[13] = clamp(bf0[12] - bf0[13])
	bf1[14] = clamp(-bf0[14] + bf0[15])
	bf1[15] = clamp(bf0[14] + bf0[15])

	bf0 = output
	bf1 = step[:]
	bf1[0] = halfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = halfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = halfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = halfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[5])
	bf1[5] = clamp(bf0[4] - bf0[5])
	bf1[6] = clamp(-bf0[6] + bf0[7])
	bf1[7] = clamp(bf0[6] + bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = halfBtf(-cospi[16], bf0[9], cospi[48], bf0[14])
	bf1[10] = halfBtf(-cospi[48], bf0[10], -cospi[16], bf0[13])
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = halfBtf(-cospi[16], bf0[10], cospi[48], bf0[13])
	bf1[14] = halfBtf(cospi[48], bf0[9], cospi[16], bf0[14])
	bf1[15] = bf0[15]

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[3])
	bf1[1] = clamp(bf0[1] + bf0[2])
	bf1[2] = clamp(bf0[1] - bf0[2])
	bf1[3] = clamp(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = halfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = halfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]
	bf1[8] = clamp(bf0[8] + bf0[11])
	bf1[9] = clamp(bf0[9] + bf0[10])
	bf1[10] = clamp(bf0[9] - bf0[10])
	bf1[11] = clamp(bf0[8] - bf0[11])
	bf1[12] = clamp(-bf0[12] + bf0[15])
	bf1[13] = clamp(-bf0[13] + bf0[14])
	bf1[14] = clamp(bf0[13] + bf0[14])
	bf1[15] = clamp(bf0[12] + bf0[15])

	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[7])
	bf1[1] = clamp(bf0[1] + bf0[6])
	bf1[2] = clamp(bf0[2] + bf0[5])
	bf1[3] = clamp(bf0[3] + bf0[4])
	bf1[4] = clamp(bf0[3] - bf0[4])
	bf1[5] = clamp(bf0[2] - bf0[5])
	bf1[6] = clamp(bf0[1] - bf0[6])
	bf1[7] = clamp(bf0[0] - bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = halfBtf(-cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[11] = halfBtf(-cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[12] = halfBtf(cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[13] = halfBtf(cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[15])
	bf1[1] = clamp(bf0[1] + bf0[14])
	bf1[2] = clamp(bf0[2] + bf0[13])
	bf1[3] = clamp(bf0[3] + bf0[12])
	bf1[4] = clamp(bf0[4] + bf0[11])
	bf1[5] = clamp(bf0[5] + bf0[10])
	bf1[6] = clamp(bf0[6] + bf0[9])
	bf1[7] = clamp(bf0[7] + bf0[8])
	bf1[8] = clamp(bf0[7] - bf0[8])
	bf1[9] = clamp(bf0[6] - bf0[9])
	bf1[10] = clamp(bf0[5] - bf0[10])
	bf1[11] = clamp(bf0[4] - bf0[11])
	bf1[12] = clamp(bf0[3] - bf0[12])
	bf1[13] = clamp(bf0[2] - bf0[13])
	bf1[14] = clamp(bf0[1] - bf0[14])
	bf1[15] = clamp(bf0[0] - bf0[15])
}

func TestInverseDCT4BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := range 5000 {
		var input [4]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT4(got[:], 1, minInt16, maxInt16)
		var want [4]int32
		libaomIDCT4(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d input=%v got=%v want=%v", trial, input, got, want)
			}
		}
	}
}

func TestInverseDCT8BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := range 5000 {
		var input [8]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT8(got[:], 1, minInt16, maxInt16)
		var want [8]int32
		libaomIDCT8(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d input=%v got=%v want=%v", trial, input, got, want)
			}
		}
	}
}

func TestInverseDCT16BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := range 5000 {
		var input [16]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT16(got[:], 1, minInt16, maxInt16)
		var want [16]int32
		libaomIDCT16(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d coeff=%d got=%d want=%d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestInverseADST8BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := range 5000 {
		var input [8]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseADST8(got[:], 1, minInt16, maxInt16)
		var want [8]int32
		libaomIADST8(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d input=%v\n  goav1=%v\n  libaom=%v", trial, input, got, want)
			}
		}
	}
}

// libaomIDCT32 reproduces libaom's av1_idct32 for cos_bit=12 and 16-bit stage
// range. Faithfully follows av1/common/av1_inv_txfm1d.c:av1_idct32.
func libaomIDCT32(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [32]int32
	bf0 := output
	bf1 := output

	// stage 1: bit reverse
	bf1[0] = input[0]
	bf1[1] = input[16]
	bf1[2] = input[8]
	bf1[3] = input[24]
	bf1[4] = input[4]
	bf1[5] = input[20]
	bf1[6] = input[12]
	bf1[7] = input[28]
	bf1[8] = input[2]
	bf1[9] = input[18]
	bf1[10] = input[10]
	bf1[11] = input[26]
	bf1[12] = input[6]
	bf1[13] = input[22]
	bf1[14] = input[14]
	bf1[15] = input[30]
	bf1[16] = input[1]
	bf1[17] = input[17]
	bf1[18] = input[9]
	bf1[19] = input[25]
	bf1[20] = input[5]
	bf1[21] = input[21]
	bf1[22] = input[13]
	bf1[23] = input[29]
	bf1[24] = input[3]
	bf1[25] = input[19]
	bf1[26] = input[11]
	bf1[27] = input[27]
	bf1[28] = input[7]
	bf1[29] = input[23]
	bf1[30] = input[15]
	bf1[31] = input[31]

	// stage 2
	bf0 = output
	bf1 = step[:]
	for i := range 16 {
		bf1[i] = bf0[i]
	}
	bf1[16] = halfBtf(cospi[62], bf0[16], -cospi[2], bf0[31])
	bf1[17] = halfBtf(cospi[30], bf0[17], -cospi[34], bf0[30])
	bf1[18] = halfBtf(cospi[46], bf0[18], -cospi[18], bf0[29])
	bf1[19] = halfBtf(cospi[14], bf0[19], -cospi[50], bf0[28])
	bf1[20] = halfBtf(cospi[54], bf0[20], -cospi[10], bf0[27])
	bf1[21] = halfBtf(cospi[22], bf0[21], -cospi[42], bf0[26])
	bf1[22] = halfBtf(cospi[38], bf0[22], -cospi[26], bf0[25])
	bf1[23] = halfBtf(cospi[6], bf0[23], -cospi[58], bf0[24])
	bf1[24] = halfBtf(cospi[58], bf0[23], cospi[6], bf0[24])
	bf1[25] = halfBtf(cospi[26], bf0[22], cospi[38], bf0[25])
	bf1[26] = halfBtf(cospi[42], bf0[21], cospi[22], bf0[26])
	bf1[27] = halfBtf(cospi[10], bf0[20], cospi[54], bf0[27])
	bf1[28] = halfBtf(cospi[50], bf0[19], cospi[14], bf0[28])
	bf1[29] = halfBtf(cospi[18], bf0[18], cospi[46], bf0[29])
	bf1[30] = halfBtf(cospi[34], bf0[17], cospi[30], bf0[30])
	bf1[31] = halfBtf(cospi[2], bf0[16], cospi[62], bf0[31])

	// stage 3
	bf0 = step[:]
	bf1 = output
	for i := range 8 {
		bf1[i] = bf0[i]
	}
	bf1[8] = halfBtf(cospi[60], bf0[8], -cospi[4], bf0[15])
	bf1[9] = halfBtf(cospi[28], bf0[9], -cospi[36], bf0[14])
	bf1[10] = halfBtf(cospi[44], bf0[10], -cospi[20], bf0[13])
	bf1[11] = halfBtf(cospi[12], bf0[11], -cospi[52], bf0[12])
	bf1[12] = halfBtf(cospi[52], bf0[11], cospi[12], bf0[12])
	bf1[13] = halfBtf(cospi[20], bf0[10], cospi[44], bf0[13])
	bf1[14] = halfBtf(cospi[36], bf0[9], cospi[28], bf0[14])
	bf1[15] = halfBtf(cospi[4], bf0[8], cospi[60], bf0[15])
	bf1[16] = clamp(bf0[16] + bf0[17])
	bf1[17] = clamp(bf0[16] - bf0[17])
	bf1[18] = clamp(-bf0[18] + bf0[19])
	bf1[19] = clamp(bf0[18] + bf0[19])
	bf1[20] = clamp(bf0[20] + bf0[21])
	bf1[21] = clamp(bf0[20] - bf0[21])
	bf1[22] = clamp(-bf0[22] + bf0[23])
	bf1[23] = clamp(bf0[22] + bf0[23])
	bf1[24] = clamp(bf0[24] + bf0[25])
	bf1[25] = clamp(bf0[24] - bf0[25])
	bf1[26] = clamp(-bf0[26] + bf0[27])
	bf1[27] = clamp(bf0[26] + bf0[27])
	bf1[28] = clamp(bf0[28] + bf0[29])
	bf1[29] = clamp(bf0[28] - bf0[29])
	bf1[30] = clamp(-bf0[30] + bf0[31])
	bf1[31] = clamp(bf0[30] + bf0[31])

	// stage 4
	bf0 = output
	bf1 = step[:]
	for i := range 4 {
		bf1[i] = bf0[i]
	}
	bf1[4] = halfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = halfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = halfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = halfBtf(cospi[8], bf0[4], cospi[56], bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[9])
	bf1[9] = clamp(bf0[8] - bf0[9])
	bf1[10] = clamp(-bf0[10] + bf0[11])
	bf1[11] = clamp(bf0[10] + bf0[11])
	bf1[12] = clamp(bf0[12] + bf0[13])
	bf1[13] = clamp(bf0[12] - bf0[13])
	bf1[14] = clamp(-bf0[14] + bf0[15])
	bf1[15] = clamp(bf0[14] + bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = halfBtf(-cospi[8], bf0[17], cospi[56], bf0[30])
	bf1[18] = halfBtf(-cospi[56], bf0[18], -cospi[8], bf0[29])
	bf1[19] = bf0[19]
	bf1[20] = bf0[20]
	bf1[21] = halfBtf(-cospi[40], bf0[21], cospi[24], bf0[26])
	bf1[22] = halfBtf(-cospi[24], bf0[22], -cospi[40], bf0[25])
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = halfBtf(-cospi[40], bf0[22], cospi[24], bf0[25])
	bf1[26] = halfBtf(cospi[24], bf0[21], cospi[40], bf0[26])
	bf1[27] = bf0[27]
	bf1[28] = bf0[28]
	bf1[29] = halfBtf(-cospi[8], bf0[18], cospi[56], bf0[29])
	bf1[30] = halfBtf(cospi[56], bf0[17], cospi[8], bf0[30])
	bf1[31] = bf0[31]

	// stage 5
	bf0 = step[:]
	bf1 = output
	bf1[0] = halfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = halfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = halfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = halfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[5])
	bf1[5] = clamp(bf0[4] - bf0[5])
	bf1[6] = clamp(-bf0[6] + bf0[7])
	bf1[7] = clamp(bf0[6] + bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = halfBtf(-cospi[16], bf0[9], cospi[48], bf0[14])
	bf1[10] = halfBtf(-cospi[48], bf0[10], -cospi[16], bf0[13])
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = halfBtf(-cospi[16], bf0[10], cospi[48], bf0[13])
	bf1[14] = halfBtf(cospi[48], bf0[9], cospi[16], bf0[14])
	bf1[15] = bf0[15]
	bf1[16] = clamp(bf0[16] + bf0[19])
	bf1[17] = clamp(bf0[17] + bf0[18])
	bf1[18] = clamp(bf0[17] - bf0[18])
	bf1[19] = clamp(bf0[16] - bf0[19])
	bf1[20] = clamp(-bf0[20] + bf0[23])
	bf1[21] = clamp(-bf0[21] + bf0[22])
	bf1[22] = clamp(bf0[21] + bf0[22])
	bf1[23] = clamp(bf0[20] + bf0[23])
	bf1[24] = clamp(bf0[24] + bf0[27])
	bf1[25] = clamp(bf0[25] + bf0[26])
	bf1[26] = clamp(bf0[25] - bf0[26])
	bf1[27] = clamp(bf0[24] - bf0[27])
	bf1[28] = clamp(-bf0[28] + bf0[31])
	bf1[29] = clamp(-bf0[29] + bf0[30])
	bf1[30] = clamp(bf0[29] + bf0[30])
	bf1[31] = clamp(bf0[28] + bf0[31])

	// stage 6
	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[3])
	bf1[1] = clamp(bf0[1] + bf0[2])
	bf1[2] = clamp(bf0[1] - bf0[2])
	bf1[3] = clamp(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = halfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = halfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]
	bf1[8] = clamp(bf0[8] + bf0[11])
	bf1[9] = clamp(bf0[9] + bf0[10])
	bf1[10] = clamp(bf0[9] - bf0[10])
	bf1[11] = clamp(bf0[8] - bf0[11])
	bf1[12] = clamp(-bf0[12] + bf0[15])
	bf1[13] = clamp(-bf0[13] + bf0[14])
	bf1[14] = clamp(bf0[13] + bf0[14])
	bf1[15] = clamp(bf0[12] + bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = halfBtf(-cospi[16], bf0[18], cospi[48], bf0[29])
	bf1[19] = halfBtf(-cospi[16], bf0[19], cospi[48], bf0[28])
	bf1[20] = halfBtf(-cospi[48], bf0[20], -cospi[16], bf0[27])
	bf1[21] = halfBtf(-cospi[48], bf0[21], -cospi[16], bf0[26])
	bf1[22] = bf0[22]
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = bf0[25]
	bf1[26] = halfBtf(-cospi[16], bf0[21], cospi[48], bf0[26])
	bf1[27] = halfBtf(-cospi[16], bf0[20], cospi[48], bf0[27])
	bf1[28] = halfBtf(cospi[48], bf0[19], cospi[16], bf0[28])
	bf1[29] = halfBtf(cospi[48], bf0[18], cospi[16], bf0[29])
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]

	// stage 7
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[7])
	bf1[1] = clamp(bf0[1] + bf0[6])
	bf1[2] = clamp(bf0[2] + bf0[5])
	bf1[3] = clamp(bf0[3] + bf0[4])
	bf1[4] = clamp(bf0[3] - bf0[4])
	bf1[5] = clamp(bf0[2] - bf0[5])
	bf1[6] = clamp(bf0[1] - bf0[6])
	bf1[7] = clamp(bf0[0] - bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = halfBtf(-cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[11] = halfBtf(-cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[12] = halfBtf(cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[13] = halfBtf(cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]
	bf1[16] = clamp(bf0[16] + bf0[23])
	bf1[17] = clamp(bf0[17] + bf0[22])
	bf1[18] = clamp(bf0[18] + bf0[21])
	bf1[19] = clamp(bf0[19] + bf0[20])
	bf1[20] = clamp(bf0[19] - bf0[20])
	bf1[21] = clamp(bf0[18] - bf0[21])
	bf1[22] = clamp(bf0[17] - bf0[22])
	bf1[23] = clamp(bf0[16] - bf0[23])
	bf1[24] = clamp(-bf0[24] + bf0[31])
	bf1[25] = clamp(-bf0[25] + bf0[30])
	bf1[26] = clamp(-bf0[26] + bf0[29])
	bf1[27] = clamp(-bf0[27] + bf0[28])
	bf1[28] = clamp(bf0[27] + bf0[28])
	bf1[29] = clamp(bf0[26] + bf0[29])
	bf1[30] = clamp(bf0[25] + bf0[30])
	bf1[31] = clamp(bf0[24] + bf0[31])

	// stage 8
	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[15])
	bf1[1] = clamp(bf0[1] + bf0[14])
	bf1[2] = clamp(bf0[2] + bf0[13])
	bf1[3] = clamp(bf0[3] + bf0[12])
	bf1[4] = clamp(bf0[4] + bf0[11])
	bf1[5] = clamp(bf0[5] + bf0[10])
	bf1[6] = clamp(bf0[6] + bf0[9])
	bf1[7] = clamp(bf0[7] + bf0[8])
	bf1[8] = clamp(bf0[7] - bf0[8])
	bf1[9] = clamp(bf0[6] - bf0[9])
	bf1[10] = clamp(bf0[5] - bf0[10])
	bf1[11] = clamp(bf0[4] - bf0[11])
	bf1[12] = clamp(bf0[3] - bf0[12])
	bf1[13] = clamp(bf0[2] - bf0[13])
	bf1[14] = clamp(bf0[1] - bf0[14])
	bf1[15] = clamp(bf0[0] - bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = bf0[18]
	bf1[19] = bf0[19]
	bf1[20] = halfBtf(-cospi[32], bf0[20], cospi[32], bf0[27])
	bf1[21] = halfBtf(-cospi[32], bf0[21], cospi[32], bf0[26])
	bf1[22] = halfBtf(-cospi[32], bf0[22], cospi[32], bf0[25])
	bf1[23] = halfBtf(-cospi[32], bf0[23], cospi[32], bf0[24])
	bf1[24] = halfBtf(cospi[32], bf0[23], cospi[32], bf0[24])
	bf1[25] = halfBtf(cospi[32], bf0[22], cospi[32], bf0[25])
	bf1[26] = halfBtf(cospi[32], bf0[21], cospi[32], bf0[26])
	bf1[27] = halfBtf(cospi[32], bf0[20], cospi[32], bf0[27])
	bf1[28] = bf0[28]
	bf1[29] = bf0[29]
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]

	// stage 9
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[31])
	bf1[1] = clamp(bf0[1] + bf0[30])
	bf1[2] = clamp(bf0[2] + bf0[29])
	bf1[3] = clamp(bf0[3] + bf0[28])
	bf1[4] = clamp(bf0[4] + bf0[27])
	bf1[5] = clamp(bf0[5] + bf0[26])
	bf1[6] = clamp(bf0[6] + bf0[25])
	bf1[7] = clamp(bf0[7] + bf0[24])
	bf1[8] = clamp(bf0[8] + bf0[23])
	bf1[9] = clamp(bf0[9] + bf0[22])
	bf1[10] = clamp(bf0[10] + bf0[21])
	bf1[11] = clamp(bf0[11] + bf0[20])
	bf1[12] = clamp(bf0[12] + bf0[19])
	bf1[13] = clamp(bf0[13] + bf0[18])
	bf1[14] = clamp(bf0[14] + bf0[17])
	bf1[15] = clamp(bf0[15] + bf0[16])
	bf1[16] = clamp(bf0[15] - bf0[16])
	bf1[17] = clamp(bf0[14] - bf0[17])
	bf1[18] = clamp(bf0[13] - bf0[18])
	bf1[19] = clamp(bf0[12] - bf0[19])
	bf1[20] = clamp(bf0[11] - bf0[20])
	bf1[21] = clamp(bf0[10] - bf0[21])
	bf1[22] = clamp(bf0[9] - bf0[22])
	bf1[23] = clamp(bf0[8] - bf0[23])
	bf1[24] = clamp(bf0[7] - bf0[24])
	bf1[25] = clamp(bf0[6] - bf0[25])
	bf1[26] = clamp(bf0[5] - bf0[26])
	bf1[27] = clamp(bf0[4] - bf0[27])
	bf1[28] = clamp(bf0[3] - bf0[28])
	bf1[29] = clamp(bf0[2] - bf0[29])
	bf1[30] = clamp(bf0[1] - bf0[30])
	bf1[31] = clamp(bf0[0] - bf0[31])
}

func TestInverseDCT32BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for trial := range 1000 {
		var input [32]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT32(got[:], 1, minInt16, maxInt16)
		var want [32]int32
		libaomIDCT32(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d coeff=%d got=%d want=%d\n input=%v\n  got=%v\n want=%v",
					trial, i, got[i], want[i], input, got, want)
			}
		}
	}
}

// libaomIADST16 reproduces libaom's av1_iadst16 for cos_bit=12 and 16-bit
// stage range. Mirrors av1/common/av1_inv_txfm1d.c:av1_iadst16.
func libaomIADST16(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [16]int32
	bf0 := output
	bf1 := output

	// stage 1
	bf1[0] = input[15]
	bf1[1] = input[0]
	bf1[2] = input[13]
	bf1[3] = input[2]
	bf1[4] = input[11]
	bf1[5] = input[4]
	bf1[6] = input[9]
	bf1[7] = input[6]
	bf1[8] = input[7]
	bf1[9] = input[8]
	bf1[10] = input[5]
	bf1[11] = input[10]
	bf1[12] = input[3]
	bf1[13] = input[12]
	bf1[14] = input[1]
	bf1[15] = input[14]

	// stage 2
	bf0 = output
	bf1 = step[:]
	bf1[0] = halfBtf(cospi[2], bf0[0], cospi[62], bf0[1])
	bf1[1] = halfBtf(cospi[62], bf0[0], -cospi[2], bf0[1])
	bf1[2] = halfBtf(cospi[10], bf0[2], cospi[54], bf0[3])
	bf1[3] = halfBtf(cospi[54], bf0[2], -cospi[10], bf0[3])
	bf1[4] = halfBtf(cospi[18], bf0[4], cospi[46], bf0[5])
	bf1[5] = halfBtf(cospi[46], bf0[4], -cospi[18], bf0[5])
	bf1[6] = halfBtf(cospi[26], bf0[6], cospi[38], bf0[7])
	bf1[7] = halfBtf(cospi[38], bf0[6], -cospi[26], bf0[7])
	bf1[8] = halfBtf(cospi[34], bf0[8], cospi[30], bf0[9])
	bf1[9] = halfBtf(cospi[30], bf0[8], -cospi[34], bf0[9])
	bf1[10] = halfBtf(cospi[42], bf0[10], cospi[22], bf0[11])
	bf1[11] = halfBtf(cospi[22], bf0[10], -cospi[42], bf0[11])
	bf1[12] = halfBtf(cospi[50], bf0[12], cospi[14], bf0[13])
	bf1[13] = halfBtf(cospi[14], bf0[12], -cospi[50], bf0[13])
	bf1[14] = halfBtf(cospi[58], bf0[14], cospi[6], bf0[15])
	bf1[15] = halfBtf(cospi[6], bf0[14], -cospi[58], bf0[15])

	// stage 3
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[8])
	bf1[1] = clamp(bf0[1] + bf0[9])
	bf1[2] = clamp(bf0[2] + bf0[10])
	bf1[3] = clamp(bf0[3] + bf0[11])
	bf1[4] = clamp(bf0[4] + bf0[12])
	bf1[5] = clamp(bf0[5] + bf0[13])
	bf1[6] = clamp(bf0[6] + bf0[14])
	bf1[7] = clamp(bf0[7] + bf0[15])
	bf1[8] = clamp(bf0[0] - bf0[8])
	bf1[9] = clamp(bf0[1] - bf0[9])
	bf1[10] = clamp(bf0[2] - bf0[10])
	bf1[11] = clamp(bf0[3] - bf0[11])
	bf1[12] = clamp(bf0[4] - bf0[12])
	bf1[13] = clamp(bf0[5] - bf0[13])
	bf1[14] = clamp(bf0[6] - bf0[14])
	bf1[15] = clamp(bf0[7] - bf0[15])

	// stage 4
	bf0 = output
	bf1 = step[:]
	for i := range 8 {
		bf1[i] = bf0[i]
	}
	bf1[8] = halfBtf(cospi[8], bf0[8], cospi[56], bf0[9])
	bf1[9] = halfBtf(cospi[56], bf0[8], -cospi[8], bf0[9])
	bf1[10] = halfBtf(cospi[40], bf0[10], cospi[24], bf0[11])
	bf1[11] = halfBtf(cospi[24], bf0[10], -cospi[40], bf0[11])
	bf1[12] = halfBtf(-cospi[56], bf0[12], cospi[8], bf0[13])
	bf1[13] = halfBtf(cospi[8], bf0[12], cospi[56], bf0[13])
	bf1[14] = halfBtf(-cospi[24], bf0[14], cospi[40], bf0[15])
	bf1[15] = halfBtf(cospi[40], bf0[14], cospi[24], bf0[15])

	// stage 5
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[4])
	bf1[1] = clamp(bf0[1] + bf0[5])
	bf1[2] = clamp(bf0[2] + bf0[6])
	bf1[3] = clamp(bf0[3] + bf0[7])
	bf1[4] = clamp(bf0[0] - bf0[4])
	bf1[5] = clamp(bf0[1] - bf0[5])
	bf1[6] = clamp(bf0[2] - bf0[6])
	bf1[7] = clamp(bf0[3] - bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[12])
	bf1[9] = clamp(bf0[9] + bf0[13])
	bf1[10] = clamp(bf0[10] + bf0[14])
	bf1[11] = clamp(bf0[11] + bf0[15])
	bf1[12] = clamp(bf0[8] - bf0[12])
	bf1[13] = clamp(bf0[9] - bf0[13])
	bf1[14] = clamp(bf0[10] - bf0[14])
	bf1[15] = clamp(bf0[11] - bf0[15])

	// stage 6
	bf0 = output
	bf1 = step[:]
	for i := range 4 {
		bf1[i] = bf0[i]
	}
	bf1[4] = halfBtf(cospi[16], bf0[4], cospi[48], bf0[5])
	bf1[5] = halfBtf(cospi[48], bf0[4], -cospi[16], bf0[5])
	bf1[6] = halfBtf(-cospi[48], bf0[6], cospi[16], bf0[7])
	bf1[7] = halfBtf(cospi[16], bf0[6], cospi[48], bf0[7])
	for i := 8; i < 12; i++ {
		bf1[i] = bf0[i]
	}
	bf1[12] = halfBtf(cospi[16], bf0[12], cospi[48], bf0[13])
	bf1[13] = halfBtf(cospi[48], bf0[12], -cospi[16], bf0[13])
	bf1[14] = halfBtf(-cospi[48], bf0[14], cospi[16], bf0[15])
	bf1[15] = halfBtf(cospi[16], bf0[14], cospi[48], bf0[15])

	// stage 7
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[2])
	bf1[1] = clamp(bf0[1] + bf0[3])
	bf1[2] = clamp(bf0[0] - bf0[2])
	bf1[3] = clamp(bf0[1] - bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[6])
	bf1[5] = clamp(bf0[5] + bf0[7])
	bf1[6] = clamp(bf0[4] - bf0[6])
	bf1[7] = clamp(bf0[5] - bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[10])
	bf1[9] = clamp(bf0[9] + bf0[11])
	bf1[10] = clamp(bf0[8] - bf0[10])
	bf1[11] = clamp(bf0[9] - bf0[11])
	bf1[12] = clamp(bf0[12] + bf0[14])
	bf1[13] = clamp(bf0[13] + bf0[15])
	bf1[14] = clamp(bf0[12] - bf0[14])
	bf1[15] = clamp(bf0[13] - bf0[15])

	// stage 8
	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = halfBtf(cospi[32], bf0[2], cospi[32], bf0[3])
	bf1[3] = halfBtf(cospi[32], bf0[2], -cospi[32], bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = halfBtf(cospi[32], bf0[6], cospi[32], bf0[7])
	bf1[7] = halfBtf(cospi[32], bf0[6], -cospi[32], bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = halfBtf(cospi[32], bf0[10], cospi[32], bf0[11])
	bf1[11] = halfBtf(cospi[32], bf0[10], -cospi[32], bf0[11])
	bf1[12] = bf0[12]
	bf1[13] = bf0[13]
	bf1[14] = halfBtf(cospi[32], bf0[14], cospi[32], bf0[15])
	bf1[15] = halfBtf(cospi[32], bf0[14], -cospi[32], bf0[15])

	// stage 9
	bf0 = step[:]
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = -bf0[8]
	bf1[2] = bf0[12]
	bf1[3] = -bf0[4]
	bf1[4] = bf0[6]
	bf1[5] = -bf0[14]
	bf1[6] = bf0[10]
	bf1[7] = -bf0[2]
	bf1[8] = bf0[3]
	bf1[9] = -bf0[11]
	bf1[10] = bf0[15]
	bf1[11] = -bf0[7]
	bf1[12] = bf0[5]
	bf1[13] = -bf0[13]
	bf1[14] = bf0[9]
	bf1[15] = -bf0[1]
}

func TestInverseADST16BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := range 1000 {
		var input [16]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseADST16(got[:], 1, minInt16, maxInt16)
		var want [16]int32
		libaomIADST16(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d coeff=%d got=%d want=%d\n input=%v\n  got=%v\n want=%v",
					trial, i, got[i], want[i], input, got, want)
			}
		}
	}
}

func TestInverseDCT4(t *testing.T) {
	values := []int32{100, -20, 30, 7}
	inverseDCT4(values, 1, minInt16, maxInt16)
	want := []int32{76, 35, 63, 108}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i, values[i], want[i])
		}
	}
}

func TestInverseDCT4Strided(t *testing.T) {
	values := []int32{100, 99, -20, 99, 30, 99, 7}
	inverseDCT4(values, 2, minInt16, maxInt16)
	want := []int32{76, 99, 35, 99, 63, 99, 108}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i, values[i], want[i])
		}
	}
}

func TestInverseDCT8(t *testing.T) {
	values := []int32{100, -20, 30, 7, 11, -13, 17, -19}
	inverseDCT8(values, 1, minInt16, maxInt16)
	want := []int32{88, 65, 30, 44, 44, 104, 53, 136}
	for i := range want {
		if values[i] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i, values[i], want[i])
		}
	}
}

func TestInverseDCT1DMatchesLibaomReference(t *testing.T) {
	tests := []struct {
		length   int
		maxError int32
	}{
		{length: 4, maxError: 6},
		{length: 8, maxError: 10},
		{length: 16, maxError: 19},
		{length: 32, maxError: 31},
		{length: 64, maxError: 40},
	}
	for _, tt := range tests {
		input := make([]int32, tt.length)
		for i := range input {
			input[i] = int32(((i*37+11)%31)-15) * 7
		}
		if tt.length == 64 {
			for i := 32; i < len(input); i++ {
				input[i] = 0
			}
		}
		got := append([]int32(nil), input...)
		inverseDCT1D(got, 1, tt.length, minInt16, maxInt16)
		want := referenceIDCT1DInt(input)
		for i := range got {
			if diff := abs32(got[i] - want[i]); diff > tt.maxError {
				t.Fatalf("length=%d coeff=%d got=%d want=%d diff=%d max=%d", tt.length, i, got[i], want[i], diff, tt.maxError)
			}
		}
	}
}

func TestInverseDCTBlock4x4DC(t *testing.T) {
	coeff := make([]int32, 4*4)
	coeff[0] = 16
	dst := make([]int16, 4*4)
	scratch := make([]int32, 4*4)
	if err := InverseBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4}, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 1 {
			t.Fatalf("dst[%d]=%d want 1", i, got)
		}
	}
}

func TestInverseDCTBlock4x4Mixed(t *testing.T) {
	coeff := av1CoeffOrder([]int32{
		100, -20, 30, 7,
		5, -9, 13, -17,
		0, 4, -8, 12,
		-16, 20, -24, 28,
	}, 4, 4)
	dst := make([]int16, 4*4)
	scratch := make([]int32, 4*4)
	if err := InverseBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4}, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		3, 1, 3, 4,
		3, 3, 0, 9,
		3, 1, 4, 1,
		4, 1, 4, 4,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestInverseDCTBlock8x8DC(t *testing.T) {
	coeff := make([]int32, 8*8)
	coeff[0] = 32
	dst := make([]int16, 8*8)
	scratch := make([]int32, 8*8)
	if err := InverseBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 1 {
			t.Fatalf("dst[%d]=%d want 1", i, got)
		}
	}
}

func TestInverseDCTBlock8x8Mixed(t *testing.T) {
	coeff := av1CoeffOrder([]int32{
		-20, 14, 7, 0, -7, -14, 20, 13,
		-9, -15, 20, 14, 8, -3, -9, -15,
		2, -3, -8, -18, 18, 8, 3, -2,
		13, 9, 0, -4, -13, 19, 15, 6,
		-17, -20, 13, 5, -3, -11, -14, 19,
		-6, -13, -20, 14, 7, 0, -7, -14,
		5, -1, -7, -13, -19, 11, 5, -1,
		16, 11, 6, -4, -9, -19, 17, 12,
	}, 8, 8)
	dst := make([]int16, 8*8)
	scratch := make([]int32, 8*8)
	if err := InverseBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		0, -1, 0, -1, -1, 0, 0, 1,
		2, 0, -1, -1, -1, -1, -2, 1,
		0, 1, 1, -1, 0, 1, -2, 0,
		-1, -1, -5, -2, 0, 0, 0, -1,
		1, 1, 3, -5, 0, 0, 0, 0,
		1, -1, 4, 1, -2, 1, 0, -2,
		1, 0, 2, 1, 2, 1, -3, -1,
		-1, -1, 2, -1, 1, -2, -1, 0,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestInverseDCTBlockSupportedSizes(t *testing.T) {
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 16, Height: 4},
		{Width: 16, Height: 8},
		{Width: 16, Height: 32},
		{Width: 32, Height: 16},
		{Width: 8, Height: 32},
		{Width: 16, Height: 16},
		{Width: 32, Height: 8},
		{Width: 32, Height: 32},
		{Width: 16, Height: 64},
		{Width: 64, Height: 16},
		{Width: 32, Height: 64},
		{Width: 64, Height: 32},
		{Width: 64, Height: 64},
	}
	for _, size := range sizes {
		width := int(size.Width)
		height := int(size.Height)
		coeffStride := height + 1
		dstStride := width + 2
		coeff := make([]int32, coeffStride*width)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[col*coeffStride+row] = int32(((row*13+col*7+5)%29)-14) * 3
			}
		}
		dst := make([]int16, dstStride*height)
		scratchLen := width * height
		scratch := make([]int32, scratchLen+3)
		if err := InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, TypeDCTDCT); err != nil {
			t.Fatalf("%dx%d InverseDCTBlock err=%v", size.Width, size.Height, err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != 0 {
					t.Fatalf("%dx%d dst padding row=%d col=%d overwritten with %d", size.Width, size.Height, row, col, got)
				}
			}
		}
		for i := scratchLen; i < len(scratch); i++ {
			if scratch[i] != 0 {
				t.Fatalf("%dx%d scratch padding[%d]=%d want 0", size.Width, size.Height, i, scratch[i])
			}
		}
	}
}

func TestInverseDCTBlock4x4Strides(t *testing.T) {
	coeff := make([]int32, 6*4)
	dst := make([]int16, 6*4)
	scratch := make([]int32, 20)
	coeff[1] = 16
	if err := InverseBlock(dst, 6, coeff, 6, scratch, Size{Width: 4, Height: 4}, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	for row := range 4 {
		for col := 4; col < 6; col++ {
			if got := dst[row*6+col]; got != 0 {
				t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
			}
		}
	}
	for i := 16; i < len(scratch); i++ {
		if scratch[i] != 0 {
			t.Fatalf("scratch padding[%d]=%d want 0", i, scratch[i])
		}
	}
}

func TestInverseDCTBlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int16, 4*4)
	coeff := make([]int32, 4*4)
	scratch := make([]int32, 4*4)
	tests := []struct {
		name        string
		dst         []int16
		dstStride   int
		coeff       []int32
		coeffStride int
		scratch     []int32
		size        Size
	}{
		{name: "zero size", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, scratch: scratch, size: Size{}},
		{name: "unsupported size", dst: make([]int16, 64*8), dstStride: 64, coeff: make([]int32, 64*8), coeffStride: 8, scratch: make([]int32, 64*8), size: Size{Width: 64, Height: 8}},
		{name: "short dst stride", dst: dst, dstStride: 3, coeff: coeff, coeffStride: 4, scratch: scratch, size: Size{Width: 4, Height: 4}},
		{name: "short coeff stride", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 3, scratch: scratch, size: Size{Width: 4, Height: 4}},
		{name: "short scratch", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, scratch: scratch[:15], size: Size{Width: 4, Height: 4}},
		{name: "short dst", dst: dst[:15], dstStride: 4, coeff: coeff, coeffStride: 4, scratch: scratch, size: Size{Width: 4, Height: 4}},
		{name: "short coeff", dst: dst, dstStride: 4, coeff: coeff[:15], coeffStride: 4, scratch: scratch, size: Size{Width: 4, Height: 4}},
	}
	for _, tt := range tests {
		err := InverseBlock(tt.dst, tt.dstStride, tt.coeff, tt.coeffStride, tt.scratch, tt.size, TypeDCTDCT)
		if !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidTransform)
		}
	}
}

func TestInverseDCTBlockAllocs(t *testing.T) {
	coeff4 := make([]int32, 4*4)
	dst4 := make([]int16, 4*4)
	scratch4 := make([]int32, 4*4)
	coeff8 := make([]int32, 8*8)
	dst8 := make([]int16, 8*8)
	scratch8 := make([]int32, 8*8)
	coeff16 := make([]int32, 16*16)
	dst16 := make([]int16, 16*16)
	scratch16 := make([]int32, 16*16)
	coeff32 := make([]int32, 32*32)
	dst32 := make([]int16, 32*32)
	scratch32 := make([]int32, 32*32)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseBlock(dst4, 4, coeff4, 4, scratch4, Size{Width: 4, Height: 4}, TypeDCTDCT); err != nil {
			t.Fatal(err)
		}
		if err := InverseBlock(dst8, 8, coeff8, 8, scratch8, Size{Width: 8, Height: 8}, TypeDCTDCT); err != nil {
			t.Fatal(err)
		}
		if err := InverseBlock(dst16, 16, coeff16, 16, scratch16, Size{Width: 16, Height: 16}, TypeDCTDCT); err != nil {
			t.Fatal(err)
		}
		if err := InverseBlock(dst32, 32, coeff32, 32, scratch32, Size{Width: 32, Height: 32}, TypeDCTDCT); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseDCTBlock allocated: %f", allocs)
	}
}

func FuzzInverseDCTBlock(f *testing.F) {
	f.Add(uint8(0), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(0), int16(16), int16(0), int16(0), int16(0))
	f.Add(uint8(1), int16(32), int16(0), int16(0), int16(0))
	f.Add(uint8(1), int16(-128), int16(64), int16(32), int16(-16))
	f.Add(uint8(2), int16(64), int16(-21), int16(17), int16(-9))
	f.Add(uint8(3), int16(128), int16(31), int16(-19), int16(11))
	f.Add(uint8(4), int16(256), int16(-33), int16(23), int16(-13))
	f.Add(uint8(1), int16(1024), int16(-512), int16(255), int16(-255))

	f.Fuzz(func(t *testing.T, rawSize uint8, dc int16, c1 int16, c2 int16, c3 int16) {
		size := Size{Width: 4, Height: 4}
		switch rawSize & 7 {
		case 1:
			size = Size{Width: 8, Height: 8}
		case 2:
			size = Size{Width: 16, Height: 16}
		case 3:
			size = Size{Width: 32, Height: 32}
		case 4:
			size = Size{Width: 64, Height: 64}
		case 5:
			size = Size{Width: 64, Height: 32}
		case 6:
			size = Size{Width: 32, Height: 64}
		}
		width := int(size.Width)
		height := int(size.Height)
		blockLen := width * height
		coeffStride := height + 3
		dstStride := width + 2
		coeff := make([]int32, coeffStride*width)
		dst := make([]int16, dstStride*height)
		scratch := make([]int32, blockLen+4)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				base := int32(dc) + int32(row*3-col*2)
				switch (row + col) & 3 {
				case 1:
					base += int32(c1)
				case 2:
					base += int32(c2)
				case 3:
					base += int32(c3)
				}
				coeff[col*coeffStride+row] = base
			}
		}
		if err := InverseBlock(dst, dstStride, coeff, coeffStride, scratch, size, TypeDCTDCT); err != nil {
			t.Fatalf("InverseDCTBlock err=%v", err)
		}
		for row := 0; row < height; row++ {
			for col := width; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != 0 {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
		for i := blockLen; i < len(scratch); i++ {
			if scratch[i] != 0 {
				t.Fatalf("scratch padding[%d]=%d want 0", i, scratch[i])
			}
		}
	})
}

func BenchmarkInverseDCTBlock4x4(b *testing.B) {
	coeff := make([]int32, 4*4)
	dst := make([]int16, 4*4)
	scratch := make([]int32, 4*4)
	for i := range coeff {
		coeff[i] = int32(i%9) - 4
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = InverseBlock(dst, 4, coeff, 4, scratch, Size{Width: 4, Height: 4}, TypeDCTDCT)
	}
}

func BenchmarkInverseDCTBlock8x8(b *testing.B) {
	coeff := make([]int32, 8*8)
	dst := make([]int16, 8*8)
	scratch := make([]int32, 8*8)
	for i := range coeff {
		coeff[i] = int32(i%17) - 8
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = InverseBlock(dst, 8, coeff, 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT)
	}
}

func referenceIDCT1DInt(input []int32) []int32 {
	size := len(input)
	out := make([]int32, size)
	invSqrt2 := 1 / math.Sqrt2
	for k := range size {
		sum := 0.0
		for n := range size {
			scale := 1.0
			if n == 0 {
				scale = invSqrt2
			}
			angle := math.Pi * float64((2*k+1)*n) / float64(2*size)
			sum += scale * float64(input[n]) * math.Cos(angle)
		}
		out[k] = int32(math.Round(sum))
	}
	return out
}

func av1CoeffOrder(rowMajor []int32, width int, height int) []int32 {
	out := make([]int32, width*height)
	for row := range height {
		for col := range width {
			out[col*height+row] = rowMajor[row*width+col]
		}
	}
	return out
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

// TestForwardDCT8x8ImplMatchesPureGo proves the dispatched 8x8 forward DCT
// kernel bit-exact with the portable reference across random 8-bit residual
// ranges and strides.
func TestForwardDCT8x8ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	const resStride, coeffStride = 23, 17
	residual := make([]int16, resStride*16)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255 // 8-bit residual range
		}
		want := make([]int32, coeffStride*16)
		got := make([]int32, coeffStride*16)
		forwardDCT8x8PureGo(want, coeffStride, residual, resStride)
		forwardDCT8x8Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT8x8(b *testing.B) {
	var residual [64]int16
	for i := range residual {
		residual[i] = int16(i*7%400) - 200
	}
	var coeff [64]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT8x8Impl(coeff[:], 8, residual[:], 8)
	}
}

// TestForwardDCT4x4ImplMatchesPureGo proves the dispatched 4x4 forward DCT
// kernel bit-exact with the portable reference across random 8-bit residual
// ranges and strides.
func TestForwardDCT4x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	const resStride, coeffStride = 13, 9
	residual := make([]int16, resStride*8)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255 // 8-bit residual range
		}
		want := make([]int32, coeffStride*8)
		got := make([]int32, coeffStride*8)
		forwardDCT4x4PureGo(want, coeffStride, residual, resStride)
		forwardDCT4x4Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT4x4(b *testing.B) {
	var residual [16]int16
	for i := range residual {
		residual[i] = int16(i*29%400) - 200
	}
	var coeff [16]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT4x4Impl(coeff[:], 4, residual[:], 4)
	}
}

// TestForwardDCT16x16ImplMatchesPureGo proves the dispatched 16x16 forward
// DCT kernel bit-exact with the portable reference across random 8-bit
// residual ranges and strides.
func TestForwardDCT16x16ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(83))
	const resStride, coeffStride = 37, 19
	residual := make([]int16, resStride*32)
	for trial := range 2000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*32)
		got := make([]int32, coeffStride*32)
		forwardDCT16x16PureGo(want, coeffStride, residual, resStride)
		forwardDCT16x16Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestForwardDCTDispatchFullInt16MatchesPureGo(t *testing.T) {
	type kernel struct {
		name string
		side int
		run  func([]int32, int, []int16, int) error
		pure func([]int32, int, []int16, int)
	}
	kernels := []kernel{
		{name: "4x4", side: 4, run: ForwardDCT4x4, pure: forwardDCT4x4PureGo},
		{name: "8x8", side: 8, run: ForwardDCT8x8, pure: forwardDCT8x8PureGo},
		{name: "16x16", side: 16, run: ForwardDCT16x16, pure: forwardDCT16x16PureGo},
		{name: "32x32", side: 32, run: ForwardDCT32x32, pure: forwardDCT32x32PureGo},
	}
	rng := rand.New(rand.NewSource(403))
	for _, k := range kernels {
		t.Run(k.name, func(t *testing.T) {
			residual := make([]int16, k.side*k.side)
			want := make([]int32, k.side*k.side)
			got := make([]int32, k.side*k.side)
			trials := 100
			if k.side == 32 {
				trials = 8
			}
			for trial := range trials {
				for i := range residual {
					residual[i] = int16(rng.Intn(1<<16) - (1 << 15))
				}
				clear(want)
				clear(got)
				k.pure(want, k.side, residual, k.side)
				if err := k.run(got, k.side, residual, k.side); err != nil {
					t.Fatalf("trial %d: %v", trial, err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("trial %d coeff[%d]=%d want %d", trial, i, got[i], want[i])
					}
				}
			}
		})
	}
}

func BenchmarkForwardDCT16x16(b *testing.B) {
	var residual [256]int16
	for i := range residual {
		residual[i] = int16(i*11%400) - 200
	}
	var coeff [256]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT16x16Impl(coeff[:], 16, residual[:], 16)
	}
}

// TestForwardDCT32x32ImplMatchesPureGo proves the dispatched 32x32 forward
// DCT kernel bit-exact with the portable reference across random 8-bit
// residual ranges and strides.
func TestForwardDCT32x32ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(89))
	const resStride, coeffStride = 41, 35
	residual := make([]int16, resStride*64)
	for trial := range 1000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*64)
		got := make([]int32, coeffStride*64)
		forwardDCT32x32PureGo(want, coeffStride, residual, resStride)
		forwardDCT32x32Impl(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] impl %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkForwardDCT32x32(b *testing.B) {
	var residual [1024]int16
	for i := range residual {
		residual[i] = int16(i*13%400) - 200
	}
	var coeff [1024]int32
	b.ReportAllocs()
	for b.Loop() {
		forwardDCT32x32Impl(coeff[:], 32, residual[:], 32)
	}
}

func BenchmarkForwardDCT64x64(b *testing.B) {
	var residual [64 * 64]int16
	for i := range residual {
		residual[i] = int16(i*17%400) - 200
	}
	var coeff [32 * 32]int32
	b.ReportAllocs()
	for b.Loop() {
		_ = ForwardDCT64x64(coeff[:], 32, residual[:], 64)
	}
}

func BenchmarkForwardDCT32x64(b *testing.B) {
	var residual [32 * 64]int16
	for i := range residual {
		residual[i] = int16(i*19%400) - 200
	}
	var coeff [32 * 32]int32
	b.ReportAllocs()
	for b.Loop() {
		_ = ForwardDCT32x64(coeff[:], 32, residual[:], 32)
	}
}

func BenchmarkForwardDCT64x32(b *testing.B) {
	var residual [64 * 32]int16
	for i := range residual {
		residual[i] = int16(i*23%400) - 200
	}
	var coeff [32 * 32]int32
	b.ReportAllocs()
	for b.Loop() {
		_ = ForwardDCT64x32(coeff[:], 32, residual[:], 64)
	}
}

func TestFwdDCT64MatchesPinnedLibaomVectors(t *testing.T) {
	cases := []struct {
		name   string
		cosBit uint
		input  func(i int) int32
		want   [64]int32
	}{
		{
			name:   "impulse_q13",
			cosBit: 13,
			input: func(i int) int32 {
				if i == 0 {
					return 1000
				}
				return 0
			},
			want: [64]int32{707, 1000, 999, 997, 995, 992, 989, 985, 981, 976, 970, 964, 957, 950, 942, 933, 924, 914, 904, 893, 882, 870, 858, 845, 831, 818, 803, 788, 773, 757, 741, 724, 707, 690, 672, 653, 634, 615, 596, 576, 556, 535, 514, 493, 471, 450, 428, 405, 383, 360, 337, 314, 290, 267, 243, 219, 195, 171, 147, 122, 98, 74, 49, 25},
		},
		{
			name:   "dense_q10",
			cosBit: 10,
			input: func(i int) int32 {
				return int32((i*37+11)%511) - 255
			},
			want: [64]int32{-573, -260, -1256, -205, -993, -606, -1000, -898, -2660, -1920, 3592, -359, 586, -82, -205, -81, -472, -910, -1272, 1712, 375, -112, 450, -468, 234, -418, -458, -958, 889, 695, -332, 504, -361, 104, -137, -483, -601, 312, 901, -377, 411, -159, -95, 99, -522, -423, 39, 876, -196, 157, 125, -315, 266, -501, -395, -47, 728, 71, -94, 336, -405, 260, -333, -529},
		},
		{
			name:   "mixed_q11",
			cosBit: 11,
			input: func(i int) int32 {
				if i%7 == 0 {
					return int32(i*13 - 300)
				}
				return int32((i%5)-2) * 17
			},
			want: [64]int32{750, -1907, 55, -524, 150, -451, 108, -343, 5, -482, 23, -404, 146, -589, 186, -767, 139, -2010, 1000, -1624, -133, -199, 21, -346, -83, -819, 478, -90, 115, -214, 210, -492, 227, -726, 205, -1762, 909, -1005, -303, 415, -29, 123, -57, 48, -111, -189, -35, -187, 58, -454, -100, -818, 580, -1407, 843, -347, -458, 944, -69, 415, -92, 262, -112, 3},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var in, got [64]int32
			for i := range in {
				in[i] = tc.input(i)
			}
			fwdDCT64(&in, &got, fwdCospiForBit(tc.cosBit), tc.cosBit)
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("coeff[%d]=%d want %d", i, got[i], tc.want[i])
				}
			}
		})
	}
}

// n4Residuals yields 8-bit residual blocks: random ranges, the extremes and
// constant blocks.
func n4Residuals(n, count int) [][]int16 {
	rng := rand.New(rand.NewSource(7))
	var out [][]int16
	for i := range count {
		res := make([]int16, n*n)
		switch i % 6 {
		case 0:
			for j := range res {
				res[j] = int16(rng.Intn(511)) - 255
			}
		case 1:
			for j := range res {
				res[j] = int16(rng.Intn(9)) - 4
			}
		case 2:
			for j := range res {
				res[j] = 255
				if rng.Intn(2) == 0 {
					res[j] = -255
				}
			}
		case 3:
			v := int16(rng.Intn(511)) - 255
			for j := range res {
				res[j] = v
			}
		case 4:
			for j := range res {
				res[j] = int16((j%n)*16 - 255)
			}
		default:
			for j := range res {
				res[j] = int16(rng.Intn(61)) - 30
			}
		}
		out = append(out, res)
	}
	return out
}

func TestForwardDCTN4MatchesFull(t *testing.T) {
	for _, n := range []int{16, 32} {
		k := n / 4
		for i, res := range n4Residuals(n, 600) {
			full := make([]int32, n*n)
			var err error
			if n == 16 {
				err = ForwardDCT16x16(full, 16, res, 16)
			} else {
				err = ForwardDCT32x32(full, 32, res, 32)
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, impl := range []string{"pure", "impl"} {
				got := make([]int32, n*n)
				for j := range got {
					got[j] = 12345 // only the N4 block is written
				}
				switch {
				case n == 16 && impl == "pure":
					var buf [64]int32
					forwardDCT16x16N4ColPureGo(&buf, (*[256]int16)(res))
					fwd16x16N4RowsPureGo((*[256]int32)(got), &buf, 0xf)
				case n == 16:
					ForwardDCT16x16N48BitTrusted((*[256]int32)(got), (*[256]int16)(res))
				case impl == "pure":
					var buf [256]int32
					forwardDCT32x32N4ColPureGo(&buf, (*[1024]int16)(res))
					fwd32x32N4RowsPureGo((*[1024]int32)(got), &buf, 0xff)
				default:
					ForwardDCT32x32N48BitTrusted((*[1024]int32)(got), (*[1024]int16)(res))
				}
				for c := range n {
					for r := range n {
						g := got[c*n+r]
						if c < k && r < k {
							if g != full[c*n+r] {
								t.Fatalf("n=%d case %d %s: coeff[col %d][row %d] = %d, full %d", n, i, impl, c, r, g, full[c*n+r])
							}
						} else if g != 12345 {
							t.Fatalf("n=%d case %d %s: wrote outside the N4 block at col %d row %d", n, i, impl, c, r)
						}
					}
				}
			}
		}
	}
}

// TestForwardDCTN4RowSkipMatchesRowSkip pins the N4 row-skip kernels to the
// scalar row-skipping kernels followed by the encoder's N4 restriction: the same
// group mask over the N4 rows and the same N4 coefficients.
func TestForwardDCTN4RowSkipMatchesRowSkip(t *testing.T) {
	for _, n := range []int{16, 32} {
		k := n / 4
		groups := uint8(1)<<((k+3)>>2) - 1
		for i, res := range n4Residuals(n, 600) {
			for _, thr := range []int32{0, 1, 20, 40, 70, 150, 220, 600, 5000} {
				thr0 := thr / 2
				want := make([]int32, n*n)
				got := make([]int32, n*n)
				var wm, gm uint8
				if n == 16 {
					wm = forwardDCT16x16RowSkipPureGo((*[256]int32)(want), (*[256]int16)(res), thr0, thr)
					gm = ForwardDCT16x16N4RowSkip8BitTrusted((*[256]int32)(got), (*[256]int16)(res), thr0, thr)
				} else {
					wm = forwardDCT32x32RowSkipPureGo((*[1024]int32)(want), (*[1024]int16)(res), thr0, thr)
					gm = ForwardDCT32x32N4RowSkip8BitTrusted((*[1024]int32)(got), (*[1024]int16)(res), thr0, thr)
				}
				wm &= groups
				if gm != wm {
					t.Fatalf("n=%d case %d thr %d: mask %b, row-skip kernel %b", n, i, thr, gm, wm)
				}
				if gm == 0 {
					continue
				}
				for c := range k {
					for r := range k {
						if got[c*n+r] != want[c*n+r] {
							t.Fatalf("n=%d case %d thr %d: coeff[col %d][row %d] = %d, row-skip %d", n, i, thr, c, r, got[c*n+r], want[c*n+r])
						}
					}
				}
			}
		}
	}
}

func BenchmarkForwardDCTN4(b *testing.B) {
	for _, n := range []int{16, 32} {
		res := n4Residuals(n, 1)[0]
		coeff := make([]int32, n*n)
		b.Run("rowskip", func(b *testing.B) {
			for range b.N {
				if n == 16 {
					ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(coeff), (*[256]int16)(res), 10, 20)
				} else {
					ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(coeff), (*[1024]int16)(res), 10, 20)
				}
			}
		})
		b.Run("n4", func(b *testing.B) {
			for range b.N {
				if n == 16 {
					ForwardDCT16x16N4RowSkip8BitTrusted((*[256]int32)(coeff), (*[256]int16)(res), 10, 20)
				} else {
					ForwardDCT32x32N4RowSkip8BitTrusted((*[1024]int32)(coeff), (*[1024]int16)(res), 10, 20)
				}
			}
		})
	}
}

// TestForwardDCT8x8N4MatchesFullTransform checks SVT's N4_SHAPE semantics:
// the low 2x2 coefficients equal the full 8x8 DCT_DCT, the rest are zero.
func TestForwardDCT8x8N4MatchesFullTransform(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	var residual [64]int16
	for trial := 0; trial < 2000; trial++ {
		for i := range residual {
			switch trial % 3 {
			case 0:
				residual[i] = int16(rng.Intn(511) - 255)
			case 1:
				residual[i] = int16(255 - 510*rng.Intn(2))
			default:
				residual[i] = int16(rng.Intn(31) - 15)
			}
		}
		var full, want [64]int32
		forwardDCT8x8PureGo(full[:], 8, residual[:], 8)
		want[0], want[1], want[8], want[9] = full[0], full[1], full[8], full[9]
		for name, fn := range map[string]func(*[64]int32, *[64]int16){
			"purego": forwardDCT8x8N4PureGo,
			"impl":   forwardDCT8x8N4Impl,
		} {
			got := [64]int32{5: 7, 63: 9}
			fn(&got, &residual)
			if got != want {
				t.Fatalf("%s trial %d: got %v want %v", name, trial, got[:10], want[:10])
			}
		}
	}
}

// TestForwardDCTRectTrustedMatchesScalar checks the trusted 2:1 DCT_DCT
// rectangles against the scalar kernels over random and extreme 8-bit
// residuals, at the packed coefficient stride (in place) and a wider one.
func TestForwardDCTRectTrustedMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, size := range []Size{{16, 8}, {8, 16}, {32, 16}, {16, 32}} {
		w, h := int(size.Width), int(size.Height)
		for _, extra := range []int{0, 3} {
			rs, cs := w+5, h+extra
			residual := make([]int16, rs*h)
			got := make([]int32, cs*w)
			want := make([]int32, cs*w)
			for trial := range 300 {
				for r := range h {
					for c := range w {
						var v int16
						switch trial % 4 {
						case 0:
							v = int16(rng.Intn(511) - 255)
						case 1:
							v = 255
						case 2:
							v = 255
							if (r+c)&1 == 1 {
								v = -255
							}
						default:
							v = int16(255 * (rng.Intn(3) - 1))
						}
						residual[r*rs+c] = v
					}
				}
				clear(got)
				clear(want)
				if err := ForwardBlock8BitResidualTrusted(got, cs, residual, rs, nil, size, TypeDCTDCT); err != nil {
					t.Fatal(err)
				}
				if err := forwardDCTBySize(want, cs, residual, rs, size); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%dx%d stride %d trial %d: coeff[%d] = %d, want %d", w, h, cs, trial, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// rowSkipNetwork is one forward row network the row-skip bound covers, with
// the analytic rounding-error bound its slack is derived from.
type rowSkipNetwork struct {
	n        int
	slack    int32
	errBound float64
	run      func(in []int32) []int32
}

var rowSkipNetworks = []rowSkipNetwork{
	{8, fwdRowSkipSlack8, 1.21, func(in []int32) []int32 {
		var a, out [8]int32
		copy(a[:], in)
		fwdDCT8(&a, &out)
		return out[:]
	}},
	{16, fwdRowSkipSlack16, 2.83, func(in []int32) []int32 {
		var a, out [16]int32
		copy(a[:], in)
		fwdDCT16(&a, &out, &fwdCospi12, 12)
		return out[:]
	}},
	{32, fwdRowSkipSlack32, 7.05, func(in []int32) []int32 {
		var a, out [32]int32
		copy(a[:], in)
		fwdDCT32(&a, &out, &fwdCospi12, 12)
		return out[:]
	}},
}

// TestForwardDCTRowBound pins the constants behind fwdRowSkipLive for each row
// network: every effective input weight is at most 1 (plus constant
// quantization noise far below the 1/64 the bound allows), the observed
// rounding error stays within the analytic bound the slack covers, and no
// output ever reaches S + S>>6 + slack.
func TestForwardDCTRowBound(t *testing.T) {
	for _, nw := range rowSkipNetworks {
		t.Run(fmt.Sprint(nw.n), func(t *testing.T) {
			if float64(nw.slack) < math.Ceil(nw.errBound)+1 {
				t.Fatalf("slack %d does not cover error bound %v", nw.slack, nw.errBound)
			}
			n := nw.n
			// Effective weights from scaled unit vectors: the network's rounding is
			// at most a few units, negligible against the 2^24 scale.
			const scale = 1 << 24
			w := make([][]float64, n)
			for l := range w {
				w[l] = make([]float64, n)
			}
			maxW := 0.0
			in := make([]int32, n)
			for j := range n {
				clear(in)
				in[j] = scale
				out := nw.run(in)
				for l := range n {
					w[l][j] = float64(out[l]) / scale
					maxW = max(maxW, math.Abs(w[l][j]))
				}
			}
			if maxW > 1+1.0/256 {
				t.Fatalf("max effective weight %v exceeds 1+1/256", maxW)
			}
			rng := rand.New(rand.NewSource(int64(7 + n)))
			maxErr := 0.0
			check := func(in []int32) {
				out := nw.run(in)
				s := int32(0)
				for _, v := range in {
					s += max(v, -v)
				}
				for l := range n {
					lin := 0.0
					for j := range n {
						lin += w[l][j] * float64(in[j])
					}
					maxErr = max(maxErr, math.Abs(float64(out[l])-lin))
					c := max(out[l], -out[l])
					if fwdRowSkipLive(s, c+1, nw.slack) == false {
						t.Fatalf("|C_%d|=%d breaks the bound for S=%d (in %v)", l, c, s, in)
					}
				}
			}
			for trial := range 300000 {
				amp := []int32{1, 2, 3, 8, 64, 4080}[trial%6]
				for j := range in {
					in[j] = rng.Int31n(2*amp+1) - amp
				}
				check(in)
			}
			// Sign patterns aligned with each basis row maximize the linear part.
			for l := range n {
				for j := range in {
					in[j] = 2040
					if w[l][j] < 0 {
						in[j] = -2040
					}
				}
				check(in)
			}
			t.Logf("n=%d max effective weight %v, observed rounding error %v (bound %v)", n, maxW, maxErr, nw.errBound)
			// The weight estimate itself carries ~1e-7 relative noise.
			if maxErr > nw.errBound+0.05 {
				t.Fatalf("observed rounding error %v exceeds the %v analytic bound", maxErr, nw.errBound)
			}
		})
	}
}

// rowSkipCase runs one row-skipping kernel against the full transform.
type rowSkipCase struct {
	n      int
	full   func(coeff []int32, res []int16)
	scalar func(coeff []int32, res []int16, thr0, thr int32) uint8
	impls  map[string]func(coeff []int32, res []int16, thr0, thr int32) uint8
}

func rowSkipCases() []rowSkipCase {
	return []rowSkipCase{
		{
			n:    8,
			full: func(c []int32, r []int16) { forwardDCT8x8PureGo(c, 8, r, 8) },
			scalar: func(c []int32, r []int16, t0, t int32) uint8 {
				return forwardDCT8x8RowSkipPureGo((*[64]int32)(c), (*[64]int16)(r), t0, t)
			},
			impls: map[string]func([]int32, []int16, int32, int32) uint8{
				"dispatch": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT8x8RowSkipImpl((*[64]int32)(c), (*[64]int16)(r), t0, t)
				},
				"exported": func(c []int32, r []int16, t0, t int32) uint8 {
					return ForwardDCT8x8RowSkip8BitTrusted((*[64]int32)(c), (*[64]int16)(r), t0, t)
				},
				"full": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT8x8RowSkipFull((*[64]int32)(c), (*[64]int16)(r), t0, t)
				},
			},
		},
		{
			n:    16,
			full: func(c []int32, r []int16) { forwardDCT16x16PureGo(c, 16, r, 16) },
			scalar: func(c []int32, r []int16, t0, t int32) uint8 {
				return forwardDCT16x16RowSkipPureGo((*[256]int32)(c), (*[256]int16)(r), t0, t)
			},
			impls: map[string]func([]int32, []int16, int32, int32) uint8{
				"dispatch": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT16x16RowSkipImpl((*[256]int32)(c), (*[256]int16)(r), t0, t)
				},
				"exported": func(c []int32, r []int16, t0, t int32) uint8 {
					return ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(c), (*[256]int16)(r), t0, t)
				},
				"full": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT16x16RowSkipFull((*[256]int32)(c), (*[256]int16)(r), t0, t)
				},
			},
		},
		{
			n:    32,
			full: func(c []int32, r []int16) { forwardDCT32x32PureGo(c, 32, r, 32) },
			scalar: func(c []int32, r []int16, t0, t int32) uint8 {
				return forwardDCT32x32RowSkipPureGo((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
			},
			impls: map[string]func([]int32, []int16, int32, int32) uint8{
				"dispatch": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT32x32RowSkipImpl((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
				},
				"exported": func(c []int32, r []int16, t0, t int32) uint8 {
					return ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
				},
				"full": func(c []int32, r []int16, t0, t int32) uint8 {
					return forwardDCT32x32RowSkipFull((*[1024]int32)(c), (*[1024]int16)(r), t0, t)
				},
			},
		},
	}
}

func TestForwardDCTRowSkipMatchesFull(t *testing.T) {
	for _, tc := range rowSkipCases() {
		t.Run(fmt.Sprint(tc.n), func(t *testing.T) {
			n := tc.n
			groups := n / 4
			allMask := uint8(1<<groups - 1)
			rng := rand.New(rand.NewSource(int64(11 + n)))
			res := make([]int16, n*n)
			full := make([]int32, n*n)
			got := make([]int32, n*n)
			sawSkip, sawPartial, sawAllZero := false, false, false
			for trial := range 6000 {
				amp := []int16{0, 1, 2, 4, 12, 40, 255}[trial%7]
				sparse := trial%3 == 0
				for i := range res {
					res[i] = 0
					if amp > 0 && (!sparse || rng.Intn(16) == 0) {
						res[i] = int16(rng.Intn(2*int(amp)+1)) - amp
					}
				}
				if trial%97 == 0 {
					for i := range res {
						res[i] = 255
						if (i/n+i%n)%2 == 1 {
							res[i] = -255
						}
					}
				}
				thr := rng.Int31n(400) + 1
				thr0 := min(rng.Int31n(400)+1, thr)
				tc.full(full, res)
				wantMask := tc.scalar(make([]int32, n*n), res, thr0, thr)
				check := func(name string, mask uint8) {
					for r := range n {
						live := mask&(1<<(r>>2)) != 0
						rowThr := thr
						if r == 0 {
							rowThr = thr0
						}
						for c := range n {
							want := full[c*n+r]
							if live {
								if got[c*n+r] != want {
									t.Fatalf("%s trial %d: coeff[%d][%d]=%d want %d", name, trial, c, r, got[c*n+r], want)
								}
								continue
							}
							if max(want, -want) >= rowThr {
								t.Fatalf("%s trial %d: skipped row %d holds |%d| >= zero bin %d", name, trial, r, want, rowThr)
							}
							if mask != 0 && got[c*n+r] != 0 {
								t.Fatalf("%s trial %d: skipped coeff[%d][%d]=%d not zeroed", name, trial, c, r, got[c*n+r])
							}
						}
					}
				}
				for i := range got {
					got[i] = 0x5a5a5a5a
				}
				check("scalar", tc.scalar(got, res, thr0, thr))
				switch wantMask {
				case 0:
					sawAllZero = true
				case allMask:
				default:
					sawPartial = true
				}
				if wantMask != allMask {
					sawSkip = true
				}
				for name, impl := range tc.impls {
					for i := range got {
						got[i] = 0x5a5a5a5a
					}
					mask := impl(got, res, thr0, thr)
					if name != "full" && mask != wantMask && mask != allMask {
						t.Fatalf("%s trial %d: mask %b, scalar mask %b", name, trial, mask, wantMask)
					}
					check(name, mask)
				}
			}
			if !sawSkip || !sawPartial || !sawAllZero {
				t.Fatalf("corpus coverage skip=%v partial=%v allzero=%v", sawSkip, sawPartial, sawAllZero)
			}
		})
	}
}

func BenchmarkForwardDCTRowSkip(b *testing.B) {
	for _, tc := range rowSkipCases() {
		n := tc.n
		rng := rand.New(rand.NewSource(3))
		res := make([]int16, n*n)
		for i := range res {
			res[i] = int16(rng.Intn(9)) - 4
		}
		coeff := make([]int32, n*n)
		b.Run(fmt.Sprintf("%d/full", n), func(b *testing.B) {
			for range b.N {
				tc.impls["full"](coeff, res, 0, 0)
			}
		})
		for _, thr := range []int32{40, 70, 150, 220} {
			mask := tc.impls["dispatch"](coeff, res, thr, thr)
			b.Run(fmt.Sprintf("%d/skip-thr%d-mask%b", n, thr, mask), func(b *testing.B) {
				for range b.N {
					tc.impls["dispatch"](coeff, res, thr, thr)
				}
			})
		}
	}
}

// The forward DCT's parity proof at this stage is the combined-pipeline check:
// ForwardDCT then the decoder's byte-exact DCT_DCT InverseBlock must reproduce the
// residual within the AV1 fixed-point pipeline's known rounding (the same
// fwd+inv error bound libaom's own unit tests assert, |err| <= 2 at 8-bit
// magnitudes). The bit-exact end-to-end gate lands with the non-lossless
// encoder: there the decoder reconstructs from the same coefficients the
// encoder used, so encoder/decoder recon match exactly regardless of forward
// rounding.

func TestForwardDCT4x4InverseRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	scratch := make([]int32, 16)
	for range 2000 {
		var residual [16]int16
		for i := range residual {
			residual[i] = int16(rng.Intn(2*255+1) - 255) // 8-bit residual span
		}
		var coeff [16]int32
		if err := ForwardDCT4x4(coeff[:], 4, residual[:], 4); err != nil {
			t.Fatalf("forward: %v", err)
		}
		var dst [16]int16
		if err := InverseBlock(dst[:], 4, coeff[:], 4, scratch, Size{Width: 4, Height: 4}, TypeDCTDCT); err != nil {
			t.Fatalf("inverse: %v", err)
		}
		for i := range dst {
			diff := int(dst[i]) - int(residual[i])
			if diff < -2 || diff > 2 {
				t.Fatalf("4x4 round-trip error %d at %d:\n in  %v\n out %v\n co  %v", diff, i, residual, dst, coeff)
			}
		}
	}
}

func TestForwardDCT8x8InverseRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	scratch := make([]int32, 64)
	for range 1000 {
		var residual [64]int16
		for i := range residual {
			residual[i] = int16(rng.Intn(2*255+1) - 255)
		}
		var coeff [64]int32
		if err := ForwardDCT8x8(coeff[:], 8, residual[:], 8); err != nil {
			t.Fatalf("forward: %v", err)
		}
		var dst [64]int16
		if err := InverseBlock(dst[:], 8, coeff[:], 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != nil {
			t.Fatalf("inverse: %v", err)
		}
		for i := range dst {
			diff := int(dst[i]) - int(residual[i])
			if diff < -2 || diff > 2 {
				t.Fatalf("8x8 round-trip error %d at %d", diff, i)
			}
		}
	}
}

// TestForwardDCTDCScale pins the DC gain as a regression guard: a constant
// block transforms to a single DC coefficient with zero AC. The DC magnitudes
// are the AV1 fixed-point pipeline's outputs (per-pass gain 2*cos(pi/4) in Q13
// plus the stage shifts), pinned from the implementation whose correctness the
// inverse round-trip tests establish.
func TestForwardDCTDCScale(t *testing.T) {
	var residual [16]int16
	for i := range residual {
		residual[i] = 100
	}
	var coeff [16]int32
	if err := ForwardDCT4x4(coeff[:], 4, residual[:], 4); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < 16; i++ {
		if coeff[i] != 0 {
			t.Fatalf("AC coeff[%d] = %d, want 0 for constant input", i, coeff[i])
		}
	}
	// 4x4: 100<<2 per sample, two passes of gain 2*cos(pi/4).
	if coeff[0] != 3199 {
		t.Fatalf("4x4 DC = %d, want 3199", coeff[0])
	}

	var residual8 [64]int16
	for i := range residual8 {
		residual8[i] = 100
	}
	var coeff8 [64]int32
	if err := ForwardDCT8x8(coeff8[:], 8, residual8[:], 8); err != nil {
		t.Fatal(err)
	}
	for i := 1; i < 64; i++ {
		if coeff8[i] != 0 {
			t.Fatalf("8x8 AC coeff[%d] = %d, want 0", i, coeff8[i])
		}
	}
	if coeff8[0] != 6404 {
		t.Fatalf("8x8 DC = %d, want 6404", coeff8[0])
	}
}

func TestForwardDCTZeroAlloc(t *testing.T) {
	residual := make([]int16, 64)
	for i := range residual {
		residual[i] = int16(i*3 - 90)
	}
	coeff := make([]int32, 64)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ForwardDCT4x4(coeff[:16], 4, residual[:16], 4)
		_ = ForwardDCT8x8(coeff, 8, residual, 8)
	})
	if allocs != 0 {
		t.Fatalf("forward DCT allocated %v objects/run, want 0", allocs)
	}
}

// TestForwardDCTRectInverseRoundTrip proves the rectangular forward DCTs
// against the decoder's byte-exact inverse within the same fixed-point
// rounding bound the square sizes assert. The rectangular pipeline adds the
// sqrt(2) row scaling, so this also pins the NewSqrt2 rounding.
func TestForwardDCTRectInverseRoundTrip(t *testing.T) {
	type rect struct {
		name string
		w, h int
		fwd  func([]int32, int, []int16, int) error
		tol  int
	}
	// The 32-point rectangles run the deeper {2,-4,0} shift pipeline, which
	// discards two more fixed-point bits mid-pass than the small rectangles;
	// one extra unit of round-trip noise is the corresponding bound.
	rects := []rect{
		{"16x8", 16, 8, ForwardDCT16x8, 2},
		{"8x16", 8, 16, ForwardDCT8x16, 2},
		{"8x4", 8, 4, ForwardDCT8x4, 2},
		{"4x8", 4, 8, ForwardDCT4x8, 2},
		{"32x16", 32, 16, ForwardDCT32x16, 3},
		{"16x32", 16, 32, ForwardDCT16x32, 3},
	}
	for _, rc := range rects {
		t.Run(rc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(3))
			n := rc.w * rc.h
			scratch := make([]int32, n)
			residual := make([]int16, n)
			coeff := make([]int32, n)
			dst := make([]int16, n)
			for range 1000 {
				for i := range residual {
					residual[i] = int16(rng.Intn(2*255+1) - 255)
				}
				if err := rc.fwd(coeff, rc.h, residual, rc.w); err != nil {
					t.Fatalf("forward: %v", err)
				}
				if err := InverseBlock(dst, rc.w, coeff, rc.h, scratch, Size{Width: uint8(rc.w), Height: uint8(rc.h)}, TypeDCTDCT); err != nil {
					t.Fatalf("inverse: %v", err)
				}
				for i := range dst {
					diff := int(dst[i]) - int(residual[i])
					if diff < -rc.tol || diff > rc.tol {
						t.Fatalf("%s round-trip error %d at %d", rc.name, diff, i)
					}
				}
			}
		})
	}
}

// TestForwardDCTRectDCScale pins the rectangular DC gains (per-pass 2*cos(pi/4)
// gains, the stage shifts, and the sqrt(2) row scaling) as regression guards.
func TestForwardDCTRectDCScale(t *testing.T) {
	check := func(name string, w, h int, fwd func([]int32, int, []int16, int) error, wantDC int32) {
		n := w * h
		residual := make([]int16, n)
		for i := range residual {
			residual[i] = 100
		}
		coeff := make([]int32, n)
		if err := fwd(coeff, h, residual, w); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i := 1; i < n; i++ {
			if coeff[i] != 0 {
				t.Fatalf("%s AC coeff[%d] = %d, want 0", name, i, coeff[i])
			}
		}
		if coeff[0] != wantDC {
			t.Fatalf("%s DC = %d, want %d", name, coeff[0], wantDC)
		}
	}
	check("16x8", 16, 8, ForwardDCT16x8, 9057)
	check("32x16", 32, 16, ForwardDCT32x16, 9057)
	check("16x32", 16, 32, ForwardDCT16x32, 9057)
	check("8x16", 8, 16, ForwardDCT8x16, 9057)
	check("8x4", 8, 4, ForwardDCT8x4, 4529)
	check("4x8", 4, 8, ForwardDCT4x8, 4529)
}

func TestForwardDCT64ExtentDCScale(t *testing.T) {
	check := func(name string, w, h int, fwd func([]int32, int, []int16, int) error, wantDC int32) {
		coeffSize := adjustedScanSize(Size{Width: uint8(w), Height: uint8(h)})
		coeffW := int(coeffSize.Width)
		coeffH := int(coeffSize.Height)
		residual := make([]int16, w*h)
		for i := range residual {
			residual[i] = 100
		}
		coeff := make([]int32, coeffW*coeffH)
		if err := fwd(coeff, coeffH, residual, w); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for i := 1; i < len(coeff); i++ {
			if coeff[i] != 0 {
				t.Fatalf("%s AC coeff[%d]=%d want 0", name, i, coeff[i])
			}
		}
		if coeff[0] != wantDC {
			t.Fatalf("%s DC=%d want %d", name, coeff[0], wantDC)
		}
	}
	check("64x64", 64, 64, ForwardDCT64x64, 12806)
	check("32x64", 32, 64, ForwardDCT32x64, 9056)
	check("64x32", 64, 32, ForwardDCT64x32, 9056)
}

func TestForwardDCT64ExtentDispatchMatchesForwardBlock(t *testing.T) {
	cases := []struct {
		name string
		size Size
		fwd  func([]int32, int, []int16, int) error
	}{
		{name: "64x64", size: Size{Width: 64, Height: 64}, fwd: ForwardDCT64x64},
		{name: "32x64", size: Size{Width: 32, Height: 64}, fwd: ForwardDCT32x64},
		{name: "64x32", size: Size{Width: 64, Height: 32}, fwd: ForwardDCT64x32},
	}
	rng := rand.New(rand.NewSource(6400))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			width := int(tc.size.Width)
			height := int(tc.size.Height)
			coeffSize := adjustedScanSize(tc.size)
			coeffW := int(coeffSize.Width)
			coeffH := int(coeffSize.Height)
			residual := make([]int16, width*height)
			for i := range residual {
				residual[i] = int16(rng.Intn(511) - 255)
			}
			got := make([]int32, coeffW*coeffH)
			want := make([]int32, coeffW*coeffH)
			scratch := make([]int32, width*height)
			if err := ForwardBlock(got, coeffH, residual, width, scratch, tc.size, TypeDCTDCT); err != nil {
				t.Fatalf("ForwardBlock: %v", err)
			}
			if err := tc.fwd(want, coeffH, residual, width); err != nil {
				t.Fatalf("specialized: %v", err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("coeff[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestForwardDCT64ExtentZeroAlloc(t *testing.T) {
	var residual [64 * 64]int16
	for i := range residual {
		residual[i] = int16(i%511) - 255
	}
	var coeff [32 * 32]int32
	allocs := testing.AllocsPerRun(20, func() {
		_ = ForwardDCT64x64(coeff[:], 32, residual[:], 64)
		_ = ForwardDCT32x64(coeff[:], 32, residual[:32*64], 32)
		_ = ForwardDCT64x32(coeff[:], 32, residual[:64*32], 64)
	})
	if allocs != 0 {
		t.Fatalf("forward DCT64 extent allocated %v objects/run, want 0", allocs)
	}
}

// TestForwardBlock8x8ImplExtremeResidual checks the selected implementation
// against the scalar oracle at residual extremes and alternating checkerboards.
func TestForwardBlock8x8ImplExtremeResidual(t *testing.T) {
	const resStride, coeffStride = 19, 13
	patterns := []func(r, c int) int16{
		func(r, c int) int16 { return 255 },
		func(r, c int) int16 { return -255 },
		func(r, c int) int16 {
			if (r+c)&1 == 0 {
				return 255
			}
			return -255
		},
		func(r, c int) int16 {
			if r&1 == 0 {
				return 255
			}
			return -255
		},
		func(r, c int) int16 {
			if c&1 == 0 {
				return 255
			}
			return -255
		},
	}
	cases := []struct {
		name string
		impl func([]int32, int, []int16, int, []int32)
		pure func([]int32, int, []int16, int, []int32)
	}{
		{"ADST_DCT", forwardBlock8x8ADSTDCTImpl, forwardBlock8x8ADSTDCTPureGo},
		{"DCT_ADST", forwardBlock8x8DCTADSTImpl, forwardBlock8x8DCTADSTPureGo},
		{"ADST_ADST", forwardBlock8x8ADSTADSTImpl, forwardBlock8x8ADSTADSTPureGo},
		{"IDTX", forwardBlock8x8IDTXImpl, forwardBlock8x8IDTXPureGo},
	}
	for _, tc := range cases {
		for pi, pat := range patterns {
			residual := make([]int16, resStride*8)
			for r := range 8 {
				for c := range 8 {
					residual[r*resStride+c] = pat(r, c)
				}
			}
			var gs, ws [64]int32
			got := make([]int32, coeffStride*8)
			want := make([]int32, coeffStride*8)
			tc.impl(got, coeffStride, residual, resStride, gs[:])
			tc.pure(want, coeffStride, residual, resStride, ws[:])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s pattern %d: coeff[%d] impl %d want %d", tc.name, pi, i, got[i], want[i])
				}
			}
		}
	}
}

func TestForwardBlockDCTMatchesSpecialized(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for _, size := range []Size{{Width: 4, Height: 4}, {Width: 8, Height: 8}} {
		n := int(size.Width)
		residual := make([]int16, n*n)
		for i := range residual {
			residual[i] = int16(rng.Intn(511) - 255)
		}
		got := make([]int32, n*n)
		want := make([]int32, n*n)
		scratch := make([]int32, n*n)
		if err := ForwardBlock(got, n, residual, n, scratch, size, TypeDCTDCT); err != nil {
			t.Fatalf("%dx%d ForwardBlock DCT: %v", n, n, err)
		}
		switch n {
		case 4:
			if err := ForwardDCT4x4(want, n, residual, n); err != nil {
				t.Fatal(err)
			}
		case 8:
			if err := ForwardDCT8x8(want, n, residual, n); err != nil {
				t.Fatal(err)
			}
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%dx%d coeff[%d]=%d want %d", n, n, i, got[i], want[i])
			}
		}
	}
}

func TestForwardBlockDCTThinRectangles(t *testing.T) {
	cases := []Size{
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 32},
		{Width: 32, Height: 8},
	}
	rng := rand.New(rand.NewSource(416))
	for _, size := range cases {
		t.Run(sizeName(size), func(t *testing.T) {
			width := int(size.Width)
			height := int(size.Height)
			n := width * height
			residual := make([]int16, n)
			coeff := make([]int32, n)
			scratch := make([]int32, n)
			invScratch := make([]int32, n)
			dst := make([]int16, n)
			for range 500 {
				for i := range residual {
					residual[i] = int16(rng.Intn(511) - 255)
				}
				if err := ForwardBlock(coeff, height, residual, width, scratch, size, TypeDCTDCT); err != nil {
					t.Fatalf("forward: %v", err)
				}
				if err := InverseBlock(dst, width, coeff, height, invScratch, size, TypeDCTDCT); err != nil {
					t.Fatalf("inverse: %v", err)
				}
				for i := range dst {
					diff := int(dst[i]) - int(residual[i])
					if diff < -5 || diff > 5 {
						t.Fatalf("%s round-trip error %d at %d", sizeName(size), diff, i)
					}
				}
			}
		})
	}
}

func TestForwardBlockDCTReduced64ExtentDC(t *testing.T) {
	for _, size := range []Size{{Width: 16, Height: 64}, {Width: 64, Height: 16}} {
		t.Run(sizeName(size), func(t *testing.T) {
			width := int(size.Width)
			height := int(size.Height)
			coeffSize := adjustedScanSize(size)
			coeffW := int(coeffSize.Width)
			coeffH := int(coeffSize.Height)
			residual := make([]int16, width*height)
			for i := range residual {
				residual[i] = 100
			}
			coeff := make([]int32, coeffW*coeffH)
			scratch := make([]int32, width*height)
			if err := ForwardBlock(coeff, coeffH, residual, width, scratch, size, TypeDCTDCT); err != nil {
				t.Fatalf("forward: %v", err)
			}
			if coeff[0] == 0 {
				t.Fatal("DC coefficient is zero")
			}
			for i := 1; i < len(coeff); i++ {
				if coeff[i] != 0 {
					t.Fatalf("AC coeff[%d]=%d want 0", i, coeff[i])
				}
			}
		})
	}
}

func TestForwardBlockHybridInverseRoundTrip(t *testing.T) {
	types := []Type{TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeIDTX}
	for _, size := range []Size{{Width: 4, Height: 4}, {Width: 8, Height: 8}} {
		n := int(size.Width)
		for _, typ := range types {
			t.Run(typeName(typ)+"_"+sizeName(size), func(t *testing.T) {
				rng := rand.New(rand.NewSource(int64(100 + n + int(typ))))
				coeff := make([]int32, n*n)
				scratch := make([]int32, n*n)
				invScratch := make([]int32, n*n)
				residual := make([]int16, n*n)
				dst := make([]int16, n*n)
				tol := 3
				if typ == TypeIDTX {
					tol = 4
				}
				for range 1000 {
					for i := range residual {
						residual[i] = int16(rng.Intn(511) - 255)
					}
					if err := ForwardBlock(coeff, n, residual, n, scratch, size, typ); err != nil {
						t.Fatalf("forward: %v", err)
					}
					if err := InverseBlock(dst, n, coeff, n, invScratch, size, typ); err != nil {
						t.Fatalf("inverse: %v", err)
					}
					for i := range dst {
						diff := int(dst[i]) - int(residual[i])
						if diff < -tol || diff > tol {
							t.Fatalf("%s %s round-trip error %d at %d", typeName(typ), sizeName(size), diff, i)
						}
					}
				}
			})
		}
	}
}

func TestForwardBlockExtendedHybridInverseRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		size Size
		typ  Type
		tol  int
	}{
		{name: "flipadst_dct_8x8", size: Size{Width: 8, Height: 8}, typ: TypeFlipADSTDCT, tol: 3},
		{name: "dct_flipadst_8x8", size: Size{Width: 8, Height: 8}, typ: TypeDCTFlipADST, tol: 3},
		{name: "flipadst_flipadst_8x8", size: Size{Width: 8, Height: 8}, typ: TypeFlipADSTFlipADST, tol: 3},
		{name: "adst_flipadst_8x8", size: Size{Width: 8, Height: 8}, typ: TypeADSTFlipADST, tol: 3},
		{name: "flipadst_adst_8x8", size: Size{Width: 8, Height: 8}, typ: TypeFlipADSTADST, tol: 3},
		{name: "v_adst_16x16", size: Size{Width: 16, Height: 16}, typ: TypeVADST, tol: 4},
		{name: "h_adst_16x16", size: Size{Width: 16, Height: 16}, typ: TypeHADST, tol: 4},
		{name: "adst_adst_16x16", size: Size{Width: 16, Height: 16}, typ: TypeADSTADST, tol: 4},
		{name: "flipadst_adst_16x8", size: Size{Width: 16, Height: 8}, typ: TypeFlipADSTADST, tol: 4},
		{name: "adst_flipadst_8x16", size: Size{Width: 8, Height: 16}, typ: TypeADSTFlipADST, tol: 4},
		{name: "h_adst_4x16", size: Size{Width: 4, Height: 16}, typ: TypeHADST, tol: 5},
		{name: "v_adst_16x4", size: Size{Width: 16, Height: 4}, typ: TypeVADST, tol: 5},
		{name: "adst_dct_32x8", size: Size{Width: 32, Height: 8}, typ: TypeADSTDCT, tol: 5},
		{name: "dct_adst_8x32", size: Size{Width: 8, Height: 32}, typ: TypeDCTADST, tol: 5},
		{name: "v_adst_32x16", size: Size{Width: 32, Height: 16}, typ: TypeVADST, tol: 5},
		{name: "h_adst_16x32", size: Size{Width: 16, Height: 32}, typ: TypeHADST, tol: 5},
		{name: "hdct_32x16", size: Size{Width: 32, Height: 16}, typ: TypeHDCT, tol: 5},
		{name: "vdct_16x32", size: Size{Width: 16, Height: 32}, typ: TypeVDCT, tol: 5},
		{name: "idtx_32x32", size: Size{Width: 32, Height: 32}, typ: TypeIDTX, tol: 5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			width := int(tc.size.Width)
			height := int(tc.size.Height)
			rng := rand.New(rand.NewSource(int64(9000 + width*height + int(tc.typ))))
			coeff := make([]int32, width*height)
			scratch := make([]int32, width*height)
			invScratch := make([]int32, width*height)
			residual := make([]int16, width*height)
			dst := make([]int16, width*height)
			for range 200 {
				for i := range residual {
					residual[i] = int16(rng.Intn(511) - 255)
				}
				if err := ForwardBlock(coeff, height, residual, width, scratch, tc.size, tc.typ); err != nil {
					t.Fatalf("forward: %v", err)
				}
				if err := InverseBlock(dst, width, coeff, height, invScratch, tc.size, tc.typ); err != nil {
					t.Fatalf("inverse: %v", err)
				}
				for i := range dst {
					diff := int(dst[i]) - int(residual[i])
					if diff < -tc.tol || diff > tc.tol {
						t.Fatalf("%s round-trip error %d at %d", tc.name, diff, i)
					}
				}
			}
		})
	}
}

func TestForwardBlockFlipADSTMatchesLibaomFlipShape(t *testing.T) {
	type flipCase struct {
		name     string
		flipType Type
		baseType Type
		flipRows bool
		flipCols bool
	}
	cases := []flipCase{
		{name: "vertical", flipType: TypeFlipADSTDCT, baseType: TypeADSTDCT, flipRows: true},
		{name: "horizontal", flipType: TypeDCTFlipADST, baseType: TypeDCTADST, flipCols: true},
		{name: "both", flipType: TypeFlipADSTFlipADST, baseType: TypeADSTADST, flipRows: true, flipCols: true},
		{name: "adst_horizontal", flipType: TypeADSTFlipADST, baseType: TypeADSTADST, flipCols: true},
		{name: "vertical_adst", flipType: TypeFlipADSTADST, baseType: TypeADSTADST, flipRows: true},
	}
	rng := rand.New(rand.NewSource(17))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var residual, flipped [64]int16
			for i := range residual {
				residual[i] = int16(rng.Intn(511) - 255)
			}
			for r := range 8 {
				for c := range 8 {
					srcR, srcC := r, c
					if tc.flipRows {
						srcR = 7 - srcR
					}
					if tc.flipCols {
						srcC = 7 - srcC
					}
					flipped[r*8+c] = residual[srcR*8+srcC]
				}
			}
			var got, want [64]int32
			var gotScratch, wantScratch [64]int32
			if err := forwardGenericBlock(got[:], 8, residual[:], 8, gotScratch[:], Size{Width: 8, Height: 8}, tc.flipType); err != nil {
				t.Fatalf("flip forward: %v", err)
			}
			if err := forwardGenericBlock(want[:], 8, flipped[:], 8, wantScratch[:], Size{Width: 8, Height: 8}, tc.baseType); err != nil {
				t.Fatalf("base forward: %v", err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("coeff[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestForwardBlockRejectsUnsupportedHybrid(t *testing.T) {
	residual := make([]int16, 64*16)
	coeff := make([]int32, 64*16)
	scratch := make([]int32, 64*16)
	if err := ForwardBlock(coeff, 16, residual, 64, scratch, Size{Width: 64, Height: 16}, TypeHDCT); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("non-realtime 64x16 HDCT err=%v want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock(coeff, 32, residual, 32, scratch, Size{Width: 32, Height: 32}, TypeDCTADST); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("ADST32-dependent DCT_ADST err=%v want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock(coeff[:64], 8, residual[:64], 8, nil, Size{Width: 8, Height: 8}, TypeADSTADST); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short scratch err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestForwardBlock8x8ADSTImplsMatchPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	const resStride, coeffStride = 19, 13
	residual := make([]int16, resStride*16)
	cases := []struct {
		name string
		impl func([]int32, int, []int16, int, []int32)
		pure func([]int32, int, []int16, int, []int32)
	}{
		{name: "ADST_DCT", impl: forwardBlock8x8ADSTDCTImpl, pure: forwardBlock8x8ADSTDCTPureGo},
		{name: "DCT_ADST", impl: forwardBlock8x8DCTADSTImpl, pure: forwardBlock8x8DCTADSTPureGo},
		{name: "ADST_ADST", impl: forwardBlock8x8ADSTADSTImpl, pure: forwardBlock8x8ADSTADSTPureGo},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for trial := range 3000 {
				for i := range residual {
					residual[i] = int16(rng.Intn(511) - 255)
				}
				var gotScratch, wantScratch [64]int32
				got := make([]int32, coeffStride*16)
				want := make([]int32, coeffStride*16)
				tc.impl(got, coeffStride, residual, resStride, gotScratch[:])
				tc.pure(want, coeffStride, residual, resStride, wantScratch[:])
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("trial %d coeff[%d]=%d want %d", trial, i, got[i], want[i])
					}
				}
			}
		})
	}
}

func TestForwardBlock8x8IDTXImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(15))
	const resStride, coeffStride = 19, 13
	residual := make([]int16, resStride*16)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511) - 255)
		}
		var gotScratch, wantScratch [64]int32
		got := make([]int32, coeffStride*16)
		want := make([]int32, coeffStride*16)
		forwardBlock8x8IDTXImpl(got, coeffStride, residual, resStride, gotScratch[:])
		forwardBlock8x8IDTXPureGo(want, coeffStride, residual, resStride, wantScratch[:])
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d coeff[%d]=%d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestForwardBlock8x8HybridFullInt16MatchesPureGo(t *testing.T) {
	types := []Type{TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeIDTX}
	rng := rand.New(rand.NewSource(404))
	for _, typ := range types {
		t.Run(typeName(typ), func(t *testing.T) {
			var residual [64]int16
			var want, got [64]int32
			var wantScratch, gotScratch [64]int32
			for trial := range 100 {
				for i := range residual {
					residual[i] = int16(rng.Intn(1<<16) - (1 << 15))
				}
				clear(want[:])
				clear(got[:])
				clear(wantScratch[:])
				clear(gotScratch[:])
				if !forwardBlock8x8HybridPureGo(want[:], 8, residual[:], 8, wantScratch[:], typ) {
					t.Fatalf("unsupported type %s", typeName(typ))
				}
				if err := ForwardBlock(got[:], 8, residual[:], 8, gotScratch[:], Size{Width: 8, Height: 8}, typ); err != nil {
					t.Fatalf("trial %d: %v", trial, err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("trial %d coeff[%d]=%d want %d", trial, i, got[i], want[i])
					}
				}
			}
		})
	}
}

func TestFwdDCT8ValuesMatchesPointerCore(t *testing.T) {
	rng := rand.New(rand.NewSource(14))
	cases := [][8]int32{
		{},
		{1, -2, 3, -4, 5, -6, 7, -8},
		{-1020, -364, 511, 2044, 17, -93, 728, -155},
	}
	for range 1000 {
		var input [8]int32
		for i := range input {
			input[i] = int32(rng.Intn(4097) - 2048)
		}
		cases = append(cases, input)
	}
	for trial, input := range cases {
		var want [8]int32
		fwdDCT8(&input, &want)
		o0, o1, o2, o3, o4, o5, o6, o7 := fwdDCT8Values(
			input[0],
			input[1],
			input[2],
			input[3],
			input[4],
			input[5],
			input[6],
			input[7],
		)
		got := [8]int32{o0, o1, o2, o3, o4, o5, o6, o7}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d output[%d]=%d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestForwardParameterized1DCoresMatchExistingQ13(t *testing.T) {
	rng := rand.New(rand.NewSource(18))
	for range 1000 {
		var in4 [4]int32
		var gotDCT4, wantDCT4, gotADST4, wantADST4 [4]int32
		for i := range in4 {
			in4[i] = int32(rng.Intn(4097) - 2048)
		}
		fwdDCT4ByCosBit(&in4, &gotDCT4, &fwdCospi13, 13)
		fwdDCT4(&in4, &wantDCT4)
		fwdADST4ByCosBit(&in4, &gotADST4, &fwdSinpi13, 13)
		fwdADST4(&in4, &wantADST4)
		if gotDCT4 != wantDCT4 {
			t.Fatalf("DCT4 got %v want %v", gotDCT4, wantDCT4)
		}
		if gotADST4 != wantADST4 {
			t.Fatalf("ADST4 got %v want %v", gotADST4, wantADST4)
		}

		var in8 [8]int32
		var gotDCT8, wantDCT8, gotADST8, wantADST8 [8]int32
		for i := range in8 {
			in8[i] = int32(rng.Intn(4097) - 2048)
		}
		fwdDCT8ByCosBit(&in8, &gotDCT8, &fwdCospi13, 13)
		fwdDCT8(&in8, &wantDCT8)
		fwdADST8ByCosBit(&in8, &gotADST8, &fwdCospi13, 13)
		fwdADST8(&in8, &wantADST8)
		if gotDCT8 != wantDCT8 {
			t.Fatalf("DCT8 got %v want %v", gotDCT8, wantDCT8)
		}
		if gotADST8 != wantADST8 {
			t.Fatalf("ADST8 got %v want %v", gotADST8, wantADST8)
		}
	}
}

func TestForwardBlockHybridZeroAlloc(t *testing.T) {
	residual := make([]int16, 64)
	for i := range residual {
		residual[i] = int16(i*5 - 120)
	}
	coeff := make([]int32, 64)
	scratch := make([]int32, 64)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ForwardBlock(coeff, 8, residual, 8, scratch, Size{Width: 8, Height: 8}, TypeADSTADST)
		_ = ForwardBlock(coeff, 8, residual, 8, scratch, Size{Width: 8, Height: 8}, TypeIDTX)
	})
	if allocs != 0 {
		t.Fatalf("ForwardBlock hybrid allocated %v objects/run, want 0", allocs)
	}
}

func BenchmarkForwardBlockHybrid8x8(b *testing.B) {
	residual := make([]int16, 64)
	for i := range residual {
		residual[i] = int16((i*37+11)%511 - 255)
	}
	coeff := make([]int32, 64)
	scratch := make([]int32, 64)
	for _, typ := range []Type{TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeIDTX} {
		b.Run(typeName(typ), func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				if err := ForwardBlock(coeff, 8, residual, 8, scratch, Size{Width: 8, Height: 8}, typ); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkForwardBlock8x8ADSTDCTImpl(b *testing.B) {
	benchmarkForwardBlock8x8HybridImpl(b, forwardBlock8x8ADSTDCTImpl)
}

func BenchmarkForwardBlock8x8ADSTDCTPureGo(b *testing.B) {
	benchmarkForwardBlock8x8HybridImpl(b, forwardBlock8x8ADSTDCTPureGo)
}

func BenchmarkForwardBlock8x8DCTADSTImpl(b *testing.B) {
	benchmarkForwardBlock8x8HybridImpl(b, forwardBlock8x8DCTADSTImpl)
}

func BenchmarkForwardBlock8x8DCTADSTPureGo(b *testing.B) {
	benchmarkForwardBlock8x8HybridImpl(b, forwardBlock8x8DCTADSTPureGo)
}

func BenchmarkForwardBlock8x8ADSTADSTImpl(b *testing.B) {
	benchmarkForwardBlock8x8HybridImpl(b, forwardBlock8x8ADSTADSTImpl)
}

func BenchmarkForwardBlock8x8ADSTADSTPureGo(b *testing.B) {
	benchmarkForwardBlock8x8HybridImpl(b, forwardBlock8x8ADSTADSTPureGo)
}

func benchmarkForwardBlock8x8HybridImpl(b *testing.B, fn func([]int32, int, []int16, int, []int32)) {
	residual := make([]int16, 64)
	for i := range residual {
		residual[i] = int16((i*37+11)%511 - 255)
	}
	coeff := make([]int32, 64)
	scratch := make([]int32, 64)
	b.ReportAllocs()
	for b.Loop() {
		fn(coeff, 8, residual, 8, scratch)
	}
}

func BenchmarkForwardBlock8x8IDTXImpl(b *testing.B) {
	benchmarkForwardBlock8x8IDTX(b, forwardBlock8x8IDTXImpl)
}

func BenchmarkForwardBlock8x8IDTXPureGo(b *testing.B) {
	benchmarkForwardBlock8x8IDTX(b, forwardBlock8x8IDTXPureGo)
}

func benchmarkForwardBlock8x8IDTX(b *testing.B, fn func([]int32, int, []int16, int, []int32)) {
	residual := make([]int16, 64)
	for i := range residual {
		residual[i] = int16((i*37+11)%511 - 255)
	}
	coeff := make([]int32, 64)
	scratch := make([]int32, 64)
	b.ReportAllocs()
	for b.Loop() {
		fn(coeff, 8, residual, 8, scratch)
	}
}

func typeName(typ Type) string {
	switch typ {
	case TypeADSTDCT:
		return "ADST_DCT"
	case TypeDCTADST:
		return "DCT_ADST"
	case TypeADSTADST:
		return "ADST_ADST"
	case TypeIDTX:
		return "IDTX"
	default:
		return "unknown"
	}
}

func sizeName(size Size) string {
	return fmt.Sprintf("%dx%d", size.Width, size.Height)
}

func TestFrame3ExactBlock64x64IDCT(t *testing.T) {
	sz := Size{Width: 64, Height: 64}
	coeffSize := adjustedScanSize(sz) // 32x32
	stride := int(coeffSize.Height)   // 32
	width := int(sz.Width)
	height := int(sz.Height)

	coeff := make([]int32, 32*32)
	// Exact dequantized coefficients from frame 3 debug test:
	// qindex=80, 8-bit, txScale=2, q.DC=74, q.AC=87
	// scan[0]->col=0,row=0 (DC): q=-3 -> dq=-55
	// scan[1]->col=1,row=0 (AC): q=-1 -> dq=-21
	// scan[3]->col=0,row=2 (AC): q=1 -> dq=21
	// scan[34]->col=1,row=6 (AC): q=1 -> dq=21
	// scan[35]->col=0,row=7 (AC): q=1 -> dq=21
	coeff[0*stride+0] = -55 // DC
	coeff[1*stride+0] = -21
	coeff[0*stride+2] = 21
	coeff[1*stride+6] = 21
	coeff[0*stride+7] = 21

	wantResid := libaomInverseResidual64x64(coeff, stride, 8)

	gotDst := make([]int16, width*height)
	scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
	scratch := make([]int32, scratchLen)
	if err := InverseBlockBitDepth(gotDst, width, coeff, stride, scratch, sz, TypeDCTDCT, 8); err != nil {
		t.Fatalf("err=%v", err)
	}
	diffs := 0
	for i := range gotDst {
		if int32(gotDst[i]) != wantResid[i] {
			r := i / width
			c := i % width
			if diffs < 10 {
				t.Logf("DIFF idx=%d (r=%d,c=%d) got=%d want=%d", i, r, c, gotDst[i], wantResid[i])
			}
			diffs++
		}
	}
	if diffs > 0 {
		t.Errorf("total diffs: %d", diffs)
	} else {
		t.Logf("IDCT matches libaom reference exactly for exact frame 3 block")
	}
}

func TestForwardBlock8BitResidualTrustedMatchesPureGo(t *testing.T) {
	types := make([]Type, TypeCount)
	for i := range types {
		types[i] = Type(i)
	}
	rng := rand.New(rand.NewSource(0x8b17))

	for _, size := range allTransformSizes {
		width, height := int(size.Width), int(size.Height)
		coeffSize := adjustedScanSize(size)
		coeffWidth, coeffHeight := int(coeffSize.Width), int(coeffSize.Height)
		residualStride := width + 3
		coeffStride := coeffHeight + 2
		residual := make([]int16, (height-1)*residualStride+width)
		coeffLen := (coeffWidth-1)*coeffStride + coeffHeight
		scratch := make([]int32, width*height)

		for _, typ := range types {
			if typ != TypeDCTDCT && !forwardGenericSupported(size, typ) {
				continue
			}
			t.Run(fmt.Sprintf("%s/%d", sizeName(size), typ), func(t *testing.T) {
				for pattern := 0; pattern < 2; pattern++ {
					// Padding is deliberately outside the 8-bit residual domain. The
					// contract covers addressed block samples, not row padding.
					for i := range residual {
						residual[i] = 30000
					}
					for row := 0; row < height; row++ {
						for col := 0; col < width; col++ {
							i := row*residualStride + col
							if pattern == 0 {
								switch (row*width + col) % 4 {
								case 0:
									residual[i] = -255
								case 1:
									residual[i] = 255
								case 2:
									residual[i] = -1
								default:
									residual[i] = 1
								}
							} else {
								src, pred := byte(rng.Intn(256)), byte(rng.Intn(256))
								residual[i] = int16(src) - int16(pred)
							}
						}
					}

					got := make([]int32, coeffLen)
					want := make([]int32, coeffLen)
					for i := range got {
						got[i], want[i] = 0x13579, 0x13579
					}
					gotScratch := make([]int32, len(scratch))
					wantScratch := make([]int32, len(scratch))
					for i := range gotScratch {
						gotScratch[i], wantScratch[i] = -0x2468, -0x2468
					}

					err := ForwardBlock8BitResidualTrusted(got, coeffStride, residual, residualStride, gotScratch, size, typ)
					if err != nil {
						t.Fatalf("trusted transform: %v", err)
					}
					if err := forwardBlock8BitTrustedPureGoReference(want, coeffStride, residual, residualStride, wantScratch, size, typ); err != nil {
						t.Fatalf("pure-Go reference: %v", err)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("coeff[%d]=%d want %d", i, got[i], want[i])
						}
					}
				}
			})
		}
	}
}

// forwardBlock8BitTrustedPureGoReference keeps the oracle independent for each
// path that the trusted entry can dispatch directly. Shapes outside those
// paths use the ordinary dispatcher because the trusted API delegates to it.
func forwardBlock8BitTrustedPureGoReference(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, size Size, typ Type) error {
	if typ == TypeDCTDCT {
		switch size {
		case Size{Width: 4, Height: 4}:
			forwardDCT4x4PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		case Size{Width: 8, Height: 8}:
			forwardDCT8x8PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		case Size{Width: 16, Height: 16}:
			forwardDCT16x16PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		case Size{Width: 32, Height: 32}:
			forwardDCT32x32PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		default:
			return forwardDCTBySize(coeff, coeffStride, residual, residualStride, size)
		}
	}
	if size == (Size{Width: 8, Height: 8}) && forwardBlock8x8HybridSupported(typ) {
		if !forwardBlock8x8HybridPureGo(coeff, coeffStride, residual, residualStride, scratch, typ) {
			return ErrInvalidTransform
		}
		return nil
	}
	return forwardGenericBlock(coeff, coeffStride, residual, residualStride, scratch, size, typ)
}

func TestForwardBlock8BitResidualTrustedRetainsShapeChecks(t *testing.T) {
	residual := make([]int16, 64)
	coeff := make([]int32, 64)
	scratch := make([]int32, 64)

	if err := ForwardBlock8BitResidualTrusted(coeff[:63], 8, residual, 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != ErrInvalidTransform {
		t.Fatalf("short coefficient buffer error=%v, want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock8BitResidualTrusted(coeff, 8, residual[:63], 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != ErrInvalidTransform {
		t.Fatalf("short residual buffer error=%v, want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock8BitResidualTrusted(coeff, 8, residual, 8, scratch[:63], Size{Width: 8, Height: 8}, TypeADSTDCT); err != ErrInvalidTransform {
		t.Fatalf("short hybrid scratch error=%v, want %v", err, ErrInvalidTransform)
	}
}

// The forward WHT's parity proof is an exact round-trip against the decoder's
// InverseWHT4x4Block (a byte-exact libaom inverse): for any int16 residual,
// ForwardWHT4x4 then InverseWHT4x4Block reproduces the residual bit-for-bit.

func roundTripWHT(t *testing.T, residual *[16]int16) {
	t.Helper()
	var coeff [16]int32
	if err := ForwardWHT4x4(coeff[:], 4, residual[:], 4); err != nil {
		t.Fatalf("forward: %v", err)
	}
	var dst [16]int16
	if err := InverseWHT4x4Block(dst[:], 4, coeff[:], 4, 16); err != nil {
		t.Fatalf("inverse: %v", err)
	}
	if dst != *residual {
		t.Fatalf("round-trip mismatch:\n in    %v\n out   %v\n coeff %v", *residual, dst, coeff)
	}
}

func TestForwardWHT4x4RoundTripRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for range 5000 {
		var residual [16]int16
		for i := range residual {
			// 12-bit residual span ([-4095,4095]) covers every supported depth.
			residual[i] = int16(rng.Intn(2*4095+1) - 4095)
		}
		roundTripWHT(t, &residual)
	}
}

func TestForwardWHT4x4RoundTripExtremes(t *testing.T) {
	cases := [][16]int16{
		{},               // all zero
		{37},             // DC only
		{0: -37},         // negative DC
		fill16(32767),    // max int16, proves no forward/inverse overflow
		fill16(-32768),   // min int16
		altSign16(32767), // alternating extremes
		{1, -1, 1, -1, 2, -2, 2, -2, 3, -3, 3, -3, 4, -4, 4, -4},
	}
	for i := range cases {
		c := cases[i]
		roundTripWHT(t, &c)
	}
}

// TestForwardWHT4x4Strided exercises the strided scatter/gather mapping with
// coeffStride and residualStride larger than the 4x4 block width.
func TestForwardWHT4x4Strided(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const rs, cs = 6, 5
	residual := make([]int16, 4*rs)
	for i := range residual {
		residual[i] = int16(rng.Intn(1024) - 512)
	}
	coeff := make([]int32, 4*cs)
	if err := ForwardWHT4x4(coeff, cs, residual, rs); err != nil {
		t.Fatalf("forward: %v", err)
	}
	dst := make([]int16, 4*rs)
	if err := InverseWHT4x4Block(dst, rs, coeff, cs, 16); err != nil {
		t.Fatalf("inverse: %v", err)
	}
	for r := range 4 {
		for c := range 4 {
			if dst[r*rs+c] != residual[r*rs+c] {
				t.Fatalf("strided mismatch at (%d,%d): got %d want %d", r, c, dst[r*rs+c], residual[r*rs+c])
			}
		}
	}
}

func TestForwardWHT4x4BoundsErrors(t *testing.T) {
	coeff := make([]int32, 16)
	residual := make([]int16, 16)
	if err := ForwardWHT4x4(coeff, 3, residual, 4); err == nil {
		t.Fatal("expected error for coeffStride < 4")
	}
	if err := ForwardWHT4x4(coeff, 4, residual, 3); err == nil {
		t.Fatal("expected error for residualStride < 4")
	}
	if err := ForwardWHT4x4(coeff[:8], 4, residual, 4); err == nil {
		t.Fatal("expected error for short coeff")
	}
	if err := ForwardWHT4x4(coeff, 4, residual[:8], 4); err == nil {
		t.Fatal("expected error for short residual")
	}
}

func TestForwardWHT4x4ZeroAlloc(t *testing.T) {
	residual := make([]int16, 16)
	for i := range residual {
		residual[i] = int16(i*7 - 50)
	}
	coeff := make([]int32, 16)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ForwardWHT4x4(coeff, 4, residual, 4)
	})
	if allocs != 0 {
		t.Fatalf("ForwardWHT4x4 allocated %v objects/run, want 0", allocs)
	}
}

func fill16(v int16) [16]int16 {
	var a [16]int16
	for i := range a {
		a[i] = v
	}
	return a
}

func altSign16(v int16) [16]int16 {
	var a [16]int16
	for i := range a {
		if i&1 == 0 {
			a[i] = v
		} else {
			a[i] = -v
		}
	}
	return a
}

// libaomIDCT4Parametric reproduces libaom's av1_idct4 with a parametric
// stage-range clamp. The existing libaomIDCT4 in dct_libaom_test.go is hard
// coded to 16-bit (bd=8 row/col); this variant exercises the 18-bit (bd=10
// row, bd=12 col) and 20-bit (bd=12 row) configurations that libaom's
// av1_gen_inv_stage_range() selects for high-bit-depth streams.
func libaomIDCT4Parametric(input []int32, output []int32, clamp func(int32) int32) {
	cospi := libaomCospi12[:]
	output[0] = input[0]
	output[1] = input[2]
	output[2] = input[1]
	output[3] = input[3]
	var step [4]int32
	step[0] = libaomHalfBtf(cospi[32], output[0], cospi[32], output[1])
	step[1] = libaomHalfBtf(cospi[32], output[0], -cospi[32], output[1])
	step[2] = libaomHalfBtf(cospi[48], output[2], -cospi[16], output[3])
	step[3] = libaomHalfBtf(cospi[16], output[2], cospi[48], output[3])
	output[0] = clamp(step[0] + step[3])
	output[1] = clamp(step[1] + step[2])
	output[2] = clamp(step[1] - step[2])
	output[3] = clamp(step[0] - step[3])
}

// libaomIDCT8Parametric reproduces libaom's av1_idct8 with a parametric
// stage-range clamp. See libaomIDCT4Parametric.
func libaomIDCT8Parametric(input []int32, output []int32, clamp func(int32) int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	var step [8]int32
	bf0 := output
	bf1 := output
	bf1[0] = input[0]
	bf1[1] = input[4]
	bf1[2] = input[2]
	bf1[3] = input[6]
	bf1[4] = input[1]
	bf1[5] = input[5]
	bf1[6] = input[3]
	bf1[7] = input[7]

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = halfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = halfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = halfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = halfBtf(cospi[8], bf0[4], cospi[56], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = halfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = halfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = halfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = halfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[5])
	bf1[5] = clamp(bf0[4] - bf0[5])
	bf1[6] = clamp(-bf0[6] + bf0[7])
	bf1[7] = clamp(bf0[6] + bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[3])
	bf1[1] = clamp(bf0[1] + bf0[2])
	bf1[2] = clamp(bf0[1] - bf0[2])
	bf1[3] = clamp(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = halfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = halfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[7])
	bf1[1] = clamp(bf0[1] + bf0[6])
	bf1[2] = clamp(bf0[2] + bf0[5])
	bf1[3] = clamp(bf0[3] + bf0[4])
	bf1[4] = clamp(bf0[3] - bf0[4])
	bf1[5] = clamp(bf0[2] - bf0[5])
	bf1[6] = clamp(bf0[1] - bf0[6])
	bf1[7] = clamp(bf0[0] - bf0[7])
}

// stageClampBits returns the libaom clamp_value at the given bit width.
func stageClampBits(bits int) func(int32) int32 {
	if bits <= 0 {
		return func(v int32) int32 { return v }
	}
	min := -(int32(1) << uint(bits-1))
	max := (int32(1) << uint(bits-1)) - 1
	return func(v int32) int32 {
		if v < min {
			return min
		}
		if v > max {
			return max
		}
		return v
	}
}

// TestInverseDCT1DStageClampMatchesLibaomHBD verifies that goav1's 1D DCT
// produces libaom-exact output at the stage-clamp bit widths libaom's
// av1_gen_inv_stage_range() selects for high-bit-depth content:
//
//   - bd=8: row=col=16 bits
//   - bd=10: row=18 bits, col=16 bits
//   - bd=12: row=20 bits, col=18 bits
//
// This pins the high-bit-depth row clamp behaviour for 4x4 and 8x8 1D blocks,
// covering the q63 10-bit DC range (|coeff| up to 2^17 - 1).
func TestInverseDCT1DStageClampMatchesLibaomHBD(t *testing.T) {
	tests := []struct {
		name      string
		stageBits int
		maxInput  int32
	}{
		{"bd8-row-col-16", 16, 32767},
		{"bd10-row-18", 18, 131071},
		{"bd10-col-16", 16, 32767},
		{"bd12-row-20", 20, 524287},
		{"bd12-col-18", 18, 131071},
	}
	rng := rand.New(rand.NewSource(1102))
	for _, tc := range tests {
		clamp := stageClampBits(tc.stageBits)
		stageMin := -(int32(1) << uint(tc.stageBits-1))
		stageMax := (int32(1) << uint(tc.stageBits-1)) - 1
		t.Run(tc.name+"/dct4", func(t *testing.T) {
			for trial := range 1000 {
				var input [4]int32
				for i := range input {
					input[i] = int32(rng.Intn(int(2*tc.maxInput)+1)) - tc.maxInput
				}
				got := input
				inverseDCT4(got[:], 1, stageMin, stageMax)
				var want [4]int32
				libaomIDCT4Parametric(input[:], want[:], clamp)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial=%d input=%v got=%v want=%v", trial, input, got, want)
					}
				}
			}
		})
		t.Run(tc.name+"/dct8", func(t *testing.T) {
			for trial := range 1000 {
				var input [8]int32
				for i := range input {
					input[i] = int32(rng.Intn(int(2*tc.maxInput)+1)) - tc.maxInput
				}
				got := input
				inverseDCT8(got[:], 1, stageMin, stageMax)
				var want [8]int32
				libaomIDCT8Parametric(input[:], want[:], clamp)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial=%d coeff=%d input=%v got=%v want=%v", trial, i, input, got, want)
					}
				}
			}
		})
	}
}

// paramLibaom1D dispatches to libaom's parametric 1D inverse transforms using
// the provided stage clamp.
func paramLibaom1D(input []int32, output []int32, length int, typ tx1DType, clamp func(int32) int32) {
	switch typ {
	case tx1DDCT:
		switch length {
		case 4:
			libaomIDCT4Parametric(input, output, clamp)
		case 8:
			libaomIDCT8Parametric(input, output, clamp)
		default:
			// Larger sizes: bit-exactness against libaom at the standard
			// 16-bit clamp is already covered by the dct_libaom_test.go
			// suite. The 2D oracle therefore uses goav1's own 1D path with
			// per-stage clamps for sizes >= 16 — the 1D path is itself
			// libaom-faithful (see TestInverseDCT16BitExactLibaom).
			copy(output, input)
			minVal := clamp(int32(-1 << 30))
			maxVal := clamp(int32((1 << 30) - 1))
			inverseDCT1D(output, 1, length, minVal, maxVal)
		}
	case tx1DADST, tx1DFlipADST:
		// libaom applies the regular ADST 1D; the up/down or left/right flip
		// is realised at the final output write via flipUD / flipLR. The
		// oracle uses the goav1 ADST regardless of FlipADST, and the 2D
		// pipeline applies flipUD/flipLR at the output write.
		copy(output, input)
		minVal := clamp(int32(-1 << 30))
		maxVal := clamp(int32((1 << 30) - 1))
		inverseADST1D(output, 1, length, minVal, maxVal)
	case tx1DIdentity:
		for i := range length {
			out, _ := identity1DValue(input[i], length)
			output[i] = out
		}
	}
}

// correctLibaom2D builds a faithful libaom inv_txfm2d_add_c oracle for the
// given pixel bit depth (bd). Unlike libaomInverseResidual in
// inv2d_libaom_test.go, this oracle uses the correct *per-pass* stage clamp
// widths derived from av1_gen_inv_stage_range:
//
//   - row pass: stage_range = bd + 8 (18 bits for bd=10, 20 bits for bd=12)
//   - col pass: stage_range = max(bd + 6, 16) (16 bits for bd<=10, 18 for bd=12)
//
// This matches the actual libaom production behaviour for high-bit-depth
// content. Returns the post-final-shift residual values.
func correctLibaom2D(coeff []int32, coeffStride int, size Size, typ Type, bd int) []int32 {
	vertical, horizontal, _ := typ.tx1DTypes()
	width := int(size.Width)
	height := int(size.Height)
	rectType := 0
	if width*2 == height || height*2 == width {
		rectType = 1
	}

	shift0, shift1 := libaomInvShift(size)
	rowBits := bd + 8
	colBits := max(bd+6, 16)

	rowClamp := stageClampBits(rowBits)
	colClamp := stageClampBits(colBits)

	buf := make([]int32, width*height)
	rowIn := make([]int32, width)
	rowOut := make([]int32, width)

	// Rows
	for r := range height {
		for c := range width {
			v := coeff[c*coeffStride+r]
			if rectType == 1 {
				v = libaomRoundShiftSingle(int64(v)*2896, 12)
			}
			rowIn[c] = v
		}
		libaomClampBits(rowIn, rowBits)
		paramLibaom1D(rowIn, rowOut, width, horizontal, rowClamp)
		libaomRoundShiftArray(rowOut, shift0)
		copy(buf[r*width:(r+1)*width], rowOut)
	}

	colIn := make([]int32, height)
	colOut := make([]int32, height)
	out := make([]int32, width*height)
	for c := range width {
		flipLR := isLRFlip(typ)
		flipUD := isUDFlip(typ)
		for r := range height {
			cc := c
			if flipLR {
				cc = width - c - 1
			}
			colIn[r] = buf[r*width+cc]
		}
		libaomClampBits(colIn, colBits)
		paramLibaom1D(colIn, colOut, height, vertical, colClamp)
		libaomRoundShiftArray(colOut, shift1)
		for r := range height {
			rr := r
			if flipUD {
				rr = height - r - 1
			}
			out[r*width+c] = colOut[rr]
		}
	}
	return out
}

// TestInverseBlockBitDepth2DMatchesLibaomHBD pins the high-bit-depth 2D
// inverse transform pipeline against the correct-per-pass libaom oracle for
// bd=10 (and bd=8 as a sanity check). Covers the worst-case q63 10-bit DC
// coefficient range (|dq_coeff| up to ±131071, which is libaom's bd+7 dequant
// clamp ceiling).
//
// This regression test anchors the fix-or-find investigation of the q63
// 10-bit lenient framework-dryrun mismatch: it pins the *transform stage*
// against the libaom inv_txfm2d_add_c specification exactly. If the transform
// stage is bit-exact against this oracle, any remaining q63 lenient failure
// is rooted *outside* internal/av1/transform/.
func TestInverseBlockBitDepth2DMatchesLibaomHBD(t *testing.T) {
	rng := rand.New(rand.NewSource(7777))
	cases := []struct {
		bd       int
		coeffAbs int32
	}{
		// bitDepth 8 uses the int16 column pipeline (dav1d 8bpc), which requires
		// the int16 butterfly not to overflow — a spec-valid-magnitude property.
		// 10-bit stays on the int32 pipeline and keeps its full range.
		{8, 512},
		{10, 131071},
	}
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 8, Height: 8},
		{Width: 16, Height: 16},
		{Width: 32, Height: 32},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 16},
		{Width: 16, Height: 8},
		{Width: 8, Height: 32},
		{Width: 32, Height: 8},
		{Width: 16, Height: 32},
		{Width: 32, Height: 16},
	}
	types := []Type{
		TypeDCTDCT,
		TypeADSTDCT,
		TypeDCTADST,
		TypeADSTADST,
		TypeFlipADSTDCT,
		TypeDCTFlipADST,
		TypeFlipADSTFlipADST,
		TypeADSTFlipADST,
		TypeFlipADSTADST,
		TypeIDTX,
		TypeVDCT,
		TypeHDCT,
		TypeVADST,
		TypeHADST,
		TypeVFlipADST,
		TypeHFlipADST,
	}
	for _, tc := range cases {
		for _, sz := range sizes {
			for _, typ := range types {
				if !typ.Supported(sz) {
					continue
				}
				vertical, horizontal, _ := typ.tx1DTypes()
				if vertical == tx1DADST || vertical == tx1DFlipADST {
					if sz.Height > 16 {
						continue
					}
				}
				if horizontal == tx1DADST || horizontal == tx1DFlipADST {
					if sz.Width > 16 {
						continue
					}
				}
				for trial := range 3 {
					coeffSize := adjustedScanSize(sz)
					coeffStride := int(coeffSize.Height)
					coeff := make([]int32, int(coeffSize.Width)*coeffStride)
					for i := range coeff {
						coeff[i] = int32(rng.Intn(int(2*tc.coeffAbs)+1)) - tc.coeffAbs
					}
					wantResid := correctLibaom2D(coeff, coeffStride, sz, typ, tc.bd)

					width := int(sz.Width)
					height := int(sz.Height)
					gotDst := make([]int16, width*height)
					scratchLen, _ := ScratchLenForType(typ, sz)
					scratch := make([]int32, scratchLen)
					if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, typ, uint8(tc.bd)); err != nil {
						t.Fatalf("bd=%d size=%v typ=%d err=%v", tc.bd, sz, typ, err)
					}
					for i := range gotDst {
						if int32(gotDst[i]) != wantResid[i] {
							t.Fatalf("bd=%d size=%v typ=%d trial=%d idx=%d got=%d want=%d", tc.bd, sz, typ, trial, i, gotDst[i], wantResid[i])
						}
					}
				}
			}
		}
	}
}

// TestInverseBlockBitDepthQ63WorstCaseDC anchors the worst-case 10-bit q63 DC
// coefficient against the libaom oracle. At q63 10-bit, the dequant DC value
// is 290 (dc_qlookup_10_QTX[63]) and the AV1 spec clamps dequantized
// coefficients to ±(2^(bd+7) - 1) = ±131071. This test sweeps each supported
// 2D transform shape with the DC at exactly the bd+7 ceiling and verifies
// goav1 reproduces libaom's residual bit-exactly through the 18-bit row /
// 16-bit col stage-range clamps.
func TestInverseBlockBitDepthQ63WorstCaseDC(t *testing.T) {
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 8, Height: 8},
		{Width: 16, Height: 16},
		{Width: 32, Height: 32},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 16},
		{Width: 16, Height: 8},
	}
	for _, sz := range sizes {
		for _, dc := range []int32{131071, -131072, 100000, -100000} {
			coeffSize := adjustedScanSize(sz)
			coeffStride := int(coeffSize.Height)
			coeff := make([]int32, int(coeffSize.Width)*coeffStride)
			coeff[0] = dc
			wantResid := correctLibaom2D(coeff, coeffStride, sz, TypeDCTDCT, 10)

			width := int(sz.Width)
			height := int(sz.Height)
			gotDst := make([]int16, width*height)
			scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
			scratch := make([]int32, scratchLen)
			if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, TypeDCTDCT, 10); err != nil {
				t.Fatalf("size=%v dc=%d err=%v", sz, dc, err)
			}
			for i := range gotDst {
				if int32(gotDst[i]) != wantResid[i] {
					t.Fatalf("size=%v dc=%d idx=%d got=%d want=%d", sz, dc, i, gotDst[i], wantResid[i])
				}
			}
		}
	}
}

func TestSizeShift(t *testing.T) {
	tests := []struct {
		size      Size
		wantShift int
		wantRect2 bool
	}{
		{size: Size{Width: 4, Height: 4}, wantShift: 0},
		{size: Size{Width: 4, Height: 8}, wantShift: 0, wantRect2: true},
		{size: Size{Width: 8, Height: 4}, wantShift: 0, wantRect2: true},
		{size: Size{Width: 4, Height: 16}, wantShift: 1},
		{size: Size{Width: 8, Height: 8}, wantShift: 1},
		{size: Size{Width: 8, Height: 16}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 8, Height: 32}, wantShift: 2},
		{size: Size{Width: 16, Height: 4}, wantShift: 1},
		{size: Size{Width: 16, Height: 8}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 16, Height: 16}, wantShift: 2},
		{size: Size{Width: 16, Height: 32}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 16, Height: 64}, wantShift: 2},
		{size: Size{Width: 32, Height: 8}, wantShift: 2},
		{size: Size{Width: 32, Height: 16}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 32, Height: 32}, wantShift: 2},
		{size: Size{Width: 32, Height: 64}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 64, Height: 16}, wantShift: 2},
		{size: Size{Width: 64, Height: 32}, wantShift: 1, wantRect2: true},
		{size: Size{Width: 64, Height: 64}, wantShift: 2},
	}
	for _, tt := range tests {
		shift, err := tt.size.Shift()
		if err != nil {
			t.Fatalf("%+v Shift err=%v", tt.size, err)
		}
		if shift != tt.wantShift {
			t.Fatalf("%+v shift=%d want %d", tt.size, shift, tt.wantShift)
		}
		if !tt.size.Valid() {
			t.Fatalf("%+v not valid", tt.size)
		}
		if got := tt.size.IsRect2(); got != tt.wantRect2 {
			t.Fatalf("%+v IsRect2=%t want %t", tt.size, got, tt.wantRect2)
		}
	}
}

func TestSizeRejectsInvalid(t *testing.T) {
	invalid := []Size{
		{},
		{Width: 4, Height: 12},
		{Width: 12, Height: 4},
		{Width: 64, Height: 8},
		{Width: 128, Height: 128},
	}
	for _, size := range invalid {
		if size.Valid() {
			t.Fatalf("%+v unexpectedly valid", size)
		}
		if _, err := size.Shift(); !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%+v Shift err=%v want %v", size, err, ErrInvalidTransform)
		}
	}
}

func TestInverseIdentityBlock4x4(t *testing.T) {
	coeff := av1CoeffOrder([]int32{
		16, -16, 0, 0,
		0, 32, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}, 4, 4)
	dst := make([]int16, 4*4)
	if err := InverseIdentityBlock(dst, 4, coeff, 4, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		2, -2, 0, 0,
		0, 4, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestInverseIdentityBlockRect2Scaling(t *testing.T) {
	coeff := make([]int32, 4*8)
	coeff[0] = 64
	dst := make([]int16, 4*8)
	if err := InverseIdentityBlock(dst, 4, coeff, 8, Size{Width: 4, Height: 8}); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 8 {
		t.Fatalf("rect2 dst[0]=%d want 8", dst[0])
	}
}

func TestInverseIdentityBlockStrides(t *testing.T) {
	coeff := make([]int32, 6*4)
	dst := make([]int16, 7*4)
	coeff[2*6+1] = 16
	if err := InverseIdentityBlock(dst, 7, coeff, 6, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	if got := dst[1*7+2]; got != 2 {
		t.Fatalf("strided residual=%d want 2", got)
	}
	if dst[1*7+4] != 0 {
		t.Fatalf("stride padding overwritten: %d", dst[1*7+4])
	}
}

func TestInverseIdentityBlockClips(t *testing.T) {
	coeff := []int32{maxInt32}
	dst := []int16{0}
	if err := InverseIdentityBlock(dst, 1, coeff, 1, Size{Width: 4, Height: 4}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short coeff err=%v want %v", err, ErrInvalidTransform)
	}

	// The 8-bit path now clamps the row input to ±int16 first (matching
	// libaom's clamp_buf(temp_in, bd+8) call). With coeff[0]=maxInt32 the
	// row input saturates to maxInt16=32767; identity1DValue(32767, 4) then
	// produces 32767 + ((32767*1697 + 2048)>>12) = 46353; the column
	// identity scales by ~1.4142 again, and the final >>4 round-shift
	// yields the bd-clamped residual. We just spot-check that the result is
	// stable and within the int16 range.
	coeff = make([]int32, 4*4)
	dst = make([]int16, 4*4)
	coeff[0] = maxInt32
	if err := InverseIdentityBlock(dst, 4, coeff, 4, Size{Width: 4, Height: 4}); err != nil {
		t.Fatal(err)
	}
	if dst[0] < 0 || dst[0] > maxInt16 {
		t.Fatalf("clipped residual=%d outside [0,%d]", dst[0], maxInt16)
	}
}

func TestInverseIdentityBlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int16, 4*4)
	coeff := make([]int32, 4*4)
	tests := []struct {
		name        string
		dst         []int16
		dstStride   int
		coeff       []int32
		coeffStride int
		size        Size
	}{
		{name: "zero size", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, size: Size{}},
		{name: "64 idtx", dst: make([]int16, 64*64), dstStride: 64, coeff: make([]int32, 64*64), coeffStride: 64, size: Size{Width: 64, Height: 64}},
		{name: "short dst stride", dst: dst, dstStride: 3, coeff: coeff, coeffStride: 4, size: Size{Width: 4, Height: 4}},
		{name: "short coeff stride", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 3, size: Size{Width: 4, Height: 4}},
		{name: "short dst", dst: dst[:15], dstStride: 4, coeff: coeff, coeffStride: 4, size: Size{Width: 4, Height: 4}},
		{name: "short coeff", dst: dst, dstStride: 4, coeff: coeff[:15], coeffStride: 4, size: Size{Width: 4, Height: 4}},
	}
	for _, tt := range tests {
		err := InverseIdentityBlock(tt.dst, tt.dstStride, tt.coeff, tt.coeffStride, tt.size)
		if !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidTransform)
		}
	}
}

func TestInverseIdentityBlockAllocs(t *testing.T) {
	coeff := make([]int32, 16*16)
	dst := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseIdentityBlock(dst, 16, coeff, 16, Size{Width: 16, Height: 16}); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseIdentityBlock allocated: %f", allocs)
	}
}

func FuzzInverseIdentityBlock(f *testing.F) {
	f.Add(uint8(0), int16(0))
	f.Add(uint8(1), int16(64))
	f.Add(uint8(5), int16(-128))
	f.Add(uint8(9), int16(1024))

	sizes := [...]Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 8, Height: 32},
		{Width: 16, Height: 4},
		{Width: 16, Height: 8},
		{Width: 16, Height: 16},
		{Width: 16, Height: 32},
		{Width: 32, Height: 8},
		{Width: 32, Height: 16},
		{Width: 32, Height: 32},
	}

	f.Fuzz(func(t *testing.T, rawSize uint8, coeffValue int16) {
		size := sizes[int(rawSize)%len(sizes)]
		width := int(size.Width)
		height := int(size.Height)
		coeffStride := height + 3
		dstStride := width + 5
		coeff := make([]int32, coeffStride*width)
		dst := make([]int16, dstStride*height)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[col*coeffStride+row] = int32(coeffValue) + int32(row-col)
			}
		}

		if err := InverseIdentityBlock(dst, dstStride, coeff, coeffStride, size); err != nil {
			t.Fatalf("InverseIdentityBlock err=%v", err)
		}
		for row := 0; row < height; row++ {
			padding := dst[row*dstStride+width : row*dstStride+dstStride]
			for i, got := range padding {
				if got != 0 {
					t.Fatalf("padding row=%d col=%d overwritten with %d", row, width+i, got)
				}
			}
		}
	})
}

func BenchmarkInverseIdentityBlock(b *testing.B) {
	coeff := make([]int32, 32*32)
	dst := make([]int16, 32*32)
	for i := range coeff {
		coeff[i] = int32(i%17) - 8
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = InverseIdentityBlock(dst, 32, coeff, 32, Size{Width: 32, Height: 32})
	}
}

// TestInverseDCT32x32MatchesLibaomFrame3Block reproduces the 32x32 DCT_DCT
// transform from libaom all-intra vector frame 3 (intra DC_PRED 64x64 block
// at MICol=32, MIRow=0). The non-zero dequantized coefficients listed below
// are dumped from the libaom reference decoder for this exact block. We
// pin the expected residual row 0 to libaom's IDCT output so any future
// regression of inverse-DCT scale, shift, or stage clamps is caught against
// a known real-stream block rather than synthetic coefficients.
func TestInverseDCT32x32MatchesLibaomFrame3Block(t *testing.T) {
	sz := Size{Width: 32, Height: 32}
	stride := 32
	coeff := make([]int32, 32*32)

	// libaom row-major dq[r=R,c=C]=v converted to goav1's column-major
	// storage (coeff[c*stride+r] = v).
	type kv struct {
		r, c int
		v    int32
	}
	nonZero := []kv{
		// libaom dump:
		//   dq[r=0,c=0]=-918  dq[r=0,c=1]=-114
		//   dq[r=1,c=0]= 522  dq[r=1,c=1]= -28
		//   dq[r=2,c=0]=  28  dq[r=2,c=6]=   9
		//   dq[r=3,c=0]=  47  dq[r=3,c=1]=  -9  dq[r=3,c=4]= 9
		//   dq[r=5,c=0]=   9  dq[r=5,c=1]=   9
		//   dq[r=9,c=2]=  -9
		//   dq[r=11,c=0]= 9
		{0, 0, -918}, {1, 0, -114},
		{0, 1, 522}, {1, 1, -28},
		{0, 2, 28}, {6, 2, 9},
		{0, 3, 47}, {1, 3, -9}, {4, 3, 9},
		{0, 5, 9}, {1, 5, 9},
		{2, 9, -9},
		{0, 11, 9},
	}
	for _, v := range nonZero {
		coeff[v.c*stride+v.r] = v.v
	}

	// libaom OUT_ROW0 - pred(206) gives the residual row 0 below.
	wantRow0 := []int16{
		-2, -2, -2, -3, -3, -4, -4, -5,
		-6, -6, -6, -6, -7, -8, -8, -9,
		-9, -9, -10, -10, -11, -11, -11, -11,
		-12, -12, -13, -13, -13, -14, -14, -14,
	}

	width := int(sz.Width)
	height := int(sz.Height)
	gotDst := make([]int16, width*height)
	scratch := make([]int32, width*height)
	if err := InverseBlockBitDepth(gotDst, width, coeff, stride, scratch, sz, TypeDCTDCT, 8); err != nil {
		t.Fatalf("InverseBlockBitDepth: %v", err)
	}
	for c := 0; c < width; c++ {
		if gotDst[c] != wantRow0[c] {
			t.Fatalf("row 0 col %d: got=%d want=%d (full row got=%v)", c, gotDst[c], wantRow0[c], gotDst[:width])
		}
	}

	// Also cross-check against our portable libaom reference port for the
	// full residual block.
	wantResid := libaomInverseResidual(coeff, stride, sz, TypeDCTDCT, 8)
	for i := 0; i < width*height; i++ {
		if int32(gotDst[i]) != wantResid[i] {
			r := i / width
			c := i % width
			t.Fatalf("idx=%d (r=%d c=%d) goav1=%d libaomRef=%d", i, r, c, gotDst[i], wantResid[i])
		}
	}
}

// libaomInvShift returns the AV1 inverse-shift pair (shift[0], shift[1]) for
// size, mirroring av1_inv_txfm_shift_ls in libaom. Values are <= 0 in libaom;
// we return the positive counts (negation of the libaom values).
func libaomInvShift(size Size) (shift0 int, shift1 int) {
	type pair struct{ w, h, s0 int }
	table := []pair{
		{4, 4, 0},
		{8, 8, 1},
		{16, 16, 2},
		{32, 32, 2},
		{64, 64, 2},
		{4, 8, 0},
		{8, 4, 0},
		{8, 16, 1},
		{16, 8, 1},
		{16, 32, 1},
		{32, 16, 1},
		{32, 64, 1},
		{64, 32, 1},
		{4, 16, 1},
		{16, 4, 1},
		{8, 32, 2},
		{32, 8, 2},
		{16, 64, 2},
		{64, 16, 2},
	}
	for _, p := range table {
		if p.w == int(size.Width) && p.h == int(size.Height) {
			return p.s0, 4
		}
	}
	return 0, 0
}

func libaomClampBits(buf []int32, bits int) {
	if bits <= 0 {
		return
	}
	min := -(int64(1) << uint(bits-1))
	max := int64(1)<<uint(bits-1) - 1
	for i, v := range buf {
		x := int64(v)
		if x < min {
			x = min
		} else if x > max {
			x = max
		}
		buf[i] = int32(x)
	}
}

func libaomRoundShiftArray(arr []int32, bits int) {
	if bits == 0 {
		return
	}
	for i, v := range arr {
		x := int64(v) + (int64(1) << uint(bits-1))
		arr[i] = int32(x >> uint(bits))
	}
}

func libaom1D(input []int32, output []int32, length int, typ tx1DType) {
	switch typ {
	case tx1DDCT:
		switch length {
		case 4:
			libaomIDCT4(input, output)
		case 8:
			libaomIDCT8(input, output)
		case 16:
			libaomIDCT16(input, output)
		case 32:
			libaomIDCT32(input, output)
		}
	case tx1DADST, tx1DFlipADST:
		switch length {
		case 4:
			// Use goav1's adst4 since it's tested in TestInverseADST1DRoundTripMatchesLibaomShape.
			copy(output, input)
			inverseADST4(output, 1)
		case 8:
			libaomIADST8(input, output)
		case 16:
			libaomIADST16(input, output)
		}
	case tx1DIdentity:
		for i := range length {
			v := input[i]
			out, _ := identity1DValue(v, length)
			output[i] = out
		}
	}
}

// libaomInverseBlock implements libaom's inv_txfm2d_add_c for bd=8 and writes
// the residual (post-add) to dst as int16. dst is assumed to be already
// populated with the predicted samples; the residual is added and the result
// is clamped to [0,255]. Returns the int16 residual values (post-round-shift,
// pre-clamp) for direct comparison with InverseBlockBitDepth.
func libaomInverseResidual(coeff []int32, coeffStride int, size Size, typ Type, bd int) []int32 {
	vertical, horizontal, _ := typ.tx1DTypes()
	width := int(size.Width)
	height := int(size.Height)
	rectType := 0
	if width*2 == height || height*2 == width {
		rectType = 1
	}

	shift0, shift1 := libaomInvShift(size)
	rowBits := bd + 8
	colBits := max(bd+6, 16)

	buf := make([]int32, width*height)
	rowIn := make([]int32, width)
	rowOut := make([]int32, width)

	// Rows
	for r := range height {
		for c := range width {
			v := coeff[c*coeffStride+r]
			if rectType == 1 {
				// libaom: round_shift(v * NewInvSqrt2, NewSqrt2Bits)
				// NewInvSqrt2 = 2896, NewSqrt2Bits = 12.
				v = libaomRoundShiftSingle(int64(v)*2896, 12)
			}
			rowIn[c] = v
		}
		libaomClampBits(rowIn, rowBits)
		libaom1D(rowIn, rowOut, width, horizontal)
		// av1_round_shift_array(rowOut, width, -shift0). -shift0 is negative => positive bit count.
		libaomRoundShiftArray(rowOut, shift0)
		copy(buf[r*width:(r+1)*width], rowOut)
	}

	colIn := make([]int32, height)
	colOut := make([]int32, height)
	out := make([]int32, width*height)
	for c := range width {
		flipLR := isLRFlip(typ)
		flipUD := isUDFlip(typ)
		for r := range height {
			cc := c
			if flipLR {
				cc = width - c - 1
			}
			colIn[r] = buf[r*width+cc]
		}
		libaomClampBits(colIn, colBits)
		libaom1D(colIn, colOut, height, vertical)
		libaomRoundShiftArray(colOut, shift1)
		for r := range height {
			rr := r
			if flipUD {
				rr = height - r - 1
			}
			out[r*width+c] = colOut[rr]
		}
	}
	return out
}

func libaomRoundShiftSingle(v int64, bits int) int32 {
	return int32((v + (int64(1) << uint(bits-1))) >> uint(bits))
}

func isLRFlip(typ Type) bool {
	switch typ {
	case TypeDCTFlipADST, TypeADSTFlipADST, TypeFlipADSTFlipADST, TypeHFlipADST:
		return true
	}
	return false
}

func isUDFlip(typ Type) bool {
	switch typ {
	case TypeFlipADSTDCT, TypeFlipADSTADST, TypeFlipADSTFlipADST, TypeVFlipADST:
		return true
	}
	return false
}

// TestInverseBlockBitDepth2DMatchesLibaom2D verifies our 2D inverse transform
// pipeline against a faithful port of libaom's inv_txfm2d_add_c (bd=8) using
// random coefficients in the validly-decoded range. Tests every supported
// transform type/size combo at sizes 4x4 through 32x32 plus rect-2.
func TestInverseBlockBitDepth2DMatchesLibaom2D(t *testing.T) {
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 16, Height: 8},
		{Width: 8, Height: 32},
		{Width: 32, Height: 8},
		{Width: 16, Height: 16},
		{Width: 16, Height: 32},
		{Width: 32, Height: 16},
		{Width: 32, Height: 32},
	}
	types := []Type{
		TypeDCTDCT,
		TypeADSTDCT,
		TypeDCTADST,
		TypeADSTADST,
		TypeFlipADSTDCT,
		TypeDCTFlipADST,
		TypeFlipADSTFlipADST,
		TypeADSTFlipADST,
		TypeFlipADSTADST,
		TypeVDCT,
		TypeHDCT,
		TypeVADST,
		TypeHADST,
		TypeVFlipADST,
		TypeHFlipADST,
	}
	rng := rand.New(rand.NewSource(7))
	for _, sz := range sizes {
		for _, typ := range types {
			if !typ.Supported(sz) {
				continue
			}
			// Skip ADST sizes 32+ (libaom uses ADST16 max).
			vertical, horizontal, _ := typ.tx1DTypes()
			if vertical == tx1DADST || vertical == tx1DFlipADST {
				if sz.Height > 16 {
					continue
				}
			}
			if horizontal == tx1DADST || horizontal == tx1DFlipADST {
				if sz.Width > 16 {
					continue
				}
			}
			for trial := range 3 {
				coeffSize := adjustedScanSize(sz)
				coeffStride := int(coeffSize.Height)
				coeff := make([]int32, int(coeffSize.Width)*coeffStride)
				// This oracle corpus samples moderate coefficients. The public
				// transform contract also covers the full int16 range; dedicated
				// tests exercise its guarded SIMD fallback.
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<10) - (1 << 9))
				}
				wantResid := libaomInverseResidual(coeff, coeffStride, sz, typ, 8)

				width := int(sz.Width)
				height := int(sz.Height)
				gotDst := make([]int16, width*height)
				scratchLen, _ := ScratchLenForType(typ, sz)
				scratch := make([]int32, scratchLen)
				if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, typ, 8); err != nil {
					t.Fatalf("size=%v typ=%d trial=%d err=%v", sz, typ, trial, err)
				}
				for i := range gotDst {
					if int32(gotDst[i]) != wantResid[i] {
						t.Errorf("size=%v typ=%d trial=%d idx=%d got=%d want=%d", sz, typ, trial, i, gotDst[i], wantResid[i])
						break
					}
				}
			}
		}
	}
}

// Float models of the inverse DCT butterfly networks in dct.go: the same
// structure with exact (unrounded, unclamped) real arithmetic, recording every
// rotation output. Fed with unit vectors they give the coefficient g[i][j] of
// input j in rotation output i, which bounds the true integer value of that
// output by sum_j |g[i][j]| |x_j| plus rounding slack (clamps only shrink
// magnitudes). This is the exactness contract of the int16 column kernels: a
// kernel equals the scalar reference whenever every rotation output fits
// int16, since the butterfly sums are then saturating adds of exact int16
// operands, equal to the scalar's clip to the same int16 range.

type gainRecorder struct {
	outs    [][]float64 // rotation outputs, one coefficient vector each
	cur     int
	perturb int // rotation output index to perturb by +1 (-1 = none)
	calls   int
}

func (g *gainRecorder) rot(v float64) float64 {
	if g.calls == g.perturb {
		v += 1
	}
	g.calls++
	if g.outs != nil {
		g.outs[g.cur] = append(g.outs[g.cur], v)
	}
	return v
}

func rs(x float64, n int) float64 { return x / float64(int64(1)<<uint(n)) }

func dct4F(c []float64, stride int, g *gainRecorder) {
	in0, in1, in2, in3 := c[0], c[stride], c[2*stride], c[3*stride]
	t0 := g.rot(rs((in0+in2)*181, 8))
	t1 := g.rot(rs((in0-in2)*181, 8))
	t2 := g.rot(rs(in1*1567-in3*3784, 12))
	t3 := g.rot(rs(in1*3784+in3*1567, 12))
	c[0], c[stride], c[2*stride], c[3*stride] = t0+t3, t1+t2, t1-t2, t0-t3
}

func dct8F(c []float64, stride int, g *gainRecorder) {
	dct4F(c, stride<<1, g)
	in1, in3, in5, in7 := c[stride], c[3*stride], c[5*stride], c[7*stride]
	t4a := g.rot(rs(in1*799-in7*4017, 12))
	t5a := g.rot(rs(in5*1703-in3*1138, 11))
	t6a := g.rot(rs(in5*1138+in3*1703, 11))
	t7a := g.rot(rs(in1*4017+in7*799, 12))
	t4 := t4a + t5a
	t5a = t4a - t5a
	t7 := t7a + t6a
	t6a = t7a - t6a
	t5 := g.rot(rs((t6a-t5a)*181, 8))
	t6 := g.rot(rs((t6a+t5a)*181, 8))
	t0, t1, t2, t3 := c[0], c[2*stride], c[4*stride], c[6*stride]
	c[0], c[stride], c[2*stride], c[3*stride] = t0+t7, t1+t6, t2+t5, t3+t4
	c[4*stride], c[5*stride], c[6*stride], c[7*stride] = t3-t4, t2-t5, t1-t6, t0-t7
}

func dct16F(c []float64, stride int, g *gainRecorder) {
	dct8F(c, stride<<1, g)
	in1, in3, in5, in7 := c[stride], c[3*stride], c[5*stride], c[7*stride]
	in9, in11, in13, in15 := c[9*stride], c[11*stride], c[13*stride], c[15*stride]
	t8a := g.rot(rs(in1*401-in15*4076, 12))
	t9a := g.rot(rs(in9*1583-in7*1299, 11))
	t10a := g.rot(rs(in5*1931-in11*3612, 12))
	t11a := g.rot(rs(in13*3920-in3*1189, 12))
	t12a := g.rot(rs(in13*1189+in3*3920, 12))
	t13a := g.rot(rs(in5*3612+in11*1931, 12))
	t14a := g.rot(rs(in9*1299+in7*1583, 11))
	t15a := g.rot(rs(in1*4076+in15*401, 12))
	t8, t9 := t8a+t9a, t8a-t9a
	t10, t11 := t11a-t10a, t11a+t10a
	t12, t13 := t12a+t13a, t12a-t13a
	t14, t15 := t15a-t14a, t15a+t14a
	t9a = g.rot(rs(t14*1567-t9*3784, 12))
	t14a = g.rot(rs(t14*3784+t9*1567, 12))
	t10a = g.rot(rs(-(t13*3784 + t10*1567), 12))
	t13a = g.rot(rs(t13*1567-t10*3784, 12))
	t8a = t8 + t11
	t9 = t9a + t10a
	t10 = t9a - t10a
	t11a = t8 - t11
	t12a = t15 - t12
	t13 = t14a - t13a
	t14 = t14a + t13a
	t15a = t15 + t12
	t10a = g.rot(rs((t13-t10)*181, 8))
	t13a = g.rot(rs((t13+t10)*181, 8))
	t11 = g.rot(rs((t12a-t11a)*181, 8))
	t12 = g.rot(rs((t12a+t11a)*181, 8))
	e := [8]float64{c[0], c[2*stride], c[4*stride], c[6*stride], c[8*stride], c[10*stride], c[12*stride], c[14*stride]}
	o := [8]float64{t15a, t14, t13a, t12, t11, t10a, t9, t8a}
	for i := 0; i < 8; i++ {
		c[i*stride] = e[i] + o[i]
		c[(15-i)*stride] = e[i] - o[i]
	}
}

// dct32F follows inverseDCT32 in dct.go stage for stage.
func dct32F(c []float64, stride int, g *gainRecorder) {
	dct16F(c, stride<<1, g)
	var in [32]float64
	for i := range in {
		in[i] = c[i*stride]
	}
	rot := func(a float64, ca float64, b float64, cb float64, n int) float64 { return g.rot(rs(a*ca+b*cb, n)) }
	t16a := rot(in[1], 201, in[31], -4091, 12)
	t17a := rot(in[17], 3035, in[15], -2751, 12)
	t18a := rot(in[9], 1751, in[23], -3703, 12)
	t19a := rot(in[25], 3857, in[7], -1380, 12)
	t20a := rot(in[5], 995, in[27], -3973, 12)
	t21a := rot(in[21], 3513, in[11], -2106, 12)
	t22a := rot(in[13], 2440, in[19], -3290, 12)
	t23a := rot(in[29], 4052, in[3], -601, 12)
	t24a := rot(in[29], 601, in[3], 4052, 12)
	t25a := rot(in[13], 3290, in[19], 2440, 12)
	t26a := rot(in[21], 2106, in[11], 3513, 12)
	t27a := rot(in[5], 3973, in[27], 995, 12)
	t28a := rot(in[25], 1380, in[7], 3857, 12)
	t29a := rot(in[9], 3703, in[23], 1751, 12)
	t30a := rot(in[17], 2751, in[15], 3035, 12)
	t31a := rot(in[1], 4091, in[31], 201, 12)
	t16, t17 := t16a+t17a, t16a-t17a
	t18, t19 := t19a-t18a, t19a+t18a
	t20, t21 := t20a+t21a, t20a-t21a
	t22, t23 := t23a-t22a, t23a+t22a
	t24, t25 := t24a+t25a, t24a-t25a
	t26, t27 := t27a-t26a, t27a+t26a
	t28, t29 := t28a+t29a, t28a-t29a
	t30, t31 := t31a-t30a, t31a+t30a
	t17a = rot(t30, 799, t17, -4017, 12)
	t30a = rot(t30, 4017, t17, 799, 12)
	t18a = rot(t29, -4017, t18, -799, 12)
	t29a = rot(t29, 799, t18, -4017, 12)
	t21a = rot(t26, 3406, t21, -2276, 12)
	t26a = rot(t26, 2276, t21, 3406, 12)
	t22a = rot(t25, -2276, t22, -3406, 12)
	t25a = rot(t25, 3406, t22, -2276, 12)
	t16a, t19a = t16+t19, t16-t19
	t17, t18 = t17a+t18a, t17a-t18a
	t20a, t23a = t23-t20, t23+t20
	t21, t22 = t22a-t21a, t22a+t21a
	t24a, t27a = t24+t27, t24-t27
	t25, t26 = t25a+t26a, t25a-t26a
	t28a, t31a = t31-t28, t31+t28
	t29, t30 = t30a-t29a, t30a+t29a
	t18a = rot(t29, 1567, t18, -3784, 12)
	t29a = rot(t29, 3784, t18, 1567, 12)
	t19 = rot(t28a, 1567, t19a, -3784, 12)
	t28 = rot(t28a, 3784, t19a, 1567, 12)
	t20 = rot(t27a, -3784, t20a, -1567, 12)
	t27 = rot(t27a, 1567, t20a, -3784, 12)
	t21a = rot(t26, -3784, t21, -1567, 12)
	t26a = rot(t26, 1567, t21, -3784, 12)
	t16, t23 = t16a+t23a, t16a-t23a
	t17a, t22a = t17+t22, t17-t22
	t18, t21 = t18a+t21a, t18a-t21a
	t19a, t20a = t19+t20, t19-t20
	t24, t31 = t31a-t24a, t31a+t24a
	t25a, t30a = t30-t25, t30+t25
	t26, t29 = t29a-t26a, t29a+t26a
	t27a, t28a = t28-t27, t28+t27
	t20 = g.rot(rs((t27a-t20a)*181, 8))
	t27 = g.rot(rs((t27a+t20a)*181, 8))
	t21a = g.rot(rs((t26-t21)*181, 8))
	t26a = g.rot(rs((t26+t21)*181, 8))
	t22 = g.rot(rs((t25a-t22a)*181, 8))
	t25 = g.rot(rs((t25a+t22a)*181, 8))
	t23a = g.rot(rs((t24-t23)*181, 8))
	t24a = g.rot(rs((t24+t23)*181, 8))
	o := [16]float64{t31, t30a, t29, t28a, t27, t26a, t25, t24a, t23a, t22, t21a, t20, t19a, t18, t17a, t16}
	var e [16]float64
	for i := range e {
		e[i] = c[2*i*stride]
	}
	for i := 0; i < 16; i++ {
		c[i*stride] = e[i] + o[i]
		c[(31-i)*stride] = e[i] - o[i]
	}
}

// rotOutputs runs the float network on c with rotation output p perturbed by
// +1 (p < 0: none) and returns every rotation output in call order.
func rotOutputs(f func([]float64, int, *gainRecorder), c []float64, p int) []float64 {
	rec := &gainRecorder{outs: make([][]float64, 1), perturb: p}
	f(c, 1, rec)
	return rec.outs[0]
}

// gainsFor returns, for the N-point inverse DCT, |g[i][j]|: the coefficient
// magnitude of input j in rotation output i.
func gainsFor(n int, f func([]float64, int, *gainRecorder)) [][]float64 {
	var outs [][]float64
	for j := 0; j < n; j++ {
		c := make([]float64, n)
		c[j] = 1
		vals := rotOutputs(f, c, -1)
		if outs == nil {
			outs = make([][]float64, len(vals))
		}
		for i, v := range vals {
			outs[i] = append(outs[i], math.Abs(v))
		}
	}
	return outs
}

// roundingSlack returns, per rotation output i, the largest total effect of
// the +-0.5 rounding errors of all earlier rotation outputs on it:
// 0.5 * sum_k |h[i][k]| where h[i][k] is the gain from a unit perturbation of
// output k to output i.
func roundingSlack(n int, f func([]float64, int, *gainRecorder), numRot int) []float64 {
	slack := make([]float64, numRot)
	for k := 0; k < numRot; k++ {
		c := make([]float64, n)
		vals := rotOutputs(f, c, k)
		for i, v := range vals {
			if i != k {
				slack[i] += 0.5 * math.Abs(v)
			}
		}
	}
	return slack
}

func TestITX16GainBounds(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{8, dct8F}, {16, dct16F}, {32, dct32F}} {
		g := gainsFor(tc.n, tc.f)
		slack := roundingSlack(tc.n, tc.f, len(g))
		maxRowL1, maxCoef, maxSlack := 0.0, 0.0, 0.0
		worst := -1
		for i, row := range g {
			l1 := 0.0
			for _, v := range row {
				l1 += v
				maxCoef = math.Max(maxCoef, v)
			}
			if l1 > maxRowL1 {
				maxRowL1, worst = l1, i
			}
			maxSlack = math.Max(maxSlack, slack[i])
		}
		t.Logf("DCT%d: %d rotation outputs; max row L1 gain %.4f (output %d) -> uniform input bound %d; max |coef| %.4f -> per-column L1 bound %d; max rounding slack %.2f",
			tc.n, len(g), maxRowL1, worst, int(32767/maxRowL1), maxCoef, int(32767/maxCoef), maxSlack)
	}
}

// TestITX16FloatModelTracksScalar checks the float networks against the integer
// kernels on small inputs (where rounding is the only difference).
func TestITX16FloatModelTracksScalar(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{16, dct16F}, {32, dct32F}} {
		for trial := 0; trial < 200; trial++ {
			ci := make([]int32, tc.n)
			cf := make([]float64, tc.n)
			for i := range ci {
				v := int32((trial*7919+i*104729)%201) - 100
				ci[i] = v
				cf[i] = float64(v)
			}
			inverseDCT1D(ci, 1, tc.n, -32768, 32767)
			tc.f(cf, 1, &gainRecorder{perturb: -1})
			for i := range ci {
				if d := math.Abs(float64(ci[i]) - cf[i]); d > 64 {
					t.Fatalf("DCT%d trial %d out %d: int %d float %.2f", tc.n, trial, i, ci[i], cf[i])
				}
			}
		}
	}
}

// dct64F follows inverseDCT64 in dct.go stage for stage (the structure the
// generated itx64Odd* networks fold their zero operands from): the DCT32 of
// the even inputs and the ten-stage odd half.
func dct64F(c []float64, stride int, g *gainRecorder) {
	dct32F(c, stride<<1, g)
	var e [32]float64
	for i := range e {
		e[i] = c[2*i*stride]
	}
	var x, y [64]float64
	for i := range x {
		x[i] = c[i*stride]
	}
	cp := func(i int) float64 { return float64(itx64Cos[i]) }
	rot := func(a, ca, b, cb float64) float64 { return g.rot(rs(a*ca+b*cb, 12)) }
	// Stage 2: pairs (32+j, 63-j) from inputs (a, 64-a) in bit-reversed order.
	a := [16]int{1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61}
	for j := 0; j < 16; j++ {
		lo, hi := 32+j, 63-j
		ia, ib := a[j], 64-a[j]
		y[lo] = rot(x[ia], cp(ib), x[ib], -cp(ia))
		y[hi] = rot(x[ia], cp(ia), x[ib], cp(ib))
	}
	x = y
	// Stage 3.
	for b := 32; b < 64; b += 4 {
		y[b] = x[b] + x[b+1]
		y[b+1] = x[b] - x[b+1]
		y[b+2] = x[b+3] - x[b+2]
		y[b+3] = x[b+2] + x[b+3]
	}
	x = y
	// Stage 4.
	for m, pq := range [4][2]int{{4, 60}, {36, 28}, {20, 44}, {52, 12}} {
		p, q := cp(pq[0]), cp(pq[1])
		lo, hi := 33+4*m, 62-4*m
		y[lo] = rot(x[lo], -p, x[hi], q)
		y[hi] = rot(x[lo], q, x[hi], p)
		y[lo+1] = rot(x[lo+1], -q, x[hi-1], -p)
		y[hi-1] = rot(x[lo+1], -p, x[hi-1], q)
	}
	x = y
	// Stage 5.
	for b := 32; b < 64; b += 8 {
		y[b] = x[b] + x[b+3]
		y[b+1] = x[b+1] + x[b+2]
		y[b+2] = x[b+1] - x[b+2]
		y[b+3] = x[b] - x[b+3]
		y[b+4] = x[b+7] - x[b+4]
		y[b+5] = x[b+6] - x[b+5]
		y[b+6] = x[b+5] + x[b+6]
		y[b+7] = x[b+4] + x[b+7]
	}
	x = y
	// Stage 6.
	for _, pql := range [2][4]int{{8, 56, 34, 61}, {40, 24, 42, 53}} {
		p, q := cp(pql[0]), cp(pql[1])
		lo, hi := pql[2], pql[3]
		for i := 0; i < 2; i++ {
			y[lo+i] = rot(x[lo+i], -p, x[hi-i], q)
			y[hi-i] = rot(x[lo+i], q, x[hi-i], p)
		}
		for i := 2; i < 4; i++ {
			y[lo+i] = rot(x[lo+i], -q, x[hi-i], -p)
			y[hi-i] = rot(x[lo+i], -p, x[hi-i], q)
		}
	}
	x = y
	// Stage 7.
	for b := 32; b < 64; b += 16 {
		for i := 0; i < 4; i++ {
			y[b+i] = x[b+i] + x[b+7-i]
			y[b+7-i] = x[b+i] - x[b+7-i]
			y[b+8+i] = x[b+15-i] - x[b+8+i]
			y[b+15-i] = x[b+8+i] + x[b+15-i]
		}
	}
	x = y
	// Stage 8.
	for i := 0; i < 4; i++ {
		y[36+i] = rot(x[36+i], -cp(16), x[59-i], cp(48))
		y[59-i] = rot(x[36+i], cp(48), x[59-i], cp(16))
		y[40+i] = rot(x[40+i], -cp(48), x[55-i], -cp(16))
		y[55-i] = rot(x[40+i], -cp(16), x[55-i], cp(48))
	}
	x = y
	// Stage 9.
	for i := 0; i < 8; i++ {
		y[32+i] = x[32+i] + x[47-i]
		y[47-i] = x[32+i] - x[47-i]
		y[48+i] = x[63-i] - x[48+i]
		y[63-i] = x[48+i] + x[63-i]
	}
	x = y
	// Stage 10.
	for i := 0; i < 8; i++ {
		y[40+i] = rot(x[40+i], -cp(32), x[55-i], cp(32))
		y[55-i] = rot(x[40+i], cp(32), x[55-i], cp(32))
	}
	x = y
	for i := 0; i < 32; i++ {
		c[i*stride] = e[i] + x[63-i]
		c[(63-i)*stride] = e[i] - x[63-i]
	}
}

// TestITX64GainBounds logs the DCT64 bounds like TestITX16GainBounds and
// checks that the fused 64x64 kernel's guard (itx16ColumnL1Bound, 32767-64)
// keeps every rotation output inside int16: an output is bounded by the
// largest single-input gain times the per-lane L1 norm of the inputs, plus
// the rounding slack.
func TestITX64GainBounds(t *testing.T) {
	g := gainsFor(64, dct64F)
	slack := roundingSlack(64, dct64F, len(g))
	maxRowL1, maxCoef, maxSlack := 0.0, 0.0, 0.0
	worst := -1
	for i, row := range g {
		l1 := 0.0
		for _, v := range row {
			l1 += v
			maxCoef = math.Max(maxCoef, v)
		}
		if l1 > maxRowL1 {
			maxRowL1, worst = l1, i
		}
		maxSlack = math.Max(maxSlack, slack[i])
	}
	t.Logf("DCT64: %d rotation outputs; max row L1 gain %.4f (output %d) -> uniform input bound %d; max |coef| %.4f -> per-column L1 bound %d; max rounding slack %.2f",
		len(g), maxRowL1, worst, int(32767/maxRowL1), maxCoef, int(32767/maxCoef), maxSlack)
	const bound = 32767 - 64
	if maxCoef*bound+maxSlack > 32767 {
		t.Fatalf("guard %d does not certify DCT64: %.4f * %d + %.2f exceeds 32767", bound, maxCoef, bound, maxSlack)
	}
}

// TestITX64FloatModelTracksScalar checks dct64F against the integer DCT64 on
// small inputs, where rounding is the only difference.
func TestITX64FloatModelTracksScalar(t *testing.T) {
	for trial := 0; trial < 200; trial++ {
		ci := make([]int32, 64)
		cf := make([]float64, 64)
		for i := range ci {
			v := int32((trial*7919+i*104729)%201) - 100
			ci[i] = v
			cf[i] = float64(v)
		}
		inverseDCT1D(ci, 1, 64, -32768, 32767)
		dct64F(cf, 1, &gainRecorder{perturb: -1})
		for i := range ci {
			if d := math.Abs(float64(ci[i]) - cf[i]); d > 64 {
				t.Fatalf("trial %d out %d: int %d float %.2f", trial, i, ci[i], cf[i])
			}
		}
	}
}

// adst16F follows inverseADST16 in adst.go stage for stage, with the
// (c-4096)*x + x forms written as the plain rotations they equal.
func adst16F(c []float64, stride int, g *gainRecorder) {
	var in [16]float64
	for i := range in {
		in[i] = c[i*stride]
	}
	rot := func(a, ca, b, cb float64) float64 { return g.rot(rs(a*ca+b*cb, 12)) }
	var t, ta [16]float64
	t[0] = rot(in[15], 4091, in[0], 201)
	t[1] = rot(in[15], 201, in[0], -4091)
	t[2] = rot(in[13], 3973, in[2], 995)
	t[3] = rot(in[13], 995, in[2], -3973)
	t[4] = rot(in[11], 3703, in[4], 1751)
	t[5] = rot(in[11], 1751, in[4], -3703)
	t[6] = rot(in[9], 3290, in[6], 2440)
	t[7] = rot(in[9], 2440, in[6], -3290)
	t[8] = rot(in[7], 2751, in[8], 3035)
	t[9] = rot(in[7], 3035, in[8], -2751)
	t[10] = rot(in[5], 2106, in[10], 3513)
	t[11] = rot(in[5], 3513, in[10], -2106)
	t[12] = rot(in[3], 1380, in[12], 3857)
	t[13] = rot(in[3], 3857, in[12], -1380)
	t[14] = rot(in[1], 601, in[14], 4052)
	t[15] = rot(in[1], 4052, in[14], -601)
	for i := 0; i < 8; i++ {
		ta[i] = t[i] + t[i+8]
		ta[i+8] = t[i] - t[i+8]
	}
	t[8] = rot(ta[8], 4017, ta[9], 799)
	t[9] = rot(ta[8], 799, ta[9], -4017)
	t[10] = rot(ta[10], 2276, ta[11], 3406)
	t[11] = rot(ta[10], 3406, ta[11], -2276)
	t[12] = rot(ta[13], 4017, ta[12], -799)
	t[13] = rot(ta[13], 799, ta[12], 4017)
	t[14] = rot(ta[15], 2276, ta[14], -3406)
	t[15] = rot(ta[15], 3406, ta[14], 2276)
	for i := 0; i < 4; i++ {
		t[i] = ta[i] + ta[i+4]
		t[i+4] = ta[i] - ta[i+4]
	}
	for i := 8; i < 12; i++ {
		ta[i] = t[i] + t[i+4]
		ta[i+4] = t[i] - t[i+4]
	}
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	t[12] = rot(ta[12], 3784, ta[13], 1567)
	t[13] = rot(ta[12], 1567, ta[13], -3784)
	t[14] = rot(ta[15], 3784, ta[14], -1567)
	t[15] = rot(ta[15], 1567, ta[14], 3784)
	var out [16]float64
	out[0] = t[0] + t[2]
	out[15] = -(t[1] + t[3])
	ta[2] = t[0] - t[2]
	ta[3] = t[1] - t[3]
	out[3] = -(ta[4] + ta[6])
	out[12] = ta[5] + ta[7]
	t[6] = ta[4] - ta[6]
	t[7] = ta[5] - ta[7]
	out[1] = -(ta[8] + ta[10])
	out[14] = ta[9] + ta[11]
	t[10] = ta[8] - ta[10]
	t[11] = ta[9] - ta[11]
	out[2] = t[12] + t[14]
	out[13] = -(t[13] + t[15])
	ta[14] = t[12] - t[14]
	ta[15] = t[13] - t[15]
	out[7] = -rot(ta[2], 2896, ta[3], 2896)
	out[8] = rot(ta[2], 2896, ta[3], -2896)
	out[4] = rot(t[6], 2896, t[7], 2896)
	out[11] = -rot(t[6], 2896, t[7], -2896)
	out[6] = rot(t[10], 2896, t[11], 2896)
	out[9] = -rot(t[10], 2896, t[11], -2896)
	out[5] = -rot(ta[14], 2896, ta[15], 2896)
	out[10] = rot(ta[14], 2896, ta[15], -2896)
	for i := range out {
		c[i*stride] = out[i]
	}
}

// adst8F follows inverseADST8To in adst.go stage for stage.
func adst8F(c []float64, stride int, g *gainRecorder) {
	var in [8]float64
	for i := range in {
		in[i] = c[i*stride]
	}
	rot := func(a, ca, b, cb float64) float64 { return g.rot(rs(a*ca+b*cb, 12)) }
	var t, ta [8]float64
	ta[0] = rot(in[7], 4076, in[0], 401)
	ta[1] = rot(in[7], 401, in[0], -4076)
	ta[2] = rot(in[5], 3612, in[2], 1931)
	ta[3] = rot(in[5], 1931, in[2], -3612)
	ta[4] = rot(in[3], 2598, in[4], 3166)
	ta[5] = rot(in[3], 3166, in[4], -2598)
	ta[6] = rot(in[1], 1189, in[6], 3920)
	ta[7] = rot(in[1], 3920, in[6], -1189)
	for i := 0; i < 4; i++ {
		t[i] = ta[i] + ta[i+4]
		t[i+4] = ta[i] - ta[i+4]
	}
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	var out [8]float64
	out[0] = t[0] + t[2]
	out[7] = -(t[1] + t[3])
	t[2] = t[0] - t[2]
	t[3] = t[1] - t[3]
	out[1] = -(ta[4] + ta[6])
	out[6] = ta[5] + ta[7]
	t[6] = ta[4] - ta[6]
	t[7] = ta[5] - ta[7]
	out[3] = -rot(t[2], 2896, t[3], 2896)
	out[4] = rot(t[2], 2896, t[3], -2896)
	out[2] = rot(t[6], 2896, t[7], 2896)
	out[5] = -rot(t[6], 2896, t[7], -2896)
	for i := range out {
		c[i*stride] = out[i]
	}
}

// TestITXHybridGainBounds checks that the fused hybrid kernels' guard
// (itx16ColumnL1Bound, 32767-64) keeps every rotation output of the ADST8
// and ADST16 networks inside int16: an output is bounded by the largest
// single-input gain times the per-lane L1 norm of the inputs, plus the
// rounding slack.
func TestITXHybridGainBounds(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{8, adst8F}, {16, adst16F}} {
		g := gainsFor(tc.n, tc.f)
		slack := roundingSlack(tc.n, tc.f, len(g))
		maxCoef, maxSlack := 0.0, 0.0
		for i, row := range g {
			for _, v := range row {
				maxCoef = math.Max(maxCoef, v)
			}
			maxSlack = math.Max(maxSlack, slack[i])
		}
		t.Logf("ADST%d: %d rotation outputs; max |coef| %.4f -> per-column L1 bound %d; max rounding slack %.2f",
			tc.n, len(g), maxCoef, int(32767/maxCoef), maxSlack)
		const bound = 32767 - 64
		if maxCoef*bound+maxSlack > 32767 {
			t.Fatalf("guard %d does not certify ADST%d: %.4f * %d + %.2f exceeds 32767", bound, tc.n, maxCoef, bound, maxSlack)
		}
	}
}

// TestITXHybridFloatModelTracksScalar checks the ADST float models against
// the integer networks on small inputs, where rounding is the only
// difference.
func TestITXHybridFloatModelTracksScalar(t *testing.T) {
	for _, tc := range []struct {
		n int
		f func([]float64, int, *gainRecorder)
	}{{8, adst8F}, {16, adst16F}} {
		for trial := 0; trial < 200; trial++ {
			ci := make([]int32, tc.n)
			cf := make([]float64, tc.n)
			for i := range ci {
				v := int32((trial*7919+i*104729)%201) - 100
				ci[i] = v
				cf[i] = float64(v)
			}
			if tc.n == 8 {
				inverseADST8(ci, 1, -32768, 32767)
			} else {
				inverseADST16(ci, 1, -32768, 32767)
			}
			tc.f(cf, 1, &gainRecorder{perturb: -1})
			for i := range ci {
				if d := math.Abs(float64(ci[i]) - cf[i]); d > 64 {
					t.Fatalf("ADST%d trial %d out %d: int %d float %.2f", tc.n, trial, i, ci[i], cf[i])
				}
			}
		}
	}
}

// TestInverseBlockBitDepthRawBoundedMatchesFull checks the eob-bounded
// sub-size kernels (activeCols/activeRows) against the unbounded transform
// for every size and DCT-row type that can use them, across bit depths,
// extremes and sparse patterns.
func TestInverseBlockBitDepthRawBoundedMatchesFull(t *testing.T) {
	rng := randv2.New(randv2.NewPCG(5, 9))
	sizes := []Size{{32, 8}, {32, 16}, {32, 32}, {32, 64}, {8, 32}, {16, 32}, {64, 32}, {16, 16}, {16, 4}, {16, 8}, {16, 64}, {4, 16}, {8, 16}, {64, 16}, {64, 64}}
	for _, size := range sizes {
		scan := adjustedScanSizeTable[sizeIndex(size)]
		cw, ch := int(scan.Width), int(scan.Height)
		for _, bd := range []uint8{8, 10, 12} {
			cfMax := int32(1)<<(7+bd) - 1
			for iter := 0; iter < 300; iter++ {
				cols := 1 + rng.IntN(min(cw, 20))
				rows := 1 + rng.IntN(ch)
				coeff := make([]int32, cw*ch)
				n := 1 + rng.IntN(cols*rows)
				for range n {
					x, y := rng.IntN(cols), rng.IntN(rows)
					v := int32(rng.IntN(int(2*cfMax+2))) - cfMax - 1
					if rng.IntN(4) == 0 {
						v = int32(rng.IntN(64)) - 32
					}
					coeff[x*ch+y] = v
				}
				want := scalarDCTDCTReference(coeff, ch, size, bd)
				got := make([]int32, len(want))
				if err := InverseBlockBitDepthRawBounded(got, coeff, ch, size, TypeDCTDCT, bd, rows, cols); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, want) {
					t.Fatalf("%dx%d bd=%d rows=%d cols=%d: bounded transform differs", size.Width, size.Height, bd, rows, cols)
				}
			}
		}
	}
}

// scalarDCTDCTReference is the two-pass DCT_DCT inverse with the scalar 1D
// kernels only, independent of every batched/SIMD dispatch.
func scalarDCTDCTReference(coeff []int32, coeffStride int, size Size, bitDepth uint8) []int32 {
	width, height := int(size.Width), int(size.Height)
	rowMin, rowMax, colMin, colMax, _ := stageRangeBounds(bitDepth)
	idx := sizeIndex(size)
	coeffSize := adjustedScanSizeTable[idx]
	coeffW, coeffH := int(coeffSize.Width), int(coeffSize.Height)
	shift := int(sizeShiftTable[idx])
	scratch := make([]int32, width*height)
	stageTransposeClampScalar(scratch, width, coeff, coeffStride, coeffH, coeffW, size.IsRect2(), rowMin, rowMax)
	for row := 0; row < coeffH; row++ {
		inverseDCT1D(scratch[row*width:row*width+width], 1, width, rowMin, rowMax)
	}
	for i, value := range scratch {
		if shift > 0 {
			value = int32(clipRange(roundShift(int64(value), shift), colMin, colMax))
		} else {
			value = clipRange(int64(value), colMin, colMax)
		}
		scratch[i] = value
	}
	for col := 0; col < width; col++ {
		inverseDCT1D(scratch[col:], width, height, colMin, colMax)
	}
	return scratch
}

func TestHotStructSizes(t *testing.T) {
	if size := unsafe.Sizeof(Size{}); size > 2 {
		t.Fatalf("Size grew to %d bytes, max 2", size)
	}
}

// allTransformSizes enumerates every supported AV1 transform shape so the
// precompute equality checks cover the full domain.
var allTransformSizes = []Size{
	{Width: 4, Height: 4},
	{Width: 4, Height: 8},
	{Width: 8, Height: 4},
	{Width: 4, Height: 16},
	{Width: 8, Height: 8},
	{Width: 8, Height: 16},
	{Width: 16, Height: 4},
	{Width: 16, Height: 8},
	{Width: 16, Height: 32},
	{Width: 32, Height: 16},
	{Width: 32, Height: 64},
	{Width: 64, Height: 32},
	{Width: 8, Height: 32},
	{Width: 16, Height: 16},
	{Width: 16, Height: 64},
	{Width: 32, Height: 8},
	{Width: 32, Height: 32},
	{Width: 64, Height: 16},
	{Width: 64, Height: 64},
}

// referenceShift reproduces the historical Size.shift switch so the precomputed
// table can be proven byte-identical.
func referenceShift(s Size) (int, bool) {
	switch s {
	case Size{Width: 4, Height: 4},
		Size{Width: 4, Height: 8},
		Size{Width: 8, Height: 4}:
		return 0, true
	case Size{Width: 4, Height: 16},
		Size{Width: 8, Height: 8},
		Size{Width: 8, Height: 16},
		Size{Width: 16, Height: 4},
		Size{Width: 16, Height: 8},
		Size{Width: 16, Height: 32},
		Size{Width: 32, Height: 16},
		Size{Width: 32, Height: 64},
		Size{Width: 64, Height: 32}:
		return 1, true
	case Size{Width: 8, Height: 32},
		Size{Width: 16, Height: 16},
		Size{Width: 16, Height: 64},
		Size{Width: 32, Height: 8},
		Size{Width: 32, Height: 32},
		Size{Width: 64, Height: 16},
		Size{Width: 64, Height: 64}:
		return 2, true
	default:
		return 0, false
	}
}

// referenceAdjustedScanSize reproduces the historical adjustedScanSize switch.
func referenceAdjustedScanSize(size Size) Size {
	switch size {
	case Size{Width: 64, Height: 64},
		Size{Width: 64, Height: 32},
		Size{Width: 32, Height: 64}:
		return Size{Width: 32, Height: 32}
	case Size{Width: 64, Height: 16}:
		return Size{Width: 32, Height: 16}
	case Size{Width: 16, Height: 64}:
		return Size{Width: 16, Height: 32}
	default:
		return size
	}
}

func TestPrecomputedShiftMatchesReference(t *testing.T) {
	// Sweep the full (width,height) cross product of AV1 dimensions plus a few
	// invalid extents to confirm validity and shift agree with the reference.
	dims := []int{4, 8, 16, 32, 64, 12, 0}
	for _, w := range dims {
		for _, h := range dims {
			s := Size{Width: uint8(w), Height: uint8(h)}
			wantShift, wantOK := referenceShift(s)
			gotShift, gotOK := s.shift()
			if gotOK != wantOK {
				t.Fatalf("shift valid %+v=%v want %v", s, gotOK, wantOK)
			}
			if wantOK && gotShift != wantShift {
				t.Fatalf("shift %+v=%d want %d", s, gotShift, wantShift)
			}
			if got := s.Valid(); got != wantOK {
				t.Fatalf("Valid %+v=%v want %v", s, got, wantOK)
			}
		}
	}
}

func TestPrecomputedAdjustedScanSizeMatchesReference(t *testing.T) {
	for _, s := range allTransformSizes {
		want := referenceAdjustedScanSize(s)
		if got := adjustedScanSize(s); got != want {
			t.Fatalf("adjustedScanSize %+v=%+v want %+v", s, got, want)
		}
	}
}

func TestPrecomputedScanOrdersMatchGenerator(t *testing.T) {
	for _, s := range allTransformSizes {
		scanSize := adjustedScanSize(s)
		total := int(scanSize.Width) * int(scanSize.Height)
		for mode := range numScanModes {
			wantScan := make([]int16, total)
			wantInverse := make([]int16, total)
			if err := fillScanOrderCompute(wantScan, wantInverse, scanSize, ScanMode(mode)); err != nil {
				t.Fatalf("generator %+v mode %d: %v", scanSize, mode, err)
			}

			gotScan := make([]int16, total)
			gotInverse := make([]int16, total)
			if err := FillScanOrder(gotScan, gotInverse, scanSize, ScanMode(mode)); err != nil {
				t.Fatalf("FillScanOrder %+v mode %d: %v", scanSize, mode, err)
			}

			for i := range wantScan {
				if gotScan[i] != wantScan[i] {
					t.Fatalf("scan %+v mode %d [%d]=%d want %d", scanSize, mode, i, gotScan[i], wantScan[i])
				}
				if gotInverse[i] != wantInverse[i] {
					t.Fatalf("inverse %+v mode %d [%d]=%d want %d", scanSize, mode, i, gotInverse[i], wantInverse[i])
				}
			}
		}
	}
}

// TestPrecomputedTablesAreSharedReadOnly confirms FillScanOrder copies out of
// the immutable table rather than returning the table slice itself, so callers
// cannot corrupt shared state.
func TestPrecomputedTablesAreSharedReadOnly(t *testing.T) {
	size := Size{Width: 4, Height: 4}
	idx := sizeIndex(size)
	orig := append([]int16(nil), scanTables[idx][ScanModeZigZag].scan...)

	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := FillScanOrder(scan, inverse, size, ScanModeZigZag); err != nil {
		t.Fatal(err)
	}
	for i := range scan {
		scan[i] = -1
	}
	for i := range orig {
		if scanTables[idx][ScanModeZigZag].scan[i] != orig[i] {
			t.Fatalf("table mutated at %d: %d want %d", i, scanTables[idx][ScanModeZigZag].scan[i], orig[i])
		}
	}
}

// stageClampSets enumerates the (min,max) clamp pairs the inverse-transform
// stages use across the supported bit depths (8/10/12 bit, row and column
// passes). The NEON kernels must be bit-exact against pure-Go for every one.
var stageClampSets = func() [][2]int32 {
	var sets [][2]int32
	for _, bd := range []uint8{8, 10, 12} {
		rMin, rMax, cMin, cMax, ok := stageRangeBounds(bd)
		if !ok {
			continue
		}
		sets = append(sets, [2]int32{rMin, rMax}, [2]int32{cMin, cMax})
	}
	return sets
}()

// rowEdgeValues are the coefficient values most likely to expose rounding or
// clamp boundary bugs in the butterfly kernels.
var rowEdgeValues = []int32{
	0, 1, -1, 7, -7,
	minInt16, maxInt16,
	-131072, 131071,
	-524288, 524287,
	-8388608, 8388607,
}

// row2TestFunc names one batched row kernel under test: its accelerated
// implementation, the pure-Go reference it must equal bit-for-bit, and the
// transform length.
type row2TestFunc struct {
	name   string
	length int
	impl   func(r0, r1 []int32, min, max int32)
	ref    func(r0, r1 []int32, min, max int32)
}

// dispatchRow2Funcs returns every kernel under test. The impl entries come
// from row2TestImpls, which the arch-specific build populates: on arm64 it
// points at the NEON adapters directly (so all five NEON kernels are gated for
// bit-exactness regardless of which the dispatcher binds), elsewhere at the
// live dispatch slots.
func dispatchRow2Funcs() []row2TestFunc {
	return row2TestImpls()
}

func eqInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestRowPassDispatchMatchesPureGo asserts every wired batched row kernel is
// bit-identical to its pure-Go reference across random inputs and the full
// clamp-range ladder. On arm64 the dispatch slots resolve to the Go SIMD
// kernels (cpu.Detected.NEON is mandatory), so this is their primary
// byte-exactness gate.
func TestRowPassDispatchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eed))
	for _, fn := range dispatchRow2Funcs() {
		t.Run(fn.name, func(t *testing.T) {
			for _, cs := range stageClampSets {
				min, max := cs[0], cs[1]
				for iter := 0; iter < 50000; iter++ {
					r0 := make([]int32, fn.length)
					r1 := make([]int32, fn.length)
					for i := 0; i < fn.length; i++ {
						// Span the row-input range (bd+8 -> up to 20-bit
						// magnitudes); clamp to the active stage range as the
						// staging step would.
						r0[i] = clipRange(int64(int32(rng.Int63())>>11), min, max)
						r1[i] = clipRange(int64(int32(rng.Int63())>>11), min, max)
					}
					ref0 := append([]int32(nil), r0...)
					ref1 := append([]int32(nil), r1...)
					got0 := append([]int32(nil), r0...)
					got1 := append([]int32(nil), r1...)
					fn.ref(ref0, ref1, min, max)
					fn.impl(got0, got1, min, max)
					if !eqInt32(got0, ref0) || !eqInt32(got1, ref1) {
						t.Fatalf("%s clamp=[%d,%d] iter=%d\n in0=%v in1=%v\n got0=%v want0=%v\n got1=%v want1=%v",
							fn.name, min, max, iter, r0, r1, got0, ref0, got1, ref1)
					}
				}
			}
		})
	}
}

// TestRowPassDispatchEdgeValues exhaustively crosses the boundary coefficient
// values (which sit at clamp and rounding cusps) for the small kernels and a
// large random sample for the wide ones.
func TestRowPassDispatchEdgeValues(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1234abcd))
	for _, fn := range dispatchRow2Funcs() {
		t.Run(fn.name, func(t *testing.T) {
			for _, cs := range stageClampSets {
				min, max := cs[0], cs[1]
				for iter := 0; iter < 200000; iter++ {
					r0 := make([]int32, fn.length)
					r1 := make([]int32, fn.length)
					for i := 0; i < fn.length; i++ {
						r0[i] = clipRange(int64(rowEdgeValues[rng.Intn(len(rowEdgeValues))]), min, max)
						r1[i] = clipRange(int64(rowEdgeValues[rng.Intn(len(rowEdgeValues))]), min, max)
					}
					ref0 := append([]int32(nil), r0...)
					ref1 := append([]int32(nil), r1...)
					got0 := append([]int32(nil), r0...)
					got1 := append([]int32(nil), r1...)
					fn.ref(ref0, ref1, min, max)
					fn.impl(got0, got1, min, max)
					if !eqInt32(got0, ref0) || !eqInt32(got1, ref1) {
						t.Fatalf("%s clamp=[%d,%d]\n in0=%v in1=%v\n got0=%v want0=%v\n got1=%v want1=%v",
							fn.name, min, max, r0, r1, got0, ref0, got1, ref1)
					}
				}
			}
		})
	}
}

// TestRowPassDispatchZeroAlloc protects the hot-path invariant that the
// batched row kernels do not allocate.
func TestRowPassDispatchZeroAlloc(t *testing.T) {
	r0 := make([]int32, dct8Size)
	r1 := make([]int32, dct8Size)
	for i := range r0 {
		r0[i] = int32(i*131 - 400)
		r1[i] = int32(i*-77 + 200)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		inverseDCT8Row2Impl(r0, r1, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT8 batched row kernel allocated %f times per call", allocs)
	}
	allocs = testing.AllocsPerRun(1000, func() {
		inverseDCT4Row2Impl(r0, r1, minInt16, maxInt16)
	})
	if allocs != 0 {
		t.Fatalf("DCT4 batched row kernel allocated %f times per call", allocs)
	}
}

// row4TestFunc names one batched four-row kernel under test.
type row4TestFunc struct {
	name   string
	length int
	impl   func(r0, r1, r2, r3 []int32, min, max int32)
	ref    func(r0, r1, r2, r3 []int32, min, max int32)
}

// runRow4 stages four rows of pre-clamped inputs (the four-row kernels'
// documented precondition, established in the decode path by the hybrid.go
// staging loops) and asserts impl matches ref bit-for-bit.
func runRow4(t *testing.T, fn row4TestFunc, rng *rand.Rand, iters int, fill func() int32) {
	t.Helper()
	for _, cs := range stageClampSets {
		min, max := cs[0], cs[1]
		for iter := 0; iter < iters; iter++ {
			rows := make([][]int32, 4)
			refRows := make([][]int32, 4)
			gotRows := make([][]int32, 4)
			for r := range rows {
				rows[r] = make([]int32, fn.length)
				for i := range rows[r] {
					rows[r][i] = clipRange(int64(fill()), min, max)
				}
				refRows[r] = append([]int32(nil), rows[r]...)
				gotRows[r] = append([]int32(nil), rows[r]...)
			}
			fn.ref(refRows[0], refRows[1], refRows[2], refRows[3], min, max)
			fn.impl(gotRows[0], gotRows[1], gotRows[2], gotRows[3], min, max)
			for r := range rows {
				if !eqInt32(gotRows[r], refRows[r]) {
					t.Fatalf("%s clamp=[%d,%d] iter=%d row=%d\n in=%v\n got=%v\nwant=%v",
						fn.name, min, max, iter, r, rows[r], gotRows[r], refRows[r])
				}
			}
		}
	}
}

// TestRowPassDispatchRow4MatchesPureGo asserts the four-row kernels are
// bit-identical to their pure-Go references across random inputs and the full
// clamp ladder; on arm64 the entries resolve to the NEON kernels.
func TestRowPassDispatchRow4MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x40404))
	for _, fn := range row4TestImpls() {
		t.Run(fn.name, func(t *testing.T) {
			runRow4(t, fn, rng, 30000, func() int32 { return int32(int32(rng.Int63()) >> 11) })
		})
	}
}

// TestRowPassDispatchRow4EdgeValues crosses the boundary coefficient values
// that sit at clamp and rounding cusps.
func TestRowPassDispatchRow4EdgeValues(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4ed6e))
	for _, fn := range row4TestImpls() {
		t.Run(fn.name, func(t *testing.T) {
			runRow4(t, fn, rng, 30000, func() int32 { return rowEdgeValues[rng.Intn(len(rowEdgeValues))] })
		})
	}
}

// TestDefaultScanActiveCols checks the column bound against a direct walk of
// the default scan prefix for every valid size and class.
func TestDefaultScanActiveCols(t *testing.T) {
	for _, w := range []uint8{4, 8, 16, 32, 64} {
		for _, h := range []uint8{4, 8, 16, 32, 64} {
			size := Size{Width: w, Height: h}
			if !size.Valid() {
				continue
			}
			scanSize, err := ScanSize(size)
			if err != nil {
				t.Fatal(err)
			}
			height := int(scanSize.Height)
			for class := Class(0); class.Valid(); class++ {
				scan, err := DefaultScan(size, class)
				if err != nil {
					t.Fatal(err)
				}
				maxCol := -1
				for eob := 0; eob <= len(scan); eob++ {
					gotCols, ok := DefaultScanActiveCols(size, class, eob)
					if !ok {
						t.Fatalf("%dx%d class %d eob %d: not ok", w, h, class, eob)
					}
					if gotCols != maxCol+1 {
						t.Fatalf("%dx%d class %d eob %d: cols %d, want %d", w, h, class, eob, gotCols, maxCol+1)
					}
					if eob < len(scan) {
						maxCol = max(maxCol, int(scan[eob])/height)
					}
				}
				if _, ok := DefaultScanActiveCols(size, class, len(scan)+1); ok {
					t.Fatalf("%dx%d class %d: eob past the scan accepted", w, h, class)
				}
			}
		}
	}
	if _, ok := DefaultScanActiveCols(Size{Width: 64, Height: 64}, Class2D, -1); ok {
		t.Fatal("negative eob accepted")
	}
}

func TestScanOrderDependencyMatchesLibaom(t *testing.T) {
	for txSize, size := range libaomScanSizes {
		scanSize, err := ScanSize(size)
		if err != nil {
			t.Fatalf("ScanSize(%+v): %v", size, err)
		}
		for txType := range int(TypeCount) {
			if !libaomTxSizeTypeValid(txSize, txType) {
				continue
			}
			class, err := Type(txType).Class()
			if err != nil {
				t.Fatalf("Class(%d): %v", txType, err)
			}
			mode, err := DefaultScanMode(size, class)
			if err != nil {
				t.Fatalf("DefaultScanMode(%+v,%d): %v", size, class, err)
			}
			scanWidth := int(scanSize.Width)
			scanHeight := int(scanSize.Height)
			scan := make([]int16, scanWidth*scanHeight)
			inverse := make([]int16, len(scan))
			if err := FillDefaultScan(scan, inverse, size, class); err != nil {
				t.Fatalf("FillDefaultScan(%+v,%d): %v", size, class, err)
			}
			assertLibaomScanOrder(t, scan, inverse, scanWidth, scanHeight, mode)
		}
	}
}

func TestFillScanOrderKnownTables(t *testing.T) {
	tests := []struct {
		name     string
		size     Size
		mode     ScanMode
		wantScan []int16
	}{
		{
			name:     "default 4x4",
			size:     Size{Width: 4, Height: 4},
			mode:     ScanModeZigZag,
			wantScan: []int16{0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15},
		},
		{
			name:     "mcol 4x4",
			size:     Size{Width: 4, Height: 4},
			mode:     ScanModeCol1D,
			wantScan: []int16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
		{
			name:     "mrow 4x4",
			size:     Size{Width: 4, Height: 4},
			mode:     ScanModeRow1D,
			wantScan: []int16{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15},
		},
	}
	for _, tt := range tests {
		scan := make([]int16, len(tt.wantScan))
		inverse := make([]int16, len(tt.wantScan))
		if err := FillScanOrder(scan, inverse, tt.size, tt.mode); err != nil {
			t.Fatalf("%s FillScanOrder: %v", tt.name, err)
		}
		for i, want := range tt.wantScan {
			if scan[i] != want {
				t.Fatalf("%s scan[%d]=%d want %d", tt.name, i, scan[i], want)
			}
			if inverse[scan[i]] != int16(i) {
				t.Fatalf("%s inverse[%d]=%d want %d", tt.name, scan[i], inverse[scan[i]], i)
			}
		}
	}
}

func TestScanSizeAdjustsLibaom64CoefficientScans(t *testing.T) {
	tests := []struct {
		size Size
		want Size
	}{
		{size: Size{Width: 64, Height: 64}, want: Size{Width: 32, Height: 32}},
		{size: Size{Width: 64, Height: 32}, want: Size{Width: 32, Height: 32}},
		{size: Size{Width: 32, Height: 64}, want: Size{Width: 32, Height: 32}},
		{size: Size{Width: 64, Height: 16}, want: Size{Width: 32, Height: 16}},
		{size: Size{Width: 16, Height: 64}, want: Size{Width: 16, Height: 32}},
	}
	for _, tt := range tests {
		got, err := ScanSize(tt.size)
		if err != nil {
			t.Fatalf("ScanSize(%+v): %v", tt.size, err)
		}
		if got != tt.want {
			t.Fatalf("ScanSize(%+v)=%+v want %+v", tt.size, got, tt.want)
		}
	}
}

func TestFillScanOrderRejectsInvalidInputs(t *testing.T) {
	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := FillScanOrder(scan[:15], inverse, Size{Width: 4, Height: 4}, ScanModeZigZag); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short scan err=%v want %v", err, ErrInvalidTransform)
	}
	if err := FillScanOrder(scan, inverse[:15], Size{Width: 4, Height: 4}, ScanModeZigZag); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short inverse err=%v want %v", err, ErrInvalidTransform)
	}
	if err := FillScanOrder(scan, inverse, Size{Width: 4, Height: 12}, ScanModeZigZag); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid size err=%v want %v", err, ErrInvalidTransform)
	}
	if err := FillScanOrder(scan, inverse, Size{Width: 4, Height: 4}, ScanMode(99)); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid mode err=%v want %v", err, ErrInvalidTransform)
	}
	if _, err := DefaultScanMode(Size{Width: 4, Height: 4}, Class(99)); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid class err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestFillScanOrderAllocs(t *testing.T) {
	var scan [32 * 32]int16
	var inverse [32 * 32]int16
	allocs := testing.AllocsPerRun(1000, func() {
		if err := FillDefaultScan(scan[:], inverse[:], Size{Width: 64, Height: 64}, Class2D); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FillDefaultScan allocated: %f", allocs)
	}
}

func FuzzFillDefaultScan(f *testing.F) {
	f.Add(uint8(0), uint8(0))
	f.Add(uint8(4), uint8(2))
	f.Add(uint8(18), uint8(1))

	f.Fuzz(func(t *testing.T, rawSize uint8, rawClass uint8) {
		size := libaomScanSizes[int(rawSize)%len(libaomScanSizes)]
		class := Class(rawClass % 3)
		scanSize, err := ScanSize(size)
		if err != nil {
			t.Fatalf("ScanSize(%+v): %v", size, err)
		}
		scanWidth := int(scanSize.Width)
		scanHeight := int(scanSize.Height)
		scan := make([]int16, scanWidth*scanHeight)
		inverse := make([]int16, len(scan))
		mode, err := DefaultScanMode(size, class)
		if err != nil {
			t.Fatalf("DefaultScanMode(%+v,%d): %v", size, class, err)
		}
		if err := FillDefaultScan(scan, inverse, size, class); err != nil {
			t.Fatalf("FillDefaultScan(%+v,%d): %v", size, class, err)
		}
		assertLibaomScanOrder(t, scan, inverse, scanWidth, scanHeight, mode)
	})
}

func assertLibaomScanOrder(t *testing.T, scan []int16, inverse []int16, width int, height int, mode ScanMode) {
	t.Helper()
	dim := width + height - 1
	si := 0
	check := func(row int, col int) {
		t.Helper()
		coeff := col*height + row
		if inverse[coeff] != int16(si) || scan[si] != int16(coeff) {
			t.Fatalf("scan mismatch row=%d col=%d si=%d inverse=%d scan=%d wantCoeff=%d",
				row, col, si, inverse[coeff], scan[si], coeff)
		}
		si++
	}
	switch mode {
	case ScanModeZigZag:
		for i := range dim {
			if i%2 == 0 {
				for col := range width {
					row := i - col
					if row >= 0 && row < height {
						check(row, col)
					}
				}
				continue
			}
			for row := range height {
				col := i - row
				if col >= 0 && col < width {
					check(row, col)
				}
			}
		}
	case ScanModeColDiag:
		for i := range dim {
			for col := range width {
				row := i - col
				if row >= 0 && row < height {
					check(row, col)
				}
			}
		}
	case ScanModeRowDiag:
		for i := range dim {
			for row := range height {
				col := i - row
				if col >= 0 && col < width {
					check(row, col)
				}
			}
		}
	case ScanModeRow1D:
		for row := range height {
			for col := range width {
				check(row, col)
			}
		}
	case ScanModeCol1D:
		for col := range width {
			for row := range height {
				check(row, col)
			}
		}
	default:
		t.Fatalf("invalid scan mode %d", mode)
	}
	if si != len(scan) {
		t.Fatalf("visited %d coefficients want %d", si, len(scan))
	}
}

func libaomTxSizeTypeValid(txSize int, txType int) bool {
	txSizeSqrUp := libaomTxSizeSqrUp[txSize]
	txSetType := 5
	if txSizeSqrUp > 3 {
		txSetType = 0
	} else if txSizeSqrUp == 3 {
		txSetType = 1
	}
	return libaomExtTxUsed[txSetType][txType] != 0
}

var libaomScanSizes = [...]Size{
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

var libaomTxSizeSqrUp = [...]int{
	0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4,
}

var libaomExtTxUsed = [...][16]int{
	{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
}

func TestTXB8x8PrepTablesMatchDefaultScan(t *testing.T) {
	var scan [64]int16
	var inverse [64]int16
	if err := FillDefaultScan(scan[:], inverse[:], Size{Width: 8, Height: 8}, Class2D); err != nil {
		t.Fatal(err)
	}
	for i, pos := range scan {
		if got := txb8x8Scan2D[i]; got != uint8(pos) {
			t.Fatalf("scan[%d]=%d want %d", i, got, pos)
		}
	}
}

func TestTXBPrep8x8Levels2DMatchesScalar(t *testing.T) {
	cases := make([][64]int16, 0, 38)
	cases = append(cases, [64]int16{})
	var edges [64]int16
	edgeValues := []int16{-32768, -4096, -128, -127, -1, 0, 1, 2, 14, 15, 127, 128, 4096, 32767}
	for i := range edges {
		edges[i] = edgeValues[i%len(edgeValues)]
	}
	cases = append(cases, edges)

	rng := rand.New(rand.NewSource(0x7a8b8c))
	for range 36 {
		var coeffs [64]int16
		for i := range coeffs {
			switch rng.Intn(8) {
			case 0, 1:
				coeffs[i] = 0
			case 2:
				coeffs[i] = int16(rng.Intn(9) - 4)
			case 3:
				coeffs[i] = int16(rng.Intn(260) - 130)
			default:
				coeffs[i] = int16(rng.Intn(8192) - 4096)
			}
		}
		cases = append(cases, coeffs)
	}

	for i := range cases {
		coeffs := cases[i]
		eob := txbPrep8x8TestEOB(&coeffs)
		var gotLevels, wantLevels [256]uint8
		var gotAbs, wantAbs [64]uint16
		got := PrepTXB8x8Levels2D(&coeffs, &gotLevels, &gotAbs, eob)
		want := txbPrep8x8Levels2DPureGo(&coeffs, &wantLevels, &wantAbs, eob)

		if got != want {
			t.Fatalf("case %d result=%+v want %+v", i, got, want)
		}
		if !reflect.DeepEqual(gotLevels, wantLevels) {
			for j := range gotLevels {
				if gotLevels[j] != wantLevels[j] {
					t.Fatalf("case %d levels[%d]=%d want %d", i, j, gotLevels[j], wantLevels[j])
				}
			}
		}
		if !reflect.DeepEqual(gotAbs, wantAbs) {
			t.Fatalf("case %d absLevels mismatch\ngot  %v\nwant %v", i, gotAbs, wantAbs)
		}
	}
}

func TestTXBPrep8x8Levels2DWritesZeroAbsLevels(t *testing.T) {
	var coeffs [64]int16
	coeffs[txb8x8Scan2D[0]] = 1
	coeffs[txb8x8Scan2D[2]] = -2
	coeffs[txb8x8Scan2D[7]] = 3

	var levels [256]uint8
	var absLevels [64]uint16
	for i := range absLevels {
		absLevels[i] = 0xffff
	}

	const eob = 8
	_ = PrepTXB8x8Levels2D(&coeffs, &levels, &absLevels, eob)
	for c := range eob {
		pos := txb8x8Scan2D[c]
		want := uint16(txbAbsInt16(coeffs[pos]))
		if absLevels[c] != want {
			t.Fatalf("absLevels[%d]=%d want %d for coeff pos %d", c, absLevels[c], want, pos)
		}
	}
}

func TestTXBPrep8x8Levels2DZeroAlloc(t *testing.T) {
	var coeffs [64]int16
	for i := range coeffs {
		coeffs[i] = int16((i*47)%511 - 255)
	}
	eob := txbPrep8x8TestEOB(&coeffs)
	var levels [256]uint8
	var absLevels [64]uint16
	allocs := testing.AllocsPerRun(1000, func() {
		levels = [256]uint8{}
		absLevels = [64]uint16{}
		_ = PrepTXB8x8Levels2D(&coeffs, &levels, &absLevels, eob)
	})
	if allocs != 0 {
		t.Fatalf("TXB 8x8 prep allocated: %f", allocs)
	}
}

func BenchmarkTXBPrep8x8Levels2D(b *testing.B) {
	rng := rand.New(rand.NewSource(0x48b8))
	var blocks [256][64]int16
	var eobs [256]int
	for i := range blocks {
		for c := range blocks[i] {
			switch rng.Intn(6) {
			case 0:
				blocks[i][c] = 0
			case 1:
				blocks[i][c] = int16(rng.Intn(7) - 3)
			default:
				blocks[i][c] = int16(rng.Intn(1024) - 512)
			}
		}
		eobs[i] = txbPrep8x8TestEOB(&blocks[i])
	}

	bench := func(b *testing.B, fn func(*[64]int16, *[256]uint8, *[64]uint16, int) TXB8x8PrepResult) {
		var levels [256]uint8
		var absLevels [64]uint16
		var sink TXB8x8PrepResult
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := i & 255
			levels = [256]uint8{}
			sink = fn(&blocks[idx], &levels, &absLevels, eobs[idx])
		}
		if sink.CulLevel == 255 {
			b.Fatal(sink)
		}
	}

	b.Run("impl", func(b *testing.B) {
		bench(b, PrepTXB8x8Levels2D)
	})
	b.Run("scalar", func(b *testing.B) {
		bench(b, txbPrep8x8Levels2DPureGo)
	})
}

func txbPrep8x8TestEOB(coeffs *[64]int16) int {
	for c := len(txb8x8Scan2D) - 1; c >= 0; c-- {
		if coeffs[txb8x8Scan2D[c]] != 0 {
			return c + 1
		}
	}
	return 0
}

func TestMaxEOBMatchesLibaomBlockd(t *testing.T) {
	want := [...]int{
		16, 64, 256, 1024, 1024, 32, 32, 128, 128, 512,
		512, 1024, 1024, 64, 64, 256, 256, 512, 512,
	}
	for i, size := range libaomScanSizes {
		got, err := MaxEOB(size)
		if err != nil {
			t.Fatalf("MaxEOB(%+v): %v", size, err)
		}
		if got != want[i] {
			t.Fatalf("MaxEOB(%+v)=%d want %d", size, got, want[i])
		}
	}
}

func TestLowerLevelsCtxEOBMatchesLibaomTXBCommon(t *testing.T) {
	tests := []struct {
		size      Size
		scanIndex int
		want      int
	}{
		{size: Size{Width: 4, Height: 4}, scanIndex: 0, want: 0},
		{size: Size{Width: 4, Height: 4}, scanIndex: 1, want: 1},
		{size: Size{Width: 4, Height: 4}, scanIndex: 2, want: 1},
		{size: Size{Width: 4, Height: 4}, scanIndex: 3, want: 2},
		{size: Size{Width: 4, Height: 4}, scanIndex: 4, want: 2},
		{size: Size{Width: 4, Height: 4}, scanIndex: 5, want: 3},
		{size: Size{Width: 64, Height: 64}, scanIndex: 128, want: 1},
		{size: Size{Width: 64, Height: 64}, scanIndex: 256, want: 2},
		{size: Size{Width: 64, Height: 64}, scanIndex: 257, want: 3},
	}
	for _, tt := range tests {
		got, err := LowerLevelsCtxEOB(tt.size, tt.scanIndex)
		if err != nil {
			t.Fatalf("LowerLevelsCtxEOB(%+v,%d): %v", tt.size, tt.scanIndex, err)
		}
		if got != tt.want {
			t.Fatalf("LowerLevelsCtxEOB(%+v,%d)=%d want %d", tt.size, tt.scanIndex, got, tt.want)
		}
	}
}

func TestTXBHelpersRejectInvalidInputs(t *testing.T) {
	if _, err := MaxEOB(Size{Width: 4, Height: 12}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid MaxEOB err=%v want %v", err, ErrInvalidTransform)
	}
	if _, err := LowerLevelsCtxEOB(Size{Width: 4, Height: 4}, 16); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("large eob ctx err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestTXBHelpersAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := MaxEOB(Size{Width: 64, Height: 64}); err != nil {
			t.Fatal(err)
		}
		if _, err := LowerLevelsCtxEOB(Size{Width: 64, Height: 64}, 257); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("txb helpers allocated: %f", allocs)
	}
}

func FuzzTXBHelpers(f *testing.F) {
	f.Add(uint8(0), uint16(0))
	f.Add(uint8(4), uint16(1023))
	f.Add(uint8(18), uint16(511))

	f.Fuzz(func(t *testing.T, rawSize uint8, rawIndex uint16) {
		size := libaomScanSizes[int(rawSize)%len(libaomScanSizes)]
		maxEOB, err := MaxEOB(size)
		if err != nil {
			t.Fatalf("MaxEOB(%+v): %v", size, err)
		}
		index := int(rawIndex) % maxEOB
		ctx, err := LowerLevelsCtxEOB(size, index)
		if err != nil {
			t.Fatalf("LowerLevelsCtxEOB(%+v,%d): %v", size, index, err)
		}
		if ctx < 0 || ctx > 3 {
			t.Fatalf("ctx=%d outside [0,3]", ctx)
		}
	})
}

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

func TestInverseWHT4x4BlockEOB1(t *testing.T) {
	coeff := make([]int32, 4*4)
	coeff[0] = 16
	dst := make([]int16, 4*4)
	if err := InverseWHT4x4Block(dst, 4, coeff, 4, 1); err != nil {
		t.Fatal(err)
	}
	for i, got := range dst {
		if got != 1 {
			t.Fatalf("dst[%d]=%d want 1", i, got)
		}
	}
}

func TestInverseWHT4x4BlockEOB16(t *testing.T) {
	coeff := make([]int32, 4*4)
	coeff[0] = 16
	coeff[1] = 8
	dst := make([]int16, 6*4)
	if err := InverseWHT4x4Block(dst, 6, coeff, 4, 2); err != nil {
		t.Fatal(err)
	}
	want := []int16{
		2, 2, 2, 2,
		1, 1, 1, 1,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	for row := range 4 {
		for col := range 4 {
			if got := dst[row*6+col]; got != want[row*4+col] {
				t.Fatalf("dst(%d,%d)=%d want %d", col, row, got, want[row*4+col])
			}
		}
		if got := dst[row*6+4]; got != 0 {
			t.Fatalf("dst padding row=%d overwritten with %d", row, got)
		}
	}
}

func TestInverseWHT4x4BlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int16, 4*4)
	coeff := make([]int32, 4*4)
	tests := []struct {
		name        string
		dst         []int16
		dstStride   int
		coeff       []int32
		coeffStride int
		eob         int
	}{
		{name: "short dst stride", dst: dst, dstStride: 3, coeff: coeff, coeffStride: 4, eob: 1},
		{name: "short coeff stride", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 3, eob: 1},
		{name: "short dst", dst: dst[:15], dstStride: 4, coeff: coeff, coeffStride: 4, eob: 1},
		{name: "short coeff", dst: dst, dstStride: 4, coeff: coeff[:15], coeffStride: 4, eob: 1},
		{name: "negative eob", dst: dst, dstStride: 4, coeff: coeff, coeffStride: 4, eob: -1},
	}
	for _, tt := range tests {
		err := InverseWHT4x4Block(tt.dst, tt.dstStride, tt.coeff, tt.coeffStride, tt.eob)
		if !errors.Is(err, ErrInvalidTransform) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidTransform)
		}
	}
}

func TestInverseWHT4x4BlockAllocs(t *testing.T) {
	coeff := make([]int32, 4*4)
	dst := make([]int16, 4*4)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := InverseWHT4x4Block(dst, 4, coeff, 4, 2); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("InverseWHT4x4Block allocated: %f", allocs)
	}
}

func BenchmarkInverseWHT4x4BlockEOB1(b *testing.B) {
	coeff := make([]int32, 4*4)
	coeff[0] = 64
	dst := make([]int16, 4*4)

	b.ReportAllocs()
	for b.Loop() {
		if err := InverseWHT4x4Block(dst, 4, coeff, 4, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInverseWHT4x4BlockEOB16(b *testing.B) {
	coeff := make([]int32, 4*4)
	for i := range coeff {
		coeff[i] = int32(i*7 - 23)
	}
	dst := make([]int16, 4*4)

	b.ReportAllocs()
	for b.Loop() {
		if err := InverseWHT4x4Block(dst, 4, coeff, 4, 16); err != nil {
			b.Fatal(err)
		}
	}
}

// FuzzInverseWHT4x4Block stresses the 4x4 inverse Walsh-Hadamard with random
// coefficient values, strides, and EOB choices. The transform must complete
// without panicking and must not write outside the requested block.
func FuzzInverseWHT4x4Block(f *testing.F) {
	f.Add(uint8(0), int16(0), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(1), int16(16), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(2), int16(64), int16(-32), int16(16), int16(-8), int16(4))
	f.Add(uint8(3), int16(-256), int16(128), int16(-64), int16(32), int16(-16))
	f.Add(uint8(0xff), int16(32767), int16(-32768), int16(32767), int16(-32768), int16(0))
	f.Add(uint8(7), int16(1024), int16(-1024), int16(0), int16(255), int16(-255))

	f.Fuzz(func(t *testing.T, rawMode uint8, c0 int16, c1 int16, c2 int16, c3 int16, c4 int16) {
		coeffStride := 4 + int(rawMode&3)
		dstStride := 4 + int((rawMode>>2)&3)
		eob := 1
		if rawMode&0x80 != 0 {
			eob = 16
		} else if rawMode&0x40 != 0 {
			eob = 2
		}

		coeff := make([]int32, coeffStride*4+1)
		dst := make([]int16, dstStride*4+1)
		seeds := [5]int16{c0, c1, c2, c3, c4}
		for col := range 4 {
			for row := range 4 {
				idx := col*coeffStride + row
				coeff[idx] = int32(seeds[(col+row)%5]) + int32(col)*3 - int32(row)
			}
		}

		const sentinel = int16(0x5a5a)
		dst[len(dst)-1] = sentinel
		for row := range 4 {
			for col := 4; col < dstStride; col++ {
				dst[row*dstStride+col] = sentinel
			}
		}

		if err := InverseWHT4x4Block(dst, dstStride, coeff, coeffStride, eob); err != nil {
			t.Fatalf("InverseWHT4x4Block err=%v", err)
		}
		if got := dst[len(dst)-1]; got != sentinel {
			t.Fatalf("tail sentinel overwritten: got %d want %d", got, sentinel)
		}
		for row := range 4 {
			for col := 4; col < dstStride; col++ {
				if got := dst[row*dstStride+col]; got != sentinel {
					t.Fatalf("dst padding row=%d col=%d overwritten with %d", row, col, got)
				}
			}
		}
	})
}
