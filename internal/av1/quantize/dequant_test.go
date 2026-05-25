package quantize

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestQuantLookupMatchesAV1Tables(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		qIndex   uint8
		wantDC   int32
		wantAC   int32
	}{
		{name: "8 bit lossless", bitDepth: 8, qIndex: 0, wantDC: 4, wantAC: 4},
		{name: "8 bit q2", bitDepth: 8, qIndex: 2, wantDC: 8, wantAC: 9},
		{name: "8 bit max", bitDepth: 8, qIndex: 255, wantDC: 1336, wantAC: 1828},
		{name: "10 bit max", bitDepth: 10, qIndex: 255, wantDC: 5347, wantAC: 7312},
		{name: "12 bit max", bitDepth: 12, qIndex: 255, wantDC: 21387, wantAC: 29247},
	}
	for _, tt := range tests {
		dc, err := DCQuant(tt.qIndex, 0, tt.bitDepth)
		if err != nil {
			t.Fatalf("%s DCQuant err=%v", tt.name, err)
		}
		ac, err := ACQuant(tt.qIndex, 0, tt.bitDepth)
		if err != nil {
			t.Fatalf("%s ACQuant err=%v", tt.name, err)
		}
		if dc != tt.wantDC || ac != tt.wantAC {
			t.Fatalf("%s quant dc/ac=%d/%d want %d/%d", tt.name, dc, ac, tt.wantDC, tt.wantAC)
		}
	}
}

func TestQuantLookupClampsDelta(t *testing.T) {
	if got := ClampQIndex(10, -100); got != 0 {
		t.Fatalf("ClampQIndex low=%d want 0", got)
	}
	if got := ClampQIndex(250, 100); got != 255 {
		t.Fatalf("ClampQIndex high=%d want 255", got)
	}
	dc, err := DCQuant(10, -100, 8)
	if err != nil {
		t.Fatal(err)
	}
	ac, err := ACQuant(250, 100, 8)
	if err != nil {
		t.Fatal(err)
	}
	if dc != 4 || ac != 1828 {
		t.Fatalf("clamped quant dc/ac=%d/%d want 4/1828", dc, ac)
	}
}

func TestPlaneQuantizer(t *testing.T) {
	params := parser.QuantizationParams{
		BaseQIdx: 96,
		YDCDelta: -2,
		UDCDelta: 3,
		UACDelta: -1,
		VDCDelta: 4,
		VACDelta: 5,
	}
	y, err := PlaneQuantizer(params, params.BaseQIdx, 8, PlaneY)
	if err != nil {
		t.Fatal(err)
	}
	u, err := PlaneQuantizer(params, params.BaseQIdx, 8, PlaneU)
	if err != nil {
		t.Fatal(err)
	}
	v, err := PlaneQuantizer(params, params.BaseQIdx, 8, PlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if y != (Quantizer{DC: 85, AC: 104}) {
		t.Fatalf("y=%+v want dc=85 ac=104", y)
	}
	if u != (Quantizer{DC: 92, AC: 102}) {
		t.Fatalf("u=%+v want dc=92 ac=102", u)
	}
	if v != (Quantizer{DC: 93, AC: 114}) {
		t.Fatalf("v=%+v want dc=93 ac=114", v)
	}
}

func TestDequantizeBlock(t *testing.T) {
	coeff := []int16{
		2, -5, 99,
		-3, 6, 99,
		4, -7, 99,
	}
	dst := []int32{
		99, 99, 99,
		99, 99, 99,
		99, 99, 99,
	}
	q := Quantizer{DC: 4, AC: 8}
	if err := DequantizeBlock(dst, 3, coeff, 3, 3, 2, q); err != nil {
		t.Fatal(err)
	}
	want := []int32{
		8, -40, 99,
		-24, 48, 99,
		32, -56, 99,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestTransformScaleMatchesLibaom(t *testing.T) {
	tests := []struct {
		name   string
		width  int
		height int
		want   uint8
	}{
		{name: "4x4", width: 4, height: 4, want: 0},
		{name: "8x32", width: 8, height: 32, want: 0},
		{name: "16x32", width: 16, height: 32, want: 1},
		{name: "32x32", width: 32, height: 32, want: 1},
		{name: "16x64", width: 16, height: 64, want: 1},
		{name: "32x64", width: 32, height: 64, want: 1},
		{name: "64x64", width: 64, height: 64, want: 1},
	}
	for _, tt := range tests {
		got, err := TransformScale(tt.width, tt.height)
		if err != nil {
			t.Fatalf("%s TransformScale err=%v", tt.name, err)
		}
		if got != tt.want {
			t.Fatalf("%s scale=%d want %d", tt.name, got, tt.want)
		}
	}
	if _, err := TransformScale(0, 4); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("zero width err=%v want %v", err, ErrInvalidQuantizer)
	}
}

func TestDequantizeBlockScaledMatchesLibaomLargeTX(t *testing.T) {
	coeff := []int16{
		7, -5, 99,
		6, -9, 99,
	}
	dst := []int32{
		99, 99, 99,
		99, 99, 99,
	}
	q := Quantizer{DC: 20, AC: 13}
	if err := DequantizeBlockScaled(dst, 3, coeff, 3, 2, 2, q, 1); err != nil {
		t.Fatal(err)
	}
	want := []int32{
		70, -32, 99,
		39, -58, 99,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestDequantizeBlockScaledQMatrixMatchesLibaomFormula(t *testing.T) {
	coeff := []int16{
		2, -1,
		3, -4,
	}
	dst := make([]int32, 4)
	q := Quantizer{DC: 64, AC: 96}
	iqMatrix := []uint16{
		32, 73,
		43, 97,
	}
	if err := DequantizeBlockScaledQMatrix(dst, 2, coeff, 2, 2, 2, q, 0, iqMatrix); err != nil {
		t.Fatal(err)
	}
	want := []int32{
		128, -129,
		657, -1164,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}

	if err := DequantizeBlockScaledQMatrix(dst, 2, coeff, 2, 2, 2, q, 0, iqMatrix[:3]); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short qmatrix err=%v want %v", err, ErrInvalidQuantizer)
	}
}

func TestDequantizeBlockScaledQMatrixUsesRowMajorMatrixPosition(t *testing.T) {
	coeff := []int16{
		1, 2,
		3, 4,
		5, 6,
	}
	dst := make([]int32, 6)
	q := Quantizer{DC: 128, AC: 64}
	iqMatrix := []uint16{
		32, 40, 48,
		56, 64, 72,
	}
	if err := DequantizeBlockScaledQMatrix(dst, 2, coeff, 2, 3, 2, q, 0, iqMatrix); err != nil {
		t.Fatal(err)
	}
	want := []int32{
		128, 224,
		240, 512,
		480, 864,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestDequantizeBlockRejectsInvalidInputs(t *testing.T) {
	dst := make([]int32, 4)
	coeff := make([]int16, 4)
	q := Quantizer{DC: 4, AC: 8}
	if _, err := DCQuant(0, 0, 9); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid bitdepth err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := PlaneQuantizer(parser.QuantizationParams{}, 0, 8, Plane(99)); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid plane err=%v want %v", err, ErrInvalidQuantizer)
	}
	if err := DequantizeBlock(dst, 2, coeff, 2, 0, 2, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("zero width err=%v want %v", err, ErrInvalidQuantizer)
	}
	if err := DequantizeBlock(dst, 1, coeff, 2, 2, 2, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short dst stride err=%v want %v", err, ErrInvalidQuantizer)
	}
	if err := DequantizeBlock(dst[:3], 2, coeff, 2, 2, 2, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short dst err=%v want %v", err, ErrInvalidQuantizer)
	}
	if err := DequantizeBlock(dst, 2, coeff, 1, 2, 2, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short coeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if err := DequantizeBlock(dst, 2, coeff, 2, 2, 2, Quantizer{AC: 8}); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid quantizer err=%v want %v", err, ErrInvalidQuantizer)
	}
	if err := DequantizeBlockScaled(dst, 2, coeff, 2, 2, 2, q, 2); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid transform scale err=%v want %v", err, ErrInvalidQuantizer)
	}
}

func TestDequantizeBlockAllocs(t *testing.T) {
	coeff := make([]int16, 16*16)
	dst := make([]int32, 16*16)
	q := Quantizer{DC: 80, AC: 97}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := DequantizeBlock(dst, 16, coeff, 16, 16, 16, q); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("DequantizeBlock allocated: %f", allocs)
	}
}

func TestQuantizeFPNoQMatrixMatchesLibaomVector(t *testing.T) {
	q := FPQuantizer{
		Dequant: [2]int16{78, 93},
		Round:   [2]int16{39, 46},
		Quant:   [2]int16{840, 704},
	}
	coeff := []int32{-449, 624, -14, 24}
	qcoeff := make([]int32, 4)
	dqcoeff := make([]int32, 4)
	scan := []int16{0, 1, 2, 3}
	eob, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q)
	if err != nil {
		t.Fatal(err)
	}
	if eob != 2 {
		t.Fatalf("eob=%d want 2", eob)
	}
	wantQCoeff := []int32{-6, 7, 0, 0}
	wantDQCoeff := []int32{-468, 651, 0, 0}
	for i := range wantQCoeff {
		if qcoeff[i] != wantQCoeff[i] {
			t.Fatalf("qcoeff[%d]=%d want %d", i, qcoeff[i], wantQCoeff[i])
		}
		if dqcoeff[i] != wantDQCoeff[i] {
			t.Fatalf("dqcoeff[%d]=%d want %d", i, dqcoeff[i], wantDQCoeff[i])
		}
	}
}

func TestQuantizeFPNoQMatrixLogScaleAndEOB(t *testing.T) {
	q := FPQuantizer{
		Dequant:  [2]int16{64, 128},
		Round:    [2]int16{32, 64},
		Quant:    [2]int16{1024, 512},
		LogScale: 1,
	}
	coeff := []int32{4, -512, 0, 1024}
	qcoeff := []int32{99, 99, 99, 99}
	dqcoeff := []int32{99, 99, 99, 99}
	scan := []int16{0, 1, 2, 3}
	eob, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q)
	if err != nil {
		t.Fatal(err)
	}
	if eob != 4 {
		t.Fatalf("eob=%d want 4", eob)
	}
	wantQCoeff := []int32{0, -8, 0, 16}
	wantDQCoeff := []int32{0, -512, 0, 1024}
	for i := range wantQCoeff {
		if qcoeff[i] != wantQCoeff[i] || dqcoeff[i] != wantDQCoeff[i] {
			t.Fatalf("coeff[%d] q/dq=%d/%d want %d/%d", i, qcoeff[i], dqcoeff[i], wantQCoeff[i], wantDQCoeff[i])
		}
	}
}

func TestQuantizeFPNoQMatrixRejectsInvalidInputs(t *testing.T) {
	q := FPQuantizer{
		Dequant: [2]int16{78, 93},
		Round:   [2]int16{39, 46},
		Quant:   [2]int16{840, 704},
	}
	coeff := make([]int32, 4)
	qcoeff := make([]int32, 4)
	dqcoeff := make([]int32, 4)
	scan := []int16{0, 1, 2, 3}
	if _, err := QuantizeFPNoQMatrix(qcoeff[:3], dqcoeff, coeff, scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short qcoeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff[:3], coeff, scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short dqcoeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff[:3], scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short coeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, []int16{0, 4}, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid scan err=%v want %v", err, ErrInvalidQuantizer)
	}
	q.Dequant[0] = 0
	if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid quantizer err=%v want %v", err, ErrInvalidQuantizer)
	}
}

func TestQuantizeFPNoQMatrixAllocs(t *testing.T) {
	q := FPQuantizer{
		Dequant: [2]int16{78, 93},
		Round:   [2]int16{39, 46},
		Quant:   [2]int16{840, 704},
	}
	coeff := make([]int32, 16)
	qcoeff := make([]int32, 16)
	dqcoeff := make([]int32, 16)
	scan := []int16{0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("QuantizeFPNoQMatrix allocated: %f", allocs)
	}
}

func FuzzDequantizeBlock(f *testing.F) {
	f.Add(uint8(8), uint8(0), int16(0), uint8(4), uint8(4), int16(1))
	f.Add(uint8(10), uint8(96), int16(-3), uint8(8), uint8(4), int16(-7))
	f.Add(uint8(12), uint8(255), int16(12), uint8(16), uint8(16), int16(31))

	f.Fuzz(func(t *testing.T, rawBitDepth uint8, qIndex uint8, delta int16, rawW uint8, rawH uint8, coeffValue int16) {
		bitDepths := [3]uint8{8, 10, 12}
		bitDepth := bitDepths[int(rawBitDepth)%len(bitDepths)]
		width := int(rawW%16) + 1
		height := int(rawH%16) + 1
		coeffStride := height + 2
		dstStride := height + 3
		coeff := make([]int16, coeffStride*width)
		dst := make([]int32, dstStride*width)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[col*coeffStride+row] = coeffValue + int16(row+col)
			}
		}
		dc, err := DCQuant(qIndex, delta, bitDepth)
		if err != nil {
			t.Fatal(err)
		}
		ac, err := ACQuant(qIndex, -delta, bitDepth)
		if err != nil {
			t.Fatal(err)
		}
		q := Quantizer{DC: dc, AC: ac}
		if err := DequantizeBlock(dst, dstStride, coeff, coeffStride, width, height, q); err != nil {
			t.Fatalf("DequantizeBlock err=%v", err)
		}
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				scale := ac
				if row == 0 && col == 0 {
					scale = dc
				}
				want := int32(coeff[col*coeffStride+row]) * scale
				got := dst[col*dstStride+row]
				if got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", col, row, got, want)
				}
			}
		}
	})
}

func FuzzQuantizeFPNoQMatrix(f *testing.F) {
	f.Add(int32(-449), int32(624), uint16(78), uint16(93), uint8(0))
	f.Add(int32(1<<20-1), int32(-(1 << 20)), uint16(4), uint16(32767), uint8(1))

	f.Fuzz(func(t *testing.T, c0 int32, c1 int32, rawDC uint16, rawAC uint16, rawLogScale uint8) {
		dequantDC := int16(rawDC%32765 + 3)
		dequantAC := int16(rawAC%32765 + 3)
		logScale := rawLogScale & 1
		q := FPQuantizer{
			Dequant:  [2]int16{dequantDC, dequantAC},
			Round:    [2]int16{dequantDC / 2, dequantAC / 2},
			Quant:    [2]int16{int16((1 << 16) / int(dequantDC)), int16((1 << 16) / int(dequantAC))},
			LogScale: logScale,
		}
		coeff := []int32{c0, c1, 0, c0 - c1}
		qcoeff := make([]int32, len(coeff))
		dqcoeff := make([]int32, len(coeff))
		scan := []int16{0, 1, 2, 3}
		eob, err := QuantizeFPNoQMatrix(qcoeff, dqcoeff, coeff, scan, q)
		if err != nil {
			t.Fatalf("QuantizeFPNoQMatrix err=%v", err)
		}
		wantEOB := 0
		for i, rawRC := range scan {
			rc := int(rawRC)
			if qcoeff[rc] == 0 {
				if dqcoeff[rc] != 0 {
					t.Fatalf("dqcoeff[%d]=%d with zero qcoeff", rc, dqcoeff[rc])
				}
				continue
			}
			wantEOB = i + 1
			absQ := qcoeff[rc]
			negative := absQ < 0
			if negative {
				absQ = -absQ
			}
			idx := 0
			if rc != 0 {
				idx = 1
			}
			wantDQ := (absQ * int32(q.Dequant[idx])) >> q.LogScale
			if negative {
				wantDQ = -wantDQ
			}
			if dqcoeff[rc] != wantDQ {
				t.Fatalf("dqcoeff[%d]=%d want %d", rc, dqcoeff[rc], wantDQ)
			}
		}
		if eob != wantEOB {
			t.Fatalf("eob=%d want %d", eob, wantEOB)
		}
	})
}

func BenchmarkQuantLookup(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = DCQuant(uint8(i), int16(i&7)-3, 10)
		_, _ = ACQuant(uint8(i), int16(i&7)-3, 10)
	}
}

func BenchmarkDequantizeBlock(b *testing.B) {
	coeff := make([]int16, 64*64)
	dst := make([]int32, 64*64)
	q := Quantizer{DC: 80, AC: 97}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = DequantizeBlock(dst, 64, coeff, 64, 64, 64, q)
	}
}
