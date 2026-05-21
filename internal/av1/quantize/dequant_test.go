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
		2, -3, 4, 99,
		-5, 6, -7, 99,
	}
	dst := []int32{
		99, 99, 99, 99,
		99, 99, 99, 99,
	}
	q := Quantizer{DC: 4, AC: 8}
	if err := DequantizeBlock(dst, 4, coeff, 4, 3, 2, q); err != nil {
		t.Fatal(err)
	}
	want := []int32{
		8, -24, 32, 99,
		-40, 48, -56, 99,
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
	if err := DequantizeBlock(dst, 2, coeff[:3], 2, 2, 2, q); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("short coeff err=%v want %v", err, ErrInvalidQuantizer)
	}
	if err := DequantizeBlock(dst, 2, coeff, 2, 2, 2, Quantizer{AC: 8}); !errors.Is(err, ErrInvalidQuantizer) {
		t.Fatalf("invalid quantizer err=%v want %v", err, ErrInvalidQuantizer)
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

func FuzzDequantizeBlock(f *testing.F) {
	f.Add(uint8(8), uint8(0), int16(0), uint8(4), uint8(4), int16(1))
	f.Add(uint8(10), uint8(96), int16(-3), uint8(8), uint8(4), int16(-7))
	f.Add(uint8(12), uint8(255), int16(12), uint8(16), uint8(16), int16(31))

	f.Fuzz(func(t *testing.T, rawBitDepth uint8, qIndex uint8, delta int16, rawW uint8, rawH uint8, coeffValue int16) {
		bitDepths := [3]uint8{8, 10, 12}
		bitDepth := bitDepths[int(rawBitDepth)%len(bitDepths)]
		width := int(rawW%16) + 1
		height := int(rawH%16) + 1
		coeffStride := width + 2
		dstStride := width + 3
		coeff := make([]int16, coeffStride*height)
		dst := make([]int32, dstStride*height)
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				coeff[row*coeffStride+col] = coeffValue + int16(row+col)
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
				want := int32(coeff[row*coeffStride+col]) * scale
				got := dst[row*dstStride+col]
				if got != want {
					t.Fatalf("sample(%d,%d)=%d want %d", col, row, got, want)
				}
			}
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
