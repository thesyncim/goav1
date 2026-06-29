package transform

import (
	"errors"
	"fmt"
	"math/rand"
	"testing"
)

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

func TestForwardBlock8x8HybridTrustedMatchesChecked(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for _, typ := range []Type{TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeIDTX} {
		for trial := range 500 {
			var residual [64]int16
			var got, want [64]int32
			var gotScratch, wantScratch [64]int32
			for i := range residual {
				residual[i] = int16(rng.Intn(511) - 255)
			}
			if err := ForwardBlock8x8HybridTrusted(got[:], 8, residual[:], 8, gotScratch[:], typ); err != nil {
				t.Fatalf("trial %d %s trusted: %v", trial, typeName(typ), err)
			}
			if err := ForwardBlock(want[:], 8, residual[:], 8, wantScratch[:], Size{Width: 8, Height: 8}, typ); err != nil {
				t.Fatalf("trial %d %s checked: %v", trial, typeName(typ), err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("trial %d %s coeff[%d]=%d want %d", trial, typeName(typ), i, got[i], want[i])
				}
			}
		}
	}
}

func TestForwardBlock8x8HybridTrustedRejectsUnsupported(t *testing.T) {
	var residual [64]int16
	var coeff [64]int32
	var scratch [64]int32
	if err := ForwardBlock8x8HybridTrusted(coeff[:], 8, residual[:], 8, scratch[:], TypeDCTDCT); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("DCT_DCT err=%v want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock8x8HybridTrusted(coeff[:], 8, residual[:], 8, scratch[:], TypeFlipADSTDCT); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("FlipADST err=%v want %v", err, ErrInvalidTransform)
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

func BenchmarkForwardBlock8x8HybridTrusted(b *testing.B) {
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
				if err := ForwardBlock8x8HybridTrusted(coeff, 8, residual, 8, scratch, typ); err != nil {
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

func BenchmarkForwardADST8(b *testing.B) {
	cases := []struct {
		name string
		base int32
	}{
		{
			name: "dense",
			base: -996,
		},
		{
			name: "zero",
			base: 0,
		},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			var inputs [16][8]int32
			for i := range inputs {
				for j := range inputs[i] {
					if tc.base == 0 {
						continue
					}
					inputs[i][j] = tc.base + int32(i*13+j*37)
				}
			}
			var out [8]int32
			sum := int32(0)
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				fwdADST8(&inputs[i&15], &out)
				sum += out[i&7]
			}
			forwardADST8BenchSink = sum
		})
	}
}

var forwardADST8BenchSink int32

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
