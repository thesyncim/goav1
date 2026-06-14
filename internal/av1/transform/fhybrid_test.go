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

func TestForwardBlockRejectsUnsupportedHybrid(t *testing.T) {
	var residual [64]int16
	var coeff [64]int32
	var scratch [64]int32
	if err := ForwardBlock(coeff[:], 8, residual[:], 8, scratch[:], Size{Width: 8, Height: 8}, TypeFlipADSTDCT); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("FlipADST err=%v want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock(coeff[:], 8, residual[:], 8, scratch[:], Size{Width: 16, Height: 16}, TypeADSTADST); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("16x16 ADST err=%v want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock(coeff[:], 8, residual[:], 8, nil, Size{Width: 8, Height: 8}, TypeADSTADST); !errors.Is(err, ErrInvalidTransform) {
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
