//go:build arm64 && !purego

package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestSADDotProdMatchesPureGo(t *testing.T) {
	if !cpu.Detected.DOTPROD {
		t.Skip("DOTPROD not detected")
	}

	rng := rand.New(rand.NewSource(461))
	const stride = 127
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}

	for range 2000 {
		off := rng.Intn(stride*96) + rng.Intn(stride-32)
		if got, want := sad16x16DotProd(src[off:], ref[off:], stride), sad16x16PureGo(src[off:], ref[off:], stride); got != want {
			t.Fatalf("16x16 off %d: dotprod %d want %d", off, got, want)
		}
		if got, want := sad32x32DotProd(src[off:], ref[off:], stride), sad32x32PureGo(src[off:], ref[off:], stride); got != want {
			t.Fatalf("32x32 off %d: dotprod %d want %d", off, got, want)
		}
		w0, w1, w2, w3 := sad16x16x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 := sad16x16x4Step4DotProd(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("16x16x4 off %d: dotprod (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
		w0, w1, w2, w3 = sad32x32x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 = sad32x32x4Step4DotProd(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("32x32x4 off %d: dotprod (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}

		row := rng.Intn(64) + 32
		col := rng.Intn(stride-72) + 32
		off = row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0, w1, w2, w3 = sad16x16x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 = sad16x16x4DotProd(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("16x16x4d off %d: dotprod (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
		w0, w1, w2, w3 = sad32x32x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 = sad32x32x4DotProd(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("32x32x4d off %d: dotprod (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

var sadDotProdBenchSink int

func BenchmarkSADDotProd(b *testing.B) {
	if !cpu.Detected.DOTPROD {
		b.Skip("DOTPROD not detected")
	}

	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}

	for _, tc := range []struct {
		name string
		fn   func([]byte, []byte, int) int
	}{
		{"16x16/NEON", sad16x16NEON},
		{"16x16/DOTPROD", sad16x16DotProd},
		{"32x32/NEON", sad32x32NEON},
		{"32x32/DOTPROD", sad32x32DotProd},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			sum := 0
			for b.Loop() {
				sum += tc.fn(src, ref, 96)
			}
			sadDotProdBenchSink = sum
		})
	}
}

func BenchmarkSADX4Step4DotProd(b *testing.B) {
	if !cpu.Detected.DOTPROD {
		b.Skip("DOTPROD not detected")
	}

	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}

	for _, tc := range []struct {
		name string
		fn   func([]byte, []byte, int) (int, int, int, int)
	}{
		{"16x16x4Step4/NEON", sad16x16x4Step4NEON},
		{"16x16x4Step4/DOTPROD", sad16x16x4Step4DotProd},
		{"32x32x4Step4/NEON", sad32x32x4Step4NEON},
		{"32x32x4Step4/DOTPROD", sad32x32x4Step4DotProd},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			sum := 0
			for b.Loop() {
				s0, s1, s2, s3 := tc.fn(src, ref, 96)
				sum += s0 + s1 + s2 + s3
			}
			sadDotProdBenchSink = sum
		})
	}
}

func BenchmarkSADX4DotProd(b *testing.B) {
	if !cpu.Detected.DOTPROD {
		b.Skip("DOTPROD not detected")
	}

	const stride = 96
	src := make([]byte, stride*stride)
	ref := make([]byte, stride*stride)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 32*stride + 32

	for _, tc := range []struct {
		name string
		fn   func([]byte, []byte, []byte, []byte, []byte, int) (int, int, int, int)
	}{
		{"16x16x4/NEON", sad16x16x4NEON},
		{"16x16x4/DOTPROD", sad16x16x4DotProd},
		{"32x32x4/NEON", sad32x32x4NEON},
		{"32x32x4/DOTPROD", sad32x32x4DotProd},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			sum := 0
			for b.Loop() {
				s0, s1, s2, s3 := tc.fn(src[off:], ref[off+2:], ref[off-2:], ref[off+2*stride:], ref[off-2*stride:], stride)
				sum += s0 + s1 + s2 + s3
			}
			sadDotProdBenchSink = sum
		})
	}
}
