// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestForwardDCTSIMDBindings(t *testing.T) {
	check := func(name string, got any, want string) {
		t.Helper()
		fn := runtime.FuncForPC(reflect.ValueOf(got).Pointer())
		if fn == nil {
			t.Fatalf("%s binding has no FuncForPC entry", name)
		}
		if !strings.Contains(fn.Name(), want) {
			t.Fatalf("%s bound to %s, want %s", name, fn.Name(), want)
		}
	}
	check("4x4", forwardDCT4x4Impl, "forwardDCT4x4SIMD")
	check("8x8", forwardDCT8x8Impl, "forwardDCT8x8SIMD")
	check("16x16", forwardDCT16x16Impl, "forwardDCT16x16SIMD")
	check("32x32", forwardDCT32x32Impl, "forwardDCT32x32NEON")
}

func TestForwardDCTSIMDDirectMatchesPureGo(t *testing.T) {
	type kernel struct {
		name string
		w, h int
		simd func([]int32, int, []int16, int)
		pure func([]int32, int, []int16, int)
		neon func([]int32, int, []int16, int)
	}
	kernels := []kernel{
		{name: "4x4", w: 4, h: 4, simd: forwardDCT4x4SIMD, pure: forwardDCT4x4PureGo, neon: forwardDCT4x4NEON},
		{name: "8x8", w: 8, h: 8, simd: forwardDCT8x8SIMD, pure: forwardDCT8x8PureGo, neon: forwardDCT8x8NEON},
		{name: "16x16", w: 16, h: 16, simd: forwardDCT16x16SIMD, pure: forwardDCT16x16PureGo, neon: forwardDCT16x16NEON},
	}
	rng := rand.New(rand.NewSource(917))
	for _, k := range kernels {
		t.Run(k.name, func(t *testing.T) {
			resStride := k.w + 7
			coeffStride := k.h + 5
			residual := make([]int16, resStride*k.h)
			want := make([]int32, coeffStride*k.w)
			got := make([]int32, coeffStride*k.w)
			asm := make([]int32, coeffStride*k.w)
			for trial := range 1000 {
				for i := range residual {
					residual[i] = int16(rng.Intn(511) - 255)
				}
				clear(want)
				clear(got)
				clear(asm)
				k.pure(want, coeffStride, residual, resStride)
				k.simd(got, coeffStride, residual, resStride)
				k.neon(asm, coeffStride, residual, resStride)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("trial %d coeff[%d]=%d want %d", trial, i, got[i], want[i])
					}
					if asm[i] != want[i] {
						t.Fatalf("trial %d coeff[%d] NEON=%d want %d", trial, i, asm[i], want[i])
					}
				}
			}
		})
	}
}

func TestForwardDCT4x4SIMDExactResidualLength(t *testing.T) {
	var residual [16]int16
	var want, got [16]int32
	rng := rand.New(rand.NewSource(401))
	for trial := range 1000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(1<<16) - (1 << 15))
		}
		clear(want[:])
		clear(got[:])
		forwardDCT4x4PureGo(want[:], 4, residual[:], 4)
		forwardDCT4x4SIMD(got[:], 4, residual[:], 4)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d coeff[%d] SIMD=%d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestForwardDCT4x4SIMDProvenInt32Range(t *testing.T) {
	var residual [16]int16
	var want, got [16]int32
	rng := rand.New(rand.NewSource(405))
	for trial := range 1000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(4097) - 2048)
		}
		clear(want[:])
		clear(got[:])
		forwardDCT4x4PureGo(want[:], 4, residual[:], 4)
		forwardDCT4x4SIMD(got[:], 4, residual[:], 4)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial %d coeff[%d] SIMD=%d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestForwardDCTSIMDFullInt16MatchesPureGo(t *testing.T) {
	type kernel struct {
		name string
		side int
		simd func([]int32, int, []int16, int)
		pure func([]int32, int, []int16, int)
	}
	kernels := []kernel{
		{name: "4x4", side: 4, simd: forwardDCT4x4SIMD, pure: forwardDCT4x4PureGo},
		{name: "8x8", side: 8, simd: forwardDCT8x8SIMD, pure: forwardDCT8x8PureGo},
		{name: "16x16", side: 16, simd: forwardDCT16x16SIMD, pure: forwardDCT16x16PureGo},
	}
	rng := rand.New(rand.NewSource(402))
	for _, k := range kernels {
		t.Run(k.name, func(t *testing.T) {
			residual := make([]int16, k.side*k.side)
			want := make([]int32, k.side*k.side)
			got := make([]int32, k.side*k.side)
			for trial := range 100 {
				for i := range residual {
					residual[i] = int16(rng.Intn(1<<16) - (1 << 15))
				}
				clear(want)
				clear(got)
				k.pure(want, k.side, residual, k.side)
				k.simd(got, k.side, residual, k.side)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("trial %d coeff[%d]=%d want %d", trial, i, got[i], want[i])
					}
				}
			}
		})
	}
}

func TestForwardDCTSIMDZeroAlloc(t *testing.T) {
	var residual [256]int16
	for i := range residual {
		residual[i] = int16(i%511) - 255
	}
	var coeff [256]int32
	allocs := testing.AllocsPerRun(100, func() {
		forwardDCT4x4SIMD(coeff[:16], 4, residual[:16], 4)
		forwardDCT8x8SIMD(coeff[:64], 8, residual[:64], 8)
		forwardDCT16x16SIMD(coeff[:], 16, residual[:], 16)
	})
	if allocs != 0 {
		t.Fatalf("SIMD forward DCT allocated %v objects/run, want 0", allocs)
	}
}

func BenchmarkForwardDCT4x4Kernels(b *testing.B) {
	var residual [16]int16
	for i := range residual {
		residual[i] = int16(i*29%400) - 200
	}
	benchmarkForwardDCTKernel(b, 4, residual[:], 4, []struct {
		name string
		fn   func([]int32, int, []int16, int)
	}{
		{name: "simd", fn: forwardDCT4x4SIMD},
		{name: "neon", fn: forwardDCT4x4NEON},
		{name: "purego", fn: forwardDCT4x4PureGo},
	})
}

func BenchmarkForwardDCT8x8Kernels(b *testing.B) {
	var residual [64]int16
	for i := range residual {
		residual[i] = int16(i*7%400) - 200
	}
	benchmarkForwardDCTKernel(b, 8, residual[:], 8, []struct {
		name string
		fn   func([]int32, int, []int16, int)
	}{
		{name: "simd", fn: forwardDCT8x8SIMD},
		{name: "neon", fn: forwardDCT8x8NEON},
		{name: "purego", fn: forwardDCT8x8PureGo},
	})
}

func BenchmarkForwardDCT16x16Kernels(b *testing.B) {
	var residual [256]int16
	for i := range residual {
		residual[i] = int16(i*11%400) - 200
	}
	benchmarkForwardDCTKernel(b, 16, residual[:], 16, []struct {
		name string
		fn   func([]int32, int, []int16, int)
	}{
		{name: "simd", fn: forwardDCT16x16SIMD},
		{name: "neon", fn: forwardDCT16x16NEON},
		{name: "purego", fn: forwardDCT16x16PureGo},
	})
}

func benchmarkForwardDCTKernel(b *testing.B, side int, residual []int16, residualStride int, kernels []struct {
	name string
	fn   func([]int32, int, []int16, int)
}) {
	b.Helper()
	for _, k := range kernels {
		b.Run(k.name, func(b *testing.B) {
			coeff := make([]int32, side*side)
			b.ReportAllocs()
			for b.Loop() {
				k.fn(coeff, side, residual, residualStride)
			}
		})
	}
}
