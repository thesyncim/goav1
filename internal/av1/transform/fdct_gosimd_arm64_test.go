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
	check("8x8", forwardDCT8x8Impl, "forwardDCT8x8SIMDGuarded")
	check("16x16", forwardDCT16x16Impl, "forwardDCT16x16SIMDGuarded")
	check("32x32", forwardDCT32x32Impl, "forwardDCT32x32SIMDGuarded")
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

func TestForwardDCTSIMDZeroAlloc(t *testing.T) {
	var residual [16]int16
	for i := range residual {
		residual[i] = int16(i%511) - 255
	}
	var coeff [16]int32
	allocs := testing.AllocsPerRun(100, func() {
		forwardDCT4x4SIMD(coeff[:], 4, residual[:], 4)
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
		{name: "simd", fn: benchmarkForwardDCTGuardedNEON(forwardDCT8x8SIMD, forwardDCT8x8PureGo, 8)},
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
		{name: "simd", fn: benchmarkForwardDCTGuardedNEON(forwardDCT16x16SIMD, forwardDCT16x16PureGo, 16)},
		{name: "purego", fn: forwardDCT16x16PureGo},
	})
}

// benchmarkForwardDCTGuardedNEON includes the same range check and scalar
// fallback as the production asm dispatch.
func benchmarkForwardDCTGuardedNEON(neon, pure func([]int32, int, []int16, int), side int) func([]int32, int, []int16, int) {
	return func(coeff []int32, coeffStride int, residual []int16, residualStride int) {
		if !residualFitsMagnitude(residual, residualStride, side, side, 255) {
			pure(coeff, coeffStride, residual, residualStride)
			return
		}
		neon(coeff, coeffStride, residual, residualStride)
	}
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
