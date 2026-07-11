// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"reflect"
	"runtime"
	"testing"
)

// TestInverseDCT32Col4SIMDMatchesScalar is the 3-way differential: the int32
// 4-wide round-once DCT32 column pass and the NEON asm adapter must both be
// byte-identical to the scalar inverseDCT32 on each of 4 columns, across the
// clamp envelope up to the SIMD guard (+/-2^15), with min/max-saturated lanes.
func TestInverseDCT32Col4SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xd32c4))
	ranges := [][2]int32{
		{-(1 << 12), (1 << 12) - 1},
		{-(1 << 14), (1 << 14) - 1},
		{-(1 << 15), (1 << 15) - 1},
	}
	for iter := 0; iter < 8000; iter++ {
		r := ranges[rng.Intn(len(ranges))]
		min, max := r[0], r[1]
		stride := 4 + rng.Intn(5)
		a := make([]int32, 32*stride)
		for k := 0; k < 32; k++ {
			for col := 0; col < 4; col++ {
				var v int32
				switch rng.Intn(4) {
				case 0:
					v = min
				case 1:
					v = max
				default:
					v = min + int32(rng.Int63n(int64(max)-int64(min)+1))
				}
				a[k*stride+col] = v
			}
		}
		b := make([]int32, len(a))
		copy(b, a)
		c := make([]int32, len(a))
		copy(c, a)
		for col := 0; col < 4; col++ {
			inverseDCT32(a[col:], stride, min, max)
		}
		inverseDCT32Col4SIMD(b, stride, min, max)
		inverseDCT32Col4NEONAdapter(c, stride, min, max)
		for k := 0; k < 32; k++ {
			for col := 0; col < 4; col++ {
				i := k*stride + col
				if a[i] != b[i] {
					t.Fatalf("SIMD iter=%d range=[%d,%d] stride=%d row=%d col=%d: scalar=%d simd=%d",
						iter, min, max, stride, k, col, a[i], b[i])
				}
				if a[i] != c[i] {
					t.Fatalf("NEON iter=%d range=[%d,%d] stride=%d row=%d col=%d: scalar=%d neon=%d",
						iter, min, max, stride, k, col, a[i], c[i])
				}
			}
		}
	}
}

// TestInverseDCT32Col4SIMDWideRangeFallback pins the out-of-envelope guard:
// 12-bit column bounds (+/-2^17) must route to the NEON adapter and stay
// byte-exact.
func TestInverseDCT32Col4SIMDWideRangeFallback(t *testing.T) {
	rng := rand.New(rand.NewSource(0x17bd))
	min, max := int32(-(1<<17)), int32((1<<17)-1)
	for iter := 0; iter < 500; iter++ {
		stride := 4 + rng.Intn(3)
		a := make([]int32, 32*stride)
		for i := range a {
			a[i] = min + int32(rng.Int63n(int64(max)-int64(min)+1))
		}
		b := make([]int32, len(a))
		copy(b, a)
		for col := 0; col < 4; col++ {
			inverseDCT32(a[col:], stride, min, max)
		}
		inverseDCT32Col4SIMD(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d idx=%d scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

// TestInverseDCT32Col4DispatchBound pins the dispatch decision: the Go-SIMD
// kernel measured ~2.5% behind the (clang-generated) NEON asm, so the slot
// keeps the asm; the SIMD kernel stays as a tested, bit-exact alternative.
func TestInverseDCT32Col4DispatchBound(t *testing.T) {
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got, want := nameOf(inverseDCT32Col4Impl), nameOf(inverseDCT32Col4NEONAdapter); got != want {
		t.Errorf("inverseDCT32Col4Impl = %s, want %s", got, want)
	}
}

func TestInverseDCT32Col4SIMDZeroAlloc(t *testing.T) {
	buf := make([]int32, 32*4)
	rng := rand.New(rand.NewSource(5))
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<15) - 1<<14)
	}
	if a := testing.AllocsPerRun(50, func() { inverseDCT32Col4SIMD(buf, 4, -(1 << 15), (1<<15)-1) }); a != 0 {
		t.Errorf("allocated %.1f objects/run, want 0", a)
	}
}

func benchDCT32Col4(b *testing.B, fn func([]int32, int, int32, int32)) {
	buf := make([]int32, 32*4)
	rng := rand.New(rand.NewSource(6))
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<15) - 1<<14)
	}
	b.ReportAllocs()
	for b.Loop() {
		fn(buf, 4, -(1 << 15), (1<<15)-1)
	}
}

func BenchmarkInverseDCT32Col4NEON(b *testing.B) { benchDCT32Col4(b, inverseDCT32Col4NEONAdapter) }
func BenchmarkInverseDCT32Col4SIMD(b *testing.B) { benchDCT32Col4(b, inverseDCT32Col4SIMD) }
