// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestInverseDCT8Col4SIMDMatchesScalar proves the int32 4-wide round-once DCT8
// column pass is byte-identical to the scalar inverseDCT8 on each of 4 columns,
// across the full int16 clamp range.
func TestInverseDCT8Col4SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xd4c4))
	ranges := [][2]int32{
		{-(1 << 12), (1 << 12) - 1},
		{-(1 << 14), (1 << 14) - 1},
		{-(1 << 15), (1 << 15) - 1},
	}
	for iter := 0; iter < 40000; iter++ {
		r := ranges[rng.Intn(len(ranges))]
		min, max := r[0], r[1]
		stride := 4 + rng.Intn(5)
		a := make([]int32, 8*stride)
		for k := 0; k < 8; k++ {
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
		for col := 0; col < 4; col++ {
			inverseDCT8(a[col:], stride, min, max)
		}
		inverseDCT8Col4SIMD(b, stride, min, max)
		for k := 0; k < 8; k++ {
			for col := 0; col < 4; col++ {
				i := k*stride + col
				if a[i] != b[i] {
					t.Fatalf("iter=%d range=[%d,%d] stride=%d row=%d col=%d: scalar=%d simd=%d",
						iter, min, max, stride, k, col, a[i], b[i])
				}
			}
		}
	}
}

func benchDCT8x4(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 4
	buf := make([]int32, 8*stride+8) // padding so 4-wide loads at buf[2:] tail don't over-read
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<13) - (1 << 12))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 14), (1<<14)-1)
	}
}

// int32 4-wide (round-once) vs int64 2-wide (x2 for 4 cols) vs NEON asm (x2).
func BenchmarkDCT8x4_Int32(b *testing.B) { benchDCT8x4(b, inverseDCT8Col4SIMD) }
func BenchmarkDCT8x4_Int64(b *testing.B) {
	benchDCT8x4(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col2SIMD(buf, s, mn, mx)
		inverseDCT8Col2SIMD(buf[2:], s, mn, mx)
	})
}
func BenchmarkDCT8x4_ASM(b *testing.B) {
	benchDCT8x4(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col2NEONAdapter(buf, s, mn, mx)
		inverseDCT8Col2NEONAdapter(buf[2:], s, mn, mx)
	})
}

func TestInverseDCT16Col4SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16c4))
	ranges := [][2]int32{{-(1 << 12), (1 << 12) - 1}, {-(1 << 15), (1 << 15) - 1}}
	for iter := 0; iter < 40000; iter++ {
		r := ranges[rng.Intn(len(ranges))]
		min, max := r[0], r[1]
		stride := 4 + rng.Intn(5)
		a := make([]int32, 16*stride)
		for k := 0; k < 16; k++ {
			for col := 0; col < 4; col++ {
				a[k*stride+col] = min + int32(rng.Int63n(int64(max)-int64(min)+1))
			}
		}
		b := make([]int32, len(a))
		copy(b, a)
		for col := 0; col < 4; col++ {
			inverseDCT16(a[col:], stride, min, max)
		}
		inverseDCT16Col4SIMD(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d range=[%d,%d] at %d: scalar=%d simd=%d", iter, min, max, i, a[i], b[i])
			}
		}
	}
}

func benchDCT16x4(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 4
	buf := make([]int32, 16*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<13) - (1 << 12))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 14), (1<<14)-1)
	}
}
func BenchmarkDCT16x4_Int32(b *testing.B) { benchDCT16x4(b, inverseDCT16Col4SIMD) }
func BenchmarkDCT16x4_ASM(b *testing.B) {
	benchDCT16x4(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col2NEONAdapter(buf, s, mn, mx)
		inverseDCT16Col2NEONAdapter(buf[2:], s, mn, mx)
	})
}
