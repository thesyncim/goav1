// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"

	"simd/archsimd"
)

func TestRoundShiftNarrowInt32x4ToInt16x8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x51f7))
	values := []int32{
		-1 << 31, -1<<31 + 1, -1 << 30, -32769, -32768, -3, -2, -1,
		0, 1, 2, 3, 32767, 32768, 1<<30 - 1, 1<<31 - 1,
	}
	for range 1024 {
		values = append(values, int32(rng.Uint32()))
	}
	for _, shift := range []uint8{1, 2, 8, 11, 12, 13, 31} {
		for start := 0; start < len(values); start += 8 {
			var lanes [8]int32
			for i := range lanes {
				if start+i < len(values) {
					lanes[i] = values[start+i]
				}
			}
			lo := archsimd.LoadInt32x4Array((*[4]int32)(lanes[:4]))
			hi := archsimd.LoadInt32x4Array((*[4]int32)(lanes[4:]))
			var got [8]int16
			roundShiftNarrowInt32x4ToInt16x8(lo, hi, shift).StoreArray(&got)
			for i, value := range lanes {
				want := clipInt16(int32(roundShift(int64(value), int(shift))))
				if got[i] != want {
					t.Fatalf("shift=%d value=%d SIMD=%d scalar=%d", shift, value, got[i], want)
				}
			}
		}
	}
}

func TestInverseDCT8Col8SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xd8c8))
	ranges := [][2]int32{{-(1 << 11), (1 << 11) - 1}, {-4095, 4095}}
	for iter := 0; iter < 40000; iter++ {
		r := ranges[rng.Intn(len(ranges))]
		min, max := r[0], r[1]
		stride := 8 + rng.Intn(3)
		a := make([]int32, 8*stride)
		for k := 0; k < 8; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = min + int32(rng.Int63n(int64(max)-int64(min)+1))
			}
		}
		b := make([]int32, len(a))
		asm := make([]int32, len(a))
		copy(b, a)
		copy(asm, a)
		for col := 0; col < 8; col++ {
			inverseDCT8(a[col:], stride, min, max)
		}
		inverseDCT8Col8SIMD(b, stride, min, max)
		for col := 0; col < 8; col += 2 {
			inverseDCT8Col2NEONAdapter(asm[col:], stride, min, max)
		}
		for k := 0; k < 8; k++ {
			for col := 0; col < 8; col++ {
				i := k*stride + col
				if a[i] != b[i] {
					t.Fatalf("iter=%d range=[%d,%d] stride=%d row=%d col=%d: scalar=%d simd=%d",
						iter, min, max, stride, k, col, a[i], b[i])
				}
				if a[i] != asm[i] {
					t.Fatalf("iter=%d range=[%d,%d] stride=%d row=%d col=%d: scalar=%d neon=%d",
						iter, min, max, stride, k, col, a[i], asm[i])
				}
			}
		}
	}
}

func benchDCT8x8(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	const stride = 8
	buf := make([]int32, 8*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 12), (1<<12)-1)
	}
}

// int16 8-wide vs int32 4-wide (x2 for 8 cols) vs NEON asm (x4 Col2).
func BenchmarkDCT8x8_Int16(b *testing.B) { benchDCT8x8(b, inverseDCT8Col8SIMD) }
func BenchmarkDCT8x8_ASMCol4(b *testing.B) {
	benchDCT8x8(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col4NEONAdapter(buf, s, mn, mx)
		inverseDCT8Col4NEONAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT8x8_ASMCol2(b *testing.B) {
	benchDCT8x8(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col2NEONAdapter(buf, s, mn, mx)
		inverseDCT8Col2NEONAdapter(buf[2:], s, mn, mx)
		inverseDCT8Col2NEONAdapter(buf[4:], s, mn, mx)
		inverseDCT8Col2NEONAdapter(buf[6:], s, mn, mx)
	})
}

func TestInverseDCT8Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16b))
	// The production dispatcher only enters this SIMD kernel when every
	// pre-column value is within the interval-certified DCT8 bound.
	min, max := -int16ColumnSIMDInputBound(8), int16ColumnSIMDInputBound(8)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 8*stride)
		for k := 0; k < 8; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(min + int32(rng.Int63n(int64(max)-int64(min)+1)))
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT8(a[col:], stride, min, max) // T=int16 via inference
		}
		inverseDCT8Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func BenchmarkDCT8x8_Int16Buf(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	const stride = 8
	buf := make([]int16, 8*stride+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int16, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		inverseDCT8Col8SIMD16(work, stride, -(1 << 12), (1<<12)-1)
	}
}

func TestInverseDCT16Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1616))
	min, max := -int16ColumnSIMDInputBound(16), int16ColumnSIMDInputBound(16)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 16*stride)
		for k := 0; k < 16; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(min + int32(rng.Int63n(int64(max)-int64(min)+1)))
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT16(a[col:], stride, min, max)
		}
		inverseDCT16Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func BenchmarkDCT16x8_Int16Buf(b *testing.B) {
	rng := rand.New(rand.NewSource(9))
	const stride = 8
	buf := make([]int16, 16*stride+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int16, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		inverseDCT16Col8SIMD16(work, stride, -(1 << 12), (1<<12)-1)
	}
}
func BenchmarkDCT16x8_ASMCol4(b *testing.B) {
	benchDCTx8ASM(b, 16, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col4NEONAdapter(buf, s, mn, mx)
		inverseDCT16Col4NEONAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT16x8_ASMCol2(b *testing.B) {
	benchDCTx8ASM(b, 16, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col2NEONAdapter(buf, s, mn, mx)
		inverseDCT16Col2NEONAdapter(buf[2:], s, mn, mx)
		inverseDCT16Col2NEONAdapter(buf[4:], s, mn, mx)
		inverseDCT16Col2NEONAdapter(buf[6:], s, mn, mx)
	})
}

func TestInverseDCT32Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x3232))
	min, max := -int16ColumnSIMDInputBound(32), int16ColumnSIMDInputBound(32)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 32*stride)
		for k := 0; k < 32; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(min + int32(rng.Int63n(int64(max)-int64(min)+1)))
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT32(a[col:], stride, min, max)
		}
		inverseDCT32Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func TestInverseDCT64Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x6464))
	min, max := -int16ColumnSIMDInputBound(64), int16ColumnSIMDInputBound(64)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 64*stride)
		for k := 0; k < 64; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(min + int32(rng.Int63n(int64(max)-int64(min)+1)))
			}
		}
		b := make([]int16, len(a))
		copy(b, a)
		for col := 0; col < 8; col++ {
			inverseDCT64(a[col:], stride, min, max)
		}
		inverseDCT64Col8SIMD16(b, stride, min, max)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("iter=%d at %d: scalar=%d simd=%d", iter, i, a[i], b[i])
			}
		}
	}
}

func benchDCTx8Int16(b *testing.B, n int, fn func([]int16, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	stride := 8
	buf := make([]int16, n*stride+8)
	for i := range buf {
		buf[i] = int16(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int16, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 12), (1<<12)-1)
	}
}
func BenchmarkDCT32x8_Int16Buf(b *testing.B) { benchDCTx8Int16(b, 32, inverseDCT32Col8SIMD16) }
func BenchmarkDCT64x8_Int16Buf(b *testing.B) { benchDCTx8Int16(b, 64, inverseDCT64Col8SIMD16) }
func benchDCTx8ASM(b *testing.B, n int, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(9))
	stride := 8
	buf := make([]int32, n*stride+8)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 12), (1<<12)-1)
	}
}
func BenchmarkDCT32x8_ASM(b *testing.B) {
	benchDCTx8ASM(b, 32, func(buf []int32, s int, mn, mx int32) {
		inverseDCT32Col4NEONAdapter(buf, s, mn, mx)
		inverseDCT32Col4NEONAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT64x8_ASM(b *testing.B) {
	benchDCTx8ASM(b, 64, func(buf []int32, s int, mn, mx int32) {
		inverseDCT64Col4NEONAdapter(buf, s, mn, mx)
		inverseDCT64Col4NEONAdapter(buf[4:], s, mn, mx)
	})
}
