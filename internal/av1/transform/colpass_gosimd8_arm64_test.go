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

func TestInt16ColumnSIMDInputGuardMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16b001))
	for _, height := range []int{8, 16, 32, 64} {
		width := height
		limit := int16ColumnSIMDInputBound(height)
		clamps := [][2]int32{{minInt16, maxInt16}, {0, maxInt16}, {minInt16, 0}, {1, maxInt16}, {minInt16, -1}, {0, 0}}
		for _, clamp := range clamps {
			for _, pattern := range []string{"random", "at-bound", "over-bound", "min-int16", "max-int16"} {
				buf := make([]int16, width*height)
				for i := range buf {
					switch pattern {
					case "random":
						buf[i] = int16(rng.Uint32())
					case "at-bound":
						if i&1 == 0 {
							buf[i] = int16(limit)
						} else {
							buf[i] = int16(-limit)
						}
					case "over-bound":
						if i&1 == 0 {
							buf[i] = int16(limit + 1)
						} else {
							buf[i] = int16(-limit - 1)
						}
					case "min-int16":
						buf[i] = minInt16
					case "max-int16":
						buf[i] = maxInt16
					}
				}
				got := int16ColumnSIMDInputSafeSIMD(buf, width, height, clamp[0], clamp[1])
				want := int16ColumnSIMDInputSafeScalar(buf, width, height, clamp[0], clamp[1])
				if got != want {
					t.Fatalf("height=%d clamp=[%d,%d] pattern=%s SIMD=%t scalar=%t", height, clamp[0], clamp[1], pattern, got, want)
				}
			}
		}
	}

	invalid := []struct {
		buf           []int16
		width, height int
		min, max      int32
	}{
		{nil, 8, 16, minInt16, maxInt16},
		{make([]int16, 8*16), 8, 16, 1, 0},
		{make([]int16, 8*16), 8, 16, -32769, maxInt16},
		{make([]int16, 8*16), 8, 16, minInt16, 32768},
	}
	for _, tc := range invalid {
		got := int16ColumnSIMDInputSafeSIMD(tc.buf, tc.width, tc.height, tc.min, tc.max)
		want := int16ColumnSIMDInputSafeScalar(tc.buf, tc.width, tc.height, tc.min, tc.max)
		if got != want {
			t.Fatalf("invalid input width=%d height=%d clamp=[%d,%d] SIMD=%t scalar=%t", tc.width, tc.height, tc.min, tc.max, got, want)
		}
	}
}

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
			inverseDCT8Col2SIMDAdapter(asm[col:], stride, min, max)
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
		inverseDCT8Col4SIMDAdapter(buf, s, mn, mx)
		inverseDCT8Col4SIMDAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT8x8_ASMCol2(b *testing.B) {
	benchDCT8x8(b, func(buf []int32, s int, mn, mx int32) {
		inverseDCT8Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[2:], s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[4:], s, mn, mx)
		inverseDCT8Col2SIMDAdapter(buf[6:], s, mn, mx)
	})
}

func TestInverseDCT8Col8SIMD16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16b))
	min, max := int32(minInt16), int32(maxInt16)
	for iter := 0; iter < 40000; iter++ {
		stride := 8 + rng.Intn(3)
		a := make([]int16, 8*stride)
		for k := 0; k < 8; k++ {
			for col := 0; col < 8; col++ {
				a[k*stride+col] = int16(rng.Uint32())
			}
		}
		if iter == 0 {
			clear(a)
			a[0], a[4*stride] = 30000, 30000
		} else if iter == 1 {
			for k := 0; k < 8; k++ {
				for col := 0; col < 8; col++ {
					if (k+col)&1 == 0 {
						a[k*stride+col] = minInt16
					} else {
						a[k*stride+col] = maxInt16
					}
				}
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

func TestInverseDCT8Col8SIMD16FullRangeClampIntervals(t *testing.T) {
	clamps := [][2]int32{{minInt16, maxInt16}, {1, maxInt16}, {minInt16, -1}, {-1024, 2047}}
	patterns := [][8]int16{
		{30000, -30000, 12000, -8000, 30000, 24000, -16000, -30000},
		{30000, minInt16, -30000, maxInt16, -30000, maxInt16, 30000, minInt16},
		{minInt16, maxInt16, minInt16, maxInt16, maxInt16, minInt16, maxInt16, minInt16},
	}
	for _, clamp := range clamps {
		for patternIndex, pattern := range patterns {
			const stride = 8
			input := make([]int16, 8*stride)
			for row, value := range pattern {
				for col := 0; col < 8; col++ {
					input[row*stride+col] = value
				}
			}
			want, got := append([]int16(nil), input...), append([]int16(nil), input...)
			for col := 0; col < 8; col++ {
				inverseDCT8(want[col:], stride, clamp[0], clamp[1])
			}
			inverseDCT8Col8SIMD16(got, stride, clamp[0], clamp[1])
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("clamp=[%d,%d] pattern=%d index=%d: scalar=%d SIMD=%d", clamp[0], clamp[1], patternIndex, i, want[i], got[i])
				}
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
		inverseDCT16Col4SIMDAdapter(buf, s, mn, mx)
		inverseDCT16Col4SIMDAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT16x8_ASMCol2(b *testing.B) {
	benchDCTx8ASM(b, 16, func(buf []int32, s int, mn, mx int32) {
		inverseDCT16Col2SIMDAdapter(buf, s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[2:], s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[4:], s, mn, mx)
		inverseDCT16Col2SIMDAdapter(buf[6:], s, mn, mx)
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
		inverseDCT32Col4SIMDAdapter(buf, s, mn, mx)
		inverseDCT32Col4SIMDAdapter(buf[4:], s, mn, mx)
	})
}
func BenchmarkDCT64x8_PureGo(b *testing.B) {
	benchDCTx8ASM(b, 64, func(buf []int32, s int, mn, mx int32) {
		inverseDCT64Col4PureGo(buf, s, mn, mx)
		inverseDCT64Col4PureGo(buf[4:], s, mn, mx)
	})
}
