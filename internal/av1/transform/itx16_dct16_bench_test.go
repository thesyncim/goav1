//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// Fair sparse-kernel benchmarks: every iteration restores a pristine block
// (the kernels run in place and the int16 kernels guard their inputs), and
// BenchmarkDCT16x8_CopyOnly measures that restore alone.

func pristine16x8(lowN int) []int32 {
	rng := rand.New(rand.NewSource(9))
	buf := make([]int32, 16*8+8)
	for i := range buf {
		if i/8 < lowN {
			buf[i] = int32(rng.Intn(1<<12) - (1 << 11))
		}
	}
	return buf
}

func benchCopyKernel(b *testing.B, lowN int, kernel func([]int32)) {
	src := pristine16x8(lowN)
	work := make([]int32, len(src))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, src)
		kernel(work)
	}
}

func BenchmarkDCT16x8_CopyOnly(b *testing.B) { benchCopyKernel(b, 4, func([]int32) {}) }
func BenchmarkDCT16x8_CopyCol4Low4x2(b *testing.B) {
	benchCopyKernel(b, 4, func(w []int32) {
		inverseDCT16LowCol4Impl(w, 8, 4, minInt16, maxInt16)
		inverseDCT16LowCol4Impl(w[4:], 8, 4, minInt16, maxInt16)
	})
}
func BenchmarkDCT16x8_CopyCol4Low8x2(b *testing.B) {
	benchCopyKernel(b, 8, func(w []int32) {
		inverseDCT16LowCol4Impl(w, 8, 8, minInt16, maxInt16)
		inverseDCT16LowCol4Impl(w[4:], 8, 8, minInt16, maxInt16)
	})
}
func BenchmarkDCT16x8_CopyCol4x2(b *testing.B) {
	benchCopyKernel(b, 16, func(w []int32) {
		inverseDCT16Col4SIMDAdapter(w, 8, minInt16, maxInt16)
		inverseDCT16Col4SIMDAdapter(w[4:], 8, minInt16, maxInt16)
	})
}
func BenchmarkDCT16x8_CopyRawLow4(b *testing.B) {
	benchCopyKernel(b, 4, func(w []int32) { inverseDCT16Col8RawLow4SIMD16(w, 8) })
}
func BenchmarkDCT16x8_CopyRawLow8(b *testing.B) {
	benchCopyKernel(b, 8, func(w []int32) { inverseDCT16Col8RawLow8SIMD16(w, 8) })
}
func BenchmarkDCT16x8_CopyRaw(b *testing.B) {
	benchCopyKernel(b, 16, func(w []int32) { inverseDCT16Col8RawSIMD16(w, 8) })
}
