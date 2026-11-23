//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"simd/archsimd"
)

func quantizePackInt16x8SIMD(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	lo16 := lo.TruncToInt16()
	hi16 := hi.TruncToInt16()
	return lo16.ToBits().ReshapeToUint64s().InterleaveLo(hi16.ToBits().ReshapeToUint64s()).ReshapeToUint16s().BitsToInt16()
}
