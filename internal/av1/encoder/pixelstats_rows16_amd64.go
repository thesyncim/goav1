//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// pixelStatsRows16Arch is pixelStatsRows16 as SVT-AV1's variance_kernel_avx2:
// each 16-sample row segment zero-extends to 16 int16 lanes (VPMOVZXBW), the
// difference squares and pair-sums straight into int32 lanes (VPMADDWD), and
// the signed sum accumulates in int16 lanes. Every 16-sample step adds one
// difference (|d| <= 255) to each int16 lane, so 64 steps fit before the flush
// to int32; the sse of a 64x64 block is at most 4096*65025 < 2^31.
//
// The kernels run only behind gosimdKernelsSupported (AVX2).
func pixelStatsRows16Arch(src []byte, srcStride int, ref []byte, refStride int, w, h int) (uint32, int32) {
	if h <= 0 {
		return 0, 0
	}
	_ = src[(h-1)*srcStride+w-1]
	_ = ref[(h-1)*refStride+w-1]
	groupRows := max(1, 1024/w)
	var sse, sum32 archsimd.Int32x8
	for r0 := 0; r0 < h; r0 += groupRows {
		var sum16 archsimd.Int16x16
		r1 := min(r0+groupRows, h)
		for r := r0; r < r1; r++ {
			s := src[r*srcStride : r*srcStride+w]
			p := ref[r*refStride : r*refStride+w]
			for c := 0; c+16 <= len(s) && c+16 <= len(p); c += 16 {
				a := archsimd.LoadUint8x16(s[c : c+16]).ExtendToUint16().BitsToInt16()
				b := archsimd.LoadUint8x16(p[c : c+16]).ExtendToUint16().BitsToInt16()
				d := a.Sub(b)
				sum16 = sum16.Add(d)
				sse = sse.Add(d.DotProductPairs(d))
			}
		}
		sum32 = sum32.Add(sum16.GetLo().ExtendToInt32()).Add(sum16.GetHi().ExtendToInt32())
	}
	return uint32(reduceInt32x8(sse)), reduceInt32x8(sum32)
}

// reduceInt32x8 returns the sum of the eight lanes.
func reduceInt32x8(v archsimd.Int32x8) int32 {
	q := v.GetLo().Add(v.GetHi())
	return q.GetElem(0) + q.GetElem(1) + q.GetElem(2) + q.GetElem(3)
}
