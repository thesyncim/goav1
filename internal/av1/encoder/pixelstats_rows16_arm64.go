//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// pixelStatsRows16Arch is pixelStatsRows16 with NEON's widening multiply: the
// int16 differences square straight into int32 lanes (SMULL on the low and
// high halves) instead of being sign-extended first, and the signed sum
// accumulates in int16 lanes (at most 510 per 16-sample step, flushed to
// int32 every 32 steps).
//
// The flush runs after each group of rows holding 32 steps rather than on a
// per-step counter: the conditional reset in the column loop made the
// register allocator swap the int16 accumulator with its zero on every
// back edge.
func pixelStatsRows16Arch(src []byte, srcStride int, ref []byte, refStride int, w, h int) (uint32, int32) {
	if h <= 0 {
		return 0, 0
	}
	_ = src[(h-1)*srcStride+w-1]
	_ = ref[(h-1)*refStride+w-1]
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	var sse0, sse1, sum32 archsimd.Int32x4
	groupRows := max(1, 512/w) // 32 steps of 16 samples
	for r0 := 0; r0 < h; r0 += groupRows {
		var sum16 archsimd.Int16x8
		for r := r0; r < min(r0+groupRows, h); r++ {
			s := unsafe.Add(sp, r*srcStride)
			p := unsafe.Add(rp, r*refStride)
			for c := 0; c < w; c += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(s, c)))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p, c)))
				dl := a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16())
				dh := a.HiToLo().ExtendLo8ToUint16().BitsToInt16().Sub(b.HiToLo().ExtendLo8ToUint16().BitsToInt16())
				sum16 = sum16.Add(dl).Add(dh)
				sse0 = sse0.Add(dl.MulWidenLo(dl)).Add(dh.MulWidenLo(dh))
				dl, dh = dl.HiToLo(), dh.HiToLo()
				sse1 = sse1.Add(dl.MulWidenLo(dl)).Add(dh.MulWidenLo(dh))
			}
		}
		sum32 = sum32.Add(sum16.ExtendLo4ToInt32()).Add(sum16.HiToLo().ExtendLo4ToInt32())
	}
	total, signed := reducePixelAcc(sse0.Add(sse1), sum32)
	return uint32(total), int32(signed)
}
