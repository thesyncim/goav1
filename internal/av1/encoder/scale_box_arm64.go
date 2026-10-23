//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// boxInterpolateRow8 is InterpolateRow_C for 8-bit samples on NEON, as
// libyuv's InterpolateRow_NEON: (s0*(256-f) + s1*f + 128) >> 8 by widening
// multiply-accumulate, and the f == 128 HalfRow as a rounding average.
func boxInterpolateRow8(dst, s0, s1 []uint8, f int) {
	n := len(s0)
	s1, dst = s1[:n], dst[:n]
	x := 0
	if n >= 16 {
		p0, p1 := unsafe.Pointer(&s0[0]), unsafe.Pointer(&s1[0])
		pd := unsafe.Pointer(&dst[0])
		if f == 128 {
			for ; x+16 <= n; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p0, x)))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p1, x)))
				a.Average(b).StoreArray((*[16]uint8)(unsafe.Add(pd, x)))
			}
		} else {
			f0 := archsimd.BroadcastUint8x16(uint8(256 - f))
			f1 := archsimd.BroadcastUint8x16(uint8(f))
			bias := archsimd.BroadcastUint16x8(128)
			sh8 := archsimd.BroadcastInt16x8(-8)
			for ; x+16 <= n; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p0, x)))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p1, x)))
				lo := a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).Shift(sh8).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				hi := a.HiToLo().MulWidenLo(f0.HiToLo()).Add(b.HiToLo().MulWidenLo(f1.HiToLo())).Add(bias).Shift(sh8).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(pd, x)))
			}
		}
	}
	if x < n {
		boxInterpolateRow(dst[x:], s0[x:], s1[x:], f, 128)
	}
}
