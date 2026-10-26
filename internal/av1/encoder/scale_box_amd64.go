//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// boxInterpolateRow8 is InterpolateRow_C for 8-bit samples on AVX2, as
// libyuv's InterpolateRow_AVX2: (s0*(256-f) + s1*f + 128) >> 8 in 16-bit
// lanes (at most 255*256 + 128, so the unsigned lane never wraps), and the
// f == 128 HalfRow as a rounding average.
func boxInterpolateRow8(dst, s0, s1 []uint8, f int) {
	n := len(s0)
	s1, dst = s1[:n], dst[:n]
	x := 0
	if encAVX2 && n >= 16 {
		p0, p1 := unsafe.Pointer(&s0[0]), unsafe.Pointer(&s1[0])
		pd := unsafe.Pointer(&dst[0])
		if f == 128 {
			for ; x+16 <= n; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p0, x)))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p1, x)))
				a.Average(b).StoreArray((*[16]uint8)(unsafe.Add(pd, x)))
			}
		} else {
			f0 := archsimd.BroadcastUint16x16(uint16(256 - f))
			f1 := archsimd.BroadcastUint16x16(uint16(f))
			bias := archsimd.BroadcastUint16x16(128)
			for ; x+16 <= n; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p0, x))).ExtendToUint16()
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p1, x))).ExtendToUint16()
				v := a.Mul(f0).Add(b.Mul(f1)).Add(bias).ShiftAllRight(8)
				packU16x16ToU8(v).StoreArray((*[16]uint8)(unsafe.Add(pd, x)))
			}
			archsimd.ClearAVXUpperBits()
		}
	}
	if x < n {
		boxInterpolateRow(dst[x:], s0[x:], s1[x:], f, 128)
	}
}

// boxColsCache and boxColsPlan are the arm64 SIMD column filter's table; amd64
// runs boxFilterCols.
type boxColsCache struct{}

type boxColsPlan struct{}

func (*boxColsCache) plan(sw, dw, x, dx int, rows int) *boxColsPlan { return nil }

func (*boxColsPlan) filter(dst, s []uint8) {}
