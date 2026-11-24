// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build goexperiment.simd && amd64 && !purego

package cdef

// High-bitdepth CDEF direction search on AVX2, ported from dav1d's
// cdef_dir_16bpc (src/x86/cdef16_avx2.asm): the rows are loaded as words,
// paired with their mirror rows, shifted down to 8 bits by coeffShift and
// handed to the shared cdef_dir body (cdefDirSearchAVX2, dav1d's .main).

import (
	"simd/archsimd"
)

// cdefDirRowPair16 packs the 8 samples at pi and pj as [row i | row j],
// shifted right by coeffShift and biased by -128. The word shift is a dword
// VPSRLVD by a count vector masked back to words (wmask = 0xffff >> shift),
// which keeps a legacy-SSE count move out of the ymm code.
func cdefDirRowPair16(pi, pj *[8]uint16, sh archsimd.Uint32x8, wmask archsimd.Uint16x16, bias archsimd.Int16x16) archsimd.Int16x16 {
	var z archsimd.Uint16x16
	v := z.SetLo(archsimd.LoadUint16x8Array(pi)).SetHi(archsimd.LoadUint16x8Array(pj))
	return v.AsUint32x8().ShiftRight(sh).AsUint16x16().And(wmask).AsInt16x16().Sub(bias)
}

func findDirectionSIMD(img []uint16, stride int, coeffShift int) (int, int32) {
	bias := archsimd.BroadcastInt16x16(128)
	sh := archsimd.BroadcastUint32x8(uint32(coeffShift))
	wmask := archsimd.BroadcastUint16x16(uint16(0xffff >> coeffShift))
	if uint(stride) > 1<<20 {
		panic("cdef: direction stride out of range")
	}
	s := stride
	r := func(i int) *[8]uint16 { return (*[8]uint16)(img[i*s : i*s+8]) }
	m0 := cdefDirRowPair16(r(0), r(7), sh, wmask, bias)
	m1 := cdefDirRowPair16(r(1), r(6), sh, wmask, bias)
	m2 := cdefDirRowPair16(r(2), r(5), sh, wmask, bias)
	m3 := cdefDirRowPair16(r(3), r(4), sh, wmask, bias)
	return cdefDirSearchAVX2(m0, m1, m2, m3)
}

func findDirectionDualSIMD(img1 []uint16, img2 []uint16, stride int, coeffShift int) (int, int32, int, int32) {
	dir1, var1 := findDirectionSIMD(img1, stride, coeffShift)
	dir2, var2 := findDirectionSIMD(img2, stride, coeffShift)
	return dir1, var1, dir2, var2
}
