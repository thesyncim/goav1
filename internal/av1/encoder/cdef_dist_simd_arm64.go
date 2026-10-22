// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/cdef"
)

// cdefDist8 is SVT-AV1's svt_aom_compute_cdef_dist_8bit_neon_dotprod
// (ASM_NEON_DOTPROD/cdef_neon_dotprod.c mse_8xn_8bit_neon_dotprod) in Go
// SIMD for the 8x8 luma blocks: rows i and i+subsampling pack into one
// 16-byte vector, |s-d| squares into 16-bit lanes and accumulates into four
// 32-bit lanes (archsimd has no UDOT, so the dot product is widened). Other
// block shapes take the scalar reference.
func cdefDist8(src []byte, sstride int, filtered []byte, fstride int, dlist []cdef.BlockPosition, bwLog2, bhLog2 int, subsampling int) uint64 {
	if bwLog2 != 3 || bhLog2 != 3 || (subsampling != 1 && subsampling != 2 && subsampling != 4) {
		return computeCDEFDist8(src, sstride, filtered, fstride, dlist, bwLog2, bhLog2, subsampling)
	}
	if len(dlist) == 0 {
		return 0
	}
	// Bounds: every block reads up to 7 rows and 7 columns past its origin.
	maxSO, maxFO := 0, 0
	for _, b := range dlist {
		maxSO = max(maxSO, (int(b.BY)*8+7)*sstride+int(b.BX)*8+7)
		maxFO = max(maxFO, (int(b.BY)*8+7)*fstride+int(b.BX)*8+7)
	}
	_ = src[maxSO]
	_ = filtered[maxFO]
	sp := unsafe.Pointer(unsafe.SliceData(src))
	fp := unsafe.Pointer(unsafe.SliceData(filtered))
	acc := archsimd.BroadcastUint32x4(0)
	// Rows (i, i+subsampling) form a pair; pairs advance by twice that,
	// so subsampling 4 measures rows 0 and 4 only.
	pairStep := 2 * subsampling
	for _, b := range dlist {
		so := int(b.BY)*8*sstride + int(b.BX)*8
		fo := int(b.BY)*8*fstride + int(b.BX)*8
		for i := 0; i < 8; i += pairStep {
			s := cdefPackRows(unsafe.Add(sp, so+i*sstride), subsampling*sstride)
			f := cdefPackRows(unsafe.Add(fp, fo+i*fstride), subsampling*fstride)
			d := s.Max(f).Sub(s.Min(f))
			lo := d.ExtendLo8ToUint16()
			hi := d.HiToLo().ExtendLo8ToUint16()
			lo = lo.Mul(lo)
			hi = hi.Mul(hi)
			acc = acc.Add(lo.ExtendLo4ToUint32()).Add(lo.HiToLo().ExtendLo4ToUint32())
			acc = acc.Add(hi.ExtendLo4ToUint32()).Add(hi.HiToLo().ExtendLo4ToUint32())
		}
	}
	return uint64(acc.ReduceSum())
}

// cdefPackRows loads two 8-byte rows (p and p+stride) into one vector.
func cdefPackRows(p unsafe.Pointer, stride int) archsimd.Uint8x16 {
	rows := [2]uint64{*(*uint64)(p), *(*uint64)(unsafe.Add(p, stride))}
	return archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s()
}
