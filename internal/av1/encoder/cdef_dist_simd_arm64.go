// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"

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
	if uint(sstride) > rowStrideMax || uint(fstride) > rowStrideMax {
		panic(errKernelArgs)
	}
	// Each row pair (i, i+subsampling) is one bounds-checked window: the
	// first row at w[:8], the second at w[sd:].
	sd, fd := subsampling*sstride, subsampling*fstride
	sn, fn := sd+8, fd+8
	pairs := 4 / subsampling
	acc := archsimd.BroadcastUint32x4(0)
	// Rows (i, i+subsampling) form a pair; pairs advance by twice that,
	// so subsampling 4 measures rows 0 and 4 only.
	for _, b := range dlist {
		so := int(b.BY)*8*sstride + int(b.BX)*8
		fo := int(b.BY)*8*fstride + int(b.BX)*8
		for k := 0; k < pairs; k++ {
			o, q := so+2*k*sd, fo+2*k*fd
			sw := src[o : o+sn : o+sn]
			fw := filtered[q : q+fn : q+fn]
			s := pack2((*[8]byte)(sw), (*[8]byte)(sw[sd:]))
			f := pack2((*[8]byte)(fw), (*[8]byte)(fw[fd:]))
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
