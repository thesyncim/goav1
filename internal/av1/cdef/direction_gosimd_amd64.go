// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build goexperiment.simd && amd64 && !purego

// Go-native SIMD CDEF direction search (simd/archsimd, GOEXPERIMENT=simd).
//
// Byte-exact twin of findDirectionScalar. The 8x8
// block is loaded as eight Int16x8 rows (one SIMD load per row), shifted right by
// coeffShift and biased by -128 in-register (matching int32(img>>shift)-128).
//
// All eight AV1 directional partial-sum arrays are built entirely in vector
// registers, exactly like dav1d's AArch64 kernel: each per-row contribution is a
// lane-shifted add, where the lane shift is the byte-lane VEXT exposed here as
// Int16x8.concatShiftRight (archsimd's ConcatShiftBytesRight). Partial-sum
// magnitudes stay in int16 (<= 8*127 = 1016). The finishDirection reduction then
// squares (MulWidenLo/Hi -> int32) and weights against the libaom div-table, and
// the argmax + variance tail matches the scalar `>` scan (lowest index wins ties,
// variance = (best - opposite) >> 10).
//
// Partial-array index layout (matches direction.go):
//
//	p0[k] = sum over i+j=k        (anti-diagonal, k=0..14): row i at lane offset i
//	p4[k] = sum over 7+i-j=k      (diagonal,      k=0..14): rev(row i) at lane i
//	p1[k] = sum over i+(j>>1)=k   (k=0..10): pairwise-sum(row i) at lane i
//	p3[k] = sum over 3+i-(j>>1)=k (k=0..10): rev(pairwise-sum(row i)) at lane i
//	p5[k] = sum over 3-(i>>1)+j=k (k=0..10): row i at lane 3-(i>>1)
//	p7[k] = sum over (i>>1)+j=k   (k=0..10): row i at lane (i>>1)
//	p2[i] = row i sum (straight);  p6[j] = column sum (straight)

package cdef

import (
	"simd/archsimd"
)

func init() {
	if archsimd.X86.AVX2() {
		findDirectionImpl = findDirectionSIMD
		findDirectionDualImpl = findDirectionDualSIMD
		findDirectionU8Impl = findDirectionU8SIMD
		findDirectionDualU8Impl = findDirectionDualU8SIMD
	}
}
