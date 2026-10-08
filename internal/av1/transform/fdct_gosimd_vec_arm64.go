// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// fwdVec is one 128-bit lane group of the forward-transform networks: four
// independent 1-D lines per vector (fdct_gosimd_net.go).
type fwdVec = archsimd.Int32x4

// fwdLanes is the number of 1-D lines a fwdVec carries.
const fwdLanes = 4

func fwdBcast(v int32) fwdVec {
	return archsimd.BroadcastInt32x4(v)
}

// Per-lane shift amounts. archsimd's immediate ShiftAllRight/ShiftAllLeft
// lower to VSSHL with the amount rebuilt on every use (a move, a duplicate and
// a negate each time); a register-amount Shift with a package-level vector
// keeps one load per use instead.
var (
	fwdShrNeg1  = archsimd.BroadcastInt32x4(-1)
	fwdShrNeg2  = archsimd.BroadcastInt32x4(-2)
	fwdShrNeg4  = archsimd.BroadcastInt32x4(-4)
	fwdShrNeg12 = archsimd.BroadcastInt32x4(-12)
	fwdShrNeg13 = archsimd.BroadcastInt32x4(-13)
	fwdShrNeg31 = archsimd.BroadcastInt32x4(-31)
	fwdShlPos2  = archsimd.BroadcastInt32x4(2)
)

// fwdShr is an arithmetic right shift by the constant n in every lane; the
// networks call it only with the constants listed here.
func fwdShr(v fwdVec, n uint64) fwdVec {
	switch n {
	case 1:
		return v.Shift(fwdShrNeg1)
	case 2:
		return v.Shift(fwdShrNeg2)
	case 4:
		return v.Shift(fwdShrNeg4)
	case 12:
		return v.Shift(fwdShrNeg12)
	case 13:
		return v.Shift(fwdShrNeg13)
	case 31:
		return v.Shift(fwdShrNeg31)
	}
	return v.ShiftAllRight(n)
}

// fwdShl is a left shift by the constant n in every lane.
func fwdShl(v fwdVec, n uint64) fwdVec {
	if n == 2 {
		return v.Shift(fwdShlPos2)
	}
	return v.ShiftAllLeft(n)
}

func fwdLoadI32(s []int32) fwdVec {
	return archsimd.LoadInt32x4(s)
}

func fwdStoreI32(s []int32, v fwdVec) {
	v.Store(s)
}

// fwdLoadRes loads the fwdLanes residual samples of column group g. s starts
// at the 8-column chunk containing g, so one full-width load serves both
// 4-lane halves; a partial load of four samples costs several times as much.
func fwdLoadRes(s []int16, g int) fwdVec {
	w := archsimd.LoadInt16x8Array((*[8]int16)(s))
	if g&4 == 0 {
		return fwdShl(w.ExtendLo4ToInt32(), 2)
	}
	return fwdShl(w.HiToLo().ExtendLo4ToInt32(), 2)
}

// fwdTranspose writes the transpose of the n x n row-major matrix src into dst
// (dst[c*n+r] = src[r*n+c]); n is a multiple of four.
func fwdTranspose(dst, src []int32, n int) {
	for bi := 0; bi < n; bi += 4 {
		for bj := 0; bj < n; bj += 4 {
			fwdTransposeBlock4(dst[bj*n+bi:], n, src[bi*n+bj:], n)
		}
	}
}

// fwdTransposeBlock4 transposes the 4x4 block at src (row stride sn) into dst
// (row stride dn) in registers: interleave the row pairs at 32 bits, then
// combine the pairs at 64 bits.
func fwdTransposeBlock4(dst []int32, dn int, src []int32, sn int) {
	v0 := archsimd.LoadInt32x4(src[0:])
	v1 := archsimd.LoadInt32x4(src[sn:])
	v2 := archsimd.LoadInt32x4(src[2*sn:])
	v3 := archsimd.LoadInt32x4(src[3*sn:])
	e0 := v0.InterleaveLo(v1)
	e1 := v0.InterleaveHi(v1)
	e2 := v2.InterleaveLo(v3)
	e3 := v2.InterleaveHi(v3)
	t0 := fwdI64AsI32(fwdI32AsI64(e0).InterleaveLo(fwdI32AsI64(e2)))
	t1 := fwdI64AsI32(fwdI32AsI64(e0).InterleaveHi(fwdI32AsI64(e2)))
	t2 := fwdI64AsI32(fwdI32AsI64(e1).InterleaveLo(fwdI32AsI64(e3)))
	t3 := fwdI64AsI32(fwdI32AsI64(e1).InterleaveHi(fwdI32AsI64(e3)))
	t0.Store(dst[0:])
	t1.Store(dst[dn:])
	t2.Store(dst[2*dn:])
	t3.Store(dst[3*dn:])
}

func fwdI32AsI64(v fwdVec) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}

func fwdI64AsI32(v archsimd.Int64x2) fwdVec {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}
