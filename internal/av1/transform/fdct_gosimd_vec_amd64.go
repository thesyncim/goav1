// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// fwdVec is one 256-bit lane group of the forward-transform networks: eight
// independent 1-D lines per vector (fdct_gosimd_net.go). Only AVX2 operations
// are used; the dispatcher checks archsimd.X86.AVX2 before binding them.
type fwdVec = archsimd.Int32x8

// fwdLanes is the number of 1-D lines a fwdVec carries.
const fwdLanes = 8

func fwdBcast(v int32) fwdVec {
	return archsimd.BroadcastInt32x8(v)
}

// fwdShr is an arithmetic right shift by the constant n in every lane (the
// AVX2 immediate form).
func fwdShr(v fwdVec, n uint64) fwdVec {
	return v.ShiftAllRight(n)
}

// fwdShl is a left shift by the constant n in every lane.
func fwdShl(v fwdVec, n uint64) fwdVec {
	return v.ShiftAllLeft(n)
}

func fwdLoadI32(s []int32) fwdVec {
	return archsimd.LoadInt32x8(s)
}

func fwdStoreI32(s []int32, v fwdVec) {
	v.Store(s)
}

// fwdLoadRes loads the fwdLanes residual samples of column group g (g is a
// multiple of eight, so s starts at the group) and widens them to the column
// pass input scale (shift[0] = 2).
func fwdLoadRes(s []int16, g int) fwdVec {
	return fwdShl(archsimd.LoadInt16x8(s[:8]).ExtendToInt32(), 2)
}

// fwdTranspose writes the transpose of the n x n row-major matrix src into dst
// (dst[c*n+r] = src[r*n+c]); n is a multiple of eight.
func fwdTranspose(dst, src []int32, n int) {
	for bi := 0; bi < n; bi += 8 {
		for bj := 0; bj < n; bj += 8 {
			fwdTransposeBlock8(dst[bj*n+bi:], n, src[bi*n+bj:], n)
		}
	}
}

// fwdTransposeBlock8 transposes the 8x8 block at src (row stride sn) into dst
// (row stride dn). Three AVX2 unpack levels (32-bit, 64-bit, then 128-bit
// lane permutes) put each output column in one register.
func fwdTransposeBlock8(dst []int32, dn int, src []int32, sn int) {
	r0 := archsimd.LoadInt32x8(src[0:])
	r1 := archsimd.LoadInt32x8(src[sn:])
	r2 := archsimd.LoadInt32x8(src[2*sn:])
	r3 := archsimd.LoadInt32x8(src[3*sn:])
	r4 := archsimd.LoadInt32x8(src[4*sn:])
	r5 := archsimd.LoadInt32x8(src[5*sn:])
	r6 := archsimd.LoadInt32x8(src[6*sn:])
	r7 := archsimd.LoadInt32x8(src[7*sn:])

	a0 := r0.InterleaveLoGrouped(r1)
	a1 := r0.InterleaveHiGrouped(r1)
	a2 := r2.InterleaveLoGrouped(r3)
	a3 := r2.InterleaveHiGrouped(r3)
	a4 := r4.InterleaveLoGrouped(r5)
	a5 := r4.InterleaveHiGrouped(r5)
	a6 := r6.InterleaveLoGrouped(r7)
	a7 := r6.InterleaveHiGrouped(r7)

	b0 := fwdI64AsI32x8(fwdI32x8AsI64(a0).InterleaveLoGrouped(fwdI32x8AsI64(a2)))
	b1 := fwdI64AsI32x8(fwdI32x8AsI64(a0).InterleaveHiGrouped(fwdI32x8AsI64(a2)))
	b2 := fwdI64AsI32x8(fwdI32x8AsI64(a1).InterleaveLoGrouped(fwdI32x8AsI64(a3)))
	b3 := fwdI64AsI32x8(fwdI32x8AsI64(a1).InterleaveHiGrouped(fwdI32x8AsI64(a3)))
	b4 := fwdI64AsI32x8(fwdI32x8AsI64(a4).InterleaveLoGrouped(fwdI32x8AsI64(a6)))
	b5 := fwdI64AsI32x8(fwdI32x8AsI64(a4).InterleaveHiGrouped(fwdI32x8AsI64(a6)))
	b6 := fwdI64AsI32x8(fwdI32x8AsI64(a5).InterleaveLoGrouped(fwdI32x8AsI64(a7)))
	b7 := fwdI64AsI32x8(fwdI32x8AsI64(a5).InterleaveHiGrouped(fwdI32x8AsI64(a7)))

	// Output column k is the low 128-bit half of b_k joined with the low half
	// of b_(k+4) (column k, rows 0-7), and likewise for the high halves.
	fwdStoreI32(dst[0*dn:], b0.ConcatPermute128Scalars(0, 2, b4))
	fwdStoreI32(dst[1*dn:], b1.ConcatPermute128Scalars(0, 2, b5))
	fwdStoreI32(dst[2*dn:], b2.ConcatPermute128Scalars(0, 2, b6))
	fwdStoreI32(dst[3*dn:], b3.ConcatPermute128Scalars(0, 2, b7))
	fwdStoreI32(dst[4*dn:], b0.ConcatPermute128Scalars(1, 3, b4))
	fwdStoreI32(dst[5*dn:], b1.ConcatPermute128Scalars(1, 3, b5))
	fwdStoreI32(dst[6*dn:], b2.ConcatPermute128Scalars(1, 3, b6))
	fwdStoreI32(dst[7*dn:], b3.ConcatPermute128Scalars(1, 3, b7))
}

func fwdI32x8AsI64(v fwdVec) archsimd.Int64x4 {
	return v.AsInt64x4()
}

func fwdI64AsI32x8(v archsimd.Int64x4) fwdVec {
	return v.AsInt32x8()
}

// The network drivers have already validated their buffers. Addressing through
// a base pointer keeps per-row bounds checks out of the vector butterflies.
func fwdLoadI32At(s []int32, i int) fwdVec {
	p := unsafe.Add(unsafe.Pointer(unsafe.SliceData(s)), i*4)
	return archsimd.LoadInt32x8Array((*[8]int32)(p))
}

func fwdStoreI32At(s []int32, i int, v fwdVec) {
	p := unsafe.Add(unsafe.Pointer(unsafe.SliceData(s)), i*4)
	v.StoreArray((*[8]int32)(p))
}

func fwdLoadResAt(s []int16, i, g int) fwdVec {
	p := unsafe.Add(unsafe.Pointer(unsafe.SliceData(s)), i*2)
	w := archsimd.LoadInt16x8Array((*[8]int16)(p))
	return fwdShl(w.ExtendToInt32(), 2)
}

// fwdMulAdd computes a*w + acc. The arm64 intrinsic maps to MLA; AVX2
// has no integer multiply-add instruction, so its port uses two operations.
func fwdMulAdd(a, w, acc fwdVec) fwdVec {
	return a.Mul(w).Add(acc)
}

// fwdHalfBtf13V computes the rounded Q13 forward butterfly.
func fwdHalfBtf13V(w0, a, w1, b fwdVec) fwdVec {
	return fwdShr(a.Mul(w0).Add(b.Mul(w1)).Add(fwdRound13), 13)
}

// fwdHalfBtf12V computes the rounded Q12 forward butterfly.
func fwdHalfBtf12V(w0, a, w1, b fwdVec) fwdVec {
	return fwdShr(a.Mul(w0).Add(b.Mul(w1)).Add(fwdRound12), 12)
}
