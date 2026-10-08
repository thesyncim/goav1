// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// The measured IDTX Go SIMD kernel replaces its NEON assembly counterpart.
// The three hybrid ADST combinations keep their faster guarded NEON kernels.
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTNEONGuarded
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTNEONGuarded
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTNEONGuarded
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXSIMD

func forwardBlock8x8Hybrid8BitResidualTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, typ Type) bool {
	switch typ {
	case TypeADSTDCT:
		forwardBlock8x8ADSTDCTNEON(coeff, coeffStride, residual, residualStride, scratch)
	case TypeDCTADST:
		forwardBlock8x8DCTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
	case TypeADSTADST:
		forwardBlock8x8ADSTADSTNEON(coeff, coeffStride, residual, residualStride, scratch)
	case TypeIDTX:
		forwardBlock8x8IDTXSIMD(coeff, coeffStride, residual, residualStride, scratch)
	default:
		return false
	}
	return true
}

func adstInt16AsInt32(v archsimd.Int16x8) archsimd.Int32x4 {
	return v.ToBits().ReshapeToUint32s().BitsToInt32()
}
func adstInt32AsInt16(v archsimd.Int32x4) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}
func adstInt16AsInt64(v archsimd.Int16x8) archsimd.Int64x2 {
	return v.ToBits().ReshapeToUint64s().BitsToInt64()
}
func adstInt64AsInt16(v archsimd.Int64x2) archsimd.Int16x8 {
	return v.ToBits().ReshapeToUint16s().BitsToInt16()
}

func idtxStoreInt16x8ToInt32(base unsafe.Pointer, v archsimd.Int16x8) {
	// Scale after sign extension so every int16 residual preserves its full
	// public value range before being promoted to the coefficient type.
	lo := v.ExtendLo4ToInt32().ShiftAllLeft(3)
	hi := v.HiToLo().ExtendLo4ToInt32().ShiftAllLeft(3)
	lo.StoreArray((*[4]int32)(base))
	hi.StoreArray((*[4]int32)(unsafe.Add(base, 16)))
}

// forwardBlock8x8IDTXSIMD is the IDTX identity scale (coeff[c*stride+r]=res[r][c]<<3).
func forwardBlock8x8IDTXSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	rbase := unsafe.Pointer(unsafe.SliceData(residual))
	rstep := uintptr(residualStride) * 2
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(rbase))
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*1)))
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*2)))
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*3)))
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*4)))
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*5)))
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*6)))
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Add(rbase, rstep*7)))

	// Transpose the residual and widen the result to coeff layout on store.
	a0 := l0.InterleaveLo(l1)
	a1 := l0.InterleaveHi(l1)
	a2 := l2.InterleaveLo(l3)
	a3 := l2.InterleaveHi(l3)
	a4 := l4.InterleaveLo(l5)
	a5 := l4.InterleaveHi(l5)
	a6 := l6.InterleaveLo(l7)
	a7 := l6.InterleaveHi(l7)
	b0 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveLo(adstInt16AsInt32(a2)))
	b1 := adstInt32AsInt16(adstInt16AsInt32(a0).InterleaveHi(adstInt16AsInt32(a2)))
	b2 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveLo(adstInt16AsInt32(a3)))
	b3 := adstInt32AsInt16(adstInt16AsInt32(a1).InterleaveHi(adstInt16AsInt32(a3)))
	b4 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveLo(adstInt16AsInt32(a6)))
	b5 := adstInt32AsInt16(adstInt16AsInt32(a4).InterleaveHi(adstInt16AsInt32(a6)))
	b6 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveLo(adstInt16AsInt32(a7)))
	b7 := adstInt32AsInt16(adstInt16AsInt32(a5).InterleaveHi(adstInt16AsInt32(a7)))
	o0 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveLo(adstInt16AsInt64(b4)))
	o1 := adstInt64AsInt16(adstInt16AsInt64(b0).InterleaveHi(adstInt16AsInt64(b4)))
	o2 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveLo(adstInt16AsInt64(b5)))
	o3 := adstInt64AsInt16(adstInt16AsInt64(b1).InterleaveHi(adstInt16AsInt64(b5)))
	o4 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveLo(adstInt16AsInt64(b6)))
	o5 := adstInt64AsInt16(adstInt16AsInt64(b2).InterleaveHi(adstInt16AsInt64(b6)))
	o6 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveLo(adstInt16AsInt64(b7)))
	o7 := adstInt64AsInt16(adstInt16AsInt64(b3).InterleaveHi(adstInt16AsInt64(b7)))

	obase := unsafe.Pointer(unsafe.SliceData(coeff))
	ostep := uintptr(coeffStride) * 4
	idtxStoreInt16x8ToInt32(obase, o0)
	idtxStoreInt16x8ToInt32(unsafe.Add(obase, ostep*1), o1)
	idtxStoreInt16x8ToInt32(unsafe.Add(obase, ostep*2), o2)
	idtxStoreInt16x8ToInt32(unsafe.Add(obase, ostep*3), o3)
	idtxStoreInt16x8ToInt32(unsafe.Add(obase, ostep*4), o4)
	idtxStoreInt16x8ToInt32(unsafe.Add(obase, ostep*5), o5)
	idtxStoreInt16x8ToInt32(unsafe.Add(obase, ostep*6), o6)
	idtxStoreInt16x8ToInt32(unsafe.Add(obase, ostep*7), o7)
}
