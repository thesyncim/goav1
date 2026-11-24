// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// The 8x8 hybrid kernels and the IDTX kernel are Go SIMD (fhybrid_idtx and
// fdct_gosimd_net.go).
var forwardBlock8x8ADSTDCTImpl = forwardBlock8x8ADSTDCTSIMDGuarded
var forwardBlock8x8DCTADSTImpl = forwardBlock8x8DCTADSTSIMDGuarded
var forwardBlock8x8ADSTADSTImpl = forwardBlock8x8ADSTADSTSIMDGuarded
var forwardBlock8x8IDTXImpl = forwardBlock8x8IDTXSIMD

func forwardBlock8x8Hybrid8BitResidualTrusted(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, typ Type) bool {
	switch typ {
	case TypeADSTDCT:
		forwardBlock8x8ADSTDCTSIMD(coeff, coeffStride, residual, residualStride, scratch)
	case TypeDCTADST:
		forwardBlock8x8DCTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
	case TypeADSTADST:
		forwardBlock8x8ADSTADSTSIMD(coeff, coeffStride, residual, residualStride, scratch)
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

func idtxStoreInt16x8ToInt32(dst *[8]int32, v archsimd.Int16x8) {
	// Scale after sign extension so every int16 residual preserves its full
	// public value range before being promoted to the coefficient type.
	lo := v.ExtendLo4ToInt32().ShiftAllLeft(3)
	hi := v.HiToLo().ExtendLo4ToInt32().ShiftAllLeft(3)
	lo.StoreArray((*[4]int32)(dst[0:4]))
	hi.StoreArray((*[4]int32)(dst[4:8]))
}

// forwardBlock8x8IDTXSIMD is the IDTX identity scale (coeff[c*stride+r]=res[r][c]<<3).
// The encoder's 8x8 blocks are packed (both strides 8), so that shape reads
// and writes one fixed 64-entry view of each buffer with no row checks; any
// other stride stages its rows through a packed copy.
func forwardBlock8x8IDTXSIMD(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32) {
	_ = scratch[63]
	if residualStride == 8 && coeffStride == 8 {
		forwardBlock8x8IDTXPacked((*[64]int32)(coeff[:64]), (*[64]int16)(residual[:64]))
		return
	}
	var r [64]int16
	var c [64]int32
	for i := range 8 {
		copy(r[i*8:i*8+8], residual[i*residualStride:i*residualStride+8])
	}
	forwardBlock8x8IDTXPacked(&c, &r)
	for i := range 8 {
		copy(coeff[i*coeffStride:i*coeffStride+8], c[i*8:i*8+8])
	}
}

// forwardBlock8x8IDTXPacked is forwardBlock8x8IDTXSIMD on packed 8x8 blocks.
func forwardBlock8x8IDTXPacked(coeff *[64]int32, residual *[64]int16) {
	l0 := archsimd.LoadInt16x8Array((*[8]int16)(residual[0:8]))
	l1 := archsimd.LoadInt16x8Array((*[8]int16)(residual[8:16]))
	l2 := archsimd.LoadInt16x8Array((*[8]int16)(residual[16:24]))
	l3 := archsimd.LoadInt16x8Array((*[8]int16)(residual[24:32]))
	l4 := archsimd.LoadInt16x8Array((*[8]int16)(residual[32:40]))
	l5 := archsimd.LoadInt16x8Array((*[8]int16)(residual[40:48]))
	l6 := archsimd.LoadInt16x8Array((*[8]int16)(residual[48:56]))
	l7 := archsimd.LoadInt16x8Array((*[8]int16)(residual[56:64]))

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

	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[0:8]), o0)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[8:16]), o1)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[16:24]), o2)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[24:32]), o3)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[32:40]), o4)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[40:48]), o5)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[48:56]), o6)
	idtxStoreInt16x8ToInt32((*[8]int32)(coeff[56:64]), o7)
}
