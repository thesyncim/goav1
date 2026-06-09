// Ported from libaom: av1/encoder/hybrid_fwd_txfm.c (av1_fwht4x4_c) with
// UNIT_QUANT_FACTOR from aom_dsp/txfm_common.h.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// unitQuantFactor is UNIT_QUANT_FACTOR == (1 << UNIT_QUANT_SHIFT). The forward
// WHT scales by it; InverseWHT4x4Block undoes it with >> unitQuantShift.
const unitQuantFactor = 1 << unitQuantShift

// ForwardWHT4x4 writes AV1's lossless 4x4 forward Walsh-Hadamard transform of a
// residual block into coeff, the exact inverse of InverseWHT4x4Block. residual
// is read as residual[row*residualStride + col]; coeff is written in the AV1
// coefficient layout that InverseWHT4x4Block consumes (coeff[row*coeffStride +
// col]), so ForwardWHT4x4 followed by InverseWHT4x4Block is an identity for any
// int16 residual.
//
// This is selected by encoding for lossless DCT_DCT 4x4 blocks; WHT is not a
// bitstream tx_type. The butterfly locals are int32 (not libaom's int64
// tran_high_t): an int16 residual makes the largest coefficient magnitude
// ~16*32767 < 2^31, matching the int32 width InverseWHT4x4Block already uses.
func ForwardWHT4x4(coeff []int32, coeffStride int, residual []int16, residualStride int) error {
	if coeffStride < 4 ||
		residualStride < 4 ||
		!blockFits(len(residual), residualStride, 4, 4) ||
		!coeffBlockFits(len(coeff), coeffStride, 4, 4) {
		return ErrInvalidTransform
	}

	// op[m*4+col] mirrors the flat libaom output buffer; pass 1 rewrites it in
	// place exactly as av1_fwht4x4_c does before the strided scatter to coeff.
	var op [16]int32

	// Pass 0: vertical butterfly down each input column, no scaling.
	for i := range 4 {
		a1 := int32(residual[0*residualStride+i])
		b1 := int32(residual[1*residualStride+i])
		c1 := int32(residual[2*residualStride+i])
		d1 := int32(residual[3*residualStride+i])

		a1 += b1
		d1 -= c1
		e1 := (a1 - d1) >> 1
		b1 = e1 - b1
		c1 = e1 - c1
		a1 -= c1
		d1 += b1

		op[i*4+0] = a1
		op[i*4+1] = c1
		op[i*4+2] = d1
		op[i*4+3] = b1
	}

	// Pass 1: butterfly across the other dimension, scaled by UNIT_QUANT_FACTOR.
	for i := range 4 {
		a1 := op[4*0+i]
		b1 := op[4*1+i]
		c1 := op[4*2+i]
		d1 := op[4*3+i]

		a1 += b1
		d1 -= c1
		e1 := (a1 - d1) >> 1
		b1 = e1 - b1
		c1 = e1 - c1
		a1 -= c1
		d1 += b1

		op[4*0+i] = a1 * unitQuantFactor
		op[4*1+i] = c1 * unitQuantFactor
		op[4*2+i] = d1 * unitQuantFactor
		op[4*3+i] = b1 * unitQuantFactor
	}

	// Scatter the flat output into the strided coefficient block. With
	// coeffStride == 4 this is the identity layout InverseWHT4x4Block reads.
	for m := range 4 {
		for i := range 4 {
			coeff[m*coeffStride+i] = op[m*4+i]
		}
	}
	return nil
}
