// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import "simd/archsimd"

// 8-bit width-4 PAETH, SMOOTH, SMOOTH_V, SMOOTH_H and CfL apply for amd64
// (dav1d's ipred_*_8bpc .w4 paths in ipred_avx2.asm): lanes 0..3 hold row r
// and lanes 4..7 row r+1, so one vector covers two rows. They are the arm64
// width-4 kernels on the shared helper layer, with the same arithmetic
// bounds (see intra_gosimd_shared.go); every height is even.

func w4Rows(a, b uint16) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array(&[8]uint16{a, a, a, a, b, b, b, b})
}

func w4Cols(v []uint16) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array(&[8]uint16{v[0], v[1], v[2], v[3], v[0], v[1], v[2], v[3]})
}

// w4Store writes the two 4-byte rows held in v's low eight byte lanes.
func w4Store(block planeBlock, row int, v archsimd.Uint16x8) {
	packed := narrowU16ToU8(v).ReshapeToUint32s()
	putU32(block.pix, row*block.stride, packed.GetElem(0))
	putU32(block.pix, (row+1)*block.stride, packed.GetElem(1))
}

func paethW4AVX(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	aV := w4Cols(above).ConvertToInt16()
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	tl2V := archsimd.BroadcastInt16x8(int16(aboveLeft) << 1)
	for row := 0; row < block.height; row += 2 {
		lV := w4Rows(left[row], left[row+1]).ConvertToInt16()
		pTop := lV.Sub(tlV).Abs()
		w4Store(block, row, paethInt16(aV, lV, tlV, lV.Sub(tl2V), pTop).ConvertToUint16())
	}
}

func smoothW4AVX(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	aV := w4Cols(above)
	wWV := w4Cols(weightsW)
	scaleV := archsimd.BroadcastUint16x8(smoothScale)
	cwRV := scaleV.Sub(wWV).Mul(archsimd.BroadcastUint16x8(rightPred))
	belowV := archsimd.BroadcastUint16x8(belowPred)
	oneV := archsimd.BroadcastUint16x8(1)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	for row := 0; row < block.height; row += 2 {
		wHV := w4Rows(weightsH[row], weightsH[row+1])
		lV := w4Rows(left[row], left[row+1])
		v := wHV.Mul(aV).Add(scaleV.Sub(wHV).Mul(belowV))
		h := wWV.Mul(lV).Add(cwRV)
		half := shrU16_1(v).Add(shrU16_1(h)).Add(v.And(h).And(oneV))
		w4Store(block, row, shrU16_8(half.Add(roundV)))
	}
}

func smoothVerticalW4AVX(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	aV := w4Cols(above)
	scaleV := archsimd.BroadcastUint16x8(smoothScale)
	belowV := archsimd.BroadcastUint16x8(belowPred)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	for row := 0; row < block.height; row += 2 {
		wV := w4Rows(weights[row], weights[row+1])
		cb := scaleV.Sub(wV).Mul(belowV).Add(roundV)
		w4Store(block, row, shrU16_8(wV.Mul(aV).Add(cb)))
	}
}

func smoothHorizontalW4AVX(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	wWV := w4Cols(weights)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	cwRV := archsimd.BroadcastUint16x8(smoothScale).Sub(wWV).Mul(archsimd.BroadcastUint16x8(rightPred)).Add(roundV)
	for row := 0; row < block.height; row += 2 {
		lV := w4Rows(left[row], left[row+1])
		w4Store(block, row, shrU16_8(wWV.Mul(lV).Add(cwRV)))
	}
}

func applyCFL4AVX(block planeBlock, visibleHeight int, acQ3 []int16, alphaQ3 int) {
	alphaV := archsimd.BroadcastInt16x8(int16(alphaQ3))
	zeroV := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(0xff)
	for row := 0; row < visibleHeight; row += 2 {
		ac0 := acQ3[row*CFLBufLine:]
		ac1 := acQ3[(row+1)*CFLBufLine:]
		v := archsimd.LoadInt16x8Array(&[8]int16{ac0[0], ac0[1], ac0[2], ac0[3], ac1[0], ac1[1], ac1[2], ac1[3]})
		loR := cflRound6(mulWidenLo16(alphaV, v))
		hiR := cflRound6(mulWidenLo16(alphaV, hiToLoI16(v)))
		l0 := block.pix[row*block.stride:]
		l1 := block.pix[(row+1)*block.stride:]
		cur := archsimd.LoadInt16x8Array(&[8]int16{
			int16(l0[0]), int16(l0[1]), int16(l0[2]), int16(l0[3]),
			int16(l1[0]), int16(l1[1]), int16(l1[2]), int16(l1[3]),
		})
		lo := cur.ExtendLo4ToInt32().Add(loR)
		hi := hiToLoI16(cur).ExtendLo4ToInt32().Add(hiR)
		w4Store(block, row, cflClampInt32Pair(lo, hi, zeroV, maxV).ConvertToUint16())
	}
}
