// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

// Go-native SIMD PAETH and SMOOTH intra static predictors. Each kernel handles
// the common 8-bit path (bytesPerSample==1) with width a multiple of 8 (every
// AV1 intra block width other than 4); widths of 4 and all high-bit-depth blocks
// fall back to the *PureGo scalar reference. Byte-identical to the references
// (TestPaethSIMDMatchesPureGo / TestSmooth*SIMDMatchesPureGo).
//
// PAETH (predictPaethSIMD): eight columns per Int16x8. Per libaom's
// paethPredictorSingle the three distances collapse to
//   pLeft    = |above - aboveLeft|
//   pTop     = |left  - aboveLeft|
//   pTopLeft = |above + left - 2*aboveLeft| = |base - aboveLeft|, base=above+left-aboveLeft
// (left and aboveLeft are the per-row/per-block scalars, broadcast once). All
// operands fit int16 (0..255 inputs => base in [-255,510], diffs in [0,510]).
// The tie-break order left -> top -> topLeft is reproduced with lane compares
// and branchless IfElse selects in the exact scalar precedence:
//   out = topLeft; out = IfElse(pTop<=pTopLeft, top, out);
//   out = IfElse(pLeft<=pTop && pLeft<=pTopLeft, left, out).
//
// SMOOTH (predictSmoothSIMD and the two 1D variants): the weighted blend
//   pred = wH*above + (256-wH)*below + wW*left + (256-wW)*right
// is a fused widening multiply-accumulate of int16 pixels by int16 weights into
// int32 lanes (SMLAL / SMLAL2 via widened multiplies and int32 adds). All products fit
// (255*255=65025) and the 4-term sum <= 130560 for real pixels, so it never
// overflows int32 and matches divideRound(pred, 9). The full predictor round-
// shifts by 1+smoothWeightLog2Scale (9); the 1D variants drop the unused term
// pair and shift by smoothWeightLog2Scale (8).
//
// The divideRound + narrow rounds each int32 lane with the equivalent
// shift/add/low-bit sequence, then packs the two halves. pred is non-negative,
// the results (<=255) do not saturate, and the low 64 bits are written with a
// single FMOV+STR. No slices in the hot loop: every load/store is an array
// pointer, so there are no bounds checks and nothing escapes to the heap.

package prediction

import (
	"simd/archsimd"
	"unsafe"
)

// smoothShiftFull is the divideRound bit count for the full SMOOTH predictor
// (1 + smoothWeightLog2Scale = 9); smoothShift1D is the count for SMOOTH_V/H (8).
const smoothShiftFull = 1 + smoothWeightLog2Scale
const smoothShift1D = smoothWeightLog2Scale

// smoothScale is the fixed-point weight scale 1<<smoothWeightLog2Scale (256).
const smoothScale = 1 << smoothWeightLog2Scale

// loadPixV8 loads 8 uint16 pixels at p as an Int16x8. The samples are 8-bit
// (0..255), so the bit pattern is a non-negative int16 and int16 multiply
// reproduces the reference's uint16-widened product exactly.
func loadPixV8(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(p)).ConvertToInt16()
}

// loadWeightV8 loads 8 uint16 weights at p as an Int16x8 (weights are 0..255).
func loadWeightV8(p unsafe.Pointer) archsimd.Int16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(p)).ConvertToInt16()
}

// store8Smooth round-shifts two int32 accumulators (lo=lanes 0..3, hi=lanes
// 4..7) right by shift with rounding, narrows to int16 (VSQRSHRN/VSQRSHRN2),
// saturates to uint8 and writes the 8 pixels at p (one 64-bit store). pred is
// non-negative and the results are <=255, so the rounding round-shift equals
// divideRound(pred, shift) and neither the int16 nor the uint8 saturation ever
// clips. shift must be a compile-time constant so the shift lowers to an
// immediate VSQRSHRN.
func store8Smooth(p unsafe.Pointer, lo, hi archsimd.Int32x4, shift uint8) {
	loRound := smoothRoundShiftInt32x4(lo, shift)
	hiRound := smoothRoundShiftInt32x4(hi, shift)
	v := cflTruncateInt32PairToInt16(loRound, hiRound).SaturateToUint8()
	*(*float64)(p) = v.ReshapeToUint64s().BitsToFloat64().GetElem(0)
}

// smoothRoundShiftInt32x4 computes the rounded arithmetic shift without adding
// a rounding bias to the accumulator, which could overflow at the int32 limit.
func smoothRoundShiftInt32x4(v archsimd.Int32x4, shift uint8) archsimd.Int32x4 {
	if shift == 0 {
		return v
	}
	s := uint64(shift)
	roundBit := v.ShiftAllRight(s - 1).And(archsimd.BroadcastInt32x4(1))
	return v.ShiftAllRight(s).Add(roundBit)
}

// store8Int16Pix writes 8 int16 pixels (each in [0,255]) as 8 uint8 at p.
func store8Int16Pix(p unsafe.Pointer, v archsimd.Int16x8) {
	*(*float64)(p) = v.SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)
}

// predictPaethSIMD is the Go-native SIMD form of predictPaethPureGo. Byte
// identical to it; the width-4 and high-bit-depth shapes fall back to it.
func predictPaethSIMD(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
		return
	}
	width := block.width
	height := block.height
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	tl2V := archsimd.BroadcastInt16x8(int16(aboveLeft) << 1) // 2*aboveLeft

	const elem = 2 // sizeof(uint16)
	dbase := unsafe.Pointer(&block.pix[0])
	abase := unsafe.Pointer(&above[0])
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		pTop := lV.Sub(tlV).Abs()    // |left - aboveLeft|, constant across the row
		leftMinus2tl := lV.Sub(tl2V) // left - 2*aboveLeft
		dp := unsafe.Add(dbase, row*block.stride)
		ap := abase
		for col := 0; col < width; col += 8 {
			aV := loadPixV8(ap)
			pLeft := aV.Sub(tlV).Abs()             // |above - aboveLeft|
			pTopLeft := aV.Add(leftMinus2tl).Abs() // |above + left - 2*aboveLeft|
			out := tlV                             // default: topLeft
			out = aV.IfElse(pTop.LessEqual(pTopLeft), out)
			leftMask := pLeft.LessEqual(pTop).And(pLeft.LessEqual(pTopLeft))
			out = lV.IfElse(leftMask, out)
			store8Int16Pix(dp, out)
			if col+8 < width {
				ap = unsafe.Add(ap, 8*elem)
				dp = unsafe.Add(dp, 8)
			}
		}
	}
}

// predictSmoothSIMD is the Go-native SIMD form of predictSmoothPureGo.
func predictSmoothSIMD(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
		return
	}
	width := block.width
	height := block.height
	const elem = 2

	belowV := archsimd.BroadcastInt16x8(int16(belowPred))
	rightV := archsimd.BroadcastInt16x8(int16(rightPred))
	scaleV := archsimd.BroadcastInt16x8(smoothScale)

	dbase := unsafe.Pointer(&block.pix[0])
	abase := unsafe.Pointer(&above[0])
	wbase := unsafe.Pointer(&weightsW[0])

	for row := 0; row < height; row++ {
		wH := int16(weightsH[row])
		wHV := archsimd.BroadcastInt16x8(wH)
		wHcV := archsimd.BroadcastInt16x8(smoothScale - wH)
		leftV := archsimd.BroadcastInt16x8(int16(left[row]))
		dp := unsafe.Add(dbase, row*block.stride)
		ap := abase
		wp := wbase
		for col := 0; col < width; col += 8 {
			aV := loadPixV8(ap)
			wWV := loadWeightV8(wp)
			wWcV := scaleV.Sub(wWV)
			// lo/hi accumulate columns 0..3 / 4..7 with SMLAL / SMLAL2.
			lo := aV.MulWidenLo(wHV).Add(belowV.MulWidenLo(wHcV)).
				Add(leftV.MulWidenLo(wWV)).Add(rightV.MulWidenLo(wWcV))
			ahi, whi := aV.HiToLo(), wHV.HiToLo()
			bhi, wchi := belowV.HiToLo(), wHcV.HiToLo()
			lhi, wwhi := leftV.HiToLo(), wWV.HiToLo()
			rhi, wwchi := rightV.HiToLo(), wWcV.HiToLo()
			hi := ahi.MulWidenLo(whi).Add(bhi.MulWidenLo(wchi)).
				Add(lhi.MulWidenLo(wwhi)).Add(rhi.MulWidenLo(wwchi))
			store8Smooth(dp, lo, hi, smoothShiftFull)
			if col+8 < width {
				ap = unsafe.Add(ap, 8*elem)
				wp = unsafe.Add(wp, 8*elem)
				dp = unsafe.Add(dp, 8)
			}
		}
	}
}

// predictSmoothVerticalSIMD is the Go-native SIMD form of
// predictSmoothVerticalPureGo: pred = w*above + (scale-w)*below, w per row.
func predictSmoothVerticalSIMD(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
		return
	}
	width := block.width
	height := block.height
	const elem = 2

	belowV := archsimd.BroadcastInt16x8(int16(belowPred))

	dbase := unsafe.Pointer(&block.pix[0])
	abase := unsafe.Pointer(&above[0])
	for row := 0; row < height; row++ {
		w := int16(weights[row])
		wV := archsimd.BroadcastInt16x8(w)
		wcV := archsimd.BroadcastInt16x8(smoothScale - w)
		dp := unsafe.Add(dbase, row*block.stride)
		ap := abase
		for col := 0; col < width; col += 8 {
			aV := loadPixV8(ap)
			lo := aV.MulWidenLo(wV).Add(belowV.MulWidenLo(wcV))
			hi := aV.HiToLo().MulWidenLo(wV.HiToLo()).Add(belowV.HiToLo().MulWidenLo(wcV.HiToLo()))
			store8Smooth(dp, lo, hi, smoothShift1D)
			if col+8 < width {
				ap = unsafe.Add(ap, 8*elem)
				dp = unsafe.Add(dp, 8)
			}
		}
	}
}

// predictSmoothHorizontalSIMD is the Go-native SIMD form of
// predictSmoothHorizontalPureGo: pred = w*left + (scale-w)*right, w per column,
// left the per-row scalar.
func predictSmoothHorizontalSIMD(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
		return
	}
	width := block.width
	height := block.height
	const elem = 2

	rightV := archsimd.BroadcastInt16x8(int16(rightPred))
	scaleV := archsimd.BroadcastInt16x8(smoothScale)

	dbase := unsafe.Pointer(&block.pix[0])
	wbase := unsafe.Pointer(&weights[0])
	for row := 0; row < height; row++ {
		leftV := archsimd.BroadcastInt16x8(int16(left[row]))
		dp := unsafe.Add(dbase, row*block.stride)
		wp := wbase
		for col := 0; col < width; col += 8 {
			wV := loadWeightV8(wp)
			wcV := scaleV.Sub(wV)
			lo := leftV.MulWidenLo(wV).Add(rightV.MulWidenLo(wcV))
			hi := leftV.HiToLo().MulWidenLo(wV.HiToLo()).Add(rightV.HiToLo().MulWidenLo(wcV.HiToLo()))
			store8Smooth(dp, lo, hi, smoothShift1D)
			if col+8 < width {
				wp = unsafe.Add(wp, 8*elem)
				dp = unsafe.Add(dp, 8)
			}
		}
	}
}
