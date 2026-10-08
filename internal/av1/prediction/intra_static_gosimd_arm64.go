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
// For valid 8-bit edges and smooth weights, pred is non-negative and at most
// 130560. Adding the divideRound bias therefore cannot overflow int32; one
// arithmetic shift rounds the full or 1D sum before packing. Results are <=255
// and the low 64 bits are written with a single FMOV+STR. No slices in the hot
// loop: every load/store is an array pointer, with no bounds checks or escapes.

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

// store8Smooth adds the rounded-division bias, shifts two int32 accumulators
// (lo=lanes 0..3, hi=lanes 4..7), narrows to int16, saturates to uint8 and
// writes the 8 pixels at p. The valid 8-bit smooth sum is non-negative and
// bounded, so adding bias cannot overflow int32.
func store8Smooth(p unsafe.Pointer, lo, hi archsimd.Int32x4, bias, shift archsimd.Int32x4) {
	loRound := lo.Add(bias).Shift(shift)
	hiRound := hi.Add(bias).Shift(shift)
	v := cflTruncateInt32PairToInt16(loRound, hiRound).SaturateToUint8()
	*(*float64)(p) = v.ReshapeToUint64s().BitsToFloat64().GetElem(0)
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
	baseV := archsimd.BroadcastInt32x4(int32(belowPred+rightPred) * smoothScale)
	roundBiasV := archsimd.BroadcastInt32x4(1 << (smoothShiftFull - 1))
	roundShiftV := archsimd.BroadcastInt32x4(-smoothShiftFull)

	dbase := unsafe.Pointer(&block.pix[0])
	abase := unsafe.Pointer(&above[0])
	wbase := unsafe.Pointer(&weightsW[0])

	for row := 0; row < height; row++ {
		wH := int16(weightsH[row])
		wHV := archsimd.BroadcastInt16x8(wH)
		leftV := archsimd.BroadcastInt16x8(int16(left[row]))
		leftRightV := leftV.Sub(rightV)
		leftRightHi := leftRightV.HiToLo()
		wHHi := wHV.HiToLo()
		dp := unsafe.Add(dbase, row*block.stride)
		ap := abase
		wp := wbase
		for col := 0; col < width; col += 8 {
			aV := loadPixV8(ap)
			wWV := loadWeightV8(wp)
			// Factor each complementary-weight pair: a*w + b*(256-w) =
			// 256*b + (a-b)*w. This keeps the exact blend while halving the
			// widening multiplies per output pixel.
			lo := baseV.Add(aV.Sub(belowV).MulWidenLo(wHV)).
				Add(leftRightV.MulWidenLo(wWV))
			ahi, wwhi := aV.HiToLo(), wWV.HiToLo()
			bhi := belowV.HiToLo()
			hi := baseV.Add(ahi.Sub(bhi).MulWidenLo(wHHi)).
				Add(leftRightHi.MulWidenLo(wwhi))
			store8Smooth(dp, lo, hi, roundBiasV, roundShiftV)
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
	baseV := archsimd.BroadcastInt32x4(int32(belowPred) * smoothScale)
	roundBiasV := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	roundShiftV := archsimd.BroadcastInt32x4(-smoothShift1D)

	dbase := unsafe.Pointer(&block.pix[0])
	abase := unsafe.Pointer(&above[0])
	for row := 0; row < height; row++ {
		w := int16(weights[row])
		wV := archsimd.BroadcastInt16x8(w)
		wHi := wV.HiToLo()
		dp := unsafe.Add(dbase, row*block.stride)
		ap := abase
		for col := 0; col < width; col += 8 {
			aV := loadPixV8(ap)
			lo := baseV.Add(aV.Sub(belowV).MulWidenLo(wV))
			hi := baseV.Add(aV.HiToLo().Sub(belowV.HiToLo()).MulWidenLo(wHi))
			store8Smooth(dp, lo, hi, roundBiasV, roundShiftV)
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
	baseV := archsimd.BroadcastInt32x4(int32(rightPred) * smoothScale)
	roundBiasV := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	roundShiftV := archsimd.BroadcastInt32x4(-smoothShift1D)

	dbase := unsafe.Pointer(&block.pix[0])
	wbase := unsafe.Pointer(&weights[0])
	for row := 0; row < height; row++ {
		leftV := archsimd.BroadcastInt16x8(int16(left[row]))
		leftRightV := leftV.Sub(rightV)
		leftRightHi := leftRightV.HiToLo()
		dp := unsafe.Add(dbase, row*block.stride)
		wp := wbase
		for col := 0; col < width; col += 8 {
			wV := loadWeightV8(wp)
			lo := baseV.Add(leftRightV.MulWidenLo(wV))
			hi := baseV.Add(leftRightHi.MulWidenLo(wV.HiToLo()))
			store8Smooth(dp, lo, hi, roundBiasV, roundShiftV)
			if col+8 < width {
				wp = unsafe.Add(wp, 8*elem)
				dp = unsafe.Add(dp, 8)
			}
		}
	}
}
