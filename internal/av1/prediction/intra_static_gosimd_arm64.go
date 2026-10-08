// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

// Go-native SIMD PAETH and SMOOTH intra static predictors for arm64. Each
// predictor has three kernel shapes that mirror the shapes the NEON asm covered:
//   - 8-bit, width a multiple of 8 (the common case),
//   - 8-bit, width 4 (two rows packed per 8-lane vector, even height only),
//   - 16-bit (high bit depth), width a multiple of 8.
// The routers at the bottom of the file select the kernel for a block shape and
// fall back to the *PureGo reference for everything else. Every kernel is
// byte-identical to the reference (see intra_static_gosimd_arm64_test.go).
//
// Arithmetic bounds that make the narrow lanes exact:
//   - PAETH: base = top+left-topLeft and the abs diffs fit int16 for 8-bit
//     samples; for 12-bit samples they still fit (|base| <= 2*4095).
//   - SMOOTH_V and SMOOTH_H are convex combinations of two 8-bit samples with
//     weights that sum to 256, so the rounded result (w*a + (256-w)*b + 128) is
//     at most 255*256+128 = 65408 and fits uint16 exactly.
//   - SMOOTH (2D) is V+H where V and H are each such convex combinations, so
//     V+H+256 can exceed 16 bits. It is computed as
//     (floor((V+H)/2) + 128) >> 8, with floor((V+H)/2) = (V>>1) + (H>>1) + (V&H&1),
//     which equals (V+H+256)>>9 exactly and stays in uint16.
//   - 16-bit SMOOTH products reach 4095*255 and need int32 lanes.

import (
	"encoding/binary"
	"simd/archsimd"
)

// storeU16x8 writes the eight uint16 lanes of v to dst[0:16], little-endian.
func storeU16x8(dst []byte, v archsimd.Int16x8) {
	r := v.ToBits().ReshapeToUint64s()
	binary.LittleEndian.PutUint64(dst, r.GetElem(0))
	binary.LittleEndian.PutUint64(dst[8:], r.GetElem(1))
}

// paethW4SIMD is the 8-bit kernel for width 4 and even height. Lanes 0..3 hold
// row r and lanes 4..7 hold row r+1, so one vector covers two rows.
func paethW4SIMD(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	height := block.height
	aV := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]}).ConvertToInt16()
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	tl2V := archsimd.BroadcastInt16x8(int16(aboveLeft) << 1)
	for row := 0; row < height; row += 2 {
		lV := archsimd.LoadInt16x8Array(&[8]int16{
			int16(left[row]), int16(left[row]), int16(left[row]), int16(left[row]),
			int16(left[row+1]), int16(left[row+1]), int16(left[row+1]), int16(left[row+1]),
		})
		pTop := lV.Sub(tlV).Abs()
		out := paethInt16(aV, lV, tlV, lV.Sub(tl2V), pTop)
		packed := out.ConvertToUint16().SaturateToUint8().ReshapeToUint32s()
		stride := block.stride
		putU32(block.pix, row*stride, packed.GetElem(0))
		putU32(block.pix, (row+1)*stride, packed.GetElem(1))
	}
}

// smooth8W4SIMD is the 8-bit 2D SMOOTH kernel for width 4 and even height, two
// rows per vector (lanes 0..3 row r, lanes 4..7 row r+1).
func smooth8W4SIMD(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	height := block.height
	aV := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]})
	wWV := archsimd.LoadUint16x8Array(&[8]uint16{weightsW[0], weightsW[1], weightsW[2], weightsW[3], weightsW[0], weightsW[1], weightsW[2], weightsW[3]})
	cwRV := archsimd.BroadcastUint16x8(smoothScale).Sub(wWV).Mul(archsimd.BroadcastUint16x8(rightPred))
	belowV := archsimd.BroadcastUint16x8(belowPred)
	oneV := archsimd.BroadcastUint16x8(1)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	for row := 0; row < height; row += 2 {
		wHV := archsimd.LoadUint16x8Array(&[8]uint16{
			weightsH[row], weightsH[row], weightsH[row], weightsH[row],
			weightsH[row+1], weightsH[row+1], weightsH[row+1], weightsH[row+1],
		})
		lV := archsimd.LoadUint16x8Array(&[8]uint16{
			left[row], left[row], left[row], left[row],
			left[row+1], left[row+1], left[row+1], left[row+1],
		})
		v := wHV.Mul(aV).Add(archsimd.BroadcastUint16x8(smoothScale).Sub(wHV).Mul(belowV))
		h := wWV.Mul(lV).Add(cwRV)
		half := v.Shift(shr1U16).Add(h.Shift(shr1U16)).Add(v.And(h).And(oneV))
		out := half.Add(roundV).Shift(shr8U16).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// smoothVerticalW4SIMD is the 8-bit SMOOTH_V kernel for width 4, even height.
func smoothVerticalW4SIMD(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	height := block.height
	aV := archsimd.LoadUint16x8Array(&[8]uint16{above[0], above[1], above[2], above[3], above[0], above[1], above[2], above[3]})
	belowV := archsimd.BroadcastUint16x8(belowPred)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	for row := 0; row < height; row += 2 {
		wV := archsimd.LoadUint16x8Array(&[8]uint16{
			weights[row], weights[row], weights[row], weights[row],
			weights[row+1], weights[row+1], weights[row+1], weights[row+1],
		})
		cb := archsimd.BroadcastUint16x8(smoothScale).Sub(wV).Mul(belowV).Add(roundV)
		out := wV.Mul(aV).Add(cb).Shift(shr8U16).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// smoothHorizontalW4SIMD is the 8-bit SMOOTH_H kernel for width 4, even height.
func smoothHorizontalW4SIMD(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	height := block.height
	wWV := archsimd.LoadUint16x8Array(&[8]uint16{weights[0], weights[1], weights[2], weights[3], weights[0], weights[1], weights[2], weights[3]})
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	cwRV := archsimd.BroadcastUint16x8(smoothScale).Sub(wWV).Mul(archsimd.BroadcastUint16x8(rightPred)).Add(roundV)
	for row := 0; row < height; row += 2 {
		lV := archsimd.LoadUint16x8Array(&[8]uint16{
			left[row], left[row], left[row], left[row],
			left[row+1], left[row+1], left[row+1], left[row+1],
		})
		out := wWV.Mul(lV).Add(cwRV).Shift(shr8U16).SaturateToUint8().ReshapeToUint32s()
		putU32(block.pix, row*block.stride, out.GetElem(0))
		putU32(block.pix, (row+1)*block.stride, out.GetElem(1))
	}
}

// paeth16SIMD is the 16-bit PAETH kernel for width a multiple of 8. Samples are
// at most 12 bits, so the int16 lanes hold base and the distances exactly.
func paeth16SIMD(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	width, height := block.width, block.height
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		pTop := archsimd.BroadcastInt16x8(absDiffInt16(left[row], aboveLeft))
		lmtV := archsimd.BroadcastInt16x8(int16(left[row]) - int16(aboveLeft)*2)
		dst := block.pix[row*block.stride:][:width*2]
		for col := 0; col < width; col += 8 {
			storeU16x8(dst[col*2:], paethInt16(loadU16x8(above[col:]), lV, tlV, lmtV, pTop))
		}
	}
}

// smooth16Cols holds the per-column 16-bit SMOOTH constants for one 8-column
// block: the weights (as int16 lanes) and the (256-wW)*rightPred products split
// into the low and high four lanes.
type smooth16Cols struct {
	wW    [8]archsimd.Int16x8
	cwRLo [8]archsimd.Int32x4
	cwRHi [8]archsimd.Int32x4
}

// smooth16SIMD is the 16-bit 2D SMOOTH kernel for width a multiple of 8. Each
// product is at most 4095*255, so the arithmetic is done in int32 lanes.
func smooth16SIMD(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	rightV := archsimd.BroadcastInt32x4(int32(rightPred))
	scale32 := archsimd.BroadcastInt32x4(smoothScale)
	belowV := archsimd.BroadcastInt32x4(int32(belowPred))
	round := archsimd.BroadcastInt32x4(256)
	var cols smooth16Cols
	for b := 0; b < blocks; b++ {
		w := loadU16x8(weightsW[b*8:])
		cols.wW[b] = w
		cols.cwRLo[b] = scale32.Sub(w.ExtendLo4ToInt32()).Mul(rightV)
		cols.cwRHi[b] = scale32.Sub(w.HiToLo().ExtendLo4ToInt32()).Mul(rightV)
	}
	for row := 0; row < height; row++ {
		wH := int16(weightsH[row])
		wHV := archsimd.BroadcastInt16x8(wH)
		cbH := scale32.Sub(archsimd.BroadcastInt32x4(int32(wH))).Mul(belowV)
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		dst := block.pix[row*block.stride:][:width*2]
		for b := 0; b < blocks; b++ {
			aV := loadU16x8(above[b*8:])
			vLo := wHV.MulWidenLo(aV).Add(cbH)
			vHi := wHV.MulWidenLo(aV.HiToLo()).Add(cbH)
			hLo := cols.wW[b].MulWidenLo(lV).Add(cols.cwRLo[b])
			hHi := cols.wW[b].HiToLo().MulWidenLo(lV).Add(cols.cwRHi[b])
			loOut := vLo.Add(hLo).Add(round).Shift(shr9I32)
			hiOut := vHi.Add(hHi).Add(round).Shift(shr9I32)
			storeU16x8(dst[b*16:], cflTruncateInt32PairToInt16(loOut, hiOut))
		}
	}
}

// smoothVertical16SIMD is the 16-bit SMOOTH_V kernel for width a multiple of 8.
func smoothVertical16SIMD(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	width, height := block.width, block.height
	scale32 := archsimd.BroadcastInt32x4(smoothScale)
	belowV := archsimd.BroadcastInt32x4(int32(belowPred))
	round := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	for row := 0; row < height; row++ {
		wH := int16(weights[row])
		wHV := archsimd.BroadcastInt16x8(wH)
		cb := scale32.Sub(archsimd.BroadcastInt32x4(int32(wH))).Mul(belowV).Add(round)
		dst := block.pix[row*block.stride:][:width*2]
		for col := 0; col < width; col += 8 {
			aV := loadU16x8(above[col:])
			loOut := wHV.MulWidenLo(aV).Add(cb).Shift(shr8I32)
			hiOut := wHV.MulWidenLo(aV.HiToLo()).Add(cb).Shift(shr8I32)
			storeU16x8(dst[col*2:], cflTruncateInt32PairToInt16(loOut, hiOut))
		}
	}
}

// smoothHorizontal16SIMD is the 16-bit SMOOTH_H kernel for width a multiple of 8.
func smoothHorizontal16SIMD(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	rightV := archsimd.BroadcastInt32x4(int32(rightPred))
	scale32 := archsimd.BroadcastInt32x4(smoothScale)
	round := archsimd.BroadcastInt32x4(1 << (smoothShift1D - 1))
	var wW [8]archsimd.Int16x8
	var cwRLo, cwRHi [8]archsimd.Int32x4
	for b := 0; b < blocks; b++ {
		w := loadU16x8(weights[b*8:])
		wW[b] = w
		cwRLo[b] = scale32.Sub(w.ExtendLo4ToInt32()).Mul(rightV).Add(round)
		cwRHi[b] = scale32.Sub(w.HiToLo().ExtendLo4ToInt32()).Mul(rightV).Add(round)
	}
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		dst := block.pix[row*block.stride:][:width*2]
		for b := 0; b < blocks; b++ {
			loOut := wW[b].MulWidenLo(lV).Add(cwRLo[b]).Shift(shr8I32)
			hiOut := wW[b].HiToLo().MulWidenLo(lV).Add(cwRHi[b]).Shift(shr8I32)
			storeU16x8(dst[b*16:], cflTruncateInt32PairToInt16(loOut, hiOut))
		}
	}
}

// predictPaethSIMD routes a PAETH block to the kernel for its shape; shapes no
// kernel covers use the pure-Go reference.
func predictPaethSIMD(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		paeth8SIMD(block, above, left, aboveLeft)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		paethW4SIMD(block, above, left, aboveLeft)
	case bytesPerSample == 2 && block.width%8 == 0:
		paeth16SIMD(block, above, left, aboveLeft)
	default:
		predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
	}
}

// predictSmoothSIMD routes a 2D SMOOTH block to the kernel for its shape.
func predictSmoothSIMD(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		smooth8SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smooth8W4SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smooth16SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
	default:
		predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
	}
}

// predictSmoothVerticalSIMD routes a SMOOTH_V block to the kernel for its shape.
func predictSmoothVerticalSIMD(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		smoothVertical8SIMD(block, weights, above, belowPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smoothVerticalW4SIMD(block, weights, above, belowPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smoothVertical16SIMD(block, weights, above, belowPred)
	default:
		predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
	}
}

// predictSmoothHorizontalSIMD routes a SMOOTH_H block to the kernel for its shape.
func predictSmoothHorizontalSIMD(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
		smoothHorizontal8SIMD(block, weights, left, rightPred)
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
		smoothHorizontalW4SIMD(block, weights, left, rightPred)
	case bytesPerSample == 2 && block.width%8 == 0:
		smoothHorizontal16SIMD(block, weights, left, rightPred)
	default:
		predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
	}
}
