// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"simd/archsimd"
	"unsafe"
)

// Go-native SIMD CfL kernels. They mirror the scalar contracts in
// intra_kernels.go:
//   - subsampleLuma8SIMD reproduces the lbd Q3 reductions with UXTL/USHLL and
//     UZP even/odd pairing.
//   - subsampleLuma16SIMD preserves the hbd max-sample validation, then performs
//     the same Q3 reductions in uint16 lanes.
//   - subtractCFLAverageSIMD computes the rounded average in uint32 lanes and
//     truncates the final signed difference exactly like int16(int(src)-avg).
//   - applyCFLSIMD uses the same abs-round-conditional-negate signed rounding as
//     the hand NEON asm, then clamps through SQXTUN for 8-bit and min/max for
//     high bit depth.

func subsampleLuma8SIMD(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
	switch {
	case subX && subY:
		if outW < 8 || outW%8 != 0 {
			subsampleLuma8NEON(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < height; row += 2 {
			outBase := (row >> 1) * CFLBufLine
			topBase := row * inputStride
			botBase := (row + 1) * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				top := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&input[topBase+ic])))
				bot := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&input[botBase+ic])))
				sum := cflPairSumUint8x16(top).Add(cflPairSumUint8x16(bot)).ShiftAllLeft(1)
				sum.StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+oc])))
			}
		}
	case subX:
		if outW < 8 || outW%8 != 0 {
			subsampleLuma8NEON(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				v := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&input[inBase+ic])))
				sum := cflPairSumUint8x16(v).ShiftAllLeft(2)
				sum.StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+oc])))
			}
		}
	default:
		if outW < 16 || outW%16 != 0 {
			subsampleLuma8NEON(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for col := 0; col < outW; col += 16 {
				v := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&input[inBase+col])))
				v.ExtendLo8ShlToUint16(3).StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+col])))
				v.ExtendHi8ShlToUint16(3).StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+col+8])))
			}
		}
	}
}

func subsampleLuma16SIMD(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool, max uint16) error {
	maxV := archsimd.BroadcastUint16x8(max)
	switch {
	case subX && subY:
		if outW < 8 || outW%8 != 0 {
			return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
		}
		for row := 0; row < height; row += 2 {
			outBase := (row >> 1) * CFLBufLine
			topBase := row * inputStride
			botBase := (row + 1) * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				top0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&input[topBase+ic])))
				top1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&input[topBase+ic+8])))
				bot0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&input[botBase+ic])))
				bot1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&input[botBase+ic+8])))
				if cflUint16AnyAbove(top0, maxV) || cflUint16AnyAbove(top1, maxV) ||
					cflUint16AnyAbove(bot0, maxV) || cflUint16AnyAbove(bot1, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				sum := cflPairSumUint16x16(top0, top1).Add(cflPairSumUint16x16(bot0, bot1)).ShiftAllLeft(1)
				sum.StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+oc])))
			}
		}
	case subX:
		if outW < 8 || outW%8 != 0 {
			return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				in0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&input[inBase+ic])))
				in1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&input[inBase+ic+8])))
				if cflUint16AnyAbove(in0, maxV) || cflUint16AnyAbove(in1, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				sum := cflPairSumUint16x16(in0, in1).ShiftAllLeft(2)
				sum.StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+oc])))
			}
		}
	default:
		if outW < 8 || outW%8 != 0 {
			return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for col := 0; col < outW; col += 8 {
				v := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&input[inBase+col])))
				if cflUint16AnyAbove(v, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				v.ShiftAllLeft(3).StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+col])))
			}
		}
	}
	return nil
}

func subtractCFLAverageSIMD(srcQ3 []uint16, dstQ3 []int16, width int, height int, numPelLog2 int) {
	if width < 8 || width%8 != 0 {
		subtractCFLAveragePureGo(srcQ3, dstQ3, width, height, numPelLog2)
		return
	}
	acc := archsimd.BroadcastUint32x4(0)
	for row := 0; row < height; row++ {
		base := row * CFLBufLine
		for col := 0; col < width; col += 8 {
			v := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&srcQ3[base+col])))
			acc = acc.Add(v.ExtendLo4ToUint32()).Add(v.ExtendHi4ToUint32())
		}
	}
	sum := int(acc.ReduceSum()) + ((width * height) >> 1)
	avg := sum >> numPelLog2
	avgV := archsimd.BroadcastInt32x4(int32(avg))
	for row := 0; row < height; row++ {
		base := row * CFLBufLine
		for col := 0; col < width; col += 8 {
			v := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&srcQ3[base+col])))
			lo := v.ExtendLo4ToUint32().ConvertToInt32().Sub(avgV)
			hi := v.ExtendHi4ToUint32().ConvertToInt32().Sub(avgV)
			lo.TruncToInt16().TruncToInt16Hi(hi).StoreArray((*[8]int16)(unsafe.Pointer(&dstQ3[base+col])))
		}
	}
}

func applyCFLSIMD(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	switch {
	case bytesPerSample == 1 && max == 0xff && visibleWidth >= 16 && visibleWidth%16 == 0:
		alpha := archsimd.BroadcastInt16x8(int16(alphaQ3))
		round := archsimd.BroadcastInt32x4(32)
		for row := 0; row < visibleHeight; row++ {
			dstBase := row * block.stride
			acBase := row * CFLBufLine
			for col := 0; col < visibleWidth; col += 16 {
				dv := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&block.pix[dstBase+col])))
				dLo := dv.ExtendLo8ToUint16().ConvertToInt16()
				dHi := dv.ExtendHi8ToUint16().ConvertToInt16()
				acLo := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&acQ3[acBase+col])))
				acHi := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&acQ3[acBase+col+8])))
				outLo := dLo.AddSaturated(cflScaleQ3SIMD(acLo, alpha, round)).SaturateToUint8()
				outHi := dHi.AddSaturated(cflScaleQ3SIMD(acHi, alpha, round))
				outLo.SaturateToUint8Hi(outHi).StoreArray((*[16]uint8)(unsafe.Pointer(&block.pix[dstBase+col])))
			}
		}
	case bytesPerSample == 2 && visibleWidth >= 8 && visibleWidth%8 == 0:
		alpha := archsimd.BroadcastInt16x8(int16(alphaQ3))
		round := archsimd.BroadcastInt32x4(32)
		zero := archsimd.BroadcastInt32x4(0)
		maxV := archsimd.BroadcastInt32x4(int32(max))
		for row := 0; row < visibleHeight; row++ {
			dstBase := row * block.stride
			acBase := row * CFLBufLine
			for col := 0; col < visibleWidth; col += 8 {
				dstOff := dstBase + col*2
				dv := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&block.pix[dstOff])))
				ac := archsimd.LoadInt16x8Array((*[8]int16)(unsafe.Pointer(&acQ3[acBase+col])))
				scaled := cflScaleQ3SIMD(ac, alpha, round)
				outLo := dv.ExtendLo4ToUint32().ConvertToInt32().Add(scaled.ExtendLo4ToInt32()).Max(zero).Min(maxV)
				outHi := dv.ExtendHi4ToUint32().ConvertToInt32().Add(scaled.HiToLo().ExtendLo4ToInt32()).Max(zero).Min(maxV)
				out := outLo.SaturateToUint16().SaturateToUint16Hi(outHi)
				out.StoreArray((*[8]uint16)(unsafe.Pointer(&block.pix[dstOff])))
			}
		}
	default:
		if bytesPerSample == 1 && max == 0xff {
			applyCFLNEON(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
			return
		}
		applyCFLPureGo(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
	}
}

func cflPairSumUint8x16(v archsimd.Uint8x16) archsimd.Uint16x8 {
	return v.ConcatEven(v).ExtendLo8ToUint16().Add(v.ConcatOdd(v).ExtendLo8ToUint16())
}

func cflPairSumUint16x16(lo, hi archsimd.Uint16x8) archsimd.Uint16x8 {
	return lo.ConcatEven(hi).Add(lo.ConcatOdd(hi))
}

func cflUint16AnyAbove(v archsimd.Uint16x8, max archsimd.Uint16x8) bool {
	return v.Greater(max).ToInt16x8().ReduceSum() != 0
}

func cflScaleQ3SIMD(ac archsimd.Int16x8, alpha archsimd.Int16x8, round archsimd.Int32x4) archsimd.Int16x8 {
	lo := cflRoundPowerOfTwoSignedInt32SIMD(ac.MulWidenLo(alpha), round)
	hi := cflRoundPowerOfTwoSignedInt32SIMD(ac.MulWidenHi(alpha), round)
	return lo.TruncToInt16().TruncToInt16Hi(hi)
}

func cflRoundPowerOfTwoSignedInt32SIMD(v archsimd.Int32x4, round archsimd.Int32x4) archsimd.Int32x4 {
	sign := v.ShiftAllRight(31)
	mag := v.Xor(sign).Sub(sign).Add(round).ShiftAllRight(6)
	return mag.Xor(sign).Sub(sign)
}
