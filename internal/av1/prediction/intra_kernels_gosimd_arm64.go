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

func subsampleLuma8SIMD(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
	switch {
	case subX && subY:
		if outW < 8 || outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
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
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
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
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			inBase := row * inputStride
			for col := 0; col < outW; col += 16 {
				v := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Pointer(&input[inBase+col])))
				v.ExtendLo8ToUint16().ShiftAllLeft(3).StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+col])))
				v.HiToLo().ExtendLo8ToUint16().ShiftAllLeft(3).StoreArray((*[8]uint16)(unsafe.Pointer(&outputQ3[outBase+col+8])))
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
			acc = acc.Add(v.ExtendLo4ToUint32()).Add(v.HiToLo().ExtendLo4ToUint32())
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
			hi := v.HiToLo().ExtendLo4ToUint32().ConvertToInt32().Sub(avgV)
			cflTruncateInt32PairToInt16(lo, hi).StoreArray((*[8]int16)(unsafe.Pointer(&dstQ3[base+col])))
		}
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

// cflTruncateInt32PairToInt16 packs the low four lanes of lo followed by the
// low four lanes of hi. ARM64 archsimd exposes single-vector truncation; a
// 64-bit interleave combines the two packed halves in lane order.
func cflTruncateInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	lo64 := lo.TruncToInt16().ToBits().ReshapeToUint64s()
	hi64 := hi.TruncToInt16().ToBits().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s().BitsToInt16()
}

// cflSaturateInt16PairToUint8 packs two groups of eight signed pixels into one
// 16-byte vector, preserving the order expected by the pixel store.
