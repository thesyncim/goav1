// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import "simd/archsimd"

// AVX (VEX-128) forms of the 16bpc CfL luma subsampling and the average
// subtraction (dav1d src/x86/ipred16_avx2.asm cfl_ac_*_16bpc). Samples are
// at most 4095 and every Q3 value at most 32760, so the int16/int32 lanes
// are exact; a sample above max falls back to the scalar path, which
// reports it.

// cfl16AnyAbove reports whether a lane of v exceeds max.
func cfl16AnyAbove(v, maxV archsimd.Uint16x8) bool {
	return !v.SubSaturated(maxV).IsZero()
}

// cfl16PairSums sums the adjacent sample pairs of a and b (sixteen inputs)
// into eight int16 lanes, in order.
func cfl16PairSums(a, b archsimd.Uint16x8, ones archsimd.Int16x8) archsimd.Int16x8 {
	return a.AsInt16x8().DotProductPairs(ones).SaturateToInt16Concat(b.AsInt16x8().DotProductPairs(ones))
}

func subsampleLuma16AVX(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool, max uint16) error {
	if outW < 8 || outW%8 != 0 {
		return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
	}
	maxV := archsimd.BroadcastUint16x8(max)
	ones := archsimd.BroadcastInt16x8(1)
	switch {
	case subX && subY:
		for row := 0; row < height; row += 2 {
			out := outputQ3[(row>>1)*CFLBufLine:]
			top := input[row*inputStride:]
			bot := input[(row+1)*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				t0, t1 := archsimd.LoadUint16x8(top[ic:ic+8]), archsimd.LoadUint16x8(top[ic+8:ic+16])
				b0, b1 := archsimd.LoadUint16x8(bot[ic:ic+8]), archsimd.LoadUint16x8(bot[ic+8:ic+16])
				if cfl16AnyAbove(t0, maxV) || cfl16AnyAbove(t1, maxV) || cfl16AnyAbove(b0, maxV) || cfl16AnyAbove(b1, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				cfl16PairSums(t0, t1, ones).Add(cfl16PairSums(b0, b1, ones)).ShiftAllLeft(1).AsUint16x8().Store(out[oc : oc+8])
			}
		}
	case subX:
		for row := 0; row < outH; row++ {
			out := outputQ3[row*CFLBufLine:]
			in := input[row*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				i0, i1 := archsimd.LoadUint16x8(in[ic:ic+8]), archsimd.LoadUint16x8(in[ic+8:ic+16])
				if cfl16AnyAbove(i0, maxV) || cfl16AnyAbove(i1, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				cfl16PairSums(i0, i1, ones).ShiftAllLeft(2).AsUint16x8().Store(out[oc : oc+8])
			}
		}
	default:
		for row := 0; row < outH; row++ {
			out := outputQ3[row*CFLBufLine:]
			in := input[row*inputStride:]
			for col := 0; col < outW; col += 8 {
				v := archsimd.LoadUint16x8(in[col : col+8])
				if cfl16AnyAbove(v, maxV) {
					return subsampleLuma16PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
				}
				v.ShiftAllLeft(3).Store(out[col : col+8])
			}
		}
	}
	return nil
}

func subtractCFLAverageAVX(srcQ3 []uint16, dstQ3 []int16, width int, height int, numPelLog2 int) {
	if width < 8 || width%8 != 0 {
		subtractCFLAveragePureGo(srcQ3, dstQ3, width, height, numPelLog2)
		return
	}
	ones := archsimd.BroadcastInt16x8(1)
	var acc archsimd.Int32x4
	for row := 0; row < height; row++ {
		src := srcQ3[row*CFLBufLine:]
		for col := 0; col < width; col += 8 {
			acc = acc.Add(archsimd.LoadUint16x8(src[col : col+8]).AsInt16x8().DotProductPairs(ones))
		}
	}
	var lanes [4]int32
	acc.StoreArray(&lanes)
	sum := int(lanes[0]) + int(lanes[1]) + int(lanes[2]) + int(lanes[3]) + ((width * height) >> 1)
	avgV := archsimd.BroadcastInt16x8(int16(sum >> numPelLog2))
	for row := 0; row < height; row++ {
		src := srcQ3[row*CFLBufLine:]
		dst := dstQ3[row*CFLBufLine:]
		for col := 0; col < width; col += 8 {
			archsimd.LoadUint16x8(src[col : col+8]).AsInt16x8().Sub(avgV).Store(dst[col : col+8])
		}
	}
}
