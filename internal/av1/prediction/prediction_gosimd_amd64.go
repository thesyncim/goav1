// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

import (
	"encoding/binary"
	"simd/archsimd"
)

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

// filterIntraPairs16[mode][k] interleaves the taps k and k+1 (k = 0, 2, 4,
// 6, the last paired with zero) of the eight outputs of one 4x2 filter-intra
// batch: lane pair i holds (taps[k][i], taps[k+1][i]) for vpmaddwd.
var filterIntraPairs16 = func() (t [FilterIntraModes][4][16]int16) {
	for mode := range FilterIntraModes {
		for p := range 4 {
			for i := range 8 {
				t[mode][p][2*i] = int16(filterIntraTaps[mode][i][2*p])
				if 2*p+1 < 7 {
					t[mode][p][2*i+1] = int16(filterIntraTaps[mode][i][2*p+1])
				}
			}
		}
	}
	return t
}()

// filterIntraPair broadcasts the neighbour pair (a, b) to every int32 lane.
func filterIntraPair(a, b uint16) archsimd.Int16x16 {
	return archsimd.BroadcastInt32x8(int32(uint32(a) | uint32(b)<<16)).AsInt16x16()
}

// predictFilterIntraBlockDirect16AVX2 is predictFilterIntraBlockDirect16 for
// 10- and 12-bit samples, in the form of dav1d's ipred16_avx2 filter: each
// 4x2 batch is four vpmaddwd of the interleaved tap pairs against the
// broadcast neighbour pairs (p0,p1), (p2,p3), (p4,p5), (p6,0), exact in
// int32, then Round2(., 4) and the [0, max] clamp, which the scalar
// roundPowerOfTwo(sum, 4) clamped to [0, max] equals sample for sample.
func predictFilterIntraBlockDirect16AVX2(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 || mode >= FilterIntraModes {
		predictFilterIntraBlockDirect16(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraPairs16[mode]
	t0 := archsimd.LoadInt16x16Array(&taps[0])
	t1 := archsimd.LoadInt16x16Array(&taps[1])
	t2 := archsimd.LoadInt16x16Array(&taps[2])
	t3 := archsimd.LoadInt16x16Array(&taps[3])
	roundV := archsimd.BroadcastInt32x8(1 << (filterIntraScaleBits - 1))
	maxV := archsimd.BroadcastInt32x8(int32(max))
	var zeroV archsimd.Int32x8

	// above[0] is p0 of the first batch and above[i+1] the top row sample i.
	var above [65]uint16
	for row := 0; row < height; row += 2 {
		if row == 0 {
			above[0] = edges.AboveLeft
			copy(above[1:width+1], edges.Above[:width])
		} else {
			above[0] = edges.Left[row-1]
			top := block.pix[(row-1)*block.stride:][:width*2]
			for i := 0; i < width; i++ {
				above[i+1] = binary.LittleEndian.Uint16(top[i<<1:])
			}
		}
		row0 := block.pix[row*block.stride:][:width*2]
		row1 := block.pix[(row+1)*block.stride:][:width*2]
		p5, p6 := edges.Left[row], edges.Left[row+1]
		for col := 0; col < width; col += 4 {
			a := above[col : col+5]
			acc := t0.DotProductPairs(filterIntraPair(a[0], a[1])).
				Add(t1.DotProductPairs(filterIntraPair(a[2], a[3]))).
				Add(t2.DotProductPairs(filterIntraPair(a[4], p5))).
				Add(t3.DotProductPairs(filterIntraPair(p6, 0))).
				Add(roundV).ShiftAllRight(filterIntraScaleBits).Max(zeroV).Min(maxV)
			// vpackssdw within lanes: lane 0 holds outputs 0..3 (row 0),
			// lane 1 outputs 4..7 (row 1).
			v := acc.SaturateToInt16ConcatGrouped(acc)
			r0 := v.GetLo().AsUint64x2().GetElem(0)
			r1 := v.GetHi().AsUint64x2().GetElem(0)
			binary.LittleEndian.PutUint64(row0[col<<1:], r0)
			binary.LittleEndian.PutUint64(row1[col<<1:], r1)
			p5, p6 = uint16(r0>>48), uint16(r1>>48)
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none.
	archsimd.ClearAVXUpperBits()
}

// filterIntraPackBytes narrows eight int16 filter outputs, each already clamped
// to [0,255], into the low eight bytes of the result. The narrowing pack
// instructions (VPMOVWB/VPMOVUSWB/VPMOVSWB) and VPERMB are EVEX-only, so the
// bytes are shuffled out of the int16 lanes with the AVX VPSHUFB instead.
func filterIntraPackBytes(out archsimd.Int16x8) archsimd.Uint8x16 {
	return out.AsUint8x16().PermuteOrZero(evenBytesIdx)
}

// filterIntraShift4 arithmetic-shifts each int16 lane right by filterIntraScaleBits
// (VPSRAW with an immediate).
func filterIntraShift4(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.ShiftAllRight(filterIntraScaleBits)
}

// init binds the Go-native SIMD intra kernels on amd64. The gate is the
// archsimd feature probe (not cpu.Detected) so the SIMD path activates under
// GOEXPERIMENT=simd on any AVX2 machine, while every other kernel keeps the
// pure-Go reference.
func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	predictPaethImpl = predictPaethSIMD
	predictSmoothImpl = predictSmoothSIMD
	predictSmoothVerticalImpl = predictSmoothVerticalSIMD
	predictSmoothHorizontalImpl = predictSmoothHorizontalSIMD
	sumSamplesImpl = sumSamplesSIMD
	applyCFLImpl = applyCFLSIMD
	subsampleLuma8Impl = subsampleLuma8SIMD
	dirRowInterp8Impl = dirRowInterp8SIMD
	dirAboveRun8Impl = dirAboveRun8SIMD
	dirLeftCol8Impl = dirLeftCol8SIMD
	predictFilterIntra8Impl = predictFilterIntraBlockDirect8SIMD
	predictFilterIntra16Impl = predictFilterIntraBlockDirect16AVX2
	subsampleLuma16Impl = subsampleLuma16AVX
	subtractCFLAverageImpl = subtractCFLAverageAVX
}

// AVX2 forms of the directional intra edge filter and upsampler (dav1d's
// ipred_filter_edge / ipred_upsample_edge in ipred_avx2.asm). Both work on
// a padded stack copy of the edge, so the clamped end taps of the scalar
// code become plain loads, and both are exact: the filter's weights are
// nonnegative and sum to 16, so 16*4095+8 fits uint16, and the upsampler's
// |-a+9b+9c-d| <= 18*1023 fits int16 up to 10 bits.

var intraEdgeAVX2 = archsimd.X86.AVX2()

// intraEdgeFilterK holds the five filter taps of each strength, broadcast.
var intraEdgeFilterK = func() (k [3][intraEdgeTaps][8]uint16) {
	for s, taps := range intraEdgeFilterKernels {
		for j, t := range taps {
			for l := range k[s][j] {
				k[s][j][l] = uint16(t)
			}
		}
	}
	return k
}()

// filterIntraEdgeArch is the body of FilterIntraEdgeBounded after its checks
// (1 <= strength <= 3); it reports false when the scalar code must run.
func filterIntraEdgeArch(edge []uint16, strength uint8) bool {
	n := len(edge)
	if !intraEdgeAVX2 || n < 2 || n > intraEdgeMaxSize {
		return false
	}
	// pad[m] is the clamped scratch[m-2].
	var pad [intraEdgeMaxSize + 32]uint16
	pad[0], pad[1] = edge[0], edge[0]
	copy(pad[2:], edge)
	last := edge[n-1]
	for i := n + 2; i < n+24; i++ {
		pad[i] = last
	}
	k := &intraEdgeFilterK[strength-1]
	k1 := archsimd.LoadUint16x8Array(&k[1])
	k2 := archsimd.LoadUint16x8Array(&k[2])
	k3 := archsimd.LoadUint16x8Array(&k[3])
	round := archsimd.BroadcastUint16x8(8)
	for i := 1; i < n; i += 8 {
		acc := archsimd.LoadUint16x8(pad[i+1:]).Mul(k1).
			Add(archsimd.LoadUint16x8(pad[i+2:]).Mul(k2)).
			Add(archsimd.LoadUint16x8(pad[i+3:]).Mul(k3)).
			Add(round)
		if strength == 3 {
			acc = acc.Add(archsimd.LoadUint16x8(pad[i:]).Add(archsimd.LoadUint16x8(pad[i+4:])).ShiftAllLeft(1))
		}
		acc = acc.ShiftAllRight(4)
		if i+8 <= n {
			acc.Store(edge[i:])
		} else {
			var t [8]uint16
			acc.StoreArray(&t)
			copy(edge[i:], t[:n-i])
		}
	}
	return true
}

// upsampleIntraEdgeArch is the body of UpsampleIntraEdgeBounded after its
// checks; it reports false when the scalar code must run.
func upsampleIntraEdgeArch(edge []uint16, origin, size int, max uint16) bool {
	if !intraEdgeAVX2 || max > 1023 || size > intraEdgeMaxUpsampleSize || origin < 2 {
		return false
	}
	// s is the scalar scratch: s[0] = s[1] = p[-1], s[2..size+1] = p[0..size-1],
	// s[size+2] = p[size-1], zero past it.
	var s [intraEdgeMaxUpsampleSize + 16]int16
	s[0] = int16(edge[origin-1])
	s[1] = s[0]
	for i := 0; i < size; i++ {
		s[i+2] = int16(edge[origin+i])
	}
	s[size+2] = int16(edge[origin+size-1])
	edge[origin-2] = uint16(s[0])
	nine := archsimd.BroadcastInt16x8(9)
	round := archsimd.BroadcastInt16x8(8)
	maxV := archsimd.BroadcastInt16x8(int16(max))
	var zero archsimd.Int16x8
	out := edge[origin-1:]
	for i := 0; i < size; i += 8 {
		a := archsimd.LoadInt16x8(s[i:])
		b := archsimd.LoadInt16x8(s[i+1:])
		c := archsimd.LoadInt16x8(s[i+2:])
		d := archsimd.LoadInt16x8(s[i+3:])
		f := b.Add(c).Mul(nine).Sub(a).Sub(d).Add(round).ShiftAllRight(4).Max(zero).Min(maxV)
		lo, hi := f.InterleaveLo(c), f.InterleaveHi(c)
		cnt := min(size-i, 8) * 2
		var t [16]int16
		lo.StoreArray((*[8]int16)(t[:8]))
		hi.StoreArray((*[8]int16)(t[8:]))
		for j := 0; j < cnt; j++ {
			out[2*i+j] = uint16(t[j])
		}
	}
	return true
}

// AVX2 implementations of the operations the shared intra kernels need. Every
// operation here lowers to a VEX (AVX/AVX2) instruction: the narrowing VPMOV*
// and VPERMB forms are EVEX-only and would SIGILL on AVX2-only CPUs, so bytes
// are gathered with VPSHUFB and 32-bit values are packed with VPACKSSDW.

// evenBytesIdx selects the even bytes of a vector into its low eight lanes; the
// high lanes are zeroed (index MSB set).
var evenBytesIdx = archsimd.LoadInt8x16Array(&[16]int8{0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128})

// narrowU16ToU8 narrows eight uint16 lanes already in [0,255] to the low eight
// bytes of the result.
func narrowU16ToU8(v archsimd.Uint16x8) archsimd.Uint8x16 {
	return v.AsUint8x16().PermuteOrZero(evenBytesIdx)
}

// hiToLoU16 moves the upper four lanes of v into the lower four (VPALIGNR).
func hiToLoU16(v archsimd.Uint16x8) archsimd.Uint16x8 {
	b := v.AsUint8x16()
	return b.ConcatShiftBytesRight(b, 8).AsUint16x8()
}

// hiToLoI16 moves the upper four lanes of v into the lower four.
func hiToLoI16(v archsimd.Int16x8) archsimd.Int16x8 {
	return hiToLoU16(v.ConvertToUint16()).ConvertToInt16()
}

// mulWidenLo16 multiplies the low four lanes of a and b into int32 lanes.
func mulWidenLo16(a, b archsimd.Int16x8) archsimd.Int32x4 {
	return a.ExtendLo4ToInt32().Mul(b.ExtendLo4ToInt32())
}

// packInt32PairToInt16 packs eight int32 values (lo's four lanes, then hi's) into
// int16 lanes with the AVX2 signed pack; callers guarantee the values fit.
func packInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	return lo.SaturateToInt16Concat(hi)
}

// reduceSumU32 returns the sum of the four uint32 lanes.
func reduceSumU32(v archsimd.Uint32x4) uint32 {
	return v.GetElem(0) + v.GetElem(1) + v.GetElem(2) + v.GetElem(3)
}

// shrU16_1, shrU16_5 and shrU16_8 are logical right shifts of uint16 lanes (VPSRLW
// with an immediate).
func shrU16_1(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.ShiftAllRight(1) }
func shrU16_5(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.ShiftAllRight(5) }
func shrU16_8(v archsimd.Uint16x8) archsimd.Uint16x8 { return v.ShiftAllRight(8) }

// shrI32_6 and shrI32_31 are arithmetic right shifts of int32 lanes (VPSRAD).
func shrI32_6(v archsimd.Int32x4) archsimd.Int32x4  { return v.ShiftAllRight(6) }
func shrI32_31(v archsimd.Int32x4) archsimd.Int32x4 { return v.ShiftAllRight(31) }

// Go-native SIMD CfL luma subsampling for amd64 (AVX2). It mirrors the arm64
// kernel and the scalar contract in subsampleLuma8PureGo. Even and odd input
// bytes are separated with VPSHUFB, widened to uint16 and summed.

// evenBytesFullIdx and oddBytesFullIdx pick the even and odd bytes of a 16-byte row
// into the low eight lanes.
var (
	evenBytesFullIdx = archsimd.LoadInt8x16Array(&[16]int8{0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128})
	oddBytesFullIdx  = archsimd.LoadInt8x16Array(&[16]int8{1, 3, 5, 7, 9, 11, 13, 15, -128, -128, -128, -128, -128, -128, -128, -128})
)

// cflPairSumBytes returns the per-column pair sums a[2c]+a[2c+1] of the 16-byte
// row v, for the eight output columns.
func cflPairSumBytes(v archsimd.Uint8x16) archsimd.Uint16x8 {
	even := v.PermuteOrZero(evenBytesFullIdx).ExtendLo8ToUint16()
	odd := v.PermuteOrZero(oddBytesFullIdx).ExtendLo8ToUint16()
	return even.Add(odd)
}

// subsampleLuma8SIMD is the amd64 form of subsampleLuma8PureGo.
func subsampleLuma8SIMD(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
	switch {
	case subX && subY:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < height; row += 2 {
			outBase := (row >> 1) * CFLBufLine
			top := input[row*inputStride:]
			bot := input[(row+1)*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				t := cflPairSumBytes(loadRow16(top[ic:]))
				b := cflPairSumBytes(loadRow16(bot[ic:]))
				t.Add(b).ShiftAllLeft(1).StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	case subX:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			in := input[row*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				ic := oc << 1
				cflPairSumBytes(loadRow16(in[ic:])).ShiftAllLeft(2).StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	default:
		if outW%8 != 0 {
			subsampleLuma8PureGo(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
			return
		}
		for row := 0; row < outH; row++ {
			outBase := row * CFLBufLine
			in := input[row*inputStride:]
			for oc := 0; oc < outW; oc += 8 {
				archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(in[oc:])).ReshapeToUint8s().ExtendLo8ToUint16().ShiftAllLeft(3).StoreArray((*[8]uint16)(outputQ3[outBase+oc:]))
			}
		}
	}
}

// loadRow16 returns the sixteen bytes at the start of s. Near the end of the
// input buffer fewer bytes may remain; those lanes are never read by the scalar
// reference, so they are zero-filled from a local copy instead of over-reading.
func loadRow16(s []byte) archsimd.Uint8x16 {
	if len(s) >= 16 {
		return archsimd.LoadUint8x16Array((*[16]uint8)(s))
	}
	var buf [16]uint8
	copy(buf[:], s)
	return archsimd.LoadUint8x16Array(&buf)
}

// amd64 routers for the 8-bit intra predictors. The Go SIMD kernels serve every
// width a multiple of 8, the width-4 kernels (intra_w4_gosimd_amd64.go) width 4
// with even height, and every other shape stays on the PureGo reference.

func predictPaethSIMD(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		paeth8SIMD(block, above, left, aboveLeft)
		return
	}
	if bytesPerSample == 1 && block.width == 4 && block.height%2 == 0 {
		paethW4AVX(block, above, left, aboveLeft)
		return
	}
	predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
}

func predictSmoothSIMD(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		smooth8SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
		return
	}
	if bytesPerSample == 1 && block.width == 4 && block.height%2 == 0 {
		smoothW4AVX(block, weightsW, weightsH, above, left, belowPred, rightPred)
		return
	}
	predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
}

func predictSmoothVerticalSIMD(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		smoothVertical8SIMD(block, weights, above, belowPred)
		return
	}
	if bytesPerSample == 1 && block.width == 4 && block.height%2 == 0 {
		smoothVerticalW4AVX(block, weights, above, belowPred)
		return
	}
	predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
}

func predictSmoothHorizontalSIMD(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		smoothHorizontal8SIMD(block, weights, left, rightPred)
		return
	}
	if bytesPerSample == 1 && block.width == 4 && block.height%2 == 0 {
		smoothHorizontalW4AVX(block, weights, left, rightPred)
		return
	}
	predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
}

// applyCFLSIMD routes the 8-bit CfL apply with width a multiple of 8 and max 255
// to the Go SIMD kernel; every other shape uses the PureGo reference.
func applyCFLSIMD(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	if bytesPerSample == 1 && visibleWidth%8 == 0 && max == 0xff {
		applyCFL8SIMD(block, visibleWidth, visibleHeight, acQ3, alphaQ3)
		return
	}
	if bytesPerSample == 1 && visibleWidth == 4 && visibleHeight%2 == 0 && max == 0xff {
		applyCFL4AVX(block, visibleHeight, acQ3, alphaQ3)
		return
	}
	applyCFLPureGo(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
}

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
