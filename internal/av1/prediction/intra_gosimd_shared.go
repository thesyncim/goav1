// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package prediction

// Go-native SIMD intra kernels shared by arm64 and amd64. They cover the 8-bit
// shapes the NEON and AVX2 asm served: PAETH and the three SMOOTH predictors
// (width a multiple of 8), CfL apply (8-bit), the DC edge sum and the directional
// Z1/Z2/Z3 interpolation. The arch-specific operations they need (byte narrowing,
// shifts by a constant, the high half of a vector, widening multiplies and
// packing) come from intra_gosimd_helpers_{arm64,amd64}.go.
//
// Arithmetic bounds that make the narrow lanes exact:
//   - PAETH: base = top+left-topLeft and the distances fit int16 for samples up
//     to 12 bits.
//   - SMOOTH_V and SMOOTH_H are convex combinations of 8-bit samples with weights
//     summing to 256, so the rounded result w*a+(256-w)*b+128 is at most
//     255*256+128 = 65408 and fits uint16.
//   - SMOOTH (2D) is V+H with V and H each a convex combination. It is computed as
//     (floor((V+H)/2)+128)>>8 with floor((V+H)/2) = (V>>1)+(H>>1)+(V&H&1), which
//     equals (V+H+256)>>9 exactly and stays in uint16.
//   - Directional interpolation operates on 8-bit edges: p0*(32-s)+p1*s+16 <=
//     255*32+16 fits uint16.

import (
	"encoding/binary"
	"simd/archsimd"
)

const (
	// smoothScale is the fixed-point weight scale (256).
	smoothScale = 1 << smoothWeightLog2Scale
	// smoothShift1D is the divideRound bit count for SMOOTH_V/H (8).
	smoothShift1D = smoothWeightLog2Scale
	// smoothShiftFull is the divideRound bit count for 2D SMOOTH (9).
	smoothShiftFull = 1 + smoothWeightLog2Scale
)

// storeU8x8 narrows eight uint16 lanes (each already in [0,255]) into dst[0:8].
func storeU8x8(dst []byte, v archsimd.Uint16x8) {
	binary.LittleEndian.PutUint64(dst, narrowU16ToU8(v).ReshapeToUint64s().GetElem(0))
}

// loadU16x8 loads eight uint16 samples (at most 12 bits) as an Int16x8.
func loadU16x8(s []uint16) archsimd.Int16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(s)).ConvertToInt16()
}

// putU32 stores v little-endian at pix[off:off+4].
func putU32(pix []byte, off int, v uint32) {
	binary.LittleEndian.PutUint32(pix[off:], v)
}

// absDiffInt16 returns |a-b| for two 12-bit samples as an int16 scalar; the
// per-row PAETH distance |left - aboveLeft| is computed here once per row.
func absDiffInt16(a, b uint16) int16 {
	d := int16(a) - int16(b)
	if d < 0 {
		return -d
	}
	return d
}

// smoothComplement returns (256-w)*edge, the per-row (or per-column) product of
// the complementary SMOOTH weight with the edge sample. The caller adds any
// rounding bias; the product is at most 256*255 and fits uint16.
func smoothComplement(w uint16, edge uint16) uint16 {
	return uint16((smoothScale - uint32(w)) * uint32(edge))
}

// paethInt16 computes PAETH for one 8-lane vector of above samples aV against
// the per-row left value lV, the block's aboveLeft constant tlV, lmtV = left-2*tl
// and pTop = |left-tl|. The three distances and the left/top/topLeft tie order
// are the same as paethPredictorSingle.
func paethInt16(aV, lV, tlV, lmtV, pTop archsimd.Int16x8) archsimd.Int16x8 {
	pLeft := aV.Sub(tlV).Abs()
	pTopLeft := aV.Add(lmtV).Abs()
	out := tlV
	out = aV.IfElse(pTop.LessEqual(pTopLeft), out)
	leftMask := pLeft.LessEqual(pTop).And(pLeft.LessEqual(pTopLeft))
	return lV.IfElse(leftMask, out)
}

// paeth8SIMD is the 8-bit kernel for width a multiple of 8.
func paeth8SIMD(block planeBlock, above []uint16, left []uint16, aboveLeft uint16) {
	width, height := block.width, block.height
	tlV := archsimd.BroadcastInt16x8(int16(aboveLeft))
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastInt16x8(int16(left[row]))
		pTop := archsimd.BroadcastInt16x8(absDiffInt16(left[row], aboveLeft))
		lmtV := archsimd.BroadcastInt16x8(int16(left[row]) - int16(aboveLeft)*2)
		dst := block.pix[row*block.stride:][:width]
		for col := 0; col < width; col += 8 {
			out := paethInt16(loadU16x8(above[col:]), lV, tlV, lmtV, pTop)
			storeU8x8(dst[col:], out.ConvertToUint16())
		}
	}
}

// smooth8Cols holds the per-column constants of the 8-bit 2D SMOOTH predictor for
// each 8-column block: the horizontal weights and (256-wW)*rightPred.
type smooth8Cols struct {
	wW  [8]archsimd.Uint16x8
	cwR [8]archsimd.Uint16x8
}

// smooth8SIMD is the 8-bit 2D SMOOTH kernel for width a multiple of 8.
func smooth8SIMD(block planeBlock, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	oneV := archsimd.BroadcastUint16x8(1)
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	rightV := archsimd.BroadcastUint16x8(rightPred)
	scaleV := archsimd.BroadcastUint16x8(smoothScale)
	var cols smooth8Cols
	for b := 0; b < blocks; b++ {
		w := archsimd.LoadUint16x8Array((*[8]uint16)(weightsW[b*8:]))
		cols.wW[b] = w
		cols.cwR[b] = scaleV.Sub(w).Mul(rightV)
	}
	for row := 0; row < height; row++ {
		wH := weightsH[row]
		wHV := archsimd.BroadcastUint16x8(wH)
		cbH := archsimd.BroadcastUint16x8(smoothComplement(wH, belowPred))
		lV := archsimd.BroadcastUint16x8(left[row])
		dst := block.pix[row*block.stride:][:width]
		for b := 0; b < blocks; b++ {
			aV := archsimd.LoadUint16x8Array((*[8]uint16)(above[b*8:]))
			v := wHV.Mul(aV).Add(cbH)
			h := cols.wW[b].Mul(lV).Add(cols.cwR[b])
			half := shrU16_1(v).Add(shrU16_1(h)).Add(v.And(h).And(oneV))
			storeU8x8(dst[b*8:], shrU16_8(half.Add(roundV)))
		}
	}
}

// smoothVertical8SIMD is the 8-bit SMOOTH_V kernel for width a multiple of 8.
func smoothVertical8SIMD(block planeBlock, weights []uint16, above []uint16, belowPred uint16) {
	width, height := block.width, block.height
	for row := 0; row < height; row++ {
		w := weights[row]
		wV := archsimd.BroadcastUint16x8(w)
		cb := archsimd.BroadcastUint16x8(smoothComplement(w, belowPred) + 1<<(smoothShift1D-1))
		dst := block.pix[row*block.stride:][:width]
		for col := 0; col < width; col += 8 {
			aV := archsimd.LoadUint16x8Array((*[8]uint16)(above[col:]))
			storeU8x8(dst[col:], shrU16_8(wV.Mul(aV).Add(cb)))
		}
	}
}

// smoothHorizontal8SIMD is the 8-bit SMOOTH_H kernel for width a multiple of 8.
func smoothHorizontal8SIMD(block planeBlock, weights []uint16, left []uint16, rightPred uint16) {
	width, height := block.width, block.height
	blocks := width / 8
	roundV := archsimd.BroadcastUint16x8(1 << (smoothShift1D - 1))
	rightV := archsimd.BroadcastUint16x8(rightPred)
	var wW, cwR [8]archsimd.Uint16x8
	for b := 0; b < blocks; b++ {
		w := archsimd.LoadUint16x8Array((*[8]uint16)(weights[b*8:]))
		wW[b] = w
		cwR[b] = archsimd.BroadcastUint16x8(smoothScale).Sub(w).Mul(rightV).Add(roundV)
	}
	for row := 0; row < height; row++ {
		lV := archsimd.BroadcastUint16x8(left[row])
		dst := block.pix[row*block.stride:][:width]
		for b := 0; b < blocks; b++ {
			storeU8x8(dst[b*8:], shrU16_8(wW[b].Mul(lV).Add(cwR[b])))
		}
	}
}

// cflRoundBias32 is the rounding bias of roundPowerOfTwoSigned(x, 6).
var cflRoundBias32 = archsimd.BroadcastInt32x4(1 << 5)

// cflRound6 returns roundPowerOfTwoSigned(x, 6) for four int32 lanes: the
// magnitude (|x|+32)>>6 with the sign restored by (r^s)-s, s = x>>31.
func cflRound6(x archsimd.Int32x4) archsimd.Int32x4 {
	mag := shrI32_6(x.Abs().Add(cflRoundBias32))
	sign := shrI32_31(x)
	return mag.Xor(sign).Sub(sign)
}

// cflClampInt32Pair clamps two int32 vectors to [0,max] and packs them to int16.
func cflClampInt32Pair(lo, hi archsimd.Int32x4, zero, max archsimd.Int32x4) archsimd.Int16x8 {
	return packInt32PairToInt16(lo.Max(zero).Min(max), hi.Max(zero).Min(max))
}

// maxSIMDSumSamples bounds the sample count for which the uint32 lane sums in
// sumSamplesSIMD are exact (65536 * 65535 < 2^32).
const maxSIMDSumSamples = 1 << 16

// sumSamplesSIMD is the Go-native SIMD form of sumSamplesPureGo.
func sumSamplesSIMD(samples []uint16) int {
	n := len(samples)
	if n > maxSIMDSumSamples {
		return sumSamplesPureGo(samples)
	}
	chunks := n &^ 7
	total := 0
	if chunks > 0 {
		acc := archsimd.BroadcastUint32x4(0)
		for i := 0; i < chunks; i += 8 {
			v := archsimd.LoadUint16x8Array((*[8]uint16)(samples[i:]))
			acc = acc.Add(v.ExtendLo4ToUint32()).Add(hiToLoU16(v).ExtendLo4ToUint32())
		}
		total = int(reduceSumU32(acc))
	}
	for i := chunks; i < n; i++ {
		total += int(samples[i])
	}
	return total
}

// loadBytes8 gathers the eight bytes b[0:8] as eight uint16 lanes.
func loadBytes8(b []byte) archsimd.Int16x8 {
	return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(b)).ReshapeToUint8s().ExtendLo8ToUint16().ConvertToInt16()
}

// applyCFL8SIMD is the 8-bit CfL apply for widths that are a multiple of 8 with
// max 255 (the saturating clamp is exactly [0,255]).
func applyCFL8SIMD(block planeBlock, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int) {
	alphaV := archsimd.BroadcastInt16x8(int16(alphaQ3))
	zeroV := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastInt32x4(0xff)
	for row := 0; row < visibleHeight; row++ {
		line := block.pix[row*block.stride:][:visibleWidth]
		ac := acQ3[row*CFLBufLine:][:visibleWidth]
		for col := 0; col < visibleWidth; col += 8 {
			v := archsimd.LoadInt16x8Array((*[8]int16)(ac[col:]))
			loR := cflRound6(mulWidenLo16(alphaV, v))
			hiR := cflRound6(mulWidenLo16(alphaV, hiToLoI16(v)))
			cur := loadBytes8(line[col:])
			lo := cur.ExtendLo4ToInt32().Add(loR)
			hi := hiToLoI16(cur).ExtendLo4ToInt32().Add(hiR)
			packed := cflClampInt32Pair(lo, hi, zeroV, maxV).ConvertToUint16()
			binary.LittleEndian.PutUint64(line[col:], narrowU16ToU8(packed).ReshapeToUint64s().GetElem(0))
		}
	}
}

// dirInterpolate8 returns roundPowerOfTwo(p0*w0+p1*w1, 5) for eight lanes, where
// w0 = 32-shift and w1 = shift.
func dirInterpolate8(p0, p1, w0, w1, round archsimd.Uint16x8) archsimd.Uint16x8 {
	return shrU16_5(p0.Mul(w0).Add(p1.Mul(w1)).Add(round))
}

// dirRowInterp8SIMD is the Go-native SIMD form of dirRowInterp8PureGo for fully
// interpolated rows (every column inside [0,maxBase)) with width a multiple of 8.
// Other rows, which touch the clamp, use the scalar reference.
func dirRowInterp8SIMD(dst []byte, above []uint16, base int, shift int, maxBase int, width int) {
	if width%8 != 0 || base < 0 || base+width > maxBase {
		dirRowInterp8PureGo(dst, above, base, shift, maxBase, width)
		return
	}
	w0 := archsimd.BroadcastUint16x8(uint16(32 - shift))
	w1 := archsimd.BroadcastUint16x8(uint16(shift))
	round := archsimd.BroadcastUint16x8(1 << 4)
	for col := 0; col < width; col += 8 {
		p0 := archsimd.LoadUint16x8Array((*[8]uint16)(above[base+col:]))
		p1 := archsimd.LoadUint16x8Array((*[8]uint16)(above[base+col+1:]))
		storeU8x8(dst[col:], dirInterpolate8(p0, p1, w0, w1, round))
	}
}

// dirAboveRun8SIMD is the Go-native SIMD form of dirAboveRun8PureGo: count
// contiguous outputs with ref[i], ref[i+1] as the pair for output i.
func dirAboveRun8SIMD(dst []byte, ref []uint16, shift int, count int) {
	w0 := archsimd.BroadcastUint16x8(uint16(32 - shift))
	w1 := archsimd.BroadcastUint16x8(uint16(shift))
	round := archsimd.BroadcastUint16x8(1 << 4)
	chunks := count &^ 7
	for i := 0; i < chunks; i += 8 {
		p0 := archsimd.LoadUint16x8Array((*[8]uint16)(ref[i:]))
		p1 := archsimd.LoadUint16x8Array((*[8]uint16)(ref[i+1:]))
		storeU8x8(dst[i:], dirInterpolate8(p0, p1, w0, w1, round))
	}
	if chunks < count {
		dirAboveRun8PureGo(dst[chunks:], ref[chunks:], shift, count-chunks)
	}
}

// dirLeftCol8SIMD is the Go-native SIMD form of dirLeftCol8PureGo: count samples
// down one column, stride bytes apart. Eight rows are interpolated per vector and
// their bytes scattered to the column.
func dirLeftCol8SIMD(dst []byte, stride int, ref []uint16, shift int, count int) {
	w0 := archsimd.BroadcastUint16x8(uint16(32 - shift))
	w1 := archsimd.BroadcastUint16x8(uint16(shift))
	round := archsimd.BroadcastUint16x8(1 << 4)
	chunks := count &^ 7
	for i := 0; i < chunks; i += 8 {
		p0 := archsimd.LoadUint16x8Array((*[8]uint16)(ref[i:]))
		p1 := archsimd.LoadUint16x8Array((*[8]uint16)(ref[i+1:]))
		out := narrowU16ToU8(dirInterpolate8(p0, p1, w0, w1, round)).ReshapeToUint64s().GetElem(0)
		for k := 0; k < 8; k++ {
			dst[(i+k)*stride] = byte(out >> (8 * k))
		}
	}
	if chunks < count {
		dirLeftCol8PureGo(dst[chunks*stride:], stride, ref[chunks:], shift, count-chunks)
	}
}
