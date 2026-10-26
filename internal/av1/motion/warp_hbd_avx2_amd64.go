// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// AVX2 10/12-bit warped motion (dav1d src/x86/mc16_avx2.asm
// warp_affine_8x8_16bpc). The horizontal pass is warpHorizontal8AVX2's on
// the samples centred by -2^15; the vertical
// intermediate does not fit int16 at these depths, so the vertical pass
// multiplies the int32 rows by per-column coefficient vectors (vpmulld),
// the eight column filters transposed into tap pairs and split into their
// sign-extended halves. Both compute the scalar formulas exactly.

func warpHorizontalHighBDResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	// The sixteen-sample loads read one sample past the resident window.
	if !warpAVX2 || !warpHorizResidentOffsInRange(sx4, alpha, beta) || ref.Stride&1 != 0 ||
		(iy4+7)*ref.Stride+(ix4+9)*2 > len(ref.Pix) {
		return warpHorizontalHighBDResident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	warpHorizontalHighBDAVX2(tmp, ref, ix4, sx4, iy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	return sy4
}

func warpVerticalHighBDFullDispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || !warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVerticalHighBDFull(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDAVX2(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
}

func warpVerticalHighBDFullGamma0Dispatch(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || !warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVerticalHighBDFullGamma0(dst, bitDepth, max, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVerticalHighBDAVX2(dst, bitDepth, max, tmp, i+rowShift, j+colShift, baseSY, 0, delta, reduceBitsVert, offsetBitsVert)
}

// warpHighBDSampleCenterBias restores the 2^15 * 128 that centring the
// samples removes from each dot product.
const warpHighBDSampleCenterBias = 1 << 22

// warpHorizontalHighBDAVX2 computes the resident 15x8 intermediate:
// tmp = (2^offset + sum(s * c) + 2^(reduce-1)) >> reduce.
func warpHorizontalHighBDAVX2(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, alpha, beta, reduce, offset int) {
	// Samples are centred by -2^15 so every uint16 value is an exact int16;
	// each filter row sums to 128, so the centring removes 2^22, added back
	// in the bias.
	bias := archsimd.BroadcastInt32x8(int32(1)<<offset + int32(1)<<(reduce-1) + warpHighBDSampleCenterBias)
	centre := archsimd.BroadcastInt16x16(-1 << 15)
	sh := archsimd.BroadcastUint32x8(uint32(reduce))
	winLo := archsimd.LoadUint32x8Array(&warpWinLo)
	winHi := archsimd.LoadUint32x8Array(&warpWinHi)
	for k := -7; k < 8; k++ {
		rowByte := (iy4+k)*ref.Stride + (ix4-7)*2
		a := archsimd.LoadUint8x32(ref.Pix[rowByte : rowByte+32]).AsInt16x16().Xor(centre).AsInt32x8()
		b := a.Permute(winLo).AsUint8x32()
		c := a.Permute(winHi).AsUint8x32()
		sx := sx4 + beta*(k+4)
		w0 := b.AsInt16x16()
		w1 := c.ConcatShiftBytesRightGrouped(b, 2).AsInt16x16()
		w2 := c.ConcatShiftBytesRightGrouped(b, 4).AsInt16x16()
		w3 := c.ConcatShiftBytesRightGrouped(b, 6).AsInt16x16()
		p0 := w0.DotProductPairs(warpFilterPair(warpPhase(sx), warpPhase(sx+4*alpha)))
		p1 := w1.DotProductPairs(warpFilterPair(warpPhase(sx+alpha), warpPhase(sx+5*alpha)))
		p2 := w2.DotProductPairs(warpFilterPair(warpPhase(sx+2*alpha), warpPhase(sx+6*alpha)))
		p3 := w3.DotProductPairs(warpFilterPair(warpPhase(sx+3*alpha), warpPhase(sx+7*alpha)))
		sum := p0.ConcatAddPairsGrouped(p1).ConcatAddPairsGrouped(p2.ConcatAddPairsGrouped(p3))
		out := (k + 7) * warpedIntermediateColumns
		sum.Add(bias).ShiftRight(sh).StoreArray((*[8]int32)(tmp[out : out+8]))
	}
	archsimd.ClearAVXUpperBits()
}

// warpVerticalHighBDAVX2 computes the 8x8 output block at (x0, y0):
// clip(((2^offset + sum(c * tmp) + 2^(reduce-1)) >> reduce) - 2^(bd-1) - 2^bd).
func warpVerticalHighBDAVX2(dst frame.Plane, bitDepth uint8, max uint16, tmp *warpTmp, y0, x0, baseSY, gamma, delta, reduce, offset int) {
	bias := archsimd.BroadcastInt32x8(int32(1)<<offset + int32(1)<<(reduce-1))
	sh := archsimd.BroadcastUint32x8(uint32(reduce))
	sub := archsimd.BroadcastInt32x8(int32(1)<<(bitDepth-1) + int32(1)<<bitDepth)
	var zero archsimd.Int32x8
	maxV := archsimd.BroadcastInt32x8(int32(max))
	for row := 0; row < 8; row++ {
		sy := baseSY + delta*row
		f0 := warpFilterPair(warpPhase(sy), warpPhase(sy+4*gamma)).AsInt32x8()
		f1 := warpFilterPair(warpPhase(sy+gamma), warpPhase(sy+5*gamma)).AsInt32x8()
		f2 := warpFilterPair(warpPhase(sy+2*gamma), warpPhase(sy+6*gamma)).AsInt32x8()
		f3 := warpFilterPair(warpPhase(sy+3*gamma), warpPhase(sy+7*gamma)).AsInt32x8()
		a0, a1 := f0.InterleaveLoGrouped(f1).AsInt64x4(), f0.InterleaveHiGrouped(f1).AsInt64x4()
		a2, a3 := f2.InterleaveLoGrouped(f3).AsInt64x4(), f2.InterleaveHiGrouped(f3).AsInt64x4()
		// Tap pairs (t, t+1) per column, columns 0..3 low lane, 4..7 high.
		k01 := a0.InterleaveLoGrouped(a2).AsInt32x8()
		k23 := a0.InterleaveHiGrouped(a2).AsInt32x8()
		k45 := a1.InterleaveLoGrouped(a3).AsInt32x8()
		k67 := a1.InterleaveHiGrouped(a3).AsInt32x8()
		// Column order of the coefficient vectors: c0..c3 | c4..c7, the
		// order of a tmp row.
		ld := func(t int) archsimd.Int32x8 {
			return archsimd.LoadInt32x8Array((*[8]int32)(tmp[(row+t)*8 : (row+t)*8+8]))
		}
		sum := bias.Add(ld(0).Mul(k01.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(1).Mul(k01.ShiftAllRight(16))).
			Add(ld(2).Mul(k23.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(3).Mul(k23.ShiftAllRight(16))).
			Add(ld(4).Mul(k45.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(5).Mul(k45.ShiftAllRight(16))).
			Add(ld(6).Mul(k67.ShiftAllLeft(16).ShiftAllRight(16))).Add(ld(7).Mul(k67.ShiftAllRight(16)))
		v := sum.ShiftRight(sh).Sub(sub).Max(zero).Min(maxV)
		p := v.SaturateToInt16ConcatGrouped(v)
		off := (y0+row)*dst.Stride + 2*x0
		d := dst.Pix[off : off+16]
		binary.LittleEndian.PutUint64(d[0:], p.GetLo().AsUint64x2().GetElem(0))
		binary.LittleEndian.PutUint64(d[8:], p.GetHi().AsUint64x2().GetElem(0))
	}
	archsimd.ClearAVXUpperBits()
}
