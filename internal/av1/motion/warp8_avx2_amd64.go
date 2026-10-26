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

// AVX2 8-bit warped motion (dav1d src/x86/mc_avx2.asm warp_affine_8x8): the
// horizontal pass forms each output's eight-tap dot product with vpmaddwd
// on centred int16 samples and its own filter row (two outputs per ymm, one
// per 128-bit lane), and two vphaddd levels gather the eight outputs in
// order; the vertical pass pairs the int16 intermediate rows and multiplies
// them against the per-column filters, transposed into tap pairs. Both are
// exact in int32 and compute the scalar warpHorizontal8Resident and
// warpVertical8Full formulas.

// warp16Filter is warpedFilter widened to int16 with 256 rows, indexed as
// [i&255] (the routing preconditions keep every index inside the filter rows).
var warp16Filter [256][8]int16

func init() {
	for i := range warpedFilter {
		warp16Filter[i] = warpedFilter[i]
	}
}

var warpAVX2 = archsimd.X86.AVX2()

func warpHorizontal8ResidentDispatch(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	if !warpAVX2 || reduceBitsHoriz != round0Bits || offsetBitsHoriz != 8+filterBits-1 || !warpHorizResidentOffsInRange(sx4, alpha, beta) {
		return warpHorizontal8Resident(tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
	}
	warpHorizontal8AVX2(tmp, ref, ix4, sx4, iy4, alpha, beta)
	return sy4
}

func warpVertical8FullDispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits || !warpVertFullOffsInRange(baseSY, gamma, delta) {
		warpVertical8Full(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8AVX2(dst, tmp, i, j, rowShift, colShift, baseSY, gamma, delta)
}

func warpVertical8FullGamma0Dispatch(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert int) {
	if !warpAVX2 || reduceBitsVert != round1Bits || offsetBitsVert != 8+2*filterBits-round0Bits || !warpVertFullOffsInRange(baseSY, 0, delta) {
		warpVertical8FullGamma0(dst, tmp, i, j, rowShift, colShift, baseSY, delta, reduceBitsVert, offsetBitsVert)
		return
	}
	warpVertical8AVX2(dst, tmp, i, j, rowShift, colShift, baseSY, 0, delta)
}

// warpFilterPair loads the int16 filter rows of the phases a and b into the
// low and high 128-bit lanes.
func warpFilterPair(a, b int) archsimd.Int16x16 {
	return archsimd.Int16x16{}.SetLo(archsimd.LoadInt16x8Array(&warp16Filter[a&255])).SetHi(archsimd.LoadInt16x8Array(&warp16Filter[b&255]))
}

// warpPhase is the filter row of a phase: 64 + round2(s, 10).
func warpPhase(s int) int { return 64 + ((s + 512) >> 10) }

var (
	// Dword selections of the widened 16-sample row: the low lane holds
	// samples 0..7 (outputs 0..3 read windows 0..10) and the high lane
	// samples 4..11 (outputs 4..7); the second vector continues each lane.
	warpWinLo = [8]uint32{0, 1, 2, 3, 2, 3, 4, 5}
	warpWinHi = [8]uint32{4, 5, 6, 7, 6, 7, 7, 7}
)

// warpHorizontal8AVX2 computes the resident 15x8 intermediate:
// tmp = (sum(s * c) + 2^14 + 4) >> 3, with s centred by -128 (every filter
// row sums to 128, so the centring removes 2^14 and the offset adds it back:
// the constant is 2^15 + 4).
func warpHorizontal8AVX2(tmp *warpTmp, ref frame.Plane, ix4, sx4, iy4, alpha, beta int) {
	centre := archsimd.BroadcastUint8x16(128)
	bias := archsimd.BroadcastInt32x8(1<<15 + 4)
	winLo := archsimd.LoadUint32x8Array(&warpWinLo)
	winHi := archsimd.LoadUint32x8Array(&warpWinHi)
	for k := -7; k < 8; k++ {
		rowByte := (iy4+k)*ref.Stride + ix4 - 7
		sx := sx4 + beta*(k+4)
		a := u8Bytes16(ref.Pix[rowByte:]).Xor(centre).AsInt8x16().ExtendToInt16().AsInt32x8()
		b := a.Permute(winLo).AsUint8x32()
		c := a.Permute(winHi).AsUint8x32()
		// Window j of each lane is b:c shifted by 2j bytes.
		w0 := b.AsInt16x16()
		w1 := c.ConcatShiftBytesRightGrouped(b, 2).AsInt16x16()
		w2 := c.ConcatShiftBytesRightGrouped(b, 4).AsInt16x16()
		w3 := c.ConcatShiftBytesRightGrouped(b, 6).AsInt16x16()
		p0 := w0.DotProductPairs(warpFilterPair(warpPhase(sx), warpPhase(sx+4*alpha)))
		p1 := w1.DotProductPairs(warpFilterPair(warpPhase(sx+alpha), warpPhase(sx+5*alpha)))
		p2 := w2.DotProductPairs(warpFilterPair(warpPhase(sx+2*alpha), warpPhase(sx+6*alpha)))
		p3 := w3.DotProductPairs(warpFilterPair(warpPhase(sx+3*alpha), warpPhase(sx+7*alpha)))
		// vphaddd twice: lane 0 ends as outputs 0..3, lane 1 as 4..7.
		sum := p0.ConcatAddPairsGrouped(p1).ConcatAddPairsGrouped(p2.ConcatAddPairsGrouped(p3))
		out := (k + 7) * warpedIntermediateColumns
		sum.Add(bias).ShiftAllRight(3).StoreArray((*[8]int32)(tmp[out : out+8]))
	}
	archsimd.ClearAVXUpperBits()
}

// warpByteIdx gathers the low byte of every dword into the low four bytes of
// its 128-bit lane.
var warpByteIdx = [32]int8{
	0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
}

// warpVertical8AVX2 computes the 8x8 vertical output block:
// clip(((2^19 + sum(c * tmp) + 2^10) >> 11) - 384). The intermediate fits
// int16, so the rows narrow exactly and pair up for vpmaddwd; the eight
// column filters of a row are transposed into four tap-pair vectors (with
// gamma = 0 every column shares one filter).
func warpVertical8AVX2(dst frame.Plane, tmp *warpTmp, i, j, rowShift, colShift, baseSY, gamma, delta int) {
	const seed = (1 << (8 + 2*filterBits - round0Bits)) + (1 << (round1Bits - 1)) - ((1<<7)+(1<<8))*(1<<round1Bits)
	bias := archsimd.BroadcastInt32x8(seed)
	var zero archsimd.Int32x8
	c255 := archsimd.BroadcastInt32x8(255)
	byteIdx := archsimd.LoadInt8x32Array(&warpByteIdx)
	// pairs[t]: the int16 rows t and t+1 interleaved, columns 0..3 in the low
	// lane and 4..7 in the high lane.
	var rows [15]archsimd.Int16x8
	for t := range rows {
		rows[t] = archsimd.LoadInt32x4Array((*[4]int32)(tmp[8*t : 8*t+4])).SaturateToInt16Concat(archsimd.LoadInt32x4Array((*[4]int32)(tmp[8*t+4 : 8*t+8])))
	}
	var pairs [14]archsimd.Int16x16
	for t := range pairs {
		pairs[t] = archsimd.Int16x16{}.SetLo(rows[t].InterleaveLo(rows[t+1])).SetHi(rows[t].InterleaveHi(rows[t+1]))
	}
	for row := 0; row < 8; row++ {
		sy := baseSY + delta*row
		// Columns c and c+4 share a ymm (low and high lane); a 4x4 dword
		// transpose per lane turns the filter rows into tap pairs.
		f0 := warpFilterPair(warpPhase(sy), warpPhase(sy+4*gamma)).AsInt32x8()
		f1 := warpFilterPair(warpPhase(sy+gamma), warpPhase(sy+5*gamma)).AsInt32x8()
		f2 := warpFilterPair(warpPhase(sy+2*gamma), warpPhase(sy+6*gamma)).AsInt32x8()
		f3 := warpFilterPair(warpPhase(sy+3*gamma), warpPhase(sy+7*gamma)).AsInt32x8()
		a0, a1 := f0.InterleaveLoGrouped(f1).AsInt64x4(), f0.InterleaveHiGrouped(f1).AsInt64x4()
		a2, a3 := f2.InterleaveLoGrouped(f3).AsInt64x4(), f2.InterleaveHiGrouped(f3).AsInt64x4()
		k01 := a0.InterleaveLoGrouped(a2).AsInt16x16()
		k23 := a0.InterleaveHiGrouped(a2).AsInt16x16()
		k45 := a1.InterleaveLoGrouped(a3).AsInt16x16()
		k67 := a1.InterleaveHiGrouped(a3).AsInt16x16()
		sum := bias.Add(pairs[row].DotProductPairs(k01)).Add(pairs[row+2].DotProductPairs(k23)).
			Add(pairs[row+4].DotProductPairs(k45)).Add(pairs[row+6].DotProductPairs(k67))
		v := sum.ShiftAllRight(round1Bits).Max(zero).Min(c255).AsUint8x32().PermuteOrZeroGrouped(byteIdx).AsUint32x8()
		px := uint64(v.GetLo().GetElem(0)) | uint64(v.GetHi().GetElem(0))<<32
		dstRow := (i+rowShift+row)*dst.Stride + j + colShift
		binary.LittleEndian.PutUint64(dst.Pix[dstRow:dstRow+8], px)
	}
	archsimd.ClearAVXUpperBits()
}
