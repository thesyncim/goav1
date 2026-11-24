// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// 8-bit single-reference warp (dav1d warp_affine_8x8, src/arm/64/mc.S): the
// same int16 15x8 intermediate as the compound block (warpCompoundH8), so the
// vertical pass multiplies the int16 rows directly instead of narrowing the
// int32 warpTmp rows for every output row. The results equal
// warpAffine8Offset's scalar formulas sample for sample.

// warpPut8x8Dispatch predicts one full 8x8 warp block into dst (row r at
// dst[r*stride:]). It reports false when the SIMD kernel does not apply; the
// caller then runs the int32 path.
func warpPut8x8Dispatch(dst []byte, stride int, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) bool {
	if !warpGoSIMDEnabled || !warpHorizResidentOffsInRange(sx4, alpha, beta) || !warpVertFullOffsInRange(sy4, gamma, delta) {
		return false
	}
	var mid warpCompoundMid
	base := (iy4-7)*ref.Stride + ix4 - 7
	if iy4 >= 7 && iy4+7 < ref.Height && ix4 >= 7 && ix4+8 <= ref.Width && base+14*ref.Stride+16 <= len(ref.Pix) {
		warpCompoundH8(&mid, &warp8FilterPadded, ref.Pix, base, ref.Stride, sx4, alpha, beta)
	} else {
		var edge [15 * 16]byte
		warpCompoundEmuEdge(&edge, ref, ix4, iy4)
		warpCompoundH8(&mid, &warp8FilterPadded, edge[:], 0, 16, sx4, alpha, beta)
	}
	if gamma == 0 {
		warpPutV8Gamma0(dst, stride, &mid, &warp8FilterPadded, sy4, delta)
	} else {
		warpPutV8(dst, stride, &mid, &warp8FilterPadded, sy4, gamma, delta)
	}
	return true
}

// warpPutSeed folds the scalar vertical constants into the accumulator:
// round2(2^19 + sum(c*(mid+4096)), 11) - 384 == (seed + sum(c*mid)) >> 11 with
// seed = 2^19 + 4096*128 + 2^10 - 384*2^11 (each filter row sums to 128, and
// subtracting a multiple of 2^11 commutes with the arithmetic shift).
const warpPutSeed = 1<<(8+2*filterBits-round0Bits) + 4096*128 + 1<<(round1Bits-1) - ((1<<7)+(1<<8))<<round1Bits

// warpPutV8 is the vertical pass with per-column filters (gamma != 0): the
// eight column filters of a row are transposed in registers into eight tap
// vectors, as dav1d's transpose_8x8b_xtl.
func warpPutV8(dst []byte, stride int, mid *warpCompoundMid, filters *[256][16]int8, sy4, gamma, delta int) {
	bias := archsimd.BroadcastInt32x4(warpPutSeed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	sy := sy4 + 64<<10 + 512
	for row := range 8 {
		f0 := archsimd.LoadInt8x16(filters[(sy>>10)&255][:])
		f1 := archsimd.LoadInt8x16(filters[((sy+gamma)>>10)&255][:])
		f2 := archsimd.LoadInt8x16(filters[((sy+2*gamma)>>10)&255][:])
		f3 := archsimd.LoadInt8x16(filters[((sy+3*gamma)>>10)&255][:])
		f4 := archsimd.LoadInt8x16(filters[((sy+4*gamma)>>10)&255][:])
		f5 := archsimd.LoadInt8x16(filters[((sy+5*gamma)>>10)&255][:])
		f6 := archsimd.LoadInt8x16(filters[((sy+6*gamma)>>10)&255][:])
		f7 := archsimd.LoadInt8x16(filters[((sy+7*gamma)>>10)&255][:])
		a01 := f0.InterleaveLo(f1).ToBits().ReshapeToUint16s()
		a23 := f2.InterleaveLo(f3).ToBits().ReshapeToUint16s()
		a45 := f4.InterleaveLo(f5).ToBits().ReshapeToUint16s()
		a67 := f6.InterleaveLo(f7).ToBits().ReshapeToUint16s()
		b03lo := a01.InterleaveLo(a23).ReshapeToUint32s()
		b03hi := a01.InterleaveHi(a23).ReshapeToUint32s()
		b47lo := a45.InterleaveLo(a67).ReshapeToUint32s()
		b47hi := a45.InterleaveHi(a67).ReshapeToUint32s()
		c01 := b03lo.InterleaveLo(b47lo).ReshapeToUint8s().BitsToInt8()
		c23 := b03lo.InterleaveHi(b47lo).ReshapeToUint8s().BitsToInt8()
		c45 := b03hi.InterleaveLo(b47hi).ReshapeToUint8s().BitsToInt8()
		c67 := b03hi.InterleaveHi(b47hi).ReshapeToUint8s().BitsToInt8()
		t0 := c01.ExtendLo8ToInt16()
		t1 := c01.HiToLo().ExtendLo8ToInt16()
		t2 := c23.ExtendLo8ToInt16()
		t3 := c23.HiToLo().ExtendLo8ToInt16()
		t4 := c45.ExtendLo8ToInt16()
		t5 := c45.HiToLo().ExtendLo8ToInt16()
		t6 := c67.ExtendLo8ToInt16()
		t7 := c67.HiToLo().ExtendLo8ToInt16()
		win := (*[64]int16)(mid[8*row : 8*row+64])
		m0 := archsimd.LoadInt16x8Array((*[8]int16)(win[0:8]))
		m1 := archsimd.LoadInt16x8Array((*[8]int16)(win[8:16]))
		m2 := archsimd.LoadInt16x8Array((*[8]int16)(win[16:24]))
		m3 := archsimd.LoadInt16x8Array((*[8]int16)(win[24:32]))
		m4 := archsimd.LoadInt16x8Array((*[8]int16)(win[32:40]))
		m5 := archsimd.LoadInt16x8Array((*[8]int16)(win[40:48]))
		m6 := archsimd.LoadInt16x8Array((*[8]int16)(win[48:56]))
		m7 := archsimd.LoadInt16x8Array((*[8]int16)(win[56:64]))
		lo := bias.Add(m0.MulWidenLo(t0)).Add(m1.MulWidenLo(t1)).Add(m2.MulWidenLo(t2)).Add(m3.MulWidenLo(t3)).
			Add(m4.MulWidenLo(t4)).Add(m5.MulWidenLo(t5)).Add(m6.MulWidenLo(t6)).Add(m7.MulWidenLo(t7))
		hi := bias.Add(m0.HiToLo().MulWidenLo(t0.HiToLo())).Add(m1.HiToLo().MulWidenLo(t1.HiToLo())).
			Add(m2.HiToLo().MulWidenLo(t2.HiToLo())).Add(m3.HiToLo().MulWidenLo(t3.HiToLo())).
			Add(m4.HiToLo().MulWidenLo(t4.HiToLo())).Add(m5.HiToLo().MulWidenLo(t5.HiToLo())).
			Add(m6.HiToLo().MulWidenLo(t6.HiToLo())).Add(m7.HiToLo().MulWidenLo(t7.HiToLo()))
		o := row * stride
		u8PackSaturate32(lo.Shift(right), hi.Shift(right)).StorePart(dst[o : o+8])
		sy += delta
	}
}

// warpLane[k] is the TBL index vector that broadcasts int16 lane k.
var warpLane = func() (t [8][16]uint8) {
	for k := range t {
		for l := 0; l < 8; l++ {
			t[k][2*l] = uint8(2 * k)
			t[k][2*l+1] = uint8(2*k + 1)
		}
	}
	return t
}()

// warpPutV8Gamma0 is warpPutV8 when every column of a row shares one filter
// (gamma == 0): the eight taps are lane broadcasts of the single filter row.
func warpPutV8Gamma0(dst []byte, stride int, mid *warpCompoundMid, filters *[256][16]int8, sy4, delta int) {
	bias := archsimd.BroadcastInt32x4(warpPutSeed)
	right := archsimd.BroadcastInt32x4(-round1Bits)
	l0, l1 := archsimd.LoadUint8x16Array(&warpLane[0]), archsimd.LoadUint8x16Array(&warpLane[1])
	l2, l3 := archsimd.LoadUint8x16Array(&warpLane[2]), archsimd.LoadUint8x16Array(&warpLane[3])
	l4, l5 := archsimd.LoadUint8x16Array(&warpLane[4]), archsimd.LoadUint8x16Array(&warpLane[5])
	l6, l7 := archsimd.LoadUint8x16Array(&warpLane[6]), archsimd.LoadUint8x16Array(&warpLane[7])
	sy := sy4 + 64<<10 + 512
	for row := range 8 {
		fb := archsimd.LoadInt8x16(filters[(sy>>10)&255][:]).ExtendLo8ToInt16().ToBits().ReshapeToUint8s()
		t0 := fb.LookupOrZero(l0).ReshapeToUint16s().BitsToInt16()
		t1 := fb.LookupOrZero(l1).ReshapeToUint16s().BitsToInt16()
		t2 := fb.LookupOrZero(l2).ReshapeToUint16s().BitsToInt16()
		t3 := fb.LookupOrZero(l3).ReshapeToUint16s().BitsToInt16()
		t4 := fb.LookupOrZero(l4).ReshapeToUint16s().BitsToInt16()
		t5 := fb.LookupOrZero(l5).ReshapeToUint16s().BitsToInt16()
		t6 := fb.LookupOrZero(l6).ReshapeToUint16s().BitsToInt16()
		t7 := fb.LookupOrZero(l7).ReshapeToUint16s().BitsToInt16()
		win := (*[64]int16)(mid[8*row : 8*row+64])
		m0 := archsimd.LoadInt16x8Array((*[8]int16)(win[0:8]))
		m1 := archsimd.LoadInt16x8Array((*[8]int16)(win[8:16]))
		m2 := archsimd.LoadInt16x8Array((*[8]int16)(win[16:24]))
		m3 := archsimd.LoadInt16x8Array((*[8]int16)(win[24:32]))
		m4 := archsimd.LoadInt16x8Array((*[8]int16)(win[32:40]))
		m5 := archsimd.LoadInt16x8Array((*[8]int16)(win[40:48]))
		m6 := archsimd.LoadInt16x8Array((*[8]int16)(win[48:56]))
		m7 := archsimd.LoadInt16x8Array((*[8]int16)(win[56:64]))
		lo := bias.Add(m0.MulWidenLo(t0)).Add(m1.MulWidenLo(t1)).Add(m2.MulWidenLo(t2)).Add(m3.MulWidenLo(t3)).
			Add(m4.MulWidenLo(t4)).Add(m5.MulWidenLo(t5)).Add(m6.MulWidenLo(t6)).Add(m7.MulWidenLo(t7))
		hi := bias.Add(m0.HiToLo().MulWidenLo(t0.HiToLo())).Add(m1.HiToLo().MulWidenLo(t1.HiToLo())).
			Add(m2.HiToLo().MulWidenLo(t2.HiToLo())).Add(m3.HiToLo().MulWidenLo(t3.HiToLo())).
			Add(m4.HiToLo().MulWidenLo(t4.HiToLo())).Add(m5.HiToLo().MulWidenLo(t5.HiToLo())).
			Add(m6.HiToLo().MulWidenLo(t6.HiToLo())).Add(m7.HiToLo().MulWidenLo(t7.HiToLo()))
		o := row * stride
		u8PackSaturate32(lo.Shift(right), hi.Shift(right)).StorePart(dst[o : o+8])
		sy += delta
	}
}
