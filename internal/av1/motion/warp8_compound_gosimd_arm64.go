// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// 8-bit compound warp (dav1d warp_affine_8x8t, src/arm/64/mc.S). The
// horizontal pass writes the 15x8 intermediate as int16 rows, and the vertical
// pass reads those rows straight into its widening multiplies, so the int32
// warpTmp round trip of the non-compound path is gone. The results equal
// PredictWarpedCompoundToConvBuf's scalar formulas sample for sample.

// warpCompoundMid is the 15x8 int16 intermediate without the +4096 offset of
// the scalar warpTmp (see warpCompoundV8 for where it goes).
type warpCompoundMid = [15 * 8]int16

// warpCompound8x8Dispatch predicts one full 8x8 compound warp block into out
// (row r at out[r*stride:]). It reports false when the SIMD kernel does not
// apply; the caller then runs the scalar block.
func warpCompound8x8Dispatch(out []uint16, stride int, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, gamma, delta int) bool {
	if !warpGoSIMDEnabled || !warpHorizResidentOffsInRange(sx4, alpha, beta) || !warpVertFullOffsInRange(sy4, gamma, delta) {
		return false
	}
	var mid warpCompoundMid
	// The resident window also needs its sixteenth byte on the last row
	// inside ref.Pix (it is loaded, never used); otherwise go through the edge
	// copy, which reads the same samples.
	base := (iy4-7)*ref.Stride + ix4 - 7
	if iy4 >= 7 && iy4+7 < ref.Height && ix4 >= 7 && ix4+8 <= ref.Width && base+14*ref.Stride+16 <= len(ref.Pix) {
		warpCompoundH8(&mid, &warp8FilterPadded, ref.Pix, base, ref.Stride, sx4, alpha, beta)
	} else {
		var edge [15 * 16]byte
		warpCompoundEmuEdge(&edge, ref, ix4, iy4)
		warpCompoundH8(&mid, &warp8FilterPadded, edge[:], 0, 16, sx4, alpha, beta)
	}
	warpCompoundV8(out, stride, &mid, &warp8FilterPadded, sy4, gamma, delta)
	return true
}

// warpCompoundEmuEdge copies the 15x16 source window at (ix4-7, iy4-7) with
// the coordinates clamped to the plane (dav1d emu_edge), which is exactly the
// sample the scalar edge path reads for every tap.
func warpCompoundEmuEdge(edge *[15 * 16]byte, ref frame.Plane, ix4, iy4 int) {
	x0 := ix4 - 7
	inside := x0 >= 0 && x0+16 <= ref.Width
	for r := range 15 {
		row := ref.Pix[clampInt(iy4-7+r, 0, ref.Height-1)*ref.Stride:]
		dst := edge[r*16 : r*16+16]
		if inside {
			copy(dst, row[x0:x0+16])
			continue
		}
		for x := range dst {
			dst[x] = row[clampInt(x0+x, 0, ref.Width-1)]
		}
	}
}

// warpCompoundH8 computes mid[k] = (sum(s*c) + 4) >> 3 for the 15 source rows
// starting at src[base] with the samples centred by -128. Every warp filter
// row sums to 128, so the centring removes 2^14 from the sum and the scalar's
// 2^14 offset cancels it; the scalar's +4096 after the shift is added back in
// the vertical accumulator.
func warpCompoundH8(mid *warpCompoundMid, filters *[256][16]int8, src []byte, base, stride, sx4, alpha, beta int) {
	centre := archsimd.BroadcastUint8x16(128)
	round := archsimd.BroadcastInt16x8(4)
	right := archsimd.BroadcastInt16x8(-3)
	// The filter row is 64 + round2(sx, 10); the 64 rides in the phase.
	sx := sx4 - 3*beta + 64<<10 + 512
	for k := range 15 {
		o := base + k*stride
		a := archsimd.LoadUint8x16(src[o : o+16]).Xor(centre)
		f0 := archsimd.LoadInt8x16(filters[(sx>>10)&255][:])
		f1 := archsimd.LoadInt8x16(filters[((sx+alpha)>>10)&255][:])
		f2 := archsimd.LoadInt8x16(filters[((sx+2*alpha)>>10)&255][:])
		f3 := archsimd.LoadInt8x16(filters[((sx+3*alpha)>>10)&255][:])
		f4 := archsimd.LoadInt8x16(filters[((sx+4*alpha)>>10)&255][:])
		f5 := archsimd.LoadInt8x16(filters[((sx+5*alpha)>>10)&255][:])
		f6 := archsimd.LoadInt8x16(filters[((sx+6*alpha)>>10)&255][:])
		f7 := archsimd.LoadInt8x16(filters[((sx+7*alpha)>>10)&255][:])
		p0 := a.BitsToInt8().MulWidenLo(f0)
		p1 := a.ConcatShiftBytesRight(a, 1).BitsToInt8().MulWidenLo(f1)
		p2 := a.ConcatShiftBytesRight(a, 2).BitsToInt8().MulWidenLo(f2)
		p3 := a.ConcatShiftBytesRight(a, 3).BitsToInt8().MulWidenLo(f3)
		p4 := a.ConcatShiftBytesRight(a, 4).BitsToInt8().MulWidenLo(f4)
		p5 := a.ConcatShiftBytesRight(a, 5).BitsToInt8().MulWidenLo(f5)
		p6 := a.ConcatShiftBytesRight(a, 6).BitsToInt8().MulWidenLo(f6)
		p7 := a.ConcatShiftBytesRight(a, 7).BitsToInt8().MulWidenLo(f7)
		low := p0.ConcatAddPairs(p1).ConcatAddPairs(p2.ConcatAddPairs(p3))
		high := p4.ConcatAddPairs(p5).ConcatAddPairs(p6.ConcatAddPairs(p7))
		low.ConcatAddPairs(high).Add(round).Shift(right).StoreArray((*[8]int16)(mid[8*k : 8*k+8]))
		sx += beta
	}
}

// warpCompoundV8 computes the 8x8 CONV_BUF block:
// (2^19 + sum(c*(mid+4096)) + 2^6) >> 7 = (2^20 + 2^6 + sum(c*mid)) >> 7, as
// each filter row sums to 128. The eight column filters of a row are
// transposed in registers into eight tap vectors. The filter table comes in
// as a parameter so its address stays in a register across the row loop.
func warpCompoundV8(out []uint16, stride int, mid *warpCompoundMid, filters *[256][16]int8, sy4, gamma, delta int) {
	const seed = 1<<(8+2*filterBits-compoundRound0Bits) + 4096*128 + 1<<(compoundRound1Bits-1)
	bias := archsimd.BroadcastInt32x4(seed)
	right := archsimd.BroadcastInt32x4(-compoundRound1Bits)
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
		// The taps of output row r read mid rows r..r+7.
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
		n := lo.Shift(right).TruncToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()
		n = n.SetElem(1, hi.Shift(right).TruncToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))
		o := row * stride
		n.ToBits().ReshapeToUint16s().StoreArray((*[8]uint16)(out[o : o+8]))
		sy += delta
	}
}
