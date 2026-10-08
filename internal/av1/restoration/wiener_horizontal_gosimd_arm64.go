// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// Go-native SIMD Wiener horizontal passes for arm64.
//
// The Wiener filter is symmetric (f0==f6, f1==f5, f2==f4, enforced by
// validWienerInfo and re-checked here), so with the libaom center reapplication
// s3<<WienerFilterBits folded into the center tap (f3' = f3 + 128) each output is
//
//	sum = seed + f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3'*s3
//
// where seed = offset + 1<<(round0-1) also folds the rounding bias, so the trailing
// shift is a plain arithmetic right shift. The symmetric pair sums fit a positive
// int16 lane (u8: <= 510; u16 samples <= 4095: <= 8190), so each term is one widening
// product and the 8 products per 8 outputs of the plain form become 4.
//
// Window construction: one 16-byte load per 8-output group, then ConcatShiftBytesRight
// on the reshaped lanes derives the shifted windows in-register. The last load of
// each row reaches two samples past the three-sample border; callers establish that
// pad with the wrappers' CanRun checks. Every memory access is a bounds-checked
// slice index, so no pointer ever leaves its allocation.

// wienerHorizontalSIMD is the Go-native SIMD form of wienerHorizontal for the uint16
// sample domain. It applies the reference validity scan first (every window sample
// must be <= max, else the call fails with false), then runs the kernel.
func wienerHorizontalSIMD(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) bool {
	if !wienerHorizontalU16SIMDCanRun(len(src), srcStride, srcOrigin, width, height, filter) {
		return wienerHorizontal(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
	}
	for row := -WienerHalfwin; row < height+WienerHalfwin; row++ {
		srcStart := srcOrigin + row*srcStride - WienerHalfwin
		for _, s := range src[srcStart : srcStart+width+2*WienerHalfwin] {
			if s > max {
				return false
			}
		}
	}
	wienerHorizontalSIMDKernel(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
	return true
}

// wienerHorizontalSIMDTrusted is the uint16 SIMD horizontal pass without the
// per-sample validity scan, used by the decoder-owned trusted entry.
func wienerHorizontalSIMDTrusted(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	if !wienerHorizontalU16SIMDCanRun(len(src), srcStride, srcOrigin, width, height, filter) {
		wienerHorizontalTrusted(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
		return
	}
	wienerHorizontalSIMDKernel(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
}

// wienerHorizontalU16SIMDCanRun reports whether the uint16 SIMD horizontal kernel
// may run: a width of whole 8-column groups, a symmetric filter, and the two-sample
// trailing pad that the 16-byte window load reads.
func wienerHorizontalU16SIMDCanRun(srcLen int, srcStride int, srcOrigin int, width int, height int, filter WienerFilter) bool {
	if width < 8 || width%8 != 0 || !wienerFilterSymmetric(filter) {
		return false
	}
	loadWidth, ok := checkedAdd(width, 2)
	return ok && borderedBlockFits(srcLen, srcStride, srcOrigin, loadWidth, height, WienerHalfwin, WienerHalfwin)
}

// wienerHorizontalSIMDKernel is the uint16 SIMD horizontal pass. The caller must
// have checked wienerHorizontalU16SIMDCanRun.
func wienerHorizontalSIMDKernel(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	limit := int32(1) << (bitDepth + 1 + WienerFilterBits - round0)
	offset := int32(1) << (bitDepth + WienerFilterBits - 1)
	seedV := archsimd.BroadcastInt32x4(offset + roundBias(round0))
	shV := archsimd.BroadcastInt32x4(-int32(round0))
	zero := archsimd.BroadcastInt32x4(0)
	maxV := archsimd.BroadcastUint16x8(uint16(limit - 1))
	g0 := archsimd.BroadcastInt16x8(filter[0])
	g1 := archsimd.BroadcastInt16x8(filter[1])
	g2 := archsimd.BroadcastInt16x8(filter[2])
	// The center tap absorbs the s3<<WienerFilterBits center reapplication.
	g3 := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))
	hg0, hg1 := g0.HiToLo(), g1.HiToLo()
	hg2, hg3 := g2.HiToLo(), g3.HiToLo()

	rows := height + 2*WienerHalfwin
	for row := 0; row < rows; row++ {
		sBase := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		dBase := row * width
		for col := 0; col < width; col += 8 {
			v0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[sBase+col])))   // s0..s7
			v1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Pointer(&src[sBase+col+8]))) // s8..s15
			lb := v0.ReshapeToUint8s()
			hb := v1.ReshapeToUint8s()
			w1 := hb.ConcatShiftBytesRight(lb, 2*1).ReshapeToUint16s().BitsToInt16() // s1..s8
			w2 := hb.ConcatShiftBytesRight(lb, 2*2).ReshapeToUint16s().BitsToInt16() // s2..s9
			w3 := hb.ConcatShiftBytesRight(lb, 2*3).ReshapeToUint16s().BitsToInt16() // s3..s10
			w4 := hb.ConcatShiftBytesRight(lb, 2*4).ReshapeToUint16s().BitsToInt16() // s4..s11
			w5 := hb.ConcatShiftBytesRight(lb, 2*5).ReshapeToUint16s().BitsToInt16() // s5..s12
			w6 := hb.ConcatShiftBytesRight(lb, 2*6).ReshapeToUint16s().BitsToInt16() // s6..s13
			a := v0.BitsToInt16().Add(w6)                                            // s0+s6 <= 8190
			b := w1.Add(w5)                                                          // s1+s5
			c := w2.Add(w4)                                                          // s2+s4
			lo := seedV.Add(a.MulWidenLo(g0)).Add(b.MulWidenLo(g1)).
				Add(c.MulWidenLo(g2)).Add(w3.MulWidenLo(g3))
			hi := seedV.Add(a.HiToLo().MulWidenLo(hg0)).Add(b.HiToLo().MulWidenLo(hg1)).
				Add(c.HiToLo().MulWidenLo(hg2)).Add(w3.HiToLo().MulWidenLo(hg3))
			lo = lo.Shift(shV)
			hi = hi.Shift(shV)
			out := restorationSaturateInt32PairToUint16(lo.Max(zero), hi.Max(zero)).Min(maxV)
			out.StoreArray((*[8]uint16)(unsafe.Pointer(&temp[dBase+col])))
		}
	}
}

// wienerHorizontalU8PackedTaps scales the four symmetric taps so the high 16
// bits of each packed accumulator are the rounded output. The bound ensures
// that the even and odd column sums both fit a signed 32-bit lane.
func wienerHorizontalU8PackedTaps(filter WienerFilter) (t0, t1, t2, t3, seedA, seedO uint32, ok bool) {
	f0, f1, f2 := int64(filter[0]), int64(filter[1]), int64(filter[2])
	f3 := int64(filter[3]) + (1 << WienerFilterBits)
	abs := func(x int64) int64 {
		if x < 0 {
			return -x
		}
		return x
	}
	s := int64(1) << (16 - WienerRound0Bits)
	c := int64(1<<(8+WienerFilterBits-1)) + int64(roundBias(WienerRound0Bits))
	if s*(510*(abs(f0)+abs(f1)+abs(f2))+255*abs(f3)+c) >= 1<<31 {
		return 0, 0, 0, 0, 0, 0, false
	}
	seedO = uint32(s * c)
	seedA = seedO + seedO<<16
	return uint32(s * f0), uint32(s * f1), uint32(s * f2), uint32(s * f3), seedA, seedO, true
}

// wienerHorizontalU8SIMD widens the byte window once, then computes adjacent
// columns with packed 32-bit multiply-accumulates. The odd column accumulator
// removes its contribution from the even column's high half. Unrepresentable
// filters and short buffers use the scalar reference.
func wienerHorizontalU8SIMD(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	t0, t1, t2, t3, seedA, seedO, ok := wienerHorizontalU8PackedTaps(filter)
	if round0 != WienerRound0Bits || !wienerFilterSymmetric(filter) || !ok ||
		!wienerHorizontalU8SIMDCanRun(len(src), srcStride, srcOrigin, width, height) ||
		!blockFits(len(temp), width, width, height+2*WienerHalfwin) {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	t0V, t1V := archsimd.BroadcastUint32x4(t0), archsimd.BroadcastUint32x4(t1)
	t2V, t3V := archsimd.BroadcastUint32x4(t2), archsimd.BroadcastUint32x4(t3)
	seedAV, seedOV := archsimd.BroadcastUint32x4(seedA), archsimd.BroadcastUint32x4(seedO)
	negK := archsimd.BroadcastUint32x4(0xffff0000)
	shr16 := archsimd.BroadcastInt32x4(-16)
	zero16 := archsimd.BroadcastInt16x8(0)
	max16 := archsimd.BroadcastInt16x8(8191)

	rows := height + 2*WienerHalfwin
	w16 := width &^ 15
	sp := unsafe.Pointer(unsafe.SliceData(src))
	tp := unsafe.Pointer(unsafe.SliceData(temp))
	for row := 0; row < rows; row++ {
		sBase := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		dBase := row * width
		for col := 0; col < w16; col += 16 {
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, sBase+col)))   // s0..s15
			v1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, sBase+col+8))) // s8..s23
			l0 := v0.ExtendLo8ToUint16()                                                // s0..s7  (u16)
			m0 := v0.HiToLo().ExtendLo8ToUint16()                                       // s8..s15 (u16)
			h1 := v1.HiToLo().ExtendLo8ToUint16()                                       // s16..s23 (u16)
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			hb := h1.ReshapeToUint8s()

			// Group A: outputs col..col+7, windows over s0..s13.
			w1a := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()  // s1..s8
			w2a := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()  // s2..s9
			w3a := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()  // s3..s10
			w4a := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()  // s4..s11
			w5a := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s() // s5..s12
			w6a := mb.ConcatShiftBytesRight(lb, 12).ReshapeToUint16s() // s6..s13
			aA := l0.Add(w6a)
			bA := w1a.Add(w5a)
			cA := w2a.Add(w4a)
			dA := w3a
			pA0 := aA.ReshapeToUint32s()
			pA1 := bA.ReshapeToUint32s()
			pA2 := cA.ReshapeToUint32s()
			pA3 := dA.ReshapeToUint32s()
			accA := pA3.MulAdd(t3V, seedAV)
			accA = pA2.MulAdd(t2V, accA)
			accA = pA1.MulAdd(t1V, accA)
			accA = pA0.MulAdd(t0V, accA)
			oddA := pA3.Shift(shr16).MulAdd(t3V, seedOV)
			oddA = pA2.Shift(shr16).MulAdd(t2V, oddA)
			oddA = pA1.Shift(shr16).MulAdd(t1V, oddA)
			oddA = pA0.Shift(shr16).MulAdd(t0V, oddA)
			evenA := oddA.MulAdd(negK, accA)
			outA := evenA.ReshapeToUint16s().InterleaveOdd(oddA.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			outA.StoreArray((*[8]uint16)(unsafe.Add(tp, (dBase+col)*2)))

			// Group B: outputs col+8..col+15, windows over s8..s21.
			w1b := hb.ConcatShiftBytesRight(mb, 2).ReshapeToUint16s()  // s9..s16
			w2b := hb.ConcatShiftBytesRight(mb, 4).ReshapeToUint16s()  // s10..s17
			w3b := hb.ConcatShiftBytesRight(mb, 6).ReshapeToUint16s()  // s11..s18
			w4b := hb.ConcatShiftBytesRight(mb, 8).ReshapeToUint16s()  // s12..s19
			w5b := hb.ConcatShiftBytesRight(mb, 10).ReshapeToUint16s() // s13..s20
			w6b := hb.ConcatShiftBytesRight(mb, 12).ReshapeToUint16s() // s14..s21
			aB := m0.Add(w6b)
			bB := w1b.Add(w5b)
			cB := w2b.Add(w4b)
			dB := w3b
			pB0 := aB.ReshapeToUint32s()
			pB1 := bB.ReshapeToUint32s()
			pB2 := cB.ReshapeToUint32s()
			pB3 := dB.ReshapeToUint32s()
			accB := pB3.MulAdd(t3V, seedAV)
			accB = pB2.MulAdd(t2V, accB)
			accB = pB1.MulAdd(t1V, accB)
			accB = pB0.MulAdd(t0V, accB)
			oddB := pB3.Shift(shr16).MulAdd(t3V, seedOV)
			oddB = pB2.Shift(shr16).MulAdd(t2V, oddB)
			oddB = pB1.Shift(shr16).MulAdd(t1V, oddB)
			oddB = pB0.Shift(shr16).MulAdd(t0V, oddB)
			evenB := oddB.MulAdd(negK, accB)
			outB := evenB.ReshapeToUint16s().InterleaveOdd(oddB.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			outB.StoreArray((*[8]uint16)(unsafe.Add(tp, (dBase+col+8)*2)))
		}
		if w16 != width { // trailing width%16 == 8 group
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(sp, sBase+w16))) // s0..s15 (needs s0..s13)
			l0 := v0.ExtendLo8ToUint16()
			m0 := v0.HiToLo().ExtendLo8ToUint16()
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			w1 := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()
			w2 := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()
			w3 := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()
			w4 := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()
			w5 := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s()
			w6 := mb.ConcatShiftBytesRight(lb, 12).ReshapeToUint16s()
			a := l0.Add(w6)
			b := w1.Add(w5)
			c := w2.Add(w4)
			d := w3
			p0 := a.ReshapeToUint32s()
			p1 := b.ReshapeToUint32s()
			p2 := c.ReshapeToUint32s()
			p3 := d.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			acc = p0.MulAdd(t0V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			odd = p0.Shift(shr16).MulAdd(t0V, odd)
			even := odd.MulAdd(negK, acc)
			out := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			out.StoreArray((*[8]uint16)(unsafe.Add(tp, (dBase+w16)*2)))
		}
	}
}
