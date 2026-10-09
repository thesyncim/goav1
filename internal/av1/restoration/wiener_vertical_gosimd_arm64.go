// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// Packed-pair Wiener vertical passes for arm64.
//
// The official archsimd API has no widening multiply-accumulate (SMLAL) and no
// high-half widening multiply, so the hand asm's widen+MAC cannot be copied op
// for op. Instead the kernels never widen at all:
//
//   - Symmetric row pairs are summed in uint16 lanes. A temp sample is at most
//     32767 (the horizontal pass clamps to [0,maxClamp], maxClamp <= 32767), so
//     a pair sum is at most 65534 and never carries out of its 16-bit lane.
//   - Each 32-bit lane of a pair vector therefore holds even + 65536*odd for an
//     adjacent (even, odd) column pair. One 32-bit multiply-accumulate (MLA)
//     per tap produces A = E + 65536*O (mod 2^32), where E and O are the even and
//     odd column sums: the even sum exact up to a 65536*O pollution term.
//   - The odd column values are extracted with one logical shift (>>16) and
//     multiply-accumulated separately into O. One more MLA, A - 65536*O, cancels
//     the pollution and leaves the exact even sum.
//   - Taps are pre-scaled by s = 2^(16-round1) and the seed folds -offset, the
//     rounding bias 2^(round1-1) and the center reapplication's offset, so the
//     high 16 bits of each 32-bit accumulator are exactly
//     floor((sum - offset + bias) / 2^round1) = roundPowerOfTwo(sum, round1).
//     One TRN2 gathers those high halves of the even and odd accumulators into
//     column order, replacing the asm's shift + two narrows.
//
// Per eight output columns that is 3 pair adds + 4 odd shifts + 9 MLAs + 1 TRN2,
// against the asm's 7 (u16) or 3 (u8) adds and 14 or 8 SMLALs plus its shift and
// narrow tail. A sliding seven-row window loads each temp row once per column
// strip instead of seven times.
//
// wienerVerticalPackedTaps validates that the packed form is exact and returns
// the scaled taps and seeds. Every intermediate value must fit int32 before
// wrapping: s*|sum - offset + bias| <= s*(32767*sum|f| + |offset - bias|) < 2^31,
// which holds for every valid Wiener filter at bit depths 8, 10 and 12. Filters
// that are not mirrored, or whose taps or rounding would break the bound, report
// ok=false and the caller takes the scalar reference.
func wienerVerticalPackedTaps(filter WienerFilter, bitDepth int, round1 int) (t0, t1, t2, t3, seedA, seedO uint32, ok bool) {
	if !wienerFilterSymmetric(filter) || round1 < 1 || round1 > 16 || bitDepth < 0 || bitDepth+round1-1 > 30 {
		return 0, 0, 0, 0, 0, 0, false
	}
	abs := func(v int64) int64 {
		if v < 0 {
			return -v
		}
		return v
	}
	f0 := int64(filter[0])
	f1 := int64(filter[1])
	f2 := int64(filter[2])
	f3 := int64(filter[3]) + (1 << WienerFilterBits)
	c := int64(roundBias(round1)) - int64(1)<<(bitDepth+round1-1)
	s := int64(1) << (16 - round1)
	bound := s * (32767*(2*abs(f0)+2*abs(f1)+2*abs(f2)+abs(f3)) + abs(c))
	if bound >= 1<<31 {
		return 0, 0, 0, 0, 0, 0, false
	}
	seedO = uint32(s * c)
	// A - 65536*O must equal s*(E + c): A carries seedA, O carries seedO.
	seedA = seedO + seedO<<16
	return uint32(s * f0), uint32(s * f1), uint32(s * f2), uint32(s * f3), seedA, seedO, true
}

// wienerVerticalFootprintFits reports whether the vertical pass's whole read
// footprint (height+6 temp rows of width samples) and write footprint (height
// dst rows of width samples) are resident, with overflow-checked arithmetic, so
// the SIMD kernels can address both buffers without per-access bounds checks.
func wienerVerticalFootprintFits(tempLen int, tempStride int, dstLen int, dstStride int, width int, height int) bool {
	tempRows, ok := checkedAdd(height, 2*WienerHalfwin)
	return ok && height > 0 &&
		blockFits(tempLen, tempStride, width, tempRows) &&
		blockFits(dstLen, dstStride, width, height)
}

// wienerVerticalU8SIMD is the Go-native SIMD form of the 8-bit Wiener vertical
// pass over the horizontal-pass intermediate, rounded by round1 and clamped to
// uint8 (see the packed-pair notes above). The high-half results are signed
// int16 values and SQXTUN's unsigned saturation is exactly clampInt32(x, 0, 255).
//
// Sixteen columns per strip (one 16-byte store per row); a trailing strip of
// eight columns is placed at width-8 when fewer than sixteen remain, recomputing
// any overlapped columns with identical values. Widths below 8, footprints that
// do not fit, and filters the packed form cannot represent take the scalar
// reference.
func wienerVerticalU8SIMD(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	t0, t1, t2, t3, seedA, seedO, ok := wienerVerticalPackedTaps(filter, 8, round1)
	if width < 8 || !ok || !wienerVerticalFootprintFits(len(temp), tempStride, len(dst), dstStride, width, height) {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	t0V := archsimd.BroadcastUint32x4(t0)
	t1V := archsimd.BroadcastUint32x4(t1)
	t2V := archsimd.BroadcastUint32x4(t2)
	t3V := archsimd.BroadcastUint32x4(t3)
	seedAV := archsimd.BroadcastUint32x4(seedA)
	seedOV := archsimd.BroadcastUint32x4(seedO)
	negK := archsimd.BroadcastUint32x4(0xffff0000) // -65536
	shr16 := archsimd.BroadcastInt32x4(-16)

	// The footprint check above proves every load and store below is in bounds,
	// so the hot loops address both buffers by byte offset from their bases.
	tp := unsafe.Pointer(unsafe.SliceData(temp))
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	ts := tempStride * 2

	col := 0
	for ; col+16 <= width; col += 16 {
		t := col * 2
		d := col
		a0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		b0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t+16)))
		t += ts
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		b1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t+16)))
		t += ts
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		b2 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t+16)))
		t += ts
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		b3 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t+16)))
		t += ts
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		b4 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t+16)))
		t += ts
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		b5 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t+16)))
		t += ts
		for row := 0; row < height; row++ {
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
			b6 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t+16)))
			t += ts

			pa0 := a0.Add(a6).ReshapeToUint32s()
			pa1 := a1.Add(a5).ReshapeToUint32s()
			pa2 := a2.Add(a4).ReshapeToUint32s()
			pa3 := a3.ReshapeToUint32s()
			accA := pa3.MulAdd(t3V, seedAV)
			accA = pa2.MulAdd(t2V, accA)
			accA = pa1.MulAdd(t1V, accA)
			accA = pa0.MulAdd(t0V, accA)
			oddA := pa3.Shift(shr16).MulAdd(t3V, seedOV)
			oddA = pa2.Shift(shr16).MulAdd(t2V, oddA)
			oddA = pa1.Shift(shr16).MulAdd(t1V, oddA)
			oddA = pa0.Shift(shr16).MulAdd(t0V, oddA)
			evenA := oddA.MulAdd(negK, accA)
			va := evenA.ReshapeToUint16s().InterleaveOdd(oddA.ReshapeToUint16s()).BitsToInt16()

			pb0 := b0.Add(b6).ReshapeToUint32s()
			pb1 := b1.Add(b5).ReshapeToUint32s()
			pb2 := b2.Add(b4).ReshapeToUint32s()
			pb3 := b3.ReshapeToUint32s()
			accB := pb3.MulAdd(t3V, seedAV)
			accB = pb2.MulAdd(t2V, accB)
			accB = pb1.MulAdd(t1V, accB)
			accB = pb0.MulAdd(t0V, accB)
			oddB := pb3.Shift(shr16).MulAdd(t3V, seedOV)
			oddB = pb2.Shift(shr16).MulAdd(t2V, oddB)
			oddB = pb1.Shift(shr16).MulAdd(t1V, oddB)
			oddB = pb0.Shift(shr16).MulAdd(t0V, oddB)
			evenB := oddB.MulAdd(negK, accB)
			vb := evenB.ReshapeToUint16s().InterleaveOdd(oddB.ReshapeToUint16s()).BitsToInt16()

			restorationSaturateInt16PairToUint8(va, vb).StoreArray((*[16]uint8)(unsafe.Add(dp, d)))
			d += dstStride
			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
			b0, b1, b2, b3, b4, b5 = b1, b2, b3, b4, b5, b6

		}
	}
	// Trailing columns: eight-wide strips, the last one shifted left to end at
	// width so it overlaps already-written columns instead of running past them.
	for ; col < width; col += 8 {
		c := min(col, width-8)
		t := c * 2
		a0 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		t += ts
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		t += ts
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		t += ts
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		t += ts
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		t += ts
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
		t += ts
		d := c
		for row := 0; row < height; row++ {
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(tp, t)))
			t += ts

			p0 := a0.Add(a6).ReshapeToUint32s()
			p1 := a1.Add(a5).ReshapeToUint32s()
			p2 := a2.Add(a4).ReshapeToUint32s()
			p3 := a3.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			acc = p0.MulAdd(t0V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			odd = p0.Shift(shr16).MulAdd(t0V, odd)
			even := odd.MulAdd(negK, acc)
			v := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16()
			*(*uint64)(unsafe.Add(dp, d)) = v.SaturateToUint8().ReshapeToUint64s().GetElem(0)
			d += dstStride

			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
		}
	}
}
