// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// wienerVerticalSIMD is the Go-native SIMD uint16 Wiener vertical pass for arm64,
// using the packed-pair form described in wiener_vertical_gosimd_arm64.go: the
// symmetric row pairs are summed in uint16 lanes (a temp sample is at most 32767,
// so a pair never exceeds 65534), multiply-accumulated as packed 32-bit lanes
// with taps pre-scaled by 2^(16-round1), and the high 16 bits of the corrected
// even and odd accumulators are exactly roundPowerOfTwo(sum - offset, round1).
// Those int16 results are clamped to [0,max] with one max and one min.
//
// Sixteen columns per strip; a trailing strip of eight columns is placed at
// width-8 when fewer than sixteen remain, recomputing any overlapped columns with
// identical values. Widths below 8, max above 32767, and filters the packed form
// cannot represent (not mirrored, or taps/rounding outside the int32 bound) take
// the scalar reference.
func wienerVerticalSIMD(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	t0, t1, t2, t3, seedA, seedO, ok := wienerVerticalPackedTaps(filter, bitDepth, round1)
	if width < 8 || max > 32767 || !ok || !wienerVerticalFootprintFits(len(temp), tempStride, len(dst), dstStride, width, height) {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
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
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(int16(max))

	tp := unsafe.Pointer(unsafe.SliceData(temp))
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	ts := tempStride * 2
	ds := dstStride * 2
	col := 0
	for ; col+16 <= width; col += 16 {
		t := col * 2
		d := col * 2
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
			va.Max(zero).Min(maxV).ToBits().StoreArray((*[8]uint16)(unsafe.Add(dp, d)))

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
			vb.Max(zero).Min(maxV).ToBits().StoreArray((*[8]uint16)(unsafe.Add(dp, d+16)))
			d += ds
			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
			b0, b1, b2, b3, b4, b5 = b1, b2, b3, b4, b5, b6
		}
	}
	// Trailing columns: eight-wide strips, the last one shifted left to end at
	// width so it overlaps already-written columns instead of running past them.
	for ; col < width; col += 8 {
		c := min(col, width-8)
		t := c * 2
		d := c * 2
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
			v.Max(zero).Min(maxV).ToBits().StoreArray((*[8]uint16)(unsafe.Add(dp, d)))
			d += ds
			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
		}
	}
}
