// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "simd/archsimd"

// Go-native SIMD directional interpolation kernels for arm64. Each computes
// roundPowerOfTwo(p0*(32-shift)+p1*shift, 5) for eight 8-bit edge pairs. The
// operands are 8-bit samples, so the weighted sum is at most 255*32+16 and the
// whole computation stays in uint16 lanes, exactly as the scalar reference.

// shr5U16 is the rounding shift by 5 as a per-lane shift vector.
var shr5U16 = archsimd.BroadcastInt16x8(-5)

// dirInterpolate8 returns roundPowerOfTwo(p0*w0+p1*w1, 5) for eight lanes, where
// w0 = 32-shift and w1 = shift.
func dirInterpolate8(p0, p1, w0, w1, round archsimd.Uint16x8) archsimd.Uint16x8 {
	return p0.Mul(w0).Add(p1.Mul(w1)).Add(round).Shift(shr5U16)
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
		out := dirInterpolate8(p0, p1, w0, w1, round).SaturateToUint8().ReshapeToUint64s().GetElem(0)
		for k := 0; k < 8; k++ {
			dst[(i+k)*stride] = byte(out >> (8 * k))
		}
	}
	if chunks < count {
		dirLeftCol8PureGo(dst[chunks*stride:], stride, ref[chunks:], shift, count-chunks)
	}
}
