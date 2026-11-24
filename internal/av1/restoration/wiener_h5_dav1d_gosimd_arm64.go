// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "simd/archsimd"

// wienerHorizontalU8Dav1d5 is the five-tap 8-bit horizontal Wiener pass in
// dav1d's wiener_filter5_h_8bpc_neon shape (src/arm/64/looprestoration.S):
// the symmetric tap pairs are summed in uint16, the products accumulate in
// wrapping int16 lanes, and the centre sample's 128*s3 plus the offsets join
// through one saturating add, whose clamp is libaom's [0, 8191] clamp:
//
//	T = 2^14 + 128*s3 + P + 4,  P = f3*s3 + f2*(s2+s4) + f1*(s1+s5)
//	out = clamp(T >> 3, 0, 8191) == (sat16(P + (128*s3 - 16380)) >> 3) + 4096
//
// (T - 32768 is what the saturating add sees; 32768 is a multiple of 8.) The
// identity needs P itself to fit int16, which wienerU8Dav1dTapsFit checks;
// every filter AV1 can code passes. Eight outputs per step from one 16-byte
// load, as bounds-checked row windows.
func wienerHorizontalU8Dav1d5(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, temp []uint16) {
	f1 := archsimd.BroadcastInt16x8(int16(filter[1]))
	f2 := archsimd.BroadcastInt16x8(int16(filter[2]))
	f3 := archsimd.BroadcastInt16x8(int16(filter[3]))
	off := archsimd.BroadcastInt16x8(1<<14 - 1<<2)
	add := archsimd.BroadcastInt16x8(4096)
	shl7 := archsimd.BroadcastInt16x8(7)
	shr3 := archsimd.BroadcastInt16x8(-3)
	rows := height + 2*WienerHalfwin
	for row := range rows {
		base := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		s := src[base : base+width+8]
		d := temp[row*width : row*width+width]
		for col := 0; col+8 <= width; col += 8 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(s[col : col+16]))
			lo := a.ExtendLo8ToUint16().ReshapeToUint8s()
			hi := a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			w1 := hi.ConcatShiftBytesRight(lo, 2).ReshapeToUint16s()
			w2 := hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s()
			w3 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16()
			w4 := hi.ConcatShiftBytesRight(lo, 8).ReshapeToUint16s()
			w5 := hi.ConcatShiftBytesRight(lo, 10).ReshapeToUint16s()
			p := w3.Mul(f3)
			p = w2.Add(w4).BitsToInt16().MulAdd(f2, p)
			p = w1.Add(w5).BitsToInt16().MulAdd(f1, p)
			c := w3.Shift(shl7).Sub(off)
			p.AddSaturated(c).Shift(shr3).Add(add).ToBits().StoreArray((*[8]uint16)(d[col : col+8]))
		}
	}
}

// wienerU8Dav1dTapsFit reports whether the int16 product sum of
// wienerHorizontalU8Dav1d5 cannot wrap for any 8-bit input.
func wienerU8Dav1dTapsFit(filter WienerFilter) bool {
	pos, neg := 0, 0
	for _, t := range [...]struct{ c, n int }{{int(filter[1]), 2}, {int(filter[2]), 2}, {int(filter[3]), 1}} {
		if t.c > 0 {
			pos += t.c * t.n
		} else {
			neg += t.c * t.n
		}
	}
	return 255*pos <= 1<<15-1 && 255*neg >= -(1<<15)
}
