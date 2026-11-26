// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

// cdefConstrainLUT[threshold][shift][min(diff, 256)+255] = constrain(diff)
// for the 8-bit CDEF kernels. Real 8-bit taps give diff in [-255, 255]; the
// only larger diff is a VeryLarge halo sentinel, whose constrain is 0 for
// every shift below cdefLUTMaxShift, so it shares the zero entry at 511.
// Shifts from 8 up to cdefLUTMaxShift behave like shift 8 on real diffs
// (a>>shift == 0), so they are folded into it.
var cdefConstrainLUT [16][9][512]int8

const cdefLUTMaxShift = 13

func init() {
	for str := 1; str < 16; str++ {
		for sh := 0; sh <= 8; sh++ {
			for diff := -255; diff <= 255; diff++ {
				cdefConstrainLUT[str][sh][diff+255] = int8(constrainShifted(diff, str, sh))
			}
		}
	}
}

// cdefLUT returns the constrain table for a nonzero threshold, or nil when
// the parameters fall outside it.
func cdefLUT(threshold int, shift int) *[512]int8 {
	if threshold <= 0 || threshold > 15 || shift > cdefLUTMaxShift {
		return nil
	}
	return &cdefConstrainLUT[threshold][min(shift, 8)]
}

func filterBlockU8SecLUT(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, width int, height int, t *cdefU8Taps, lut *[512]int8) {
	sec0, sec1, sec2, sec3 := t.sec0, t.sec1, t.sec2, t.sec3
	for row := range height {
		base := inputOrigin + row*BStride
		d := dst[dstOrigin+row*dstStride : dstOrigin+row*dstStride+width]
		c := input[base : base+width]
		s0, s1 := input[base+sec0:base+sec0+width], input[base-sec0:base-sec0+width]
		s2, s3 := input[base+sec2:base+sec2+width], input[base-sec2:base-sec2+width]
		s4, s5 := input[base+sec1:base+sec1+width], input[base-sec1:base-sec1+width]
		s6, s7 := input[base+sec3:base+sec3+width], input[base-sec3:base-sec3+width]
		i := 0
		for ; i+3 < width; i += 4 {
			x0, x1, x2, x3 := int(c[i]), int(c[i+1]), int(c[i+2]), int(c[i+3])
			b0, b1, b2, b3 := uint(255-x0), uint(255-x1), uint(255-x2), uint(255-x3)
			sum0 := 2*(int(lut[min(uint(s0[i])+b0, 511)])+int(lut[min(uint(s1[i])+b0, 511)])+
				int(lut[min(uint(s2[i])+b0, 511)])+int(lut[min(uint(s3[i])+b0, 511)])) +
				int(lut[min(uint(s4[i])+b0, 511)]) + int(lut[min(uint(s5[i])+b0, 511)]) +
				int(lut[min(uint(s6[i])+b0, 511)]) + int(lut[min(uint(s7[i])+b0, 511)])
			sum1 := 2*(int(lut[min(uint(s0[i+1])+b1, 511)])+int(lut[min(uint(s1[i+1])+b1, 511)])+
				int(lut[min(uint(s2[i+1])+b1, 511)])+int(lut[min(uint(s3[i+1])+b1, 511)])) +
				int(lut[min(uint(s4[i+1])+b1, 511)]) + int(lut[min(uint(s5[i+1])+b1, 511)]) +
				int(lut[min(uint(s6[i+1])+b1, 511)]) + int(lut[min(uint(s7[i+1])+b1, 511)])
			sum2 := 2*(int(lut[min(uint(s0[i+2])+b2, 511)])+int(lut[min(uint(s1[i+2])+b2, 511)])+
				int(lut[min(uint(s2[i+2])+b2, 511)])+int(lut[min(uint(s3[i+2])+b2, 511)])) +
				int(lut[min(uint(s4[i+2])+b2, 511)]) + int(lut[min(uint(s5[i+2])+b2, 511)]) +
				int(lut[min(uint(s6[i+2])+b2, 511)]) + int(lut[min(uint(s7[i+2])+b2, 511)])
			sum3 := 2*(int(lut[min(uint(s0[i+3])+b3, 511)])+int(lut[min(uint(s1[i+3])+b3, 511)])+
				int(lut[min(uint(s2[i+3])+b3, 511)])+int(lut[min(uint(s3[i+3])+b3, 511)])) +
				int(lut[min(uint(s4[i+3])+b3, 511)]) + int(lut[min(uint(s5[i+3])+b3, 511)]) +
				int(lut[min(uint(s6[i+3])+b3, 511)]) + int(lut[min(uint(s7[i+3])+b3, 511)])
			d[i] = byte(cdefRound(x0, sum0))
			d[i+1] = byte(cdefRound(x1, sum1))
			d[i+2] = byte(cdefRound(x2, sum2))
			d[i+3] = byte(cdefRound(x3, sum3))
		}
		for ; i < width; i++ {
			x := int(c[i])
			b := uint(255 - x)
			sum := 2*(int(lut[min(uint(s0[i])+b, 511)])+int(lut[min(uint(s1[i])+b, 511)])+
				int(lut[min(uint(s2[i])+b, 511)])+int(lut[min(uint(s3[i])+b, 511)])) +
				int(lut[min(uint(s4[i])+b, 511)]) + int(lut[min(uint(s5[i])+b, 511)]) +
				int(lut[min(uint(s6[i])+b, 511)]) + int(lut[min(uint(s7[i])+b, 511)])
			d[i] = byte(cdefRound(x, sum))
		}
	}
}

func filterBlockU8PriLUT(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, width int, height int, t *cdefU8Taps, lut *[512]int8) {
	pri0, pri1 := t.pri0, t.pri1
	tap0, tap1 := t.priTap0, t.priTap1
	for row := range height {
		base := inputOrigin + row*BStride
		d := dst[dstOrigin+row*dstStride : dstOrigin+row*dstStride+width]
		c := input[base : base+width]
		p0, p1 := input[base+pri0:base+pri0+width], input[base-pri0:base-pri0+width]
		p2, p3 := input[base+pri1:base+pri1+width], input[base-pri1:base-pri1+width]
		i := 0
		for ; i+3 < width; i += 4 {
			x0, x1, x2, x3 := int(c[i]), int(c[i+1]), int(c[i+2]), int(c[i+3])
			b0, b1, b2, b3 := uint(255-x0), uint(255-x1), uint(255-x2), uint(255-x3)
			sum0 := tap0*(int(lut[min(uint(p0[i])+b0, 511)])+int(lut[min(uint(p1[i])+b0, 511)])) +
				tap1*(int(lut[min(uint(p2[i])+b0, 511)])+int(lut[min(uint(p3[i])+b0, 511)]))
			sum1 := tap0*(int(lut[min(uint(p0[i+1])+b1, 511)])+int(lut[min(uint(p1[i+1])+b1, 511)])) +
				tap1*(int(lut[min(uint(p2[i+1])+b1, 511)])+int(lut[min(uint(p3[i+1])+b1, 511)]))
			sum2 := tap0*(int(lut[min(uint(p0[i+2])+b2, 511)])+int(lut[min(uint(p1[i+2])+b2, 511)])) +
				tap1*(int(lut[min(uint(p2[i+2])+b2, 511)])+int(lut[min(uint(p3[i+2])+b2, 511)]))
			sum3 := tap0*(int(lut[min(uint(p0[i+3])+b3, 511)])+int(lut[min(uint(p1[i+3])+b3, 511)])) +
				tap1*(int(lut[min(uint(p2[i+3])+b3, 511)])+int(lut[min(uint(p3[i+3])+b3, 511)]))
			d[i] = byte(cdefRound(x0, sum0))
			d[i+1] = byte(cdefRound(x1, sum1))
			d[i+2] = byte(cdefRound(x2, sum2))
			d[i+3] = byte(cdefRound(x3, sum3))
		}
		for ; i < width; i++ {
			x := int(c[i])
			b := uint(255 - x)
			sum := tap0*(int(lut[min(uint(p0[i])+b, 511)])+int(lut[min(uint(p1[i])+b, 511)])) +
				tap1*(int(lut[min(uint(p2[i])+b, 511)])+int(lut[min(uint(p3[i])+b, 511)]))
			d[i] = byte(cdefRound(x, sum))
		}
	}
}

func filterBlockU8PriSecLUT(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, width int, height int, t *cdefU8Taps, plut *[512]int8, slut *[512]int8) {
	pri0, pri1 := t.pri0, t.pri1
	sec0, sec1, sec2, sec3 := t.sec0, t.sec1, t.sec2, t.sec3
	tap0, tap1 := t.priTap0, t.priTap1
	const mask = VeryLarge - 1
	for row := range height {
		base := inputOrigin + row*BStride
		d := dst[dstOrigin+row*dstStride : dstOrigin+row*dstStride+width]
		c := input[base : base+width]
		p0, p1 := input[base+pri0:base+pri0+width], input[base-pri0:base-pri0+width]
		p2, p3 := input[base+pri1:base+pri1+width], input[base-pri1:base-pri1+width]
		s0, s1 := input[base+sec0:base+sec0+width], input[base-sec0:base-sec0+width]
		s2, s3 := input[base+sec2:base+sec2+width], input[base-sec2:base-sec2+width]
		s4, s5 := input[base+sec1:base+sec1+width], input[base-sec1:base-sec1+width]
		s6, s7 := input[base+sec3:base+sec3+width], input[base-sec3:base-sec3+width]
		for i := range d {
			x := int(c[i])
			b := uint(255 - x)
			a0, a1, a2, a3 := int(p0[i]), int(p1[i]), int(p2[i]), int(p3[i])
			b0, b1, b2, b3 := int(s0[i]), int(s1[i]), int(s2[i]), int(s3[i])
			b4, b5, b6, b7 := int(s4[i]), int(s5[i]), int(s6[i]), int(s7[i])
			sum := tap0*(int(plut[min(uint(a0)+b, 511)])+int(plut[min(uint(a1)+b, 511)])) +
				tap1*(int(plut[min(uint(a2)+b, 511)])+int(plut[min(uint(a3)+b, 511)])) +
				2*(int(slut[min(uint(b0)+b, 511)])+int(slut[min(uint(b1)+b, 511)])+
					int(slut[min(uint(b2)+b, 511)])+int(slut[min(uint(b3)+b, 511)])) +
				int(slut[min(uint(b4)+b, 511)]) + int(slut[min(uint(b5)+b, 511)]) +
				int(slut[min(uint(b6)+b, 511)]) + int(slut[min(uint(b7)+b, 511)])
			hi := max(x, a0&mask, a1&mask, a2&mask, a3&mask, b0&mask, b1&mask, b2&mask, b3&mask,
				b4&mask, b5&mask, b6&mask, b7&mask)
			lo := min(x, a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7)
			d[i] = byte(min(max(cdefRound(x, sum), lo), hi))
		}
	}
}
