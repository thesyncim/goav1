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

// cdefLUTIndex maps a tap difference to its constrain-table slot.
func cdefLUTIndex(diff int) int {
	return (min(diff, 256) + 255) & 511
}

func filterRowU8SecLUT(d []byte, input []uint16, base int, t *cdefU8Taps, lut *[512]int8) {
	n := len(d)
	c := input[base:][:n]
	s0, s1 := cdefTapRow(input, base, t.sec0, n), cdefTapRow(input, base, -t.sec0, n)
	s2, s3 := cdefTapRow(input, base, t.sec2, n), cdefTapRow(input, base, -t.sec2, n)
	s4, s5 := cdefTapRow(input, base, t.sec1, n), cdefTapRow(input, base, -t.sec1, n)
	s6, s7 := cdefTapRow(input, base, t.sec3, n), cdefTapRow(input, base, -t.sec3, n)
	for i := range d {
		x := int(c[i])
		sum := 2*(int(lut[cdefLUTIndex(int(s0[i])-x)])+int(lut[cdefLUTIndex(int(s1[i])-x)])+
			int(lut[cdefLUTIndex(int(s2[i])-x)])+int(lut[cdefLUTIndex(int(s3[i])-x)])) +
			int(lut[cdefLUTIndex(int(s4[i])-x)]) + int(lut[cdefLUTIndex(int(s5[i])-x)]) +
			int(lut[cdefLUTIndex(int(s6[i])-x)]) + int(lut[cdefLUTIndex(int(s7[i])-x)])
		d[i] = byte(cdefRound(x, sum))
	}
}

func filterRowU8PriLUT(d []byte, input []uint16, base int, t *cdefU8Taps, lut *[512]int8) {
	n := len(d)
	c := input[base:][:n]
	p0, p1 := cdefTapRow(input, base, t.pri0, n), cdefTapRow(input, base, -t.pri0, n)
	p2, p3 := cdefTapRow(input, base, t.pri1, n), cdefTapRow(input, base, -t.pri1, n)
	tap0, tap1 := t.priTap0, t.priTap1
	for i := range d {
		x := int(c[i])
		sum := tap0*(int(lut[cdefLUTIndex(int(p0[i])-x)])+int(lut[cdefLUTIndex(int(p1[i])-x)])) +
			tap1*(int(lut[cdefLUTIndex(int(p2[i])-x)])+int(lut[cdefLUTIndex(int(p3[i])-x)]))
		d[i] = byte(cdefRound(x, sum))
	}
}

// filterRowU8PriSecLUT is filterRowU8PriSec with table constrain.
func filterRowU8PriSecLUT(d []byte, input []uint16, base int, t *cdefU8Taps, plut *[512]int8, slut *[512]int8) {
	n := len(d)
	c := input[base:][:n]
	p0, p1 := cdefTapRow(input, base, t.pri0, n), cdefTapRow(input, base, -t.pri0, n)
	p2, p3 := cdefTapRow(input, base, t.pri1, n), cdefTapRow(input, base, -t.pri1, n)
	s0, s1 := cdefTapRow(input, base, t.sec0, n), cdefTapRow(input, base, -t.sec0, n)
	s2, s3 := cdefTapRow(input, base, t.sec2, n), cdefTapRow(input, base, -t.sec2, n)
	s4, s5 := cdefTapRow(input, base, t.sec1, n), cdefTapRow(input, base, -t.sec1, n)
	s6, s7 := cdefTapRow(input, base, t.sec3, n), cdefTapRow(input, base, -t.sec3, n)
	tap0, tap1 := t.priTap0, t.priTap1
	const mask = VeryLarge - 1
	for i := range d {
		x := int(c[i])
		a0, a1, a2, a3 := int(p0[i]), int(p1[i]), int(p2[i]), int(p3[i])
		b0, b1, b2, b3 := int(s0[i]), int(s1[i]), int(s2[i]), int(s3[i])
		b4, b5, b6, b7 := int(s4[i]), int(s5[i]), int(s6[i]), int(s7[i])
		sum := tap0*(int(plut[cdefLUTIndex(a0-x)])+int(plut[cdefLUTIndex(a1-x)])) +
			tap1*(int(plut[cdefLUTIndex(a2-x)])+int(plut[cdefLUTIndex(a3-x)])) +
			2*(int(slut[cdefLUTIndex(b0-x)])+int(slut[cdefLUTIndex(b1-x)])+
				int(slut[cdefLUTIndex(b2-x)])+int(slut[cdefLUTIndex(b3-x)])) +
			int(slut[cdefLUTIndex(b4-x)]) + int(slut[cdefLUTIndex(b5-x)]) +
			int(slut[cdefLUTIndex(b6-x)]) + int(slut[cdefLUTIndex(b7-x)])
		hi := max(x, a0&mask, a1&mask, a2&mask, a3&mask, b0&mask, b1&mask, b2&mask, b3&mask,
			b4&mask, b5&mask, b6&mask, b7&mask)
		lo := min(x, a0, a1, a2, a3, b0, b1, b2, b3, b4, b5, b6, b7)
		d[i] = byte(min(max(cdefRound(x, sum), lo), hi))
	}
}
