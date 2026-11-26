// Ported from libaom:
//   av1/common/restoration.c
//   av1/common/restoration.h
// 8-bit sample domain following dav1d's 8bpc split (dav1d reads uint8_t source
// pixels and writes uint8_t output in src/looprestoration_tmpl.c and
// src/arm/64/looprestoration.S wiener_filter7, widening to 16 bits inside the
// kernel), while keeping goav1's two-pass Wiener structure and libaom
// arithmetic unchanged.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package restoration

// ApplyWienerRestorationU8 is the 8-bit-pixel variant of
// ApplyWienerRestorationTrusted: the source and destination are uint8 sample
// planes and bitDepth is fixed at 8, so the tile layer can run Wiener
// restoration directly on 8-bit planes without widening them to uint16 first.
//
// Contract (mirrors the uint16 entry exactly, so wiring is mechanical):
//   - src holds 8-bit samples; srcOrigin indexes the unit's top-left pixel and
//     src must include a WienerHalfwin(=3)-pixel border around the
//     width x height unit (same borderedBlockFits geometry as the u16 entry;
//     the caller prepares stripe/frame borders exactly as it does today).
//   - dst is the unit's output block: dst[row*dstStride+col] for
//     row in [0,height), col in [0,width). Only those samples are written.
//   - scratch is the same uint16 buffer sized by WienerScratchLen(width,
//     height); the 16-bit horizontal intermediate is unchanged from the u16
//     pipeline.
//   - Output is bit-identical to ApplyWienerRestorationTrusted called with
//     the same samples widened to uint16 and bitDepth=8. The trusted/checked
//     split disappears at 8 bits: a uint8 sample can never exceed max=255, so
//     the per-sample validation scan is vacuous and this single entry covers
//     both.
func ApplyWienerRestorationU8(src []uint8, srcStride int, srcOrigin int, dst []uint8, dstStride int, width int, height int, info WienerInfo, scratch []uint16) error {
	if !validRestorationBlock(width, height) ||
		!validWienerInfo(info) ||
		!borderedBlockFits(len(src), srcStride, srcOrigin, width, height, WienerHalfwin, WienerHalfwin) ||
		!blockFits(len(dst), dstStride, width, height) {
		return ErrInvalidRestoration
	}
	need, err := WienerScratchLen(width, height)
	if err != nil || len(scratch) < need {
		return ErrInvalidRestoration
	}

	round0, round1 := wienerRounds(8)
	temp := scratch[:need]
	wienerHorizontalU8Impl(src, srcStride, srcOrigin, width, height, info.HFilter, round0, temp)
	wienerVerticalU8Impl(temp, width, dst, dstStride, width, height, info.VFilter, round1)
	return nil
}

// wienerHorizontalU8Impl and wienerVerticalU8Impl are the dispatch slots for
// the 8-bit-pixel Wiener passes, resolved once at package init (see
// u8_dispatch_*.go) exactly like the uint16 slots above. wienerHorizontalU8 /
// wienerVerticalU8 below are the canonical bit-exact references: they are the
// uint16 references specialized to bitDepth=8 with only the sample domain
// changed (int32(uint8 s) == int32(uint16 s) for 8-bit samples, so every
// intermediate value is identical). Every tuned variant MUST match them sample
// for sample.
var (
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl   = wienerVerticalU8
)

// wienerHorizontalU8 is wienerHorizontalTrusted with a uint8 source at
// bitDepth=8 (round0 is always 3 for 8-bit input; see wienerRounds). The
// 16-bit temp intermediate is unchanged.
func wienerHorizontalU8(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if width <= 0 || height <= 0 {
		return
	}
	const bitDepth = 8
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	maxClamp := limit - 1
	r0Shift := uint(round0)
	offsetRound := offset
	if round0 > 0 {
		offsetRound += int32(1 << (round0 - 1))
	}
	f0 := int32(filter[0])
	f1 := int32(filter[1])
	f2 := int32(filter[2])
	f3 := int32(filter[3])
	f4 := int32(filter[4])
	f5 := int32(filter[5])
	f6 := int32(filter[6])
	f3c := f3 + (1 << WienerFilterBits)
	symmetric := f0 == f6 && f1 == f5 && f2 == f4
	fiveTap := symmetric && f0 == 0
	maxU := uint32(maxClamp)

	for row := -WienerHalfwin; row < height+WienerHalfwin; row++ {
		dstRow := (row + WienerHalfwin) * width
		srcStart := srcOrigin + row*srcStride - WienerHalfwin
		srcRow := src[srcStart : srcStart+width+2*WienerHalfwin]
		dstSlice := temp[dstRow : dstRow+width]
		if fiveTap {
			s1, s2, s3, s4 := int32(srcRow[1]), int32(srcRow[2]), int32(srcRow[3]), int32(srcRow[4])
			col := 0
			for ; col+3 < width; col += 4 {
				s5 := int32(srcRow[col+5])
				s6 := int32(srcRow[col+6])
				s7 := int32(srcRow[col+7])
				s8 := int32(srcRow[col+8])
				v0 := (offsetRound + f1*(s1+s5) + f2*(s2+s4) + f3c*s3) >> r0Shift
				v1 := (offsetRound + f1*(s2+s6) + f2*(s3+s5) + f3c*s4) >> r0Shift
				v2 := (offsetRound + f1*(s3+s7) + f2*(s4+s6) + f3c*s5) >> r0Shift
				v3 := (offsetRound + f1*(s4+s8) + f2*(s5+s7) + f3c*s6) >> r0Shift
				if uint32(v0|v1|v2|v3) > maxU {
					v0 = clampInt32(v0, 0, maxClamp)
					v1 = clampInt32(v1, 0, maxClamp)
					v2 = clampInt32(v2, 0, maxClamp)
					v3 = clampInt32(v3, 0, maxClamp)
				}
				dstSlice[col] = uint16(v0)
				dstSlice[col+1] = uint16(v1)
				dstSlice[col+2] = uint16(v2)
				dstSlice[col+3] = uint16(v3)
				s1, s2, s3, s4 = s5, s6, s7, s8
			}
			for ; col < width; col++ {
				s5 := int32(srcRow[col+5])
				v := (offsetRound + f1*(s1+s5) + f2*(s2+s4) + f3c*s3) >> r0Shift
				if uint32(v) > maxU {
					if v < 0 {
						v = 0
					} else {
						v = maxClamp
					}
				}
				dstSlice[col] = uint16(v)
				s1, s2, s3, s4 = s2, s3, s4, s5
			}
			continue
		}
		if symmetric {
			s0, s1, s2, s3, s4, s5 := int32(srcRow[0]), int32(srcRow[1]), int32(srcRow[2]), int32(srcRow[3]), int32(srcRow[4]), int32(srcRow[5])
			col := 0
			for ; col+3 < width; col += 4 {
				s6 := int32(srcRow[col+6])
				s7 := int32(srcRow[col+7])
				s8 := int32(srcRow[col+8])
				s9 := int32(srcRow[col+9])
				v0 := (offsetRound + f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3c*s3) >> r0Shift
				v1 := (offsetRound + f0*(s1+s7) + f1*(s2+s6) + f2*(s3+s5) + f3c*s4) >> r0Shift
				v2 := (offsetRound + f0*(s2+s8) + f1*(s3+s7) + f2*(s4+s6) + f3c*s5) >> r0Shift
				v3 := (offsetRound + f0*(s3+s9) + f1*(s4+s8) + f2*(s5+s7) + f3c*s6) >> r0Shift
				if uint32(v0|v1|v2|v3) > maxU {
					v0 = clampInt32(v0, 0, maxClamp)
					v1 = clampInt32(v1, 0, maxClamp)
					v2 = clampInt32(v2, 0, maxClamp)
					v3 = clampInt32(v3, 0, maxClamp)
				}
				dstSlice[col] = uint16(v0)
				dstSlice[col+1] = uint16(v1)
				dstSlice[col+2] = uint16(v2)
				dstSlice[col+3] = uint16(v3)
				s0, s1, s2, s3, s4, s5 = s4, s5, s6, s7, s8, s9
			}
			for ; col < width; col++ {
				s6 := int32(srcRow[col+6])
				v := (offsetRound + f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3c*s3) >> r0Shift
				if uint32(v) > maxU {
					if v < 0 {
						v = 0
					} else {
						v = maxClamp
					}
				}
				dstSlice[col] = uint16(v)
				s0, s1, s2, s3, s4, s5 = s1, s2, s3, s4, s5, s6
			}
			continue
		}
		for col := range width {
			w := srcRow[col : col+WienerWin : col+WienerWin]
			s0, s1, s2, s3, s4, s5, s6 := w[0], w[1], w[2], w[3], w[4], w[5], w[6]
			sum := int32(s3)*f3c + offsetRound +
				int32(s0)*f0 + int32(s1)*f1 + int32(s2)*f2 +
				int32(s4)*f4 + int32(s5)*f5 + int32(s6)*f6
			dstSlice[col] = uint16(clampInt32(sum>>r0Shift, 0, maxClamp))
		}
	}
}

// wienerVerticalU8 is wienerVertical with a uint8 destination at bitDepth=8
// (round1 is always 11 for 8-bit input). The final clamp bound max=255 equals
// the uint8 value range, so the narrowing store is exact.
func wienerVerticalU8(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if width <= 0 || height <= 0 {
		return
	}
	const bitDepth = 8
	offset := int32(1 << (bitDepth + round1 - 1))
	r1Shift := uint(round1)
	bias := -offset
	if round1 > 0 {
		bias += int32(1 << (round1 - 1))
	}
	f0 := int32(filter[0])
	f1 := int32(filter[1])
	f2 := int32(filter[2])
	f3 := int32(filter[3])
	f4 := int32(filter[4])
	f5 := int32(filter[5])
	f6 := int32(filter[6])
	f3c := f3 + (1 << WienerFilterBits)
	symmetric := f0 == f6 && f1 == f5 && f2 == f4
	fiveTap := symmetric && f0 == 0

	for row := range height {
		r1 := temp[(row+1)*tempStride : (row+1)*tempStride+width]
		r2 := temp[(row+2)*tempStride : (row+2)*tempStride+width]
		r3 := temp[(row+3)*tempStride : (row+3)*tempStride+width]
		r4 := temp[(row+4)*tempStride : (row+4)*tempStride+width]
		r5 := temp[(row+5)*tempStride : (row+5)*tempStride+width]
		dstSlice := dst[row*dstStride : row*dstStride+width]
		if fiveTap {
			col := 0
			for ; col+3 < width; col += 4 {
				v0 := (bias + f1*(int32(r1[col])+int32(r5[col])) + f2*(int32(r2[col])+int32(r4[col])) + f3c*int32(r3[col])) >> r1Shift
				v1 := (bias + f1*(int32(r1[col+1])+int32(r5[col+1])) + f2*(int32(r2[col+1])+int32(r4[col+1])) + f3c*int32(r3[col+1])) >> r1Shift
				v2 := (bias + f1*(int32(r1[col+2])+int32(r5[col+2])) + f2*(int32(r2[col+2])+int32(r4[col+2])) + f3c*int32(r3[col+2])) >> r1Shift
				v3 := (bias + f1*(int32(r1[col+3])+int32(r5[col+3])) + f2*(int32(r2[col+3])+int32(r4[col+3])) + f3c*int32(r3[col+3])) >> r1Shift
				if uint32(v0|v1|v2|v3) > 255 {
					v0 = clampInt32(v0, 0, 255)
					v1 = clampInt32(v1, 0, 255)
					v2 = clampInt32(v2, 0, 255)
					v3 = clampInt32(v3, 0, 255)
				}
				dstSlice[col] = uint8(v0)
				dstSlice[col+1] = uint8(v1)
				dstSlice[col+2] = uint8(v2)
				dstSlice[col+3] = uint8(v3)
			}
			for ; col < width; col++ {
				v := (bias + f1*(int32(r1[col])+int32(r5[col])) + f2*(int32(r2[col])+int32(r4[col])) + f3c*int32(r3[col])) >> r1Shift
				if uint32(v) > 255 {
					if v < 0 {
						v = 0
					} else {
						v = 255
					}
				}
				dstSlice[col] = uint8(v)
			}
			continue
		}
		r0 := temp[(row+0)*tempStride : (row+0)*tempStride+width]
		r6 := temp[(row+6)*tempStride : (row+6)*tempStride+width]
		if symmetric {
			col := 0
			for ; col+3 < width; col += 4 {
				v0 := (bias + f0*(int32(r0[col])+int32(r6[col])) + f1*(int32(r1[col])+int32(r5[col])) + f2*(int32(r2[col])+int32(r4[col])) + f3c*int32(r3[col])) >> r1Shift
				v1 := (bias + f0*(int32(r0[col+1])+int32(r6[col+1])) + f1*(int32(r1[col+1])+int32(r5[col+1])) + f2*(int32(r2[col+1])+int32(r4[col+1])) + f3c*int32(r3[col+1])) >> r1Shift
				v2 := (bias + f0*(int32(r0[col+2])+int32(r6[col+2])) + f1*(int32(r1[col+2])+int32(r5[col+2])) + f2*(int32(r2[col+2])+int32(r4[col+2])) + f3c*int32(r3[col+2])) >> r1Shift
				v3 := (bias + f0*(int32(r0[col+3])+int32(r6[col+3])) + f1*(int32(r1[col+3])+int32(r5[col+3])) + f2*(int32(r2[col+3])+int32(r4[col+3])) + f3c*int32(r3[col+3])) >> r1Shift
				if uint32(v0|v1|v2|v3) > 255 {
					v0 = clampInt32(v0, 0, 255)
					v1 = clampInt32(v1, 0, 255)
					v2 = clampInt32(v2, 0, 255)
					v3 = clampInt32(v3, 0, 255)
				}
				dstSlice[col] = uint8(v0)
				dstSlice[col+1] = uint8(v1)
				dstSlice[col+2] = uint8(v2)
				dstSlice[col+3] = uint8(v3)
			}
			for ; col < width; col++ {
				v := (bias + f0*(int32(r0[col])+int32(r6[col])) + f1*(int32(r1[col])+int32(r5[col])) + f2*(int32(r2[col])+int32(r4[col])) + f3c*int32(r3[col])) >> r1Shift
				if uint32(v) > 255 {
					if v < 0 {
						v = 0
					} else {
						v = 255
					}
				}
				dstSlice[col] = uint8(v)
			}
			continue
		}
		for col, c0 := range r0 {
			c3 := int32(r3[col])
			sum := c3*f3c + bias +
				int32(c0)*f0 + int32(r1[col])*f1 + int32(r2[col])*f2 +
				int32(r4[col])*f4 + int32(r5[col])*f5 + int32(r6[col])*f6
			dstSlice[col] = uint8(clampInt32(sum>>r1Shift, 0, 255))
		}
	}
}
