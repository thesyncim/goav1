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
//   - Output is bit-identical to ApplyWienerRestoration{,Trusted} called with
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
	const bitDepth = 8
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	maxClamp := limit - 1
	f0 := int32(filter[0])
	f1 := int32(filter[1])
	f2 := int32(filter[2])
	f3 := int32(filter[3])
	f4 := int32(filter[4])
	f5 := int32(filter[5])
	f6 := int32(filter[6])
	for row := -WienerHalfwin; row < height+WienerHalfwin; row++ {
		dstRow := (row + WienerHalfwin) * width
		srcStart := srcOrigin + row*srcStride - WienerHalfwin
		srcRow := src[srcStart : srcStart+width+2*WienerHalfwin]
		dstSlice := temp[dstRow : dstRow+width]
		for col := range width {
			w := srcRow[col : col+WienerWin : col+WienerWin]
			s0, s1, s2, s3, s4, s5, s6 := w[0], w[1], w[2], w[3], w[4], w[5], w[6]
			sum := int32(s3)<<WienerFilterBits + offset
			sum += int32(s0)*f0 + int32(s1)*f1 + int32(s2)*f2 + int32(s3)*f3 +
				int32(s4)*f4 + int32(s5)*f5 + int32(s6)*f6
			dstSlice[col] = uint16(clampInt32(roundPowerOfTwo(sum, round0), 0, maxClamp))
		}
	}
}

// wienerVerticalU8 is wienerVertical with a uint8 destination at bitDepth=8
// (round1 is always 11 for 8-bit input). The final clamp bound max=255 equals
// the uint8 value range, so the narrowing store is exact.
func wienerVerticalU8(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	const bitDepth = 8
	offset := int32(1 << (bitDepth + round1 - 1))
	const maxI = int32(255)
	f0 := int32(filter[0])
	f1 := int32(filter[1])
	f2 := int32(filter[2])
	f3 := int32(filter[3])
	f4 := int32(filter[4])
	f5 := int32(filter[5])
	f6 := int32(filter[6])
	for row := range height {
		r0 := temp[(row+0)*tempStride : (row+0)*tempStride+width]
		r1 := temp[(row+1)*tempStride : (row+1)*tempStride+width]
		r2 := temp[(row+2)*tempStride : (row+2)*tempStride+width]
		r3 := temp[(row+3)*tempStride : (row+3)*tempStride+width]
		r4 := temp[(row+4)*tempStride : (row+4)*tempStride+width]
		r5 := temp[(row+5)*tempStride : (row+5)*tempStride+width]
		r6 := temp[(row+6)*tempStride : (row+6)*tempStride+width]
		dstSlice := dst[row*dstStride : row*dstStride+width]
		r1 = r1[:width]
		r2 = r2[:width]
		r3 = r3[:width]
		r4 = r4[:width]
		r5 = r5[:width]
		r6 = r6[:width]
		dstSlice = dstSlice[:width]
		for col, c0 := range r0 {
			c3 := int32(r3[col])
			sum := c3<<WienerFilterBits - offset
			sum += int32(c0)*f0 + int32(r1[col])*f1 + int32(r2[col])*f2 +
				c3*f3 + int32(r4[col])*f4 + int32(r5[col])*f5 + int32(r6[col])*f6
			dstSlice[col] = uint8(clampInt32(roundPowerOfTwo(sum, round1), 0, maxI))
		}
	}
}
