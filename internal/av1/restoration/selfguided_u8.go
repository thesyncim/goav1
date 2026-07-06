// Ported from libaom:
//   av1/common/restoration.c
//   av1/common/restoration.h
// 8-bit sample domain following dav1d's 8bpc split (dav1d's sgr filters read
// uint8_t pixels and its weighted blend — sgr_weighted1/sgr_weighted2 in
// src/looprestoration_tmpl.c and src/arm/64/looprestoration.S — writes uint8_t
// output directly), while keeping goav1's libaom SGR integer pipeline and the
// int32 A/B/flt precision unchanged.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package restoration

// ApplySelfguidedRestorationU8 is the 8-bit-pixel variant of
// ApplySelfguidedRestoration: the source and destination are uint8 sample
// planes and bitDepth is fixed at 8, so the tile layer can run self-guided
// restoration directly on 8-bit planes without widening them to uint16 first.
//
// Contract (mirrors the uint16 entry exactly, so wiring is mechanical):
//   - src holds 8-bit samples; srcOrigin indexes the unit's top-left pixel and
//     src must include a 3-pixel border (SGRProjBorderHorz/Vert) around the
//     width x height unit, prepared by the caller exactly as for the u16
//     entry.
//   - dst is the unit's output block: dst[row*dstStride+col] for
//     row in [0,height), col in [0,width). Only those samples are written.
//   - scratch is the same int32 buffer sized by SelfguidedScratchLen(width,
//     height); the internal dgd/flt/A/B staging keeps today's int32 precision
//     exactly (the AV1 SGR integer pipeline is exact).
//   - Output is bit-identical to ApplySelfguidedRestoration called with the
//     same samples widened to uint16 and bitDepth=8: a uint8 sample can never
//     exceed max=255, so the per-sample validation of the u16 entry is
//     vacuous here, and the final projection keeps libaom's int16 wrap before
//     the [0,255] clip.
func ApplySelfguidedRestorationU8(src []uint8, srcStride int, srcOrigin int, dst []uint8, dstStride int, width int, height int, paramsIndex int, xqd [2]int8, scratch []int32) error {
	const bitDepth = 8
	if paramsIndex < 0 || paramsIndex >= len(SGRParameterTable) ||
		!validSGRBlock(width, height) ||
		!borderedBlockFits(len(src), srcStride, srcOrigin, width, height, SGRProjBorderHorz, SGRProjBorderVert) ||
		!blockFits(len(dst), dstStride, width, height) {
		return ErrInvalidRestoration
	}
	need, err := SelfguidedScratchLen(width, height)
	if err != nil || len(scratch) < need {
		return ErrInvalidRestoration
	}

	dgdStride := width + 2*SGRProjBorderHorz
	dgdLen := dgdStride * (height + 2*SGRProjBorderVert)
	fltLen := width * height
	bufStride := sgrBufferStride(width)
	bufLen := bufStride * (height + 2*SGRProjBorderVert)

	dgd := scratch[:dgdLen]
	flt0 := scratch[dgdLen : dgdLen+fltLen]
	flt1 := scratch[dgdLen+fltLen : dgdLen+2*fltLen]
	ab := scratch[dgdLen+2*fltLen:]
	aBuf := ab[:bufLen]
	bBuf := ab[bufLen : 2*bufLen]
	clearInt32s(dgd)
	clearInt32s(flt0)
	clearInt32s(flt1)
	clearInt32s(aBuf)
	clearInt32s(bBuf)

	dgdOrigin := SGRProjBorderVert*dgdStride + SGRProjBorderHorz
	extWidth := width + 2*SGRProjBorderHorz
	for row := -SGRProjBorderVert; row < height+SGRProjBorderVert; row++ {
		srcStart := srcOrigin + row*srcStride - SGRProjBorderHorz
		dgdStart := dgdOrigin + row*dgdStride - SGRProjBorderHorz
		srcRow := src[srcStart : srcStart+extWidth]
		dgdRow := dgd[dgdStart : dgdStart+extWidth]
		for col, sample := range srcRow {
			dgdRow[col] = int32(sample)
		}
	}

	params := SGRParameterTable[paramsIndex]
	if params.Radius[0] == 0 && params.Radius[1] == 0 {
		return ErrInvalidRestoration
	}
	if params.Radius[0] > 0 {
		selfguidedFastImpl(dgd, dgdOrigin, width, height, dgdStride, flt0, width, bitDepth, paramsIndex, 0, aBuf, bBuf, bufStride)
	}
	if params.Radius[1] > 0 {
		selfguidedImpl(dgd, dgdOrigin, width, height, dgdStride, flt1, width, bitDepth, paramsIndex, 1, aBuf, bBuf, bufStride)
	}

	xq := DecodeSGRXQ(xqd, params)
	xq0 := int32(xq[0])
	xq1 := int32(xq[1])
	// The u16 reference guards each projection term with use0/use1, but
	// DecodeSGRXQ pins the unused filter's xq to exactly 0 (Radius[0]==0 =>
	// xq[0]=0, Radius[1]==0 => xq[1]=0) and the unused flt buffer is zeroed
	// above, so applying both terms unconditionally is bit-identical:
	// 0*(f-u) == 0 in two's-complement int32 for every f.
	for row := range height {
		srcRow := src[srcOrigin+row*srcStride : srcOrigin+row*srcStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		f0Row := flt0[row*width : row*width+width]
		f1Row := flt1[row*width : row*width+width]
		sgrWeightedRowU8Impl(dstRow, srcRow, f0Row, f1Row, xq0, xq1)
	}
	return nil
}

// sgrWeightedRowU8Impl is the dispatch slot for the final SGR projection row
// over 8-bit pixels, resolved once at package init (see u8_dispatch_*.go).
// sgrWeightedRowU8 below is the canonical bit-exact reference; every tuned
// variant MUST match it sample for sample (including the int16 wrap before
// the [0,255] clip).
var sgrWeightedRowU8Impl = sgrWeightedRowU8

// sgrWeightedRowU8 applies the final self-guided projection for one row of
// 8-bit pixels. It is the per-pixel tail of ApplySelfguidedRestoration
// specialized to bitDepth=8 (u16 samples <= 255 widen to the same int32
// values), in the role of dav1d's sgr_weighted2 8bpc (weights applied to the
// filtered planes, uint8 store) but keeping libaom's exact arithmetic: the
// rounded projection is cast to int16 (wrapping) before the [0,255] clip.
// dst, src, f0, and f1 all hold len(dst) samples of the same row.
func sgrWeightedRowU8(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	for col, s := range src {
		u := int32(s) << SGRProjRstBits
		v := u << SGRProjPrjBits
		v += xq0 * (f0[col] - u)
		v += xq1 * (f1[col] - u)
		rounded := roundPowerOfTwo(v, SGRProjPrjBits+SGRProjRstBits)
		w := int32(int16(rounded))
		dst[col] = uint8(clampInt32(w, 0, 255))
	}
}
