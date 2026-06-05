// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

// NEON-accelerated Wiener separable passes. The .s file implements the inner
// per-row loop for widths that are a multiple of 8 (i.e. the common 64-wide
// processing units and every other 8-aligned restoration unit). The Go wrappers
// here route narrow / non-8-aligned widths to the pure-Go reference, which keeps
// the asm to a single code path and the byte-exactness contract easy to audit.
//
// Bit-exactness with wienerHorizontal / wienerVertical:
//   - Samples are 8/10/12-bit (<= 4095), so they fit in a positive int16 lane
//     and the signed widening MAC (smlal/smlal2) reproduces s_i*f_i exactly.
//   - The libaom "center reapplication" term s3<<7 is folded into the center tap
//     (tap3 += 1<<WienerFilterBits) by the wrappers, so the asm runs a plain
//     7-tap MAC.
//   - The rounding bias 1<<(round-1) is folded into the accumulator seed
//     (offset for the horizontal pass, -offset for the vertical pass), then the
//     accumulator is arithmetically shifted right by `round` with a per-lane
//     SSHL (negative shift). This equals roundPowerOfTwo(sum, round) bit-for-bit.
//   - The clamp is SQXTUN (signed s32 -> unsigned u16, saturating, lower bound 0)
//     followed by UMIN against the broadcast max, identical to
//     clampInt32(x, 0, maxClamp) / clampInt32(x, 0, max).

// wienerNEONHorizCtx is the asm calling context for the horizontal pass. Field
// order and sizes are part of the ABI shared with wiener_neon_arm64.s; do not
// reorder.
type wienerNEONHorizCtx struct {
	dst    *uint16 // temp output base
	src    *uint16 // first tap sample of row -WienerHalfwin (== srcStart for that row)
	srcStr uintptr // src stride in elements
	width  uintptr // output columns (multiple of 8)
	rows   uintptr // number of rows: height + 2*WienerHalfwin
	taps   *int16  // 8 lanes: 7 adjusted taps (tap3 += 128) + 1 pad
	seed   int32   // accumulator seed: offset + (1<<(round0-1))
	shift  int32   // -round0 (negative => arithmetic right shift)
	maxCl  uint16  // upper clamp (maxClamp = limit-1)
}

// wienerNEONVertCtx is the asm calling context for the vertical pass. Field
// order and sizes are part of the ABI shared with wiener_neon_arm64.s; do not
// reorder.
type wienerNEONVertCtx struct {
	dst    *uint16 // output base
	src    *uint16 // temp base (row 0)
	dstStr uintptr // dst stride in elements
	srcStr uintptr // temp stride in elements (== width)
	width  uintptr // output columns (multiple of 8)
	rows   uintptr // output rows (== height)
	taps   *int16  // 8 lanes: 7 adjusted taps (tap3 += 128) + 1 pad
	seed   int32   // accumulator seed: -offset + (1<<(round1-1))
	shift  int32   // -round1
	maxCl  uint16  // upper clamp (max)
}

//go:noescape
func wienerHorizontalNEONAsm(ctx *wienerNEONHorizCtx)

//go:noescape
func wienerVerticalNEONAsm(ctx *wienerNEONVertCtx)

// adjustedWienerTaps returns the 7 distinct taps with the center tap raised by
// 1<<WienerFilterBits so the asm's plain 7-tap MAC reproduces the libaom
// s3<<WienerFilterBits center reapplication. Lane 7 is padding (the asm loads a
// full 8-lane vector but only multiplies by lanes 0..6). All values stay well
// inside int16: the largest center tap is WienerTap2Max-derived plus 128.
func adjustedWienerTaps(filter WienerFilter) [8]int16 {
	var t [8]int16
	for i := 0; i < 7; i++ {
		t[i] = filter[i]
	}
	t[3] += 1 << WienerFilterBits
	return t
}

func wienerHorizontalNEON(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) bool {
	if width < 8 || width%8 != 0 {
		return wienerHorizontal(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
	}
	// Replicate the pure-Go validity check: every sample touched by the inner
	// window must be <= max. The window for row r spans columns [col-3..col+3]
	// for col in [0,width), i.e. the full reslice src[srcStart : srcStart+width+6].
	for row := -WienerHalfwin; row < height+WienerHalfwin; row++ {
		srcStart := srcOrigin + row*srcStride - WienerHalfwin
		w := src[srcStart : srcStart+width+2*WienerHalfwin]
		for _, s := range w {
			if s > max {
				return false
			}
		}
	}

	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	taps := adjustedWienerTaps(filter)
	ctx := wienerNEONHorizCtx{
		dst:    &temp[0],
		src:    &src[srcOrigin-WienerHalfwin*srcStride-WienerHalfwin],
		srcStr: uintptr(srcStride),
		width:  uintptr(width),
		rows:   uintptr(height + 2*WienerHalfwin),
		taps:   &taps[0],
		seed:   offset + roundBias(round0),
		shift:  int32(-round0),
		maxCl:  uint16(limit - 1),
	}
	wienerHorizontalNEONAsm(&ctx)
	return true
}

func wienerHorizontalNEONTrusted(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	if width < 8 || width%8 != 0 {
		wienerHorizontalTrusted(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
		return
	}
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	taps := adjustedWienerTaps(filter)
	ctx := wienerNEONHorizCtx{
		dst:    &temp[0],
		src:    &src[srcOrigin-WienerHalfwin*srcStride-WienerHalfwin],
		srcStr: uintptr(srcStride),
		width:  uintptr(width),
		rows:   uintptr(height + 2*WienerHalfwin),
		taps:   &taps[0],
		seed:   offset + roundBias(round0),
		shift:  int32(-round0),
		maxCl:  uint16(limit - 1),
	}
	wienerHorizontalNEONAsm(&ctx)
}

func wienerVerticalNEON(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	if width < 8 || width%8 != 0 {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
		return
	}
	offset := int32(1 << (bitDepth + round1 - 1))
	taps := adjustedWienerTaps(filter)
	ctx := wienerNEONVertCtx{
		dst:    &dst[0],
		src:    &temp[0],
		dstStr: uintptr(dstStride),
		srcStr: uintptr(tempStride),
		width:  uintptr(width),
		rows:   uintptr(height),
		taps:   &taps[0],
		seed:   -offset + roundBias(round1),
		shift:  int32(-round1),
		maxCl:  max,
	}
	wienerVerticalNEONAsm(&ctx)
}

// roundBias returns the rounding term 1<<(bits-1) folded into the accumulator
// seed so the asm's arithmetic right shift reproduces roundPowerOfTwo.
func roundBias(bits int) int32 {
	if bits <= 0 {
		return 0
	}
	return 1 << (bits - 1)
}
