// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

// AVX2-accelerated Wiener separable passes. The .s file implements the inner
// per-row loop for widths that are a multiple of 8; the Go wrappers route
// narrow / non-8-aligned widths to the pure-Go reference so the asm only ever
// handles full 8-wide column groups and the byte-exactness contract stays easy
// to audit.
//
// Bit-exactness with wienerHorizontal / wienerVertical:
//   - Each source sample is 8/10/12-bit (<= 4095), so VPMOVZXWD zero-extends it
//     to a positive int32 lane and VPMULLD reproduces s_i*f_i exactly (the
//     product never overflows int32).
//   - The libaom "center reapplication" term s3<<7 is folded into the center tap
//     (tap3 += 1<<WienerFilterBits) by the wrappers, so the asm runs a plain
//     7-tap MAC.
//   - The rounding bias 1<<(round-1) is folded into the accumulator seed
//     (+offset horizontal, -offset vertical), then VPSRAD arithmetic-shifts the
//     accumulator right by `round`. This equals roundPowerOfTwo(sum, round)
//     bit-for-bit.
//   - The clamp is VPMAXSD against 0, VPMINSD against the broadcast upper bound,
//     then VPACKUSDW down to u16 (the values are already in range so the pack's
//     own saturation never triggers), identical to clampInt32(x, 0, bound).

// wienerAVX2HorizCtx is the asm calling context for the horizontal pass. Field
// order and sizes are part of the ABI shared with wiener_avx2_amd64.s; do not
// reorder. The int32 vectors (taps, seed, shift, maxCl) are broadcast from
// memory inside the asm, so they must stay 32-bit.
type wienerAVX2HorizCtx struct {
	dst    *uint16   // temp output base
	src    *uint16   // first tap sample of row -WienerHalfwin
	srcStr uintptr   // src stride in elements
	width  uintptr   // output columns (multiple of 8)
	rows   uintptr   // number of rows: height + 2*WienerHalfwin
	taps   *[7]int32 // 7 adjusted taps (tap3 += 128), int32 for VPMULLD
	seed   int32     // accumulator seed: offset + (1<<(round0-1))
	shift  int32     // round0 (positive shift count for VPSRAD)
	maxCl  int32     // upper clamp (maxClamp = limit-1)
}

// wienerAVX2VertCtx is the asm calling context for the vertical pass. Field
// order and sizes are part of the ABI shared with wiener_avx2_amd64.s; do not
// reorder.
type wienerAVX2VertCtx struct {
	dst    *uint16   // output base
	src    *uint16   // temp base (row 0)
	dstStr uintptr   // dst stride in elements
	srcStr uintptr   // temp stride in elements (== width)
	width  uintptr   // output columns (multiple of 8)
	rows   uintptr   // output rows (== height)
	taps   *[7]int32 // 7 adjusted taps (tap3 += 128), int32 for VPMULLD
	seed   int32     // accumulator seed: -offset + (1<<(round1-1))
	shift  int32     // round1
	maxCl  int32     // upper clamp (max)
}

//go:noescape
func wienerHorizontalAVX2Asm(ctx *wienerAVX2HorizCtx)

//go:noescape
func wienerVerticalAVX2Asm(ctx *wienerAVX2VertCtx)

// adjustedWienerTapsI32 returns the 7 distinct taps with the center tap raised
// by 1<<WienerFilterBits so the asm's plain 7-tap MAC reproduces the libaom
// s3<<WienerFilterBits center reapplication, widened to int32 for VPMULLD.
func adjustedWienerTapsI32(filter WienerFilter) [7]int32 {
	var t [7]int32
	for i := 0; i < 7; i++ {
		t[i] = int32(filter[i])
	}
	t[3] += 1 << WienerFilterBits
	return t
}

func wienerHorizontalAVX2(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) bool {
	if width < 8 || width%8 != 0 {
		return wienerHorizontal(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
	}
	// Replicate the pure-Go validity check: every sample touched by the inner
	// window must be <= max.
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
	taps := adjustedWienerTapsI32(filter)
	ctx := wienerAVX2HorizCtx{
		dst:    &temp[0],
		src:    &src[srcOrigin-WienerHalfwin*srcStride-WienerHalfwin],
		srcStr: uintptr(srcStride),
		width:  uintptr(width),
		rows:   uintptr(height + 2*WienerHalfwin),
		taps:   &taps,
		seed:   offset + roundBiasAVX2(round0),
		shift:  int32(round0),
		maxCl:  limit - 1,
	}
	wienerHorizontalAVX2Asm(&ctx)
	return true
}

func wienerVerticalAVX2(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	if width < 8 || width%8 != 0 {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
		return
	}
	offset := int32(1 << (bitDepth + round1 - 1))
	taps := adjustedWienerTapsI32(filter)
	ctx := wienerAVX2VertCtx{
		dst:    &dst[0],
		src:    &temp[0],
		dstStr: uintptr(dstStride),
		srcStr: uintptr(tempStride),
		width:  uintptr(width),
		rows:   uintptr(height),
		taps:   &taps,
		seed:   -offset + roundBiasAVX2(round1),
		shift:  int32(round1),
		maxCl:  int32(max),
	}
	wienerVerticalAVX2Asm(&ctx)
}

// roundBiasAVX2 returns the rounding term 1<<(bits-1) folded into the
// accumulator seed so the asm's arithmetic right shift reproduces
// roundPowerOfTwo.
func roundBiasAVX2(bits int) int32 {
	if bits <= 0 {
		return 0
	}
	return 1 << (bits - 1)
}
