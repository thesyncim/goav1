// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

// AVX2-accelerated 8-bit-pixel Wiener separable passes, the amd64 counterpart
// of wiener_u8_neon_arm64.go. Everything except the sample loads/stores is
// instruction-for-instruction the u16 AVX2 kernels in wiener_avx2_amd64.s:
//   - The horizontal pass loads each tap window with VPMOVZXBD (8 u8 samples
//     zero-extended to 8 s32 lanes), so the MACs see exactly the values the
//     u16 kernel loads with VPMOVZXWD. The 16-bit temp intermediate is
//     unchanged.
//   - The vertical pass clamps to [0,255] (VPMAXSD/VPMINSD), packs to u16
//     (VPACKUSDW, in-range so its saturation never triggers), then packs to
//     u8 (VPACKUSWB, likewise in-range) and stores 8 bytes.

// wienerU8AVX2HorizCtx is the asm calling context for the 8-bit horizontal
// pass. Field order and sizes are part of the ABI shared with
// wiener_u8_avx2_amd64.s; do not reorder.
type wienerU8AVX2HorizCtx struct {
	dst    *uint16   // temp output base (16-bit intermediate, unchanged)
	src    *uint8    // first tap sample of row -WienerHalfwin
	srcStr uintptr   // src stride in elements (== bytes for uint8)
	width  uintptr   // output columns (multiple of 8)
	rows   uintptr   // number of rows: height + 2*WienerHalfwin
	taps   *[7]int32 // 7 adjusted taps (tap3 += 128), int32 for VPMULLD
	seed   int32     // accumulator seed: offset + (1<<(round0-1))
	shift  int32     // round0 (positive shift count for VPSRAD)
	maxCl  int32     // upper clamp (maxClamp = limit-1)
}

// wienerU8AVX2VertCtx is the asm calling context for the 8-bit vertical pass.
// Field order and sizes are part of the ABI shared with
// wiener_u8_avx2_amd64.s; do not reorder.
type wienerU8AVX2VertCtx struct {
	dst    *uint8    // output base
	src    *uint16   // temp base (row 0)
	dstStr uintptr   // dst stride in elements (== bytes for uint8)
	srcStr uintptr   // temp stride in elements (== width)
	width  uintptr   // output columns (multiple of 8)
	rows   uintptr   // output rows (== height)
	taps   *[7]int32 // 7 adjusted taps (tap3 += 128), int32 for VPMULLD
	seed   int32     // accumulator seed: -offset + (1<<(round1-1))
	shift  int32     // round1
	maxCl  int32     // upper clamp (255)
}

//go:noescape
func wienerHorizontalU8AVX2Asm(ctx *wienerU8AVX2HorizCtx)

//go:noescape
func wienerVerticalU8AVX2Asm(ctx *wienerU8AVX2VertCtx)

func wienerHorizontalU8AVX2(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if width < 8 || width%8 != 0 {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	const bitDepth = 8
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	taps := adjustedWienerTapsI32(filter)
	ctx := wienerU8AVX2HorizCtx{
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
	wienerHorizontalU8AVX2Asm(&ctx)
}

func wienerVerticalU8AVX2(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if width < 8 || width%8 != 0 {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	const bitDepth = 8
	offset := int32(1 << (bitDepth + round1 - 1))
	taps := adjustedWienerTapsI32(filter)
	ctx := wienerU8AVX2VertCtx{
		dst:    &dst[0],
		src:    &temp[0],
		dstStr: uintptr(dstStride),
		srcStr: uintptr(tempStride),
		width:  uintptr(width),
		rows:   uintptr(height),
		taps:   &taps,
		seed:   -offset + roundBiasAVX2(round1),
		shift:  int32(round1),
		maxCl:  255,
	}
	wienerVerticalU8AVX2Asm(&ctx)
}
