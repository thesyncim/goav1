// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

// NEON-accelerated 8-bit-pixel Wiener separable passes. These follow dav1d's
// 8bpc kernels (src/arm/64/looprestoration.S wiener_filter7: the horizontal
// pass loads uint8 pixels and widens with UXTL inside the kernel; the vertical
// pass narrows the int32 accumulator straight to uint8 with saturating
// narrows) mapped onto goav1's existing two-pass kernel structure: everything
// except the sample loads/stores is instruction-for-instruction the u16
// kernels in wiener_neon_arm64.s, so bit-exactness with the pure-Go u8
// reference follows from the same argument (see wiener_neon_arm64.go), plus:
//   - UXTL/UXTL2 zero-extend uint8 samples to the same positive int16 lanes
//     the u16 kernel loads directly, so the widening MACs see identical
//     inputs.
//   - The final u8 store is SQXTUN (s32 -> u16 saturating, lower bound 0)
//     followed by UQXTN (u16 -> u8 saturating, upper bound 255), which equals
//     clampInt32(x, 0, 255) exactly for every int32 x.
//
// Like the u16 horizontal kernel, the u8 horizontal kernel loads a full
// 16-sample window per 8 outputs and consumes only the first 14 lanes, so its
// last load of each row reaches up to 2 samples past the 3-pixel border
// (identical overshoot contract to the production u16 kernel; the extra lanes
// never contribute to any output).

// wienerU8NEONHorizCtx is the asm calling context for the 8-bit horizontal
// pass. Field order and sizes are part of the ABI shared with
// wiener_u8_neon_arm64.s; do not reorder.
type wienerU8NEONHorizCtx struct {
	dst    *uint16 // temp output base (16-bit intermediate, unchanged)
	src    *uint8  // first tap sample of row -WienerHalfwin (== srcStart for that row)
	srcStr uintptr // src stride in elements (== bytes for uint8)
	width  uintptr // output columns (multiple of 8)
	rows   uintptr // number of rows: height + 2*WienerHalfwin
	taps   *int16  // 8 lanes: 7 adjusted taps (tap3 += 128) + 1 pad
	seed   int32   // accumulator seed: offset + (1<<(round0-1))
	shift  int32   // -round0 (negative => arithmetic right shift)
	maxCl  uint16  // upper clamp (maxClamp = limit-1)
}

// wienerU8NEONVertCtx is the asm calling context for the 8-bit vertical pass.
// Field order and sizes are part of the ABI shared with
// wiener_u8_neon_arm64.s; do not reorder. There is no maxCl field: the [0,255]
// clamp is hardwired by the SQXTUN+UQXTN narrowing chain.
type wienerU8NEONVertCtx struct {
	dst    *uint8  // output base
	src    *uint16 // temp base (row 0)
	dstStr uintptr // dst stride in elements (== bytes for uint8)
	srcStr uintptr // temp stride in elements (== width)
	width  uintptr // output columns (multiple of 8)
	rows   uintptr // output rows (== height)
	taps   *int16  // 8 lanes: 7 adjusted taps (tap3 += 128) + 1 pad
	seed   int32   // accumulator seed: -offset + (1<<(round1-1))
	shift  int32   // -round1
}

//go:noescape
func wienerHorizontalU8NEONAsm(ctx *wienerU8NEONHorizCtx)

//go:noescape
func wienerVerticalU8NEONAsm(ctx *wienerU8NEONVertCtx)

func wienerHorizontalU8NEON(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if width < 8 || width%8 != 0 {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	// The u8 window load (ld1 {v1.16b}) pulls 16 u8 for the last 8-column group,
	// reaching 2 samples past the width+2*WienerHalfwin reslice the scalar
	// reference validates. Those trailing lanes never feed a stored result, but
	// they must be resident: require the 2-sample trailing pad (a borderedBlockFits
	// check with a width widened by 2) before dispatching; otherwise run scalar.
	if !borderedBlockFits(len(src), srcStride, srcOrigin, width+2, height, WienerHalfwin, WienerHalfwin) {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	const bitDepth = 8
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	offset := int32(1 << (bitDepth + WienerFilterBits - 1))
	taps := adjustedWienerTaps(filter)
	ctx := wienerU8NEONHorizCtx{
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
	wienerHorizontalU8NEONAsm(&ctx)
}

func wienerVerticalU8NEON(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if width < 8 || width%8 != 0 {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	const bitDepth = 8
	offset := int32(1 << (bitDepth + round1 - 1))
	taps := adjustedWienerTaps(filter)
	ctx := wienerU8NEONVertCtx{
		dst:    &dst[0],
		src:    &temp[0],
		dstStr: uintptr(dstStride),
		srcStr: uintptr(tempStride),
		width:  uintptr(width),
		rows:   uintptr(height),
		taps:   &taps[0],
		seed:   -offset + roundBias(round1),
		shift:  int32(-round1),
	}
	wienerVerticalU8NEONAsm(&ctx)
}
