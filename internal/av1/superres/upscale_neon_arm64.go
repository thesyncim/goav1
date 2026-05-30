// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package superres

// NEON-accelerated super-res horizontal upscale row kernel.
//
// The AV1 normative super-res filter is a per-output-pixel 8-tap polyphase
// resampler: each destination pixel x derives its own source base column and
// its own subpel phase (and hence its own 8-tap coefficient set) from a 14-bit
// fixed-point cursor srcX. Because srcX increases monotonically across the row,
// the destination pixels whose full 8-tap source window lies in bounds form a
// single contiguous central span. The asm kernel processes that span; the few
// edge pixels that need per-tap clamping fall back to the pure-Go reference.
//
// Bit-exactness with upscaleRowPureGo (central span):
//   - For each output pixel the 8 source samples src[base..base+7] are loaded
//     as a uint16x8 vector and the 8 phase coefficients upscaleFilter[subpel]
//     as an int16x8 vector. The source values are non-negative (<= 4095) so a
//     signed widening multiply reproduces the int product exactly.
//   - smull/smlal2 widen the lane products to int32 and addv reduces them to
//     the scalar sum, identical to the unrolled int accumulator.
//   - srshr-equivalent rounding: (sum + 64) >> 7 == roundPowerOfTwo(sum, 7).
//     The reference sum fits well within int32 for all <=12-bit inputs, so the
//     scalar rounding/clamp is computed in GP registers exactly as the
//     reference clipInt(roundPowerOfTwo(sum, 7), 0, maxValue).

// upscaleRowNEONCtx is the asm calling context. Field order and sizes are part
// of the ABI shared with upscale_neon_arm64.s; do not reorder.
type upscaleRowNEONCtx struct {
	src      *uint16 // &srcRow[0]
	dst      *uint16 // &dstRow[xLo] (first central output pixel)
	filter   *int16  // &upscaleFilter[0][0]
	count    int64   // number of central output pixels to produce
	srcX0    int64   // srcX cursor at the first central pixel
	stepX    int64   // srcX increment per output pixel
	maxValue int64   // (1<<bitDepth)-1
}

//go:noescape
func upscaleRowNEONAsm(ctx *upscaleRowNEONCtx)

// upscaleRowNEON implements the dispatch slot using the NEON asm kernel for the
// in-bounds central span and the pure-Go reference for the boundary-clamped
// edge pixels. It is bit-identical to upscaleRowPureGo.
func upscaleRowNEON(srcRow []uint16, dstRow []uint16, dstWidth, stepX, initialSubpelX, srcLast, maxValue int) {
	// Find the contiguous central span [xLo, xHi) whose full 8-tap window is
	// in bounds: base >= 0 && base+7 <= srcLast, where
	// base = ((-(1<<ScaleBits) + initialSubpelX + x*stepX) >> ScaleBits) - FilterOffset.
	// base is non-decreasing in x (stepX > 0), so the predicate is a single
	// contiguous interval. Solve the bounds with integer math, scanning from
	// the analytic estimates to stay exact against the reference truncation.
	xLo := 0
	for xLo < dstWidth {
		srcX := -(1 << ScaleBits) + initialSubpelX + xLo*stepX
		base := (srcX >> ScaleBits) - FilterOffset
		if base >= 0 && base+FilterTaps-1 <= srcLast {
			break
		}
		xLo++
	}
	xHi := dstWidth
	for xHi > xLo {
		x := xHi - 1
		srcX := -(1 << ScaleBits) + initialSubpelX + x*stepX
		base := (srcX >> ScaleBits) - FilterOffset
		if base >= 0 && base+FilterTaps-1 <= srcLast {
			break
		}
		xHi--
	}

	// Leading edge pixels (boundary-clamped).
	for x := 0; x < xLo; x++ {
		upscaleOnePureGo(srcRow, dstRow, x, stepX, initialSubpelX, srcLast, maxValue)
	}

	if xHi > xLo {
		srcX0 := -(1 << ScaleBits) + initialSubpelX + xLo*stepX
		ctx := upscaleRowNEONCtx{
			src:      &srcRow[0],
			dst:      &dstRow[xLo],
			filter:   &upscaleFilter[0][0],
			count:    int64(xHi - xLo),
			srcX0:    int64(srcX0),
			stepX:    int64(stepX),
			maxValue: int64(maxValue),
		}
		upscaleRowNEONAsm(&ctx)
	}

	// Trailing edge pixels (boundary-clamped).
	for x := xHi; x < dstWidth; x++ {
		upscaleOnePureGo(srcRow, dstRow, x, stepX, initialSubpelX, srcLast, maxValue)
	}
}

// upscaleOnePureGo computes a single output pixel using the pure-Go reference
// arithmetic, including per-tap boundary clamping. Used for the edge pixels of
// the NEON path.
func upscaleOnePureGo(srcRow []uint16, dstRow []uint16, x, stepX, initialSubpelX, srcLast, maxValue int) {
	srcX := -(1 << ScaleBits) + initialSubpelX + x*stepX
	srcXPx := srcX >> ScaleBits
	srcXSubpel := (srcX & ScaleMask) >> ExtraBits
	filter := &upscaleFilter[srcXSubpel]
	base := srcXPx - FilterOffset
	var sum int
	for k := range FilterTaps {
		sampleX := clipInt(base+k, 0, srcLast)
		sum += int(srcRow[sampleX]) * int(filter[k])
	}
	dstRow[x] = uint16(clipInt(roundPowerOfTwo(sum, filterRoundBits), 0, maxValue))
}
