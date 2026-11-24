// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (amd64 || arm64) && !purego

package superres

import "simd/archsimd"

// Go-native SIMD super-res horizontal upscale row kernel (simd/archsimd,
// GOEXPERIMENT=simd). The AV1 normative super-res filter is a per-output-pixel
// 8-tap polyphase resampler: each destination pixel x derives its own source
// base column and subpel phase (and so its own 8-tap coefficient set) from a
// 14-bit fixed-point cursor srcX. srcX increases monotonically across the row,
// so the destination pixels whose full 8-tap window lies in bounds form one
// contiguous central span. That span is computed here; the few edge pixels
// that need per-tap clamping take the pure-Go path.
//
// Bit-exactness with upscaleRowPureGo (central span):
//   - Each output pixel loads the 8 source samples src[base..base+7] as one
//     Int16x8 and the 8 phase coefficients upscaleFilter[subpel] as one
//     Int16x8. Source samples are at most 4095, so the signed-16 view is exact.
//   - superresTapFold forms the eight 16x16 -> 32 products exactly and folds
//     them into four int32 partial sums per pixel. Eight pixels are batched:
//     pairwise adds (ConcatAddPairs) reduce the partials of all eight pixels to
//     one int32 sum per lane, which fits int32 for every <=12-bit input.
//   - The rounding and clamp run lane-wise in int32 and match the reference
//     clipInt(roundPowerOfTwo(sum, 7), 0, maxValue): (sum+64)>>7 is an
//     arithmetic shift, then Max(0) and Min(maxValue) clamp the lanes.

// upscaleRowSIMD is the Go SIMD row kernel. It is bit-identical to
// upscaleRowPureGo for every input the dispatcher routes here.
func upscaleRowSIMD(srcRow []uint16, dstRow []uint16, dstWidth, stepX, initialSubpelX, srcLast, maxValue int) {
	// Find the contiguous central span [xLo, xHi) whose full 8-tap window is
	// in bounds: base >= 0 && base+7 <= srcLast, where
	// base = ((-(1<<ScaleBits) + initialSubpelX + x*stepX) >> ScaleBits) - FilterOffset.
	// base is non-decreasing in x (stepX > 0), so the predicate is a single
	// contiguous interval.
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

	// Central span: every window [base, base+8) is inside srcRow by the
	// predicate above, so the raw 16-byte loads in superresFold stay in bounds.
	round := archsimd.BroadcastInt32x4(1 << (filterRoundBits - 1))
	zero := archsimd.BroadcastInt32x4(0)
	high := archsimd.BroadcastInt32x4(int32(maxValue))
	srcX := -(1 << ScaleBits) + initialSubpelX + xLo*stepX
	// A group's eight windows start (r0+k*stepX)>>ScaleBits samples after its
	// first column, r0 < 1<<ScaleBits, so they stay within 15 when
	// 7*stepX <= 15<<ScaleBits (every AV1 super-res ratio, at most 2x).
	winStep := 7*stepX <= 15<<ScaleBits
	x := xLo
	// Eight pixels' four partial sums each are held in registers (no array:
	// a stack-resident array of vectors spilled on every pixel). One checked
	// window holds all eight 8-tap windows; the windows near the row's end
	// that it would overrun take one checked load each.
	for ; winStep && x+8 <= xHi; x += 8 {
		w0 := (srcX >> ScaleBits) - FilterOffset
		if w0 > len(srcRow)-superresWinLen {
			break
		}
		w := (*[superresWinLen]uint16)(srcRow[w0 : w0+superresWinLen])
		r0, us := uint(srcX)&ScaleMask, uint(stepX)
		superresStore8(dstRow, x,
			superresFoldWin(w, r0), superresFoldWin(w, r0+us),
			superresFoldWin(w, r0+2*us), superresFoldWin(w, r0+3*us),
			superresFoldWin(w, r0+4*us), superresFoldWin(w, r0+5*us),
			superresFoldWin(w, r0+6*us), superresFoldWin(w, r0+7*us),
			round, zero, high)
		srcX += 8 * stepX
	}
	for ; x+8 <= xHi; x += 8 {
		superresStore8(dstRow, x,
			superresFold(srcRow, srcX), superresFold(srcRow, srcX+stepX),
			superresFold(srcRow, srcX+2*stepX), superresFold(srcRow, srcX+3*stepX),
			superresFold(srcRow, srcX+4*stepX), superresFold(srcRow, srcX+5*stepX),
			superresFold(srcRow, srcX+6*stepX), superresFold(srcRow, srcX+7*stepX),
			round, zero, high)
		srcX += 8 * stepX
	}
	for ; x < xHi; x++ {
		t := superresFold(srcRow, srcX)
		q := t.ConcatAddPairs(t)
		sum := q.ConcatAddPairs(q).GetElem(0)
		dstRow[x] = uint16(clipInt(roundPowerOfTwo(int(sum), filterRoundBits), 0, maxValue))
		srcX += stepX
	}

	// Trailing edge pixels (boundary-clamped).
	for x := xHi; x < dstWidth; x++ {
		upscaleOnePureGo(srcRow, dstRow, x, stepX, initialSubpelX, srcLast, maxValue)
	}
}

// superresFold computes the four partial int32 sums of the 8-tap dot product
// for the output pixel whose fixed-point source cursor is srcX: it loads the
// window srcRow[base..base+7] and the phase coefficients upscaleFilter[subpel]
// exactly as upscaleRowPureGo derives them. The caller guarantees the window is
// in bounds (central span), so the checked 16-byte load never fails.
func superresFold(srcRow []uint16, srcX int) archsimd.Int32x4 {
	base := (srcX >> ScaleBits) - FilterOffset
	filter := &upscaleFilter[(srcX&ScaleMask)>>ExtraBits]
	s := archsimd.LoadUint16x8Array((*[8]uint16)(srcRow[base : base+8])).BitsToInt16()
	return superresTapFold(s, archsimd.LoadInt16x8Array(filter))
}

// superresStore8 reduces eight pixels' four partial sums to one sum each,
// rounds, clamps and stores them at dstRow[x:x+8]. Two levels of pairwise adds
// do the reduction: pairs of pixels interleave first, then the pair results
// gather the four pixel sums (one per lane).
func superresStore8(dstRow []uint16, x int, t0, t1, t2, t3, t4, t5, t6, t7, round, zero, high archsimd.Int32x4) {
	lo := t0.ConcatAddPairs(t1).ConcatAddPairs(t2.ConcatAddPairs(t3))
	hi := t4.ConcatAddPairs(t5).ConcatAddPairs(t6.ConcatAddPairs(t7))
	lo = lo.Add(round).ShiftAllRight(filterRoundBits).Max(zero).Min(high)
	hi = hi.Add(round).ShiftAllRight(filterRoundBits).Max(zero).Min(high)
	superresPack8(lo, hi).StoreArray((*[8]uint16)(dstRow[x:]))
}

// superresWinLen spans eight consecutive 8-tap windows: starts 0..15 past
// the first window plus the 8 taps.
const superresWinLen = 24

// superresFoldWin is superresFold for a window w starting at source column
// (srcX>>ScaleBits)-FilterOffset of the group's first pixel; r is the pixel's
// cursor relative to that column, so its window starts r>>ScaleBits samples
// in, a value in [0, 15] that the mask states for the bounds prover.
func superresFoldWin(w *[superresWinLen]uint16, r uint) archsimd.Int32x4 {
	d := (r >> ScaleBits) & 15
	filter := &upscaleFilter[(r&ScaleMask)>>ExtraBits]
	s := archsimd.LoadUint16x8Array((*[8]uint16)(w[d : d+8])).BitsToInt16()
	return superresTapFold(s, archsimd.LoadInt16x8Array(filter))
}

// superresHi4 moves the high four 16-bit lanes of x into the low four lanes
// with one byte-granular concatenate-shift (EXT / PALIGNR by 8 bytes); the high
// four lanes of the result are not used.
func superresHi4(x archsimd.Int16x8) archsimd.Int16x8 {
	b := x.ToBits().ReshapeToUint8s()
	return b.ConcatShiftBytesRight(b, 8).ReshapeToUint16s().BitsToInt16()
}

// upscaleOnePureGo computes a single output pixel using the pure-Go reference
// arithmetic, including per-tap boundary clamping. It handles the edge pixels
// the Go SIMD kernel does not cover.
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
