// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

// NEON-accelerated intra static predictors. The .s file implements the per-row
// inner loops for the common 8-bit path with width a multiple of 8 (every AV1
// intra block width >= 8). The Go wrappers below resolve base pointers and route
// narrow (width 4) or high-bit-depth blocks to the pure-Go reference, which
// keeps the asm to a single code path and the byte-exactness contract easy to
// audit.
//
// Bit-exactness with the *PureGo references:
//   - PAETH computes base = top + left - topLeft in int16 lanes (values fit:
//     0..255 inputs, base in [-255, 510], diffs in [0, 510]); the three abs
//     diffs and the libaom tie-break order (left, then top, then topLeft) are
//     reproduced with lane compares and bit-selects identical to
//     paethPredictorSingle.
//   - SMOOTH accumulates weightsH*above + (256-weightsH)*below +
//     weightsW*left + (256-weightsW)*right in u32 lanes (max ~261120 fits in
//     u32) and rounds with (sum + (1<<(bits-1))) >> bits, identical to
//     divideRound; the 1D variants drop the unused pair of terms and shift by 8.

// paethNEONCtx is the asm calling context. Field order and sizes are part of
// the ABI shared with intra_neon_arm64.s; do not reorder.
type paethNEONCtx struct {
	dst       *byte
	above     *uint16
	left      *uint16
	dstStr    uintptr
	width     uintptr
	height    uintptr
	aboveLeft uintptr
}

// smoothNEONCtx is the asm calling context for the full SMOOTH predictor.
type smoothNEONCtx struct {
	dst       *byte
	above     *uint16
	left      *uint16
	weightsW  *uint16
	weightsH  *uint16
	dstStr    uintptr
	width     uintptr
	height    uintptr
	belowPred uintptr
	rightPred uintptr
}

// smooth1DNEONCtx is the asm calling context for the SMOOTH_V / SMOOTH_H
// predictors. The meaning of the fields differs per direction (see wrappers).
type smooth1DNEONCtx struct {
	dst       *byte
	primary   *uint16 // SMOOTH_V: above[]; SMOOTH_H: left[]
	weights   *uint16 // SMOOTH_V: weightsH (row-indexed); SMOOTH_H: weightsW (col-indexed)
	dstStr    uintptr
	width     uintptr
	height    uintptr
	secondary uintptr // SMOOTH_V: belowPred; SMOOTH_H: rightPred
}

//go:noescape
func predictPaethNEONAsm(ctx *paethNEONCtx)

//go:noescape
func predictSmoothNEONAsm(ctx *smoothNEONCtx)

//go:noescape
func predictSmoothVerticalNEONAsm(ctx *smooth1DNEONCtx)

//go:noescape
func predictSmoothHorizontalNEONAsm(ctx *smooth1DNEONCtx)

func predictPaethNEON(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
		return
	}
	ctx := paethNEONCtx{
		dst:       &block.pix[0],
		above:     &above[0],
		left:      &left[0],
		dstStr:    uintptr(block.stride),
		width:     uintptr(block.width),
		height:    uintptr(block.height),
		aboveLeft: uintptr(aboveLeft),
	}
	predictPaethNEONAsm(&ctx)
}

func predictSmoothNEON(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
		return
	}
	ctx := smoothNEONCtx{
		dst:       &block.pix[0],
		above:     &above[0],
		left:      &left[0],
		weightsW:  &weightsW[0],
		weightsH:  &weightsH[0],
		dstStr:    uintptr(block.stride),
		width:     uintptr(block.width),
		height:    uintptr(block.height),
		belowPred: uintptr(belowPred),
		rightPred: uintptr(rightPred),
	}
	predictSmoothNEONAsm(&ctx)
}

func predictSmoothVerticalNEON(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
		return
	}
	ctx := smooth1DNEONCtx{
		dst:       &block.pix[0],
		primary:   &above[0],
		weights:   &weights[0],
		dstStr:    uintptr(block.stride),
		width:     uintptr(block.width),
		height:    uintptr(block.height),
		secondary: uintptr(belowPred),
	}
	predictSmoothVerticalNEONAsm(&ctx)
}

func predictSmoothHorizontalNEON(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	if bytesPerSample != 1 || block.width%8 != 0 {
		predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
		return
	}
	ctx := smooth1DNEONCtx{
		dst:       &block.pix[0],
		primary:   &left[0],
		weights:   &weights[0],
		dstStr:    uintptr(block.stride),
		width:     uintptr(block.width),
		height:    uintptr(block.height),
		secondary: uintptr(rightPred),
	}
	predictSmoothHorizontalNEONAsm(&ctx)
}
