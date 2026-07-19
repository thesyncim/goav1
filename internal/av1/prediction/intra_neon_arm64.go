// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

// NEON-accelerated intra static predictors. intra_neon_arm64.s implements the
// per-row inner loops for the 8-bit path with width a multiple of 8;
// intra_static16_neon_arm64.s adds the high-bit-depth (bytesPerSample==2)
// counterparts (identical arithmetic, uint16 stores); intra_static_w4_neon_arm64.s
// adds the width-4 kernels (two rows packed per vector). The Go wrappers below
// resolve base pointers and pick the kernel for the block shape, falling back to
// the pure-Go reference for the shapes none of the kernels cover (high-bit-depth
// width 4, and the odd-height width-4 frame-edge remainder).
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
	// AV1 block widths are one of {4,8,16,32,64}, so width%8==0 selects the
	// 8-wide kernels and width==4 the packed two-rows-per-iteration kernels.
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
	case bytesPerSample == 2 && block.width%8 == 0:
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
	default:
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
	switch {
	case bytesPerSample == 2:
		predictPaeth16NEONAsm(&ctx)
	case block.width == 4:
		predictPaethW4NEONAsm(&ctx)
	default:
		predictPaethNEONAsm(&ctx)
	}
}

func predictSmoothNEON(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
	case bytesPerSample == 2 && block.width%8 == 0:
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
	default:
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
	switch {
	case bytesPerSample == 2:
		predictSmooth16NEONAsm(&ctx)
	case block.width == 4:
		predictSmoothW4NEONAsm(&ctx)
	default:
		predictSmoothNEONAsm(&ctx)
	}
}

func predictSmoothVerticalNEON(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
	case bytesPerSample == 2 && block.width%8 == 0:
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
	default:
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
	switch {
	case bytesPerSample == 2:
		predictSmoothVertical16NEONAsm(&ctx)
	case block.width == 4:
		predictSmoothVerticalW4NEONAsm(&ctx)
	default:
		predictSmoothVerticalNEONAsm(&ctx)
	}
}

func predictSmoothHorizontalNEON(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	switch {
	case bytesPerSample == 1 && block.width%8 == 0:
	case bytesPerSample == 2 && block.width%8 == 0:
	case bytesPerSample == 1 && block.width == 4 && block.height%2 == 0:
	default:
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
	switch {
	case bytesPerSample == 2:
		predictSmoothHorizontal16NEONAsm(&ctx)
	case block.width == 4:
		predictSmoothHorizontalW4NEONAsm(&ctx)
	default:
		predictSmoothHorizontalNEONAsm(&ctx)
	}
}
