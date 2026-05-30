// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package prediction

// AVX2-accelerated intra static predictors (PAETH, SMOOTH and the two 1D
// variants), the CfL apply and the DC edge sum. The .s file implements the
// per-row inner loops for the common 8-bit path with width a multiple of 16;
// the Go wrappers route narrower widths and high-bit-depth blocks to the
// pure-Go reference, keeping the asm single-path and the byte-exactness
// contract easy to audit.
//
// Bit-exactness with the *PureGo references:
//   - PAETH computes base = top + left - topLeft in int16 lanes; the three abs
//     diffs use VPABSW and the libaom tie-break order (left, then top, then
//     topLeft) is reproduced with VPCMPGTW masks and blends, matching
//     paethPredictorSingle.
//   - SMOOTH accumulates the four weighted terms in u32 lanes and rounds with
//     (sum + 256) >> 9 (VPADDD + VPSRLD), identical to divideRound; the 1D
//     variants drop one weighted pair and shift by 8.
//   - applyCFL reproduces sign(ac*alpha)*((|ac*alpha|+32)>>6), the widened DC
//     add and the [0,255] clamp.
//   - sumSamples folds uint16 lanes into u64 accumulators.

//go:noescape
func predictPaethAVX2Asm(ctx *paethAVX2Ctx)

//go:noescape
func predictSmoothAVX2Asm(ctx *smoothAVX2Ctx)

//go:noescape
func predictSmooth1DAVX2Asm(ctx *smooth1DAVX2Ctx)

//go:noescape
func applyCFLRowAVX2Asm(dst *byte, ac *int16, alpha uint64, n uint64)

//go:noescape
func sumSamplesAVX2Asm(samples *uint16, n uint64) uint64

// paethAVX2Ctx is the asm calling context. Field order is part of the ABI
// shared with intra_avx2_amd64.s; do not reorder.
type paethAVX2Ctx struct {
	dst       *byte
	above     *uint16
	left      *uint16
	dstStr    uintptr
	width     uintptr
	height    uintptr
	aboveLeft uintptr
}

type smoothAVX2Ctx struct {
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

// smooth1DAVX2Ctx serves SMOOTH_V (primary=above, weights row-indexed,
// secondary=belowPred) and SMOOTH_H (primary=left, weights col-indexed,
// secondary=rightPred). isH selects the direction inside the asm.
type smooth1DAVX2Ctx struct {
	dst       *byte
	primary   *uint16
	weights   *uint16
	dstStr    uintptr
	width     uintptr
	height    uintptr
	secondary uintptr
	isH       uintptr
}

func predictPaethAVX2(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	if bytesPerSample != 1 || block.width%16 != 0 {
		predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
		return
	}
	ctx := paethAVX2Ctx{
		dst:       &block.pix[0],
		above:     &above[0],
		left:      &left[0],
		dstStr:    uintptr(block.stride),
		width:     uintptr(block.width),
		height:    uintptr(block.height),
		aboveLeft: uintptr(aboveLeft),
	}
	predictPaethAVX2Asm(&ctx)
}

func predictSmoothAVX2(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	if bytesPerSample != 1 || block.width%16 != 0 {
		predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
		return
	}
	ctx := smoothAVX2Ctx{
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
	predictSmoothAVX2Asm(&ctx)
}

func predictSmoothVerticalAVX2(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	if bytesPerSample != 1 || block.width%16 != 0 {
		predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
		return
	}
	ctx := smooth1DAVX2Ctx{
		dst:       &block.pix[0],
		primary:   &above[0],
		weights:   &weights[0],
		dstStr:    uintptr(block.stride),
		width:     uintptr(block.width),
		height:    uintptr(block.height),
		secondary: uintptr(belowPred),
		isH:       0,
	}
	predictSmooth1DAVX2Asm(&ctx)
}

func predictSmoothHorizontalAVX2(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	if bytesPerSample != 1 || block.width%16 != 0 {
		predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
		return
	}
	ctx := smooth1DAVX2Ctx{
		dst:       &block.pix[0],
		primary:   &left[0],
		weights:   &weights[0],
		dstStr:    uintptr(block.stride),
		width:     uintptr(block.width),
		height:    uintptr(block.height),
		secondary: uintptr(rightPred),
		isH:       1,
	}
	predictSmooth1DAVX2Asm(&ctx)
}

func applyCFLAVX2(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	if bytesPerSample != 1 || visibleWidth%16 != 0 || max != 0xff {
		applyCFLPureGo(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
		return
	}
	for row := 0; row < visibleHeight; row++ {
		line := block.pix[row*block.stride:]
		ac := acQ3[row*CFLBufLine:]
		applyCFLRowAVX2Asm(&line[0], &ac[0], uint64(uint32(int32(alphaQ3))), uint64(visibleWidth))
	}
}

func sumSamplesAVX2(samples []uint16) int {
	n := len(samples)
	chunks := n &^ 15
	total := 0
	if chunks > 0 {
		total = int(sumSamplesAVX2Asm(&samples[0], uint64(chunks)))
	}
	for i := chunks; i < n; i++ {
		total += int(samples[i])
	}
	return total
}
