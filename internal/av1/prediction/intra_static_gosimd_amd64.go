// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package prediction

// amd64 routers for the 8-bit intra predictors. The AVX2 asm this replaces served
// 8-bit blocks with width a multiple of 16; the Go SIMD kernels serve every width
// a multiple of 8 and every other shape stays on the PureGo reference.

func predictPaethSIMD(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		paeth8SIMD(block, above, left, aboveLeft)
		return
	}
	predictPaethPureGo(block, bytesPerSample, above, left, aboveLeft)
}

func predictSmoothSIMD(block planeBlock, bytesPerSample int, weightsW []uint16, weightsH []uint16, above []uint16, left []uint16, belowPred uint16, rightPred uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		smooth8SIMD(block, weightsW, weightsH, above, left, belowPred, rightPred)
		return
	}
	predictSmoothPureGo(block, bytesPerSample, weightsW, weightsH, above, left, belowPred, rightPred)
}

func predictSmoothVerticalSIMD(block planeBlock, bytesPerSample int, weights []uint16, above []uint16, belowPred uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		smoothVertical8SIMD(block, weights, above, belowPred)
		return
	}
	predictSmoothVerticalPureGo(block, bytesPerSample, weights, above, belowPred)
}

func predictSmoothHorizontalSIMD(block planeBlock, bytesPerSample int, weights []uint16, left []uint16, rightPred uint16) {
	if bytesPerSample == 1 && block.width%8 == 0 {
		smoothHorizontal8SIMD(block, weights, left, rightPred)
		return
	}
	predictSmoothHorizontalPureGo(block, bytesPerSample, weights, left, rightPred)
}

// applyCFLSIMD routes the 8-bit CfL apply with width a multiple of 8 and max 255
// to the Go SIMD kernel; every other shape uses the PureGo reference.
func applyCFLSIMD(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	if bytesPerSample == 1 && visibleWidth%8 == 0 && max == 0xff {
		applyCFL8SIMD(block, visibleWidth, visibleHeight, acQ3, alphaQ3)
		return
	}
	applyCFLPureGo(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
}
