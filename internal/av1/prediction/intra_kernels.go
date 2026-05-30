// Ported from libaom: av1/common/reconintra.c and av1/common/cfl.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package prediction

// This file holds the canonical pure-Go references for the dispatched DC,
// CfL and directional-interpolation kernels. They are the bit-exact contract
// every SIMD variant (NEON on arm64, AVX2 on amd64) must reproduce. The DC sum,
// the CfL subsample/apply and the directional row interpolation all funnel
// through these functions, with the SIMD variants bound in the per-arch
// dispatch files and falling back here for the shapes/bit-depths they do not
// special-case.

// sumSamplesPureGo returns the integer sum of the edge samples. The DC
// predictor adds the above and left reductions before the rounding division.
func sumSamplesPureGo(samples []uint16) int {
	sum := 0
	for _, s := range samples {
		sum += int(s)
	}
	return sum
}

// applyCFLPureGo applies the CfL residual to the already-written chroma DC
// prediction in block. It mirrors PredictCFLPlaneBlockVisible's inner loop
// exactly: scaled = roundPowerOfTwoSigned(alpha*ac, 6); out = clamp(dc+scaled).
func applyCFLPureGo(block planeBlock, bytesPerSample int, visibleWidth int, visibleHeight int, acQ3 []int16, alphaQ3 int, max uint16) {
	for row := 0; row < visibleHeight; row++ {
		line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
		for col := 0; col < visibleWidth; col++ {
			scaled := roundPowerOfTwoSigned(alphaQ3*int(acQ3[row*CFLBufLine+col]), 6)
			current := readCFLPlaneSample(line, bytesPerSample, col)
			writeCFLPlaneSample(line, bytesPerSample, col, uint16(clampInt(current+scaled, 0, int(max))))
		}
	}
}

// subsampleLuma8PureGo subsamples an 8-bit luma block to Q3. It reproduces the
// three libaom lbd cfl_luma_subsampling_*_c reductions and shifts exactly.
func subsampleLuma8PureGo(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool) {
	switch {
	case subX && subY:
		for row := 0; row < height; row += 2 {
			outRow := row >> 1
			for col := 0; col < width; col += 2 {
				bot := (row+1)*inputStride + col
				sum := int(input[row*inputStride+col]) + int(input[row*inputStride+col+1]) +
					int(input[bot]) + int(input[bot+1])
				outputQ3[outRow*CFLBufLine+(col>>1)] = uint16(sum << 1)
			}
		}
	case subX:
		for row := 0; row < outH; row++ {
			for col := 0; col < width; col += 2 {
				sum := int(input[row*inputStride+col]) + int(input[row*inputStride+col+1])
				outputQ3[row*CFLBufLine+(col>>1)] = uint16(sum << 2)
			}
		}
	default:
		for row := 0; row < outH; row++ {
			for col := 0; col < outW; col++ {
				outputQ3[row*CFLBufLine+col] = uint16(input[row*inputStride+col]) << 3
			}
		}
	}
}

// dirRowInterp8PureGo fills one 8-bit destination row of the Z1/Z3 directional
// predictor with the non-upsampled (baseInc==1) per-column interpolation. base
// advances by one per column; once base reaches maxBase the remaining columns
// clamp to above[maxBase]. The interpolation matches the pure-Go reference's
// roundPowerOfTwo(p0*(32-shift)+p1*shift, 5).
func dirRowInterp8PureGo(dst []byte, above []uint16, base int, shift int, maxBase int, width int) {
	for col := 0; col < width; col++ {
		var value uint16
		if base < maxBase {
			p0 := int(above[base])
			p1 := int(above[base+1])
			value = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
		} else {
			value = above[maxBase]
		}
		dst[col] = byte(value)
		base++
	}
}
