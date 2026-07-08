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

// subsampleLuma16PureGo subsamples a high-bit-depth luma block to Q3 and keeps
// the scalar range checks from SubsampleLuma16ToQ3.
func subsampleLuma16PureGo(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, outW int, outH int, subX bool, subY bool, max uint16) error {
	switch {
	case subX && subY:
		for row := 0; row < height; row += 2 {
			outRow := row >> 1
			for col := 0; col < width; col += 2 {
				p0 := input[row*inputStride+col]
				p1 := input[row*inputStride+col+1]
				p2 := input[(row+1)*inputStride+col]
				p3 := input[(row+1)*inputStride+col+1]
				if p0 > max || p1 > max || p2 > max || p3 > max {
					return ErrInvalidPrediction
				}
				outputQ3[outRow*CFLBufLine+(col>>1)] = (p0 + p1 + p2 + p3) << 1
			}
		}
	case subX:
		for row := 0; row < outH; row++ {
			for col := 0; col < width; col += 2 {
				p0 := input[row*inputStride+col]
				p1 := input[row*inputStride+col+1]
				if p0 > max || p1 > max {
					return ErrInvalidPrediction
				}
				outputQ3[row*CFLBufLine+(col>>1)] = (p0 + p1) << 2
			}
		}
	default:
		for row := 0; row < outH; row++ {
			for col := 0; col < outW; col++ {
				p := input[row*inputStride+col]
				if p > max {
					return ErrInvalidPrediction
				}
				outputQ3[row*CFLBufLine+col] = p << 3
			}
		}
	}
	return nil
}

// subtractCFLAveragePureGo subtracts the rounded block average from Q3 luma
// samples. Buffers use CFLBufLine stride and are already size-validated.
func subtractCFLAveragePureGo(srcQ3 []uint16, dstQ3 []int16, width int, height int, numPelLog2 int) {
	sum := (width * height) >> 1
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			sum += int(srcQ3[row*CFLBufLine+col])
		}
	}
	avg := sum >> numPelLog2
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			dstQ3[row*CFLBufLine+col] = int16(int(srcQ3[row*CFLBufLine+col]) - avg)
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

// dirAboveRun8PureGo fills count contiguous 8-bit destination bytes for the
// zone-2 "above" branch and the unclamped portion of any forward edge run. ref
// is the edge slice positioned at the first source index (so ref[0] and ref[1]
// are the first pixel's neighbour pair); the source index advances by one per
// output. No clamp: callers (zone 2, validated reference span) only invoke this
// for runs fully inside the reference. The interpolation is identical to
// roundPowerOfTwo(ref[i]*(32-shift)+ref[i+1]*shift, 5).
func dirAboveRun8PureGo(dst []byte, ref []uint16, shift int, count int) {
	for i := 0; i < count; i++ {
		p0 := int(ref[i])
		p1 := int(ref[i+1])
		dst[i] = byte(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
	}
}

// dirLeftCol8PureGo fills count 8-bit destination samples down a single column
// of the zone-3 predictor. ref is the left edge positioned at the first source
// index (ref[0]/ref[1] are the first row's neighbour pair); the source index
// advances by one per row. dst holds the column's top sample and stride is the
// per-row byte step. No clamp: callers only invoke this for the run of rows
// strictly inside [base, maxBase). The interpolation matches
// roundPowerOfTwo(ref[i]*(32-shift)+ref[i+1]*shift, 5).
func dirLeftCol8PureGo(dst []byte, stride int, ref []uint16, shift int, count int) {
	for i := 0; i < count; i++ {
		p0 := int(ref[i])
		p1 := int(ref[i+1])
		dst[i*stride] = byte(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
	}
}
