// Ported from libaom: av1/common/cfl.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package prediction

import "github.com/thesyncim/goav1/internal/av1/frame"

const (
	CFLBufLine   = 32
	CFLBufSquare = CFLBufLine * CFLBufLine
)

type CFLPredType uint8

const (
	CFLPredU CFLPredType = iota
	CFLPredV
)

const (
	cflAlphabetSizeLog2 = 4
	cflAlphabetSize     = 1 << cflAlphabetSizeLog2
	cflSignZero         = 0
	cflSignNegative     = 1
	cflSignPositive     = 2
	cflSigns            = 3
	cflJointSigns       = cflSigns*cflSigns - 1
)

// CFLAlphaQ3 ports libaom's cfl_idx_to_alpha helper.
func CFLAlphaQ3(alphaIndex uint8, jointSign int8, predType CFLPredType) (int, error) {
	if jointSign < 0 || int(jointSign) >= cflJointSigns {
		return 0, ErrInvalidPrediction
	}
	sign := cflSignU(int(jointSign))
	magnitude := int(alphaIndex) >> cflAlphabetSizeLog2
	switch predType {
	case CFLPredU:
	case CFLPredV:
		sign = cflSignV(int(jointSign))
		magnitude = int(alphaIndex) & (cflAlphabetSize - 1)
	default:
		return 0, ErrInvalidPrediction
	}
	switch sign {
	case cflSignZero:
		return 0, nil
	case cflSignPositive:
		return magnitude + 1, nil
	case cflSignNegative:
		return -magnitude - 1, nil
	default:
		return 0, ErrInvalidPrediction
	}
}

// SubsampleLuma8ToQ3 ports libaom's lbd cfl_luma_subsampling_*_c helpers.
// width and height describe the luma input block. Output uses CFLBufLine stride.
func SubsampleLuma8ToQ3(outputQ3 []uint16, input []uint8, inputStride int, width int, height int, subX bool, subY bool) error {
	outW, outH, err := validateCFLSubsample(outputQ3, len(input), inputStride, width, height, subX, subY)
	if err != nil {
		return err
	}
	subsampleLuma8Impl(outputQ3, input, inputStride, width, height, outW, outH, subX, subY)
	return nil
}

// SubsampleLuma16ToQ3 ports libaom's hbd cfl_luma_subsampling_*_c helpers.
// width and height describe the luma input block. Output uses CFLBufLine stride.
func SubsampleLuma16ToQ3(outputQ3 []uint16, input []uint16, inputStride int, width int, height int, subX bool, subY bool, bitDepth uint8) error {
	max, err := sampleMax(2, bitDepth)
	if err != nil {
		return err
	}
	outW, outH, err := validateCFLSubsample(outputQ3, len(input), inputStride, width, height, subX, subY)
	if err != nil {
		return err
	}
	return subsampleLuma16Impl(outputQ3, input, inputStride, width, height, outW, outH, subX, subY, max)
}

// PadCFLReconQ3 ports libaom's cfl_pad for frame-boundary overrun handling.
func PadCFLReconQ3(reconQ3 []uint16, bufWidth int, bufHeight int, width int, height int) (int, int, error) {
	if !validCFLSize(width, height) || bufWidth <= 0 || bufHeight <= 0 || bufWidth > width || bufHeight > height ||
		!fixedCFLBlockFits(len(reconQ3), width, height) {
		return 0, 0, ErrInvalidPrediction
	}
	if width > bufWidth {
		diffWidth := width - bufWidth
		for row := 0; row < bufHeight; row++ {
			line := row * CFLBufLine
			last := reconQ3[line+bufWidth-1]
			for col := range diffWidth {
				reconQ3[line+bufWidth+col] = last
			}
		}
		bufWidth = width
	}
	if height > bufHeight {
		for row := bufHeight; row < height; row++ {
			copy(reconQ3[row*CFLBufLine:row*CFLBufLine+width], reconQ3[(row-1)*CFLBufLine:(row-1)*CFLBufLine+width])
		}
		bufHeight = height
	}
	return bufWidth, bufHeight, nil
}

// SubtractCFLAverage ports libaom's subtract_average_c. Buffers use CFLBufLine stride.
func SubtractCFLAverage(srcQ3 []uint16, dstQ3 []int16, width int, height int) error {
	if !validCFLSize(width, height) ||
		!fixedCFLBlockFits(len(srcQ3), width, height) ||
		!fixedCFLBlockFits(len(dstQ3), width, height) {
		return ErrInvalidPrediction
	}
	numPelLog2, ok := log2PowerOfTwoInt(width * height)
	if !ok {
		return ErrInvalidPrediction
	}
	subtractCFLAverageImpl(srcQ3, dstQ3, width, height, numPelLog2)
	return nil
}

// PredictCFLPlaneBlock ports libaom's cfl_predict_lbd_c/hbd_c over a plane
// block that already contains the chroma DC prediction.
func PredictCFLPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, acQ3 []int16, alphaQ3 int) error {
	return PredictCFLPlaneBlockVisible(dst, bytesPerSample, bitDepth, x, y, width, height, width, height, acQ3, alphaQ3)
}

// PredictCFLPlaneBlockVisible applies CfL to a clipped visible rectangle using
// an AC buffer computed over the full padded CfL block.
func PredictCFLPlaneBlockVisible(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, acWidth int, acHeight int, acQ3 []int16, alphaQ3 int) error {
	block, err := planeBlockWindow(dst, bytesPerSample, x, y, visibleWidth, visibleHeight)
	if err != nil {
		return err
	}
	max, err := sampleMax(bytesPerSample, bitDepth)
	if err != nil {
		return err
	}
	if visibleWidth <= 0 || visibleHeight <= 0 || visibleWidth > acWidth || visibleHeight > acHeight ||
		!validCFLSize(acWidth, acHeight) || !fixedCFLBlockFits(len(acQ3), acWidth, acHeight) ||
		alphaQ3 < -16 || alphaQ3 > 16 {
		return ErrInvalidPrediction
	}
	applyCFLImpl(block, bytesPerSample, visibleWidth, visibleHeight, acQ3, alphaQ3, max)
	return nil
}

func validateCFLSubsample(outputQ3 []uint16, inputLen int, inputStride int, width int, height int, subX bool, subY bool) (int, int, error) {
	if width <= 0 || height <= 0 || width > CFLBufLine || height > CFLBufLine || inputStride < width {
		return 0, 0, ErrInvalidPrediction
	}
	if subX && width&1 != 0 {
		return 0, 0, ErrInvalidPrediction
	}
	if subY && height&1 != 0 {
		return 0, 0, ErrInvalidPrediction
	}
	if !blockFits(inputLen, inputStride, width, height) {
		return 0, 0, ErrInvalidPrediction
	}
	outW := width
	outH := height
	if subX {
		outW >>= 1
	}
	if subY {
		outH >>= 1
	}
	if outW <= 0 || outH <= 0 || !fixedCFLBlockFits(len(outputQ3), outW, outH) {
		return 0, 0, ErrInvalidPrediction
	}
	return outW, outH, nil
}

func validCFLSize(width int, height int) bool {
	switch {
	case width == 4 && (height == 4 || height == 8 || height == 16):
		return true
	case width == 8 && (height == 4 || height == 8 || height == 16 || height == 32):
		return true
	case width == 16 && (height == 4 || height == 8 || height == 16 || height == 32):
		return true
	case width == 32 && (height == 8 || height == 16 || height == 32):
		return true
	default:
		return false
	}
}

func fixedCFLBlockFits(length int, width int, height int) bool {
	return blockFits(length, CFLBufLine, width, height)
}

func readCFLPlaneSample(line []byte, bytesPerSample int, col int) int {
	if bytesPerSample == 1 {
		return int(line[col])
	}
	i := col * 2
	return int(uint16(line[i]) | uint16(line[i+1])<<8)
}

func writeCFLPlaneSample(line []byte, bytesPerSample int, col int, sample uint16) {
	if bytesPerSample == 1 {
		line[col] = byte(sample)
		return
	}
	i := col * 2
	line[i] = byte(sample)
	line[i+1] = byte(sample >> 8)
}

func cflSignU(jointSign int) int {
	return ((jointSign + 1) * 11) >> 5
}

func cflSignV(jointSign int) int {
	signU := cflSignU(jointSign)
	return jointSign + 1 - cflSigns*signU
}

func roundPowerOfTwoSigned(value int, bits int) int {
	if value < 0 {
		return -((-value + (1 << (bits - 1))) >> bits)
	}
	return (value + (1 << (bits - 1))) >> bits
}

func clampInt(v int, lo int, hi int) int {
	return min(max(v, lo), hi)
}

func blockFits(length int, stride int, width int, height int) bool {
	if stride <= 0 || width <= 0 || height <= 0 || stride < width {
		return false
	}
	lastRow, ok := checkedMul(height-1, stride)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastRow, width)
	return ok && needed <= length
}
