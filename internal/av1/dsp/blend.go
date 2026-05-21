package dsp

const blendA64MaxAlpha = 64

// BlendA64Mask blends src0 and src1 into dst using libaom's AOM_BLEND_A64
// alpha mask. Strides are expressed in samples, not bytes. subX and subY match
// libaom's subw/subh mask subsampling flags.
func BlendA64Mask(dst []uint16, dstStride int, src0 []uint16, src0Stride int, src1 []uint16, src1Stride int, mask []uint8, maskStride int, width int, height int, subX bool, subY bool, bitDepth uint8) error {
	max, err := sampleMaxForBitDepth(bitDepth)
	if err != nil {
		return err
	}
	if width <= 0 || height <= 0 || !isPowerOfTwo(width) || !isPowerOfTwo(height) {
		return ErrInvalidBlock
	}
	if dstStride < width || src0Stride < width || src1Stride < width || maskStride < maskWidth(width, subX) {
		return ErrInvalidBlock
	}
	if !sampleBlockFits(len(dst), dstStride, width, height) ||
		!sampleBlockFits(len(src0), src0Stride, width, height) ||
		!sampleBlockFits(len(src1), src1Stride, width, height) ||
		!sampleBlockFits(len(mask), maskStride, maskWidth(width, subX), maskHeight(height, subY)) {
		return ErrInvalidBlock
	}

	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			s0 := src0[row*src0Stride+col]
			s1 := src1[row*src1Stride+col]
			if s0 > max || s1 > max {
				return ErrInvalidBlock
			}
			m, err := blendMaskSample(mask, maskStride, row, col, subX, subY)
			if err != nil {
				return err
			}
			dst[row*dstStride+col] = blendA64(m, s0, s1)
		}
	}
	return nil
}

func blendMaskSample(mask []uint8, maskStride int, row int, col int, subX bool, subY bool) (uint8, error) {
	var m int
	switch {
	case !subX && !subY:
		m = int(mask[row*maskStride+col])
	case subX && subY:
		m = roundPowerOfTwoPositive(
			int(mask[(2*row)*maskStride+2*col])+
				int(mask[(2*row+1)*maskStride+2*col])+
				int(mask[(2*row)*maskStride+2*col+1])+
				int(mask[(2*row+1)*maskStride+2*col+1]),
			2,
		)
	case subX:
		m = blendAvg(int(mask[row*maskStride+2*col]), int(mask[row*maskStride+2*col+1]))
	default:
		m = blendAvg(int(mask[(2*row)*maskStride+col]), int(mask[(2*row+1)*maskStride+col]))
	}
	if m < 0 || m > blendA64MaxAlpha {
		return 0, ErrInvalidBlock
	}
	return uint8(m), nil
}

func blendA64(alpha uint8, src0 uint16, src1 uint16) uint16 {
	a := int(alpha)
	value := a*int(src0) + (blendA64MaxAlpha-a)*int(src1)
	return uint16(roundPowerOfTwoPositive(value, 6))
}

func blendAvg(a int, b int) int {
	return roundPowerOfTwoPositive(a+b, 1)
}

func roundPowerOfTwoPositive(value int, bits int) int {
	return (value + (1 << (bits - 1))) >> bits
}

func sampleMaxForBitDepth(bitDepth uint8) (uint16, error) {
	switch bitDepth {
	case 8, 10, 12:
		return uint16((1 << bitDepth) - 1), nil
	default:
		return 0, ErrInvalidBlock
	}
}

func sampleBlockFits(length int, stride int, width int, height int) bool {
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

func maskWidth(width int, subX bool) int {
	if subX {
		return 2 * width
	}
	return width
}

func maskHeight(height int, subY bool) int {
	if subY {
		return 2 * height
	}
	return height
}

func isPowerOfTwo(v int) bool {
	return v > 0 && v&(v-1) == 0
}
