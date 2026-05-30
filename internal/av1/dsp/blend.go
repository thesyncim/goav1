// Ported from libaom: aom_dsp/blend_a64_mask.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package dsp

const blendA64MaxAlpha = 64

// blendA64MaskArgs bundles the structurally-validated arguments handed to the
// per-block blend kernel. Extents and strides are already known to fit; the
// kernel still enforces the per-sample range (s0,s1 <= max) and per-weight
// range (m <= 64), reporting any violation back to BlendA64Mask.
type blendA64MaskArgs struct {
	dst        []uint16
	dstStride  int
	src0       []uint16
	src0Stride int
	src1       []uint16
	src1Stride int
	mask       []uint8
	maskStride int
	width      int
	height     int
	max        uint16
	subX       bool
	subY       bool
}

// blendA64MaskImpl is the dispatch slot for the BlendA64Mask inner loop. It is
// resolved exactly once at package init (see blend_dispatch_*.go), so the
// per-call cost is a single indirect call. It returns false if any sample or
// mask weight was out of range, in which case BlendA64Mask reports
// ErrInvalidBlock.
var blendA64MaskImpl = blendA64MaskPureGo

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

	if !blendA64MaskImpl(blendA64MaskArgs{
		dst: dst, dstStride: dstStride,
		src0: src0, src0Stride: src0Stride,
		src1: src1, src1Stride: src1Stride,
		mask: mask, maskStride: maskStride,
		width: width, height: height, max: max,
		subX: subX, subY: subY,
	}) {
		return ErrInvalidBlock
	}
	return nil
}

// blendA64MaskPureGo is the portable reference for the blend inner loop. Every
// SIMD variant the dispatcher selects MUST be bit-exact with it on valid input
// and MUST agree on the valid/invalid verdict. Structural validation has
// already been performed by BlendA64Mask.
func blendA64MaskPureGo(a blendA64MaskArgs) bool {
	for row := range a.height {
		for col := range a.width {
			s0 := a.src0[row*a.src0Stride+col]
			s1 := a.src1[row*a.src1Stride+col]
			if s0 > a.max || s1 > a.max {
				return false
			}
			m, err := blendMaskSample(a.mask, a.maskStride, row, col, a.subX, a.subY)
			if err != nil {
				return false
			}
			a.dst[row*a.dstStride+col] = blendA64(m, s0, s1)
		}
	}
	return true
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
