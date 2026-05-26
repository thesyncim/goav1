// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package reconstruct

import (
	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// Block describes the residual transform block to reconstruct.
type Block struct {
	Size           transform.Size
	Transform      transform.Type
	Quantizer      quantize.Quantizer
	InverseQMatrix []uint16
	Lossless       bool
	EOB            int
}

// ScratchLen returns the int32 and int16 scratch lengths needed by cfg.
func ScratchLen(cfg Block) (int32Len int, int16Len int, err error) {
	if cfg.Lossless && !losslessWHTSupported(cfg) {
		return 0, 0, ErrInvalidBlock
	}
	if !cfg.Transform.Supported(cfg.Size) {
		return 0, 0, ErrInvalidBlock
	}
	blockLen, ok := checkedMul(cfg.Size.Width, cfg.Size.Height)
	if !ok {
		return 0, 0, ErrInvalidBlock
	}
	transformLen, err := transform.ScratchLenForType(cfg.Transform, cfg.Size)
	if err != nil {
		return 0, 0, ErrInvalidBlock
	}
	total32, ok := checkedAdd(blockLen, transformLen)
	if !ok {
		return 0, 0, ErrInvalidBlock
	}
	return total32, blockLen, nil
}

// ReconstructPlaneBlock dequantizes quantized coefficients, applies the inverse
// transform, and adds the resulting residual to a predicted plane block.
func ReconstructPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlock(dst, bytesPerSample, bitDepth, x, y, cfg.Size.Width, cfg.Size.Height, quantized, quantizedStride, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisible dequantizes a full transform block and writes
// only the visible rectangle. This handles frame-edge transform blocks whose
// coded transform extends beyond the clipped output plane.
func ReconstructPlaneBlockVisible(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlock(dst, bytesPerSample, bitDepth, x, y, visibleWidth, visibleHeight, quantized, quantizedStride, int32Scratch, residualScratch, cfg)
}

func reconstructPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	int32Len, residualLen, err := ScratchLen(cfg)
	if err != nil ||
		len(int32Scratch) < int32Len ||
		len(residualScratch) < residualLen {
		return ErrInvalidBlock
	}

	width := cfg.Size.Width
	height := cfg.Size.Height
	if visibleWidth <= 0 || visibleHeight <= 0 || visibleWidth > width || visibleHeight > height {
		return ErrInvalidBlock
	}
	coeffSize, err := transform.ScanSize(cfg.Size)
	if err != nil {
		return ErrInvalidBlock
	}
	dequantLen, ok := checkedMul(coeffSize.Width, coeffSize.Height)
	if !ok || residualLen < dequantLen {
		return ErrInvalidBlock
	}
	dequant := int32Scratch[:dequantLen]
	transformScratch := int32Scratch[residualLen:int32Len]
	residual := residualScratch[:residualLen]

	txScale, err := quantize.TransformScale(width, height)
	if err != nil {
		return ErrInvalidBlock
	}
	if cfg.InverseQMatrix != nil {
		if err := quantize.DequantizeBlockScaledQMatrixBitDepth(dequant, coeffSize.Height, quantized, quantizedStride, coeffSize.Width, coeffSize.Height, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else {
		if err := quantize.DequantizeBlockScaledBitDepth(dequant, coeffSize.Height, quantized, quantizedStride, coeffSize.Width, coeffSize.Height, cfg.Quantizer, txScale, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	}
	if cfg.Lossless {
		if err := transform.InverseWHT4x4Block(residual, width, dequant, coeffSize.Height, cfg.EOB); err != nil {
			return ErrInvalidBlock
		}
	} else if err := transform.InverseBlockBitDepth(residual, width, dequant, coeffSize.Height, transformScratch, cfg.Size, cfg.Transform, bitDepth); err != nil {
		return ErrInvalidBlock
	}
	if err := dsp.AddResidualPlaneBlock(dst, bytesPerSample, bitDepth, x, y, visibleWidth, visibleHeight, residual, width); err != nil {
		return ErrInvalidBlock
	}
	return nil
}

func losslessWHTSupported(cfg Block) bool {
	return cfg.Size == (transform.Size{Width: 4, Height: 4}) &&
		cfg.Transform == transform.TypeDCTDCT &&
		cfg.EOB >= 0
}

func checkedAdd(a int, b int) (int, bool) {
	c := a + b
	if c < a {
		return 0, false
	}
	return c, true
}

func checkedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}
