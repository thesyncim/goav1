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
	InverseQMatrix []uint16
	Quantizer      quantize.Quantizer
	Size           transform.Size
	Transform      transform.Type
	EOB            int16
	Lossless       bool
}

// ScratchLen returns the int32 and int16 scratch lengths needed by cfg.
func ScratchLen(cfg Block) (int32Len int, int16Len int, err error) {
	if cfg.Lossless && !losslessWHTSupported(cfg) {
		return 0, 0, ErrInvalidBlock
	}
	if !cfg.Transform.Supported(cfg.Size) {
		return 0, 0, ErrInvalidBlock
	}
	blockLen, ok := checkedMul(int(cfg.Size.Width), int(cfg.Size.Height))
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
	return reconstructPlaneBlock(dst, bytesPerSample, bitDepth, x, y, int(cfg.Size.Width), int(cfg.Size.Height), quantized, quantizedStride, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisible dequantizes a full transform block and writes
// only the visible rectangle. This handles frame-edge transform blocks whose
// coded transform extends beyond the clipped output plane.
func ReconstructPlaneBlockVisible(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlock(dst, bytesPerSample, bitDepth, x, y, visibleWidth, visibleHeight, quantized, quantizedStride, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisibleWithGeometry is the hot-path variant of
// ReconstructPlaneBlockVisible for callers that already resolved the adjusted
// coefficient scan size and transform scale while deriving block geometry.
func ReconstructPlaneBlockVisibleWithGeometry(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlockWithGeometry(dst, bytesPerSample, bitDepth, x, y, visibleWidth, visibleHeight, quantized, quantizedStride, nil, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisibleWithGeometryAndScan is
// ReconstructPlaneBlockVisibleWithGeometry with the transform scan order
// supplied by the coefficient decoder. Sparse blocks can then dequantize only
// the EOB prefix instead of multiplying the full coefficient rectangle.
func ReconstructPlaneBlockVisibleWithGeometryAndScan(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlockWithGeometry(dst, bytesPerSample, bitDepth, x, y, visibleWidth, visibleHeight, quantized, quantizedStride, scan, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisibleTrustedWithGeometryAndScan is the decoder hot
// path after FrameWorkBatch has resolved and validated block geometry, plane
// windows, transform shape, scratch capacity, and visible extents.
func ReconstructPlaneBlockVisibleTrustedWithGeometryAndScan(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlockTrustedWithGeometry(dst, bytesPerSample, bitDepth, x, y, visibleWidth, visibleHeight, quantized, quantizedStride, scan, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisibleTrustedAtWithGeometryAndScan is the trusted
// decoder hot path when the caller already sliced dst to the block origin.
func ReconstructPlaneBlockVisibleTrustedAtWithGeometryAndScan(dst []byte, dstStride int, bytesPerSample int, bitDepth uint8, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlockTrustedAtWithGeometry(dst, dstStride, bytesPerSample, bitDepth, visibleWidth, visibleHeight, quantized, quantizedStride, scan, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

func reconstructPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	int32Len, residualLen, err := ScratchLen(cfg)
	if err != nil ||
		len(int32Scratch) < int32Len ||
		len(residualScratch) < residualLen {
		return ErrInvalidBlock
	}

	width := int(cfg.Size.Width)
	height := int(cfg.Size.Height)
	if visibleWidth <= 0 || visibleHeight <= 0 || visibleWidth > width || visibleHeight > height {
		return ErrInvalidBlock
	}
	coeffSize, err := transform.ScanSize(cfg.Size)
	if err != nil {
		return ErrInvalidBlock
	}
	coeffWidth := int(coeffSize.Width)
	coeffHeight := int(coeffSize.Height)
	dequantLen, ok := checkedMul(coeffWidth, coeffHeight)
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
		if err := quantize.DequantizeBlockScaledQMatrixBitDepth(dequant, coeffHeight, quantized, quantizedStride, coeffWidth, coeffHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else {
		if err := quantize.DequantizeBlockScaledBitDepth(dequant, coeffHeight, quantized, quantizedStride, coeffWidth, coeffHeight, cfg.Quantizer, txScale, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	}
	eob := int(cfg.EOB)
	if cfg.Lossless {
		if err := transform.InverseWHT4x4Block(residual, width, dequant, coeffHeight, eob); err != nil {
			return ErrInvalidBlock
		}
	} else if cfg.Transform == transform.TypeDCTDCT && eob == 1 {
		if err := transform.InverseDCTDCOnlyBlockBitDepth(residual, width, dequant[0], transformScratch, cfg.Size, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else if err := transform.InverseBlockBitDepth(residual, width, dequant, coeffHeight, transformScratch, cfg.Size, cfg.Transform, bitDepth); err != nil {
		return ErrInvalidBlock
	}
	if err := dsp.AddResidualPlaneBlock(dst, bytesPerSample, bitDepth, x, y, visibleWidth, visibleHeight, residual, width); err != nil {
		return ErrInvalidBlock
	}
	return nil
}

func reconstructPlaneBlockTrustedWithGeometry(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	dstOffset := y*dst.Stride + x*bytesPerSample
	rowBytes := visibleWidth * bytesPerSample
	dstLen := (visibleHeight-1)*dst.Stride + rowBytes
	return reconstructPlaneBlockTrustedAtWithGeometry(dst.Pix[dstOffset:dstOffset+dstLen:dstOffset+dstLen], dst.Stride, bytesPerSample, bitDepth, visibleWidth, visibleHeight, quantized, quantizedStride, scan, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

func reconstructPlaneBlockTrustedAtWithGeometry(dst []byte, dstStride int, bytesPerSample int, bitDepth uint8, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	width := int(cfg.Size.Width)
	height := int(cfg.Size.Height)
	blockLen := width * height
	scanWidth := int(scanSize.Width)
	scanHeight := int(scanSize.Height)
	dequantLen := scanWidth * scanHeight
	needed32 := blockLen
	if !cfg.Lossless && cfg.Transform != transform.TypeIDTX {
		needed32 += blockLen
	}

	dequant := int32Scratch[:dequantLen:dequantLen]
	transformScratch := int32Scratch[blockLen:needed32:needed32]
	residual := residualScratch[:blockLen:blockLen]

	eob := int(cfg.EOB)
	useSparseDequant := eob > 0 && len(scan) >= eob && eob*4 <= dequantLen
	if cfg.InverseQMatrix != nil {
		if useSparseDequant {
			quantize.DequantizeBlockScaledQMatrixBitDepthEOBTrusted(dequant, scanHeight, quantized, quantizedStride, scan, eob, scanWidth, scanHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth)
		} else if err := quantize.DequantizeBlockScaledQMatrixBitDepth(dequant, scanHeight, quantized, quantizedStride, scanWidth, scanHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else {
		if useSparseDequant {
			quantize.DequantizeBlockScaledBitDepthEOBTrusted(dequant, scanHeight, quantized, quantizedStride, scan, eob, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth)
		} else if err := quantize.DequantizeBlockScaledBitDepth(dequant, scanHeight, quantized, quantizedStride, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	}
	if cfg.Lossless {
		if err := transform.InverseWHT4x4Block(residual, width, dequant, scanHeight, eob); err != nil {
			return ErrInvalidBlock
		}
	} else if cfg.Transform == transform.TypeDCTDCT && eob == 1 {
		if err := transform.InverseDCTDCOnlyBlockBitDepth(residual, width, dequant[0], transformScratch, cfg.Size, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else if err := transform.InverseBlockBitDepth(residual, width, dequant, scanHeight, transformScratch, cfg.Size, cfg.Transform, bitDepth); err != nil {
		return ErrInvalidBlock
	}
	max := uint16((1 << bitDepth) - 1)
	dsp.AddResidualPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, residual, width)
	return nil
}

func reconstructPlaneBlockWithGeometry(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	width := int(cfg.Size.Width)
	height := int(cfg.Size.Height)
	scanWidth := int(scanSize.Width)
	scanHeight := int(scanSize.Height)
	if width <= 0 || height <= 0 ||
		visibleWidth <= 0 || visibleHeight <= 0 || visibleWidth > width || visibleHeight > height ||
		scanWidth <= 0 || scanHeight <= 0 || scanWidth > width || scanHeight > height ||
		txScale > 2 {
		return ErrInvalidBlock
	}
	if cfg.Lossless && !losslessWHTSupported(cfg) {
		return ErrInvalidBlock
	}
	blockLen, ok := checkedMul(width, height)
	if !ok || len(residualScratch) < blockLen || len(int32Scratch) < blockLen {
		return ErrInvalidBlock
	}
	dequantLen, ok := checkedMul(scanWidth, scanHeight)
	if !ok || dequantLen > blockLen {
		return ErrInvalidBlock
	}
	needed32 := blockLen
	if !cfg.Lossless && cfg.Transform != transform.TypeIDTX {
		needed32 += blockLen
	}
	if len(int32Scratch) < needed32 {
		return ErrInvalidBlock
	}

	dequant := int32Scratch[:dequantLen]
	transformScratch := int32Scratch[blockLen:needed32]
	residual := residualScratch[:blockLen]

	eob := int(cfg.EOB)
	useSparseDequant := eob > 0 && len(scan) >= eob && eob*4 <= dequantLen
	if cfg.InverseQMatrix != nil {
		if useSparseDequant {
			if err := quantize.DequantizeBlockScaledQMatrixBitDepthEOB(dequant, scanHeight, quantized, quantizedStride, scan, eob, scanWidth, scanHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth); err != nil {
				return ErrInvalidBlock
			}
		} else if err := quantize.DequantizeBlockScaledQMatrixBitDepth(dequant, scanHeight, quantized, quantizedStride, scanWidth, scanHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else {
		if useSparseDequant {
			if err := quantize.DequantizeBlockScaledBitDepthEOB(dequant, scanHeight, quantized, quantizedStride, scan, eob, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth); err != nil {
				return ErrInvalidBlock
			}
		} else if err := quantize.DequantizeBlockScaledBitDepth(dequant, scanHeight, quantized, quantizedStride, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	}
	if cfg.Lossless {
		if err := transform.InverseWHT4x4Block(residual, width, dequant, scanHeight, eob); err != nil {
			return ErrInvalidBlock
		}
	} else if cfg.Transform == transform.TypeDCTDCT && eob == 1 {
		if err := transform.InverseDCTDCOnlyBlockBitDepth(residual, width, dequant[0], transformScratch, cfg.Size, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else if err := transform.InverseBlockBitDepth(residual, width, dequant, scanHeight, transformScratch, cfg.Size, cfg.Transform, bitDepth); err != nil {
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
