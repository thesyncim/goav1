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

const sparseDequantWorkFactor = 8
const nonzeroDequantWorkFactor = 16

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
	return reconstructPlaneBlockTrustedAtWithGeometry(dst, dstStride, bytesPerSample, bitDepth, visibleWidth, visibleHeight, quantized, quantizedStride, scan, nil, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisibleTrustedAtWithGeometryScanAndNonZero is the
// decoder hot path used when coefficient decode already produced a compact
// non-zero raster-position list.
func ReconstructPlaneBlockVisibleTrustedAtWithGeometryScanAndNonZero(dst []byte, dstStride int, bytesPerSample int, bitDepth uint8, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, nonzero []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return reconstructPlaneBlockTrustedAtWithGeometry(dst, dstStride, bytesPerSample, bitDepth, visibleWidth, visibleHeight, quantized, quantizedStride, scan, nonzero, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisibleTrustedAtWithDequantized is the fused decoder
// path for a tracked 2D TXB whose entropy reader has already written
// dequantized coefficients into int32Scratch[:scanSize.Width*scanSize.Height].
// The ordinary quantized reconstruction path remains separate so public
// visitors, deferred reconstruction, and unsupported coefficient modes retain
// their established behavior and error ordering. As with the existing trusted
// entry points, the caller validates geometry and scratch capacity and only
// uses this for non-lossless, no-qmatrix blocks; dcOnly must match an EOB-1
// DCT_DCT block. Those contracts are intentionally not rechecked here.
func ReconstructPlaneBlockVisibleTrustedAtWithDequantized(dst []byte, dstStride int, bytesPerSample int, bitDepth uint8, visibleWidth int, visibleHeight int, scanSize transform.Size, activeRows int, dcOnly bool, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	return ReconstructPlaneBlockVisibleTrustedAtFromDequantized(dst, dstStride, bytesPerSample, bitDepth, visibleWidth, visibleHeight, scanSize, activeRows, 0, dcOnly, int32Scratch, int32Scratch, residualScratch, cfg)
}

// ReconstructPlaneBlockVisibleTrustedAtFromDequantized is
// ReconstructPlaneBlockVisibleTrustedAtWithDequantized with the dequantized
// coefficients in their own slice (dav1d's frame-thread cf arena). dequant
// may alias the start of int32Scratch; the transform scratch begins after
// the coded block.
func ReconstructPlaneBlockVisibleTrustedAtFromDequantized(dst []byte, dstStride int, bytesPerSample int, bitDepth uint8, visibleWidth int, visibleHeight int, scanSize transform.Size, activeRows int, activeCols int, dcOnly bool, dequantSrc []int32, int32Scratch []int32, residualScratch []int16, cfg Block) error {
	width := int(cfg.Size.Width)
	height := int(cfg.Size.Height)
	scanWidth := int(scanSize.Width)
	scanHeight := int(scanSize.Height)
	blockLen := width * height
	dequantLen := scanWidth * scanHeight
	needed32 := blockLen
	if cfg.Transform != transform.TypeIDTX {
		needed32 += blockLen
	}
	dequant := dequantSrc[:dequantLen:dequantLen]
	// The transform scratch always begins after the full coded block, including
	// 64x64 TXBs whose coefficient scan is adjusted to 32x32.
	transformScratch := int32Scratch[blockLen:needed32:needed32]
	residual := residualScratch[:blockLen:blockLen]
	if dcOnly {
		sample := transform.InverseDCTDCOnlySample(dequant[0], cfg.Size, bitDepth)
		max := uint16((1 << bitDepth) - 1)
		dsp.AddConstantResidualPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, sample)
		return nil
	}
	if cfg.Transform == transform.TypeIDTX {
		if err := transform.InverseBlockBitDepth(residual, width, dequant, scanHeight, transformScratch, cfg.Size, cfg.Transform, bitDepth); err != nil {
			return ErrInvalidBlock
		}
		max := uint16((1 << bitDepth) - 1)
		dsp.AddResidualPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, residual, width)
		return nil
	}
	// dav1d's two-pass structure with the add fused (transform itx_fused*):
	// the eob-bounded 8-bit square DCT_DCT block, and the 16x16 and 8x8
	// DCT/ADST pairings, reconstruct straight from the coefficients into
	// dst; a declined block takes the staged path.
	if bitDepth == 8 && bytesPerSample == 1 && width == height && visibleWidth == width && visibleHeight == height &&
		activeRows > 0 && activeCols > 0 {
		if cfg.Transform == transform.TypeDCTDCT {
			switch width {
			case 32:
				if activeRows <= 8 && activeCols <= 8 {
					if transform.InverseAddDCT32x32Low8(dst, dstStride, dst, dstStride, dequant, scanHeight) {
						return nil
					}
				} else if activeRows <= 16 && activeCols <= 16 &&
					transform.InverseAddDCT32x32Low16(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows) {
					return nil
				} else if transform.InverseAddDCT32x32Full(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols) {
					return nil
				}
			case 64:
				if transform.InverseAddDCT64x64(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols) {
					return nil
				}
			case 16:
				if transform.InverseAddDCT16x16(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols) {
					return nil
				}
			case 8:
				if transform.InverseAddDCT8x8(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols) {
					return nil
				}
			}
		} else if width == 16 {
			if transform.InverseAdd16x16Hybrid(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
				return nil
			}
		} else if width == 8 {
			if transform.InverseAdd8x8Hybrid(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
				return nil
			}
		}
	} else if bytesPerSample == 2 && width == height &&
		visibleWidth == width && visibleHeight == height && activeRows > 0 && activeCols > 0 {
		// The 10- and 12-bit square DCT_DCT blocks (8x8 to 64x64) and 16x16 / 8x8
		// DCT/ADST pairings: the same int16 passes under the same guards,
		// added to 16bpc samples (dav1d itx16 reuses its 8bpc column
		// functions at 10 bits).
		if cfg.Transform == transform.TypeDCTDCT {
			switch width {
			case 64:
				if transform.InverseAddDCT64x64HBD(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols, bitDepth) {
					return nil
				}
			case 32:
				if transform.InverseAddDCT32x32HBD(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols, bitDepth) {
					return nil
				}
			case 16:
				if transform.InverseAddDCT16x16HBD(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols, bitDepth) {
					return nil
				}
			case 8:
				if transform.InverseAddDCT8x8HBD(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols, bitDepth) {
					return nil
				}
			}
		} else if width == 16 {
			if transform.InverseAdd16x16HybridHBD(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols, bitDepth) {
				return nil
			}
		} else if width == 8 {
			if transform.InverseAdd8x8HybridHBD(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols, bitDepth) {
				return nil
			}
		}
	} else if bitDepth == 8 && bytesPerSample == 1 && cfg.Transform == transform.TypeDCTDCT &&
		visibleWidth == width && visibleHeight == height && activeRows > 0 && activeCols > 0 {
		// The 2:1 rectangular DCT_DCT blocks (rect2 coefficient scale).
		if width == 16 && height == 32 {
			if transform.InverseAddDCT16x32(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols) {
				return nil
			}
		} else if width == 32 && height == 16 {
			if transform.InverseAddDCT32x16(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols) {
				return nil
			}
		}
	}
	if err := transform.InverseBlockBitDepthRawBounded(transformScratch, dequant, scanHeight, cfg.Size, cfg.Transform, bitDepth, activeRows, activeCols); err != nil {
		return ErrInvalidBlock
	}
	max := uint16((1 << bitDepth) - 1)
	dsp.AddRawTransformPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, transformScratch, width)
	return nil
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
	return reconstructPlaneBlockTrustedAtWithGeometry(dst.Pix[dstOffset:dstOffset+dstLen:dstOffset+dstLen], dst.Stride, bytesPerSample, bitDepth, visibleWidth, visibleHeight, quantized, quantizedStride, scan, nil, scanSize, txScale, int32Scratch, residualScratch, cfg)
}

func reconstructPlaneBlockTrustedAtWithGeometry(dst []byte, dstStride int, bytesPerSample int, bitDepth uint8, visibleWidth int, visibleHeight int, quantized []int16, quantizedStride int, scan []int16, nonzero []int16, scanSize transform.Size, txScale uint8, int32Scratch []int32, residualScratch []int16, cfg Block) error {
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
	if !cfg.Lossless && cfg.Transform == transform.TypeDCTDCT && eob == 1 {
		dc := quantize.DequantizeDCCoeffBitDepthTrusted(quantized[0], cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth)
		sample := transform.InverseDCTDCOnlySample(dc, cfg.Size, bitDepth)
		max := uint16((1 << bitDepth) - 1)
		dsp.AddConstantResidualPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, sample)
		return nil
	}

	useNonZeroDequant := len(nonzero) > 0 && len(nonzero)*nonzeroDequantWorkFactor <= dequantLen
	useSparseDequant := !useNonZeroDequant && eob > 0 && len(scan) >= eob && eob*sparseDequantWorkFactor <= dequantLen
	activeRows := 0
	if useSparseDequant {
		activeRows = activeTransformRowsFromScan(scan, eob, scanHeight, cfg.Size, cfg.Transform)
	} else if !useNonZeroDequant && !cfg.Lossless && cfg.Transform != transform.TypeIDTX && eob > 0 {
		// Trusted decoder coefficients are zero outside the decoded EOB prefix.
		// Reuse the exact O(1) row bound only for the immutable default scan;
		// copied or custom scans retain the full-row fallback.
		if class, err := cfg.Transform.Class(); err == nil {
			if rows, ok := transform.DefaultScanActiveRowsForScan(cfg.Size, class, scan, eob); ok && rows > 0 {
				activeRows = rows
			}
		}
	}
	if cfg.InverseQMatrix != nil {
		if useNonZeroDequant {
			activeRows = quantize.DequantizeBlockScaledQMatrixBitDepthNonZeroRowsTrusted(dequant, scanHeight, quantized, quantizedStride, nonzero, scanWidth, scanHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth)
		} else if useSparseDequant {
			quantize.DequantizeBlockScaledQMatrixBitDepthEOBTrusted(dequant, scanHeight, quantized, quantizedStride, scan, eob, scanWidth, scanHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth)
		} else if err := quantize.DequantizeBlockScaledQMatrixBitDepth(dequant, scanHeight, quantized, quantizedStride, scanWidth, scanHeight, cfg.Quantizer, txScale, cfg.InverseQMatrix, bitDepth); err != nil {
			return ErrInvalidBlock
		}
	} else {
		if useNonZeroDequant {
			activeRows = quantize.DequantizeBlockScaledBitDepthNonZeroRowsTrusted(dequant, scanHeight, quantized, quantizedStride, nonzero, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth)
		} else if useSparseDequant {
			quantize.DequantizeBlockScaledBitDepthEOBTrusted(dequant, scanHeight, quantized, quantizedStride, scan, eob, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth)
		} else {
			quantize.DequantizeBlockScaledBitDepthFullTrusted(dequant, scanHeight, quantized, quantizedStride, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth)
		}
	}
	if cfg.Lossless {
		if err := transform.InverseWHT4x4Block(residual, width, dequant, scanHeight, eob); err != nil {
			return ErrInvalidBlock
		}
		max := uint16((1 << bitDepth) - 1)
		dsp.AddResidualPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, residual, width)
		return nil
	}
	if cfg.Transform == transform.TypeIDTX {
		if err := transform.InverseBlockBitDepth(residual, width, dequant, scanHeight, transformScratch, cfg.Size, cfg.Transform, bitDepth); err != nil {
			return ErrInvalidBlock
		}
		max := uint16((1 << bitDepth) - 1)
		dsp.AddResidualPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, residual, width)
		return nil
	}
	if err := transform.InverseBlockBitDepthRaw(transformScratch, dequant, scanHeight, cfg.Size, cfg.Transform, bitDepth, activeRows); err != nil {
		return ErrInvalidBlock
	}
	max := uint16((1 << bitDepth) - 1)
	dsp.AddRawTransformPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, transformScratch, width)
	return nil
}

func activeTransformRowsFromScan(scan []int16, eob int, scanHeight int, size transform.Size, typ transform.Type) int {
	if class, err := typ.Class(); err == nil {
		if activeRows, ok := transform.DefaultScanActiveRowsForScan(size, class, scan, eob); ok {
			return activeRows
		}
	}
	activeRows := 0
	for i := 0; i < eob; i++ {
		pos := int(scan[i])
		row := pos - (pos/scanHeight)*scanHeight
		if row >= activeRows {
			activeRows = row + 1
		}
	}
	return activeRows
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
	useSparseDequant := eob > 0 && len(scan) >= eob && eob*sparseDequantWorkFactor <= dequantLen
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
