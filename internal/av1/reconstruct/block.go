// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package reconstruct

import (
	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// Block describes the residual transform block to reconstruct.
type Block struct {
	Size      transform.Size
	Transform transform.Type
	EOB       int16
	Lossless  bool
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

// ReconstructPlaneBlockVisibleTrustedAtFromDequantized is dav1d's itxfm_add
// for one TXB whose coefficients decode_coefs already dequantized into
// dequantSrc (t->cf or the frame-thread cf arena): the inverse transform of
// cfg.Transform added to dst, which the caller sliced to the block origin.
// The caller validates geometry and scratch capacity; dcOnly must match an
// EOB-1 DCT_DCT block. dequantSrc may alias the start of int32Scratch; the
// transform scratch begins after the coded block.
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
	if cfg.Transform == transform.TypeWHTWHT {
		if err := transform.InverseWHT4x4Block(residual, width, dequant, scanHeight, int(cfg.EOB)); err != nil {
			return ErrInvalidBlock
		}
		max := uint16((1 << bitDepth) - 1)
		dsp.AddResidualPlaneBlockTrusted(dst, dstStride, bytesPerSample, max, visibleWidth, visibleHeight, residual, width)
		return nil
	}
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
		if width == 4 {
			// Every 4x4 type but IDTX (amd64: dav1d's inv_txfm_add_4x4).
			if transform.InverseAdd4x4(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows) {
				return nil
			}
		} else if cfg.Transform == transform.TypeDCTDCT {
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
	} else if bitDepth == 8 && bytesPerSample == 1 && width*height == 32 &&
		visibleWidth == width && visibleHeight == height && activeRows > 0 && activeCols > 0 {
		// The 4x8 and 8x4 blocks of every type but IDTX (amd64: dav1d's
		// inv_txfm_add_4x8 / inv_txfm_add_8x4).
		if width == 4 {
			if transform.InverseAdd4x8(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
				return nil
			}
		} else if transform.InverseAdd8x4(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
			return nil
		}
	} else if bitDepth == 8 && bytesPerSample == 1 && width*height == 64 && width != height &&
		visibleWidth == width && visibleHeight == height && activeRows > 0 && activeCols > 0 {
		// The 4x16 and 16x4 blocks of every type but IDTX (amd64: dav1d's
		// inv_txfm_add_4x16 / inv_txfm_add_16x4).
		if width == 4 {
			if transform.InverseAdd4x16(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
				return nil
			}
		} else if transform.InverseAdd16x4(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
			return nil
		}
	} else if bitDepth == 8 && bytesPerSample == 1 && width*height == 128 &&
		visibleWidth == width && visibleHeight == height && activeRows > 0 && activeCols > 0 {
		// The 8x16 and 16x8 blocks of every type but IDTX (amd64: dav1d's
		// inv_txfm_add_8x16 / inv_txfm_add_16x8).
		if width == 8 {
			if transform.InverseAdd8x16(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
				return nil
			}
		} else if transform.InverseAdd16x8(dst, dstStride, dst, dstStride, dequant, scanHeight, cfg.Transform, activeRows, activeCols) {
			return nil
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
	} else if bytesPerSample == 2 && cfg.Transform == transform.TypeDCTDCT &&
		visibleWidth == width && visibleHeight == height && activeRows > 0 && activeCols > 0 {
		// The 10- and 12-bit 2:1 rectangular DCT_DCT blocks.
		if width == 16 && height == 32 {
			if transform.InverseAddDCT16x32HBD(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols, bitDepth) {
				return nil
			}
		} else if width == 32 && height == 16 {
			if transform.InverseAddDCT32x16HBD(dst, dstStride, dst, dstStride, dequant, scanHeight, activeRows, activeCols, bitDepth) {
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
