package reconstruct

import (
	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// Block describes the residual transform block to reconstruct.
type Block struct {
	Size      transform.Size
	Transform transform.Type
	Quantizer quantize.Quantizer
	Lossless  bool
	EOB       int
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
	int32Len, residualLen, err := ScratchLen(cfg)
	if err != nil ||
		len(int32Scratch) < int32Len ||
		len(residualScratch) < residualLen {
		return ErrInvalidBlock
	}

	width := cfg.Size.Width
	height := cfg.Size.Height
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

	if err := quantize.DequantizeBlock(dequant, coeffSize.Height, quantized, quantizedStride, coeffSize.Width, coeffSize.Height, cfg.Quantizer); err != nil {
		return ErrInvalidBlock
	}
	if cfg.Lossless {
		if err := transform.InverseWHT4x4Block(residual, width, dequant, coeffSize.Height, cfg.EOB); err != nil {
			return ErrInvalidBlock
		}
	} else if err := transform.InverseBlock(residual, width, dequant, coeffSize.Height, transformScratch, cfg.Size, cfg.Transform); err != nil {
		return ErrInvalidBlock
	}
	if err := dsp.AddResidualPlaneBlock(dst, bytesPerSample, bitDepth, x, y, width, height, residual, width); err != nil {
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
