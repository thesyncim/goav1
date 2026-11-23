package reconstruct

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// testBlock is a Block together with the quantizer of the tests' quantized
// coefficients.
type testBlock struct {
	Size      transform.Size
	Transform transform.Type
	EOB       int16
	Lossless  bool
	Quantizer quantize.Quantizer
}

func (b testBlock) block() Block {
	return Block{Size: b.Size, Transform: b.Transform, EOB: b.EOB, Lossless: b.Lossless}
}

// reconstructQuantized dequantizes quantized coefficients the way decode_coefs
// does and reconstructs them through ReconstructPlaneBlockVisibleTrustedAtFromDequantized
// into the block at (x, y) of dst.
func reconstructQuantized(dst frame.Plane, bytesPerSample int, bitDepth uint8, x, y int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg testBlock) error {
	return reconstructQuantizedVisible(dst, bytesPerSample, bitDepth, x, y, int(cfg.Size.Width), int(cfg.Size.Height), quantized, quantizedStride, int32Scratch, residualScratch, cfg)
}

func reconstructQuantizedVisible(dst frame.Plane, bytesPerSample int, bitDepth uint8, x, y, visibleWidth, visibleHeight int, quantized []int16, quantizedStride int, int32Scratch []int32, residualScratch []int16, cfg testBlock) error {
	scanSize, err := transform.ScanSize(cfg.Size)
	if err != nil {
		return err
	}
	txScale, err := quantize.TransformScale(int(cfg.Size.Width), int(cfg.Size.Height))
	if err != nil {
		return err
	}
	scanW, scanH := int(scanSize.Width), int(scanSize.Height)
	cf := make([]int32, scanW*scanH)
	if err := quantize.DequantizeBlockScaledBitDepth(cf, scanH, quantized, quantizedStride, scanW, scanH, cfg.Quantizer, txScale, bitDepth); err != nil {
		return err
	}
	activeRows, activeCols := 0, 0
	for i, v := range cf {
		if v != 0 {
			activeRows = max(activeRows, i%scanH+1)
			activeCols = max(activeCols, i/scanH+1)
		}
	}
	if activeRows == 0 {
		return nil
	}
	b := cfg.block()
	if b.Lossless {
		b.Transform = transform.TypeWHTWHT
	}
	dcOnly := b.Transform == transform.TypeDCTDCT && b.EOB == 1
	off := y*dst.Stride + x*bytesPerSample
	n := (visibleHeight-1)*dst.Stride + visibleWidth*bytesPerSample
	return ReconstructPlaneBlockVisibleTrustedAtFromDequantized(dst.Pix[off:off+n:off+n], dst.Stride, bytesPerSample, bitDepth, visibleWidth, visibleHeight, scanSize, activeRows, activeCols, dcOnly, cf, int32Scratch, residualScratch, b)
}
