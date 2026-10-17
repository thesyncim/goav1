package reconstruct

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestReconstructPlaneBlockVisibleTrustedAtWithDequantizedMatchesQuantizedPath(t *testing.T) {
	tests := []struct {
		name string
		size transform.Size
		typ  transform.Type
	}{
		{name: "8x8-dct", size: transform.Size{Width: 8, Height: 8}, typ: transform.TypeDCTDCT},
		{name: "16x16-adst", size: transform.Size{Width: 16, Height: 16}, typ: transform.TypeADSTADST},
		{name: "32x32-idtx", size: transform.Size{Width: 32, Height: 32}, typ: transform.TypeIDTX},
		{name: "64x64-adjusted-dct", size: transform.Size{Width: 64, Height: 64}, typ: transform.TypeDCTDCT},
		{name: "64x16-adjusted-dct", size: transform.Size{Width: 64, Height: 16}, typ: transform.TypeDCTDCT},
	}
	for _, test := range tests {
		for _, bitDepth := range []uint8{8, 10, 12} {
			for _, eob := range []int{1, 9} {
				if !test.typ.Supported(test.size) {
					continue
				}
				for _, pattern := range []string{"small-alternating", "large-signed"} {
					t.Run(fmt.Sprintf("%s/bd%d/eob%d/%s", test.name, bitDepth, eob, pattern), func(t *testing.T) {
						cfg := Block{
							Size:      test.size,
							Transform: test.typ,
							Quantizer: quantize.Quantizer{DC: 211, AC: 337},
							EOB:       int16(eob),
						}
						scanSize, err := transform.ScanSize(cfg.Size)
						if err != nil {
							t.Fatal(err)
						}
						scan, err := transform.DefaultScan(cfg.Size, transform.Class2D)
						if err != nil {
							t.Fatal(err)
						}
						scanWidth, scanHeight := int(scanSize.Width), int(scanSize.Height)
						scanLen := scanWidth * scanHeight
						if eob > len(scan) {
							t.Skip("scan is shorter than the requested EOB")
						}
						coeff := make([]int16, scanLen)
						activeRows := 0
						for i := 0; i < eob; i++ {
							pos := int(scan[i])
							var level int16
							if pattern == "small-alternating" {
								level = [...]int16{1, -1, 2, -2}[i%4]
							} else {
								level = int16((i*37)%601 - 300)
								if level == 0 {
									level = 1
								}
							}
							coeff[pos] = level
							row := pos % scanHeight
							if row+1 > activeRows {
								activeRows = row + 1
							}
						}

						int32Len, residualLen, err := ScratchLen(cfg)
						if err != nil {
							t.Fatal(err)
						}
						oldScratch := make([]int32, int32Len)
						newScratch := make([]int32, int32Len)
						blockLen := int(cfg.Size.Width) * int(cfg.Size.Height)
						for i := scanLen; i < blockLen; i++ {
							newScratch[i] = int32(0x5a5a0000 | i)
						}
						residualOld := make([]int16, residualLen)
						residualNew := make([]int16, residualLen)
						txScale, err := quantize.TransformScale(int(cfg.Size.Width), int(cfg.Size.Height))
						if err != nil {
							t.Fatal(err)
						}
						quantize.DequantizeBlockScaledBitDepthFullTrusted(newScratch[:scanLen], scanHeight, coeff, scanHeight, scanWidth, scanHeight, cfg.Quantizer, txScale, bitDepth)

						bytesPerSample := 1
						if bitDepth > 8 {
							bytesPerSample = 2
						}
						frameWidth, frameHeight := int(cfg.Size.Width)+16, int(cfg.Size.Height)+16
						stride := frameWidth * bytesPerSample
						oldPlane := frame.Plane{Pix: make([]byte, stride*frameHeight), Stride: stride, Width: frameWidth, Height: frameHeight}
						newPlane := frame.Plane{Pix: make([]byte, stride*frameHeight), Stride: stride, Width: frameWidth, Height: frameHeight}
						fillPlane(oldPlane, bytesPerSample, 1<<(bitDepth-1))
						fillPlane(newPlane, bytesPerSample, 1<<(bitDepth-1))
						visibleWidth := int(cfg.Size.Width)
						visibleHeight := int(cfg.Size.Height)
						if visibleWidth > 8 {
							visibleWidth -= 2
						}
						if visibleHeight > 8 {
							visibleHeight--
						}
						x, y := 3, 4
						offset := y*stride + x*bytesPerSample
						dstLen := (visibleHeight-1)*stride + visibleWidth*bytesPerSample
						oldDst := oldPlane.Pix[offset : offset+dstLen : offset+dstLen]
						newDst := newPlane.Pix[offset : offset+dstLen : offset+dstLen]
						if err := ReconstructPlaneBlockVisibleTrustedAtWithGeometryAndScan(oldDst, stride, bytesPerSample, bitDepth, visibleWidth, visibleHeight,
							coeff, scanHeight, scan, scanSize, txScale, oldScratch, residualOld, cfg); err != nil {
							t.Fatal(err)
						}
						dcOnly := cfg.Transform == transform.TypeDCTDCT && eob == 1
						if err := ReconstructPlaneBlockVisibleTrustedAtWithDequantized(newDst, stride, bytesPerSample, bitDepth, visibleWidth, visibleHeight,
							scanSize, activeRows, dcOnly, newScratch, residualNew, cfg); err != nil {
							t.Fatal(err)
						}
						if !bytes.Equal(newPlane.Pix, oldPlane.Pix) {
							t.Fatalf("dequantized reconstruction differs: size=%+v transform=%v bitDepth=%d eob=%d activeRows=%d pattern=%s", cfg.Size, cfg.Transform, bitDepth, eob, activeRows, pattern)
						}
						for i := scanLen; i < blockLen; i++ {
							if got, want := newScratch[i], int32(0x5a5a0000|i); got != want {
								t.Fatalf("coefficient-to-transform scratch gap overwritten at %d: got=%08x want=%08x size=%+v", i, got, want, cfg.Size)
							}
						}
					})
				}
			}
		}
	}
}
