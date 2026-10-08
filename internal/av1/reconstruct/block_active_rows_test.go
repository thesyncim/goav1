package reconstruct

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestTrustedFullDequantDefaultScanBoundMatchesFullTransform(t *testing.T) {
	types := []transform.Type{
		transform.TypeDCTDCT,
		transform.TypeADSTDCT,
		transform.TypeDCTADST,
		transform.TypeADSTADST,
		transform.TypeHDCT,
		transform.TypeVDCT,
	}
	sizes := []transform.Size{
		{Width: 8, Height: 8},
		{Width: 16, Height: 8},
		{Width: 8, Height: 16},
		{Width: 16, Height: 16},
		{Width: 32, Height: 32},
		{Width: 64, Height: 32},
		{Width: 32, Height: 64},
		{Width: 64, Height: 64},
	}
	tested := 0
	for _, size := range sizes {
		scanSize, err := transform.ScanSize(size)
		if err != nil {
			t.Fatal(err)
		}
		scanHeight := int(scanSize.Height)
		area := int(scanSize.Width) * scanHeight
		eob := area/sparseDequantWorkFactor + 1
		for _, typ := range types {
			if !typ.Supported(size) {
				continue
			}
			class, err := typ.Class()
			if err != nil {
				t.Fatal(err)
			}
			scan, err := transform.DefaultScan(size, class)
			if err != nil {
				t.Fatalf("DefaultScan(%+v,%v): %v", size, typ, err)
			}
			bound, exact := transform.DefaultScanActiveRowsForScan(size, class, scan, eob)
			fullRows := min(int(size.Height), scanHeight)
			if !exact || bound <= 0 || bound >= fullRows {
				continue
			}
			for _, bitDepth := range []uint8{8, 10, 12} {
				for _, qmatrix := range []bool{false, true} {
					name := fmt.Sprintf("%dx%d/type%d/bd%d/qmatrix=%v", size.Width, size.Height, typ, bitDepth, qmatrix)
					t.Run(name, func(t *testing.T) {
						cfg := Block{
							Size:      size,
							Transform: typ,
							Quantizer: quantize.Quantizer{DC: 4, AC: 8},
							EOB:       int16(eob),
						}
						quantized := make([]int16, area)
						nonzeroCount := area/nonzeroDequantWorkFactor + 1
						if nonzeroCount >= eob {
							t.Fatalf("test setup needs EOB %d greater than NNZ threshold %d", eob, nonzeroCount)
						}
						nonzero := make([]int16, 0, nonzeroCount+1)
						for i := 0; i < nonzeroCount; i++ {
							pos := scan[i]
							value := int16(i%11 - 5)
							if value == 0 {
								value = 2
							}
							setScannedCoeff(quantized, scanHeight, int(pos), value)
							nonzero = append(nonzero, pos)
						}
						lastPos := scan[eob-1]
						setScannedCoeff(quantized, scanHeight, int(lastPos), -5)
						nonzero = append(nonzero, lastPos)
						if eob*sparseDequantWorkFactor <= area {
							t.Fatalf("test setup selected sparse dequant: eob=%d area=%d", eob, area)
						}
						if len(nonzero)*nonzeroDequantWorkFactor <= area {
							t.Fatalf("test setup selected non-zero dequant: nnz=%d area=%d", len(nonzero), area)
						}
						if qmatrix {
							cfg.InverseQMatrix = make([]uint16, area)
							for i := range cfg.InverseQMatrix {
								cfg.InverseQMatrix[i] = 1 << 5
							}
						}
						assertTrustedMatchesFullReconstruction(t, cfg, quantized, scanHeight, scan, nonzero, scanSize, bitDepth)
					})
					tested++
				}
			}
		}
	}
	if tested == 0 {
		t.Fatal("no full-dequant default-scan case had rows to skip")
	}
}

func TestTrustedFullDequantCopiedScanPreservesFullFallback(t *testing.T) {
	size := transform.Size{Width: 16, Height: 16}
	scanSize, err := transform.ScanSize(size)
	if err != nil {
		t.Fatal(err)
	}
	scanHeight := int(scanSize.Height)
	area := int(scanSize.Width) * scanHeight
	eob := area/sparseDequantWorkFactor + 1
	class := transform.Class2D
	defaultScan, err := transform.DefaultScan(size, class)
	if err != nil {
		t.Fatal(err)
	}
	defaultBound, exact := transform.DefaultScanActiveRowsForScan(size, class, defaultScan, eob)
	if !exact || defaultBound >= scanHeight {
		t.Fatalf("default scan did not provide a useful bound: exact=%v rows=%d height=%d", exact, defaultBound, scanHeight)
	}
	customScan := append([]int16(nil), defaultScan...)
	if _, exact := transform.DefaultScanActiveRowsForScan(size, class, customScan, eob); exact {
		t.Fatal("copied scan unexpectedly matched the immutable default scan")
	}
	quantized := make([]int16, area)
	setScannedCoeff(quantized, scanHeight, int(defaultScan[eob-1]), -5)
	var outside int
	foundOutside := false
	for i := eob; i < len(defaultScan); i++ {
		pos := int(defaultScan[i])
		if pos%scanHeight >= defaultBound {
			outside = pos
			foundOutside = true
			break
		}
	}
	if !foundOutside {
		t.Fatal("default scan has no post-EOB coefficient outside the EOB row bound")
	}
	setScannedCoeff(quantized, scanHeight, outside, 64)
	cfg := Block{
		Size:      size,
		Transform: transform.TypeDCTDCT,
		Quantizer: quantize.Quantizer{DC: 4, AC: 8},
		EOB:       int16(eob),
	}
	assertTrustedMatchesFullReconstruction(t, cfg, quantized, scanHeight, customScan, nil, scanSize, 8)
}

func TestTrustedFullDequantEOBZeroKeepsFullRowSentinel(t *testing.T) {
	size := transform.Size{Width: 16, Height: 16}
	scanSize, err := transform.ScanSize(size)
	if err != nil {
		t.Fatal(err)
	}
	scan, err := transform.DefaultScan(size, transform.Class2D)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Block{Size: size, Transform: transform.TypeDCTDCT, Quantizer: quantize.Quantizer{DC: 4, AC: 8}}
	assertTrustedMatchesFullReconstruction(t, cfg, make([]int16, int(scanSize.Width)*int(scanSize.Height)), int(scanSize.Height), scan, nil, scanSize, 8)
}

func setScannedCoeff(coeff []int16, scanHeight, position int, value int16) {
	coeff[(position/scanHeight)*scanHeight+position%scanHeight] = value
}

func assertTrustedMatchesFullReconstruction(t *testing.T, cfg Block, quantized []int16, quantizedStride int, scan []int16, nonzero []int16, scanSize transform.Size, bitDepth uint8) {
	t.Helper()
	width, height := int(cfg.Size.Width), int(cfg.Size.Height)
	bytesPerSample := 1
	if bitDepth > 8 {
		bytesPerSample = 2
	}
	txScale, err := quantize.TransformScale(width, height)
	if err != nil {
		t.Fatal(err)
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]byte, width*height*bytesPerSample)
	want := make([]byte, width*height*bytesPerSample)
	predictor := uint16(1 << (bitDepth - 1))
	for i := 0; i < width*height; i++ {
		setPixelSample(got, i, bytesPerSample, predictor)
		setPixelSample(want, i, bytesPerSample, predictor)
	}
	transformScratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	for i := range transformScratch {
		transformScratch[i] = int32(0x5a5a5a5a)
	}
	for i := range residualScratch {
		residualScratch[i] = -12345
	}
	if nonzero == nil {
		if err := ReconstructPlaneBlockVisibleTrustedAtWithGeometryAndScan(
			got, width*bytesPerSample, bytesPerSample, bitDepth, width, height, quantized, quantizedStride, scan, scanSize, txScale,
			transformScratch, residualScratch, cfg,
		); err != nil {
			t.Fatalf("trusted reconstruction: %v", err)
		}
	} else {
		if err := ReconstructPlaneBlockVisibleTrustedAtWithGeometryScanAndNonZero(
			got, width*bytesPerSample, bytesPerSample, bitDepth, width, height, quantized, quantizedStride, scan, nonzero, scanSize, txScale,
			transformScratch, residualScratch, cfg,
		); err != nil {
			t.Fatalf("trusted non-zero reconstruction: %v", err)
		}
	}
	plane := frame.Plane{Pix: want, Stride: width * bytesPerSample, Width: width, Height: height}
	oracleTransformScratch := make([]int32, int32Len)
	oracleResidualScratch := make([]int16, int16Len)
	for i := range oracleTransformScratch {
		oracleTransformScratch[i] = int32(0x3c3c3c3c)
	}
	for i := range oracleResidualScratch {
		oracleResidualScratch[i] = 23456
	}
	if err := ReconstructPlaneBlockVisible(
		plane, bytesPerSample, bitDepth, 0, 0, width, height, quantized, quantizedStride,
		oracleTransformScratch, oracleResidualScratch, cfg,
	); err != nil {
		t.Fatalf("full-transform oracle: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("trusted EOB-bounded reconstruction differs from full dequant/full transform")
	}
}

func setPixelSample(dst []byte, index, bytesPerSample int, value uint16) {
	if bytesPerSample == 1 {
		dst[index] = byte(value)
		return
	}
	offset := index * 2
	dst[offset] = byte(value)
	dst[offset+1] = byte(value >> 8)
}

var benchmarkTrustedFullDequantSink byte

func BenchmarkTrustedFullDequantActiveRows(b *testing.B) {
	for _, size := range []transform.Size{
		{Width: 16, Height: 16},
		{Width: 32, Height: 32},
	} {
		scanSize, err := transform.ScanSize(size)
		if err != nil {
			b.Fatal(err)
		}
		width, height := int(size.Width), int(size.Height)
		scanWidth, scanHeight := int(scanSize.Width), int(scanSize.Height)
		area := scanWidth * scanHeight
		eob := area/sparseDequantWorkFactor + 1
		scan, err := transform.DefaultScan(size, transform.Class2D)
		if err != nil {
			b.Fatal(err)
		}
		rows, ok := transform.DefaultScanActiveRowsForScan(size, transform.Class2D, scan, eob)
		if !ok || rows <= 0 || rows >= min(height, scanHeight) {
			b.Fatalf("%dx%d has no useful default-scan row bound: rows=%d height=%d", width, height, rows, min(height, scanHeight))
		}
		quantized := make([]int16, area)
		for i := 0; i < eob; i++ {
			value := int16(i%13 - 6)
			setScannedCoeff(quantized, scanHeight, int(scan[i]), value)
		}
		setScannedCoeff(quantized, scanHeight, int(scan[eob-1]), 4)
		cfg := Block{
			Size:      size,
			Transform: transform.TypeDCTDCT,
			Quantizer: quantize.Quantizer{DC: 4, AC: 8},
			EOB:       int16(eob),
		}
		txScale, err := quantize.TransformScale(width, height)
		if err != nil {
			b.Fatal(err)
		}
		int32Len, int16Len, err := ScratchLen(cfg)
		if err != nil {
			b.Fatal(err)
		}
		predictor := make([]byte, width*height)
		for i := range predictor {
			predictor[i] = 128
		}
		benchmarks := []struct {
			name string
			scan []int16
		}{
			{name: fmt.Sprintf("bound_%d_rows", rows), scan: scan},
			{name: "full_rows_copied_scan", scan: append([]int16(nil), scan...)},
		}
		for _, benchmark := range benchmarks {
			b.Run(fmt.Sprintf("%dx%d/%s", width, height, benchmark.name), func(b *testing.B) {
				dst := make([]byte, len(predictor))
				int32Scratch := make([]int32, int32Len)
				residualScratch := make([]int16, int16Len)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(dst, predictor)
					if err := reconstructPlaneBlockTrustedAtWithGeometry(
						dst, width, 1, 8, width, height, quantized, scanHeight, benchmark.scan, nil,
						scanSize, txScale, int32Scratch, residualScratch, cfg,
					); err != nil {
						b.Fatal(err)
					}
				}
				benchmarkTrustedFullDequantSink = dst[0]
			})
		}
	}
}
