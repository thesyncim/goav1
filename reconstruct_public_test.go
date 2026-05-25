package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicTransformScanAndScratchHelpers(t *testing.T) {
	size, err := av1.TransformSizeFromDimensions(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !size.Valid() {
		t.Fatalf("size %+v is not valid", size)
	}
	class, err := av1.TransformTypeDCTDCT.Class()
	if err != nil {
		t.Fatal(err)
	}
	if class != av1.TransformClass2D {
		t.Fatalf("class=%v want %v", class, av1.TransformClass2D)
	}
	mode, err := av1.TransformDefaultScanMode(size, class)
	if err != nil {
		t.Fatal(err)
	}
	if mode != av1.TransformScanModeZigZag {
		t.Fatalf("mode=%v want %v", mode, av1.TransformScanModeZigZag)
	}

	scanSize, err := av1.TransformScanSize(av1.TransformSize{Width: 64, Height: 16})
	if err != nil {
		t.Fatal(err)
	}
	if scanSize != (av1.TransformSize{Width: 32, Height: 16}) {
		t.Fatalf("scanSize=%+v want 32x16", scanSize)
	}

	scan := make([]int16, size.Width*size.Height)
	inverse := make([]int16, len(scan))
	if err := av1.FillTransformDefaultScan(scan, inverse, size, class); err != nil {
		t.Fatal(err)
	}
	for i, coeff := range scan {
		if coeff < 0 || int(coeff) >= len(scan) {
			t.Fatalf("scan[%d]=%d outside range", i, coeff)
		}
		if inverse[coeff] != int16(i) {
			t.Fatalf("inverse[%d]=%d want %d", coeff, inverse[coeff], i)
		}
	}
	scratch, err := av1.TransformScratchLenForType(av1.TransformTypeDCTDCT, size)
	if err != nil {
		t.Fatal(err)
	}
	if scratch != 16 {
		t.Fatalf("scratch=%d want 16", scratch)
	}
}

func TestPublicPlaneQuantizerAndDequantizeBlock(t *testing.T) {
	params := av1.QuantizationParams{
		BaseQIdx: 96,
		YDCDelta: -2,
		UDCDelta: 3,
		UACDelta: -1,
		VDCDelta: 4,
		VACDelta: 5,
	}
	y, err := av1.PlaneQuantizer(params, params.BaseQIdx, 8, av1.QuantizerPlaneY)
	if err != nil {
		t.Fatal(err)
	}
	u, err := av1.PlaneQuantizer(params, params.BaseQIdx, 8, av1.QuantizerPlaneU)
	if err != nil {
		t.Fatal(err)
	}
	v, err := av1.PlaneQuantizer(params, params.BaseQIdx, 8, av1.QuantizerPlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if y != (av1.Quantizer{DC: 85, AC: 104}) ||
		u != (av1.Quantizer{DC: 92, AC: 102}) ||
		v != (av1.Quantizer{DC: 93, AC: 114}) {
		t.Fatalf("quantizers y=%+v u=%+v v=%+v", y, u, v)
	}

	scale, err := av1.TransformScale(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	if scale != 2 {
		t.Fatalf("scale=%d want 2", scale)
	}
	if got := av1.ClampQIndex(2, -9); got != 0 {
		t.Fatalf("clamped low=%d want 0", got)
	}
	if got := av1.ClampQIndex(av1.QuantizerMaxQIndex, 9); got != av1.QuantizerMaxQIndex {
		t.Fatalf("clamped high=%d want %d", got, av1.QuantizerMaxQIndex)
	}

	coeff := []int16{
		2, -5, 99,
		-3, 6, 99,
		4, -7, 99,
	}
	dst := []int32{
		99, 99, 99,
		99, 99, 99,
		99, 99, 99,
	}
	if err := av1.DequantizeBlock(dst, 3, coeff, 3, 3, 2, av1.Quantizer{DC: 4, AC: 8}); err != nil {
		t.Fatal(err)
	}
	want := []int32{
		8, -40, 99,
		-24, 48, 99,
		32, -56, 99,
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestPublicReconstructPlaneBlockDCT4x4(t *testing.T) {
	plane := publicReconstructPlane(6, 5, 1, 8)
	fillPublicReconstructPlane(plane, 1, 100)
	quantized := make([]int16, 4*4)
	quantized[0] = 4
	cfg := av1.ReconstructBlock{
		Size:      av1.TransformSize{Width: 4, Height: 4},
		Transform: av1.TransformTypeDCTDCT,
		Quantizer: av1.Quantizer{DC: 4, AC: 8},
	}

	int32Len, int16Len, err := av1.ReconstructBlockScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if int32Len != 32 || int16Len != 16 {
		t.Fatalf("scratch=%d/%d want 32/16", int32Len, int16Len)
	}
	if err := av1.ReconstructPlaneBlock(
		plane,
		1,
		8,
		1,
		1,
		quantized,
		4,
		make([]int32, int32Len),
		make([]int16, int16Len),
		cfg,
	); err != nil {
		t.Fatal(err)
	}

	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			want := uint16(100)
			if x >= 1 && x < 5 && y >= 1 && y < 5 {
				want = 101
			}
			if got := getPublicFrameSample(plane, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPublicReconstructPlaneBlockRejectsInvalid(t *testing.T) {
	plane := publicReconstructPlane(4, 4, 1, 4)
	quantized := make([]int16, 16)
	cfg := av1.ReconstructBlock{
		Size:      av1.TransformSize{Width: 4, Height: 4},
		Transform: av1.TransformTypeDCTDCT,
		Quantizer: av1.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := av1.ReconstructBlockScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := av1.ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 4, make([]int32, int32Len-1), make([]int16, int16Len), cfg); !errors.Is(err, av1.ErrReconstructInvalidBlock) {
		t.Fatalf("short scratch err=%v want %v", err, av1.ErrReconstructInvalidBlock)
	}
	invalid := cfg
	invalid.Size = av1.TransformSize{Width: 64, Height: 8}
	if _, _, err := av1.ReconstructBlockScratchLen(invalid); !errors.Is(err, av1.ErrReconstructInvalidBlock) {
		t.Fatalf("invalid size err=%v want %v", err, av1.ErrReconstructInvalidBlock)
	}
	if _, err := av1.DCQuant(0, 0, 9); !errors.Is(err, av1.ErrQuantizeInvalidQuantizer) {
		t.Fatalf("invalid quant err=%v want %v", err, av1.ErrQuantizeInvalidQuantizer)
	}
	if _, err := av1.TransformSizeFromDimensions(12, 12); !errors.Is(err, av1.ErrTransformInvalidTransform) {
		t.Fatalf("invalid transform err=%v want %v", err, av1.ErrTransformInvalidTransform)
	}
}

func TestPublicReconstructPlaneBlockAllocs(t *testing.T) {
	plane := publicReconstructPlane(8, 8, 1, 8)
	quantized := make([]int16, 8*8)
	quantized[0] = 8
	cfg := av1.ReconstructBlock{
		Size:      av1.TransformSize{Width: 8, Height: 8},
		Transform: av1.TransformTypeDCTDCT,
		Quantizer: av1.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := av1.ReconstructBlockScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	int32Scratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	allocs := testing.AllocsPerRun(1000, func() {
		fillPublicReconstructPlane(plane, 1, 100)
		if err := av1.ReconstructPlaneBlock(plane, 1, 8, 0, 0, quantized, 8, int32Scratch, residualScratch, cfg); err != nil {
			t.Fatalf("ReconstructPlaneBlock err=%v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicReconstructPlane(width int, height int, bytesPerSample int, strideSamples int) av1.FramePlane {
	return av1.FramePlane{
		Pix:    make([]byte, strideSamples*height*bytesPerSample),
		Stride: strideSamples * bytesPerSample,
		Width:  width,
		Height: height,
	}
}

func fillPublicReconstructPlane(plane av1.FramePlane, bytesPerSample int, sample uint16) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, bytesPerSample, x, y, sample)
		}
	}
}
