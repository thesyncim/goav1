package goav1_test

import (
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

func TestPublicPlaneQuantizer(t *testing.T) {
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

}

func TestPublicReconstructBlockMaxScratchLen(t *testing.T) {
	int32Len, int16Len, err := av1.ReconstructBlockMaxScratchLen(false)
	if err != nil {
		t.Fatal(err)
	}
	want32, want16, err := av1.ReconstructBlockScratchLen(av1.ReconstructBlock{
		Size:      av1.TransformSize{Width: 64, Height: 64},
		Transform: av1.TransformTypeDCTDCT,
	})
	if err != nil {
		t.Fatal(err)
	}
	if int32Len != want32 || int16Len != want16 {
		t.Fatalf("max scratch=%d/%d want %d/%d", int32Len, int16Len, want32, want16)
	}

	lossless32, lossless16, err := av1.ReconstructBlockMaxScratchLen(true)
	if err != nil {
		t.Fatal(err)
	}
	wantLossless32, wantLossless16, err := av1.ReconstructBlockScratchLen(av1.ReconstructBlock{
		Size:      av1.TransformSize{Width: 4, Height: 4},
		Transform: av1.TransformTypeDCTDCT,
		Lossless:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if lossless32 != wantLossless32 || lossless16 != wantLossless16 {
		t.Fatalf("lossless max scratch=%d/%d want %d/%d", lossless32, lossless16, wantLossless32, wantLossless16)
	}
}

func fillPublicReconstructPlane(plane av1.FramePlane, bytesPerSample int, sample uint16) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, bytesPerSample, x, y, sample)
		}
	}
}
