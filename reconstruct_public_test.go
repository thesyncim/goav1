package goav1

import (
	"testing"

	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalquantize "github.com/thesyncim/goav1/internal/av1/quantize"
	internalreconstruct "github.com/thesyncim/goav1/internal/av1/reconstruct"
	internaltransform "github.com/thesyncim/goav1/internal/av1/transform"
)

func TestPublicTransformScanAndScratchHelpers(t *testing.T) {
	size, err := internaltransform.SizeFromDimensions(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if !size.Valid() {
		t.Fatalf("size %+v is not valid", size)
	}
	class, err := internaltransform.TypeDCTDCT.Class()
	if err != nil {
		t.Fatal(err)
	}
	if class != internaltransform.Class2D {
		t.Fatalf("class=%v want %v", class, internaltransform.Class2D)
	}
	mode, err := internaltransform.DefaultScanMode(size, class)
	if err != nil {
		t.Fatal(err)
	}
	if mode != internaltransform.ScanModeZigZag {
		t.Fatalf("mode=%v want %v", mode, internaltransform.ScanModeZigZag)
	}

	scanSize, err := internaltransform.ScanSize(internaltransform.Size{Width: 64, Height: 16})
	if err != nil {
		t.Fatal(err)
	}
	if scanSize != (internaltransform.Size{Width: 32, Height: 16}) {
		t.Fatalf("scanSize=%+v want 32x16", scanSize)
	}

	scan := make([]int16, size.Width*size.Height)
	inverse := make([]int16, len(scan))
	if err := internaltransform.FillDefaultScan(scan, inverse, size, class); err != nil {
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
	scratch, err := internaltransform.ScratchLenForType(internaltransform.TypeDCTDCT, size)
	if err != nil {
		t.Fatal(err)
	}
	if scratch != 16 {
		t.Fatalf("scratch=%d want 16", scratch)
	}
}

func TestPublicPlaneQuantizer(t *testing.T) {
	params := internalparser.QuantizationParams{
		BaseQIdx: 96,
		YDCDelta: -2,
		UDCDelta: 3,
		UACDelta: -1,
		VDCDelta: 4,
		VACDelta: 5,
	}
	y, err := internalquantize.PlaneQuantizer(params, params.BaseQIdx, 8, internalquantize.PlaneY)
	if err != nil {
		t.Fatal(err)
	}
	u, err := internalquantize.PlaneQuantizer(params, params.BaseQIdx, 8, internalquantize.PlaneU)
	if err != nil {
		t.Fatal(err)
	}
	v, err := internalquantize.PlaneQuantizer(params, params.BaseQIdx, 8, internalquantize.PlaneV)
	if err != nil {
		t.Fatal(err)
	}
	if y != (internalquantize.Quantizer{DC: 85, AC: 104}) ||
		u != (internalquantize.Quantizer{DC: 92, AC: 102}) ||
		v != (internalquantize.Quantizer{DC: 93, AC: 114}) {
		t.Fatalf("quantizers y=%+v u=%+v v=%+v", y, u, v)
	}

	scale, err := internalquantize.TransformScale(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	if scale != 2 {
		t.Fatalf("scale=%d want 2", scale)
	}
	if got := internalquantize.ClampQIndex(2, -9); got != 0 {
		t.Fatalf("clamped low=%d want 0", got)
	}
	if got := internalquantize.ClampQIndex(internalquantize.MaxQIndex, 9); got != internalquantize.MaxQIndex {
		t.Fatalf("clamped high=%d want %d", got, internalquantize.MaxQIndex)
	}

}

func TestPublicReconstructBlockMaxScratchLen(t *testing.T) {
	int32Len, int16Len, err := reconstructBlockMaxScratchLen(false)
	if err != nil {
		t.Fatal(err)
	}
	want32, want16, err := internalreconstruct.ScratchLen(internalreconstruct.Block{
		Size:      internaltransform.Size{Width: 64, Height: 64},
		Transform: internaltransform.TypeDCTDCT,
	})
	if err != nil {
		t.Fatal(err)
	}
	if int32Len != want32 || int16Len != want16 {
		t.Fatalf("max scratch=%d/%d want %d/%d", int32Len, int16Len, want32, want16)
	}

	lossless32, lossless16, err := reconstructBlockMaxScratchLen(true)
	if err != nil {
		t.Fatal(err)
	}
	wantLossless32, wantLossless16, err := internalreconstruct.ScratchLen(internalreconstruct.Block{
		Size:      internaltransform.Size{Width: 4, Height: 4},
		Transform: internaltransform.TypeDCTDCT,
		Lossless:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if lossless32 != wantLossless32 || lossless16 != wantLossless16 {
		t.Fatalf("lossless max scratch=%d/%d want %d/%d", lossless32, lossless16, wantLossless32, wantLossless16)
	}
}

func fillPublicReconstructPlane(plane FramePlane, bytesPerSample int, sample uint16) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setPublicFrameSample(plane, bytesPerSample, x, y, sample)
		}
	}
}
