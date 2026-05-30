// Regression test for the full reconstruct pipeline applied to the 32x32
// D157 first-block fixture of the libaom 8-bit 34x34 vector. The fixture
// coefficients are bit-exact entropy decode, the predictor is the bit-exact
// 32x32 D157 corner-frame snapshot, and the expected output is what the
// goav1 quantize + transform stack produces when invoked directly. This
// pins the ReconstructPlaneBlock glue between dequant, inverse-transform,
// and add-residual: any future regression that perturbs the dequant call,
// the scratch slicing, the txScale derivation, or the visible-rectangle
// trimming for this representative block will regress here.

package reconstruct

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestReconstructPlaneBlockD157Block0FullPipeline(t *testing.T) {
	const W, H = 32, 32

	// Build expected residual via direct dequant + InverseBlockBitDepth.
	q, err := quantize.PlaneQuantizer(parser.QuantizationParams{BaseQIdx: 109}, 109, 8, quantize.PlaneY)
	if err != nil {
		t.Fatal(err)
	}
	if q.DC != 107 || q.AC != 130 {
		t.Fatalf("unexpected quantizer DC=%d AC=%d (want 107/130)", q.DC, q.AC)
	}
	txScale, err := quantize.TransformScale(W, H)
	if err != nil {
		t.Fatal(err)
	}
	dq := make([]int32, W*H)
	if err := quantize.DequantizeBlockScaledBitDepth(dq, H, d157Block0Coeffs[:], H, W, H, q, txScale, 8); err != nil {
		t.Fatal(err)
	}
	wantRes := make([]int16, W*H)
	scratch := make([]int32, W*H)
	if err := transform.InverseBlockBitDepth(wantRes, W, dq, H, scratch,
		transform.Size{Width: W, Height: H}, transform.TypeDCTDCT, 8); err != nil {
		t.Fatal(err)
	}

	// Build the predictor reference plane and add+clip via dsp directly.
	wantPlane, _ := testPlane(W, H, 1, W)
	for r := range H {
		for c := range W {
			wantPlane.Pix[r*wantPlane.Stride+c] = d157Block0Predictor[r*W+c]
		}
	}
	if err := dsp.AddResidualPlaneBlock(wantPlane, 1, 8, 0, 0, W, H, wantRes, W); err != nil {
		t.Fatal(err)
	}

	// Run ReconstructPlaneBlock on the same predictor and assert byte-exact
	// equality with the hand-composed reference.
	gotPlane, _ := testPlane(W, H, 1, W)
	for r := range H {
		for c := range W {
			gotPlane.Pix[r*gotPlane.Stride+c] = d157Block0Predictor[r*W+c]
		}
	}
	cfg := Block{
		Size:      transform.Size{Width: W, Height: H},
		Transform: transform.TypeDCTDCT,
		Quantizer: q,
		EOB:       845,
	}
	int32Len, int16Len, err := ScratchLen(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ReconstructPlaneBlock(gotPlane, 1, 8, 0, 0,
		d157Block0Coeffs[:], H,
		make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}

	for r := range H {
		for c := range W {
			got := gotPlane.Pix[r*gotPlane.Stride+c]
			want := wantPlane.Pix[r*wantPlane.Stride+c]
			if got != want {
				t.Fatalf("D157 blk0 reconstruct mismatch at (%d,%d): got=%d want=%d", c, r, got, want)
			}
		}
	}
}
