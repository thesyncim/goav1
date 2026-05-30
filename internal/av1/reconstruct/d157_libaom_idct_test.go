package reconstruct

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// d157Block0LibaomPreFilterRow0 is the byte-exact pre-postfilter row 0 of
// libaom's 32x32 reconstruct for the D157 first block of frame 0 of the
// libaom 8-bit 34x34 test vector. The 32 values were captured by passing the
// goav1-computed dequantized coefficient block through libaom's
// av1_inv_txfm2d_add_32x32_c with the all-zero predictor (so each value IS
// the residual). Cross-checked: predicting onto the goav1 corner-frame D157
// predictor and adding the residual + saturating to [0,255] reproduces the
// pre-postfilter byte. Postfilter (deblock/CDEF/restoration) is applied
// later in the pipeline and is not validated by this test.
var d157Block0LibaomPreFilterRow0 = [32]int16{
	44, 56, 74, 85, 89, 73, 68, 67,
	56, 71, 62, 41, 59, 68, 73, 77,
	71, 97, 108, 77, 72, 80, 83, 76,
	55, 62, 62, 44, 52, 68, 62, 43,
}

// TestReconstructD157Block0Row0MatchesLibaomIDCT pins the goav1
// reconstruct-path residual for row 0 of the D157 first block against the
// byte-exact reference produced by libaom v3.13.1's
// av1_inv_txfm2d_add_32x32_c when invoked with the same dequantized
// coefficients and a zero predictor. This guards the entire dequant +
// inverse-transform path against any rounding/clamp regression that would
// shift the row-0 residual by even ±1, which would silently divert the
// end-to-end 34x34 MD5 on the opt-in extended cohort.
func TestReconstructD157Block0Row0MatchesLibaomIDCT(t *testing.T) {
	const W, H = 32, 32

	q, err := quantize.PlaneQuantizer(parser.QuantizationParams{BaseQIdx: 109}, 109, 8, quantize.PlaneY)
	if err != nil {
		t.Fatal(err)
	}
	plane, _ := testPlane(W, H, 1, W)
	// Zero predictor so plane output equals residual (saturated to [0,255]).
	for r := range H {
		for c := range W {
			plane.Pix[r*plane.Stride+c] = 0
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
	if err := ReconstructPlaneBlock(plane, 1, 8, 0, 0,
		d157Block0Coeffs[:], H,
		make([]int32, int32Len), make([]int16, int16Len), cfg); err != nil {
		t.Fatal(err)
	}
	for c := range W {
		want := int16(d157Block0LibaomPreFilterRow0[c])
		wantSat := max(want, 0)
		if wantSat > 255 {
			wantSat = 255
		}
		got := int16(plane.Pix[0*plane.Stride+c])
		if got != wantSat {
			t.Fatalf("D157 blk0 row 0 col %d: goav1=%d want=%d (libaom v3.13.1 av1_inv_txfm2d_add_32x32_c residual)", c, got, wantSat)
		}
	}
}
