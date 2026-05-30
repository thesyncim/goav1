// Regression test for the 32x32 D157 first-block dequant + inverse-transform
// pipeline of the libaom 8-bit 34x34 test vector.
//
// The fixture is the exact set of entropy-decoded coefficients for the 32x32
// luma TXB at (mi=0,0) of frame 0. The coefficients were captured under the
// goav1_coeff_trace build tag and cross-validated bit-exact against libaom's
// read_coeffs_txb. For an intra 32x32 block libaom uses ExtTXSetDCTOnly which
// forces the transform type to DCT_DCT regardless of intra mode; we mirror
// that here. The dequantizer (DC=107, AC=130) corresponds to base_qidx=109
// with no segmentation or qmatrix.
//
// This test pins the dequant + InverseBlockBitDepth output to a libaom-port
// reference (see libaomInverseResidual) so any future change to either path
// that perturbs reconstruction for the D157 corner regresses here rather than
// only at end-to-end MD5 time on the opt-in 34x34 extended vector.

package transform

import "testing"

func TestD157Block0InverseTransformMatchesLibaomReference(t *testing.T) {
	const W, H = 32, 32
	const DC, AC = 107, 130
	const txScale = 1

	dqMin := -int32(1) << 15
	dqMax := int32(1)<<15 - 1
	dq := make([]int32, W*H)
	for col := range W {
		for row := range H {
			scale := int32(AC)
			if col == 0 && row == 0 {
				scale = DC
			}
			lvl := int32(d157Block0Coeffs[col*H+row])
			negative := lvl < 0
			if negative {
				lvl = -lvl
			}
			product := (lvl * scale) & 0xffffff
			out := product >> txScale
			if negative {
				out = -out
			}
			if out < dqMin {
				out = dqMin
			} else if out > dqMax {
				out = dqMax
			}
			dq[col*H+row] = out
		}
	}

	// goav1 inverse-transform path.
	got := make([]int16, W*H)
	scratch := make([]int32, W*H)
	if err := InverseBlockBitDepth(got, W, dq, H, scratch,
		Size{Width: W, Height: H}, TypeDCTDCT, 8); err != nil {
		t.Fatal(err)
	}

	// libaom-port reference (faithful inv_txfm2d_add_c reproduction).
	want := libaomInverseResidual(dq, H, Size{Width: W, Height: H}, TypeDCTDCT, 8)

	for i := range W * H {
		if int32(got[i]) != want[i] {
			t.Fatalf("D157 blk0 residual[%d]: got=%d want=%d", i, got[i], want[i])
		}
	}
}
