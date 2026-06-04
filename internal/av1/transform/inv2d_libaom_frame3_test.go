package transform

import (
	"testing"
)

// TestInverseDCT32x32MatchesLibaomFrame3Block reproduces the 32x32 DCT_DCT
// transform from libaom all-intra vector frame 3 (intra DC_PRED 64x64 block
// at MICol=32, MIRow=0). The non-zero dequantized coefficients listed below
// are dumped from the libaom reference decoder for this exact block. We
// pin the expected residual row 0 to libaom's IDCT output so any future
// regression of inverse-DCT scale, shift, or stage clamps is caught against
// a known real-stream block rather than synthetic coefficients.
func TestInverseDCT32x32MatchesLibaomFrame3Block(t *testing.T) {
	sz := Size{Width: 32, Height: 32}
	stride := 32
	coeff := make([]int32, 32*32)

	// libaom row-major dq[r=R,c=C]=v converted to goav1's column-major
	// storage (coeff[c*stride+r] = v).
	type kv struct {
		r, c int
		v    int32
	}
	nonZero := []kv{
		// libaom dump:
		//   dq[r=0,c=0]=-918  dq[r=0,c=1]=-114
		//   dq[r=1,c=0]= 522  dq[r=1,c=1]= -28
		//   dq[r=2,c=0]=  28  dq[r=2,c=6]=   9
		//   dq[r=3,c=0]=  47  dq[r=3,c=1]=  -9  dq[r=3,c=4]= 9
		//   dq[r=5,c=0]=   9  dq[r=5,c=1]=   9
		//   dq[r=9,c=2]=  -9
		//   dq[r=11,c=0]= 9
		{0, 0, -918}, {1, 0, -114},
		{0, 1, 522}, {1, 1, -28},
		{0, 2, 28}, {6, 2, 9},
		{0, 3, 47}, {1, 3, -9}, {4, 3, 9},
		{0, 5, 9}, {1, 5, 9},
		{2, 9, -9},
		{0, 11, 9},
	}
	for _, v := range nonZero {
		coeff[v.c*stride+v.r] = v.v
	}

	// libaom OUT_ROW0 - pred(206) gives the residual row 0 below.
	wantRow0 := []int16{
		-2, -2, -2, -3, -3, -4, -4, -5,
		-6, -6, -6, -6, -7, -8, -8, -9,
		-9, -9, -10, -10, -11, -11, -11, -11,
		-12, -12, -13, -13, -13, -14, -14, -14,
	}

	width := int(sz.Width)
	height := int(sz.Height)
	gotDst := make([]int16, width*height)
	scratch := make([]int32, width*height)
	if err := InverseBlockBitDepth(gotDst, width, coeff, stride, scratch, sz, TypeDCTDCT, 8); err != nil {
		t.Fatalf("InverseBlockBitDepth: %v", err)
	}
	for c := 0; c < width; c++ {
		if gotDst[c] != wantRow0[c] {
			t.Fatalf("row 0 col %d: got=%d want=%d (full row got=%v)", c, gotDst[c], wantRow0[c], gotDst[:width])
		}
	}

	// Also cross-check against our portable libaom reference port for the
	// full residual block.
	wantResid := libaomInverseResidual(coeff, stride, sz, TypeDCTDCT, 8)
	for i := 0; i < width*height; i++ {
		if int32(gotDst[i]) != wantResid[i] {
			r := i / width
			c := i % width
			t.Fatalf("idx=%d (r=%d c=%d) goav1=%d libaomRef=%d", i, r, c, gotDst[i], wantResid[i])
		}
	}
}
