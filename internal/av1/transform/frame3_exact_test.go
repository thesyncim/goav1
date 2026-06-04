package transform

import "testing"

func TestFrame3ExactBlock64x64IDCT(t *testing.T) {
	sz := Size{Width: 64, Height: 64}
	coeffSize := adjustedScanSize(sz) // 32x32
	stride := int(coeffSize.Height)   // 32
	width := int(sz.Width)
	height := int(sz.Height)

	coeff := make([]int32, 32*32)
	// Exact dequantized coefficients from frame 3 debug test:
	// qindex=80, 8-bit, txScale=2, q.DC=74, q.AC=87
	// scan[0]->col=0,row=0 (DC): q=-3 -> dq=-55
	// scan[1]->col=1,row=0 (AC): q=-1 -> dq=-21
	// scan[3]->col=0,row=2 (AC): q=1 -> dq=21
	// scan[34]->col=1,row=6 (AC): q=1 -> dq=21
	// scan[35]->col=0,row=7 (AC): q=1 -> dq=21
	coeff[0*stride+0] = -55 // DC
	coeff[1*stride+0] = -21
	coeff[0*stride+2] = 21
	coeff[1*stride+6] = 21
	coeff[0*stride+7] = 21

	wantResid := libaomInverseResidual64x64(coeff, stride, 8)

	gotDst := make([]int16, width*height)
	scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
	scratch := make([]int32, scratchLen)
	if err := InverseBlockBitDepth(gotDst, width, coeff, stride, scratch, sz, TypeDCTDCT, 8); err != nil {
		t.Fatalf("err=%v", err)
	}
	diffs := 0
	for i := range gotDst {
		if int32(gotDst[i]) != wantResid[i] {
			r := i / width
			c := i % width
			if diffs < 10 {
				t.Logf("DIFF idx=%d (r=%d,c=%d) got=%d want=%d", i, r, c, gotDst[i], wantResid[i])
			}
			diffs++
		}
	}
	if diffs > 0 {
		t.Errorf("total diffs: %d", diffs)
	} else {
		t.Logf("IDCT matches libaom reference exactly for exact frame 3 block")
	}
}
