// SPDX-License-Identifier: BSD-2-Clause
//
// Verify goav1's inverseDCT64 against libaom's av1_idct64 at the 16-bit
// stage clamp (bd=8). Pins the 64x64 DCT_DCT row/column pipeline against
// the reference; specifically guards against the regression where
// inverseDCT64 produced a growing-with-column residual error for sparse
// 64x64 DCT_DCT blocks (input positions 33, 35, ..., 63 of the column
// pass were silently treated as zero).

package transform

import (
	"math/rand"
	"testing"
)

func TestInverseDCT64BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(64))
	for trial := range 200 {
		var input [64]int32
		// Mostly sparse: zero everywhere then plant 1-4 random coefficients.
		nz := 1 + rng.Intn(4)
		for range nz {
			pos := rng.Intn(64)
			input[pos] = int32(rng.Intn(1<<15) - (1 << 14))
		}
		var want [64]int32
		var got [64]int32
		copy(got[:], input[:])
		libaomIDCT64(input[:], want[:])
		inverseDCT64(got[:], 1, -(1 << 15), (1<<15)-1)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial=%d idx=%d got=%d want=%d input=%v", trial, i, got[i], want[i], input)
			}
		}
	}
}

func TestInverseDCT64BitExactLibaomDense(t *testing.T) {
	rng := rand.New(rand.NewSource(1064))
	for trial := range 200 {
		var input [64]int32
		for i := range input {
			input[i] = int32(rng.Intn(1<<15) - (1 << 14))
		}
		var want [64]int32
		var got [64]int32
		copy(got[:], input[:])
		libaomIDCT64(input[:], want[:])
		inverseDCT64(got[:], 1, -(1 << 15), (1<<15)-1)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial=%d idx=%d got=%d want=%d", trial, i, got[i], want[i])
			}
		}
	}
}

// TestInverseDCT64BitExactLibaomImpulse pins the impulse response (single
// non-zero input) of inverseDCT64 against libaom's av1_idct64 at each of
// the 64 input positions. The 32..63 high-frequency positions are the
// ones whose impulse response was previously wrong (returned zero) and
// drove the 64x64 DCT_DCT residual error on the column pass.
func TestInverseDCT64BitExactLibaomImpulse(t *testing.T) {
	for pos := range 64 {
		var input [64]int32
		input[pos] = 1000
		var want [64]int32
		var got [64]int32
		copy(got[:], input[:])
		libaomIDCT64(input[:], want[:])
		inverseDCT64(got[:], 1, -(1 << 15), (1<<15)-1)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("pos=%d idx=%d got=%d want=%d", pos, i, got[i], want[i])
			}
		}
	}
}

// libaom1DAll dispatches all supported 1D inverse transforms used by the 2D
// pipeline, including length 64.
func libaom1DAll(input []int32, output []int32, length int, typ tx1DType) {
	if typ == tx1DDCT && length == 64 {
		libaomIDCT64(input, output)
		return
	}
	libaom1D(input, output, length, typ)
}

// libaomInverseResidual64x64 mirrors libaom's inv_txfm2d_add_64x64_c at bd=8.
// The first 32 columns / 32 rows of the input are read from `coeff` (column-
// major with stride coeffStride); the rest of the implicit 64x64 input is
// zero. The residual returned is row-major width*height int32, matching
// libaomInverseResidual.
func libaomInverseResidual64x64(coeff []int32, coeffStride int, bd int) []int32 {
	width := 64
	height := 64
	shift0 := 2
	shift1 := 4
	rowBits := bd + 8
	colBits := max(bd+6, 16)

	buf := make([]int32, width*height)
	rowIn := make([]int32, width)
	rowOut := make([]int32, width)

	// Rows
	for r := range height {
		for c := range width {
			var v int32
			if c < 32 && r < 32 {
				v = coeff[c*coeffStride+r]
			}
			rowIn[c] = v
		}
		libaomClampBits(rowIn, rowBits)
		libaom1DAll(rowIn, rowOut, width, tx1DDCT)
		libaomRoundShiftArray(rowOut, shift0)
		copy(buf[r*width:(r+1)*width], rowOut)
	}

	colIn := make([]int32, height)
	colOut := make([]int32, height)
	out := make([]int32, width*height)
	for c := range width {
		for r := range height {
			colIn[r] = buf[r*width+c]
		}
		libaomClampBits(colIn, colBits)
		libaom1DAll(colIn, colOut, height, tx1DDCT)
		libaomRoundShiftArray(colOut, shift1)
		for r := range height {
			out[r*width+c] = colOut[r]
		}
	}
	return out
}

// TestInverseBlockBitDepth64x64DCTDCTRandom exercises the full 2D 64x64
// DCT_DCT inverse pipeline against a libaom-faithful reference. Uses
// sparse-ish (90% zero) coefficients to mimic real coded residuals,
// reproducing the row/column shape that drove the frame-3 monochrome
// growing-with-column residual error.
func TestInverseBlockBitDepth64x64DCTDCTRandom(t *testing.T) {
	sz := Size{Width: 64, Height: 64}
	coeffSize := adjustedScanSize(sz) // 32x32

	rng := rand.New(rand.NewSource(20260527))
	for trial := range 20 {
		coeffStride := int(coeffSize.Height)
		coeff := make([]int32, int(coeffSize.Width)*coeffStride)
		for i := range coeff {
			if rng.Intn(10) == 0 {
				coeff[i] = int32(rng.Intn(1<<15) - (1 << 14))
			}
		}
		wantResid := libaomInverseResidual64x64(coeff, coeffStride, 8)

		width := int(sz.Width)
		height := int(sz.Height)
		gotDst := make([]int16, width*height)
		scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
		scratch := make([]int32, scratchLen)
		if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, TypeDCTDCT, 8); err != nil {
			t.Fatalf("trial=%d err=%v", trial, err)
		}
		for i := range gotDst {
			if int32(gotDst[i]) != wantResid[i] {
				r := i / width
				c := i % width
				t.Fatalf("trial=%d idx=%d (r=%d c=%d) got=%d want=%d", trial, i, r, c, gotDst[i], wantResid[i])
			}
		}
	}
}

// TestInverseBlockBitDepth64x64DCTDCTSparseFrame3 pins the specific
// sparse-coefficient pattern referenced by the prior-agent diagnostic for
// the monochrome frame-3 mismatch (block mi=(32,16) BLOCK_64X64,
// compound NEAREST_NEAREST, three non-zero coefficients producing a
// smooth-varying-with-column residual). The exact coefficients used are
// synthetic but exercise the row+column 64-length DCT path that previously
// produced growing-with-column residual error.
func TestInverseBlockBitDepth64x64DCTDCTSparseFrame3(t *testing.T) {
	sz := Size{Width: 64, Height: 64}
	coeffSize := adjustedScanSize(sz) // 32x32
	stride := int(coeffSize.Height)

	coeff := make([]int32, int(coeffSize.Width)*stride)
	// Three non-zero coefficients mimicking the sparse residual that
	// produced the row 64 residual error in frame 3 of the monochrome
	// vector. The exact coefficient indices/values are chosen to
	// exercise low- and mid-frequency rows that recurse into the
	// length-64 column pass on positions 33..63.
	coeff[0*stride+0] = -200 // DC
	coeff[0*stride+1] = 150  // (col=0,row=1)
	coeff[1*stride+2] = -80  // (col=1,row=2)
	wantResid := libaomInverseResidual64x64(coeff, stride, 8)

	width := int(sz.Width)
	height := int(sz.Height)
	gotDst := make([]int16, width*height)
	scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
	scratch := make([]int32, scratchLen)
	if err := InverseBlockBitDepth(gotDst, width, coeff, stride, scratch, sz, TypeDCTDCT, 8); err != nil {
		t.Fatalf("err=%v", err)
	}
	for i := range gotDst {
		if int32(gotDst[i]) != wantResid[i] {
			r := i / width
			c := i % width
			t.Fatalf("idx=%d (r=%d c=%d) got=%d want=%d", i, r, c, gotDst[i], wantResid[i])
		}
	}
}
