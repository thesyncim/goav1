package transform

import (
	"math/rand"
	"testing"
)

// libaomInvShift returns the AV1 inverse-shift pair (shift[0], shift[1]) for
// size, mirroring av1_inv_txfm_shift_ls in libaom. Values are <= 0 in libaom;
// we return the positive counts (negation of the libaom values).
func libaomInvShift(size Size) (shift0 int, shift1 int) {
	type pair struct{ w, h, s0 int }
	table := []pair{
		{4, 4, 0},
		{8, 8, 1},
		{16, 16, 2},
		{32, 32, 2},
		{64, 64, 2},
		{4, 8, 0},
		{8, 4, 0},
		{8, 16, 1},
		{16, 8, 1},
		{16, 32, 1},
		{32, 16, 1},
		{32, 64, 1},
		{64, 32, 1},
		{4, 16, 1},
		{16, 4, 1},
		{8, 32, 2},
		{32, 8, 2},
		{16, 64, 2},
		{64, 16, 2},
	}
	for _, p := range table {
		if p.w == int(size.Width) && p.h == int(size.Height) {
			return p.s0, 4
		}
	}
	return 0, 0
}

func libaomClampBits(buf []int32, bits int) {
	if bits <= 0 {
		return
	}
	min := -(int64(1) << uint(bits-1))
	max := int64(1)<<uint(bits-1) - 1
	for i, v := range buf {
		x := int64(v)
		if x < min {
			x = min
		} else if x > max {
			x = max
		}
		buf[i] = int32(x)
	}
}

func libaomRoundShiftArray(arr []int32, bits int) {
	if bits == 0 {
		return
	}
	for i, v := range arr {
		x := int64(v) + (int64(1) << uint(bits-1))
		arr[i] = int32(x >> uint(bits))
	}
}

func libaom1D(input []int32, output []int32, length int, typ tx1DType) {
	switch typ {
	case tx1DDCT:
		switch length {
		case 4:
			libaomIDCT4(input, output)
		case 8:
			libaomIDCT8(input, output)
		case 16:
			libaomIDCT16(input, output)
		case 32:
			libaomIDCT32(input, output)
		}
	case tx1DADST, tx1DFlipADST:
		switch length {
		case 4:
			// Use goav1's adst4 since it's tested in TestInverseADST1DRoundTripMatchesLibaomShape.
			copy(output, input)
			inverseADST4(output, 1)
		case 8:
			libaomIADST8(input, output)
		case 16:
			libaomIADST16(input, output)
		}
	case tx1DIdentity:
		for i := range length {
			v := input[i]
			out, _ := identity1DValue(v, length)
			output[i] = out
		}
	}
}

// libaomInverseBlock implements libaom's inv_txfm2d_add_c for bd=8 and writes
// the residual (post-add) to dst as int16. dst is assumed to be already
// populated with the predicted samples; the residual is added and the result
// is clamped to [0,255]. Returns the int16 residual values (post-round-shift,
// pre-clamp) for direct comparison with InverseBlockBitDepth.
func libaomInverseResidual(coeff []int32, coeffStride int, size Size, typ Type, bd int) []int32 {
	vertical, horizontal, _ := typ.tx1DTypes()
	width := int(size.Width)
	height := int(size.Height)
	rectType := 0
	if width*2 == height || height*2 == width {
		rectType = 1
	}

	shift0, shift1 := libaomInvShift(size)
	rowBits := bd + 8
	colBits := max(bd+6, 16)

	buf := make([]int32, width*height)
	rowIn := make([]int32, width)
	rowOut := make([]int32, width)

	// Rows
	for r := range height {
		for c := range width {
			v := coeff[c*coeffStride+r]
			if rectType == 1 {
				// libaom: round_shift(v * NewInvSqrt2, NewSqrt2Bits)
				// NewInvSqrt2 = 2896, NewSqrt2Bits = 12.
				v = libaomRoundShiftSingle(int64(v)*2896, 12)
			}
			rowIn[c] = v
		}
		libaomClampBits(rowIn, rowBits)
		libaom1D(rowIn, rowOut, width, horizontal)
		// av1_round_shift_array(rowOut, width, -shift0). -shift0 is negative => positive bit count.
		libaomRoundShiftArray(rowOut, shift0)
		copy(buf[r*width:(r+1)*width], rowOut)
	}

	colIn := make([]int32, height)
	colOut := make([]int32, height)
	out := make([]int32, width*height)
	for c := range width {
		flipLR := isLRFlip(typ)
		flipUD := isUDFlip(typ)
		for r := range height {
			cc := c
			if flipLR {
				cc = width - c - 1
			}
			colIn[r] = buf[r*width+cc]
		}
		libaomClampBits(colIn, colBits)
		libaom1D(colIn, colOut, height, vertical)
		libaomRoundShiftArray(colOut, shift1)
		for r := range height {
			rr := r
			if flipUD {
				rr = height - r - 1
			}
			out[r*width+c] = colOut[rr]
		}
	}
	return out
}

func libaomRoundShiftSingle(v int64, bits int) int32 {
	return int32((v + (int64(1) << uint(bits-1))) >> uint(bits))
}

func isLRFlip(typ Type) bool {
	switch typ {
	case TypeDCTFlipADST, TypeADSTFlipADST, TypeFlipADSTFlipADST, TypeHFlipADST:
		return true
	}
	return false
}

func isUDFlip(typ Type) bool {
	switch typ {
	case TypeFlipADSTDCT, TypeFlipADSTADST, TypeFlipADSTFlipADST, TypeVFlipADST:
		return true
	}
	return false
}

// TestInverseBlockBitDepth2DMatchesLibaom2D verifies our 2D inverse transform
// pipeline against a faithful port of libaom's inv_txfm2d_add_c (bd=8) using
// random coefficients in the validly-decoded range. Tests every supported
// transform type/size combo at sizes 4x4 through 32x32 plus rect-2.
func TestInverseBlockBitDepth2DMatchesLibaom2D(t *testing.T) {
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 8},
		{Width: 8, Height: 16},
		{Width: 16, Height: 8},
		{Width: 8, Height: 32},
		{Width: 32, Height: 8},
		{Width: 16, Height: 16},
		{Width: 16, Height: 32},
		{Width: 32, Height: 16},
		{Width: 32, Height: 32},
	}
	types := []Type{
		TypeDCTDCT,
		TypeADSTDCT,
		TypeDCTADST,
		TypeADSTADST,
		TypeFlipADSTDCT,
		TypeDCTFlipADST,
		TypeFlipADSTFlipADST,
		TypeADSTFlipADST,
		TypeFlipADSTADST,
		TypeVDCT,
		TypeHDCT,
		TypeVADST,
		TypeHADST,
		TypeVFlipADST,
		TypeHFlipADST,
	}
	rng := rand.New(rand.NewSource(7))
	for _, sz := range sizes {
		for _, typ := range types {
			if !typ.Supported(sz) {
				continue
			}
			// Skip ADST sizes 32+ (libaom uses ADST16 max).
			vertical, horizontal, _ := typ.tx1DTypes()
			if vertical == tx1DADST || vertical == tx1DFlipADST {
				if sz.Height > 16 {
					continue
				}
			}
			if horizontal == tx1DADST || horizontal == tx1DFlipADST {
				if sz.Width > 16 {
					continue
				}
			}
			for trial := range 3 {
				coeffSize := adjustedScanSize(sz)
				coeffStride := int(coeffSize.Height)
				coeff := make([]int32, int(coeffSize.Width)*coeffStride)
				// 8-bit dq_coeff is up to ±(1<<15) per libaom decodetxb clamp.
				for i := range coeff {
					coeff[i] = int32(rng.Intn(1<<15) - (1 << 14))
				}
				wantResid := libaomInverseResidual(coeff, coeffStride, sz, typ, 8)

				width := int(sz.Width)
				height := int(sz.Height)
				gotDst := make([]int16, width*height)
				scratchLen, _ := ScratchLenForType(typ, sz)
				scratch := make([]int32, scratchLen)
				if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, typ, 8); err != nil {
					t.Fatalf("size=%v typ=%d trial=%d err=%v", sz, typ, trial, err)
				}
				for i := range gotDst {
					if int32(gotDst[i]) != wantResid[i] {
						t.Errorf("size=%v typ=%d trial=%d idx=%d got=%d want=%d", sz, typ, trial, i, gotDst[i], wantResid[i])
						break
					}
				}
			}
		}
	}
}
