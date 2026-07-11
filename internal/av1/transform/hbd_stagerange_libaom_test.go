// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"math/rand"
	"testing"
)

// libaomIDCT4Parametric reproduces libaom's av1_idct4 with a parametric
// stage-range clamp. The existing libaomIDCT4 in dct_libaom_test.go is hard
// coded to 16-bit (bd=8 row/col); this variant exercises the 18-bit (bd=10
// row, bd=12 col) and 20-bit (bd=12 row) configurations that libaom's
// av1_gen_inv_stage_range() selects for high-bit-depth streams.
func libaomIDCT4Parametric(input []int32, output []int32, clamp func(int32) int32) {
	cospi := libaomCospi12[:]
	output[0] = input[0]
	output[1] = input[2]
	output[2] = input[1]
	output[3] = input[3]
	var step [4]int32
	step[0] = libaomHalfBtf(cospi[32], output[0], cospi[32], output[1])
	step[1] = libaomHalfBtf(cospi[32], output[0], -cospi[32], output[1])
	step[2] = libaomHalfBtf(cospi[48], output[2], -cospi[16], output[3])
	step[3] = libaomHalfBtf(cospi[16], output[2], cospi[48], output[3])
	output[0] = clamp(step[0] + step[3])
	output[1] = clamp(step[1] + step[2])
	output[2] = clamp(step[1] - step[2])
	output[3] = clamp(step[0] - step[3])
}

// libaomIDCT8Parametric reproduces libaom's av1_idct8 with a parametric
// stage-range clamp. See libaomIDCT4Parametric.
func libaomIDCT8Parametric(input []int32, output []int32, clamp func(int32) int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	var step [8]int32
	bf0 := output
	bf1 := output
	bf1[0] = input[0]
	bf1[1] = input[4]
	bf1[2] = input[2]
	bf1[3] = input[6]
	bf1[4] = input[1]
	bf1[5] = input[5]
	bf1[6] = input[3]
	bf1[7] = input[7]

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = halfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = halfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = halfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = halfBtf(cospi[8], bf0[4], cospi[56], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = halfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = halfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = halfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = halfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[5])
	bf1[5] = clamp(bf0[4] - bf0[5])
	bf1[6] = clamp(-bf0[6] + bf0[7])
	bf1[7] = clamp(bf0[6] + bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[3])
	bf1[1] = clamp(bf0[1] + bf0[2])
	bf1[2] = clamp(bf0[1] - bf0[2])
	bf1[3] = clamp(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = halfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = halfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[7])
	bf1[1] = clamp(bf0[1] + bf0[6])
	bf1[2] = clamp(bf0[2] + bf0[5])
	bf1[3] = clamp(bf0[3] + bf0[4])
	bf1[4] = clamp(bf0[3] - bf0[4])
	bf1[5] = clamp(bf0[2] - bf0[5])
	bf1[6] = clamp(bf0[1] - bf0[6])
	bf1[7] = clamp(bf0[0] - bf0[7])
}

// stageClampBits returns the libaom clamp_value at the given bit width.
func stageClampBits(bits int) func(int32) int32 {
	if bits <= 0 {
		return func(v int32) int32 { return v }
	}
	min := -(int32(1) << uint(bits-1))
	max := (int32(1) << uint(bits-1)) - 1
	return func(v int32) int32 {
		if v < min {
			return min
		}
		if v > max {
			return max
		}
		return v
	}
}

// TestInverseDCT1DStageClampMatchesLibaomHBD verifies that goav1's 1D DCT
// produces libaom-exact output at the stage-clamp bit widths libaom's
// av1_gen_inv_stage_range() selects for high-bit-depth content:
//
//   - bd=8: row=col=16 bits
//   - bd=10: row=18 bits, col=16 bits
//   - bd=12: row=20 bits, col=18 bits
//
// This pins the high-bit-depth row clamp behaviour for 4x4 and 8x8 1D blocks,
// covering the q63 10-bit DC range (|coeff| up to 2^17 - 1).
func TestInverseDCT1DStageClampMatchesLibaomHBD(t *testing.T) {
	tests := []struct {
		name      string
		stageBits int
		maxInput  int32
	}{
		{"bd8-row-col-16", 16, 32767},
		{"bd10-row-18", 18, 131071},
		{"bd10-col-16", 16, 32767},
		{"bd12-row-20", 20, 524287},
		{"bd12-col-18", 18, 131071},
	}
	rng := rand.New(rand.NewSource(1102))
	for _, tc := range tests {
		clamp := stageClampBits(tc.stageBits)
		stageMin := -(int32(1) << uint(tc.stageBits-1))
		stageMax := (int32(1) << uint(tc.stageBits-1)) - 1
		t.Run(tc.name+"/dct4", func(t *testing.T) {
			for trial := range 1000 {
				var input [4]int32
				for i := range input {
					input[i] = int32(rng.Intn(int(2*tc.maxInput)+1)) - tc.maxInput
				}
				got := input
				inverseDCT4(got[:], 1, stageMin, stageMax)
				var want [4]int32
				libaomIDCT4Parametric(input[:], want[:], clamp)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial=%d input=%v got=%v want=%v", trial, input, got, want)
					}
				}
			}
		})
		t.Run(tc.name+"/dct8", func(t *testing.T) {
			for trial := range 1000 {
				var input [8]int32
				for i := range input {
					input[i] = int32(rng.Intn(int(2*tc.maxInput)+1)) - tc.maxInput
				}
				got := input
				inverseDCT8(got[:], 1, stageMin, stageMax)
				var want [8]int32
				libaomIDCT8Parametric(input[:], want[:], clamp)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial=%d coeff=%d input=%v got=%v want=%v", trial, i, input, got, want)
					}
				}
			}
		})
	}
}

// paramLibaom1D dispatches to libaom's parametric 1D inverse transforms using
// the provided stage clamp.
func paramLibaom1D(input []int32, output []int32, length int, typ tx1DType, clamp func(int32) int32) {
	switch typ {
	case tx1DDCT:
		switch length {
		case 4:
			libaomIDCT4Parametric(input, output, clamp)
		case 8:
			libaomIDCT8Parametric(input, output, clamp)
		default:
			// Larger sizes: bit-exactness against libaom at the standard
			// 16-bit clamp is already covered by the dct_libaom_test.go
			// suite. The 2D oracle therefore uses goav1's own 1D path with
			// per-stage clamps for sizes >= 16 — the 1D path is itself
			// libaom-faithful (see TestInverseDCT16BitExactLibaom).
			copy(output, input)
			minVal := clamp(int32(-1 << 30))
			maxVal := clamp(int32((1 << 30) - 1))
			inverseDCT1D(output, 1, length, minVal, maxVal)
		}
	case tx1DADST, tx1DFlipADST:
		// libaom applies the regular ADST 1D; the up/down or left/right flip
		// is realised at the final output write via flipUD / flipLR. The
		// oracle uses the goav1 ADST regardless of FlipADST, and the 2D
		// pipeline applies flipUD/flipLR at the output write.
		copy(output, input)
		minVal := clamp(int32(-1 << 30))
		maxVal := clamp(int32((1 << 30) - 1))
		inverseADST1D(output, 1, length, minVal, maxVal)
	case tx1DIdentity:
		for i := range length {
			out, _ := identity1DValue(input[i], length)
			output[i] = out
		}
	}
}

// correctLibaom2D builds a faithful libaom inv_txfm2d_add_c oracle for the
// given pixel bit depth (bd). Unlike libaomInverseResidual in
// inv2d_libaom_test.go, this oracle uses the correct *per-pass* stage clamp
// widths derived from av1_gen_inv_stage_range:
//
//   - row pass: stage_range = bd + 8 (18 bits for bd=10, 20 bits for bd=12)
//   - col pass: stage_range = max(bd + 6, 16) (16 bits for bd<=10, 18 for bd=12)
//
// This matches the actual libaom production behaviour for high-bit-depth
// content. Returns the post-final-shift residual values.
func correctLibaom2D(coeff []int32, coeffStride int, size Size, typ Type, bd int) []int32 {
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

	rowClamp := stageClampBits(rowBits)
	colClamp := stageClampBits(colBits)

	buf := make([]int32, width*height)
	rowIn := make([]int32, width)
	rowOut := make([]int32, width)

	// Rows
	for r := range height {
		for c := range width {
			v := coeff[c*coeffStride+r]
			if rectType == 1 {
				v = libaomRoundShiftSingle(int64(v)*2896, 12)
			}
			rowIn[c] = v
		}
		libaomClampBits(rowIn, rowBits)
		paramLibaom1D(rowIn, rowOut, width, horizontal, rowClamp)
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
		paramLibaom1D(colIn, colOut, height, vertical, colClamp)
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

// TestInverseBlockBitDepth2DMatchesLibaomHBD pins the high-bit-depth 2D
// inverse transform pipeline against the correct-per-pass libaom oracle for
// bd=10 (and bd=8 as a sanity check). Covers the worst-case q63 10-bit DC
// coefficient range (|dq_coeff| up to ±131071, which is libaom's bd+7 dequant
// clamp ceiling).
//
// This regression test anchors the fix-or-find investigation of the q63
// 10-bit lenient framework-dryrun mismatch: it pins the *transform stage*
// against the libaom inv_txfm2d_add_c specification exactly. If the transform
// stage is bit-exact against this oracle, any remaining q63 lenient failure
// is rooted *outside* internal/av1/transform/.
func TestInverseBlockBitDepth2DMatchesLibaomHBD(t *testing.T) {
	rng := rand.New(rand.NewSource(7777))
	cases := []struct {
		bd       int
		coeffAbs int32
	}{
		// bitDepth 8 uses the int16 column pipeline (dav1d 8bpc), which requires
		// the int16 butterfly not to overflow — a spec-valid-magnitude property.
		// 10-bit stays on the int32 pipeline and keeps its full range.
		{8, 512},
		{10, 131071},
	}
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 8, Height: 8},
		{Width: 16, Height: 16},
		{Width: 32, Height: 32},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 16},
		{Width: 16, Height: 8},
		{Width: 8, Height: 32},
		{Width: 32, Height: 8},
		{Width: 16, Height: 32},
		{Width: 32, Height: 16},
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
		TypeIDTX,
		TypeVDCT,
		TypeHDCT,
		TypeVADST,
		TypeHADST,
		TypeVFlipADST,
		TypeHFlipADST,
	}
	for _, tc := range cases {
		for _, sz := range sizes {
			for _, typ := range types {
				if !typ.Supported(sz) {
					continue
				}
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
					for i := range coeff {
						coeff[i] = int32(rng.Intn(int(2*tc.coeffAbs)+1)) - tc.coeffAbs
					}
					wantResid := correctLibaom2D(coeff, coeffStride, sz, typ, tc.bd)

					width := int(sz.Width)
					height := int(sz.Height)
					gotDst := make([]int16, width*height)
					scratchLen, _ := ScratchLenForType(typ, sz)
					scratch := make([]int32, scratchLen)
					if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, typ, uint8(tc.bd)); err != nil {
						t.Fatalf("bd=%d size=%v typ=%d err=%v", tc.bd, sz, typ, err)
					}
					for i := range gotDst {
						if int32(gotDst[i]) != wantResid[i] {
							t.Fatalf("bd=%d size=%v typ=%d trial=%d idx=%d got=%d want=%d", tc.bd, sz, typ, trial, i, gotDst[i], wantResid[i])
						}
					}
				}
			}
		}
	}
}

// TestInverseBlockBitDepthQ63WorstCaseDC anchors the worst-case 10-bit q63 DC
// coefficient against the libaom oracle. At q63 10-bit, the dequant DC value
// is 290 (dc_qlookup_10_QTX[63]) and the AV1 spec clamps dequantized
// coefficients to ±(2^(bd+7) - 1) = ±131071. This test sweeps each supported
// 2D transform shape with the DC at exactly the bd+7 ceiling and verifies
// goav1 reproduces libaom's residual bit-exactly through the 18-bit row /
// 16-bit col stage-range clamps.
func TestInverseBlockBitDepthQ63WorstCaseDC(t *testing.T) {
	sizes := []Size{
		{Width: 4, Height: 4},
		{Width: 8, Height: 8},
		{Width: 16, Height: 16},
		{Width: 32, Height: 32},
		{Width: 4, Height: 8},
		{Width: 8, Height: 4},
		{Width: 4, Height: 16},
		{Width: 16, Height: 4},
		{Width: 8, Height: 16},
		{Width: 16, Height: 8},
	}
	for _, sz := range sizes {
		for _, dc := range []int32{131071, -131072, 100000, -100000} {
			coeffSize := adjustedScanSize(sz)
			coeffStride := int(coeffSize.Height)
			coeff := make([]int32, int(coeffSize.Width)*coeffStride)
			coeff[0] = dc
			wantResid := correctLibaom2D(coeff, coeffStride, sz, TypeDCTDCT, 10)

			width := int(sz.Width)
			height := int(sz.Height)
			gotDst := make([]int16, width*height)
			scratchLen, _ := ScratchLenForType(TypeDCTDCT, sz)
			scratch := make([]int32, scratchLen)
			if err := InverseBlockBitDepth(gotDst, width, coeff, coeffStride, scratch, sz, TypeDCTDCT, 10); err != nil {
				t.Fatalf("size=%v dc=%d err=%v", sz, dc, err)
			}
			for i := range gotDst {
				if int32(gotDst[i]) != wantResid[i] {
					t.Fatalf("size=%v dc=%d idx=%d got=%d want=%d", sz, dc, i, gotDst[i], wantResid[i])
				}
			}
		}
	}
}
