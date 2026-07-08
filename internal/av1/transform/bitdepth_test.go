// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"testing"
)

// TestStageRangeBoundsMatchLibaom anchors the bit-depth-dependent clamp bounds
// to the values libaom's av1_gen_inv_stage_range() derives.
func TestStageRangeBoundsMatchLibaom(t *testing.T) {
	cases := []struct {
		bd                     uint8
		wantRowMin, wantRowMax int32
		wantColMin, wantColMax int32
	}{
		{bd: 8, wantRowMin: -32768, wantRowMax: 32767, wantColMin: -32768, wantColMax: 32767},
		{bd: 10, wantRowMin: -131072, wantRowMax: 131071, wantColMin: -32768, wantColMax: 32767},
		{bd: 12, wantRowMin: -524288, wantRowMax: 524287, wantColMin: -131072, wantColMax: 131071},
	}
	for _, tc := range cases {
		rmin, rmax, cmin, cmax, ok := stageRangeBounds(tc.bd)
		if !ok {
			t.Fatalf("bd=%d: stageRangeBounds returned !ok", tc.bd)
		}
		if rmin != tc.wantRowMin || rmax != tc.wantRowMax {
			t.Fatalf("bd=%d: row=[%d,%d] want [%d,%d]", tc.bd, rmin, rmax, tc.wantRowMin, tc.wantRowMax)
		}
		if cmin != tc.wantColMin || cmax != tc.wantColMax {
			t.Fatalf("bd=%d: col=[%d,%d] want [%d,%d]", tc.bd, cmin, cmax, tc.wantColMin, tc.wantColMax)
		}
	}
}

// TestStageRangeBoundsRejectsInvalidBitDepth ensures only 8, 10, 12 are
// accepted.
func TestStageRangeBoundsRejectsInvalidBitDepth(t *testing.T) {
	for _, bd := range []uint8{0, 1, 6, 7, 9, 11, 13, 16, 255} {
		if _, _, _, _, ok := stageRangeBounds(bd); ok {
			t.Fatalf("bd=%d: stageRangeBounds unexpectedly accepted", bd)
		}
	}
}

// TestInverseBlockBitDepthRowClampDiverges exercises a 4x4 DCT_DCT block whose
// dequantized coefficient exceeds int16 to verify the 10-bit row clamp keeps
// more headroom than the legacy 8-bit clamp. This anchors the fix for the
// 10-bit quantizer-32 / quantizer-63 cohort: with the 8-bit clamp the row
// transform input saturates to ±32767, but libaom's bd+8=18-bit clamp lets the
// real value flow through.
func TestInverseBlockBitDepthRowClampDiverges(t *testing.T) {
	size := Size{Width: 4, Height: 4}
	// 200000 > int16 max (32767); fits in 18 bits (131071? no, 200000 > 131071
	// too). We want a value that the 8-bit clamp saturates and the 10-bit
	// clamp does not — so 100000, well within ±131071 (bd+8=18) but outside
	// ±32767.
	coeff := []int32{
		100000, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	}
	scratch8 := make([]int32, 16)
	scratch10 := make([]int32, 16)
	dst8 := make([]int16, 16)
	dst10 := make([]int16, 16)
	if err := InverseBlock(dst8, 4, coeff, 4, scratch8, size, TypeDCTDCT); err != nil {
		t.Fatal(err)
	}
	if err := InverseBlockBitDepth(dst10, 4, coeff, 4, scratch10, size, TypeDCTDCT, 10); err != nil {
		t.Fatal(err)
	}
	if dst8[0] == dst10[0] {
		t.Fatalf("expected 8-bit (clamped) and 10-bit (unclamped) residuals to differ at [0]; both produced %d", dst8[0])
	}
}

// TestInverseBlockBitDepthMatchesInverseBlockAt8Bit anchors the equivalence
// guarantee: bd=8 must reproduce the legacy InverseBlock output bit-for-bit.
func TestInverseBlockBitDepthMatchesInverseBlockAt8Bit(t *testing.T) {
	size := Size{Width: 8, Height: 8}
	coeff := make([]int32, 64)
	for i := range coeff {
		// Spec-valid magnitude: the int16 8bpc column path (dav1d) requires the
		// int16 butterfly intermediates not to overflow, which real decode
		// guarantees. Full-range random coeffs are not valid transform inputs.
		coeff[i] = int32((i*37)%1024) - 512
	}
	scratchA := make([]int32, 64)
	scratchB := make([]int32, 64)
	dstA := make([]int16, 64)
	dstB := make([]int16, 64)
	for _, typ := range []Type{TypeDCTDCT, TypeADSTDCT, TypeADSTADST, TypeIDTX, TypeVDCT, TypeHADST} {
		if err := InverseBlock(dstA, 8, coeff, 8, scratchA, size, typ); err != nil {
			t.Fatalf("InverseBlock type=%d: %v", typ, err)
		}
		if err := InverseBlockBitDepth(dstB, 8, coeff, 8, scratchB, size, typ, 8); err != nil {
			t.Fatalf("InverseBlockBitDepth type=%d: %v", typ, err)
		}
		for i := range dstA {
			if dstA[i] != dstB[i] {
				t.Fatalf("type=%d index=%d: InverseBlock=%d InverseBlockBitDepth(bd=8)=%d", typ, i, dstA[i], dstB[i])
			}
		}
	}
}
