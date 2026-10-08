// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import "testing"

// warpDiffCases sweeps the warp sub-pel / shear parameter space so the filter
// index (offs) selection and per-column/per-row filter variation are all
// exercised against the scalar reference.
var warpDiffCases = []struct {
	name                     string
	baseSX, baseSY           int
	alpha, beta, gamma, delta int
}{
	{"zero", 0, 0, 0, 0, 0, 0},
	{"smallshear", 512, -256, 32, -16, 24, -8},
	{"bigshear", -1 << 18, 1 << 17, -320, -64, 128, -64},
	{"mixed", 1024, 2048, 96, 48, -96, 40},
	{"gamma0", 300, -700, 64, -32, 0, -64},
}

func TestWarpHorizontal8ResidentSIMDMatchesScalar(t *testing.T) {
	ref, _ := testPlane(48, 48, 1, 48)
	for i := range ref.Pix {
		ref.Pix[i] = byte((i*61 + (i/48)*29 + 7) & 0xff)
	}
	const reduceBitsHoriz = round0Bits
	const offsetBitsHoriz = 8 + filterBits - 1
	const ix4, iy4 = 20, 20 // resident: ix4-7>=0, ix4+7<W, iy4-7>=0, iy4+7<H
	for _, tc := range warpDiffCases {
		var want, got [warpedIntermediateRows * warpedIntermediateColumns]int32
		wRet := warpHorizontal8Resident(&want, ref, ix4, tc.baseSX, iy4, tc.baseSY, tc.alpha, tc.beta, reduceBitsHoriz, offsetBitsHoriz)
		gRet := warpHorizontal8ResidentSIMD(&got, ref, ix4, tc.baseSX, iy4, tc.baseSY, tc.alpha, tc.beta, reduceBitsHoriz, offsetBitsHoriz)
		if wRet != gRet {
			t.Fatalf("%s: sy4 ret mismatch got=%d want=%d", tc.name, gRet, wRet)
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("%s: tmp[%d]=%d want %d", tc.name, i, got[i], want[i])
			}
		}
	}
}

func TestWarpVertical8FullSIMDMatchesScalar(t *testing.T) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	for i := range tmp {
		tmp[i] = int32((i*97+31)%8192 - 4096)
	}
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	for _, tc := range warpDiffCases {
		want, _ := testPlane(48, 48, 1, 48)
		got, _ := testPlane(48, 48, 1, 48)
		for i := range want.Pix {
			want.Pix[i], got.Pix[i] = 0xaa, 0xaa
		}
		// General (per-column gamma) path.
		warpVertical8Full(want, &tmp, 16, 16, 1, 2, tc.baseSY, tc.gamma, tc.delta, reduceBitsVert, offsetBitsVert)
		warpVertical8FullSIMD(got, &tmp, 16, 16, 1, 2, tc.baseSY, tc.gamma, tc.delta, reduceBitsVert, offsetBitsVert)
		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("%s Full: pix[%d]=%d want %d", tc.name, i, got.Pix[i], want.Pix[i])
			}
		}
	}
}

func TestWarpVertical8FullGamma0SIMDMatchesScalar(t *testing.T) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	for i := range tmp {
		tmp[i] = int32((i*131+17)%8192 - 4096)
	}
	const reduceBitsVert = round1Bits
	const offsetBitsVert = 8 + 2*filterBits - round0Bits
	for _, tc := range warpDiffCases {
		want, _ := testPlane(48, 48, 1, 48)
		got, _ := testPlane(48, 48, 1, 48)
		for i := range want.Pix {
			want.Pix[i], got.Pix[i] = 0x5a, 0x5a
		}
		warpVertical8FullGamma0(want, &tmp, 16, 16, 1, 2, tc.baseSY, tc.delta, reduceBitsVert, offsetBitsVert)
		warpVertical8FullGamma0SIMD(got, &tmp, 16, 16, 1, 2, tc.baseSY, tc.delta, reduceBitsVert, offsetBitsVert)
		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("%s Gamma0: pix[%d]=%d want %d", tc.name, i, got.Pix[i], want.Pix[i])
			}
		}
	}
}
