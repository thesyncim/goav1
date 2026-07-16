// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestWarpHorizontal8DotProdMatchesScalar(t *testing.T) {
	if !cpu.Detected.DOTPROD {
		t.Skip("DOTPROD unavailable")
	}
	ref, _ := testPlane(48, 48, 1, 48)
	for i := range ref.Pix {
		ref.Pix[i] = byte((i*61 + (i/48)*29 + 7) & 0xff)
	}
	cases := []struct {
		name             string
		sx4, alpha, beta int
	}{
		{name: "zero", sx4: 0},
		{name: "positive", sx4: 512, alpha: 32, beta: -16},
		{name: "negative", sx4: -4096, alpha: -320, beta: -64},
		{name: "mixed", sx4: 2048, alpha: 96, beta: 48},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !warpHorizontalFilterRangeOK(tc.sx4, tc.alpha, tc.beta) {
				t.Fatal("test phase unexpectedly outside filter table")
			}
			var want, got [warpedIntermediateRows * warpedIntermediateColumns]int32
			warpHorizontal8Resident(&want, ref, 20, tc.sx4, 20, 0, tc.alpha, tc.beta, round0Bits, 8+filterBits-1)
			ctx := warpHorizontal8DotProdCtx{
				tmp:     &got[0],
				ref:     &ref.Pix[(20-7)*ref.Stride+20-7],
				filter:  &warpedFilterI8Packed[0],
				permute: &warpHorizontalPermute[0],
				refStr:  uintptr(ref.Stride),
				sxStart: int64(tc.sx4 - 3*tc.beta),
				alpha:   int64(tc.alpha),
				beta:    int64(tc.beta),
			}
			warpHorizontal8DotProdAsm(&ctx)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("tmp[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func BenchmarkWarpHorizontal8ResidentScalar(b *testing.B) {
	ref, _ := testPlane(48, 48, 1, 48)
	for i := range ref.Pix {
		ref.Pix[i] = byte((i*61 + (i/48)*29 + 7) & 0xff)
	}
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpHorizontal8Resident(&tmp, ref, 20, 512, 20, -256, 32, -16, round0Bits, 8+filterBits-1)
	}
	sink32 = tmp[0]
}

func BenchmarkWarpHorizontal8ResidentDotProd(b *testing.B) {
	if !cpu.Detected.DOTPROD {
		b.Skip("DOTPROD unavailable")
	}
	ref, _ := testPlane(48, 48, 1, 48)
	for i := range ref.Pix {
		ref.Pix[i] = byte((i*61 + (i/48)*29 + 7) & 0xff)
	}
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	ctx := warpHorizontal8DotProdCtx{
		tmp:     &tmp[0],
		ref:     &ref.Pix[(20-7)*ref.Stride+20-7],
		filter:  &warpedFilterI8Packed[0],
		permute: &warpHorizontalPermute[0],
		refStr:  uintptr(ref.Stride),
		sxStart: int64(512 - 3*(-16)),
		alpha:   32,
		beta:    -16,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		warpHorizontal8DotProdAsm(&ctx)
	}
	sink32 = tmp[0]
}

var sink32 int32
