// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func compoundRoundOffset8() int {
	round0 := compoundRound0(8)
	offsetBits := 8 + 2*filterBits - round0
	return (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
}

func TestCompoundX8DotProdMatchesPureGo(t *testing.T) {
	if !cpu.Detected.DOTPROD {
		t.Skip("DOTPROD not detected")
	}
	ref, _ := testPlane(96, 48, 1, 96)
	fillMotionTestPlane(ref)
	roundOffset := compoundRoundOffset8()
	cases := []struct {
		name          string
		width, height int
		subX          int
		filter        InterpFilter
	}{
		{"regular_8x4", 8, 4, 3, InterpEightTapRegular},
		{"regular_16x8", 16, 8, 5, InterpEightTapRegular},
		{"smooth_32x13", 32, 13, 7, InterpEightTapSmooth},
		{"sharp_64x16", 64, 16, 11, InterpMultiTapSharp},
		{"bilinear_32x8", 32, 8, 8, InterpBilinear},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			kernel, err := interpKernel(tc.filter, tc.width, tc.subX)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			refX, refY := filterTaps, filterTaps
			predictInterCompoundRef8ToConvBufXDotProd(got, ref, refX, refY, tc.width, tc.height, kernel, roundOffset)
			predictInterCompoundRef8ToConvBufXPureGo(want, ref, refX, refY, tc.width, tc.height, kernel, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample %d: DOTPROD=%d PureGo=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompoundX8DotProdZeroAlloc(t *testing.T) {
	if !cpu.Detected.DOTPROD {
		t.Skip("DOTPROD not detected")
	}
	ref, _ := testPlane(64, 64, 1, 64)
	fillMotionTestPlane(ref)
	out := make([]uint16, 32*32)
	kernel, err := interpKernel(InterpEightTapRegular, 32, 5)
	if err != nil {
		t.Fatal(err)
	}
	roundOffset := compoundRoundOffset8()
	allocs := testing.AllocsPerRun(50, func() {
		predictInterCompoundRef8ToConvBufXDotProd(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
	if allocs != 0 {
		t.Fatalf("compound X DOTPROD allocated %v times, want 0", allocs)
	}
}

func BenchmarkCompoundConvBufX8DotProd_32(b *testing.B) {
	if !cpu.Detected.DOTPROD {
		b.Skip("DOTPROD not detected")
	}
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	kernel, err := interpKernel(InterpEightTapRegular, 32, 3)
	if err != nil {
		b.Fatal(err)
	}
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	roundOffset := compoundRoundOffset8()
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRef8ToConvBufXDotProd(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
}

func BenchmarkCompoundConvBufX8NEONDirect_32(b *testing.B) {
	_, ref := benchPlanes(32, 8)
	var buf CompoundConvBuf
	kernel, err := interpKernel(InterpEightTapRegular, 32, 3)
	if err != nil {
		b.Fatal(err)
	}
	out, ok := compoundConvBufView(&buf, 32, 32)
	if !ok {
		b.Fatal("invalid convbuf")
	}
	roundOffset := compoundRoundOffset8()
	runConvolveBench(b, 32, 32, func() {
		predictInterCompoundRef8ToConvBufXNEON(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
}
