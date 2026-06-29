// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestCompoundX8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(128, 80, 1, 128)
	fillMotionTestPlane(ref)
	roundOffset := compoundRoundOffset8()
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []struct {
		width  int
		height int
	}{
		{8, 1},
		{8, 4},
		{16, 7},
		{32, 13},
		{64, 16},
	}
	for _, filter := range filters {
		for _, size := range sizes {
			for subX := 1; subX <= subpelQ4Mask; subX++ {
				kernel, err := interpKernel(filter, size.width, subX)
				if err != nil {
					t.Fatal(err)
				}
				got := make([]uint16, size.width*size.height)
				want := make([]uint16, size.width*size.height)
				refX, refY := filterTaps, filterTaps
				predictInterCompoundRef8ToConvBufXI8MM(got, ref, refX, refY, size.width, size.height, kernel, roundOffset)
				predictInterCompoundRef8ToConvBufXPureGo(want, ref, refX, refY, size.width, size.height, kernel, roundOffset)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("filter=%d size=%dx%d subX=%d sample=%d I8MM=%d PureGo=%d",
							filter, size.width, size.height, subX, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestCompoundX8I8MMFallbackMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(ref)
	roundOffset := compoundRoundOffset8()
	kernel, err := interpKernel(InterpEightTapRegular, 8, 5)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name          string
		refX, refY    int
		width, height int
	}{
		{name: "left_edge", refX: 1, refY: filterTaps, width: 8, height: 8},
		{name: "right_edge", refX: 27, refY: filterTaps, width: 8, height: 8},
		{name: "odd_width", refX: filterTaps, refY: filterTaps, width: 12, height: 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]uint16, tc.width*tc.height)
			want := make([]uint16, tc.width*tc.height)
			predictInterCompoundRef8ToConvBufXI8MM(got, ref, tc.refX, tc.refY, tc.width, tc.height, kernel, roundOffset)
			predictInterCompoundRef8ToConvBufXPureGo(want, ref, tc.refX, tc.refY, tc.width, tc.height, kernel, roundOffset)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s sample=%d I8MM wrapper=%d PureGo=%d", tc.name, i, got[i], want[i])
				}
			}
		})
	}
}

func TestCompoundX8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
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
		predictInterCompoundRef8ToConvBufXI8MM(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
	if allocs != 0 {
		t.Fatalf("compound X I8MM allocated %v times, want 0", allocs)
	}
}

func BenchmarkCompoundConvBufX8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
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
		predictInterCompoundRef8ToConvBufXI8MM(out, ref, filterTaps, filterTaps, 32, 32, kernel, roundOffset)
	})
}

func TestConvolveX8I8MMMatchesPureGo(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []struct {
		width  int
		height int
	}{
		{8, 1},
		{8, 4},
		{16, 7},
		{32, 13},
		{64, 16},
	}
	for _, filter := range filters {
		for _, size := range sizes {
			ref, _ := testPlane(size.width+2*filterTaps, size.height+2*filterTaps, 1, size.width+2*filterTaps)
			fillMotionTestPlane(ref)
			got, _ := testPlane(size.width, size.height, 1, size.width)
			want, _ := testPlane(size.width, size.height, 1, size.width)
			for subX := 1; subX <= subpelQ4Mask; subX++ {
				kernel, err := interpKernel(filter, size.width, subX)
				if err != nil {
					t.Fatal(err)
				}
				clear(got.Pix)
				clear(want.Pix)
				convolveX8I8MM(got, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, kernel)
				convolveX8PureGo(want, ref, 0, 0, filterTaps, filterTaps, size.width, size.height, kernel)
				for y := 0; y < size.height; y++ {
					row := y * size.width
					for x := 0; x < size.width; x++ {
						i := row + x
						if got.Pix[i] != want.Pix[i] {
							t.Fatalf("filter=%d size=%dx%d subX=%d sample=(%d,%d) I8MM=%d PureGo=%d",
								filter, size.width, size.height, subX, x, y, got.Pix[i], want.Pix[i])
						}
					}
				}
			}
		}
	}
}

func TestConvolveX8I8MMFallbackMatchesNEON(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	ref, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(ref)
	kernel, err := interpKernel(InterpEightTapRegular, 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := testPlane(4, 16, 1, 4)
	want, _ := testPlane(4, 16, 1, 4)
	convolveX8I8MM(got, ref, 0, 0, filterTaps, filterTaps, 4, 16, kernel)
	convolveX8NEON(want, ref, 0, 0, filterTaps, filterTaps, 4, 16, kernel)
	for i := range got.Pix {
		if got.Pix[i] != want.Pix[i] {
			t.Fatalf("width-4 fallback sample=%d I8MM wrapper=%d NEON=%d", i, got.Pix[i], want.Pix[i])
		}
	}
}

func TestConvolveX8I8MMZeroAlloc(t *testing.T) {
	if !cpu.Detected.I8MM {
		t.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	kernel, err := interpKernel(InterpEightTapRegular, 32, 5)
	if err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(50, func() {
		convolveX8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, kernel)
	})
	if allocs != 0 {
		t.Fatalf("convolve X I8MM allocated %v times, want 0", allocs)
	}
}

func BenchmarkConvolveX8I8MM_32(b *testing.B) {
	if !cpu.Detected.I8MM {
		b.Skip("I8MM not detected")
	}
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveX8I8MM(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}

func BenchmarkConvolveX8NEONDirect_32(b *testing.B) {
	dst, ref := benchPlanes(32, 8)
	xk := subpelFilters8[3]
	runConvolveBench(b, 32, 32, func() {
		convolveX8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 32, 32, xk)
	})
}
