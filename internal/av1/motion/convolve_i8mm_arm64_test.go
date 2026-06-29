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
