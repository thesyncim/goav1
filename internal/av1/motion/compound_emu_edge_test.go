// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package motion

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestCompound8XYClampedEmuEdgeMatchesPureGo drives the 8-bit compound
// horizontal-only and vertical-only emu_edge paths over edge-overhanging tap
// windows and asserts bit-identity with the pure-Go clamped references. The
// width/height matrix covers every AV1 block size class from 4 through 128
// (including the width-4 four-tap kernels); the filter/phase matrix covers the
// regular, smooth, sharp and bilinear tables at every nonzero subpel phase.
func TestCompound8XYClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x8ced802d))
	const refW, refH = 37, 29
	ref := frame.Plane{Pix: make([]byte, refW*refH), Stride: refW, Width: refW, Height: refH}
	for y := 0; y < refH; y++ {
		for x := 0; x < refW; x++ {
			v := byte(rng.Intn(256))
			if (x == 0 || x == refW-1) && (y == 0 || y == refH-1) {
				v = byte((x*y + x + y) & 0xff)
			}
			ref.Pix[y*ref.Stride+x] = v
		}
	}

	round0 := compoundRound0(8)
	offsetBits := 8 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	filters := []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
	sizes := []int{4, 8, 16, 32, 64, 128}
	offsets := [][2]int{
		{-3, -2},
		{refW - 2, 5},
		{5, -3},
		{7, refH - 2},
		{-96, refH + 32},
	}
	for _, w := range sizes {
		for _, h := range sizes {
			gotX := make([]uint16, w*h)
			gotY := make([]uint16, w*h)
			wantX := make([]uint16, w*h)
			wantY := make([]uint16, w*h)
			var scratch CompoundConvolveScratch
			for _, filter := range filters {
				for sub := 1; sub <= subpelQ4Mask; sub++ {
					xKernel, err := interpKernel(filter, w, sub)
					if err != nil {
						t.Fatal(err)
					}
					yKernel, err := interpKernel(filter, h, sub)
					if err != nil {
						t.Fatal(err)
					}
					for _, o := range offsets {
						predictInterCompoundRef8ToConvBufXEmuEdge(gotX, ref, o[0], o[1], w, h, xKernel, roundOffset, &scratch.edge)
						predictInterCompoundRef8ToConvBufXPureGo(wantX, ref, o[0], o[1], w, h, xKernel, roundOffset)
						predictInterCompoundRef8ToConvBufYEmuEdge(gotY, ref, o[0], o[1], w, h, yKernel, round0, roundOffset, &scratch.edge)
						predictInterCompoundRef8ToConvBufYPureGo(wantY, ref, o[0], o[1], w, h, yKernel, round0, roundOffset)
						for i := range wantX {
							if gotX[i] != wantX[i] {
								t.Fatalf("X filter=%d size=%dx%d sub=%d off=%v sample=%d emu=%d clamped=%d",
									filter, w, h, sub, o, i, gotX[i], wantX[i])
							}
							if gotY[i] != wantY[i] {
								t.Fatalf("Y filter=%d size=%dx%d sub=%d off=%v sample=%d emu=%d clamped=%d",
									filter, w, h, sub, o, i, gotY[i], wantY[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestCompound8XYClampedEmuEdgeEntryPoint proves the public convbuf entry
// routes edge-overhanging X-only and Y-only 8-bit blocks through the scratch
// emu_edge path byte-identically to the scratchless scalar route. Resident
// cases are included to keep the direct SIMD route covered by the same table.
func TestCompound8XYClampedEmuEdgeEntryPoint(t *testing.T) {
	const refW, refH = 31, 23
	ref := frame.Plane{Pix: make([]byte, refW*refH), Stride: refW, Width: refW, Height: refH}
	fillMotionTestPlane(ref)
	cases := []struct {
		name          string
		refX, refY    int
		width, height int
		subX, subY    int
		filters       InterpFilters
	}{
		{"x resident", 8, 8, 16, 8, 5, 0, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"x left edge", -3, 4, 16, 8, 5, 0, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapRegular}},
		{"x right edge w4", refW - 2, 8, 4, 16, 7, 0, InterpFilters{X: InterpEightTapSmooth, Y: InterpEightTapRegular}},
		{"x vertical edge", 5, -2, 32, 8, 11, 0, InterpFilters{X: InterpBilinear, Y: InterpEightTapRegular}},
		{"y resident", 8, 8, 16, 8, 0, 5, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"y top edge", 5, -3, 16, 8, 0, 11, InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}},
		{"y bottom edge w4", 9, refH - 2, 4, 16, 0, 7, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}},
		{"y horizontal edge", -2, 5, 32, 8, 0, 8, InterpFilters{X: InterpEightTapRegular, Y: InterpBilinear}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got, want CompoundConvBuf
			var scratch CompoundConvolveScratch
			if err := PredictInterCompoundRefToConvBufWithScratch(&got, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters, &scratch); err != nil {
				t.Fatalf("with scratch: %v", err)
			}
			if err := PredictInterCompoundRefToConvBuf(&want, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters); err != nil {
				t.Fatalf("without scratch: %v", err)
			}
			for i := range tc.width * tc.height {
				if got.Data[i] != want.Data[i] {
					t.Fatalf("sample=%d scratch=%d default=%d", i, got.Data[i], want.Data[i])
				}
			}
		})
	}
}

// TestCompound8XYClampedEmuEdgeZeroAlloc proves the lowbd directional emu_edge
// paths do not allocate when the caller supplies the halo scratch.
func TestCompound8XYClampedEmuEdgeZeroAlloc(t *testing.T) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*refH), Stride: refW, Width: refW, Height: refH}
	fillMotionTestPlane(ref)
	round0 := compoundRound0(8)
	offsetBits := 8 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[6]
	out := make([]uint16, 64*64)
	var scratch CompoundConvolveScratch
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRef8ToConvBufXEmuEdge(out, ref, -3, -3, 64, 64, kernel, roundOffset, &scratch.edge)
	}); a != 0 {
		t.Fatalf("emu_edge compound X8 allocated %v times, want 0", a)
	}
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRef8ToConvBufYEmuEdge(out, ref, -3, -3, 64, 64, kernel, round0, roundOffset, &scratch.edge)
	}); a != 0 {
		t.Fatalf("emu_edge compound Y8 allocated %v times, want 0", a)
	}
}

// TestCompoundHighBDXYClampedEmuEdgeMatchesPureGo drives the HBD compound
// horizontal-only and vertical-only emu_edge paths (dav1d src/recon_tmpl.c
// mc(), src/mc_tmpl.c emu_edge_c) over tap windows that overhang the reference
// plane on every side and asserts bit-identity with the pure-Go
// per-tap-clamping references at bit depths 10 and 12, across every block
// width class the resident kernels dispatch on (including the width-2 chroma
// and width-4 four-tap shapes).
func TestCompoundHighBDXYClampedEmuEdgeMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x16bced2d))
	const refW, refH = 24, 20
	widths := []int{2, 4, 8, 12, 16, 32, 64, 128}
	heights := []int{2, 4, 8, 16, 32, 128}
	for _, bitDepth := range []uint8{10, 12} {
		max := uint16((1 << int(bitDepth)) - 1)
		round0 := compoundRound0(bitDepth)
		offsetBits := int(bitDepth) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
		for y := 0; y < refH; y++ {
			for x := 0; x < refW; x++ {
				v := uint16(rng.Intn(int(max) + 1))
				if (x == 0 || x == refW-1) && (y == 0 || y == refH-1) {
					if (x+y)&1 == 0 {
						v = 0
					} else {
						v = max
					}
				}
				storeHighBDSample(ref, x, y, v)
			}
		}
		for _, w := range widths {
			// Four-tap kernels for the width<=4 dispatch class, full
			// eight-tap otherwise (mirrors interpKernel's selection).
			xKernel := subpelFilters8[6]
			yKernel := subpelFilters8[9]
			if w <= 4 {
				xKernel = subpelFilters4[6]
				yKernel = subpelFilters4[9]
			}
			for _, h := range heights {
				offs := [][2]int{
					{-w, -h}, {-3, -3}, {2, -5}, {refW - 2, -1},
					{refW - 1, 5}, {refW, refH}, {5, refH - 2},
					{-4, refH - 3}, {-6, 4}, {-200, -200}, {refW + 50, 3},
				}
				for _, o := range offs {
					gotX := make([]uint16, w*h)
					gotY := make([]uint16, w*h)
					wantX := make([]uint16, w*h)
					wantY := make([]uint16, w*h)
					var edge emuEdge16Buf
					// Poison the caller-owned edge window to prove every
					// sample read by the resident kernel was materialized.
					for i := range edge {
						edge[i] = 0xa5
					}
					predictInterCompoundRefHighBDToConvBufXEmuEdge(gotX, ref, o[0], o[1], w, h, xKernel, round0, roundOffset, &edge)
					predictInterCompoundRefHighBDToConvBufXClamped(wantX, ref, o[0], o[1], w, h, xKernel, round0, roundOffset)
					for i := range edge {
						edge[i] = 0x5a
					}
					predictInterCompoundRefHighBDToConvBufYEmuEdge(gotY, ref, o[0], o[1], w, h, yKernel, round0, roundOffset, &edge)
					predictInterCompoundRefHighBDToConvBufYClamped(wantY, ref, o[0], o[1], w, h, yKernel, round0, roundOffset)
					for i := range wantX {
						if gotX[i] != wantX[i] {
							t.Fatalf("X bd=%d %dx%d off=%v sample=%d emu=%d clamped=%d",
								bitDepth, w, h, o, i, gotX[i], wantX[i])
						}
						if gotY[i] != wantY[i] {
							t.Fatalf("Y bd=%d %dx%d off=%v sample=%d emu=%d clamped=%d",
								bitDepth, w, h, o, i, gotY[i], wantY[i])
						}
					}
				}
			}
		}
	}
}

// TestCompoundHighBDXYClampedEmuEdgeEntryPoint proves the public convbuf entry
// routes edge-overhanging X-only and Y-only HBD blocks through the emu_edge
// path when scratch is provided, byte-identically to the scratchless (scalar
// clamped) route.
func TestCompoundHighBDXYClampedEmuEdgeEntryPoint(t *testing.T) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	filters := InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}
	for _, bitDepth := range []uint8{10, 12} {
		for _, sub := range [][2]int{{5, 0}, {0, 11}} {
			for _, o := range [][2]int{{-3, 4}, {refW - 2, 4}, {4, -5}, {4, refH - 2}} {
				var got, want CompoundConvBuf
				var scratch CompoundConvolveScratch
				if err := PredictInterCompoundRefToConvBufWithScratch(&got, ref, 2, bitDepth, o[0], o[1], 16, 8, sub[0], sub[1], filters, &scratch); err != nil {
					t.Fatalf("with scratch: %v", err)
				}
				if err := PredictInterCompoundRefToConvBuf(&want, ref, 2, bitDepth, o[0], o[1], 16, 8, sub[0], sub[1], filters); err != nil {
					t.Fatalf("without scratch: %v", err)
				}
				for i := range 16 * 8 {
					if got.Data[i] != want.Data[i] {
						t.Fatalf("bd=%d sub=%v off=%v sample=%d scratch=%d default=%d",
							bitDepth, sub, o, i, got.Data[i], want.Data[i])
					}
				}
			}
		}
	}
}

// TestCompoundHighBDXYClampedEmuEdgeZeroAlloc proves the emu_edge X/Y paths do
// not allocate when the caller supplies the halo scratch.
func TestCompoundHighBDXYClampedEmuEdgeZeroAlloc(t *testing.T) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[6]
	out := make([]uint16, 64*64)
	var edge emuEdge16Buf
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBufXEmuEdge(out, ref, -3, -3, 64, 64, kernel, round0, roundOffset, &edge)
	}); a != 0 {
		t.Fatalf("emu_edge compound X clamped allocated %v times, want 0", a)
	}
	if a := testing.AllocsPerRun(20, func() {
		predictInterCompoundRefHighBDToConvBufYEmuEdge(out, ref, -3, -3, 64, 64, kernel, round0, roundOffset, &edge)
	}); a != 0 {
		t.Fatalf("emu_edge compound Y clamped allocated %v times, want 0", a)
	}
}

// BenchmarkCompoundHighBDXClampedEmuEdge compares the emu_edge horizontal-only
// path against the scalar per-tap-clamping reference on an edge-overhanging
// block.
func BenchmarkCompoundHighBDXClampedEmuEdge(b *testing.B) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[6]
	for _, w := range []int{8, 16, 32, 64} {
		out := make([]uint16, w*w)
		var edge emuEdge16Buf
		b.Run("emu_"+strconv.Itoa(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBufXEmuEdge(out, ref, -3, -3, w, w, kernel, round0, roundOffset, &edge)
			}
		})
		b.Run("clamped_"+strconv.Itoa(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBufXClamped(out, ref, -3, -3, w, w, kernel, round0, roundOffset)
			}
		})
	}
}

// BenchmarkCompoundHighBDYClampedEmuEdge is the vertical-only sibling of
// BenchmarkCompoundHighBDXClampedEmuEdge.
func BenchmarkCompoundHighBDYClampedEmuEdge(b *testing.B) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*2*refH), Stride: refW * 2, Width: refW, Height: refH}
	fillHighBDMotionTestPlane(ref, 0x3ff)
	round0 := compoundRound0(10)
	offsetBits := 10 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[9]
	for _, w := range []int{8, 16, 32, 64} {
		out := make([]uint16, w*w)
		var edge emuEdge16Buf
		b.Run("emu_"+strconv.Itoa(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBufYEmuEdge(out, ref, -3, -3, w, w, kernel, round0, roundOffset, &edge)
			}
		})
		b.Run("clamped_"+strconv.Itoa(w), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				predictInterCompoundRefHighBDToConvBufYClamped(out, ref, -3, -3, w, w, kernel, round0, roundOffset)
			}
		})
	}
}

// BenchmarkCompound8XClampedEmuEdge compares the scratch emu_edge
// horizontal-only path against the scalar per-tap-clamping reference on an
// edge-overhanging 8-bit compound block.
func BenchmarkCompound8XClampedEmuEdge(b *testing.B) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*refH), Stride: refW, Width: refW, Height: refH}
	fillMotionTestPlane(ref)
	round0 := compoundRound0(8)
	offsetBits := 8 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[6]
	for _, w := range []int{4, 8, 16, 32, 64} {
		out := make([]uint16, w*w)
		var scratch CompoundConvolveScratch
		b.Run("emu_"+strconv.Itoa(w), func(b *testing.B) {
			for b.Loop() {
				predictInterCompoundRef8ToConvBufXEmuEdge(out, ref, -3, -3, w, w, kernel, roundOffset, &scratch.edge)
			}
		})
		b.Run("clamped_"+strconv.Itoa(w), func(b *testing.B) {
			for b.Loop() {
				predictInterCompoundRef8ToConvBufXPureGo(out, ref, -3, -3, w, w, kernel, roundOffset)
			}
		})
	}
}

// BenchmarkCompound8YClampedEmuEdge is the vertical-only sibling of
// BenchmarkCompound8XClampedEmuEdge.
func BenchmarkCompound8YClampedEmuEdge(b *testing.B) {
	const refW, refH = 24, 20
	ref := frame.Plane{Pix: make([]byte, refW*refH), Stride: refW, Width: refW, Height: refH}
	fillMotionTestPlane(ref)
	round0 := compoundRound0(8)
	offsetBits := 8 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	kernel := subpelFilters8[9]
	for _, w := range []int{4, 8, 16, 32, 64} {
		out := make([]uint16, w*w)
		var scratch CompoundConvolveScratch
		b.Run("emu_"+strconv.Itoa(w), func(b *testing.B) {
			for b.Loop() {
				predictInterCompoundRef8ToConvBufYEmuEdge(out, ref, -3, -3, w, w, kernel, round0, roundOffset, &scratch.edge)
			}
		})
		b.Run("clamped_"+strconv.Itoa(w), func(b *testing.B) {
			for b.Loop() {
				predictInterCompoundRef8ToConvBufYPureGo(out, ref, -3, -3, w, w, kernel, round0, roundOffset)
			}
		})
	}
}
