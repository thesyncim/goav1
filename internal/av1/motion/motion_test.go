package motion

import (
	"bytes"
	"compress/zlib"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"io"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"testing"
	"unsafe"
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
// cases are included to keep the direct optimized route covered by the same table.
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
			if err := PredictInterCompoundRefToConvBuf(&got, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.width, tc.height, tc.subX, tc.subY, tc.filters, &scratch); err != nil {
				t.Fatalf("with scratch: %v", err)
			}
			if err := PredictInterCompoundRefToConvBuf(&want, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.width, tc.height, tc.subX, tc.subY, tc.filters, nil); err != nil {
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
				if err := PredictInterCompoundRefToConvBuf(&got, ref, 2, bitDepth, o[0], o[1], 16, 8, 16, 8, sub[0], sub[1], filters, &scratch); err != nil {
					t.Fatalf("with scratch: %v", err)
				}
				if err := PredictInterCompoundRefToConvBuf(&want, ref, 2, bitDepth, o[0], o[1], 16, 8, 16, 8, sub[0], sub[1], filters, nil); err != nil {
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

// TestCompoundRound0 verifies the round_0 selection ported from libaom's
// get_conv_params_no_round() (av1/common/convolve.c). round_0 = ROUND0_BITS
// (3) for bit depths <= 10. At bd == 12 the highbd intbufrange adjustment
// (intbufrange = bd + FILTER_BITS - round_0 + 2 > 16) raises round_0 to 5.
func TestCompoundRound0(t *testing.T) {
	cases := []struct {
		bitDepth uint8
		want     int
	}{
		// intbufrange = bd + 7 - 3 + 2 = bd + 6.
		{8, 3},  // intbufrange = 14, not > 16.
		{10, 3}, // intbufrange = 16, not > 16.
		{12, 5}, // intbufrange = 18 > 16 => round0 += 2.
	}
	for _, tc := range cases {
		if got := compoundRound0(tc.bitDepth); got != tc.want {
			t.Errorf("compoundRound0(%d)=%d want %d", tc.bitDepth, got, tc.want)
		}
	}
}

// TestRoundPowerOfTwoCompound verifies the rounding helper used by the d16
// mask subsampling path (matches libaom's ROUND_POWER_OF_TWO macro).
func TestRoundPowerOfTwoCompound(t *testing.T) {
	cases := []struct {
		value, bits, want int
	}{
		{0, 2, 0},
		{1, 2, 0},    // (1 + 2) >> 2 = 0
		{2, 2, 1},    // (2 + 2) >> 2 = 1
		{100, 2, 25}, // (100 + 2) >> 2 = 25
		{7, 1, 4},    // (7 + 1) >> 1 = 4
	}
	for _, tc := range cases {
		if got := roundPowerOfTwoCompound(tc.value, tc.bits); got != tc.want {
			t.Errorf("roundPowerOfTwoCompound(%d,%d)=%d want %d", tc.value, tc.bits, got, tc.want)
		}
	}
}

// TestCompoundMaskSample verifies the (possibly subsampled) A64 mask read that
// mirrors aom_*_blend_a64_d16_mask_c subw/subh handling.
func TestCompoundMaskSample(t *testing.T) {
	// 4x4 mask. Lay out so subsampling reductions are easy to verify.
	stride := 4
	mask := []byte{
		10, 21, 30, 40,
		20, 30, 50, 60,
		1, 2, 3, 4,
		5, 6, 7, 8,
	}

	// No subsampling: direct read.
	if v, ok := compoundMaskSample(mask, stride, 1, 2, false, false); !ok || v != 50 {
		t.Errorf("noSub mask[1][2]=%d ok=%v want 50", v, ok)
	}
	// subX only: (a + b + 1) >> 1 over horizontally adjacent pair at row*stride+2*col.
	// row=0 col=0 -> (10 + 21 + 1) >> 1 = 16.
	if v, ok := compoundMaskSample(mask, stride, 0, 0, true, false); !ok || v != 16 {
		t.Errorf("subX mask[0][0]=%d ok=%v want 16", v, ok)
	}
	// subY only: (top + bottom + 1) >> 1 over vertically adjacent rows.
	// row=0 col=0 -> (mask[0][0]=10 + mask[1][0]=20 + 1) >> 1 = 15.
	if v, ok := compoundMaskSample(mask, stride, 0, 0, false, true); !ok || v != 15 {
		t.Errorf("subY mask[0][0]=%d ok=%v want 15", v, ok)
	}
	// subX && subY: ROUND_POWER_OF_TWO(sum of 2x2, 2).
	// row=0 col=0 -> sum(10,21,20,30)=81 -> (81 + 2) >> 2 = 20.
	if v, ok := compoundMaskSample(mask, stride, 0, 0, true, true); !ok || v != 20 {
		t.Errorf("subXY mask[0][0]=%d ok=%v want 20", v, ok)
	}
	// Out-of-bounds index returns ok=false.
	if _, ok := compoundMaskSample(mask, stride, 3, 3, false, true); ok {
		t.Errorf("subY out-of-range expected ok=false")
	}
}

// flatCompoundBuf builds a CompoundConvBuf of the given size filled with a
// single CONV_BUF value.
func flatCompoundBuf(width, height int, value uint16) *CompoundConvBuf {
	buf := &CompoundConvBuf{Width: uint8(width), Height: uint8(height)}
	for i := 0; i < width*height; i++ {
		buf.Data[i] = value
	}
	return buf
}

// convValForPixel returns the CONV_BUF d16 value whose equal-weight blend
// (fwd==bck==8) inverts exactly to pixel P at the given bit depth. With
// fwd==bck==8 the BlendCompoundAvg arithmetic is:
//
//	tmp = v*8 + v*8 = v*16; tmp >>= 4 (=> v); tmp -= roundOffset
//	out = ROUND_POWER_OF_TWO(v - roundOffset, roundBits)
//
// so v = (P << roundBits) + roundOffset yields out == P exactly.
func convValForPixel(p int, bitDepth uint8) uint16 {
	round0 := compoundRound0(bitDepth)
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits := 2*filterBits - round0 - compoundRound1Bits
	return uint16((p << roundBits) + roundOffset)
}

func readPixel8(dst frame.Plane, x, y int) int { return int(dst.Pix[y*dst.Stride+x]) }
func readPixel16(dst frame.Plane, x, y int) int {
	off := y*dst.Stride + x*2
	return int(dst.Pix[off]) | int(dst.Pix[off+1])<<8
}

// TestBlendCompoundAvg verifies the distance-weighted compound average
// (av1_dist_wtd_convolve_* do_average blend). With equal forward/backward
// offsets and equal predictors, the output must invert exactly to the seeded
// pixel value at each bit depth.
func TestBlendCompoundAvg(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		bps      int
		pixel    int
	}{
		{"bd8", 8, 1, 100},
		{"bd10", 10, 2, 600},
		{"bd12", 12, 2, 2500},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 4, 4
			v := convValForPixel(tc.pixel, tc.bitDepth)
			buf0 := flatCompoundBuf(w, h, v)
			buf1 := flatCompoundBuf(w, h, v)
			dst := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			// fwdOffset + bckOffset must sum to 16 (DIST_PRECISION); equal split.
			if err := BlendCompoundAvg(dst, buf0, buf1, tc.bps, tc.bitDepth, 0, 0, w, h, 8, 8); err != nil {
				t.Fatalf("BlendCompoundAvg: %v", err)
			}
			for y := range h {
				for x := range w {
					var got int
					if tc.bps == 1 {
						got = readPixel8(dst, x, y)
					} else {
						got = readPixel16(dst, x, y)
					}
					if got != tc.pixel {
						t.Fatalf("pixel(%d,%d)=%d want %d", x, y, got, tc.pixel)
					}
				}
			}
		})
	}
}

func TestBlendCompoundAvg8ClippedWidthsFallback(t *testing.T) {
	const h = 3
	// Clipped inter targets can leave widths that are divisible by 4 but not
	// 8. The pinned libaom av1_dist_wtd_convolve_2d_copy_c iterates x < w, so
	// these rows must not inherit the SIMD kernel's 8-pixel tail assumptions.
	for _, width := range []int{12, 20, 28, 36} {
		t.Run(fmt.Sprintf("width-%d", width), func(t *testing.T) {
			var buf0, buf1 CompoundConvBuf
			buf0.Width, buf1.Width = uint8(width), uint8(width)
			buf0.Height, buf1.Height = h, h
			for i := range width * h {
				buf0.Data[i] = uint16(6144 + 16*((i*17+3)%256))
				buf1.Data[i] = uint16(6144 + 16*((i*29+111)%256))
			}

			const dstX, dstY = 3, 1
			dstWidth := width + 8
			stride := dstWidth + 5
			dstHeight := h + 2
			pix := make([]byte, stride*dstHeight)
			for i := range pix {
				pix[i] = 0xa5
			}
			unchanged := frame.Plane{Pix: slices.Clone(pix), Stride: stride, Width: dstWidth, Height: dstHeight}
			want := frame.Plane{Pix: slices.Clone(pix), Stride: stride, Width: dstWidth, Height: dstHeight}
			blendCompoundAvg8PureGo(want, buf0.Data[:width*h], buf1.Data[:width*h], dstX, dstY, width, h, 8, 8, 6144, 4)
			got := frame.Plane{Pix: slices.Clone(pix), Stride: stride, Width: dstWidth, Height: dstHeight}
			if err := BlendCompoundAvg(got, &buf0, &buf1, 1, 8, dstX, dstY, width, h, 8, 8); err != nil {
				t.Fatalf("BlendCompoundAvg: %v", err)
			}
			if !bytes.Equal(got.Pix, want.Pix) {
				for i := range got.Pix {
					if got.Pix[i] != want.Pix[i] {
						t.Fatalf("BlendCompoundAvg byte %d=%d want %d (pure-Go oracle plus row guards)", i, got.Pix[i], want.Pix[i])
					}
				}
			}

			fastDst := frame.Plane{Pix: slices.Clone(pix), Stride: stride, Width: dstWidth, Height: dstHeight}
			if BlendAvg8(&fastDst, &buf0, &buf1, dstX, dstY, width, h) {
				if !bytes.Equal(fastDst.Pix, want.Pix) {
					t.Fatal("BlendAvg8 differs from pure-Go oracle or changed destination guards")
				}
			} else if !bytes.Equal(fastDst.Pix, unchanged.Pix) {
				t.Fatal("declined BlendAvg8 changed destination pixels or row guards")
			}
		})
	}
}

// TestBlendCompoundAvgErrors verifies the validation guards reject mismatched
// buffer dimensions and out-of-range destination regions.
func TestBlendCompoundAvgErrors(t *testing.T) {
	buf0 := flatCompoundBuf(4, 4, 1000)
	buf1 := flatCompoundBuf(4, 4, 1000)
	dst := frame.Plane{Pix: make([]byte, 16), Stride: 4, Width: 4, Height: 4}

	if err := BlendCompoundAvg(dst, nil, buf1, 1, 8, 0, 0, 4, 4, 8, 8); err != ErrInvalidMotion {
		t.Errorf("nil buf0 err=%v want ErrInvalidMotion", err)
	}
	mismatched := flatCompoundBuf(2, 2, 1000)
	if err := BlendCompoundAvg(dst, mismatched, buf1, 1, 8, 0, 0, 4, 4, 8, 8); err != ErrInvalidMotion {
		t.Errorf("dim mismatch err=%v want ErrInvalidMotion", err)
	}
	if err := BlendCompoundAvg(dst, buf0, buf1, 1, 8, 2, 2, 4, 4, 8, 8); err != ErrInvalidMotion {
		t.Errorf("region overflow err=%v want ErrInvalidMotion", err)
	}
}

// TestBlendCompoundMaskD16 verifies the soft-mask d16 blend
// (aom_blend_a64_d16_mask_c). A full mask (m=64) selects src0 entirely; an
// empty mask (m=0) selects src1 entirely. With the CONV_BUF value chosen to
// invert to a known pixel, the selected predictor's pixel must appear.
func TestBlendCompoundMaskD16(t *testing.T) {
	const w, h = 4, 4
	const bd = uint8(8)
	pixel := 100
	v := convValForPixel(pixel, bd)
	buf0 := flatCompoundBuf(w, h, v)
	buf1 := flatCompoundBuf(w, h, 0)

	// Full mask selects src0.
	maskFull := make([]byte, w*h)
	for i := range maskFull {
		maskFull[i] = compoundBlendA64MaxAlpha
	}
	dst := frame.Plane{Pix: make([]byte, w*h), Stride: w, Width: w, Height: h}
	if err := BlendCompoundMaskD16(dst, buf0, buf1, 1, bd, 0, 0, w, h, maskFull, w, false, false); err != nil {
		t.Fatalf("BlendCompoundMaskD16 full: %v", err)
	}
	if got := readPixel8(dst, 1, 1); got != pixel {
		t.Errorf("full-mask pixel=%d want %d", got, pixel)
	}

	// Empty mask selects src1; swap which buffer carries v.
	buf0b := flatCompoundBuf(w, h, 0)
	buf1b := flatCompoundBuf(w, h, v)
	maskEmpty := make([]byte, w*h) // all zero
	dst2 := frame.Plane{Pix: make([]byte, w*h), Stride: w, Width: w, Height: h}
	if err := BlendCompoundMaskD16(dst2, buf0b, buf1b, 1, bd, 0, 0, w, h, maskEmpty, w, false, false); err != nil {
		t.Fatalf("BlendCompoundMaskD16 empty: %v", err)
	}
	if got := readPixel8(dst2, 1, 1); got != pixel {
		t.Errorf("empty-mask pixel=%d want %d", got, pixel)
	}
}

// TestBuildDiffWtdMaskD16 verifies the difference-weighted mask construction
// (av1_build_compound_diffwtd_mask_d16_c). For bd=8 the round shift is
// 2*7 - 3 - 7 + 0 = 4, then m = 38 + ROUND_POWER_OF_TWO(|s0-s1|, 4) / 16,
// clamped to [0, 64].
func TestBuildDiffWtdMaskD16(t *testing.T) {
	const w, h = 2, 2
	const bd = uint8(8)
	maskStride := w

	cases := []struct {
		name   string
		s0, s1 uint16
		invert bool
		want   byte
	}{
		// diff=0 -> rpot(0,4)=0 -> m=38.
		{"equal", 1000, 1000, false, 38},
		// diff=160 -> rpot(160,4)=10 -> m=38+0=38 (10/16=0).
		{"smalldiff", 1160, 1000, false, 38},
		// diff=16000 -> rpot=1000 -> m=38+62=100 clamp 64.
		{"saturate", 17000, 1000, false, 64},
		// invert of equal: 64-38 = 26.
		{"equal-inverted", 1000, 1000, true, 26},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf0 := flatCompoundBuf(w, h, tc.s0)
			buf1 := flatCompoundBuf(w, h, tc.s1)
			mask := make([]byte, w*h)
			if err := BuildDiffWtdMaskD16(mask, maskStride, buf0, buf1, bd, w, h, tc.invert); err != nil {
				t.Fatalf("BuildDiffWtdMaskD16: %v", err)
			}
			for i, m := range mask {
				if m != tc.want {
					t.Fatalf("mask[%d]=%d want %d", i, m, tc.want)
				}
			}
		})
	}
}

// TestBuildDiffWtdMaskD16Errors verifies the validation guards reject
// undersized mask buffers and mismatched predictor dimensions.
func TestBuildDiffWtdMaskD16Errors(t *testing.T) {
	buf0 := flatCompoundBuf(4, 4, 1000)
	buf1 := flatCompoundBuf(4, 4, 2000)
	short := make([]byte, 4) // too small for 4x4
	if err := BuildDiffWtdMaskD16(short, 4, buf0, buf1, 8, 4, 4, false); err != ErrInvalidMotion {
		t.Errorf("short mask err=%v want ErrInvalidMotion", err)
	}
	mismatched := flatCompoundBuf(2, 2, 1000)
	mask := make([]byte, 16)
	if err := BuildDiffWtdMaskD16(mask, 4, mismatched, buf1, 8, 4, 4, false); err != ErrInvalidMotion {
		t.Errorf("dim mismatch err=%v want ErrInvalidMotion", err)
	}
}

// TestPredictInterCompoundRefToConvBufCopyRoundTrip exercises the integer
// (subX==0, subY==0) compound convolve path (av1_dist_wtd_convolve_2d_copy):
// each CONV_BUF sample is (src << bits) + roundOffset. Feeding two identical
// copy predictors through the equal-weight BlendCompoundAvg must reproduce the
// source pixels exactly, validating that the convolve-to-convbuf scaling and
// the blend-back rounding are mutually consistent across bit depths. This is
// the same precision contract libaom relies on: keep the predictor un-rounded
// at d16 precision, round once after combining.
func TestPredictInterCompoundRefToConvBufCopyRoundTrip(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		bps      int
	}{
		{"bd8", 8, 1},
		{"bd10", 10, 2},
		{"bd12", 12, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 8, 8
			// Build a ref plane with a per-pixel gradient (kept within the bit
			// depth range). loadSample*Clamped clamps to plane bounds, and the
			// integer-MV (subX=subY=0) copy reads exactly (refX+x, refY+y), so
			// place the block at (0,0) within an exactly-sized plane.
			ref := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			want := make([]int, w*h)
			for y := range h {
				for x := range w {
					p := (x*7 + y*11) & ((1 << tc.bitDepth) - 1)
					want[y*w+x] = p
					if tc.bps == 1 {
						ref.Pix[y*ref.Stride+x] = byte(p)
					} else {
						off := y*ref.Stride + x*2
						ref.Pix[off] = byte(p)
						ref.Pix[off+1] = byte(p >> 8)
					}
				}
			}

			var buf0, buf1 CompoundConvBuf
			if err := PredictInterCompoundRefToConvBuf(&buf0, ref, tc.bps, tc.bitDepth, 0, 0, w, h, w, h, 0, 0, RegularFilters, nil); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf buf0: %v", err)
			}
			if err := PredictInterCompoundRefToConvBuf(&buf1, ref, tc.bps, tc.bitDepth, 0, 0, w, h, w, h, 0, 0, RegularFilters, nil); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf buf1: %v", err)
			}

			dst := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			if err := BlendCompoundAvg(dst, &buf0, &buf1, tc.bps, tc.bitDepth, 0, 0, w, h, 8, 8); err != nil {
				t.Fatalf("BlendCompoundAvg: %v", err)
			}
			for y := range h {
				for x := range w {
					var got int
					if tc.bps == 1 {
						got = readPixel8(dst, x, y)
					} else {
						got = readPixel16(dst, x, y)
					}
					if got != want[y*w+x] {
						t.Fatalf("pixel(%d,%d)=%d want %d", x, y, got, want[y*w+x])
					}
				}
			}
		})
	}
}

func TestPredictInterCompoundRefToConvBuf8OptimizedMatchesReference(t *testing.T) {
	const (
		refW   = 23
		refH   = 19
		stride = 32
	)
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	for y := range refH {
		for x := range refW {
			ref.Pix[y*stride+x] = byte((x*37 + y*53 + x*y*3 + 11) & 0xff)
		}
	}

	cases := []struct {
		name          string
		refX, refY    int
		width, height int
		subX, subY    int
		filters       InterpFilters
	}{
		{"copy interior", 3, 4, 16, 8, 0, 0, RegularFilters},
		{"copy width4 interior", 7, 6, 4, 8, 0, 0, RegularFilters},
		{"copy edge", -2, 16, 16, 8, 0, 0, RegularFilters},
		{"x regular interior", 2, 4, 16, 8, 5, 0, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"x width4 interior", 7, 6, 4, 8, 5, 0, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"x smooth right edge", 13, 5, 8, 8, 7, 0, InterpFilters{X: InterpEightTapSmooth, Y: InterpEightTapRegular}},
		{"y sharp top edge", 4, -3, 16, 8, 0, 11, InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}},
		{"y width4 interior", 7, 6, 4, 8, 0, 11, InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}},
		{"2d regular interior", 4, 4, 16, 16, 3, 9, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}},
		{"2d width4 interior", 7, 6, 4, 8, 3, 9, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}},
		{"2d fourtap left edge", -2, 3, 4, 16, 4, 12, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"2d bilinear bottom edge", 9, 14, 8, 4, 8, 8, InterpFilters{X: InterpBilinear, Y: InterpBilinear}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got, gotScratch CompoundConvBuf
			if err := PredictInterCompoundRefToConvBuf(&got, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.width, tc.height, tc.subX, tc.subY, tc.filters, nil); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf: %v", err)
			}
			var scratch CompoundConvolveScratch
			if err := PredictInterCompoundRefToConvBuf(&gotScratch, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.width, tc.height, tc.subX, tc.subY, tc.filters, &scratch); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf scratch: %v", err)
			}
			want, err := referenceCompoundConvBuf8(ref, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters)
			if err != nil {
				t.Fatalf("reference: %v", err)
			}
			if int(got.Width) != tc.width || int(got.Height) != tc.height || int(gotScratch.Width) != tc.width || int(gotScratch.Height) != tc.height {
				t.Fatalf("dims default=%dx%d scratch=%dx%d want %dx%d", got.Width, got.Height, gotScratch.Width, gotScratch.Height, tc.width, tc.height)
			}
			for i := range tc.width * tc.height {
				if got.Data[i] != want[i] {
					t.Fatalf("convbuf default[%d]=%d want %d", i, got.Data[i], want[i])
				}
				if gotScratch.Data[i] != want[i] {
					t.Fatalf("convbuf scratch[%d]=%d want %d", i, gotScratch.Data[i], want[i])
				}
			}
		})
	}
}

func TestPredictInterCompoundRefToConvBufHighBDOptimizedMatchesClamped(t *testing.T) {
	const (
		refW   = 37
		refH   = 31
		stride = 96
	)
	cases := []struct {
		name          string
		refX, refY    int
		width, height int
		subX, subY    int
		filters       InterpFilters
	}{
		{"copy interior", 5, 6, 16, 8, 0, 0, RegularFilters},
		{"copy edge", -1, 27, 16, 4, 0, 0, RegularFilters},
		{"x regular interior", 5, 6, 16, 8, 5, 0, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"x smooth edge", 28, 8, 8, 8, 7, 0, InterpFilters{X: InterpEightTapSmooth, Y: InterpEightTapRegular}},
		{"y sharp interior", 6, 6, 16, 8, 0, 11, InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}},
		{"y bilinear edge", 4, -2, 16, 8, 0, 8, InterpFilters{X: InterpEightTapRegular, Y: InterpBilinear}},
		{"2d regular interior", 6, 6, 16, 16, 3, 9, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}},
		{"2d fourtap edge", -2, 3, 4, 16, 4, 12, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
	}
	for _, bitDepth := range []uint8{10, 12} {
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		mask := (1 << bitDepth) - 1
		for y := range refH {
			for x := range refW {
				p := (x*43 + y*59 + x*y*5 + 17) & mask
				off := y*stride + x*2
				ref.Pix[off] = byte(p)
				ref.Pix[off+1] = byte(p >> 8)
			}
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				var got, gotScratch, want CompoundConvBuf
				if err := PredictInterCompoundRefToConvBuf(&got, ref, 2, bitDepth, tc.refX, tc.refY, tc.width, tc.height, tc.width, tc.height, tc.subX, tc.subY, tc.filters, nil); err != nil {
					t.Fatalf("PredictInterCompoundRefToConvBuf: %v", err)
				}
				var scratch CompoundConvolveScratch
				if err := PredictInterCompoundRefToConvBuf(&gotScratch, ref, 2, bitDepth, tc.refX, tc.refY, tc.width, tc.height, tc.width, tc.height, tc.subX, tc.subY, tc.filters, &scratch); err != nil {
					t.Fatalf("PredictInterCompoundRefToConvBuf scratch: %v", err)
				}
				wantOut, ok := compoundConvBufView(&want, tc.width, tc.height)
				if !ok {
					t.Fatal("invalid reference convbuf dims")
				}
				xKernel, err := interpKernel(tc.filters.X, tc.width, tc.subX)
				if err != nil {
					t.Fatalf("x kernel: %v", err)
				}
				yKernel, err := interpKernel(tc.filters.Y, tc.height, tc.subY)
				if err != nil {
					t.Fatalf("y kernel: %v", err)
				}
				round0 := compoundRound0(bitDepth)
				offsetBits := int(bitDepth) + 2*filterBits - round0
				roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
				switch {
				case tc.subX != 0 && tc.subY != 0:
					var im compoundIM
					predictInterCompoundRefHighBDToConvBuf2DClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, xKernel, yKernel, round0, offsetBits, int(bitDepth), &im)
				case tc.subX != 0:
					predictInterCompoundRefHighBDToConvBufXClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, xKernel, round0, roundOffset)
				case tc.subY != 0:
					predictInterCompoundRefHighBDToConvBufYClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, yKernel, round0, roundOffset)
				default:
					predictInterCompoundRefHighBDToConvBufCopyClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, round0, roundOffset)
				}
				if int(got.Width) != tc.width || int(got.Height) != tc.height || int(gotScratch.Width) != tc.width || int(gotScratch.Height) != tc.height {
					t.Fatalf("dims default=%dx%d scratch=%dx%d want %dx%d", got.Width, got.Height, gotScratch.Width, gotScratch.Height, tc.width, tc.height)
				}
				for i := range tc.width * tc.height {
					if got.Data[i] != want.Data[i] {
						t.Fatalf("convbuf default[%d]=%d want %d", i, got.Data[i], want.Data[i])
					}
					if gotScratch.Data[i] != want.Data[i] {
						t.Fatalf("convbuf scratch[%d]=%d want %d", i, gotScratch.Data[i], want.Data[i])
					}
				}
			})
		}
	}
}

func referenceCompoundConvBuf8(ref frame.Plane, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) ([]uint16, error) {
	xKernel, err := interpKernel(filters.X, width, subX)
	if err != nil {
		return nil, err
	}
	yKernel, err := interpKernel(filters.Y, height, subY)
	if err != nil {
		return nil, err
	}
	out := make([]uint16, width*height)
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	round0 := compoundRound0(8)
	offsetBits := 8 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	load := func(x, y int) int {
		return int(loadSample8Clamped(ref, x, y))
	}
	switch {
	case subX != 0 && subY != 0:
		const imStride = maxBlockSize
		var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
		imH := height + filterTaps - 1
		for y := range imH {
			for x := range width {
				sum := 1 << (8 + filterBits - 1)
				for k := range filterTaps {
					sum += int(xKernel[k]) * load(refX+x-foX+k, refY-foY+y)
				}
				im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0))
			}
		}
		for y := range height {
			for x := range width {
				sum := 1 << offsetBits
				for k := range filterTaps {
					sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
				}
				out[y*width+x] = uint16(roundPowerOfTwo(sum, compoundRound1Bits))
			}
		}
	case subX != 0:
		bits := filterBits - compoundRound1Bits
		for y := range height {
			for x := range width {
				res := 0
				for k := range filterTaps {
					res += int(xKernel[k]) * load(refX+x-foX+k, refY+y)
				}
				res = (1 << bits) * roundPowerOfTwo(res, round0)
				res += roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	case subY != 0:
		bits := filterBits - round0
		for y := range height {
			for x := range width {
				res := 0
				for k := range filterTaps {
					res += int(yKernel[k]) * load(refX+x, refY+y-foY+k)
				}
				res *= 1 << bits
				res = roundPowerOfTwo(res, compoundRound1Bits) + roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	default:
		bits := 2*filterBits - compoundRound1Bits - round0
		for y := range height {
			for x := range width {
				res := load(refX+x, refY+y) << bits
				res += roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	}
	return out, nil
}

func TestPredictScaledCompoundRefToConvBufIdentityMatchesUnscaled(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		bps      int
	}{
		{"bd8", 8, 1},
		{"bd10", 10, 2},
		{"bd12", 12, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 16, 16
			ref := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			mask := (1 << tc.bitDepth) - 1
			for y := range h {
				for x := range w {
					p := (x*17 + y*29 + 11) & mask
					if tc.bps == 1 {
						ref.Pix[y*ref.Stride+x] = byte(p)
					} else {
						off := y*ref.Stride + x*2
						ref.Pix[off] = byte(p)
						ref.Pix[off+1] = byte(p >> 8)
					}
				}
			}

			sf, err := NewScaleFactors(w, h, w, h)
			if err != nil {
				t.Fatal(err)
			}
			startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
			if err != nil {
				t.Fatal(err)
			}
			xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
			if err != nil {
				t.Fatal(err)
			}
			yTable, err := SubpelKernelTableFor(InterpEightTapRegular, h)
			if err != nil {
				t.Fatal(err)
			}

			var scaled, unscaled CompoundConvBuf
			if err := PredictScaledCompoundRefToConvBufWithScratch(&scaled, ref, tc.bps, tc.bitDepth, w, h, startX, xStep, startY, yStep, xTable, yTable, nil); err != nil {
				t.Fatalf("PredictScaledCompoundRefToConvBuf: %v", err)
			}
			if err := PredictInterCompoundRefToConvBuf(&unscaled, ref, tc.bps, tc.bitDepth, 0, 0, w, h, w, h, 0, 0, RegularFilters, nil); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf: %v", err)
			}
			if scaled.Width != unscaled.Width || scaled.Height != unscaled.Height {
				t.Fatalf("dims got=%dx%d want=%dx%d", scaled.Width, scaled.Height, unscaled.Width, unscaled.Height)
			}
			for i := range w * h {
				if scaled.Data[i] != unscaled.Data[i] {
					t.Fatalf("convbuf[%d]=%d want %d", i, scaled.Data[i], unscaled.Data[i])
				}
			}
		})
	}
}

func TestPredictScaledCompoundRefToConvBufSubpelIdentityMatchesUnscaled(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		bps      int
		subX     int
		subY     int
	}{
		{"bd8_x", 8, 1, 5, 0},
		{"bd8_y", 8, 1, 0, 7},
		{"bd8_2d", 8, 1, 3, 5},
		{"bd10_2d", 10, 2, 3, 5},
		{"bd12_2d", 12, 2, 11, 13},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const (
				w    = 16
				h    = 16
				refW = 40
				refH = 40
				refX = 8
				refY = 8
			)
			ref := frame.Plane{Pix: make([]byte, refW*refH*tc.bps), Stride: refW * tc.bps, Width: refW, Height: refH}
			mask := (1 << tc.bitDepth) - 1
			for y := range refH {
				for x := range refW {
					p := (x*31 + y*47 + x*y*7 + int(tc.bitDepth)*13) & mask
					if tc.bps == 1 {
						ref.Pix[y*ref.Stride+x] = byte(p)
					} else {
						off := y*ref.Stride + x*2
						ref.Pix[off] = byte(p)
						ref.Pix[off+1] = byte(p >> 8)
					}
				}
			}
			xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
			if err != nil {
				t.Fatal(err)
			}
			yTable, err := SubpelKernelTableFor(InterpEightTapSmooth, h)
			if err != nil {
				t.Fatal(err)
			}
			startX := int64(refX)*ScaleSubpelScale + int64(tc.subX<<ScaleExtraBits)
			startY := int64(refY)*ScaleSubpelScale + int64(tc.subY<<ScaleExtraBits)
			var scaled, unscaled CompoundConvBuf
			if err := PredictScaledCompoundRefToConvBufWithScratch(&scaled, ref, tc.bps, tc.bitDepth, w, h, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, nil); err != nil {
				t.Fatalf("PredictScaledCompoundRefToConvBuf: %v", err)
			}
			if err := PredictInterCompoundRefToConvBuf(&unscaled, ref, tc.bps, tc.bitDepth, refX, refY, w, h, w, h, tc.subX, tc.subY, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}, nil); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf: %v", err)
			}
			for i := range w * h {
				if scaled.Data[i] != unscaled.Data[i] {
					t.Fatalf("convbuf[%d]=%d want %d", i, scaled.Data[i], unscaled.Data[i])
				}
			}
		})
	}
}

func TestPredictScaledCompoundRefToConvBufYIdentityMatchesGeneric(t *testing.T) {
	const (
		w    = 16
		h    = 16
		refW = 48
		refH = 32
	)
	ref := frame.Plane{Pix: make([]byte, refW*refH), Stride: refW, Width: refW, Height: refH}
	for y := range refH {
		for x := range refW {
			ref.Pix[y*ref.Stride+x] = byte((x*37 + y*53 + x*y*3 + 19) & 0xff)
		}
	}
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
	if err != nil {
		t.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, h)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name           string
		startX, startY int64
		xStep          int64
	}{
		{
			name:   "interior_three_quarter_step",
			startX: int64(12)*ScaleSubpelScale + ScaleSubpelScale/4,
			startY: int64(8) * ScaleSubpelScale,
			xStep:  ScaleSubpelScale * 3 / 4,
		},
		{
			name:   "left_edge_half_step",
			startX: -ScaleSubpelScale / 4,
			startY: int64(4) * ScaleSubpelScale,
			xStep:  ScaleSubpelScale / 2,
		},
		{
			name:   "right_edge_double_step",
			startX: int64(refW-w/2) * ScaleSubpelScale,
			startY: int64(refH-h) * ScaleSubpelScale,
			xStep:  2 * ScaleSubpelScale,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got, want CompoundConvBuf
			var scratch ScaledConvolveScratch
			if err := PredictScaledCompoundRefToConvBufWithScratch(&got, ref, 1, 8, w, h,
				tc.startX, tc.xStep, tc.startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
				t.Fatalf("PredictScaledCompoundRefToConvBufWithScratch: %v", err)
			}
			out, ok := compoundConvBufView(&want, w, h)
			if !ok {
				t.Fatal("invalid convbuf")
			}
			imH, ok := scaledIMHeight(h, tc.startY, ScaleSubpelScale)
			if !ok {
				t.Fatal("invalid intermediate height")
			}
			predictScaledCompoundRefToConvBufGeneric(out, ref, 1, 8, w, h,
				tc.startX, tc.xStep, tc.startY, ScaleSubpelScale, xTable, yTable, imH, &scratch)
			for i := range w * h {
				if got.Data[i] != want.Data[i] {
					t.Fatalf("convbuf[%d]=%d want %d", i, got.Data[i], want.Data[i])
				}
			}
		})
	}
}

func TestPredictScaledCompoundRefToConvBufHighBDYIdentityMatchesGeneric(t *testing.T) {
	const (
		w    = 16
		h    = 16
		refW = 48
		refH = 32
	)
	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
	if err != nil {
		t.Fatal(err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, h)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name           string
		bitDepth       uint8
		startX, startY int64
		xStep          int64
	}{
		{
			name:     "bd10_interior_three_quarter_step",
			bitDepth: 10,
			startX:   int64(12)*ScaleSubpelScale + ScaleSubpelScale/4,
			startY:   int64(8) * ScaleSubpelScale,
			xStep:    ScaleSubpelScale * 3 / 4,
		},
		{
			name:     "bd10_left_edge_half_step",
			bitDepth: 10,
			startX:   -ScaleSubpelScale / 4,
			startY:   int64(4) * ScaleSubpelScale,
			xStep:    ScaleSubpelScale / 2,
		},
		{
			name:     "bd12_right_edge_double_step",
			bitDepth: 12,
			startX:   int64(refW-w/2) * ScaleSubpelScale,
			startY:   int64(refH-h) * ScaleSubpelScale,
			xStep:    2 * ScaleSubpelScale,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ref := highBDPlane(refW, refH, 0x534d+int(tc.bitDepth), int(tc.bitDepth))
			var got, want CompoundConvBuf
			var scratch ScaledConvolveScratch
			if err := PredictScaledCompoundRefToConvBufWithScratch(&got, ref, 2, tc.bitDepth, w, h,
				tc.startX, tc.xStep, tc.startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
				t.Fatalf("PredictScaledCompoundRefToConvBufWithScratch: %v", err)
			}
			out, ok := compoundConvBufView(&want, w, h)
			if !ok {
				t.Fatal("invalid convbuf")
			}
			imH, ok := scaledIMHeight(h, tc.startY, ScaleSubpelScale)
			if !ok {
				t.Fatal("invalid intermediate height")
			}
			predictScaledCompoundRefToConvBufGeneric(out, ref, 2, tc.bitDepth, w, h,
				tc.startX, tc.xStep, tc.startY, ScaleSubpelScale, xTable, yTable, imH, &scratch)
			for i := range w * h {
				if got.Data[i] != want.Data[i] {
					t.Fatalf("convbuf[%d]=%d want %d", i, got.Data[i], want.Data[i])
				}
			}
		})
	}
}

// TestPredictInterCompoundRefToConvBufErrors verifies the parameter guards.
func TestPredictInterCompoundRefToConvBufErrors(t *testing.T) {
	ref := frame.Plane{Pix: make([]byte, 64), Stride: 8, Width: 8, Height: 8}
	var buf CompoundConvBuf
	// nil buffer.
	if err := PredictInterCompoundRefToConvBuf(nil, ref, 1, 8, 0, 0, 8, 8, 8, 8, 0, 0, RegularFilters, nil); err != ErrInvalidMotion {
		t.Errorf("nil buf err=%v want ErrInvalidMotion", err)
	}
	// Out-of-range subpel position.
	if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, 0, 0, 8, 8, 8, 8, -1, 0, RegularFilters, nil); err != ErrInvalidMotion {
		t.Errorf("bad subX err=%v want ErrInvalidMotion", err)
	}
	// Bit-depth / sample-width mismatch (8-bit declared but 2 bytes/sample).
	if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, 8, 0, 0, 8, 8, 8, 8, 0, 0, RegularFilters, nil); err != ErrInvalidMotion {
		t.Errorf("bps mismatch err=%v want ErrInvalidMotion", err)
	}
}

// TestPredictScaledCompoundIdentityStep8ZeroAlloc keeps the 1:1-step scaled
// compound path (a scaled reference whose step resolves to identity)
// allocation-free for every filter and subpel phase. Its 2D SIMD kernel falls
// back to a heap intermediate when it is called without scratch, so the path
// must pass the caller's scratch through.
func TestPredictScaledCompoundIdentityStep8ZeroAlloc(t *testing.T) {
	const (
		w    = 16
		h    = 16
		refW = 48
		refH = 32
	)
	ref := frame.Plane{Pix: make([]byte, refW*refH), Stride: refW, Width: refW, Height: refH}
	for i := range ref.Pix {
		ref.Pix[i] = byte(i * 37)
	}
	var buf CompoundConvBuf
	scratch := new(ScaledConvolveScratch)
	for _, filter := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
		table, err := SubpelKernelTableFor(filter, w)
		if err != nil {
			t.Fatal(err)
		}
		// Fractional starts in both axes select the separable 2D kernel.
		for frac := int64(1); frac < 16; frac++ {
			startX := int64(12)*ScaleSubpelScale + frac*ScaleSubpelScale/16
			startY := int64(8)*ScaleSubpelScale + (16-frac)*ScaleSubpelScale/16
			allocs := testing.AllocsPerRun(5, func() {
				if err := PredictScaledCompoundRefToConvBufWithScratch(&buf, ref, 1, 8, w, h,
					startX, ScaleSubpelScale, startY, ScaleSubpelScale, table, table, scratch); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Fatalf("filter %d frac %d/16: identity-step scaled compound allocates %v times per call", filter, frac, allocs)
			}
		}
	}
}

// This file pins the per-tap-clamping pure-Go convolve reference (the spec
// oracle, identical to the original convolve*ClampedPureGo before the
// edge/interior split optimization) and asserts the production clamped
// functions stay bit-identical to it across the width/tap/edge matrix.

func refConvolveX8Clamped(dst frame.Plane, ref frame.Plane, dstX, dstY, refX, refY, width, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	for y := 0; y < height; y++ {
		sy := refY + y
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		for x := 0; x < width; x++ {
			sx := refX + x - fo
			sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
				k1*int(loadSample8Clamped(ref, sx+1, sy)) +
				k2*int(loadSample8Clamped(ref, sx+2, sy)) +
				k3*int(loadSample8Clamped(ref, sx+3, sy)) +
				k4*int(loadSample8Clamped(ref, sx+4, sy)) +
				k5*int(loadSample8Clamped(ref, sx+5, sy)) +
				k6*int(loadSample8Clamped(ref, sx+6, sy)) +
				k7*int(loadSample8Clamped(ref, sx+7, sy))
			res := roundPowerOfTwo(sum, round0Bits)
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func refConvolveY8Clamped(dst frame.Plane, ref frame.Plane, dstX, dstY, refX, refY, width, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	k0, k1, k2, k3, k4, k5, k6, k7 := int(kernel[0]), int(kernel[1]), int(kernel[2]), int(kernel[3]), int(kernel[4]), int(kernel[5]), int(kernel[6]), int(kernel[7])
	for y := 0; y < height; y++ {
		sy := refY + y - fo
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		for x := 0; x < width; x++ {
			sx := refX + x
			sum := k0*int(loadSample8Clamped(ref, sx, sy)) +
				k1*int(loadSample8Clamped(ref, sx, sy+1)) +
				k2*int(loadSample8Clamped(ref, sx, sy+2)) +
				k3*int(loadSample8Clamped(ref, sx, sy+3)) +
				k4*int(loadSample8Clamped(ref, sx, sy+4)) +
				k5*int(loadSample8Clamped(ref, sx, sy+5)) +
				k6*int(loadSample8Clamped(ref, sx, sy+6)) +
				k7*int(loadSample8Clamped(ref, sx, sy+7))
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(sum, filterBits)))
		}
	}
}

func refConvolve2D8Clamped(dst frame.Plane, ref frame.Plane, dstX, dstY, refX, refY, width, height int, xKernel, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	imH := height + filterTaps - 1
	const xBias = 1 << (8 + filterBits - 1)
	xk0, xk1, xk2, xk3, xk4, xk5, xk6, xk7 := int(xKernel[0]), int(xKernel[1]), int(xKernel[2]), int(xKernel[3]), int(xKernel[4]), int(xKernel[5]), int(xKernel[6]), int(xKernel[7])
	for y := 0; y < imH; y++ {
		sy := refY - foY + y
		imRow := im[y*imStride:]
		for x := 0; x < width; x++ {
			sx := refX + x - foX
			sum := xBias +
				xk0*int(loadSample8Clamped(ref, sx, sy)) +
				xk1*int(loadSample8Clamped(ref, sx+1, sy)) +
				xk2*int(loadSample8Clamped(ref, sx+2, sy)) +
				xk3*int(loadSample8Clamped(ref, sx+3, sy)) +
				xk4*int(loadSample8Clamped(ref, sx+4, sy)) +
				xk5*int(loadSample8Clamped(ref, sx+5, sy)) +
				xk6*int(loadSample8Clamped(ref, sx+6, sy)) +
				xk7*int(loadSample8Clamped(ref, sx+7, sy))
			imRow[x] = int16(roundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	yBias := 1 << offsetBits
	yk0, yk1, yk2, yk3, yk4, yk5, yk6, yk7 := int(yKernel[0]), int(yKernel[1]), int(yKernel[2]), int(yKernel[3]), int(yKernel[4]), int(yKernel[5]), int(yKernel[6]), int(yKernel[7])
	for y := 0; y < height; y++ {
		dstRow := dst.Pix[(dstY+y)*dst.Stride+dstX:]
		col := im[y*imStride : (y+7)*imStride+width]
		for x := 0; x < width; x++ {
			sum := yBias + yk0*int(col[x]) + yk1*int(col[x+imStride]) + yk2*int(col[x+2*imStride]) +
				yk3*int(col[x+3*imStride]) + yk4*int(col[x+4*imStride]) + yk5*int(col[x+5*imStride]) +
				yk6*int(col[x+6*imStride]) + yk7*int(col[x+7*imStride])
			res := roundPowerOfTwo(sum, round1Bits) - roundOffset
			dstRow[x] = byte(clipPixel(roundPowerOfTwo(res, bits)))
		}
	}
}

// clampedRefCase describes one block placement against the reference plane.
type clampedRefCase struct {
	name               string
	refX, refY         int
	width, height      int
	fourTapX, fourTapY bool
}

func TestConvolveClampedMatchesPerTapReference(t *testing.T) {
	const planeW, planeH = 48, 48
	ref, _ := testPlane(planeW, planeH, 1, planeW)
	fillMotionTestPlane(ref)

	// Subpel index 3 (8-tap regular) and 5 (8-tap, different phase); 4-tap from
	// subpelFilters4. The clamped functions branch on whether the kernel has
	// zero end taps, so cover both shapes.
	xk8 := subpelFilters8[3]
	yk8 := subpelFilters8[5]
	xk4 := subpelFilters4[3]
	yk4 := subpelFilters4[5]

	pick := func(four bool, eight, fourK [filterTaps]int16) [filterTaps]int16 {
		if four {
			return fourK
		}
		return eight
	}

	cases := []clampedRefCase{
		// Interior (fully resident halo, tap window inside the plane).
		{"interior_16x16_8tap", 12, 12, 16, 16, false, false},
		{"interior_8x8_8tap", 16, 16, 8, 8, false, false},
		{"interior_4x4_8tap", 20, 20, 4, 4, false, false},
		{"interior_4x16_4tapX_8tapY", 20, 12, 4, 16, true, false},
		{"interior_16x16_4tap", 12, 12, 16, 16, true, true},
		// Top-left corner (negative tap coordinates -> clamp active).
		{"corner_16x16_8tap", 0, 0, 16, 16, false, false},
		{"corner_8x8_8tap", 0, 0, 8, 8, false, false},
		{"corner_4x4_8tap", 0, 0, 4, 4, false, false},
		{"corner_4x16_4tap", 0, 0, 4, 16, true, true},
		// Left edge only / top edge only.
		{"left_8x8_8tap", 0, 20, 8, 8, false, false},
		{"top_8x8_8tap", 20, 0, 8, 8, false, false},
		// Bottom-right corner (tap coordinates beyond Width/Height).
		{"br_16x16_8tap", planeW - 16, planeH - 16, 16, 16, false, false},
		{"br_8x8_8tap", planeW - 8, planeH - 8, 8, 8, false, false},
		{"br_4x4_4tap", planeW - 4, planeH - 4, 4, 4, true, true},
		// Block straddling the right edge so part of every row needs clamping.
		{"right_straddle_8x8_8tap", planeW - 4, 16, 8, 8, false, false},
		{"bottom_straddle_8x8_8tap", 16, planeH - 4, 8, 8, false, false},
	}

	for _, c := range cases {
		xk := pick(c.fourTapX, xk8, xk4)
		yk := pick(c.fourTapY, yk8, yk4)

		t.Run("2D/"+c.name, func(t *testing.T) {
			got, _ := testPlane(c.width, c.height, 1, c.width)
			want, _ := testPlane(c.width, c.height, 1, c.width)
			convolve2D8ClampedPureGo(got, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk, yk)
			refConvolve2D8Clamped(want, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk, yk)
			assertClampedEqual(t, got, want, c.width, c.height)
		})
		t.Run("X/"+c.name, func(t *testing.T) {
			got, _ := testPlane(c.width, c.height, 1, c.width)
			want, _ := testPlane(c.width, c.height, 1, c.width)
			convolveX8ClampedPureGo(got, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk)
			refConvolveX8Clamped(want, ref, 0, 0, c.refX, c.refY, c.width, c.height, xk)
			assertClampedEqual(t, got, want, c.width, c.height)
		})
		t.Run("Y/"+c.name, func(t *testing.T) {
			got, _ := testPlane(c.width, c.height, 1, c.width)
			want, _ := testPlane(c.width, c.height, 1, c.width)
			convolveY8ClampedPureGo(got, ref, 0, 0, c.refX, c.refY, c.width, c.height, yk)
			refConvolveY8Clamped(want, ref, 0, 0, c.refX, c.refY, c.width, c.height, yk)
			assertClampedEqual(t, got, want, c.width, c.height)
		})
	}
}

func assertClampedEqual(t *testing.T, got, want frame.Plane, width, height int) {
	t.Helper()
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			g := got.Pix[y*got.Stride+x]
			w := want.Pix[y*want.Stride+x]
			if g != w {
				t.Fatalf("mismatch at (%d,%d): got %d want %d", x, y, g, w)
			}
		}
	}
}

// TestConvolve2DHighBDScratchMatchesNoScratch asserts the pure-Go
// scratch-carrying HBD 2D convolves and the bound dispatch slots are
// bit-identical to their no-scratch references with a deliberately poisoned
// scratch, proving every intermediate sample read is written first. Runs on
// every build (including purego), exercising whichever variant the dispatch
// slots bound at init.
func TestConvolve2DHighBDScratchMatchesNoScratch(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2d5c))
	const pad = filterTaps
	scratch := &ConvolveScratch{}
	poison := func() {
		for i := range scratch.imHBD {
			scratch.imHBD[i] = int32(0x7bad0000) | int32(i&0xffff)
		}
		for i := range scratch.edge16 {
			scratch.edge16[i] = byte(0xa5)
		}
	}
	xk := subpelFilters8[6]
	yk := subpelFilters8[9]
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		for _, w := range []int{4, 8, 16, 32, 64, 128} {
			for _, h := range []int{4, 8, 16, 32, 64} {
				side := w
				if h > side {
					side = h
				}
				refSide := side + 2*pad
				ref, _ := testPlane(refSide, refSide, 2, refSide*2)
				for y := 0; y < refSide; y++ {
					for x := 0; x < refSide; x++ {
						setSample(ref, 2, x, y, uint16(rng.Intn(int(max)+1)))
					}
				}
				// Pure-Go WithScratch, resident window.
				g, _ := testPlane(w, h, 2, w*2)
				wn, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDPureGoWithScratch(g, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDPureGo(wn, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				diffHighBDScratchPlanes(t, g, wn, w, h, "puregoscratch", bd)
				// Pure-Go WithScratch, clamped overhang.
				gc, _ := testPlane(w, h, 2, w*2)
				wc, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedPureGoWithScratch(gc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedPureGo(wc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				diffHighBDScratchPlanes(t, gc, wc, w, h, "puregoscratch-clamped", bd)
				// Dispatch slots (whatever the arch bound), scratch vs no-scratch.
				gd, _ := testPlane(w, h, 2, w*2)
				wd, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDWithScratchImpl(gd, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk, scratch)
				convolve2DHighBDImpl(wd, ref, bd, max, 0, 0, pad, pad, w, h, xk, yk)
				diffHighBDScratchPlanes(t, gd, wd, w, h, "dispatchscratch", bd)
				gdc, _ := testPlane(w, h, 2, w*2)
				wdc, _ := testPlane(w, h, 2, w*2)
				poison()
				convolve2DHighBDClampedWithScratchImpl(gdc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk, scratch)
				convolve2DHighBDClampedImpl(wdc, ref, bd, max, 0, 0, -3, -3, w, h, xk, yk)
				diffHighBDScratchPlanes(t, gdc, wdc, w, h, "dispatchscratch-clamped", bd)
			}
		}
	}
}

// TestBlendCompoundAvgHighBDImplMatchesReference asserts the bound HBD blend
// dispatch slot (GoSIMD on arm64, pure-Go elsewhere) matches the canonical
// pure-Go reference on every build.
func TestBlendCompoundAvgHighBDImplMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xb1e42))
	for _, bd := range []uint8{10, 12} {
		max, _ := highBDMax(bd)
		round0 := compoundRound0(bd)
		offsetBits := int(bd) + 2*filterBits - round0
		roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
		roundBits := 2*filterBits - round0 - compoundRound1Bits
		for _, w := range []int{4, 8, 16, 32, 64, 128} {
			for _, h := range []int{1, 3, 4, 8, 32} {
				src0 := make([]uint16, w*h)
				src1 := make([]uint16, w*h)
				for i := range src0 {
					src0[i] = uint16(rng.Intn(1 << 16))
					src1[i] = uint16(rng.Intn(1 << 16))
				}
				for _, wt := range [][2]int{{8, 8}, {9, 7}, {13, 3}} {
					got, _ := testPlane(w, h, 2, w*2)
					want, _ := testPlane(w, h, 2, w*2)
					blendCompoundAvgHighBDImpl(got, src0, src1, max, 0, 0, w, h, wt[0], wt[1], roundOffset, roundBits)
					blendCompoundAvgHighBD(want, src0, src1, max, 0, 0, w, h, wt[0], wt[1], roundOffset, roundBits)
					diffHighBDScratchPlanes(t, got, want, w, h, "blendimpl", bd, w, h, wt)
				}
			}
		}
	}
}

func diffHighBDScratchPlanes(t *testing.T, got, want frame.Plane, w, h int, tag string, ctx ...any) {
	t.Helper()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			g := getSample(got, 2, x, y)
			e := getSample(want, 2, x, y)
			if g != e {
				t.Fatalf("%s (%d,%d): scratch=%d ref=%d ctx=%v", tag, x, y, g, e, ctx)
			}
		}
	}
}

const (
	libaomInputPadding = 12
	libaomInputOrigin  = libaomInputPadding / 2
)

type libaomConvolveBlockSize struct {
	width  int
	height int
}

func TestLibaomConvolveParametersPort(t *testing.T) {
	sizes := libaomConvolveTestSizes()
	if len(sizes) != 27 {
		t.Fatalf("len(sizes)=%d want 27", len(sizes))
	}
	if sizes[0] != (libaomConvolveBlockSize{width: 2, height: 2}) {
		t.Fatalf("first size=%+v want 2x2", sizes[0])
	}
	if sizes[len(sizes)-1] != (libaomConvolveBlockSize{width: 128, height: 128}) {
		t.Fatalf("last size=%+v want 128x128", sizes[len(sizes)-1])
	}
}

func TestLibaomConvolveXRunTestPort(t *testing.T) {
	for _, size := range libaomConvolveTestSizes() {
		for subX := 1; subX < 16; subX++ {
			for _, filter := range libaomConvolveFilters() {
				runLibaomConvolveXCase(t, size, subX, filter)
			}
		}
	}
}

func TestLibaomConvolveYRunTestPort(t *testing.T) {
	for _, size := range libaomConvolveTestSizes() {
		for subY := 1; subY < 16; subY++ {
			for _, filter := range libaomConvolveFilters() {
				runLibaomConvolveYCase(t, size, subY, filter)
			}
		}
	}
}

func TestLibaomConvolve2DQuickPort(t *testing.T) {
	for _, size := range []libaomConvolveBlockSize{
		{width: 2, height: 2},
		{width: 4, height: 16},
		{width: 8, height: 8},
		{width: 16, height: 64},
		{width: 32, height: 16},
		{width: 64, height: 64},
		{width: 128, height: 128},
	} {
		for _, subX := range []int{1, 2, 7, 8, 15} {
			for _, subY := range []int{1, 2, 7, 8, 15} {
				for _, filters := range []InterpFilters{
					{X: InterpEightTapRegular, Y: InterpEightTapRegular},
					{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
					{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
					{X: InterpBilinear, Y: InterpBilinear},
					{X: InterpEightTapRegular, Y: InterpEightTapSmooth},
					{X: InterpMultiTapSharp, Y: InterpBilinear},
				} {
					runLibaomConvolve2DCase(t, size, subX, subY, filters)
				}
			}
		}
	}
}

func TestLibaomConvolve2DRunTestPort(t *testing.T) {
	if os.Getenv("GOAV1_FULL_LIBAOM_CONVOLVE") == "" {
		t.Skip("set GOAV1_FULL_LIBAOM_CONVOLVE=1 to run the full libaom 2D convolve matrix")
	}
	for _, size := range libaomConvolveTestSizes() {
		for subX := 1; subX < 16; subX++ {
			for subY := 1; subY < 16; subY++ {
				for _, filterX := range libaomConvolveFilters() {
					for _, filterY := range libaomConvolveFilters() {
						runLibaomConvolve2DCase(t, size, subX, subY, InterpFilters{X: filterX, Y: filterY})
					}
				}
			}
		}
	}
}

func runLibaomConvolveXCase(t *testing.T, size libaomConvolveBlockSize, subX int, filter InterpFilter) {
	t.Helper()
	src := libaomConvolveInput8(size)
	got, _ := testPlane(size.width, size.height, 1, size.width)
	want, _ := testPlane(size.width, size.height, 1, size.width)
	kernel, err := interpKernel(filter, size.width, subX)
	if err != nil {
		t.Fatalf("x %dx%d sub=%d filter=%d: %v", size.width, size.height, subX, filter, err)
	}
	convolveX8Impl(got, src, 0, 0, libaomInputOrigin, libaomInputOrigin, size.width, size.height, kernel)
	libaomConvolveXRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, size.width, size.height, kernel)
	assertLibaomConvolveEqual(t, "x", got, want, size, subX, 0, InterpFilters{X: filter})
}

func runLibaomConvolveYCase(t *testing.T, size libaomConvolveBlockSize, subY int, filter InterpFilter) {
	t.Helper()
	src := libaomConvolveInput8(size)
	got, _ := testPlane(size.width, size.height, 1, size.width)
	want, _ := testPlane(size.width, size.height, 1, size.width)
	kernel, err := interpKernel(filter, size.height, subY)
	if err != nil {
		t.Fatalf("y %dx%d sub=%d filter=%d: %v", size.width, size.height, subY, filter, err)
	}
	convolveY8Impl(got, src, 0, 0, libaomInputOrigin, libaomInputOrigin, size.width, size.height, kernel)
	libaomConvolveYRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, size.width, size.height, kernel)
	assertLibaomConvolveEqual(t, "y", got, want, size, 0, subY, InterpFilters{Y: filter})
}

func runLibaomConvolve2DCase(t *testing.T, size libaomConvolveBlockSize, subX int, subY int, filters InterpFilters) {
	t.Helper()
	src := libaomConvolveInput8(size)
	got, _ := testPlane(size.width, size.height, 1, size.width)
	want, _ := testPlane(size.width, size.height, 1, size.width)
	xKernel, err := interpKernel(filters.X, size.width, subX)
	if err != nil {
		t.Fatalf("2d %dx%d sub=%d,%d filters=%d,%d: %v", size.width, size.height, subX, subY, filters.X, filters.Y, err)
	}
	yKernel, err := interpKernel(filters.Y, size.height, subY)
	if err != nil {
		t.Fatalf("2d %dx%d sub=%d,%d filters=%d,%d: %v", size.width, size.height, subX, subY, filters.X, filters.Y, err)
	}
	convolve2D8Impl(got, src, 0, 0, libaomInputOrigin, libaomInputOrigin, size.width, size.height, xKernel, yKernel)
	libaomConvolve2DRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, size.width, size.height, xKernel, yKernel)
	assertLibaomConvolveEqual(t, "2d", got, want, size, subX, subY, filters)
}

func assertLibaomConvolveEqual(t *testing.T, name string, got frame.Plane, want frame.Plane, size libaomConvolveBlockSize, subX int, subY int, filters InterpFilters) {
	t.Helper()
	for y := 0; y < size.height; y++ {
		for x := 0; x < size.width; x++ {
			g := got.Pix[y*got.Stride+x]
			w := want.Pix[y*want.Stride+x]
			if g != w {
				t.Fatalf("%s %dx%d sub=%d,%d filters=%d,%d sample(%d,%d)=%d want %d", name, size.width, size.height, subX, subY, filters.X, filters.Y, x, y, g, w)
			}
		}
	}
}

func libaomConvolveInput8(size libaomConvolveBlockSize) frame.Plane {
	plane, _ := testPlane(size.width+libaomInputPadding, size.height+libaomInputPadding, 1, size.width+libaomInputPadding)
	var state uint32 = 0xbaba
	for i := range plane.Pix {
		state = state*1664525 + 1013904223
		plane.Pix[i] = byte(state >> 24)
	}
	return plane
}

func libaomConvolveFilters() []InterpFilter {
	return []InterpFilter{
		InterpEightTapRegular,
		InterpEightTapSmooth,
		InterpMultiTapSharp,
		InterpBilinear,
	}
}

func libaomConvolveTestSizes() []libaomConvolveBlockSize {
	return []libaomConvolveBlockSize{
		{width: 2, height: 2},
		{width: 2, height: 4},
		{width: 2, height: 8},
		{width: 4, height: 2},
		{width: 4, height: 4},
		{width: 4, height: 8},
		{width: 4, height: 16},
		{width: 8, height: 2},
		{width: 8, height: 4},
		{width: 8, height: 8},
		{width: 8, height: 16},
		{width: 8, height: 32},
		{width: 16, height: 4},
		{width: 16, height: 8},
		{width: 16, height: 16},
		{width: 16, height: 32},
		{width: 16, height: 64},
		{width: 32, height: 8},
		{width: 32, height: 16},
		{width: 32, height: 32},
		{width: 32, height: 64},
		{width: 64, height: 16},
		{width: 64, height: 32},
		{width: 64, height: 64},
		{width: 64, height: 128},
		{width: 128, height: 64},
		{width: 128, height: 128},
	}
}

// libaomConvolveScale2DRef is a faithful port of libaom's
// av1_convolve_2d_scale_c (av1/common/convolve.c). It takes the same
// post-pos>>10 source pointer convention as libaom: the caller pre-offsets
// the reference plane by (refX0, refY0) integer-pel, and passes only the Q10
// sub-pel fractional component plus per-output Q10 step.
func libaomConvolveScale2DRef(dst frame.Plane, src frame.Plane, refX0, refY0 int,
	dstX, dstY, width, height int, subpelXQN, xStepQN, subpelYQN, yStepQN int,
	xTable, yTable *SubpelKernelTable, conv_params_round_0, conv_params_round_1 int) {
	const imStride = maxBlockSize
	var im [scaledIMMaxHeight * maxBlockSize]int16
	imH := (((height-1)*yStepQN + subpelYQN) >> ScaleSubpelBits) + filterTaps
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	bd := 8

	// Horizontal pass: src_horiz = src - fo_vert * src_stride
	startRow := refY0 - foVert
	for y := range imH {
		srcRow := startRow + y
		xQN := subpelXQN
		for x := range width {
			xInt := (xQN >> ScaleSubpelBits)
			xFilterIdx := (xQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := xTable[xFilterIdx]
			sum := 1 << (bd + filterBits - 1)
			for k := range filterTaps {
				col := refX0 + xInt + k - foHoriz
				sum += int(kernel[k]) * int(src.Pix[srcRow*src.Stride+col])
			}
			im[y*imStride+x] = int16(libaomRoundPowerOfTwo(sum, conv_params_round_0))
			xQN += xStepQN
		}
	}

	// Vertical pass: src_vert = im_block + fo_vert * im_stride
	offsetBits := bd + 2*filterBits - conv_params_round_0
	roundOffset := (1 << (offsetBits - conv_params_round_1)) + (1 << (offsetBits - conv_params_round_1 - 1))
	bits := 2*filterBits - conv_params_round_0 - conv_params_round_1
	for x := range width {
		yQN := subpelYQN
		for y := range height {
			yInt := (yQN >> ScaleSubpelBits)
			yFilterIdx := (yQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := yTable[yFilterIdx]
			sum := 1 << offsetBits
			// src_y = &src_vert[(yQN>>10) * im_stride]; tap k reads
			// src_y[(k - fo_vert) * im_stride] = im_block[(fo_vert + yInt + k - fo_vert) * im_stride] = im[(yInt + k) * im_stride]
			for k := range filterTaps {
				sum += int(kernel[k]) * int(im[(yInt+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, conv_params_round_1) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, bits)))
			yQN += yStepQN
		}
	}
}

func libaomConvolveScale2DHighBDRef(dst frame.Plane, src frame.Plane, bd int, refX0, refY0 int,
	dstX, dstY, width, height int, subpelXQN, xStepQN, subpelYQN, yStepQN int,
	xTable, yTable *SubpelKernelTable, conv_params_round_0, conv_params_round_1 int) {
	const imStride = maxBlockSize
	var im [scaledIMMaxHeight * maxBlockSize]int32
	imH := (((height-1)*yStepQN + subpelYQN) >> ScaleSubpelBits) + filterTaps
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	max := uint16((1 << bd) - 1)

	startRow := refY0 - foVert
	for y := range imH {
		srcRow := startRow + y
		xQN := subpelXQN
		for x := range width {
			xInt := (xQN >> ScaleSubpelBits)
			xFilterIdx := (xQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := xTable[xFilterIdx]
			sum := 1 << (bd + filterBits - 1)
			for k := range filterTaps {
				col := refX0 + xInt + k - foHoriz
				sum += int(kernel[k]) * int(getSample(src, 2, col, srcRow))
			}
			im[y*imStride+x] = int32(libaomRoundPowerOfTwo(sum, conv_params_round_0))
			xQN += xStepQN
		}
	}

	offsetBits := bd + 2*filterBits - conv_params_round_0
	roundOffset := (1 << (offsetBits - conv_params_round_1)) + (1 << (offsetBits - conv_params_round_1 - 1))
	bits := 2*filterBits - conv_params_round_0 - conv_params_round_1
	for x := range width {
		yQN := subpelYQN
		for y := range height {
			yInt := (yQN >> ScaleSubpelBits)
			yFilterIdx := (yQN & ScaleSubpelMask) >> ScaleExtraBits
			kernel := yTable[yFilterIdx]
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(kernel[k]) * int(im[(yInt+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, conv_params_round_1) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), max))
			yQN += yStepQN
		}
	}
}

// scaledTestSourcePlane fills a plane with deterministic pseudo-random bytes
// for libaom-shaped scaled convolver fuzzing.
func scaledTestSourcePlane(width, height, bytesPerSample, seed int) frame.Plane {
	plane, _ := testPlane(width, height, bytesPerSample, width*bytesPerSample)
	state := uint32(0xc0ffee | uint32(seed))
	for i := range plane.Pix {
		state = state*1664525 + 1013904223
		plane.Pix[i] = byte(state >> 24)
	}
	return plane
}

// TestConvolveScale2D8MatchesLibaomIdentity verifies the scaled convolver
// matches the same-size 2D 8-tap convolver under identity scaling. With
// xStep = yStep = ScaleSubpelScale and zero subpel, the result must equal the
// existing convolve2D8 output bit-exact.
func TestConvolveScale2D8MatchesLibaomIdentity(t *testing.T) {
	const w, h = 16, 8
	src := scaledTestSourcePlane(w+filterTaps+8, h+filterTaps+8, 1, 0xabcd)
	got, _ := testPlane(w, h, 1, w)
	want, _ := testPlane(w, h, 1, w)

	// Pre-offset position so the centre tap is at the block origin (8, 8) in
	// the input plane, leaving foX=foY=3 taps of headroom on each edge.
	const refX0, refY0 = 8, 8
	startX := int64(refX0) * ScaleSubpelScale
	startY := int64(refY0) * ScaleSubpelScale

	for _, filter := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
		xTable, err := SubpelKernelTableFor(filter, w)
		if err != nil {
			t.Fatalf("xTable: %v", err)
		}
		yTable, err := SubpelKernelTableFor(filter, h)
		if err != nil {
			t.Fatalf("yTable: %v", err)
		}
		clearPlane(got)
		clearPlane(want)
		if err := ConvolveScale2D8ClampedWithScratch(got, src, 0, 0, w, h, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, nil); err != nil {
			t.Fatalf("filter %d: ConvolveScale2D8ClampedWithScratch: %v", filter, err)
		}
		// Same-size reference convolver uses subpel=0 → integer-pel only.
		xKern, _ := interpKernel(filter, w, 0)
		yKern, _ := interpKernel(filter, h, 0)
		libaomConvolve2DRef(want, src, refX0, refY0, 0, 0, w, h, xKern, yKern)
		comparePlanes(t, "identity_"+filterName(filter), got, want, w, h, 1)
	}
}

// TestConvolveScale2D8MatchesLibaomReference confirms the new scaled
// convolver matches libaom's av1_convolve_2d_scale_c bit-exact under a range
// of typical scaled-position / step configurations.
func TestConvolveScale2D8MatchesLibaomReference(t *testing.T) {
	const w, h = 16, 16
	const srcPad = 32 // generous padding for tap headroom across scaled steps
	src := scaledTestSourcePlane(w+srcPad, h+srcPad, 1, 0x1234)

	type tc struct {
		name                                         string
		refX0, refY0, subpelX, xStep, subpelY, yStep int
	}
	cases := []tc{
		{"identity", 8, 8, 0, ScaleSubpelScale, 0, ScaleSubpelScale},
		{"halfStep_zeroSubpel", 8, 8, 0, ScaleSubpelScale / 2, 0, ScaleSubpelScale / 2}, // 2x upscale
		{"halfStep_halfSubpel", 8, 8, ScaleSubpelScale / 2, ScaleSubpelScale / 2, ScaleSubpelScale / 2, ScaleSubpelScale / 2},
		{"doubleStep_zeroSubpel", 8, 8, 0, 2 * ScaleSubpelScale, 0, 2 * ScaleSubpelScale}, // 2x downscale
		{"3over4Step_q4subpel", 8, 8, 256, (ScaleSubpelScale * 3) / 4, 192, (ScaleSubpelScale * 3) / 4},
		{"asymmetric", 8, 8, 64, ScaleSubpelScale - 13, 320, ScaleSubpelScale + 47},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, filterX := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
				for _, filterY := range []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear} {
					xTable, err := SubpelKernelTableFor(filterX, w)
					if err != nil {
						t.Fatalf("xTable: %v", err)
					}
					yTable, err := SubpelKernelTableFor(filterY, h)
					if err != nil {
						t.Fatalf("yTable: %v", err)
					}
					got, _ := testPlane(w, h, 1, w)
					want, _ := testPlane(w, h, 1, w)
					startX := int64(c.refX0)*ScaleSubpelScale + int64(c.subpelX)
					startY := int64(c.refY0)*ScaleSubpelScale + int64(c.subpelY)
					if err := ConvolveScale2D8ClampedWithScratch(got, src, 0, 0, w, h, startX, int64(c.xStep), startY, int64(c.yStep), xTable, yTable, nil); err != nil {
						t.Fatalf("%s X=%d Y=%d: %v", c.name, filterX, filterY, err)
					}
					libaomConvolveScale2DRef(want, src, c.refX0, c.refY0, 0, 0, w, h,
						c.subpelX, c.xStep, c.subpelY, c.yStep, xTable, yTable,
						round0Bits, round1Bits)
					comparePlanes(t, c.name+"/"+filterName(filterX)+"/"+filterName(filterY), got, want, w, h, 1)
				}
			}
		})
	}
}

func TestConvolveScale2D8ClampedMatchesLibaomReferenceAtEdge(t *testing.T) {
	const w, h = 8, 8
	src := scaledTestSourcePlane(w, h, 1, 0xfeed)
	// Position the block so the leftmost taps fall outside the source — only
	// the *Clamped* variant should handle this.
	const refX0, refY0 = 0, 0
	startX := int64(refX0) * ScaleSubpelScale
	startY := int64(refY0) * ScaleSubpelScale

	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
	if err != nil {
		t.Fatalf("xTable: %v", err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, h)
	if err != nil {
		t.Fatalf("yTable: %v", err)
	}
	got, _ := testPlane(w, h, 1, w)

	// Bordered reference for the libaom-shape reference impl: replicate src
	// edges into a padded buffer.
	const pad = 12
	padded, _ := testPlane(w+2*pad, h+2*pad, 1, w+2*pad)
	for y := range h + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= h {
			sy = h - 1
		}
		for x := range w + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= w {
				sx = w - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}

	want, _ := testPlane(w, h, 1, w)
	libaomConvolveScale2DRef(want, padded, pad+refX0, pad+refY0, 0, 0, w, h,
		0, ScaleSubpelScale, 0, ScaleSubpelScale, xTable, yTable, round0Bits, round1Bits)
	if err := ConvolveScale2D8ClampedWithScratch(got, src, 0, 0, w, h, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, nil); err != nil {
		t.Fatalf("ConvolveScale2D8ClampedWithScratch: %v", err)
	}
	comparePlanes(t, "clamped_edge", got, want, w, h, 1)
}

func TestConvolveScale2DHighBDMatchesLibaomReference(t *testing.T) {
	const w, h = 16, 16
	const srcPad = 32
	for _, bd := range []int{10, 12} {
		bd := bd
		t.Run(fmt.Sprintf("%dbit", bd), func(t *testing.T) {
			src := highBDPlane(w+srcPad, h+srcPad, 0x9999, bd)
			round0, round1 := highBDRoundBits(uint8(bd))

			xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
			if err != nil {
				t.Fatalf("xTable: %v", err)
			}
			yTable, err := SubpelKernelTableFor(InterpEightTapSmooth, h)
			if err != nil {
				t.Fatalf("yTable: %v", err)
			}

			for _, step := range []int{ScaleSubpelScale, ScaleSubpelScale / 2, 2 * ScaleSubpelScale, ScaleSubpelScale - 17} {
				got, _ := testPlane(w, h, 2, w*2)
				want, _ := testPlane(w, h, 2, w*2)
				const refX0, refY0 = 8, 8
				const subpelX, subpelY = 31, 47
				startX := int64(refX0)*ScaleSubpelScale + int64(subpelX)
				startY := int64(refY0)*ScaleSubpelScale + int64(subpelY)
				if err := ConvolveScale2DHighBDClampedWithScratch(got, src, uint8(bd), 0, 0, w, h, startX, int64(step), startY, int64(step), xTable, yTable, nil); err != nil {
					t.Fatalf("step=%d: %v", step, err)
				}
				libaomConvolveScale2DHighBDRef(want, src, bd, refX0, refY0, 0, 0, w, h,
					subpelX, step, subpelY, step, xTable, yTable, round0, round1)
				comparePlanes(t, fmt.Sprintf("hb_%dbit_step_%d", bd, step), got, want, w, h, 2)
			}
		})
	}
}

func TestConvolveScale2DHighBDClampedMatchesSuperResReference(t *testing.T) {
	const srcW, srcH = 160, 128
	const curW, curH = 107, 128
	for _, bd := range []int{10, 12} {
		bd := bd
		t.Run(fmt.Sprintf("%dbit", bd), func(t *testing.T) {
			src := highBDPlane(srcW, srcH, 0x5eed, bd)
			round0, round1 := highBDRoundBits(uint8(bd))
			sf, err := NewScaleFactors(srcW, srcH, curW, curH)
			if err != nil {
				t.Fatalf("NewScaleFactors: %v", err)
			}

			const pad = 24
			padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 2, (srcW+2*pad)*2)
			for y := range srcH + 2*pad {
				sy := y - pad
				if sy < 0 {
					sy = 0
				} else if sy >= srcH {
					sy = srcH - 1
				}
				for x := range srcW + 2*pad {
					sx := x - pad
					if sx < 0 {
						sx = 0
					} else if sx >= srcW {
						sx = srcW - 1
					}
					setSample(padded, 2, x, y, getSample(src, 2, sx, sy))
				}
			}

			for _, tc := range []struct {
				name           string
				x, y           int
				blockW, blockH int
				mv             Vector
				filterX        InterpFilter
				filterY        InterpFilter
			}{
				{name: "top_left", x: 0, y: 0, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
				{name: "middle", x: 48, y: 0, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
				{name: "right_edge", x: curW - 16, y: 0, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
				{name: "bottom_right", x: curW - 16, y: curH - 16, blockW: 16, blockH: 16, filterX: InterpEightTapRegular, filterY: InterpEightTapSmooth},
				{name: "moving_obmc_area", x: 96, y: 16, blockW: 8, blockH: 16, mv: Vector{Col: 28, Row: 24}, filterX: InterpEightTapRegular, filterY: InterpEightTapRegular},
				{name: "moving_direct_area", x: 96, y: 0, blockW: 11, blockH: 16, mv: Vector{Col: 28, Row: 24}, filterX: InterpEightTapRegular, filterY: InterpEightTapRegular},
			} {
				xTable, err := SubpelKernelTableFor(tc.filterX, tc.blockW)
				if err != nil {
					t.Fatalf("%s xTable: %v", tc.name, err)
				}
				yTable, err := SubpelKernelTableFor(tc.filterY, tc.blockH)
				if err != nil {
					t.Fatalf("%s yTable: %v", tc.name, err)
				}
				startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(tc.x, tc.y, tc.mv, false, false)
				if err != nil {
					t.Fatalf("%s ScaledBlockOrigin: %v", tc.name, err)
				}
				got, _ := testPlane(tc.blockW, tc.blockH, 2, tc.blockW*2)
				if err := ConvolveScale2DHighBDClampedWithScratch(got, src, uint8(bd), 0, 0, tc.blockW, tc.blockH, startX, xStep, startY, yStep, xTable, yTable, nil); err != nil {
					t.Fatalf("%s ConvolveScale2DHighBDClampedWithScratch: %v", tc.name, err)
				}

				intX := int(startX >> ScaleSubpelBits)
				intY := int(startY >> ScaleSubpelBits)
				subpelX := int(startX & ScaleSubpelMask)
				subpelY := int(startY & ScaleSubpelMask)
				want, _ := testPlane(tc.blockW, tc.blockH, 2, tc.blockW*2)
				libaomConvolveScale2DHighBDRef(want, padded, bd, pad+intX, pad+intY, 0, 0, tc.blockW, tc.blockH,
					subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0, round1)
				comparePlanes(t, fmt.Sprintf("superres%d_%s", bd, tc.name), got, want, tc.blockW, tc.blockH, 2)
			}
		})
	}
}

// TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FrameTopLeft pins the
// scaled 8-tap convolver against libaom for the exact SVC L2T1 spatial=1
// frame-top-left block: a 1280x720 enhancement layer references the 640x360
// base, so the start position is libaom's biased (-224, -224) Q10 and the
// step is exactly 0.5 sample per output pel. The leftmost / topmost taps
// fall outside the source plane and must be edge-replicated, exercising the
// Clamped variant against a libaom-shape padded reference. Locks the bias
// (3d13790) + clamping behaviour against silent SVC L2T1 spatial=1 drift.
func TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FrameTopLeft(t *testing.T) {
	const blockW, blockH = 16, 16
	const srcW, srcH = 64, 64
	src := scaledTestSourcePlane(srcW, srcH, 1, 0x5717)

	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	if startX >= 0 || startY >= 0 {
		t.Fatalf("expected biased negative startX/startY, got startX=%d startY=%d", startX, startY)
	}

	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, blockW)
	if err != nil {
		t.Fatalf("xTable: %v", err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapRegular, blockH)
	if err != nil {
		t.Fatalf("yTable: %v", err)
	}

	got, _ := testPlane(blockW, blockH, 1, blockW)
	if err := ConvolveScale2D8ClampedWithScratch(got, src, 0, 0, blockW, blockH, startX, xStep, startY, yStep, xTable, yTable, nil); err != nil {
		t.Fatalf("ConvolveScale2D8ClampedWithScratch: %v", err)
	}

	const pad = 16
	padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 1, srcW+2*pad)
	for y := range srcH + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= srcH {
			sy = srcH - 1
		}
		for x := range srcW + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}
	intX := int(startX >> ScaleSubpelBits)
	intY := int(startY >> ScaleSubpelBits)
	subpelX := int(startX & ScaleSubpelMask)
	subpelY := int(startY & ScaleSubpelMask)

	want, _ := testPlane(blockW, blockH, 1, blockW)
	libaomConvolveScale2DRef(want, padded, pad+intX, pad+intY, 0, 0, blockW, blockH,
		subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0Bits, round1Bits)

	comparePlanes(t, "svc_l2t1_top_left", got, want, blockW, blockH, 1)
}

// TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1RightEdge pins the analogous
// right-edge case: the last column block of an SVC L2T1 spatial=1 frame
// references the rightmost columns of the base plane, with several taps
// overhanging the source plane and replicated.
func TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1RightEdge(t *testing.T) {
	const blockW, blockH = 16, 16
	const srcW, srcH = 64, 64
	src := scaledTestSourcePlane(srcW, srcH, 1, 0x9b0a)

	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	dstX := 2*srcW - blockW
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(dstX, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}

	xTable, err := SubpelKernelTableFor(InterpEightTapRegular, blockW)
	if err != nil {
		t.Fatalf("xTable: %v", err)
	}
	yTable, err := SubpelKernelTableFor(InterpEightTapSmooth, blockH)
	if err != nil {
		t.Fatalf("yTable: %v", err)
	}

	got, _ := testPlane(blockW, blockH, 1, blockW)
	if err := ConvolveScale2D8ClampedWithScratch(got, src, 0, 0, blockW, blockH, startX, xStep, startY, yStep, xTable, yTable, nil); err != nil {
		t.Fatalf("ConvolveScale2D8ClampedWithScratch: %v", err)
	}

	const pad = 24
	padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 1, srcW+2*pad)
	for y := range srcH + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= srcH {
			sy = srcH - 1
		}
		for x := range srcW + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}
	intX := int(startX >> ScaleSubpelBits)
	intY := int(startY >> ScaleSubpelBits)
	subpelX := int(startX & ScaleSubpelMask)
	subpelY := int(startY & ScaleSubpelMask)

	want, _ := testPlane(blockW, blockH, 1, blockW)
	libaomConvolveScale2DRef(want, padded, pad+intX, pad+intY, 0, 0, blockW, blockH,
		subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0Bits, round1Bits)

	comparePlanes(t, "svc_l2t1_right_edge", got, want, blockW, blockH, 1)
}

// TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FuzzGrid sweeps the L2T1
// scaled convolver across a coarse grid of destination positions, motion
// vectors, and filter combinations, comparing every pixel against the
// libaom-faithful reference implementation. Any drift in the dispatched
// scale-factor / sub-pel split / kernel-table lookup would manifest as a
// non-matching sample here.
func TestConvolveScale2D8ClampedMatchesLibaomSVCL2T1FuzzGrid(t *testing.T) {
	const srcW, srcH = 96, 96
	src := scaledTestSourcePlane(srcW, srcH, 1, 0xa5a5)
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	const blockW, blockH = 16, 16

	const pad = 32
	padded, _ := testPlane(srcW+2*pad, srcH+2*pad, 1, srcW+2*pad)
	for y := range srcH + 2*pad {
		sy := y - pad
		if sy < 0 {
			sy = 0
		} else if sy >= srcH {
			sy = srcH - 1
		}
		for x := range srcW + 2*pad {
			sx := x - pad
			if sx < 0 {
				sx = 0
			} else if sx >= srcW {
				sx = srcW - 1
			}
			padded.Pix[y*padded.Stride+x] = src.Pix[sy*src.Stride+sx]
		}
	}

	dstPositions := []struct{ x, y int }{
		{0, 0},
		{16, 0},
		{0, 16},
		{32, 32},
		{2*srcW - blockW, 0},
		{0, 2*srcH - blockH},
		{2*srcW - blockW, 2*srcH - blockH},
	}
	mvs := []Vector{
		{Row: 0, Col: 0},
		{Row: 4, Col: 4},
		{Row: -3, Col: 5},
		{Row: -8, Col: 0},
	}
	filters := []InterpFilter{InterpEightTapRegular, InterpEightTapSmooth, InterpMultiTapSharp, InterpBilinear}

	for _, dst := range dstPositions {
		for _, mv := range mvs {
			for _, fX := range filters {
				for _, fY := range filters {
					fX, fY := fX, fY
					name := fmt.Sprintf("dst%dx%d_mv(%d,%d)_%s/%s", dst.x, dst.y, mv.Row, mv.Col, filterName(fX), filterName(fY))
					t.Run(name, func(t *testing.T) {
						startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(dst.x, dst.y, mv, false, false)
						if err != nil {
							t.Fatalf("ScaledBlockOrigin: %v", err)
						}
						xTable, err := SubpelKernelTableFor(fX, blockW)
						if err != nil {
							t.Fatalf("xTable: %v", err)
						}
						yTable, err := SubpelKernelTableFor(fY, blockH)
						if err != nil {
							t.Fatalf("yTable: %v", err)
						}
						got, _ := testPlane(blockW, blockH, 1, blockW)
						if err := ConvolveScale2D8ClampedWithScratch(got, src, 0, 0, blockW, blockH, startX, xStep, startY, yStep, xTable, yTable, nil); err != nil {
							t.Fatalf("ConvolveScale2D8ClampedWithScratch: %v", err)
						}
						intX := int(startX >> ScaleSubpelBits)
						intY := int(startY >> ScaleSubpelBits)
						subpelX := int(startX & ScaleSubpelMask)
						subpelY := int(startY & ScaleSubpelMask)
						want, _ := testPlane(blockW, blockH, 1, blockW)
						libaomConvolveScale2DRef(want, padded, pad+intX, pad+intY, 0, 0, blockW, blockH,
							subpelX, int(xStep), subpelY, int(yStep), xTable, yTable, round0Bits, round1Bits)
						comparePlanes(t, name, got, want, blockW, blockH, 1)
					})
				}
			}
		}
	}
}

func TestSubpelKernelTableForInvalidFilter(t *testing.T) {
	if _, err := SubpelKernelTableFor(interpFilterCount, 16); err == nil {
		t.Fatal("expected ErrInvalidMotion for invalid filter")
	}
}

func TestConvolveScale2D8ZeroAllocation(t *testing.T) {
	src := scaledTestSourcePlane(32, 32, 1, 0x55)
	dst, _ := testPlane(16, 16, 1, 16)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 16)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 16)
	startX := int64(8) * ScaleSubpelScale
	startY := int64(8) * ScaleSubpelScale
	var scratch ScaledConvolveScratch

	allocs := testing.AllocsPerRun(50, func() {
		if err := ConvolveScale2D8ClampedWithScratch(dst, src, 0, 0, 16, 16, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, nil); err != nil {
			t.Fatalf("ConvolveScale2D8ClampedWithScratch(nil): %v", err)
		}
		if err := ConvolveScale2D8ClampedWithScratch(dst, src, 0, 0, 16, 16, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			t.Fatalf("ConvolveScale2D8ClampedWithScratch: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("expected 0 allocs, got %v", allocs)
	}
}

func BenchmarkConvolveScale2DHighBDClampedInterior32(b *testing.B) {
	const bd = 10
	src := highBDPlane(96, 96, 0xace, bd)
	dst, _ := testPlane(32, 32, 2, 64)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16) * ScaleSubpelScale
	startY := int64(16) * ScaleSubpelScale
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2DHighBDClampedWithScratch(dst, src, bd, 0, 0, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2DHighBDClampedEdge32(b *testing.B) {
	const bd = 10
	src := highBDPlane(40, 40, 0xace, bd)
	dst, _ := testPlane(32, 32, 2, 64)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2DHighBDClampedWithScratch(dst, src, bd, 0, 0, 32, 32, 0, ScaleSubpelScale, 0, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2D8ClampedInterior32(b *testing.B) {
	src := scaledTestSourcePlane(96, 96, 1, 0x55)
	dst, _ := testPlane(32, 32, 1, 32)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16) * ScaleSubpelScale
	startY := int64(16) * ScaleSubpelScale

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2D8ClampedWithScratch(dst, src, 0, 0, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2D8ClampedInterior32WithScratch(b *testing.B) {
	src := scaledTestSourcePlane(96, 96, 1, 0x55)
	dst, _ := testPlane(32, 32, 1, 32)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16) * ScaleSubpelScale
	startY := int64(16) * ScaleSubpelScale
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2D8ClampedWithScratch(dst, src, 0, 0, 32, 32, startX, ScaleSubpelScale, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2D8ClampedYIdentity32WithScratch(b *testing.B) {
	src := scaledTestSourcePlane(96, 96, 1, 0x55)
	dst, _ := testPlane(32, 32, 1, 32)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16)*ScaleSubpelScale + ScaleSubpelScale/4
	startY := int64(16) * ScaleSubpelScale
	xStep := int64(ScaleSubpelScale * 3 / 4)
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2D8ClampedWithScratch(dst, src, 0, 0, 32, 32, startX, xStep, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConvolveScale2DHighBDClampedYIdentity32WithScratch(b *testing.B) {
	const bd = 10
	src := highBDPlane(96, 96, 0xace, bd)
	dst, _ := testPlane(32, 32, 2, 64)
	xTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	yTable, _ := SubpelKernelTableFor(InterpEightTapRegular, 32)
	startX := int64(16)*ScaleSubpelScale + ScaleSubpelScale/4
	startY := int64(16) * ScaleSubpelScale
	xStep := int64(ScaleSubpelScale * 3 / 4)
	var scratch ScaledConvolveScratch

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ConvolveScale2DHighBDClampedWithScratch(dst, src, bd, 0, 0, 32, 32, startX, xStep, startY, ScaleSubpelScale, xTable, yTable, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func filterName(f InterpFilter) string {
	switch f {
	case InterpEightTapRegular:
		return "EightTapRegular"
	case InterpEightTapSmooth:
		return "EightTapSmooth"
	case InterpMultiTapSharp:
		return "MultiTapSharp"
	case InterpBilinear:
		return "Bilinear"
	default:
		return "unknown"
	}
}

func comparePlanes(t *testing.T, name string, got, want frame.Plane, w, h, bytesPerSample int) {
	t.Helper()
	for y := range h {
		for x := range w {
			g := getSample(got, bytesPerSample, x, y)
			wv := getSample(want, bytesPerSample, x, y)
			if g != wv {
				t.Fatalf("%s sample(%d,%d)=%d want %d", name, x, y, g, wv)
			}
		}
	}
}

func clearPlane(p frame.Plane) {
	for i := range p.Pix {
		p.Pix[i] = 0
	}
}

func highBDPlane(width, height, seed, bd int) frame.Plane {
	plane, _ := testPlane(width, height, 2, width*2)
	state := uint32(0x55aa55aa | uint32(seed))
	max := uint16((1 << bd) - 1)
	for y := range height {
		for x := range width {
			state = state*1664525 + 1013904223
			v := uint16(state>>16) & max
			setSample(plane, 2, x, y, v)
		}
	}
	return plane
}

// copyPlaneBlockClampedNaive is the pre-optimization per-pixel-clamp reference
// (emu_edge), kept here as the differential oracle for the region-split fast
// path in copyPlaneBlockClamped.
func copyPlaneBlockClampedNaive(dst frame.Plane, ref frame.Plane, bps, dstX, dstY, refX, refY, width, height int) {
	for y := range height {
		sy := clampInt(refY+y, 0, ref.Height-1)
		for x := range width {
			sx := clampInt(refX+x, 0, ref.Width-1)
			switch bps {
			case 1:
				dst.Pix[(dstY+y)*dst.Stride+dstX+x] = ref.Pix[sy*ref.Stride+sx]
			case 2:
				do := (dstY+y)*dst.Stride + (dstX+x)*2
				ro := sy*ref.Stride + sx*2
				dst.Pix[do] = ref.Pix[ro]
				dst.Pix[do+1] = ref.Pix[ro+1]
			}
		}
	}
}

func makeRefPlane(w, h, bps int) frame.Plane {
	stride := w * bps
	pix := make([]byte, stride*h)
	for i := range pix {
		// Deterministic non-uniform content so any region mismatch shows up.
		pix[i] = byte((i*37 + 11) & 0xff)
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

func TestCopyPlaneBlockClampedMatchesNaive(t *testing.T) {
	refDims := [][2]int{{1, 1}, {2, 3}, {8, 8}, {16, 9}, {13, 7}}
	blocks := [][2]int{{1, 1}, {4, 4}, {8, 8}, {5, 3}, {16, 16}}
	// Offsets span fully-left, straddle-left, in-bounds, straddle-right,
	// fully-right, and the exact-edge cases in both axes.
	offs := []int{-20, -9, -3, -1, 0, 1, 3, 7, 12, 20}
	for _, bps := range []int{1, 2} {
		for _, rd := range refDims {
			ref := makeRefPlane(rd[0], rd[1], bps)
			for _, blk := range blocks {
				w, h := blk[0], blk[1]
				dstStride := (w + 4) * bps
				got := frame.Plane{Pix: make([]byte, dstStride*(h+2)), Stride: dstStride, Width: w + 4, Height: h + 2}
				want := frame.Plane{Pix: make([]byte, dstStride*(h+2)), Stride: dstStride, Width: w + 4, Height: h + 2}
				for _, rx := range offs {
					for _, ry := range offs {
						for i := range got.Pix {
							got.Pix[i] = 0xAA
							want.Pix[i] = 0xAA
						}
						copyPlaneBlockClampedNaive(want, ref, bps, 1, 1, rx, ry, w, h)
						if err := copyPlaneBlockClamped(got, ref, bps, 1, 1, rx, ry, w, h); err != nil {
							t.Fatalf("bps=%d ref=%dx%d blk=%dx%d rx=%d ry=%d: err %v", bps, rd[0], rd[1], w, h, rx, ry, err)
						}
						for i := range want.Pix {
							if got.Pix[i] != want.Pix[i] {
								t.Fatalf("bps=%d ref=%dx%d blk=%dx%d rx=%d ry=%d: byte %d got %d want %d",
									bps, rd[0], rd[1], w, h, rx, ry, i, got.Pix[i], want.Pix[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestEmuEdge8MatchesClampedReference proves emuEdge8 materializes exactly
// the samples a per-pixel clamped load would produce (dav1d src/mc_tmpl.c
// emu_edge_c semantics) for windows overhanging any combination of plane
// edges, including windows entirely outside the plane.
func TestEmuEdge8MatchesClampedReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xed6e))
	for trial := 0; trial < 2000; trial++ {
		iw := 1 + rng.Intn(64)
		ih := 1 + rng.Intn(64)
		stride := iw + rng.Intn(8)
		ref, _ := testPlane(iw, ih, 1, stride)
		for i := range ref.Pix {
			ref.Pix[i] = byte(rng.Intn(256))
		}
		bw := 1 + rng.Intn(emuEdgeStride)
		bh := 1 + rng.Intn(emuEdgeRows)
		// Window origin anywhere from fully above/left to fully below/right.
		x := rng.Intn(iw+2*bw+8) - bw - 4
		y := rng.Intn(ih+2*bh+8) - bh - 4

		var edge [emuEdgeStride * emuEdgeRows]byte
		emuEdge8(bw, bh, ref, x, y, edge[:], emuEdgeStride)

		for yy := 0; yy < bh; yy++ {
			sy := clampInt(y+yy, 0, ih-1)
			for xx := 0; xx < bw; xx++ {
				sx := clampInt(x+xx, 0, iw-1)
				want := ref.Pix[sy*ref.Stride+sx]
				got := edge[yy*emuEdgeStride+xx]
				if got != want {
					t.Fatalf("trial=%d plane=%dx%d window=%dx%d@(%d,%d) sample=(%d,%d): emu=%d clamped=%d",
						trial, iw, ih, bw, bh, x, y, xx, yy, got, want)
				}
			}
		}
	}
}

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Vector", size: unsafe.Sizeof(Vector{}), max: 4},
		{name: "ScaleFactors", size: unsafe.Sizeof(ScaleFactors{}), max: 10},
		{name: "CompoundConvBuf", size: unsafe.Sizeof(CompoundConvBuf{}), max: compoundMaxConvSamples*2 + 2},
	}
	for _, tc := range tests {
		t.Logf("%s size=%d max=%d", tc.name, tc.size, tc.max)
		if tc.size > tc.max {
			t.Fatalf("%s grew to %d bytes, max %d", tc.name, tc.size, tc.max)
		}
	}
}

func put8UVTestPlane(rng *rand.Rand, w, h, stride int) frame.Plane {
	pix := make([]byte, stride*h)
	for i := range pix {
		pix[i] = byte(rng.Intn(256))
	}
	return frame.Plane{Pix: pix, Stride: stride, Width: w, Height: h}
}

// TestPut8UVMatchesPerPlanePut8 checks the paired chroma put against one Put8
// per plane: whenever Put8UV reports success both planes hold exactly the
// per-plane samples, and on planes of one geometry it succeeds exactly when
// both per-plane puts do. The cases cover every kernel shape (copy, h, v, hv,
// widths 2 to 64, the eight-tap sharp filter), sources inside the plane and
// clamped at each edge (the two planes then share the edge buffer in turn),
// destinations at and past the plane edge, and reference planes whose
// geometries differ.
func TestPut8UVMatchesPerPlanePut8(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const pw, ph = 96, 80
	type geom struct{ w, h, stride int }
	for _, g := range []struct {
		name   string
		u, v   geom
		dv, du geom
	}{
		{"same", geom{pw, ph, 112}, geom{pw, ph, 112}, geom{pw, ph, 128}, geom{pw, ph, 128}},
		{"refstride", geom{pw, ph, 112}, geom{pw, ph, 120}, geom{pw, ph, 128}, geom{pw, ph, 128}},
		{"dstsize", geom{pw, ph, 112}, geom{pw, ph, 112}, geom{pw, ph - 8, 128}, geom{pw, ph, 128}},
	} {
		refU := put8UVTestPlane(rng, g.u.w, g.u.h, g.u.stride)
		refV := put8UVTestPlane(rng, g.v.w, g.v.h, g.v.stride)
		for _, sz := range [][2]int{{2, 4}, {2, 16}, {4, 4}, {4, 16}, {8, 8}, {8, 32}, {16, 16}, {32, 8}, {64, 64}} {
			w, h := sz[0], sz[1]
			for _, f := range []InterpFilters{RegularFilters, {X: InterpEightTapSmooth, Y: InterpBilinear}, {X: InterpMultiTapSharp, Y: InterpEightTapRegular}} {
				for _, sub := range [][2]int{{0, 0}, {5, 0}, {0, 11}, {5, 11}, {8, 8}} {
					for _, pos := range [][4]int{{8, 8, 20, 12}, {0, 0, -5, -3}, {pw - w, ph - h, pw - w + 3, ph - h + 2}, {pw - w + 4, 0, 10, 10}, {16, 8, -40, ph + 10}} {
						dx, dy, rx, ry := pos[0], pos[1], pos[2], pos[3]
						name := fmt.Sprintf("%s/%dx%d/f%d%d/s%d,%d/d%d,%d/r%d,%d", g.name, w, h, f.X, f.Y, sub[0], sub[1], dx, dy, rx, ry)
						var s1, s2 ConvolveScratch
						wantU := put8UVTestPlane(rng, g.du.w, g.du.h, g.du.stride)
						wantV := put8UVTestPlane(rng, g.dv.w, g.dv.h, g.dv.stride)
						gotU := frame.Plane{Pix: bytes.Clone(wantU.Pix), Stride: wantU.Stride, Width: wantU.Width, Height: wantU.Height}
						gotV := frame.Plane{Pix: bytes.Clone(wantV.Pix), Stride: wantV.Stride, Width: wantV.Width, Height: wantV.Height}
						okU := Put8(&wantU, &refU, dx, dy, rx, ry, w, h, w, h, sub[0], sub[1], f, &s1)
						okV := Put8(&wantV, &refV, dx, dy, rx, ry, w, h, w, h, sub[0], sub[1], f, &s1)
						ok := Put8UV(&gotU, &gotV, &refU, &refV, dx, dy, rx, ry, w, h, w, h, sub[0], sub[1], f, &s2)
						if g.name == "same" && ok != (okU && okV) {
							t.Fatalf("%s: Put8UV ok=%v, per-plane %v/%v", name, ok, okU, okV)
						}
						if !ok {
							continue
						}
						if !okU || !okV {
							t.Fatalf("%s: Put8UV wrote a block a per-plane put declines (%v/%v)", name, okU, okV)
						}
						if !bytes.Equal(gotU.Pix, wantU.Pix) || !bytes.Equal(gotV.Pix, wantV.Pix) {
							t.Fatalf("%s: samples differ from the per-plane puts", name)
						}
					}
				}
			}
		}
	}
}

func TestNewScaleFactorsIdentity(t *testing.T) {
	sf, err := NewScaleFactors(1280, 720, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	if !sf.Identity {
		t.Fatalf("expected identity scaling, got Identity=false")
	}
	if got, want := sf.XScaleFP, uint16(1<<RefScaleShift); got != want {
		t.Errorf("XScaleFP=%d want %d", got, want)
	}
	if got, want := sf.YScaleFP, uint16(1<<RefScaleShift); got != want {
		t.Errorf("YScaleFP=%d want %d", got, want)
	}
	if got, want := sf.XStepQN, uint16(1<<ScaleSubpelBits); got != want {
		t.Errorf("XStepQN=%d want %d", got, want)
	}
	if got, want := sf.YStepQN, uint16(1<<ScaleSubpelBits); got != want {
		t.Errorf("YStepQN=%d want %d", got, want)
	}
}

func TestNewScaleFactorsSVCEnhancementUpscale(t *testing.T) {
	// libaom SVC L2T1: S=1 enhancement layer (1280x720) references the S=0
	// base layer (640x360). For the enhancement-layer block consuming a
	// half-size base reference, x_scale_fp == 1/2 in Q14.
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	if sf.Identity {
		t.Fatalf("expected non-identity scaling for SVC enhancement layer")
	}
	wantScale := uint16(((640 << RefScaleShift) + 1280/2) / 1280)
	if sf.XScaleFP != wantScale {
		t.Errorf("XScaleFP=%d want %d", sf.XScaleFP, wantScale)
	}
	if sf.YScaleFP != wantScale {
		t.Errorf("YScaleFP=%d want %d", sf.YScaleFP, wantScale)
	}
	wantStep := uint16(((640 << ScaleSubpelBits) + 1280/2) / 1280)
	if sf.XStepQN != wantStep {
		t.Errorf("XStepQN=%d want %d", sf.XStepQN, wantStep)
	}
	if sf.YStepQN != wantStep {
		t.Errorf("YStepQN=%d want %d", sf.YStepQN, wantStep)
	}
	// For an exact 2:1 ratio the step is exactly 0.5 in Q10.
	if sf.XStepQN != uint16(ScaleSubpelScale/2) {
		t.Errorf("XStepQN=%d, expected exact 0.5 Q10 step=%d", sf.XStepQN, ScaleSubpelScale/2)
	}
}

func TestNewScaleFactorsRejectsOutOfRange(t *testing.T) {
	tests := []struct {
		name                   string
		refW, refH, curW, curH int
	}{
		{"zero width", 0, 360, 640, 360},
		{"zero height", 640, 0, 640, 360},
		{"zero cur width", 640, 360, 0, 360},
		{"zero cur height", 640, 360, 640, 0},
		// Upscale > 16x (ref tiny, cur huge): 16 * 32 < 1280  => 512 < 1280 → reject.
		{"upscale beyond 16x width", 32, 360, 1280, 720},
		{"upscale beyond 16x height", 640, 32, 1280, 1280},
		// Downscale > 2x (ref huge, cur small): cur*2 < ref.
		{"downscale beyond 2x width", 2000, 360, 640, 360},
		{"downscale beyond 2x height", 640, 1600, 640, 720},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewScaleFactors(tc.refW, tc.refH, tc.curW, tc.curH); err == nil {
				t.Fatalf("NewScaleFactors(%d,%d,%d,%d): expected error, got nil",
					tc.refW, tc.refH, tc.curW, tc.curH)
			}
		})
	}
}

func TestNewScaleFactorsStorageBounds(t *testing.T) {
	cases := []struct {
		name                   string
		refW, refH, curW, curH int
		wantScale              uint16
		wantStep               uint16
	}{
		{
			name:      "max downscale",
			refW:      2,
			refH:      2,
			curW:      1,
			curH:      1,
			wantScale: 2 << RefScaleShift,
			wantStep:  2 << ScaleSubpelBits,
		},
		{
			name:      "max upscale",
			refW:      1,
			refH:      1,
			curW:      16,
			curH:      16,
			wantScale: 1 << (RefScaleShift - 4),
			wantStep:  1 << (ScaleSubpelBits - 4),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sf, err := NewScaleFactors(tc.refW, tc.refH, tc.curW, tc.curH)
			if err != nil {
				t.Fatalf("NewScaleFactors: %v", err)
			}
			if sf.XScaleFP != tc.wantScale || sf.YScaleFP != tc.wantScale ||
				sf.XStepQN != tc.wantStep || sf.YStepQN != tc.wantStep {
				t.Fatalf("scale=%d,%d step=%d,%d want scale=%d step=%d",
					sf.XScaleFP, sf.YScaleFP, sf.XStepQN, sf.YStepQN, tc.wantScale, tc.wantStep)
			}
		})
	}
}

func TestScaledBlockOriginIdentityMatchesSameSize(t *testing.T) {
	// For identity scaling the scaled origin must agree with the existing
	// same-size ReferenceOriginSubsampled path: same integer-pel reference
	// origin and same Q4 sub-pel fractional offsets.
	sf, err := NewScaleFactors(1280, 720, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	cases := []struct {
		name string
		mv   Vector
	}{
		{"zero", Vector{}},
		{"integer +1 col", Vector{Col: 1 << SubpelBits}},
		{"integer -2 row", Vector{Row: -2 << SubpelBits}},
		{"half-pel col", Vector{Col: 4}},    // 4/8 sample
		{"quarter-pel row", Vector{Row: 2}}, // 2/8 sample
		{"eighth-pel both", Vector{Row: 1, Col: 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(40, 24, tc.mv, false, false)
			if err != nil {
				t.Fatalf("ScaledBlockOrigin: %v", err)
			}
			if xStep != ScaleSubpelScale {
				t.Errorf("xStep=%d want %d", xStep, ScaleSubpelScale)
			}
			if yStep != ScaleSubpelScale {
				t.Errorf("yStep=%d want %d", yStep, ScaleSubpelScale)
			}
			refX, refY, subX, subY, err := ReferenceOriginSubsampled(40, 24, tc.mv, false, false)
			if err != nil {
				t.Fatalf("ReferenceOriginSubsampled: %v", err)
			}
			gotIntX, gotSubQ4X := SplitScaledPosition(startX)
			gotIntY, gotSubQ4Y := SplitScaledPosition(startY)
			if int(gotIntX) != refX {
				t.Errorf("intX=%d want %d", gotIntX, refX)
			}
			if int(gotIntY) != refY {
				t.Errorf("intY=%d want %d", gotIntY, refY)
			}
			if gotSubQ4X != subX {
				t.Errorf("subX(Q4)=%d want %d", gotSubQ4X, subX)
			}
			if gotSubQ4Y != subY {
				t.Errorf("subY(Q4)=%d want %d", gotSubQ4Y, subY)
			}
		})
	}
}

func TestScaledBlockOriginUpscaledRef(t *testing.T) {
	// Upscaled current frame (1280x720) referencing half-size base (640x360):
	// each output pel maps to 0.5 source pel, so the stepper is 0.5 Q10 per
	// output pel. The destination block origin (16, 8) plus libaom's
	// av1_scaled_x bias (off = (x_scale_fp - REF_NO_SCALE) * SUBPEL_OFF; for
	// 2:1 that bias is half a source pel) lands the first tap at source
	// integer column 7 with Q4 phase 12, not at "src 8.0 / phase 0". This
	// libaom behaviour is what SVC L2T1 spatial=1 enhancement layers expect.
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(16, 8, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	if xStep != ScaleSubpelScale/2 {
		t.Errorf("xStep=%d want %d", xStep, ScaleSubpelScale/2)
	}
	if yStep != ScaleSubpelScale/2 {
		t.Errorf("yStep=%d want %d", yStep, ScaleSubpelScale/2)
	}
	intX, fracQ4X := SplitScaledPosition(startX)
	intY, fracQ4Y := SplitScaledPosition(startY)
	if intX != 7 || fracQ4X != 12 {
		t.Errorf("startX split=(%d, %d) want (7, 12)", intX, fracQ4X)
	}
	if intY != 3 || fracQ4Y != 12 {
		t.Errorf("startY split=(%d, %d) want (3, 12)", intY, fracQ4Y)
	}
}

func TestScaledBlockOriginCarriesNegativeMV(t *testing.T) {
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	// MV = (-8, 0) in Q3: 1 luma pel to the left. With 2:1 downscale that's
	// 0.5 source pel less. Combined with libaom's av1_scaled_x bias for the
	// 2:1 ratio, the start lands at source integer column 7 with Q4 phase 4
	// — a half-pel offset to the LEFT of phase 12 (the zero-MV case at
	// dstX=16 — see TestScaledBlockOriginUpscaledRef).
	startX, _, _, _, err := sf.ScaledBlockOrigin(16, 8, Vector{Col: -8}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	intX, fracQ4X := SplitScaledPosition(startX)
	if intX != 7 || fracQ4X != 4 {
		t.Errorf("start=(%d, %d) want (7, 4)", intX, fracQ4X)
	}
}

func TestScaledBlockOriginRejectsZeroFactors(t *testing.T) {
	var sf ScaleFactors
	if _, _, _, _, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false); err == nil {
		t.Fatal("expected error for zero ScaleFactors")
	}
}

// TestScaledBlockOriginLibaomBiasFrameTopLeft pins the bias-aware Q10 origin
// for the very first block of an SVC L2T1 enhancement layer (1280x720
// referencing the 640x360 base): a 128x128 mi-(0,0) NEARESTMV block with
// zero MV. Before the libaom bias fix, this case returned (0,0) at xStep=512
// with no SCALE_EXTRA_OFF, which drove the convolver to filter phase 0 at
// each output integer column. libaom's av1_scaled_x bias instead yields a
// negative integer-pel origin with Q4 phase 12 — the half-pel offset the
// convolver expects for non-identity ratios.
func TestScaledBlockOriginLibaomBiasFrameTopLeft(t *testing.T) {
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	if xStep != ScaleSubpelScale/2 || yStep != ScaleSubpelScale/2 {
		t.Fatalf("xStep=%d yStep=%d want %d", xStep, yStep, ScaleSubpelScale/2)
	}
	// libaom for x=0, mv=0, 2:1 downscale:
	//   off = (8192-16384)*8 = -65536
	//   tval = 0 + (-65536) = -65536
	//   pos = ROUND_POWER_OF_TWO_SIGNED_64(-65536, 8) = -256
	//   startX = -256 + SCALE_EXTRA_OFF(=32) = -224
	if want := int64(-224); startX != want {
		t.Errorf("startX=%d want %d", startX, want)
	}
	if want := int64(-224); startY != want {
		t.Errorf("startY=%d want %d", startY, want)
	}
}

// TestScaledBlockOriginLibaomBiasStepsCleanly walks the first eight output
// integer columns of the same SVC L2T1 frame-top-left block and pins each
// Q10 stepper position, integer-pel index, and Q4 sub-pel filter index.
// Phase alternation (12/4/12/4...) and the negative integer-pel origin at
// the left edge are the libaom-faithful behaviour; the AV1 spec convolver
// (av1_convolve_2d_scale_c) requires these for bit-exact output.
func TestScaledBlockOriginLibaomBiasStepsCleanly(t *testing.T) {
	sf, err := NewScaleFactors(640, 360, 1280, 720)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	startX, _, xStep, _, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	cases := []struct {
		x       int
		wantPos int64
		wantInt int64
		wantQ4  int
	}{
		// startX=-224 + n*512, then >>10 / phase4 derivation:
		// x=0: -224 (intX=-1, phase=12)
		// x=1: 288  (intX=0,  phase=4)
		// x=2: 800  (intX=0,  phase=12)
		// x=3: 1312 (intX=1,  phase=4)
		// x=4: 1824 (intX=1,  phase=12)
		// x=5: 2336 (intX=2,  phase=4)
		// x=6: 2848 (intX=2,  phase=12)
		// x=7: 3360 (intX=3,  phase=4)
		{0, -224, -1, 12},
		{1, 288, 0, 4},
		{2, 800, 0, 12},
		{3, 1312, 1, 4},
		{4, 1824, 1, 12},
		{5, 2336, 2, 4},
		{6, 2848, 2, 12},
		{7, 3360, 3, 4},
	}
	for _, tc := range cases {
		pos := startX + int64(tc.x)*xStep
		if pos != tc.wantPos {
			t.Errorf("x=%d pos=%d want %d", tc.x, pos, tc.wantPos)
		}
		intPart, subQ4 := SplitScaledPosition(pos)
		if intPart != tc.wantInt || subQ4 != tc.wantQ4 {
			t.Errorf("x=%d split=(%d, %d) want (%d, %d)", tc.x, intPart, subQ4, tc.wantInt, tc.wantQ4)
		}
	}
}

// TestScaledBlockOriginLibaomBiasChromaSubsampling pins the chroma path:
// the av1_scaled_x bias applies independently of subsampling, but the MV
// component scale (×2 for luma, ×1 for subsampled chroma) survives.
func TestScaledBlockOriginLibaomBiasChromaSubsampling(t *testing.T) {
	sf, err := NewScaleFactors(320, 180, 640, 360)
	if err != nil {
		t.Fatalf("NewScaleFactors: %v", err)
	}
	// Chroma block at (0,0) with non-zero Q3 MV. Subsampled chroma uses the
	// MV components directly (×1), not ×2 like luma.
	startX, _, _, _, err := sf.ScaledBlockOrigin(0, 0, Vector{Col: 4}, true, true)
	if err != nil {
		t.Fatalf("ScaledBlockOrigin: %v", err)
	}
	// val_Q4 = 0*16 + 4*1 = 4.
	// off    = (8192-16384)*8 = -65536.
	// tval   = 4*8192 - 65536 = 32768 - 65536 = -32768.
	// pos    = ROUND_POWER_OF_TWO_SIGNED_64(-32768, 8) = -((32768+128)>>8) = -128.
	// startX = -128 + 32 = -96.
	if want := int64(-96); startX != want {
		t.Errorf("startX=%d want %d", startX, want)
	}
}

func TestSplitScaledPositionZero(t *testing.T) {
	intPart, sub := SplitScaledPosition(0)
	if intPart != 0 || sub != 0 {
		t.Fatalf("split(0)=(%d, %d) want (0, 0)", intPart, sub)
	}
}

func TestSplitScaledPositionAtOneSample(t *testing.T) {
	// Q10 position of exactly 1 sample: int=1, frac=0.
	intPart, sub := SplitScaledPosition(ScaleSubpelScale)
	if intPart != 1 || sub != 0 {
		t.Fatalf("split(1.0)=(%d, %d) want (1, 0)", intPart, sub)
	}
}

func TestSplitScaledPositionHalfSample(t *testing.T) {
	// Q10 position of 0.5 sample: int=0, frac=8/16 (Q4).
	intPart, sub := SplitScaledPosition(ScaleSubpelScale / 2)
	if intPart != 0 || sub != 8 {
		t.Fatalf("split(0.5)=(%d, %d) want (0, 8)", intPart, sub)
	}
}

func TestFullpelVectorAndReferenceOrigin(t *testing.T) {
	mv := Vector{Row: -1 << 3, Col: 2 << 3}
	if !mv.IsFullpel() {
		t.Fatal("fullpel vector reported fractional")
	}
	dx, dy, err := mv.FullpelOffset()
	if err != nil {
		t.Fatal(err)
	}
	if dx != 2 || dy != -1 {
		t.Fatalf("offset=%d,%d want 2,-1", dx, dy)
	}
	if (Vector{Col: 4}).IsFullpel() {
		t.Fatal("fractional vector reported fullpel")
	}
	if _, _, err := (Vector{Col: 4}).FullpelOffset(); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("fractional offset err=%v want %v", err, ErrInvalidMotion)
	}
}

func TestReferenceOriginSubsampledMatchesLibaomPositionRule(t *testing.T) {
	tests := []struct {
		name               string
		dstX, dstY         int
		mv                 Vector
		subsamplingX       bool
		subsamplingY       bool
		wantX, wantY       int
		wantSubX, wantSubY int
	}{
		{
			name:     "luma",
			dstX:     5,
			dstY:     6,
			mv:       Vector{Col: -1, Row: 5},
			wantX:    4,
			wantY:    6,
			wantSubX: 14,
			wantSubY: 10,
		},
		{
			name:         "subsampled positive fractional",
			dstX:         8,
			dstY:         9,
			mv:           Vector{Col: 3, Row: 5},
			subsamplingX: true,
			subsamplingY: true,
			wantX:        8,
			wantY:        9,
			wantSubX:     3,
			wantSubY:     5,
		},
		{
			name:         "subsampled negative fractional floors",
			dstX:         8,
			dstY:         9,
			mv:           Vector{Col: -1, Row: -9},
			subsamplingX: true,
			subsamplingY: true,
			wantX:        7,
			wantY:        8,
			wantSubX:     15,
			wantSubY:     7,
		},
		{
			name:         "mixed subsampling axes",
			dstX:         4,
			dstY:         7,
			mv:           Vector{Col: 7, Row: 7},
			subsamplingY: true,
			wantX:        4,
			wantY:        7,
			wantSubX:     14,
			wantSubY:     7,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotX, gotY, gotSubX, gotSubY, err := ReferenceOriginSubsampled(tt.dstX, tt.dstY, tt.mv, tt.subsamplingX, tt.subsamplingY)
			if err != nil {
				t.Fatal(err)
			}
			if gotX != tt.wantX || gotY != tt.wantY || gotSubX != tt.wantSubX || gotSubY != tt.wantSubY {
				t.Fatalf("origin=%d,%d sub=%d,%d want %d,%d sub=%d,%d", gotX, gotY, gotSubX, gotSubY, tt.wantX, tt.wantY, tt.wantSubX, tt.wantSubY)
			}
		})
	}
}

func TestLowerPrecisionMatchesLibaom(t *testing.T) {
	tests := []struct {
		name               string
		in                 Vector
		allowHighPrecision bool
		forceInteger       bool
		want               Vector
	}{
		{name: "high precision keeps eighth pel", in: Vector{Row: 3, Col: -5}, allowHighPrecision: true, want: Vector{Row: 3, Col: -5}},
		{name: "low precision moves odd components toward zero", in: Vector{Row: 3, Col: -3}, want: Vector{Row: 2, Col: -2}},
		{name: "low precision keeps even components", in: Vector{Row: 6, Col: -4}, want: Vector{Row: 6, Col: -4}},
		{name: "integer rounds positive over half", in: Vector{Row: 5, Col: 12}, allowHighPrecision: true, forceInteger: true, want: Vector{Row: 8, Col: 8}},
		{name: "integer rounds negative over half", in: Vector{Row: -5, Col: -12}, forceInteger: true, want: Vector{Row: -8, Col: -8}},
		{name: "integer ties round toward zero", in: Vector{Row: 4, Col: -4}, forceInteger: true, want: Vector{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LowerPrecision(tt.in, tt.allowHighPrecision, tt.forceInteger)
			if got != tt.want {
				t.Fatalf("LowerPrecision(%+v)=%+v want %+v", tt.in, got, tt.want)
			}
		})
	}
}

func TestPredictInterPlaneBlock8Bit(t *testing.T) {
	src, _ := testPlane(7, 6, 1, 9)
	dst, _ := testPlane(7, 6, 1, 9)
	for i := range dst.Pix {
		dst.Pix[i] = 0xee
	}
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setSample(src, 1, x, y, uint16(10*y+x))
		}
	}
	mv := Vector{Row: 1 << 3, Col: -1 << 3}
	if err := predictLumaWithVector(dst, src, 1, 8, 2, 1, 3, 2, mv, RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < dst.Height; y++ {
		for x := 0; x < dst.Width; x++ {
			got := getSample(dst, 1, x, y)
			want := uint16(0xee)
			if x >= 2 && x < 5 && y >= 1 && y < 3 {
				want = uint16(10*(y+1) + (x - 1))
			}
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictInterPlaneBlockHighBitDepth(t *testing.T) {
	src, _ := testPlane(6, 5, 2, 14)
	dst, _ := testPlane(6, 5, 2, 14)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setSample(src, 2, x, y, uint16(1000+y*10+x))
		}
	}
	mv := Vector{Row: -1 << 3, Col: 1 << 3}
	if err := predictLumaWithVector(dst, src, 2, 12, 1, 2, 3, 2, mv, RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := range 2 {
		for x := range 3 {
			got := getSample(dst, 2, 1+x, 2+y)
			want := uint16(1000 + (1+y)*10 + 2 + x)
			if got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestInterpFilterTablesMatchLibaom(t *testing.T) {
	tests := []struct {
		name      string
		filter    InterpFilter
		blockSize int
		subpel    int
		want      [filterTaps]int16
	}{
		{name: "regular 8 tap", filter: InterpEightTapRegular, blockSize: 8, subpel: 1, want: [filterTaps]int16{0, 2, -6, 126, 8, -2, 0, 0}},
		{name: "smooth 8 tap", filter: InterpEightTapSmooth, blockSize: 8, subpel: 8, want: [filterTaps]int16{0, -2, 14, 52, 52, 14, -2, 0}},
		{name: "sharp 8 tap", filter: InterpMultiTapSharp, blockSize: 8, subpel: 15, want: [filterTaps]int16{0, 2, -2, 8, 126, -6, 2, -2}},
		{name: "bilinear", filter: InterpBilinear, blockSize: 4, subpel: 8, want: [filterTaps]int16{0, 0, 0, 64, 64, 0, 0, 0}},
		{name: "regular 4 tap", filter: InterpEightTapRegular, blockSize: 4, subpel: 4, want: [filterTaps]int16{0, 0, -12, 110, 38, -8, 0, 0}},
		{name: "sharp uses regular 4 tap", filter: InterpMultiTapSharp, blockSize: 4, subpel: 4, want: [filterTaps]int16{0, 0, -12, 110, 38, -8, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := interpKernel(tt.filter, tt.blockSize, tt.subpel)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("kernel=%v want %v", got, tt.want)
			}
		})
	}
}

func TestPredictInterPlaneBlockFractionalXMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: 4, Row: 0}
	if err := predictLumaWithVector(dst, src, 1, 8, 8, 8, 8, 8, mv, RegularFilters); err != nil {
		t.Fatal(err)
	}
	kernel, err := interpKernel(InterpEightTapRegular, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolveXRef(want, src, 8, 8, 8, 8, 8, 8, kernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockFractionalYMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: 0, Row: 6}
	filters := InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}
	if err := predictLumaWithVector(dst, src, 1, 8, 8, 8, 8, 8, mv, filters); err != nil {
		t.Fatal(err)
	}
	kernel, err := interpKernel(InterpEightTapSmooth, 8, 12)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolveYRef(want, src, 8, 8, 8, 8, 8, 8, kernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockFractional2DMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: 3, Row: 5}
	filters := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}
	if err := predictLumaWithVector(dst, src, 1, 8, 8, 8, 8, 8, mv, filters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(InterpMultiTapSharp, 8, 6)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(InterpEightTapSmooth, 8, 10)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolve2DRef(want, src, 8, 8, 8, 8, 8, 8, xKernel, yKernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockNegativeFractionalMatchesLibaom(t *testing.T) {
	src, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	want, _ := testPlane(32, 32, 1, 32)
	fillMotionTestPlane(src)

	mv := Vector{Col: -1, Row: -3}
	if err := predictLumaWithVector(dst, src, 1, 8, 8, 8, 8, 8, mv, RegularFilters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(InterpEightTapRegular, 8, 14)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(InterpEightTapRegular, 8, 10)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolve2DRef(want, src, 7, 7, 8, 8, 8, 8, xKernel, yKernel)
	assertPlaneBlockEqual(t, dst, want, 1, 8, 8, 8, 8)
}

func TestPredictInterPlaneBlockFullpelExtendsReferenceEdges(t *testing.T) {
	src, _ := testPlane(4, 4, 1, 4)
	dst, _ := testPlane(5, 3, 1, 5)
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			setSample(src, 1, x, y, uint16(10*y+x))
		}
	}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, -2, 2, 5, 3, 0, 0, RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 5 {
			want := getSampleClamped(src, 1, x-2, y+2)
			if got := getSample(dst, 1, x, y); got != want {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}

	srcHigh, _ := testPlane(4, 4, 2, 8)
	dstHigh, _ := testPlane(5, 3, 2, 10)
	for y := 0; y < srcHigh.Height; y++ {
		for x := 0; x < srcHigh.Width; x++ {
			setSample(srcHigh, 2, x, y, uint16(100+10*y+x))
		}
	}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstHigh, srcHigh, 2, 12, 0, 0, 2, -2, 5, 3, 0, 0, RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 5 {
			want := getSampleClamped(srcHigh, 2, x+2, y-2)
			if got := getSample(dstHigh, 2, x, y); got != want {
				t.Fatalf("highbd sample(%d,%d)=%d want %d", x, y, got, want)
			}
		}
	}
}

func TestPredictInterPlaneBlockFractionalExtendsReferenceEdges8Bit(t *testing.T) {
	src, _ := testPlane(8, 8, 1, 8)
	got, _ := testPlane(4, 4, 1, 4)
	want, _ := testPlane(4, 4, 1, 4)
	fillMotionTestPlane(src)

	filters := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 1, 8, 0, 0, 0, 0, 4, 4, 3, 5, filters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(filters.X, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(filters.Y, 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	libaomConvolve2DClampedRef(want, src, 0, 0, 0, 0, 4, 4, xKernel, yKernel)
	assertPlaneBlockEqual(t, got, want, 1, 0, 0, 4, 4)
}

func TestPredictInterPlaneBlockOneDimensionalFractionalExtendsReferenceEdges8Bit(t *testing.T) {
	src, _ := testPlane(8, 8, 1, 8)
	fillMotionTestPlane(src)
	filters := InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}

	t.Run("x", func(t *testing.T) {
		got, _ := testPlane(4, 4, 1, 4)
		want, _ := testPlane(4, 4, 1, 4)
		if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 1, 8, 0, 0, 0, 2, 4, 4, 3, 0, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.X, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		libaomConvolveXClampedRef(want, src, 0, 2, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 1, 0, 0, 4, 4)
	})

	t.Run("y", func(t *testing.T) {
		got, _ := testPlane(4, 4, 1, 4)
		want, _ := testPlane(4, 4, 1, 4)
		if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 1, 8, 0, 0, 2, 0, 4, 4, 0, 5, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.Y, 4, 5)
		if err != nil {
			t.Fatal(err)
		}
		libaomConvolveYClampedRef(want, src, 2, 0, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 1, 0, 0, 4, 4)
	})
}

func TestPredictInterPlaneBlockHighBitDepthFractionalMatchesLibaom(t *testing.T) {
	tests := []struct {
		name     string
		bitDepth uint8
		mv       Vector
		filters  InterpFilters
		ref      func(dst frame.Plane, src frame.Plane, max uint16, kernelX [filterTaps]int16, kernelY [filterTaps]int16)
	}{
		{
			name:     "x",
			bitDepth: 10,
			mv:       Vector{Col: 4, Row: 0},
			filters:  RegularFilters,
			ref: func(dst frame.Plane, src frame.Plane, max uint16, kernelX [filterTaps]int16, _ [filterTaps]int16) {
				libaomHighBDConvolveXRef(dst, src, max, 8, 8, 8, 8, 8, 8, kernelX)
			},
		},
		{
			name:     "y",
			bitDepth: 12,
			mv:       Vector{Col: 0, Row: 6},
			filters:  InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth},
			ref: func(dst frame.Plane, src frame.Plane, max uint16, _ [filterTaps]int16, kernelY [filterTaps]int16) {
				libaomHighBDConvolveYRef(dst, src, max, 8, 8, 8, 8, 8, 8, kernelY)
			},
		},
		{
			name:     "2d",
			bitDepth: 12,
			mv:       Vector{Col: 3, Row: 5},
			filters:  InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth},
			ref: func(dst frame.Plane, src frame.Plane, max uint16, kernelX [filterTaps]int16, kernelY [filterTaps]int16) {
				libaomHighBDConvolve2DRef(dst, src, 12, max, 8, 8, 8, 8, 8, 8, kernelX, kernelY)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src, _ := testPlane(32, 32, 2, 64)
			dst, _ := testPlane(32, 32, 2, 64)
			want, _ := testPlane(32, 32, 2, 64)
			max := uint16((1 << tt.bitDepth) - 1)
			fillHighBDMotionTestPlane(src, max)
			if err := predictLumaWithVector(dst, src, 2, tt.bitDepth, 8, 8, 8, 8, tt.mv, tt.filters); err != nil {
				t.Fatal(err)
			}
			_, _, subX, subY, err := ReferenceOriginSubsampled(8, 8, tt.mv, false, false)
			if err != nil {
				t.Fatal(err)
			}
			xKernel, err := interpKernel(tt.filters.X, 8, subX)
			if err != nil {
				t.Fatal(err)
			}
			yKernel, err := interpKernel(tt.filters.Y, 8, subY)
			if err != nil {
				t.Fatal(err)
			}
			tt.ref(want, src, max, xKernel, yKernel)
			assertPlaneBlockEqual(t, dst, want, 2, 8, 8, 8, 8)
		})
	}
}

func TestPredictInterPlaneBlockFractionalExtendsReferenceEdgesHighBD(t *testing.T) {
	src, _ := testPlane(8, 8, 2, 16)
	got, _ := testPlane(4, 4, 2, 8)
	want, _ := testPlane(4, 4, 2, 8)
	fillHighBDMotionTestPlane(src, 0x3ff)

	filters := InterpFilters{X: InterpEightTapSmooth, Y: InterpMultiTapSharp}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 2, 10, 0, 0, 5, 5, 4, 4, 9, 11, filters); err != nil {
		t.Fatal(err)
	}
	xKernel, err := interpKernel(filters.X, 4, 9)
	if err != nil {
		t.Fatal(err)
	}
	yKernel, err := interpKernel(filters.Y, 4, 11)
	if err != nil {
		t.Fatal(err)
	}
	libaomHighBDConvolve2DClampedRef(want, src, 10, 0x3ff, 5, 5, 0, 0, 4, 4, xKernel, yKernel)
	assertPlaneBlockEqual(t, got, want, 2, 0, 0, 4, 4)
}

func TestPredictInterPlaneBlockOneDimensionalFractionalExtendsReferenceEdgesHighBD(t *testing.T) {
	src, _ := testPlane(8, 8, 2, 16)
	fillHighBDMotionTestPlane(src, 0x3ff)
	filters := InterpFilters{X: InterpEightTapSmooth, Y: InterpMultiTapSharp}

	t.Run("x", func(t *testing.T) {
		got, _ := testPlane(4, 4, 2, 8)
		want, _ := testPlane(4, 4, 2, 8)
		if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 2, 10, 0, 0, 0, 2, 4, 4, 9, 0, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.X, 4, 9)
		if err != nil {
			t.Fatal(err)
		}
		libaomHighBDConvolveXClampedRef(want, src, 0x3ff, 0, 2, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 2, 0, 0, 4, 4)
	})

	t.Run("y", func(t *testing.T) {
		got, _ := testPlane(4, 4, 2, 8)
		want, _ := testPlane(4, 4, 2, 8)
		if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 2, 10, 0, 0, 2, 0, 4, 4, 0, 11, filters); err != nil {
			t.Fatal(err)
		}
		kernel, err := interpKernel(filters.Y, 4, 11)
		if err != nil {
			t.Fatal(err)
		}
		libaomHighBDConvolveYClampedRef(want, src, 0x3ff, 2, 0, 0, 0, 4, 4, kernel)
		assertPlaneBlockEqual(t, got, want, 2, 0, 0, 4, 4)
	})
}

func TestPredictInterPlaneBlockFromOriginLibaomConvolvePort(t *testing.T) {
	tests := []struct {
		name    string
		size    libaomConvolveBlockSize
		subX    int
		subY    int
		filters InterpFilters
	}{
		{name: "x", size: libaomConvolveBlockSize{width: 4, height: 16}, subX: 7, filters: InterpFilters{X: InterpEightTapSmooth, Y: InterpEightTapRegular}},
		{name: "y", size: libaomConvolveBlockSize{width: 16, height: 4}, subY: 11, filters: InterpFilters{X: InterpEightTapRegular, Y: InterpBilinear}},
		{name: "2d-small", size: libaomConvolveBlockSize{width: 8, height: 8}, subX: 3, subY: 5, filters: InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}},
		{name: "2d-wide", size: libaomConvolveBlockSize{width: 64, height: 32}, subX: 15, subY: 1, filters: InterpFilters{X: InterpBilinear, Y: InterpMultiTapSharp}},
		{name: "2d-max", size: libaomConvolveBlockSize{width: 128, height: 128}, subX: 8, subY: 8, filters: InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := libaomConvolveInput8(tt.size)
			got, _ := testPlane(tt.size.width, tt.size.height, 1, tt.size.width)
			want, _ := testPlane(tt.size.width, tt.size.height, 1, tt.size.width)
			if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 1, 8, 0, 0, libaomInputOrigin, libaomInputOrigin, tt.size.width, tt.size.height, tt.subX, tt.subY, tt.filters); err != nil {
				t.Fatal(err)
			}
			xKernel, err := interpKernel(tt.filters.X, tt.size.width, tt.subX)
			if err != nil {
				t.Fatal(err)
			}
			yKernel, err := interpKernel(tt.filters.Y, tt.size.height, tt.subY)
			if err != nil {
				t.Fatal(err)
			}
			switch {
			case tt.subX != 0 && tt.subY != 0:
				libaomConvolve2DRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, tt.size.width, tt.size.height, xKernel, yKernel)
			case tt.subX != 0:
				libaomConvolveXRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, tt.size.width, tt.size.height, xKernel)
			case tt.subY != 0:
				libaomConvolveYRef(want, src, libaomInputOrigin, libaomInputOrigin, 0, 0, tt.size.width, tt.size.height, yKernel)
			}
			assertLibaomConvolveEqual(t, tt.name, got, want, tt.size, tt.subX, tt.subY, tt.filters)
		})
	}
}

func TestPredictInterPlaneBlockFromOriginFullpelHighBitDepth(t *testing.T) {
	src, _ := testPlane(12, 12, 2, 24)
	got, _ := testPlane(4, 3, 2, 8)
	fillHighBDMotionTestPlane(src, 0xfff)

	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(got, src, 2, 12, 0, 0, 3, 4, 4, 3, 0, 0, RegularFilters); err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 4 {
			if g, w := getSample(got, 2, x, y), getSample(src, 2, 3+x, 4+y); g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, g, w)
			}
		}
	}
}

func TestPredictInterPlaneBlockFromOriginRejectsInvalidInputs(t *testing.T) {
	src, _ := testPlane(16, 16, 1, 16)
	dst, _ := testPlane(8, 8, 1, 8)
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, 4, 4, 4, 4, 16, 0, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad subX err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, 4, 4, 4, 4, 0, -1, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad subY err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, 4, 4, 4, 4, 0, 0, InterpFilters{X: interpFilterCount}); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad filter err=%v want %v", err, ErrInvalidMotion)
	}
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, -1, 0, 4, 4, 0, 0, RegularFilters); err != nil {
		t.Fatalf("edge-extended fullpel err=%v", err)
	}

	srcHigh, _ := testPlane(16, 16, 2, 32)
	dstHigh, _ := testPlane(8, 8, 2, 16)
	if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstHigh, srcHigh, 2, 8, 0, 0, 4, 4, 4, 4, 0, 0, RegularFilters); !errors.Is(err, ErrInvalidMotion) {
		t.Fatalf("bad highbd bitdepth err=%v want %v", err, ErrInvalidMotion)
	}
}

func TestPredictInterPlaneBlockFromOriginAllocs(t *testing.T) {
	t.Run("lowbd", func(t *testing.T) {
		src, _ := testPlane(32, 32, 1, 32)
		dst, _ := testPlane(8, 8, 1, 8)
		fillMotionTestPlane(src)
		allocs := testing.AllocsPerRun(1000, func() {
			if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 1, 8, 0, 0, 8, 8, 8, 8, 3, 5, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("explicit-origin inter prediction allocated: %f", allocs)
		}
	})
	t.Run("highbd", func(t *testing.T) {
		src, _ := testPlane(32, 32, 2, 64)
		dst, _ := testPlane(8, 8, 2, 16)
		fillHighBDMotionTestPlane(src, 0x3ff)
		allocs := testing.AllocsPerRun(1000, func() {
			if err := PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, src, 2, 10, 0, 0, 8, 8, 8, 8, 3, 5, InterpFilters{X: InterpMultiTapSharp, Y: InterpEightTapSmooth}); err != nil {
				t.Fatal(err)
			}
		})
		if allocs != 0 {
			t.Fatalf("explicit-origin highbd inter prediction allocated: %f", allocs)
		}
	})
}

// TestConvolveHighBDIdentityKernel asserts that running the High-BD
// convolvers with the subpel=0 identity kernel ({0,0,0,128,0,0,0,0})
// reproduces the input sample byte-for-byte at bd=10 (q32) and bd=12.
//
// This pins the dispatch + rounding shape for the simplest reproducible
// case: any drift here would imply a bias-shift or final-round bug that
// libaom does not have, because libaom's
// av1_highbd_convolve_{x,y,2d}_sr_c are bit-exact identity at the
// identity kernel by construction (see av1/common/convolve.c lines
// 689-790 in libaom-v3.13.1).
func TestConvolveHighBDIdentityKernel(t *testing.T) {
	identity := [filterTaps]int16{0, 0, 0, 128, 0, 0, 0, 0}
	for _, tc := range []struct {
		name     string
		bitDepth uint8
	}{
		{"bd=10", 10},
		{"bd=12", 12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			max, ok := highBDMax(tc.bitDepth)
			if !ok {
				t.Fatalf("highBDMax(%d) not supported", tc.bitDepth)
			}
			// Sized so that the full convolve window (block + filter
			// halo) fits inside the source plane without clamping.
			const stride = 64
			src, _ := testPlane(32, 32, 2, stride)
			fillHighBDMotionTestPlane(src, max)
			t.Run("X", func(t *testing.T) {
				dst, _ := testPlane(32, 32, 2, stride)
				convolveXHighBDImpl(dst, src, tc.bitDepth, max, 8, 8, 8, 8, 8, 8, identity)
				for row := range 8 {
					for col := range 8 {
						got := getSample(dst, 2, 8+col, 8+row)
						want := getSample(src, 2, 8+col, 8+row)
						if got != want {
							t.Fatalf("X identity sample(%d,%d)=%d want %d", col, row, got, want)
						}
					}
				}
			})
			t.Run("Y", func(t *testing.T) {
				dst, _ := testPlane(32, 32, 2, stride)
				convolveYHighBDImpl(dst, src, tc.bitDepth, max, 8, 8, 8, 8, 8, 8, identity)
				for row := range 8 {
					for col := range 8 {
						got := getSample(dst, 2, 8+col, 8+row)
						want := getSample(src, 2, 8+col, 8+row)
						if got != want {
							t.Fatalf("Y identity sample(%d,%d)=%d want %d", col, row, got, want)
						}
					}
				}
			})
			t.Run("2D", func(t *testing.T) {
				dst, _ := testPlane(32, 32, 2, stride)
				convolve2DHighBDImpl(dst, src, tc.bitDepth, max, 8, 8, 8, 8, 8, 8, identity, identity)
				for row := range 8 {
					for col := range 8 {
						got := getSample(dst, 2, 8+col, 8+row)
						want := getSample(src, 2, 8+col, 8+row)
						if got != want {
							t.Fatalf("2D identity sample(%d,%d)=%d want %d", col, row, got, want)
						}
					}
				}
			})
		})
	}
}

// TestConvolveHighBDQ10ClampedPath pins the *clamped* HighBD convolvers
// (the path that fires when the convolve halo overruns the reference
// edge) against a from-scratch libaom-shaped reference. Frame 1 of the
// q32 vector exercises this path at every block whose MV pulls the
// filter halo across the top/left frame edge — any drift in the edge
// extension would surface as a wave of ±N pixel diffs along the top
// and left of the frame.
func TestConvolveHighBDQ10ClampedPath(t *testing.T) {
	const bd = uint8(10)
	max := uint16((1 << bd) - 1)
	// Use a tiny source plane so the convolve halo overruns the edge
	// at every position; this forces the *Clamped paths exclusively.
	src, _ := testPlane(6, 6, 2, 12)
	fillHighBDMotionTestPlane(src, max)
	filters := []InterpFilters{
		{X: InterpEightTapRegular, Y: InterpEightTapRegular},
		{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
		{X: InterpBilinear, Y: InterpBilinear},
	}
	sizes := []libaomConvolveBlockSize{
		{width: 4, height: 4},
		{width: 8, height: 4},
		{width: 4, height: 8},
	}
	for _, sz := range sizes {
		for _, f := range filters {
			for _, sx := range []int{1, 7, 8, 15} {
				for _, sy := range []int{1, 7, 8, 15} {
					t.Run(libaomHighBDCaseName(sz, sx, sy, f)+"_clamped", func(t *testing.T) {
						got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						xKernel, err := interpKernel(f.X, sz.width, sx)
						if err != nil {
							t.Fatal(err)
						}
						yKernel, err := interpKernel(f.Y, sz.height, sy)
						if err != nil {
							t.Fatal(err)
						}
						convolve2DHighBDClampedImpl(got, src, bd, max, 0, 0, 0, 0, sz.width, sz.height, xKernel, yKernel)
						libaomVerbatimHighBD2DSRClamped(want, src, bd, max, 0, 0, 0, 0, sz.width, sz.height, xKernel, yKernel)
						for row := 0; row < sz.height; row++ {
							for col := 0; col < sz.width; col++ {
								g := getSample(got, 2, col, row)
								w := getSample(want, 2, col, row)
								if g != w {
									t.Fatalf("clamped size=%dx%d sub=(%d,%d) filters=(%d,%d) sample(%d,%d)=%d want %d",
										sz.width, sz.height, sx, sy, f.X, f.Y, col, row, g, w)
								}
							}
						}
					})
				}
			}
		}
	}
}

// TestConvolveHighBDQ10OneDimensional pins the X-only and Y-only
// HighBD convolve paths (subX=0 vs subY=0) against a from-scratch
// libaom-shaped reference. Frame 1 of q32 has many blocks that decode
// to a one-axis MV (e.g. cell.MV = (3, 0) or (0, 5)), and those land
// on the simpler convolveXHighBD / convolveYHighBD paths rather than
// the 2D path.
func TestConvolveHighBDQ10OneDimensional(t *testing.T) {
	const bd = uint8(10)
	max := uint16((1 << bd) - 1)
	src, _ := testPlane(32, 32, 2, 64)
	fillHighBDMotionTestPlane(src, max)
	filters := []InterpFilters{
		{X: InterpEightTapRegular, Y: InterpEightTapRegular},
		{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
		{X: InterpBilinear, Y: InterpBilinear},
	}
	sizes := []libaomConvolveBlockSize{
		{width: 4, height: 4},
		{width: 8, height: 8},
		{width: 16, height: 8},
		{width: 16, height: 16},
	}
	for _, sz := range sizes {
		for _, f := range filters {
			for _, sub := range []int{1, 4, 7, 8, 11, 15} {
				t.Run("X_"+libaomHighBDCaseName(sz, sub, 0, f), func(t *testing.T) {
					got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					xKernel, err := interpKernel(f.X, sz.width, sub)
					if err != nil {
						t.Fatal(err)
					}
					convolveXHighBDImpl(got, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel)
					libaomVerbatimHighBDXSR(want, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel)
					for row := 0; row < sz.height; row++ {
						for col := 0; col < sz.width; col++ {
							g := getSample(got, 2, col, row)
							w := getSample(want, 2, col, row)
							if g != w {
								t.Fatalf("X size=%dx%d sub=%d filter=%d sample(%d,%d)=%d want %d", sz.width, sz.height, sub, f.X, col, row, g, w)
							}
						}
					}
				})
				t.Run("Y_"+libaomHighBDCaseName(sz, 0, sub, f), func(t *testing.T) {
					got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
					yKernel, err := interpKernel(f.Y, sz.height, sub)
					if err != nil {
						t.Fatal(err)
					}
					convolveYHighBDImpl(got, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, yKernel)
					libaomVerbatimHighBDYSR(want, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, yKernel)
					for row := 0; row < sz.height; row++ {
						for col := 0; col < sz.width; col++ {
							g := getSample(got, 2, col, row)
							w := getSample(want, 2, col, row)
							if g != w {
								t.Fatalf("Y size=%dx%d sub=%d filter=%d sample(%d,%d)=%d want %d", sz.width, sz.height, sub, f.Y, col, row, g, w)
							}
						}
					}
				})
			}
		}
	}
}

// libaomVerbatimHighBDXSR ports libaom av1_highbd_convolve_x_sr_c.
func libaomVerbatimHighBDXSR(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, xFilter [filterTaps]int16) {
	foHoriz := filterTaps/2 - 1
	bits := filterBits - round0Bits
	for y := range h {
		for x := range w {
			res := int32(0)
			for k := range filterTaps {
				res += int32(xFilter[k]) * int32(getSample(src, 2, refX+x-foHoriz+k, refY+y))
			}
			res = int32(libaomRoundPowerOfTwo(int(res), round0Bits))
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(int(res), bits), maxVal))
		}
	}
	_ = bd
}

// libaomVerbatimHighBDYSR ports libaom av1_highbd_convolve_y_sr_c.
func libaomVerbatimHighBDYSR(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, yFilter [filterTaps]int16) {
	foVert := filterTaps/2 - 1
	for y := range h {
		for x := range w {
			res := int32(0)
			for k := range filterTaps {
				res += int32(yFilter[k]) * int32(getSample(src, 2, refX+x, refY+y-foVert+k))
			}
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(int(res), filterBits), maxVal))
		}
	}
	_ = bd
}

// libaomVerbatimHighBD2DSRClamped is the clamped counterpart of
// libaomVerbatimHighBD2DSR: every reference load goes through edge
// replication (libaom's highbd_build_mc_border equivalent).
func libaomVerbatimHighBD2DSRClamped(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, xFilter [filterTaps]int16, yFilter [filterTaps]int16) {
	const imStride = maxBlockSize
	var imBlock [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	imH := h + filterTaps - 1
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	for y := range imH {
		for x := range w {
			sum := int32(1) << (uint32(bd) + uint32(filterBits) - 1)
			for k := range filterTaps {
				sum += int32(xFilter[k]) * int32(getSampleClamped(src, 2, refX+x-foHoriz+k, refY-foVert+y))
			}
			imBlock[y*imStride+x] = int16(libaomRoundPowerOfTwo(int(sum), round0Bits))
		}
	}
	offsetBits := int(bd) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range h {
		for x := range w {
			sum := int32(1) << uint32(offsetBits)
			for k := range filterTaps {
				sum += int32(yFilter[k]) * int32(imBlock[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(int(sum), round1Bits) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), maxVal))
		}
	}
}

// TestConvolveHighBDLibaomQ10MatchesAlgorithm pins the HighBD convolvers
// against an independent libaom-shaped reference implementation written
// inline (see libaomVerbatimHighBD2DSR) for a representative grid of
// block sizes, subpel positions, and switchable interpolation filters
// at bd=10. The reference is rewritten from scratch in this test (it
// does not call the libaom*Ref helpers above) so any algebra mistake in
// the production path that the helper might also have inherited (same
// formula, same bug) shows up here as a divergence.
func TestConvolveHighBDLibaomQ10MatchesAlgorithm(t *testing.T) {
	const bd = uint8(10)
	max := uint16((1 << bd) - 1)
	sizes := []libaomConvolveBlockSize{
		{width: 4, height: 4},
		{width: 4, height: 8},
		{width: 8, height: 4},
		{width: 8, height: 8},
		{width: 8, height: 16},
		{width: 16, height: 16},
		{width: 32, height: 32},
	}
	filters := []InterpFilters{
		{X: InterpEightTapRegular, Y: InterpEightTapRegular},
		{X: InterpEightTapSmooth, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpMultiTapSharp},
		{X: InterpBilinear, Y: InterpBilinear},
		{X: InterpEightTapRegular, Y: InterpEightTapSmooth},
		{X: InterpMultiTapSharp, Y: InterpBilinear},
	}
	subpels := []int{1, 2, 4, 7, 8, 11, 13, 15}
	for _, sz := range sizes {
		for _, f := range filters {
			for _, sx := range subpels {
				for _, sy := range subpels {
					t.Run(libaomHighBDCaseName(sz, sx, sy, f), func(t *testing.T) {
						src := libaomHighBDConvolveInput(sz, max)
						got, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						want, _ := testPlane(sz.width, sz.height, 2, sz.width*2)
						xKernel, err := interpKernel(f.X, sz.width, sx)
						if err != nil {
							t.Fatal(err)
						}
						yKernel, err := interpKernel(f.Y, sz.height, sy)
						if err != nil {
							t.Fatal(err)
						}
						// Production call (anchored at the input
						// origin so the convolve halo lies inside
						// the padded source plane).
						convolve2DHighBDImpl(got, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel, yKernel)
						// Independent libaom-shaped reference,
						// rewritten verbatim from libaom-v3.13.1
						// av1/common/convolve.c
						// av1_highbd_convolve_2d_sr_c.
						libaomVerbatimHighBD2DSR(want, src, bd, max, 0, 0, libaomInputOrigin, libaomInputOrigin, sz.width, sz.height, xKernel, yKernel)
						for row := 0; row < sz.height; row++ {
							for col := 0; col < sz.width; col++ {
								g := getSample(got, 2, col, row)
								w := getSample(want, 2, col, row)
								if g != w {
									t.Fatalf("size=%dx%d sub=(%d,%d) filters=(%d,%d) sample(%d,%d)=%d want %d",
										sz.width, sz.height, sx, sy, f.X, f.Y, col, row, g, w)
								}
							}
						}
					})
				}
			}
		}
	}
}

func libaomHighBDCaseName(sz libaomConvolveBlockSize, sx int, sy int, f InterpFilters) string {
	return libaomHighBDFormat(sz.width) + "x" + libaomHighBDFormat(sz.height) + "_sx" + libaomHighBDFormat(sx) + "_sy" + libaomHighBDFormat(sy) + "_fx" + libaomHighBDFormat(int(f.X)) + "_fy" + libaomHighBDFormat(int(f.Y))
}

func libaomHighBDFormat(v int) string {
	digits := "0123456789"
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var buf [12]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = digits[v%10]
		v /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func libaomHighBDConvolveInput(size libaomConvolveBlockSize, max uint16) frame.Plane {
	const padding = libaomInputPadding
	plane, _ := testPlane(size.width+padding, size.height+padding, 2, (size.width+padding)*2)
	state := uint32(0xcafe)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			state = state*1664525 + 1013904223
			setSample(plane, 2, x, y, uint16(state>>16)&max)
		}
	}
	return plane
}

// libaomVerbatimHighBD2DSR is a from-scratch port of libaom-v3.13.1
// av1/common/convolve.c av1_highbd_convolve_2d_sr_c. It intentionally
// uses libaom's exact intermediate types (int16 im_block, int32
// accumulators) so the test cross-checks the production path against
// the reference algebra rather than against a Go helper that may share
// a refactoring bug.
func libaomVerbatimHighBD2DSR(dst frame.Plane, src frame.Plane, bd uint8, maxVal uint16, dstX int, dstY int, refX int, refY int, w int, h int, xFilter [filterTaps]int16, yFilter [filterTaps]int16) {
	const imStride = maxBlockSize
	var imBlock [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	imH := h + filterTaps - 1
	foHoriz := filterTaps/2 - 1
	foVert := filterTaps/2 - 1
	// Horizontal filter — bias is (1 << (bd + FILTER_BITS - 1)).
	for y := range imH {
		for x := range w {
			sum := int32(1) << (uint32(bd) + uint32(filterBits) - 1)
			for k := range filterTaps {
				sum += int32(xFilter[k]) * int32(getSample(src, 2, refX+x-foHoriz+k, refY-foVert+y))
			}
			imBlock[y*imStride+x] = int16(libaomRoundPowerOfTwo(int(sum), round0Bits))
		}
	}
	// Vertical filter — offset_bits = bd + 2 * FILTER_BITS - round_0.
	offsetBits := int(bd) + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range h {
		for x := range w {
			sum := int32(1) << uint32(offsetBits)
			for k := range filterTaps {
				sum += int32(yFilter[k]) * int32(imBlock[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(int(sum), round1Bits) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), maxVal))
		}
	}
}

func testPlane(width int, height int, bytesPerSample int, stride int) (frame.Plane, []byte) {
	buf := make([]byte, stride*height)
	return frame.Plane{
		Pix:    buf,
		Stride: stride,
		Width:  width,
		Height: height,
	}, buf
}

func getSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func getSampleClamped(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	if x < 0 {
		x = 0
	} else if x >= plane.Width {
		x = plane.Width - 1
	}
	if y < 0 {
		y = 0
	} else if y >= plane.Height {
		y = plane.Height - 1
	}
	return getSample(plane, bytesPerSample, x, y)
}

func setSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func fillMotionTestPlane(plane frame.Plane) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte((x*x + 3*y*y + 17*x + 11*y) & 0xff)
		}
	}
}

func fillHighBDMotionTestPlane(plane frame.Plane, max uint16) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 2, x, y, uint16((x*x+3*y*y+17*x+11*y)&int(max)))
		}
	}
}

func assertPlaneBlockEqual(t *testing.T, got frame.Plane, want frame.Plane, bytesPerSample int, x int, y int, width int, height int) {
	t.Helper()
	for row := range height {
		for col := range width {
			g := getSample(got, bytesPerSample, x+col, y+row)
			w := getSample(want, bytesPerSample, x+col, y+row)
			if g != w {
				t.Fatalf("sample(%d,%d)=%d want %d", x+col, y+row, g, w)
			}
		}
	}
}

func libaomHighBDConvolveXRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSample(src, 2, refX+x-fo+k, refY+y))
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, filterBits-round0Bits), max))
		}
	}
}

func libaomHighBDConvolveYRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSample(src, 2, refX+x, refY+y-fo+k))
			}
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func libaomHighBDConvolveXClampedRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 2, refX+x-fo+k, refY+y))
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, filterBits-round0Bits), max))
		}
	}
}

func libaomHighBDConvolveYClampedRef(dst frame.Plane, src frame.Plane, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 2, refX+x, refY+y-fo+k))
			}
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(sum, filterBits), max))
		}
	}
}

func libaomHighBDConvolve2DRef(dst frame.Plane, src frame.Plane, bitDepth uint8, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	// get_conv_params_no_round bumps round_0/round_1 at bd == 12.
	round0, round1 := highBDRoundBits(bitDepth)
	for y := 0; y < height+filterTaps-1; y++ {
		for x := range width {
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(getSample(src, 2, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(libaomRoundPowerOfTwo(sum, round0))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), max))
		}
	}
}

func libaomHighBDConvolve2DClampedRef(dst frame.Plane, src frame.Plane, bitDepth uint8, max uint16, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	// get_conv_params_no_round bumps round_0/round_1 at bd == 12.
	round0, round1 := highBDRoundBits(bitDepth)
	for y := 0; y < height+filterTaps-1; y++ {
		for x := range width {
			sum := 1 << (int(bitDepth) + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(getSampleClamped(src, 2, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int32(libaomRoundPowerOfTwo(sum, round0))
		}
	}
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - round1)) + (1 << (offsetBits - round1 - 1))
	bits := 2*filterBits - round0 - round1
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1) - roundOffset
			setSample(dst, 2, dstX+x, dstY+y, libaomClipPixelHighBD(libaomRoundPowerOfTwo(res, bits), max))
		}
	}
}

func libaomConvolveXRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(src.Pix[(refY+y)*src.Stride+refX+x-fo+k])
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func libaomConvolveYRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(src.Pix[(refY+y-fo+k)*src.Stride+refX+x])
			}
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(sum, filterBits)))
		}
	}
}

func libaomConvolveXClampedRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 1, refX+x-fo+k, refY+y))
			}
			res := libaomRoundPowerOfTwo(sum, round0Bits)
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, filterBits-round0Bits)))
		}
	}
}

func libaomConvolveYClampedRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, kernel [filterTaps]int16) {
	fo := filterTaps/2 - 1
	for y := range height {
		for x := range width {
			sum := 0
			for k := range filterTaps {
				sum += int(kernel[k]) * int(getSampleClamped(src, 1, refX+x, refY+y-fo+k))
			}
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(sum, filterBits)))
		}
	}
}

func libaomConvolve2DRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	for y := 0; y < height+filterTaps-1; y++ {
		for x := range width {
			sum := 1 << (8 + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(src.Pix[(refY-foY+y)*src.Stride+refX+x-foX+k])
			}
			im[y*imStride+x] = int16(libaomRoundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, bits)))
		}
	}
}

func libaomConvolve2DClampedRef(dst frame.Plane, src frame.Plane, refX int, refY int, dstX int, dstY int, width int, height int, xKernel [filterTaps]int16, yKernel [filterTaps]int16) {
	const imStride = maxBlockSize
	var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int16
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	for y := 0; y < height+filterTaps-1; y++ {
		for x := range width {
			sum := 1 << (8 + filterBits - 1)
			for k := range filterTaps {
				sum += int(xKernel[k]) * int(getSampleClamped(src, 1, refX+x-foX+k, refY-foY+y))
			}
			im[y*imStride+x] = int16(libaomRoundPowerOfTwo(sum, round0Bits))
		}
	}
	offsetBits := 8 + 2*filterBits - round0Bits
	roundOffset := (1 << (offsetBits - round1Bits)) + (1 << (offsetBits - round1Bits - 1))
	bits := 2*filterBits - round0Bits - round1Bits
	for y := range height {
		for x := range width {
			sum := 1 << offsetBits
			for k := range filterTaps {
				sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
			}
			res := libaomRoundPowerOfTwo(sum, round1Bits) - roundOffset
			dst.Pix[(dstY+y)*dst.Stride+dstX+x] = byte(libaomClipPixel(libaomRoundPowerOfTwo(res, bits)))
		}
	}
}

func libaomRoundPowerOfTwo(value int, bits int) int {
	if bits <= 0 {
		return value
	}
	return (value + (1 << (bits - 1))) >> bits
}

func libaomClipPixel(value int) uint8 {
	if value < 0 {
		return 0
	}
	if value > 255 {
		return 255
	}
	return uint8(value)
}

func libaomClipPixelHighBD(value int, max uint16) uint16 {
	if value < 0 {
		return 0
	}
	if value > int(max) {
		return max
	}
	return uint16(value)
}

// predictLumaWithVector predicts a luma block displaced by mv: the luma
// ReferenceOriginSubsampled origin fed to the explicit-origin predictor.
func predictLumaWithVector(dst frame.Plane, ref frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mv Vector, filters InterpFilters) error {
	refX, refY, subX, subY, err := ReferenceOriginSubsampled(dstX, dstY, mv, false, false)
	if err != nil {
		return err
	}
	return PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, refX, refY, width, height, subX, subY, filters)
}

// libaomWarpAffineC is a hand-translated, intentionally faithful port of
// av1/common/warped_motion.c:av1_warp_affine_c (single-reference, 8-bit, not
// compound). It is used purely as a parity oracle to discover any drift in our
// production warpAffine8 implementation.
func libaomWarpAffineC(pred []byte, predStride int, ref []byte, refStride, refWidth, refHeight int,
	pCol, pRow, pWidth, pHeight int, mat [6]int32, alpha, beta, gamma, delta int16,
	subsamplingX, subsamplingY int,
) {
	const (
		WARPEDMODEL_PREC_BITS   = 16
		WARP_PARAM_REDUCE       = 6
		WARPEDDIFF_PREC_BITS    = WARPEDMODEL_PREC_BITS - 6 // = 10
		WARPEDPIXEL_PREC_SHIFTS = 64
		FILTER_BITS             = 7
		bd                      = 8
		reduceBitsHoriz         = 3
		offsetBitsHoriz         = bd + FILTER_BITS - 1                 // 14
		reduceBitsVert          = 2*FILTER_BITS - reduceBitsHoriz      // 11
		offsetBitsVert          = bd + 2*FILTER_BITS - reduceBitsHoriz // 19
	)

	var tmp [15 * 8]int32

	round := func(v int, n int) int {
		bias := 1 << (n - 1)
		// C arithmetic shift semantics: (v+bias) >> n on signed int.
		return (v + bias) >> n
	}
	clamp := func(v, lo, hi int) int {
		if v < lo {
			return lo
		}
		if v > hi {
			return hi
		}
		return v
	}
	clipPix := func(v int) byte {
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return byte(v)
	}

	for i := pRow; i < pRow+pHeight; i += 8 {
		for j := pCol; j < pCol+pWidth; j += 8 {
			srcX := int64(j+4) << uint(subsamplingX)
			srcY := int64(i+4) << uint(subsamplingY)
			dstX := int64(mat[2])*srcX + int64(mat[3])*srcY + int64(mat[0])
			dstY := int64(mat[4])*srcX + int64(mat[5])*srcY + int64(mat[1])
			x4 := dstX >> uint(subsamplingX)
			y4 := dstY >> uint(subsamplingY)

			ix4 := int32(x4 >> WARPEDMODEL_PREC_BITS)
			sx4 := int32(x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1))
			iy4 := int32(y4 >> WARPEDMODEL_PREC_BITS)
			sy4 := int32(y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1))

			sx4 += int32(alpha)*-4 + int32(beta)*-4
			sy4 += int32(gamma)*-4 + int32(delta)*-4

			sx4 &^= (1 << WARP_PARAM_REDUCE) - 1
			sy4 &^= (1 << WARP_PARAM_REDUCE) - 1

			// Horizontal filter
			for k := -7; k < 8; k++ {
				iy := clamp(int(iy4)+k, 0, refHeight-1)
				sx := int(sx4) + int(beta)*(k+4)
				for l := -4; l < 4; l++ {
					ix := int(ix4) + l - 3
					offs := round(sx, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsHoriz
					for m := range 8 {
						sampleX := clamp(ix+m, 0, refWidth-1)
						sum += int(ref[iy*refStride+sampleX]) * int(coeffs[m])
					}
					tmp[(k+7)*8+(l+4)] = int32(round(sum, reduceBitsHoriz))
					sx += int(alpha)
				}
			}

			// Vertical filter
			rowEnd := min(pRow+pHeight-i-4, 4)
			colEnd := min(pCol+pWidth-j-4, 4)
			for k := -4; k < rowEnd; k++ {
				sy := int(sy4) + int(delta)*(k+4)
				for l := -4; l < colEnd; l++ {
					offs := round(sy, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS
					coeffs := warpedFilter[offs]
					sum := 1 << offsetBitsVert
					for m := range 8 {
						sum += int(tmp[(k+m+4)*8+(l+4)]) * int(coeffs[m])
					}
					sum = round(sum, reduceBitsVert)
					row := i - pRow + k + 4
					col := j - pCol + l + 4
					pred[row*predStride+col] = clipPix(sum - (1 << (bd - 1)) - (1 << bd))
					sy += int(gamma)
				}
			}
		}
	}
}

// TestWarpAffineMatchesLibaomReference compares the production warpAffine8
// against the hand-rolled libaom-faithful oracle on a battery of randomized
// inputs that mirror the bitstream contract for local-warp shear parameters.
func TestWarpAffineMatchesLibaomReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA0BEEF))

	const refW = 96
	const refH = 96
	ref := make([]byte, refW*refH)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}

	// Block sizes that occur in practice (powers of 2 × pairs).
	cases := []struct {
		w, h int
	}{
		{8, 8}, {16, 16}, {8, 16}, {16, 8}, {32, 8}, {8, 32}, {32, 16}, {16, 32},
	}

	for caseIdx := range 64 {
		for _, sz := range cases {
			pCol := rng.Intn(refW-sz.w+1) &^ 7
			pRow := rng.Intn(refH-sz.h+1) &^ 7
			// Build a representative warp matrix: small perturbation around identity.
			matrix := [6]int32{
				int32(rng.Intn(1<<16)) - (1 << 15),   // mat[0]: trans X (Q0 << 16)
				int32(rng.Intn(1<<16)) - (1 << 15),   // mat[1]: trans Y
				1<<16 + int32(rng.Intn(8192)) - 4096, // mat[2] ≈ 1 + small
				int32(rng.Intn(8192)) - 4096,         // mat[3]
				int32(rng.Intn(8192)) - 4096,         // mat[4]
				1<<16 + int32(rng.Intn(8192)) - 4096, // mat[5]
			}
			// Pick alpha/beta/gamma/delta consistent with WARP_PARAM_REDUCE.
			roundReduce := func(v int) int16 {
				if v < 0 {
					return int16(-((-v + 32) >> 6) * 64)
				}
				return int16(((v + 32) >> 6) * 64)
			}
			alpha := roundReduce(rng.Intn(1024) - 512)
			beta := roundReduce(rng.Intn(1024) - 512)
			gamma := roundReduce(rng.Intn(1024) - 512)
			delta := roundReduce(rng.Intn(1024) - 512)

			// Filter the "interesting" cases: must satisfy is_affine_shear_allowed.
			abs := func(x int16) int {
				if x < 0 {
					return -int(x)
				}
				return int(x)
			}
			if 4*abs(alpha)+7*abs(beta) >= 1<<16 || 4*abs(gamma)+4*abs(delta) >= 1<<16 {
				continue
			}

			expected := make([]byte, sz.w*sz.h)
			actual := make([]byte, sz.w*sz.h)

			libaomWarpAffineC(expected, sz.w, ref, refW, refW, refH,
				pCol, pRow, sz.w, sz.h, matrix, alpha, beta, gamma, delta, 0, 0)

			// Wrap our destination into a frame.Plane covering the block region exactly.
			dst, _ := testPlane(sz.w, sz.h, 1, sz.w)
			refPlane, _ := testPlane(refW, refH, 1, refW)
			copy(refPlane.Pix, ref)

			if err := PredictWarpedPlaneBlockBitDepth(dst, refPlane, 1, 8, 0, 0, sz.w, sz.h, matrix, alpha, beta, gamma, delta, false, false); err != nil {
				t.Fatalf("case %d sz=%dx%d: %v", caseIdx, sz.w, sz.h, err)
			}

			// Because the production function uses absolute coordinates, recompute
			// expected with pCol=0, pRow=0.
			expected = make([]byte, sz.w*sz.h)
			libaomWarpAffineC(expected, sz.w, ref, refW, refW, refH,
				0, 0, sz.w, sz.h, matrix, alpha, beta, gamma, delta, 0, 0)
			for i := range expected {
				actual[i] = dst.Pix[i]
			}
			for y := 0; y < sz.h; y++ {
				for x := 0; x < sz.w; x++ {
					if expected[y*sz.w+x] != actual[y*sz.w+x] {
						t.Fatalf("case %d sz=%dx%d mismatch at (%d,%d): expected=%d actual=%d matrix=%v alpha=%d beta=%d gamma=%d delta=%d",
							caseIdx, sz.w, sz.h, x, y, expected[y*sz.w+x], actual[y*sz.w+x], matrix, alpha, beta, gamma, delta)
					}
				}
			}
		}
	}
}

// libaomQ00WarpRefBin returns the captured Y-plane of libaom's reconstruction
// of quantizer_00 frame 0, which serves as the reference frame for frame 1.
// The bytes are read from testdata/warp_q00_ref.bin.zlib so the snapshot
// stays out of the binary even though the warp regression below needs to
// drive a full 352x288 reference window through PredictWarpedPlaneBlockBitDepth.
func libaomQ00WarpRefBin(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/warp_q00_ref.bin.zlib")
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}
	zr, err := zlib.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("zlib reader: %v", err)
	}
	defer zr.Close()
	out, err := io.ReadAll(zr)
	if err != nil {
		t.Fatalf("zlib decompress: %v", err)
	}
	const expected = 352 * 288
	if len(out) != expected {
		t.Fatalf("snapshot size = %d want %d", len(out), expected)
	}
	return out
}

// TestPredictWarpedPlaneBlockQ00Frame1MI80x4 pins our warpAffine8 dispatch
// against av1_warp_affine_c() on the first WARPED_CAUSAL block of libaom's
// quantizer_00 frame 1 (mi=(80,4), 8x16 luma, MotionMode=WARPED_CAUSAL with
// MV (-1,-7) sampling frame 0). The warp inputs are derived from a real
// bitstream decode pass and the expected output bytes were produced by
// invoking av1_warp_affine_c() directly with the same inputs.
//
// This pins both the shear-parameter dispatch (alpha=-320, beta=-64,
// gamma=0, delta=0) and the filter table indexing for a block whose
// initial sub-block lands at the upper-left corner with no edge clamping,
// catching ±1 rounding drift in the horizontal/vertical reduce stages.
func TestPredictWarpedPlaneBlockQ00Frame1MI80x4(t *testing.T) {
	refBytes := libaomQ00WarpRefBin(t)
	refPlane, _ := testPlane(352, 288, 1, 352)
	copy(refPlane.Pix, refBytes)

	dst, _ := testPlane(352, 288, 1, 352)

	matrix := [6]int32{42805, -7571, 65230, -57, 0, 65509}
	alpha := int16(-320)
	beta := int16(-64)
	gamma := int16(0)
	delta := int16(0)

	if err := PredictWarpedPlaneBlockBitDepth(dst, refPlane, 1, 8, 320, 16, 8, 16,
		matrix, alpha, beta, gamma, delta, false, false); err != nil {
		t.Fatalf("PredictWarpedPlaneBlockBitDepth: %v", err)
	}

	// Expected bytes produced by av1_warp_affine_c() (libaom v3.13.1) with the
	// inputs above on the same reference plane.
	want := [16][8]byte{
		{98, 110, 99, 90, 93, 91, 99, 98},
		{89, 99, 90, 82, 82, 83, 92, 90},
		{87, 97, 91, 82, 84, 87, 92, 90},
		{86, 95, 87, 78, 79, 81, 90, 90},
		{90, 99, 93, 86, 88, 89, 101, 101},
		{102, 106, 99, 95, 95, 95, 104, 99},
		{94, 98, 89, 84, 86, 84, 94, 93},
		{92, 94, 87, 83, 84, 84, 93, 94},
		{97, 93, 88, 84, 84, 84, 92, 90},
		{88, 102, 93, 80, 95, 90, 93, 102},
		{100, 117, 106, 110, 103, 101, 112, 99},
		{115, 102, 129, 134, 83, 124, 124, 78},
		{100, 107, 120, 97, 107, 121, 100, 116},
		{106, 121, 89, 93, 119, 89, 109, 131},
		{117, 102, 103, 121, 88, 104, 125, 91},
		{111, 110, 121, 118, 114, 114, 104, 106},
	}

	for y := range 16 {
		for x := range 8 {
			got := dst.Pix[(16+y)*dst.Stride+(320+x)]
			if got != want[y][x] {
				t.Fatalf("dst[%d,%d] = %d want %d (libaom av1_warp_affine_c)",
					320+x, 16+y, got, want[y][x])
			}
		}
	}
}

func TestPredictWarpedPlaneBlockIdentityConstant8Bit(t *testing.T) {
	ref, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	for i := range ref.Pix {
		ref.Pix[i] = 91
	}
	matrix := [6]int32{0, 0, warpedModelOne, 0, 0, warpedModelOne}
	if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 1, 8, 8, 8, 16, 16, matrix, 0, 0, 0, 0, false, false); err != nil {
		t.Fatal(err)
	}
	for y := 8; y < 24; y++ {
		for x := 8; x < 24; x++ {
			if got := dst.Pix[y*dst.Stride+x]; got != 91 {
				t.Fatalf("sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestPredictWarpedPlaneBlockIdentityConstantHighBD(t *testing.T) {
	ref, _ := testPlane(32, 32, 2, 64)
	dst, _ := testPlane(32, 32, 2, 64)
	for y := 0; y < ref.Height; y++ {
		for x := 0; x < ref.Width; x++ {
			storeHighBDSample(ref, x, y, 777)
		}
	}
	matrix := [6]int32{0, 0, warpedModelOne, 0, 0, warpedModelOne}
	if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 2, 10, 8, 8, 16, 16, matrix, 0, 0, 0, 0, false, false); err != nil {
		t.Fatal(err)
	}
	for y := 8; y < 24; y++ {
		for x := 8; x < 24; x++ {
			if got := loadHighBDSample(dst, x, y); got != 777 {
				t.Fatalf("sample(%d,%d)=%d want 777", x, y, got)
			}
		}
	}
}

func TestPredictWarpedPlaneBlockAllocs(t *testing.T) {
	ref, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	matrix := [6]int32{0, 0, warpedModelOne, 0, 0, warpedModelOne}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 1, 8, 8, 8, 16, 16, matrix, 0, 0, 0, 0, false, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warped prediction allocated: %f", allocs)
	}
}

func TestWarpVertical8FullGamma0MatchesGeneric(t *testing.T) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	for i := range tmp {
		tmp[i] = int32((i*97+31)%4096 - 2048)
	}

	cases := []struct {
		baseSY int
		delta  int
	}{
		{baseSY: 0, delta: -64},
		{baseSY: 512, delta: 128},
		{baseSY: -1 << 20, delta: 0},
	}
	for _, tc := range cases {
		want, _ := testPlane(32, 32, 1, 32)
		got, _ := testPlane(32, 32, 1, 32)
		for i := range want.Pix {
			want.Pix[i] = 0xaa
			got.Pix[i] = 0xaa
		}

		warpVertical8Full(want, &tmp, 8, 8, 1, 2, tc.baseSY, 0, tc.delta, round1Bits, 8+2*filterBits-round0Bits)
		warpVertical8FullGamma0(got, &tmp, 8, 8, 1, 2, tc.baseSY, tc.delta, round1Bits, 8+2*filterBits-round0Bits)

		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("baseSY=%d delta=%d pix[%d]=%d want %d", tc.baseSY, tc.delta, i, got.Pix[i], want.Pix[i])
			}
		}
	}
}

func TestWarpVerticalHighBDFullGamma0MatchesGeneric(t *testing.T) {
	var tmp [warpedIntermediateRows * warpedIntermediateColumns]int32
	for i := range tmp {
		tmp[i] = int32((i*131+17)%8192 - 4096)
	}

	round0, round1 := highBDRoundBits(10)
	cases := []struct {
		baseSY int
		delta  int
	}{
		{baseSY: 0, delta: -64},
		{baseSY: 512, delta: 128},
		{baseSY: -1 << 20, delta: 0},
	}
	for _, tc := range cases {
		want, _ := testPlane(32, 32, 2, 64)
		got, _ := testPlane(32, 32, 2, 64)
		for i := range want.Pix {
			want.Pix[i] = 0x5a
			got.Pix[i] = 0x5a
		}

		warpVerticalHighBDFull(want, 10, 1023, &tmp, 8, 8, 1, 2, tc.baseSY, 0, tc.delta, round1, 10+2*filterBits-round0)
		warpVerticalHighBDFullGamma0(got, 10, 1023, &tmp, 8, 8, 1, 2, tc.baseSY, tc.delta, round1, 10+2*filterBits-round0)

		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("baseSY=%d delta=%d pix[%d]=%d want %d", tc.baseSY, tc.delta, i, got.Pix[i], want.Pix[i])
			}
		}
	}
}

func BenchmarkPredictWarpedPlaneBlock8Bit64(b *testing.B) {
	ref, _ := testPlane(96, 96, 1, 96)
	dst, _ := testPlane(96, 96, 1, 96)
	for y := 0; y < ref.Height; y++ {
		for x := 0; x < ref.Width; x++ {
			ref.Pix[y*ref.Stride+x] = byte((x*17 + y*29 + x*y) & 0xff)
		}
	}
	matrix := [6]int32{42805, -7571, 65230, -57, 0, 65509}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 1, 8, 16, 16, 64, 64, matrix, -320, -64, 0, -64, false, false); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPredictWarpedPlaneBlockHighBD64(b *testing.B) {
	ref, _ := testPlane(96, 96, 2, 192)
	dst, _ := testPlane(96, 96, 2, 192)
	for y := 0; y < ref.Height; y++ {
		for x := 0; x < ref.Width; x++ {
			storeHighBDSample(ref, x, y, uint16((x*17+y*29+x*y)&0x3ff))
		}
	}
	matrix := [6]int32{42805, -7571, 65230, -57, 0, 65509}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 2, 10, 16, 16, 64, 64, matrix, -320, -64, 0, -64, false, false); err != nil {
			b.Fatal(err)
		}
	}
}

// FuzzPredictWarpedPlaneBlockBitDepth stresses the warped affine prediction
// dispatcher with random matrix coefficients, alpha/beta/gamma/delta scales,
// block placements, and subsampling. Out-of-range parameters are clipped to
// the contract validated by PredictWarpedPlaneBlockBitDepth so the function
// must return cleanly without panicking or writing past the destination
// region.
func FuzzPredictWarpedPlaneBlockBitDepth(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(1), uint8(2), uint8(8), uint8(8), uint8(7), uint8(7), int16(64), int16(-32), int16(16), int16(-8))
	f.Add(uint8(2), uint8(5), uint8(2), uint8(3), uint8(15), uint8(7), int16(255), int16(-255), int16(127), int16(-127))
	f.Add(uint8(3), uint8(7), uint8(1), uint8(0), uint8(3), uint8(15), int16(-512), int16(256), int16(-128), int16(64))
	f.Add(uint8(4), uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), int16(32767), int16(-32768), int16(32767), int16(-32768))

	f.Fuzz(func(t *testing.T, rawShape uint8, rawSubs uint8, rawDstX uint8, rawDstY uint8, rawW uint8, rawH uint8, alphaRaw int16, betaRaw int16, gammaRaw int16, deltaRaw int16) {
		bytesPerSample := 1
		bitDepth := uint8(8)
		switch rawShape & 3 {
		case 1:
			bytesPerSample = 2
			bitDepth = 10
		case 2:
			bytesPerSample = 2
			bitDepth = 12
		}

		planeW := 64
		planeH := 64
		blockW := 8 + 8*(int(rawShape>>2)&3)
		blockH := 8 + 8*(int(rawShape>>4)&3)
		if blockW > planeW {
			blockW = planeW
		}
		if blockH > planeH {
			blockH = planeH
		}
		blockW &^= 7
		blockH &^= 7
		if blockW == 0 {
			blockW = 8
		}
		if blockH == 0 {
			blockH = 8
		}

		dstX := int(rawDstX) % (planeW - blockW + 1)
		dstY := int(rawDstY) % (planeH - blockH + 1)
		dstX &^= 7
		dstY &^= 7

		stride := planeW * bytesPerSample
		ref, _ := testPlane(planeW, planeH, bytesPerSample, stride)
		dst, _ := testPlane(planeW, planeH, bytesPerSample, stride)

		maxSample := uint16((1 << bitDepth) - 1)
		for y := 0; y < ref.Height; y++ {
			for x := 0; x < ref.Width; x++ {
				v := uint16((x*7 + y*11) & 0xff)
				if bytesPerSample == 2 {
					v = uint16((x*13+y*17)&0x3ff) & maxSample
				}
				if bytesPerSample == 1 {
					ref.Pix[y*ref.Stride+x] = byte(v)
				} else {
					storeHighBDSample(ref, x, y, v)
				}
			}
		}

		const warpRange = warpedModelOne / 2
		matrix := [6]int32{
			int32((int(alphaRaw) % (planeW + 1)) << 4),
			int32((int(betaRaw) % (planeH + 1)) << 4),
			warpedModelOne + int32(alphaRaw)%warpRange,
			int32(gammaRaw) % warpRange,
			int32(deltaRaw) % warpRange,
			warpedModelOne + int32(betaRaw)%warpRange,
		}
		// libaom keeps alpha/beta/gamma/delta in [-512, 511] after parameter
		// reduction; clamp to mirror that bitstream contract.
		clamp := func(v int16) int16 {
			if v > 511 {
				return 511
			}
			if v < -512 {
				return -512
			}
			return v
		}
		alpha := clamp(alphaRaw)
		beta := clamp(betaRaw)
		gamma := clamp(gammaRaw)
		delta := clamp(deltaRaw)

		subsamplingX := rawSubs&1 != 0
		subsamplingY := rawSubs&2 != 0

		err := PredictWarpedPlaneBlockBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, blockW, blockH, matrix, alpha, beta, gamma, delta, subsamplingX, subsamplingY)
		if err != nil {
			// Any rejection must come from contract validation, not a panic.
			return
		}

		for y := 0; y < dst.Height; y++ {
			for x := 0; x < dst.Width; x++ {
				if x >= dstX && x < dstX+blockW && y >= dstY && y < dstY+blockH {
					continue
				}
				if bytesPerSample == 1 {
					if got := dst.Pix[y*dst.Stride+x]; got != 0 {
						t.Fatalf("dst outside region modified at (%d,%d): %d", x, y, got)
					}
				} else {
					if got := loadHighBDSample(dst, x, y); got != 0 {
						t.Fatalf("dst outside region modified at (%d,%d): %d", x, y, got)
					}
				}
			}
		}
		for y := dstY; y < dstY+blockH; y++ {
			for x := dstX; x < dstX+blockW; x++ {
				var got uint16
				if bytesPerSample == 1 {
					got = uint16(dst.Pix[y*dst.Stride+x])
				} else {
					got = loadHighBDSample(dst, x, y)
				}
				if got > maxSample {
					t.Fatalf("dst sample(%d,%d)=%d exceeds max %d", x, y, got, maxSample)
				}
			}
		}
	})
}
