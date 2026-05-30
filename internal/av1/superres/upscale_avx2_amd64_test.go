// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package superres

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// runRowImplsAVX2 upscales the same input with both the AVX2 kernel and the
// pure-Go reference and returns the two destination rows for comparison.
func runRowImplsAVX2(t *testing.T, srcRow []uint16, dstWidth, codedWidth, bitDepth int) ([]uint16, []uint16) {
	t.Helper()
	srcWidth := len(srcRow)
	stepX := ((codedWidth << ScaleBits) + (dstWidth / 2)) / dstWidth
	errTerm := (dstWidth * stepX) - (codedWidth << ScaleBits)
	initialSubpelX := ((-((dstWidth - codedWidth) << (ScaleBits - 1))) + dstWidth/2) / dstWidth
	initialSubpelX += (1 << (ExtraBits - 1)) - errTerm/2
	initialSubpelX &= ScaleMask
	maxValue := (1 << bitDepth) - 1
	srcLast := srcWidth - 1

	got := make([]uint16, dstWidth)
	want := make([]uint16, dstWidth)
	upscaleRowAVX2(srcRow, got, dstWidth, stepX, initialSubpelX, srcLast, maxValue)
	upscaleRowPureGo(srcRow, want, dstWidth, stepX, initialSubpelX, srcLast, maxValue)
	return got, want
}

func TestUpscaleRowAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	bitDepths := []int{8, 10, 12}
	// Exercise a spread of coded/dst widths, including the awkward small and
	// odd shapes and the boundary-heavy near-1:1 ratios.
	type shape struct{ coded, dst int }
	shapes := []shape{
		{8, 13}, {4, 7}, {2, 16}, {9, 16}, {15, 16}, {16, 31},
		{100, 160}, {120, 121}, {480, 640}, {1, 16}, {3, 5},
		{63, 64}, {64, 127}, {255, 256}, {200, 201},
	}
	for _, bd := range bitDepths {
		maxVal := uint16((1 << bd) - 1)
		for _, sh := range shapes {
			// src physical width >= coded width; add a little padding to
			// model the MI-aligned span.
			srcWidth := sh.coded + rng.Intn(4)
			src := make([]uint16, srcWidth)
			for i := range src {
				src[i] = uint16(rng.Intn(int(maxVal) + 1))
			}
			got, want := runRowImplsAVX2(t, src, sh.dst, sh.coded, bd)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("bd=%d coded=%d dst=%d srcW=%d: dst[%d]=%d want %d\nsrc=%v\ngot=%v\nwant=%v",
						bd, sh.coded, sh.dst, srcWidth, i, got[i], want[i], src, got, want)
				}
			}
		}
	}
}

// TestUpscalePlaneAVX2MatchesPureGo drives the full plane entry point against a
// pure-Go-only reference computed with upscaleRowPureGo, over many rows. Note
// it only exercises the AVX2 row kernel through the dispatcher if
// cpu.Detected.AVX2 is set; the direct-kernel TestUpscaleRowAVX2MatchesPureGo
// validates the AVX2 path unconditionally.
func TestUpscalePlaneAVX2MatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, bd := range []uint8{8, 10, 12} {
		maxVal := uint16((1 << bd) - 1)
		coded, dst, height := 200, 320, 17
		srcWidth := coded + 5
		src := make([]uint16, srcWidth*height)
		for i := range src {
			src[i] = uint16(rng.Intn(int(maxVal) + 1))
		}
		gotPix := make([]uint16, dst*height)
		srcPlane := frame.SamplePlane{Pix: src, Stride: srcWidth, Width: srcWidth, Height: height}
		dstPlane := frame.SamplePlane{Pix: gotPix, Stride: dst, Width: dst, Height: height}
		if err := UpscalePlaneCoded(srcPlane, dstPlane, coded, bd); err != nil {
			t.Fatal(err)
		}

		// Reference: same orchestration but force pure-Go row kernel.
		wantPix := make([]uint16, dst*height)
		stepX := ((coded << ScaleBits) + (dst / 2)) / dst
		errTerm := (dst * stepX) - (coded << ScaleBits)
		initialSubpelX := ((-((dst - coded) << (ScaleBits - 1))) + dst/2) / dst
		initialSubpelX += (1 << (ExtraBits - 1)) - errTerm/2
		initialSubpelX &= ScaleMask
		srcLast := srcWidth - 1
		maxValue := (1 << int(bd)) - 1
		for y := 0; y < height; y++ {
			sr := src[y*srcWidth : y*srcWidth+srcWidth : y*srcWidth+srcWidth]
			dr := wantPix[y*dst : y*dst+dst : y*dst+dst]
			upscaleRowPureGo(sr, dr, dst, stepX, initialSubpelX, srcLast, maxValue)
		}
		for i := range wantPix {
			if gotPix[i] != wantPix[i] {
				t.Fatalf("bd=%d idx=%d got=%d want=%d", bd, i, gotPix[i], wantPix[i])
			}
		}
	}
}

func TestUpscaleRowAVX2ZeroAlloc(t *testing.T) {
	coded, dst := 480, 640
	srcWidth := coded + 3
	src := make([]uint16, srcWidth)
	for i := range src {
		src[i] = uint16(i*97) & 0xff
	}
	out := make([]uint16, dst)
	stepX := ((coded << ScaleBits) + (dst / 2)) / dst
	errTerm := (dst * stepX) - (coded << ScaleBits)
	initialSubpelX := ((-((dst - coded) << (ScaleBits - 1))) + dst/2) / dst
	initialSubpelX += (1 << (ExtraBits - 1)) - errTerm/2
	initialSubpelX &= ScaleMask
	srcLast := srcWidth - 1
	maxValue := 255

	allocs := testing.AllocsPerRun(100, func() {
		upscaleRowAVX2(src, out, dst, stepX, initialSubpelX, srcLast, maxValue)
	})
	if allocs != 0 {
		t.Fatalf("upscaleRowAVX2 allocated %v objects/run, want 0", allocs)
	}
}

// TestReportAVX2DispatchStatus prints whether the dispatcher bound the AVX2
// kernel (cpu.Detected.AVX2) or fell back to pure-Go.
func TestReportAVX2DispatchStatus(t *testing.T) {
	t.Logf("cpu.Detected.AVX2 = %v", cpu.Detected.AVX2)
	if cpu.Detected.AVX2 {
		t.Logf("super-res dispatch bound AVX2 kernel")
	} else {
		t.Logf("super-res dispatch fell back to pure-Go (cpu package does not detect AVX2)")
	}
}

func benchRowImplAVX2(b *testing.B, impl func([]uint16, []uint16, int, int, int, int, int)) {
	b.Helper()
	coded, dst := 480, 640
	srcWidth := coded + 3
	src := make([]uint16, srcWidth)
	for i := range src {
		src[i] = uint16(i*97) & 0xff
	}
	out := make([]uint16, dst)
	stepX := ((coded << ScaleBits) + (dst / 2)) / dst
	e := (dst * stepX) - (coded << ScaleBits)
	isp := ((-((dst - coded) << (ScaleBits - 1))) + dst/2) / dst
	isp += (1 << (ExtraBits - 1)) - e/2
	isp &= ScaleMask
	srcLast := srcWidth - 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		impl(src, out, dst, stepX, isp, srcLast, 255)
	}
}

func BenchmarkRowPureGoAMD64(b *testing.B) { benchRowImplAVX2(b, upscaleRowPureGo) }
func BenchmarkRowAVX2(b *testing.B)        { benchRowImplAVX2(b, upscaleRowAVX2) }
