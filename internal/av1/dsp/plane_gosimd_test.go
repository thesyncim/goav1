// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package dsp

import (
	"math/rand"
	"testing"
)

// planeSIMDBlock returns a block of height rows with stride bytes per row, filled
// from rng so that 16-bit samples cover the whole u16 range (including values
// above max, which the references clamp).
func planeSIMDBlock(rng *rand.Rand, height, stride int) []byte {
	pix := make([]byte, height*stride)
	for i := range pix {
		pix[i] = byte(rng.Intn(256))
	}
	return pix
}

// planeSIMDCheckResidual runs addResidualPlaneBlockSIMD and the scalar reference
// over identical copies and reports the first byte that differs.
func planeSIMDCheckResidual(t *testing.T, pix []byte, stride int, bps int, max uint16, width int, height int, residual []int16, resStride int) {
	t.Helper()
	got := append([]byte(nil), pix...)
	want := append([]byte(nil), pix...)
	addResidualPlaneBlockSIMD(planeBlock{pix: got, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, residual, resStride)
	addResidualPlaneBlockPureGo(planeBlock{pix: want, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, residual, resStride)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("residual bps=%d max=%d w=%d h=%d stride=%d: byte %d simd=%#x ref=%#x", bps, max, width, height, stride, i, got[i], want[i])
		}
	}
}

// planeSIMDCheckRaw runs addRawTransformPlaneBlockSIMD and the scalar reference
// over identical copies and reports the first byte that differs.
func planeSIMDCheckRaw(t *testing.T, pix []byte, stride int, bps int, max uint16, width int, height int, raw []int32, rawStride int) {
	t.Helper()
	got := append([]byte(nil), pix...)
	want := append([]byte(nil), pix...)
	addRawTransformPlaneBlockSIMD(planeBlock{pix: got, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, raw, rawStride)
	addRawTransformPlaneBlockPureGo(planeBlock{pix: want, stride: stride, width: width, height: height, rowBytes: width * bps}, bps, max, width, raw, rawStride)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("raw bps=%d max=%d w=%d h=%d stride=%d: byte %d simd=%#x ref=%#x", bps, max, width, height, stride, i, got[i], want[i])
		}
	}
}

// planeSIMDShapes lists the widths the Go SIMD kernels accept, plus widths they
// route to the reference, and every height from one to five so that odd row
// counts reach the paired-row tail.
var planeSIMDShapes = []struct{ width, height int }{
	{4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5},
	{8, 1}, {8, 4}, {8, 8},
	{12, 3}, {16, 2}, {16, 8}, {32, 4}, {64, 2}, {5, 2},
}

// TestAddResidualPlaneBlockSIMDMatchesPureGo sweeps widths, heights, strides and
// both sample widths and bit depths with random destinations and residuals.
func TestAddResidualPlaneBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095, 0xffff}
		}
		for _, max := range maxes {
			for _, shape := range planeSIMDShapes {
				for _, pad := range []int{0, 3} {
					stride := shape.width*bps + pad
					for trial := range 16 {
						pix := planeSIMDBlock(rng, shape.height, stride)
						res := make([]int16, shape.height*shape.width+8)
						for i := range res {
							switch {
							case trial%4 == 0 && rng.Intn(3) == 0:
								res[i] = []int16{-32768, -32767, -4096, -1, 0, 1, 4096, 32767}[rng.Intn(8)]
							default:
								res[i] = int16(rng.Intn(65536) - 32768)
							}
						}
						planeSIMDCheckResidual(t, pix, stride, bps, max, shape.width, shape.height, res, shape.width)
					}
				}
			}
		}
	}
}

// TestAddRawTransformPlaneBlockSIMDMatchesPureGo sweeps the same shapes with raw
// int32 inverse-transform samples across their full range.
func TestAddRawTransformPlaneBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	edges := []int32{
		-1 << 31, -1<<31 + 1, -1<<31 + 7, -1<<31 + 8, -(1 << 20), -4096, -129, -128, -121, -120,
		-9, -8, -7, -1, 0, 1, 7, 8, 9, 120, 121, 127, 128, 4095, 4096, 1 << 20, 1<<31 - 9,
		1<<31 - 8, 1<<31 - 1,
	}
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095}
		}
		for _, max := range maxes {
			for _, shape := range planeSIMDShapes {
				for _, pad := range []int{0, 5} {
					stride := shape.width*bps + pad
					for trial := range 16 {
						pix := planeSIMDBlock(rng, shape.height, stride)
						raw := make([]int32, shape.height*shape.width+8)
						for i := range raw {
							if trial%2 == 0 {
								raw[i] = edges[rng.Intn(len(edges))]
							} else {
								raw[i] = int32(rng.Uint32())
							}
						}
						planeSIMDCheckRaw(t, pix, stride, bps, max, shape.width, shape.height, raw, shape.width)
					}
				}
			}
		}
	}
}

// TestPlaneSIMDExhaustiveResidualEdges pairs every destination edge value with
// every residual edge value on a single row, for each sample width and maximum.
func TestPlaneSIMDExhaustiveResidualEdges(t *testing.T) {
	dstEdges := []uint16{0, 1, 127, 128, 255, 256, 1023, 1024, 4095, 4096, 0x7fff, 0x8000, 0xfffe, 0xffff}
	resEdges := []int16{-32768, -32767, -4096, -1024, -256, -128, -1, 0, 1, 127, 128, 255, 256, 4095, 4096, 32766, 32767}
	for _, bps := range []int{1, 2} {
		maxes := []uint16{0xff}
		if bps == 2 {
			maxes = []uint16{1023, 4095}
		}
		for _, max := range maxes {
			for _, width := range []int{8, 4} {
				for _, d := range dstEdges {
					for _, r := range resEdges {
						pix := make([]byte, 2*width*bps)
						res := make([]int16, 2*width)
						for i := range 2 * width {
							if bps == 1 {
								pix[i] = byte(d)
							} else {
								pix[2*i] = byte(d)
								pix[2*i+1] = byte(d >> 8)
							}
							res[i] = r
						}
						planeSIMDCheckResidual(t, pix, width*bps, bps, max, width, 2, res, width)
					}
				}
			}
		}
	}
}

// TestPlaneSIMDRoutesOtherWidthsToReference confirms widths the SIMD kernels do
// not vectorise still produce the reference output through the same entry point.
func TestPlaneSIMDRoutesOtherWidthsToReference(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, width := range []int{1, 2, 3, 5, 6, 7, 9, 12} {
		pix := planeSIMDBlock(rng, 3, width*2+1)
		res := make([]int16, 3*width)
		for i := range res {
			res[i] = int16(rng.Intn(2000) - 1000)
		}
		planeSIMDCheckResidual(t, pix, width*2+1, 2, 4095, width, 3, res, width)
		raw := make([]int32, 3*width)
		for i := range raw {
			raw[i] = int32(rng.Intn(1<<20) - 1<<19)
		}
		planeSIMDCheckRaw(t, pix, width+1, 1, 255, width, 3, raw, width)
	}
}

// TestPlaneSIMDIsZeroAlloc keeps both Go SIMD kernels allocation-free.
func TestPlaneSIMDIsZeroAlloc(t *testing.T) {
	const width, height = 16, 8
	pix := make([]byte, height*width*2)
	res := make([]int16, height*width)
	raw := make([]int32, height*width)
	for _, bps := range []int{1, 2} {
		block := planeBlock{pix: pix, stride: width * 2, width: width, height: height, rowBytes: width * bps}
		allocs := testing.AllocsPerRun(200, func() {
			addResidualPlaneBlockSIMD(block, bps, 255, width, res, width)
			addRawTransformPlaneBlockSIMD(block, bps, 255, width, raw, width)
		})
		if allocs != 0 {
			t.Fatalf("bps=%d: Go SIMD plane kernels allocated %f times per call", bps, allocs)
		}
	}
}
