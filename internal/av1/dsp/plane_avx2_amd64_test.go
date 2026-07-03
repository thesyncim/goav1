// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// addResidualLCG is a tiny deterministic generator so the differential test is
// reproducible without pulling in math/rand.
type addResidualLCG struct{ s uint64 }

func (g *addResidualLCG) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s >> 11
}

// TestAddResidualPlaneBlockAVX2MatchesPureGo runs the AVX2 inner loop and the
// pure-Go reference over identical inputs and asserts byte-for-byte equality.
// It sweeps power-of-two and edge (non-multiple-of-8) widths to exercise both
// the vector body and the scalar tail, and pushes residual values past the
// int16 add range to confirm the s32 intermediate clamps correctly. It is
// skipped on hosts without AVX2 (e.g. amd64 under Rosetta 2).
func TestAddResidualPlaneBlockAVX2MatchesPureGo(t *testing.T) {
	type shape struct {
		bytesPerSample int
		bitDepth       uint8
		max            uint16
	}
	shapes := []shape{
		{1, 8, 0xff},
		{2, 10, 0x3ff},
		{2, 12, 0xfff},
	}
	widths := []int{1, 3, 7, 8, 9, 13, 16, 17, 24, 32, 33, 64}
	heights := []int{1, 2, 4, 8, 16}

	for _, sh := range shapes {
		for _, width := range widths {
			for _, height := range heights {
				g := addResidualLCG{s: uint64(width*131 + height*17 + int(sh.bitDepth))}
				stride := (width + 7) * sh.bytesPerSample
				if stride < width*sh.bytesPerSample {
					stride = width * sh.bytesPerSample
				}
				residualStride := width + 3

				basePix := make([]byte, stride*height)
				for i := range basePix {
					basePix[i] = byte(g.next())
				}
				residual := make([]int16, residualStride*height)
				for i := range residual {
					switch g.next() & 3 {
					case 0:
						residual[i] = int16(g.next())
					case 1:
						residual[i] = -int16(g.next())
					default:
						residual[i] = int16(g.next()%64) - 32
					}
				}

				gotPix := append([]byte(nil), basePix...)
				wantPix := append([]byte(nil), basePix...)

				gotPlane := frame.Plane{Pix: gotPix, Stride: stride, Width: width, Height: height}
				if err := AddResidualPlaneBlock(gotPlane, sh.bytesPerSample, sh.bitDepth, 0, 0, width, height, residual, residualStride); err != nil {
					t.Fatalf("AVX2 dispatch err: bps=%d w=%d h=%d: %v", sh.bytesPerSample, width, height, err)
				}

				wantBlock, err := planeBlockWindow(frame.Plane{Pix: wantPix, Stride: stride, Width: width, Height: height}, sh.bytesPerSample, 0, 0, width, height)
				if err != nil {
					t.Fatalf("reference window err: %v", err)
				}
				addResidualPlaneBlockPureGo(wantBlock, sh.bytesPerSample, sh.max, width, residual, residualStride)

				for i := range gotPix {
					if gotPix[i] != wantPix[i] {
						t.Fatalf("bps=%d w=%d h=%d byte %d: avx2=%#02x ref=%#02x", sh.bytesPerSample, width, height, i, gotPix[i], wantPix[i])
					}
				}
			}
		}
	}
}

// TestAddRawTransformPlaneBlockAVX2MatchesPureGo runs the AVX2 raw-transform
// fused-add inner loop and the pure-Go reference over identical inputs and
// asserts byte-for-byte equality. It calls addRawTransformPlaneBlockAVX2
// directly (not through the public dispatch) so the vector path executes even on
// hosts that do not advertise AVX2 in CPUID — notably Rosetta 2, which runs AVX2
// but reports it absent. It sweeps power-of-two and edge widths to exercise both
// the vector body and the scalar tail, with raw values spanning the full int32
// range to confirm the round/int16-saturate/clamp matches the reference.
func TestAddRawTransformPlaneBlockAVX2MatchesPureGo(t *testing.T) {
	type shape struct {
		bytesPerSample int
		max            uint16
	}
	shapes := []shape{
		{1, 0xff},
		{2, 0x3ff},
		{2, 0xfff},
	}
	widths := []int{1, 3, 7, 8, 9, 13, 16, 17, 24, 32, 33, 64}
	heights := []int{1, 2, 4, 8, 16}

	for _, sh := range shapes {
		for _, width := range widths {
			for _, height := range heights {
				g := addResidualLCG{s: uint64(width*193 + height*29 + int(sh.max))}
				stride := (width + 7) * sh.bytesPerSample
				if stride < width*sh.bytesPerSample {
					stride = width * sh.bytesPerSample
				}
				rawStride := width + 3

				basePix := make([]byte, stride*height)
				for i := range basePix {
					basePix[i] = byte(g.next())
				}
				raw := make([]int32, rawStride*height)
				for i := range raw {
					switch g.next() & 3 {
					case 0:
						raw[i] = int32(g.next())
					case 1:
						raw[i] = -int32(g.next())
					default:
						raw[i] = int32(g.next()%4096) - 2048
					}
				}

				gotPix := append([]byte(nil), basePix...)
				wantPix := append([]byte(nil), basePix...)

				gotBlock, err := planeBlockWindow(frame.Plane{Pix: gotPix, Stride: stride, Width: width, Height: height}, sh.bytesPerSample, 0, 0, width, height)
				if err != nil {
					t.Fatalf("got window err: %v", err)
				}
				addRawTransformPlaneBlockAVX2(gotBlock, sh.bytesPerSample, sh.max, width, raw, rawStride)

				wantBlock, err := planeBlockWindow(frame.Plane{Pix: wantPix, Stride: stride, Width: width, Height: height}, sh.bytesPerSample, 0, 0, width, height)
				if err != nil {
					t.Fatalf("reference window err: %v", err)
				}
				addRawTransformPlaneBlockPureGo(wantBlock, sh.bytesPerSample, sh.max, width, raw, rawStride)

				for i := range gotPix {
					if gotPix[i] != wantPix[i] {
						t.Fatalf("bps=%d w=%d h=%d byte %d: avx2=%#02x ref=%#02x", sh.bytesPerSample, width, height, i, gotPix[i], wantPix[i])
					}
				}
			}
		}
	}
}

func TestAddRawTransformPlaneBlockAVX2IsZeroAlloc(t *testing.T) {
	basePix := make([]byte, 64*64)
	for i := range basePix {
		basePix[i] = byte(i)
	}
	raw := make([]int32, 64*64)
	for i := range raw {
		raw[i] = int32(i%4096) - 2048
	}
	block, err := planeBlockWindow(frame.Plane{Pix: basePix, Stride: 64, Width: 64, Height: 64}, 1, 0, 0, 64, 64)
	if err != nil {
		t.Fatalf("window err: %v", err)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		addRawTransformPlaneBlockAVX2(block, 1, 0xff, 64, raw, 64)
	})
	if allocs != 0 {
		t.Fatalf("addRawTransformPlaneBlockAVX2 allocated: %f", allocs)
	}
}

func TestAddResidualPlaneBlockAVX2IsZeroAlloc(t *testing.T) {
	plane, _ := testPlane(64, 64, 1, 64)
	residual := make([]int16, 64*64)
	for i := range residual {
		residual[i] = int16(i%512) - 256
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := AddResidualPlaneBlock(plane, 1, 8, 0, 0, 64, 64, residual, 64); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("AddResidualPlaneBlock allocated: %f", allocs)
	}
}
