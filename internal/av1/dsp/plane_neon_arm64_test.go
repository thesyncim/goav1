// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

// addResidualLCG is a tiny deterministic generator so the differential test is
// reproducible without pulling in math/rand.
type addResidualLCG struct{ s uint64 }

func (g *addResidualLCG) next() uint64 {
	g.s = g.s*6364136223846793005 + 1442695040888963407
	return g.s >> 11
}

// TestAddResidualPlaneBlockNEONMatchesPureGo runs the NEON inner loop and the
// pure-Go reference over identical inputs and asserts byte-for-byte equality.
// It sweeps power-of-two and edge (non-multiple-of-8) widths to exercise both
// the vector body and the scalar tail, and pushes residual values past the
// int16 add range to confirm the s32 intermediate clamps correctly.
func TestAddResidualPlaneBlockNEONMatchesPureGo(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON not detected")
	}
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
	widths := []int{1, 3, 4, 7, 8, 9, 13, 16, 17, 24, 32, 33, 64}
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
					// Mix of small and large (overflow-inducing) signed values.
					switch g.next() & 3 {
					case 0:
						residual[i] = int16(g.next()) // full int16 range
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
					t.Fatalf("NEON dispatch err: bps=%d w=%d h=%d: %v", sh.bytesPerSample, width, height, err)
				}

				wantBlock, err := planeBlockWindow(frame.Plane{Pix: wantPix, Stride: stride, Width: width, Height: height}, sh.bytesPerSample, 0, 0, width, height)
				if err != nil {
					t.Fatalf("reference window err: %v", err)
				}
				addResidualPlaneBlockPureGo(wantBlock, sh.bytesPerSample, sh.max, width, residual, residualStride)

				for i := range gotPix {
					if gotPix[i] != wantPix[i] {
						t.Fatalf("bps=%d w=%d h=%d byte %d: neon=%#02x ref=%#02x", sh.bytesPerSample, width, height, i, gotPix[i], wantPix[i])
					}
				}
			}
		}
	}
}

func TestAddResidualPlaneBlockNEONIsZeroAlloc(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("NEON not detected")
	}
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
