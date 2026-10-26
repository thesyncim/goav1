// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import (
	"simd/archsimd"
	"testing"
)

// TestAMD64CDEFByteRouteDispatch asserts that an AVX2 CPU takes the byte-lane
// interior route (dav1d's 8bpc edges == 0xf kernels) instead of declining to
// the uint16 tap buffer.
func TestAMD64CDEFByteRouteDispatch(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("CPU does not advertise AVX2")
	}
	if !U8ByteInteriorAvailable() {
		t.Fatal("AVX2 CPU declines the byte-interior CDEF route")
	}
}

// BenchmarkAMD64CDEFUnitU8 times a full 8-bit unit through the byte-lane AVX2
// route (impl=avx2) against the uint16 tap buffer route amd64 used before
// (impl=generic: FilterFrameBlocksU8Trusted with the 128-bit block kernel).
// Directions are precomputed, so only the filter is timed.
func BenchmarkAMD64CDEFUnitU8(b *testing.B) {
	type shape struct {
		name       string
		dec        uint8
		plane      Plane
		level, sec uint8
	}
	for _, s := range []shape{
		{"luma8x8/prisec", 0, PlaneY, 6, 2},
		{"luma8x8/pri", 0, PlaneY, 6, 0},
		{"luma8x8/sec", 0, PlaneY, 0, 2},
		{"chroma4x4/prisec", 1, PlaneU, 5, 1},
		{"chroma4x4/pri", 1, PlaneU, 5, 0},
	} {
		rng := byteUnitRNG(0x1234567)
		unitW, unitH := BlockSize>>s.dec, BlockSize>>s.dec
		bw, bh := 8>>s.dec, 8>>s.dec
		input, wide, origin := makeByteUnitInput(&rng, unitW, unitH, 5)
		var blocks []BlockPosition
		for by := 0; by < unitH/bh; by++ {
			for bx := 0; bx < unitW/bw; bx++ {
				blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
			}
		}
		var dirs DirectionGrid
		var vars VarianceGrid
		for i := range dirs {
			for j := range dirs[i] {
				dirs[i][j] = uint8(rng.next() % 8)
				vars[i][j] = 1 << 12
			}
		}
		params := FrameFilterParams{XDec: s.dec, YDec: s.dec, Plane: s.plane, Level: s.level, SecondaryStrength: s.sec, Damping: 5}
		dst := make([]byte, unitW*unitH)
		b.Run("shape="+s.name+"/impl=generic", func(b *testing.B) {
			for b.Loop() {
				_ = FilterFrameBlocksU8TrustedNoPrimaryDirection(dst, unitW, wide, origin, blocks, &dirs, &vars, params)
			}
		})
		b.Run("shape="+s.name+"/impl=avx2", func(b *testing.B) {
			if !U8ByteInteriorAvailable() {
				b.Skip("no AVX2")
			}
			for b.Loop() {
				if !FilterFrameBlocksU8ByteInteriorTrusted(dst, unitW, input, origin, unitW, unitH, blocks, &dirs, &vars, params, false) {
					b.Fatal("declined")
				}
			}
		})
	}
}
