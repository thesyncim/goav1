// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package prediction

import "testing"

// filterIntraSIMDTestSizes are every valid AV1 filter-intra transform size
// (width multiple of four, height multiple of two, both <= 32), covering the
// square and rectangular shapes plus the width tail of the vector loop.
var filterIntraSIMDTestSizes = [...]directionalTestSize{
	{4, 4}, {8, 8}, {16, 16}, {32, 32},
	{4, 8}, {8, 4}, {8, 16}, {16, 8},
	{16, 32}, {32, 16}, {4, 16}, {16, 4},
	{8, 32}, {32, 8}, {4, 2}, {32, 2},
	{12, 8}, {20, 16}, {28, 32},
}

func filterIntraSIMDEdges(width, height int, seed uint32) IntraEdges {
	above, s1 := samplesFor(width, 0xff, seed)
	left, _ := samplesFor(height, 0xff, s1)
	return IntraEdges{
		Above:              above,
		Left:               left,
		AboveLeft:          uint16((seed >> 3) & 0xff),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
}

// TestFilterIntraTapSumsFitInt16 proves the 8-bit SIMD kernel's int16 lanes
// cannot overflow: for every mode and output, the largest and smallest tap sums
// over 8-bit neighbours, plus the rounding bias, must fit in int16.
func TestFilterIntraTapSumsFitInt16(t *testing.T) {
	for mode := range filterIntraTaps {
		for output := 0; output < 8; output++ {
			hi, lo := 0, 0
			for tap := 0; tap < 7; tap++ {
				c := int(filterIntraTaps[mode][output][tap])
				if c > 0 {
					hi += c * 0xff
				} else {
					lo += c * 0xff
				}
			}
			if hi+(1<<(filterIntraScaleBits-1)) > 1<<15-1 || lo < -(1<<15) {
				t.Fatalf("mode=%d output=%d tap sums [%d,%d] overflow int16", mode, output, lo, hi)
			}
		}
	}
}

// TestFilterIntra8SIMDMatchesPureGo executes the 8-bit SIMD kernel directly
// against the pure-Go reference for every mode, every valid transform size and
// a spread of pseudo-random edges. Every output sample must match bit for bit.
func TestFilterIntra8SIMDMatchesPureGo(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df, 0xdeadbeef} {
				edges := filterIntraSIMDEdges(sz.width, sz.height, seed+uint32(mode)*131)
				base := makeDispatchBlock(sz.width, sz.height, 1)
				got := cloneBlock(base)
				want := cloneBlock(base)
				predictFilterIntraBlockDirect8SIMD(got, sz.width, sz.height, mode, edges, 0xff)
				predictFilterIntraBlockDirect8(want, sz.width, sz.height, mode, edges, 0xff)
				diffBlocks(t, "filter-intra-simd", 0xff, sz.width, sz.height, got, want)
			}
		}
	}
}

// TestFilterIntra8SIMDEdgeValues stresses the clamp corners (all zero, all max,
// alternating, ramp) which exercise the [0,255] saturation on both the negative
// and positive tap sums, including the top-left and left-column chaining.
func TestFilterIntra8SIMDEdgeValues(t *testing.T) {
	patterns := []func(i int) uint16{
		func(i int) uint16 { return 0 },
		func(i int) uint16 { return 255 },
		func(i int) uint16 {
			if i%2 == 0 {
				return 0
			}
			return 255
		},
		func(i int) uint16 { return uint16(i * 37 % 256) },
	}
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, ap := range patterns {
				for _, lp := range patterns {
					above := make([]uint16, sz.width)
					left := make([]uint16, sz.height)
					for i := range above {
						above[i] = ap(i)
					}
					for i := range left {
						left[i] = lp(i)
					}
					for _, al := range []uint16{0, 128, 255} {
						edges := IntraEdges{
							Above:              above,
							Left:               left,
							AboveLeft:          al,
							AboveAvailable:     true,
							LeftAvailable:      true,
							AboveLeftAvailable: true,
						}
						base := makeDispatchBlock(sz.width, sz.height, 1)
						got := cloneBlock(base)
						want := cloneBlock(base)
						predictFilterIntraBlockDirect8SIMD(got, sz.width, sz.height, mode, edges, 0xff)
						predictFilterIntraBlockDirect8(want, sz.width, sz.height, mode, edges, 0xff)
						diffBlocks(t, "filter-intra-simd-edge", 0xff, sz.width, sz.height, got, want)
					}
				}
			}
		}
	}
}

// TestFilterIntra8SIMDZeroAlloc protects the hot-path contract: the SIMD
// predictor must not allocate per call.
func TestFilterIntra8SIMDZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	edges := filterIntraSIMDEdges(w, h, 0x24)
	block := makeDispatchBlock(w, h, 1)
	fn := func() {
		predictFilterIntraBlockDirect8SIMD(block, w, h, FilterIntraModePaeth, edges, 0xff)
	}
	if allocs := testing.AllocsPerRun(1000, fn); allocs != 0 {
		t.Fatalf("filter-intra SIMD allocated %f times per call", allocs)
	}
}
