// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package prediction

import "testing"

// filterIntraAVX2TestSizes are every valid AV1 filter-intra transform size
// (width multiple of four, height multiple of two, both <= 32), covering the
// square and rectangular shapes plus the width tail of the vector loop.
var filterIntraAVX2TestSizes = [...]directionalTestSize{
	{4, 4}, {8, 8}, {16, 16}, {32, 32},
	{4, 8}, {8, 4}, {8, 16}, {16, 8},
	{16, 32}, {32, 16}, {4, 16}, {16, 4},
	{8, 32}, {32, 8}, {4, 2}, {32, 2},
	{12, 8}, {20, 16}, {28, 32},
}

func filterIntraEdges(width, height int, seed uint32) IntraEdges {
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

// TestFilterIntraAVX2MatchesPureGo executes the AVX2 kernel directly against the
// pure-Go reference for every mode, every valid transform size and a spread of
// pseudo-random edges. Every output sample must match bit for bit.
func TestFilterIntraAVX2MatchesPureGo(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraAVX2TestSizes {
			for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df, 0xdeadbeef} {
				edges := filterIntraEdges(sz.width, sz.height, seed+uint32(mode)*131)
				base := makeDispatchBlock(sz.width, sz.height, 1)
				got := cloneBlock(base)
				want := cloneBlock(base)
				predictFilterIntraBlockDirect8AVX2(got, sz.width, sz.height, mode, edges, 0xff)
				predictFilterIntraBlockDirect8(want, sz.width, sz.height, mode, edges, 0xff)
				diffBlocks(t, "filter-intra-avx2", 0xff, sz.width, sz.height, got, want)
			}
		}
	}
}

// TestFilterIntraAVX2EdgeValues stresses the clamp corners of the packed
// saturation (all zero, all max, alternating, ramp) which exercise the [0,255]
// clamp on both the negative and positive tap sums.
func TestFilterIntraAVX2EdgeValues(t *testing.T) {
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
		for _, sz := range filterIntraAVX2TestSizes {
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
						predictFilterIntraBlockDirect8AVX2(got, sz.width, sz.height, mode, edges, 0xff)
						predictFilterIntraBlockDirect8(want, sz.width, sz.height, mode, edges, 0xff)
						diffBlocks(t, "filter-intra-avx2-edge", 0xff, sz.width, sz.height, got, want)
					}
				}
			}
		}
	}
}

// TestFilterIntraAVX2ZeroAlloc protects the hot-path contract: the AVX2
// predictor must not allocate per call.
func TestFilterIntraAVX2ZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	edges := filterIntraEdges(w, h, 0x24)
	block := makeDispatchBlock(w, h, 1)
	fn := func() {
		predictFilterIntraBlockDirect8AVX2(block, w, h, FilterIntraModePaeth, edges, 0xff)
	}
	if allocs := testing.AllocsPerRun(1000, fn); allocs != 0 {
		t.Fatalf("filter-intra AVX2 allocated %f times per call", allocs)
	}
}
