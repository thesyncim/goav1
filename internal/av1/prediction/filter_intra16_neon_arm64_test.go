// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

import "testing"

// filterIntra16Depths are the high-bit-depth clamps the NEON kernel must honour:
// 10-bit (max 1023) and 12-bit (max 4095, whose tap sums overflow int16 and so
// exercise the 32-bit widening path).
var filterIntra16Depths = [...]int{1023, 4095}

func filterIntra16Edges(width, height int, max int, seed uint32) IntraEdges {
	above, s1 := samplesFor(width, uint16(max), seed)
	left, s2 := samplesFor(height, uint16(max), s1)
	_ = s2
	return IntraEdges{
		Above:              above,
		Left:               left,
		AboveLeft:          uint16((seed >> 3) % uint32(max+1)),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
}

// TestFilterIntra16NEONMatchesPureGo executes the 16-bit NEON kernel directly
// against the pure-Go reference for every mode, every valid transform size, both
// high bit depths and a spread of pseudo-random edges. Every output sample must
// match bit for bit.
func TestFilterIntra16NEONMatchesPureGo(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraNEONTestSizes {
			for _, max := range filterIntra16Depths {
				for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df, 0xdeadbeef} {
					edges := filterIntra16Edges(sz.width, sz.height, max, seed+uint32(mode)*131)
					base := makeDispatchBlock(sz.width, sz.height, 2)
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictFilterIntraBlockDirect16NEON(got, sz.width, sz.height, mode, edges, max)
					predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
					diffBlocks(t, "filter-intra16-neon", max, sz.width*2, sz.height, got, want)
				}
			}
		}
	}
}

// TestFilterIntra16NEONEdgeValues stresses the clamp corners (all zero, all max,
// alternating, ramp) which exercise the [0,max] saturation on both the negative
// tap sums (clamp low via sqrshrun) and positive tap sums (clamp high via smin).
func TestFilterIntra16NEONEdgeValues(t *testing.T) {
	for _, max := range filterIntra16Depths {
		patterns := []func(i int) uint16{
			func(i int) uint16 { return 0 },
			func(i int) uint16 { return uint16(max) },
			func(i int) uint16 {
				if i%2 == 0 {
					return 0
				}
				return uint16(max)
			},
			func(i int) uint16 { return uint16(i * 37 % (max + 1)) },
		}
		for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
			for _, sz := range filterIntraNEONTestSizes {
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
						for _, al := range []uint16{0, uint16(max / 2), uint16(max)} {
							edges := IntraEdges{
								Above:              above,
								Left:               left,
								AboveLeft:          al,
								AboveAvailable:     true,
								LeftAvailable:      true,
								AboveLeftAvailable: true,
							}
							base := makeDispatchBlock(sz.width, sz.height, 2)
							got := cloneBlock(base)
							want := cloneBlock(base)
							predictFilterIntraBlockDirect16NEON(got, sz.width, sz.height, mode, edges, max)
							predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
							diffBlocks(t, "filter-intra16-neon-edge", max, sz.width*2, sz.height, got, want)
						}
					}
				}
			}
		}
	}
}

// TestFilterIntra16NEONZeroAlloc protects the hot-path contract: the 16-bit NEON
// predictor must not allocate per call.
func TestFilterIntra16NEONZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	edges := filterIntra16Edges(w, h, 1023, 0x24)
	block := makeDispatchBlock(w, h, 2)
	fn := func() {
		predictFilterIntraBlockDirect16NEON(block, w, h, FilterIntraModePaeth, edges, 1023)
	}
	if allocs := testing.AllocsPerRun(1000, fn); allocs != 0 {
		t.Fatalf("filter-intra16 NEON allocated %f times per call", allocs)
	}
}
