// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "testing"

// filterIntra16Depths are the high-bit-depth clamps the SIMD kernel must honour:
// 10-bit (max 1023) and 12-bit (max 4095, whose tap sums overflow int16 and so
// exercise the 32-bit widening path).
var filterIntra16Depths = [...]int{1023, 4095}

func filterIntra16SIMDEdges(width, height int, max int, seed uint32) IntraEdges {
	above, s1 := samplesFor(width, uint16(max), seed)
	left, _ := samplesFor(height, uint16(max), s1)
	return IntraEdges{
		Above:              above,
		Left:               left,
		AboveLeft:          uint16((seed >> 3) % uint32(max+1)),
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
}

// TestFilterIntraSIMDBinding asserts the arm64 SIMD build binds the Go-native
// filter-intra kernels, so the differential tests below exercise the production
// dispatch target.
func TestFilterIntraSIMDBinding(t *testing.T) {
	assertDispatchTarget(t, "predictFilterIntra8Impl", predictFilterIntra8Impl, "predictFilterIntraBlockDirect8SIMD")
	assertDispatchTarget(t, "predictFilterIntra16Impl", predictFilterIntra16Impl, "predictFilterIntraBlockDirect16SIMD")
}

// TestFilterIntra16SIMDMatchesPureGo executes the 16-bit SIMD kernel directly
// against the pure-Go reference for every mode, every valid transform size, both
// high bit depths and a spread of pseudo-random edges.
func TestFilterIntra16SIMDMatchesPureGo(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, max := range filterIntra16Depths {
				for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df, 0xdeadbeef} {
					edges := filterIntra16SIMDEdges(sz.width, sz.height, max, seed+uint32(mode)*131)
					base := makeDispatchBlock(sz.width, sz.height, 2)
					got := cloneBlock(base)
					want := cloneBlock(base)
					predictFilterIntraBlockDirect16SIMD(got, sz.width, sz.height, mode, edges, max)
					predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
					diffBlocks(t, "filter-intra16-simd", max, sz.width*2, sz.height, got, want)
				}
			}
		}
	}
}

// TestFilterIntra16SIMDEdgeValues stresses the [0,max] clamp corners at both
// high bit depths, where the 32-bit tap sums saturate in both directions.
func TestFilterIntra16SIMDEdgeValues(t *testing.T) {
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for _, max := range filterIntra16Depths {
				for _, pat := range []uint16{0, uint16(max), uint16(max / 2), uint16(max - 1)} {
					above := make([]uint16, sz.width)
					left := make([]uint16, sz.height)
					for i := range above {
						above[i] = pat
						if i%3 == 0 {
							above[i] = uint16(max) - pat
						}
					}
					for i := range left {
						left[i] = uint16(max) - pat
						if i%2 == 0 {
							left[i] = pat
						}
					}
					for _, al := range []uint16{0, uint16(max)} {
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
						predictFilterIntraBlockDirect16SIMD(got, sz.width, sz.height, mode, edges, max)
						predictFilterIntraBlockDirect16(want, sz.width, sz.height, mode, edges, max)
						diffBlocks(t, "filter-intra16-simd-edge", max, sz.width*2, sz.height, got, want)
					}
				}
			}
		}
	}
}

// TestFilterIntra16SIMDZeroAlloc protects the hot-path contract: the 16-bit SIMD
// predictor must not allocate per call.
func TestFilterIntra16SIMDZeroAlloc(t *testing.T) {
	const w, h = 32, 32
	edges := filterIntra16SIMDEdges(w, h, 1023, 0x24)
	block := makeDispatchBlock(w, h, 2)
	fn := func() {
		predictFilterIntraBlockDirect16SIMD(block, w, h, FilterIntraModePaeth, edges, 1023)
	}
	if allocs := testing.AllocsPerRun(1000, fn); allocs != 0 {
		t.Fatalf("filter-intra16 SIMD allocated %f times per call", allocs)
	}
}
