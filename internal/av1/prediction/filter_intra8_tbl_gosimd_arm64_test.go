// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "testing"

func TestFilterIntra8TBLMatchesPureGo(t *testing.T) {
	patterns := []func(i int, seed uint32) uint16{
		func(i int, seed uint32) uint16 { return uint16((uint32(i)*2654435761 + seed) >> 7 & 0xff) },
		func(i int, seed uint32) uint16 { return 0 },
		func(i int, seed uint32) uint16 { return 255 },
		func(i int, seed uint32) uint16 { return uint16(255 * (i & 1)) },
	}
	for mode := FilterIntraMode(0); mode < FilterIntraModes; mode++ {
		for _, sz := range filterIntraSIMDTestSizes {
			for pi, ap := range patterns {
				for _, lp := range patterns {
					for _, seed := range []uint32{0x1, 0xabcd, 0x5f3759df} {
						above := make([]uint16, sz.width)
						left := make([]uint16, sz.height)
						for i := range above {
							above[i] = ap(i, seed)
						}
						for i := range left {
							left[i] = lp(i, seed*3+uint32(pi))
						}
						edges := IntraEdges{Above: above, Left: left, AboveLeft: uint16(seed & 0xff),
							AboveAvailable: true, LeftAvailable: true, AboveLeftAvailable: true}
						base := makeDispatchBlock(sz.width, sz.height, 1)
						got := cloneBlock(base)
						want := cloneBlock(base)
						predictFilterIntraBlockDirect8TBL(got, sz.width, sz.height, mode, edges, 0xff)
						predictFilterIntraBlockDirect8(want, sz.width, sz.height, mode, edges, 0xff)
						diffBlocks(t, "filter-intra-tbl", 0xff, sz.width, sz.height, got, want)
					}
				}
			}
		}
	}
}

func BenchmarkFilterIntra8Kernels(b *testing.B) {
	edges := filterIntraSIMDEdges(32, 32, 0xabcd)
	blk := makeDispatchBlock(32, 32, 1)
	b.Run("tbl", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8TBL(blk, 32, 32, FilterIntraModeD157, edges, 0xff)
		}
	})
	b.Run("bcast", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8SIMD(blk, 32, 32, FilterIntraModeD157, edges, 0xff)
		}
	})
	edges8 := filterIntraSIMDEdges(8, 8, 0xabcd)
	b.Run("tbl8x8", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8TBL(blk, 8, 8, FilterIntraModeD157, edges8, 0xff)
		}
	})
	b.Run("bcast8x8", func(b *testing.B) {
		for b.Loop() {
			predictFilterIntraBlockDirect8SIMD(blk, 8, 8, FilterIntraModeD157, edges8, 0xff)
		}
	})
}
