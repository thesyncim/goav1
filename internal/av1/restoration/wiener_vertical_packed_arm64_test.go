// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import "testing"

func TestWienerVerticalPackedEdgesAndTails(t *testing.T) {
	filters := []WienerFilter{
		NewWienerFilter(WienerTap0Min, WienerTap1Min, WienerTap2Min),
		NewWienerFilter(WienerTap0Max, WienerTap1Max, WienerTap2Max),
		DefaultWienerInfo().VFilter,
	}
	for _, info := range wienerDispatchFilters() {
		filters = append(filters, info.VFilter)
	}
	for _, bitDepth := range []int{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(bitDepth)
		maxClamp := uint16((1 << (bitDepth + 1 + WienerFilterBits - round0)) - 1)
		for _, width := range []int{1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 64} {
			const height = 5
			tempStride := width + 3
			dstStride := width + 2
			temp := make([]uint16, tempStride*(height+2*WienerHalfwin))
			for i := range temp {
				if i%3 != 0 {
					temp[i] = maxClamp
				}
			}
			for _, filter := range filters {
				want := make([]uint16, dstStride*height)
				got := make([]uint16, dstStride*height)
				wienerVertical(temp, tempStride, want, dstStride, width, height, filter, bitDepth, round1, max)
				wienerVerticalSIMD(temp, tempStride, got, dstStride, width, height, filter, bitDepth, round1, max)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("u16 depth=%d width=%d filter=%v index=%d got=%d want=%d", bitDepth, width, filter, i, got[i], want[i])
					}
				}
				if bitDepth == 8 {
					want8 := make([]uint8, dstStride*height)
					got8 := make([]uint8, dstStride*height)
					wienerVerticalU8(temp, tempStride, want8, dstStride, width, height, filter, round1)
					wienerVerticalU8SIMD(temp, tempStride, got8, dstStride, width, height, filter, round1)
					for i := range want8 {
						if got8[i] != want8[i] {
							t.Fatalf("u8 width=%d filter=%v index=%d got=%d want=%d", width, filter, i, got8[i], want8[i])
						}
					}
				}
			}
		}
	}
}
