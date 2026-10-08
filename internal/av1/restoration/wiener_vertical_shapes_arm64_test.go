// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"fmt"
	"testing"
)

func BenchmarkWienerVerticalU8Shape(b *testing.B) {
	_, round1 := wienerRounds(8)
	filter := DefaultWienerInfo().VFilter
	for _, shape := range [][2]int{{8, 8}, {16, 8}, {32, 8}, {64, 8}, {64, 64}} {
		width, height := shape[0], shape[1]
		b.Run(fmt.Sprintf("%dx%d", width, height), func(b *testing.B) {
			temp := make([]uint16, width*(height+2*WienerHalfwin))
			for i := range temp {
				temp[i] = uint16((i*97 + 31) % 8192)
			}
			dst := make([]uint8, width*height)
			b.ReportAllocs()
			for b.Loop() {
				wienerVerticalU8SIMD(temp, width, dst, width, width, height, filter, round1)
			}
		})
	}
}

func BenchmarkWienerVerticalU16Shape(b *testing.B) {
	_, round1 := wienerRounds(12)
	filter := DefaultWienerInfo().VFilter
	for _, shape := range [][2]int{{8, 8}, {16, 8}, {32, 8}, {64, 8}, {64, 64}} {
		width, height := shape[0], shape[1]
		b.Run(fmt.Sprintf("%dx%d", width, height), func(b *testing.B) {
			temp := make([]uint16, width*(height+2*WienerHalfwin))
			for i := range temp {
				temp[i] = uint16((i*97 + 31) % 32768)
			}
			dst := make([]uint16, width*height)
			b.ReportAllocs()
			for b.Loop() {
				wienerVerticalSIMD(temp, width, dst, width, width, height, filter, 12, round1, 4095)
			}
		})
	}
}

func BenchmarkWienerHorizontalU16Shape(b *testing.B) {
	round0, _ := wienerRounds(12)
	filter := DefaultWienerInfo().HFilter
	for _, width := range []int{8, 32, 64, 256} {
		b.Run(fmt.Sprint(width), func(b *testing.B) {
			const height = 64
			stride := width + 2*WienerHalfwin + 2
			origin := WienerHalfwin*stride + WienerHalfwin
			src := make([]uint16, stride*(height+2*WienerHalfwin))
			for i := range src {
				src[i] = uint16((i*97 + 31) % 4096)
			}
			temp := make([]uint16, width*(height+2*WienerHalfwin))
			b.ReportAllocs()
			for b.Loop() {
				wienerHorizontalSIMDTrusted(src, stride, origin, width, height, filter, 12, round0, 4095, temp)
			}
		})
	}
}
