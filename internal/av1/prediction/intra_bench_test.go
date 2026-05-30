// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package prediction

import (
	"fmt"
	"testing"
)

func benchInputs(w, h int) (above, left, weightsW, weightsH []uint16) {
	above, _ = samplesFor(w, 0xff, 7)
	left, _ = samplesFor(h, 0xff, 99)
	weightsW, _ = smoothWeightsForSize(w)
	weightsH, _ = smoothWeightsForSize(h)
	return
}

func benchSizes() [][2]int {
	return [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}}
}

func BenchmarkPaeth(b *testing.B) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, _, _ := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		for _, v := range []struct {
			name string
			fn   predictPaethFunc
		}{{"NEON", predictPaethImpl}, {"PureGo", predictPaethPureGo}} {
			b.Run(fmt.Sprintf("%dx%d/%s", w, h, v.name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					v.fn(block, 1, above, left, 123)
				}
			})
		}
	}
}

func BenchmarkSmooth(b *testing.B) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, weightsW, weightsH := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		for _, v := range []struct {
			name string
			fn   predictSmoothFunc
		}{{"NEON", predictSmoothImpl}, {"PureGo", predictSmoothPureGo}} {
			b.Run(fmt.Sprintf("%dx%d/%s", w, h, v.name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					v.fn(block, 1, weightsW, weightsH, above, left, left[h-1], above[w-1])
				}
			})
		}
	}
}

func BenchmarkSmoothVertical(b *testing.B) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, _, weightsH := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		for _, v := range []struct {
			name string
			fn   predictSmoothVerticalFunc
		}{{"NEON", predictSmoothVerticalImpl}, {"PureGo", predictSmoothVerticalPureGo}} {
			b.Run(fmt.Sprintf("%dx%d/%s", w, h, v.name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					v.fn(block, 1, weightsH, above, left[h-1])
				}
			})
		}
	}
}

func BenchmarkSmoothHorizontal(b *testing.B) {
	for _, s := range benchSizes() {
		w, h := s[0], s[1]
		above, left, weightsW, _ := benchInputs(w, h)
		block := makeDispatchBlock(w, h, 1)
		for _, v := range []struct {
			name string
			fn   predictSmoothHorizontalFunc
		}{{"NEON", predictSmoothHorizontalImpl}, {"PureGo", predictSmoothHorizontalPureGo}} {
			b.Run(fmt.Sprintf("%dx%d/%s", w, h, v.name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					v.fn(block, 1, weightsW, left, above[w-1])
				}
			})
		}
	}
}
