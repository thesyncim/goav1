// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// BenchmarkMC8Put measures the dav1d-style put kernels on interior blocks
// (regular filter, both axes fractional unless noted).
func BenchmarkMC8Put(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	ref := mc8TestRef(rng, 320, 320, 352, 0)
	var scratch ConvolveScratch
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		w, h := sz[0], sz[1]
		dst := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
		for _, mode := range []string{"hv", "h", "v"} {
			subX, subY := 5, 11
			switch mode {
			case "h":
				subY = 0
			case "v":
				subX = 0
			}
			b.Run(fmt.Sprintf("%s_%dx%d", mode, w, h), func(b *testing.B) {
				for b.Loop() {
					Put8(&dst, &ref, 0, 0, 40, 40, w, h, w, h, subX, subY, RegularFilters, &scratch)
				}
			})
		}
	}
}

// BenchmarkMC8Prep measures the dav1d-style prep kernels on interior blocks.
func BenchmarkMC8Prep(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	ref := mc8TestRef(rng, 320, 320, 352, 0)
	scratch := new(CompoundConvolveScratch)
	out := make([]uint16, 128*128)
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}} {
		w, h := sz[0], sz[1]
		for _, mode := range []string{"hv", "h", "v", "copy"} {
			subX, subY := 5, 11
			switch mode {
			case "h":
				subY = 0
			case "v":
				subX = 0
			case "copy":
				subX, subY = 0, 0
			}
			b.Run(fmt.Sprintf("%s_%dx%d", mode, w, h), func(b *testing.B) {
				for b.Loop() {
					mc8PrepFast(out[:w*h], &ref, 40, 40, w, h, w, h, subX, subY, RegularFilters, scratch)
				}
			})
		}
	}
}

// BenchmarkMC8BlendAvg compares the int16 plain average with the general
// distance-weighted SIMD blend it replaces for fwd = bck = 8.
func BenchmarkMC8BlendAvg(b *testing.B) {
	rng := rand.New(rand.NewSource(6))
	a := make([]uint16, 128*128)
	c := make([]uint16, 128*128)
	for i := range a {
		a[i] = uint16(4000 + rng.Intn(8000))
		c[i] = uint16(4000 + rng.Intn(8000))
	}
	dst := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}} {
		w, h := sz[0], sz[1]
		if !mc8BlendAvgFast(&dst, a, c, 0, 0, w, h) {
			b.Fatalf("mc8BlendAvgFast declined benchmark size %dx%d", w, h)
		}
		b.Run(fmt.Sprintf("int16_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				mc8BlendAvgFast(&dst, a, c, 0, 0, w, h)
			}
		})
		b.Run(fmt.Sprintf("general_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				blendCompoundAvg8GoSIMD(dst, a, c, 0, 0, w, h, 8, 8, 6144, 4)
			}
		})
	}
}
