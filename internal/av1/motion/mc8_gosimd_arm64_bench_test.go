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
	for _, sz := range [][2]int{{2, 8}, {4, 4}, {4, 16}, {8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		w, h := sz[0], sz[1]
		dst := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
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
					Put8(&dst, &ref, 0, 0, 40, 40, w, h, w, h, subX, subY, RegularFilters, &scratch)
				}
			})
		}
	}
}

// BenchmarkMC8PutUV measures a chroma block pair: two Put8 calls (one per
// plane) against one Put8UV.
func BenchmarkMC8PutUV(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	refU := mc8TestRef(rng, 320, 320, 352, 0)
	refV := mc8TestRef(rng, 320, 320, 352, 0)
	var scratch ConvolveScratch
	dstU := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
	dstV := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
	for _, sz := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}} {
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
			b.Run(fmt.Sprintf("pair_%s_%dx%d", mode, w, h), func(b *testing.B) {
				for b.Loop() {
					Put8(&dstU, &refU, 0, 0, 40, 40, w, h, w, h, subX, subY, RegularFilters, &scratch)
					Put8(&dstV, &refV, 0, 0, 40, 40, w, h, w, h, subX, subY, RegularFilters, &scratch)
				}
			})
			b.Run(fmt.Sprintf("uv_%s_%dx%d", mode, w, h), func(b *testing.B) {
				for b.Loop() {
					Put8UV(&dstU, &dstV, &refU, &refV, 0, 0, 40, 40, w, h, w, h, subX, subY, RegularFilters, &scratch)
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
	for _, sz := range [][2]int{{4, 4}, {4, 16}, {8, 8}, {16, 16}, {32, 32}} {
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
	for _, sz := range [][2]int{{4, 8}, {8, 8}, {16, 16}, {32, 32}} {
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

// BenchmarkMC8Mask measures the soft-mask blend (full-resolution and 4:2:0
// mask) and the difference-weighted mask builder.
func BenchmarkMC8Mask(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	a := make([]uint16, 128*128)
	c := make([]uint16, 128*128)
	for i := range a {
		a[i] = uint16(4000 + rng.Intn(8000))
		c[i] = uint16(4000 + rng.Intn(8000))
	}
	mask := make([]byte, 256*256+16)
	for i := range mask {
		mask[i] = byte(rng.Intn(65))
	}
	dst := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
	for _, sz := range [][2]int{{8, 8}, {16, 16}, {32, 32}} {
		w, h := sz[0], sz[1]
		for _, sub := range []bool{false, true} {
			ms := w
			if sub {
				ms = 2 * w
			}
			b.Run(fmt.Sprintf("blend_sub%v_%dx%d", sub, w, h), func(b *testing.B) {
				for b.Loop() {
					mc8BlendMaskFast(dst, a, c, 0, 0, w, h, mask, ms, sub, sub)
				}
			})
		}
		b.Run(fmt.Sprintf("diffwtd_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				mc8DiffWtdMaskFast(mask, w, a, c, w, h, false)
			}
		})
	}
}

// BenchmarkMC16 measures the 10-bit dav1d-style put/prep kernels (regular
// and sharp), the 16bpc copy, and the 16bpc average, mask blend and
// difference-weighted mask.
func BenchmarkMC16(b *testing.B) {
	rng := rand.New(rand.NewSource(8))
	ref := mc16TestRef(rng, 320, 320, 352, 10, 0)
	var scratch ConvolveScratch
	cscratch := new(CompoundConvolveScratch)
	out := make([]uint16, 128*128)
	dst := frame.Plane{Pix: make([]byte, 2*128*128), Stride: 256, Width: 128, Height: 128}
	sharp := InterpFilters{X: InterpMultiTapSharp, Y: InterpMultiTapSharp}
	for _, sz := range [][2]int{{4, 4}, {4, 16}, {8, 8}, {16, 16}, {32, 32}} {
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
			for _, fn := range []struct {
				name string
				f    InterpFilters
			}{{"reg", RegularFilters}, {"sharp", sharp}} {
				if mode == "copy" && fn.name == "sharp" {
					continue
				}
				if mode != "copy" {
					b.Run(fmt.Sprintf("put_%s_%s_%dx%d", fn.name, mode, w, h), func(b *testing.B) {
						for b.Loop() {
							mc16PutFast(&dst, &ref, 10, 0, 0, 40, 40, w, h, w, h, subX, subY, fn.f, &scratch)
						}
					})
				} else {
					b.Run(fmt.Sprintf("copy_%dx%d", w, h), func(b *testing.B) {
						for b.Loop() {
							mc16CopyFast(&dst, &ref, 0, 0, 40, 40, w, h)
						}
					})
				}
				b.Run(fmt.Sprintf("prep_%s_%s_%dx%d", fn.name, mode, w, h), func(b *testing.B) {
					for b.Loop() {
						mc16PrepFast(out[:w*h], &ref, 10, 40, 40, w, h, w, h, subX, subY, fn.f, cscratch)
					}
				})
			}
		}
	}
	a := make([]uint16, 128*128)
	c := make([]uint16, 128*128)
	for i := range a {
		a[i] = uint16(rng.Intn(1 << 14))
		c[i] = uint16(rng.Intn(1 << 14))
	}
	mask := make([]byte, 256*256+16)
	for i := range mask {
		mask[i] = byte(rng.Intn(65))
	}
	for _, sz := range [][2]int{{4, 8}, {8, 8}, {16, 16}, {32, 32}} {
		w, h := sz[0], sz[1]
		b.Run(fmt.Sprintf("avg_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				mc16BlendAvgFast(&dst, a, c, 10, 0, 0, w, h)
			}
		})
		if w < 8 {
			continue
		}
		for _, sub := range []bool{false, true} {
			ms := w
			if sub {
				ms = 2 * w
			}
			b.Run(fmt.Sprintf("mask_sub%v_%dx%d", sub, w, h), func(b *testing.B) {
				for b.Loop() {
					mc16BlendMaskFast(dst, a, c, 10, 0, 0, w, h, mask, ms, sub, sub)
				}
			})
		}
		b.Run(fmt.Sprintf("diffwtd_%dx%d", w, h), func(b *testing.B) {
			for b.Loop() {
				mc16DiffWtdMaskFast(mask, w, a, c, 10, w, h, false)
			}
		})
	}
}
