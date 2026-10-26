// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// The AMD64 MC benchmarks run the public 8-bit predictors with the AVX2
// kernels (impl=avx2) and with them switched off (impl=generic, the general
// Go SIMD path amd64 ran before).

var mc8BenchImpls = []struct {
	name string
	on   bool
}{{"generic", false}, {"avx2", true}}

func mc8BenchMV(mode string) Vector {
	// Q3 luma vector: column phase 5, row phase 11 (in 1/16 units 10, 6).
	mv := Vector{Row: 8*3 + 3, Col: 8*2 + 5}
	switch mode {
	case "h", "hsharp":
		mv.Row = 8 * 3
	case "v", "vsharp":
		mv.Col = 8 * 2
	case "copy":
		mv = Vector{Row: 8 * 3, Col: 8 * 2}
	}
	return mv
}

func BenchmarkAMD64MC8Put(b *testing.B) {
	requireAVX2MC8(b)
	rng := rand.New(rand.NewSource(3))
	ref := mc8TestRef(rng, 320, 320, 352, 0)
	var scratch ConvolveScratch
	dst := frame.Plane{Pix: make([]byte, 192*192), Stride: 192, Width: 192, Height: 192}
	for _, sz := range [][2]int{{4, 4}, {4, 8}, {8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		w, h := sz[0], sz[1]
		for _, mode := range []string{"hv", "h", "v", "hvsharp", "hsharp", "vsharp"} {
			mv := mc8BenchMV(mode)
			filters := RegularFilters
			if strings.HasSuffix(mode, "sharp") {
				filters = InterpFilters{X: InterpMultiTapSharp, Y: InterpMultiTapSharp}
			}
			for _, impl := range mc8BenchImpls {
				b.Run(fmt.Sprintf("mode=%s/shape=%dx%d/impl=%s", mode, w, h, impl.name), func(b *testing.B) {
					mc8AVX2 = impl.on
					defer func() { mc8AVX2 = true }()
					for b.Loop() {
						if err := PredictTranslation8(&dst, &ref, 64, 64, w, h, w, h, mv, false, false, filters, &scratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

func BenchmarkAMD64MC8Prep(b *testing.B) {
	requireAVX2MC8(b)
	rng := rand.New(rand.NewSource(4))
	ref := mc8TestRef(rng, 320, 320, 352, 0)
	scratch := new(CompoundConvolveScratch)
	buf := new(CompoundConvBuf)
	for _, sz := range [][2]int{{4, 4}, {4, 8}, {8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		w, h := sz[0], sz[1]
		for _, mode := range []string{"hv", "h", "v", "hvsharp", "hsharp", "vsharp", "copy"} {
			mv := mc8BenchMV(mode)
			filters := RegularFilters
			if strings.HasSuffix(mode, "sharp") {
				filters = InterpFilters{X: InterpMultiTapSharp, Y: InterpMultiTapSharp}
			}
			for _, impl := range mc8BenchImpls {
				b.Run(fmt.Sprintf("mode=%s/shape=%dx%d/impl=%s", mode, w, h, impl.name), func(b *testing.B) {
					mc8AVX2 = impl.on
					defer func() { mc8AVX2 = true }()
					for b.Loop() {
						if err := PrepTranslation8(buf, &ref, 64, 64, w, h, w, h, mv, false, false, filters, scratch); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

func BenchmarkAMD64MC8BlendAvg(b *testing.B) {
	requireAVX2MC8(b)
	rng := rand.New(rand.NewSource(6))
	b0, b1 := new(CompoundConvBuf), new(CompoundConvBuf)
	for i := range b0.Data {
		b0.Data[i] = uint16(4000 + rng.Intn(8000))
		b1.Data[i] = uint16(4000 + rng.Intn(8000))
	}
	dst := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
	for _, sz := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		w, h := sz[0], sz[1]
		b0.Width, b0.Height, b1.Width, b1.Height = uint8(w), uint8(h), uint8(w), uint8(h)
		for _, impl := range mc8BenchImpls {
			b.Run(fmt.Sprintf("shape=%dx%d/impl=%s", w, h, impl.name), func(b *testing.B) {
				mc8AVX2 = impl.on
				defer func() { mc8AVX2 = true }()
				for b.Loop() {
					if err := BlendCompoundAvg(dst, b0, b1, 1, 8, 0, 0, w, h, 8, 8); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkAMD64MC8MaskBlend(b *testing.B) {
	requireAVX2MC8(b)
	rng := rand.New(rand.NewSource(11))
	b0, b1 := new(CompoundConvBuf), new(CompoundConvBuf)
	for i := range b0.Data {
		b0.Data[i] = uint16(4000 + rng.Intn(8000))
		b1.Data[i] = uint16(4000 + rng.Intn(8000))
	}
	mask := make([]byte, 256*256)
	for i := range mask {
		mask[i] = byte(rng.Intn(65))
	}
	dst := frame.Plane{Pix: make([]byte, 128*128), Stride: 128, Width: 128, Height: 128}
	for _, sz := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {64, 64}} {
		w, h := sz[0], sz[1]
		b0.Width, b0.Height, b1.Width, b1.Height = uint8(w), uint8(h), uint8(w), uint8(h)
		for _, op := range []string{"mask", "mask420", "diffwtd", "wavg"} {
			for _, impl := range mc8BenchImpls {
				b.Run(fmt.Sprintf("op=%s/shape=%dx%d/impl=%s", op, w, h, impl.name), func(b *testing.B) {
					mc8AVX2 = impl.on
					defer func() { mc8AVX2 = true }()
					for b.Loop() {
						var err error
						switch op {
						case "mask":
							err = BlendCompoundMaskD16(dst, b0, b1, 1, 8, 0, 0, w, h, mask, 128, false, false)
						case "mask420":
							err = BlendCompoundMaskD16(dst, b0, b1, 1, 8, 0, 0, w, h, mask, 256, true, true)
						case "diffwtd":
							err = BuildDiffWtdMaskD16(mask, 128, b0, b1, 8, w, h, false)
						default:
							err = BlendCompoundAvg(dst, b0, b1, 1, 8, 0, 0, w, h, 9, 7)
						}
						if err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
