// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func TestGoSIMDDispatchProbe(t *testing.T) {
	if !cpu.Detected.NEON {
		t.Skip("arm64 NEON not detected")
	}
	assertDispatchTarget(t, "applyCFLImpl", applyCFLImpl, "applyCFLNEON")
	assertDispatchTarget(t, "subsampleLuma8Impl", subsampleLuma8Impl, "subsampleLuma8SIMD")
	assertDispatchTarget(t, "subsampleLuma16Impl", subsampleLuma16Impl, "subsampleLuma16SIMD")
	assertDispatchTarget(t, "subtractCFLAverageImpl", subtractCFLAverageImpl, "subtractCFLAverageSIMD")

	assertDispatchTarget(t, "predictPaethImpl", predictPaethImpl, "predictPaethNEON")
	assertDispatchTarget(t, "predictSmoothImpl", predictSmoothImpl, "predictSmoothNEON")
	assertDispatchTarget(t, "predictSmoothVerticalImpl", predictSmoothVerticalImpl, "predictSmoothVerticalNEON")
	assertDispatchTarget(t, "predictSmoothHorizontalImpl", predictSmoothHorizontalImpl, "predictSmoothHorizontalNEON")
	assertDispatchTarget(t, "sumSamplesImpl", sumSamplesImpl, "sumSamplesNEON")
	assertDispatchTarget(t, "dirRowInterp8Impl", dirRowInterp8Impl, "dirRowInterp8NEON")
	assertDispatchTarget(t, "predictFilterIntra8Impl", predictFilterIntra8Impl, "predictFilterIntraBlockDirect8NEON")
	assertDispatchTarget(t, "predictFilterIntra16Impl", predictFilterIntra16Impl, "predictFilterIntraBlockDirect16NEON")
}

func assertDispatchTarget(t *testing.T, slot string, fn any, want string) {
	t.Helper()
	pc := reflect.ValueOf(fn).Pointer()
	got := runtime.FuncForPC(pc).Name()
	if !strings.Contains(got, want) {
		t.Fatalf("%s bound to %s, want %s", slot, got, want)
	}
}

func BenchmarkCFLSubsample8Arm64(b *testing.B) {
	input := make([]uint8, 32*32)
	output := make([]uint16, CFLBufSquare)
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for i := range input {
		input[i] = uint8(rnd.pseudoUniform(256))
	}
	for _, mode := range []struct {
		name       string
		subX, subY bool
		outW, outH int
	}{
		{"444", false, false, 32, 32},
		{"422", true, false, 16, 32},
		{"420", true, true, 16, 16},
	} {
		for _, v := range []struct {
			name string
			fn   subsampleLuma8Func
		}{
			{"SIMD", subsampleLuma8SIMD},
			{"NEON", subsampleLuma8NEON},
			{"PureGo", subsampleLuma8PureGo},
		} {
			b.Run(fmt.Sprintf("%s/%s", mode.name, v.name), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					v.fn(output, input, 32, 32, 32, mode.outW, mode.outH, mode.subX, mode.subY)
				}
			})
		}
	}
}

func BenchmarkCFLSubtractAverageArm64(b *testing.B) {
	src := make([]uint16, CFLBufSquare)
	dst := make([]int16, CFLBufSquare)
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 15))
	}
	log2, _ := log2PowerOfTwoInt(32 * 32)
	for _, v := range []struct {
		name string
		fn   subtractCFLAverageFunc
	}{
		{"SIMD", subtractCFLAverageSIMD},
		{"PureGo", subtractCFLAveragePureGo},
	} {
		b.Run(v.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				v.fn(src, dst, 32, 32, log2)
			}
		})
	}
}

func BenchmarkCFLSubsample16Arm64(b *testing.B) {
	const width, height = 32, 32
	stride := width + 3
	rnd := newLibaomIntraEdgeRandom(libaomIntraEdgeDeterministicSeed)
	for _, depth := range []uint8{10, 12} {
		max := uint16((1 << depth) - 1)
		input := make([]uint16, stride*height)
		for i := range input {
			input[i] = uint16(rnd.pseudoUniform(int(max) + 1))
		}
		for _, mode := range []struct {
			name       string
			subX, subY bool
			outW, outH int
		}{
			{"444", false, false, width, height},
			{"422", true, false, width / 2, height},
			{"420", true, true, width / 2, height / 2},
		} {
			for _, variant := range []struct {
				name string
				fn   subsampleLuma16Func
			}{
				{"SIMD", subsampleLuma16SIMD},
				{"PureGo", subsampleLuma16PureGo},
			} {
				b.Run(fmt.Sprintf("%dbit/%s/%s", depth, mode.name, variant.name), func(b *testing.B) {
					output := make([]uint16, CFLBufSquare)
					b.ReportAllocs()
					for b.Loop() {
						if err := variant.fn(output, input, stride, width, height, mode.outW, mode.outH, mode.subX, mode.subY, max); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
