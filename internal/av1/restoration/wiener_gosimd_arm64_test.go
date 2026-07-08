// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"math/rand"
	"reflect"
	"runtime"
	"testing"
)

// TestWienerVerticalSIMDIsBound guards the wiring: under the goexperiment.simd
// build the vertical dispatch slot must resolve to the Go-native SIMD kernel,
// while the horizontal slots keep the NEON asm (which is still compiled here).
func TestWienerVerticalSIMDIsBound(t *testing.T) {
	nameOf := func(v any) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got := nameOf(wienerVerticalImpl); got != nameOf(wienerVerticalSIMD) {
		t.Fatalf("wienerVerticalImpl = %s, want wienerVerticalSIMD", got)
	}
	if got := nameOf(wienerHorizontalImpl); got != nameOf(wienerHorizontalNEON) {
		t.Fatalf("wienerHorizontalImpl = %s, want wienerHorizontalNEON", got)
	}
	if got := nameOf(wienerHorizontalTrustedImpl); got != nameOf(wienerHorizontalNEONTrusted) {
		t.Fatalf("wienerHorizontalTrustedImpl = %s, want wienerHorizontalNEONTrusted", got)
	}
}

// TestWienerVerticalSIMDMatchesPureGo is the byte-exactness differential test:
// the Go-native SIMD vertical kernel must equal the scalar wienerVertical
// sample-for-sample over random temp buffers, filters, bit depths, widths and
// heights. Widths cover 8-aligned (SIMD) and non-8-aligned (scalar fallback)
// shapes; temp values span the valid [0,maxClamp] range produced by the
// horizontal pass.
func TestWienerVerticalSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5115))
	sizes := []struct{ width, height int }{
		{8, 8}, {16, 12}, {64, 64}, {64, 31}, {24, 7},
		{13, 9}, {1, 1}, {40, 64}, {8, 1}, {56, 3}, {72, 5},
	}
	for iter := 0; iter < 400; iter++ {
		bitDepth := []int{8, 10, 12}[rng.Intn(3)]
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(bitDepth)
		// maxClamp is the horizontal pass's output ceiling; the vertical pass
		// only ever sees temp samples in [0,maxClamp].
		maxClamp := int(1<<(bitDepth+1+WienerFilterBits-round0)) - 1

		var filter WienerFilter
		switch rng.Intn(2) {
		case 0:
			// A random valid Wiener filter (taps sum to zero, mirrored).
			t0 := int16(WienerTap0Min + rng.Intn(WienerTap0Max-WienerTap0Min+1))
			t1 := int16(WienerTap1Min + rng.Intn(WienerTap1Max-WienerTap1Min+1))
			t2 := int16(WienerTap2Min + rng.Intn(WienerTap2Max-WienerTap2Min+1))
			filter = NewWienerFilter(t0, t1, t2)
		default:
			// A general 7-tap filter over the int16 tap window the reference
			// accepts, exercising the kernel beyond the canonical Wiener set.
			for i := 0; i < 7; i++ {
				filter[i] = int16(-64 + rng.Intn(129))
			}
			filter[7] = 0
		}

		sz := sizes[rng.Intn(len(sizes))]
		tempStride := sz.width
		temp := make([]uint16, tempStride*(sz.height+2*WienerHalfwin))
		for i := range temp {
			temp[i] = uint16(rng.Intn(maxClamp + 1))
		}
		dstStride := sz.width + rng.Intn(5)
		want := make([]uint16, dstStride*sz.height)
		got := make([]uint16, dstStride*sz.height)

		wienerVertical(temp, tempStride, want, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)
		wienerVerticalSIMD(temp, tempStride, got, dstStride, sz.width, sz.height, filter, bitDepth, round1, max)

		for r := 0; r < sz.height; r++ {
			for c := 0; c < sz.width; c++ {
				i := r*dstStride + c
				if want[i] != got[i] {
					t.Fatalf("iter=%d bd=%d sz=%dx%d filter=%v pos=(%d,%d) got=%d want=%d",
						iter, bitDepth, sz.width, sz.height, filter, r, c, got[i], want[i])
				}
			}
		}
	}
}

// TestWienerVerticalSIMDIsZeroAlloc protects the hot-path zero-allocation
// contract for the SIMD vertical kernel.
func TestWienerVerticalSIMDIsZeroAlloc(t *testing.T) {
	const width, height = 64, 64
	const bitDepth = 12
	max := uint16((1 << bitDepth) - 1)
	_, round1 := wienerRounds(bitDepth)
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	rng := rand.New(rand.NewSource(0x2727))
	for i := range temp {
		temp[i] = uint16(rng.Intn(int(max) + 1))
	}
	dst := make([]uint16, width*height)
	info := DefaultWienerInfo()
	if allocs := testing.AllocsPerRun(200, func() {
		wienerVerticalSIMD(temp, width, dst, width, width, height, info.VFilter, bitDepth, round1, max)
	}); allocs != 0 {
		t.Fatalf("wienerVerticalSIMD allocated %f times per call", allocs)
	}
}

func benchWienerVertical(b *testing.B, fn func([]uint16, int, []uint16, int, int, int, WienerFilter, int, int, uint16), width int) {
	const height = 64
	const bitDepth = 12
	max := uint16((1 << bitDepth) - 1)
	_, round1 := wienerRounds(bitDepth)
	rng := rand.New(rand.NewSource(0x6363))
	maxClamp := int(1<<(bitDepth+1+WienerFilterBits-5)) - 1
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	for i := range temp {
		temp[i] = uint16(rng.Intn(maxClamp + 1))
	}
	dst := make([]uint16, width*height)
	filter := DefaultWienerInfo().VFilter
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		fn(temp, width, dst, width, width, height, filter, bitDepth, round1, max)
	}
}

func BenchmarkWienerVertical8_SIMD(b *testing.B)   { benchWienerVertical(b, wienerVerticalSIMD, 8) }
func BenchmarkWienerVertical8_NEON(b *testing.B)   { benchWienerVertical(b, wienerVerticalNEON, 8) }
func BenchmarkWienerVertical8_PureGo(b *testing.B) { benchWienerVertical(b, wienerVertical, 8) }

func BenchmarkWienerVertical32_SIMD(b *testing.B)   { benchWienerVertical(b, wienerVerticalSIMD, 32) }
func BenchmarkWienerVertical32_NEON(b *testing.B)   { benchWienerVertical(b, wienerVerticalNEON, 32) }
func BenchmarkWienerVertical32_PureGo(b *testing.B) { benchWienerVertical(b, wienerVertical, 32) }

func BenchmarkWienerVertical64_SIMD(b *testing.B)   { benchWienerVertical(b, wienerVerticalSIMD, 64) }
func BenchmarkWienerVertical64_NEON(b *testing.B)   { benchWienerVertical(b, wienerVerticalNEON, 64) }
func BenchmarkWienerVertical64_PureGo(b *testing.B) { benchWienerVertical(b, wienerVertical, 64) }

func BenchmarkWienerVertical256_SIMD(b *testing.B)   { benchWienerVertical(b, wienerVerticalSIMD, 256) }
func BenchmarkWienerVertical256_NEON(b *testing.B)   { benchWienerVertical(b, wienerVerticalNEON, 256) }
func BenchmarkWienerVertical256_PureGo(b *testing.B) { benchWienerVertical(b, wienerVertical, 256) }
