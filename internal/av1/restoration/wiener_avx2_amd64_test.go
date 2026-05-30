// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// TestWienerAVX2ReportsDetection prints whether the running host advertises
// AVX2 (and so whether the dispatcher binds the AVX2 kernels or falls back to
// pure-Go). Under Rosetta 2 on Apple silicon CPUID does not advertise AVX2, so
// auto-dispatch stays pure-Go even though the instructions themselves execute.
func TestWienerAVX2ReportsDetection(t *testing.T) {
	t.Logf("cpu.Detected.AVX2 = %v (SSE41=%v SSE42=%v AVX512=%v)",
		cpu.Detected.AVX2, cpu.Detected.SSE41, cpu.Detected.SSE42, cpu.Detected.AVX512)
	if cpu.Detected.AVX2 {
		t.Logf("AVX2 detected: Wiener dispatch is bound to the AVX2 kernels")
	} else {
		t.Logf("AVX2 NOT detected: Wiener dispatch falls back to pure-Go")
	}
}

// TestWienerAVX2MatchesPureGo is the bit-exactness guard for the AVX2 Wiener
// passes. It calls the AVX2 kernels directly (independent of cpu.Detected, so
// it exercises the SIMD even when auto-dispatch falls back) and compares every
// temp and destination sample against the canonical pure-Go reference across a
// matrix of bit depths, filters, and 8-aligned / non-8-aligned widths.
func TestWienerAVX2MatchesPureGo(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x7A2)
	sizes := []struct{ width, height int }{
		{8, 8}, {16, 12}, {64, 64}, {64, 31}, {24, 7}, {13, 9}, {1, 1}, {40, 64}, {48, 1},
	}
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		round0, round1 := wienerRounds(int(bitDepth))
		for _, sz := range sizes {
			for fi, info := range wienerDispatchFilters() {
				stride := sz.width + 2*WienerHalfwin + 4
				origin := WienerHalfwin*stride + WienerHalfwin + 2
				src := make([]uint16, stride*(sz.height+2*WienerHalfwin))
				for i := range src {
					src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				tempLen := sz.width * (sz.height + 2*WienerHalfwin)

				wantTemp := make([]uint16, tempLen)
				gotTemp := make([]uint16, tempLen)
				okWant := wienerHorizontal(src, stride, origin, sz.width, sz.height, info.HFilter, int(bitDepth), round0, max, wantTemp)
				okGot := wienerHorizontalAVX2(src, stride, origin, sz.width, sz.height, info.HFilter, int(bitDepth), round0, max, gotTemp)
				if okWant != okGot {
					t.Fatalf("bd=%d sz=%dx%d f=%d horizontal ok mismatch: want=%v got=%v", bitDepth, sz.width, sz.height, fi, okWant, okGot)
				}
				for i := range wantTemp {
					if wantTemp[i] != gotTemp[i] {
						t.Fatalf("bd=%d sz=%dx%d f=%d horizontal temp[%d]=%d want %d", bitDepth, sz.width, sz.height, fi, i, gotTemp[i], wantTemp[i])
					}
				}

				dstStride := sz.width + 3
				wantDst := make([]uint16, dstStride*sz.height)
				gotDst := make([]uint16, dstStride*sz.height)
				wienerVertical(wantTemp, sz.width, wantDst, dstStride, sz.width, sz.height, info.VFilter, int(bitDepth), round1, max)
				wienerVerticalAVX2(wantTemp, sz.width, gotDst, dstStride, sz.width, sz.height, info.VFilter, int(bitDepth), round1, max)
				for i := range wantDst {
					if wantDst[i] != gotDst[i] {
						t.Fatalf("bd=%d sz=%dx%d f=%d vertical dst[%d]=%d want %d", bitDepth, sz.width, sz.height, fi, i, gotDst[i], wantDst[i])
					}
				}
			}
		}
	}
}

// TestWienerAVX2RejectsOutOfRange verifies the AVX2 horizontal pass reproduces
// the reference validity bail-out (sample > max => false).
func TestWienerAVX2RejectsOutOfRange(t *testing.T) {
	const width, height = 64, 8
	const bitDepth uint8 = 8
	max := uint16((1 << bitDepth) - 1)
	round0, _ := wienerRounds(int(bitDepth))
	stride := width + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(height+2*WienerHalfwin))
	for i := range src {
		src[i] = uint16(i % int(max+1))
	}
	src[origin+2*stride+5] = max + 1
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	if wienerHorizontalAVX2(src, stride, origin, width, height, DefaultWienerInfo().HFilter, int(bitDepth), round0, max, temp) {
		t.Fatal("AVX2 accepted out-of-range sample")
	}
}

// TestWienerAVX2IsZeroAlloc protects the hot-path contract that the AVX2 passes
// do not allocate per call.
func TestWienerAVX2IsZeroAlloc(t *testing.T) {
	const width, height = 64, 64
	const bitDepth uint8 = 12
	max := uint16((1 << bitDepth) - 1)
	round0, round1 := wienerRounds(int(bitDepth))
	stride := width + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(height+2*WienerHalfwin))
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x4242)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}
	temp := make([]uint16, width*(height+2*WienerHalfwin))
	dst := make([]uint16, width*height)
	info := DefaultWienerInfo()
	if allocs := testing.AllocsPerRun(200, func() {
		wienerHorizontalAVX2(src, stride, origin, width, height, info.HFilter, int(bitDepth), round0, max, temp)
	}); allocs != 0 {
		t.Fatalf("AVX2 horizontal allocated %f times per call", allocs)
	}
	if allocs := testing.AllocsPerRun(200, func() {
		wienerVerticalAVX2(temp, width, dst, width, width, height, info.VFilter, int(bitDepth), round1, max)
	}); allocs != 0 {
		t.Fatalf("AVX2 vertical allocated %f times per call", allocs)
	}
}

func BenchmarkWienerHorizontalAVX2(b *testing.B) {
	src, temp, _, stride, origin, info := benchWienerSetup()
	round0, _ := wienerRounds(12)
	max := uint16((1 << 12) - 1)
	b.ReportAllocs()
	for b.Loop() {
		wienerHorizontalAVX2(src, stride, origin, 64, 64, info.HFilter, 12, round0, max, temp)
	}
}

func BenchmarkWienerVerticalAVX2(b *testing.B) {
	src, temp, dst, stride, origin, info := benchWienerSetup()
	round0, round1 := wienerRounds(12)
	max := uint16((1 << 12) - 1)
	wienerHorizontal(src, stride, origin, 64, 64, info.HFilter, 12, round0, max, temp)
	b.ReportAllocs()
	for b.Loop() {
		wienerVerticalAVX2(temp, 64, dst, 64, 64, 64, info.VFilter, 12, round1, max)
	}
}
