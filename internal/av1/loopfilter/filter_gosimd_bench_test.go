// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// benchWideEdge sets up a horizontal-edge 8-bit buffer of edgeLen contiguous
// positions with thresholds chosen so most positions pass needsFilter and split
// across the hev/flat branches.
func benchWideEdge(edgeLen int) ([]byte, int, int, int, filter4Params) {
	const stride = 256
	buf := make([]byte, stride*16)
	rng := rand.New(rand.NewSource(99))
	for i := range buf {
		buf[i] = byte(100 + rng.Intn(40))
	}
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	q0Base := 8 * stride
	return buf, q0Base, stride, edgeLen, params
}

func BenchmarkFilter4EdgePureGo_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter4EdgePureGo(buf, q0, step, 1, n, p)
	}
}

func BenchmarkFilter4EdgeNEON_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter4EdgeNEON(buf, q0, step, 1, n, p)
	}
}

func BenchmarkFilter4EdgeSIMD_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter4EdgeSIMD(buf, q0, step, 1, n, p)
	}
}

func BenchmarkFilter8EdgePureGo_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter8EdgePureGo(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter8EdgeNEON_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter8EdgeNEON(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter8EdgeSIMD_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter8EdgeSIMD(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter6EdgePureGo_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter6EdgePureGo(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter6EdgeNEON_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter6EdgeNEON(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter6EdgeSIMD_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter6EdgeSIMD(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter6EdgeNEON_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	b.ReportAllocs()
	for b.Loop() {
		filter6EdgeNEON(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter6EdgeSIMD_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	b.ReportAllocs()
	for b.Loop() {
		filter6EdgeSIMD(buf, q0, step, 1, n, 1, p)
	}
}

// benchWideEdgeFlat sets up a horizontal-edge buffer of near-constant samples
// so every lane takes the fourteen-tap wide path (flat8in && flat8out), the
// heaviest branch and the common case on smooth content.
func benchWideEdgeFlat(edgeLen int) ([]byte, int, int, int, filter4Params) {
	const stride = 256
	buf := make([]byte, stride*16)
	rng := rand.New(rand.NewSource(42))
	for i := range buf {
		buf[i] = byte(127 + rng.Intn(2))
	}
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	q0Base := 8 * stride
	return buf, q0Base, stride, edgeLen, params
}

var filter14BenchmarkSink byte

type filter14BenchmarkFn func([]byte, int, int, int, int, int, filter4Params)

// benchFilter14Reset restores the same input before each iteration and feeds
// both competing kernels through identical copy cost. Several filter cases
// mutate their edge block, so repeatedly timing one buffer would benchmark a
// different branch pattern after the first iteration.
func benchFilter14Reset(b *testing.B, initial []byte, q0Base, step, outer, length, scale int, params filter4Params, fn filter14BenchmarkFn) {
	work := make([]byte, len(initial))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		copy(work, initial)
		fn(work, q0Base, step, outer, length, scale, params)
	}
	b.StopTimer()
	filter14BenchmarkSink = work[q0Base]
}

func BenchmarkFilter14EdgePureGo_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	benchFilter14Reset(b, buf, q0, step, 1, n, 1, p, filter14EdgePureGo)
}

func BenchmarkFilter14EdgeNEON_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	benchFilter14Reset(b, buf, q0, step, 1, n, 1, p, filter14EdgeNEON)
}

func BenchmarkFilter14EdgeSIMD_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	benchFilter14Reset(b, buf, q0, step, 1, n, 1, p, filter14EdgeSIMD)
}

func BenchmarkFilter14EdgePureGo_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	benchFilter14Reset(b, buf, q0, step, 1, n, 1, p, filter14EdgePureGo)
}

func BenchmarkFilter14EdgeNEON_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	benchFilter14Reset(b, buf, q0, step, 1, n, 1, p, filter14EdgeNEON)
}

func BenchmarkFilter14EdgeSIMD_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	benchFilter14Reset(b, buf, q0, step, 1, n, 1, p, filter14EdgeSIMD)
}

// benchWideEdgeVertFlat sets up a vertical-edge buffer (taps contiguous within
// a row, positions advancing by the row stride) of near-constant samples so
// the batched transpose + wide path runs end to end.
func benchWideEdgeVertFlat() ([]byte, filter4Params) {
	const stride = 96
	buf := make([]byte, stride*80)
	rng := rand.New(rand.NewSource(43))
	for i := range buf {
		buf[i] = byte(127 + rng.Intn(2))
	}
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	return buf, params
}

func BenchmarkFilter14EdgeNEON_VFlat(b *testing.B) {
	buf, p := benchWideEdgeVertFlat()
	benchFilter14Reset(b, buf, 16, 1, 96, 64, 1, p, filter14VertNEON)
}

func BenchmarkFilter14EdgeSIMD_VFlat(b *testing.B) {
	buf, p := benchWideEdgeVertFlat()
	benchFilter14Reset(b, buf, 16, 1, 96, 64, 1, p, filter14VertSIMD)
}

// benchWide16Edge sets up a 10/12-bit horizontal-edge buffer. Flat content
// takes the fourteen-tap wide path in every lane; mixed content exercises the
// branch ladder the way real frames do.
func benchWide16Edge(bitDepth uint8, flat bool) ([]byte, int, int, int, filter4Params) {
	const strideBytes = 512
	buf := make([]byte, strideBytes*16)
	rng := rand.New(rand.NewSource(77))
	maxVal := (1 << bitDepth) - 1
	mid := maxVal / 2
	for i := 0; i+1 < len(buf); i += 2 {
		var v int
		if flat {
			v = mid + rng.Intn(2)
		} else {
			v = mid - 80 + rng.Intn(160)
		}
		buf[i] = byte(v)
		buf[i+1] = byte(v >> 8)
	}
	scale := 1 << int(bitDepth-8)
	params := filter4Params{
		limit: int16(16 * scale), blimit: int16(40 * scale), hev: int16(8 * scale),
		min: int16(-128 * scale), max: int16(128*scale - 1), center: int16(128 * scale),
	}
	return buf, 8 * strideBytes, strideBytes, 64, params
}

func BenchmarkFilter14Edge16PureGo_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16PureGo)
}

func BenchmarkFilter14Edge16NEON_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16NEON)
}

func BenchmarkFilter14Edge16SIMD_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16SIMD)
}

func BenchmarkFilter14Edge16NEON_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16NEON)
}

func BenchmarkFilter14Edge16SIMD_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16SIMD)
}

// 12-bit: the NEON wrapper refuses (center != 512) and runs pure-Go, so this
// pair measures the SIMD kernel against today's actual 12-bit dispatch.
func BenchmarkFilter14Edge16NEON_H12Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(12, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 16, p, filter14Edge16NEON)
}

func BenchmarkFilter14Edge16SIMD_H12Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(12, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 16, p, filter14Edge16SIMD)
}

func BenchmarkFilter6Edge16NEON_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	b.ReportAllocs()
	for b.Loop() {
		filter6Edge16NEON(buf, q0, step, 2, n, 4, p)
	}
}

func BenchmarkFilter6Edge16SIMD_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	b.ReportAllocs()
	for b.Loop() {
		filter6Edge16SIMD(buf, q0, step, 2, n, 4, p)
	}
}

func BenchmarkFilter6Edge16NEON_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	b.ReportAllocs()
	for b.Loop() {
		filter6Edge16NEON(buf, q0, step, 2, n, 4, p)
	}
}

func BenchmarkFilter6Edge16SIMD_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	b.ReportAllocs()
	for b.Loop() {
		filter6Edge16SIMD(buf, q0, step, 2, n, 4, p)
	}
}

// benchWide16Vert sets up a 10-bit vertical-edge buffer of near-flat samples.
func benchWide16Vert() ([]byte, filter4Params) {
	const strideBytes = 192
	buf := make([]byte, strideBytes*80)
	rng := rand.New(rand.NewSource(78))
	for i := 0; i+1 < len(buf); i += 2 {
		v := 511 + rng.Intn(2)
		buf[i] = byte(v)
		buf[i+1] = byte(v >> 8)
	}
	params := filter4Params{limit: 64, blimit: 160, hev: 32, min: -512, max: 511, center: 512}
	return buf, params
}

func BenchmarkFilter14Edge16NEON_V10Flat(b *testing.B) {
	buf, p := benchWide16Vert()
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter14Vert16NEON)
}

func BenchmarkFilter14Edge16SIMD_V10Flat(b *testing.B) {
	buf, p := benchWide16Vert()
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter14Vert16SIMD)
}
