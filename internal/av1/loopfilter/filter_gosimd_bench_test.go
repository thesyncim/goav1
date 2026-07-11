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

func BenchmarkFilter14EdgePureGo_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter14EdgePureGo(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter14EdgeNEON_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter14EdgeNEON(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter14EdgeSIMD_H(b *testing.B) {
	buf, q0, step, n, p := benchWideEdge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter14EdgeSIMD(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter14EdgePureGo_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	b.ReportAllocs()
	for b.Loop() {
		filter14EdgePureGo(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter14EdgeNEON_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	b.ReportAllocs()
	for b.Loop() {
		filter14EdgeNEON(buf, q0, step, 1, n, 1, p)
	}
}

func BenchmarkFilter14EdgeSIMD_HFlat(b *testing.B) {
	buf, q0, step, n, p := benchWideEdgeFlat(64)
	b.ReportAllocs()
	for b.Loop() {
		filter14EdgeSIMD(buf, q0, step, 1, n, 1, p)
	}
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
	b.ReportAllocs()
	for b.Loop() {
		filter14VertNEON(buf, 16, 1, 96, 64, 1, p)
	}
}

func BenchmarkFilter14EdgeSIMD_VFlat(b *testing.B) {
	buf, p := benchWideEdgeVertFlat()
	b.ReportAllocs()
	for b.Loop() {
		filter14VertSIMD(buf, 16, 1, 96, 64, 1, p)
	}
}
