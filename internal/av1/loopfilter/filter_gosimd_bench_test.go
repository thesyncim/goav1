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
