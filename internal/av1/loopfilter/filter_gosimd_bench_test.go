// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

var filter14BenchmarkSink byte

type filter14BenchmarkFn func([]byte, int, int, int, int, int, filter4Params)

// benchFilter14Reset restores the same input before each iteration and feeds
// both competing kernels through identical copy cost.
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

func BenchmarkFilter14Edge16SIMD_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16SIMD)
}

func BenchmarkFilter14Edge16PureGo_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16PureGo)
}

func BenchmarkFilter14Edge16SIMD_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter14Edge16SIMD)
}

// 12-bit: the 10/12-bit SIMD kernel is measured against the pure-Go reference, so this
// pair measures the SIMD kernel against today's actual 12-bit dispatch.
func BenchmarkFilter14Edge16PureGo_H12Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(12, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 16, p, filter14Edge16PureGo)
}

func BenchmarkFilter14Edge16SIMD_H12Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(12, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 16, p, filter14Edge16SIMD)
}

func BenchmarkFilter6Edge16PureGo_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter6Edge16PureGo)
}

func BenchmarkFilter6Edge16SIMD_H10Flat(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, true)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter6Edge16SIMD)
}

func BenchmarkFilter6Edge16PureGo_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter6Edge16PureGo)
}

func BenchmarkFilter6Edge16SIMD_H10Mixed(b *testing.B) {
	buf, q0, step, n, p := benchWide16Edge(10, false)
	benchFilter14Reset(b, buf, q0, step, 2, n, 4, p, filter6Edge16SIMD)
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

func BenchmarkFilter14Edge16PureGo_V10Flat(b *testing.B) {
	buf, p := benchWide16Vert()
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter14Edge16PureGo)
}

func BenchmarkFilter14Edge16SIMD_V10Flat(b *testing.B) {
	buf, p := benchWide16Vert()
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter14Edge16SIMD)
}

func benchWide16VertDepth(bitDepth uint8, flat bool) ([]byte, filter4Params) {
	const strideBytes = 192
	buf := make([]byte, strideBytes*80)
	rng := rand.New(rand.NewSource(int64(80 + bitDepth)))
	maxVal := (1 << bitDepth) - 1
	mid := maxVal / 2
	for i := 0; i+1 < len(buf); i += 2 {
		v := mid - 80 + rng.Intn(160)
		if flat {
			v = mid + rng.Intn(2)
		}
		buf[i], buf[i+1] = byte(v), byte(v>>8)
	}
	scale := 1 << int(bitDepth-8)
	params := filter4Params{
		limit: int16(16 * scale), blimit: int16(40 * scale), hev: int16(8 * scale),
		min: int16(-128 * scale), max: int16(128*scale - 1), center: int16(128 * scale),
	}
	return buf, params
}

func BenchmarkFilter6Edge16PureGo_V10Flat(b *testing.B) {
	buf, p := benchWide16VertDepth(10, true)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter6Edge16PureGo)
}
func BenchmarkFilter6Edge16SIMD_V10Flat(b *testing.B) {
	buf, p := benchWide16VertDepth(10, true)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter6Edge16SIMD)
}
func BenchmarkFilter6Edge16PureGo_V10Mixed(b *testing.B) {
	buf, p := benchWide16VertDepth(10, false)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter6Edge16PureGo)
}
func BenchmarkFilter6Edge16SIMD_V10Mixed(b *testing.B) {
	buf, p := benchWide16VertDepth(10, false)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 4, p, filter6Edge16SIMD)
}
func BenchmarkFilter6Edge16PureGo_V12Flat(b *testing.B) {
	buf, p := benchWide16VertDepth(12, true)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 16, p, filter6Edge16PureGo)
}
func BenchmarkFilter6Edge16SIMD_V12Flat(b *testing.B) {
	buf, p := benchWide16VertDepth(12, true)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 16, p, filter6Edge16SIMD)
}
func BenchmarkFilter6Edge16PureGo_V12Mixed(b *testing.B) {
	buf, p := benchWide16VertDepth(12, false)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 16, p, filter6Edge16PureGo)
}
func BenchmarkFilter6Edge16SIMD_V12Mixed(b *testing.B) {
	buf, p := benchWide16VertDepth(12, false)
	benchFilter14Reset(b, buf, 16, 2, 192, 64, 16, p, filter6Edge16SIMD)
}
