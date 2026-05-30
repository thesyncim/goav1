// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

import "testing"

// Per-kernel width-4 benchmarks comparing the new NEON 4-lane kernels against
// the pure-Go reference they replace. Decode wall-clock is too noisy under
// contention to attribute to a single kernel, so these isolate the width-4
// convolve cost directly.

func benchW4(b *testing.B, h int, fn func()) {
	b.ReportAllocs()
	for b.Loop() {
		fn()
	}
	_ = h
}

func BenchmarkConvolve2D8W4_4x4_NEON(b *testing.B) {
	dst, ref := benchPlanes(4, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	benchW4(b, 4, func() { convolve2D8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 4, 4, xk, yk) })
}

func BenchmarkConvolve2D8W4_4x4_PureGo(b *testing.B) {
	dst, ref := benchPlanes(4, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	benchW4(b, 4, func() { convolve2D8PureGo(dst, ref, 0, 0, filterTaps, filterTaps, 4, 4, xk, yk) })
}

func BenchmarkConvolve2D8W4_4x16_NEON(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	benchW4(b, 16, func() { convolve2D8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xk, yk) })
}

func BenchmarkConvolve2D8W4_4x16_PureGo(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters8[3]
	yk := subpelFilters8[5]
	benchW4(b, 16, func() { convolve2D8PureGo(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xk, yk) })
}

func BenchmarkConvolveX8W4_4x16_NEON(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters8[3]
	benchW4(b, 16, func() { convolveX8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xk) })
}

func BenchmarkConvolveX8W4_4x16_PureGo(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	xk := subpelFilters8[3]
	benchW4(b, 16, func() { convolveX8PureGo(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, xk) })
}

func BenchmarkConvolveY8W4_4x16_NEON(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	yk := subpelFilters8[5]
	benchW4(b, 16, func() { convolveY8NEON(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, yk) })
}

func BenchmarkConvolveY8W4_4x16_PureGo(b *testing.B) {
	dst, ref := benchPlanes(16, 8)
	yk := subpelFilters8[5]
	benchW4(b, 16, func() { convolveY8PureGo(dst, ref, 0, 0, filterTaps, filterTaps, 4, 16, yk) })
}
