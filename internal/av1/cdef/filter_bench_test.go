// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

import "testing"

// benchFilterBlockParams is the worst-case 8x8 shape: both primary and
// secondary taps active (so the clip path runs) at 12-bit precision.
func benchFilterBlockParams() ([]uint16, []uint16, int, BlockFilterParams) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 12, 0, 0)
	dst := make([]uint16, 64)
	params := BlockFilterParams{
		PrimaryStrength:   15 << 4,
		SecondaryStrength: 4 << 4,
		Direction:         4,
		PrimaryDamping:    7,
		SecondaryDamping:  7,
		CoeffShift:        4,
		Width:             8,
		Height:            8,
	}
	return dst, input, cdefBlockOrigin(), params
}

// BenchmarkFilterBlockPureGo and BenchmarkFilterBlockDispatch bracket the
// per-kernel speedup: the first always runs the pure-Go reference; the second
// runs whatever the dispatcher resolved (NEON on arm64). Both skip the
// FilterBlock validation wrapper so they time only the inner kernel.
func BenchmarkFilterBlockPureGo(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockPureGo(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockDispatch(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	b.ReportAllocs()
	for b.Loop() {
		filterBlockImpl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockDispatch4x8(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 8
	b.ReportAllocs()
	for b.Loop() {
		filterBlockImpl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockPureGo4x8(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 8
	b.ReportAllocs()
	for b.Loop() {
		filterBlockPureGo(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockDispatch4x4(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 4
	b.ReportAllocs()
	for b.Loop() {
		filterBlockImpl(dst, 8, 0, input, origin, params)
	}
}

func BenchmarkFilterBlockPureGo4x4(b *testing.B) {
	dst, input, origin, params := benchFilterBlockParams()
	params.Width = 4
	params.Height = 4
	b.ReportAllocs()
	for b.Loop() {
		filterBlockPureGo(dst, 8, 0, input, origin, params)
	}
}
