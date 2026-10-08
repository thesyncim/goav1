// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

import "testing"

// The interior-vs-.8h benchmarks run the worst-case pri+sec block (clip path
// active) on a fully interior tap buffer, so the new .16b kernel and the
// existing .8h kernel do identical work; the delta is the 16-lane throughput.

func benchU8InteriorCtx(width, height int) ([]byte, []uint16, filterBlockU8NEONCtx) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 24*height)
	params := BlockFilterParams{
		PrimaryStrength:   15,
		SecondaryStrength: 4,
		Direction:         4,
		PrimaryDamping:    5,
		SecondaryDamping:  5,
		CoeffShift:        0,
		Width:             uint8(width),
		Height:            uint8(height),
	}
	ctx := buildU8NEONCtx(dst, 24, 0, input, cdefBlockOrigin(), params)
	return dst, input, ctx
}

func BenchmarkFilterBlockU8_8x8_Edge8h(b *testing.B) {
	_, _, ctx := benchU8InteriorCtx(8, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock8U8NEON(&ctx)
	}
}

func BenchmarkFilterBlockU8_8x8_Interior16b(b *testing.B) {
	_, _, ctx := benchU8InteriorCtx(8, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock8InteriorU8NEON(&ctx)
	}
}

func BenchmarkFilterBlockU8_4x8_Edge8h(b *testing.B) {
	_, _, ctx := benchU8InteriorCtx(4, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4U8NEON(&ctx)
	}
}

func BenchmarkFilterBlockU8_4x8_Interior16b(b *testing.B) {
	_, _, ctx := benchU8InteriorCtx(4, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4InteriorU8NEON(&ctx)
	}
}
