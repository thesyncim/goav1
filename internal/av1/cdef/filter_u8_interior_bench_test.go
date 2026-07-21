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

func BenchmarkFilterBlockU8_4x4_Interior16b(b *testing.B) {
	_, _, ctx := benchU8InteriorCtx(4, 4)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4InteriorU8NEON(&ctx)
	}
}

func BenchmarkFilterBlockU8_4x4_InteriorByte(b *testing.B) {
	dst, input16, wide := benchU8InteriorCtx(4, 4)
	input8 := make([]byte, len(input16))
	for i := range input16 {
		input8[i] = byte(input16[i])
	}
	ctx := filterBlockU8ByteNEONCtx{
		dst:         &dst[0],
		input:       &input8[cdefBlockOrigin()],
		dstStr:      wide.dstStr,
		height:      4,
		pri0:        wide.pri0,
		pri1:        wide.pri1,
		sec0:        wide.sec0,
		sec1:        wide.sec1,
		sec2:        wide.sec2,
		sec3:        wide.sec3,
		priTap0:     wide.priTap0,
		priTap1:     wide.priTap1,
		secTap0:     wide.secTap0,
		secTap1:     wide.secTap1,
		priStrength: wide.priStrength,
		secStrength: wide.secStrength,
		priShift:    wide.priShift,
		secShift:    wide.secShift,
	}
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4FusedByteU8NEON(&ctx)
	}
}
