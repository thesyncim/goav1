// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "testing"

// secondaryU8Ctx builds the NEON/SIMD calling ctx for a secondary-only block,
// mirroring filterBlockU8NEON's ctx construction.
func secondaryU8Ctx(dst []byte, dstStride int, input []uint16, inputOrigin int, params BlockFilterParams) filterBlockU8NEONCtx {
	direction := int(params.Direction)
	secondaryStrength := int(params.SecondaryStrength)
	return filterBlockU8NEONCtx{
		dst:    &dst[0],
		input:  &input[inputOrigin],
		dstStr: int64(dstStride),
		height: int64(params.Height),

		sec0: int64(cdefDirections[direction+4][0]),
		sec1: int64(cdefDirections[direction][0]),
		sec2: int64(cdefDirections[direction+4][1]),
		sec3: int64(cdefDirections[direction][1]),

		secTap0: int64(cdefSecondaryTaps[0]),
		secTap1: int64(cdefSecondaryTaps[1]),

		secStrength:     int64(secondaryStrength),
		secShift:        int64(constrainShift(secondaryStrength, int(params.SecondaryDamping))),
		enableSecondary: 1,
	}
}

// TestCDEFSecondaryU8SIMDMatches is the 3-way differential for the Go-native
// SIMD secondary-only kernels: SIMD and the NEON asm must both match the
// pure-Go reference byte-for-byte across directions, strengths, dampings,
// widths and halo-sentinel boundary patterns.
func TestCDEFSecondaryU8SIMDMatches(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {4, 8}, {4, 4},
	}
	for boundary := 0; boundary < 16; boundary++ {
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x51D51D), 8, boundary, boundary+1)
		for _, shape := range shapes {
			for dir := 0; dir <= 7; dir++ {
				for _, sec := range cdefSecondaryStrengthCorpus(0) {
					if sec == 0 {
						continue
					}
					for _, damping := range []int{3, 4, 6} {
						params := BlockFilterParams{
							SecondaryStrength: uint8(sec),
							Direction:         uint8(dir),
							PrimaryDamping:    uint8(damping),
							SecondaryDamping:  uint8(damping),
							Width:             uint8(shape.width),
							Height:            uint8(shape.height),
						}
						want := make([]byte, dstStride*8)
						got := make([]byte, dstStride*8)
						asm := make([]byte, dstStride*8)
						filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
						gctx := secondaryU8Ctx(got, dstStride, input, origin, params)
						actx := secondaryU8Ctx(asm, dstStride, input, origin, params)
						if shape.width == 8 {
							cdefFilterBlock8SecondaryU8SIMD(&gctx)
							cdefFilterBlock8SecondaryU8NEON(&actx)
						} else {
							cdefFilterBlock4SecondaryU8SIMD(&gctx)
							cdefFilterBlock4SecondaryU8NEON(&actx)
						}
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("SIMD boundary=%d shape=%dx%d dir=%d sec=%d damp=%d idx=%d got=%d want=%d",
									boundary, shape.width, shape.height, dir, sec, damping, i, got[i], want[i])
							}
							if asm[i] != want[i] {
								t.Fatalf("NEON boundary=%d shape=%dx%d dir=%d sec=%d damp=%d idx=%d asm=%d want=%d",
									boundary, shape.width, shape.height, dir, sec, damping, i, asm[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestCDEFSecondaryU8SIMDExtremes drives the kernels over extreme content:
// all-zero, all-255, alternating 0/255 (maximal diffs against every tap), and
// interior pixels adjacent to VeryLarge halo sentinels on every side, at the
// largest secondary strength and both damping extremes.
func TestCDEFSecondaryU8SIMDExtremes(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	fills := []func(i int) uint16{
		func(i int) uint16 { return 0 },
		func(i int) uint16 { return 255 },
		func(i int) uint16 {
			if i%2 == 0 {
				return 0
			}
			return 255
		},
		func(i int) uint16 {
			if (i/BStride)%2 == 0 {
				return 255
			}
			return 0
		},
	}
	for fi, fill := range fills {
		input := make([]uint16, InputBufferSize)
		for i := range input {
			input[i] = fill(i)
		}
		// Sentinel frame: mark everything outside the 8x8 interior VeryLarge.
		sentinel := make([]uint16, InputBufferSize)
		copy(sentinel, input)
		for i := range sentinel {
			row := i / BStride
			col := i % BStride
			if row < VerticalBorder || row >= VerticalBorder+8 || col < HorizontalBorder || col >= HorizontalBorder+8 {
				sentinel[i] = VeryLarge
			}
		}
		for _, in := range [][]uint16{input, sentinel} {
			for dir := 0; dir <= 7; dir++ {
				for _, damping := range []int{3, 6} {
					for _, shape := range [...]struct{ width, height int }{{8, 8}, {4, 4}} {
						params := BlockFilterParams{
							SecondaryStrength: 4,
							Direction:         uint8(dir),
							PrimaryDamping:    uint8(damping),
							SecondaryDamping:  uint8(damping),
							Width:             uint8(shape.width),
							Height:            uint8(shape.height),
						}
						want := make([]byte, dstStride*8)
						got := make([]byte, dstStride*8)
						filterBlockU8PureGo(want, dstStride, 0, in, origin, params)
						ctx := secondaryU8Ctx(got, dstStride, in, origin, params)
						if shape.width == 8 {
							cdefFilterBlock8SecondaryU8SIMD(&ctx)
						} else {
							cdefFilterBlock4SecondaryU8SIMD(&ctx)
						}
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("fill=%d dir=%d damp=%d w=%d idx=%d got=%d want=%d",
									fi, dir, damping, shape.width, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

// TestCDEFSecondaryU8SIMDZeroAlloc pins the kernels and the dispatch route as
// allocation-free.
func TestCDEFSecondaryU8SIMDZeroAlloc(t *testing.T) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	params := BlockFilterParams{
		SecondaryStrength: 4,
		Direction:         3,
		PrimaryDamping:    5,
		SecondaryDamping:  5,
		Width:             8,
		Height:            8,
	}
	origin := cdefBlockOrigin()
	ctx := secondaryU8Ctx(dst, 8, input, origin, params)
	cases := []struct {
		name string
		fn   func()
	}{
		{"cdefFilterBlock8SecondaryU8SIMD", func() { cdefFilterBlock8SecondaryU8SIMD(&ctx) }},
		{"cdefFilterBlock4SecondaryU8SIMD", func() {
			c := ctx
			c.height = 8
			cdefFilterBlock4SecondaryU8SIMD(&c)
		}},
		{"filterBlockU8Impl/secondary", func() { filterBlockU8Impl(dst, 8, 0, input, origin, params) }},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(50, c.fn); a != 0 {
			t.Errorf("%s allocated %.1f objects/run, want 0", c.name, a)
		}
	}
}

// --- benches: head-to-head vs the NEON asm ------------------------------------

func benchSecondaryU8(width, height int) ([]byte, filterBlockU8NEONCtx) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	params := BlockFilterParams{
		SecondaryStrength: 4,
		Direction:         4,
		PrimaryDamping:    5,
		SecondaryDamping:  5,
		Width:             uint8(width),
		Height:            uint8(height),
	}
	return dst, secondaryU8Ctx(dst, 8, input, cdefBlockOrigin(), params)
}

func BenchmarkCDEFSecondaryU8_8x8_NEON(b *testing.B) {
	_, ctx := benchSecondaryU8(8, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock8SecondaryU8NEON(&ctx)
	}
}

func BenchmarkCDEFSecondaryU8_8x8_SIMD(b *testing.B) {
	_, ctx := benchSecondaryU8(8, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock8SecondaryU8SIMD(&ctx)
	}
}

func BenchmarkCDEFSecondaryU8_4x8_NEON(b *testing.B) {
	_, ctx := benchSecondaryU8(4, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4SecondaryU8NEON(&ctx)
	}
}

func BenchmarkCDEFSecondaryU8_4x8_SIMD(b *testing.B) {
	_, ctx := benchSecondaryU8(4, 8)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4SecondaryU8SIMD(&ctx)
	}
}

func BenchmarkCDEFSecondaryU8_4x4_NEON(b *testing.B) {
	_, ctx := benchSecondaryU8(4, 4)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4SecondaryU8NEON(&ctx)
	}
}

func BenchmarkCDEFSecondaryU8_4x4_SIMD(b *testing.B) {
	_, ctx := benchSecondaryU8(4, 4)
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock4SecondaryU8SIMD(&ctx)
	}
}
