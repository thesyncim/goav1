// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

import "testing"

// buildU8NEONCtx prepares a filterBlockU8NEONCtx for a single block exactly as
// filterBlockU8NEON does, so the interior kernels can be driven directly from a
// BlockFilterParams in tests.
func buildU8NEONCtx(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) filterBlockU8NEONCtx {
	primaryStrength := int(params.PrimaryStrength)
	secondaryStrength := int(params.SecondaryStrength)
	direction := int(params.Direction)
	coeffShift := int(params.CoeffShift)
	priTaps := cdefPrimaryTaps[(primaryStrength>>coeffShift)&1]
	ctx := filterBlockU8NEONCtx{
		dst:    &dst[dstOrigin],
		input:  &input[inputOrigin],
		dstStr: int64(dstStride),
		height: int64(params.Height),

		pri0: int64(cdefDirections[direction+2][0]),
		pri1: int64(cdefDirections[direction+2][1]),
		sec0: int64(cdefDirections[direction+4][0]),
		sec1: int64(cdefDirections[direction][0]),
		sec2: int64(cdefDirections[direction+4][1]),
		sec3: int64(cdefDirections[direction][1]),

		priTap0: int64(priTaps[0]),
		priTap1: int64(priTaps[1]),
		secTap0: int64(cdefSecondaryTaps[0]),
		secTap1: int64(cdefSecondaryTaps[1]),

		priStrength: int64(primaryStrength),
		secStrength: int64(secondaryStrength),
		priShift:    int64(constrainShift(primaryStrength, int(params.PrimaryDamping))),
		secShift:    int64(constrainShift(secondaryStrength, int(params.SecondaryDamping))),
	}
	if primaryStrength != 0 {
		ctx.enablePrimary = 1
	}
	if secondaryStrength != 0 {
		ctx.enableSecondary = 1
	}
	if primaryStrength != 0 && secondaryStrength != 0 {
		ctx.clipping = 1
	}
	return ctx
}

// TestFilterBlockU8InteriorNEONMatchesPureGo is the safety net for the CDEF
// interior .16b kernels: over sentinel-free (fully interior) tap buffers it
// pins each of the six kernels — fused / primary-only / secondary-only at
// widths 8 and 4 — against filterBlockU8PureGo, the goav1 scalar reference,
// across every direction, the strength/damping corpus, and many random inputs.
// A byte mismatch fails, so any divergence in the ported dav1d arithmetic or in
// a hand-written encoding is caught here before it can reach a frame.
func TestFilterBlockU8InteriorNEONMatchesPureGo(t *testing.T) {
	const dstStride = 24
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {8, 2}, {4, 8}, {4, 4},
	}
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x1c7e2d05)
	for iter := 0; iter < 48; iter++ {
		// boundary == 0: no VeryLarge sentinel anywhere, i.e. a fully interior
		// block, which is the sole precondition of the interior kernels.
		input := makeCDEFBlockInput(rnd, 8, 0, iter+1)
		if !cdefUnitInteriorU8(input, origin, []BlockPosition{{BY: 0, BX: 0}}, 3, 3) {
			t.Fatalf("iter=%d: interior tap buffer flagged as boundary", iter)
		}
		for _, shape := range shapes {
			width := shape.width
			height := shape.height
			if width == 8 && height%2 != 0 {
				continue
			}
			if width == 4 && height%4 != 0 {
				continue
			}
			for dir := 0; dir <= 7; dir++ {
				for _, pri := range cdefPrimaryStrengthCorpus(0) {
					for _, sec := range cdefSecondaryStrengthCorpus(0) {
						if pri == 0 && sec == 0 {
							continue
						}
						for _, damping := range []int{3, 4, 5, 6} {
							params := BlockFilterParams{
								PrimaryStrength:   uint8(pri),
								SecondaryStrength: uint8(sec),
								Direction:         uint8(dir),
								PrimaryDamping:    uint8(damping),
								SecondaryDamping:  uint8(damping),
								CoeffShift:        0,
								Width:             uint8(width),
								Height:            uint8(height),
							}
							want := make([]byte, dstStride*height)
							got := make([]byte, dstStride*height)
							filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
							ctx := buildU8NEONCtx(got, dstStride, 0, input, origin, params)
							dispatchFilterBlockU8InteriorNEON(&ctx, width, pri, sec)
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("iter=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
										iter, width, height, dir, pri, sec, damping, i, got[i], want[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestCDEFUnitInteriorU8Predicate checks the interior predicate: a sentinel-free
// footprint is interior, and a VeryLarge sentinel anywhere the taps reach flags
// the unit as a boundary (forcing the .8h path).
func TestCDEFUnitInteriorU8Predicate(t *testing.T) {
	origin := cdefBlockOrigin()
	blocks := []BlockPosition{{BY: 0, BX: 0}}
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x51de77a1)
	clean := makeCDEFBlockInput(rnd, 8, 0, 1)
	if !cdefUnitInteriorU8(clean, origin, blocks, 3, 3) {
		t.Fatalf("clean buffer must be interior")
	}
	// A sentinel two columns to the left of the block (within tap reach) must
	// flag boundary; the same sentinel far in the outer halo must not.
	for _, off := range []int{-cdefTapReach, -HorizontalBorder} {
		buf := make([]uint16, len(clean))
		copy(buf, clean)
		buf[origin+off] = VeryLarge
		interior := cdefUnitInteriorU8(buf, origin, blocks, 3, 3)
		wantInterior := off < -cdefTapReach
		if interior != wantInterior {
			t.Fatalf("off=%d interior=%v want=%v", off, interior, wantInterior)
		}
	}
}
