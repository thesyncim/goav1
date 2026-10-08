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

func TestCDEFUnitInteriorU8PredicateTightTailAndBounds(t *testing.T) {
	block := []BlockPosition{{BY: 0, BX: 0}}
	inputOrigin := 2*BStride + cdefTapReach
	start, rows, cols := 0, 8+2*cdefTapReach, 8+2*cdefTapReach
	end := start + (rows-1)*BStride + cols
	input := make([]uint16, end)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			input[start+row*BStride+col] = 0xFF
		}
	}
	if !cdefUnitInteriorU8(input, inputOrigin, block, 3, 3) {
		t.Fatal("byte-range footprint ending at the slice boundary must be interior")
	}

	// The 12-column footprint has an 8-lane vector and a four-sample tail. Pin
	// values at both vector and tail positions, including the final sample.
	for _, tc := range []struct {
		row, col int
		value    uint16
	}{
		{0, 0, 256}, {0, 7, 0xFFFF}, {0, 8, VeryLarge}, {0, 11, 256},
		{rows - 1, 0, VeryLarge}, {rows - 1, 7, 0xFFFF},
		{rows - 1, 8, 256}, {rows - 1, cols - 1, 0xFFFF},
	} {
		withHighSample := append([]uint16(nil), input...)
		withHighSample[start+tc.row*BStride+tc.col] = tc.value
		if cdefUnitInteriorU8(withHighSample, inputOrigin, block, 3, 3) {
			t.Fatalf("row=%d col=%d value=%#x inside footprint was accepted", tc.row, tc.col, tc.value)
		}
	}

	// Values immediately outside the scanned rectangle do not change the
	// predicate, while a short final tail and a negative scan start fail closed.
	outside := append([]uint16(nil), input...)
	outside[start+cols] = 0xFFFF
	if !cdefUnitInteriorU8(outside, inputOrigin, block, 3, 3) {
		t.Fatal("sample just outside the footprint must not affect the result")
	}
	if cdefUnitInteriorU8(input[:end-1], inputOrigin, block, 3, 3) {
		t.Fatal("truncated final row tail must fail closed")
	}
	if cdefUnitInteriorU8(input, inputOrigin-1, block, 3, 3) {
		t.Fatal("negative scan start must fail closed")
	}
}

func TestCDEFUnitInteriorU8PredicateFullUnitTail(t *testing.T) {
	blocks := make([]BlockPosition, 0, 64)
	for by := range 8 {
		for bx := range 8 {
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	inputOrigin := cdefBlockOrigin()
	start := inputOrigin - cdefTapReach*BStride - cdefTapReach
	rows, cols := 64+2*cdefTapReach, 64+2*cdefTapReach
	end := start + (rows-1)*BStride + cols
	input := make([]uint16, end)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			input[start+row*BStride+col] = 255
		}
	}
	if !cdefUnitInteriorU8(input, inputOrigin, blocks, 3, 3) {
		t.Fatal("full 64x64 byte-range footprint must be interior")
	}
	input[end-1] = 0xFFFF // final lane in the last row's four-sample tail
	if cdefUnitInteriorU8(input, inputOrigin, blocks, 3, 3) {
		t.Fatal("full-unit final tail sentinel was missed")
	}
	if cdefUnitInteriorU8(input[:end-1], inputOrigin, blocks, 3, 3) {
		t.Fatal("full-unit scan must reject an incomplete final row")
	}
}

func TestCDEFUnitInteriorU8PredicateDifferential(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x51de77a2)
	full := make([]BlockPosition, 0, 64)
	for by := range 8 {
		for bx := range 8 {
			full = append(full, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	cases := []struct {
		name   string
		blocks []BlockPosition
	}{
		{name: "full64x64", blocks: full},
		{name: "sparse64x64", blocks: []BlockPosition{{BY: 0, BX: 0}, {BY: 0, BX: 7}, {BY: 7, BX: 0}, {BY: 7, BX: 7}}},
		{name: "single8x8", blocks: []BlockPosition{{BY: 3, BX: 3}}},
		{name: "disjoint", blocks: []BlockPosition{{BY: 1, BX: 6}, {BY: 6, BX: 1}}},
	}
	for _, tc := range cases {
		for iter := 0; iter < 64; iter++ {
			input := make([]uint16, InputBufferSize)
			for i := range input {
				input[i] = uint16(rnd.generate(256))
			}
			for i := 0; i < 32; i++ {
				pos := int(rnd.generate(uint32(len(input))))
				switch i % 4 {
				case 0:
					input[pos] = 255
				case 1:
					input[pos] = 256
				case 2:
					input[pos] = VeryLarge
				default:
					input[pos] = 0xFFFF
				}
			}
			want := cdefUnitInteriorU8ScalarOracle(input, cdefBlockOrigin(), tc.blocks, 3, 3)
			got := cdefUnitInteriorU8(input, cdefBlockOrigin(), tc.blocks, 3, 3)
			if got != want {
				t.Fatalf("case=%s iter=%d got=%v want=%v", tc.name, iter, got, want)
			}
		}
	}
}

// cdefUnitInteriorU8ScalarOracle independently walks the exact bounding
// footprint and is intentionally scalar so SIMD reductions are checked against
// the same byte-range contract without sharing the production scan helper.
func cdefUnitInteriorU8ScalarOracle(input []uint16, inputOrigin int, blocks []BlockPosition, bwLog2, bhLog2 int) bool {
	if len(input) == 0 || len(blocks) == 0 {
		return false
	}
	blockWidth, blockHeight := 1<<bwLog2, 1<<bhLog2
	minBX, maxBX := int(blocks[0].BX), int(blocks[0].BX)
	minBY, maxBY := int(blocks[0].BY), int(blocks[0].BY)
	for _, block := range blocks[1:] {
		bx, by := int(block.BX), int(block.BY)
		minBX, maxBX = min(minBX, bx), max(maxBX, bx)
		minBY, maxBY = min(minBY, by), max(maxBY, by)
	}
	start := inputOrigin + ((minBY * BStride) << bhLog2) + (minBX << bwLog2) - cdefTapReach*BStride - cdefTapReach
	rows := (maxBY-minBY+1)*blockHeight + 2*cdefTapReach
	cols := (maxBX-minBX+1)*blockWidth + 2*cdefTapReach
	if start < 0 || start+(rows-1)*BStride+cols > len(input) {
		return false
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			if input[start+row*BStride+col] > 255 {
				return false
			}
		}
	}
	return true
}
