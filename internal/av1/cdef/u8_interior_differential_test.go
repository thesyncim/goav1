// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "testing"

// TestFilterBlockU8InteriorSIMDMatchesPureGo is the safety net for the interior
// fused kernel: over sentinel-free tap buffers (every tap a real 8-bit sample)
// it pins the 8-wide fused block against filterBlockU8PureGo across every
// direction, the strength/damping corpus, both 8-wide heights, and many random
// inputs. A byte mismatch fails, so the byte-lane constrain must reproduce the
// reference exactly.
func TestFilterBlockU8InteriorSIMDMatchesPureGo(t *testing.T) {
	const dstStride = 24
	origin := cdefBlockOrigin()
	for trial := 0; trial < 3000; trial++ {
		rnd := newCDEFRandom(uint32(0x1a7e0000 + trial))
		input := makeCDEFBlockInput(rnd, 8, trial%16, trial%16+1)
		for i := range input {
			if input[i] == VeryLarge {
				input[i] = uint16(rnd.generate(256))
			}
		}
		for _, height := range []int{8, 4} {
			params := BlockFilterParams{
				PrimaryStrength:   uint8(1 + rnd.generate(15)),
				SecondaryStrength: uint8(1 + rnd.generate(4)),
				Direction:         uint8(rnd.generate(8)),
				PrimaryDamping:    uint8(3 + rnd.generate(4)),
				SecondaryDamping:  uint8(3 + rnd.generate(4)),
				Width:             8,
				Height:            uint8(height),
			}
			want := make([]byte, dstStride*8)
			got := make([]byte, dstStride*8)
			filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
			filterBlockU8InteriorSIMD(got, dstStride, 0, input, origin, params)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("trial=%d params=%+v idx=%d interior=%d want=%d", trial, params, i, got[i], want[i])
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

// TestFilterBlockU8InteriorSIMDClipCorpus drives the interior kernel's clamp:
// narrow-spread, sentinel-free neighbourhoods with strong fused strengths are
// the only inputs where the summed taps overshoot the neighbourhood, so the
// random corpus above does not reach the clamp by itself.
func TestFilterBlockU8InteriorSIMDClipCorpus(t *testing.T) {
	rng := newCDEFRandom(0x1c11b0a7)
	input := make([]uint16, BStride*24)
	origin := cdefBlockOrigin()
	const dstStride = 24
	for trial := 0; trial < 6000; trial++ {
		base := rng.generate(256)
		spread := uint32(1) << rng.generate(6)
		for i := range input {
			input[i] = uint16(min(base+rng.generate(spread+1), 255))
		}
		params := BlockFilterParams{
			PrimaryStrength:   uint8(1 + rng.generate(15)),
			SecondaryStrength: uint8(1 + rng.generate(4)),
			Direction:         uint8(rng.generate(8)),
			PrimaryDamping:    uint8(3 + rng.generate(4)),
			SecondaryDamping:  uint8(3 + rng.generate(4)),
			Width:             8,
			Height:            8,
		}
		want := make([]byte, dstStride*8)
		got := make([]byte, dstStride*8)
		filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
		filterBlockU8InteriorSIMD(got, dstStride, 0, input, origin, params)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial=%d params=%+v base=%d spread=%d idx=%d interior=%d want=%d", trial, params, base, spread, i, got[i], want[i])
			}
		}
	}
}
