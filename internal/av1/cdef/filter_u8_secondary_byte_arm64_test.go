// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"fmt"
	"reflect"
	"simd/archsimd"
	"testing"
)

func TestConstrainSecondaryByteSIMDExhaustive(t *testing.T) {
	for _, strength := range []int{1, 2, 4} {
		str := archsimd.BroadcastUint8x16(uint8(strength))
		for shift := 0; shift <= 8; shift++ {
			sh := archsimd.BroadcastInt8x16(-int8(shift))
			for x := 0; x < 256; x++ {
				xv := archsimd.BroadcastUint8x16(uint8(x))
				for tap := 0; tap < 256; tap++ {
					tv := archsimd.BroadcastUint8x16(uint8(tap))
					got := int(cdefConstrainSecondaryByteSIMD(tv, xv, str, sh).GetElem(0))
					want := constrainShifted(tap-x, strength, shift)
					if got != want {
						t.Fatalf("strength=%d shift=%d tap=%d center=%d got=%d want=%d", strength, shift, tap, x, got, want)
					}
				}
			}
		}
	}
}

func TestRoundSecondaryByteSIMDAllPossibleSums(t *testing.T) {
	for sum := -48; sum <= 48; sum++ {
		got := int(cdefRoundSecondaryByteSIMD(archsimd.BroadcastInt8x16(int8(sum))).GetElem(0))
		want := (sum + (sum >> 7) + 8) >> 4
		if got != want {
			t.Fatalf("sum=%d got=%d want=%d", sum, got, want)
		}
	}
}

func TestFilterFrameBlocksU8SecondaryOnlyByteInteriorSIMDParity(t *testing.T) {
	if !U8SecondaryOnlyByteInteriorAvailable() {
		t.Fatal("SIMD secondary-only byte kernel must be available on this build")
	}
	for _, dec := range [][2]uint8{{0, 0}, {0, 1}, {1, 0}, {1, 1}} {
		blockWidth := 8 >> dec[0]
		blockHeight := 8 >> dec[1]
		unitWidth := blockWidth * 2
		unitHeight := blockHeight * 2
		blocks := []BlockPosition{{BX: 0, BY: 0}, {BX: 1, BY: 0}, {BX: 0, BY: 1}, {BX: 1, BY: 1}}
		for _, pattern := range []string{"mixed-byte-ramp", "low-range-random", "alternating-extremes", "constant-zero", "constant-mid", "constant-maximum"} {
			t.Run(fmt.Sprintf("dec%v/%s", dec, pattern), func(t *testing.T) {
				input, origin := makeSecondaryByteUnitInputPattern(unitWidth, unitHeight, 0x3952, pattern)
				wide := make([]uint16, len(input))
				for i, sample := range input {
					wide[i] = uint16(sample)
				}
				for _, plane := range []Plane{PlaneY, PlaneU, PlaneV} {
					for _, strength := range []uint8{1, 2, 4} {
						for damping := uint8(3); damping <= 6; damping++ {
							params := FrameFilterParams{
								XDec: dec[0], YDec: dec[1], Plane: plane,
								Level: 0, SecondaryStrength: strength, Damping: damping,
							}
							const padding = 5
							dstStride := unitWidth + padding
							dstLen := (unitHeight-1)*dstStride + unitWidth
							got := make([]byte, dstLen)
							want := make([]byte, dstLen)
							for i := range got {
								got[i] = byte(i*37 + int(strength)*11 + int(plane)*53)
								want[i] = got[i]
							}
							if !FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(got, dstStride, input, origin, unitWidth, unitHeight, blocks, params) {
								t.Fatalf("route declined plane=%d strength=%d damping=%d", plane, strength, damping)
							}
							effectiveDamping := int(damping)
							if plane != PlaneY {
								effectiveDamping--
							}
							for _, block := range blocks {
								srcOrigin := origin + int(block.BY)*blockHeight*BStride + int(block.BX)*blockWidth
								dstOrigin := int(block.BY)*blockHeight*dstStride + int(block.BX)*blockWidth
								filterBlockU8PureGo(want, dstStride, dstOrigin, wide, srcOrigin, BlockFilterParams{
									SecondaryStrength: strength,
									Direction:         0,
									SecondaryDamping:  uint8(effectiveDamping),
									Width:             uint8(blockWidth),
									Height:            uint8(blockHeight),
								})
							}
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("plane=%d strength=%d damping=%d idx=%d got=%d want=%d", plane, strength, damping, i, got[i], want[i])
								}
							}
						}
					}
				}
			})
		}
	}
}

func TestSecondaryOnlyByteGreedyGroupEntries(t *testing.T) {
	tests := []struct {
		name       string
		blockWidth int
		blocks     []BlockPosition
		want       []int
	}{
		{name: "width8_k_minus_1", blockWidth: 8, blocks: []BlockPosition{{BX: 3, BY: 1}}, want: []int{1}},
		{name: "width8_k", blockWidth: 8, blocks: []BlockPosition{{BX: 3, BY: 1}, {BX: 4, BY: 1}}, want: []int{2}},
		{name: "width8_k_plus_1", blockWidth: 8, blocks: []BlockPosition{{BX: 3, BY: 1}, {BX: 4, BY: 1}, {BX: 5, BY: 1}}, want: []int{2, 1}},
		{name: "width8_two_k", blockWidth: 8, blocks: []BlockPosition{{BX: 3, BY: 1}, {BX: 4, BY: 1}, {BX: 5, BY: 1}, {BX: 6, BY: 1}}, want: []int{2, 2}},
		{name: "width8_gap_then_pair", blockWidth: 8, blocks: []BlockPosition{{BX: 0, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}}, want: []int{1, 2}},
		{name: "width8_row_break", blockWidth: 8, blocks: []BlockPosition{{BX: 0, BY: 0}, {BX: 1, BY: 1}, {BX: 2, BY: 1}}, want: []int{1, 2}},
		{name: "width8_reversed", blockWidth: 8, blocks: []BlockPosition{{BX: 2, BY: 0}, {BX: 1, BY: 0}}, want: []int{1, 1}},
		{name: "width8_duplicate", blockWidth: 8, blocks: []BlockPosition{{BX: 1, BY: 0}, {BX: 1, BY: 0}}, want: []int{1, 1}},
		{name: "width4_k_minus_1", blockWidth: 4, blocks: []BlockPosition{{BX: 1, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}}, want: []int{1, 1, 1}},
		{name: "width4_k", blockWidth: 4, blocks: []BlockPosition{{BX: 1, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}, {BX: 4, BY: 0}}, want: []int{4}},
		{name: "width4_k_plus_1", blockWidth: 4, blocks: []BlockPosition{{BX: 1, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}, {BX: 4, BY: 0}, {BX: 5, BY: 0}}, want: []int{4, 1}},
		{name: "width4_two_k", blockWidth: 4, blocks: []BlockPosition{{BX: 1, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}, {BX: 4, BY: 0}, {BX: 5, BY: 0}, {BX: 6, BY: 0}, {BX: 7, BY: 0}, {BX: 8, BY: 0}}, want: []int{4, 4}},
		{name: "width4_gap_then_group", blockWidth: 4, blocks: []BlockPosition{{BX: 0, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}, {BX: 4, BY: 0}, {BX: 5, BY: 0}}, want: []int{1, 4}},
		{name: "width4_row_break", blockWidth: 4, blocks: []BlockPosition{{BX: 0, BY: 0}, {BX: 1, BY: 1}, {BX: 2, BY: 1}, {BX: 3, BY: 1}, {BX: 4, BY: 1}}, want: []int{1, 4}},
		{name: "width4_reversed", blockWidth: 4, blocks: []BlockPosition{{BX: 4, BY: 0}, {BX: 3, BY: 0}, {BX: 2, BY: 0}, {BX: 1, BY: 0}}, want: []int{1, 1, 1, 1}},
		{name: "width4_duplicate", blockWidth: 4, blocks: []BlockPosition{{BX: 1, BY: 0}, {BX: 1, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}}, want: []int{1, 1, 1, 1}},
		{name: "unsupported_width", blockWidth: 16, blocks: []BlockPosition{{BX: 0, BY: 0}, {BX: 1, BY: 0}}, want: []int{1, 1}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got []int
			for i := 0; i < len(tc.blocks); {
				n := secondaryOnlyByteGroupEntries(tc.blocks, i, tc.blockWidth)
				if n < 1 || n > len(tc.blocks)-i {
					t.Fatalf("invalid group length %d at index %d", n, i)
				}
				got = append(got, n)
				i += n
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("greedy groups=%v want=%v", got, tc.want)
			}
		})
	}
}

func TestFilterFrameBlocksU8SecondaryOnlyByteInteriorSIMDCoalescedRowsParity(t *testing.T) {
	if !U8SecondaryOnlyByteInteriorAvailable() {
		t.Fatal("SIMD secondary-only byte kernel must be available on this build")
	}
	for _, dec := range [][2]uint8{{0, 0}, {0, 1}, {1, 0}, {1, 1}} {
		blockWidth := 8 >> dec[0]
		blockHeight := 8 >> dec[1]
		groupEntries := 2
		unitWidths := []int{8, 16, 24, 32, 40}
		if blockWidth == 4 {
			groupEntries = 4
			unitWidths = []int{12, 16, 20, 24, 28, 32, 40}
		}
		for _, unitWidth := range unitWidths {
			blockCols := unitWidth / blockWidth
			unitHeight := blockHeight * 2
			blockRows := unitHeight / blockHeight
			seed := uint32(unitWidth*31 + int(dec[0])*11 + int(dec[1])*7)
			for _, by := range []int{0, blockRows - 1} {
				if blockCols >= groupEntries {
					startBX := 0
					if blockCols > groupEntries {
						startBX = 1
					}
					blocks := make([]BlockPosition, groupEntries)
					for i := range groupEntries {
						blocks[i] = BlockPosition{BX: uint8(startBX + i), BY: uint8(by)}
					}
					if got := secondaryOnlyByteGroupEntries(blocks, 0, blockWidth); got != groupEntries {
						t.Fatalf("dec=%v width=%d by=%d failed to form exact group: %d", dec, unitWidth, by, got)
					}
					t.Run(fmt.Sprintf("dec%v/width%d/by%d/group", dec, unitWidth, by), func(t *testing.T) {
						runSecondaryOnlyByteBlockListParity(t, unitWidth, unitHeight, blockWidth, blockHeight, dec, blocks, seed, "low-range-random", 2)
					})
				}
				if blockCols >= groupEntries+1 {
					startBX := 0
					if blockCols >= groupEntries+2 {
						startBX = 1
					}
					blocks := make([]BlockPosition, groupEntries+1)
					for i := range groupEntries + 1 {
						blocks[i] = BlockPosition{BX: uint8(startBX + i), BY: uint8(by)}
					}
					t.Run(fmt.Sprintf("dec%v/width%d/by%d/group_plus_tail", dec, unitWidth, by), func(t *testing.T) {
						runSecondaryOnlyByteBlockListParity(t, unitWidth, unitHeight, blockWidth, blockHeight, dec, blocks, seed+1, "low-range-random", 2)
					})
				}
			}
			if blockCols >= groupEntries+2 {
				blocks := []BlockPosition{{BX: 0, BY: 0}}
				for i := range groupEntries {
					blocks = append(blocks, BlockPosition{BX: uint8(2 + i), BY: 0})
				}
				if got := secondaryOnlyByteGroupEntries(blocks, 0, blockWidth); got != 1 || secondaryOnlyByteGroupEntries(blocks, 1, blockWidth) != groupEntries {
					t.Fatalf("dec=%v width=%d gap did not preserve the next greedy group", dec, unitWidth)
				}
				t.Run(fmt.Sprintf("dec%v/width%d/gap_then_group", dec, unitWidth), func(t *testing.T) {
					runSecondaryOnlyByteBlockListParity(t, unitWidth, unitHeight, blockWidth, blockHeight, dec, blocks, seed+2, "low-range-random", 2)
				})
			}
			if blockCols >= 2*groupEntries {
				startBX := 0
				if blockCols >= 2*groupEntries+1 {
					startBX = 1
				}
				blocks := make([]BlockPosition, 2*groupEntries)
				for i := range 2 * groupEntries {
					blocks[i] = BlockPosition{BX: uint8(startBX + i), BY: 0}
				}
				t.Run(fmt.Sprintf("dec%v/width%d/two_groups", dec, unitWidth), func(t *testing.T) {
					runSecondaryOnlyByteBlockListParity(t, unitWidth, unitHeight, blockWidth, blockHeight, dec, blocks, seed+3, "low-range-random", 2)
				})
			}
		}
	}
}

func TestFilterFrameBlocksU8SecondaryOnlyByteInteriorSIMDCoalescedWidth4Patterns(t *testing.T) {
	const unitWidth, unitHeight = 16, 8
	dec := [2]uint8{1, 1}
	blocks := []BlockPosition{{BX: 0, BY: 0}, {BX: 1, BY: 0}, {BX: 2, BY: 0}, {BX: 3, BY: 0}}
	for _, pattern := range []string{"low-range-random", "alternating-extremes", "constant-zero", "constant-mid", "constant-maximum"} {
		for _, strength := range []uint8{1, 2, 4} {
			t.Run(fmt.Sprintf("%s/strength%d", pattern, strength), func(t *testing.T) {
				runSecondaryOnlyByteBlockListParity(t, unitWidth, unitHeight, 4, 4, dec, blocks, 0x2351+uint32(strength), pattern, strength)
			})
		}
	}
}

func TestFilterFrameBlocksU8SecondaryOnlyByteInteriorSIMDCoalescedPreservesListSemantics(t *testing.T) {
	const unitWidth, unitHeight = 64, 16
	input, origin := makeSecondaryByteUnitInputPattern(unitWidth, unitHeight, 0x61d3, "low-range-random")
	inputBefore := append([]byte(nil), input...)
	wide := make([]uint16, len(input))
	for i, sample := range input {
		wide[i] = uint16(sample)
	}
	// The reversed and duplicate entries must remain separate operations. The
	// later same-row run may be partially coalesced; no earlier hole is filled.
	blocks := []BlockPosition{
		{BX: 2, BY: 0}, {BX: 1, BY: 0}, {BX: 1, BY: 0},
		{BX: 0, BY: 1}, {BX: 1, BY: 1}, {BX: 2, BY: 1},
	}
	const padding = 7
	dstStride := unitWidth + padding
	dstLen := (unitHeight-1)*dstStride + unitWidth
	got := make([]byte, dstLen)
	want := make([]byte, dstLen)
	for i := range got {
		got[i] = byte(i*19 + 3)
		want[i] = got[i]
	}
	params := FrameFilterParams{SecondaryStrength: 4, Damping: 6}
	if !FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(got, dstStride, input, origin, unitWidth, unitHeight, blocks, params) {
		t.Fatal("mixed sparse block list was declined")
	}
	for _, block := range blocks {
		srcOrigin := origin + int(block.BY)*8*BStride + int(block.BX)*8
		dstOrigin := int(block.BY)*8*dstStride + int(block.BX)*8
		filterBlockU8PureGo(want, dstStride, dstOrigin, wide, srcOrigin, BlockFilterParams{
			SecondaryStrength: params.SecondaryStrength,
			Direction:         0,
			SecondaryDamping:  params.Damping,
			Width:             8,
			Height:            8,
		})
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("idx=%d got=%d want=%d", i, got[i], want[i])
		}
	}
	for i := range input {
		if input[i] != inputBefore[i] {
			t.Fatalf("immutable input changed at %d: got=%d want=%d", i, input[i], inputBefore[i])
		}
	}
}

func runSecondaryOnlyByteBlockListParity(t *testing.T, unitWidth int, unitHeight int, blockWidth int, blockHeight int, dec [2]uint8, blocks []BlockPosition, seed uint32, pattern string, strength uint8) {
	t.Helper()
	input, origin := makeSecondaryByteUnitInputPattern(unitWidth, unitHeight, seed, pattern)
	wide := make([]uint16, len(input))
	for i, sample := range input {
		wide[i] = uint16(sample)
	}
	const padding = 5
	dstStride := unitWidth + padding
	dstLen := (unitHeight-1)*dstStride + unitWidth
	got := make([]byte, dstLen)
	want := make([]byte, dstLen)
	for i := range got {
		got[i] = byte(i*53 + int(seed))
		want[i] = got[i]
	}
	params := FrameFilterParams{XDec: dec[0], YDec: dec[1], SecondaryStrength: strength, Damping: 5}
	if !FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(got, dstStride, input, origin, unitWidth, unitHeight, blocks, params) {
		t.Fatal("coalesced block list was declined")
	}
	for _, block := range blocks {
		srcOrigin := origin + (int(block.BY)*blockHeight)*BStride + int(block.BX)*blockWidth
		dstOrigin := (int(block.BY)*blockHeight)*dstStride + int(block.BX)*blockWidth
		filterBlockU8PureGo(want, dstStride, dstOrigin, wide, srcOrigin, BlockFilterParams{
			SecondaryStrength: params.SecondaryStrength,
			Direction:         0,
			SecondaryDamping:  params.Damping,
			Width:             uint8(blockWidth),
			Height:            uint8(blockHeight),
		})
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("idx=%d got=%d want=%d blocks=%v", i, got[i], want[i], blocks)
		}
	}
}

func TestFilterFrameBlocksU8SecondaryOnlyByteInteriorSIMDRejectsWithoutMutation(t *testing.T) {
	input, origin := makeSecondaryByteUnitInput(16, 16, 0x17a3)
	blocks := []BlockPosition{{BX: 0, BY: 0}}
	params := FrameFilterParams{SecondaryStrength: 2, Damping: 4}
	dst := make([]byte, 16*16)
	for i := range dst {
		dst[i] = byte(i*13 + 7)
	}
	checkDeclines := func(name string, in []byte, w, h int, p FrameFilterParams) {
		t.Helper()
		before := append([]byte(nil), dst...)
		if FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(dst, 16, in, origin, w, h, blocks, p) {
			t.Fatalf("%s unexpectedly selected byte route", name)
		}
		for i := range dst {
			if dst[i] != before[i] {
				t.Fatalf("%s mutated dst at %d", name, i)
			}
		}
	}
	bad := params
	bad.Level = 1
	checkDeclines("primary", input, 16, 16, bad)
	bad = params
	bad.SecondaryStrength = 0
	checkDeclines("zero-secondary", input, 16, 16, bad)
	checkDeclines("short-input", input[:len(input)-1], 16, 16, params)
	checkDeclines("misaligned-unit", input, 15, 16, params)
}

func TestFilterFrameBlocksU8SecondaryOnlyByteInteriorSIMDZeroAlloc(t *testing.T) {
	const unitWidth, unitHeight = 64, 64
	input, origin := makeSecondaryByteUnitInput(unitWidth, unitHeight, 0x20ab)
	blocks := make([]BlockPosition, 0, 64)
	for by := range 8 {
		for bx := range 8 {
			blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
		}
	}
	dst := make([]byte, unitWidth*unitHeight)
	params := FrameFilterParams{SecondaryStrength: 4, Damping: 6}
	allocs := testing.AllocsPerRun(100, func() {
		if !FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(dst, unitWidth, input, origin, unitWidth, unitHeight, blocks, params) {
			t.Fatal("byte route declined")
		}
	})
	if allocs != 0 {
		t.Fatalf("secondary byte route allocates: %v allocs/op", allocs)
	}
}

func makeSecondaryByteUnitInput(unitWidth int, unitHeight int, seed uint32) ([]byte, int) {
	return makeSecondaryByteUnitInputPattern(unitWidth, unitHeight, seed, "mixed-byte-ramp")
}

func makeSecondaryByteUnitInputPattern(unitWidth int, unitHeight int, seed uint32, pattern string) ([]byte, int) {
	origin := VerticalBorder*BStride + HorizontalBorder
	end := origin + (unitHeight+1)*BStride + unitWidth + 2
	input := make([]byte, end)
	for i := range input {
		input[i] = 0xa5
	}
	rng := seed
	for row := -VerticalBorder; row < unitHeight+VerticalBorder; row++ {
		for col := -2; col < unitWidth+2; col++ {
			var value byte
			switch pattern {
			case "low-range-random":
				rng ^= rng << 13
				rng ^= rng >> 17
				rng ^= rng << 5
				value = byte(rng & 15)
			case "alternating-extremes":
				if (row+col)&1 == 0 {
					value = 0
				} else {
					value = 255
				}
			case "constant-zero":
				value = 0
			case "constant-mid":
				value = 127
			case "constant-maximum":
				value = 255
			default:
				value = byte(uint32(row*73+col*29) + seed)
			}
			input[origin+row*BStride+col] = value
		}
	}
	return input, origin
}
