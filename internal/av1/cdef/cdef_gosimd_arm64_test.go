//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"fmt"
	"math/rand"
	"reflect"
	"simd/archsimd"
	"testing"
)

// directionSIMDBound reports whether the Go SIMD direction kernels are the
// dispatch targets: always on arm64 (NEON is mandatory).
func directionSIMDBound() bool { return true }

func TestFilterUnitBlocksSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x11cdef))
	input := make([]uint16, InputBufferSize)
	inputOrigin := VerticalBorder*BStride + HorizontalBorder
	const dstStride = BStride
	for trial := 0; trial < 2000; trial++ {
		// 8, 9, 10 and 12 bits: the secondary-only, primary-only and fused
		// kernels at every depth.
		coeffShift := []int{0, 1, 2, 4}[rng.Intn(4)]
		maxSample := (1 << (8 + coeffShift)) - 1
		for i := range input {
			if rng.Intn(13) == 0 {
				input[i] = VeryLarge
			} else {
				input[i] = uint16(rng.Intn(maxSample + 1))
			}
		}
		xDec := rng.Intn(2)
		yDec := rng.Intn(2)
		lumaAdjust := rng.Intn(2) == 0
		if lumaAdjust {
			xDec, yDec = 0, 0
		}
		bwLog2 := 3 - xDec
		bhLog2 := 3 - yDec
		u := unitFilterParams{
			primaryStrength:   rng.Intn(16) << coeffShift,
			secondaryStrength: []int{0, 1, 2, 4}[rng.Intn(4)] << coeffShift,
			damping:           3 + coeffShift + rng.Intn(3),
			coeffShift:        coeffShift,
			bwLog2:            bwLog2,
			bhLog2:            bhLog2,
			blockWidth:        1 << bwLog2,
			blockHeight:       1 << bhLog2,
			lumaAdjust:        lumaAdjust,
		}
		var directions DirectionGrid
		var variances VarianceGrid
		var blocks []BlockPosition
		for by := 0; by < 8; by++ {
			for bx := 0; bx < 8; bx++ {
				directions[by][bx] = uint8(rng.Intn(8))
				variances[by][bx] = int32(rng.Intn(1 << 20))
				if rng.Intn(4) == 0 {
					variances[by][bx] = 0
				}
				if rng.Intn(3) != 0 {
					blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		if len(blocks) == 0 {
			blocks = append(blocks, BlockPosition{})
		}
		want := make([]uint16, dstStride*BlockSize)
		got := make([]uint16, dstStride*BlockSize)
		for i := range want {
			want[i] = 0xdead
			got[i] = 0xdead
		}
		filterUnitBlocksReference(want, dstStride, input, inputOrigin, blocks, &directions, &variances, u)
		if err := filterUnitBlocksSIMD(got, dstStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("trial=%d u=%+v sample=%d: SIMD=%d reference=%d", trial, u, i, got[i], want[i])
			}
		}
	}
}

func TestFilterUnitBlocksSIMDZeroAlloc(t *testing.T) {
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(i * 29 & 1023)
	}
	inputOrigin := VerticalBorder*BStride + HorizontalBorder
	var directions DirectionGrid
	var variances VarianceGrid
	var blocks []BlockPosition
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			directions[by][bx] = uint8((by + bx) & 7)
			variances[by][bx] = int32(bx * by << 10)
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	dst := make([]uint16, BStride*BlockSize)
	u := unitFilterParams{
		primaryStrength:   5,
		secondaryStrength: 2,
		damping:           4,
		bwLog2:            3,
		bhLog2:            3,
		blockWidth:        8,
		blockHeight:       8,
		lumaAdjust:        true,
	}
	allocs := testing.AllocsPerRun(50, func() {
		if err := filterUnitBlocksSIMD(dst, BStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("filterUnitBlocksSIMD allocates: %v allocs/run", allocs)
	}
}

func benchFilterUnitBlocks(b *testing.B, fn func(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error) {
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(i * 29 & 1023)
	}
	inputOrigin := VerticalBorder*BStride + HorizontalBorder
	var directions DirectionGrid
	var variances VarianceGrid
	var blocks []BlockPosition
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			directions[by][bx] = uint8((by + bx) & 7)
			variances[by][bx] = int32(bx * by << 10)
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	dst := make([]uint16, BStride*BlockSize)
	u := unitFilterParams{
		primaryStrength:   5,
		secondaryStrength: 2,
		damping:           4,
		bwLog2:            3,
		bhLog2:            3,
		blockWidth:        8,
		blockHeight:       8,
		lumaAdjust:        true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := fn(dst, BStride, input, inputOrigin, blocks, &directions, &variances, u, true); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFilterUnitBlocksSIMD_64Blocks(b *testing.B) {
	benchFilterUnitBlocks(b, filterUnitBlocksSIMD)
}

func BenchmarkFilterUnitBlocksPerBlockCtx_64Blocks(b *testing.B) {
	benchFilterUnitBlocks(b, filterUnitBlocksPureGo)
}

func TestCDEFUnitInteriorScanUsesGoSIMD(t *testing.T) {
	input := make([]uint16, BStride+8)
	for i := range input {
		input[i] = 255
	}
	if !cdefScanU16RowsAtMost255(input, 0, 2, 8) {
		t.Fatal("byte-range input must pass the Go SIMD scan")
	}
	input[BStride+7] = VeryLarge
	if cdefScanU16RowsAtMost255(input, 0, 2, 8) {
		t.Fatal("high sample in the final vector must fail the Go SIMD scan")
	}
}

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

// TestFilterUnitBlocksU8SIMDLoopMatchesPureGo pins the Go SIMD 8-bit unit-level
// loop (strength adjust, zero-strength skip, kernel dispatch) against the
// pure-Go unit loop on whole synthetic units.
func TestFilterUnitBlocksU8SIMDLoopMatchesPureGo(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x384e454f)
	for _, geom := range []struct{ xDec, yDec int }{{0, 0}, {1, 1}, {1, 0}} {
		blockW := 8 >> geom.xDec
		blockH := 8 >> geom.yDec
		unitW := BlockSize >> geom.xDec
		unitH := BlockSize >> geom.yDec
		stride := unitW + 11
		input := make([]uint16, InputBufferSize)
		for i := range input {
			input[i] = uint16(rnd.generate(256))
		}
		var dirs DirectionGrid
		var vars VarianceGrid
		for by := range NBlocks {
			for bx := range NBlocks {
				dirs[by][bx] = uint8(rnd.generate(8))
				vars[by][bx] = int32(rnd.generate(1 << 22))
			}
		}
		blocks := make([]BlockPosition, 0, 64)
		for by := 0; by < unitH/blockH; by++ {
			for bx := 0; bx < unitW/blockW; bx++ {
				if rnd.generate(4) == 0 {
					continue
				}
				blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
			}
		}
		frame := make([]byte, stride*(unitH+8))
		for i := range frame {
			frame[i] = byte(rnd.generate(256))
		}
		for _, lumaAdjust := range []bool{true, false} {
			for _, str := range []struct{ pri, sec int }{{9, 0}, {0, 2}, {15, 4}, {1, 1}} {
				u := unitFilterParams{
					primaryStrength:   str.pri,
					secondaryStrength: str.sec,
					damping:           5,
					bwLog2:            3 - geom.xDec,
					bhLog2:            3 - geom.yDec,
					blockWidth:        blockW,
					blockHeight:       blockH,
					lumaAdjust:        lumaAdjust,
				}
				want := make([]byte, len(frame))
				copy(want, frame)
				got := make([]byte, len(frame))
				copy(got, frame)
				if err := filterUnitBlocksU8PureGo(want, stride, input, cdefBlockOrigin(), blocks, &dirs, &vars, u); err != nil {
					t.Fatal(err)
				}
				if err := filterUnitBlocksU8SIMD(got, stride, input, cdefBlockOrigin(), blocks, &dirs, &vars, u); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("xdec=%d ydec=%d pri=%d sec=%d lumaAdjust=%v idx=%d got=%d want=%d",
							geom.xDec, geom.yDec, str.pri, str.sec, lumaAdjust, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestCDEFPrimarySecondaryU8UnitDispatchMatchesPureGo checks a single-block
// unit against filterBlockU8PureGo, with the oracle built independently of the
// dispatch slot so a shared SIMD bug cannot make both sides agree.
func TestCDEFPrimarySecondaryU8UnitDispatchMatchesPureGo(t *testing.T) {
	blocks := []BlockPosition{{BY: 0, BX: 0}}
	for _, boundary := range []int{0, 15} {
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x77AA55), 8, boundary, boundary+1)
		for _, strengths := range [][2]int{{0, 4}, {15, 0}, {15, 4}} {
			var directions DirectionGrid
			directions[0][0] = 3
			var variances VarianceGrid
			u := unitFilterParams{
				primaryStrength: strengths[0], secondaryStrength: strengths[1], damping: 5,
				bwLog2: 3, bhLog2: 3, blockWidth: 8, blockHeight: 8,
			}
			want := make([]byte, 64)
			got := make([]byte, 64)
			dir := 0
			if strengths[0] != 0 {
				dir = int(directions[0][0])
			}
			params := BlockFilterParams{
				PrimaryStrength: uint8(strengths[0]), SecondaryStrength: uint8(strengths[1]),
				Direction: uint8(dir), PrimaryDamping: uint8(u.damping), SecondaryDamping: uint8(u.damping),
				Width: uint8(u.blockWidth), Height: uint8(u.blockHeight),
			}
			filterBlockU8PureGo(want, 8, 0, input, cdefBlockOrigin(), params)
			if err := filterUnitBlocksU8SIMD(got, 8, input, cdefBlockOrigin(), blocks, &directions, &variances, u); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("boundary=%d pri=%d sec=%d byte=%d got=%d want=%d", boundary, strengths[0], strengths[1], i, got[i], want[i])
				}
			}
		}
	}
}

func TestCDEFPrimarySecondaryU8SIMDZeroAlloc(t *testing.T) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	params := BlockFilterParams{SecondaryStrength: 4, Direction: 3, PrimaryDamping: 5, SecondaryDamping: 5, Width: 8, Height: 8}
	if allocs := testing.AllocsPerRun(50, func() { filterBlockU8Impl(dst, 8, 0, input, cdefBlockOrigin(), params) }); allocs != 0 {
		t.Fatalf("selected secondary kernel allocated %.1f objects/run", allocs)
	}
	blocks := []BlockPosition{{BY: 0, BX: 0}}
	var directions DirectionGrid
	var variances VarianceGrid
	u := unitFilterParams{secondaryStrength: 4, damping: 5, bwLog2: 3, bhLog2: 3, blockWidth: 8, blockHeight: 8}
	if allocs := testing.AllocsPerRun(50, func() {
		if err := filterUnitBlocksU8SIMD(dst, 8, input, cdefBlockOrigin(), blocks, &directions, &variances, u); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("selected unit path allocated %.1f objects/run", allocs)
	}
}

var cdefU8SIMDBenchmarkSink byte

func benchCDEFU8SIMD(b *testing.B, width, height int, primary, secondary int) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 8*8)
	params := BlockFilterParams{
		PrimaryStrength: uint8(primary), SecondaryStrength: uint8(secondary), Direction: 4,
		PrimaryDamping: 5, SecondaryDamping: 5, Width: uint8(width), Height: uint8(height),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		filterBlockU8Impl(dst, 8, 0, input, cdefBlockOrigin(), params)
	}
	cdefU8SIMDBenchmarkSink = dst[0]
}

func BenchmarkCDEFSecondaryU8_8x8_SIMD(b *testing.B) { benchCDEFU8SIMD(b, 8, 8, 0, 4) }
func BenchmarkCDEFSecondaryU8_4x8_SIMD(b *testing.B) { benchCDEFU8SIMD(b, 4, 8, 0, 4) }
func BenchmarkCDEFSecondaryU8_4x4_SIMD(b *testing.B) { benchCDEFU8SIMD(b, 4, 4, 0, 4) }
func BenchmarkCDEFPrimaryU8_8x8_SIMD(b *testing.B)   { benchCDEFU8SIMD(b, 8, 8, 15, 0) }
func BenchmarkCDEFPrimaryU8_4x8_SIMD(b *testing.B)   { benchCDEFU8SIMD(b, 4, 8, 15, 0) }

// filterBlockU8InteriorSIMD runs the interior kernel on one fused 8-wide block.
func filterBlockU8InteriorSIMD(dst []byte, dstStride int, dstOrigin int, input []uint16, inputOrigin int, params BlockFilterParams) {
	var ctx cdefSIMDCtx
	ctx.setBlock(nil, dst, dstOrigin, dstStride, input, inputOrigin, params, true)
	cdefFilterBlock8InteriorSIMD(&ctx)
}

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
