// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "testing"

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
