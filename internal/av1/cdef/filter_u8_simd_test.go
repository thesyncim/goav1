// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package cdef

import (
	"math/rand"
	"testing"
)

// TestFilterBlockU8SIMDMatchesPureGo pins the Go SIMD 8-bit block
// filter against the pure-Go reference. It covers every legal 8-bit primary
// strength (0..15), the full secondary range accepted by this package (0..4),
// all directions, all halo-sentinel patterns, all u8 CDEF block shapes, and
// representative damping values without going through cpu.Detected dispatch.
func TestFilterBlockU8SIMDMatchesPureGo(t *testing.T) {
	const (
		dstStride = 19
		dstOrigin = 5
	)
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {4, 8}, {4, 4},
	}
	for boundary := 0; boundary < 16; boundary++ {
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x38415658), 8, boundary, boundary+2)
		for _, shape := range shapes {
			for dir := 0; dir <= 7; dir++ {
				for pri := 0; pri <= 15; pri++ {
					for sec := 0; sec <= 4; sec++ {
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
								Width:             uint8(shape.width),
								Height:            uint8(shape.height),
							}
							want := makeGuardedU8Dst(dstOrigin + dstStride*8 + 8)
							got := makeGuardedU8Dst(len(want))
							filterBlockU8PureGo(want, dstStride, dstOrigin, input, origin, params)
							filterBlockU8SIMD(got, dstStride, dstOrigin, input, origin, params)
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
										boundary, shape.width, shape.height, dir, pri, sec, damping, i, got[i], want[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestFilterUnitBlocksU8SIMDMatchesPureGo proves the unit loop
// remains byte-identical when every per-block call is forced through the Go
// SIMD u8 kernel directly. This covers partial units and skip masks without
// relying on the runtime dispatch slot.
func TestFilterUnitBlocksU8SIMDMatchesPureGo(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x41563832)
	cases := make([]cdefU8UnitCase, 0, 5120)
	for _, geom := range []struct {
		plane      Plane
		xDec, yDec int
	}{
		{PlaneY, 0, 0},
		{PlaneU, 1, 1},
		{PlaneU, 1, 0},
		{PlaneU, 0, 0},
	} {
		unitWFull := BlockSize >> geom.xDec
		unitHFull := BlockSize >> geom.yDec
		blockW := 8 >> geom.xDec
		blockH := 8 >> geom.yDec
		for _, size := range []struct{ w, h int }{
			{unitWFull, unitHFull},
			{unitWFull - blockW, unitHFull - blockH},
			{blockW, blockH},
			{unitWFull, blockH * 2},
		} {
			for level := 0; level <= 15; level++ {
				for sec := 0; sec <= 4; sec++ {
					for _, damping := range []int{3, 4, 5, 6} {
						cases = append(cases, cdefU8UnitCase{
							plane: geom.plane, xDec: geom.xDec, yDec: geom.yDec,
							unitW: size.w, unitH: size.h,
							level: level, sec: sec, damping: damping,
							skipMask: uint64(rnd.generate(1<<16)) | uint64(rnd.generate(1<<16))<<16,
							halo:     uint8(rnd.generate(16)),
						})
					}
				}
			}
		}
	}
	for i, tc := range cases {
		runCDEFU8UnitSIMDDifferential(t, rnd, tc, i)
	}
}

func makeGuardedU8Dst(n int) []byte {
	dst := make([]byte, n)
	for i := range dst {
		dst[i] = byte((i*37 + 113) & 0xff)
	}
	return dst
}

func runCDEFU8UnitSIMDDifferential(t *testing.T, rnd *cdefRandom, tc cdefU8UnitCase, caseIdx int) {
	t.Helper()
	blockW := 8 >> tc.xDec
	blockH := 8 >> tc.yDec
	blockCols := (tc.unitW + blockW - 1) / blockW
	blockRows := (tc.unitH + blockH - 1) / blockH

	const margin = 24
	stride := tc.unitW + 2*margin + 5
	height := tc.unitH + 2*margin
	frame := make([]byte, stride*height)
	for i := range frame {
		frame[i] = byte(rnd.generate(256))
	}
	unitOrigin := margin*stride + margin

	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(256))
	}
	origin := cdefBlockOrigin()
	for row := 0; row < tc.unitH; row++ {
		for col := 0; col < tc.unitW; col++ {
			input[origin+row*BStride+col] = uint16(frame[unitOrigin+row*stride+col])
		}
	}
	applyCDEFU8UnitHalo(input, tc.unitW, tc.unitH, tc.halo)

	blocks := make([]BlockPosition, 0, blockCols*blockRows)
	bit := 0
	for by := 0; by < blockRows; by++ {
		for bx := 0; bx < blockCols; bx++ {
			if tc.skipMask&(1<<uint(bit%64)) == 0 {
				blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
			}
			bit++
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, BlockPosition{BY: 0, BX: 0})
	}

	var dirs DirectionGrid
	var vars VarianceGrid
	for by := range NBlocks {
		for bx := range NBlocks {
			dirs[by][bx] = uint8(rnd.generate(8))
			vars[by][bx] = int32(rnd.generate(1 << 20))
		}
	}
	wantDirs := dirs
	gotDirs := dirs
	wantVars := vars
	gotVars := vars
	u := unitFilterParams{
		primaryStrength:   tc.level,
		secondaryStrength: tc.sec,
		damping:           tc.damping,
		coeffShift:        0,
		bwLog2:            3 - tc.xDec,
		bhLog2:            3 - tc.yDec,
		blockWidth:        blockW,
		blockHeight:       blockH,
		lumaAdjust:        tc.plane == PlaneY,
	}

	want := make([]byte, len(frame))
	copy(want, frame)
	got := make([]byte, len(frame))
	copy(got, frame)
	if err := filterUnitBlocksU8WithBlockForTest(want[unitOrigin:], stride, input, origin, blocks, &wantDirs, &wantVars, u, filterBlockU8PureGo); err != nil {
		t.Fatalf("case %d: pure-Go unit filter: %v", caseIdx, err)
	}
	if err := filterUnitBlocksU8WithBlockForTest(got[unitOrigin:], stride, input, origin, blocks, &gotDirs, &gotVars, u, filterBlockU8SIMD); err != nil {
		t.Fatalf("case %d: SIMD unit filter: %v", caseIdx, err)
	}
	if gotDirs != wantDirs {
		t.Fatalf("case %d (%+v): direction grids differ", caseIdx, tc)
	}
	if gotVars != wantVars {
		t.Fatalf("case %d (%+v): variance grids differ", caseIdx, tc)
	}
	for i := range got {
		if got[i] != want[i] {
			row := i / stride
			col := i % stride
			t.Fatalf("case %d (%+v): byte mismatch at row=%d col=%d got=%d want=%d", caseIdx, tc, row, col, got[i], want[i])
		}
	}
}

func filterUnitBlocksU8WithBlockForTest(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, filterBlock func([]byte, int, int, []uint16, int, BlockFilterParams)) error {
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := u.primaryStrength
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
		}
		if strength == 0 && u.secondaryStrength == 0 {
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		params := BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(u.secondaryStrength),
			Direction:         uint8(dir),
			PrimaryDamping:    uint8(u.damping),
			SecondaryDamping:  uint8(u.damping),
			CoeffShift:        0,
			Width:             uint8(u.blockWidth),
			Height:            uint8(u.blockHeight),
		}
		filterBlock(dst, dstStride, dstOrigin, input, srcOrigin, params)
	}
	return nil
}

// compareFilterBlockU8SIMD checks one u8 block against the pure-Go reference.
func compareFilterBlockU8SIMD(t *testing.T, input []uint16, origin int, params BlockFilterParams) {
	t.Helper()
	const dstStride = 16
	want := make([]byte, dstStride*8)
	got := make([]byte, dstStride*8)
	filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
	filterBlockU8SIMD(got, dstStride, 0, input, origin, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("w=%d h=%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
				params.Width, params.Height, params.Direction, params.PrimaryStrength,
				params.SecondaryStrength, params.PrimaryDamping, i, got[i], want[i])
		}
	}
}

// TestFilterBlockU8SIMDExtremes drives all-zero, all-255, and alternating
// 0/255 neighbourhoods with VeryLarge halos through every direction and both
// single-strength shapes; the extremes sit on the clamp and saturation edges.
func TestFilterBlockU8SIMDExtremes(t *testing.T) {
	origin := cdefBlockOrigin()
	fills := []func(int) uint16{
		func(int) uint16 { return 0 },
		func(int) uint16 { return 255 },
		func(i int) uint16 {
			if i&1 == 0 {
				return 0
			}
			return 255
		},
	}
	for _, fill := range fills {
		input := make([]uint16, InputBufferSize)
		for i := range input {
			input[i] = fill(i)
		}
		for i := range input {
			row, col := i/BStride, i%BStride
			if row < VerticalBorder || row >= VerticalBorder+8 || col < HorizontalBorder || col >= HorizontalBorder+8 {
				input[i] = VeryLarge
			}
		}
		for dir := 0; dir < 8; dir++ {
			for _, shape := range [...]struct{ width, height int }{{8, 8}, {4, 4}} {
				for _, strengths := range [][2]uint8{{15, 0}, {0, 4}, {15, 4}} {
					compareFilterBlockU8SIMD(t, input, origin, BlockFilterParams{
						PrimaryStrength: strengths[0], SecondaryStrength: strengths[1],
						Direction: uint8(dir), PrimaryDamping: 3, SecondaryDamping: 3,
						Width: uint8(shape.width), Height: uint8(shape.height),
					})
				}
			}
		}
	}
}

// TestFilterBlockU8SIMDClipCorpus is the 8-bit counterpart of
// TestFilterBlockSIMDClipCorpus: narrow-spread neighbourhoods at 8-bit depth
// with strong strengths, so the fused clamp and the sentinel skip are reached.
func TestFilterBlockU8SIMDClipCorpus(t *testing.T) {
	rng := rand.New(rand.NewSource(0x8b175ca1))
	input := make([]uint16, BStride*24)
	origin := cdefBlockOrigin()
	for trial := 0; trial < 6000; trial++ {
		base := rng.Intn(256)
		spread := 1 << rng.Intn(6)
		for i := range input {
			v := min(base+rng.Intn(spread+1), 255)
			if rng.Intn(8) == 0 {
				v = VeryLarge
			}
			input[i] = uint16(v)
		}
		width, height := 8, 8
		if rng.Intn(2) == 0 {
			width, height = 4, 4
		}
		params := BlockFilterParams{
			PrimaryStrength:   uint8(1 + rng.Intn(15)),
			SecondaryStrength: uint8(1 + rng.Intn(4)),
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + rng.Intn(4)),
			SecondaryDamping:  uint8(3 + rng.Intn(4)),
			Width:             uint8(width),
			Height:            uint8(height),
		}
		compareFilterBlockU8SIMD(t, input, origin, params)
	}
}
