// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cdef

import "testing"

// The AVX2-backed 8-bit wrappers are validated by DIRECT calls (not through
// cpu.Detected dispatch) so the differential also runs under Rosetta, where
// CPUID reports AVX2 false but the instructions execute fine.

// TestFilterBlockU8AVX2MatchesPureGo pins the native u8-store AVX2 block
// filter against the pure-Go reference. It covers every legal 8-bit primary
// strength (0..15), the full secondary range accepted by this package (0..4),
// all directions, all halo-sentinel patterns, all u8 CDEF block shapes, and
// representative damping values without going through cpu.Detected dispatch.
func TestFilterBlockU8AVX2MatchesPureGo(t *testing.T) {
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
							want := makeGuardedU8AVX2Dst(dstOrigin + dstStride*8 + 8)
							got := makeGuardedU8AVX2Dst(len(want))
							filterBlockU8PureGo(want, dstStride, dstOrigin, input, origin, params)
							filterBlockU8AVX2(got, dstStride, dstOrigin, input, origin, params)
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

// TestFilterUnitBlocksU8AVX2MatchesPureGo proves the generic amd64 unit loop
// remains byte-identical when every per-block call is forced through the AVX2
// u8 kernel directly. This covers partial units and skip masks without relying
// on the runtime dispatch slot, which is deliberately false under Rosetta.
func TestFilterUnitBlocksU8AVX2MatchesPureGo(t *testing.T) {
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
		runCDEFU8UnitAVX2Differential(t, rnd, tc, i)
	}
}

// TestFindDirectionU8AVX2MatchesScalar pins the AVX2-backed 8-bit direction
// wrapper (single and dual) against the scalar uint8 reference.
func TestFindDirectionU8AVX2MatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x38445236)
	for _, stride := range []int{8, 23, 320} {
		for iter := range 128 {
			img := make([]byte, stride*8+8)
			for i := range img {
				switch iter % 3 {
				case 0:
					img[i] = byte(rnd.generate(256))
				case 1:
					img[i] = byte(rnd.generate(5))
				default:
					img[i] = byte(251 + rnd.generate(5))
				}
			}
			wantDir, wantVar := findDirectionU8Scalar(img, stride)
			gotDir, gotVar := findDirectionU8AVX2(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			if stride >= 16 {
				wd1, wv1, wd2, wv2 := findDirectionDualU8Scalar(img, img[8:], stride)
				gd1, gv1, gd2, gv2 := findDirectionDualU8AVX2(img, img[8:], stride)
				if gd1 != wd1 || gv1 != wv1 || gd2 != wd2 || gv2 != wv2 {
					t.Fatalf("dual: stride=%d iter=%d got=(%d,%d,%d,%d) want=(%d,%d,%d,%d)",
						stride, iter, gd1, gv1, gd2, gv2, wd1, wv1, wd2, wv2)
				}
			}
		}
	}
}

func makeGuardedU8AVX2Dst(n int) []byte {
	dst := make([]byte, n)
	for i := range dst {
		dst[i] = byte((i*37 + 113) & 0xff)
	}
	return dst
}

func runCDEFU8UnitAVX2Differential(t *testing.T, rnd *cdefRandom, tc cdefU8UnitCase, caseIdx int) {
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
	if err := filterUnitBlocksU8WithBlockForTest(got[unitOrigin:], stride, input, origin, blocks, &gotDirs, &gotVars, u, filterBlockU8AVX2); err != nil {
		t.Fatalf("case %d: avx2 unit filter: %v", caseIdx, err)
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
