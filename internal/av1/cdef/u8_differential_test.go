// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

import "testing"

// Differential proof for the 8-bit-pixel CDEF kernel family (filter_u8.go,
// direction_u8.go): every entry must be bit-identical to the uint16 kernels
// fed the same widened input + 0x4000 sentinels, followed by a uint8
// narrowing store. The tests run the resolved dispatch slots, so on arm64
// they cover the NEON kernels and on native amd64 the AVX2-backed wrappers;
// the pure-Go references are asserted explicitly as well.

// TestFilterBlockU8MatchesU16Narrow proves the per-block u8 kernel identity
// across directions, strengths, dampings, block shapes, and all sixteen
// halo-sentinel patterns, and that the uint16 result always fits a byte under
// the 8-bit contract (real interior pixels).
func TestFilterBlockU8MatchesU16Narrow(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {4, 8}, {4, 4},
	}
	for boundary := 0; boundary < 16; boundary++ {
		for iter := range 3 {
			input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^uint32(iter*977)), 8, boundary, iter)
			for _, shape := range shapes {
				for dir := 0; dir <= 7; dir++ {
					for _, pri := range cdefPrimaryStrengthCorpus(0) {
						for _, sec := range cdefSecondaryStrengthCorpus(0) {
							if pri == 0 && sec == 0 {
								continue
							}
							for _, damping := range []int{3, 5, 6} {
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
								want16 := make([]uint16, dstStride*8)
								filterBlockPureGo(want16, dstStride, 0, input, origin, params)

								gotRef := make([]byte, dstStride*8)
								filterBlockU8PureGo(gotRef, dstStride, 0, input, origin, params)
								gotImpl := make([]byte, dstStride*8)
								filterBlockU8Impl(gotImpl, dstStride, 0, input, origin, params)

								for row := 0; row < shape.height; row++ {
									for col := 0; col < shape.width; col++ {
										i := row*dstStride + col
										if want16[i] > 255 {
											t.Fatalf("u16 result exceeds byte range: boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d val=%d",
												boundary, shape.width, shape.height, dir, pri, sec, damping, i, want16[i])
										}
										if gotRef[i] != byte(want16[i]) {
											t.Fatalf("pure-Go u8 mismatch: boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
												boundary, shape.width, shape.height, dir, pri, sec, damping, i, gotRef[i], want16[i])
										}
										if gotImpl[i] != byte(want16[i]) {
											t.Fatalf("dispatched u8 mismatch: boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
												boundary, shape.width, shape.height, dir, pri, sec, damping, i, gotImpl[i], want16[i])
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestFindDirectionU8MatchesU16 proves the 8-bit direction search (scalar
// reference and the resolved dispatch slot, single and dual) returns exactly
// the uint16 search's direction and variance on widened pixels.
func TestFindDirectionU8MatchesU16(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x5538_4449)
	for _, stride := range []int{8, 11, 64, 640} {
		for iter := range 64 {
			img8 := make([]byte, stride*8+8)
			img16 := make([]uint16, len(img8))
			for i := range img8 {
				var v uint32
				switch iter % 4 {
				case 0:
					v = rnd.generate(256)
				case 1:
					v = rnd.generate(8) // flat block, exercises variance 0
				case 2:
					v = 255 - rnd.generate(4)
				default:
					v = rnd.generate(2) * 255
				}
				img8[i] = byte(v)
				img16[i] = uint16(v)
			}
			wantDir, wantVar := findDirectionScalar(img16, stride, 0)
			gotDir, gotVar := findDirectionU8Scalar(img8, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("scalar u8 direction mismatch: stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			gotDir, gotVar = findDirectionU8Unchecked(img8, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("dispatched u8 direction mismatch: stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			if stride >= 16 {
				wd1, wv1, wd2, wv2 := findDirectionDualUnchecked(img16, img16[8:], stride, 0)
				gd1, gv1, gd2, gv2 := findDirectionDualU8Unchecked(img8, img8[8:], stride)
				if gd1 != wd1 || gv1 != wv1 || gd2 != wd2 || gv2 != wv2 {
					t.Fatalf("dual u8 direction mismatch: stride=%d iter=%d got=(%d,%d,%d,%d) want=(%d,%d,%d,%d)",
						stride, iter, gd1, gv1, gd2, gv2, wd1, wv1, wd2, wv2)
				}
			}
		}
	}
}

// cdefU8UnitCase describes one synthetic filter unit for the frame-level
// differential.
type cdefU8UnitCase struct {
	plane    Plane
	xDec     int
	yDec     int
	unitW    int
	unitH    int
	level    int
	sec      int
	damping  int
	skipMask uint64 // bit per block position: 1 = drop from the block list
	halo     uint8  // applyCDEFBoundary-style sentinel pattern
}

// TestFilterFrameBlocksU8MatchesU16 is the unit-level differential: for a
// matrix of planes/subsampling/unit sizes/strengths/skip patterns/sentinel
// halos it runs the trusted uint16 frame filter into a prefilled uint16 unit
// and the trusted uint8 frame filter in place on a byte plane, then requires
// identical bytes everywhere (filtered blocks, skipped blocks, and pixels
// outside the unit) plus identical direction/variance grids.
func TestFilterFrameBlocksU8MatchesU16(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x5538_4642)
	cases := make([]cdefU8UnitCase, 0, 256)
	for _, geom := range []struct {
		plane      Plane
		xDec, yDec int
	}{
		{PlaneY, 0, 0},
		{PlaneU, 1, 1},
		{PlaneV, 1, 1},
		{PlaneU, 1, 0}, // 4:2:2 (chroma direction conversion path)
		{PlaneU, 0, 0}, // 4:4:4
	} {
		unitWFull := BlockSize >> geom.xDec
		unitHFull := BlockSize >> geom.yDec
		blockW := 8 >> geom.xDec
		blockH := 8 >> geom.yDec
		for _, size := range []struct{ w, h int }{
			{unitWFull, unitHFull},                   // full unit
			{unitWFull - blockW, unitHFull - blockH}, // partial right/bottom
			{blockW, blockH},                         // single block
			{unitWFull, blockH * 2},                  // short unit
		} {
			for _, str := range []struct{ level, sec int }{
				{0, 0}, // direction-only luma pass shape
				{9, 0}, {0, 2}, {15, 4}, {1, 1}, {4, 2},
			} {
				for _, damping := range []int{3, 6} {
					cases = append(cases, cdefU8UnitCase{
						plane: geom.plane, xDec: geom.xDec, yDec: geom.yDec,
						unitW: size.w, unitH: size.h,
						level: str.level, sec: str.sec, damping: damping,
						skipMask: uint64(rnd.generate(1<<16)) | uint64(rnd.generate(1<<16))<<16,
						halo:     uint8(rnd.generate(16)),
					})
				}
			}
		}
	}
	for i, tc := range cases {
		runCDEFU8UnitDifferential(t, rnd, tc, i)
	}
}

func runCDEFU8UnitDifferential(t *testing.T, rnd *cdefRandom, tc cdefU8UnitCase, caseIdx int) {
	t.Helper()
	blockW := 8 >> tc.xDec
	blockH := 8 >> tc.yDec
	blockCols := (tc.unitW + blockW - 1) / blockW
	blockRows := (tc.unitH + blockH - 1) / blockH

	// Frame plane: the unit sits at an offset inside a larger byte plane so
	// out-of-unit writes would be caught.
	const margin = 24
	stride := tc.unitW + 2*margin + 3
	height := tc.unitH + 2*margin
	frame := make([]byte, stride*height)
	for i := range frame {
		frame[i] = byte(rnd.generate(256))
	}
	unitOrigin := margin*stride + margin

	// CDEF input buffer: interior = frame unit pixels, halo = random real
	// neighbours, then the sentinel pattern (frame edges) on top.
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(256))
	}
	origin := VerticalBorder*BStride + HorizontalBorder
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

	params := FrameFilterParams{
		XDec:              uint8(tc.xDec),
		YDec:              uint8(tc.yDec),
		Plane:             tc.plane,
		Level:             uint8(tc.level),
		SecondaryStrength: uint8(tc.sec),
		Damping:           uint8(tc.damping),
		CoeffShift:        0,
	}

	// uint16 path: prefill the unit dst with the source pixels (the decoder's
	// skip prefill), filter, then narrow.
	var wantDirs, gotDirs DirectionGrid
	var wantVars, gotVars VarianceGrid
	if tc.plane != PlaneY {
		// Chroma reuses luma directions; seed both grids identically.
		for by := range NBlocks {
			for bx := range NBlocks {
				d := uint8(rnd.generate(8))
				wantDirs[by][bx] = d
				gotDirs[by][bx] = d
				v := int32(rnd.generate(1 << 20))
				wantVars[by][bx] = v
				gotVars[by][bx] = v
			}
		}
	}
	want16 := make([]uint16, stride*height)
	for i, v := range frame {
		want16[i] = uint16(v)
	}
	if err := FilterFrameBlocksTrusted(want16[unitOrigin:], stride, input, origin, blocks, &wantDirs, &wantVars, params); err != nil {
		t.Fatalf("case %d: u16 filter: %v", caseIdx, err)
	}

	// uint8 path: in place on a copy of the frame.
	got := make([]byte, len(frame))
	copy(got, frame)
	if err := FilterFrameBlocksU8Trusted(got[unitOrigin:], stride, input, origin, blocks, &gotDirs, &gotVars, params); err != nil {
		t.Fatalf("case %d: u8 filter: %v", caseIdx, err)
	}

	if gotDirs != wantDirs {
		t.Fatalf("case %d (%+v): direction grids differ", caseIdx, tc)
	}
	if gotVars != wantVars {
		t.Fatalf("case %d (%+v): variance grids differ", caseIdx, tc)
	}
	for i := range got {
		if want16[i] > 255 {
			t.Fatalf("case %d (%+v): u16 result exceeds byte range at %d: %d", caseIdx, tc, i, want16[i])
		}
		if got[i] != byte(want16[i]) {
			row := i / stride
			col := i % stride
			t.Fatalf("case %d (%+v): byte mismatch at row=%d col=%d got=%d want=%d", caseIdx, tc, row, col, got[i], want16[i])
		}
	}
}

// applyCDEFU8UnitHalo overlays the VeryLarge sentinel on the unit halo in the
// same four-edge pattern as applyCDEFBoundary, but for an arbitrary unit
// geometry (the decoder writes sentinels for missing frame edges).
func applyCDEFU8UnitHalo(input []uint16, unitW int, unitH int, halo uint8) {
	fillH := unitH + 2*VerticalBorder
	if halo&1 != 0 { // left
		for row := 0; row < fillH; row++ {
			for col := 0; col < HorizontalBorder; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if halo&2 != 0 { // right
		for row := 0; row < fillH; row++ {
			for col := HorizontalBorder + unitW; col < BStride; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if halo&4 != 0 { // top
		for row := 0; row < VerticalBorder; row++ {
			for col := 0; col < BStride; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
	if halo&8 != 0 { // bottom
		for row := VerticalBorder + unitH; row < fillH; row++ {
			for col := 0; col < BStride; col++ {
				input[row*BStride+col] = VeryLarge
			}
		}
	}
}

// TestFilterFrameBlocksU8RejectsHighBitDepth pins the 8-bit-only contract.
func TestFilterFrameBlocksU8RejectsHighBitDepth(t *testing.T) {
	var dirs DirectionGrid
	var vars VarianceGrid
	params := FrameFilterParams{Plane: PlaneY, Level: 4, Damping: 4, CoeffShift: 2}
	err := FilterFrameBlocksU8Trusted(make([]byte, 64*64), 64, make([]uint16, InputBufferSize), cdefBlockOrigin(), []BlockPosition{{}}, &dirs, &vars, params)
	if err == nil {
		t.Fatal("expected CoeffShift != 0 to be rejected")
	}
}

// TestFilterFrameBlocksU8IsZeroAlloc protects the hot-path contract.
func TestFilterFrameBlocksU8IsZeroAlloc(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x414c_4c38)
	stride := 96
	frame := make([]byte, stride*96)
	for i := range frame {
		frame[i] = byte(rnd.generate(256))
	}
	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(256))
	}
	blocks := make([]BlockPosition, 0, 64)
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	var dirs DirectionGrid
	var vars VarianceGrid
	params := FrameFilterParams{Plane: PlaneY, Level: 9, SecondaryStrength: 2, Damping: 5}
	allocs := testing.AllocsPerRun(32, func() {
		if err := FilterFrameBlocksU8Trusted(frame, stride, input, cdefBlockOrigin(), blocks, &dirs, &vars, params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FilterFrameBlocksU8Trusted allocates: %v allocs/op", allocs)
	}
}
