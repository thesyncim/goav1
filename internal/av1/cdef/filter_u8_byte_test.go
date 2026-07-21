// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

import "testing"

func TestFilterFrameBlocksU8ByteTrustedMatchesWidenedInput(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x4259_5445)
	geometries := []struct {
		plane      Plane
		xDec, yDec int
	}{
		{PlaneY, 0, 0},
		{PlaneU, 1, 1},
		{PlaneV, 1, 1},
		{PlaneU, 1, 0},
		{PlaneU, 0, 0},
	}
	strengths := []struct{ primary, secondary int }{{9, 0}, {0, 2}, {15, 4}}
	for _, geom := range geometries {
		unitW := BlockSize >> geom.xDec
		unitH := BlockSize >> geom.yDec
		blockW := 8 >> geom.xDec
		blockH := 8 >> geom.yDec
		stride := unitW + 13
		blocks := make([]BlockPosition, 0, NBlocks*NBlocks)
		for by := 0; by < unitH/blockH; by++ {
			for bx := 0; bx < unitW/blockW; bx++ {
				blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
			}
		}
		input8 := make([]byte, InputBufferSize)
		input16 := make([]uint16, InputBufferSize)
		for i := range input8 {
			input8[i] = byte(rnd.generate(256))
			input16[i] = uint16(input8[i])
		}
		frame := make([]byte, stride*unitH)
		for i := range frame {
			frame[i] = byte(rnd.generate(256))
		}
		for _, strength := range strengths {
			want := append([]byte(nil), frame...)
			got := append([]byte(nil), frame...)
			var wantDirs, gotDirs DirectionGrid
			var wantVars, gotVars VarianceGrid
			for by := range NBlocks {
				for bx := range NBlocks {
					dir := uint8(rnd.generate(8))
					variance := int32(rnd.generate(1 << 20))
					wantDirs[by][bx], gotDirs[by][bx] = dir, dir
					wantVars[by][bx], gotVars[by][bx] = variance, variance
				}
			}
			params := FrameFilterParams{
				Plane:             geom.plane,
				XDec:              uint8(geom.xDec),
				YDec:              uint8(geom.yDec),
				Level:             uint8(strength.primary),
				SecondaryStrength: uint8(strength.secondary),
				Damping:           5,
			}
			if err := FilterFrameBlocksU8Trusted(want, stride, input16, cdefBlockOrigin(), blocks, &wantDirs, &wantVars, params); err != nil {
				t.Fatalf("plane=%d xdec=%d ydec=%d widened: %v", geom.plane, geom.xDec, geom.yDec, err)
			}
			if err := FilterFrameBlocksU8ByteTrusted(got, stride, input8, cdefBlockOrigin(), blocks, &gotDirs, &gotVars, params); err != nil {
				t.Fatalf("plane=%d xdec=%d ydec=%d byte: %v", geom.plane, geom.xDec, geom.yDec, err)
			}
			if wantDirs != gotDirs || wantVars != gotVars {
				t.Fatalf("plane=%d xdec=%d ydec=%d direction results differ", geom.plane, geom.xDec, geom.yDec)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("plane=%d xdec=%d ydec=%d primary=%d secondary=%d idx=%d got=%d want=%d", geom.plane, geom.xDec, geom.yDec, strength.primary, strength.secondary, i, got[i], want[i])
				}
			}
		}
	}
}

func TestFilterFrameBlocksU8ByteTrustedIsZeroAlloc(t *testing.T) {
	const stride = 77
	frame := make([]byte, stride*64)
	input := make([]byte, InputBufferSize)
	blocks := make([]BlockPosition, 0, NBlocks*NBlocks)
	for by := range NBlocks {
		for bx := range NBlocks {
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	var directions DirectionGrid
	var variances VarianceGrid
	params := FrameFilterParams{Plane: PlaneU, XDec: 1, YDec: 1, Level: 9, SecondaryStrength: 2, Damping: 5}
	allocs := testing.AllocsPerRun(32, func() {
		if err := FilterFrameBlocksU8ByteTrusted(frame, stride, input, cdefBlockOrigin(), blocks, &directions, &variances, params); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FilterFrameBlocksU8ByteTrusted allocates: %v allocs/op", allocs)
	}
}
