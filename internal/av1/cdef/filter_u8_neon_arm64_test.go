// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

import "testing"

// TestFilterBlockU8NEONMatchesPureGo pins each of the six strength-split
// 8-bit-dst NEON kernels (fused / primary-only / secondary-only at widths 8
// and 4) against the pure-Go reference across directions, dampings, shapes,
// and halo-sentinel patterns.
func TestFilterBlockU8NEONMatchesPureGo(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {4, 8}, {4, 4},
	}
	for boundary := 0; boundary < 16; boundary++ {
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x38553845), 8, boundary, boundary+1)
		for _, shape := range shapes {
			for dir := 0; dir <= 7; dir++ {
				for _, pri := range cdefPrimaryStrengthCorpus(0) {
					for _, sec := range cdefSecondaryStrengthCorpus(0) {
						if pri == 0 && sec == 0 {
							continue
						}
						for _, damping := range []int{3, 4, 6} {
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
							want := make([]byte, dstStride*8)
							got := make([]byte, dstStride*8)
							filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
							filterBlockU8NEON(got, dstStride, 0, input, origin, params)
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

// TestFilterUnitBlocksU8NEONMatchesPureGo pins the NEON unit-level loop
// (strength adjust, zero-strength skip, kernel dispatch) against the pure-Go
// unit loop on whole synthetic units.
func TestFilterUnitBlocksU8NEONMatchesPureGo(t *testing.T) {
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
				if err := filterUnitBlocksU8NEON(got, stride, input, cdefBlockOrigin(), blocks, &dirs, &vars, u); err != nil {
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

// TestFindDirectionU8NEONMatchesScalar pins the 8-bit NEON direction kernel
// against the scalar uint8 reference.
func TestFindDirectionU8NEONMatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x4e384e38)
	for _, stride := range []int{8, 17, 160, 640} {
		for iter := range 256 {
			img := make([]byte, stride*8+8)
			for i := range img {
				switch iter % 3 {
				case 0:
					img[i] = byte(rnd.generate(256))
				case 1:
					img[i] = byte(rnd.generate(6))
				default:
					img[i] = byte(250 + rnd.generate(6))
				}
			}
			wantDir, wantVar := findDirectionU8Scalar(img, stride)
			gotDir, gotVar := findDirectionU8NEON(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}
