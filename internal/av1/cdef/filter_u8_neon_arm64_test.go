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

func TestFilterBlockU8SecondaryByteNEONMatchesPureGo(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x38425954)
	origin := cdefBlockOrigin()
	for iter := range 512 {
		input := make([]byte, InputBufferSize)
		for i := range input {
			switch iter % 4 {
			case 0:
				input[i] = byte(rnd.generate(256))
			case 1:
				input[i] = byte(rnd.generate(8))
			case 2:
				input[i] = byte(248 + rnd.generate(8))
			default:
				input[i] = byte((i + iter) & 255)
			}
		}
		for _, strength := range []int{1, 2, 4} {
			for _, damping := range []int{3, 4, 5, 6} {
				want := make([]byte, 8*8)
				got := make([]byte, 8*8)
				filterBlockU8SecondaryBytePureGo(want, 8, 0, input, origin, strength, damping)
				ctx := filterBlockU8SecondaryByteNEONCtx{
					dst:         &got[0],
					input:       &input[origin],
					dstStr:      8,
					secStrength: int64(strength),
					secShift:    int64(constrainShift(strength, damping)),
				}
				cdefFilterBlock8SecondaryByteU8NEON(&ctx)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("iter=%d strength=%d damping=%d idx=%d got=%d want=%d", iter, strength, damping, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestFilterBlockU8GeneralByteNEONMatchesUint16(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x3842474e)
	origin := cdefBlockOrigin()
	strengths := []struct{ pri, sec int }{{1, 0}, {15, 0}, {0, 1}, {0, 4}, {1, 1}, {9, 2}, {15, 4}}
	for iter := range 128 {
		input8 := make([]byte, InputBufferSize)
		input16 := make([]uint16, InputBufferSize)
		for i := range input8 {
			var sample int
			switch iter % 4 {
			case 0:
				sample = int(rnd.generate(256))
			case 1:
				sample = int(rnd.generate(8))
			case 2:
				sample = 248 + int(rnd.generate(8))
			default:
				sample = (i + iter) & 255
			}
			input8[i] = byte(sample)
			input16[i] = uint16(sample)
		}
		for dir := range 8 {
			for _, strength := range strengths {
				for _, damping := range []int{3, 5, 6} {
					params := BlockFilterParams{
						PrimaryStrength:   uint8(strength.pri),
						SecondaryStrength: uint8(strength.sec),
						Direction:         uint8(dir),
						PrimaryDamping:    uint8(damping),
						SecondaryDamping:  uint8(damping),
						Width:             8,
						Height:            8,
					}
					want := make([]byte, 64)
					filterBlockU8PureGo(want, 8, 0, input16, origin, params)
					byteRef := make([]byte, 64)
					filterBlockU8BytePureGo(byteRef, 8, 0, input8, origin, params)
					got := make([]byte, 64)
					priTaps := cdefPrimaryTaps[strength.pri&1]
					ctx := filterBlockU8ByteNEONCtx{
						dst:         &got[0],
						input:       &input8[origin],
						dstStr:      8,
						height:      8,
						pri0:        int64(cdefDirections[dir+2][0]),
						pri1:        int64(cdefDirections[dir+2][1]),
						sec0:        int64(cdefDirections[dir+4][0]),
						sec1:        int64(cdefDirections[dir][0]),
						sec2:        int64(cdefDirections[dir+4][1]),
						sec3:        int64(cdefDirections[dir][1]),
						priTap0:     int64(priTaps[0]),
						priTap1:     int64(priTaps[1]),
						secTap0:     2,
						secTap1:     1,
						priStrength: int64(strength.pri),
						secStrength: int64(strength.sec),
						priShift:    int64(constrainShift(strength.pri, damping)),
						secShift:    int64(constrainShift(strength.sec, damping)),
					}
					switch {
					case strength.pri != 0 && strength.sec == 0:
						cdefFilterBlock8PrimaryByteU8NEON(&ctx)
					case strength.pri == 0:
						cdefFilterBlock8SecondaryGeneralByteU8NEON(&ctx)
					default:
						cdefFilterBlock8FusedByteU8NEON(&ctx)
					}
					for i := range want {
						if byteRef[i] != want[i] || got[i] != want[i] {
							t.Fatalf("iter=%d dir=%d pri=%d sec=%d damping=%d idx=%d got=%d byteRef=%d want=%d", iter, dir, strength.pri, strength.sec, damping, i, got[i], byteRef[i], want[i])
						}
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

func BenchmarkFindDirectionU8NEON(b *testing.B) {
	img := make([]byte, 8*8)
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x4e384e38)
	for i := range img {
		img[i] = byte(rnd.generate(256))
	}
	var dir, variance int32
	b.ReportAllocs()
	for b.Loop() {
		cdefFindDirectionU8NEONAsm(&img[0], 8, &dir, &variance)
	}
}

func benchmarkSecondaryU8NEON(b *testing.B, width, height int) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	const dstStride = 8
	const direction = 4
	const strength = 4
	const damping = 5
	ctx := filterBlockU8NEONCtx{
		dst:         &dst[0],
		input:       &input[cdefBlockOrigin()],
		dstStr:      dstStride,
		height:      int64(height),
		sec0:        int64(cdefDirections[direction+4][0]),
		sec1:        int64(cdefDirections[direction][0]),
		sec2:        int64(cdefDirections[direction+4][1]),
		sec3:        int64(cdefDirections[direction][1]),
		secTap0:     int64(cdefSecondaryTaps[0]),
		secTap1:     int64(cdefSecondaryTaps[1]),
		secStrength: strength,
		secShift:    int64(constrainShift(strength, damping)),
	}
	b.ReportAllocs()
	if width == 8 {
		for b.Loop() {
			cdefFilterBlock8SecondaryU8NEON(&ctx)
		}
		return
	}
	for b.Loop() {
		cdefFilterBlock4SecondaryU8NEON(&ctx)
	}
}

func BenchmarkCDEFSecondaryU8NEON8x8(b *testing.B) {
	benchmarkSecondaryU8NEON(b, 8, 8)
}

func BenchmarkCDEFSecondaryU8NEON4x8(b *testing.B) {
	benchmarkSecondaryU8NEON(b, 4, 8)
}

func BenchmarkCDEFSecondaryByteU8NEON8x8(b *testing.B) {
	input := make([]byte, InputBufferSize)
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x38425954)
	for i := range input {
		input[i] = byte(rnd.generate(256))
	}
	dst := make([]byte, 64)
	const strength = 4
	const damping = 5
	ctx := filterBlockU8SecondaryByteNEONCtx{
		dst:         &dst[0],
		input:       &input[cdefBlockOrigin()],
		dstStr:      8,
		secStrength: strength,
		secShift:    int64(constrainShift(strength, damping)),
	}
	b.ReportAllocs()
	for b.Loop() {
		cdefFilterBlock8SecondaryByteU8NEON(&ctx)
	}
}

func benchmarkGeneralByteU8NEON(b *testing.B, primaryStrength, secondaryStrength int) {
	input := make([]byte, InputBufferSize)
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x3842474e)
	for i := range input {
		input[i] = byte(rnd.generate(256))
	}
	dst := make([]byte, 64)
	const direction = 4
	const damping = 5
	priTaps := cdefPrimaryTaps[primaryStrength&1]
	ctx := filterBlockU8ByteNEONCtx{
		dst:         &dst[0],
		input:       &input[cdefBlockOrigin()],
		dstStr:      8,
		height:      8,
		pri0:        int64(cdefDirections[direction+2][0]),
		pri1:        int64(cdefDirections[direction+2][1]),
		sec0:        int64(cdefDirections[direction+4][0]),
		sec1:        int64(cdefDirections[direction][0]),
		sec2:        int64(cdefDirections[direction+4][1]),
		sec3:        int64(cdefDirections[direction][1]),
		priTap0:     int64(priTaps[0]),
		priTap1:     int64(priTaps[1]),
		priStrength: int64(primaryStrength),
		secStrength: int64(secondaryStrength),
		priShift:    int64(constrainShift(primaryStrength, damping)),
		secShift:    int64(constrainShift(secondaryStrength, damping)),
	}
	b.ReportAllocs()
	for b.Loop() {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryByteU8NEON(&ctx)
		case primaryStrength == 0:
			cdefFilterBlock8SecondaryGeneralByteU8NEON(&ctx)
		default:
			cdefFilterBlock8FusedByteU8NEON(&ctx)
		}
	}
}

func BenchmarkCDEFPrimaryByteU8NEON8x8(b *testing.B) {
	benchmarkGeneralByteU8NEON(b, 15, 0)
}

func BenchmarkCDEFSecondaryGeneralByteU8NEON8x8(b *testing.B) {
	benchmarkGeneralByteU8NEON(b, 0, 4)
}

func BenchmarkCDEFFusedByteU8NEON8x8(b *testing.B) {
	benchmarkGeneralByteU8NEON(b, 15, 4)
}

func benchmarkPrimaryU8NEON(b *testing.B, width, height int) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	const dstStride = 8
	const direction = 4
	const strength = 15
	const damping = 5
	priTaps := cdefPrimaryTaps[strength&1]
	ctx := filterBlockU8NEONCtx{
		dst:         &dst[0],
		input:       &input[cdefBlockOrigin()],
		dstStr:      dstStride,
		height:      int64(height),
		pri0:        int64(cdefDirections[direction+2][0]),
		pri1:        int64(cdefDirections[direction+2][1]),
		priTap0:     int64(priTaps[0]),
		priTap1:     int64(priTaps[1]),
		priStrength: strength,
		priShift:    int64(constrainShift(strength, damping)),
	}
	b.ReportAllocs()
	if width == 8 {
		for b.Loop() {
			cdefFilterBlock8PrimaryU8NEON(&ctx)
		}
		return
	}
	for b.Loop() {
		cdefFilterBlock4PrimaryU8NEON(&ctx)
	}
}

func BenchmarkCDEFPrimaryU8NEON8x8(b *testing.B) {
	benchmarkPrimaryU8NEON(b, 8, 8)
}

func BenchmarkCDEFPrimaryU8NEON4x8(b *testing.B) {
	benchmarkPrimaryU8NEON(b, 4, 8)
}

func benchmarkFusedU8NEON(b *testing.B, width, height int) {
	input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed), 8, 0, 0)
	dst := make([]byte, 64)
	const dstStride = 8
	const direction = 4
	const primaryStrength = 15
	const secondaryStrength = 4
	const damping = 5
	priTaps := cdefPrimaryTaps[primaryStrength&1]
	ctx := filterBlockU8NEONCtx{
		dst:         &dst[0],
		input:       &input[cdefBlockOrigin()],
		dstStr:      dstStride,
		height:      int64(height),
		pri0:        int64(cdefDirections[direction+2][0]),
		pri1:        int64(cdefDirections[direction+2][1]),
		sec0:        int64(cdefDirections[direction+4][0]),
		sec1:        int64(cdefDirections[direction][0]),
		sec2:        int64(cdefDirections[direction+4][1]),
		sec3:        int64(cdefDirections[direction][1]),
		priTap0:     int64(priTaps[0]),
		priTap1:     int64(priTaps[1]),
		secTap0:     int64(cdefSecondaryTaps[0]),
		secTap1:     int64(cdefSecondaryTaps[1]),
		priStrength: primaryStrength,
		secStrength: secondaryStrength,
		priShift:    int64(constrainShift(primaryStrength, damping)),
		secShift:    int64(constrainShift(secondaryStrength, damping)),
	}
	b.ReportAllocs()
	if width == 8 {
		for b.Loop() {
			cdefFilterBlock8U8NEON(&ctx)
		}
		return
	}
	for b.Loop() {
		cdefFilterBlock4U8NEON(&ctx)
	}
}

func BenchmarkCDEFFusedU8NEON8x8(b *testing.B) {
	benchmarkFusedU8NEON(b, 8, 8)
}

func BenchmarkCDEFFusedU8NEON4x8(b *testing.B) {
	benchmarkFusedU8NEON(b, 4, 8)
}
