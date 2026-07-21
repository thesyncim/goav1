// SPDX-License-Identifier: BSD-2-Clause

//go:build arm64 && !purego

package cdef

import "testing"

func TestFilterBlockU8SecondaryByteNEONMatchesPureGo(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x38425954)
	origin := cdefBlockOrigin()
	for iter := range 128 {
		input := make([]byte, InputBufferSize)
		for i := range input {
			input[i] = byte(rnd.generate(256))
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
	for iter := range 64 {
		input8 := make([]byte, InputBufferSize)
		input16 := make([]uint16, InputBufferSize)
		for i := range input8 {
			sample := int(rnd.generate(256))
			input8[i] = byte(sample)
			input16[i] = uint16(sample)
		}
		for _, width := range []int{8, 4} {
			for dir := range 8 {
				for _, strength := range strengths {
					for _, damping := range []int{3, 5, 6} {
						params := BlockFilterParams{
							PrimaryStrength:   uint8(strength.pri),
							SecondaryStrength: uint8(strength.sec),
							Direction:         uint8(dir),
							PrimaryDamping:    uint8(damping),
							SecondaryDamping:  uint8(damping),
							Width:             uint8(width),
							Height:            uint8(width),
						}
						want := make([]byte, width*width)
						filterBlockU8PureGo(want, width, 0, input16, origin, params)
						got := make([]byte, width*width)
						priTaps := cdefPrimaryTaps[strength.pri&1]
						ctx := filterBlockU8ByteNEONCtx{
							dst:         &got[0],
							input:       &input8[origin],
							dstStr:      int64(width),
							height:      int64(width),
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
						if width == 8 {
							switch {
							case strength.pri != 0 && strength.sec == 0:
								cdefFilterBlock8PrimaryByteU8NEON(&ctx)
							case strength.pri == 0:
								cdefFilterBlock8SecondaryGeneralByteU8NEON(&ctx)
							default:
								cdefFilterBlock8FusedByteU8NEON(&ctx)
							}
						} else {
							switch {
							case strength.pri != 0 && strength.sec == 0:
								cdefFilterBlock4PrimaryByteU8NEON(&ctx)
							case strength.pri == 0:
								cdefFilterBlock4SecondaryByteU8NEON(&ctx)
							default:
								cdefFilterBlock4FusedByteU8NEON(&ctx)
							}
						}
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("iter=%d width=%d dir=%d pri=%d sec=%d damping=%d idx=%d got=%d want=%d", iter, width, dir, strength.pri, strength.sec, damping, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}
