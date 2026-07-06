// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cdef

import "testing"

// The AVX2-backed 8-bit wrappers are validated by DIRECT calls (not through
// cpu.Detected dispatch) so the differential also runs under Rosetta, where
// CPUID reports AVX2 false but the instructions execute fine.

// TestFilterBlockU8AVX2MatchesPureGo pins the AVX2-backed 8-bit block filter
// wrapper against the pure-Go reference.
func TestFilterBlockU8AVX2MatchesPureGo(t *testing.T) {
	const dstStride = 16
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {4, 8}, {4, 4},
	}
	for boundary := 0; boundary < 16; boundary++ {
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x38415658), 8, boundary, boundary+2)
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
							want := make([]byte, dstStride*8)
							got := make([]byte, dstStride*8)
							filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
							filterBlockU8AVX2(got, dstStride, 0, input, origin, params)
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
