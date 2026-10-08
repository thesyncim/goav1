// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cdef

import "testing"

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
