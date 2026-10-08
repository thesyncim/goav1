// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import "testing"

// The AVX2 Go SIMD direction kernels are called directly (not through the
// dispatch slot), so they are checked against the scalar reference regardless
// of CPU feature detection.

func TestFindDirectionAVX2MatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x41565832)
	for coeffShift := range 5 {
		max := uint16((1 << (8 + coeffShift)) - 1)
		for _, stride := range []int{8, 9, 13, 16, 31} {
			for iter := range 128 {
				img := make([]uint16, stride*8)
				for i := range img {
					img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
				gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
				if gotDir != wantDir || gotVar != wantVar {
					t.Fatalf("coeffShift=%d stride=%d iter=%d dir,var=%d,%d want %d,%d", coeffShift, stride, iter, gotDir, gotVar, wantDir, wantVar)
				}
			}
		}
	}
}

// TestFindDirectionAVX2Directional exercises every one of the 8 directions with
// deterministic gradient patterns so the argmax and tie-break paths are all hit.
func TestFindDirectionAVX2Directional(t *testing.T) {
	const stride = 8
	patterns := []func(row, col int) uint16{
		func(row, col int) uint16 { return uint16((row + col) * 16) },       // diag 0
		func(row, col int) uint16 { return uint16((2*row + col) * 8) },      // dir 1
		func(row, col int) uint16 { return uint16(row * 32) },               // horizontal 2
		func(row, col int) uint16 { return uint16((2*row - col + 8) * 8) },  // dir 3
		func(row, col int) uint16 { return uint16((row - col + 8) * 16) },   // anti-diag 4
		func(row, col int) uint16 { return uint16((row - 2*col + 16) * 8) }, // dir 5
		func(row, col int) uint16 { return uint16(col * 32) },               // vertical 6
		func(row, col int) uint16 { return uint16((-row + 2*col + 8) * 8) }, // dir 7
		func(row, col int) uint16 { return 128 },                            // constant -> dir 0, var 0
	}
	for coeffShift := range 5 {
		for pi, fill := range patterns {
			img := make([]uint16, stride*8)
			for row := range 8 {
				for col := range 8 {
					img[row*stride+col] = fill(row, col) << coeffShift
				}
			}
			wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
			gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("pattern=%d coeffShift=%d dir,var=%d,%d want %d,%d", pi, coeffShift, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionDualAVX2MatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x44554132)
	for coeffShift := range 5 {
		max := uint16((1 << (8 + coeffShift)) - 1)
		for _, stride := range []int{16, 19, 32} {
			for iter := range 96 {
				img := make([]uint16, stride*8)
				for i := range img {
					img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				wantDir1, wantVar1, wantDir2, wantVar2 := findDirectionDualScalar(img, img[8:], stride, coeffShift)
				gotDir1, gotVar1, gotDir2, gotVar2 := findDirectionDualSIMD(img, img[8:], stride, coeffShift)
				if gotDir1 != wantDir1 || gotVar1 != wantVar1 || gotDir2 != wantDir2 || gotVar2 != wantVar2 {
					t.Fatalf("coeffShift=%d stride=%d iter=%d dual=(%d,%d),(%d,%d) want (%d,%d),(%d,%d)",
						coeffShift, stride, iter,
						gotDir1, gotVar1, gotDir2, gotVar2,
						wantDir1, wantVar1, wantDir2, wantVar2)
				}
			}
		}
	}
}

func TestFindDirectionAVX2ZeroAlloc(t *testing.T) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 37) & 0xfff)
	}
	var dir int32
	var variance int32
	allocs := testing.AllocsPerRun(1000, func() {
		dir, variance = int32(0), int32(0)
		d, v := findDirectionSIMD(img, 8, 4)
		dir, variance = int32(d), v
	})
	_ = dir
	_ = variance
	if allocs != 0 {
		t.Fatalf("findDirectionSIMD allocated: %f", allocs)
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
			gotDir, gotVar := findDirectionU8SIMD(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			if stride >= 16 {
				wd1, wv1, wd2, wv2 := findDirectionDualU8Scalar(img, img[8:], stride)
				gd1, gv1, gd2, gv2 := findDirectionDualU8SIMD(img, img[8:], stride)
				if gd1 != wd1 || gv1 != wv1 || gd2 != wd2 || gv2 != wv2 {
					t.Fatalf("dual: stride=%d iter=%d got=(%d,%d,%d,%d) want=(%d,%d,%d,%d)",
						stride, iter, gd1, gv1, gd2, gv2, wd1, wv1, wd2, wv2)
				}
			}
		}
	}
}
