// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for upstream attribution.

//go:build amd64 && !purego

package cdef

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the AVX2-backed 8-bit direction search when the CPU advertises
// AVX2; otherwise the scalar uint8 reference stays in place.
func init() {
	_ = cpu.Detected
	if cpu.Detected.AVX2 {
		findDirectionU8Impl = findDirectionU8AVX2
		findDirectionDualU8Impl = findDirectionDualU8AVX2
	}
}

// findDirectionU8AVX2 widens the 8x8 block into a stack buffer and reuses the
// proven AVX2 uint16 kernel with coeffShift 0. The widen is 64 bytes; the
// direction outputs are bit-identical to the scalar uint8 search because the
// uint16 kernel computes the same partial sums on the same values.
func findDirectionU8AVX2(img []byte, stride int) (int, int32) {
	var block [64]uint16
	for row := 0; row < 8; row++ {
		src := img[row*stride : row*stride+8]
		dst := block[row*8 : row*8+8]
		for col, v := range src {
			dst[col] = uint16(v)
		}
	}
	var dir, variance int32
	cdefFindDirectionAVX2Asm(&block[0], 8, 0, &dir, &variance)
	return int(dir), variance
}

func findDirectionDualU8AVX2(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	dir1, var1 := findDirectionU8AVX2(img1, stride)
	dir2, var2 := findDirectionU8AVX2(img2, stride)
	return dir1, var1, dir2, var2
}
