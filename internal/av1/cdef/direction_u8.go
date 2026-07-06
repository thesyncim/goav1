// Ported from libaom: av1/common/cdef_block.c
// with the 8bpc pixel split taken from dav1d: src/cdef_tmpl.c
// (cdef_find_dir_c, pixel=uint8_t).
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

// 8-bit-pixel direction search. dav1d's 8bpc cdef_find_dir reads the uint8
// frame directly (bitdepth_min_8 == 0, so px = img[x] - 128); goav1's uint16
// search over the widened CDEF buffer computes the identical partial sums, so
// the two are bit-identical on the same pixels. The uint8 entry lets the
// 8-bit frame walk search directions without staging the unit first.

type findDirectionU8Func func([]byte, int) (int, int32)
type findDirectionDualU8Func func([]byte, []byte, int) (int, int32, int, int32)

var (
	findDirectionU8Impl     findDirectionU8Func     = findDirectionU8Scalar
	findDirectionDualU8Impl findDirectionDualU8Func = findDirectionDualU8Scalar
)

func findDirectionU8Unchecked(img []byte, stride int) (int, int32) {
	return findDirectionU8Impl(img, stride)
}

func findDirectionDualU8Unchecked(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	return findDirectionDualU8Impl(img1, img2, stride)
}

// findDirectionU8Scalar mirrors findDirection8Unchecked (direction.go) with
// uint8 loads; the partial-sum layout and finishDirection reduction are
// shared, so outputs match the uint16 search exactly.
func findDirectionU8Scalar(img []byte, stride int) (int, int32) {
	var partial [8][15]int32
	for i := range 8 {
		rowBase := i * stride
		iHalf := i >> 1
		for j := range 8 {
			x := int32(img[rowBase+j]) - 128
			partial[0][i+j] += x
			partial[1][i+(j>>1)] += x
			partial[2][i] += x
			partial[3][3+i-(j>>1)] += x
			partial[4][7+i-j] += x
			partial[5][3-iHalf+j] += x
			partial[6][j] += x
			partial[7][iHalf+j] += x
		}
	}
	return finishDirection(&partial)
}

func findDirectionDualU8Scalar(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	dir1, var1 := findDirectionU8Scalar(img1, stride)
	dir2, var2 := findDirectionU8Scalar(img2, stride)
	return dir1, var1, dir2, var2
}
