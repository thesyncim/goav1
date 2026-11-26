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
	var c0, c1, c2, c3, c4, c5, c6, c7 int32
	for i := 0; i < 8; i += 2 {
		r0 := img[i*stride : i*stride+8]
		r1 := img[(i+1)*stride : (i+1)*stride+8]
		a0, a1, a2, a3 := int32(r0[0])-128, int32(r0[1])-128, int32(r0[2])-128, int32(r0[3])-128
		a4, a5, a6, a7 := int32(r0[4])-128, int32(r0[5])-128, int32(r0[6])-128, int32(r0[7])-128
		b0, b1, b2, b3 := int32(r1[0])-128, int32(r1[1])-128, int32(r1[2])-128, int32(r1[3])-128
		b4, b5, b6, b7 := int32(r1[4])-128, int32(r1[5])-128, int32(r1[6])-128, int32(r1[7])-128

		// Direction 2: horizontal rows.
		ap01, ap23, ap45, ap67 := a0+a1, a2+a3, a4+a5, a6+a7
		bp01, bp23, bp45, bp67 := b0+b1, b2+b3, b4+b5, b6+b7
		partial[2][i] = ap01 + ap23 + ap45 + ap67
		partial[2][i+1] = bp01 + bp23 + bp45 + bp67

		// Direction 1 & 3: 2:1 shallow slopes share horizontal pairs.
		partial[1][i] += ap01
		partial[1][i+1] += ap23 + bp01
		partial[1][i+2] += ap45 + bp23
		partial[1][i+3] += ap67 + bp45
		partial[1][i+4] += bp67

		partial[3][3+i] += ap01
		partial[3][2+i] += ap23
		partial[3][1+i] += ap45
		partial[3][i] += ap67
		partial[3][4+i] += bp01
		partial[3][3+i] += bp23
		partial[3][2+i] += bp45
		partial[3][1+i] += bp67

		// Direction 0: 45-degree diagonals.
		partial[0][i] += a0
		partial[0][i+1] += a1 + b0
		partial[0][i+2] += a2 + b1
		partial[0][i+3] += a3 + b2
		partial[0][i+4] += a4 + b3
		partial[0][i+5] += a5 + b4
		partial[0][i+6] += a6 + b5
		partial[0][i+7] += a7 + b6
		partial[0][i+8] += b7

		// Direction 4: 135-degree diagonals.
		partial[4][7+i] += a0
		partial[4][6+i] += a1
		partial[4][5+i] += a2
		partial[4][4+i] += a3
		partial[4][3+i] += a4
		partial[4][2+i] += a5
		partial[4][1+i] += a6
		partial[4][i] += a7
		partial[4][8+i] += b0
		partial[4][7+i] += b1
		partial[4][6+i] += b2
		partial[4][5+i] += b3
		partial[4][4+i] += b4
		partial[4][3+i] += b5
		partial[4][2+i] += b6
		partial[4][1+i] += b7

		// Vertical pairs shared by directions 5, 6, 7.
		v0, v1, v2, v3 := a0+b0, a1+b1, a2+b2, a3+b3
		v4, v5, v6, v7 := a4+b4, a5+b5, a6+b6, a7+b7
		c0 += v0
		c1 += v1
		c2 += v2
		c3 += v3
		c4 += v4
		c5 += v5
		c6 += v6
		c7 += v7

		ih := i >> 1
		p5 := partial[5][3-ih:]
		p5[0] += v0
		p5[1] += v1
		p5[2] += v2
		p5[3] += v3
		p5[4] += v4
		p5[5] += v5
		p5[6] += v6
		p5[7] += v7

		p7 := partial[7][ih:]
		p7[0] += v0
		p7[1] += v1
		p7[2] += v2
		p7[3] += v3
		p7[4] += v4
		p7[5] += v5
		p7[6] += v6
		p7[7] += v7
	}
	partial[6][0] = c0
	partial[6][1] = c1
	partial[6][2] = c2
	partial[6][3] = c3
	partial[6][4] = c4
	partial[6][5] = c5
	partial[6][6] = c6
	partial[6][7] = c7
	return finishDirection(&partial)
}

func findDirectionDualU8Scalar(img1 []byte, img2 []byte, stride int) (int, int32, int, int32) {
	dir1, var1 := findDirectionU8Scalar(img1, stride)
	dir2, var2 := findDirectionU8Scalar(img2, stride)
	return dir1, var1, dir2, var2
}
