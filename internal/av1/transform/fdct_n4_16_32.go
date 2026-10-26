// Ported from SVT-AV1 v4.1.0:
//   Source/Lib/Codec/transforms.c (svt_av1_fdct16_new_N4,
//   svt_av1_fdct32_new_N4, av1_tranform_two_d_core_N4_c)
//   Source/Lib/ASM_NEON/highbd_fwd_txfm_neon.c (svt_av1_fwd_txfm2d_16x16_N4_neon,
//   svt_av1_fwd_txfm2d_32x32_N4_neon)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

// SVT-AV1's N4_SHAPE 16x16 and 32x32 DCT_DCT: the column pass keeps only
// outputs 0..n/4-1 of each column's DCT and the row pass only outputs
// 0..n/4-1 of those rows, so only the n/4 x n/4 lowest-frequency
// coefficients are computed. The pruned networks drop the butterflies that
// feed the other outputs only, so the kept coefficients equal the full
// transform's.
//
// The coefficients are written in the decoder layout coeff[col*n+row], and
// only the n/4 x n/4 low block is written: the caller zeroes the rest (the
// encoder's N4 path clears it before quantizing).

// ForwardDCT16x16N48BitTrusted writes the 4x4 low coefficients of the 16x16
// DCT_DCT of an 8-bit residual block (stride 16).
func ForwardDCT16x16N48BitTrusted(coeff *[256]int32, residual *[256]int16) {
	var buf [4 * 16]int32
	forwardDCT16x16N4Col(&buf, residual)
	fwd16x16N4Rows(coeff, &buf, 0xf)
}

// ForwardDCT32x32N48BitTrusted writes the 8x8 low coefficients of the 32x32
// DCT_DCT of an 8-bit residual block (stride 32).
func ForwardDCT32x32N48BitTrusted(coeff *[1024]int32, residual *[1024]int16) {
	var buf [8 * 32]int32
	forwardDCT32x32N4Col(&buf, residual)
	fwd32x32N4Rows(coeff, &buf, 0xff)
}

// ForwardDCT16x16N4RowSkip8BitTrusted is ForwardDCT16x16RowSkip8BitTrusted
// restricted to N4_SHAPE: the column-pass rows 0..3 (row group 0) are bounded
// as the row-skipping kernel bounds them, and when the group is live the 4x4
// low coefficients are written and 1 returned. A zero result means every N4
// coefficient quantizes to zero and coeff was not written.
func ForwardDCT16x16N4RowSkip8BitTrusted(coeff *[256]int32, residual *[256]int16, thr0, thr int32) uint8 {
	var buf [4 * 16]int32
	forwardDCT16x16N4Col(&buf, residual)
	mask := fwdRowSkipMaskRows(buf[:], 16, 4, thr0, thr, fwdRowSkipSlack16)
	if mask == 0 {
		return 0
	}
	fwd16x16N4Rows(coeff, &buf, mask)
	return mask
}

// ForwardDCT32x32N4RowSkip8BitTrusted is ForwardDCT16x16N4RowSkip8BitTrusted
// for 32x32 blocks: column-pass rows 0..7 (groups 0 and 1, mask bits 0 and
// 1), the 8x8 low coefficients. A group that is not live writes zeros when
// the other one is.
func ForwardDCT32x32N4RowSkip8BitTrusted(coeff *[1024]int32, residual *[1024]int16, thr0, thr int32) uint8 {
	var buf [8 * 32]int32
	forwardDCT32x32N4Col(&buf, residual)
	mask := fwdRowSkipMaskRows(buf[:], 32, 8, thr0, thr, fwdRowSkipSlack32)
	if mask == 0 {
		return 0
	}
	fwd32x32N4Rows(coeff, &buf, mask)
	return mask
}

// fwdRowSkipMaskRows is fwdRowSkipMask over the first rows rows of an
// n-wide row-major column-pass output.
func fwdRowSkipMaskRows(buf []int32, n, rows int, thr0, thr, slack int32) uint8 {
	var mask uint8
	for r := range rows {
		s := int32(0)
		for _, v := range buf[r*n : r*n+n] {
			s += max(v, -v)
		}
		t := thr
		if r == 0 {
			t = thr0
		}
		if fwdRowSkipLive(s, t, slack) {
			mask |= 1 << (r >> 2)
		}
	}
	return mask
}

// The column passes (forwardDCT16x16N4Col, forwardDCT32x32N4Col) and row
// passes (fwd16x16N4Rows, fwd32x32N4Rows) are bound per build by plain
// wrappers rather than function variables, which would move the stack
// scratch to the heap.

// fwd16x16N4RowsPureGo is the N4 row pass (fwdDCT16N4 at cos_bit 12,
// shift[2] = 0) of the rounded column outputs buf (row-major, 4 x 16).
func fwd16x16N4RowsPureGo(coeff *[256]int32, buf *[4 * 16]int32, mask uint8) {
	_ = mask // the 4 rows form one group, live whenever this runs
	for r := range 4 {
		o := fwdDCT16N4((*[16]int32)(buf[r*16:r*16+16]), &fwdCospi12, 12)
		for k, v := range o {
			coeff[k*16+r] = v
		}
	}
}

// fwd32x32N4RowsPureGo is the N4 row pass (fwdDCT32N4 at cos_bit 12) of the
// live four-row groups of buf (row-major, 8 x 32); rows of a dead group read 0.
func fwd32x32N4RowsPureGo(coeff *[1024]int32, buf *[8 * 32]int32, mask uint8) {
	for r := range 8 {
		if mask&(1<<(r>>2)) == 0 {
			for k := range 8 {
				coeff[k*32+r] = 0
			}
			continue
		}
		o := fwdDCT32N4((*[32]int32)(buf[r*32:r*32+32]), &fwdCospi12, 12)
		for k, v := range o {
			coeff[k*32+r] = v
		}
	}
}

// forwardDCT16x16N4ColPureGo writes the rounded N4 column outputs
// (fwdDCT16N4 at cos_bit 13 after shift[0] = 2, shift[1] = 2) of the 16
// columns into buf[k*16+c].
func forwardDCT16x16N4ColPureGo(buf *[4 * 16]int32, residual *[256]int16) {
	var in [16]int32
	for c := range 16 {
		for r := range 16 {
			in[r] = int32(residual[r*16+c]) << 2
		}
		o := fwdDCT16N4(&in, &fwdCospi13, 13)
		for k, v := range o {
			buf[k*16+c] = (v + 2) >> 2
		}
	}
}

// forwardDCT32x32N4ColPureGo writes the rounded N4 column outputs
// (fwdDCT32N4 at cos_bit 12 after shift[0] = 2, shift[1] = 4) of the 32
// columns into buf[k*32+c].
func forwardDCT32x32N4ColPureGo(buf *[8 * 32]int32, residual *[1024]int16) {
	var in [32]int32
	for c := range 32 {
		for r := range 32 {
			in[r] = int32(residual[r*32+c]) << 2
		}
		o := fwdDCT32N4(&in, &fwdCospi12, 12)
		for k, v := range o {
			buf[k*32+c] = (v + 8) >> 4
		}
	}
}

// fwdDCT16N4 returns outputs 0..3 of fwdDCT16 (SVT-AV1 svt_av1_fdct16_new_N4).
func fwdDCT16N4(x *[16]int32, c *[64]int32, cosBit uint) (o [4]int32) {
	x0 := x[0]
	x1 := x[1]
	x2 := x[2]
	x3 := x[3]
	x4 := x[4]
	x5 := x[5]
	x6 := x[6]
	x7 := x[7]
	x8 := x[8]
	x9 := x[9]
	x10 := x[10]
	x11 := x[11]
	x12 := x[12]
	x13 := x[13]
	x14 := x[14]
	x15 := x[15]
	t0 := x0 + x15
	t1 := x1 + x14
	t2 := x2 + x13
	t3 := x3 + x12
	t4 := x4 + x11
	t5 := x5 + x10
	t6 := x6 + x9
	t7 := x7 + x8
	t8 := x7 - x8
	t9 := x6 - x9
	t10 := x5 - x10
	t11 := x4 - x11
	t12 := x3 - x12
	t13 := x2 - x13
	t14 := x1 - x14
	t15 := x0 - x15
	t16 := t0 + t7
	t17 := t1 + t6
	t18 := t2 + t5
	t19 := t3 + t4
	t20 := t3 - t4
	t21 := t2 - t5
	t22 := t1 - t6
	t23 := t0 - t7
	t24 := fwdHalfBtf(-c[32], t10, c[32], t13, cosBit)
	t25 := fwdHalfBtf(-c[32], t11, c[32], t12, cosBit)
	t26 := fwdHalfBtf(c[32], t12, c[32], t11, cosBit)
	t27 := fwdHalfBtf(c[32], t13, c[32], t10, cosBit)
	t28 := t16 + t19
	t29 := t17 + t18
	t30 := fwdHalfBtf(-c[32], t21, c[32], t22, cosBit)
	t31 := fwdHalfBtf(c[32], t22, c[32], t21, cosBit)
	t32 := t8 + t25
	t33 := t9 + t24
	t34 := t9 - t24
	t35 := t8 - t25
	t36 := t15 - t26
	t37 := t14 - t27
	t38 := t14 + t27
	t39 := t15 + t26
	t40 := fwdHalfBtf(c[32], t28, c[32], t29, cosBit)
	t41 := t20 + t30
	t42 := t23 + t31
	t43 := fwdHalfBtf(-c[16], t33, c[48], t38, cosBit)
	t44 := fwdHalfBtf(-c[48], t34, -c[16], t37, cosBit)
	t45 := fwdHalfBtf(c[48], t37, -c[16], t34, cosBit)
	t46 := fwdHalfBtf(c[16], t38, c[48], t33, cosBit)
	t47 := fwdHalfBtf(c[56], t41, c[8], t42, cosBit)
	t48 := t32 + t43
	t49 := t35 + t44
	t50 := t36 + t45
	t51 := t39 + t46
	t52 := fwdHalfBtf(c[60], t48, c[4], t51, cosBit)
	t53 := fwdHalfBtf(c[12], t50, -c[52], t49, cosBit)
	o[0] = t40
	o[1] = t52
	o[2] = t47
	o[3] = t53
	return o
}

// fwdDCT32N4 returns outputs 0..7 of fwdDCT32 (SVT-AV1 svt_av1_fdct32_new_N4).
func fwdDCT32N4(x *[32]int32, c *[64]int32, cosBit uint) (o [8]int32) {
	x0 := x[0]
	x1 := x[1]
	x2 := x[2]
	x3 := x[3]
	x4 := x[4]
	x5 := x[5]
	x6 := x[6]
	x7 := x[7]
	x8 := x[8]
	x9 := x[9]
	x10 := x[10]
	x11 := x[11]
	x12 := x[12]
	x13 := x[13]
	x14 := x[14]
	x15 := x[15]
	x16 := x[16]
	x17 := x[17]
	x18 := x[18]
	x19 := x[19]
	x20 := x[20]
	x21 := x[21]
	x22 := x[22]
	x23 := x[23]
	x24 := x[24]
	x25 := x[25]
	x26 := x[26]
	x27 := x[27]
	x28 := x[28]
	x29 := x[29]
	x30 := x[30]
	x31 := x[31]
	t0 := x0 + x31
	t1 := x1 + x30
	t2 := x2 + x29
	t3 := x3 + x28
	t4 := x4 + x27
	t5 := x5 + x26
	t6 := x6 + x25
	t7 := x7 + x24
	t8 := x8 + x23
	t9 := x9 + x22
	t10 := x10 + x21
	t11 := x11 + x20
	t12 := x12 + x19
	t13 := x13 + x18
	t14 := x14 + x17
	t15 := x15 + x16
	t16 := x15 - x16
	t17 := x14 - x17
	t18 := x13 - x18
	t19 := x12 - x19
	t20 := x11 - x20
	t21 := x10 - x21
	t22 := x9 - x22
	t23 := x8 - x23
	t24 := x7 - x24
	t25 := x6 - x25
	t26 := x5 - x26
	t27 := x4 - x27
	t28 := x3 - x28
	t29 := x2 - x29
	t30 := x1 - x30
	t31 := x0 - x31
	t32 := t0 + t15
	t33 := t1 + t14
	t34 := t2 + t13
	t35 := t3 + t12
	t36 := t4 + t11
	t37 := t5 + t10
	t38 := t6 + t9
	t39 := t7 + t8
	t40 := t7 - t8
	t41 := t6 - t9
	t42 := t5 - t10
	t43 := t4 - t11
	t44 := t3 - t12
	t45 := t2 - t13
	t46 := t1 - t14
	t47 := t0 - t15
	t48 := fwdHalfBtf(-c[32], t20, c[32], t27, cosBit)
	t49 := fwdHalfBtf(-c[32], t21, c[32], t26, cosBit)
	t50 := fwdHalfBtf(-c[32], t22, c[32], t25, cosBit)
	t51 := fwdHalfBtf(-c[32], t23, c[32], t24, cosBit)
	t52 := fwdHalfBtf(c[32], t24, c[32], t23, cosBit)
	t53 := fwdHalfBtf(c[32], t25, c[32], t22, cosBit)
	t54 := fwdHalfBtf(c[32], t26, c[32], t21, cosBit)
	t55 := fwdHalfBtf(c[32], t27, c[32], t20, cosBit)
	t56 := t32 + t39
	t57 := t33 + t38
	t58 := t34 + t37
	t59 := t35 + t36
	t60 := t35 - t36
	t61 := t34 - t37
	t62 := t33 - t38
	t63 := t32 - t39
	t64 := fwdHalfBtf(-c[32], t42, c[32], t45, cosBit)
	t65 := fwdHalfBtf(-c[32], t43, c[32], t44, cosBit)
	t66 := fwdHalfBtf(c[32], t44, c[32], t43, cosBit)
	t67 := fwdHalfBtf(c[32], t45, c[32], t42, cosBit)
	t68 := t16 + t51
	t69 := t17 + t50
	t70 := t18 + t49
	t71 := t19 + t48
	t72 := t19 - t48
	t73 := t18 - t49
	t74 := t17 - t50
	t75 := t16 - t51
	t76 := t31 - t52
	t77 := t30 - t53
	t78 := t29 - t54
	t79 := t28 - t55
	t80 := t28 + t55
	t81 := t29 + t54
	t82 := t30 + t53
	t83 := t31 + t52
	t84 := t56 + t59
	t85 := t57 + t58
	t86 := fwdHalfBtf(-c[32], t61, c[32], t62, cosBit)
	t87 := fwdHalfBtf(c[32], t62, c[32], t61, cosBit)
	t88 := t40 + t65
	t89 := t41 + t64
	t90 := t41 - t64
	t91 := t40 - t65
	t92 := t47 - t66
	t93 := t46 - t67
	t94 := t46 + t67
	t95 := t47 + t66
	t96 := fwdHalfBtf(-c[16], t70, c[48], t81, cosBit)
	t97 := fwdHalfBtf(-c[16], t71, c[48], t80, cosBit)
	t98 := fwdHalfBtf(-c[48], t72, -c[16], t79, cosBit)
	t99 := fwdHalfBtf(-c[48], t73, -c[16], t78, cosBit)
	t100 := fwdHalfBtf(c[48], t78, -c[16], t73, cosBit)
	t101 := fwdHalfBtf(c[48], t79, -c[16], t72, cosBit)
	t102 := fwdHalfBtf(c[16], t80, c[48], t71, cosBit)
	t103 := fwdHalfBtf(c[16], t81, c[48], t70, cosBit)
	t104 := fwdHalfBtf(c[32], t84, c[32], t85, cosBit)
	t105 := t60 + t86
	t106 := t63 + t87
	t107 := fwdHalfBtf(-c[16], t89, c[48], t94, cosBit)
	t108 := fwdHalfBtf(-c[48], t90, -c[16], t93, cosBit)
	t109 := fwdHalfBtf(c[48], t93, -c[16], t90, cosBit)
	t110 := fwdHalfBtf(c[16], t94, c[48], t89, cosBit)
	t111 := t68 + t97
	t112 := t69 + t96
	t113 := t69 - t96
	t114 := t68 - t97
	t115 := t75 - t98
	t116 := t74 - t99
	t117 := t74 + t99
	t118 := t75 + t98
	t119 := t76 + t101
	t120 := t77 + t100
	t121 := t77 - t100
	t122 := t76 - t101
	t123 := t83 - t102
	t124 := t82 - t103
	t125 := t82 + t103
	t126 := t83 + t102
	t127 := fwdHalfBtf(c[56], t105, c[8], t106, cosBit)
	t128 := t88 + t107
	t129 := t91 + t108
	t130 := t92 + t109
	t131 := t95 + t110
	t132 := fwdHalfBtf(-c[8], t112, c[56], t125, cosBit)
	t133 := fwdHalfBtf(-c[56], t113, -c[8], t124, cosBit)
	t134 := fwdHalfBtf(-c[40], t116, c[24], t121, cosBit)
	t135 := fwdHalfBtf(-c[24], t117, -c[40], t120, cosBit)
	t136 := fwdHalfBtf(c[24], t120, -c[40], t117, cosBit)
	t137 := fwdHalfBtf(c[40], t121, c[24], t116, cosBit)
	t138 := fwdHalfBtf(c[56], t124, -c[8], t113, cosBit)
	t139 := fwdHalfBtf(c[8], t125, c[56], t112, cosBit)
	t140 := fwdHalfBtf(c[60], t128, c[4], t131, cosBit)
	t141 := fwdHalfBtf(c[12], t130, -c[52], t129, cosBit)
	t142 := t111 + t132
	t143 := t114 + t133
	t144 := t115 + t134
	t145 := t118 + t135
	t146 := t119 + t136
	t147 := t122 + t137
	t148 := t123 + t138
	t149 := t126 + t139
	t150 := fwdHalfBtf(c[62], t142, c[2], t149, cosBit)
	t151 := fwdHalfBtf(c[54], t144, c[10], t147, cosBit)
	t152 := fwdHalfBtf(c[6], t146, -c[58], t145, cosBit)
	t153 := fwdHalfBtf(c[14], t148, -c[50], t143, cosBit)
	o[0] = t104
	o[1] = t150
	o[2] = t140
	o[3] = t152
	o[4] = t127
	o[5] = t151
	o[6] = t141
	o[7] = t153
	return o
}
