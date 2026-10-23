// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

// itx_fused64: dav1d's pass structure (itx_fused32) for the 8-bit 64x64
// DCT_DCT block, whose coded coefficients lie in the 32x32 corner. The row
// pass runs up to four eight-row strips straight from the column-major
// coefficients with the DCT64 network form that matches the coded columns
// (8, 16 or 32 inputs), rounds by 2 in registers and transposes 8x8 into a
// 32x64 int16 tmp; the eight column strips run the form matching the coded
// rows with Round2(., 4), the prediction add and the byte saturation fused
// into their final stage. The even half of the network is the DCT32 of
// itx_fused32 on the even inputs; the odd half (itx64Odd*) is generated
// from the stage structure of inverseDCT64 with the zero operands of each
// form folded. itx64_gain_test.go certifies the per-lane L1 guard for it.

// itx64K[i] broadcasts cospi(i) and itx64K[64+i] broadcasts -cospi(i).
var itx64K = func() (k [128][8]int16) {
	for i, c := range itx64Cos {
		for l := 0; l < 8; l++ {
			k[i][l] = c
			k[64+i][l] = -c
		}
	}
	return k
}()

// itx64Odd32 is the odd half of inverseDCT64 on eight lanes for odd inputs
// below 32 (the rest zero): in[k] is coefficient k and o[j] is stage-10
// value 32+j. Generated from the stage structure of inverseDCT64 with the
// zero operands folded.
func itx64Odd32(in *[32]archsimd.Int16x8, o *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[1], k(63), s12)
	a2 := itx16Rot1(in[1], k(1), s12)
	a3 := itx16Rot1(in[31], kn(33), s12)
	a4 := itx16Rot1(in[31], k(31), s12)
	a5 := itx16Rot1(in[17], k(47), s12)
	a6 := itx16Rot1(in[17], k(17), s12)
	a7 := itx16Rot1(in[15], kn(49), s12)
	a8 := itx16Rot1(in[15], k(15), s12)
	a9 := itx16Rot1(in[9], k(55), s12)
	a10 := itx16Rot1(in[9], k(9), s12)
	a11 := itx16Rot1(in[23], kn(41), s12)
	a12 := itx16Rot1(in[23], k(23), s12)
	a13 := itx16Rot1(in[25], k(39), s12)
	a14 := itx16Rot1(in[25], k(25), s12)
	a15 := itx16Rot1(in[7], kn(57), s12)
	a16 := itx16Rot1(in[7], k(7), s12)
	a17 := itx16Rot1(in[5], k(59), s12)
	a18 := itx16Rot1(in[5], k(5), s12)
	a19 := itx16Rot1(in[27], kn(37), s12)
	a20 := itx16Rot1(in[27], k(27), s12)
	a21 := itx16Rot1(in[21], k(43), s12)
	a22 := itx16Rot1(in[21], k(21), s12)
	a23 := itx16Rot1(in[11], kn(53), s12)
	a24 := itx16Rot1(in[11], k(11), s12)
	a25 := itx16Rot1(in[13], k(51), s12)
	a26 := itx16Rot1(in[13], k(13), s12)
	a27 := itx16Rot1(in[19], kn(45), s12)
	a28 := itx16Rot1(in[19], k(19), s12)
	a29 := itx16Rot1(in[29], k(35), s12)
	a30 := itx16Rot1(in[29], k(29), s12)
	a31 := itx16Rot1(in[3], kn(61), s12)
	a32 := itx16Rot1(in[3], k(3), s12)
	b33 := a1.AddSaturated(a3)
	b34 := a1.SubSaturated(a3)
	b35 := a7.SubSaturated(a5)
	b36 := a5.AddSaturated(a7)
	b37 := a9.AddSaturated(a11)
	b38 := a9.SubSaturated(a11)
	b39 := a15.SubSaturated(a13)
	b40 := a13.AddSaturated(a15)
	b41 := a17.AddSaturated(a19)
	b42 := a17.SubSaturated(a19)
	b43 := a23.SubSaturated(a21)
	b44 := a21.AddSaturated(a23)
	b45 := a25.AddSaturated(a27)
	b46 := a25.SubSaturated(a27)
	b47 := a31.SubSaturated(a29)
	b48 := a29.AddSaturated(a31)
	b49 := a32.AddSaturated(a30)
	b50 := a32.SubSaturated(a30)
	b51 := a26.SubSaturated(a28)
	b52 := a28.AddSaturated(a26)
	b53 := a24.AddSaturated(a22)
	b54 := a24.SubSaturated(a22)
	b55 := a18.SubSaturated(a20)
	b56 := a20.AddSaturated(a18)
	b57 := a16.AddSaturated(a14)
	b58 := a16.SubSaturated(a14)
	b59 := a10.SubSaturated(a12)
	b60 := a12.AddSaturated(a10)
	b61 := a8.AddSaturated(a6)
	b62 := a8.SubSaturated(a6)
	b63 := a2.SubSaturated(a4)
	b64 := a4.AddSaturated(a2)
	c65 := itx16Rot(b34, b63, kn(4), k(60), s12)
	c66 := itx16Rot(b34, b63, k(60), k(4), s12)
	c67 := itx16Rot(b35, b62, kn(60), kn(4), s12)
	c68 := itx16Rot(b35, b62, kn(4), k(60), s12)
	c69 := itx16Rot(b38, b59, kn(36), k(28), s12)
	c70 := itx16Rot(b38, b59, k(28), k(36), s12)
	c71 := itx16Rot(b39, b58, kn(28), kn(36), s12)
	c72 := itx16Rot(b39, b58, kn(36), k(28), s12)
	c73 := itx16Rot(b42, b55, kn(20), k(44), s12)
	c74 := itx16Rot(b42, b55, k(44), k(20), s12)
	c75 := itx16Rot(b43, b54, kn(44), kn(20), s12)
	c76 := itx16Rot(b43, b54, kn(20), k(44), s12)
	c77 := itx16Rot(b46, b51, kn(52), k(12), s12)
	c78 := itx16Rot(b46, b51, k(12), k(52), s12)
	c79 := itx16Rot(b47, b50, kn(12), kn(52), s12)
	c80 := itx16Rot(b47, b50, kn(52), k(12), s12)
	d81 := b33.AddSaturated(b36)
	d82 := c65.AddSaturated(c67)
	d83 := c65.SubSaturated(c67)
	d84 := b33.SubSaturated(b36)
	d85 := b40.SubSaturated(b37)
	d86 := c71.SubSaturated(c69)
	d87 := c69.AddSaturated(c71)
	d88 := b37.AddSaturated(b40)
	d89 := b41.AddSaturated(b44)
	d90 := c73.AddSaturated(c75)
	d91 := c73.SubSaturated(c75)
	d92 := b41.SubSaturated(b44)
	d93 := b48.SubSaturated(b45)
	d94 := c79.SubSaturated(c77)
	d95 := c77.AddSaturated(c79)
	d96 := b45.AddSaturated(b48)
	d97 := b49.AddSaturated(b52)
	d98 := c80.AddSaturated(c78)
	d99 := c80.SubSaturated(c78)
	d100 := b49.SubSaturated(b52)
	d101 := b56.SubSaturated(b53)
	d102 := c74.SubSaturated(c76)
	d103 := c76.AddSaturated(c74)
	d104 := b53.AddSaturated(b56)
	d105 := b57.AddSaturated(b60)
	d106 := c72.AddSaturated(c70)
	d107 := c72.SubSaturated(c70)
	d108 := b57.SubSaturated(b60)
	d109 := b64.SubSaturated(b61)
	d110 := c66.SubSaturated(c68)
	d111 := c68.AddSaturated(c66)
	d112 := b61.AddSaturated(b64)
	e113 := itx16Rot(d83, d110, kn(8), k(56), s12)
	e114 := itx16Rot(d83, d110, k(56), k(8), s12)
	e115 := itx16Rot(d84, d109, kn(8), k(56), s12)
	e116 := itx16Rot(d84, d109, k(56), k(8), s12)
	e117 := itx16Rot(d85, d108, kn(56), kn(8), s12)
	e118 := itx16Rot(d85, d108, kn(8), k(56), s12)
	e119 := itx16Rot(d86, d107, kn(56), kn(8), s12)
	e120 := itx16Rot(d86, d107, kn(8), k(56), s12)
	e121 := itx16Rot(d91, d102, kn(40), k(24), s12)
	e122 := itx16Rot(d91, d102, k(24), k(40), s12)
	e123 := itx16Rot(d92, d101, kn(40), k(24), s12)
	e124 := itx16Rot(d92, d101, k(24), k(40), s12)
	e125 := itx16Rot(d93, d100, kn(24), kn(40), s12)
	e126 := itx16Rot(d93, d100, kn(40), k(24), s12)
	e127 := itx16Rot(d94, d99, kn(24), kn(40), s12)
	e128 := itx16Rot(d94, d99, kn(40), k(24), s12)
	f129 := d81.AddSaturated(d88)
	f130 := d81.SubSaturated(d88)
	f131 := d96.SubSaturated(d89)
	f132 := d89.AddSaturated(d96)
	f133 := d82.AddSaturated(d87)
	f134 := d82.SubSaturated(d87)
	f135 := d95.SubSaturated(d90)
	f136 := d90.AddSaturated(d95)
	f137 := e113.AddSaturated(e119)
	f138 := e113.SubSaturated(e119)
	f139 := e127.SubSaturated(e121)
	f140 := e121.AddSaturated(e127)
	f141 := e115.AddSaturated(e117)
	f142 := e115.SubSaturated(e117)
	f143 := e125.SubSaturated(e123)
	f144 := e123.AddSaturated(e125)
	f145 := d97.AddSaturated(d104)
	f146 := d97.SubSaturated(d104)
	f147 := d112.SubSaturated(d105)
	f148 := d105.AddSaturated(d112)
	f149 := d98.AddSaturated(d103)
	f150 := d98.SubSaturated(d103)
	f151 := d111.SubSaturated(d106)
	f152 := d106.AddSaturated(d111)
	f153 := e128.AddSaturated(e122)
	f154 := e128.SubSaturated(e122)
	f155 := e114.SubSaturated(e120)
	f156 := e120.AddSaturated(e114)
	f157 := e126.AddSaturated(e124)
	f158 := e126.SubSaturated(e124)
	f159 := e116.SubSaturated(e118)
	f160 := e118.AddSaturated(e116)
	g161 := itx16Rot(f142, f159, kn(16), k(48), s12)
	g162 := itx16Rot(f142, f159, k(48), k(16), s12)
	g163 := itx16Rot(f131, f146, kn(48), kn(16), s12)
	g164 := itx16Rot(f131, f146, kn(16), k(48), s12)
	g165 := itx16Rot(f138, f155, kn(16), k(48), s12)
	g166 := itx16Rot(f138, f155, k(48), k(16), s12)
	g167 := itx16Rot(f135, f150, kn(48), kn(16), s12)
	g168 := itx16Rot(f135, f150, kn(16), k(48), s12)
	g169 := itx16Rot(f134, f151, kn(16), k(48), s12)
	g170 := itx16Rot(f134, f151, k(48), k(16), s12)
	g171 := itx16Rot(f139, f154, kn(48), kn(16), s12)
	g172 := itx16Rot(f139, f154, kn(16), k(48), s12)
	g173 := itx16Rot(f130, f147, kn(16), k(48), s12)
	g174 := itx16Rot(f130, f147, k(48), k(16), s12)
	g175 := itx16Rot(f143, f158, kn(48), kn(16), s12)
	g176 := itx16Rot(f143, f158, kn(16), k(48), s12)
	h177 := f129.AddSaturated(f132)
	h178 := f129.SubSaturated(f132)
	h179 := f148.SubSaturated(f145)
	h180 := f145.AddSaturated(f148)
	h181 := f133.AddSaturated(f136)
	h182 := f133.SubSaturated(f136)
	h183 := f152.SubSaturated(f149)
	h184 := f149.AddSaturated(f152)
	h185 := f137.AddSaturated(f140)
	h186 := f137.SubSaturated(f140)
	h187 := f156.SubSaturated(f153)
	h188 := f153.AddSaturated(f156)
	h189 := f141.AddSaturated(f144)
	h190 := f141.SubSaturated(f144)
	h191 := f160.SubSaturated(f157)
	h192 := f157.AddSaturated(f160)
	h193 := g161.AddSaturated(g175)
	h194 := g161.SubSaturated(g175)
	h195 := g162.SubSaturated(g176)
	h196 := g176.AddSaturated(g162)
	h197 := g165.AddSaturated(g171)
	h198 := g165.SubSaturated(g171)
	h199 := g166.SubSaturated(g172)
	h200 := g172.AddSaturated(g166)
	h201 := g169.AddSaturated(g167)
	h202 := g169.SubSaturated(g167)
	h203 := g170.SubSaturated(g168)
	h204 := g168.AddSaturated(g170)
	h205 := g173.AddSaturated(g163)
	h206 := g173.SubSaturated(g163)
	h207 := g174.SubSaturated(g164)
	h208 := g164.AddSaturated(g174)
	i209 := itx16Rot(h206, h207, kn(32), k(32), s12)
	i210 := itx16Rot(h206, h207, k(32), k(32), s12)
	i211 := itx16Rot(h202, h203, kn(32), k(32), s12)
	i212 := itx16Rot(h202, h203, k(32), k(32), s12)
	i213 := itx16Rot(h198, h199, kn(32), k(32), s12)
	i214 := itx16Rot(h198, h199, k(32), k(32), s12)
	i215 := itx16Rot(h194, h195, kn(32), k(32), s12)
	i216 := itx16Rot(h194, h195, k(32), k(32), s12)
	i217 := itx16Rot(h190, h191, kn(32), k(32), s12)
	i218 := itx16Rot(h190, h191, k(32), k(32), s12)
	i219 := itx16Rot(h186, h187, kn(32), k(32), s12)
	i220 := itx16Rot(h186, h187, k(32), k(32), s12)
	i221 := itx16Rot(h182, h183, kn(32), k(32), s12)
	i222 := itx16Rot(h182, h183, k(32), k(32), s12)
	i223 := itx16Rot(h178, h179, kn(32), k(32), s12)
	i224 := itx16Rot(h178, h179, k(32), k(32), s12)
	o[0] = h177
	o[1] = h181
	o[2] = h185
	o[3] = h189
	o[4] = h193
	o[5] = h197
	o[6] = h201
	o[7] = h205
	o[8] = i209
	o[9] = i211
	o[10] = i213
	o[11] = i215
	o[12] = i217
	o[13] = i219
	o[14] = i221
	o[15] = i223
	o[16] = i224
	o[17] = i222
	o[18] = i220
	o[19] = i218
	o[20] = i216
	o[21] = i214
	o[22] = i212
	o[23] = i210
	o[24] = h208
	o[25] = h204
	o[26] = h200
	o[27] = h196
	o[28] = h192
	o[29] = h188
	o[30] = h184
	o[31] = h180
}

// itx64Odd16 is the odd half of inverseDCT64 on eight lanes for odd inputs
// below 16 (the rest zero): in[k] is coefficient k and o[j] is stage-10
// value 32+j. Generated from the stage structure of inverseDCT64 with the
// zero operands folded.
func itx64Odd16(in *[32]archsimd.Int16x8, o *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[1], k(63), s12)
	a2 := itx16Rot1(in[1], k(1), s12)
	a3 := itx16Rot1(in[15], kn(49), s12)
	a4 := itx16Rot1(in[15], k(15), s12)
	a5 := itx16Rot1(in[9], k(55), s12)
	a6 := itx16Rot1(in[9], k(9), s12)
	a7 := itx16Rot1(in[7], kn(57), s12)
	a8 := itx16Rot1(in[7], k(7), s12)
	a9 := itx16Rot1(in[5], k(59), s12)
	a10 := itx16Rot1(in[5], k(5), s12)
	a11 := itx16Rot1(in[11], kn(53), s12)
	a12 := itx16Rot1(in[11], k(11), s12)
	a13 := itx16Rot1(in[13], k(51), s12)
	a14 := itx16Rot1(in[13], k(13), s12)
	a15 := itx16Rot1(in[3], kn(61), s12)
	a16 := itx16Rot1(in[3], k(3), s12)
	c17 := itx16Rot(a1, a2, kn(4), k(60), s12)
	c18 := itx16Rot(a1, a2, k(60), k(4), s12)
	c19 := itx16Rot(a3, a4, kn(60), kn(4), s12)
	c20 := itx16Rot(a3, a4, kn(4), k(60), s12)
	c21 := itx16Rot(a5, a6, kn(36), k(28), s12)
	c22 := itx16Rot(a5, a6, k(28), k(36), s12)
	c23 := itx16Rot(a7, a8, kn(28), kn(36), s12)
	c24 := itx16Rot(a7, a8, kn(36), k(28), s12)
	c25 := itx16Rot(a9, a10, kn(20), k(44), s12)
	c26 := itx16Rot(a9, a10, k(44), k(20), s12)
	c27 := itx16Rot(a11, a12, kn(44), kn(20), s12)
	c28 := itx16Rot(a11, a12, kn(20), k(44), s12)
	c29 := itx16Rot(a13, a14, kn(52), k(12), s12)
	c30 := itx16Rot(a13, a14, k(12), k(52), s12)
	c31 := itx16Rot(a15, a16, kn(12), kn(52), s12)
	c32 := itx16Rot(a15, a16, kn(52), k(12), s12)
	d33 := a1.AddSaturated(a3)
	d34 := c17.AddSaturated(c19)
	d35 := c17.SubSaturated(c19)
	d36 := a1.SubSaturated(a3)
	d37 := a7.SubSaturated(a5)
	d38 := c23.SubSaturated(c21)
	d39 := c21.AddSaturated(c23)
	d40 := a5.AddSaturated(a7)
	d41 := a9.AddSaturated(a11)
	d42 := c25.AddSaturated(c27)
	d43 := c25.SubSaturated(c27)
	d44 := a9.SubSaturated(a11)
	d45 := a15.SubSaturated(a13)
	d46 := c31.SubSaturated(c29)
	d47 := c29.AddSaturated(c31)
	d48 := a13.AddSaturated(a15)
	d49 := a16.AddSaturated(a14)
	d50 := c32.AddSaturated(c30)
	d51 := c32.SubSaturated(c30)
	d52 := a16.SubSaturated(a14)
	d53 := a10.SubSaturated(a12)
	d54 := c26.SubSaturated(c28)
	d55 := c28.AddSaturated(c26)
	d56 := a12.AddSaturated(a10)
	d57 := a8.AddSaturated(a6)
	d58 := c24.AddSaturated(c22)
	d59 := c24.SubSaturated(c22)
	d60 := a8.SubSaturated(a6)
	d61 := a2.SubSaturated(a4)
	d62 := c18.SubSaturated(c20)
	d63 := c20.AddSaturated(c18)
	d64 := a4.AddSaturated(a2)
	e65 := itx16Rot(d35, d62, kn(8), k(56), s12)
	e66 := itx16Rot(d35, d62, k(56), k(8), s12)
	e67 := itx16Rot(d36, d61, kn(8), k(56), s12)
	e68 := itx16Rot(d36, d61, k(56), k(8), s12)
	e69 := itx16Rot(d37, d60, kn(56), kn(8), s12)
	e70 := itx16Rot(d37, d60, kn(8), k(56), s12)
	e71 := itx16Rot(d38, d59, kn(56), kn(8), s12)
	e72 := itx16Rot(d38, d59, kn(8), k(56), s12)
	e73 := itx16Rot(d43, d54, kn(40), k(24), s12)
	e74 := itx16Rot(d43, d54, k(24), k(40), s12)
	e75 := itx16Rot(d44, d53, kn(40), k(24), s12)
	e76 := itx16Rot(d44, d53, k(24), k(40), s12)
	e77 := itx16Rot(d45, d52, kn(24), kn(40), s12)
	e78 := itx16Rot(d45, d52, kn(40), k(24), s12)
	e79 := itx16Rot(d46, d51, kn(24), kn(40), s12)
	e80 := itx16Rot(d46, d51, kn(40), k(24), s12)
	f81 := d33.AddSaturated(d40)
	f82 := d33.SubSaturated(d40)
	f83 := d48.SubSaturated(d41)
	f84 := d41.AddSaturated(d48)
	f85 := d34.AddSaturated(d39)
	f86 := d34.SubSaturated(d39)
	f87 := d47.SubSaturated(d42)
	f88 := d42.AddSaturated(d47)
	f89 := e65.AddSaturated(e71)
	f90 := e65.SubSaturated(e71)
	f91 := e79.SubSaturated(e73)
	f92 := e73.AddSaturated(e79)
	f93 := e67.AddSaturated(e69)
	f94 := e67.SubSaturated(e69)
	f95 := e77.SubSaturated(e75)
	f96 := e75.AddSaturated(e77)
	f97 := d49.AddSaturated(d56)
	f98 := d49.SubSaturated(d56)
	f99 := d64.SubSaturated(d57)
	f100 := d57.AddSaturated(d64)
	f101 := d50.AddSaturated(d55)
	f102 := d50.SubSaturated(d55)
	f103 := d63.SubSaturated(d58)
	f104 := d58.AddSaturated(d63)
	f105 := e80.AddSaturated(e74)
	f106 := e80.SubSaturated(e74)
	f107 := e66.SubSaturated(e72)
	f108 := e72.AddSaturated(e66)
	f109 := e78.AddSaturated(e76)
	f110 := e78.SubSaturated(e76)
	f111 := e68.SubSaturated(e70)
	f112 := e70.AddSaturated(e68)
	g113 := itx16Rot(f94, f111, kn(16), k(48), s12)
	g114 := itx16Rot(f94, f111, k(48), k(16), s12)
	g115 := itx16Rot(f83, f98, kn(48), kn(16), s12)
	g116 := itx16Rot(f83, f98, kn(16), k(48), s12)
	g117 := itx16Rot(f90, f107, kn(16), k(48), s12)
	g118 := itx16Rot(f90, f107, k(48), k(16), s12)
	g119 := itx16Rot(f87, f102, kn(48), kn(16), s12)
	g120 := itx16Rot(f87, f102, kn(16), k(48), s12)
	g121 := itx16Rot(f86, f103, kn(16), k(48), s12)
	g122 := itx16Rot(f86, f103, k(48), k(16), s12)
	g123 := itx16Rot(f91, f106, kn(48), kn(16), s12)
	g124 := itx16Rot(f91, f106, kn(16), k(48), s12)
	g125 := itx16Rot(f82, f99, kn(16), k(48), s12)
	g126 := itx16Rot(f82, f99, k(48), k(16), s12)
	g127 := itx16Rot(f95, f110, kn(48), kn(16), s12)
	g128 := itx16Rot(f95, f110, kn(16), k(48), s12)
	h129 := f81.AddSaturated(f84)
	h130 := f81.SubSaturated(f84)
	h131 := f100.SubSaturated(f97)
	h132 := f97.AddSaturated(f100)
	h133 := f85.AddSaturated(f88)
	h134 := f85.SubSaturated(f88)
	h135 := f104.SubSaturated(f101)
	h136 := f101.AddSaturated(f104)
	h137 := f89.AddSaturated(f92)
	h138 := f89.SubSaturated(f92)
	h139 := f108.SubSaturated(f105)
	h140 := f105.AddSaturated(f108)
	h141 := f93.AddSaturated(f96)
	h142 := f93.SubSaturated(f96)
	h143 := f112.SubSaturated(f109)
	h144 := f109.AddSaturated(f112)
	h145 := g113.AddSaturated(g127)
	h146 := g113.SubSaturated(g127)
	h147 := g114.SubSaturated(g128)
	h148 := g128.AddSaturated(g114)
	h149 := g117.AddSaturated(g123)
	h150 := g117.SubSaturated(g123)
	h151 := g118.SubSaturated(g124)
	h152 := g124.AddSaturated(g118)
	h153 := g121.AddSaturated(g119)
	h154 := g121.SubSaturated(g119)
	h155 := g122.SubSaturated(g120)
	h156 := g120.AddSaturated(g122)
	h157 := g125.AddSaturated(g115)
	h158 := g125.SubSaturated(g115)
	h159 := g126.SubSaturated(g116)
	h160 := g116.AddSaturated(g126)
	i161 := itx16Rot(h158, h159, kn(32), k(32), s12)
	i162 := itx16Rot(h158, h159, k(32), k(32), s12)
	i163 := itx16Rot(h154, h155, kn(32), k(32), s12)
	i164 := itx16Rot(h154, h155, k(32), k(32), s12)
	i165 := itx16Rot(h150, h151, kn(32), k(32), s12)
	i166 := itx16Rot(h150, h151, k(32), k(32), s12)
	i167 := itx16Rot(h146, h147, kn(32), k(32), s12)
	i168 := itx16Rot(h146, h147, k(32), k(32), s12)
	i169 := itx16Rot(h142, h143, kn(32), k(32), s12)
	i170 := itx16Rot(h142, h143, k(32), k(32), s12)
	i171 := itx16Rot(h138, h139, kn(32), k(32), s12)
	i172 := itx16Rot(h138, h139, k(32), k(32), s12)
	i173 := itx16Rot(h134, h135, kn(32), k(32), s12)
	i174 := itx16Rot(h134, h135, k(32), k(32), s12)
	i175 := itx16Rot(h130, h131, kn(32), k(32), s12)
	i176 := itx16Rot(h130, h131, k(32), k(32), s12)
	o[0] = h129
	o[1] = h133
	o[2] = h137
	o[3] = h141
	o[4] = h145
	o[5] = h149
	o[6] = h153
	o[7] = h157
	o[8] = i161
	o[9] = i163
	o[10] = i165
	o[11] = i167
	o[12] = i169
	o[13] = i171
	o[14] = i173
	o[15] = i175
	o[16] = i176
	o[17] = i174
	o[18] = i172
	o[19] = i170
	o[20] = i168
	o[21] = i166
	o[22] = i164
	o[23] = i162
	o[24] = h160
	o[25] = h156
	o[26] = h152
	o[27] = h148
	o[28] = h144
	o[29] = h140
	o[30] = h136
	o[31] = h132
}

// itx64Odd8 is the odd half of inverseDCT64 on eight lanes for odd inputs
// below 8 (the rest zero): in[k] is coefficient k and o[j] is stage-10
// value 32+j. Generated from the stage structure of inverseDCT64 with the
// zero operands folded.
func itx64Odd8(in *[32]archsimd.Int16x8, o *[32]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[1], k(63), s12)
	a2 := itx16Rot1(in[1], k(1), s12)
	a3 := itx16Rot1(in[7], kn(57), s12)
	a4 := itx16Rot1(in[7], k(7), s12)
	a5 := itx16Rot1(in[5], k(59), s12)
	a6 := itx16Rot1(in[5], k(5), s12)
	a7 := itx16Rot1(in[3], kn(61), s12)
	a8 := itx16Rot1(in[3], k(3), s12)
	c9 := itx16Rot(a1, a2, kn(4), k(60), s12)
	c10 := itx16Rot(a1, a2, k(60), k(4), s12)
	c11 := itx16Rot(a3, a4, kn(28), kn(36), s12)
	c12 := itx16Rot(a3, a4, kn(36), k(28), s12)
	c13 := itx16Rot(a5, a6, kn(20), k(44), s12)
	c14 := itx16Rot(a5, a6, k(44), k(20), s12)
	c15 := itx16Rot(a7, a8, kn(12), kn(52), s12)
	c16 := itx16Rot(a7, a8, kn(52), k(12), s12)
	e17 := itx16Rot(c9, c10, kn(8), k(56), s12)
	e18 := itx16Rot(c9, c10, k(56), k(8), s12)
	e19 := itx16Rot(a1, a2, kn(8), k(56), s12)
	e20 := itx16Rot(a1, a2, k(56), k(8), s12)
	e21 := itx16Rot(a3, a4, kn(56), kn(8), s12)
	e22 := itx16Rot(a3, a4, kn(8), k(56), s12)
	e23 := itx16Rot(c11, c12, kn(56), kn(8), s12)
	e24 := itx16Rot(c11, c12, kn(8), k(56), s12)
	e25 := itx16Rot(c13, c14, kn(40), k(24), s12)
	e26 := itx16Rot(c13, c14, k(24), k(40), s12)
	e27 := itx16Rot(a5, a6, kn(40), k(24), s12)
	e28 := itx16Rot(a5, a6, k(24), k(40), s12)
	e29 := itx16Rot(a7, a8, kn(24), kn(40), s12)
	e30 := itx16Rot(a7, a8, kn(40), k(24), s12)
	e31 := itx16Rot(c15, c16, kn(24), kn(40), s12)
	e32 := itx16Rot(c15, c16, kn(40), k(24), s12)
	f33 := a1.AddSaturated(a3)
	f34 := a1.SubSaturated(a3)
	f35 := a7.SubSaturated(a5)
	f36 := a5.AddSaturated(a7)
	f37 := c9.AddSaturated(c11)
	f38 := c9.SubSaturated(c11)
	f39 := c15.SubSaturated(c13)
	f40 := c13.AddSaturated(c15)
	f41 := e17.AddSaturated(e23)
	f42 := e17.SubSaturated(e23)
	f43 := e31.SubSaturated(e25)
	f44 := e25.AddSaturated(e31)
	f45 := e19.AddSaturated(e21)
	f46 := e19.SubSaturated(e21)
	f47 := e29.SubSaturated(e27)
	f48 := e27.AddSaturated(e29)
	f49 := a8.AddSaturated(a6)
	f50 := a8.SubSaturated(a6)
	f51 := a2.SubSaturated(a4)
	f52 := a4.AddSaturated(a2)
	f53 := c16.AddSaturated(c14)
	f54 := c16.SubSaturated(c14)
	f55 := c10.SubSaturated(c12)
	f56 := c12.AddSaturated(c10)
	f57 := e32.AddSaturated(e26)
	f58 := e32.SubSaturated(e26)
	f59 := e18.SubSaturated(e24)
	f60 := e24.AddSaturated(e18)
	f61 := e30.AddSaturated(e28)
	f62 := e30.SubSaturated(e28)
	f63 := e20.SubSaturated(e22)
	f64 := e22.AddSaturated(e20)
	g65 := itx16Rot(f46, f63, kn(16), k(48), s12)
	g66 := itx16Rot(f46, f63, k(48), k(16), s12)
	g67 := itx16Rot(f35, f50, kn(48), kn(16), s12)
	g68 := itx16Rot(f35, f50, kn(16), k(48), s12)
	g69 := itx16Rot(f42, f59, kn(16), k(48), s12)
	g70 := itx16Rot(f42, f59, k(48), k(16), s12)
	g71 := itx16Rot(f39, f54, kn(48), kn(16), s12)
	g72 := itx16Rot(f39, f54, kn(16), k(48), s12)
	g73 := itx16Rot(f38, f55, kn(16), k(48), s12)
	g74 := itx16Rot(f38, f55, k(48), k(16), s12)
	g75 := itx16Rot(f43, f58, kn(48), kn(16), s12)
	g76 := itx16Rot(f43, f58, kn(16), k(48), s12)
	g77 := itx16Rot(f34, f51, kn(16), k(48), s12)
	g78 := itx16Rot(f34, f51, k(48), k(16), s12)
	g79 := itx16Rot(f47, f62, kn(48), kn(16), s12)
	g80 := itx16Rot(f47, f62, kn(16), k(48), s12)
	h81 := f33.AddSaturated(f36)
	h82 := f33.SubSaturated(f36)
	h83 := f52.SubSaturated(f49)
	h84 := f49.AddSaturated(f52)
	h85 := f37.AddSaturated(f40)
	h86 := f37.SubSaturated(f40)
	h87 := f56.SubSaturated(f53)
	h88 := f53.AddSaturated(f56)
	h89 := f41.AddSaturated(f44)
	h90 := f41.SubSaturated(f44)
	h91 := f60.SubSaturated(f57)
	h92 := f57.AddSaturated(f60)
	h93 := f45.AddSaturated(f48)
	h94 := f45.SubSaturated(f48)
	h95 := f64.SubSaturated(f61)
	h96 := f61.AddSaturated(f64)
	h97 := g65.AddSaturated(g79)
	h98 := g65.SubSaturated(g79)
	h99 := g66.SubSaturated(g80)
	h100 := g80.AddSaturated(g66)
	h101 := g69.AddSaturated(g75)
	h102 := g69.SubSaturated(g75)
	h103 := g70.SubSaturated(g76)
	h104 := g76.AddSaturated(g70)
	h105 := g73.AddSaturated(g71)
	h106 := g73.SubSaturated(g71)
	h107 := g74.SubSaturated(g72)
	h108 := g72.AddSaturated(g74)
	h109 := g77.AddSaturated(g67)
	h110 := g77.SubSaturated(g67)
	h111 := g78.SubSaturated(g68)
	h112 := g68.AddSaturated(g78)
	i113 := itx16Rot(h110, h111, kn(32), k(32), s12)
	i114 := itx16Rot(h110, h111, k(32), k(32), s12)
	i115 := itx16Rot(h106, h107, kn(32), k(32), s12)
	i116 := itx16Rot(h106, h107, k(32), k(32), s12)
	i117 := itx16Rot(h102, h103, kn(32), k(32), s12)
	i118 := itx16Rot(h102, h103, k(32), k(32), s12)
	i119 := itx16Rot(h98, h99, kn(32), k(32), s12)
	i120 := itx16Rot(h98, h99, k(32), k(32), s12)
	i121 := itx16Rot(h94, h95, kn(32), k(32), s12)
	i122 := itx16Rot(h94, h95, k(32), k(32), s12)
	i123 := itx16Rot(h90, h91, kn(32), k(32), s12)
	i124 := itx16Rot(h90, h91, k(32), k(32), s12)
	i125 := itx16Rot(h86, h87, kn(32), k(32), s12)
	i126 := itx16Rot(h86, h87, k(32), k(32), s12)
	i127 := itx16Rot(h82, h83, kn(32), k(32), s12)
	i128 := itx16Rot(h82, h83, k(32), k(32), s12)
	o[0] = h81
	o[1] = h85
	o[2] = h89
	o[3] = h93
	o[4] = h97
	o[5] = h101
	o[6] = h105
	o[7] = h109
	o[8] = i113
	o[9] = i115
	o[10] = i117
	o[11] = i119
	o[12] = i121
	o[13] = i123
	o[14] = i125
	o[15] = i127
	o[16] = i128
	o[17] = i126
	o[18] = i124
	o[19] = i122
	o[20] = i120
	o[21] = i118
	o[22] = i116
	o[23] = i114
	o[24] = h112
	o[25] = h108
	o[26] = h104
	o[27] = h100
	o[28] = h96
	o[29] = h92
	o[30] = h88
	o[31] = h84
}

// itx64Net is inverseDCT64 on eight lanes for n nonzero inputs (8, 16 or
// 32): in[k] is coefficient k, out[r] is output r.
func itx64Net(n int, in *[32]archsimd.Int16x8, out *[64]archsimd.Int16x8) {
	var ev [32]archsimd.Int16x8
	var o [32]archsimd.Int16x8
	switch {
	case n <= 8:
		// Only coefficients 0, 2, 4, 6 reach the even half (a DCT32): its
		// Low4 network.
		e := [4]archsimd.Int16x8{in[0], in[2], in[4], in[6]}
		itx32Low4Net(&e, &ev)
		itx64Odd8(in, &o)
	case n <= 16:
		var e [8]archsimd.Int16x8
		for k := 0; k < 8; k++ {
			e[k] = in[2*k]
		}
		itx32Low8Net(&e, &ev)
		itx64Odd16(in, &o)
	default:
		var e [16]archsimd.Int16x8
		for k := 0; k < 16; k++ {
			e[k] = in[2*k]
		}
		itx32Low16Net(&e, &ev)
		itx64Odd32(in, &o)
	}
	for i := 0; i < 32; i++ {
		out[i] = ev[i].AddSaturated(o[31-i])
		out[63-i] = ev[i].SubSaturated(o[31-i])
	}
}

// itx64Form rounds a coded extent up to the network form that covers it.
func itx64Form(n int) int {
	switch {
	case n <= 8:
		return 8
	case n <= 16:
		return 16
	default:
		return 32
	}
}

// itx64RowStrip runs the 64x64 row pass for coefficient rows r0..r0+7 with
// inCols coded coefficient columns (8, 16 or 32) into tmp rows r0..r0+7 and
// adds the stored values' magnitudes to the eight column guards. It reports
// false, with tmp untouched, when a row fails the L1 guard.
func itx64RowStrip(coeff []int32, stride int, r0 int, inCols int, tmp *[32 * 64]int16, colL1 *[8]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [32]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [64]archsimd.Int16x8
	itx64Net(inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 8; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*64+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itx64ColStripAdd runs the 64x64 column pass for columns c0..c0+7 from
// inRows coded tmp rows (8, 16 or 32) and adds the result to the block.
func itx64ColStripAdd(tmp *[32 * 64]int16, c0 int, inRows int, dst unsafe.Pointer, dstStride int, pred unsafe.Pointer, predStride int) {
	var in [32]archsimd.Int16x8
	for r := 0; r < inRows; r++ {
		in[r] = archsimd.LoadInt16x8Array((*[8]int16)(tmp[r*64+c0:]))
	}
	var out [64]archsimd.Int16x8
	itx64Net(inRows, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	sh3 := archsimd.BroadcastInt16x8(-3)
	for r := 0; r < 64; r++ {
		res := itx32Round2(out[r], sh3, sh1, one)
		rows := [2]uint64{*(*uint64)(unsafe.Add(pred, r*predStride+c0)), 0}
		p := archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
		*(*uint64)(unsafe.Add(dst, r*dstStride+c0)) = p.Add(res).SaturateToUint8().ReshapeToUint64s().GetElem(0)
	}
}

// InverseAddDCT64x64 reconstructs one 8-bit 64x64 DCT_DCT block whose coded
// coefficients lie in rows 0..activeRows-1 and columns 0..activeCols-1 of
// the 32x32 corner: dst = clip(pred + inverse(coeff)), with the dequantized
// coefficients column-major at coeffStride (the 32x32 scan). pred may be
// dst. It reports false, with dst untouched, when the block is outside the
// kernel's contract or a guard declines; the caller then runs the staged
// path, which produces the same bytes.
func InverseAddDCT64x64(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 32 || activeCols <= 0 || activeCols > 32 ||
		coeffStride < 32 || len(coeff) < 31*coeffStride+32 || dstStride < 64 || predStride < 64 ||
		len(dst) < 63*dstStride+64 || len(pred) < 63*predStride+64 {
		return false
	}
	inCols := itx64Form(activeCols)
	inRows := itx64Form(activeRows)
	var tmp [32 * 64]int16
	var colL1 [8]archsimd.Uint16x8
	written := 0
	for ; written < activeRows; written += 8 {
		if !itx64RowStrip(coeff, coeffStride, written, inCols, &tmp, &colL1) {
			return false
		}
	}
	// The column form may read tmp rows past the strips that ran; those
	// rows are the transform of zero coefficient rows, so zero.
	if written < inRows {
		clear(tmp[written*64 : inRows*64])
	}
	m := colL1[0]
	for g := 1; g < 8; g++ {
		m = m.Max(colL1[g])
	}
	if m.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	dp := unsafe.Pointer(unsafe.SliceData(dst))
	pp := unsafe.Pointer(unsafe.SliceData(pred))
	for c0 := 0; c0 < 64; c0 += 8 {
		itx64ColStripAdd(&tmp, c0, inRows, dp, dstStride, pp, predStride)
	}
	return true
}
