// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// itx_fused_hybrid: the fused two-pass structure of itx_fused16 and
// itx_fused8 for the 8-bit 16x16 and 8x8 blocks whose transform pairs the
// DCT with the ADST or the flipped ADST in either direction (the intra
// types). The strips are those of the DCT kernels with the one-dimensional
// network chosen per pass; a flipped ADST reverses its outputs before the
// transpose (rows) or before the add (columns), as inverseFlipADST1D does.
// The ADST networks below are generated from the stage structure of
// inverseADST16 and inverseADST8To with the zero operands of each form
// folded; itx_hybrid_gain_test.go certifies the shared per-lane L1 guard
// for them.

// itx16ADSTNet16 is inverseADST16 on eight lanes for inputs below 16 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST16 with the zero operands folded.
func itx16ADSTNet16(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot(in[15], in[0], k(2), k(62), s12)
	a2 := itx16Rot(in[15], in[0], k(62), kn(2), s12)
	a3 := itx16Rot(in[13], in[2], k(10), k(54), s12)
	a4 := itx16Rot(in[13], in[2], k(54), kn(10), s12)
	a5 := itx16Rot(in[11], in[4], k(18), k(46), s12)
	a6 := itx16Rot(in[11], in[4], k(46), kn(18), s12)
	a7 := itx16Rot(in[9], in[6], k(26), k(38), s12)
	a8 := itx16Rot(in[9], in[6], k(38), kn(26), s12)
	a9 := itx16Rot(in[7], in[8], k(34), k(30), s12)
	a10 := itx16Rot(in[7], in[8], k(30), kn(34), s12)
	a11 := itx16Rot(in[5], in[10], k(42), k(22), s12)
	a12 := itx16Rot(in[5], in[10], k(22), kn(42), s12)
	a13 := itx16Rot(in[3], in[12], k(50), k(14), s12)
	a14 := itx16Rot(in[3], in[12], k(14), kn(50), s12)
	a15 := itx16Rot(in[1], in[14], k(58), k(6), s12)
	a16 := itx16Rot(in[1], in[14], k(6), kn(58), s12)
	b17 := a1.AddSaturated(a9)
	b18 := a1.SubSaturated(a9)
	b19 := a2.AddSaturated(a10)
	b20 := a2.SubSaturated(a10)
	b21 := a3.AddSaturated(a11)
	b22 := a3.SubSaturated(a11)
	b23 := a4.AddSaturated(a12)
	b24 := a4.SubSaturated(a12)
	b25 := a5.AddSaturated(a13)
	b26 := a5.SubSaturated(a13)
	b27 := a6.AddSaturated(a14)
	b28 := a6.SubSaturated(a14)
	b29 := a7.AddSaturated(a15)
	b30 := a7.SubSaturated(a15)
	b31 := a8.AddSaturated(a16)
	b32 := a8.SubSaturated(a16)
	c33 := itx16Rot(b18, b20, k(8), k(56), s12)
	c34 := itx16Rot(b18, b20, k(56), kn(8), s12)
	c35 := itx16Rot(b22, b24, k(40), k(24), s12)
	c36 := itx16Rot(b22, b24, k(24), kn(40), s12)
	c37 := itx16Rot(b28, b26, k(8), kn(56), s12)
	c38 := itx16Rot(b28, b26, k(56), k(8), s12)
	c39 := itx16Rot(b32, b30, k(40), kn(24), s12)
	c40 := itx16Rot(b32, b30, k(24), k(40), s12)
	d41 := b17.AddSaturated(b25)
	d42 := b17.SubSaturated(b25)
	d43 := b19.AddSaturated(b27)
	d44 := b19.SubSaturated(b27)
	d45 := b21.AddSaturated(b29)
	d46 := b21.SubSaturated(b29)
	d47 := b23.AddSaturated(b31)
	d48 := b23.SubSaturated(b31)
	d49 := c33.AddSaturated(c37)
	d50 := c33.SubSaturated(c37)
	d51 := c34.AddSaturated(c38)
	d52 := c34.SubSaturated(c38)
	d53 := c35.AddSaturated(c39)
	d54 := c35.SubSaturated(c39)
	d55 := c36.AddSaturated(c40)
	d56 := c36.SubSaturated(c40)
	e57 := itx16Rot(d42, d44, k(16), k(48), s12)
	e58 := itx16Rot(d42, d44, k(48), kn(16), s12)
	e59 := itx16Rot(d48, d46, k(16), kn(48), s12)
	e60 := itx16Rot(d48, d46, k(48), k(16), s12)
	e61 := itx16Rot(d50, d52, k(16), k(48), s12)
	e62 := itx16Rot(d50, d52, k(48), kn(16), s12)
	e63 := itx16Rot(d56, d54, k(16), kn(48), s12)
	e64 := itx16Rot(d56, d54, k(48), k(16), s12)
	f65 := d41.AddSaturated(d45)
	f66 := d43.AddSaturated(d47)
	f67 := zero.SubSaturated(f66)
	f68 := d41.SubSaturated(d45)
	f69 := d43.SubSaturated(d47)
	f70 := e57.AddSaturated(e59)
	f71 := zero.SubSaturated(f70)
	f72 := e58.AddSaturated(e60)
	f73 := e57.SubSaturated(e59)
	f74 := e58.SubSaturated(e60)
	f75 := d49.AddSaturated(d53)
	f76 := zero.SubSaturated(f75)
	f77 := d51.AddSaturated(d55)
	f78 := d49.SubSaturated(d53)
	f79 := d51.SubSaturated(d55)
	f80 := e61.AddSaturated(e63)
	f81 := e62.AddSaturated(e64)
	f82 := zero.SubSaturated(f81)
	f83 := e61.SubSaturated(e63)
	f84 := e62.SubSaturated(e64)
	g85 := itx16Rot(f68, f69, k(32), k(32), s12)
	g86 := zero.SubSaturated(g85)
	g87 := itx16Rot(f68, f69, k(32), kn(32), s12)
	g88 := itx16Rot(f73, f74, k(32), k(32), s12)
	g89 := itx16Rot(f73, f74, k(32), kn(32), s12)
	g90 := zero.SubSaturated(g89)
	g91 := itx16Rot(f78, f79, k(32), k(32), s12)
	g92 := itx16Rot(f78, f79, k(32), kn(32), s12)
	g93 := zero.SubSaturated(g92)
	g94 := itx16Rot(f83, f84, k(32), k(32), s12)
	g95 := zero.SubSaturated(g94)
	g96 := itx16Rot(f83, f84, k(32), kn(32), s12)
	out[0] = f65
	out[1] = f76
	out[2] = f80
	out[3] = f71
	out[4] = g88
	out[5] = g95
	out[6] = g91
	out[7] = g86
	out[8] = g87
	out[9] = g93
	out[10] = g96
	out[11] = g90
	out[12] = f72
	out[13] = f82
	out[14] = f77
	out[15] = f67
}

// itx16ADSTNet8 is inverseADST16 on eight lanes for inputs below 8 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST16 with the zero operands folded.
func itx16ADSTNet8(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot1(in[0], k(62), s12)
	a2 := itx16Rot1(in[0], kn(2), s12)
	a3 := itx16Rot1(in[2], k(54), s12)
	a4 := itx16Rot1(in[2], kn(10), s12)
	a5 := itx16Rot1(in[4], k(46), s12)
	a6 := itx16Rot1(in[4], kn(18), s12)
	a7 := itx16Rot1(in[6], k(38), s12)
	a8 := itx16Rot1(in[6], kn(26), s12)
	a9 := itx16Rot1(in[7], k(34), s12)
	a10 := itx16Rot1(in[7], k(30), s12)
	a11 := itx16Rot1(in[5], k(42), s12)
	a12 := itx16Rot1(in[5], k(22), s12)
	a13 := itx16Rot1(in[3], k(50), s12)
	a14 := itx16Rot1(in[3], k(14), s12)
	a15 := itx16Rot1(in[1], k(58), s12)
	a16 := itx16Rot1(in[1], k(6), s12)
	b17 := a1.AddSaturated(a9)
	b18 := a1.SubSaturated(a9)
	b19 := a2.AddSaturated(a10)
	b20 := a2.SubSaturated(a10)
	b21 := a3.AddSaturated(a11)
	b22 := a3.SubSaturated(a11)
	b23 := a4.AddSaturated(a12)
	b24 := a4.SubSaturated(a12)
	b25 := a5.AddSaturated(a13)
	b26 := a5.SubSaturated(a13)
	b27 := a6.AddSaturated(a14)
	b28 := a6.SubSaturated(a14)
	b29 := a7.AddSaturated(a15)
	b30 := a7.SubSaturated(a15)
	b31 := a8.AddSaturated(a16)
	b32 := a8.SubSaturated(a16)
	c33 := itx16Rot(b18, b20, k(8), k(56), s12)
	c34 := itx16Rot(b18, b20, k(56), kn(8), s12)
	c35 := itx16Rot(b22, b24, k(40), k(24), s12)
	c36 := itx16Rot(b22, b24, k(24), kn(40), s12)
	c37 := itx16Rot(b28, b26, k(8), kn(56), s12)
	c38 := itx16Rot(b28, b26, k(56), k(8), s12)
	c39 := itx16Rot(b32, b30, k(40), kn(24), s12)
	c40 := itx16Rot(b32, b30, k(24), k(40), s12)
	d41 := b17.AddSaturated(b25)
	d42 := b17.SubSaturated(b25)
	d43 := b19.AddSaturated(b27)
	d44 := b19.SubSaturated(b27)
	d45 := b21.AddSaturated(b29)
	d46 := b21.SubSaturated(b29)
	d47 := b23.AddSaturated(b31)
	d48 := b23.SubSaturated(b31)
	d49 := c33.AddSaturated(c37)
	d50 := c33.SubSaturated(c37)
	d51 := c34.AddSaturated(c38)
	d52 := c34.SubSaturated(c38)
	d53 := c35.AddSaturated(c39)
	d54 := c35.SubSaturated(c39)
	d55 := c36.AddSaturated(c40)
	d56 := c36.SubSaturated(c40)
	e57 := itx16Rot(d42, d44, k(16), k(48), s12)
	e58 := itx16Rot(d42, d44, k(48), kn(16), s12)
	e59 := itx16Rot(d48, d46, k(16), kn(48), s12)
	e60 := itx16Rot(d48, d46, k(48), k(16), s12)
	e61 := itx16Rot(d50, d52, k(16), k(48), s12)
	e62 := itx16Rot(d50, d52, k(48), kn(16), s12)
	e63 := itx16Rot(d56, d54, k(16), kn(48), s12)
	e64 := itx16Rot(d56, d54, k(48), k(16), s12)
	f65 := d41.AddSaturated(d45)
	f66 := d43.AddSaturated(d47)
	f67 := zero.SubSaturated(f66)
	f68 := d41.SubSaturated(d45)
	f69 := d43.SubSaturated(d47)
	f70 := e57.AddSaturated(e59)
	f71 := zero.SubSaturated(f70)
	f72 := e58.AddSaturated(e60)
	f73 := e57.SubSaturated(e59)
	f74 := e58.SubSaturated(e60)
	f75 := d49.AddSaturated(d53)
	f76 := zero.SubSaturated(f75)
	f77 := d51.AddSaturated(d55)
	f78 := d49.SubSaturated(d53)
	f79 := d51.SubSaturated(d55)
	f80 := e61.AddSaturated(e63)
	f81 := e62.AddSaturated(e64)
	f82 := zero.SubSaturated(f81)
	f83 := e61.SubSaturated(e63)
	f84 := e62.SubSaturated(e64)
	g85 := itx16Rot(f68, f69, k(32), k(32), s12)
	g86 := zero.SubSaturated(g85)
	g87 := itx16Rot(f68, f69, k(32), kn(32), s12)
	g88 := itx16Rot(f73, f74, k(32), k(32), s12)
	g89 := itx16Rot(f73, f74, k(32), kn(32), s12)
	g90 := zero.SubSaturated(g89)
	g91 := itx16Rot(f78, f79, k(32), k(32), s12)
	g92 := itx16Rot(f78, f79, k(32), kn(32), s12)
	g93 := zero.SubSaturated(g92)
	g94 := itx16Rot(f83, f84, k(32), k(32), s12)
	g95 := zero.SubSaturated(g94)
	g96 := itx16Rot(f83, f84, k(32), kn(32), s12)
	out[0] = f65
	out[1] = f76
	out[2] = f80
	out[3] = f71
	out[4] = g88
	out[5] = g95
	out[6] = g91
	out[7] = g86
	out[8] = g87
	out[9] = g93
	out[10] = g96
	out[11] = g90
	out[12] = f72
	out[13] = f82
	out[14] = f77
	out[15] = f67
}

// itx16ADSTNet4 is inverseADST16 on eight lanes for inputs below 4 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST16 with the zero operands folded.
func itx16ADSTNet4(in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot1(in[0], k(62), s12)
	a2 := itx16Rot1(in[0], kn(2), s12)
	a3 := itx16Rot1(in[2], k(54), s12)
	a4 := itx16Rot1(in[2], kn(10), s12)
	a5 := itx16Rot1(in[3], k(50), s12)
	a6 := itx16Rot1(in[3], k(14), s12)
	a7 := itx16Rot1(in[1], k(58), s12)
	a8 := itx16Rot1(in[1], k(6), s12)
	b9 := zero.SubSaturated(a5)
	b10 := zero.SubSaturated(a6)
	b11 := zero.SubSaturated(a7)
	b12 := zero.SubSaturated(a8)
	c13 := itx16Rot(a1, a2, k(8), k(56), s12)
	c14 := itx16Rot(a1, a2, k(56), kn(8), s12)
	c15 := itx16Rot(a3, a4, k(40), k(24), s12)
	c16 := itx16Rot(a3, a4, k(24), kn(40), s12)
	c17 := itx16Rot(b10, b9, k(8), kn(56), s12)
	c18 := itx16Rot(b10, b9, k(56), k(8), s12)
	c19 := itx16Rot(b12, b11, k(40), kn(24), s12)
	c20 := itx16Rot(b12, b11, k(24), k(40), s12)
	d21 := a1.AddSaturated(a5)
	d22 := a1.SubSaturated(a5)
	d23 := a2.AddSaturated(a6)
	d24 := a2.SubSaturated(a6)
	d25 := a3.AddSaturated(a7)
	d26 := a3.SubSaturated(a7)
	d27 := a4.AddSaturated(a8)
	d28 := a4.SubSaturated(a8)
	d29 := c13.AddSaturated(c17)
	d30 := c13.SubSaturated(c17)
	d31 := c14.AddSaturated(c18)
	d32 := c14.SubSaturated(c18)
	d33 := c15.AddSaturated(c19)
	d34 := c15.SubSaturated(c19)
	d35 := c16.AddSaturated(c20)
	d36 := c16.SubSaturated(c20)
	e37 := itx16Rot(d22, d24, k(16), k(48), s12)
	e38 := itx16Rot(d22, d24, k(48), kn(16), s12)
	e39 := itx16Rot(d28, d26, k(16), kn(48), s12)
	e40 := itx16Rot(d28, d26, k(48), k(16), s12)
	e41 := itx16Rot(d30, d32, k(16), k(48), s12)
	e42 := itx16Rot(d30, d32, k(48), kn(16), s12)
	e43 := itx16Rot(d36, d34, k(16), kn(48), s12)
	e44 := itx16Rot(d36, d34, k(48), k(16), s12)
	f45 := d21.AddSaturated(d25)
	f46 := d23.AddSaturated(d27)
	f47 := zero.SubSaturated(f46)
	f48 := d21.SubSaturated(d25)
	f49 := d23.SubSaturated(d27)
	f50 := e37.AddSaturated(e39)
	f51 := zero.SubSaturated(f50)
	f52 := e38.AddSaturated(e40)
	f53 := e37.SubSaturated(e39)
	f54 := e38.SubSaturated(e40)
	f55 := d29.AddSaturated(d33)
	f56 := zero.SubSaturated(f55)
	f57 := d31.AddSaturated(d35)
	f58 := d29.SubSaturated(d33)
	f59 := d31.SubSaturated(d35)
	f60 := e41.AddSaturated(e43)
	f61 := e42.AddSaturated(e44)
	f62 := zero.SubSaturated(f61)
	f63 := e41.SubSaturated(e43)
	f64 := e42.SubSaturated(e44)
	g65 := itx16Rot(f48, f49, k(32), k(32), s12)
	g66 := zero.SubSaturated(g65)
	g67 := itx16Rot(f48, f49, k(32), kn(32), s12)
	g68 := itx16Rot(f53, f54, k(32), k(32), s12)
	g69 := itx16Rot(f53, f54, k(32), kn(32), s12)
	g70 := zero.SubSaturated(g69)
	g71 := itx16Rot(f58, f59, k(32), k(32), s12)
	g72 := itx16Rot(f58, f59, k(32), kn(32), s12)
	g73 := zero.SubSaturated(g72)
	g74 := itx16Rot(f63, f64, k(32), k(32), s12)
	g75 := zero.SubSaturated(g74)
	g76 := itx16Rot(f63, f64, k(32), kn(32), s12)
	out[0] = f45
	out[1] = f56
	out[2] = f60
	out[3] = f51
	out[4] = g68
	out[5] = g75
	out[6] = g71
	out[7] = g66
	out[8] = g67
	out[9] = g73
	out[10] = g76
	out[11] = g70
	out[12] = f52
	out[13] = f62
	out[14] = f57
	out[15] = f47
}

// itx8ADSTNet8 is inverseADST8 on eight lanes for inputs below 8 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST8To with the zero operands folded.
func itx8ADSTNet8(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot(in[7], in[0], k(4), k(60), s12)
	a2 := itx16Rot(in[7], in[0], k(60), kn(4), s12)
	a3 := itx16Rot(in[5], in[2], k(20), k(44), s12)
	a4 := itx16Rot(in[5], in[2], k(44), kn(20), s12)
	a5 := itx16Rot(in[3], in[4], k(36), k(28), s12)
	a6 := itx16Rot(in[3], in[4], k(28), kn(36), s12)
	a7 := itx16Rot(in[1], in[6], k(52), k(12), s12)
	a8 := itx16Rot(in[1], in[6], k(12), kn(52), s12)
	b9 := a1.AddSaturated(a5)
	b10 := a1.SubSaturated(a5)
	b11 := a2.AddSaturated(a6)
	b12 := a2.SubSaturated(a6)
	b13 := a3.AddSaturated(a7)
	b14 := a3.SubSaturated(a7)
	b15 := a4.AddSaturated(a8)
	b16 := a4.SubSaturated(a8)
	c17 := itx16Rot(b10, b12, k(16), k(48), s12)
	c18 := itx16Rot(b10, b12, k(48), kn(16), s12)
	c19 := itx16Rot(b16, b14, k(16), kn(48), s12)
	c20 := itx16Rot(b16, b14, k(48), k(16), s12)
	d21 := b9.AddSaturated(b13)
	d22 := b11.AddSaturated(b15)
	d23 := zero.SubSaturated(d22)
	d24 := b9.SubSaturated(b13)
	d25 := b11.SubSaturated(b15)
	d26 := c17.AddSaturated(c19)
	d27 := zero.SubSaturated(d26)
	d28 := c18.AddSaturated(c20)
	d29 := c17.SubSaturated(c19)
	d30 := c18.SubSaturated(c20)
	e31 := itx16Rot(d24, d25, k(32), k(32), s12)
	e32 := zero.SubSaturated(e31)
	e33 := itx16Rot(d24, d25, k(32), kn(32), s12)
	e34 := itx16Rot(d29, d30, k(32), k(32), s12)
	e35 := itx16Rot(d29, d30, k(32), kn(32), s12)
	e36 := zero.SubSaturated(e35)
	out[0] = d21
	out[1] = d27
	out[2] = e34
	out[3] = e32
	out[4] = e33
	out[5] = e36
	out[6] = d28
	out[7] = d23
}

// itx8ADSTNet4 is inverseADST8 on eight lanes for inputs below 4 (the rest
// zero): in[k] is coefficient k, out[r] output r. Generated from the stage
// structure of inverseADST8To with the zero operands folded.
func itx8ADSTNet4(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	var zero archsimd.Int16x8
	a1 := itx16Rot1(in[0], k(60), s12)
	a2 := itx16Rot1(in[0], kn(4), s12)
	a3 := itx16Rot1(in[2], k(44), s12)
	a4 := itx16Rot1(in[2], kn(20), s12)
	a5 := itx16Rot1(in[3], k(36), s12)
	a6 := itx16Rot1(in[3], k(28), s12)
	a7 := itx16Rot1(in[1], k(52), s12)
	a8 := itx16Rot1(in[1], k(12), s12)
	b9 := a1.AddSaturated(a5)
	b10 := a1.SubSaturated(a5)
	b11 := a2.AddSaturated(a6)
	b12 := a2.SubSaturated(a6)
	b13 := a3.AddSaturated(a7)
	b14 := a3.SubSaturated(a7)
	b15 := a4.AddSaturated(a8)
	b16 := a4.SubSaturated(a8)
	c17 := itx16Rot(b10, b12, k(16), k(48), s12)
	c18 := itx16Rot(b10, b12, k(48), kn(16), s12)
	c19 := itx16Rot(b16, b14, k(16), kn(48), s12)
	c20 := itx16Rot(b16, b14, k(48), k(16), s12)
	d21 := b9.AddSaturated(b13)
	d22 := b11.AddSaturated(b15)
	d23 := zero.SubSaturated(d22)
	d24 := b9.SubSaturated(b13)
	d25 := b11.SubSaturated(b15)
	d26 := c17.AddSaturated(c19)
	d27 := zero.SubSaturated(d26)
	d28 := c18.AddSaturated(c20)
	d29 := c17.SubSaturated(c19)
	d30 := c18.SubSaturated(c20)
	e31 := itx16Rot(d24, d25, k(32), k(32), s12)
	e32 := zero.SubSaturated(e31)
	e33 := itx16Rot(d24, d25, k(32), kn(32), s12)
	e34 := itx16Rot(d29, d30, k(32), k(32), s12)
	e35 := itx16Rot(d29, d30, k(32), kn(32), s12)
	e36 := zero.SubSaturated(e35)
	out[0] = d21
	out[1] = d27
	out[2] = e34
	out[3] = e32
	out[4] = e33
	out[5] = e36
	out[6] = d28
	out[7] = d23
}

// itx16Net1D runs the 16-point network of t for n nonzero inputs (4, 8 or
// 16), reversing the outputs for the flipped ADST.
func itx16Net1D(t tx1DType, n int, in *[16]archsimd.Int16x8, out *[16]archsimd.Int16x8) {
	if t == tx1DDCT {
		itx16Net(n, in, out)
		return
	}
	switch {
	case n <= 4:
		itx16ADSTNet4(in, out)
	case n <= 8:
		itx16ADSTNet8(in, out)
	default:
		itx16ADSTNet16(in, out)
	}
	if t == tx1DFlipADST {
		for i := 0; i < 8; i++ {
			out[i], out[15-i] = out[15-i], out[i]
		}
	}
}

// itx8Net1D is itx16Net1D for the 8-point networks (n is 4 or 8).
func itx8Net1D(t tx1DType, n int, in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	if t == tx1DDCT {
		itx8Net(in, out)
		return
	}
	if n <= 4 {
		itx8ADSTNet4(in, out)
	} else {
		itx8ADSTNet8(in, out)
	}
	if t == tx1DFlipADST {
		for i := 0; i < 4; i++ {
			out[i], out[7-i] = out[7-i], out[i]
		}
	}
}

// itx16RowStripT is itx16RowStrip with the row network t.
func itx16RowStripT(coeff []int32, stride int, r0 int, inCols int, t tx1DType, tmp *[16 * 16]int16, colL1 *[2]archsimd.Uint16x8) bool {
	_ = coeff[(inCols-1)*stride+r0+7]
	var in [16]archsimd.Int16x8
	var l1 archsimd.Uint16x8
	for k := 0; k < inCols; k++ {
		x := itx16LoadNarrow(coeff[k*stride+r0 : k*stride+r0+8])
		l1 = l1.AddSaturated(x.Abs().ToBits())
		in[k] = x
	}
	if l1.ReduceMax() > itx16ColumnL1Bound {
		return false
	}
	var out [16]archsimd.Int16x8
	itx16Net1D(t, inCols, &in, &out)
	one := archsimd.BroadcastInt16x8(1)
	sh1 := archsimd.BroadcastInt16x8(-1)
	for g := 0; g < 2; g++ {
		var v [8]archsimd.Int16x8
		for j := 0; j < 8; j++ {
			v[j] = itx32Round2(out[8*g+j], sh1, sh1, one)
		}
		itx16Transpose8x8(&v)
		gl1 := colL1[g]
		for r := 0; r < 8; r++ {
			v[r].StoreArray((*[8]int16)(tmp[(r0+r)*16+8*g:]))
			gl1 = gl1.AddSaturated(v[r].Abs().ToBits())
		}
		colL1[g] = gl1
	}
	return true
}

// itxHybridTypes returns the per-pass networks of the DCT/ADST/flipped-ADST
// pairings the fused hybrid kernels cover; ok is false for every other
// transform type (including DCT_DCT, which has its own kernels).
func itxHybridTypes(txType Type) (vertical, horizontal tx1DType, ok bool) {
	switch txType {
	case TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
		TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST:
		return txType.tx1DTypes()
	}
	return 0, 0, false
}

// InverseAdd16x16Hybrid reconstructs one 8-bit 16x16 block of a
// DCT/ADST/flipped-ADST/identity transform pairing other than DCT_DCT and
// IDTX with coded coefficients in rows 0..activeRows-1 and columns
// 0..activeCols-1: dst = clip(pred + inverse(coeff)), with the dequantized
// coefficients column-major at coeffStride (16). pred may be dst. It
// reports false, with dst untouched, when the block is outside the
// kernel's contract or a guard declines; the caller then runs the staged
// path, which produces the same bytes. The kernels are generated
// (itx_fused_gen_arm64.go) per type and network form.
func InverseAdd16x16Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 16 || activeCols <= 0 || activeCols > 16 ||
		coeffStride != 16 || len(coeff) < 256 || dstStride < 16 || predStride < 16 ||
		len(dst) < 15*dstStride+16 || len(pred) < 15*predStride+16 {
		return false
	}
	return itxGen16x16Hybrid((*[256]int32)(coeff[:256]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd8x8Hybrid is InverseAdd16x16Hybrid for the 8x8 block (every
// pairing but DCT_DCT and IDTX; coeffStride 8).
func InverseAdd8x8Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if activeRows <= 0 || activeRows > 8 || activeCols <= 0 || activeCols > 8 ||
		coeffStride != 8 || len(coeff) < 64 || dstStride < 8 || predStride < 8 ||
		len(dst) < 7*dstStride+8 || len(pred) < 7*predStride+8 {
		return false
	}
	return itxGen8x8Hybrid((*[64]int32)(coeff[:64]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// The 4x4 and the 4- and 8/16-point rectangular 8-bit blocks of every type
// but IDTX: the generated kernels (itx_fused_gen_arm64.go), DCT_DCT per
// network form of the coded columns and rows, every other type in the
// full form.

// itxSmallFits reports whether a w x h block's buffers satisfy the
// generated kernels' contract: coefficients column-major at stride h, dst
// and pred rows of w bytes at their strides.
func itxSmallFits(w, h int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride, activeRows, activeCols int) bool {
	return activeRows > 0 && activeRows <= h && activeCols > 0 && activeCols <= w &&
		coeffStride == h && len(coeff) >= w*h && dstStride >= w && predStride >= w &&
		len(dst) >= (h-1)*dstStride+w && len(pred) >= (h-1)*predStride+w
}

// InverseAdd4x4 reconstructs one 8-bit 4x4 block of any type but IDTX:
// dst = clip(pred + inverse(coeff)), with the dequantized coefficients
// column-major at coeffStride (4). pred may be dst. It reports false, with
// dst untouched, when the block is outside the kernels' contract or a
// guard declines; the caller then runs the staged path, which produces the
// same bytes.
func InverseAdd4x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows int) bool {
	if !itxSmallFits(4, 4, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, 4) {
		return false
	}
	return itxGen4x4((*[16]int32)(coeff[:16]), dst, dstStride, pred, predStride, txType, 4, activeRows)
}

// InverseAdd4x8 is InverseAdd4x4 for the 4 wide, 8 high block.
func InverseAdd4x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(4, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen4x8((*[32]int32)(coeff[:32]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd8x4 is InverseAdd4x4 for the 8 wide, 4 high block.
func InverseAdd8x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(8, 4, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen8x4((*[32]int32)(coeff[:32]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd4x16 is InverseAdd4x4 for the 4 wide, 16 high block.
func InverseAdd4x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(4, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen4x16((*[64]int32)(coeff[:64]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd16x4 is InverseAdd4x4 for the 16 wide, 4 high block.
func InverseAdd16x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(16, 4, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen16x4((*[64]int32)(coeff[:64]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd8x16 is InverseAdd4x4 for the 8 wide, 16 high block.
func InverseAdd8x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(8, 16, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen8x16((*[128]int32)(coeff[:128]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}

// InverseAdd16x8 is InverseAdd4x4 for the 16 wide, 8 high block.
func InverseAdd16x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	if !itxSmallFits(16, 8, dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols) {
		return false
	}
	return itxGen16x8((*[128]int32)(coeff[:128]), dst, dstStride, pred, predStride, txType, activeCols, activeRows)
}
