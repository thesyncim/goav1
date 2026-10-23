// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// itx8NetLow4 is inverseDCT8 on eight lanes for inputs whose indices 4..7
// are zero: in[k] is coefficient k, out[r] output r. Generated from the
// network of itx8Net with the zero operands folded.
func itx8NetLow4(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {
	k := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }
	kn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }
	s12 := archsimd.LoadInt32x4Array(&itx16Scale12)
	a1 := itx16Rot1(in[0], k(32), s12)
	a2 := itx16Rot1(in[2], k(48), s12)
	a3 := itx16Rot1(in[2], k(16), s12)
	b4 := a1.AddSaturated(a3)
	b5 := a1.AddSaturated(a2)
	b6 := a1.SubSaturated(a2)
	b7 := a1.SubSaturated(a3)
	c8 := itx16Rot1(in[1], k(56), s12)
	c9 := itx16Rot1(in[3], kn(40), s12)
	c10 := itx16Rot1(in[3], k(24), s12)
	c11 := itx16Rot1(in[1], k(8), s12)
	d12 := c8.AddSaturated(c9)
	d13 := c8.SubSaturated(c9)
	d14 := c11.AddSaturated(c10)
	d15 := c11.SubSaturated(c10)
	e16 := itx16Rot(d15, d13, k(32), kn(32), s12)
	e17 := itx16Rot(d15, d13, k(32), k(32), s12)
	f18 := b4.AddSaturated(d14)
	f19 := b5.AddSaturated(e17)
	f20 := b6.AddSaturated(e16)
	f21 := b7.AddSaturated(d12)
	f22 := b7.SubSaturated(d12)
	f23 := b6.SubSaturated(e16)
	f24 := b5.SubSaturated(e17)
	f25 := b4.SubSaturated(d14)
	out[0] = f18
	out[1] = f19
	out[2] = f20
	out[3] = f21
	out[4] = f22
	out[5] = f23
	out[6] = f24
	out[7] = f25
}
