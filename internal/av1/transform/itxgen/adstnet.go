// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

// The ADST networks, written once over the symbolic operations both
// generators provide: the stage structure of inverseADST16 and
// inverseADST8To (adst.go), with every roundShift written as the rotation
// Round2(a*ca + b*cb, 12) it equals and every clipped sum as a saturating
// int16 add.

// netOps is a generator's symbolic arithmetic over its value type V.
type netOps[V any] interface {
	rotc(a V, ca int, b V, cb int) V
	add(a, b V) V
	sub(a, b V) V
	neg(a V) V
	// stageMark names the stage the following values belong to.
	stageMark(s string)
}

// adst16Net is inverseADST16 on in[0..15].
func adst16Net[V any](g netOps[V], in []V) []V {
	rot := g.rotc
	g.stageMark("a")
	t := make([]V, 16)
	t[0] = rot(in[15], 4091, in[0], 201)
	t[1] = rot(in[15], 201, in[0], -4091)
	t[2] = rot(in[13], 3973, in[2], 995)
	t[3] = rot(in[13], 995, in[2], -3973)
	t[4] = rot(in[11], 3703, in[4], 1751)
	t[5] = rot(in[11], 1751, in[4], -3703)
	t[6] = rot(in[9], 3290, in[6], 2440)
	t[7] = rot(in[9], 2440, in[6], -3290)
	t[8] = rot(in[7], 2751, in[8], 3035)
	t[9] = rot(in[7], 3035, in[8], -2751)
	t[10] = rot(in[5], 2106, in[10], 3513)
	t[11] = rot(in[5], 3513, in[10], -2106)
	t[12] = rot(in[3], 1380, in[12], 3857)
	t[13] = rot(in[3], 3857, in[12], -1380)
	t[14] = rot(in[1], 601, in[14], 4052)
	t[15] = rot(in[1], 4052, in[14], -601)
	g.stageMark("b")
	ta := make([]V, 16)
	for i := 0; i < 8; i++ {
		ta[i] = g.add(t[i], t[i+8])
		ta[i+8] = g.sub(t[i], t[i+8])
	}
	g.stageMark("c")
	t[8] = rot(ta[8], 4017, ta[9], 799)
	t[9] = rot(ta[8], 799, ta[9], -4017)
	t[10] = rot(ta[10], 2276, ta[11], 3406)
	t[11] = rot(ta[10], 3406, ta[11], -2276)
	t[12] = rot(ta[13], 4017, ta[12], -799)
	t[13] = rot(ta[13], 799, ta[12], 4017)
	t[14] = rot(ta[15], 2276, ta[14], -3406)
	t[15] = rot(ta[15], 3406, ta[14], 2276)
	g.stageMark("d")
	for i := 0; i < 4; i++ {
		t[i] = g.add(ta[i], ta[i+4])
		t[i+4] = g.sub(ta[i], ta[i+4])
	}
	for i := 8; i < 12; i++ {
		ta[i] = g.add(t[i], t[i+4])
		ta[i+4] = g.sub(t[i], t[i+4])
	}
	g.stageMark("e")
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	t[12] = rot(ta[12], 3784, ta[13], 1567)
	t[13] = rot(ta[12], 1567, ta[13], -3784)
	t[14] = rot(ta[15], 3784, ta[14], -1567)
	t[15] = rot(ta[15], 1567, ta[14], 3784)
	g.stageMark("f")
	out := make([]V, 16)
	out[0] = g.add(t[0], t[2])
	out[15] = g.neg(g.add(t[1], t[3]))
	ta[2] = g.sub(t[0], t[2])
	ta[3] = g.sub(t[1], t[3])
	out[3] = g.neg(g.add(ta[4], ta[6]))
	out[12] = g.add(ta[5], ta[7])
	t[6] = g.sub(ta[4], ta[6])
	t[7] = g.sub(ta[5], ta[7])
	out[1] = g.neg(g.add(ta[8], ta[10]))
	out[14] = g.add(ta[9], ta[11])
	t[10] = g.sub(ta[8], ta[10])
	t[11] = g.sub(ta[9], ta[11])
	out[2] = g.add(t[12], t[14])
	out[13] = g.neg(g.add(t[13], t[15]))
	ta[14] = g.sub(t[12], t[14])
	ta[15] = g.sub(t[13], t[15])
	g.stageMark("g")
	out[7] = g.neg(rot(ta[2], 2896, ta[3], 2896))
	out[8] = rot(ta[2], 2896, ta[3], -2896)
	out[4] = rot(t[6], 2896, t[7], 2896)
	out[11] = g.neg(rot(t[6], 2896, t[7], -2896))
	out[6] = rot(t[10], 2896, t[11], 2896)
	out[9] = g.neg(rot(t[10], 2896, t[11], -2896))
	out[5] = g.neg(rot(ta[14], 2896, ta[15], 2896))
	out[10] = rot(ta[14], 2896, ta[15], -2896)
	return out
}

// adst8Net is inverseADST8To on in[0..7].
func adst8Net[V any](g netOps[V], in []V) []V {
	rot := g.rotc
	g.stageMark("a")
	ta := make([]V, 8)
	ta[0] = rot(in[7], 4076, in[0], 401)
	ta[1] = rot(in[7], 401, in[0], -4076)
	ta[2] = rot(in[5], 3612, in[2], 1931)
	ta[3] = rot(in[5], 1931, in[2], -3612)
	ta[4] = rot(in[3], 2598, in[4], 3166)
	ta[5] = rot(in[3], 3166, in[4], -2598)
	ta[6] = rot(in[1], 1189, in[6], 3920)
	ta[7] = rot(in[1], 3920, in[6], -1189)
	g.stageMark("b")
	t := make([]V, 8)
	for i := 0; i < 4; i++ {
		t[i] = g.add(ta[i], ta[i+4])
		t[i+4] = g.sub(ta[i], ta[i+4])
	}
	g.stageMark("c")
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	g.stageMark("d")
	out := make([]V, 8)
	out[0] = g.add(t[0], t[2])
	out[7] = g.neg(g.add(t[1], t[3]))
	t[2] = g.sub(t[0], t[2])
	t[3] = g.sub(t[1], t[3])
	out[1] = g.neg(g.add(ta[4], ta[6]))
	out[6] = g.add(ta[5], ta[7])
	t[6] = g.sub(ta[4], ta[6])
	t[7] = g.sub(ta[5], ta[7])
	g.stageMark("e")
	out[3] = g.neg(rot(t[2], 2896, t[3], 2896))
	out[4] = rot(t[2], 2896, t[3], -2896)
	out[2] = rot(t[6], 2896, t[7], 2896)
	out[5] = g.neg(rot(t[6], 2896, t[7], -2896))
	return out
}
