// avx2gen emits the four-lane AVX2 inverse-DCT32/DCT64 column/row kernels for
// amd64 (internal/av1/transform/dct4lane_avx2_amd64.{s,go}).
//
// It is the amd64/AVX2 sibling of tools/itxgen (which produced the arm64 NEON
// four-lane kernels). Where the NEON path packs four columns into the four
// int32 lanes of a .4s register and rewrites |w|>2048 as w-4096 to keep the
// int32 accumulators from overflowing, the AVX2 path is simpler: it packs four
// rows/columns into the four int64 lanes of a YMM register and multiplies with
// VPMULDQ (signed 32x32 -> 64). Every multiplicand in the AV1 inverse DCT is
// either a raw input coefficient or a clipRange'd stage value, both clamped to
// the +/-2^19 stage-range envelope, so the low-32-bit VPMULDQ products are
// exact and the whole butterfly is bit-for-bit identical to the pure-Go
// reference (inverseDCT32 / inverseDCT64 with the matching stride). The
// rounding shift and per-stage clamp reuse the same AVX2-only integer
// emulation as the existing dct_avx2_amd64.s (roundShift(x,N) via a +2^62
// bias; clamp via VPCMPGTQ+VPBLENDVB).
//
// dav1d reference shape: third_party/upstream/dav1d/src/x86/itx_avx2.asm,
// inv_txfm_add_dct_dct_32x32_8bpc / _64x64_8bpc and the inv_dct32 / inv_dct64
// butterfly macros (the same staged half-butterfly structure this transliterates
// from libaom's av1_idct32 / av1_idct64, which the pure-Go port follows).
//
// SPDX-License-Identifier: BSD-2-Clause
package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
)

// ---------------------------------------------------------------------------
// IR
// ---------------------------------------------------------------------------

// vref is an operand: either a mutable c-slot (memory location, index 0..N-1,
// matching pure-Go's in-place c[] array) or an SSA temporary (a vreg id).
type vref struct {
	isC bool
	id  int
}

type opKind int

const (
	opAdd   opKind = iota // dst = a + b
	opSub                 // dst = a - b
	opClamp               // dst = clamp(a)
	opRsMul               // dst = roundShift((a op b)*w0, n)
	opHB                  // dst = roundShift(a*w0 + b*w1, n)
	opOut                 // c-slot ci = a  (store)
	opMov                 // dst = a  (snapshot a c-slot into a temp)
)

type op struct {
	kind       opKind
	dst        int // temp vreg id (producer ops)
	a, b       vref
	w0, w1, n  int
	subOp      bool // for opRsMul: true => a-b, false => a+b
	ci         int  // for opOut: destination c-slot
}

type gen struct {
	n     int // number of c-slots (transform length)
	ops   []op
	nextV int
	// collected constants / shifts
	weights map[int]bool
	shifts  map[int]bool
}

func newGen(n int) *gen {
	return &gen{n: n, weights: map[int]bool{}, shifts: map[int]bool{}}
}

func (g *gen) cin(i int) vref { return vref{isC: true, id: i} }

// load snapshots c-slot i into a fresh temp. The even-half output combines in
// dct8/dct16/dct32 read the transformed even coefficients while the output
// stores overwrite those same c-slots, so the reads must be captured first
// (matching pure-Go's "t0 := int64(c[0*stride])" locals).
func (g *gen) load(i int) vref {
	d := g.newTemp()
	g.ops = append(g.ops, op{kind: opMov, dst: d.id, a: g.cin(i)})
	return d
}

func (g *gen) newTemp() vref {
	id := g.nextV
	g.nextV++
	return vref{isC: false, id: id}
}

func (g *gen) add(a, b vref) vref {
	d := g.newTemp()
	g.ops = append(g.ops, op{kind: opAdd, dst: d.id, a: a, b: b})
	return d
}

func (g *gen) sub(a, b vref) vref {
	d := g.newTemp()
	g.ops = append(g.ops, op{kind: opSub, dst: d.id, a: a, b: b})
	return d
}

func (g *gen) clampv(a vref) vref {
	d := g.newTemp()
	g.ops = append(g.ops, op{kind: opClamp, dst: d.id, a: a})
	return d
}

// rsmul emits roundShift((a op b)*w, n).
func (g *gen) rsmul(a vref, subtract bool, b vref, w, n int) vref {
	g.weights[w] = true
	g.shifts[n] = true
	d := g.newTemp()
	g.ops = append(g.ops, op{kind: opRsMul, dst: d.id, a: a, b: b, w0: w, n: n, subOp: subtract})
	return d
}

// hb emits roundShift(x0*w0 + x1*w1, n) (libaom half_btf).
func (g *gen) hb(w0 int, x0 vref, w1 int, x1 vref, n int) vref {
	g.weights[w0] = true
	g.weights[w1] = true
	g.shifts[n] = true
	d := g.newTemp()
	g.ops = append(g.ops, op{kind: opHB, dst: d.id, a: x0, b: x1, w0: w0, w1: w1, n: n})
	return d
}

func (g *gen) out(i int, a vref) {
	g.ops = append(g.ops, op{kind: opOut, ci: i, a: a})
}

// ---------------------------------------------------------------------------
// Butterfly transliterations (verbatim from internal/av1/transform/dct.go).
// c is the list of c-slot indices for this (sub-)transform's c[0..len-1].
// ---------------------------------------------------------------------------

func (g *gen) dct4(c []int) {
	in0, in1, in2, in3 := g.cin(c[0]), g.cin(c[1]), g.cin(c[2]), g.cin(c[3])
	t0 := g.rsmul(in0, false, in2, 181, 8)
	t1 := g.rsmul(in0, true, in2, 181, 8)
	t2 := g.sub(g.hb(1567, in1, -(3784 - 4096), in3, 12), in3)
	t3 := g.add(g.hb((3784 - 4096), in1, 1567, in3, 12), in1)
	o0 := g.clampv(g.add(t0, t3))
	o1 := g.clampv(g.add(t1, t2))
	o2 := g.clampv(g.sub(t1, t2))
	o3 := g.clampv(g.sub(t0, t3))
	g.out(c[0], o0)
	g.out(c[1], o1)
	g.out(c[2], o2)
	g.out(c[3], o3)
}

func (g *gen) dct8(c []int) {
	g.dct4([]int{c[0], c[2], c[4], c[6]})
	in1, in3, in5, in7 := g.cin(c[1]), g.cin(c[3]), g.cin(c[5]), g.cin(c[7])
	t4a := g.sub(g.hb(799, in1, -(4017 - 4096), in7, 12), in7)
	t5a := g.hb(1703, in5, -1138, in3, 11)
	t6a := g.hb(1138, in5, 1703, in3, 11)
	t7a := g.add(g.hb((4017 - 4096), in1, 799, in7, 12), in1)
	t4 := g.clampv(g.add(t4a, t5a))
	t5a = g.clampv(g.sub(t4a, t5a))
	t7 := g.clampv(g.add(t7a, t6a))
	t6a = g.clampv(g.sub(t7a, t6a))
	t5 := g.rsmul(t6a, true, t5a, 181, 8)
	t6 := g.rsmul(t6a, false, t5a, 181, 8)
	t0 := g.load(c[0])
	t1 := g.load(c[2])
	t2 := g.load(c[4])
	t3 := g.load(c[6])
	g.out(c[0], g.clampv(g.add(t0, t7)))
	g.out(c[1], g.clampv(g.add(t1, t6)))
	g.out(c[2], g.clampv(g.add(t2, t5)))
	g.out(c[3], g.clampv(g.add(t3, t4)))
	g.out(c[4], g.clampv(g.sub(t3, t4)))
	g.out(c[5], g.clampv(g.sub(t2, t5)))
	g.out(c[6], g.clampv(g.sub(t1, t6)))
	g.out(c[7], g.clampv(g.sub(t0, t7)))
}

func (g *gen) dct16(c []int) {
	g.dct8([]int{c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]})
	in1, in3, in5, in7 := g.cin(c[1]), g.cin(c[3]), g.cin(c[5]), g.cin(c[7])
	in9, in11, in13, in15 := g.cin(c[9]), g.cin(c[11]), g.cin(c[13]), g.cin(c[15])
	t8a := g.sub(g.hb(401, in1, -(4076 - 4096), in15, 12), in15)
	t9a := g.hb(1583, in9, -1299, in7, 11)
	t10a := g.sub(g.hb(1931, in5, -(3612 - 4096), in11, 12), in11)
	t11a := g.add(g.hb((3920 - 4096), in13, -1189, in3, 12), in13)
	t12a := g.add(g.hb(1189, in13, (3920 - 4096), in3, 12), in3)
	t13a := g.add(g.hb((3612 - 4096), in5, 1931, in11, 12), in5)
	t14a := g.hb(1299, in9, 1583, in7, 11)
	t15a := g.add(g.hb((4076 - 4096), in1, 401, in15, 12), in1)
	t8 := g.clampv(g.add(t8a, t9a))
	t9 := g.clampv(g.sub(t8a, t9a))
	t10 := g.clampv(g.sub(t11a, t10a))
	t11 := g.clampv(g.add(t11a, t10a))
	t12 := g.clampv(g.add(t12a, t13a))
	t13 := g.clampv(g.sub(t12a, t13a))
	t14 := g.clampv(g.sub(t15a, t14a))
	t15 := g.clampv(g.add(t15a, t14a))
	t9a = g.sub(g.hb(1567, t14, -(3784-4096), t9, 12), t9)
	t14a = g.add(g.hb((3784-4096), t14, 1567, t9, 12), t14)
	t10a = g.sub(g.hb(-(3784-4096), t13, -1567, t10, 12), t13)
	t13a = g.sub(g.hb(1567, t13, -(3784-4096), t10, 12), t10)
	t8a = g.clampv(g.add(t8, t11))
	t9 = g.clampv(g.add(t9a, t10a))
	t10 = g.clampv(g.sub(t9a, t10a))
	t11a = g.clampv(g.sub(t8, t11))
	t12a = g.clampv(g.sub(t15, t12))
	t13 = g.clampv(g.sub(t14a, t13a))
	t14 = g.clampv(g.add(t14a, t13a))
	t15a = g.clampv(g.add(t15, t12))
	t10a = g.rsmul(t13, true, t10, 181, 8)
	t13a = g.rsmul(t13, false, t10, 181, 8)
	t11 = g.rsmul(t12a, true, t11a, 181, 8)
	t12 = g.rsmul(t12a, false, t11a, 181, 8)
	t0 := g.load(c[0])
	t1 := g.load(c[2])
	t2 := g.load(c[4])
	t3 := g.load(c[6])
	t4 := g.load(c[8])
	t5 := g.load(c[10])
	t6 := g.load(c[12])
	t7 := g.load(c[14])
	g.out(c[0], g.clampv(g.add(t0, t15a)))
	g.out(c[1], g.clampv(g.add(t1, t14)))
	g.out(c[2], g.clampv(g.add(t2, t13a)))
	g.out(c[3], g.clampv(g.add(t3, t12)))
	g.out(c[4], g.clampv(g.add(t4, t11)))
	g.out(c[5], g.clampv(g.add(t5, t10a)))
	g.out(c[6], g.clampv(g.add(t6, t9)))
	g.out(c[7], g.clampv(g.add(t7, t8a)))
	g.out(c[8], g.clampv(g.sub(t7, t8a)))
	g.out(c[9], g.clampv(g.sub(t6, t9)))
	g.out(c[10], g.clampv(g.sub(t5, t10a)))
	g.out(c[11], g.clampv(g.sub(t4, t11)))
	g.out(c[12], g.clampv(g.sub(t3, t12)))
	g.out(c[13], g.clampv(g.sub(t2, t13a)))
	g.out(c[14], g.clampv(g.sub(t1, t14)))
	g.out(c[15], g.clampv(g.sub(t0, t15a)))
}

func (g *gen) dct32(c []int) {
	even := make([]int, 16)
	for i := 0; i < 16; i++ {
		even[i] = c[2*i]
	}
	g.dct16(even)

	in1, in3, in5, in7 := g.cin(c[1]), g.cin(c[3]), g.cin(c[5]), g.cin(c[7])
	in9, in11, in13, in15 := g.cin(c[9]), g.cin(c[11]), g.cin(c[13]), g.cin(c[15])
	in17, in19, in21, in23 := g.cin(c[17]), g.cin(c[19]), g.cin(c[21]), g.cin(c[23])
	in25, in27, in29, in31 := g.cin(c[25]), g.cin(c[27]), g.cin(c[29]), g.cin(c[31])

	t16a := g.sub(g.hb(201, in1, -(4091-4096), in31, 12), in31)
	t17a := g.add(g.hb((3035-4096), in17, -2751, in15, 12), in17)
	t18a := g.sub(g.hb(1751, in9, -(3703-4096), in23, 12), in23)
	t19a := g.add(g.hb((3857-4096), in25, -1380, in7, 12), in25)
	t20a := g.sub(g.hb(995, in5, -(3973-4096), in27, 12), in27)
	t21a := g.add(g.hb((3513-4096), in21, -2106, in11, 12), in21)
	t22a := g.hb(1220, in13, -1645, in19, 11)
	t23a := g.add(g.hb((4052-4096), in29, -601, in3, 12), in29)
	t24a := g.add(g.hb(601, in29, (4052-4096), in3, 12), in3)
	t25a := g.hb(1645, in13, 1220, in19, 11)
	t26a := g.add(g.hb(2106, in21, (3513-4096), in11, 12), in11)
	t27a := g.add(g.hb((3973-4096), in5, 995, in27, 12), in5)
	t28a := g.add(g.hb(1380, in25, (3857-4096), in7, 12), in7)
	t29a := g.add(g.hb((3703-4096), in9, 1751, in23, 12), in9)
	t30a := g.add(g.hb(2751, in17, (3035-4096), in15, 12), in15)
	t31a := g.add(g.hb((4091-4096), in1, 201, in31, 12), in1)

	t16 := g.clampv(g.add(t16a, t17a))
	t17 := g.clampv(g.sub(t16a, t17a))
	t18 := g.clampv(g.sub(t19a, t18a))
	t19 := g.clampv(g.add(t19a, t18a))
	t20 := g.clampv(g.add(t20a, t21a))
	t21 := g.clampv(g.sub(t20a, t21a))
	t22 := g.clampv(g.sub(t23a, t22a))
	t23 := g.clampv(g.add(t23a, t22a))
	t24 := g.clampv(g.add(t24a, t25a))
	t25 := g.clampv(g.sub(t24a, t25a))
	t26 := g.clampv(g.sub(t27a, t26a))
	t27 := g.clampv(g.add(t27a, t26a))
	t28 := g.clampv(g.add(t28a, t29a))
	t29 := g.clampv(g.sub(t28a, t29a))
	t30 := g.clampv(g.sub(t31a, t30a))
	t31 := g.clampv(g.add(t31a, t30a))

	t17a = g.sub(g.hb(799, t30, -(4017-4096), t17, 12), t17)
	t30a = g.add(g.hb((4017-4096), t30, 799, t17, 12), t30)
	t18a = g.sub(g.hb(-(4017-4096), t29, -799, t18, 12), t29)
	t29a = g.sub(g.hb(799, t29, -(4017-4096), t18, 12), t18)
	t21a = g.hb(1703, t26, -1138, t21, 11)
	t26a = g.hb(1138, t26, 1703, t21, 11)
	t22a = g.hb(-1138, t25, -1703, t22, 11)
	t25a = g.hb(1703, t25, -1138, t22, 11)

	t16a = g.clampv(g.add(t16, t19))
	t17 = g.clampv(g.add(t17a, t18a))
	t18 = g.clampv(g.sub(t17a, t18a))
	t19a = g.clampv(g.sub(t16, t19))
	t20a = g.clampv(g.sub(t23, t20))
	t21 = g.clampv(g.sub(t22a, t21a))
	t22 = g.clampv(g.add(t22a, t21a))
	t23a = g.clampv(g.add(t23, t20))
	t24a = g.clampv(g.add(t24, t27))
	t25 = g.clampv(g.add(t25a, t26a))
	t26 = g.clampv(g.sub(t25a, t26a))
	t27a = g.clampv(g.sub(t24, t27))
	t28a = g.clampv(g.sub(t31, t28))
	t29 = g.clampv(g.sub(t30a, t29a))
	t30 = g.clampv(g.add(t30a, t29a))
	t31a = g.clampv(g.add(t31, t28))

	t18a = g.sub(g.hb(1567, t29, -(3784-4096), t18, 12), t18)
	t29a = g.add(g.hb((3784-4096), t29, 1567, t18, 12), t29)
	t19 = g.sub(g.hb(1567, t28a, -(3784-4096), t19a, 12), t19a)
	t28 = g.add(g.hb((3784-4096), t28a, 1567, t19a, 12), t28a)
	t20 = g.sub(g.hb(-(3784-4096), t27a, -1567, t20a, 12), t27a)
	t27 = g.sub(g.hb(1567, t27a, -(3784-4096), t20a, 12), t20a)
	t21a = g.sub(g.hb(-(3784-4096), t26, -1567, t21, 12), t26)
	t26a = g.sub(g.hb(1567, t26, -(3784-4096), t21, 12), t21)

	t16 = g.clampv(g.add(t16a, t23a))
	t17a = g.clampv(g.add(t17, t22))
	t18 = g.clampv(g.add(t18a, t21a))
	t19a = g.clampv(g.add(t19, t20))
	t20a = g.clampv(g.sub(t19, t20))
	t21 = g.clampv(g.sub(t18a, t21a))
	t22a = g.clampv(g.sub(t17, t22))
	t23 = g.clampv(g.sub(t16a, t23a))
	t24 = g.clampv(g.sub(t31a, t24a))
	t25a = g.clampv(g.sub(t30, t25))
	t26 = g.clampv(g.sub(t29a, t26a))
	t27a = g.clampv(g.sub(t28, t27))
	t28a = g.clampv(g.add(t28, t27))
	t29 = g.clampv(g.add(t29a, t26a))
	t30a = g.clampv(g.add(t30, t25))
	t31 = g.clampv(g.add(t31a, t24a))

	t20 = g.rsmul(t27a, true, t20a, 181, 8)
	t27 = g.rsmul(t27a, false, t20a, 181, 8)
	t21a = g.rsmul(t26, true, t21, 181, 8)
	t26a = g.rsmul(t26, false, t21, 181, 8)
	t22 = g.rsmul(t25a, true, t22a, 181, 8)
	t25 = g.rsmul(t25a, false, t22a, 181, 8)
	t23a = g.rsmul(t24, true, t23, 181, 8)
	t24a = g.rsmul(t24, false, t23, 181, 8)

	t0 := g.load(c[0])
	t1 := g.load(c[2])
	t2 := g.load(c[4])
	t3 := g.load(c[6])
	t4 := g.load(c[8])
	t5 := g.load(c[10])
	t6 := g.load(c[12])
	t7 := g.load(c[14])
	t8 := g.load(c[16])
	t9 := g.load(c[18])
	t10 := g.load(c[20])
	t11 := g.load(c[22])
	t12 := g.load(c[24])
	t13 := g.load(c[26])
	t14 := g.load(c[28])
	t15 := g.load(c[30])

	g.out(c[0], g.clampv(g.add(t0, t31)))
	g.out(c[1], g.clampv(g.add(t1, t30a)))
	g.out(c[2], g.clampv(g.add(t2, t29)))
	g.out(c[3], g.clampv(g.add(t3, t28a)))
	g.out(c[4], g.clampv(g.add(t4, t27)))
	g.out(c[5], g.clampv(g.add(t5, t26a)))
	g.out(c[6], g.clampv(g.add(t6, t25)))
	g.out(c[7], g.clampv(g.add(t7, t24a)))
	g.out(c[8], g.clampv(g.add(t8, t23a)))
	g.out(c[9], g.clampv(g.add(t9, t22)))
	g.out(c[10], g.clampv(g.add(t10, t21a)))
	g.out(c[11], g.clampv(g.add(t11, t20)))
	g.out(c[12], g.clampv(g.add(t12, t19a)))
	g.out(c[13], g.clampv(g.add(t13, t18)))
	g.out(c[14], g.clampv(g.add(t14, t17a)))
	g.out(c[15], g.clampv(g.add(t15, t16)))
	g.out(c[16], g.clampv(g.sub(t15, t16)))
	g.out(c[17], g.clampv(g.sub(t14, t17a)))
	g.out(c[18], g.clampv(g.sub(t13, t18)))
	g.out(c[19], g.clampv(g.sub(t12, t19a)))
	g.out(c[20], g.clampv(g.sub(t11, t20)))
	g.out(c[21], g.clampv(g.sub(t10, t21a)))
	g.out(c[22], g.clampv(g.sub(t9, t22)))
	g.out(c[23], g.clampv(g.sub(t8, t23a)))
	g.out(c[24], g.clampv(g.sub(t7, t24a)))
	g.out(c[25], g.clampv(g.sub(t6, t25)))
	g.out(c[26], g.clampv(g.sub(t5, t26a)))
	g.out(c[27], g.clampv(g.sub(t4, t27)))
	g.out(c[28], g.clampv(g.sub(t3, t28a)))
	g.out(c[29], g.clampv(g.sub(t2, t29)))
	g.out(c[30], g.clampv(g.sub(t1, t30a)))
	g.out(c[31], g.clampv(g.sub(t0, t31)))
}

// dct64 transliterates inverseDCT64 (dct.go). bf0/bf1 hold vrefs; "bf0 = bf1"
// aliases the vref arrays (matching the snapshot); each "bf1[i] = ..." rebinds
// to a fresh SSA temp. cospi weights are the full libaom constants.
func (g *gen) dct64(c []int) {
	const (
		cospi1  = 4095
		cospi2  = 4091
		cospi3  = 4085
		cospi4  = 4076
		cospi5  = 4065
		cospi6  = 4052
		cospi7  = 4036
		cospi8  = 4017
		cospi9  = 3996
		cospi10 = 3973
		cospi11 = 3948
		cospi12 = 3920
		cospi13 = 3889
		cospi14 = 3857
		cospi15 = 3822
		cospi16 = 3784
		cospi17 = 3745
		cospi18 = 3703
		cospi19 = 3659
		cospi20 = 3612
		cospi21 = 3564
		cospi22 = 3513
		cospi23 = 3461
		cospi24 = 3406
		cospi25 = 3349
		cospi26 = 3290
		cospi27 = 3229
		cospi28 = 3166
		cospi29 = 3102
		cospi30 = 3035
		cospi31 = 2967
		cospi32 = 2896
		cospi33 = 2824
		cospi34 = 2751
		cospi35 = 2675
		cospi36 = 2598
		cospi37 = 2520
		cospi38 = 2440
		cospi39 = 2359
		cospi40 = 2276
		cospi41 = 2191
		cospi42 = 2106
		cospi43 = 2019
		cospi44 = 1931
		cospi45 = 1842
		cospi46 = 1751
		cospi47 = 1660
		cospi48 = 1567
		cospi49 = 1474
		cospi50 = 1380
		cospi51 = 1285
		cospi52 = 1189
		cospi53 = 1092
		cospi54 = 995
		cospi55 = 897
		cospi56 = 799
		cospi57 = 700
		cospi58 = 601
		cospi59 = 501
		cospi60 = 401
		cospi61 = 301
		cospi62 = 201
		cospi63 = 101
	)
	hbtf := func(w0 int, in0 vref, w1 int, in1 vref) vref { return g.hb(w0, in0, w1, in1, 12) }
	clamp := func(v vref) vref { return g.clampv(v) }
	add := g.add
	sub := g.sub

	var bf0, bf1 [64]vref

	perm := []int{0, 32, 16, 48, 8, 40, 24, 56, 4, 36, 20, 52, 12, 44, 28, 60,
		2, 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62,
		1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61,
		3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 23, 55, 15, 47, 31, 63}
	for i := 0; i < 64; i++ {
		bf1[i] = g.cin(c[perm[i]])
	}

	// stage 2
	bf0 = bf1
	bf1[32] = hbtf(cospi63, bf0[32], -cospi1, bf0[63])
	bf1[33] = hbtf(cospi31, bf0[33], -cospi33, bf0[62])
	bf1[34] = hbtf(cospi47, bf0[34], -cospi17, bf0[61])
	bf1[35] = hbtf(cospi15, bf0[35], -cospi49, bf0[60])
	bf1[36] = hbtf(cospi55, bf0[36], -cospi9, bf0[59])
	bf1[37] = hbtf(cospi23, bf0[37], -cospi41, bf0[58])
	bf1[38] = hbtf(cospi39, bf0[38], -cospi25, bf0[57])
	bf1[39] = hbtf(cospi7, bf0[39], -cospi57, bf0[56])
	bf1[40] = hbtf(cospi59, bf0[40], -cospi5, bf0[55])
	bf1[41] = hbtf(cospi27, bf0[41], -cospi37, bf0[54])
	bf1[42] = hbtf(cospi43, bf0[42], -cospi21, bf0[53])
	bf1[43] = hbtf(cospi11, bf0[43], -cospi53, bf0[52])
	bf1[44] = hbtf(cospi51, bf0[44], -cospi13, bf0[51])
	bf1[45] = hbtf(cospi19, bf0[45], -cospi45, bf0[50])
	bf1[46] = hbtf(cospi35, bf0[46], -cospi29, bf0[49])
	bf1[47] = hbtf(cospi3, bf0[47], -cospi61, bf0[48])
	bf1[48] = hbtf(cospi61, bf0[47], cospi3, bf0[48])
	bf1[49] = hbtf(cospi29, bf0[46], cospi35, bf0[49])
	bf1[50] = hbtf(cospi45, bf0[45], cospi19, bf0[50])
	bf1[51] = hbtf(cospi13, bf0[44], cospi51, bf0[51])
	bf1[52] = hbtf(cospi53, bf0[43], cospi11, bf0[52])
	bf1[53] = hbtf(cospi21, bf0[42], cospi43, bf0[53])
	bf1[54] = hbtf(cospi37, bf0[41], cospi27, bf0[54])
	bf1[55] = hbtf(cospi5, bf0[40], cospi59, bf0[55])
	bf1[56] = hbtf(cospi57, bf0[39], cospi7, bf0[56])
	bf1[57] = hbtf(cospi25, bf0[38], cospi39, bf0[57])
	bf1[58] = hbtf(cospi41, bf0[37], cospi23, bf0[58])
	bf1[59] = hbtf(cospi9, bf0[36], cospi55, bf0[59])
	bf1[60] = hbtf(cospi49, bf0[35], cospi15, bf0[60])
	bf1[61] = hbtf(cospi17, bf0[34], cospi47, bf0[61])
	bf1[62] = hbtf(cospi33, bf0[33], cospi31, bf0[62])
	bf1[63] = hbtf(cospi1, bf0[32], cospi63, bf0[63])

	// stage 3
	bf0 = bf1
	bf1[16] = hbtf(cospi62, bf0[16], -cospi2, bf0[31])
	bf1[17] = hbtf(cospi30, bf0[17], -cospi34, bf0[30])
	bf1[18] = hbtf(cospi46, bf0[18], -cospi18, bf0[29])
	bf1[19] = hbtf(cospi14, bf0[19], -cospi50, bf0[28])
	bf1[20] = hbtf(cospi54, bf0[20], -cospi10, bf0[27])
	bf1[21] = hbtf(cospi22, bf0[21], -cospi42, bf0[26])
	bf1[22] = hbtf(cospi38, bf0[22], -cospi26, bf0[25])
	bf1[23] = hbtf(cospi6, bf0[23], -cospi58, bf0[24])
	bf1[24] = hbtf(cospi58, bf0[23], cospi6, bf0[24])
	bf1[25] = hbtf(cospi26, bf0[22], cospi38, bf0[25])
	bf1[26] = hbtf(cospi42, bf0[21], cospi22, bf0[26])
	bf1[27] = hbtf(cospi10, bf0[20], cospi54, bf0[27])
	bf1[28] = hbtf(cospi50, bf0[19], cospi14, bf0[28])
	bf1[29] = hbtf(cospi18, bf0[18], cospi46, bf0[29])
	bf1[30] = hbtf(cospi34, bf0[17], cospi30, bf0[30])
	bf1[31] = hbtf(cospi2, bf0[16], cospi62, bf0[31])
	bf1[32] = clamp(add(bf0[32], bf0[33]))
	bf1[33] = clamp(sub(bf0[32], bf0[33]))
	bf1[34] = clamp(sub(bf0[35], bf0[34]))
	bf1[35] = clamp(add(bf0[34], bf0[35]))
	bf1[36] = clamp(add(bf0[36], bf0[37]))
	bf1[37] = clamp(sub(bf0[36], bf0[37]))
	bf1[38] = clamp(sub(bf0[39], bf0[38]))
	bf1[39] = clamp(add(bf0[38], bf0[39]))
	bf1[40] = clamp(add(bf0[40], bf0[41]))
	bf1[41] = clamp(sub(bf0[40], bf0[41]))
	bf1[42] = clamp(sub(bf0[43], bf0[42]))
	bf1[43] = clamp(add(bf0[42], bf0[43]))
	bf1[44] = clamp(add(bf0[44], bf0[45]))
	bf1[45] = clamp(sub(bf0[44], bf0[45]))
	bf1[46] = clamp(sub(bf0[47], bf0[46]))
	bf1[47] = clamp(add(bf0[46], bf0[47]))
	bf1[48] = clamp(add(bf0[48], bf0[49]))
	bf1[49] = clamp(sub(bf0[48], bf0[49]))
	bf1[50] = clamp(sub(bf0[51], bf0[50]))
	bf1[51] = clamp(add(bf0[50], bf0[51]))
	bf1[52] = clamp(add(bf0[52], bf0[53]))
	bf1[53] = clamp(sub(bf0[52], bf0[53]))
	bf1[54] = clamp(sub(bf0[55], bf0[54]))
	bf1[55] = clamp(add(bf0[54], bf0[55]))
	bf1[56] = clamp(add(bf0[56], bf0[57]))
	bf1[57] = clamp(sub(bf0[56], bf0[57]))
	bf1[58] = clamp(sub(bf0[59], bf0[58]))
	bf1[59] = clamp(add(bf0[58], bf0[59]))
	bf1[60] = clamp(add(bf0[60], bf0[61]))
	bf1[61] = clamp(sub(bf0[60], bf0[61]))
	bf1[62] = clamp(sub(bf0[63], bf0[62]))
	bf1[63] = clamp(add(bf0[62], bf0[63]))

	// stage 4
	bf0 = bf1
	bf1[8] = hbtf(cospi60, bf0[8], -cospi4, bf0[15])
	bf1[9] = hbtf(cospi28, bf0[9], -cospi36, bf0[14])
	bf1[10] = hbtf(cospi44, bf0[10], -cospi20, bf0[13])
	bf1[11] = hbtf(cospi12, bf0[11], -cospi52, bf0[12])
	bf1[12] = hbtf(cospi52, bf0[11], cospi12, bf0[12])
	bf1[13] = hbtf(cospi20, bf0[10], cospi44, bf0[13])
	bf1[14] = hbtf(cospi36, bf0[9], cospi28, bf0[14])
	bf1[15] = hbtf(cospi4, bf0[8], cospi60, bf0[15])
	bf1[16] = clamp(add(bf0[16], bf0[17]))
	bf1[17] = clamp(sub(bf0[16], bf0[17]))
	bf1[18] = clamp(sub(bf0[19], bf0[18]))
	bf1[19] = clamp(add(bf0[18], bf0[19]))
	bf1[20] = clamp(add(bf0[20], bf0[21]))
	bf1[21] = clamp(sub(bf0[20], bf0[21]))
	bf1[22] = clamp(sub(bf0[23], bf0[22]))
	bf1[23] = clamp(add(bf0[22], bf0[23]))
	bf1[24] = clamp(add(bf0[24], bf0[25]))
	bf1[25] = clamp(sub(bf0[24], bf0[25]))
	bf1[26] = clamp(sub(bf0[27], bf0[26]))
	bf1[27] = clamp(add(bf0[26], bf0[27]))
	bf1[28] = clamp(add(bf0[28], bf0[29]))
	bf1[29] = clamp(sub(bf0[28], bf0[29]))
	bf1[30] = clamp(sub(bf0[31], bf0[30]))
	bf1[31] = clamp(add(bf0[30], bf0[31]))
	bf1[33] = hbtf(-cospi4, bf0[33], cospi60, bf0[62])
	bf1[34] = hbtf(-cospi60, bf0[34], -cospi4, bf0[61])
	bf1[37] = hbtf(-cospi36, bf0[37], cospi28, bf0[58])
	bf1[38] = hbtf(-cospi28, bf0[38], -cospi36, bf0[57])
	bf1[41] = hbtf(-cospi20, bf0[41], cospi44, bf0[54])
	bf1[42] = hbtf(-cospi44, bf0[42], -cospi20, bf0[53])
	bf1[45] = hbtf(-cospi52, bf0[45], cospi12, bf0[50])
	bf1[46] = hbtf(-cospi12, bf0[46], -cospi52, bf0[49])
	bf1[49] = hbtf(-cospi52, bf0[46], cospi12, bf0[49])
	bf1[50] = hbtf(cospi12, bf0[45], cospi52, bf0[50])
	bf1[53] = hbtf(-cospi20, bf0[42], cospi44, bf0[53])
	bf1[54] = hbtf(cospi44, bf0[41], cospi20, bf0[54])
	bf1[57] = hbtf(-cospi36, bf0[38], cospi28, bf0[57])
	bf1[58] = hbtf(cospi28, bf0[37], cospi36, bf0[58])
	bf1[61] = hbtf(-cospi4, bf0[34], cospi60, bf0[61])
	bf1[62] = hbtf(cospi60, bf0[33], cospi4, bf0[62])

	// stage 5
	bf0 = bf1
	bf1[4] = hbtf(cospi56, bf0[4], -cospi8, bf0[7])
	bf1[5] = hbtf(cospi24, bf0[5], -cospi40, bf0[6])
	bf1[6] = hbtf(cospi40, bf0[5], cospi24, bf0[6])
	bf1[7] = hbtf(cospi8, bf0[4], cospi56, bf0[7])
	bf1[8] = clamp(add(bf0[8], bf0[9]))
	bf1[9] = clamp(sub(bf0[8], bf0[9]))
	bf1[10] = clamp(sub(bf0[11], bf0[10]))
	bf1[11] = clamp(add(bf0[10], bf0[11]))
	bf1[12] = clamp(add(bf0[12], bf0[13]))
	bf1[13] = clamp(sub(bf0[12], bf0[13]))
	bf1[14] = clamp(sub(bf0[15], bf0[14]))
	bf1[15] = clamp(add(bf0[14], bf0[15]))
	bf1[17] = hbtf(-cospi8, bf0[17], cospi56, bf0[30])
	bf1[18] = hbtf(-cospi56, bf0[18], -cospi8, bf0[29])
	bf1[21] = hbtf(-cospi40, bf0[21], cospi24, bf0[26])
	bf1[22] = hbtf(-cospi24, bf0[22], -cospi40, bf0[25])
	bf1[25] = hbtf(-cospi40, bf0[22], cospi24, bf0[25])
	bf1[26] = hbtf(cospi24, bf0[21], cospi40, bf0[26])
	bf1[29] = hbtf(-cospi8, bf0[18], cospi56, bf0[29])
	bf1[30] = hbtf(cospi56, bf0[17], cospi8, bf0[30])
	bf1[32] = clamp(add(bf0[32], bf0[35]))
	bf1[33] = clamp(add(bf0[33], bf0[34]))
	bf1[34] = clamp(sub(bf0[33], bf0[34]))
	bf1[35] = clamp(sub(bf0[32], bf0[35]))
	bf1[36] = clamp(sub(bf0[39], bf0[36]))
	bf1[37] = clamp(sub(bf0[38], bf0[37]))
	bf1[38] = clamp(add(bf0[37], bf0[38]))
	bf1[39] = clamp(add(bf0[36], bf0[39]))
	bf1[40] = clamp(add(bf0[40], bf0[43]))
	bf1[41] = clamp(add(bf0[41], bf0[42]))
	bf1[42] = clamp(sub(bf0[41], bf0[42]))
	bf1[43] = clamp(sub(bf0[40], bf0[43]))
	bf1[44] = clamp(sub(bf0[47], bf0[44]))
	bf1[45] = clamp(sub(bf0[46], bf0[45]))
	bf1[46] = clamp(add(bf0[45], bf0[46]))
	bf1[47] = clamp(add(bf0[44], bf0[47]))
	bf1[48] = clamp(add(bf0[48], bf0[51]))
	bf1[49] = clamp(add(bf0[49], bf0[50]))
	bf1[50] = clamp(sub(bf0[49], bf0[50]))
	bf1[51] = clamp(sub(bf0[48], bf0[51]))
	bf1[52] = clamp(sub(bf0[55], bf0[52]))
	bf1[53] = clamp(sub(bf0[54], bf0[53]))
	bf1[54] = clamp(add(bf0[53], bf0[54]))
	bf1[55] = clamp(add(bf0[52], bf0[55]))
	bf1[56] = clamp(add(bf0[56], bf0[59]))
	bf1[57] = clamp(add(bf0[57], bf0[58]))
	bf1[58] = clamp(sub(bf0[57], bf0[58]))
	bf1[59] = clamp(sub(bf0[56], bf0[59]))
	bf1[60] = clamp(sub(bf0[63], bf0[60]))
	bf1[61] = clamp(sub(bf0[62], bf0[61]))
	bf1[62] = clamp(add(bf0[61], bf0[62]))
	bf1[63] = clamp(add(bf0[60], bf0[63]))

	// stage 6
	bf0 = bf1
	bf1[0] = hbtf(cospi32, bf0[0], cospi32, bf0[1])
	bf1[1] = hbtf(cospi32, bf0[0], -cospi32, bf0[1])
	bf1[2] = hbtf(cospi48, bf0[2], -cospi16, bf0[3])
	bf1[3] = hbtf(cospi16, bf0[2], cospi48, bf0[3])
	bf1[4] = clamp(add(bf0[4], bf0[5]))
	bf1[5] = clamp(sub(bf0[4], bf0[5]))
	bf1[6] = clamp(sub(bf0[7], bf0[6]))
	bf1[7] = clamp(add(bf0[6], bf0[7]))
	bf1[9] = hbtf(-cospi16, bf0[9], cospi48, bf0[14])
	bf1[10] = hbtf(-cospi48, bf0[10], -cospi16, bf0[13])
	bf1[13] = hbtf(-cospi16, bf0[10], cospi48, bf0[13])
	bf1[14] = hbtf(cospi48, bf0[9], cospi16, bf0[14])
	bf1[16] = clamp(add(bf0[16], bf0[19]))
	bf1[17] = clamp(add(bf0[17], bf0[18]))
	bf1[18] = clamp(sub(bf0[17], bf0[18]))
	bf1[19] = clamp(sub(bf0[16], bf0[19]))
	bf1[20] = clamp(sub(bf0[23], bf0[20]))
	bf1[21] = clamp(sub(bf0[22], bf0[21]))
	bf1[22] = clamp(add(bf0[21], bf0[22]))
	bf1[23] = clamp(add(bf0[20], bf0[23]))
	bf1[24] = clamp(add(bf0[24], bf0[27]))
	bf1[25] = clamp(add(bf0[25], bf0[26]))
	bf1[26] = clamp(sub(bf0[25], bf0[26]))
	bf1[27] = clamp(sub(bf0[24], bf0[27]))
	bf1[28] = clamp(sub(bf0[31], bf0[28]))
	bf1[29] = clamp(sub(bf0[30], bf0[29]))
	bf1[30] = clamp(add(bf0[29], bf0[30]))
	bf1[31] = clamp(add(bf0[28], bf0[31]))
	bf1[34] = hbtf(-cospi8, bf0[34], cospi56, bf0[61])
	bf1[35] = hbtf(-cospi8, bf0[35], cospi56, bf0[60])
	bf1[36] = hbtf(-cospi56, bf0[36], -cospi8, bf0[59])
	bf1[37] = hbtf(-cospi56, bf0[37], -cospi8, bf0[58])
	bf1[42] = hbtf(-cospi40, bf0[42], cospi24, bf0[53])
	bf1[43] = hbtf(-cospi40, bf0[43], cospi24, bf0[52])
	bf1[44] = hbtf(-cospi24, bf0[44], -cospi40, bf0[51])
	bf1[45] = hbtf(-cospi24, bf0[45], -cospi40, bf0[50])
	bf1[50] = hbtf(-cospi40, bf0[45], cospi24, bf0[50])
	bf1[51] = hbtf(-cospi40, bf0[44], cospi24, bf0[51])
	bf1[52] = hbtf(cospi24, bf0[43], cospi40, bf0[52])
	bf1[53] = hbtf(cospi24, bf0[42], cospi40, bf0[53])
	bf1[58] = hbtf(-cospi8, bf0[37], cospi56, bf0[58])
	bf1[59] = hbtf(-cospi8, bf0[36], cospi56, bf0[59])
	bf1[60] = hbtf(cospi56, bf0[35], cospi8, bf0[60])
	bf1[61] = hbtf(cospi56, bf0[34], cospi8, bf0[61])

	// stage 7
	bf0 = bf1
	bf1[0] = clamp(add(bf0[0], bf0[3]))
	bf1[1] = clamp(add(bf0[1], bf0[2]))
	bf1[2] = clamp(sub(bf0[1], bf0[2]))
	bf1[3] = clamp(sub(bf0[0], bf0[3]))
	bf1[5] = hbtf(-cospi32, bf0[5], cospi32, bf0[6])
	bf1[6] = hbtf(cospi32, bf0[5], cospi32, bf0[6])
	bf1[8] = clamp(add(bf0[8], bf0[11]))
	bf1[9] = clamp(add(bf0[9], bf0[10]))
	bf1[10] = clamp(sub(bf0[9], bf0[10]))
	bf1[11] = clamp(sub(bf0[8], bf0[11]))
	bf1[12] = clamp(sub(bf0[15], bf0[12]))
	bf1[13] = clamp(sub(bf0[14], bf0[13]))
	bf1[14] = clamp(add(bf0[13], bf0[14]))
	bf1[15] = clamp(add(bf0[12], bf0[15]))
	bf1[18] = hbtf(-cospi16, bf0[18], cospi48, bf0[29])
	bf1[19] = hbtf(-cospi16, bf0[19], cospi48, bf0[28])
	bf1[20] = hbtf(-cospi48, bf0[20], -cospi16, bf0[27])
	bf1[21] = hbtf(-cospi48, bf0[21], -cospi16, bf0[26])
	bf1[26] = hbtf(-cospi16, bf0[21], cospi48, bf0[26])
	bf1[27] = hbtf(-cospi16, bf0[20], cospi48, bf0[27])
	bf1[28] = hbtf(cospi48, bf0[19], cospi16, bf0[28])
	bf1[29] = hbtf(cospi48, bf0[18], cospi16, bf0[29])
	bf1[32] = clamp(add(bf0[32], bf0[39]))
	bf1[33] = clamp(add(bf0[33], bf0[38]))
	bf1[34] = clamp(add(bf0[34], bf0[37]))
	bf1[35] = clamp(add(bf0[35], bf0[36]))
	bf1[36] = clamp(sub(bf0[35], bf0[36]))
	bf1[37] = clamp(sub(bf0[34], bf0[37]))
	bf1[38] = clamp(sub(bf0[33], bf0[38]))
	bf1[39] = clamp(sub(bf0[32], bf0[39]))
	bf1[40] = clamp(sub(bf0[47], bf0[40]))
	bf1[41] = clamp(sub(bf0[46], bf0[41]))
	bf1[42] = clamp(sub(bf0[45], bf0[42]))
	bf1[43] = clamp(sub(bf0[44], bf0[43]))
	bf1[44] = clamp(add(bf0[43], bf0[44]))
	bf1[45] = clamp(add(bf0[42], bf0[45]))
	bf1[46] = clamp(add(bf0[41], bf0[46]))
	bf1[47] = clamp(add(bf0[40], bf0[47]))
	bf1[48] = clamp(add(bf0[48], bf0[55]))
	bf1[49] = clamp(add(bf0[49], bf0[54]))
	bf1[50] = clamp(add(bf0[50], bf0[53]))
	bf1[51] = clamp(add(bf0[51], bf0[52]))
	bf1[52] = clamp(sub(bf0[51], bf0[52]))
	bf1[53] = clamp(sub(bf0[50], bf0[53]))
	bf1[54] = clamp(sub(bf0[49], bf0[54]))
	bf1[55] = clamp(sub(bf0[48], bf0[55]))
	bf1[56] = clamp(sub(bf0[63], bf0[56]))
	bf1[57] = clamp(sub(bf0[62], bf0[57]))
	bf1[58] = clamp(sub(bf0[61], bf0[58]))
	bf1[59] = clamp(sub(bf0[60], bf0[59]))
	bf1[60] = clamp(add(bf0[59], bf0[60]))
	bf1[61] = clamp(add(bf0[58], bf0[61]))
	bf1[62] = clamp(add(bf0[57], bf0[62]))
	bf1[63] = clamp(add(bf0[56], bf0[63]))

	// stage 8
	bf0 = bf1
	bf1[0] = clamp(add(bf0[0], bf0[7]))
	bf1[1] = clamp(add(bf0[1], bf0[6]))
	bf1[2] = clamp(add(bf0[2], bf0[5]))
	bf1[3] = clamp(add(bf0[3], bf0[4]))
	bf1[4] = clamp(sub(bf0[3], bf0[4]))
	bf1[5] = clamp(sub(bf0[2], bf0[5]))
	bf1[6] = clamp(sub(bf0[1], bf0[6]))
	bf1[7] = clamp(sub(bf0[0], bf0[7]))
	bf1[10] = hbtf(-cospi32, bf0[10], cospi32, bf0[13])
	bf1[11] = hbtf(-cospi32, bf0[11], cospi32, bf0[12])
	bf1[12] = hbtf(cospi32, bf0[11], cospi32, bf0[12])
	bf1[13] = hbtf(cospi32, bf0[10], cospi32, bf0[13])
	bf1[16] = clamp(add(bf0[16], bf0[23]))
	bf1[17] = clamp(add(bf0[17], bf0[22]))
	bf1[18] = clamp(add(bf0[18], bf0[21]))
	bf1[19] = clamp(add(bf0[19], bf0[20]))
	bf1[20] = clamp(sub(bf0[19], bf0[20]))
	bf1[21] = clamp(sub(bf0[18], bf0[21]))
	bf1[22] = clamp(sub(bf0[17], bf0[22]))
	bf1[23] = clamp(sub(bf0[16], bf0[23]))
	bf1[24] = clamp(sub(bf0[31], bf0[24]))
	bf1[25] = clamp(sub(bf0[30], bf0[25]))
	bf1[26] = clamp(sub(bf0[29], bf0[26]))
	bf1[27] = clamp(sub(bf0[28], bf0[27]))
	bf1[28] = clamp(add(bf0[27], bf0[28]))
	bf1[29] = clamp(add(bf0[26], bf0[29]))
	bf1[30] = clamp(add(bf0[25], bf0[30]))
	bf1[31] = clamp(add(bf0[24], bf0[31]))
	bf1[36] = hbtf(-cospi16, bf0[36], cospi48, bf0[59])
	bf1[37] = hbtf(-cospi16, bf0[37], cospi48, bf0[58])
	bf1[38] = hbtf(-cospi16, bf0[38], cospi48, bf0[57])
	bf1[39] = hbtf(-cospi16, bf0[39], cospi48, bf0[56])
	bf1[40] = hbtf(-cospi48, bf0[40], -cospi16, bf0[55])
	bf1[41] = hbtf(-cospi48, bf0[41], -cospi16, bf0[54])
	bf1[42] = hbtf(-cospi48, bf0[42], -cospi16, bf0[53])
	bf1[43] = hbtf(-cospi48, bf0[43], -cospi16, bf0[52])
	bf1[52] = hbtf(-cospi16, bf0[43], cospi48, bf0[52])
	bf1[53] = hbtf(-cospi16, bf0[42], cospi48, bf0[53])
	bf1[54] = hbtf(-cospi16, bf0[41], cospi48, bf0[54])
	bf1[55] = hbtf(-cospi16, bf0[40], cospi48, bf0[55])
	bf1[56] = hbtf(cospi48, bf0[39], cospi16, bf0[56])
	bf1[57] = hbtf(cospi48, bf0[38], cospi16, bf0[57])
	bf1[58] = hbtf(cospi48, bf0[37], cospi16, bf0[58])
	bf1[59] = hbtf(cospi48, bf0[36], cospi16, bf0[59])

	// stage 9
	bf0 = bf1
	bf1[0] = clamp(add(bf0[0], bf0[15]))
	bf1[1] = clamp(add(bf0[1], bf0[14]))
	bf1[2] = clamp(add(bf0[2], bf0[13]))
	bf1[3] = clamp(add(bf0[3], bf0[12]))
	bf1[4] = clamp(add(bf0[4], bf0[11]))
	bf1[5] = clamp(add(bf0[5], bf0[10]))
	bf1[6] = clamp(add(bf0[6], bf0[9]))
	bf1[7] = clamp(add(bf0[7], bf0[8]))
	bf1[8] = clamp(sub(bf0[7], bf0[8]))
	bf1[9] = clamp(sub(bf0[6], bf0[9]))
	bf1[10] = clamp(sub(bf0[5], bf0[10]))
	bf1[11] = clamp(sub(bf0[4], bf0[11]))
	bf1[12] = clamp(sub(bf0[3], bf0[12]))
	bf1[13] = clamp(sub(bf0[2], bf0[13]))
	bf1[14] = clamp(sub(bf0[1], bf0[14]))
	bf1[15] = clamp(sub(bf0[0], bf0[15]))
	bf1[20] = hbtf(-cospi32, bf0[20], cospi32, bf0[27])
	bf1[21] = hbtf(-cospi32, bf0[21], cospi32, bf0[26])
	bf1[22] = hbtf(-cospi32, bf0[22], cospi32, bf0[25])
	bf1[23] = hbtf(-cospi32, bf0[23], cospi32, bf0[24])
	bf1[24] = hbtf(cospi32, bf0[23], cospi32, bf0[24])
	bf1[25] = hbtf(cospi32, bf0[22], cospi32, bf0[25])
	bf1[26] = hbtf(cospi32, bf0[21], cospi32, bf0[26])
	bf1[27] = hbtf(cospi32, bf0[20], cospi32, bf0[27])
	bf1[32] = clamp(add(bf0[32], bf0[47]))
	bf1[33] = clamp(add(bf0[33], bf0[46]))
	bf1[34] = clamp(add(bf0[34], bf0[45]))
	bf1[35] = clamp(add(bf0[35], bf0[44]))
	bf1[36] = clamp(add(bf0[36], bf0[43]))
	bf1[37] = clamp(add(bf0[37], bf0[42]))
	bf1[38] = clamp(add(bf0[38], bf0[41]))
	bf1[39] = clamp(add(bf0[39], bf0[40]))
	bf1[40] = clamp(sub(bf0[39], bf0[40]))
	bf1[41] = clamp(sub(bf0[38], bf0[41]))
	bf1[42] = clamp(sub(bf0[37], bf0[42]))
	bf1[43] = clamp(sub(bf0[36], bf0[43]))
	bf1[44] = clamp(sub(bf0[35], bf0[44]))
	bf1[45] = clamp(sub(bf0[34], bf0[45]))
	bf1[46] = clamp(sub(bf0[33], bf0[46]))
	bf1[47] = clamp(sub(bf0[32], bf0[47]))
	bf1[48] = clamp(sub(bf0[63], bf0[48]))
	bf1[49] = clamp(sub(bf0[62], bf0[49]))
	bf1[50] = clamp(sub(bf0[61], bf0[50]))
	bf1[51] = clamp(sub(bf0[60], bf0[51]))
	bf1[52] = clamp(sub(bf0[59], bf0[52]))
	bf1[53] = clamp(sub(bf0[58], bf0[53]))
	bf1[54] = clamp(sub(bf0[57], bf0[54]))
	bf1[55] = clamp(sub(bf0[56], bf0[55]))
	bf1[56] = clamp(add(bf0[55], bf0[56]))
	bf1[57] = clamp(add(bf0[54], bf0[57]))
	bf1[58] = clamp(add(bf0[53], bf0[58]))
	bf1[59] = clamp(add(bf0[52], bf0[59]))
	bf1[60] = clamp(add(bf0[51], bf0[60]))
	bf1[61] = clamp(add(bf0[50], bf0[61]))
	bf1[62] = clamp(add(bf0[49], bf0[62]))
	bf1[63] = clamp(add(bf0[48], bf0[63]))

	// stage 10
	bf0 = bf1
	bf1[0] = clamp(add(bf0[0], bf0[31]))
	bf1[1] = clamp(add(bf0[1], bf0[30]))
	bf1[2] = clamp(add(bf0[2], bf0[29]))
	bf1[3] = clamp(add(bf0[3], bf0[28]))
	bf1[4] = clamp(add(bf0[4], bf0[27]))
	bf1[5] = clamp(add(bf0[5], bf0[26]))
	bf1[6] = clamp(add(bf0[6], bf0[25]))
	bf1[7] = clamp(add(bf0[7], bf0[24]))
	bf1[8] = clamp(add(bf0[8], bf0[23]))
	bf1[9] = clamp(add(bf0[9], bf0[22]))
	bf1[10] = clamp(add(bf0[10], bf0[21]))
	bf1[11] = clamp(add(bf0[11], bf0[20]))
	bf1[12] = clamp(add(bf0[12], bf0[19]))
	bf1[13] = clamp(add(bf0[13], bf0[18]))
	bf1[14] = clamp(add(bf0[14], bf0[17]))
	bf1[15] = clamp(add(bf0[15], bf0[16]))
	bf1[16] = clamp(sub(bf0[15], bf0[16]))
	bf1[17] = clamp(sub(bf0[14], bf0[17]))
	bf1[18] = clamp(sub(bf0[13], bf0[18]))
	bf1[19] = clamp(sub(bf0[12], bf0[19]))
	bf1[20] = clamp(sub(bf0[11], bf0[20]))
	bf1[21] = clamp(sub(bf0[10], bf0[21]))
	bf1[22] = clamp(sub(bf0[9], bf0[22]))
	bf1[23] = clamp(sub(bf0[8], bf0[23]))
	bf1[24] = clamp(sub(bf0[7], bf0[24]))
	bf1[25] = clamp(sub(bf0[6], bf0[25]))
	bf1[26] = clamp(sub(bf0[5], bf0[26]))
	bf1[27] = clamp(sub(bf0[4], bf0[27]))
	bf1[28] = clamp(sub(bf0[3], bf0[28]))
	bf1[29] = clamp(sub(bf0[2], bf0[29]))
	bf1[30] = clamp(sub(bf0[1], bf0[30]))
	bf1[31] = clamp(sub(bf0[0], bf0[31]))
	bf1[40] = hbtf(-cospi32, bf0[40], cospi32, bf0[55])
	bf1[41] = hbtf(-cospi32, bf0[41], cospi32, bf0[54])
	bf1[42] = hbtf(-cospi32, bf0[42], cospi32, bf0[53])
	bf1[43] = hbtf(-cospi32, bf0[43], cospi32, bf0[52])
	bf1[44] = hbtf(-cospi32, bf0[44], cospi32, bf0[51])
	bf1[45] = hbtf(-cospi32, bf0[45], cospi32, bf0[50])
	bf1[46] = hbtf(-cospi32, bf0[46], cospi32, bf0[49])
	bf1[47] = hbtf(-cospi32, bf0[47], cospi32, bf0[48])
	bf1[48] = hbtf(cospi32, bf0[47], cospi32, bf0[48])
	bf1[49] = hbtf(cospi32, bf0[46], cospi32, bf0[49])
	bf1[50] = hbtf(cospi32, bf0[45], cospi32, bf0[50])
	bf1[51] = hbtf(cospi32, bf0[44], cospi32, bf0[51])
	bf1[52] = hbtf(cospi32, bf0[43], cospi32, bf0[52])
	bf1[53] = hbtf(cospi32, bf0[42], cospi32, bf0[53])
	bf1[54] = hbtf(cospi32, bf0[41], cospi32, bf0[54])
	bf1[55] = hbtf(cospi32, bf0[40], cospi32, bf0[55])

	// stage 11 — final outputs
	bf0 = bf1
	for i := 0; i < 32; i++ {
		g.out(c[i], clamp(add(bf0[i], bf0[63-i])))
	}
	for i := 32; i < 64; i++ {
		g.out(c[i], clamp(sub(bf0[63-i], bf0[i])))
	}
}

// ---------------------------------------------------------------------------
// Linear-scan slot allocation for temporaries + assembly emission.
// ---------------------------------------------------------------------------

// allocate assigns each temp vreg a physical slot >= n (c-slots occupy 0..n-1).
// Returns phys[vreg] and the total physical slot count (high-water mark).
func (g *gen) allocate() (phys []int, total int) {
	phys = make([]int, g.nextV)
	for i := range phys {
		phys[i] = -1
	}
	lastUse := make([]int, g.nextV)
	for i := range lastUse {
		lastUse[i] = -1
	}
	useTemp := func(v vref, idx int) {
		if !v.isC && idx > lastUse[v.id] {
			lastUse[v.id] = idx
		}
	}
	for idx, o := range g.ops {
		useTemp(o.a, idx)
		useTemp(o.b, idx)
	}
	// free list of physical slots (>= n)
	var free []int
	nextPhys := g.n
	maxPhys := g.n
	alloc := func() int {
		if len(free) > 0 {
			p := free[len(free)-1]
			free = free[:len(free)-1]
			return p
		}
		p := nextPhys
		nextPhys++
		if nextPhys > maxPhys {
			maxPhys = nextPhys
		}
		return p
	}
	for idx, o := range g.ops {
		if o.kind != opOut {
			phys[o.dst] = alloc()
		}
		// free temps whose last use is this op
		freeIfDead := func(v vref) {
			if !v.isC && lastUse[v.id] == idx {
				free = append(free, phys[v.id])
			}
		}
		freeIfDead(o.a)
		freeIfDead(o.b)
	}
	return phys, maxPhys
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

type emitter struct {
	weights map[int]string // value -> symbol
	worder  []int
}

func (e *emitter) sym(w int) string {
	if s, ok := e.weights[w]; ok {
		return s
	}
	s := fmt.Sprintf("kw%d", len(e.worder))
	e.weights[w] = s
	e.worder = append(e.worder, w)
	return s
}

func addr(slot int) string { return fmt.Sprintf("%d(R12)", slot*32) }

func (g *gen) emitCore(buf *bytes.Buffer, e *emitter, phys []int) {
	pa := func(v vref) string {
		if v.isC {
			return addr(v.id)
		}
		return addr(phys[v.id])
	}
	rs := func(n int) {
		fmt.Fprintf(buf, "\tRS%d(Y0)\n", n)
	}
	for _, o := range g.ops {
		switch o.kind {
		case opAdd:
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y0\n\tVMOVDQU %s, Y1\n\tVPADDQ Y1, Y0, Y2\n\tVMOVDQU Y2, %s\n", pa(o.a), pa(o.b), addr(phys[o.dst]))
		case opSub:
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y0\n\tVMOVDQU %s, Y1\n\tVPSUBQ Y1, Y0, Y2\n\tVMOVDQU Y2, %s\n", pa(o.a), pa(o.b), addr(phys[o.dst]))
		case opClamp:
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y0\n\tCLAMP(Y0)\n\tVMOVDQU Y0, %s\n", pa(o.a), addr(phys[o.dst]))
		case opRsMul:
			pm := "VPADDQ"
			if o.subOp {
				pm = "VPSUBQ"
			}
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y0\n\tVMOVDQU %s, Y1\n\t%s Y1, Y0, Y0\n\tVPBROADCASTQ %s<>(SB), Y1\n\tVPMULDQ Y1, Y0, Y0\n", pa(o.a), pa(o.b), pm, e.sym(o.w0))
			rs(o.n)
			fmt.Fprintf(buf, "\tVMOVDQU Y0, %s\n", addr(phys[o.dst]))
		case opHB:
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y0\n\tVPBROADCASTQ %s<>(SB), Y1\n\tVPMULDQ Y1, Y0, Y0\n", pa(o.a), e.sym(o.w0))
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y2\n\tVPBROADCASTQ %s<>(SB), Y1\n\tVPMULDQ Y1, Y2, Y2\n\tVPADDQ Y2, Y0, Y0\n", pa(o.b), e.sym(o.w1))
			rs(o.n)
			fmt.Fprintf(buf, "\tVMOVDQU Y0, %s\n", addr(phys[o.dst]))
		case opOut:
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y0\n\tVMOVDQU Y0, %s\n", pa(o.a), addr(o.ci))
		case opMov:
			fmt.Fprintf(buf, "\tVMOVDQU %s, Y0\n\tVMOVDQU Y0, %s\n", pa(o.a), addr(phys[o.dst]))
		}
	}
}

// ---------------------------------------------------------------------------
// Top-level file assembly
// ---------------------------------------------------------------------------

func main() {
	outDir := "internal/av1/transform"
	if len(os.Args) > 1 {
		outDir = os.Args[1]
	}

	kernels := []struct {
		name string // e.g. DCT32
		n    int
	}{
		{"DCT32", 32},
		{"DCT64", 64},
	}

	var s bytes.Buffer
	e := &emitter{weights: map[int]string{}}

	// header + macros
	s.WriteString(asmHeader)

	type built struct {
		name string
		n    int
		core *bytes.Buffer
	}
	var bs []built
	maxScratch := 0
	for _, k := range kernels {
		g := newGen(k.n)
		switch k.n {
		case 32:
			c := make([]int, 32)
			for i := range c {
				c[i] = i
			}
			g.dct32(c)
		case 64:
			c := make([]int, 64)
			for i := range c {
				c[i] = i
			}
			g.dct64(c)
		}
		phys, total := g.allocate()
		coreBuf := &bytes.Buffer{}
		g.emitCore(coreBuf, e, phys)
		bs = append(bs, built{k.name, k.n, coreBuf})
		if total > maxScratch {
			maxScratch = total
		}
		fmt.Fprintf(os.Stderr, "%s: temps=%d totalSlots=%d ops=%d\n", k.name, g.nextV, total, len(g.ops))
	}

	// Emit constant pool (weights) + RS biases.
	// Weights:
	ws := append([]int(nil), e.worder...)
	for _, w := range ws {
		sym := e.weights[w]
		fmt.Fprintf(&s, "DATA %s<>+0(SB)/8, $%d\nGLOBL %s<>(SB), RODATA|NOPTR, $8\n", sym, w, sym)
	}
	// RS biases for shifts {8,11,12}
	shifts := []int{8, 11, 12}
	for _, n := range shifts {
		fmt.Fprintf(&s, "DATA radd%d<>+0(SB)/8, $%d\nGLOBL radd%d<>(SB), RODATA|NOPTR, $8\n", n, (1<<(uint(n)-1))+(int64(1)<<62), n)
		fmt.Fprintf(&s, "DATA rsub%d<>+0(SB)/8, $%d\nGLOBL rsub%d<>(SB), RODATA|NOPTR, $8\n", n, int64(1)<<uint(62-n), n)
	}
	s.WriteString("\n")
	s.WriteString(asmMacros)

	// Emit the four kernels (row + col for each length).
	for _, b := range bs {
		emitRowKernel(&s, b.name, b.n, b.core)
		emitColKernel(&s, b.name, b.n, b.core)
	}

	if err := os.WriteFile(filepath.Join(outDir, "dct4lane_avx2_amd64.s"), s.Bytes(), 0o644); err != nil {
		panic(err)
	}

	// Emit the .go declarations + adapters wiring is hand-written separately;
	// here we only emit the func decls + scratch size const.
	scratchInts := maxScratch * 8 // 8 int32 per 32-byte slot
	var gof bytes.Buffer
	fmt.Fprintf(&gof, goHeader, scratchInts)
	if err := os.WriteFile(filepath.Join(outDir, "dct4lane_avx2_amd64.go"), gof.Bytes(), 0o644); err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "scratch ints = %d (%d bytes)\n", scratchInts, scratchInts*4)
}

const asmHeader = `// Code generated by tools/itxgen/avx2gen. DO NOT EDIT.
//
// Four-lane AVX2 inverse-DCT32/DCT64 column & row kernels (amd64). Four rows
// (row pass) or four columns (column pass) occupy the four int64 lanes of each
// YMM register; multiplies use VPMULDQ (signed 32x32 -> 64, exact because every
// AV1 inverse-DCT multiplicand is a raw input or a clipRange'd value inside the
// +/-2^19 stage-range envelope), and the rounding shift and per-stage clamp use
// the same AVX2 integer emulation as dct_avx2_amd64.s. The result is bit-for-bit
// identical to the pure-Go inverseDCT32 / inverseDCT64 references (asserted by
// the dispatch differential tests, executed even under Rosetta 2).
//
// dav1d reference: third_party/upstream/dav1d/src/x86/itx_avx2.asm
// (inv_txfm_add_dct_dct_32x32 / _64x64, inv_dct32 / inv_dct64 butterflies).
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

#include "textflag.h"

`

const asmMacros = `// min in Y14, max in Y15, scratch Y13.
#define CLAMP(r) \
    VPCMPGTQ r, Y14, Y13 \
    VPBLENDVB Y13, Y14, r, r \
    VPCMPGTQ Y15, r, Y13 \
    VPBLENDVB Y13, Y15, r, r
// roundShift(x,N) = ((x + 2^(N-1) + 2^62) >>u N) - 2^(62-N)
#define RS8(r)  VPBROADCASTQ radd8<>(SB), Y13; VPADDQ Y13, r, r; VPSRLQ $8, r, r; VPBROADCASTQ rsub8<>(SB), Y13; VPSUBQ Y13, r, r
#define RS11(r) VPBROADCASTQ radd11<>(SB), Y13; VPADDQ Y13, r, r; VPSRLQ $11, r, r; VPBROADCASTQ rsub11<>(SB), Y13; VPSUBQ Y13, r, r
#define RS12(r) VPBROADCASTQ radd12<>(SB), Y13; VPADDQ Y13, r, r; VPSRLQ $12, r, r; VPBROADCASTQ rsub12<>(SB), Y13; VPSUBQ Y13, r, r

`

// emitRowKernel writes the row-pass wrapper: gather 4 contiguous rows into the
// c-slots, run the shared core, scatter back.
func emitRowKernel(s *bytes.Buffer, name string, n int, core *bytes.Buffer) {
	fmt.Fprintf(s, "// func inverse%sRow4AVX2(r0, r1, r2, r3 *int32, minv, maxv int64, scratch *int32)\n", name)
	fmt.Fprintf(s, "TEXT ·inverse%sRow4AVX2(SB), NOSPLIT, $0-56\n", name)
	s.WriteString("\tMOVQ r0+0(FP), R8\n\tMOVQ r1+8(FP), R9\n\tMOVQ r2+16(FP), R10\n\tMOVQ r3+24(FP), R11\n")
	s.WriteString("\tMOVQ minv+32(FP), AX\n\tMOVQ AX, X0\n\tVPBROADCASTQ X0, Y14\n")
	s.WriteString("\tMOVQ maxv+40(FP), AX\n\tMOVQ AX, X0\n\tVPBROADCASTQ X0, Y15\n")
	s.WriteString("\tMOVQ scratch+48(FP), R12\n")
	s.WriteString("\tMOVQ R12, R13\n\tMOVQ $0, CX\n")
	fmt.Fprintf(s, "%srow_load:\n", lname(name))
	s.WriteString("\tMOVLQSX (R8), AX\n\tMOVQ AX, 0(R13)\n\tADDQ $4, R8\n")
	s.WriteString("\tMOVLQSX (R9), AX\n\tMOVQ AX, 8(R13)\n\tADDQ $4, R9\n")
	s.WriteString("\tMOVLQSX (R10), AX\n\tMOVQ AX, 16(R13)\n\tADDQ $4, R10\n")
	s.WriteString("\tMOVLQSX (R11), AX\n\tMOVQ AX, 24(R13)\n\tADDQ $4, R11\n")
	s.WriteString("\tADDQ $32, R13\n\tINCQ CX\n")
	fmt.Fprintf(s, "\tCMPQ CX, $%d\n\tJL %srow_load\n", n, lname(name))
	s.Write(core.Bytes())
	// scatter
	s.WriteString("\tMOVQ r0+0(FP), R8\n\tMOVQ r1+8(FP), R9\n\tMOVQ r2+16(FP), R10\n\tMOVQ r3+24(FP), R11\n")
	s.WriteString("\tMOVQ R12, R13\n\tMOVQ $0, CX\n")
	fmt.Fprintf(s, "%srow_store:\n", lname(name))
	s.WriteString("\tMOVQ 0(R13), AX\n\tMOVL AX, (R8)\n\tADDQ $4, R8\n")
	s.WriteString("\tMOVQ 8(R13), AX\n\tMOVL AX, (R9)\n\tADDQ $4, R9\n")
	s.WriteString("\tMOVQ 16(R13), AX\n\tMOVL AX, (R10)\n\tADDQ $4, R10\n")
	s.WriteString("\tMOVQ 24(R13), AX\n\tMOVL AX, (R11)\n\tADDQ $4, R11\n")
	s.WriteString("\tADDQ $32, R13\n\tINCQ CX\n")
	fmt.Fprintf(s, "\tCMPQ CX, $%d\n\tJL %srow_store\n", n, lname(name))
	s.WriteString("\tVZEROUPPER\n\tRET\n\n")
}

// emitColKernel writes the column-pass wrapper: gather 4 adjacent columns
// (contiguous within each row) into the c-slots, run the shared core, scatter.
func emitColKernel(s *bytes.Buffer, name string, n int, core *bytes.Buffer) {
	fmt.Fprintf(s, "// func inverse%sCol4AVX2(base *int32, rowStrideBytes, minv, maxv int64, scratch *int32)\n", name)
	fmt.Fprintf(s, "TEXT ·inverse%sCol4AVX2(SB), NOSPLIT, $0-40\n", name)
	s.WriteString("\tMOVQ base+0(FP), R8\n\tMOVQ rowStrideBytes+8(FP), R10\n")
	s.WriteString("\tMOVQ minv+16(FP), AX\n\tMOVQ AX, X0\n\tVPBROADCASTQ X0, Y14\n")
	s.WriteString("\tMOVQ maxv+24(FP), AX\n\tMOVQ AX, X0\n\tVPBROADCASTQ X0, Y15\n")
	s.WriteString("\tMOVQ scratch+32(FP), R12\n")
	s.WriteString("\tMOVQ R8, R11\n\tMOVQ R12, R13\n\tMOVQ $0, CX\n")
	fmt.Fprintf(s, "%scol_load:\n", lname(name))
	s.WriteString("\tMOVLQSX 0(R11), AX\n\tMOVQ AX, 0(R13)\n")
	s.WriteString("\tMOVLQSX 4(R11), AX\n\tMOVQ AX, 8(R13)\n")
	s.WriteString("\tMOVLQSX 8(R11), AX\n\tMOVQ AX, 16(R13)\n")
	s.WriteString("\tMOVLQSX 12(R11), AX\n\tMOVQ AX, 24(R13)\n")
	s.WriteString("\tADDQ R10, R11\n\tADDQ $32, R13\n\tINCQ CX\n")
	fmt.Fprintf(s, "\tCMPQ CX, $%d\n\tJL %scol_load\n", n, lname(name))
	s.Write(core.Bytes())
	// scatter
	s.WriteString("\tMOVQ base+0(FP), R8\n\tMOVQ R8, R11\n\tMOVQ R12, R13\n\tMOVQ $0, CX\n")
	fmt.Fprintf(s, "%scol_store:\n", lname(name))
	s.WriteString("\tMOVQ 0(R13), AX\n\tMOVL AX, 0(R11)\n")
	s.WriteString("\tMOVQ 8(R13), AX\n\tMOVL AX, 4(R11)\n")
	s.WriteString("\tMOVQ 16(R13), AX\n\tMOVL AX, 8(R11)\n")
	s.WriteString("\tMOVQ 24(R13), AX\n\tMOVL AX, 12(R11)\n")
	s.WriteString("\tADDQ R10, R11\n\tADDQ $32, R13\n\tINCQ CX\n")
	fmt.Fprintf(s, "\tCMPQ CX, $%d\n\tJL %scol_store\n", n, lname(name))
	s.WriteString("\tVZEROUPPER\n\tRET\n\n")
}

func lname(name string) string { return name }

const goHeader = `// Code generated by tools/itxgen/avx2gen. DO NOT EDIT.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

package transform

// avx2Scratch4Ints is the int32 scratch length the four-lane AVX2 kernels need
// (the largest kernel's physical slot high-water mark, 8 int32 per YMM slot).
const avx2Scratch4Ints = %d

//go:noescape
func inverseDCT32Row4AVX2(r0, r1, r2, r3 *int32, minv, maxv int64, scratch *int32)

//go:noescape
func inverseDCT64Row4AVX2(r0, r1, r2, r3 *int32, minv, maxv int64, scratch *int32)

//go:noescape
func inverseDCT32Col4AVX2(base *int32, rowStrideBytes, minv, maxv int64, scratch *int32)

//go:noescape
func inverseDCT64Col4AVX2(base *int32, rowStrideBytes, minv, maxv int64, scratch *int32)
`
