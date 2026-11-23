// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

import (
	"fmt"
	"slices"
	"strings"
)

type writer struct{ strings.Builder }

func (w *writer) line(format string, args ...any) {
	fmt.Fprintf(w, format+"\n", args...)
}

// body writes the live nodes of nw in creation order; btf spells one
// butterfly round_shift(a*w0 + b*w1).
func (w *writer) body(nw *network, btf func(v *node) string) {
	for _, v := range nw.nodes {
		if !v.live {
			continue
		}
		switch v.op {
		case '+':
			w.line("\t%s := %s.Add(%s)", v.name, v.a.name, v.b.name)
		case '-':
			w.line("\t%s := %s.Sub(%s)", v.name, v.a.name, v.b.name)
		default:
			w.line("\t%s := %s", v.name, btf(v))
		}
	}
}

// arm64Btf indexes the twiddle table t (fwd64Twiddles) and folds the
// rounding bias into the first multiply-accumulate.
func arm64Btf(v *node) string {
	tw := func(x weight) string {
		if x.neg {
			return fmt.Sprintf("t.n[%d]", x.idx)
		}
		return fmt.Sprintf("t.c[%d]", x.idx)
	}
	return fmt.Sprintf("%s.MulAdd(%s, %s.MulAdd(%s, t.rnd)).Shift(t.shr)", v.b.name, tw(v.w1), v.a.name, tw(v.w0))
}

// arm64Round is the shift[1]/shift[2] rounding (v + c) >> s with hoisted
// constant and negated-shift vectors.
type arm64Round struct{ add, shr string }

// arm64Col writes a column pass over fwdLanes columns from g. Rows
// 0..n-2 load the eight samples from column g (the read ends inside the next
// row) and keep the low four; the last row takes the in-row load. The kept
// outputs are rounded and stored transposed into out: out[c*stride+k] is
// output k of column c.
func arm64Col(w *writer, nw *network, fn, tab string, shl bool, rnd arm64Round, stride int) {
	w.line("func %s(out []int32, g int, residual []int16, rs int) {", fn)
	w.line("\tt := %s", tab)
	w.line("\tr, s := %s, %s", rnd.add, rnd.shr)
	ext, last := ".ExtendLo4ToInt32()", "fwdLoadResRawAt"
	if shl {
		w.line("\tsh := fwdShlPos2")
		ext, last = ".ExtendLo4ToInt32().Shift(sh)", "fwdLoadResAt"
	}
	w.line("\to := g")
	n := nw.n
	for i := range n - 1 {
		if i > 0 {
			w.line("\to += rs")
		}
		w.line("\tx%d := fwdLoadI16At(residual, o)%s", i, ext)
	}
	w.line("\tx%d := %s(residual, %d*rs+g&^7, g)", n-1, last, n-1)
	w.body(nw, arm64Btf)
	w.line("\tdst := (*[4*%d]int32)(out[g*%d:])", stride, stride)
	for k := 0; k < len(nw.out); k += 4 {
		q := k / 4
		for j, l := range "abcd" {
			w.line("\tt%d%c := %s.Add(r).Shift(s)", q, l, nw.out[k+j].name)
		}
		w.line("\te%d0, e%d1 := fwdI32AsI64(t%da.InterleaveLo(t%db)), fwdI32AsI64(t%da.InterleaveHi(t%db))", q, q, q, q, q, q)
		w.line("\te%d2, e%d3 := fwdI32AsI64(t%dc.InterleaveLo(t%dd)), fwdI32AsI64(t%dc.InterleaveHi(t%dd))", q, q, q, q, q, q)
		w.line("\tfwdStoreI32Arr(dst[:], %d, fwdI64AsI32(e%d0.InterleaveLo(e%d2)))", k, q, q)
		w.line("\tfwdStoreI32Arr(dst[:], %d, fwdI64AsI32(e%d0.InterleaveHi(e%d2)))", k+stride, q, q)
		w.line("\tfwdStoreI32Arr(dst[:], %d, fwdI64AsI32(e%d1.InterleaveLo(e%d3)))", k+2*stride, q, q)
		w.line("\tfwdStoreI32Arr(dst[:], %d, fwdI64AsI32(e%d1.InterleaveHi(e%d3)))", k+3*stride, q, q)
	}
	w.line("}")
}

// arm64Row writes a row pass over the fwdLanes rows from h of the
// transposed column output in (in[c*stride+h], c < n). The outputs are
// rounded when rnd is set and written to out[k*stride+h]; in and out may be
// the same slice, every input being loaded before the first store.
func arm64Row(w *writer, nw *network, sig, tab, in, out string, rnd *arm64Round, stride int) {
	w.line("func %s {", sig)
	w.line("\tt := %s", tab)
	if rnd != nil {
		w.line("\tr, s := %s, %s", rnd.add, rnd.shr)
	}
	n := nw.n
	w.line("\tsrc := (*[%d*%d + fwdLanes]int32)(%s[h:])", n-1, stride, in)
	for i := range n {
		w.line("\tx%d := fwdLoadI32Arr(src[:], %d*%d)", i, i, stride)
	}
	w.body(nw, arm64Btf)
	dst := "src"
	if out != in {
		dst = "dst"
		w.line("\tdst := (*[%d*%d + fwdLanes]int32)(%s[h:])", len(nw.out)-1, stride, out)
	}
	for k, v := range nw.out {
		val := v.name
		if rnd != nil {
			val += ".Add(r).Shift(s)"
		}
		w.line("\tfwdStoreI32Arr(%s[:], %d*%d, %s)", dst, k, stride, val)
	}
	w.line("}")
}

// portable writes the amd64 forms: twiddles hoisted from the package-level
// broadcast constants (fdct_gosimd_constants.go) into locals and the
// fwdHalfBtf<bit>V butterflies.
type portable struct {
	bit   int
	cospi [64]int32
}

func (p portable) constName(x weight) string {
	v := p.cospi[x.idx]
	if x.neg {
		return fmt.Sprintf("kn%d", v)
	}
	return fmt.Sprintf("k%d", v)
}

func (p portable) hoist(w *writer, nw *network) {
	var pos, neg []int32
	seen := map[weight]bool{}
	for _, v := range nw.nodes {
		if !v.live || v.op != 'b' {
			continue
		}
		for _, x := range []weight{v.w0, v.w1} {
			if seen[x] {
				continue
			}
			seen[x] = true
			if x.neg {
				neg = append(neg, p.cospi[x.idx])
			} else {
				pos = append(pos, p.cospi[x.idx])
			}
		}
	}
	slices.Sort(pos)
	slices.Sort(neg)
	for _, v := range pos {
		w.line("\tk%d := fwdConst%d", v, v)
	}
	for _, v := range neg {
		w.line("\tkn%d := fwdConstN%d", v, v)
	}
}

func (p portable) btf(v *node) string {
	return fmt.Sprintf("fwdHalfBtf%dV(%s, %s, %s, %s)", p.bit, p.constName(v.w0), v.a.name, p.constName(v.w1), v.b.name)
}

// portableCol writes the column pass over fwdLanes columns from g, rounded
// by round and stored row-major into buf[k*n+g].
func portableCol(w *writer, nw *network, fn string, p portable, round string) {
	n := nw.n
	w.line("func %s(buf []int32, g int, residual []int16, rs int) {", fn)
	p.hoist(w, nw)
	for i := range n {
		w.line("\tx%d := fwdLoadResAt(residual, %d*rs+g&^7, g)", i, i)
	}
	w.body(nw, p.btf)
	for k, v := range nw.out {
		w.line("\tfwdStoreI32At(buf, %d*%d+g, %s(%s))", k, n, round, v.name)
	}
	w.line("}")
}

// portableRow writes the row pass over the fwdLanes rows from h of the
// transposed column output bufT, written to coeff[k*coeffStride+h].
func portableRow(w *writer, nw *network, fn string, p portable) {
	n := nw.n
	w.line("func %s(coeff []int32, coeffStride int, bufT []int32, h int) {", fn)
	p.hoist(w, nw)
	for i := range n {
		w.line("\tx%d := fwdLoadI32At(bufT, %d*%d+h)", i, i, n)
	}
	w.body(nw, p.btf)
	for k, v := range nw.out {
		w.line("\tfwdStoreI32At(coeff, %d*coeffStride+h, %s)", k, v.name)
	}
	w.line("}")
}
