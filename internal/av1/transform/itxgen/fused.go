// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

import (
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// The fused generator writes the 8-bit DCT_DCT inverse-transform-and-add
// kernels of itx_fused_gen_arm64.go as straight-line code: the coefficient
// loads, both passes with the zero inputs of the coded extent folded, the
// guards, the transposes, the rounding and the prediction add, with no
// helper calls, loops or arrays inside a pass and every constant one load
// at a fixed offset from a single table pointer. 8x8 and 16x16 are one
// function per (coded columns, coded rows) network form; 32x32 and 64x64,
// whose row-pass output is far beyond the register file, are a thin block
// function over generated row-strip and column-strip functions with the
// rows kept in a stack tmp, as dav1d does.
//
// Exactness is the hand-written kernels' contract: every rotation output
// fits int16 under the per-lane L1 guards (itx*_gain_test.go), so the int32
// multiply-accumulate sums and the narrowing are exact, and the saturating
// adds are the 8-bit int16 clamps of the scalar transforms.

// fv is one symbolic eight-lane int16 vector; zero marks a known-zero one.
// A nonzero id marks the pending n-point identity of the named vector: the
// rounding that follows it (round2) evaluates both exactly in one step.
type fv struct {
	zero bool
	name string
	id   int
}

type fgen struct {
	sb     strings.Builder
	n      int
	memo   map[string]fv
	wide   map[string][2]string
	consts map[int]bool
	// raw holds the int32 constants loaded as they are (field name to value).
	raw map[string]int
	// half marks a four-lane network instance.
	half bool
	// split runs a 32- or 64-point strip in dav1d's halves: the even half
	// parked in a stack array while the odd half runs, the inputs loaded at
	// their first use and the output butterflies written as they are
	// consumed. Without it the scheduler keeps all 32 inputs and their
	// widened halves live at once and spills them.
	split bool
	// inner marks the 32-point even half of a split 64-point network, whose
	// own even half stays in registers (parking both measured slower).
	inner bool
	// pend holds the values written at their first use (use).
	pend map[string]func()
}

// use writes the pending definition of v, if any, and returns its name.
func (g *fgen) use(v fv) string {
	if f, ok := g.pend[v.name]; ok {
		delete(g.pend, v.name)
		f()
	}
	return v.name
}

// lazy returns the vector name whose definition emit writes at its first
// use.
func (g *fgen) lazy(name string, emit func()) fv {
	if g.pend == nil {
		g.pend = map[string]func(){}
	}
	g.pend[name] = emit
	return fv{name: name}
}

// park stores the live vectors of vs in a stack array and returns them as
// reloads at their first use (dav1d's idct32 keeps its even half in memory
// while the odd half runs).
func (g *fgen) park(vs []fv) []fv {
	if !g.split || g.inner {
		return vs
	}
	a := g.tmp("pk")
	var live []fv
	for _, v := range vs {
		if !v.zero {
			live = append(live, v)
		}
	}
	g.line("var %s [%d]archsimd.Int16x8", a, len(live))
	out := make([]fv, len(vs))
	j := 0
	for i, v := range vs {
		if v.zero {
			out[i] = v
			continue
		}
		g.line("%s[%d] = %s", a, j, g.use(v))
		n, slot := g.tmp("e"), j
		out[i] = g.lazy(n, func() { g.line("%s := %s[%d]", n, a, slot) })
		j++
	}
	return out
}

// combineAll returns the output butterflies out[i] = e[i] + o[i] and
// out[n-1-i] = e[i] - o[i]; split writes each when it is first used.
func (g *fgen) combineAll(e, o []fv) []fv {
	h := len(e)
	out := make([]fv, 2*h)
	for i := 0; i < h; i++ {
		ei, oi := e[i], o[i]
		if !g.split || ei.zero || oi.zero {
			out[i], out[2*h-1-i] = g.combine(ei, oi)
			continue
		}
		s, d := g.tmp("a"), g.tmp("a")
		out[i] = g.lazy(s, func() { g.line("%s := %s.AddSaturated(%s)", s, g.use(ei), g.use(oi)) })
		out[2*h-1-i] = g.lazy(d, func() { g.line("%s := %s.SubSaturated(%s)", d, g.use(ei), g.use(oi)) })
	}
	return out
}

// The 10- and 12-bit (16bpc) kernels run the same int16 passes under the
// same guards and differ only in the prediction add (dav1d itx16 reuses its
// 8bpc int16 functions at 10 bits): the row strips are the 8-bit ones and
// the column output is added to the 16-bit prediction row with a saturating
// int16 add clipped to [0, mx], which equals the scalar clip(pred + residual)
// for any int16 residual and pred <= 4095. Under the guards no coefficient
// or butterfly leaves int16, so the staged path's bd + 8 row and
// max(bd + 6, 16) column clamps never bite.

// combine returns the butterfly e + o, e - o of a network's output stage.
func (g *fgen) combine(e, o fv) (fv, fv) { return g.add(e, o), g.sub(e, o) }

func (g *fgen) line(format string, args ...any) {
	fmt.Fprintf(&g.sb, "\t"+format+"\n", args...)
}

func (g *fgen) tmp(p string) string {
	g.n++
	return fmt.Sprintf("%s%d", p, g.n)
}

// reset starts a network instance: no rotation or widening is shared with
// the previous one.
func (g *fgen) reset() {
	g.memo = map[string]fv{}
	g.wide = map[string][2]string{}
}

func constField(c int) string {
	if c < 0 {
		return fmt.Sprintf("n%d", -c)
	}
	return fmt.Sprintf("p%d", c)
}

// k is the load of the broadcast int32 constant 16*c.
func (g *fgen) k(c int) string {
	g.consts[c] = true
	return fmt.Sprintf("archsimd.LoadInt32x4Array(&t.%s)", constField(c))
}

// widen returns the sign-extended low and high halves of v, once per
// network instance (only the low half in a four-lane network).
func (g *fgen) widen(v fv) (string, string) {
	if w, ok := g.wide[v.name]; ok {
		return w[0], w[1]
	}
	lo, hi := g.tmp("w"), ""
	x := g.use(v)
	g.line("%s := %s.ExtendLo4ToInt32()", lo, x)
	if !g.half {
		hi = g.tmp("w")
		g.line("%s := %s.HiToLo().ExtendLo4ToInt32()", hi, x)
	}
	g.wide[v.name] = [2]string{lo, hi}
	return lo, hi
}

// term is one operand of a dot product: v times the cospi-scale constant c.
type term struct {
	v fv
	c int
}

// rot is Round2(a*ca + b*cb, 12) on eight int16 lanes (dot).
func (g *fgen) rot(a fv, ca int, b fv, cb int) fv {
	return g.dot([]term{{a, ca}, {b, cb}})
}

// dot is Round2(sum v*c, 12) on eight int16 lanes: on each sign-extended
// half, 2^15 + sum v*(16c) by multiply-accumulate (the final sum fits
// int32, so a wrapped partial sum is harmless), then the high halves of
// the 32-bit sums. Zero operands fold away and a repeated product is
// computed once. A four-lane network computes the low half only (the high
// lanes of its vectors are don't-cares).
func (g *fgen) dot(ts []term) fv {
	var live []term
	var key strings.Builder
	// The key keeps the zero operands' constants (as #328's rotation did):
	// folding a*c + 0*c1 and a*c + 0*c2 into one product saves an
	// instruction but moves the register allocation of the 64-point strips
	// (+3% cycles on DCT64 Col8).
	for _, t := range ts {
		if !t.v.zero {
			live = append(live, t)
		}
		fmt.Fprintf(&key, "%s|%d|", t.v.name, t.c)
	}
	if len(live) == 0 {
		return fv{zero: true}
	}
	if v, ok := g.memo[key.String()]; ok {
		return v
	}
	n := g.tmp("r")
	for i, t := range live {
		lo, hi := g.widen(t.v)
		k := g.tmp("k")
		g.line("%s := %s", k, g.k(t.c))
		if i == 0 {
			g.line("%sl := %s.MulAdd(%s, bias)", n, lo, k)
			if !g.half {
				g.line("%sh := %s.MulAdd(%s, bias)", n, hi, k)
			}
		} else {
			g.line("%sl = %s.MulAdd(%s, %sl)", n, lo, k, n)
			if !g.half {
				g.line("%sh = %s.MulAdd(%s, %sh)", n, hi, k, n)
			}
		}
	}
	g.narrowHi(n)
	v := fv{name: n}
	g.memo[key.String()] = v
	return v
}

// narrowHi writes n, the high halves of the 32-bit sums nl and nh (nl and
// nl again in a four-lane network).
func (g *fgen) narrowHi(n string) {
	hi := n + "h"
	if g.half {
		hi = n + "l"
	}
	g.line("%s := %sl.ToBits().ReshapeToUint16s().ConcatOdd(%s.ToBits().ReshapeToUint16s()).BitsToInt16()", n, n, hi)
}

func (g *fgen) add(a, b fv) fv {
	switch {
	case a.zero:
		return b
	case b.zero:
		return a
	}
	n := g.tmp("a")
	x, y := g.use(a), g.use(b)
	g.line("%s := %s.AddSaturated(%s)", n, x, y)
	return fv{name: n}
}

func (g *fgen) sub(a, b fv) fv {
	switch {
	case b.zero:
		return a
	case a.zero:
		n := g.tmp("a")
		g.line("%s := zero.SubSaturated(%s)", n, g.use(b))
		return fv{name: n}
	}
	n := g.tmp("a")
	x, y := g.use(a), g.use(b)
	g.line("%s := %s.SubSaturated(%s)", n, x, y)
	return fv{name: n}
}

// inputs pads the live vectors with zeros to n.
func inputs(live []fv, n int) []fv {
	in := make([]fv, n)
	copy(in, live)
	for i := len(live); i < n; i++ {
		in[i] = fv{zero: true}
	}
	return in
}

// l1 writes the per-lane L1 sum of the live vectors of vs into acc
// (declared when decl is set) and reports whether any was live.
func (g *fgen) l1(acc string, decl bool, vs []fv) bool {
	any := false
	for _, v := range vs {
		if v.zero {
			continue
		}
		switch {
		case decl && !any:
			g.line("%s := %s.Abs().ToBits()", acc, g.use(v))
		default:
			g.line("%s = %s.AddSaturated(%s.Abs().ToBits())", acc, acc, g.use(v))
		}
		any = true
	}
	return any
}

// guard writes the per-lane L1 decline check over the live vectors.
func (g *fgen) guard(vs []fv) {
	acc := g.tmp("g")
	if !g.l1(acc, true, vs) {
		return
	}
	g.line("if %s.ReduceMax() > itx16ColumnL1Bound {", acc)
	g.line("\treturn false")
	g.line("}")
}

// transpose8 transposes the 8x8 int16 matrix rows (rows[r] = row r) with
// three interleave stages; a zero row is a zero operand and an interleave
// of two zero operands is a zero result.
func (g *fgen) transpose8(rows []fv) []fv {
	p := g.tmp("x")
	// a/b stage values, named or zero ("").
	var a, b [8]string
	op := func(x, y, zeroOf string) (string, string) {
		if x == "" {
			x = zeroOf
		}
		if y == "" {
			y = zeroOf
		}
		return x, y
	}
	for i := 0; i < 8; i += 2 {
		if rows[i].zero && rows[i+1].zero {
			continue
		}
		x, y := op(g.use(rows[i]), g.use(rows[i+1]), "zero")
		a[i], a[i+1] = fmt.Sprintf("%sa%d", p, i), fmt.Sprintf("%sa%d", p, i+1)
		g.line("%s := %s.InterleaveLo(%s).ToBits().ReshapeToUint32s()", a[i], x, y)
		g.line("%s := %s.InterleaveHi(%s).ToBits().ReshapeToUint32s()", a[i+1], x, y)
	}
	for _, q := range [][4]int{{0, 1, 0, 2}, {2, 3, 1, 3}, {4, 5, 4, 6}, {6, 7, 5, 7}} {
		if a[q[2]] == "" && a[q[3]] == "" {
			continue
		}
		x, y := op(a[q[2]], a[q[3]], "zero.ToBits().ReshapeToUint32s()")
		b[q[0]], b[q[1]] = fmt.Sprintf("%sb%d", p, q[0]), fmt.Sprintf("%sb%d", p, q[1])
		g.line("%s := %s.InterleaveLo(%s).ReshapeToUint64s()", b[q[0]], x, y)
		g.line("%s := %s.InterleaveHi(%s).ReshapeToUint64s()", b[q[1]], x, y)
	}
	out := make([]fv, 8)
	for i, q := range [][2]int{{0, 4}, {1, 5}, {2, 6}, {3, 7}} {
		if b[q[0]] == "" && b[q[1]] == "" {
			out[2*i], out[2*i+1] = fv{zero: true}, fv{zero: true}
			continue
		}
		x, y := op(b[q[0]], b[q[1]], "zero.ToBits().ReshapeToUint64s()")
		lo, hi := fmt.Sprintf("%st%d", p, 2*i), fmt.Sprintf("%st%d", p, 2*i+1)
		g.line("%s := %s.InterleaveLo(%s).ReshapeToUint16s().BitsToInt16()", lo, x, y)
		g.line("%s := %s.InterleaveHi(%s).ReshapeToUint16s().BitsToInt16()", hi, x, y)
		out[2*i], out[2*i+1] = fv{name: lo}, fv{name: hi}
	}
	return out
}

// round2 writes Round2(v, s) for s >= 1 as ((v >> (s-1)) + 1) >> 1, the
// last step a rounding halving add of zero.
func (g *fgen) round2(v fv, s int) fv {
	if v.zero {
		return v
	}
	if v.id != 0 {
		return g.idRound(v, s)
	}
	if s == 0 {
		return v
	}
	n := g.tmp("s")
	if s == 1 {
		g.line("%s := %s.Average(zero)", n, g.use(v))
	} else {
		g.line("%s := %s.Shift(sh%d).Average(zero)", n, g.use(v), s-1)
	}
	return fv{name: n}
}

// load writes the int16-saturating load of the lanes (8 or 4) int32
// coefficients at c[off:] (the row clamp of the 8-bit row pass), after the
// rect2 scale Round2(x*2896, 12) = (x*181 + 128) >> 8 when rect2 is set
// (with x first clamped to +-2^22, which keeps the product inside int32
// and changes no result: the scaled value saturates either way).
func (g *fgen) load(off, lanes int, rect2 bool) fv {
	return fv{name: g.loadAs("", off, lanes, rect2)}
}

// loadAs is load into the vector named v (a new one if v is empty), whose
// name it returns.
func (g *fgen) loadAs(v string, off, lanes int, rect2 bool) string {
	half := func(o int) string {
		x := fmt.Sprintf("archsimd.LoadInt32x4Array((*[4]int32)(c[%d:%d]))", o, o+4)
		if rect2 {
			x += fmt.Sprintf(".Max(%s).Min(%s).MulAdd(%s, %s).Shift(%s)",
				g.kraw("rlo", -(1<<22)), g.kraw("rhi", 1<<22), g.kraw("r181", 181), g.kraw("r128", 128), g.kraw("sh8", -8))
		}
		return x + ".SaturateToInt16()"
	}
	if lanes == 4 {
		if v == "" {
			v = g.tmp("c")
		}
		g.line("%s := %s", v, half(off))
		return v
	}
	n := g.tmp("c")
	g.line("%s := %s.ToBits().ReshapeToUint64s().BitsToFloat64()", n, half(off))
	g.line("%s = %s.SetElem(1, %s.ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))", n, n, half(off+4))
	if v == "" {
		v = g.tmp("c")
	}
	g.line("%s := %s.ToBits().ReshapeToUint16s().BitsToInt16()", v, n)
	return v
}

// Prediction row loads of an eight-column strip. Every load stays inside
// the block's own prediction rows (another worker may be writing the
// neighbouring blocks): predLo loads the strip's eight bytes and the next
// eight, predHi the eight before and the strip's (pq is the strip's column
// minus 8), predSel sixteen bytes zero-extended by the caller's TBL
// selector (either of the two), predD8 only the eight bytes (8 wide) and
// predD4 the four (4 wide).
const (
	predLo = iota
	predHi
	predSel
	predD8
	predD4
	// predH16 loads the strip's eight 16-bit prediction samples (16 bytes).
	predH16
)

// addRowsGroup is the number of rows whose dst and prediction windows
// addRows takes before their loads and stores: the bounds checks of a group
// run back to back ahead of its rows, so the transform's outputs are not
// split across a branch per row (which spilled vector registers).
const addRowsGroup = 8

// addRows writes the column output of one eight-column strip: Round2(., 4)
// of out[r], added to prediction row r and saturated into dst row r, for
// the n rows, through bounds-checked row windows of d and p (row r at
// r*ds / r*ps).
func (g *fgen) addRows(out []fv, pred int) {
	dw, pw := 8, 16
	switch pred {
	case predD8:
		pw = 8
	case predD4:
		dw, pw = 4, 4
	case predH16:
		dw = 16
	}
	// One compare per window: with the row offset o in [0, len-w] the
	// window [o, o+w) is in bounds, which the compiler proves from the
	// explicit test (the slice expression alone costs an add and a compare).
	dl, pl := g.tmp("dl"), g.tmp("pl")
	g.line("%s, %s := len(d)-%d, len(p)-%d", dl, pl, dw, pw)
	wins := make([][2]string, len(out))
	for r := range out {
		if r%addRowsGroup == 0 {
			for j := r; j < min(r+addRowsGroup, len(out)); j++ {
				dn, pn := g.tmp("dw"), g.tmp("pw")
				do, po := "0", "0"
				if j > 0 {
					do, po = fmt.Sprintf("%d*ds", j), fmt.Sprintf("%d*ps", j)
				}
				g.line("if %s > %s || %s > %s {\n\t\tpanic(errItxGenWindow)\n\t}", do, dl, po, pl)
				g.line("%s, %s := (*[%d]byte)(d[%s:%s+%d]), (*[%d]byte)(p[%s:%s+%d])", dn, pn, dw, do, do, dw, pw, po, po, pw)
				wins[j] = [2]string{dn, pn}
			}
		}
		dq, pq := wins[r][0], wins[r][1]
		res := g.round2(out[r], 4)
		p := g.tmp("p")
		switch pred {
		case predLo:
			g.line("%s := archsimd.LoadUint8x16Array(%s).ExtendLo8ToUint16().BitsToInt16()", p, pq)
		case predHi:
			g.line("%s := archsimd.LoadUint8x16Array(%s).HiToLo().ExtendLo8ToUint16().BitsToInt16()", p, pq)
		case predSel:
			g.line("%s := archsimd.LoadUint8x16Array(%s).LookupOrZero(sel).ReshapeToUint16s().BitsToInt16()", p, pq)
		case predD8:
			g.line("%s := pz.SetElem(0, %s).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()", p, leLoad(pq, 8))
		case predD4:
			g.line("%s := pf.SetElem(0, %s).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()", p, leLoad(pq, 4))
		case predH16:
			g.line("%s := archsimd.LoadUint8x16Array(%s).ReshapeToUint16s().BitsToInt16()", p, pq)
			if res.zero {
				g.line("%s.ToBits().ReshapeToUint8s().StoreArray(%s)", p, dq)
			} else {
				g.line("%s.AddSaturated(%s).Max(zero).Min(mxv).ToBits().ReshapeToUint8s().StoreArray(%s)", p, res.name, dq)
			}
			continue
		}
		sum := p
		if !res.zero {
			sum = fmt.Sprintf("%s.Add(%s)", p, res.name)
		}
		v := g.tmp("o")
		if pred == predD4 {
			g.line("%s := %s.SaturateToUint8().ReshapeToUint32s().GetElem(0)", v, sum)
			g.line("%s", leStore(dq, v, 4))
			continue
		}
		g.line("%s := %s.SaturateToUint8().ReshapeToUint64s().GetElem(0)", v, sum)
		g.line("%s", leStore(dq, v, 8))
	}
}

// strideGuard declines strides above itxGenMaxStride. With the strides in
// that range the row offsets r*ds stay far from overflow, so each row
// window d[r*ds : r*ds+8] is one bounds check.
const strideGuard = "\tif uint(ds) > itxGenMaxStride || uint(ps) > itxGenMaxStride {\n\t\treturn false\n\t}\n"

// leLoad is the little-endian load of the n (4 or 8) bytes of the array
// window w, written out: the compiler merges it into one load, where a call
// to encoding/binary would not be inlined into the larger kernels.
func leLoad(w string, n int) string {
	t := "uint64"
	if n == 4 {
		t = "uint32"
	}
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteString(" | ")
		}
		fmt.Fprintf(&b, "%s(%s[%d])", t, w, i)
		if i > 0 {
			fmt.Fprintf(&b, "<<%d", 8*i)
		}
	}
	return b.String()
}

// leStore is the little-endian store of v into the n (4 or 8) bytes of the
// array window w, merged by the compiler into one store.
func leStore(w, v string, n int) string {
	var l, r []string
	for i := 0; i < n; i++ {
		l = append(l, fmt.Sprintf("%s[%d]", w, i))
		if i == 0 {
			r = append(r, fmt.Sprintf("byte(%s)", v))
		} else {
			r = append(r, fmt.Sprintf("byte(%s>>%d)", v, 8*i))
		}
	}
	return strings.Join(l, ", ") + " = " + strings.Join(r, ", ")
}

// kernel is one generated block function: an n wide, m high (m 0: n) block
// of the vertical (column) and horizontal (row) 1-D types v and h with
// coded coefficients within the cols x rows network forms.
type kernel struct {
	n, m, cols, rows int
	v, h             string
	// hbd selects the 10/12-bit (16bpc) prediction add.
	hbd bool
}

// height is the block's height.
func (k kernel) height() int {
	if k.m == 0 {
		return k.n
	}
	return k.m
}

func (k kernel) name() string {
	if k.m == 0 && k.v == tDCT && k.h == tDCT {
		return fmt.Sprintf("itxGen%sDCT%dx%dC%dR%d", k.depth(), k.n, k.n, k.cols, k.rows)
	}
	return fmt.Sprintf("itxGen%dx%d%s%sC%dR%d", k.n, k.height(), k.v, k.h, k.cols, k.rows)
}

// depth is the name infix of the kernel's bit depth.
func (k kernel) depth() string {
	if k.hbd {
		return "HBD"
	}
	return ""
}

// depthDoc is the kernel's bit depth in comments.
func depthDoc(hbd bool) string {
	if hbd {
		return "10/12-bit"
	}
	return "8-bit"
}

// typeDoc is the block type's name in comments.
func (k kernel) typeDoc() string {
	return typeDoc(k.v) + "_" + typeDoc(k.h)
}

// rowShift is the inter-pass Round2 of the n x n 8-bit transform.
func rowShift(n int) int { return rowShiftWH(n, n) }

// rowShiftWH is the inter-pass Round2 of the w x h 8-bit transform
// (sizeShiftTable, precompute.go).
func rowShiftWH(w, h int) int {
	switch {
	case w == h:
		return min(w/8, 2)
	case w == 2*h || h == 2*w:
		return min(w*h/64, 1)
	}
	// 4:1
	return min(w*h/64, 2)
}

// coeffStride is the column-major coefficient stride of an n x n block
// (64-point blocks code a 32 x 32 corner).
func coeffStride(n int) int { return min(n, 32) }

// header writes a function's constant loads for the body it precedes.
func header(body string) string {
	var out strings.Builder
	out.WriteString("\tt := itxGenKp\n")
	if strings.Contains(body, ", bias)") {
		out.WriteString("\tbias := archsimd.LoadInt32x4Array(&t.bias)\n")
	}
	if strings.Contains(body, "zero") {
		out.WriteString("\tvar zero archsimd.Int16x8\n")
	}
	if strings.Contains(body, "pz.") {
		out.WriteString("\tvar pz archsimd.Uint64x2\n")
	}
	if strings.Contains(body, "pf.") {
		out.WriteString("\tvar pf archsimd.Uint32x4\n")
	}
	if strings.Contains(body, "mxv") {
		out.WriteString("\tmxv := archsimd.BroadcastInt16x8(mx)\n")
	}
	for s := 1; s <= 3; s++ {
		if strings.Contains(body, fmt.Sprintf("sh%d)", s)) {
			fmt.Fprintf(&out, "\tsh%d := archsimd.LoadInt16x8Array(&t.shr%d)\n", s, s)
		}
	}
	return out.String()
}

// wholeKernel writes a block function with the whole block in registers.
// The row pass runs row strips of up to eight rows (lanes = rows) over the
// coded columns; strips past the coded rows are zero. Each strip's outputs
// are rounded and transposed in 8x8 tiles; the column strips of up to
// eight columns (lanes = columns) then run over the coded rows, round by 4
// and add to the prediction. Four-row and four-column strips run four-lane
// networks.
func (g *fgen) wholeKernel(k kernel) string {
	g.sb.Reset()
	g.n = 0
	w, ht := k.n, k.height()
	cs := min(ht, 32)
	rect2 := w == 2*ht || ht == 2*w
	rl, cl := min(ht, 8), min(w, 8)
	rs, cq := (ht+7)/8, (w+7)/8
	// tile[s][q]: the transposed tile of row strip s, column strip q; tile
	// rows are block rows 8s.., lanes the columns 8q...
	tile := make([][][]fv, rs)
	for s := 0; s < rs; s++ {
		tile[s] = make([][]fv, cq)
		if 8*s >= k.rows {
			for q := range tile[s] {
				tile[s][q] = inputs(nil, rl)
			}
			continue
		}
		g.line("// Row strip %d: rows %d..%d, coded columns 0..%d.", s, 8*s, 8*s+rl-1, k.cols-1)
		var live []fv
		for c := 0; c < k.cols; c++ {
			live = append(live, g.load(c*cs+8*s, rl, rect2))
		}
		if k.h != tIden {
			g.guard(live)
		}
		g.reset()
		g.half = rl == 4
		out := g.network1D(k.h, w, inputs(live, w))
		for q := 0; q < cq; q++ {
			rows := make([]fv, 8)
			for j := range rows {
				if 8*q+j < w {
					rows[j] = g.round2(out[8*q+j], rowShiftWH(w, ht))
				} else {
					rows[j] = fv{zero: true}
				}
			}
			t := g.transpose8(rows)[:rl]
			// Rows past the coded ones transform to zero lanes, so the
			// transposed rows 8s+r for 8s+r >= k.rows are zero.
			for r := range t {
				if 8*s+r >= k.rows {
					t[r] = fv{zero: true}
				}
			}
			tile[s][q] = t
		}
	}
	g.half = false
	if k.hbd && w == 4 {
		panic("itxgen: no 4-wide 16bpc kernel")
	}
	col := func(q int) []fv {
		var c []fv
		for s := 0; s < rs; s++ {
			c = append(c, tile[s][q]...)
		}
		return c
	}
	for q := 0; q < cq && k.v != tIden; q++ {
		g.guard(col(q))
	}
	for q := 0; q < cq; q++ {
		g.line("// Column strip %d: columns %d..%d.", q, 8*q, 8*q+cl-1)
		g.reset()
		g.half = cl == 4
		out := g.network1D(k.v, ht, col(q))
		switch {
		case k.hbd && q == 0:
			g.addRows(out, predH16)
		case k.hbd:
			g.line("d, p = d[16:], p[16:]")
			g.addRows(out, predH16)
		case w == 4:
			g.addRows(out, predD4)
		case w == 8:
			g.addRows(out, predD8)
		case q == 0:
			g.addRows(out, predLo)
		default:
			// The right strip of a 16-wide block: its prediction loads
			// start at the block's row start (column 8q-8 = 0).
			g.line("d = d[8:]")
			g.addRows(out, predHi)
		}
		g.half = false
	}
	body := deadStrip(g.sb.String())
	var out strings.Builder
	fmt.Fprintf(&out, "// %s is the %s %dx%d %s inverse transform and add for\n", k.name(), depthDoc(k.hbd), w, ht, k.typeDoc())
	fmt.Fprintf(&out, "// coefficients in the first %d columns and %d rows (column-major at stride\n", k.cols, k.rows)
	fmt.Fprintf(&out, "// %d); d/p hold the destination and prediction rows at strides ds/ps.\n", cs)
	if k.hbd {
		fmt.Fprintf(&out, "// mx is the pixel maximum. It reports false, with dst untouched, when a\n// guard declines or a stride exceeds itxGenMaxStride.\n")
		fmt.Fprintf(&out, "func %s(c *[%d]int32, d []byte, ds int, p []byte, ps int, mx int16) bool {\n", k.name(), w*cs)
	} else {
		fmt.Fprintf(&out, "// It reports false, with dst untouched, when a guard declines or a stride\n// exceeds itxGenMaxStride.\n")
		fmt.Fprintf(&out, "func %s(c *[%d]int32, d []byte, ds int, p []byte, ps int) bool {\n", k.name(), w*cs)
	}
	out.WriteString(strideGuard)
	out.WriteString(header(body))
	out.WriteString(body)
	out.WriteString("\treturn true\n}\n\n")
	return out.String()
}

func rowStripName(t string, n, cols int) string { return fmt.Sprintf("itxGen%s%dRow%d", t, n, cols) }
func colStripName(t string, n, rows int) string { return fmt.Sprintf("itxGen%s%dCol%d", t, n, rows) }

// colStripNameHBD names the 10/12-bit form of a column strip.
func colStripNameHBD(t string, n, rows int) string {
	return fmt.Sprintf("itxGenHBD%s%dCol%d", t, n, rows)
}

// rowStrip writes the row pass of one eight-row strip of an n x n block
// with cols coded columns: the narrowed coefficient loads and their guard,
// the network, the inter-pass rounding and the 8x8 transposes into the
// strip's eight n-wide tmp rows, whose magnitudes it adds to the column
// strips' guards gc.
func (g *fgen) rowStrip(t string, n, cols int) string {
	g.sb.Reset()
	g.n = 0
	cs := coeffStride(n)
	var live []fv
	g.split = splitOn(t, n, cols)
	if g.split {
		live = g.guardedLoads(cols, cs)
	} else {
		for c := 0; c < cols; c++ {
			live = append(live, g.load(c*cs, 8, false))
		}
		if t != tIden {
			g.guard(live)
		}
	}
	g.reset()
	out := g.network1D(t, n, inputs(live, n))
	for _, q := range g.tileOrder(n) {
		rows := make([]fv, 8)
		for j := range rows {
			rows[j] = g.round2(out[8*q+j], rowShift(n))
		}
		tr := g.transpose8(rows)
		acc := g.tmp("g")
		g.line("%s := gc[%d]", acc, q)
		for r, v := range tr {
			// A zero row stays as the zeroed tmp has it.
			if v.zero {
				continue
			}
			off := r*n + 8*q
			g.line("%s.StoreArray((*[8]int16)(m[%d:%d]))", v.name, off, off+8)
		}
		g.l1(acc, false, tr)
		g.line("gc[%d] = %s", q, acc)
	}
	g.done()
	body := deadStrip(g.sb.String())
	var out2 strings.Builder
	name := rowStripName(t, n, cols)
	fmt.Fprintf(&out2, "// %s is the %s row pass of one eight-row strip of an 8-bit %dx%d\n", name, typeDoc(t), n, n)
	fmt.Fprintf(&out2, "// block for %d coded columns: c holds the strip's coefficients\n", cols)
	fmt.Fprintf(&out2, "// (column-major at stride %d), m receives its eight %d-wide rows and gc\n", cs, n)
	fmt.Fprintf(&out2, "// accumulates the column strips' guards. It reports false when the row\n// guard declines.\n")
	fmt.Fprintf(&out2, "func %s(c *[%d]int32, m *[%d]int16, gc *[%d]archsimd.Uint16x8) bool {\n", name, (cols-1)*cs+8, 8*n, n/8)
	out2.WriteString(header(body))
	out2.WriteString(body)
	out2.WriteString("\treturn true\n}\n\n")
	return out2.String()
}

// splitOn reports whether the n-point strip of type t over f coded inputs
// runs split (fgen.split): only the 32-input forms spill enough to pay for
// the parked half (the 8- and 16-input forms measured neutral to +3%).
func splitOn(t string, n, f int) bool {
	return t == tDCT && n >= 32 && f == 32
}

// guardedLoads returns the loads of a row strip's cols coded columns, each
// written at its first use with its L1 added to the row guard, which the
// last one checks.
func (g *fgen) guardedLoads(cols, cs int) []fv {
	acc := g.tmp("g")
	left := cols
	live := make([]fv, cols)
	for c := range live {
		v := g.tmp("c")
		live[c] = g.lazy(v, func() {
			g.loadAs(v, c*cs, 8, false)
			if left == cols {
				g.line("%s := %s.Abs().ToBits()", acc, v)
			} else {
				g.line("%s = %s.AddSaturated(%s.Abs().ToBits())", acc, acc, v)
			}
			left--
			if left == 0 {
				g.line("if %s.ReduceMax() > itx16ColumnL1Bound {", acc)
				g.line("\treturn false")
				g.line("}")
			}
		})
	}
	return live
}

// tileOrder is the order a row strip writes its n/8 output tiles: a split
// strip takes tile q with tile n/8-1-q, which share their butterflies.
func (g *fgen) tileOrder(n int) []int {
	var qs []int
	if g.split {
		for q := 0; q < n/16; q++ {
			qs = append(qs, q, n/8-1-q)
		}
		return qs
	}
	for q := 0; q < n/8; q++ {
		qs = append(qs, q)
	}
	return qs
}

// done ends a strip: every pending value was written.
func (g *fgen) done() {
	if len(g.pend) != 0 {
		panic(fmt.Sprintf("itxgen: %d values never used", len(g.pend)))
	}
	g.split = false
}

// colStrip writes the column pass of one eight-column strip of an n x n
// block from its rows coded tmp rows, with the Round2(., 4) and the
// prediction add.
func (g *fgen) colStrip(t string, n, rows int) string {
	return g.colStripDepth(t, n, rows, false)
}

// colStripDepth is colStrip at either depth: hbd adds the column output to
// eight 16-bit prediction samples per row clipped to [0, mx].
func (g *fgen) colStripDepth(t string, n, rows int, hbd bool) string {
	g.sb.Reset()
	g.n = 0
	var live []fv
	g.split = splitOn(t, n, rows)
	for r := 0; r < rows; r++ {
		v := g.tmp("m")
		ld := func() { g.line("%s := archsimd.LoadInt16x8Array((*[8]int16)(m[%d:%d]))", v, r*n, r*n+8) }
		if g.split {
			live = append(live, g.lazy(v, ld))
			continue
		}
		ld()
		live = append(live, fv{name: v})
	}
	g.reset()
	pred := predSel
	if hbd {
		pred = predH16
	}
	g.addRows(g.network1D(t, n, inputs(live, n)), pred)
	g.done()
	body := deadStrip(g.sb.String())
	var out strings.Builder
	if hbd {
		name := colStripNameHBD(t, n, rows)
		fmt.Fprintf(&out, "// %s is the %s column pass of one eight-column strip of a\n", name, typeDoc(t))
		fmt.Fprintf(&out, "// 10/12-bit %dx%d block from %d coded tmp rows (m, %d wide), added to the\n", n, n, rows, n)
		fmt.Fprintf(&out, "// eight 16-bit prediction samples of each row of p into d (strides ps/ds\n")
		fmt.Fprintf(&out, "// in bytes, at most itxGenMaxStride, checked by the block function),\n")
		fmt.Fprintf(&out, "// clipped to [0, mx].\n")
		fmt.Fprintf(&out, "func %s(m *[%d]int16, d []byte, ds int, p []byte, ps int, mx int16) {\n", name, (rows-1)*n+8)
		out.WriteString("\t// The block function checked the strides; the conversion carries that\n")
		out.WriteString("\t// range here, so each row window costs one bounds check.\n")
		out.WriteString("\tds, ps = int(uint32(ds)), int(uint32(ps))\n")
		out.WriteString(header(body))
		out.WriteString(body)
		out.WriteString("}\n\n")
		return out.String()
	}
	name := colStripName(t, n, rows)
	fmt.Fprintf(&out, "// %s is the %s column pass of one eight-column strip of an\n", name, typeDoc(t))
	fmt.Fprintf(&out, "// 8-bit %dx%d block from %d coded tmp rows (m, %d wide), added to the\n", n, n, rows, n)
	fmt.Fprintf(&out, "// prediction into d (strides ds/ps, at most itxGenMaxStride, checked by\n")
	fmt.Fprintf(&out, "// the block function): p holds sixteen prediction bytes of each row\n")
	fmt.Fprintf(&out, "// inside the block and selp the TBL selector that zero-extends the\n// strip's eight.\n")
	fmt.Fprintf(&out, "func %s(m *[%d]int16, d []byte, ds int, p []byte, ps int, selp *[16]uint8) {\n", name, (rows-1)*n+8)
	out.WriteString("\t// The block function checked the strides; the conversion carries that\n")
	out.WriteString("\t// range here, so each row window costs one bounds check.\n")
	out.WriteString("\tds, ps = int(uint32(ds)), int(uint32(ps))\n")
	out.WriteString("\tsel := archsimd.LoadUint8x16Array(selp)\n")
	out.WriteString(header(body))
	out.WriteString(body)
	out.WriteString("}\n\n")
	return out.String()
}

// stripKernel writes a block function over the strip functions.
func stripKernel(k kernel) string {
	n, cs := k.n, coeffStride(k.n)
	rows8 := (k.rows + 7) / 8 * 8
	var b strings.Builder
	fmt.Fprintf(&b, "// %s is the %s %dx%d %s inverse transform and add for\n", k.name(), depthDoc(k.hbd), n, n, k.typeDoc())
	fmt.Fprintf(&b, "// coefficients in the first %d columns and %d rows (column-major at stride\n", k.cols, k.rows)
	fmt.Fprintf(&b, "// %d); d/p hold the destination and prediction rows at strides ds/ps.\n", cs)
	fmt.Fprintf(&b, "// Coefficient rows from rows on are zero: the row strips past them are\n")
	fmt.Fprintf(&b, "// skipped (tmp starts zero). It reports false, with dst untouched, when a\n// guard declines or a stride exceeds itxGenMaxStride.\n")
	mx := ""
	if k.hbd {
		mx = ", mx int16"
	}
	fmt.Fprintf(&b, "func %s(c *[%d]int32, d []byte, ds int, p []byte, ps int, rows int%s) bool {\n", k.name(), cs*cs, mx)
	b.WriteString(strideGuard)
	fmt.Fprintf(&b, "\tvar tmp [%d]int16\n", rows8*n)
	fmt.Fprintf(&b, "\tvar gc [%d]archsimd.Uint16x8\n", n/8)
	for s := 0; s < rows8/8; s++ {
		cond := ""
		if s > 0 {
			cond = fmt.Sprintf("rows > %d && ", 8*s)
		}
		fmt.Fprintf(&b, "\tif %s!%s((*[%d]int32)(c[%d:]), (*[%d]int16)(tmp[%d:]), &gc) {\n\t\treturn false\n\t}\n",
			cond, rowStripName(k.h, n, k.cols), (k.cols-1)*cs+8, 8*s, 8*n, 8*n*s)
	}
	if k.v != tIden {
		b.WriteString("\tm := gc[0]")
		for q := 1; q < n/8; q++ {
			fmt.Fprintf(&b, ".Max(gc[%d])", q)
		}
		b.WriteString("\n\tif m.ReduceMax() > itx16ColumnL1Bound {\n\t\treturn false\n\t}\n")
	}
	if k.hbd {
		// Strip q reads and writes its own eight 16-bit samples per row.
		for q := 0; q < n/8; q++ {
			if q == 0 {
				fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[0:]), d, ds, p, ps, mx)\n", colStripNameHBD(k.v, n, k.rows), (k.rows-1)*n+8)
				continue
			}
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[%d:]), d[%d:], ds, p[%d:], ps, mx)\n", colStripNameHBD(k.v, n, k.rows), (k.rows-1)*n+8, 8*q, 16*q, 16*q)
		}
		b.WriteString("\treturn true\n}\n\n")
		return b.String()
	}
	// Strip 0 zero-extends the low eight of the sixteen bytes at its own
	// columns, every later strip the high eight of the sixteen ending at
	// its last column.
	b.WriteString("\tsel := &itxGenKp.sel\n")
	for q := 0; q < n/8; q++ {
		switch q {
		case 0:
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[0:]), d, ds, p, ps, &sel[0])\n", colStripName(k.v, n, k.rows), (k.rows-1)*n+8)
		default:
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[%d:]), d[%d:], ds, p[%d:], ps, &sel[1])\n", colStripName(k.v, n, k.rows), (k.rows-1)*n+8, 8*q, 8*q, 8*q-8)
		}
	}
	b.WriteString("\treturn true\n}\n\n")
	return b.String()
}

// fusedSizes lists the generated block sizes and their network forms.
var fusedSizes = []struct {
	n     int
	forms []int
}{{8, []int{4, 8}}, {16, []int{4, 8, 16}}, {32, []int{8, 16, 32}}, {64, []int{8, 16, 32}}}

// fusedKernels lists the generated block functions: every pair of network
// forms of the coded columns and rows.
var fusedKernels = func() []kernel {
	var ks []kernel
	for _, f := range fusedSizes {
		for _, c := range f.forms {
			for _, r := range f.forms {
				ks = append(ks, kernel{n: f.n, cols: c, rows: r, v: tDCT, h: tDCT})
			}
		}
	}
	return ks
}()

// dispatch writes the block function of size n that selects the kernel of
// the network forms covering cols x rows by direct calls.
func dispatch(n int, forms []int) string { return dispatchDepth(n, forms, false) }

// dispatchDepth is dispatch at either depth (hbd: the 10/12-bit kernels,
// with the pixel maximum mx).
func dispatchDepth(n int, forms []int, hbd bool) string {
	var b strings.Builder
	cs := coeffStride(n)
	extra := ""
	if n > 16 {
		extra = ", rows"
	}
	dep, mx := "", ""
	if hbd {
		dep, mx = "HBD", ", mx int16"
		extra += ", mx"
		fmt.Fprintf(&b, "// itxGenHBDDCT%dx%d runs the generated 10/12-bit %dx%d kernel of the network\n", n, n, n, n)
		fmt.Fprintf(&b, "// forms covering cols coded columns and rows coded rows (1..%d each).\n", forms[len(forms)-1])
	} else {
		fmt.Fprintf(&b, "// itxGenDCT%dx%d runs the generated %dx%d kernel of the network forms covering\n", n, n, n, n)
		fmt.Fprintf(&b, "// cols coded columns and rows coded rows (1..%d each).\n", forms[len(forms)-1])
	}
	fmt.Fprintf(&b, "func itxGen%sDCT%dx%d(c *[%d]int32, d []byte, ds int, p []byte, ps int, cols, rows int%s) bool {\n", dep, n, n, cs*cs, mx)
	sw := func(v string, body func(f int)) {
		b.WriteString("\tswitch {\n")
		for i, f := range forms {
			if i < len(forms)-1 {
				fmt.Fprintf(&b, "\tcase %s <= %d:\n", v, f)
			} else {
				b.WriteString("\tdefault:\n")
			}
			body(f)
		}
		b.WriteString("\t}\n")
	}
	sw("cols", func(c int) {
		sw("rows", func(r int) {
			fmt.Fprintf(&b, "\t\treturn %s(c, d, ds, p, ps%s)\n", kernel{n: n, cols: c, rows: r, v: tDCT, h: tDCT, hbd: hbd}.name(), extra)
		})
	})
	b.WriteString("}\n\n")
	return b.String()
}

// generateFused returns the generated kernel file keyed by its path under dir.
func generateFused(dir string) map[string][]byte {
	g := &fgen{consts: map[int]bool{}, raw: map[string]int{}}
	var funcs strings.Builder
	// 8x8 and 16x16 switch in their entry points (one call level less).
	for _, f := range fusedSizes {
		if f.n > 16 {
			funcs.WriteString(dispatch(f.n, f.forms))
		}
	}
	for _, k := range fusedKernels {
		if k.n <= 16 {
			funcs.WriteString(g.wholeKernel(k))
		} else {
			funcs.WriteString(stripKernel(k))
		}
	}
	for _, n := range []int{32, 64} {
		for _, f := range []int{8, 16, 32} {
			funcs.WriteString(g.rowStrip(tDCT, n, f))
		}
		for _, f := range []int{8, 16, 32} {
			funcs.WriteString(g.colStrip(tDCT, n, f))
		}
	}
	g.generateHybrid(&funcs)
	// The 10/12-bit DCT_DCT kernels: their own column passes over the 8-bit
	// row strips.
	for _, f := range fusedSizes {
		funcs.WriteString(dispatchDepth(f.n, f.forms, true))
	}
	for _, k := range fusedKernels {
		k.hbd = true
		if k.n <= 16 {
			funcs.WriteString(g.wholeKernel(k))
		} else {
			funcs.WriteString(stripKernel(k))
		}
	}
	for _, n := range []int{32, 64} {
		for _, f := range []int{8, 16, 32} {
			funcs.WriteString(g.colStripDepth(tDCT, n, f, true))
		}
	}
	var cs []int
	for c := range g.consts {
		cs = append(cs, c)
	}
	slices.Sort(cs)
	var out strings.Builder
	out.WriteString("// Code generated by itxgen from the DCT networks of dct.go. DO NOT EDIT.\n\n")
	out.WriteString("//go:build goexperiment.simd && arm64 && !purego\n\n")
	out.WriteString("package transform\n\n")
	out.WriteString("import (\n\t\"errors\"\n\t\"simd/archsimd\"\n)\n\n")
	out.WriteString("// The kernels read the prediction and write the destination through\n")
	out.WriteString("// bounds-checked row windows of d and p. The block functions decline\n")
	out.WriteString("// strides above itxGenMaxStride (the caller then runs the staged path).\n")
	out.WriteString("const itxGenMaxStride = 1 << 30\n\n")
	out.WriteString("// errItxGenWindow reports a row window outside d or p: a caller that\n")
	out.WriteString("// passed slices shorter than the block's rows.\n")
	out.WriteString("var errItxGenWindow = errors.New(\"transform: inverse-add row outside the destination or prediction\")\n\n")
	out.WriteString("// itxGenConsts holds the constants of the generated kernels: the rotation\n")
	out.WriteString("// multipliers 16*C (pC / nC for the cospi constant +C / -C), the negated\n")
	out.WriteString("// shift counts, the Round2(., 12) bias 2^15 of the multiply-accumulate and\n")
	out.WriteString("// the TBL selectors sel[0] / sel[1] that zero-extend bytes 0..7 / 8..15 to\n")
	out.WriteString("// 16-bit lanes (an index of 255 reads zero).\n")
	out.WriteString("type itxGenConsts struct {\n")
	for _, c := range cs {
		fmt.Fprintf(&out, "\t%s [4]int32\n", constField(c))
	}
	var rs []string
	for f := range g.raw {
		rs = append(rs, f)
	}
	slices.Sort(rs)
	for _, f := range rs {
		fmt.Fprintf(&out, "\t%s [4]int32\n", f)
	}
	out.WriteString("\tshr1, shr2, shr3 [8]int16\n")
	out.WriteString("\tbias       [4]int32\n")
	out.WriteString("\tsel        [2][16]uint8\n}\n\n")
	out.WriteString("var itxGenK = itxGenConsts{\n")
	for _, c := range cs {
		fmt.Fprintf(&out, "\t%s: [4]int32{%d, %d, %d, %d},\n", constField(c), 16*c, 16*c, 16*c, 16*c)
	}
	for _, f := range rs {
		v := g.raw[f]
		fmt.Fprintf(&out, "\t%s: [4]int32{%d, %d, %d, %d},\n", f, v, v, v, v)
	}
	out.WriteString("\tshr1: [8]int16{-1, -1, -1, -1, -1, -1, -1, -1},\n")
	out.WriteString("\tshr2: [8]int16{-2, -2, -2, -2, -2, -2, -2, -2},\n")
	out.WriteString("\tshr3: [8]int16{-3, -3, -3, -3, -3, -3, -3, -3},\n")
	out.WriteString("\tbias: [4]int32{32768, 32768, 32768, 32768},\n")
	out.WriteString("\tsel: [2][16]uint8{\n")
	out.WriteString("\t\t{0, 255, 1, 255, 2, 255, 3, 255, 4, 255, 5, 255, 6, 255, 7, 255},\n")
	out.WriteString("\t\t{8, 255, 9, 255, 10, 255, 11, 255, 12, 255, 13, 255, 14, 255, 15, 255},\n\t},\n}\n\n")
	out.WriteString("// itxGenKp is loaded once per function; every constant is then one load\n")
	out.WriteString("// at a fixed offset from it (the address of itxGenK itself would be\n")
	out.WriteString("// rematerialized with ADRP+ADD at each use).\n")
	out.WriteString("var itxGenKp = &itxGenK\n\n")
	out.WriteString(funcs.String())
	src, err := format.Source([]byte(out.String()))
	if err != nil {
		panic(err)
	}
	return map[string][]byte{filepath.Join(dir, "itx_fused_gen_arm64.go"): src}
}

// writeFused regenerates the fused kernel file in dir.
func writeFused(dir string) error {
	for name, src := range generateFused(dir) {
		if err := os.WriteFile(name, src, 0o644); err != nil {
			return err
		}
	}
	return nil
}

var (
	declRe  = regexp.MustCompile(`^\t([A-Za-z_][A-Za-z0-9_]*) := `)
	identRe = regexp.MustCompile(`[A-Za-z_][A-Za-z0-9_]*`)
)

// deadStrip drops the single-name declarations nothing reads (the transposed
// rows a zero-folded network never consumes and their interleave inputs).
func deadStrip(body string) string {
	lines := strings.Split(strings.TrimSuffix(body, "\n"), "\n")
	for {
		uses := map[string]int{}
		for _, l := range lines {
			start := 0
			if m := declRe.FindStringSubmatchIndex(l); m != nil {
				start = m[1]
			}
			for _, id := range identRe.FindAllString(l[start:], -1) {
				uses[id]++
			}
		}
		var kept []string
		for _, l := range lines {
			if m := declRe.FindStringSubmatch(l); m != nil && uses[m[1]] == 0 {
				continue
			}
			kept = append(kept, l)
		}
		if len(kept) == len(lines) {
			return strings.Join(lines, "\n") + "\n"
		}
		lines = kept
	}
}
