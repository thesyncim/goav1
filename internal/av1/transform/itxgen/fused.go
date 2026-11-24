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
	x := v.name
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
	x, y := a.name, b.name
	g.line("%s := %s.AddSaturated(%s)", n, x, y)
	return fv{name: n}
}

func (g *fgen) sub(a, b fv) fv {
	switch {
	case b.zero:
		return a
	case a.zero:
		n := g.tmp("a")
		g.line("%s := zero.SubSaturated(%s)", n, b.name)
		return fv{name: n}
	}
	n := g.tmp("a")
	x, y := a.name, b.name
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
			g.line("%s := %s.Abs().ToBits()", acc, v.name)
		default:
			g.line("%s = %s.AddSaturated(%s.Abs().ToBits())", acc, acc, v.name)
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
		x, y := op(rows[i].name, rows[i+1].name, "zero")
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
		g.line("%s := %s.Average(zero)", n, v.name)
	} else {
		g.line("%s := %s.Shift(sh%d).Average(zero)", n, v.name, s-1)
	}
	return fv{name: n}
}

// load writes the int16-saturating load of the lanes (8 or 4) int32
// coefficients at c[off:] (the row clamp of the 8-bit row pass), after the
// rect2 scale Round2(x*2896, 12) = (x*181 + 128) >> 8 when rect2 is set
// (with x first clamped to +-2^22, which keeps the product inside int32
// and changes no result: the scaled value saturates either way).
func (g *fgen) load(off, lanes int, rect2 bool) fv {
	half := func(o int) string {
		x := fmt.Sprintf("archsimd.LoadInt32x4Array((*[4]int32)(c[%d:%d]))", o, o+4)
		if rect2 {
			x += fmt.Sprintf(".Max(%s).Min(%s).MulAdd(%s, %s).ShiftAllRight(8)",
				g.kraw("rlo", -(1<<22)), g.kraw("rhi", 1<<22), g.kraw("r181", 181), g.kraw("r128", 128))
		}
		return x + ".SaturateToInt16()"
	}
	if lanes == 4 {
		v := g.tmp("c")
		g.line("%s := %s", v, half(off))
		return fv{name: v}
	}
	n := g.tmp("c")
	g.line("%s := %s.ToBits().ReshapeToUint64s().BitsToFloat64()", n, half(off))
	g.line("%s = %s.SetElem(1, %s.ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))", n, n, half(off+4))
	v := g.tmp("c")
	g.line("%s := %s.ToBits().ReshapeToUint16s().BitsToInt16()", v, n)
	return fv{name: v}
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

// addRows writes the column output of one eight-column strip: Round2(., 4)
// of out[r], added to prediction row r at pq and saturated into dst row r
// at dq, for the n rows, with the dst and prediction strip pointers taken
// from dp and pp.
func (g *fgen) addRows(out []fv, first string, pred int) {
	g.line("dq, pq %s dp, pp", first)
	for r := range out {
		res := g.round2(out[r], 4)
		if r > 0 {
			g.line("dq, pq = unsafe.Add(dq, ds), unsafe.Add(pq, ps)")
		}
		p := g.tmp("p")
		switch pred {
		case predLo:
			g.line("%s := archsimd.LoadUint8x16Array((*[16]uint8)(pq)).ExtendLo8ToUint16().BitsToInt16()", p)
		case predHi:
			g.line("%s := archsimd.LoadUint8x16Array((*[16]uint8)(pq)).HiToLo().ExtendLo8ToUint16().BitsToInt16()", p)
		case predSel:
			g.line("%s := archsimd.LoadUint8x16Array((*[16]uint8)(pq)).LookupOrZero(sel).ReshapeToUint16s().BitsToInt16()", p)
		case predD8:
			g.line("%s := pz.SetElem(0, *(*float64)(pq)).ToBits().ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()", p)
		case predD4:
			g.line("%s := pf.SetElem(0, *(*float32)(pq)).ToBits().ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()", p)
		case predH16:
			g.line("%s := archsimd.LoadInt16x8Array((*[8]int16)(pq))", p)
			if res.zero {
				g.line("%s.StoreArray((*[8]int16)(dq))", p)
			} else {
				g.line("%s.AddSaturated(%s).Max(zero).Min(mxv).StoreArray((*[8]int16)(dq))", p, res.name)
			}
			continue
		}
		if pred == predD4 {
			sum := p
			if !res.zero {
				sum = fmt.Sprintf("%s.Add(%s)", p, res.name)
			}
			g.line("*(*float32)(dq) = %s.SaturateToUint8().ReshapeToUint32s().BitsToFloat32().GetElem(0)", sum)
			continue
		}
		if res.zero {
			g.line("*(*float64)(dq) = %s.SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)", p)
		} else {
			g.line("*(*float64)(dq) = %s.Add(%s).SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)", p, res.name)
		}
	}
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
		out.WriteString("\tvar pz archsimd.Float64x2\n")
	}
	if strings.Contains(body, "pf.") {
		out.WriteString("\tvar pf archsimd.Float32x4\n")
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
			g.addRows(out, ":=", predH16)
		case k.hbd:
			g.line("dp, pp = unsafe.Add(dp, 16), unsafe.Add(pp, 16)")
			g.addRows(out, "=", predH16)
		case w == 4:
			g.addRows(out, ":=", predD4)
		case w == 8:
			g.addRows(out, ":=", predD8)
		case q == 0:
			g.addRows(out, ":=", predLo)
		default:
			// The right strip of a 16-wide block: its prediction loads
			// start at the block's row start (column 8q-8 = 0).
			g.line("dp = unsafe.Add(dp, 8)")
			g.addRows(out, "=", predHi)
		}
		g.half = false
	}
	body := deadStrip(g.sb.String())
	var out strings.Builder
	fmt.Fprintf(&out, "// %s is the %s %dx%d %s inverse transform and add for\n", k.name(), depthDoc(k.hbd), w, ht, k.typeDoc())
	fmt.Fprintf(&out, "// coefficients in the first %d columns and %d rows (column-major at stride\n", k.cols, k.rows)
	fmt.Fprintf(&out, "// %d); dp/pp address the destination and prediction at strides ds/ps,\n", cs)
	if k.hbd {
		fmt.Fprintf(&out, "// which the caller has bounds-checked, and mx is the pixel maximum. It\n// reports false, with dst untouched, when a guard declines.\n")
		fmt.Fprintf(&out, "func %s(c *[%d]int32, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, mx int16) bool {\n", k.name(), w*cs)
	} else {
		fmt.Fprintf(&out, "// which the caller has bounds-checked. It reports false, with dst\n// untouched, when a guard declines.\n")
		fmt.Fprintf(&out, "func %s(c *[%d]int32, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int) bool {\n", k.name(), w*cs)
	}
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
	for c := 0; c < cols; c++ {
		live = append(live, g.load(c*cs, 8, false))
	}
	if t != tIden {
		g.guard(live)
	}
	g.reset()
	out := g.network1D(t, n, inputs(live, n))
	for q := 0; q < n/8; q++ {
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
	for r := 0; r < rows; r++ {
		v := g.tmp("m")
		g.line("%s := archsimd.LoadInt16x8Array((*[8]int16)(m[%d:%d]))", v, r*n, r*n+8)
		live = append(live, fv{name: v})
	}
	g.reset()
	pred := predSel
	if hbd {
		pred = predH16
	}
	g.addRows(g.network1D(t, n, inputs(live, n)), ":=", pred)
	body := deadStrip(g.sb.String())
	var out strings.Builder
	if hbd {
		name := colStripNameHBD(t, n, rows)
		fmt.Fprintf(&out, "// %s is the %s column pass of one eight-column strip of a\n", name, typeDoc(t))
		fmt.Fprintf(&out, "// 10/12-bit %dx%d block from %d coded tmp rows (m, %d wide), added to the\n", n, n, rows, n)
		fmt.Fprintf(&out, "// eight 16-bit prediction samples of each row at pp into dp (strides\n")
		fmt.Fprintf(&out, "// ps/ds in bytes, bounds-checked by the caller), clipped to [0, mx].\n")
		fmt.Fprintf(&out, "func %s(m *[%d]int16, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, mx int16) {\n", name, (rows-1)*n+8)
		out.WriteString(header(body))
		out.WriteString(body)
		out.WriteString("}\n\n")
		return out.String()
	}
	name := colStripName(t, n, rows)
	fmt.Fprintf(&out, "// %s is the %s column pass of one eight-column strip of an\n", name, typeDoc(t))
	fmt.Fprintf(&out, "// 8-bit %dx%d block from %d coded tmp rows (m, %d wide), added to the\n", n, n, rows, n)
	fmt.Fprintf(&out, "// prediction into dp (strides ps/ds, bounds-checked by the caller): pp\n")
	fmt.Fprintf(&out, "// addresses sixteen prediction bytes of each row inside the block and\n")
	fmt.Fprintf(&out, "// selp the TBL selector that zero-extends the strip's eight.\n")
	fmt.Fprintf(&out, "func %s(m *[%d]int16, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, selp *[16]uint8) {\n", name, (rows-1)*n+8)
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
	fmt.Fprintf(&b, "// %d); dp/pp address the destination and prediction at strides ds/ps,\n", cs)
	fmt.Fprintf(&b, "// which the caller has bounds-checked. Coefficient rows from rows on are\n")
	fmt.Fprintf(&b, "// zero: the row strips past them are skipped (tmp starts zero). It reports\n// false, with dst untouched, when a guard declines.\n")
	mx := ""
	if k.hbd {
		mx = ", mx int16"
	}
	fmt.Fprintf(&b, "func %s(c *[%d]int32, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, rows int%s) bool {\n", k.name(), cs*cs, mx)
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
				fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[0:]), dp, ds, pp, ps, mx)\n", colStripNameHBD(k.v, n, k.rows), (k.rows-1)*n+8)
				continue
			}
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[%d:]), unsafe.Add(dp, %d), ds, unsafe.Add(pp, %d), ps, mx)\n", colStripNameHBD(k.v, n, k.rows), (k.rows-1)*n+8, 8*q, 16*q, 16*q)
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
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[0:]), dp, ds, pp, ps, &sel[0])\n", colStripName(k.v, n, k.rows), (k.rows-1)*n+8)
		default:
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[%d:]), unsafe.Add(dp, %d), ds, unsafe.Add(pp, %d), ps, &sel[1])\n", colStripName(k.v, n, k.rows), (k.rows-1)*n+8, 8*q, 8*q, 8*q-8)
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
	d, mx := "", ""
	if hbd {
		d, mx = "HBD", ", mx int16"
		extra += ", mx"
		fmt.Fprintf(&b, "// itxGenHBDDCT%dx%d runs the generated 10/12-bit %dx%d kernel of the network\n", n, n, n, n)
		fmt.Fprintf(&b, "// forms covering cols coded columns and rows coded rows (1..%d each).\n", forms[len(forms)-1])
	} else {
		fmt.Fprintf(&b, "// itxGenDCT%dx%d runs the generated %dx%d kernel of the network forms covering\n", n, n, n, n)
		fmt.Fprintf(&b, "// cols coded columns and rows coded rows (1..%d each).\n", forms[len(forms)-1])
	}
	fmt.Fprintf(&b, "func itxGen%sDCT%dx%d(c *[%d]int32, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, cols, rows int%s) bool {\n", d, n, n, cs*cs, mx)
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
			fmt.Fprintf(&b, "\t\treturn %s(c, dp, ds, pp, ps%s)\n", kernel{n: n, cols: c, rows: r, v: tDCT, h: tDCT, hbd: hbd}.name(), extra)
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
	out.WriteString("import (\n\t\"simd/archsimd\"\n\t\"unsafe\"\n)\n\n")
	out.WriteString("// The kernels step dp/pp through the destination and prediction rows as\n")
	out.WriteString("// pointers inside the extent their callers checked: these are residual-add\n")
	out.WriteString("// rows, where per-row checked slices measured 9-86% slower.\n\n")
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
