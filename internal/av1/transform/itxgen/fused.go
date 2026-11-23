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
type fv struct {
	zero bool
	name string
}

type fgen struct {
	sb     strings.Builder
	n      int
	memo   map[string]fv
	wide   map[string][2]string
	consts map[int]bool
}

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
// network instance.
func (g *fgen) widen(v fv) (string, string) {
	if w, ok := g.wide[v.name]; ok {
		return w[0], w[1]
	}
	lo, hi := g.tmp("w"), g.tmp("w")
	x := v.name
	g.line("%s := %s.ExtendLo4ToInt32()", lo, x)
	g.line("%s := %s.HiToLo().ExtendLo4ToInt32()", hi, x)
	g.wide[v.name] = [2]string{lo, hi}
	return lo, hi
}

// rot is Round2(a*ca + b*cb, 12) on eight int16 lanes: on each
// sign-extended half, 2^15 + a*(16ca) + b*(16cb) by multiply-accumulate
// (the final sum fits int32, so a wrapped partial sum is harmless), then
// the high halves of the 32-bit sums. Zero operands fold away and a
// repeated rotation is computed once.
func (g *fgen) rot(a fv, ca int, b fv, cb int) fv {
	if a.zero && b.zero {
		return fv{zero: true}
	}
	if a.zero {
		a, ca, b, cb = b, cb, fv{zero: true}, 0
	}
	key := fmt.Sprintf("%s|%d|%s|%d", a.name, ca, b.name, cb)
	if v, ok := g.memo[key]; ok {
		return v
	}
	n := g.tmp("r")
	al, ah := g.widen(a)
	ka := g.tmp("k")
	g.line("%s := %s", ka, g.k(ca))
	g.line("%sl := %s.MulAdd(%s, bias)", n, al, ka)
	g.line("%sh := %s.MulAdd(%s, bias)", n, ah, ka)
	if !b.zero {
		bl, bh := g.widen(b)
		kb := g.tmp("k")
		g.line("%s := %s", kb, g.k(cb))
		g.line("%sl = %s.MulAdd(%s, %sl)", n, bl, kb, n)
		g.line("%sh = %s.MulAdd(%s, %sh)", n, bh, kb, n)
	}
	g.line("%s := %sl.ToBits().ReshapeToUint16s().ConcatOdd(%sh.ToBits().ReshapeToUint16s()).BitsToInt16()", n, n, n)
	v := fv{name: n}
	g.memo[key] = v
	return v
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
// three interleave stages; a zero row is a zero operand.
func (g *fgen) transpose8(rows []fv) []fv {
	name := func(v fv) string {
		if v.zero {
			return "zero"
		}
		return v.name
	}
	p := g.tmp("x")
	for i := 0; i < 8; i += 2 {
		g.line("%sa%d := %s.InterleaveLo(%s).ToBits().ReshapeToUint32s()", p, i, name(rows[i]), name(rows[i+1]))
		g.line("%sa%d := %s.InterleaveHi(%s).ToBits().ReshapeToUint32s()", p, i+1, name(rows[i]), name(rows[i+1]))
	}
	for _, q := range [][4]int{{0, 1, 0, 2}, {2, 3, 1, 3}, {4, 5, 4, 6}, {6, 7, 5, 7}} {
		g.line("%sb%d := %sa%d.InterleaveLo(%sa%d).ReshapeToUint64s()", p, q[0], p, q[2], p, q[3])
		g.line("%sb%d := %sa%d.InterleaveHi(%sa%d).ReshapeToUint64s()", p, q[1], p, q[2], p, q[3])
	}
	out := make([]fv, 8)
	for i, q := range [][2]int{{0, 4}, {1, 5}, {2, 6}, {3, 7}} {
		lo, hi := fmt.Sprintf("%st%d", p, 2*i), fmt.Sprintf("%st%d", p, 2*i+1)
		g.line("%s := %sb%d.InterleaveLo(%sb%d).ReshapeToUint16s().BitsToInt16()", lo, p, q[0], p, q[1])
		g.line("%s := %sb%d.InterleaveHi(%sb%d).ReshapeToUint16s().BitsToInt16()", hi, p, q[0], p, q[1])
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
	n := g.tmp("s")
	if s == 1 {
		g.line("%s := %s.Average(zero)", n, v.name)
	} else {
		g.line("%s := %s.Shift(sh%d).Average(zero)", n, v.name, s-1)
	}
	return fv{name: n}
}

// load writes the int16-saturating load of the eight int32 coefficients at
// c[off:off+8] (the row clamp of the 8-bit row pass).
func (g *fgen) load(off int) fv {
	n := g.tmp("c")
	g.line("%s := archsimd.LoadInt32x4Array((*[4]int32)(c[%d:%d])).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()", n, off, off+4)
	g.line("%s = %s.SetElem(1, archsimd.LoadInt32x4Array((*[4]int32)(c[%d:%d])).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))", n, n, off+4, off+8)
	v := g.tmp("c")
	g.line("%s := %s.ToBits().ReshapeToUint16s().BitsToInt16()", v, n)
	return fv{name: v}
}

// Prediction row loads of an eight-column strip. Every load stays inside
// the block's own prediction rows (another worker may be writing the
// neighbouring blocks): predLo loads the strip's eight bytes and the next
// eight, predHi the eight before and the strip's (pq is the strip's column
// minus 8), predSel sixteen bytes zero-extended by the caller's TBL
// selector (either of the two), predD8 only the eight bytes (8x8).
const (
	predLo = iota
	predHi
	predSel
	predD8
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
		}
		if res.zero {
			g.line("*(*float64)(dq) = %s.SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)", p)
		} else {
			g.line("*(*float64)(dq) = %s.Add(%s).SaturateToUint8().ReshapeToUint64s().BitsToFloat64().GetElem(0)", p, res.name)
		}
	}
}

// kernel is one generated block function: an n x n DCT_DCT block with
// coded coefficients within the cols x rows network forms.
type kernel struct {
	n, cols, rows int
}

func (k kernel) name() string {
	return fmt.Sprintf("itxGenDCT%dx%dC%dR%d", k.n, k.n, k.cols, k.rows)
}

// rowShift is the inter-pass Round2 of the n x n 8-bit transform.
func rowShift(n int) int {
	if n == 8 {
		return 1
	}
	return 2
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
	for s := 1; s <= 3; s++ {
		if strings.Contains(body, fmt.Sprintf("sh%d)", s)) {
			fmt.Fprintf(&out, "\tsh%d := archsimd.LoadInt16x8Array(&t.shr%d)\n", s, s)
		}
	}
	return out.String()
}

// wholeKernel writes an 8x8 or 16x16 block function. The row pass runs
// eight-row strips (lanes = rows) over the coded columns; strips past the
// coded rows are zero. Each strip's outputs are rounded and transposed in
// 8x8 tiles; the column strips (lanes = columns) then run over the coded
// rows, round by 4 and add to the prediction.
func (g *fgen) wholeKernel(k kernel) string {
	g.sb.Reset()
	g.n = 0
	n := k.n
	strips := n / 8
	// tile[s][q]: the transposed 8x8 tile of row strip s, column strip q;
	// tile rows are block rows 8s..8s+7, lanes the columns 8q..8q+7.
	tile := make([][][]fv, strips)
	for s := 0; s < strips; s++ {
		tile[s] = make([][]fv, strips)
		if 8*s >= k.rows {
			for q := range tile[s] {
				tile[s][q] = inputs(nil, 8)
			}
			continue
		}
		g.line("// Row strip %d: rows %d..%d, coded columns 0..%d.", s, 8*s, 8*s+7, k.cols-1)
		var live []fv
		for c := 0; c < k.cols; c++ {
			live = append(live, g.load(c*n+8*s))
		}
		g.guard(live)
		g.reset()
		out := g.network(n, inputs(live, n))
		for q := 0; q < strips; q++ {
			rows := make([]fv, 8)
			for j := range rows {
				rows[j] = g.round2(out[8*q+j], rowShift(n))
			}
			t := g.transpose8(rows)
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
	col := func(q int) []fv {
		var c []fv
		for s := 0; s < strips; s++ {
			c = append(c, tile[s][q]...)
		}
		return c
	}
	for q := 0; q < strips; q++ {
		g.guard(col(q))
	}
	for q := 0; q < strips; q++ {
		g.line("// Column strip %d: columns %d..%d.", q, 8*q, 8*q+7)
		g.reset()
		out := g.network(n, col(q))
		switch {
		case n == 8:
			g.addRows(out, ":=", predD8)
		case q == 0:
			g.addRows(out, ":=", predLo)
		default:
			// The right strip of 16x16: its prediction loads start at
			// the block's row start (column 8q-8 = 0).
			g.line("dp = unsafe.Add(dp, 8)")
			g.addRows(out, "=", predHi)
		}
	}
	body := deadStrip(g.sb.String())
	var out strings.Builder
	fmt.Fprintf(&out, "// %s is the 8-bit %dx%d DCT_DCT inverse transform and add for\n", k.name(), n, n)
	fmt.Fprintf(&out, "// coefficients in the first %d columns and %d rows (column-major at stride\n", k.cols, k.rows)
	fmt.Fprintf(&out, "// %d); dp/pp address the destination and prediction at strides ds/ps,\n", coeffStride(n))
	fmt.Fprintf(&out, "// which the caller has bounds-checked. It reports false, with dst\n// untouched, when a guard declines.\n")
	fmt.Fprintf(&out, "func %s(c *[%d]int32, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int) bool {\n", k.name(), n*n)
	out.WriteString(header(body))
	out.WriteString(body)
	out.WriteString("\treturn true\n}\n\n")
	return out.String()
}

func rowStripName(n, cols int) string { return fmt.Sprintf("itxGenDCT%dRow%d", n, cols) }
func colStripName(n, rows int) string { return fmt.Sprintf("itxGenDCT%dCol%d", n, rows) }

// rowStrip writes the row pass of one eight-row strip of an n x n block
// with cols coded columns: the narrowed coefficient loads and their guard,
// the network, the inter-pass rounding and the 8x8 transposes into the
// strip's eight n-wide tmp rows, whose magnitudes it adds to the column
// strips' guards gc.
func (g *fgen) rowStrip(n, cols int) string {
	g.sb.Reset()
	g.n = 0
	cs := coeffStride(n)
	var live []fv
	for c := 0; c < cols; c++ {
		live = append(live, g.load(c*cs))
	}
	g.guard(live)
	g.reset()
	out := g.network(n, inputs(live, n))
	for q := 0; q < n/8; q++ {
		rows := make([]fv, 8)
		for j := range rows {
			rows[j] = g.round2(out[8*q+j], rowShift(n))
		}
		t := g.transpose8(rows)
		acc := g.tmp("g")
		g.line("%s := gc[%d]", acc, q)
		for r, v := range t {
			off := r*n + 8*q
			g.line("%s.StoreArray((*[8]int16)(m[%d:%d]))", v.name, off, off+8)
		}
		g.l1(acc, false, t)
		g.line("gc[%d] = %s", q, acc)
	}
	body := deadStrip(g.sb.String())
	var out2 strings.Builder
	name := rowStripName(n, cols)
	fmt.Fprintf(&out2, "// %s is the row pass of one eight-row strip of the 8-bit %dx%d\n", name, n, n)
	fmt.Fprintf(&out2, "// DCT_DCT block for %d coded columns: c holds the strip's coefficients\n", cols)
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
func (g *fgen) colStrip(n, rows int) string {
	g.sb.Reset()
	g.n = 0
	var live []fv
	for r := 0; r < rows; r++ {
		v := g.tmp("m")
		g.line("%s := archsimd.LoadInt16x8Array((*[8]int16)(m[%d:%d]))", v, r*n, r*n+8)
		live = append(live, fv{name: v})
	}
	g.reset()
	g.addRows(g.network(n, inputs(live, n)), ":=", predSel)
	body := deadStrip(g.sb.String())
	var out strings.Builder
	name := colStripName(n, rows)
	fmt.Fprintf(&out, "// %s is the column pass of one eight-column strip of the 8-bit\n", name)
	fmt.Fprintf(&out, "// %dx%d DCT_DCT block from %d coded tmp rows (m, %d wide), added to the\n", n, n, rows, n)
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

// stripKernel writes a 32x32 or 64x64 block function over the strip
// functions.
func stripKernel(k kernel) string {
	n, cs := k.n, coeffStride(k.n)
	var b strings.Builder
	fmt.Fprintf(&b, "// %s is the 8-bit %dx%d DCT_DCT inverse transform and add for\n", k.name(), n, n)
	fmt.Fprintf(&b, "// coefficients in the first %d columns and %d rows (column-major at stride\n", k.cols, k.rows)
	fmt.Fprintf(&b, "// %d); dp/pp address the destination and prediction at strides ds/ps,\n", cs)
	fmt.Fprintf(&b, "// which the caller has bounds-checked. Coefficient rows from rows on are\n")
	fmt.Fprintf(&b, "// zero: the row strips past them are skipped (tmp starts zero). It reports\n// false, with dst untouched, when a guard declines.\n")
	fmt.Fprintf(&b, "func %s(c *[%d]int32, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, rows int) bool {\n", k.name(), cs*cs)
	fmt.Fprintf(&b, "\tvar tmp [%d]int16\n", k.rows*n)
	fmt.Fprintf(&b, "\tvar gc [%d]archsimd.Uint16x8\n", n/8)
	for s := 0; s < k.rows/8; s++ {
		cond := ""
		if s > 0 {
			cond = fmt.Sprintf("rows > %d && ", 8*s)
		}
		fmt.Fprintf(&b, "\tif %s!%s((*[%d]int32)(c[%d:]), (*[%d]int16)(tmp[%d:]), &gc) {\n\t\treturn false\n\t}\n",
			cond, rowStripName(n, k.cols), (k.cols-1)*cs+8, 8*s, 8*n, 8*n*s)
	}
	b.WriteString("\tm := gc[0]")
	for q := 1; q < n/8; q++ {
		fmt.Fprintf(&b, ".Max(gc[%d])", q)
	}
	b.WriteString("\n\tif m.ReduceMax() > itx16ColumnL1Bound {\n\t\treturn false\n\t}\n")
	// Strip 0 zero-extends the low eight of the sixteen bytes at its own
	// columns, every later strip the high eight of the sixteen ending at
	// its last column.
	b.WriteString("\tsel := &itxGenKp.sel\n")
	for q := 0; q < n/8; q++ {
		switch q {
		case 0:
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[0:]), dp, ds, pp, ps, &sel[0])\n", colStripName(n, k.rows), (k.rows-1)*n+8)
		default:
			fmt.Fprintf(&b, "\t%s((*[%d]int16)(tmp[%d:]), unsafe.Add(dp, %d), ds, unsafe.Add(pp, %d), ps, &sel[1])\n", colStripName(n, k.rows), (k.rows-1)*n+8, 8*q, 8*q, 8*q-8)
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
				ks = append(ks, kernel{n: f.n, cols: c, rows: r})
			}
		}
	}
	return ks
}()

// dispatch writes the block function of size n that selects the kernel of
// the network forms covering cols x rows by direct calls.
func dispatch(n int, forms []int) string {
	var b strings.Builder
	cs := coeffStride(n)
	extra := ""
	if n > 16 {
		extra = ", rows"
	}
	fmt.Fprintf(&b, "// itxGenDCT%dx%d runs the generated %dx%d kernel of the network forms covering\n", n, n, n, n)
	fmt.Fprintf(&b, "// cols coded columns and rows coded rows (1..%d each).\n", forms[len(forms)-1])
	fmt.Fprintf(&b, "func itxGenDCT%dx%d(c *[%d]int32, dp unsafe.Pointer, ds int, pp unsafe.Pointer, ps int, cols, rows int) bool {\n", n, n, cs*cs)
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
			fmt.Fprintf(&b, "\t\treturn %s(c, dp, ds, pp, ps%s)\n", kernel{n: n, cols: c, rows: r}.name(), extra)
		})
	})
	b.WriteString("}\n\n")
	return b.String()
}

// generateFused returns the generated kernel file keyed by its path under dir.
func generateFused(dir string) map[string][]byte {
	g := &fgen{consts: map[int]bool{}}
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
			funcs.WriteString(g.rowStrip(n, f))
		}
		for _, f := range []int{8, 16, 32} {
			funcs.WriteString(g.colStrip(n, f))
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
	out.WriteString("// itxGenConsts holds the constants of the generated kernels: the rotation\n")
	out.WriteString("// multipliers 16*C (pC / nC for the cospi constant +C / -C), the negated\n")
	out.WriteString("// shift counts, the Round2(., 12) bias 2^15 of the multiply-accumulate and\n")
	out.WriteString("// the TBL selectors sel[0] / sel[1] that zero-extend bytes 0..7 / 8..15 to\n")
	out.WriteString("// 16-bit lanes (an index of 255 reads zero).\n")
	out.WriteString("type itxGenConsts struct {\n")
	for _, c := range cs {
		fmt.Fprintf(&out, "\t%s [4]int32\n", constField(c))
	}
	out.WriteString("\tshr1, shr3 [8]int16\n")
	out.WriteString("\tbias       [4]int32\n")
	out.WriteString("\tsel        [2][16]uint8\n}\n\n")
	out.WriteString("var itxGenK = itxGenConsts{\n")
	for _, c := range cs {
		fmt.Fprintf(&out, "\t%s: [4]int32{%d, %d, %d, %d},\n", constField(c), 16*c, 16*c, 16*c, 16*c)
	}
	out.WriteString("\tshr1: [8]int16{-1, -1, -1, -1, -1, -1, -1, -1},\n")
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
