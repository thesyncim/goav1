// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

import (
	"fmt"
	"strings"
)

// The hybrid kernels: every 8x8 and 16x16 pairing of the DCT, the ADST,
// the flipped ADST and the identity per direction but DCT_DCT (generated
// above) and IDTX (the staged identity path), from the same networks.
//
// The identity is exact on any int16 input when it is evaluated together
// with the rounding that follows it (identity1DValue, identity.go):
//
//	8-point, row (Round2(., 1)):     Round2(2x, 1) = x
//	8-point, column (Round2(., 4)):  Round2(2x, 4) = Round2(x, 3)
//	16-point, row (Round2(., 2)):    floor((5793x + 5120) / 2^13)
//	16-point, column (Round2(., 4)): floor((5793x + 17408) / 2^15)
//
// (2x + ((1697x + 1024) >> 11) folded with the shift by the nested-floor
// identity floor((floor(a/m) + k)/n) = floor((a + km)/(mn))), so it needs
// no guard: the 16-point forms are one 32-bit multiply-accumulate per half
// and the high halves.

// The 1-D types.
const (
	tDCT  = "DCT"
	tADST = "ADST"
	tFlip = "Flip"
	tIden = "Iden"
)

// typeDoc is the AV1 name of a 1-D type.
func typeDoc(t string) string {
	switch t {
	case tADST:
		return "ADST"
	case tFlip:
		return "FLIPADST"
	case tIden:
		return "IDTX"
	}
	return "DCT"
}

// fgen implements netOps (adstnet.go).
func (g *fgen) rotc(a fv, ca int, b fv, cb int) fv { return g.rot(a, ca, b, cb) }
func (g *fgen) stageMark(string)                   {}

// neg returns -a with the zero operand folded.
func (g *fgen) neg(a fv) fv {
	if a.zero {
		return a
	}
	n := g.tmp("a")
	g.line("%s := zero.SubSaturated(%s)", n, a.name)
	return fv{name: n}
}

// network1D runs the n-point network of type t: the flipped ADST is the
// ADST with its outputs reversed (inverseFlipADST1D) and the identity is
// left pending for the rounding that follows it.
func (g *fgen) network1D(t string, n int, in []fv) []fv {
	switch t {
	case tDCT:
		return g.network(n, in)
	case tIden:
		out := make([]fv, n)
		for i, v := range in {
			out[i] = v
			if !v.zero {
				out[i].id = n
			}
		}
		return out
	}
	var out []fv
	switch n {
	case 4:
		out = g.adst4(in)
	case 8:
		out = adst8Net[fv](g, in)
	default:
		out = adst16Net[fv](g, in)
	}
	if t == tFlip {
		for i := 0; i < n/2; i++ {
			out[i], out[n-1-i] = out[n-1-i], out[i]
		}
	}
	return out
}

// kraw is the load of the broadcast int32 constant v (field f).
func (g *fgen) kraw(f string, v int) string {
	g.raw[f] = v
	return fmt.Sprintf("archsimd.LoadInt32x4Array(&t.%s)", f)
}

// idRound writes Round2(identity_n(x), s) of the pending identity v, then
// the int16 clamp of the scalar path (only the s < 2 row forms of the 4-
// and 16-point identities can leave int16).
func (g *fgen) idRound(v fv, s int) fv {
	x := fv{name: v.name}
	switch {
	case v.id == 4 && s == 0:
		return g.mulShiftSat(x, 5793, 2048, 12)
	case v.id == 4 && s == 1:
		return g.mulHi(x, 46344, 49152)
	case v.id == 4 && s == 4:
		return g.mulHi(x, 5793, 34816)
	case v.id == 8 && s == 0:
		n := g.tmp("a")
		g.line("%s := %s.AddSaturated(%s)", n, x.name, x.name)
		return fv{name: n}
	case v.id == 8 && s == 1:
		return x
	case v.id == 8 && s == 4:
		return g.round2(x, 3)
	case v.id == 16 && s == 1:
		return g.mulShiftSat(x, 5793, 3072, 12)
	case v.id == 16 && s == 2:
		return g.mulHi(x, 46344, 40960)
	case v.id == 16 && s == 4:
		return g.mulHi(x, 11586, 34816)
	}
	panic(fmt.Sprintf("identity %d with Round2 by %d", v.id, s))
}

// mulHi writes (x*m + b) >> 16 on the sign-extended halves of x, for a
// result that fits int16.
func (g *fgen) mulHi(x fv, m, b int) fv {
	lo, hi := g.widen(x)
	k, kb := g.tmp("k"), g.tmp("k")
	g.line("%s := %s", k, g.kraw(fmt.Sprintf("m%d", m), m))
	g.line("%s := %s", kb, g.kraw(fmt.Sprintf("b%d", b), b))
	n := g.tmp("r")
	g.line("%sl := %s.MulAdd(%s, %s)", n, lo, k, kb)
	if !g.half {
		g.line("%sh := %s.MulAdd(%s, %s)", n, hi, k, kb)
	}
	g.narrowHi(n)
	return fv{name: n}
}

// mulShiftSat writes (x*m + b) >> s on the sign-extended halves of x,
// saturated to int16.
func (g *fgen) mulShiftSat(x fv, m, b, s int) fv {
	lo, hi := g.widen(x)
	k, kb := g.tmp("k"), g.tmp("k")
	g.line("%s := %s", k, g.kraw(fmt.Sprintf("m%d", m), m))
	g.line("%s := %s", kb, g.kraw(fmt.Sprintf("b%d", b), b))
	n := g.tmp("r")
	if g.half {
		g.line("%s := %s.MulAdd(%s, %s).ShiftAllRight(%d).SaturateToInt16()", n, lo, k, kb, s)
		return fv{name: n}
	}
	f := g.tmp("f")
	g.line("%s := %s.MulAdd(%s, %s).ShiftAllRight(%d).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64()", f, lo, k, kb, s)
	g.line("%s = %s.SetElem(1, %s.MulAdd(%s, %s).ShiftAllRight(%d).SaturateToInt16().ToBits().ReshapeToUint64s().BitsToFloat64().GetElem(0))", f, f, hi, k, kb, s)
	g.line("%s := %s.ToBits().ReshapeToUint16s().BitsToInt16()", n, f)
	return fv{name: n}
}

// hybType is one generated transform type: its Go constant and the
// vertical and horizontal 1-D types.
type hybType struct {
	goName string
	v, h   string
}

// hybTypes8 are the 8x8 hybrid types; hybTypes16 the 16x16 ones (AV1 has
// no 16x16 V_ADST, H_ADST, V_FLIPADST or H_FLIPADST).
var (
	hybTypes16 = []hybType{
		{"TypeADSTDCT", tADST, tDCT}, {"TypeDCTADST", tDCT, tADST}, {"TypeADSTADST", tADST, tADST},
		{"TypeFlipADSTDCT", tFlip, tDCT}, {"TypeDCTFlipADST", tDCT, tFlip},
		{"TypeFlipADSTFlipADST", tFlip, tFlip}, {"TypeADSTFlipADST", tADST, tFlip},
		{"TypeFlipADSTADST", tFlip, tADST}, {"TypeVDCT", tDCT, tIden}, {"TypeHDCT", tIden, tDCT},
	}
	hybTypes8 = append(append([]hybType{}, hybTypes16...),
		hybType{"TypeVADST", tADST, tIden}, hybType{"TypeHADST", tIden, tADST},
		hybType{"TypeVFlipADST", tFlip, tIden}, hybType{"TypeHFlipADST", tIden, tFlip})
	dctDCT = hybType{"TypeDCTDCT", tDCT, tDCT}
)

// formsUpTo are the network forms of an n-point pass.
func formsUpTo(n int) []int {
	var fs []int
	for _, f := range []int{4, 8, 16} {
		if f <= n {
			fs = append(fs, f)
		}
	}
	return fs
}

// cross is every (columns, rows) pairing of the forms of a w x h block.
func cross(w, h int) [][2]int {
	var ps [][2]int
	for _, c := range formsUpTo(w) {
		for _, r := range formsUpTo(h) {
			ps = append(ps, [2]int{c, r})
		}
	}
	return ps
}

// lowFull is the coded 4x4 corner form and the full form of a w x h block
// of type t: the low corner dominates the hybrid and small blocks (the
// intermediate forms are a few percent of them) and an identity pass gains
// nothing from a narrower form of its own direction.
func lowFull(t hybType, w, h int) [][2]int {
	c, r := 4, 4
	if t.h == tIden {
		c = w
	}
	if t.v == tIden {
		r = h
	}
	if c == w && r == h {
		return [][2]int{{w, h}}
	}
	return [][2]int{{c, r}, {w, h}}
}

// tdEntry is one transform type of a dispatch function with its (columns,
// rows) network forms, the most specific first.
type tdEntry struct {
	t     hybType
	forms [][2]int
}

// typeDispatch writes name, which runs the w x h kernel of type t and the
// first network form covering cols x rows, or reports false for a type
// without one. extra is appended to the kernel arguments.
func typeDispatch(name string, w, h int, es []tdEntry, extra string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "// %s runs the generated %dx%d kernel of type t and the network forms\n", name, w, h)
	fmt.Fprintf(&b, "// covering cols coded columns and rows coded rows; it reports false for a\n")
	fmt.Fprintf(&b, "// type without one and when a guard declines.\n")
	fmt.Fprintf(&b, "func %s(c *[%d]int32, d []byte, ds int, p []byte, ps int, t Type, cols, rows int) bool {\n", name, w*min(h, 32))
	m := h
	if w == h {
		m = 0
	}
	b.WriteString("\tswitch t {\n")
	for _, e := range es {
		fmt.Fprintf(&b, "\tcase %s:\n", e.t.goName)
		for i, f := range e.forms {
			k := kernel{n: w, m: m, cols: f[0], rows: f[1], v: e.t.v, h: e.t.h}
			call := fmt.Sprintf("return %s(c, d, ds, p, ps%s)", k.name(), extra)
			var cond []string
			if f[0] < w {
				cond = append(cond, fmt.Sprintf("cols <= %d", f[0]))
			}
			if f[1] < h {
				cond = append(cond, fmt.Sprintf("rows <= %d", f[1]))
			}
			if i == len(e.forms)-1 || len(cond) == 0 {
				fmt.Fprintf(&b, "\t\t%s\n", call)
				break
			}
			fmt.Fprintf(&b, "\t\tif %s {\n\t\t\t%s\n\t\t}\n", strings.Join(cond, " && "), call)
		}
	}
	b.WriteString("\t}\n\treturn false\n}\n\n")
	return b.String()
}

// generateHybrid writes the hybrid 8x8 and 16x16 kernels, the 4x4 and the
// 4- and 8/16-point rectangular kernels of every type, and their dispatch
// functions. The 16x16 hybrids are row and column strip functions over a
// stack tmp (as 32x32): one function per block form costs three times the
// code for 3-5% fewer cycles per call. Every other block is one function
// per lowFull form (DCT_DCT rectangles too; the 4x4 and rectangular
// hybrids, a few hundred blocks a clip, in the full form only).
func (g *fgen) generateHybrid(funcs *strings.Builder) {
	whole := func(w, h int, es []tdEntry) {
		m := h
		if w == h {
			m = 0
		}
		for _, e := range es {
			for _, f := range e.forms {
				funcs.WriteString(g.wholeKernel(kernel{n: w, m: m, cols: f[0], rows: f[1], v: e.t.v, h: e.t.h}))
			}
		}
	}
	var es []tdEntry
	for _, ht := range hybTypes8 {
		es = append(es, tdEntry{ht, lowFull(ht, 8, 8)})
	}
	funcs.WriteString(typeDispatch("itxGen8x8Hybrid", 8, 8, es, ""))
	whole(8, 8, es)

	es = nil
	for _, ht := range hybTypes16 {
		es = append(es, tdEntry{ht, cross(16, 16)})
	}
	funcs.WriteString(typeDispatch("itxGen16x16Hybrid", 16, 16, es, ", rows"))
	for _, e := range es {
		for _, f := range e.forms {
			funcs.WriteString(stripKernel(kernel{n: 16, cols: f[0], rows: f[1], v: e.t.v, h: e.t.h}))
		}
	}
	for _, t := range []string{tDCT, tADST, tFlip, tIden} {
		for _, f := range formsUpTo(16) {
			funcs.WriteString(g.rowStrip(t, 16, f))
		}
		for _, f := range formsUpTo(16) {
			funcs.WriteString(g.colStrip(t, 16, f))
		}
	}

	for _, sz := range [][2]int{{4, 4}, {4, 8}, {8, 4}, {4, 16}, {16, 4}, {8, 16}, {16, 8}} {
		w, h := sz[0], sz[1]
		types := hybTypes16
		if max(w, h) <= 8 {
			types = hybTypes8
		}
		es := []tdEntry{{dctDCT, lowFull(dctDCT, w, h)}}
		for _, ht := range types {
			es = append(es, tdEntry{ht, [][2]int{{w, h}}})
		}
		funcs.WriteString(typeDispatch(fmt.Sprintf("itxGen%dx%d", w, h), w, h, es, ""))
		whole(w, h, es)
	}
}
