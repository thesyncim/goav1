// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Command mcgridgen writes the unrolled six-tap Grid kernels of package
// motion (arm64 Go SIMD):
//
//   - mc8v6grid_gosimd_arm64.go: the 8-bit vertical-only put and prep passes
//     over sixteen-column strips, one eight-column strip and a four-column
//     block;
//   - mc8h6grid_gosimd_arm64.go: the 8-bit horizontal-only put and prep
//     passes, same shapes;
//   - mc16v6grid_gosimd_arm64.go: the 16bpc vertical-only put and prep passes
//     over eight-sample strips and a four-sample block.
//
// Every kernel walks its rows in chunks of eight, then two, then one; each
// chunk is a fixed-shape sub-view whose Fits test is the only check its rows
// pay. The vertical kernels keep five source rows resident and rotate them
// once per chunk. The generator spells out each chunk's rows, so a change to
// the row pattern is made once here rather than in every copy.
//
// Run it from the motion directory (go generate ./internal/av1/motion).
// TestGeneratedUpToDate fails when the committed files differ from its output.
package main

import (
	"fmt"
	"go/format"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	files, err := generate(".")
	if err != nil {
		fmt.Fprintln(os.Stderr, "mcgridgen:", err)
		os.Exit(1)
	}
	for name, src := range files {
		if err := os.WriteFile(name, src, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "mcgridgen:", err)
			os.Exit(1)
		}
	}
}

const fileHeader = `// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"encoding/binary"
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)
`

// generate returns the generated files, by name, for the motion package
// directory dir.
func generate(dir string) (map[string][]byte, error) {
	out := map[string][]byte{}
	for name, body := range map[string]string{
		"mc8v6grid_gosimd_arm64.go":  mc8V6File(),
		"mc8h6grid_gosimd_arm64.go":  mc8H6File(),
		"mc16v6grid_gosimd_arm64.go": mc16V6File(),
	} {
		src, err := format.Source([]byte(fileHeader + body))
		if err != nil {
			return nil, fmt.Errorf("%s: %v", name, err)
		}
		out[filepath.Join(dir, name)] = src
	}
	return out, nil
}

// chunks are the row counts of the chunk loops, in order.
var chunks = []int{8, 2, 1}

// kernel describes one generated function.
type kernel struct {
	doc      string   // the doc comment, without the "// " prefixes
	sig      string   // name and parameters
	prologue []string // the constant set-up, ending with the h (and w) line
	strip    *strip   // the column-strip loop; nil for a single-strip kernel
	vert     bool     // vertical: five resident rows, source rows from r+5
	sw, ow   int      // source and destination chunk window widths

	// headDecl is the declaration before the five head-row loads (vertical).
	headDecl string
	// load returns the lines that load source row k of view v into
	// resident register set i (vertical).
	load func(v string, k, i int) []string
	// store returns the line that filters and stores output row k of the
	// chunk from the six register sets base..base+5 (vertical) or from source
	// row k of sw (horizontal).
	store func(k, base int) []string
	// rot names register set i, for the rotation after each chunk.
	rot func(i int) string
}

// strip is the column-strip loop of a multi-strip kernel.
type strip struct {
	step     int    // columns per strip
	col      string // the strip's first source/destination column expression
	srcCol   string
	srcRows  string // "h+5" or "h"
	dstW     int    // the strip's destination width
	preamble string // a declaration at the top of each strip, or ""
}

type writer struct{ strings.Builder }

func (w *writer) line(format string, args ...any) {
	fmt.Fprintf(w, format, args...)
	w.WriteByte('\n')
}

func (w *writer) comment(text string) {
	for _, l := range strings.Split(text, "\n") {
		if l == "" {
			w.line("//")
		} else {
			w.line("// %s", l)
		}
	}
}

func (w *writer) kernel(k kernel) {
	w.line("")
	w.comment(k.doc)
	w.line("func %s {", k.sig)
	for i := 0; i < 6; i++ {
		w.line("%s", strings.ReplaceAll(k.prologue[0], "#", fmt.Sprint(i)))
	}
	for _, l := range k.prologue[1:] {
		w.line("%s", l)
	}
	s, o := "s", "o"
	if k.strip != nil {
		st := k.strip
		w.line("for c := 0; c+%d <= w; c += %d {", st.step, st.step)
		w.line("if !sg.Fits(0, %s, %d, %s) || !og.Fits(0, %s, %d, h) {", st.srcCol, k.sw, st.srcRows, st.col, st.dstW)
		w.line("return")
		w.line("}")
		w.line("s, o := sg.Sub(0, %s, %d, %s), og.Sub(0, %s, %d, h)", st.srcCol, k.sw, st.srcRows, st.col, st.dstW)
		if st.preamble != "" {
			w.line("%s", st.preamble)
		}
	}
	if k.vert {
		w.line("if !%s.Fits(0, 0, 16, 5) {", s)
		w.line("return")
		w.line("}")
		w.line("hd := %s.Sub(0, 0, 16, 5)", s)
		if k.headDecl != "" {
			w.line("%s", k.headDecl)
		}
		for i := 0; i < 5; i++ {
			for _, l := range k.load("hd", i, i) {
				w.line("%s", l)
			}
		}
	}
	w.line("r := 0")
	for _, n := range chunks {
		if n == 1 {
			w.line("for ; r < h; r += 1 {")
		} else {
			w.line("for ; r+%d <= h; r += %d {", n, n)
		}
		row := "r"
		if k.vert {
			row = "r+5"
		}
		w.line("if !%s.Fits(%s, 0, %d, %d) || !%s.Fits(r, 0, %d, %d) {", s, row, k.sw, n, o, k.ow, n)
		w.line("return")
		w.line("}")
		w.line("sw, ow := %s.Sub(%s, 0, %d, %d), %s.Sub(r, 0, %d, %d)", s, row, k.sw, n, o, k.ow, n)
		for r := 0; r < n; r++ {
			if k.vert {
				for _, l := range k.load("sw", r, 5+r) {
					w.line("%s", l)
				}
			}
			for _, l := range k.store(r, r) {
				w.line("%s", l)
			}
		}
		if k.vert {
			var lhs, rhs []string
			for i := 0; i < 5; i++ {
				lhs = append(lhs, k.rot(i))
				rhs = append(rhs, k.rot(n+i))
			}
			w.line("%s = %s", strings.Join(lhs, ", "), strings.Join(rhs, ", "))
		}
		w.line("}")
	}
	if k.strip != nil {
		w.line("}")
	}
	w.line("}")
}

// regs lists the names of the six register sets base..base+5 of prefix p.
func regs(p string, base int) string {
	var n []string
	for i := base; i < base+6; i++ {
		n = append(n, fmt.Sprintf("%s%d", p, i))
	}
	return strings.Join(n, ", ")
}

const taps = "c0, c1, c2, c3, c4, c5, rnd"

// pair names register set i of a two-half (lo/hi) kernel, for the rotation.
func pair(i int) string { return fmt.Sprintf("l%d, h%d", i, i) }

func single(p string) func(int) string {
	return func(i int) string { return fmt.Sprintf("%s%d", p, i) }
}

// 8-bit prologues: halved taps on int16 lanes.
func mc8Prologue(put bool, hw string) []string {
	rnd, sh := "2 + 6144<<2", "-2"
	if put {
		rnd, sh = "32", "-6"
	}
	return []string{
		"c# := archsimd.LoadInt16x8Array(&ph.h[#])",
		fmt.Sprintf("rnd := archsimd.BroadcastInt16x8(%s)", rnd),
		fmt.Sprintf("sh := archsimd.BroadcastInt16x8(%s)", sh),
		hw,
	}
}

func mc8Load(two bool) func(v string, k, i int) []string {
	return func(v string, k, i int) []string {
		ld := fmt.Sprintf("archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](%s, %d, 0))", v, k)
		if !two {
			return []string{fmt.Sprintf("r%d := %s.ExtendLo8ToUint16().BitsToInt16()", i, ld)}
		}
		op := "="
		if v == "hd" && k == 0 {
			op = ":="
		}
		return []string{
			fmt.Sprintf("x %s %s", op, ld),
			fmt.Sprintf("l%d, h%d := x.ExtendLo8ToUint16().BitsToInt16(), x.HiToLo().ExtendLo8ToUint16().BitsToInt16()", i, i),
		}
	}
}

// The 8-bit store expressions, from a filtered Int16x8 sum expression.
func put8x16(lo, hi string, k int) string {
	return fmt.Sprintf("%s.Shift(sh).SaturateToUint8().ReshapeToUint64s().InterleaveLo(%s.Shift(sh).SaturateToUint8().ReshapeToUint64s()).ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, %d, 0))", lo, hi, k)
}

func put8x8(sum string, k int) string {
	return fmt.Sprintf("binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, %d, 0)[:], %s.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0))", k, sum)
}

func put8x4(sum string, k int) string {
	return fmt.Sprintf("binary.LittleEndian.PutUint32(unsafeutil.Row[[4]byte](ow, %d, 0)[:], uint32(%s.Shift(sh).SaturateToUint8().ReshapeToUint64s().GetElem(0)))", k, sum)
}

func prep8x8(sum string, k, c int) string {
	return fmt.Sprintf("%s.ToBits().Shift(sh).StoreArray(unsafeutil.Row[[8]uint16](ow, %d, %d))", sum, k, c)
}

func prep8x4(sum string, k int) string {
	return fmt.Sprintf("mc8Store4U16(unsafeutil.Row[[4]uint16](ow, %d, 0), %s.ToBits().Shift(sh).ReshapeToUint64s().GetElem(0))", k, sum)
}

func v6(p string, base int) string { return fmt.Sprintf("mc8V6(%s, %s)", regs(p, base), taps) }

const (
	putV6Desc  = "the put vertical-only pass, clip((s + 32) >> 6)"
	prepV6Desc = "the prep vertical-only pass, ((s + 2) >> 2) + 6144 (the offset sums lie in [0, 2^16) and shift as unsigned)"
	putH6Desc  = "the put horizontal-only pass, clip((s + 34) >> 6)"
	prepH6Desc = "the prep horizontal-only pass, ((s + 2) >> 2) + 6144 (the offset sums lie in [0, 2^16) and shift as unsigned)"
)

func mc8V6File() string {
	var w writer
	w.line("")
	w.comment(`The 8-bit six-tap vertical kernels (dav1d put_6tap/prep_6tap v) over Grid
views. Each column strip keeps five source rows in registers and filters
eight output rows per chunk (two, then one, for the last rows), so the
resident rows rotate once per eight output rows. Every chunk is a
fixed-shape sub-view whose Fits test is the only check its rows pay; the
tests cannot fail (the entries sized the views), and with no panic behind
them the kernels stay leaf functions.`)
	for _, put := range []bool{true, false} {
		name, desc, dst, dstName := "Put", putV6Desc, "uint8", "destination"
		if !put {
			name, desc, dst, dstName = "Prep", prepV6Desc, "uint16", "CONV_BUF"
		}
		w16 := kernel{
			doc: fmt.Sprintf(`mc8%sV6W16 is %s,
over sixteen-column strips: sg holds the h+5 source rows (h = og.Height(),
og.Width() a multiple of 16) and og the h %s rows. Each source
row is one load for both eight-sample halves.`, name, desc, dstName),
			sig:      fmt.Sprintf("mc8%sV6W16(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[%s])", name, dst),
			prologue: mc8Prologue(put, "h, w := og.Height(), og.Width()"),
			strip:    &strip{step: 16, col: "c", srcCol: "c", srcRows: "h+5", dstW: 16},
			vert:     true, sw: 16, ow: 16,
			load: mc8Load(true),
			rot:  pair,
		}
		if put {
			w16.store = func(k, b int) []string { return []string{put8x16(v6("l", b), v6("h", b), k)} }
		} else {
			w16.store = func(k, b int) []string {
				return []string{prep8x8(v6("l", b), k, 0), prep8x8(v6("h", b), k, 8)}
			}
		}
		w.kernel(w16)

		s8 := kernel{
			doc: fmt.Sprintf(`mc8%sV6S8 is mc8%sV6W16 over one eight-column strip: s holds its
h+5 source rows of sixteen readable bytes, o its h %s rows.`, name, name, dstName),
			sig:      fmt.Sprintf("mc8%sV6S8(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[%s])", name, dst),
			prologue: mc8Prologue(put, "h := o.Height()"),
			vert:     true, sw: 16, ow: 8,
			load: mc8Load(false),
			rot:  single("r"),
		}
		if put {
			s8.store = func(k, b int) []string { return []string{put8x8(v6("r", b), k)} }
		} else {
			s8.store = func(k, b int) []string { return []string{prep8x8(v6("r", b), k, 0)} }
		}
		w.kernel(s8)

		s4 := s8
		s4.doc = fmt.Sprintf(`mc8%sV6S4 is mc8%sV6S8 for a four-column block (dav1d's w4 v path):
s holds its h+5 source rows of sixteen readable bytes, o its h %s rows
of four samples.`, name, name, dstName)
		s4.sig = fmt.Sprintf("mc8%sV6S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[%s])", name, dst)
		s4.ow = 4
		if put {
			s4.store = func(k, b int) []string { return []string{put8x4(v6("r", b), k)} }
		} else {
			s4.store = func(k, b int) []string { return []string{prep8x4(v6("r", b), k)} }
		}
		w.kernel(s4)
	}
	return w.String()
}

func mc8H6File() string {
	var w writer
	w.line("")
	w.comment(`The 8-bit six-tap horizontal kernels (dav1d put_6tap/prep_6tap h) over
Grid views, eight rows per chunk (two, then one, for the last rows) of each
column strip. Sixteen-column strips read one 24-byte window per row (the
two eight-output tap windows are its bytes 0..15 and 8..23); every chunk is
a fixed-shape sub-view whose Fits test is the only check its rows pay, a
test the entries make unreachable, so the kernels stay leaf functions.`)
	w.line("")
	w.comment(`mc8H6S is the six-tap halved horizontal sum (plus rnd) of the eight output
columns whose tap window starts at p (taps read lanes 0..12 of the widened
bytes). It calls no helper, so a call on the line of the vector store that
uses it leaves no inline-mark NOP.`)
	w.line(`func mc8H6S(p *[16]uint8, c0, c1, c2, c3, c4, c5, rnd archsimd.Int16x8) archsimd.Int16x8 {
	a := archsimd.LoadUint8x16Array(p)
	lo, hi := a.ExtendLo8ToUint16().ReshapeToUint8s(), a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
	s := lo.ReshapeToUint16s().BitsToInt16().MulAdd(c0, rnd)`)
	for t := 1; t < 5; t++ {
		w.line("s = hi.ConcatShiftBytesRight(lo, %d).ReshapeToUint16s().BitsToInt16().MulAdd(c%d, s)", 2*t, t)
	}
	w.line("return hi.ConcatShiftBytesRight(lo, 10).ReshapeToUint16s().BitsToInt16().MulAdd(c5, s)")
	w.line("}")

	h6 := func(win string) string { return fmt.Sprintf("mc8H6S(%s, %s)", win, taps) }
	const lo, hi = "(*[16]uint8)(win[0:16])", "(*[16]uint8)(win[8:24])"
	winRow := func(k int) string { return fmt.Sprintf("win = unsafeutil.Row[[24]byte](sw, %d, 0)", k) }
	row16 := func(k int) string { return fmt.Sprintf("unsafeutil.Row[[16]byte](sw, %d, 0)", k) }
	for _, put := range []bool{true, false} {
		name, desc, dst, dstName := "Put", putH6Desc, "uint8", "destination"
		pro := func(hw string) []string {
			p := mc8Prologue(put, hw)
			p[1] = "rnd := archsimd.BroadcastInt16x8(34)"
			return p
		}
		if !put {
			name, desc, dst, dstName = "Prep", prepH6Desc, "uint16", "CONV_BUF"
			pro = func(hw string) []string { return mc8Prologue(false, hw) }
		}
		w16 := kernel{
			doc: fmt.Sprintf(`mc8%sH6W16 is %s,
over sixteen-column strips: sg holds the h = og.Height() source rows, each
readable for 24 bytes from every strip (og.Width() a multiple of 16), and
og the h %s rows.`, name, desc, dstName),
			sig:      fmt.Sprintf("mc8%sH6W16(sg unsafeutil.Grid[uint8], ph *mc8Phase, og unsafeutil.Grid[%s])", name, dst),
			prologue: pro("h, w := og.Height(), og.Width()"),
			strip:    &strip{step: 16, col: "c", srcCol: "c", srcRows: "h", dstW: 16, preamble: "var win *[24]byte"},
			sw:       24, ow: 16,
		}
		if put {
			w16.store = func(k, _ int) []string {
				return []string{winRow(k), put8x16(h6(lo), h6(hi), k)}
			}
		} else {
			w16.store = func(k, _ int) []string {
				return []string{winRow(k), prep8x8(h6(lo), k, 0), prep8x8(h6(hi), k, 8)}
			}
		}
		w.kernel(w16)

		s8 := kernel{
			doc: fmt.Sprintf(`mc8%sH6S8 is mc8%sH6W16 over one eight-column strip: s holds its
h source rows of sixteen readable bytes, o its h %s rows.`, name, name, dstName),
			sig:      fmt.Sprintf("mc8%sH6S8(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[%s])", name, dst),
			prologue: pro("h := o.Height()"),
			sw:       16, ow: 8,
		}
		if put {
			s8.store = func(k, _ int) []string { return []string{put8x8(h6(row16(k)), k)} }
		} else {
			s8.store = func(k, _ int) []string { return []string{prep8x8(h6(row16(k)), k, 0)} }
		}
		w.kernel(s8)

		s4 := s8
		s4.doc = fmt.Sprintf(`mc8%sH6S4 is mc8%sH6S8 for a four-column block: s holds its h
source rows of sixteen readable bytes, o its h %s rows of four samples.`, name, name, dstName)
		s4.sig = fmt.Sprintf("mc8%sH6S4(s unsafeutil.Grid[uint8], ph *mc8Phase, o unsafeutil.Grid[%s])", name, dst)
		s4.ow = 4
		if put {
			s4.store = func(k, _ int) []string { return []string{put8x4(h6(row16(k)), k)} }
		} else {
			s4.store = func(k, _ int) []string { return []string{prep8x4(h6(row16(k)), k)} }
		}
		w.kernel(s4)
	}
	return w.String()
}

func mc16V6File() string {
	var w writer
	w.line("")
	w.comment(`The 16bpc six-tap vertical kernels (dav1d mc16 put_6tap/prep_6tap v) over
Grid views. Each source row is widened to int32 once, when it is loaded,
and the halved taps run as int32 multiply-accumulates (mc8Phase.w): a
halved-tap sum of 10- or 12-bit samples needs 32-bit lanes, and widening
the rows instead of the products drops the per-tap widening multiplies,
their adds and the high-half moves (the sums are the same integers). Each
eight-sample column strip keeps five rows resident and filters eight
output rows per chunk (two, then one, for the last rows); the chunks'
Fits tests are the only checks the rows pay, and the entries make them
unreachable, so the kernels stay leaf functions.`)
	w.line("")
	w.comment("mc16V6I is the six-tap int32 vertical sum of rows r0..r5, plus rnd.")
	w.line("func mc16V6I(r0, r1, r2, r3, r4, r5, %s archsimd.Int32x4) archsimd.Int32x4 {", taps)
	w.line("a := r0.MulAdd(c0, rnd)")
	for t := 1; t < 5; t++ {
		w.line("a = r%d.MulAdd(c%d, a)", t, t)
	}
	w.line("return r5.MulAdd(c5, a)")
	w.line("}")

	load := func(two bool) func(v string, k, i int) []string {
		return func(v string, k, i int) []string {
			l := []string{fmt.Sprintf("x = archsimd.LoadUint8x16Array(unsafeutil.Row[[16]byte](%s, %d, 0)).ReshapeToUint16s().BitsToInt16()", v, k)}
			if two {
				return append(l, fmt.Sprintf("l%d, h%d := x.ExtendLo4ToInt32(), x.HiToLo().ExtendLo4ToInt32()", i, i))
			}
			return append(l, fmt.Sprintf("l%d := x.ExtendLo4ToInt32()", i))
		}
	}
	v6i := func(p string, b int) string {
		return fmt.Sprintf("mc16V6I(%s, %s).Shift(shv)", regs(p, b), taps)
	}
	coef := "c# := archsimd.LoadInt32x4Array(&ph.w[#])"
	putPro := []string{coef,
		"rnd := archsimd.BroadcastInt32x4(32)",
		"shv := archsimd.BroadcastInt32x4(-6)",
		"zero := archsimd.BroadcastInt16x8(0)",
		"maxV := archsimd.BroadcastInt16x8(maxPix)",
	}
	prepPro := []string{coef,
		"rnd := archsimd.BroadcastInt32x4(int32(1)<<(sh-1) + int32(mc16PrepRoundOffset)<<sh)",
		"shv := archsimd.BroadcastInt32x4(int32(-sh))",
	}
	with := func(p []string, extra ...string) []string { return append(append([]string{}, p...), extra...) }

	w.kernel(kernel{
		doc: `mc16PutV6G is the 16bpc put vertical-only pass, clip((hs + 32) >> 6),
for w = og.Width()/2 a multiple of 8: sg holds the h+5 source rows
(h = og.Height()), each readable for 16 bytes from every eight-sample
strip, and og the h destination rows.`,
		sig:      "mc16PutV6G(sg unsafeutil.Grid[uint8], ph *mc8Phase, maxPix int16, og unsafeutil.Grid[uint8])",
		prologue: with(putPro, "h, w := og.Height(), og.Width()/2"),
		strip:    &strip{step: 8, col: "2*c", srcCol: "2*c", srcRows: "h+5", dstW: 16},
		vert:     true, sw: 16, ow: 16,
		headDecl: "var x archsimd.Int16x8",
		load:     load(true),
		store: func(k, b int) []string {
			return []string{fmt.Sprintf("mc16Clip(%s, %s, zero, maxV).ToBits().ReshapeToUint8s().StoreArray(unsafeutil.Row[[16]byte](ow, %d, 0))", v6i("l", b), v6i("h", b), k)}
		},
		rot: pair,
	})
	w.kernel(kernel{
		doc:      "mc16PutV6GW4 is mc16PutV6G for a four-sample-wide block.",
		sig:      "mc16PutV6GW4(s unsafeutil.Grid[uint8], ph *mc8Phase, maxPix int16, o unsafeutil.Grid[uint8])",
		prologue: with(putPro, "var zero32 archsimd.Int32x4", "h := o.Height()"),
		vert:     true, sw: 16, ow: 8,
		headDecl: "var x archsimd.Int16x8",
		load:     load(false),
		store: func(k, b int) []string {
			return []string{fmt.Sprintf("binary.LittleEndian.PutUint64(unsafeutil.Row[[8]byte](ow, %d, 0)[:], mc16Clip(%s, zero32, zero, maxV).ToBits().ReshapeToUint64s().GetElem(0))", k, v6i("l", b))}
		},
		rot: single("l"),
	})
	w.kernel(kernel{
		doc: `mc16PrepV6G is the 16bpc prep vertical-only pass, ((hs + 2^(sh-1)) >> sh) + roundOffset,
for w = og.Width() a multiple of 8: sg holds the h+5 source rows
(h = og.Height()), each readable for 16 bytes from every eight-sample
strip, and og the h CONV_BUF rows.`,
		sig:      "mc16PrepV6G(sg unsafeutil.Grid[uint8], ph *mc8Phase, sh int, og unsafeutil.Grid[uint16])",
		prologue: with(prepPro, "h, w := og.Height(), og.Width()"),
		strip:    &strip{step: 8, col: "c", srcCol: "2*c", srcRows: "h+5", dstW: 8},
		vert:     true, sw: 16, ow: 8,
		headDecl: "var x archsimd.Int16x8",
		load:     load(true),
		store: func(k, b int) []string {
			return []string{fmt.Sprintf("mc16PackU16(%s, %s).StoreArray(unsafeutil.Row[[8]uint16](ow, %d, 0))", v6i("l", b), v6i("h", b), k)}
		},
		rot: pair,
	})
	w.kernel(kernel{
		doc:      "mc16PrepV6GW4 is mc16PrepV6G for a four-sample-wide block.",
		sig:      "mc16PrepV6GW4(s unsafeutil.Grid[uint8], ph *mc8Phase, sh int, o unsafeutil.Grid[uint16])",
		prologue: with(prepPro, "var zero32 archsimd.Int32x4", "h := o.Height()"),
		vert:     true, sw: 16, ow: 4,
		headDecl: "var x archsimd.Int16x8",
		load:     load(false),
		store: func(k, b int) []string {
			return []string{fmt.Sprintf("mc8Store4U16(unsafeutil.Row[[4]uint16](ow, %d, 0), mc16PackU16(%s, zero32).ReshapeToUint64s().GetElem(0))", k, v6i("l", b))}
		},
		rot: single("l"),
	})
	return w.String()
}
