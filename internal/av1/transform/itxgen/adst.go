package main

import (
	"fmt"
	"strings"
)

// neg returns -a with the zero operand folded.
func (g *gen) neg(a val) val {
	if a.zero {
		return val{zero: true}
	}
	n := g.nv()
	g.emit("%s := zero.SubSaturated(%s)", n, a.name)
	return val{name: n}
}

// cospi index helpers: the constant c appears in itx64K at the index whose
// itx64Cos entry equals c.
var cosIndex = map[int]int{}

func init() {
	cos := [64]int{
		4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
		3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
		2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
		1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101,
	}
	for i, c := range cos {
		cosIndex[c] = i
	}
}

// kc returns the itx64K accessor for the signed constant c.
func kc(c int) string {
	if c < 0 {
		return fmt.Sprintf("kn(%d)", cosIndex[-c])
	}
	return fmt.Sprintf("k(%d)", cosIndex[c])
}

// gen implements netOps for the old network emitters.
func (g *gen) rotc(a val, ca int, b val, cb int) val { return g.rot(a, kc(ca), b, kc(cb)) }
func (g *gen) stageMark(s string)                    { g.stage = s }

// adstEmit emits the n-point ADST network (adstnet.go) for inputs below
// live.
func (g *gen) adstEmit(n, live int) string {
	g.sb.Reset()
	g.tmpN = 0
	g.memo = nil
	in := make([]val, n)
	for i := range in {
		if i < live {
			in[i] = val{name: fmt.Sprintf("in[%d]", i)}
		} else {
			in[i] = val{zero: true}
		}
	}
	var out []val
	if n == 16 {
		out = adst16Net[val](g, in)
	} else {
		out = adst8Net[val](g, in)
	}
	for i := 0; i < n; i++ {
		if out[i].zero {
			g.emit("out[%d] = zero", i)
		} else {
			g.emit("out[%d] = %s", i, out[i].name)
		}
	}
	return g.sb.String()
}

func emitADST() string {
	var out strings.Builder
	emit := func(name, doc, size string, body string) {
		usesZero := strings.Contains(body, "zero")
		fmt.Fprintf(&out, "// %s\n", doc)
		fmt.Fprintf(&out, "func %s(in *[%s]archsimd.Int16x8, out *[%s]archsimd.Int16x8) {\n", name, size, size)
		fmt.Fprintf(&out, "\tk := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }\n")
		fmt.Fprintf(&out, "\tkn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }\n")
		fmt.Fprintf(&out, "\ts12 := archsimd.LoadInt32x4Array(&itx16Scale12)\n")
		if usesZero {
			fmt.Fprintf(&out, "\tvar zero archsimd.Int16x8\n")
		}
		out.WriteString(body)
		fmt.Fprintf(&out, "}\n\n")
	}
	for _, n := range []int{16, 8, 4} {
		g := &gen{}
		emit(fmt.Sprintf("itx16ADSTNet%d", n),
			fmt.Sprintf("itx16ADSTNet%d is inverseADST16 on eight lanes for inputs below %d (the rest\n// zero): in[k] is coefficient k, out[r] output r. Generated from the stage\n// structure of inverseADST16 with the zero operands folded.", n, n),
			"16", g.adstEmit(16, n))
	}
	for _, n := range []int{8, 4} {
		g := &gen{}
		emit(fmt.Sprintf("itx8ADSTNet%d", n),
			fmt.Sprintf("itx8ADSTNet%d is inverseADST8 on eight lanes for inputs below %d (the rest\n// zero): in[k] is coefficient k, out[r] output r. Generated from the stage\n// structure of inverseADST8To with the zero operands folded.", n, n),
			"8", g.adstEmit(8, n))
	}
	return out.String()
}
