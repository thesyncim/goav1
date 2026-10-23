package main

import (
	"fmt"
	"strings"
)

// dct8 emits inverseDCT8 (the network of itx8Net) for inputs below n.
func (g *gen) dct8(n int) string {
	g.sb.Reset()
	g.tmpN = 0
	g.memo = nil
	in := make([]val, 8)
	for i := range in {
		if i < n {
			in[i] = val{name: fmt.Sprintf("in[%d]", i)}
		} else {
			in[i] = val{zero: true}
		}
	}
	rot := func(a val, ca int, b val, cb int) val { return g.rot(a, kc(ca), b, kc(cb)) }
	g.stage = "a"
	t0 := rot(in[0], 2896, in[4], 2896)
	t1 := rot(in[0], 2896, in[4], -2896)
	t2 := rot(in[2], 1567, in[6], -3784)
	t3 := rot(in[2], 3784, in[6], 1567)
	g.stage = "b"
	d0 := g.add(t0, t3)
	d1 := g.add(t1, t2)
	d2 := g.sub(t1, t2)
	d3 := g.sub(t0, t3)
	g.stage = "c"
	t4a := rot(in[1], 799, in[7], -4017)
	t5a := rot(in[5], 3406, in[3], -2276)
	t6a := rot(in[5], 2276, in[3], 3406)
	t7a := rot(in[1], 4017, in[7], 799)
	g.stage = "d"
	t4 := g.add(t4a, t5a)
	t5b := g.sub(t4a, t5a)
	t7 := g.add(t7a, t6a)
	t6b := g.sub(t7a, t6a)
	g.stage = "e"
	t5 := rot(t6b, 2896, t5b, -2896)
	t6 := rot(t6b, 2896, t5b, 2896)
	g.stage = "f"
	out := [8]val{
		g.add(d0, t7), g.add(d1, t6), g.add(d2, t5), g.add(d3, t4),
		g.sub(d3, t4), g.sub(d2, t5), g.sub(d1, t6), g.sub(d0, t7),
	}
	for i := 0; i < 8; i++ {
		if out[i].zero {
			g.emit("out[%d] = zero", i)
		} else {
			g.emit("out[%d] = %s", i, out[i].name)
		}
	}
	return g.sb.String()
}

func emitDCT8() string {
	var out strings.Builder
	g := &gen{}
	body := g.dct8(4)
	fmt.Fprintf(&out, "// itx8NetLow4 is inverseDCT8 on eight lanes for inputs whose indices 4..7\n// are zero: in[k] is coefficient k, out[r] output r. Generated from the\n// network of itx8Net with the zero operands folded.\n")
	fmt.Fprintf(&out, "func itx8NetLow4(in *[8]archsimd.Int16x8, out *[8]archsimd.Int16x8) {\n")
	fmt.Fprintf(&out, "\tk := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }\n")
	fmt.Fprintf(&out, "\tkn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }\n")
	fmt.Fprintf(&out, "\ts12 := archsimd.LoadInt32x4Array(&itx16Scale12)\n")
	if strings.Contains(body, "zero") {
		fmt.Fprintf(&out, "\tvar zero archsimd.Int16x8\n")
	}
	out.WriteString(body)
	fmt.Fprintf(&out, "}\n")
	return out.String()
}
