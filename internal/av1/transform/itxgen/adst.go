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

// adst16 emits inverseADST16 (adst.go inverseADST16Core) for inputs below n.
func (g *gen) adst16(n int) string {
	g.sb.Reset()
	g.tmpN = 0
	g.memo = nil
	in := make([]val, 16)
	for i := range in {
		if i < n {
			in[i] = val{name: fmt.Sprintf("in[%d]", i)}
		} else {
			in[i] = val{zero: true}
		}
	}
	rot := func(a val, ca int, b val, cb int) val { return g.rot(a, kc(ca), b, kc(cb)) }
	g.stage = "a"
	var t [16]val
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
	g.stage = "b"
	var ta [16]val
	for i := 0; i < 8; i++ {
		ta[i] = g.add(t[i], t[i+8])
		ta[i+8] = g.sub(t[i], t[i+8])
	}
	g.stage = "c"
	t[8] = rot(ta[8], 4017, ta[9], 799)
	t[9] = rot(ta[8], 799, ta[9], -4017)
	t[10] = rot(ta[10], 2276, ta[11], 3406)
	t[11] = rot(ta[10], 3406, ta[11], -2276)
	t[12] = rot(ta[13], 4017, ta[12], -799)
	t[13] = rot(ta[13], 799, ta[12], 4017)
	t[14] = rot(ta[15], 2276, ta[14], -3406)
	t[15] = rot(ta[15], 3406, ta[14], 2276)
	g.stage = "d"
	for i := 0; i < 4; i++ {
		t[i] = g.add(ta[i], ta[i+4])
		t[i+4] = g.sub(ta[i], ta[i+4])
	}
	for i := 8; i < 12; i++ {
		ta[i] = g.add(t[i], t[i+4])
		ta[i+4] = g.sub(t[i], t[i+4])
	}
	g.stage = "e"
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	t[12] = rot(ta[12], 3784, ta[13], 1567)
	t[13] = rot(ta[12], 1567, ta[13], -3784)
	t[14] = rot(ta[15], 3784, ta[14], -1567)
	t[15] = rot(ta[15], 1567, ta[14], 3784)
	g.stage = "f"
	var out [16]val
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
	g.stage = "g"
	out[7] = g.neg(rot(ta[2], 2896, ta[3], 2896))
	out[8] = rot(ta[2], 2896, ta[3], -2896)
	out[4] = rot(t[6], 2896, t[7], 2896)
	out[11] = g.neg(rot(t[6], 2896, t[7], -2896))
	out[6] = rot(t[10], 2896, t[11], 2896)
	out[9] = g.neg(rot(t[10], 2896, t[11], -2896))
	out[5] = g.neg(rot(ta[14], 2896, ta[15], 2896))
	out[10] = rot(ta[14], 2896, ta[15], -2896)
	for i := 0; i < 16; i++ {
		if out[i].zero {
			g.emit("out[%d] = zero", i)
		} else {
			g.emit("out[%d] = %s", i, out[i].name)
		}
	}
	return g.sb.String()
}

// adst8 emits inverseADST8 (adst.go inverseADST8To) for inputs below n.
func (g *gen) adst8(n int) string {
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
	var ta [8]val
	ta[0] = rot(in[7], 4076, in[0], 401)
	ta[1] = rot(in[7], 401, in[0], -4076)
	ta[2] = rot(in[5], 3612, in[2], 1931)
	ta[3] = rot(in[5], 1931, in[2], -3612)
	ta[4] = rot(in[3], 2598, in[4], 3166)
	ta[5] = rot(in[3], 3166, in[4], -2598)
	ta[6] = rot(in[1], 1189, in[6], 3920)
	ta[7] = rot(in[1], 3920, in[6], -1189)
	g.stage = "b"
	var t [8]val
	for i := 0; i < 4; i++ {
		t[i] = g.add(ta[i], ta[i+4])
		t[i+4] = g.sub(ta[i], ta[i+4])
	}
	g.stage = "c"
	ta[4] = rot(t[4], 3784, t[5], 1567)
	ta[5] = rot(t[4], 1567, t[5], -3784)
	ta[6] = rot(t[7], 3784, t[6], -1567)
	ta[7] = rot(t[7], 1567, t[6], 3784)
	g.stage = "d"
	var out [8]val
	out[0] = g.add(t[0], t[2])
	out[7] = g.neg(g.add(t[1], t[3]))
	t[2] = g.sub(t[0], t[2])
	t[3] = g.sub(t[1], t[3])
	out[1] = g.neg(g.add(ta[4], ta[6]))
	out[6] = g.add(ta[5], ta[7])
	t[6] = g.sub(ta[4], ta[6])
	t[7] = g.sub(ta[5], ta[7])
	g.stage = "e"
	out[3] = g.neg(rot(t[2], 2896, t[3], 2896))
	out[4] = rot(t[2], 2896, t[3], -2896)
	out[2] = rot(t[6], 2896, t[7], 2896)
	out[5] = g.neg(rot(t[6], 2896, t[7], -2896))
	for i := 0; i < 8; i++ {
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
			fmt.Sprintf("itx16ADSTNet%d is inverseADST16 on eight lanes for inputs below %d (the rest\n// zero): in[k] is coefficient k, out[r] output r. Generated from the stage\n// structure of inverseADST16Core with the zero operands folded.", n, n),
			"16", g.adst16(n))
	}
	for _, n := range []int{8, 4} {
		g := &gen{}
		emit(fmt.Sprintf("itx8ADSTNet%d", n),
			fmt.Sprintf("itx8ADSTNet%d is inverseADST8 on eight lanes for inputs below %d (the rest\n// zero): in[k] is coefficient k, out[r] output r. Generated from the stage\n// structure of inverseADST8To with the zero operands folded.", n, n),
			"8", g.adst8(n))
	}
	return out.String()
}
