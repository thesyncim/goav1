// gen64 emits the DCT64 odd-half networks (itx64Odd32/16/8) from the stage
// structure of inverseDCT64 (dct.go / colpass_gosimd8_arm64.go), folding the
// zero inputs of each form symbolically.
package main

import (
	"fmt"
	"os"
	"strings"
)

type val struct {
	zero bool
	name string
}

type gen struct {
	sb    strings.Builder
	memo  map[string]val
	x     [64]val
	tmpN  int
	stage string
}

func (g *gen) nv() string {
	g.tmpN++
	return fmt.Sprintf("%s%d", g.stage, g.tmpN)
}

func (g *gen) emit(format string, args ...any) {
	fmt.Fprintf(&g.sb, "\t"+format+"\n", args...)
}

// rot returns Round2(a*ca + b*cb, 12) with the zero operands folded.
func (g *gen) rot(a val, ca string, b val, cb string) val {
	key := a.name + "|" + ca + "|" + b.name + "|" + cb
	if a.zero {
		key = "|" + b.name + "|" + cb
	} else if b.zero {
		key = a.name + "|" + ca + "|"
	}
	if v, ok := g.memo[key]; ok && !(a.zero && b.zero) {
		return v
	}
	v := g.rotRaw(a, ca, b, cb)
	if g.memo == nil {
		g.memo = map[string]val{}
	}
	g.memo[key] = v
	return v
}

func (g *gen) rotRaw(a val, ca string, b val, cb string) val {
	switch {
	case a.zero && b.zero:
		return val{zero: true}
	case b.zero:
		n := g.nv()
		g.emit("%s := itx16Rot1(%s, %s, s12)", n, a.name, ca)
		return val{name: n}
	case a.zero:
		n := g.nv()
		g.emit("%s := itx16Rot1(%s, %s, s12)", n, b.name, cb)
		return val{name: n}
	}
	n := g.nv()
	g.emit("%s := itx16Rot(%s, %s, %s, %s, s12)", n, a.name, b.name, ca, cb)
	return val{name: n}
}

func (g *gen) add(a, b val) val {
	switch {
	case a.zero && b.zero:
		return val{zero: true}
	case b.zero:
		return a
	case a.zero:
		return b
	}
	n := g.nv()
	g.emit("%s := %s.AddSaturated(%s)", n, a.name, b.name)
	return val{name: n}
}

// sub returns a - b.
func (g *gen) sub(a, b val) val {
	switch {
	case a.zero && b.zero:
		return val{zero: true}
	case b.zero:
		return a
	case a.zero:
		n := g.nv()
		g.emit("%s := zero.SubSaturated(%s)", n, b.name)
		return val{name: n}
	}
	n := g.nv()
	g.emit("%s := %s.SubSaturated(%s)", n, a.name, b.name)
	return val{name: n}
}

func kp(i int) string { return fmt.Sprintf("k(%d)", i) }
func kn(i int) string { return fmt.Sprintf("kn(%d)", i) }

func (g *gen) network(n int) string {
	g.sb.Reset()
	g.tmpN = 0
	g.memo = nil
	var in [64]val
	for i := range in {
		if i < n {
			in[i] = val{name: fmt.Sprintf("in[%d]", i)}
		} else {
			in[i] = val{zero: true}
		}
	}
	var x [64]val
	// Stage 2: pairs (32+j, 63-j) from inputs (a, 64-a) in bit-reversed order.
	a := [16]int{1, 33, 17, 49, 9, 41, 25, 57, 5, 37, 21, 53, 13, 45, 29, 61}
	g.stage = "a"
	for j := 0; j < 16; j++ {
		lo, hi := 32+j, 63-j
		ia, ib := a[j], 64-a[j]
		x[lo] = g.rot(in[ia], kp(ib), in[ib], kn(ia))
		x[hi] = g.rot(in[ia], kp(ia), in[ib], kp(ib))
	}
	// Stage 3.
	g.stage = "b"
	var y [64]val
	for b := 32; b < 64; b += 4 {
		y[b] = g.add(x[b], x[b+1])
		y[b+1] = g.sub(x[b], x[b+1])
		y[b+2] = g.sub(x[b+3], x[b+2])
		y[b+3] = g.add(x[b+2], x[b+3])
	}
	x = y
	// Stage 4.
	g.stage = "c"
	y = x
	for m, pq := range [4][2]int{{4, 60}, {36, 28}, {20, 44}, {52, 12}} {
		p, q := pq[0], pq[1]
		lo, hi := 33+4*m, 62-4*m
		y[lo] = g.rot(x[lo], kn(p), x[hi], kp(q))
		y[hi] = g.rot(x[lo], kp(q), x[hi], kp(p))
		y[lo+1] = g.rot(x[lo+1], kn(q), x[hi-1], kn(p))
		y[hi-1] = g.rot(x[lo+1], kn(p), x[hi-1], kp(q))
	}
	x = y
	// Stage 5.
	g.stage = "d"
	for b := 32; b < 64; b += 8 {
		y[b] = g.add(x[b], x[b+3])
		y[b+1] = g.add(x[b+1], x[b+2])
		y[b+2] = g.sub(x[b+1], x[b+2])
		y[b+3] = g.sub(x[b], x[b+3])
		y[b+4] = g.sub(x[b+7], x[b+4])
		y[b+5] = g.sub(x[b+6], x[b+5])
		y[b+6] = g.add(x[b+5], x[b+6])
		y[b+7] = g.add(x[b+4], x[b+7])
	}
	x = y
	// Stage 6.
	g.stage = "e"
	y = x
	for _, pql := range [2][4]int{{8, 56, 34, 61}, {40, 24, 42, 53}} {
		p, q, lo, hi := pql[0], pql[1], pql[2], pql[3]
		for i := 0; i < 2; i++ {
			y[lo+i] = g.rot(x[lo+i], kn(p), x[hi-i], kp(q))
			y[hi-i] = g.rot(x[lo+i], kp(q), x[hi-i], kp(p))
		}
		for i := 2; i < 4; i++ {
			y[lo+i] = g.rot(x[lo+i], kn(q), x[hi-i], kn(p))
			y[hi-i] = g.rot(x[lo+i], kn(p), x[hi-i], kp(q))
		}
	}
	x = y
	// Stage 7.
	g.stage = "f"
	for b := 32; b < 64; b += 16 {
		for i := 0; i < 4; i++ {
			y[b+i] = g.add(x[b+i], x[b+7-i])
			y[b+7-i] = g.sub(x[b+i], x[b+7-i])
			y[b+8+i] = g.sub(x[b+15-i], x[b+8+i])
			y[b+15-i] = g.add(x[b+8+i], x[b+15-i])
		}
	}
	x = y
	// Stage 8.
	g.stage = "g"
	y = x
	for i := 0; i < 4; i++ {
		y[36+i] = g.rot(x[36+i], kn(16), x[59-i], kp(48))
		y[59-i] = g.rot(x[36+i], kp(48), x[59-i], kp(16))
		y[40+i] = g.rot(x[40+i], kn(48), x[55-i], kn(16))
		y[55-i] = g.rot(x[40+i], kn(16), x[55-i], kp(48))
	}
	x = y
	// Stage 9.
	g.stage = "h"
	for i := 0; i < 8; i++ {
		y[32+i] = g.add(x[32+i], x[47-i])
		y[47-i] = g.sub(x[32+i], x[47-i])
		y[48+i] = g.sub(x[63-i], x[48+i])
		y[63-i] = g.add(x[48+i], x[63-i])
	}
	x = y
	// Stage 10.
	g.stage = "i"
	y = x
	for i := 0; i < 8; i++ {
		y[40+i] = g.rot(x[40+i], kn(32), x[55-i], kp(32))
		y[55-i] = g.rot(x[40+i], kp(32), x[55-i], kp(32))
	}
	x = y
	for j := 0; j < 32; j++ {
		if x[32+j].zero {
			g.emit("o[%d] = zero", j)
		} else {
			g.emit("o[%d] = %s", j, x[32+j].name)
		}
	}
	return g.sb.String()
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "dct8" {
		os.Stdout.WriteString(emitDCT8())
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "adst" {
		os.Stdout.WriteString(emitADST())
		return
	}
	var out strings.Builder
	for _, n := range []int{32, 16, 8} {
		g := &gen{}
		body := g.network(n)
		usesZero := strings.Contains(body, "zero")
		fmt.Fprintf(&out, "// itx64Odd%d is the odd half of inverseDCT64 on eight lanes for odd inputs\n", n)
		fmt.Fprintf(&out, "// below %d (the rest zero): in[k] is coefficient k and o[j] is stage-10\n", n)
		fmt.Fprintf(&out, "// value 32+j. Generated from the stage structure of inverseDCT64 with the\n")
		fmt.Fprintf(&out, "// zero operands folded.\n")
		fmt.Fprintf(&out, "func itx64Odd%d(in *[32]archsimd.Int16x8, o *[32]archsimd.Int16x8) {\n", n)
		fmt.Fprintf(&out, "\tk := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[i]) }\n")
		fmt.Fprintf(&out, "\tkn := func(i int) archsimd.Int16x8 { return archsimd.LoadInt16x8Array(&itx64K[64+i]) }\n")
		fmt.Fprintf(&out, "\ts12 := archsimd.LoadInt32x4Array(&itx16Scale12)\n")
		if usesZero {
			fmt.Fprintf(&out, "\tvar zero archsimd.Int16x8\n")
		}
		out.WriteString(body)
		fmt.Fprintf(&out, "}\n\n")
	}
	os.Stdout.WriteString(out.String())
}
