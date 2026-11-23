// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
)

// weight is one half_btf twiddle: cospi[idx], negated when neg.
type weight struct {
	idx int
	neg bool
}

// node is one vector value of a network: an input line, a sum, a difference
// or a half butterfly round_shift(a*w0 + b*w1, cos_bit).
type node struct {
	op     byte // 'x' input, '+', '-', 'b' butterfly
	in     int  // input line for 'x'
	a, b   *node
	w0, w1 weight
	name   string
	live   bool
}

// network is the straight-line DAG of one scalar 1-D DCT, read from its Go
// source by netOf.
type network struct {
	n      int
	inputs []*node
	nodes  []*node // creation order, inputs excluded
	out    []*node
}

// netOf symbolically runs the scalar forward DCT fn (fwdDCT16, fwdDCT32 or
// fwdDCT64 from fdct.go / fdct64.go) over n input lines and keeps only the
// nodes that feed its first outs outputs.
func netOf(file, fn string, n, outs int) (*network, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, file, nil, 0)
	if err != nil {
		return nil, err
	}
	var body *ast.BlockStmt
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == fn {
			body = fd.Body
		}
	}
	if body == nil {
		return nil, fmt.Errorf("%s: no func %s", file, fn)
	}
	r := &runner{fset: fset, arrays: map[string][]*node{}, ptr: map[string]string{}, ints: map[string]int{}}
	r.net = &network{n: n}
	in := make([]*node, n)
	for i := range in {
		in[i] = &node{op: 'x', in: i}
	}
	r.net.inputs = in
	r.arrays["input"] = in
	r.arrays["output"] = make([]*node, n)
	r.ptr["input"], r.ptr["output"] = "input", "output"
	if err := r.block(body); err != nil {
		return nil, err
	}
	out := r.arrays["output"][:outs]
	for i, v := range out {
		if v == nil {
			return nil, fmt.Errorf("%s: output %d unset", fn, i)
		}
		mark(v)
	}
	r.net.out = out
	return r.net, nil
}

func mark(v *node) {
	if v.live {
		return
	}
	v.live = true
	if v.a != nil {
		mark(v.a)
	}
	if v.b != nil {
		mark(v.b)
	}
}

// runner interprets the restricted statement forms of the scalar DCTs:
// stage arrays and the bf0/bf1 pointers swapped between them, element
// assignments of sums, differences and fwdHalfBtf calls, and constant
// range loops.
type runner struct {
	fset   *token.FileSet
	net    *network
	arrays map[string][]*node // storage by array name
	ptr    map[string]string  // pointer variable -> array name
	ints   map[string]int     // loop variables
	memo   map[string]*node
}

func (r *runner) errorf(p ast.Node, format string, args ...any) error {
	return fmt.Errorf("%s: %s", r.fset.Position(p.Pos()), fmt.Sprintf(format, args...))
}

func (r *runner) block(b *ast.BlockStmt) error {
	for _, s := range b.List {
		if err := r.stmt(s); err != nil {
			return err
		}
	}
	return nil
}

func (r *runner) stmt(s ast.Stmt) error {
	switch s := s.(type) {
	case *ast.DeclStmt:
		gd := s.Decl.(*ast.GenDecl)
		for _, sp := range gd.Specs {
			vs := sp.(*ast.ValueSpec)
			if at, ok := vs.Type.(*ast.ArrayType); ok {
				size, err := r.intExpr(at.Len)
				if err != nil {
					return err
				}
				for _, nm := range vs.Names {
					r.arrays[nm.Name] = make([]*node, size)
				}
			}
		}
		return nil
	case *ast.AssignStmt:
		return r.assign(s)
	case *ast.RangeStmt:
		cnt, err := r.intExpr(s.X)
		if err != nil {
			return err
		}
		k := s.Key.(*ast.Ident).Name
		for i := range cnt {
			r.ints[k] = i
			if err := r.block(s.Body); err != nil {
				return err
			}
		}
		delete(r.ints, k)
		return nil
	}
	return r.errorf(s, "unsupported statement %T", s)
}

func (r *runner) assign(s *ast.AssignStmt) error {
	if len(s.Lhs) != len(s.Rhs) {
		return r.errorf(s, "unbalanced assignment")
	}
	type store struct {
		arr []*node
		i   int
		v   *node
		ptr string
		to  string
	}
	var stores []store
	for k, lhs := range s.Lhs {
		rhs := s.Rhs[k]
		switch l := lhs.(type) {
		case *ast.Ident:
			if l.Name == "_" {
				continue
			}
			to, err := r.pointee(rhs)
			if err != nil {
				return err
			}
			stores = append(stores, store{ptr: l.Name, to: to})
		case *ast.IndexExpr:
			arr, err := r.array(l.X)
			if err != nil {
				return err
			}
			i, err := r.intExpr(l.Index)
			if err != nil {
				return err
			}
			v, err := r.value(rhs)
			if err != nil {
				return err
			}
			stores = append(stores, store{arr: arr, i: i, v: v})
		default:
			return r.errorf(s, "unsupported assignment target %T", lhs)
		}
	}
	for _, st := range stores {
		if st.arr != nil {
			st.arr[st.i] = st.v
		} else {
			r.ptr[st.ptr] = st.to
		}
	}
	return nil
}

// pointee resolves the right side of a pointer assignment: an array
// pointer variable, &array, or a nil conversion.
func (r *runner) pointee(e ast.Expr) (string, error) {
	switch e := e.(type) {
	case *ast.Ident:
		if to, ok := r.ptr[e.Name]; ok {
			return to, nil
		}
		if _, ok := r.arrays[e.Name]; ok {
			return e.Name, nil
		}
	case *ast.UnaryExpr:
		if id, ok := e.X.(*ast.Ident); ok && e.Op == token.AND {
			return id.Name, nil
		}
	case *ast.CallExpr:
		return "", nil
	}
	return "", r.errorf(e, "unsupported pointer value")
}

func (r *runner) array(e ast.Expr) ([]*node, error) {
	id, ok := e.(*ast.Ident)
	if !ok {
		return nil, r.errorf(e, "unsupported array expression")
	}
	name := id.Name
	if to, ok := r.ptr[name]; ok {
		name = to
	}
	arr, ok := r.arrays[name]
	if !ok {
		return nil, r.errorf(e, "unknown array %s", id.Name)
	}
	return arr, nil
}

func (r *runner) intExpr(e ast.Expr) (int, error) {
	switch e := e.(type) {
	case *ast.BasicLit:
		return strconv.Atoi(e.Value)
	case *ast.Ident:
		if v, ok := r.ints[e.Name]; ok {
			return v, nil
		}
	case *ast.ParenExpr:
		return r.intExpr(e.X)
	case *ast.BinaryExpr:
		a, err := r.intExpr(e.X)
		if err != nil {
			return 0, err
		}
		b, err := r.intExpr(e.Y)
		if err != nil {
			return 0, err
		}
		switch e.Op {
		case token.ADD:
			return a + b, nil
		case token.SUB:
			return a - b, nil
		case token.MUL:
			return a * b, nil
		}
	}
	return 0, r.errorf(e, "unsupported integer expression")
}

func (r *runner) value(e ast.Expr) (*node, error) {
	switch e := e.(type) {
	case *ast.IndexExpr:
		arr, err := r.array(e.X)
		if err != nil {
			return nil, err
		}
		i, err := r.intExpr(e.Index)
		if err != nil {
			return nil, err
		}
		if arr[i] == nil {
			return nil, r.errorf(e, "read of unset element %d", i)
		}
		return arr[i], nil
	case *ast.BinaryExpr:
		// -a + b is the scalar spelling of b - a.
		if u, ok := e.X.(*ast.UnaryExpr); ok && u.Op == token.SUB && e.Op == token.ADD {
			a, err := r.value(u.X)
			if err != nil {
				return nil, err
			}
			b, err := r.value(e.Y)
			if err != nil {
				return nil, err
			}
			return r.make(&node{op: '-', a: b, b: a}), nil
		}
		a, err := r.value(e.X)
		if err != nil {
			return nil, err
		}
		b, err := r.value(e.Y)
		if err != nil {
			return nil, err
		}
		switch e.Op {
		case token.ADD:
			return r.make(&node{op: '+', a: a, b: b}), nil
		case token.SUB:
			return r.make(&node{op: '-', a: a, b: b}), nil
		}
	case *ast.CallExpr:
		if id, ok := e.Fun.(*ast.Ident); ok && id.Name == "fwdHalfBtf" && len(e.Args) == 5 {
			w0, err := r.weight(e.Args[0])
			if err != nil {
				return nil, err
			}
			a, err := r.value(e.Args[1])
			if err != nil {
				return nil, err
			}
			w1, err := r.weight(e.Args[2])
			if err != nil {
				return nil, err
			}
			b, err := r.value(e.Args[3])
			if err != nil {
				return nil, err
			}
			return r.make(&node{op: 'b', a: a, w0: w0, b: b, w1: w1}), nil
		}
	}
	return nil, r.errorf(e, "unsupported value expression")
}

func (r *runner) weight(e ast.Expr) (weight, error) {
	neg := false
	if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.SUB {
		neg, e = true, u.X
	}
	ix, ok := e.(*ast.IndexExpr)
	if !ok {
		return weight{}, r.errorf(e, "unsupported twiddle")
	}
	if id, ok := ix.X.(*ast.Ident); !ok || id.Name != "cospi" {
		return weight{}, r.errorf(e, "unsupported twiddle")
	}
	i, err := r.intExpr(ix.Index)
	return weight{idx: i, neg: neg}, err
}

// make records v, sharing an identical earlier node.
func (r *runner) make(v *node) *node {
	key := fmt.Sprintf("%c %p %p %v %v", v.op, v.a, v.b, v.w0, v.w1)
	if old, ok := r.memo[key]; ok {
		return old
	}
	if r.memo == nil {
		r.memo = map[string]*node{}
	}
	r.memo[key] = v
	r.net.nodes = append(r.net.nodes, v)
	return v
}

// name assigns the value names of the live nodes: x<line> for inputs and
// v<k> in creation order.
func (nw *network) name() {
	for _, x := range nw.inputs {
		x.name = fmt.Sprintf("x%d", x.in)
	}
	k := 0
	for _, v := range nw.nodes {
		if v.live {
			v.name = fmt.Sprintf("v%d", k)
			k++
		}
	}
}

// cospiTable reads the [64]int32 literal of the package variable name.
func cospiTable(file, name string) ([64]int32, error) {
	var t [64]int32
	f, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
	if err != nil {
		return t, err
	}
	for _, d := range f.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, sp := range gd.Specs {
			vs, ok := sp.(*ast.ValueSpec)
			if !ok || len(vs.Names) != 1 || vs.Names[0].Name != name {
				continue
			}
			lit := vs.Values[0].(*ast.CompositeLit)
			if len(lit.Elts) != 64 {
				return t, fmt.Errorf("%s: %d entries", name, len(lit.Elts))
			}
			for i, el := range lit.Elts {
				v, err := strconv.Atoi(el.(*ast.BasicLit).Value)
				if err != nil {
					return t, err
				}
				t[i] = int32(v)
			}
			return t, nil
		}
	}
	return t, fmt.Errorf("%s: no var %s", file, name)
}
