// Command itxzero derives a sparse-input variant of a generated inverse
// transform kernel: inputs x<N>.. are known zero, so their loads (or
// transposes) are dropped and zero is propagated through the SSA-style
// statements with exact identities (0*k = 0, 0>>s = 0, avg(0,0) = 0,
// clamp(0) = 0, a+0 = a, a-0 = a). All output stores are kept.
//
// usage: itxzero -src file.go -func name -n N -name newName
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strconv"
	"strings"
)

var zeros = map[string]bool{"zero": true}

// alias maps a variable whose value equals an earlier one (CSE) to it.
var alias = map[string]string{}
var seen = map[string]string{}

func exprString(fset *token.FileSet, e ast.Expr) string {
	var b bytes.Buffer
	format.Node(&b, fset, e)
	return b.String()
}

func isZeroIdent(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	return ok && zeros[id.Name]
}

func zeroIdent() ast.Expr { return ast.NewIdent("zero") }

// simplify returns the rewritten expression and whether it is exactly zero.
func simplify(e ast.Expr) (ast.Expr, bool) {
	switch x := e.(type) {
	case *ast.Ident:
		if zeros[x.Name] {
			return zeroIdent(), true
		}
		if a, ok := alias[x.Name]; ok {
			return ast.NewIdent(a), false
		}
		return x, false
	case *ast.CallExpr:
		sel, ok := x.Fun.(*ast.SelectorExpr)
		if !ok {
			for i, a := range x.Args {
				na, z := simplify(a)
				if z {
					na = zeroIdent()
				}
				x.Args[i] = na
			}
			return x, false
		}
		// Package-level call (archsimd.X): simplify args only.
		if id, ok := sel.X.(*ast.Ident); ok && (id.Name == "archsimd") {
			return x, false
		}
		recv, rz := simplify(sel.X)
		args := make([]ast.Expr, len(x.Args))
		argz := make([]bool, len(x.Args))
		for i, a := range x.Args {
			args[i], argz[i] = simplify(a)
		}
		m := sel.Sel.Name
		switch m {
		case "Mul", "ShiftAllRight", "ShiftAllLeft", "Max", "Min":
			if rz {
				return zeroIdent(), true
			}
		case "Average":
			if rz && argz[0] {
				return zeroIdent(), true
			}
		case "Add":
			if rz && argz[0] {
				return zeroIdent(), true
			}
			if rz {
				return args[0], false
			}
			if argz[0] {
				return recv, false
			}
		case "Sub":
			if rz && argz[0] {
				return zeroIdent(), true
			}
			if argz[0] {
				return recv, false
			}
		case "MulAdd":
			// recv*k + y
			if rz && argz[1] {
				return zeroIdent(), true
			}
			if rz {
				return args[1], false
			}
			if argz[1] {
				return &ast.CallExpr{Fun: &ast.SelectorExpr{X: recv, Sel: ast.NewIdent("Mul")}, Args: []ast.Expr{args[0]}}, false
			}
		}
		if rz {
			recv = zeroIdent()
		}
		for i := range args {
			if argz[i] {
				args[i] = zeroIdent()
			}
		}
		x.Fun = &ast.SelectorExpr{X: recv, Sel: sel.Sel}
		x.Args = args
		return x, false
	case *ast.ParenExpr:
		return simplify(x.X)
	}
	return e, false
}

func inputIndex(name string) (int, bool) {
	if len(name) < 2 || name[0] != 'x' {
		return 0, false
	}
	n, err := strconv.Atoi(name[1:])
	return n, err == nil
}

func main() {
	src := flag.String("src", "", "")
	fn := flag.String("func", "", "")
	n := flag.Int("n", 8, "")
	name := flag.String("name", "", "")
	flag.Parse()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, *src, nil, parser.ParseComments)
	if err != nil {
		panic(err)
	}
	var fd *ast.FuncDecl
	for _, d := range f.Decls {
		if g, ok := d.(*ast.FuncDecl); ok && g.Name.Name == *fn {
			fd = g
		}
	}
	if fd == nil {
		panic("func not found")
	}
	var out []ast.Stmt
	for _, st := range fd.Body.List {
		as, ok := st.(*ast.AssignStmt)
		if !ok {
			if es, ok := st.(*ast.ExprStmt); ok {
				ne, z := simplify(es.X)
				if z {
					ne = zeroIdent()
				}
				es.X = ne
			}
			out = append(out, st)
			continue
		}
		// Input definitions: drop those whose every lhs index >= n.
		allInputs, allZero := true, true
		for _, l := range as.Lhs {
			id := l.(*ast.Ident)
			k, isIn := inputIndex(id.Name)
			if !isIn {
				allInputs = false
				break
			}
			if k < *n {
				allZero = false
			}
		}
		if allInputs && len(as.Lhs) > 0 {
			if allZero {
				for _, l := range as.Lhs {
					zeros[l.(*ast.Ident).Name] = true
				}
				continue
			}
			for _, l := range as.Lhs {
				if k, _ := inputIndex(l.(*ast.Ident).Name); k >= *n {
					panic("mixed input group straddles n")
				}
			}
			out = append(out, st)
			continue
		}
		if len(as.Lhs) == 1 && len(as.Rhs) == 1 {
			id := as.Lhs[0].(*ast.Ident)
			ne, z := simplify(as.Rhs[0])
			if z && as.Tok == token.DEFINE && id.Name != "zero" {
				zeros[id.Name] = true
				continue
			}
			as.Rhs[0] = ne
			if as.Tok == token.DEFINE {
				key := exprString(fset, ne)
				if prev, ok := seen[key]; ok && strings.HasPrefix(id.Name, "v") {
					alias[id.Name] = prev
					continue
				}
				seen[key] = id.Name
			}
			out = append(out, st)
			continue
		}
		for i, r := range as.Rhs {
			ne, z := simplify(r)
			if z {
				ne = zeroIdent()
			}
			as.Rhs[i] = ne
		}
		out = append(out, st)
	}
	// Drop unused constant loads (iterate to fixpoint).
	for {
		used := map[string]int{}
		for _, st := range out {
			ast.Inspect(st, func(nd ast.Node) bool {
				if id, ok := nd.(*ast.Ident); ok {
					used[id.Name]++
				}
				return true
			})
		}
		var keep []ast.Stmt
		removed := false
		for _, st := range out {
			if as, ok := st.(*ast.AssignStmt); ok && as.Tok == token.DEFINE && len(as.Lhs) == 1 {
				nm := as.Lhs[0].(*ast.Ident).Name
				if nm != "_" && used[nm] == 1 {
					removed = true
					continue
				}
			}
			keep = append(keep, st)
		}
		out = keep
		if !removed {
			break
		}
	}
	fd.Body.List = out
	fd.Name = ast.NewIdent(*name)
	fd.Doc = nil
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, fd); err != nil {
		panic(err)
	}
	s := buf.String()
	// zero may now be unused if every Average(zero) disappeared; keep it referenced.
	if !strings.Contains(strings.SplitN(s, "var zero archsimd.Int32x4", 2)[1], "zero") {
		s = strings.Replace(s, "var zero archsimd.Int32x4", "var zero archsimd.Int32x4\n\t_ = zero", 1)
	}
	fmt.Printf("// %s is %s with inputs %d.. known zero, derived mechanically by\n// zero propagation (exact identities only); all output stores are kept.\n%s\n\n", *name, *fn, *n, s)
}
