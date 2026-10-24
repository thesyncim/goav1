package entropy

import (
	"math/rand/v2"
	"testing"
)

func TestCursorReadCDFSymbolsMatchesReader(t *testing.T) {
	rnd := rand.New(rand.NewPCG(5, 6))
	src := make([]byte, 4096)
	for i := range src {
		src[i] = byte(rnd.Uint32())
	}
	for n := 2; n <= 16; n++ {
		r1 := NewReader(src)
		r2 := NewReader(src)
		var a, b [5]CDF
		for k := range a {
			a[k].InitUniform(n)
			b[k].InitUniform(n)
		}
		for it := 0; it < 3000; it++ {
			k := rnd.IntN(5)
			x, _ := r1.ReadCDF(&a[k])
			c := r2.Cursor()
			y := c.ReadCDFSymbolsUnchecked(&b[k], n)
			c.CommitStateTo(&r2)
			if x != y || a[k] != b[k] {
				t.Fatalf("n=%d it=%d: reader %d cursor %d\n%v\n%v", n, it, x, y, a[k], b[k])
			}
		}
	}
}
