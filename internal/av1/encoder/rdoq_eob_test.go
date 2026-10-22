package encoder

import (
	"math/rand"
	"testing"
)

func TestRDOQEOBMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, n := range []int{16, 64, 256, 1024} {
		iscan := rng.Perm(n)
		is := make([]int16, n)
		for i, v := range iscan {
			is[i] = int16(v)
		}
		q := make([]int16, n)
		for trial := range 500 {
			clear(q)
			for k := rng.Intn(6); k > 0; k-- {
				q[rng.Intn(n)] = int16(rng.Intn(9) - 4)
			}
			want := 0
			for ci, v := range q {
				if v != 0 {
					want = max(want, int(is[ci])+1)
				}
			}
			if got := rdoqEOBFromIScan(q, is); got != want {
				t.Fatalf("n=%d trial %d: eob %d want %d", n, trial, got, want)
			}
			if got := rdoqEOBPureGo(q, is); got != want {
				t.Fatalf("n=%d trial %d: pure eob %d want %d", n, trial, got, want)
			}
		}
	}
}
