package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
)

// TestCDEFDist8MatchesScalar: the dispatched cdefDist8 equals the scalar
// svt_aom_compute_cdef_dist_8bit_c port for every block shape and
// subsampling factor the search uses.
func TestCDEFDist8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	src := make([]byte, 80*70)
	flt := make([]byte, 64*64)
	for trial := 0; trial < 300; trial++ {
		for i := range src {
			src[i] = byte(rng.Intn(256))
		}
		for i := range flt {
			flt[i] = byte(rng.Intn(256))
		}
		dec := rng.Intn(2)
		n := 8 >> dec
		var dlist []cdef.BlockPosition
		for by := 0; by < n; by++ {
			for bx := 0; bx < n; bx++ {
				if rng.Intn(3) != 0 {
					dlist = append(dlist, cdef.BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		for _, ss := range []int{1, 2, 4} {
			bl := 3 - dec
			want := computeCDEFDist8(src, 80, flt, 64>>dec, dlist, bl, bl, ss)
			got := cdefDist8(src, 80, flt, 64>>dec, dlist, bl, bl, ss)
			if got != want {
				t.Fatalf("trial %d dec %d ss %d: %d != %d", trial, dec, ss, got, want)
			}
		}
	}
}
