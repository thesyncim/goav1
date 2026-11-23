package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestInterTXBResultEndTablesMatchScan(t *testing.T) {
	var st lossyEncodeState
	if err := st.initScans(); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	for _, e := range sharedScans.ends {
		n := len(e.endP1)
		var sc []int16
		for _, cand := range [][]int16{sharedScans.scan4, sharedScans.scan8, sharedScans.scan16, sharedScans.scan32,
			sharedScans.scan16x8, sharedScans.scan8x16, sharedScans.scan8x4, sharedScans.scan4x8,
			sharedScans.scan32x16, sharedScans.scan16x32, sharedScans.scan4x16, sharedScans.scan16x4,
			sharedScans.scan8x32, sharedScans.scan32x8, sharedScans.scan16x64, sharedScans.scan64x16} {
			if &cand[0] == e.scan {
				sc = cand
			}
		}
		for trial := range 300 {
			q := make([]int16, n)
			for k := 0; k < 1+trial%7; k++ {
				q[rng.Intn(n)] = int16(rng.Intn(9) - 4)
			}
			want := 0
			for c := n - 1; c >= 0; c-- {
				if q[sc[c]] != 0 {
					want = c + 1
					break
				}
			}
			got := interTXBResult(q, sc, n)
			if want == 0 {
				if got != (tile.TXBDecodeResult{AllZero: true}) {
					t.Fatalf("n=%d: got %+v want all-zero", n, got)
				}
			} else if got != (tile.TXBDecodeResult{EOB: uint16(want)}) {
				t.Fatalf("n=%d trial %d: got %+v want eob %d", n, trial, got, want)
			}
		}
	}
}
