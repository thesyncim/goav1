// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package quantize

import (
	"math/rand"
	"reflect"
	"testing"
)

// TestQuantizeBlockScaledBEOB checks that QuantizeBlockScaledBEOB writes the
// levels QuantizeBlockScaledB writes and reports their end of block.
func TestQuantizeBlockScaledBEOB(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for _, dims := range [][2]int{{4, 4}, {8, 8}, {16, 16}, {32, 32}, {16, 8}, {8, 32}} {
		w, h := dims[0], dims[1]
		n := w * h
		for trial := range 300 {
			coeff := make([]int32, n)
			amp := []int32{1, 40, 400, 4000, 40000}[trial%5]
			for i := range coeff {
				if rng.Intn(4) == 0 {
					coeff[i] = rng.Int31n(2*amp+1) - amp
				}
			}
			if trial%7 == 0 {
				clear(coeff)
				coeff[0] = rng.Int31n(200) - 100
			}
			q := Quantizer{DC: 4 + rng.Int31n(2000), AC: 4 + rng.Int31n(2000)}
			ts := uint8(trial % 3)
			want := make([]int16, n)
			if err := QuantizeBlockScaledB(want, h, coeff, h, w, h, q, ts); err != nil {
				t.Fatal(err)
			}
			got := make([]int16, n)
			perm := rng.Perm(n - 1)
			iscan := make([]int16, n)
			for pos := 1; pos < n; pos++ {
				iscan[pos] = int16(perm[pos-1] + 1)
			}
			eob, err := QuantizeBlockScaledBEOB(got, h, coeff, h, w, h, q, ts, iscan)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%dx%d trial %d: levels differ", w, h, trial)
			}
			if wantEOB := testEOB(want, iscan); eob != wantEOB {
				t.Fatalf("%dx%d trial %d: eob %d want %d", w, h, trial, eob, wantEOB)
			}
		}
	}
}
