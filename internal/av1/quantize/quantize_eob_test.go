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

// TestTryQuantizeBBlockEOBTrusted checks the prebuilt-constant square kernel
// against the independent scalar rule (originalRowGroupLevel) over every
// size, tx scale and live-column bound, on sparse blocks around both zero
// bins with int32 extremes mixed in. The columns past the bound hold
// garbage that must be neither read nor coded.
func TestTryQuantizeBBlockEOBTrusted(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eb))
	quants := []Quantizer{{4, 4}, {120, 140}, {147, 148}, {148, 147}, {1828, 4}, {4, 1828}, {29247, 29247}, {65535, 65535}}
	handledCount := 0
	for _, n := range []int{8, 16, 32} {
		for ts := uint8(0); ts <= 2; ts++ {
			for qi, q := range quants {
				p := testBParams(q, ts)
				dcZ, acZ := ZeroBinB(q, ts)
				for trial := range 24 {
					cols := []int{n, n / 2, max(n/4, 1), 1}[trial%4]
					coeff := make([]int32, n*n)
					for i := range coeff {
						switch {
						case i/n >= cols:
							coeff[i] = int32(rng.Uint32()) // garbage
						case rng.Intn(3) == 0:
							z := []int32{acZ, dcZ}[i%2]
							coeff[i] = rng.Int31n(2*z+5) - z - 2
						}
					}
					edges := []int32{acZ, -acZ, acZ - 1, dcZ, -dcZ, dcZ - 1, -2147483648, 2147483647, 32767, -32768}
					for range trial % 5 {
						coeff[rng.Intn(cols*n)] = edges[rng.Intn(len(edges))]
					}
					if trial%6 == 0 {
						coeff[0] = []int32{dcZ, -dcZ, dcZ - 1, acZ}[trial/6%4]
					}
					want := make([]int16, n*n)
					for i := range want[:cols*n] {
						step := q.AC
						if i == 0 {
							step = q.DC
						}
						want[i] = originalRowGroupLevel(coeff[i], step, q.DC, ts)
					}
					iscan := testISCAN(rng, n)
					got := make([]int16, n*n+8)
					for i := range got {
						got[i] = 0x5a5a
					}
					before := append([]int32(nil), coeff...)
					eob, handled := TryQuantizeBBlockEOBTrusted(got, coeff, n, cols, p, iscan)
					wantHandled := quantizeBBlockEOBTrustedImpl != nil && p.vector
					if handled != wantHandled {
						t.Fatalf("n=%d ts=%d q=%+v: handled=%v want %v", n, ts, q, handled, wantHandled)
					}
					if !handled {
						for _, v := range got {
							if v != 0x5a5a {
								t.Fatal("declined call wrote levels")
							}
						}
						continue
					}
					handledCount++
					if !reflect.DeepEqual(got[:n*n], want) {
						t.Fatalf("n=%d ts=%d q=%d cols=%d trial=%d: levels differ", n, ts, qi, cols, trial)
					}
					if wantEOB := testEOB(want, iscan); eob != wantEOB {
						t.Fatalf("n=%d ts=%d q=%d cols=%d: eob %d want %d", n, ts, qi, cols, eob, wantEOB)
					}
					for _, v := range got[n*n:] {
						if v != 0x5a5a {
							t.Fatal("output tail changed")
						}
					}
					if !reflect.DeepEqual(coeff, before) {
						t.Fatal("input changed")
					}
				}
			}
		}
	}
	if quantizeBBlockEOBTrustedImpl != nil && handledCount == 0 {
		t.Fatal("kernel did not execute")
	}
}
