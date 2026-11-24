// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestInverseAddDCT32x64GuardEdge drives the 32-coded-column row strips (the
// split forms, whose row guard is checked after the even half has run) with
// one coefficient row whose L1 is exactly the guard bound and one past it:
// past the bound the block declines with dst untouched; at the bound it
// declines untouched or matches the staged path byte for byte.
func TestInverseAddDCT32x64GuardEdge(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	for _, n := range []int{32, 64} {
		stride := n + 8
		coeff := make([]int32, 32*32)
		pred := make([]byte, n*stride)
		got := make([]byte, n*stride)
		want := make([]byte, n*stride)
		accepted := 0
		for trial := 0; trial < 400; trial++ {
			clear(coeff)
			l1 := itx16ColumnL1Bound + trial%2
			row := rng.Intn(8)
			activeRows := 24 + rng.Intn(9)
			if row >= activeRows {
				row = 0
			}
			// Split l1 over the 32 columns of one row, random signs.
			left := l1
			for c := 0; c < 32; c++ {
				v := left / (32 - c)
				if c < 31 {
					v += rng.Intn(v/2+1) - v/4
				}
				v = min(v, left)
				left -= v
				if rng.Intn(2) == 0 {
					v = -v
				}
				coeff[c*32+row] = int32(v)
			}
			for c := 0; c < 32; c++ {
				for r := 0; r < activeRows; r++ {
					if r != row && rng.Intn(4) == 0 {
						coeff[c*32+r] = int32(rng.Intn(65) - 32)
					}
				}
			}
			for i := range pred {
				pred[i] = byte(rng.Intn(256))
			}
			copy(got, pred)
			var ok bool
			if n == 32 {
				ok = InverseAddDCT32x32Full(got, stride, got, stride, coeff, 32, activeRows, 32)
			} else {
				ok = InverseAddDCT64x64(got, stride, got, stride, coeff, 32, activeRows, 32)
			}
			if !ok {
				for i := range got {
					if got[i] != pred[i] {
						t.Fatalf("n %d trial %d: declined block modified dst at %d", n, trial, i)
					}
				}
				continue
			}
			if l1 > itx16ColumnL1Bound {
				t.Fatalf("n %d trial %d: row L1 %d past the bound accepted", n, trial, l1)
			}
			accepted++
			copy(want, pred)
			if n == 32 {
				itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, 32)
			} else {
				itxFused64Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, 32)
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("n %d trial %d: mismatch at row %d col %d", n, trial, i/stride, i%stride)
				}
			}
		}
		if accepted == 0 {
			t.Fatalf("n %d: no block at the bound accepted", n)
		}
		t.Logf("n %d: %d of 200 at the bound accepted", n, accepted)
	}
}
