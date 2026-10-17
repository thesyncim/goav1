package quantize

import (
	"fmt"
	"math/rand"
	"testing"
)

func BenchmarkQuantizeBRowGroups16x16(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	coeff := make([]int32, 256)
	for i := range coeff {
		coeff[i] = rng.Int31n(201) - 100
	}
	qcoeff := make([]int16, 256)
	q := Quantizer{DC: 120, AC: 140}
	for _, live := range []uint8{0xf, 0x1, 0x3} {
		b.Run(fmt.Sprintf("live%04b", live), func(b *testing.B) {
			for range b.N {
				QuantizeBRowGroupsTrusted(qcoeff, coeff, 16, live, q, 0)
			}
		})
	}
	b.Run("block", func(b *testing.B) {
		for range b.N {
			_ = QuantizeBlockScaledB(qcoeff, 16, coeff, 16, 16, 16, q, 0)
		}
	})
}

func TestQuantizeBRowGroupsMatchesBlock(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	impls := map[string]func([]int16, []int32, int, uint8, Quantizer, uint8) bool{
		"scalar":   quantizeBRowGroupsScalar,
		"dispatch": quantizeBRowGroupsImpl,
		"exported": QuantizeBRowGroupsTrusted,
	}
	for _, n := range []int{8, 16, 32} {
		txScale := uint8(0)
		if n == 32 {
			txScale = 1
		}
		coeff := make([]int32, n*n)
		want := make([]int16, n*n)
		got := make([]int16, n*n)
		for trial := range 3000 {
			q := Quantizer{DC: int32(rng.Intn(1500) + 4), AC: int32(rng.Intn(1500) + 4)}
			if trial%50 == 0 {
				q = Quantizer{DC: int32(rng.Intn(29000) + 4), AC: int32(rng.Intn(29000) + 4)}
			}
			dcZ, acZ := ZeroBinB(q, txScale)
			live := uint8(rng.Intn(1 << (n / 4)))
			amp := []int32{0, 1, 3 * acZ / 2, 4 * acZ, 70000}[trial%5] + 1
			for col := range n {
				for r := range n {
					i := col*n + r
					if live&(1<<(r/4)) != 0 {
						coeff[i] = rng.Int31n(2*amp+1) - amp
						continue
					}
					// Dead rows: provably below their zero bin.
					z := acZ
					if i == 0 {
						z = min(dcZ, acZ)
					}
					coeff[i] = 0
					if z > 1 {
						coeff[i] = rng.Int31n(2*z-1) - (z - 1)
					}
				}
			}
			if err := QuantizeBlockScaledB(want, n, coeff, n, n, n, q, txScale); err != nil {
				t.Fatal(err)
			}
			wantNZ := false
			for _, v := range want {
				wantNZ = wantNZ || v != 0
			}
			for name, impl := range impls {
				for i := range got {
					got[i] = 0x5a5a
				}
				nz := impl(got, coeff, n, live, q, txScale)
				if nz != wantNZ {
					t.Fatalf("%s n=%d trial %d: nonzero=%v want %v", name, n, trial, nz, wantNZ)
				}
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("%s n=%d trial %d: level[%d]=%d want %d (c=%d live=%b q=%+v)", name, n, trial, i, got[i], want[i], coeff[i], live, q)
					}
				}
			}
		}
	}
}
