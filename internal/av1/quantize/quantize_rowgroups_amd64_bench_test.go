//go:build goexperiment.simd && amd64 && !purego

package quantize

import (
	"fmt"
	"math/rand"
	"simd/archsimd"
	"testing"
)

// BenchmarkAMD64QuantizeBRowGroups compares the scalar row-group quantizer
// with the AVX2 kernel (impl=generic / impl=avx2 for the amd64-bench
// workflow). The diff variant also produces the skip statistics.
func BenchmarkAMD64QuantizeBRowGroups(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{8, 16, 32} {
		coeff := make([]int32, n*n)
		for i := range coeff {
			coeff[i] = rng.Int31n(401) - 200
		}
		qcoeff := make([]int16, n*n)
		q := Quantizer{DC: 120, AC: 140}
		txScale := uint8(0)
		if n == 32 {
			txScale = 1
		}
		live := uint8(1<<(n/4) - 1)
		for _, kind := range []string{"quant", "diff"} {
			b.Run(fmt.Sprintf("%s/n=%d/impl=generic", kind, n), func(b *testing.B) {
				for b.Loop() {
					quantizeBRowGroupsScalar(qcoeff, coeff, n, live, q, txScale)
				}
			})
			b.Run(fmt.Sprintf("%s/n=%d/impl=avx2", kind, n), func(b *testing.B) {
				if !archsimd.X86.AVX2() {
					b.Skip("no AVX2")
				}
				for b.Loop() {
					if kind == "diff" {
						quantizeBRowGroupsDiffAVX2(qcoeff, coeff, n, live, q, txScale)
					} else {
						quantizeBRowGroupsAVX2(qcoeff, coeff, n, live, q, txScale)
					}
				}
			})
		}
	}
}
