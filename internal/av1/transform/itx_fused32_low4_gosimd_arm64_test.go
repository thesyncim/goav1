//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"simd/archsimd"
	"testing"
)

// TestItx32Low4NetMatchesLow8 pins the zero-folded network to itx32Low8Net
// with inputs 4..7 zero, lane for lane, over random, sparse and saturating
// int16 inputs.
func TestItx32Low4NetMatchesLow8(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	for trial := 0; trial < 20000; trial++ {
		var raw [4][8]int16
		for k := range raw {
			for l := range raw[k] {
				switch rng.Intn(6) {
				case 0:
					raw[k][l] = 0
				case 1:
					raw[k][l] = -32768
				case 2:
					raw[k][l] = 32767
				case 3:
					raw[k][l] = int16(rng.Intn(65536) - 32768)
				case 4:
					raw[k][l] = int16(rng.Intn(2049) - 1024)
				default:
					raw[k][l] = int16(rng.Intn(21) - 10)
				}
			}
		}
		var in4 [4]archsimd.Int16x8
		var in8 [8]archsimd.Int16x8
		for k := range raw {
			in4[k] = archsimd.LoadInt16x8Array(&raw[k])
			in8[k] = in4[k]
		}
		var got, want [32]archsimd.Int16x8
		itx32Low4Net(&in4, &got)
		itx32Low8Net(&in8, &want)
		for r := 0; r < 32; r++ {
			var g, w [8]int16
			got[r].StoreArray(&g)
			want[r].StoreArray(&w)
			if g != w {
				t.Fatalf("trial %d output %d: low4 %v low8 %v (inputs %v)", trial, r, g, w, raw)
			}
		}
	}
}
