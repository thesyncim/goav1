package transform

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// TestInverseBlockBitDepthRawBoundedMatchesFull checks the eob-bounded
// sub-size kernels (activeCols/activeRows) against the unbounded transform
// for every size and DCT-row type that can use them, across bit depths,
// extremes and sparse patterns.
func TestInverseBlockBitDepthRawBoundedMatchesFull(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 9))
	sizes := []Size{{32, 8}, {32, 16}, {32, 32}, {32, 64}, {8, 32}, {16, 32}, {64, 32}, {16, 16}, {16, 4}, {16, 8}, {16, 64}, {4, 16}, {8, 16}, {64, 16}}
	for _, size := range sizes {
		scan := adjustedScanSizeTable[sizeIndex(size)]
		cw, ch := int(scan.Width), int(scan.Height)
		for _, bd := range []uint8{8, 10, 12} {
			cfMax := int32(1)<<(7+bd) - 1
			for iter := 0; iter < 300; iter++ {
				cols := 1 + rng.IntN(min(cw, 20))
				rows := 1 + rng.IntN(ch)
				coeff := make([]int32, cw*ch)
				n := 1 + rng.IntN(cols*rows)
				for range n {
					x, y := rng.IntN(cols), rng.IntN(rows)
					v := int32(rng.IntN(int(2*cfMax+2))) - cfMax - 1
					if rng.IntN(4) == 0 {
						v = int32(rng.IntN(64)) - 32
					}
					coeff[x*ch+y] = v
				}
				want := make([]int32, int(size.Width)*int(size.Height))
				got := make([]int32, len(want))
				if err := InverseBlockBitDepthRaw(want, coeff, ch, size, TypeDCTDCT, bd, 0); err != nil {
					t.Fatal(err)
				}
				if err := InverseBlockBitDepthRawBounded(got, coeff, ch, size, TypeDCTDCT, bd, rows, cols); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got, want) {
					t.Fatalf("%dx%d bd=%d rows=%d cols=%d: bounded transform differs", size.Width, size.Height, bd, rows, cols)
				}
			}
		}
	}
}
