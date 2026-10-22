package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestCoeffTrustedContextsMatchChecked(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, size := range []TransformSize{TransformSize4x4, TransformSize8x8, TransformSize16x16, TransformSize32x32} {
		n := int(coeffGeometryTable[size].maxEOB)
		coeffs := make([]int16, n)
		levels := make([]uint8, coeffGeometryTable[size].scratchLen)
		for trial := range 50 {
			for i := range coeffs {
				coeffs[i] = 0
				if rng.Intn(3) == 0 {
					coeffs[i] = int16(rng.Intn(40) - 20)
				}
			}
			if err := CoeffInitLevels(coeffs, size, levels); err != nil {
				t.Fatal(err)
			}
			for ci := range n {
				want, err := CoeffLowerLevelsContext(levels, size, transform.Class2D, ci)
				if err != nil {
					t.Fatal(err)
				}
				if got := CoeffLowerLevelsContextTrusted(levels, size, transform.Class2D, ci); got != want {
					t.Fatalf("size %d trial %d ci %d: lower levels ctx %d want %d", size, trial, ci, got, want)
				}
				wantBR, err := CoeffBRContext(levels, size, transform.Class2D, ci)
				if err != nil {
					t.Fatal(err)
				}
				if got := CoeffBRContextTrusted(levels, size, transform.Class2D, ci); got != wantBR {
					t.Fatalf("size %d trial %d ci %d: br ctx %d want %d", size, trial, ci, got, wantBR)
				}
			}
		}
	}
}
