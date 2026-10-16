package tile

import (
	"math/rand"
	"testing"
)

// TestTXBEOBMatchesScan checks every txbEOB form against the backwards scan
// walk it replaces, over sparse random blocks and every single-coefficient
// position.
func TestTXBEOBMatchesScan(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	scanEOB := func(coeffs []int16, scan []coeffScanHot) int {
		for c := len(scan) - 1; c >= 0; c-- {
			if coeffs[scan[c].pos] != 0 {
				return c + 1
			}
		}
		return 0
	}
	fill := func(coeffs []int16, trial int) {
		clear(coeffs)
		switch {
		case trial < len(coeffs):
			coeffs[trial] = int16(trial%7) - 3
			if coeffs[trial] == 0 {
				coeffs[trial] = -32768
			}
		default:
			k := rng.Intn(6)
			for range k {
				coeffs[rng.Intn(len(coeffs))] = int16(rng.Intn(65536) - 32768)
			}
		}
	}
	check := func(name string, coeffs []int16, scan []coeffScanHot, got int) {
		if want := scanEOB(coeffs, scan); got != want {
			t.Fatalf("%s: eob %d want %d", name, got, want)
		}
	}
	var c16 [16]int16
	var c64 [64]int16
	var c256 [256]int16
	var c1024 [1024]int16
	for trial := range 3000 {
		fill(c16[:], trial)
		check("4x4", c16[:], coeffScanHot4x4Y2D[:], txbEOB16(&c16))
		fill(c64[:], trial)
		check("8x8", c64[:], coeffScanHot8x8Y2D[:], txbEOB64(&c64))
		fill(c256[:], trial)
		check("16x16", c256[:], coeffScanHot16x16Y2D[:], txbEOB256(&c256))
		fill(c1024[:], trial)
		check("32x32", c1024[:], coeffScanHot32x32Y2D[:], txbEOB1024(&c1024))
	}
}
