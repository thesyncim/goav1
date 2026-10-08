//go:build goexperiment.simd && (arm64 || amd64) && !purego

package quantize

import (
	"math"
	"testing"
)

// dequantSIMDCoeffDomain covers every int16 coefficient value once, followed
// by a repeat of the first seven values so every n%8 tail length is exercised
// after a full-width body.
func dequantSIMDCoeffDomain() []int16 {
	coeff := make([]int16, 65536+8)
	for i := range 65536 {
		coeff[i] = int16(i)
	}
	for i := range 8 {
		coeff[65536+i] = int16(i)
	}
	return coeff
}

// TestDequantColumnSIMDMatchesPureGoExhaustive proves the SIMD dequant kernel
// bit-exact with dequantColumnPureGo over the full int16 coefficient domain,
// every tx scale and bit-depth clamp, and scales spanning the table range plus
// values whose products wrap int32 (only the low 24 bits matter).
func TestDequantColumnSIMDMatchesPureGoExhaustive(t *testing.T) {
	if !dequantSIMDSupported() {
		t.Skip("SIMD dequant kernel not supported on this CPU")
	}
	coeff := dequantSIMDCoeffDomain()
	scales := []int32{1, 4, 9, 38, 44, 74, 88, 1828, 7312, 29247, 65535, 1<<20 + 3}
	bounds := [][2]int32{
		{-1 << 15, 1<<15 - 1},
		{-1 << 17, 1<<17 - 1},
		{-1 << 19, 1<<19 - 1},
		{math.MinInt32, math.MaxInt32},
	}
	for _, n := range []int{65536, 65536 + 1, 65536 + 3, 65536 + 7, 65536 + 8} {
		c := coeff[:n]
		for _, scale := range scales {
			for txScale := uint8(0); txScale <= 2; txScale++ {
				for _, b := range bounds {
					want := make([]int32, n)
					got := make([]int32, n)
					dequantColumnPureGo(want, c, scale, txScale, b[0], b[1])
					dequantColumnSIMD(got, c, scale, txScale, b[0], b[1])
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("n=%d scale=%d txScale=%d bounds=%v coeff[%d]=%d: got %d want %d",
								n, scale, txScale, b, i, c[i], got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestDequantColumnSIMDZeroAlloc(t *testing.T) {
	if !dequantSIMDSupported() {
		t.Skip("SIMD dequant kernel not supported on this CPU")
	}
	coeff := dequantSIMDCoeffDomain()[:1024]
	dst := make([]int32, 1024)
	allocs := testing.AllocsPerRun(200, func() {
		dequantColumnSIMD(dst, coeff, 88, 1, -32768, 32767)
	})
	if allocs != 0 {
		t.Fatalf("dequantColumnSIMD allocs = %v, want 0", allocs)
	}
}
