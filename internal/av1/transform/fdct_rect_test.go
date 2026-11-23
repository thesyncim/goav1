// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"math/rand"
	"testing"
)

// TestForwardDCTRectTrustedMatchesScalar checks the trusted 2:1 DCT_DCT
// rectangles against the scalar kernels over random and extreme 8-bit
// residuals, at the packed coefficient stride (in place) and a wider one.
func TestForwardDCTRectTrustedMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for _, size := range []Size{{16, 8}, {8, 16}, {32, 16}, {16, 32}} {
		w, h := int(size.Width), int(size.Height)
		for _, extra := range []int{0, 3} {
			rs, cs := w+5, h+extra
			residual := make([]int16, rs*h)
			got := make([]int32, cs*w)
			want := make([]int32, cs*w)
			for trial := range 300 {
				for r := range h {
					for c := range w {
						var v int16
						switch trial % 4 {
						case 0:
							v = int16(rng.Intn(511) - 255)
						case 1:
							v = 255
						case 2:
							v = 255
							if (r+c)&1 == 1 {
								v = -255
							}
						default:
							v = int16(255 * (rng.Intn(3) - 1))
						}
						residual[r*rs+c] = v
					}
				}
				clear(got)
				clear(want)
				if err := ForwardBlock8BitResidualTrusted(got, cs, residual, rs, nil, size, TypeDCTDCT); err != nil {
					t.Fatal(err)
				}
				if err := forwardDCTBySize(want, cs, residual, rs, size); err != nil {
					t.Fatal(err)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%dx%d stride %d trial %d: coeff[%d] = %d, want %d", w, h, cs, trial, i, got[i], want[i])
					}
				}
			}
		}
	}
}
