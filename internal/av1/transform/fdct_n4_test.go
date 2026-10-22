// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package transform

import (
	"math/rand"
	"testing"
)

// TestForwardDCT8x8N4MatchesFullTransform checks SVT's N4_SHAPE semantics:
// the low 2x2 coefficients equal the full 8x8 DCT_DCT, the rest are zero.
func TestForwardDCT8x8N4MatchesFullTransform(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	var residual [64]int16
	for trial := 0; trial < 2000; trial++ {
		for i := range residual {
			switch trial % 3 {
			case 0:
				residual[i] = int16(rng.Intn(511) - 255)
			case 1:
				residual[i] = int16(255 - 510*rng.Intn(2))
			default:
				residual[i] = int16(rng.Intn(31) - 15)
			}
		}
		var full, want [64]int32
		forwardDCT8x8PureGo(full[:], 8, residual[:], 8)
		want[0], want[1], want[8], want[9] = full[0], full[1], full[8], full[9]
		for name, fn := range map[string]func(*[64]int32, *[64]int16){
			"purego": forwardDCT8x8N4PureGo,
			"impl":   forwardDCT8x8N4Impl,
		} {
			got := [64]int32{5: 7, 63: 9}
			fn(&got, &residual)
			if got != want {
				t.Fatalf("%s trial %d: got %v want %v", name, trial, got[:10], want[:10])
			}
		}
	}
}
