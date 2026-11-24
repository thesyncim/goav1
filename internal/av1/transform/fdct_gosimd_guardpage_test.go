// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego && (darwin || linux)

package transform

import (
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
)

func TestForwardDCT4x4SIMDDoesNotReadPastResidual(t *testing.T) {
	// Place exactly 16 int16 samples at the end of the readable memory. A
	// full 8-lane load from the last row crosses into the inaccessible page.
	residual := guardpage.After[int16](t, 16)
	for i := range residual {
		residual[i] = int16(i*17 - 128)
	}
	var want, got [16]int32
	forwardDCT4x4PureGo(want[:], 4, residual, 4)
	guardpage.Check(t, "forwardDCT4x4SIMD", func() { forwardDCT4x4SIMD(got[:], 4, residual, 4) })
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("coeff[%d]=%d want %d", i, got[i], want[i])
		}
	}
}
