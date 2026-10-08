// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"testing"

	"simd/archsimd"
)

func TestCDEFInt16AbsDiffPreservesLaneWrap(t *testing.T) {
	const minInt16 = -32768
	const maxInt16 = 32767
	a := [8]int16{minInt16, maxInt16, minInt16, maxInt16, -1, 0, 1, -30000}
	b := [8]int16{maxInt16, minInt16, minInt16, maxInt16, 0, -1, 30000, 30000}
	got := [8]int16{}
	cdefAbsDiffInt16x8(archsimd.LoadInt16x8Array(&a), archsimd.LoadInt16x8Array(&b)).StoreArray(&got)

	for i := range a {
		delta := int32(a[i]) - int32(b[i])
		if delta < 0 {
			delta = -delta
		}
		want := int16(uint16(delta)) // SABD returns the difference's low 16 bits.
		if got[i] != want {
			t.Fatalf("lane %d: absdiff(%d, %d) = %d, want wrapped %d", i, a[i], b[i], got[i], want)
		}
	}
}
