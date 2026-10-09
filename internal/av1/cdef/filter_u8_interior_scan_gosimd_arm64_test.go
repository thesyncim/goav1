// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "testing"

func TestCDEFUnitInteriorScanUsesGoSIMD(t *testing.T) {
	if cdefUnitInteriorScanFlavor != "simd" {
		t.Fatalf("interior scan flavor=%q, want simd", cdefUnitInteriorScanFlavor)
	}
	input := make([]uint16, BStride+8)
	for i := range input {
		input[i] = 255
	}
	if !cdefScanU16RowsAtMost255(input, 0, 2, 8) {
		t.Fatal("byte-range input must pass the Go SIMD scan")
	}
	input[BStride+7] = VeryLarge
	if cdefScanU16RowsAtMost255(input, 0, 2, 8) {
		t.Fatal("high sample in the final vector must fail the Go SIMD scan")
	}
}
