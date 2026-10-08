// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !arm64 || purego

package cdef

const cdefUnitInteriorScanFlavor = "scalar"

// cdefScanU16RowsAtMost255 is the scalar fallback for builds without the
// arm64 Go SIMD experiment. The rows are the exact tap footprint selected by
// cdefUnitInteriorU8; values outside the byte range include CDEF's border
// sentinel and require the wider filter path.
func cdefScanU16RowsAtMost255(input []uint16, start, rows, cols int) bool {
	for r := 0; r < rows; r++ {
		rowStart := start + r*BStride
		for c := 0; c < cols; c++ {
			if input[rowStart+c] > 0xFF {
				return false
			}
		}
	}
	return true
}
