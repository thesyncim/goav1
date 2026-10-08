// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "simd/archsimd"

const cdefUnitInteriorScanFlavor = "simd"

// cdefScanU16RowsAtMost255 checks eight samples per official Go SIMD vector.
// It only loads complete vectors inside each row and checks the exact row tail
// with scalar loads, so the final row can end at the slice boundary.
func cdefScanU16RowsAtMost255(input []uint16, start, rows, cols int) bool {
	for r := 0; r < rows; r++ {
		rowStart := start + r*BStride
		row := input[rowStart : rowStart+cols]
		var acc = archsimd.BroadcastUint16x8(0)
		c := 0
		for ; c+8 <= cols; c += 8 {
			chunk := row[c : c+8]
			acc = acc.Or(archsimd.LoadUint16x8Array((*[8]uint16)(chunk)))
		}
		for ; c < cols; c++ {
			if row[c] > 0xFF {
				return false
			}
		}
		if acc.ReduceMax() > 0xFF {
			return false
		}
	}
	return true
}
