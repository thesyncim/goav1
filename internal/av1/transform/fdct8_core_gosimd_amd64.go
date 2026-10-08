// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package transform

// The amd64 core retains the eight-lane scratch transpose.
func fwd8DCTCore(coeff []int32, coeffStride int, residual []int16, residualStride int) {
	var buf, bufT [64]int32
	for g := 0; g < 8; g += fwdLanes {
		fwdColDCT8(buf[:], g, residual, residualStride)
	}
	fwdTranspose(bufT[:], buf[:], 8)
	for h := 0; h < 8; h += fwdLanes {
		fwdRowDCT8(coeff, coeffStride, bufT[:], h)
	}
}
