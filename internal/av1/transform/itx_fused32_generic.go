// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package transform

// InverseAddDCT32x32Low8 reports that this build has no fused 32x32 path;
// the caller runs the staged transform and the separate residual add.
func InverseAddDCT32x32Low8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int) bool {
	return false
}

// InverseAddDCT32x32Low16 is InverseAddDCT32x32Low8 for the 16x16 corner.
func InverseAddDCT32x32Low16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows int) bool {
	return false
}
