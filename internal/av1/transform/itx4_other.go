// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && amd64 && !purego)

package transform

// InverseAdd4x4 reports that this build has no fused 4x4 path; the caller
// runs the staged transform and the separate residual add.
func InverseAdd4x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows int) bool {
	return false
}

// InverseAdd4x8 is InverseAdd4x4 for the 4x8 block.
func InverseAdd4x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}

// InverseAdd8x4 is InverseAdd4x4 for the 8x4 block.
func InverseAdd8x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}
