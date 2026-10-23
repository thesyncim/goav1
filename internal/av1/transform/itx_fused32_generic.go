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

// InverseAddDCT16x16 is the fused path for the 8-bit 16x16 DCT_DCT block.
func InverseAddDCT16x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	return false
}

// InverseAddDCT8x8 is the fused path for the 8-bit 8x8 DCT_DCT block.
func InverseAddDCT8x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	return false
}

// InverseAddDCT32x32Full is the fused path for the 8-bit 32x32 DCT_DCT block
// past the 16x16 corner.
func InverseAddDCT32x32Full(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	return false
}

// InverseAddDCT64x64 is the fused path for the 8-bit 64x64 DCT_DCT block.
func InverseAddDCT64x64(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	return false
}

// InverseAdd16x16Hybrid is the fused path for the 8-bit 16x16 DCT/ADST
// pairings.
func InverseAdd16x16Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}

// InverseAdd8x8Hybrid is the fused path for the 8-bit 8x8 DCT/ADST pairings.
func InverseAdd8x8Hybrid(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}
