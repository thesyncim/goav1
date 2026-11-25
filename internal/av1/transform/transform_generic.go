// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package transform

func forwardDCT16x16N4Col(buf *[4 * 16]int32, residual *[256]int16) {
	forwardDCT16x16N4ColPureGo(buf, residual)
}

func forwardDCT32x32N4Col(buf *[8 * 32]int32, residual *[1024]int16) {
	forwardDCT32x32N4ColPureGo(buf, residual)
}

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

// InverseAdd4x16 is InverseAdd4x4 for the 4x16 block.
func InverseAdd4x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}

// InverseAdd16x4 is InverseAdd4x4 for the 16x4 block.
func InverseAdd16x4(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}

// InverseAdd8x16 is InverseAdd4x4 for the 8x16 block.
func InverseAdd8x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}

// InverseAdd16x8 is InverseAdd4x4 for the 16x8 block.
func InverseAdd16x8(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
	return false
}

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

// InverseAddDCT16x32 is the fused path for the 8-bit 16x32 DCT_DCT block.
func InverseAddDCT16x32(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	return false
}

// InverseAddDCT32x16 is the fused path for the 8-bit 32x16 DCT_DCT block.
func InverseAddDCT32x16(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool {
	return false
}

// InverseAddDCT8x8HBD reports that this build has no fused 16bpc path.
func InverseAddDCT8x8HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}

// InverseAddDCT16x16HBD reports that this build has no fused 16bpc path.
func InverseAddDCT16x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}

// InverseAddDCT32x32HBD reports that this build has no fused 16bpc path.
func InverseAddDCT32x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}

// InverseAdd16x16HybridHBD reports that this build has no fused 16bpc path.
func InverseAdd16x16HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}

// InverseAdd8x8HybridHBD reports that this build has no fused 16bpc path.
func InverseAdd8x8HybridHBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}

// InverseAddDCT64x64HBD reports that this build has no fused 16bpc path.
func InverseAddDCT64x64HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}

// InverseAddDCT16x32HBD reports that this build has no fused 16bpc path.
func InverseAddDCT16x32HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}

// InverseAddDCT32x16HBD reports that this build has no fused 16bpc path.
func InverseAddDCT32x16HBD(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	return false
}
