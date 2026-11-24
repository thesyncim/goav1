// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && amd64 && !purego

package cdef

// The amd64 secondary-only byte route runs the AVX2 byte-lane sec-only
// kernels (dav1d cdef_avx2.asm CDEF_FILTER .sec_only, direction 0), the same
// kernels the general byte route picks for a primary-strength-zero unit, so
// the decoder takes the identical route on amd64 and arm64.
func u8SecondaryOnlyByteInteriorAvailable() bool { return cdefHasAVX2 }

func filterFrameBlocksU8SecondaryOnlyByteInterior(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, blockWidth int, blockHeight int, strength int, shift int) bool {
	if !cdefHasAVX2 || (blockWidth != 4 && blockWidth != 8) || (blockHeight != 4 && blockHeight != 8) {
		return false
	}
	// The AVX2 kernels load 16 bytes per tap row; the exported entry proves
	// only the 2-pixel halo. Decline before any write unless the whole
	// over-read window of the blocks' bounding box is inside input, the same
	// slack FilterFrameBlocksU8ByteInteriorTrusted requires.
	rowsEnd, colsEnd := 0, 0
	for _, block := range blocks {
		rowsEnd = max(rowsEnd, (int(block.BY)+1)*blockHeight)
		colsEnd = max(colsEnd, (int(block.BX)+1)*blockWidth)
	}
	if inputOrigin+(rowsEnd+1)*BStride+colsEnd+16 > len(input) {
		return false
	}
	ctx := cdefByteAVX2Ctx{dstStr: dstStride, height: blockHeight, secStr: uint8(strength), secSh: uint64(shift)}
	ctx.taps = &cdefByteTapIdx[0]
	ctx.input, ctx.dst = input, dst
	bwLog2, bhLog2 := 3, 3
	if blockWidth == 4 {
		bwLog2 = 2
	}
	if blockHeight == 4 {
		bhLog2 = 2
	}
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		ctx.srcOff = inputOrigin + ((by * BStride) << bhLog2) + (bx << bwLog2)
		ctx.dstOff = (by<<bhLog2)*dstStride + (bx << bwLog2)
		if blockWidth == 8 {
			cdefByteAVX2Filter8Sec(&ctx)
		} else {
			cdefByteAVX2Filter4Sec(&ctx)
		}
	}
	return true
}
