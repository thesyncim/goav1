// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import "unsafe"

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
	ctx.sec0 = int(cdefDirections[4][0])
	ctx.sec1 = int(cdefDirections[0][0])
	ctx.sec2 = int(cdefDirections[4][1])
	ctx.sec3 = int(cdefDirections[0][1])
	bwLog2, bhLog2 := 3, 3
	if blockWidth == 4 {
		bwLog2 = 2
	}
	if blockHeight == 4 {
		bhLog2 = 2
	}
	base := unsafe.Pointer(unsafe.SliceData(input))
	dstBase := unsafe.Pointer(unsafe.SliceData(dst))
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		ctx.src = unsafe.Add(base, inputOrigin+((by*BStride)<<bhLog2)+(bx<<bwLog2))
		ctx.dst = unsafe.Add(dstBase, (by<<bhLog2)*dstStride+(bx<<bwLog2))
		if blockWidth == 8 {
			cdefByteAVX2Filter8Sec(&ctx)
		} else {
			cdefByteAVX2Filter4Sec(&ctx)
		}
	}
	return true
}
