// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build purego || !arm64 || !goexperiment.simd

package cdef

func u8SecondaryOnlyByteInteriorAvailable() bool { return false }

func filterFrameBlocksU8SecondaryOnlyByteInterior(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, blockWidth int, blockHeight int, strength int, shift int) bool {
	return false
}
