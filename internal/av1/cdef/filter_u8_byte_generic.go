// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build purego || !arm64 || !goexperiment.simd

package cdef

func u8ByteInteriorAvailable() bool { return false }

func filterUnitBlocksU8Byte(dst []byte, dstStride int, input []byte, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) {
}
