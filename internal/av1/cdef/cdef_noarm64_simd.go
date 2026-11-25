// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !arm64 || !goexperiment.simd

package cdef

// filterUnitBlocksU8 binds the canonical 8-bit per-unit block loop on targets
// without a tuned unit-level variant (and on purego builds). Per-block
// filtering still routes through the filterBlockU8Impl dispatch slot, so
// amd64 keeps its AVX2-backed block kernel. See filter_u8_dispatch.go.
func filterUnitBlocksU8(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	return filterUnitBlocksU8PureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
}

// filterUnitBlocks binds the canonical per-unit block loop on targets without
// a tuned unit-level variant (and on purego builds). Per-block filtering
// still routes through the filterBlockImpl dispatch slot, so amd64 keeps its
// AVX2 block kernel. See filter_dispatch.go for why this is a build-tag
// binding instead of a func variable.
func filterUnitBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	return filterUnitBlocksPureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
}

func filterUnitBlocksU16LE(dst []byte, dstStrideB int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	return filterUnitBlocksU16LEPureGo(dst, dstStrideB, input, inputOrigin, blocks, directions, variances, u)
}
