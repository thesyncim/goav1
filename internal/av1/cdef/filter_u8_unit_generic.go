// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || purego

package cdef

// filterUnitBlocksU8 binds the canonical 8-bit per-unit block loop on targets
// without a tuned unit-level variant (and on purego builds). Per-block
// filtering still routes through the filterBlockU8Impl dispatch slot, so
// amd64 keeps its AVX2-backed block kernel. See filter_u8_dispatch.go.
func filterUnitBlocksU8(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams) error {
	return filterUnitBlocksU8PureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u)
}
