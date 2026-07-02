// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || purego

package cdef

// filterUnitBlocks binds the canonical per-unit block loop on targets without
// a tuned unit-level variant (and on purego builds). Per-block filtering
// still routes through the filterBlockImpl dispatch slot, so amd64 keeps its
// AVX2 block kernel. See filter_dispatch.go for why this is a build-tag
// binding instead of a func variable.
func filterUnitBlocks(dst []uint16, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, trusted bool) error {
	return filterUnitBlocksPureGo(dst, dstStride, input, inputOrigin, blocks, directions, variances, u, trusted)
}
