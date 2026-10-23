// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && arm64 && !purego)

package cdef

// searchBlocksU8 reports that this build has no batched search kernel; the
// encoder keeps its per-trial filter-then-measure loop.
func searchBlocksU8(src []byte, srcStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, damping int, trials []SearchTrial, rowStep int, sse []uint64) bool {
	return false
}
