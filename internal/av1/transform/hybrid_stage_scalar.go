// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !arm64 || purego

package transform

// stageTransposeClamp stages the rows x cols rectangle of the column-major
// coefficient block into row-major scratch lines, clamping to [lo, hi] (rect2
// applies the sqrt2 scale first). This build binds the scalar reference; the
// goexperiment.simd arm64 build overrides it with a 4x4-block SIMD transpose
// (hybrid_stage_gosimd_arm64.go).
func stageTransposeClamp(scratch []int32, width int, coeff []int32, coeffStride int, rows int, cols int, rect2 bool, lo int32, hi int32) {
	stageTransposeClampScalar(scratch, width, coeff, coeffStride, rows, cols, rect2, lo, hi)
}
