// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

// inverseDCT8Col2NEON and inverseDCT16Col2NEON are the NEON implementations of
// the batched (two-column) inverse DCT column kernels. base points at the first
// element of column col in the row-major scratch buffer; the two adjacent
// columns (col, col+1) occupy lanes 0 and 1 of every loaded vector and are
// transformed together. rowStrideBytes is the byte distance between successive
// rows (== width*4). The arithmetic, rounding and clamping are bit-for-bit
// identical to inverseDCT8 / inverseDCT16 run on each column (asserted by the
// column dispatch differential test).
//
//go:noescape
func inverseDCT8Col2NEON(base *int32, rowStrideBytes, min, max int64)

//go:noescape
func inverseDCT16Col2NEON(base *int32, rowStrideBytes, min, max int64)
