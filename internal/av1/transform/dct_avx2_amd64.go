// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build amd64 && !purego

package transform

// AVX2 implementations of the batched (two-row / two-column) inverse DCT
// kernels. The two values for each coefficient index occupy the low two 64-bit
// lanes of a YMM register; the products, rounding shifts and stage clamps are
// bit-for-bit identical to the pure-Go references (asserted by the dispatch
// differential tests). They are bound only when cpu.Detected.AVX2 is set.
//
// The row kernels take element pointers to the two rows and int64 clamp bounds;
// each row must hold at least the transform length. The column kernels take a
// base element pointer, the row stride in bytes (== width*4) and int64 bounds;
// the two adjacent columns occupy lanes 0 and 1.

//go:noescape
func inverseDCT8Row2AVX2(r0, r1 *int32, minv, maxv int64)

//go:noescape
func inverseDCT16Row2AVX2(r0, r1 *int32, minv, maxv int64)

//go:noescape
func inverseDCT8Col2AVX2(base *int32, rowStrideBytes, minv, maxv int64)

//go:noescape
func inverseDCT16Col2AVX2(base *int32, rowStrideBytes, minv, maxv int64)
