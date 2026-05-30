// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

// inverseDCT4Row2NEON and inverseDCT8Row2NEON are the NEON implementations of
// the batched (two-row) inverse DCT row kernels. They transform two adjacent
// stride-1 rows in place; r0 and r1 must each hold at least the transform
// length (4 and 8 int32 respectively). The arithmetic, rounding and clamping
// are bit-for-bit identical to the pure-Go references inverseDCT4Row2PureGo /
// inverseDCT8Row2PureGo (asserted by the dispatch differential test).
//
//go:noescape
func inverseDCT4Row2NEON(r0, r1 *int32, min, max int64)

//go:noescape
func inverseDCT8Row2NEON(r0, r1 *int32, min, max int64)

// inverseADST4Row2NEON / inverseADST8Row2NEON are the NEON implementations of
// the batched (two-row) inverse ADST row kernels. ADST4 needs no clamp bounds
// (its outputs pass only through clipInt32, which the narrowing covers); ADST8
// takes the stage clamp bounds. Both are bit-for-bit identical to their
// pure-Go references.
//
//go:noescape
func inverseADST4Row2NEON(r0, r1 *int32)

//go:noescape
func inverseADST8Row2NEON(r0, r1 *int32, min, max int64)

// inverseDCT16Row2NEON is the NEON implementation of the batched (two-row)
// inverse DCT16 row kernel. r0 and r1 must each hold at least 16 int32. It is
// bit-for-bit identical to inverseDCT16Row2PureGo.
//
//go:noescape
func inverseDCT16Row2NEON(r0, r1 *int32, min, max int64)
