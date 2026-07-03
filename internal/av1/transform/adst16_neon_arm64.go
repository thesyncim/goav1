// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

// inverseADST16Col4NEON is the NEON implementation of the batched
// (four-column) inverse ADST16 column kernel, ported from dav1d's
// high-bitdepth arm64 inverse transform shape (src/arm/64/itx16.S,
// inv_adst_4s_x16_neon): four adjacent columns occupy the four int32 lanes of
// every .4s register. base points at the first element of column col in the
// row-major scratch buffer; rowStrideBytes is the byte distance between
// successive rows (== width*4).
//
// Preconditions (enforced by the adapter): the buffer holds all four columns
// across all 16 rows, every input value is already clamped to [min, max], and
// [min, max] lies within the +/-2^19 stage-range envelope. Under those the
// kernel is bit-for-bit identical to inverseADST16 run on each column
// (asserted by the column dispatch differential tests).
//
//go:noescape
func inverseADST16Col4NEON(base *int32, rowStrideBytes, min, max int64)

// inverseADST16Row4NEON transforms four adjacent stride-1 rows in place, using
// the same int32-lane ADST16 butterfly as the column kernel with a 4x4
// transpose on the way in and out. Bit-for-bit identical to inverseADST16 run
// on each row (asserted by the row dispatch differential tests).
//
//go:noescape
func inverseADST16Row4NEON(r0, r1, r2, r3 *int32, min, max int64)
