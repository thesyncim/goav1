// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

// inverseDCT32Col4NEON is the NEON implementation of the batched
// (four-column) inverse DCT32 column kernel, ported from dav1d's
// high-bitdepth arm64 inverse transform shape (dav1d src/arm/64/itx16.S,
// inv_dct_4s_x16_neon): four adjacent columns occupy the four int32 lanes of
// every .4s register. base points at the first element of column col in the
// row-major scratch buffer; rowStrideBytes is the byte distance between
// successive rows (== width*4).
//
// Preconditions (enforced by the adapter): the buffer holds all four columns
// across all 32 rows, every input value is already clamped to [min, max], and
// [min, max] lies within the +/-2^19 stage-range envelope. Under those the
// kernel is bit-for-bit identical to inverseDCT32 run on each column
// (asserted by the column dispatch differential tests).
//
//go:noescape
func inverseDCT32Col4NEON(base *int32, rowStrideBytes, min, max int64)
