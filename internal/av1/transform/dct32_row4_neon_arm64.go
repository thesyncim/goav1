// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

// inverseDCT32Row4NEON is the NEON implementation of the batched (four-row)
// inverse DCT32 row kernel, ported from dav1d's high-bitdepth arm64 inverse
// transform shape (dav1d src/arm/64/itx16.S, inv_dct_4s_x16_neon with its
// 4x4 transposes): the four rows are transposed in 4x4 blocks so coefficient
// index i of each row shares the four int32 lanes of one .4s register.
//
// Preconditions (enforced by the adapter): each row holds at least 32 int32,
// every input value is already clamped to [min, max], and [min, max] lies
// within the +/-2^19 stage-range envelope. Under those the kernel is
// bit-for-bit identical to inverseDCT32Row run on each row (asserted by the
// row dispatch differential tests).
//
//go:noescape
func inverseDCT32Row4NEON(r0, r1, r2, r3 *int32, min, max int64)
