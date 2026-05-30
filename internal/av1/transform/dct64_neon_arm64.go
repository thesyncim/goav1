// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

// inverseDCT64Row2NEON is the NEON implementation of the batched (two-row)
// inverse DCT64 row kernel. r0 and r1 must each hold at least 64 int32. It is
// bit-for-bit identical to inverseDCT64Row run on each row (asserted by the row
// dispatch differential test).
//
//go:noescape
func inverseDCT64Row2NEON(r0, r1 *int32, min, max int64)

// inverseDCT64Col2NEON is the NEON implementation of the batched (two-column)
// inverse DCT64 column kernel. base points at the first element of column col
// in the row-major scratch buffer; the two adjacent columns (col, col+1) occupy
// lanes 0 and 1 of every loaded vector and are transformed together.
// rowStrideBytes is the byte distance between successive rows (== width*4). It
// is bit-for-bit identical to inverseDCT64 run on each column (asserted by the
// column dispatch differential test).
//
//go:noescape
func inverseDCT64Col2NEON(base *int32, rowStrideBytes, min, max int64)
