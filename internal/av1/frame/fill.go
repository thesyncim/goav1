// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

// FillBytes sets every byte of dst to v. The edge-replication passes (a
// plane's border extension here, the encoder's padded references) fill
// border runs of 64 to 160 bytes per row; the Go compiler lowers a byte fill
// loop to one store per byte, so the SIMD builds store sixteen at a time.
func FillBytes(dst []byte, v byte) {
	fillBytes(dst, v)
}

// fillBytesScalar is the reference fill.
func fillBytesScalar(dst []byte, v byte) {
	for i := range dst {
		dst[i] = v
	}
}
