// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package frame

import "simd/archsimd"

// fillBytes stores the broadcast value sixteen bytes at a time; a tail
// shorter than a vector is stored through an overlapping final vector when
// the run is at least sixteen bytes, else byte by byte.
func fillBytes(dst []byte, v byte) {
	n := len(dst)
	if n < 16 {
		fillBytesScalar(dst, v)
		return
	}
	vec := archsimd.BroadcastUint8x16(v)
	i := 0
	for ; i+32 <= n; i += 32 {
		vec.StoreArray((*[16]uint8)(dst[i:]))
		vec.StoreArray((*[16]uint8)(dst[i+16:]))
	}
	if i+16 <= n {
		vec.StoreArray((*[16]uint8)(dst[i:]))
		i += 16
	}
	if i < n {
		vec.StoreArray((*[16]uint8)(dst[n-16:]))
	}
}
