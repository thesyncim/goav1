// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package prediction

import "simd/archsimd"

// widenEdge8 widens the bytes src[:len(dst)] into dst. It loads sixteen bytes
// per vector while src holds them, so src may extend past len(dst).
func widenEdge8(dst []uint16, src []byte) {
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		v := archsimd.LoadUint8x16Array((*[16]uint8)(src[i:]))
		v.ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[i:]))
		v.HiToLo().ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[i+8:]))
	}
	if i+8 <= len(dst) && i+16 <= len(src) {
		archsimd.LoadUint8x16Array((*[16]uint8)(src[i:])).ExtendLo8ToUint16().StoreArray((*[8]uint16)(dst[i:]))
		i += 8
	}
	for ; i < len(dst); i++ {
		dst[i] = uint16(src[i])
	}
}
