// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package decoder

import (
	"encoding/binary"
	"simd/archsimd"
)

func copyBytes68(dst, src []byte) {
	d, s := (*[68]byte)(dst), (*[68]byte)(src)
	archsimd.LoadUint8x16Array((*[16]byte)(s[0:16])).StoreArray((*[16]byte)(d[0:16]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[16:32])).StoreArray((*[16]byte)(d[16:32]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[32:48])).StoreArray((*[16]byte)(d[32:48]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[48:64])).StoreArray((*[16]byte)(d[48:64]))
	binary.LittleEndian.PutUint32(d[64:68], binary.LittleEndian.Uint32(s[64:68]))
}

func copyBytes66(dst, src []byte) {
	d, s := (*[66]byte)(dst), (*[66]byte)(src)
	archsimd.LoadUint8x16Array((*[16]byte)(s[0:16])).StoreArray((*[16]byte)(d[0:16]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[16:32])).StoreArray((*[16]byte)(d[16:32]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[32:48])).StoreArray((*[16]byte)(d[32:48]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[48:64])).StoreArray((*[16]byte)(d[48:64]))
	d[64] = s[64]
	d[65] = s[65]
}

func copyBytes36(dst, src []byte) {
	d, s := (*[36]byte)(dst), (*[36]byte)(src)
	archsimd.LoadUint8x16Array((*[16]byte)(s[0:16])).StoreArray((*[16]byte)(d[0:16]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[16:32])).StoreArray((*[16]byte)(d[16:32]))
	binary.LittleEndian.PutUint32(d[32:36], binary.LittleEndian.Uint32(s[32:36]))
}

func copyBytes34(dst, src []byte) {
	d, s := (*[34]byte)(dst), (*[34]byte)(src)
	archsimd.LoadUint8x16Array((*[16]byte)(s[0:16])).StoreArray((*[16]byte)(d[0:16]))
	archsimd.LoadUint8x16Array((*[16]byte)(s[16:32])).StoreArray((*[16]byte)(d[16:32]))
	d[32] = s[32]
	d[33] = s[33]
}
