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
	_ = dst[67]
	_ = src[67]
	archsimd.LoadUint8x16(src[0:16]).Store(dst[0:16])
	archsimd.LoadUint8x16(src[16:32]).Store(dst[16:32])
	archsimd.LoadUint8x16(src[32:48]).Store(dst[32:48])
	archsimd.LoadUint8x16(src[48:64]).Store(dst[48:64])
	binary.LittleEndian.PutUint32(dst[64:68], binary.LittleEndian.Uint32(src[64:68]))
}

func copyBytes66(dst, src []byte) {
	_ = dst[65]
	_ = src[65]
	archsimd.LoadUint8x16(src[0:16]).Store(dst[0:16])
	archsimd.LoadUint8x16(src[16:32]).Store(dst[16:32])
	archsimd.LoadUint8x16(src[32:48]).Store(dst[32:48])
	archsimd.LoadUint8x16(src[48:64]).Store(dst[48:64])
	dst[64] = src[64]
	dst[65] = src[65]
}

func copyBytes36(dst, src []byte) {
	_ = dst[35]
	_ = src[35]
	archsimd.LoadUint8x16(src[0:16]).Store(dst[0:16])
	archsimd.LoadUint8x16(src[16:32]).Store(dst[16:32])
	binary.LittleEndian.PutUint32(dst[32:36], binary.LittleEndian.Uint32(src[32:36]))
}

func copyBytes34(dst, src []byte) {
	_ = dst[33]
	_ = src[33]
	archsimd.LoadUint8x16(src[0:16]).Store(dst[0:16])
	archsimd.LoadUint8x16(src[16:32]).Store(dst[16:32])
	dst[32] = src[32]
	dst[33] = src[33]
}
