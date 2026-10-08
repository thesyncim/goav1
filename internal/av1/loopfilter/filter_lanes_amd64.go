// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

import (
	"encoding/binary"
	"simd/archsimd"
)

// lfStore8 writes the first eight lanes of v, which lie inside 0..255, as bytes
// at pix[off:off+8]. The byte-narrowing vector ops on amd64 (VPMOVUSWB, VPMOVWB)
// are AVX-512 only and would fault on AVX2 hosts, so the narrowing is a scalar
// copy through a stack array instead.
func lfStore8(pix []byte, off int, v archsimd.Int16x8) {
	var lanes [8]int16
	v.StoreArray(&lanes)
	dst := pix[off : off+8]
	for i, lane := range lanes {
		dst[i] = byte(lane)
	}
}

// lfLoad8 reads eight bytes at pix[off:off+8] as signed 16-bit lanes. The bytes
// are staged through a stack array as one 64-bit word, then loaded as a single
// vector, so no lane inserts are needed.
func lfLoad8(pix []byte, off int) archsimd.Int16x8 {
	var tmp [16]uint8
	binary.LittleEndian.PutUint64(tmp[:8], binary.LittleEndian.Uint64(pix[off:off+8]))
	return archsimd.LoadUint8x16Array(&tmp).ExtendLo8ToUint16().BitsToInt16()
}
