// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"simd/archsimd"
)

// lfStore8 narrows eight 16-bit lanes already inside 0..255 to bytes and writes
// them at pix[off:off+8]. Saturating narrow is exact on in-range lanes and
// compiles to a NEON SQXTUN. The bytes are staged through a stack array and
// copied as one 64-bit word, so the write is a single store with no lane
// inserts and no unsafe pointer that would make the staging array escape.
func lfStore8(pix []byte, off int, v archsimd.Int16x8) {
	var tmp [16]uint8
	v.ToBits().SaturateToUint8().StoreArray(&tmp)
	binary.LittleEndian.PutUint64(pix[off:off+8], binary.LittleEndian.Uint64(tmp[:8]))
}

// lfLoad8 reads eight bytes at pix[off:off+8] as signed 16-bit lanes. The bytes
// are staged through a stack array as one 64-bit word, then loaded as a single
// vector, so no lane inserts are needed.
func lfLoad8(pix []byte, off int) archsimd.Int16x8 {
	var tmp [16]uint8
	binary.LittleEndian.PutUint64(tmp[:8], binary.LittleEndian.Uint64(pix[off:off+8]))
	return archsimd.LoadUint8x16Array(&tmp).ExtendLo8ToUint16().BitsToInt16()
}

// lfAny reports whether any lane of the mask is set.
func lfAny(m archsimd.Mask16x8) bool {
	return m.ToInt16x8().ToBits().ReduceSum() != 0
}
