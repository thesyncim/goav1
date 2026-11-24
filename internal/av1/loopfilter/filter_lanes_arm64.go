// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"encoding/binary"
	"simd/archsimd"
)

// lfStore8 narrows eight in-range lanes and writes exactly eight bytes. The
// narrow is a NEON SQXTUN and the low 64-bit lane becomes one store.
func lfStore8(pix []byte, off int, v archsimd.Int16x8) {
	binary.LittleEndian.PutUint64(pix[off:off+8], v.ToBits().SaturateToUint8().ReshapeToUint64s().GetElem(0))
}

// lfLoad8 reads exactly eight bytes and zero-extends them to signed lanes.
func lfLoad8(pix []byte, off int) archsimd.Int16x8 {
	return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(pix[off : off+8])).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
}

// lfAny reports whether any lane of the mask is set.
func lfAny(m archsimd.Mask16x8) bool {
	return m.ToInt16x8().ToBits().ReduceSum() != 0
}

func lfAll(m archsimd.Mask16x8) bool {
	return m.ToInt16x8().ToBits().ReduceSum() == 65528
}

// lfShift1 shifts by a reusable vector on arm64.
func lfShift1(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}

// lfShift3 shifts by a reusable vector on arm64.
func lfShift3(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}

// lfShift4 shifts by a reusable vector on arm64.
func lfShift4(v, by archsimd.Int16x8) archsimd.Int16x8 {
	return v.Shift(by)
}
