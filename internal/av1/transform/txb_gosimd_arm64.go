// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"encoding/binary"
	"simd/archsimd"
)

// txbInitLevels8x8SIMD is the Go SIMD form of the 8x8 TXB level init:
// levels[col*12+row] = min(abs(coeff[col*8+row]), 127) for row 0..7, with the
// four pad bytes at col*12+8..11 cleared. abs wraps MinInt16 to the unsigned
// pattern 0x8000 (32768), which the unsigned min clips to 127 exactly as the
// saturating abs did.
func txbInitLevels8x8SIMD(coeffs *[64]int16, levels *[256]uint8) {
	lim := archsimd.BroadcastUint16x8(127)
	for col := range 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(coeffs[col*8:]))
		lv := v.Abs().ToBits().Min(lim).TruncToUint8()
		base := col * 12
		binary.LittleEndian.PutUint64(levels[base:], lv.ReshapeToUint64s().GetElem(0))
		binary.LittleEndian.PutUint32(levels[base+8:], 0)
	}
}

func txbPrep8x8Levels2DSIMD(coeffs *[64]int16, levels *[256]uint8, absLevels *[64]uint16, eob int) TXB8x8PrepResult {
	txbInitLevels8x8SIMD(coeffs, levels)
	return txbPrep8x8Summary(coeffs, absLevels, eob)
}
