// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package tile

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// coeffInitLevelsArch is the Go SIMD slice of SVT-AV1's
// svt_av1_txb_init_levels_neon adapted to goav1's column-major level scratch:
//
//	levels[col*(height+TX_PAD_HOR)+row] = min(abs(coeff[col*height+row]), 127)
//
// The full scratch window is cleared first (libaom's av1_txb_init_levels_c
// padding semantics); only the live rows of each column are written.
func coeffInitLevelsArch(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) bool {
	if !cpu.Detected.NEON {
		return false
	}
	switch scanHeight {
	case 4, 8, 16, 32:
	default:
		return false
	}
	clear(levels[:scratchLen])
	coeffInitLevelsSIMD(coeffs, scanWidth, scanHeight, levels)
	return true
}

// coeffInitAbs127x8 loads eight int16 coefficients and returns min(|c|, 127)
// in the low eight bytes of the result (upper eight zero). The unsigned min
// after Abs maps INT16_MIN (|c| wraps to 0x8000) to 127, matching
// coeffAbsClamp127.
func coeffInitAbs127x8(p *[8]int16, cap127 archsimd.Uint16x8) archsimd.Uint8x16 {
	return archsimd.LoadInt16x8Array(p).Abs().ToBits().Min(cap127).SaturateToUint8()
}

// coeffInitPack2x8 joins two packed eight-byte halves into one 16-byte vector
// (lo in bytes 0..7, hi in bytes 8..15) with a single EXT and ORR.
func coeffInitPack2x8(lo, hi archsimd.Uint8x16) archsimd.Uint8x16 {
	return lo.Or(hi.ConcatShiftBytesRight(lo, 8))
}

// coeffInitLevelsSIMD fills the live rows of each column. Columns are written in
// increasing order and a store may spill into the next column's leading bytes
// or the padding; those spilled bytes are zero and are overwritten by the next
// column's own store, while the final spill stays inside the scratch window
// ((scanWidth+TX_PAD_HOR) columns long), so the padding of every column ends up
// zero as the clear left it. The loop bounds the coefficient windows and the
// explicit length test bounds each column's store window.
func coeffInitLevelsSIMD(coeffs []int16, scanWidth int, scanHeight int, levels []uint8) {
	stride := scanHeight + txPadHorizontal
	n := scanWidth * scanHeight
	src := coeffs[:n:n]
	m := (scanWidth + txPadHorizontal) * stride
	dst := levels[:m:m]
	cap127 := archsimd.BroadcastUint16x8(127)
	switch scanHeight {
	case 4:
		// One vector covers two 4-row columns: bytes 0..3 are column c, bytes
		// 4..7 are column c+1. The lookup spreads them to the padded stride.
		pad4 := archsimd.LoadUint8x16Array(&coeffInitLevelsPad4)
		o := 0
		for i := 0; i <= len(src)-8; i += 8 {
			if o > len(dst)-16 {
				panic("tile: coefficient level scratch too short")
			}
			v := coeffInitAbs127x8((*[8]int16)(src[i:i+8]), cap127)
			v.LookupOrZero(pad4).StoreArray((*[16]uint8)(dst[o : o+16]))
			o += 2 * (4 + txPadHorizontal)
		}
	case 8:
		o := 0
		for i := 0; i <= len(src)-8; i += 8 {
			if o > len(dst)-16 {
				panic("tile: coefficient level scratch too short")
			}
			v := coeffInitAbs127x8((*[8]int16)(src[i:i+8]), cap127)
			// The upper eight bytes are zero: they land in this column's
			// padding and the next column's head, which its own store rewrites.
			v.StoreArray((*[16]uint8)(dst[o : o+16]))
			o += 8 + txPadHorizontal
		}
	case 16:
		o := 0
		for i := 0; i <= len(src)-16; i += 16 {
			if o > len(dst)-16 {
				panic("tile: coefficient level scratch too short")
			}
			w := (*[16]int16)(src[i : i+16])
			lo := coeffInitAbs127x8((*[8]int16)(w[0:8]), cap127)
			hi := coeffInitAbs127x8((*[8]int16)(w[8:16]), cap127)
			coeffInitPack2x8(lo, hi).StoreArray((*[16]uint8)(dst[o : o+16]))
			o += 16 + txPadHorizontal
		}
	case 32:
		o := 0
		for i := 0; i <= len(src)-32; i += 32 {
			if o > len(dst)-32 {
				panic("tile: coefficient level scratch too short")
			}
			w := (*[32]int16)(src[i : i+32])
			a := coeffInitAbs127x8((*[8]int16)(w[0:8]), cap127)
			b := coeffInitAbs127x8((*[8]int16)(w[8:16]), cap127)
			c := coeffInitAbs127x8((*[8]int16)(w[16:24]), cap127)
			d := coeffInitAbs127x8((*[8]int16)(w[24:32]), cap127)
			out := (*[32]uint8)(dst[o : o+32])
			coeffInitPack2x8(a, b).StoreArray((*[16]uint8)(out[0:16]))
			coeffInitPack2x8(c, d).StoreArray((*[16]uint8)(out[16:32]))
			o += 32 + txPadHorizontal
		}
	}
}

// coeffInitLevelsPad4 maps the packed pair vector of two 4-row columns onto
// the padded stride-8 layout: column c in bytes 0..3, zero padding in 4..7,
// column c+1 in bytes 8..11, zero padding in 12..15. Index 0xFF is out of
// range for the table lookup and yields zero.
var coeffInitLevelsPad4 = [16]uint8{
	0, 1, 2, 3, 0xff, 0xff, 0xff, 0xff, 4, 5, 6, 7, 0xff, 0xff, 0xff, 0xff,
}
