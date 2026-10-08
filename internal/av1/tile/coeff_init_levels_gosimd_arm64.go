// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package tile

import (
	"simd/archsimd"
	"unsafe"

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

// coeffInitAbs127x8 loads eight int16 coefficients at p and returns
// min(|c|, 127) in the low eight bytes of the result (upper eight zero). The
// unsigned min after Abs maps INT16_MIN (|c| wraps to 0x8000) to 127, matching
// coeffAbsClamp127.
func coeffInitAbs127x8(p unsafe.Pointer, cap127 archsimd.Uint16x8) archsimd.Uint8x16 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p)).Abs().ToBits().Min(cap127).SaturateToUint8()
}

// coeffInitPack2x8 joins two packed eight-byte halves into one 16-byte vector
// (lo in bytes 0..7, hi in bytes 8..15) with a single EXT and ORR.
func coeffInitPack2x8(lo, hi archsimd.Uint8x16) archsimd.Uint8x16 {
	return lo.Or(hi.ConcatShiftBytesRight(lo, 8))
}

// coeffInitStore16 writes all sixteen bytes of v at p.
func coeffInitStore16(p unsafe.Pointer, v archsimd.Uint8x16) {
	v.Store(unsafe.Slice((*uint8)(p), 16))
}

// coeffInitLevelsSIMD fills the live rows of each column. Columns are written in
// increasing order and a store may spill into the next column's leading bytes
// or the padding; those spilled bytes are zero and are overwritten by the next
// column's own store, while the final spill stays inside the scratch window
// ((scanWidth+TX_PAD_HOR) columns long), so the padding of every column ends up
// zero as the clear left it.
func coeffInitLevelsSIMD(coeffs []int16, scanWidth int, scanHeight int, levels []uint8) {
	stride := scanHeight + txPadHorizontal
	coeffs = coeffs[:scanWidth*scanHeight]
	levels = levels[:(scanWidth+txPadHorizontal)*stride]
	src := unsafe.Pointer(unsafe.SliceData(coeffs))
	dst := unsafe.Pointer(unsafe.SliceData(levels))
	cap127 := archsimd.BroadcastUint16x8(127)
	switch scanHeight {
	case 4:
		// One vector covers two 4-row columns: bytes 0..3 are column c, bytes
		// 4..7 are column c+1. The lookup spreads them to the padded stride.
		pad4 := archsimd.LoadUint8x16Array(&coeffInitLevelsPad4)
		for col := 0; col < scanWidth; col += 2 {
			v := coeffInitAbs127x8(unsafe.Add(src, 2*col*4), cap127)
			coeffInitStore16(unsafe.Add(dst, col*stride), v.LookupOrZero(pad4))
		}
	case 8:
		for col := range scanWidth {
			v := coeffInitAbs127x8(unsafe.Add(src, 2*col*8), cap127)
			*(*uint64)(unsafe.Add(dst, col*stride)) = v.ReshapeToUint64s().GetElem(0)
		}
	case 16:
		for col := range scanWidth {
			p := unsafe.Add(src, 2*col*16)
			lo := coeffInitAbs127x8(p, cap127)
			hi := coeffInitAbs127x8(unsafe.Add(p, 16), cap127)
			coeffInitStore16(unsafe.Add(dst, col*stride), coeffInitPack2x8(lo, hi))
		}
	case 32:
		for col := range scanWidth {
			p := unsafe.Add(src, 2*col*32)
			a := coeffInitAbs127x8(p, cap127)
			b := coeffInitAbs127x8(unsafe.Add(p, 16), cap127)
			c := coeffInitAbs127x8(unsafe.Add(p, 32), cap127)
			d := coeffInitAbs127x8(unsafe.Add(p, 48), cap127)
			out := unsafe.Add(dst, col*stride)
			coeffInitStore16(out, coeffInitPack2x8(a, b))
			coeffInitStore16(unsafe.Add(out, 16), coeffInitPack2x8(c, d))
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
