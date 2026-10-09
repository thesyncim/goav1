// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && amd64 && !purego

package tile

import (
	"encoding/binary"
	"simd/archsimd"
)

// coeffInitLevelsPickEven gathers the low byte of each int16 lane within each
// 128-bit half (VPSHUFB); indices with the high bit set produce zero. After the
// gather, each half's low eight bytes hold eight clamped levels.
var coeffInitLevelsPickEven = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
	0, 2, 4, 6, 8, 10, 12, 14, -128, -128, -128, -128, -128, -128, -128, -128,
}

// coeffInitLevelsArch is the amd64 AVX2 slice of libaom's av1_txb_init_levels_c
// adapted to goav1's column-major levels[col*(height+TX_PAD_HOR)+row] scratch.
// Bit-identical to coeffInitLevelsPureGo; the dispatch is gated on OS-enabled
// AVX2 through archsimd.
func coeffInitLevelsArch(coeffs []int16, scanWidth int, scanHeight int, levels []uint8, scratchLen int) bool {
	if !archsimd.X86.AVX2() {
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

// coeffInitAbs127x16 returns min(|c|, 127) for sixteen int16 coefficients as
// two uint64 words: lo holds coefficients 0..7 and hi holds 8..15, one byte
// each. The unsigned min after Abs maps INT16_MIN (|c| wraps to 0x8000) to 127,
// matching coeffAbsClamp127.
func coeffInitAbs127x16(p *[16]int16, cap127 archsimd.Uint16x16, pick archsimd.Int8x32) (lo, hi uint64) {
	v := archsimd.LoadInt16x16Array(p).Abs().ToBits().Min(cap127).ReshapeToUint8s().PermuteOrZeroGrouped(pick).ReshapeToUint64s()
	return v.GetLo().GetElem(0), v.GetHi().GetElem(0)
}

func coeffInitLevelsSIMD(coeffs []int16, scanWidth int, scanHeight int, levels []uint8) {
	cap127 := archsimd.BroadcastUint16x16(127)
	pick := archsimd.LoadInt8x32Array(&coeffInitLevelsPickEven)
	stride := scanHeight + txPadHorizontal
	switch scanHeight {
	case 4:
		// Each 16-coefficient chunk is four 4-row columns: lo carries
		// columns c and c+1, hi carries c+2 and c+3, four bytes each.
		for col := 0; col < scanWidth; col += 4 {
			lo, hi := coeffInitAbs127x16((*[16]int16)(coeffs[col*4:]), cap127, pick)
			binary.LittleEndian.PutUint32(levels[col*stride:], uint32(lo))
			binary.LittleEndian.PutUint32(levels[(col+1)*stride:], uint32(lo>>32))
			binary.LittleEndian.PutUint32(levels[(col+2)*stride:], uint32(hi))
			binary.LittleEndian.PutUint32(levels[(col+3)*stride:], uint32(hi>>32))
		}
	case 8:
		// Each chunk is two 8-row columns: lo is column c, hi is column c+1.
		for col := 0; col < scanWidth; col += 2 {
			lo, hi := coeffInitAbs127x16((*[16]int16)(coeffs[col*8:]), cap127, pick)
			binary.LittleEndian.PutUint64(levels[col*stride:], lo)
			binary.LittleEndian.PutUint64(levels[(col+1)*stride:], hi)
		}
	case 16:
		for col := range scanWidth {
			lo, hi := coeffInitAbs127x16((*[16]int16)(coeffs[col*16:]), cap127, pick)
			dst := levels[col*stride:]
			binary.LittleEndian.PutUint64(dst, lo)
			binary.LittleEndian.PutUint64(dst[8:], hi)
		}
	case 32:
		for col := range scanWidth {
			dst := levels[col*stride:]
			for k := range 2 {
				lo, hi := coeffInitAbs127x16((*[16]int16)(coeffs[col*32+16*k:]), cap127, pick)
				binary.LittleEndian.PutUint64(dst[16*k:], lo)
				binary.LittleEndian.PutUint64(dst[16*k+8:], hi)
			}
		}
	}
}
