// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// warpHPerm{01,23,45,67} permute the 16 loaded source bytes into the four USDOT
// sample-pair layouts in one VTBL each: pair 2c holds col-2c's window
// (raw[2c..2c+7]) in bytes 0..7 and col-(2c+1)'s window (raw[2c+1..2c+8]) in
// bytes 8..15. Every index is in [0,14] so the TBL never zeroes. Replaces the
// EXT + UZP1 shuffle chain (11 ops) with four table lookups.
var (
	warpHPerm01Arr = [16]uint8{0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8}
	warpHPerm23Arr = [16]uint8{2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10}
	warpHPerm45Arr = [16]uint8{4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12}
	warpHPerm67Arr = [16]uint8{6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14}
)

// warpedFilterI8x8 is warpedFilter pre-narrowed to int8 (every tap is in
// [-22,127]) with each row's 8 taps packed little-endian into one int64. Built
// once at package init so the hot gather packs a filter pair with two scalar
// int64 stores + one vector load instead of copying 16 bytes and re-narrowing.
var warpedFilterI8x8 = func() [len(warpedFilter)]int64 {
	var t [len(warpedFilter)]int64
	for i := range warpedFilter {
		var p uint64
		for j := 0; j < filterTaps; j++ {
			p |= uint64(uint8(int8(warpedFilter[i][j]))) << (8 * uint(j))
		}
		t[i] = int64(p)
	}
	return t
}()

// warpHorizontal8ResidentSIMD is the Go-native SIMD form of the 8-bit
// warped-motion horizontal pass (interior/resident block, no edge clamp). It is
// byte-identical to the scalar reference warpHorizontal8Resident and returns
// sy4 unchanged.
//
// Algorithm (libaom av1/common/arm/warp_plane_neon_i8mm.c
// horizontal_filter_8x1_f8): each of the 15 rows produces 8 output columns, and
// every column uses a DIFFERENT 8-tap filter selected by
//
//	offs = roundPowerOfTwo(sx, warpedDiffPrecBits) + warpedPixelPrecShifts
//	sx += alpha        (per column)
//	sx  = sx4 + beta*(k+4)   (per row)
//
// with the 8 windows read from overlapping 1-pixel-shifted slices of the same
// row (col n reads ref[base + n .. base + n + 7], base = row + ix4 - 7). The
// eight per-column filters are gathered scalar-side (there is no SIMD gather),
// packed as four int8x16 tap-pairs (each pair = col2c taps || col2c+1 taps), the
// 16 source bytes are permuted into the four sample-pair layouts with one VTBL
// each, and USDOT dots them. The USDOT seed carries the horizontal bias plus the
// round0 rounding constant in its even lanes only (odd lanes zero) so the
// pairwise ConcatAddPairs that collapses the two half-sums does not double-count
// it; an arithmetic right shift by reduceBitsHoriz then yields the exact
// roundPowerOfTwo(sum, reduceBitsHoriz) int32 result (sum is non-negative for
// every resident phase so the shift is bit-identical). Four USDOT + two ADDP +
// one shift replace 64 scalar MACs per row.
func warpHorizontal8ResidentSIMD(tmp *[warpedIntermediateRows * warpedIntermediateColumns]int32, ref frame.Plane, ix4, sx4, iy4, sy4, alpha, beta, reduceBitsHoriz, offsetBitsHoriz int) int {
	// Even-lane bias = (1<<offsetBitsHoriz) + round const (1<<(reduceBitsHoriz-1)),
	// odd lanes zero: after ConcatAddPairs collapses each column's two 4-tap
	// half-sums, every output lane carries exactly one copy of bias+round, so the
	// subsequent >>reduceBitsHoriz is (sum + (1<<(reduceBitsHoriz-1))) >>
	// reduceBitsHoriz == roundPowerOfTwo(sum, reduceBitsHoriz).
	bias := int32((1 << offsetBitsHoriz) + (1 << (reduceBitsHoriz - 1)))
	seed := archsimd.LoadInt32x4Array(&[4]int32{bias, 0, bias, 0})
	shift := uint64(reduceBitsHoriz)

	perm01 := archsimd.LoadUint8x16Array(&warpHPerm01Arr)
	perm23 := archsimd.LoadUint8x16Array(&warpHPerm23Arr)
	perm45 := archsimd.LoadUint8x16Array(&warpHPerm45Arr)
	perm67 := archsimd.LoadUint8x16Array(&warpHPerm67Arr)

	rowStride := ref.Stride
	// Pointer-walk the source rows (row k=-7 starts at column ix4-7) to avoid the
	// per-iteration base+(k+7)*stride address recompute.
	rowp := unsafe.Pointer(&ref.Pix[(iy4-7)*rowStride+(ix4-7)])

	// offs = roundPowerOfTwo(s,10)+64 == ((s + 512 + 64*1024) >> 10). idxAdd folds
	// the round bias and the +64 (as 64<<10, an exact multiple of 1<<10) into the
	// shift input so each column index is a single arithmetic shift.
	const roundAdd = 1 << (warpedDiffPrecBits - 1)                          // 512
	const idxAdd = roundAdd + (warpedPixelPrecShifts << warpedDiffPrecBits) // 512 + 64*1024
	const maxIdx = len(warpedFilter) - 1

	var taps [8]int64
	dp := unsafe.Pointer(&tmp[0])
	for k := -7; k < 8; k++ {
		// Eight per-column filter offsets, computed serially in scalar registers
		// (kept out of memory so each feeds its table gather directly). offs advances
		// by alpha per column, so the eight raw indices are monotonic and the whole
		// row's range is bounded by its two endpoints off0/off7. When both endpoints
		// are in [0,maxIdx] (always true for resident warp params) the entire row is
		// in range and the per-column clamp is skipped; otherwise fall to the clamped
		// path that reproduces the scalar reference's out-of-range guard exactly.
		base0 := sx4 + beta*(k+4) + idxAdd
		off0 := base0 >> warpedDiffPrecBits
		off7 := (base0 + 7*alpha) >> warpedDiffPrecBits
		lo7, hi7 := off0, off7
		if off7 < off0 {
			lo7, hi7 = off7, off0
		}
		if uint(lo7) <= uint(maxIdx) && uint(hi7) <= uint(maxIdx) {
			acc := base0
			taps[0] = warpedFilterI8x8[acc>>warpedDiffPrecBits]
			acc += alpha
			taps[1] = warpedFilterI8x8[acc>>warpedDiffPrecBits]
			acc += alpha
			taps[2] = warpedFilterI8x8[acc>>warpedDiffPrecBits]
			acc += alpha
			taps[3] = warpedFilterI8x8[acc>>warpedDiffPrecBits]
			acc += alpha
			taps[4] = warpedFilterI8x8[acc>>warpedDiffPrecBits]
			acc += alpha
			taps[5] = warpedFilterI8x8[acc>>warpedDiffPrecBits]
			acc += alpha
			taps[6] = warpedFilterI8x8[acc>>warpedDiffPrecBits]
			taps[7] = warpedFilterI8x8[off7]
		} else {
			acc := base0
			taps[0] = warpedFilterI8x8[clampIdx(acc>>warpedDiffPrecBits, maxIdx)]
			acc += alpha
			taps[1] = warpedFilterI8x8[clampIdx(acc>>warpedDiffPrecBits, maxIdx)]
			acc += alpha
			taps[2] = warpedFilterI8x8[clampIdx(acc>>warpedDiffPrecBits, maxIdx)]
			acc += alpha
			taps[3] = warpedFilterI8x8[clampIdx(acc>>warpedDiffPrecBits, maxIdx)]
			acc += alpha
			taps[4] = warpedFilterI8x8[clampIdx(acc>>warpedDiffPrecBits, maxIdx)]
			acc += alpha
			taps[5] = warpedFilterI8x8[clampIdx(acc>>warpedDiffPrecBits, maxIdx)]
			acc += alpha
			taps[6] = warpedFilterI8x8[clampIdx(acc>>warpedDiffPrecBits, maxIdx)]
			taps[7] = warpedFilterI8x8[clampIdx(off7, maxIdx)]
		}
		fv01 := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&taps[0])))
		fv23 := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&taps[2])))
		fv45 := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&taps[4])))
		fv67 := archsimd.LoadInt8x16Array((*[16]int8)(unsafe.Pointer(&taps[6])))

		// Load 16 source bytes at the col-0 window start; the resident guarantee
		// (ix4-7>=0, ix4+7<W) keeps all 16 bytes in-bounds. One VTBL per pair builds
		// the two 1-pixel-shifted 8-byte sample windows.
		raw := archsimd.LoadUint8x16Array((*[16]uint8)(rowp))
		m01 := seed.DotProdUS(raw.LookupOrZero(perm01), fv01)
		m23 := seed.DotProdUS(raw.LookupOrZero(perm23), fv23)
		m45 := seed.DotProdUS(raw.LookupOrZero(perm45), fv45)
		m67 := seed.DotProdUS(raw.LookupOrZero(perm67), fv67)

		// ConcatAddPairs collapses each column's two 4-tap half-sums; ShiftAllRight
		// finishes roundPowerOfTwo. Cols 0..3 then 4..7.
		m01.ConcatAddPairs(m23).ShiftAllRight(shift).StoreArray((*[4]int32)(dp))
		m45.ConcatAddPairs(m67).ShiftAllRight(shift).StoreArray((*[4]int32)(unsafe.Add(dp, 16)))

		rowp = unsafe.Add(rowp, rowStride)
		dp = unsafe.Add(dp, warpedIntermediateColumns*4)
	}
	return sy4
}

// clampIdx maps a raw filter offset to [0, maxIdx] branchlessly, matching the
// scalar reference's out-of-range guard (offs<0 -> 0, offs>=len -> len-1).
func clampIdx(offs, maxIdx int) int {
	return min(max(offs, 0), maxIdx)
}
