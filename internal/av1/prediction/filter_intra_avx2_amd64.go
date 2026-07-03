// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package prediction

// AVX2-accelerated 8-bit AV1 filter-intra predictor. The asm kernel
// filterIntraRow8AVX2Asm fills one row pair (see filter_intra_avx2_amd64.s);
// this wrapper walks the row pairs, builds the padded `above` buffer for each
// pair and resolves the left-edge seeds, so the steady-state math stays inside
// the vector loop. It mirrors the arm64 NEON kernel in
// filter_intra_neon_arm64.s and is bit-exact with
// predictFilterIntraBlockDirect8: the dispatch slot only reaches this path for
// 8-bit samples, and filter-intra widths are always a multiple of four, but a
// width that is not is routed back to the scalar reference for safety.
//
// filterIntraTapsAVX2 is the transpose of filterIntraTaps: [mode][tap][output],
// so the asm can load seven 8-lane int16 vectors (one per tap) and broadcast
// the seven neighbour samples across them.
//
// dav1d reference: third_party/upstream/dav1d/src/x86/ipred_avx2.asm
// (ipred_filter_8bpc / filter_intra_taps).

//go:noescape
func filterIntraRow8AVX2Asm(dst0 *byte, dst1 *byte, above *uint8, left0 uint32, left1 uint32, taps *int16, n uint64)

var filterIntraTapsAVX2 [FilterIntraModes][7][8]int16

func init() {
	for mode := range filterIntraTaps {
		for output := 0; output < 8; output++ {
			for tap := 0; tap < 7; tap++ {
				filterIntraTapsAVX2[mode][tap][output] = int16(filterIntraTaps[mode][output][tap])
			}
		}
	}
}

func predictFilterIntraBlockDirect8AVX2(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 || max != 0xff {
		predictFilterIntraBlockDirect8(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraTapsAVX2[mode][0][0]
	batches := uint64(width >> 2)
	// above holds p0 (top-left) at index 0 and the top row at indices 1..width.
	// The asm reads p0..p4 (indices up to width) per row pair, so the array is
	// padded past width to keep every load in bounds without a heap allocation.
	var above [48]byte
	for row := 0; row < height; row += 2 {
		if row == 0 {
			above[0] = byte(edges.AboveLeft)
			for i := 0; i < width; i++ {
				above[i+1] = byte(edges.Above[i])
			}
		} else {
			above[0] = byte(edges.Left[row-1])
			top := block.pix[(row-1)*block.stride:]
			for i := 0; i < width; i++ {
				above[i+1] = top[i]
			}
		}
		filterIntraRow8AVX2Asm(
			&block.pix[row*block.stride],
			&block.pix[(row+1)*block.stride],
			&above[0],
			uint32(edges.Left[row]),
			uint32(edges.Left[row+1]),
			taps,
			batches,
		)
	}
}
