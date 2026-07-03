// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

// NEON-accelerated 8-bit AV1 filter-intra predictor. The asm kernel
// filterIntraRow8NEONAsm fills one row pair (see filter_intra_neon_arm64.s);
// this wrapper walks the row pairs, builds the padded `above` buffer for each
// pair and resolves the left-edge seeds, so the steady-state math stays inside
// the vector loop. It is bit-exact with predictFilterIntraBlockDirect8: the
// dispatch slot only reaches this path for 8-bit samples, and filter-intra
// widths are always a multiple of four, but a width that is not is routed back
// to the scalar reference for safety.
//
// filterIntraTapsNEON is the transpose of filterIntraTaps: [mode][tap][output],
// so the asm can load seven 8-lane int16 vectors (one per tap) and broadcast
// the seven neighbour samples across them.

//go:noescape
func filterIntraRow8NEONAsm(dst0 *byte, dst1 *byte, above *uint8, left0 uint32, left1 uint32, taps *int16, n uintptr)

var filterIntraTapsNEON [FilterIntraModes][7][8]int16

func init() {
	for mode := range filterIntraTaps {
		for output := 0; output < 8; output++ {
			for tap := 0; tap < 7; tap++ {
				filterIntraTapsNEON[mode][tap][output] = int16(filterIntraTaps[mode][output][tap])
			}
		}
	}
}

func predictFilterIntraBlockDirect8NEON(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 || max != 0xff {
		predictFilterIntraBlockDirect8(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraTapsNEON[mode][0][0]
	batches := uintptr(width >> 2)
	// above holds p0 (top-left) at index 0 and the top row at indices 1..width.
	// The asm reads eight bytes per batch (p0..p4 plus slack), so the array is
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
		filterIntraRow8NEONAsm(
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
