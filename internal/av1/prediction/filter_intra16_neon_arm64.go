// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package prediction

// NEON-accelerated 16-bit (high-bit-depth) AV1 filter-intra predictor. The asm
// kernel filterIntraRow16NEONAsm fills one row pair (see
// filter_intra16_neon_arm64.s); this wrapper walks the row pairs, builds the
// padded uint16 `above` buffer for each pair and resolves the left-edge seeds,
// so the steady-state math stays inside the vector loop. It is bit-exact with
// predictFilterIntraBlockDirect16: the dispatch slot only reaches this path for
// 2-byte samples, and filter-intra widths are always a multiple of four, but a
// width that is not is routed back to the scalar reference for safety.
//
// It reuses filterIntraTapsNEON (the transpose of filterIntraTaps built by the
// 8-bit path's init): [mode][tap][output], so the asm can load seven 8-lane
// int16 vectors (one per tap) and broadcast the seven neighbour samples across
// them.

//go:noescape
func filterIntraRow16NEONAsm(dst0 *byte, dst1 *byte, above *uint16, left0 uint32, left1 uint32, taps *int16, n uintptr, max uint32)

func predictFilterIntraBlockDirect16NEON(block planeBlock, width int, height int, mode FilterIntraMode, edges IntraEdges, max int) {
	if width%4 != 0 {
		predictFilterIntraBlockDirect16(block, width, height, mode, edges, max)
		return
	}
	taps := &filterIntraTapsNEON[mode][0][0]
	batches := uintptr(width >> 2)
	maxU := uint32(max)
	// above holds p0 (top-left) at index 0 and the top row at indices 1..width.
	// The asm reads eight halfwords per batch (p0..p4 plus slack), so the array
	// is padded past width to keep every load in bounds without a heap alloc.
	var above [48]uint16
	for row := 0; row < height; row += 2 {
		if row == 0 {
			above[0] = edges.AboveLeft
			for i := 0; i < width; i++ {
				above[i+1] = edges.Above[i]
			}
		} else {
			above[0] = edges.Left[row-1]
			top := block.pix[(row-1)*block.stride:]
			for i := 0; i < width; i++ {
				above[i+1] = uint16(top[i<<1]) | uint16(top[i<<1+1])<<8
			}
		}
		filterIntraRow16NEONAsm(
			&block.pix[row*block.stride],
			&block.pix[(row+1)*block.stride],
			&above[0],
			uint32(edges.Left[row]),
			uint32(edges.Left[row+1]),
			taps,
			batches,
			maxU,
		)
	}
}
