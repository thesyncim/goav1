package threading

import "encoding/binary"

// FrameWorkLoopFilterNoSkip is dav1d's per-superblock CDEF noskip_mask
// (src/lf_mask.h Av1Filter.noskip_mask, written in src/decode.c decode_b for
// every block with coded residual and read by src/cdef_apply_tmpl.c to skip
// all-skip 8x8 blocks), stored as eight row bytes per 64x64 CDEF unit with
// one bit per 8x8 block: the same layout the compact single-tile path keeps,
// so CDEF reads one 64-bit word per unit. Tiles are superblock aligned, so
// concurrently decoded tiles write disjoint bytes. Caller owned; grown on
// demand and reused across frames.
type FrameWorkLoopFilterNoSkip struct {
	buf      []byte
	unitCols int
	unitRows int
}

// Preseed hands n caller-owned backing storage (capacity for the largest frame)
// so binding never allocates.
func (n *FrameWorkLoopFilterNoSkip) Preseed(buf []byte) { n.buf = buf[:0] }

// initFrameWorkLoopFilterNoSkip clears noskip for a cols x rows MI frame and
// pre-sets the out-of-frame bits of the final unit column/row, matching the
// record-map CDEF skip test (blocks outside the decoded grid are never
// all-skip). cols and rows must be even.
func initFrameWorkLoopFilterNoSkip(noskip []byte, cols, rows, unitCols, unitRows int) {
	clear(noskip)
	validCols := cols >> 1
	validRows := rows >> 1
	if tail := validCols & 7; tail != 0 {
		rightMask := uint8(0xff << uint(tail))
		for unitY := 0; unitY < unitRows; unitY++ {
			base := (unitY*unitCols + unitCols - 1) * 8
			for by := 0; by < 8; by++ {
				noskip[base+by] |= rightMask
			}
		}
	}
	if tail := validRows & 7; tail != 0 {
		for unitX := 0; unitX < unitCols; unitX++ {
			base := ((unitRows-1)*unitCols + unitX) * 8
			for by := tail; by < 8; by++ {
				noskip[base+by] = 0xff
			}
		}
	}
}

// markFrameWorkLoopFilterNoSkip sets the noskip bit of every 8x8 block the MI
// rectangle touches (dav1d decode.c: noskip_mask[by4 >> 1][bx_idx] |= mask
// for each 8-row pair of a block with !b->skip).
func markFrameWorkLoopFilterNoSkip(noskip []byte, unitCols, miColStart, miRowStart, miColEnd, miRowEnd int) {
	startX := miColStart >> 1
	startY := miRowStart >> 1
	endX := (miColEnd + 1) >> 1
	endY := (miRowEnd + 1) >> 1
	for y := startY; y < endY; y++ {
		unitY := y >> 3
		localY := y & 7
		for x := startX; x < endX; x++ {
			unitX := x >> 3
			off := (unitY*unitCols+unitX)*8 + localY
			if off >= 0 && off < len(noskip) {
				noskip[off] |= 1 << uint(x&7)
			}
		}
	}
}

// BindNoSkip sizes, clears and attaches dav1d noskip storage for this frame so
// the decode-time build records which 8x8 blocks carry residual. Frames with
// odd MI extents keep the record-map CDEF skip source.
func (h *FrameWorkLoopFilterMasks) BindNoSkip(n *FrameWorkLoopFilterNoSkip) {
	h.noskip = nil
	if n == nil || !h.Valid() || h.Cols&1 != 0 || h.Rows&1 != 0 {
		return
	}
	n.unitCols = (h.Cols + 15) >> 4
	n.unitRows = (h.Rows + 15) >> 4
	n.buf = growBytes(n.buf, n.unitCols*n.unitRows*8)
	initFrameWorkLoopFilterNoSkip(n.buf, h.Cols, h.Rows, n.unitCols, n.unitRows)
	h.noskip = n
}

// NoSkipUnitGrid reports whether dav1d noskip is bound for exactly the
// requested CDEF 64x64-unit grid.
func (h *FrameWorkLoopFilterMasks) NoSkipUnitGrid(unitCols, unitRows int) bool {
	return h != nil && h.noskip != nil && h.noskip.unitCols == unitCols && h.noskip.unitRows == unitRows &&
		len(h.noskip.buf) >= unitCols*unitRows*8
}

// NoSkipUnitMask returns the noskip word of one 64x64 unit: bit by*8+bx is set
// when 8x8 block (by, bx) carries residual. Bytes are little-endian row-major.
func (h *FrameWorkLoopFilterMasks) NoSkipUnitMask(unitRow, unitCol int) uint64 {
	n := h.noskip
	off := (unitRow*n.unitCols + unitCol) * 8
	return binary.LittleEndian.Uint64(n.buf[off : off+8])
}

// MapFree reports that this frame's decode-time build carries everything the
// post-filter needs (dav1d's lflvl structures: levels, noskip and, for
// multi-tile frames, the tile-edge backups), so the per-MI loop-filter record
// map need not be marked.
func (h *FrameWorkLoopFilterMasks) MapFree() bool {
	return h != nil && h.levels != nil && h.noskip != nil && (h.tileEdges != nil || h.singleTile)
}
