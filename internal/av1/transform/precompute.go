// Precomputed, immutable lookup tables for transform metadata.
//
// AV1 transform sizes draw width/height from the fixed set {4,8,16,32,64} and
// scan orders depend only on (size x scan-mode). Both domains are tiny and
// constant, so this file builds the derived metadata (validity, post-column
// shift, adjusted scan size) and every scan/inverse-scan order exactly once at
// package init and serves reads from the resulting read-only tables. The
// tables are written only during init and never mutated afterwards, so they
// are safe to share across decode goroutines without synchronisation.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// A transform dimension (4,8,16,32,64) maps to a compact 0..4 slot, or -1 when
// the dimension is not an AV1 transform extent. dimSlotTable performs this map
// indexed by the raw pixel dimension; out-of-range dimensions are handled by
// dimSlot/sizeIndex.
const (
	numDimSlots  = 5  // {4,8,16,32,64}
	numSizeSlots = 25 // numDimSlots * numDimSlots
	invalidSize  = -1
)

// dimSlotTable maps a raw transform dimension (0..64) to its compact 0..4 slot,
// or invalidSize for any value that is not an AV1 transform extent. Indexing a
// fixed array avoids the per-call branch ladder of a switch on this hot path.
// It is fully specified at initialisation (no reliance on init ordering) so it
// is safe for the package-level metadata init to consult via sizeIndex.
var dimSlotTable = buildDimSlotTable()

func buildDimSlotTable() [65]int8 {
	var t [65]int8
	for i := range t {
		t[i] = invalidSize
	}
	t[4] = 0
	t[8] = 1
	t[16] = 2
	t[32] = 3
	t[64] = 4
	return t
}

// dimSlot returns the compact 0..4 slot for an AV1 transform dimension, or -1.
func dimSlot(d int) int {
	if uint(d) >= uint(len(dimSlotTable)) {
		return invalidSize
	}
	return int(dimSlotTable[d])
}

// sizeIndex returns a compact 0..24 index for s, or -1 when either dimension is
// not an AV1 transform extent. The index space includes some shapes that are
// not valid AV1 transforms (e.g. 4x32); per-index validity is tracked in
// sizeValidTable.
func sizeIndex(s Size) int {
	w := dimSlot(int(s.Width))
	h := dimSlot(int(s.Height))
	if w < 0 || h < 0 {
		return invalidSize
	}
	return w*numDimSlots + h
}

var (
	// sizeValidTable[idx] reports whether the size at compact index idx is a
	// supported AV1 transform shape.
	sizeValidTable [numSizeSlots]bool
	// sizeShiftTable[idx] holds the inverse-transform post-column shift for a
	// valid size; the entry is meaningless when sizeValidTable[idx] is false.
	sizeShiftTable [numSizeSlots]int8
	// adjustedScanSizeTable[idx] holds the coefficient scan dimensions for a
	// valid size (the libaom 64-extent clamp applied).
	adjustedScanSizeTable [numSizeSlots]Size

	// scanTables holds the precomputed forward and inverse scan orders for
	// every (compact size index x scan mode). A nil entry means the
	// combination is not realised (invalid size); valid entries are filled at
	// init and never mutated afterwards.
	scanTables [numSizeSlots][numScanModes]scanPair
)

const numScanModes = 5 // ScanModeZigZag..ScanModeRow1D

type scanPair struct {
	scan    []int16
	inverse []int16
}

func init() {
	// Enumerate the canonical valid AV1 transform sizes and record their
	// derived metadata. The shape set and shifts mirror Size.shift exactly.
	type sizeShift struct {
		size  Size
		shift int8
	}
	valid := []sizeShift{
		{Size{Width: 4, Height: 4}, 0},
		{Size{Width: 4, Height: 8}, 0},
		{Size{Width: 8, Height: 4}, 0},

		{Size{Width: 4, Height: 16}, 1},
		{Size{Width: 8, Height: 8}, 1},
		{Size{Width: 8, Height: 16}, 1},
		{Size{Width: 16, Height: 4}, 1},
		{Size{Width: 16, Height: 8}, 1},
		{Size{Width: 16, Height: 32}, 1},
		{Size{Width: 32, Height: 16}, 1},
		{Size{Width: 32, Height: 64}, 1},
		{Size{Width: 64, Height: 32}, 1},

		{Size{Width: 8, Height: 32}, 2},
		{Size{Width: 16, Height: 16}, 2},
		{Size{Width: 16, Height: 64}, 2},
		{Size{Width: 32, Height: 8}, 2},
		{Size{Width: 32, Height: 32}, 2},
		{Size{Width: 64, Height: 16}, 2},
		{Size{Width: 64, Height: 64}, 2},
	}

	for _, vs := range valid {
		idx := sizeIndex(vs.size)
		sizeValidTable[idx] = true
		sizeShiftTable[idx] = vs.shift
		adjustedScanSizeTable[idx] = computeAdjustedScanSize(vs.size)
	}

	// Precompute the scan and inverse-scan order for every valid size and
	// scan mode. The forward fill uses the canonical generator once per
	// combination; callers later copy out of these immutable buffers.
	for _, vs := range valid {
		idx := sizeIndex(vs.size)
		scanSize := adjustedScanSizeTable[idx]
		total := int(scanSize.Width) * int(scanSize.Height)
		for mode := range numScanModes {
			scan := make([]int16, total)
			inverse := make([]int16, total)
			fillScanOrderCompute(scan, inverse, scanSize, ScanMode(mode))
			scanTables[idx][mode] = scanPair{scan: scan, inverse: inverse}
		}
	}
}

// computeAdjustedScanSize mirrors the historical adjustedScanSize switch and is
// used only at init to seed adjustedScanSizeTable.
func computeAdjustedScanSize(size Size) Size {
	switch size {
	case Size{Width: 64, Height: 64},
		Size{Width: 64, Height: 32},
		Size{Width: 32, Height: 64}:
		return Size{Width: 32, Height: 32}
	case Size{Width: 64, Height: 16}:
		return Size{Width: 32, Height: 16}
	case Size{Width: 16, Height: 64}:
		return Size{Width: 16, Height: 32}
	default:
		return size
	}
}
