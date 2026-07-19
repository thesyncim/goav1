// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

// CDEF interior 8-bit-dst .16b kernels: goav1's port of dav1d's fully-edged
// 8bpc fast path (src/arm/64/cdef.S cdef_filter{8,4}_*_edged_8bpc_neon). dav1d
// takes this path when edges == 0xf — every CDEF neighbour is a real pixel, so
// there is no CDEF_VERY_LARGE border and the whole block filters in uint8 .16b
// lanes, 16 samples per pass (two rows for w=8, four for w=4), doubling the
// throughput of the .8h path in filter_u8_neon_arm64.s. goav1 keeps its tap
// buffer in uint16 (libaom's 0x4000 sentinel), so cdefUnitInteriorU8 first
// proves the unit's whole tap footprint is sentinel-free; only then are its
// blocks routed here, where each tap pair is narrowed (xtn/xtn2) to uint8
// losslessly. Output is bit-identical to filterBlockU8PureGo — see
// filter_u8_interior_neon_arm64.s and u8_interior_differential_test.go.

//go:noescape
func cdefFilterBlock8InteriorU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock8PrimaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock8SecondaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock4InteriorU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock4PrimaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)

//go:noescape
func cdefFilterBlock4SecondaryInteriorU8NEON(ctx *filterBlockU8NEONCtx)

// dispatchFilterBlockU8InteriorNEON is the interior counterpart of
// dispatchFilterBlockU8NEON: it routes a prepared ctx to the width- and
// strength-specialized .16b kernel (dav1d's pri/sec/pri_sec split of the edged
// 8bpc path). The caller guarantees the block is interior (sentinel-free tap
// footprint) and that the height matches the kernel's row packing (even for
// w=8, a multiple of four for w=4).
func dispatchFilterBlockU8InteriorNEON(ctx *filterBlockU8NEONCtx, width int, primaryStrength int, secondaryStrength int) {
	if width == 8 {
		switch {
		case primaryStrength != 0 && secondaryStrength == 0:
			cdefFilterBlock8PrimaryInteriorU8NEON(ctx)
		case primaryStrength == 0 && secondaryStrength != 0:
			cdefFilterBlock8SecondaryInteriorU8NEON(ctx)
		default:
			cdefFilterBlock8InteriorU8NEON(ctx)
		}
		return
	}
	switch {
	case primaryStrength != 0 && secondaryStrength == 0:
		cdefFilterBlock4PrimaryInteriorU8NEON(ctx)
	case primaryStrength == 0 && secondaryStrength != 0:
		cdefFilterBlock4SecondaryInteriorU8NEON(ctx)
	default:
		cdefFilterBlock4InteriorU8NEON(ctx)
	}
}

// cdefTapReach is the maximum |offset|, in samples, that any CDEF tap reads
// beyond the current pixel along either axis (cdefDirections secondary taps
// reach two rows and two columns). A block whose whole footprint expanded by
// this reach holds only real samples has no VeryLarge sentinel in any tap.
const cdefTapReach = 2

// cdefUnitInteriorU8 reports whether every CDEF tap the unit's blocks read
// lands on a real 0..255 sample, i.e. the tap footprint of the block bounding
// box (expanded by cdefTapReach on every side) is free of the VeryLarge border
// sentinel. That is exactly dav1d's edges == 0xf predicate for goav1's tap
// buffer: when it holds, the interior .16b kernels may narrow the uint16 taps
// to uint8 losslessly. The scan is one strided pass over the footprint, run
// once per unit and amortized across all of the unit's blocks.
func cdefUnitInteriorU8(input []uint16, inputOrigin int, blocks []BlockPosition, bwLog2 int, bhLog2 int) bool {
	if len(input) == 0 || len(blocks) == 0 {
		return false
	}
	blockWidth := 1 << bwLog2
	blockHeight := 1 << bhLog2
	minBX := int(blocks[0].BX)
	maxBX := minBX
	minBY := int(blocks[0].BY)
	maxBY := minBY
	for i := 1; i < len(blocks); i++ {
		bx := int(blocks[i].BX)
		by := int(blocks[i].BY)
		if bx < minBX {
			minBX = bx
		}
		if bx > maxBX {
			maxBX = bx
		}
		if by < minBY {
			minBY = by
		}
		if by > maxBY {
			maxBY = by
		}
	}
	// inputOrigin addresses the unit's first interior sample (buffer row
	// VerticalBorder, column HorizontalBorder). Walk out cdefTapReach on each
	// side of the block bounding box.
	topLeft := inputOrigin + ((minBY*BStride)<<bhLog2) + (minBX << bwLog2)
	scanStart := topLeft - cdefTapReach*BStride - cdefTapReach
	nRows := (maxBY-minBY+1)*blockHeight + 2*cdefTapReach
	nCols := (maxBX-minBX+1)*blockWidth + 2*cdefTapReach
	if scanStart < 0 {
		return false
	}
	end := scanStart + (nRows-1)*BStride + nCols
	if end > len(input) {
		return false
	}
	var acc uint16
	for r := 0; r < nRows; r++ {
		row := input[scanStart+r*BStride:]
		for c := 0; c < nCols; c++ {
			acc |= row[c]
		}
		if acc > 0xFF {
			return false
		}
	}
	return acc <= 0xFF
}
