// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

// U8ByteInteriorAvailable reports whether this build has the byte-tap
// kernels for real-interior 8-bit units (dav1d's 8bpc edges == 0xf path:
// src/arm/64/cdef.S cdef_padding*_edged_8bpc_neon + cdef_filter*_edged_8bpc_neon).
// Builds without them decline before the caller assembles byte scratch.
func U8ByteInteriorAvailable() bool {
	return u8ByteInteriorAvailable()
}

// FilterFrameBlocksU8ByteInteriorTrusted is FilterFrameBlocksU8Trusted (or
// its NoPrimaryDirection twin when deriveDirections is false) for a
// decoder-owned 8-bit unit whose whole 2-pixel tap footprint is real: input
// is a byte tap buffer with BStride spacing, inputOrigin at the unit's first
// pixel, holding the pre-filter pixels of the unit and that halo. This is
// the shape of dav1d's 8bpc edged CDEF, where every tap is an 8-bit sample
// and the filter runs in byte lanes with no sentinel handling.
//
// It returns false without touching dst or the grids when the shape,
// parameters, or build are unsupported; callers then use the widened path.
// The buffer must also allow 16-byte vector reads that start inside the
// read window: inputOrigin+(unitHeight+1)*BStride+unitWidth+16 <= len(input).
func FilterFrameBlocksU8ByteInteriorTrusted(dst []byte, dstStride int, input []byte, inputOrigin int, unitWidth int, unitHeight int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, params FrameFilterParams, deriveDirections bool) bool {
	if !u8ByteInteriorAvailable() || len(blocks) == 0 || params.CoeffShift != 0 ||
		params.Level > 15 ||
		(params.SecondaryStrength != 0 && params.SecondaryStrength != 1 && params.SecondaryStrength != 2 && params.SecondaryStrength != 4) ||
		params.XDec > 1 || params.YDec > 1 || params.Plane > PlaneV {
		return false
	}
	blockWidth := 8 >> params.XDec
	blockHeight := 8 >> params.YDec
	if unitWidth <= 0 || unitHeight <= 0 || unitWidth%blockWidth != 0 || unitHeight%blockHeight != 0 ||
		unitWidth+4 > BStride || dstStride < unitWidth {
		return false
	}
	damping := int(params.Damping)
	if params.Plane != PlaneY {
		damping--
	}
	if damping < 0 || damping > 7 {
		return false
	}
	if inputOrigin < 2*BStride+2 || inputOrigin+(unitHeight+1)*BStride+unitWidth+16 > len(input) {
		return false
	}
	if (unitHeight-1)*dstStride+unitWidth > len(dst) {
		return false
	}
	for _, block := range blocks {
		if (int(block.BX)+1)*blockWidth > unitWidth || (int(block.BY)+1)*blockHeight > unitHeight {
			return false
		}
	}
	if deriveDirections && !deriveU8Directions(dst, dstStride, blocks, directions, variances, params) {
		return false
	}
	primaryStrength := int(params.Level)
	secondaryStrength := int(params.SecondaryStrength)
	if primaryStrength == 0 && secondaryStrength == 0 {
		return true
	}
	bwLog2 := 3 - int(params.XDec)
	bhLog2 := 3 - int(params.YDec)
	filterUnitBlocksU8Byte(dst, dstStride, input, inputOrigin, blocks, directions, variances, unitFilterParams{
		primaryStrength:   primaryStrength,
		secondaryStrength: secondaryStrength,
		damping:           damping,
		bwLog2:            bwLog2,
		bhLog2:            bhLog2,
		blockWidth:        blockWidth,
		blockHeight:       blockHeight,
		lumaAdjust:        params.Plane == PlaneY && (deriveDirections || primaryStrength != 0),
	})
	return true
}
