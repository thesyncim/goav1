// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

// U8SecondaryOnlyByteInteriorAvailable reports whether this build has the
// private byte-tap kernel used by the decoder's real-interior secondary-only
// path. The generic and pure-Go builds decline before the caller prepares its
// byte scratch, leaving the existing U8 filter path untouched.
func U8SecondaryOnlyByteInteriorAvailable() bool {
	return u8SecondaryOnlyByteInteriorAvailable()
}

// FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted filters a decoder-owned
// U8 unit whose complete 2-pixel tap footprint is real. input is a byte view
// with BStride spacing and inputOrigin at the unit's first pixel. The unit
// dimensions must be exact multiples of the block shape selected by params.
//
// The entry only accepts primary strength zero and secondary strength 1, 2,
// or 4, and leaves direction and variance grids untouched. It returns false
// without modifying dst if the shape, spans, parameters, or build-specific
// implementation are unsupported; callers may then assemble the existing
// widened scratch and use the normal trusted filter.
func FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(dst []byte, dstStride int, input []byte, inputOrigin int, unitWidth int, unitHeight int, blocks []BlockPosition, params FrameFilterParams) bool {
	if !u8SecondaryOnlyByteInteriorAvailable() || len(blocks) == 0 ||
		params.Level != 0 || params.CoeffShift != 0 ||
		(params.SecondaryStrength != 1 && params.SecondaryStrength != 2 && params.SecondaryStrength != 4) ||
		params.XDec > 1 || params.YDec > 1 || params.Plane > PlaneV {
		return false
	}
	blockWidth := 8 >> params.XDec
	blockHeight := 8 >> params.YDec
	if (blockWidth != 4 && blockWidth != 8) || (blockHeight != 4 && blockHeight != 8) ||
		unitWidth <= 0 || unitHeight <= 0 || unitWidth%blockWidth != 0 || unitHeight%blockHeight != 0 ||
		unitWidth+4 > BStride || dstStride < unitWidth {
		return false
	}
	effectiveDamping := int(params.Damping)
	if params.Plane != PlaneY {
		effectiveDamping--
	}
	if effectiveDamping < 0 {
		return false
	}
	// The kernel reads exactly the unit plus a 2-pixel real-sample halo.
	// Requiring the complete rectangle here makes every later narrow load
	// provably in-bounds and prevents a partial mutation before fallback.
	if inputOrigin < 2*BStride+2 || inputOrigin+(unitHeight+1)*BStride+unitWidth+2 > len(input) {
		return false
	}
	if (unitHeight-1)*dstStride+unitWidth > len(dst) {
		return false
	}
	for _, block := range blocks {
		bx := int(block.BX)
		by := int(block.BY)
		if (bx+1)*blockWidth > unitWidth || (by+1)*blockHeight > unitHeight {
			return false
		}
	}
	shift := constrainShift(int(params.SecondaryStrength), effectiveDamping)
	if shift > 127 {
		return false
	}
	return filterFrameBlocksU8SecondaryOnlyByteInterior(dst, dstStride, input, inputOrigin, blocks, blockWidth, blockHeight, int(params.SecondaryStrength), shift)
}
