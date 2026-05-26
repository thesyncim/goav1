// Ported from libaom: av1/common/cdef.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package cdef

// DecodeStrength splits an AV1 packed CDEF strength into primary level and
// secondary strength. The syntax packs a 4-bit primary level with a 2-bit
// secondary strength; secondary value 3 maps to 4.
func DecodeStrength(packed uint8) (level int, secondary int, err error) {
	if packed > 0x3f {
		return 0, 0, ErrInvalidCDEF
	}
	level = int(packed >> 2)
	secondary = int(packed & 3)
	if secondary == 3 {
		secondary = 4
	}
	return level, secondary, nil
}

// FrameFilterParamsFromStrength builds FilterFrameBlocks parameters from one
// packed frame-header CDEF strength entry.
func FrameFilterParamsFromStrength(plane Plane, xDec int, yDec int, packed uint8, damping int, coeffShift int) (FrameFilterParams, error) {
	level, secondary, err := DecodeStrength(packed)
	if err != nil {
		return FrameFilterParams{}, err
	}
	if plane > PlaneV || xDec < 0 || xDec > 1 || yDec < 0 || yDec > 1 ||
		damping < 0 || coeffShift < 0 || coeffShift > 4 {
		return FrameFilterParams{}, ErrInvalidCDEF
	}
	return FrameFilterParams{
		XDec:              xDec,
		YDec:              yDec,
		Plane:             plane,
		Level:             level,
		SecondaryStrength: secondary,
		Damping:           damping,
		CoeffShift:        coeffShift,
	}, nil
}
