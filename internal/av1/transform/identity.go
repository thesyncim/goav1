// Ported from libaom: av1/common/av1_inv_txfm2d.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// Size identifies an AV1 transform block shape in pixels.
type Size struct {
	Width  int
	Height int
}

// SizeFromDimensions returns the AV1 transform size with the given dimensions.
func SizeFromDimensions(width int, height int) (Size, error) {
	size := Size{Width: width, Height: height}
	if !size.Valid() {
		return Size{}, ErrInvalidTransform
	}
	return size, nil
}

// Valid reports whether s is one of the transform sizes supported by AV1.
func (s Size) Valid() bool {
	_, ok := s.shift()
	return ok
}

// IsRect2 reports whether s is a 1:2 or 2:1 rectangular transform. AV1 applies
// an extra scale before the first 1D pass for these shapes.
func (s Size) IsRect2() bool {
	return s.Width*2 == s.Height || s.Height*2 == s.Width
}

// Shift returns the AV1 inverse-transform post-column shift for s.
func (s Size) Shift() (int, error) {
	shift, ok := s.shift()
	if !ok {
		return 0, ErrInvalidTransform
	}
	return shift, nil
}

func (s Size) shift() (int, bool) {
	switch s {
	case Size{Width: 4, Height: 4},
		Size{Width: 4, Height: 8},
		Size{Width: 8, Height: 4}:
		return 0, true
	case Size{Width: 4, Height: 16},
		Size{Width: 8, Height: 8},
		Size{Width: 8, Height: 16},
		Size{Width: 16, Height: 4},
		Size{Width: 16, Height: 8},
		Size{Width: 16, Height: 32},
		Size{Width: 32, Height: 16},
		Size{Width: 32, Height: 64},
		Size{Width: 64, Height: 32}:
		return 1, true
	case Size{Width: 8, Height: 32},
		Size{Width: 16, Height: 16},
		Size{Width: 16, Height: 64},
		Size{Width: 32, Height: 8},
		Size{Width: 32, Height: 32},
		Size{Width: 64, Height: 16},
		Size{Width: 64, Height: 64}:
		return 2, true
	default:
		return 0, false
	}
}

func identityBlockSupported(size Size) bool {
	return size.Valid() && size.Width <= 32 && size.Height <= 32
}

// RoundShift applies AV1's signed rounded right shift.
func RoundShift(value int64, bits uint8) (int32, error) {
	if bits == 0 || bits > 62 {
		return 0, ErrInvalidTransform
	}
	bias := int64(1) << (bits - 1)
	if value > maxInt64-bias {
		return 0, ErrInvalidTransform
	}
	shifted := (value + bias) >> bits
	if shifted < minInt32 || shifted > maxInt32 {
		return 0, ErrInvalidTransform
	}
	return int32(shifted), nil
}

// RoundShiftArray applies AV1's round-shift-array primitive in place.
//
// Positive bit values round-shift right; negative values saturating-shift left.
func RoundShiftArray(values []int32, bit int) error {
	if bit > 62 || bit < -62 {
		return ErrInvalidTransform
	}
	if bit == 0 {
		return nil
	}
	if bit > 0 {
		for i, v := range values {
			values[i] = clipInt32(roundShift(int64(v), bit))
		}
		return nil
	}

	shift := -bit
	for i, v := range values {
		values[i] = saturatingLeftShiftInt32(v, shift)
	}
	return nil
}

func saturatingLeftShiftInt32(v int32, shift int) int32 {
	if v == 0 || shift == 0 {
		return v
	}
	if shift >= 31 {
		if v < 0 {
			return minInt32
		}
		return maxInt32
	}
	return clipInt32(int64(v) << uint(shift))
}

// InverseIdentity1DValue applies AV1's 1D identity inverse transform scale for
// a single coefficient.
func InverseIdentity1DValue(v int32, length int) (int32, error) {
	out, ok := identity1DValue(v, length)
	if !ok {
		return 0, ErrInvalidTransform
	}
	return out, nil
}

// InverseIdentityBlock writes an AV1 IDTX residual block to dst. The source
// coefficients are dequantized transform coefficients in AV1 coefficient order:
// coeff_idx = col * coeffStride + row. The 8-bit stage-range clamps are used.
func InverseIdentityBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, size Size) error {
	return inverseIdentityBlockClamped(dst, dstStride, coeff, coeffStride, size, minInt16, maxInt16, minInt16, maxInt16)
}

// inverseIdentityBlockClamped is the bit-depth-aware variant of
// InverseIdentityBlock. The clamps mirror clamp_buf() in
// av1/common/av1_inv_txfm2d.c, which the IDTX path also passes through (it
// shares inv_txfm2d_add_c with the other inverse transforms).
func inverseIdentityBlockClamped(dst []int16, dstStride int, coeff []int32, coeffStride int, size Size, rowMin int32, rowMax int32, colMin int32, colMax int32) error {
	shift, ok := size.shift()
	coeffSize := adjustedScanSize(size)
	if !ok ||
		!identityBlockSupported(size) ||
		dstStride < size.Width ||
		coeffStride < coeffSize.Height ||
		!blockFits(len(dst), dstStride, size.Width, size.Height) ||
		!coeffBlockFits(len(coeff), coeffStride, coeffSize.Width, coeffSize.Height) {
		return ErrInvalidTransform
	}

	width := size.Width
	height := size.Height
	rect2 := size.IsRect2()
	for row := 0; row < height; row++ {
		dstLine := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		for col := range dstLine {
			v := coeff[col*coeffStride+row]
			if rect2 {
				v = rect2Scale(v)
			}
			v = clipRange(int64(v), rowMin, rowMax)
			v, _ = identity1DValue(v, width)
			if shift > 0 {
				v = clipRange(roundShift(int64(v), shift), colMin, colMax)
			} else {
				v = clipRange(int64(v), colMin, colMax)
			}
			v, _ = identity1DValue(v, height)
			dstLine[col] = clipInt16(clipInt32(roundShift(int64(v), 4)))
		}
	}
	return nil
}

func identity1DValue(v int32, length int) (int32, bool) {
	x := int64(v)
	switch length {
	case 4:
		return clipInt32(x + ((x*1697 + 2048) >> 12)), true
	case 8:
		return clipInt32(x * 2), true
	case 16:
		return clipInt32(x*2 + ((x*1697 + 1024) >> 11)), true
	case 32:
		return clipInt32(x * 4), true
	default:
		return 0, false
	}
}

func rect2Scale(v int32) int32 {
	return clipInt32((int64(v)*181 + 128) >> 8)
}

func roundShift(v int64, bits int) int64 {
	return (v + (int64(1) << (bits - 1))) >> bits
}

func clipInt16(v int32) int16 {
	if v < minInt16 {
		return minInt16
	}
	if v > maxInt16 {
		return maxInt16
	}
	return int16(v)
}

func clipInt32(v int64) int32 {
	if v < minInt32 {
		return minInt32
	}
	if v > maxInt32 {
		return maxInt32
	}
	return int32(v)
}

func blockFits(length int, stride int, width int, height int) bool {
	if stride <= 0 || width <= 0 || height <= 0 {
		return false
	}
	lastRowOffset, ok := checkedMul(height-1, stride)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastRowOffset, width)
	return ok && needed <= length
}

func coeffBlockFits(length int, stride int, width int, height int) bool {
	if stride <= 0 || width <= 0 || height <= 0 {
		return false
	}
	lastColOffset, ok := checkedMul(width-1, stride)
	if !ok {
		return false
	}
	needed, ok := checkedAdd(lastColOffset, height)
	return ok && needed <= length
}

func checkedAdd(a int, b int) (int, bool) {
	c := a + b
	if c < a {
		return 0, false
	}
	return c, true
}

func checkedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

const (
	minInt16 = -1 << 15
	maxInt16 = 1<<15 - 1
	minInt32 = -1 << 31
	maxInt32 = 1<<31 - 1
	maxInt64 = 1<<63 - 1
)
