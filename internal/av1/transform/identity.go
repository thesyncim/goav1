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
// coefficients are dequantized transform coefficients in row-major order.
func InverseIdentityBlock(dst []int16, dstStride int, coeff []int32, coeffStride int, size Size) error {
	shift, ok := size.shift()
	if !ok ||
		!identityBlockSupported(size) ||
		dstStride < size.Width ||
		coeffStride < size.Width ||
		!blockFits(len(dst), dstStride, size.Width, size.Height) ||
		!blockFits(len(coeff), coeffStride, size.Width, size.Height) {
		return ErrInvalidTransform
	}

	for row := 0; row < size.Height; row++ {
		dstLine := dst[row*dstStride : row*dstStride+size.Width]
		coeffLine := coeff[row*coeffStride : row*coeffStride+size.Width]
		for col := 0; col < size.Width; col++ {
			v := coeffLine[col]
			if size.IsRect2() {
				v = rect2Scale(v)
			}
			v, _ = identity1DValue(v, size.Width)
			if shift > 0 {
				v = clipInt32(roundShift(int64(v), shift))
			}
			v, _ = identity1DValue(v, size.Height)
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
