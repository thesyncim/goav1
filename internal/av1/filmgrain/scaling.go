// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

const ScalingLUTSize = 256

// ScalingPoint is one film grain scaling point in the 8-bit lookup domain.
type ScalingPoint struct {
	Value   uint8
	Scaling uint8
}

// BuildScalingLUT constructs AV1's 256-entry film grain scaling lookup table.
func BuildScalingLUT(dst []uint8, points []ScalingPoint) error {
	if len(dst) < ScalingLUTSize {
		return ErrInvalidParams
	}
	dst = dst[:ScalingLUTSize]
	for i := range dst {
		dst[i] = 0
	}
	if len(points) == 0 {
		return nil
	}
	if err := validateScalingPoints(points); err != nil {
		return err
	}

	first := points[0]
	for x := 0; x < int(first.Value); x++ {
		dst[x] = first.Scaling
	}
	for point := 0; point < len(points)-1; point++ {
		left := points[point]
		right := points[point+1]
		deltaY := int(right.Scaling) - int(left.Scaling)
		deltaX := int(right.Value) - int(left.Value)
		delta := int64(deltaY) * int64((65536+(deltaX>>1))/deltaX)
		for x := range deltaX {
			v := int(left.Scaling) + int((int64(x)*delta+32768)>>16)
			dst[int(left.Value)+x] = uint8(v)
		}
	}
	last := points[len(points)-1]
	for x := int(last.Value); x < ScalingLUTSize; x++ {
		dst[x] = last.Scaling
	}
	return nil
}

// ScaleLUT returns the scaling value for an image sample at bitDepth.
func ScaleLUT(lut []uint8, index int, bitDepth uint8) (uint8, error) {
	if len(lut) < ScalingLUTSize || bitDepth < 8 || bitDepth > 12 ||
		index < 0 || index >= 1<<bitDepth {
		return 0, ErrInvalidParams
	}
	return scaleLUT(lut, index, bitDepth), nil
}

func scaleLUT(lut []uint8, index int, bitDepth uint8) uint8 {
	// Defense in depth: clip the input sample to the legal bitdepth range
	// before deriving the LUT index. Internal callers receive uint16 samples
	// from decoded planes; a malformed bitstream that left a value above
	// (1<<bitDepth)-1 in a plane would otherwise drive lut[x] out of range.
	maxIndex := (1 << bitDepth) - 1
	if index < 0 {
		index = 0
	} else if index > maxIndex {
		index = maxIndex
	}
	shift := int(bitDepth - 8)
	x := index >> shift
	if bitDepth == 8 || x == ScalingLUTSize-1 {
		return lut[x]
	}
	rem := index - (x << shift)
	start := int(lut[x])
	end := int(lut[x+1])
	return uint8(start + roundPowerOfTwo((end-start)*rem, shift))
}

func validateScalingPoints(points []ScalingPoint) error {
	previous := int(-1)
	for _, point := range points {
		if int(point.Value) <= previous {
			return ErrInvalidParams
		}
		previous = int(point.Value)
	}
	return nil
}

func roundPowerOfTwo(value int, bits int) int {
	if bits <= 0 {
		return value
	}
	return (value + (1 << (bits - 1))) >> bits
}
