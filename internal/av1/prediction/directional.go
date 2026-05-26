// Ported from libaom: av1/common/reconintra.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package prediction

import "github.com/thesyncim/goav1/internal/av1/frame"

var drIntraDerivative = [90]int16{
	0, 0, 0,
	1023, 0, 0,
	547, 0, 0,
	372, 0, 0, 0, 0,
	273, 0, 0,
	215, 0, 0,
	178, 0, 0,
	151, 0, 0,
	132, 0, 0,
	116, 0, 0,
	102, 0, 0, 0,
	90, 0, 0,
	80, 0, 0,
	71, 0, 0,
	64, 0, 0,
	57, 0, 0,
	51, 0, 0,
	45, 0, 0, 0,
	40, 0, 0,
	35, 0, 0,
	31, 0, 0,
	27, 0, 0,
	23, 0, 0,
	19, 0, 0,
	15, 0, 0, 0, 0,
	11, 0, 0,
	7, 0, 0,
	3, 0, 0,
}

// DirectionalEdges carries neighboring samples for directional intra
// prediction. AboveOrigin and LeftOrigin identify libaom's above[0] and left[0]
// positions inside their slices so zone 2 can address the negative edge taps
// used by the C implementation.
type DirectionalEdges struct {
	Above []uint16
	Left  []uint16

	AboveOrigin int
	LeftOrigin  int

	UpsampleAbove bool
	UpsampleLeft  bool
}

// DirectionalDX matches libaom's av1_get_dx helper.
func DirectionalDX(angle int) int {
	if angle > 0 && angle < 90 {
		return int(drIntraDerivative[angle])
	}
	if angle > 90 && angle < 180 {
		return int(drIntraDerivative[180-angle])
	}
	return 1
}

// PredictDirectionalIntraPlaneBlock writes libaom-compatible directional intra
// prediction for angles in (0, 270). Angles 90 and 180 use the vertical and
// horizontal predictors; other angles use the AV1 directional zones.
func PredictDirectionalIntraPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, angle int, edges DirectionalEdges) error {
	block, err := planeBlockWindow(dst, bytesPerSample, x, y, width, height)
	if err != nil {
		return err
	}
	max, err := sampleMax(bytesPerSample, bitDepth)
	if err != nil {
		return err
	}
	if angle <= 0 || angle >= 270 {
		return ErrInvalidPrediction
	}

	switch {
	case angle == 90:
		if err := validateDirectionalRange(edges.Above, edges.AboveOrigin, 0, width-1, max); err != nil {
			return err
		}
		predictVertical(block, bytesPerSample, edges.Above[edges.AboveOrigin:edges.AboveOrigin+width])
	case angle == 180:
		if err := validateDirectionalRange(edges.Left, edges.LeftOrigin, 0, height-1, max); err != nil {
			return err
		}
		predictHorizontal(block, bytesPerSample, edges.Left[edges.LeftOrigin:edges.LeftOrigin+height])
	case angle < 90:
		dx := DirectionalDX(angle)
		if dx <= 0 {
			return ErrInvalidPrediction
		}
		return predictDirectionalZ1(block, bytesPerSample, edges, boolToInt(edges.UpsampleAbove), dx, max)
	case angle < 180:
		dx := DirectionalDX(angle)
		dy := DirectionalDY(angle)
		if dx <= 0 || dy <= 0 {
			return ErrInvalidPrediction
		}
		return predictDirectionalZ2(block, bytesPerSample, edges, boolToInt(edges.UpsampleAbove), boolToInt(edges.UpsampleLeft), dx, dy, max)
	default:
		dy := DirectionalDY(angle)
		if dy <= 0 {
			return ErrInvalidPrediction
		}
		return predictDirectionalZ3(block, bytesPerSample, edges, boolToInt(edges.UpsampleLeft), dy, max)
	}
	return nil
}

func predictDirectionalZ1(block planeBlock, bytesPerSample int, edges DirectionalEdges, upsampleAbove int, dx int, max uint16) error {
	maxBaseX := (block.width + block.height - 1) << upsampleAbove
	if err := validateDirectionalRange(edges.Above, edges.AboveOrigin, 0, maxBaseX, max); err != nil {
		return err
	}
	fracBits := 6 - upsampleAbove
	baseInc := 1 << upsampleAbove
	x := dx
	for row := 0; row < block.height; row++ {
		base := x >> fracBits
		shift := ((x << upsampleAbove) & 0x3f) >> 1
		if base >= maxBaseX {
			for ; row < block.height; row++ {
				for col := 0; col < block.width; col++ {
					setBlockSample(block, bytesPerSample, row, col, edges.Above[edges.AboveOrigin+maxBaseX])
				}
			}
			return nil
		}
		for col := 0; col < block.width; col++ {
			value := edges.Above[edges.AboveOrigin+maxBaseX]
			if base < maxBaseX {
				p0 := int(edges.Above[edges.AboveOrigin+base])
				p1 := int(edges.Above[edges.AboveOrigin+base+1])
				value = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
			}
			setBlockSample(block, bytesPerSample, row, col, value)
			base += baseInc
		}
		x += dx
	}
	return nil
}

func predictDirectionalZ2(block planeBlock, bytesPerSample int, edges DirectionalEdges, upsampleAbove int, upsampleLeft int, dx int, dy int, max uint16) error {
	aboveRange, leftRange, err := directionalZ2Ranges(block.width, block.height, upsampleAbove, upsampleLeft, dx, dy)
	if err != nil {
		return err
	}
	if err := aboveRange.validate(edges.Above, edges.AboveOrigin, max); err != nil {
		return err
	}
	if err := leftRange.validate(edges.Left, edges.LeftOrigin, max); err != nil {
		return err
	}

	fracBitsX := 6 - upsampleAbove
	fracBitsY := 6 - upsampleLeft
	for row := 0; row < block.height; row++ {
		for col := 0; col < block.width; col++ {
			y := row + 1
			x := (col << 6) - y*dx
			baseX := x >> fracBitsX
			var value uint16
			if baseX >= -(1 << upsampleAbove) {
				shift := ((x * (1 << upsampleAbove)) & 0x3f) >> 1
				p0 := int(edges.Above[edges.AboveOrigin+baseX])
				p1 := int(edges.Above[edges.AboveOrigin+baseX+1])
				value = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
			} else {
				x = col + 1
				y = (row << 6) - x*dy
				baseY := y >> fracBitsY
				shift := ((y * (1 << upsampleLeft)) & 0x3f) >> 1
				p0 := int(edges.Left[edges.LeftOrigin+baseY])
				p1 := int(edges.Left[edges.LeftOrigin+baseY+1])
				value = uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
			}
			setBlockSample(block, bytesPerSample, row, col, value)
		}
	}
	return nil
}

func predictDirectionalZ3(block planeBlock, bytesPerSample int, edges DirectionalEdges, upsampleLeft int, dy int, max uint16) error {
	maxBaseY := (block.width + block.height - 1) << upsampleLeft
	if err := validateDirectionalRange(edges.Left, edges.LeftOrigin, 0, maxBaseY, max); err != nil {
		return err
	}
	fracBits := 6 - upsampleLeft
	baseInc := 1 << upsampleLeft
	y := dy
	for col := 0; col < block.width; col++ {
		base := y >> fracBits
		shift := ((y << upsampleLeft) & 0x3f) >> 1
		for row := 0; row < block.height; row++ {
			if base < maxBaseY {
				p0 := int(edges.Left[edges.LeftOrigin+base])
				p1 := int(edges.Left[edges.LeftOrigin+base+1])
				value := uint16(roundPowerOfTwo(p0*(32-shift)+p1*shift, 5))
				setBlockSample(block, bytesPerSample, row, col, value)
			} else {
				for ; row < block.height; row++ {
					setBlockSample(block, bytesPerSample, row, col, edges.Left[edges.LeftOrigin+maxBaseY])
				}
				break
			}
			base += baseInc
		}
		y += dy
	}
	return nil
}

func directionalZ2Ranges(width int, height int, upsampleAbove int, upsampleLeft int, dx int, dy int) (sampleIndexRange, sampleIndexRange, error) {
	minBaseX := -(1 << upsampleAbove)
	minBaseY := -(1 << upsampleLeft)
	fracBitsX := 6 - upsampleAbove
	fracBitsY := 6 - upsampleLeft
	var aboveRange sampleIndexRange
	var leftRange sampleIndexRange
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			y := row + 1
			x := (col << 6) - y*dx
			baseX := x >> fracBitsX
			if baseX >= minBaseX {
				aboveRange.add(baseX)
				aboveRange.add(baseX + 1)
				continue
			}
			x = col + 1
			y = (row << 6) - x*dy
			baseY := y >> fracBitsY
			if baseY < minBaseY {
				return sampleIndexRange{}, sampleIndexRange{}, ErrInvalidPrediction
			}
			leftRange.add(baseY)
			leftRange.add(baseY + 1)
		}
	}
	return aboveRange, leftRange, nil
}

type sampleIndexRange struct {
	used bool
	min  int
	max  int
}

func (r *sampleIndexRange) add(index int) {
	if !r.used {
		r.used = true
		r.min = index
		r.max = index
		return
	}
	if index < r.min {
		r.min = index
	}
	if index > r.max {
		r.max = index
	}
}

func (r sampleIndexRange) validate(samples []uint16, origin int, max uint16) error {
	if !r.used {
		return nil
	}
	return validateDirectionalRange(samples, origin, r.min, r.max, max)
}

func validateDirectionalRange(samples []uint16, origin int, minIndex int, maxIndex int, max uint16) error {
	if minIndex > maxIndex || origin < 0 {
		return ErrInvalidPrediction
	}
	start := origin + minIndex
	end := origin + maxIndex + 1
	if start < 0 || end < start || end > len(samples) {
		return ErrInvalidPrediction
	}
	return validateSamples(samples[start:end], max)
}

func setBlockSample(block planeBlock, bytesPerSample int, row int, col int, value uint16) {
	offset := row*block.stride + col*bytesPerSample
	switch bytesPerSample {
	case 1:
		block.pix[offset] = byte(value)
	case 2:
		block.pix[offset] = byte(value)
		block.pix[offset+1] = byte(value >> 8)
	}
}

func roundPowerOfTwo(value int, bit int) int {
	return (value + (1 << (bit - 1))) >> bit
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

// DirectionalDY matches libaom's av1_get_dy helper.
func DirectionalDY(angle int) int {
	if angle > 90 && angle < 180 {
		return int(drIntraDerivative[angle-90])
	}
	if angle > 180 && angle < 270 {
		return int(drIntraDerivative[270-angle])
	}
	return 1
}
