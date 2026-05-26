// Ported from libaom: av1/common/reconintra.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package prediction

import "github.com/thesyncim/goav1/internal/av1/frame"

// IntraMode identifies the prediction modes implemented by PredictIntraPlaneBlock.
type IntraMode uint8

const (
	IntraModeDC IntraMode = iota
	IntraModeVertical
	IntraModeHorizontal
	IntraModePaeth
	IntraModeSmooth
	IntraModeSmoothVertical
	IntraModeSmoothHorizontal
)

// IntraEdges carries caller-owned neighboring samples for intra prediction.
// Above must contain at least block width samples when AboveAvailable is true;
// Left must contain at least block height samples when LeftAvailable is true.
type IntraEdges struct {
	Above []uint16
	Left  []uint16

	AboveLeft uint16

	AboveAvailable     bool
	LeftAvailable      bool
	AboveLeftAvailable bool
}

// PredictIntraPlaneBlock writes an intra-predicted rectangular block into dst.
// High-bit-depth samples are stored little-endian in two bytes. Edge storage is
// supplied by the caller so reconstruction hot paths can reuse buffers.
func PredictIntraPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, mode IntraMode, edges IntraEdges) error {
	return PredictIntraPlaneBlockWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, width, height, mode, edges)
}

// PredictIntraPlaneBlockWithExtent writes a visible sub-rectangle of an AV1
// intra predictor whose coded transform extent may continue past the frame
// boundary. predWidth/predHeight select the libaom smooth/DC weights and edge
// lengths; width/height select the pixels addressable in dst.
func PredictIntraPlaneBlockWithExtent(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, predWidth int, predHeight int, mode IntraMode, edges IntraEdges) error {
	if predWidth < width || predHeight < height {
		return ErrInvalidPrediction
	}
	block, err := planeBlockWindow(dst, bytesPerSample, x, y, width, height)
	if err != nil {
		return err
	}
	max, err := sampleMax(bytesPerSample, bitDepth)
	if err != nil {
		return err
	}

	switch mode {
	case IntraModeDC:
		value, err := dcPrediction(predWidth, predHeight, max, edges)
		if err != nil {
			return err
		}
		fillBlock(block, bytesPerSample, value)
	case IntraModeVertical:
		if !edges.AboveAvailable || len(edges.Above) < predWidth {
			return ErrInvalidPrediction
		}
		if err := validateSamples(edges.Above[:predWidth], max); err != nil {
			return err
		}
		predictVertical(block, bytesPerSample, edges.Above[:width])
	case IntraModeHorizontal:
		if !edges.LeftAvailable || len(edges.Left) < predHeight {
			return ErrInvalidPrediction
		}
		if err := validateSamples(edges.Left[:predHeight], max); err != nil {
			return err
		}
		predictHorizontal(block, bytesPerSample, edges.Left[:height])
	case IntraModePaeth:
		if err := validateIntraDirectionalEdges(predWidth, predHeight, max, edges, true); err != nil {
			return err
		}
		predictPaeth(block, bytesPerSample, edges.Above[:width], edges.Left[:height], edges.AboveLeft)
	case IntraModeSmooth:
		if err := validateIntraDirectionalEdges(predWidth, predHeight, max, edges, false); err != nil {
			return err
		}
		if err := predictSmoothWithExtent(block, bytesPerSample, predWidth, predHeight, edges.Above[:predWidth], edges.Left[:predHeight]); err != nil {
			return err
		}
	case IntraModeSmoothVertical:
		if err := validateIntraDirectionalEdges(predWidth, predHeight, max, edges, false); err != nil {
			return err
		}
		if err := predictSmoothVerticalWithExtent(block, bytesPerSample, predHeight, edges.Above[:width], edges.Left[:predHeight]); err != nil {
			return err
		}
	case IntraModeSmoothHorizontal:
		if err := validateIntraDirectionalEdges(predWidth, predHeight, max, edges, false); err != nil {
			return err
		}
		if err := predictSmoothHorizontalWithExtent(block, bytesPerSample, predWidth, edges.Above[:predWidth], edges.Left[:height]); err != nil {
			return err
		}
	default:
		return ErrInvalidPrediction
	}
	return nil
}

func dcPrediction(width int, height int, max uint16, edges IntraEdges) (uint16, error) {
	sum := 0
	count := 0
	if edges.AboveAvailable {
		if len(edges.Above) < width {
			return 0, ErrInvalidPrediction
		}
		for _, sample := range edges.Above[:width] {
			if sample > max {
				return 0, ErrInvalidPrediction
			}
			sum += int(sample)
		}
		count += width
	}
	if edges.LeftAvailable {
		if len(edges.Left) < height {
			return 0, ErrInvalidPrediction
		}
		for _, sample := range edges.Left[:height] {
			if sample > max {
				return 0, ErrInvalidPrediction
			}
			sum += int(sample)
		}
		count += height
	}
	if count == 0 {
		return uint16((int(max) + 1) >> 1), nil
	}
	if edges.AboveAvailable && edges.LeftAvailable {
		if value, ok := dcRectPredictionValue(width, height, max, sum); ok {
			return value, nil
		}
	}
	return uint16((sum + (count >> 1)) / count), nil
}

type planeBlock struct {
	pix      []byte
	stride   int
	width    int
	height   int
	rowBytes int
}

func planeBlockWindow(plane frame.Plane, bytesPerSample int, x int, y int, width int, height int) (planeBlock, error) {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return planeBlock{}, ErrInvalidPrediction
	}
	if plane.Stride <= 0 ||
		plane.Width <= 0 ||
		plane.Height <= 0 ||
		x < 0 ||
		y < 0 ||
		width <= 0 ||
		height <= 0 ||
		x > plane.Width-width ||
		y > plane.Height-height {
		return planeBlock{}, ErrInvalidPrediction
	}
	rowBytes, ok := checkedMul(width, bytesPerSample)
	if !ok || rowBytes <= 0 || rowBytes > plane.Stride {
		return planeBlock{}, ErrInvalidPrediction
	}
	rowOffset, ok := checkedMul(y, plane.Stride)
	if !ok {
		return planeBlock{}, ErrInvalidPrediction
	}
	colOffset, ok := checkedMul(x, bytesPerSample)
	if !ok {
		return planeBlock{}, ErrInvalidPrediction
	}
	offset, ok := checkedAdd(rowOffset, colOffset)
	if !ok {
		return planeBlock{}, ErrInvalidPrediction
	}
	lastRowOffset, ok := checkedMul(height-1, plane.Stride)
	if !ok {
		return planeBlock{}, ErrInvalidPrediction
	}
	windowLen, ok := checkedAdd(lastRowOffset, rowBytes)
	if !ok {
		return planeBlock{}, ErrInvalidPrediction
	}
	end, ok := checkedAdd(offset, windowLen)
	if !ok || offset < 0 || end > len(plane.Pix) {
		return planeBlock{}, ErrInvalidPrediction
	}
	return planeBlock{
		pix:      plane.Pix[offset:end],
		stride:   plane.Stride,
		width:    width,
		height:   height,
		rowBytes: rowBytes,
	}, nil
}

func fillBlock(block planeBlock, bytesPerSample int, value uint16) {
	switch bytesPerSample {
	case 1:
		v := byte(value)
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			for i := 0; i < len(line); i++ {
				line[i] = v
			}
		}
	case 2:
		lo := byte(value)
		hi := byte(value >> 8)
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			for i := 0; i < len(line); i += 2 {
				line[i] = lo
				line[i+1] = hi
			}
		}
	}
}

func predictVertical(block planeBlock, bytesPerSample int, above []uint16) {
	switch bytesPerSample {
	case 1:
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			for col := 0; col < block.width; col++ {
				line[col] = byte(above[col])
			}
		}
	case 2:
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			for col := 0; col < block.width; col++ {
				i := col * 2
				sample := above[col]
				line[i] = byte(sample)
				line[i+1] = byte(sample >> 8)
			}
		}
	}
}

func predictHorizontal(block planeBlock, bytesPerSample int, left []uint16) {
	switch bytesPerSample {
	case 1:
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			sample := byte(left[row])
			for col := 0; col < block.width; col++ {
				line[col] = sample
			}
		}
	case 2:
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			sample := left[row]
			lo := byte(sample)
			hi := byte(sample >> 8)
			for col := 0; col < block.width; col++ {
				i := col * 2
				line[i] = lo
				line[i+1] = hi
			}
		}
	}
}

func validateIntraDirectionalEdges(width int, height int, max uint16, edges IntraEdges, requireAboveLeft bool) error {
	if !edges.AboveAvailable || len(edges.Above) < width || !edges.LeftAvailable || len(edges.Left) < height {
		return ErrInvalidPrediction
	}
	if requireAboveLeft {
		if !edges.AboveLeftAvailable || edges.AboveLeft > max {
			return ErrInvalidPrediction
		}
	}
	if err := validateSamples(edges.Above[:width], max); err != nil {
		return err
	}
	return validateSamples(edges.Left[:height], max)
}

func dcRectPredictionValue(width int, height int, max uint16, sum int) (uint16, bool) {
	small := width
	large := height
	if small > large {
		small, large = large, small
	}
	if small <= 0 || (large != 2*small && large != 4*small) {
		return 0, false
	}
	shift1, ok := log2PowerOfTwoInt(small)
	if !ok {
		return 0, false
	}
	multiplier := 0x5556
	shift2 := 16
	if large == 4*small {
		multiplier = 0x3334
	}
	if max > 0xff {
		shift2 = 17
		if large == 2*small {
			multiplier = 0xaaab
		} else {
			multiplier = 0x6667
		}
	}
	value := divideUsingMultiplyShift(sum+((width+height)>>1), shift1, multiplier, shift2)
	return uint16(value), true
}

func divideUsingMultiplyShift(num int, shift1 int, multiplier int, shift2 int) int {
	intermediate := num >> shift1
	return intermediate * multiplier >> shift2
}

func log2PowerOfTwoInt(v int) (int, bool) {
	if v <= 0 || v&(v-1) != 0 {
		return 0, false
	}
	log := 0
	for v > 1 {
		v >>= 1
		log++
	}
	return log, true
}

func validateSamples(samples []uint16, max uint16) error {
	for _, sample := range samples {
		if sample > max {
			return ErrInvalidPrediction
		}
	}
	return nil
}

func sampleMax(bytesPerSample int, bitDepth uint8) (uint16, error) {
	switch {
	case bytesPerSample == 1 && bitDepth == 8:
		return 0xff, nil
	case bytesPerSample == 2 && (bitDepth == 10 || bitDepth == 12):
		return uint16((1 << bitDepth) - 1), nil
	default:
		return 0, ErrInvalidPrediction
	}
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
