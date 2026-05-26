// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package dsp

import "github.com/thesyncim/goav1/internal/av1/frame"

// FillPlaneBlock writes value to a rectangular plane block. High-bit-depth
// samples are stored little-endian in two bytes.
func FillPlaneBlock(dst frame.Plane, bytesPerSample int, x int, y int, width int, height int, value uint16) error {
	block, err := planeBlockWindow(dst, bytesPerSample, x, y, width, height)
	if err != nil {
		return err
	}
	if bytesPerSample == 1 && value > 0xff {
		return ErrInvalidBlock
	}
	switch bytesPerSample {
	case 1:
		v := byte(value)
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			for i := range line {
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
	default:
		return ErrInvalidBlock
	}
	return nil
}

// CopyPlaneBlock copies a rectangular plane block from src to dst. The source
// and destination use the same sample width.
func CopyPlaneBlock(dst frame.Plane, src frame.Plane, bytesPerSample int, dstX int, dstY int, srcX int, srcY int, width int, height int) error {
	dstBlock, err := planeBlockWindow(dst, bytesPerSample, dstX, dstY, width, height)
	if err != nil {
		return err
	}
	srcBlock, err := planeBlockWindow(src, bytesPerSample, srcX, srcY, width, height)
	if err != nil {
		return err
	}
	for row := 0; row < dstBlock.height; row++ {
		dstLine := dstBlock.pix[row*dstBlock.stride : row*dstBlock.stride+dstBlock.rowBytes]
		srcLine := srcBlock.pix[row*srcBlock.stride : row*srcBlock.stride+srcBlock.rowBytes]
		copy(dstLine, srcLine)
	}
	return nil
}

// AddResidualPlaneBlock adds signed transform residuals to an already-predicted
// destination block and clips to bitDepth.
func AddResidualPlaneBlock(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, residual []int16, residualStride int) error {
	block, err := planeBlockWindow(dst, bytesPerSample, x, y, width, height)
	if err != nil {
		return err
	}
	max, err := sampleMax(bytesPerSample, bitDepth)
	if err != nil {
		return err
	}
	if residualStride < width || residualStride <= 0 {
		return ErrInvalidBlock
	}
	lastRowOffset, ok := checkedMul(height-1, residualStride)
	if !ok {
		return ErrInvalidBlock
	}
	needed, ok := checkedAdd(lastRowOffset, width)
	if !ok || needed > len(residual) {
		return ErrInvalidBlock
	}

	switch bytesPerSample {
	case 1:
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			resLine := residual[row*residualStride : row*residualStride+width]
			for col := range width {
				line[col] = byte(clipSample(int(line[col])+int(resLine[col]), max))
			}
		}
	case 2:
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+block.rowBytes]
			resLine := residual[row*residualStride : row*residualStride+width]
			for col := range width {
				i := col * 2
				sample := int(uint16(line[i]) | uint16(line[i+1])<<8)
				out := clipSample(sample+int(resLine[col]), max)
				line[i] = byte(out)
				line[i+1] = byte(out >> 8)
			}
		}
	default:
		return ErrInvalidBlock
	}
	return nil
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
		return planeBlock{}, ErrInvalidBlock
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
		return planeBlock{}, ErrInvalidBlock
	}
	rowBytes, ok := checkedMul(width, bytesPerSample)
	if !ok || rowBytes <= 0 || rowBytes > plane.Stride {
		return planeBlock{}, ErrInvalidBlock
	}
	rowOffset, ok := checkedMul(y, plane.Stride)
	if !ok {
		return planeBlock{}, ErrInvalidBlock
	}
	colOffset, ok := checkedMul(x, bytesPerSample)
	if !ok {
		return planeBlock{}, ErrInvalidBlock
	}
	offset, ok := checkedAdd(rowOffset, colOffset)
	if !ok {
		return planeBlock{}, ErrInvalidBlock
	}
	lastRowOffset, ok := checkedMul(height-1, plane.Stride)
	if !ok {
		return planeBlock{}, ErrInvalidBlock
	}
	windowLen, ok := checkedAdd(lastRowOffset, rowBytes)
	if !ok {
		return planeBlock{}, ErrInvalidBlock
	}
	end, ok := checkedAdd(offset, windowLen)
	if !ok || offset < 0 || end > len(plane.Pix) {
		return planeBlock{}, ErrInvalidBlock
	}
	return planeBlock{
		pix:      plane.Pix[offset:end],
		stride:   plane.Stride,
		width:    width,
		height:   height,
		rowBytes: rowBytes,
	}, nil
}

func sampleMax(bytesPerSample int, bitDepth uint8) (uint16, error) {
	switch {
	case bytesPerSample == 1 && bitDepth == 8:
		return 0xff, nil
	case bytesPerSample == 2 && (bitDepth == 10 || bitDepth == 12):
		return uint16((1 << bitDepth) - 1), nil
	default:
		return 0, ErrInvalidBlock
	}
}

func clipSample(v int, max uint16) uint16 {
	if v < 0 {
		return 0
	}
	if v > int(max) {
		return max
	}
	return uint16(v)
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
