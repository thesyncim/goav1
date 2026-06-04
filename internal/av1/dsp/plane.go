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

// addResidualPlaneBlockImpl is the dispatch slot for the AddResidualPlaneBlock
// inner loop. It is resolved exactly once, at package init (see
// plane_dispatch_*.go), so the per-call cost is a single indirect call with no
// feature-detection branches. It is only invoked after AddResidualPlaneBlock
// has validated every input, so it never has to re-check bounds.
var addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo

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

	addResidualPlaneBlockImpl(block, bytesPerSample, max, width, residual, residualStride)
	return nil
}

// AddResidualPlaneBlockTrusted is AddResidualPlaneBlock after the caller has
// already validated the destination window, sample width, bit-depth maximum,
// residual extent, and residual stride. dst must start at the block origin and
// contain every row touched by height/stride.
func AddResidualPlaneBlockTrusted(dst []byte, dstStride int, bytesPerSample int, max uint16, width int, height int, residual []int16, residualStride int) {
	block := planeBlock{
		pix:      dst,
		stride:   dstStride,
		width:    width,
		height:   height,
		rowBytes: width * bytesPerSample,
	}
	addResidualPlaneBlockImpl(block, bytesPerSample, max, width, residual, residualStride)
}

// addResidualPlaneBlockPureGo is the portable reference for the residual-add
// inner loop. Every SIMD variant the dispatcher selects MUST produce bit-exact
// output relative to this function. AddResidualPlaneBlock has already validated
// the block window, the residual extent, and that bytesPerSample is 1 or 2.
func addResidualPlaneBlockPureGo(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	switch bytesPerSample {
	case 1:
		maxInt := int(max)
		if width == 4 {
			for row := 0; row < block.height; row++ {
				line := block.pix[row*block.stride : row*block.stride+4 : row*block.stride+4]
				resLine := residual[row*residualStride : row*residualStride+4 : row*residualStride+4]
				v0 := int(line[0]) + int(resLine[0])
				if v0 < 0 {
					v0 = 0
				} else if v0 > maxInt {
					v0 = maxInt
				}
				v1 := int(line[1]) + int(resLine[1])
				if v1 < 0 {
					v1 = 0
				} else if v1 > maxInt {
					v1 = maxInt
				}
				v2 := int(line[2]) + int(resLine[2])
				if v2 < 0 {
					v2 = 0
				} else if v2 > maxInt {
					v2 = maxInt
				}
				v3 := int(line[3]) + int(resLine[3])
				if v3 < 0 {
					v3 = 0
				} else if v3 > maxInt {
					v3 = maxInt
				}
				line[0] = byte(v0)
				line[1] = byte(v1)
				line[2] = byte(v2)
				line[3] = byte(v3)
			}
			return
		}
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+width : row*block.stride+width]
			resLine := residual[row*residualStride : row*residualStride+width : row*residualStride+width]
			line = line[:len(resLine)]
			for col, r := range resLine {
				v := int(line[col]) + int(r)
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				line[col] = byte(v)
			}
		}
	case 2:
		maxInt := int(max)
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+2*width : row*block.stride+2*width]
			resLine := residual[row*residualStride : row*residualStride+width : row*residualStride+width]
			line = line[:2*len(resLine)]
			for col, r := range resLine {
				pair := line[col*2 : col*2+2 : col*2+2]
				v := int(uint16(pair[0])|uint16(pair[1])<<8) + int(r)
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				pair[0] = byte(v)
				pair[1] = byte(v >> 8)
			}
		}
	}
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
