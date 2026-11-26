// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package dsp

import (
	"encoding/binary"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// CopyPlaneBlock copies a rectangular plane block from src to dst. The source
// and destination use the same sample width.
//
// Accesses stay inside the width x height block of each plane or buffer.
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

// addResidualPlaneBlockImpl is the dispatch slot for the
// AddResidualPlaneBlockTrusted inner loop. It is resolved exactly once, at package init (see
// plane_dispatch_*.go), so the per-call cost is a single indirect call with no
// feature-detection branches. Its caller has validated every input, so it never
// has to re-check bounds.
var addResidualPlaneBlockImpl = addResidualPlaneBlockPureGo
var addRawTransformPlaneBlockImpl = addRawTransformPlaneBlockPureGo

// addConstantPlaneBlock8Impl adds one DC-only residual to an 8-bit block,
// clamping to [0, 255]; bound like the slots above.
var addConstantPlaneBlock8Impl = addConstantPlaneBlock8PureGo

// addConstantPlaneBlock16Impl is the 16bpc (2-byte sample) sibling, clamping
// to [0, max]; bound like the slots above.
var addConstantPlaneBlock16Impl = addConstantPlaneBlock16PureGo

func addConstantPlaneBlock16PureGo(dst []byte, dstStride int, width int, height int, max uint16, residual int16) {
	maxInt := int(max)
	r := int(residual)
	rowBytes := width * 2
	for row := 0; row < height; row++ {
		line := dst[row*dstStride : row*dstStride+rowBytes : row*dstStride+rowBytes]
		for col := 0; col < rowBytes; col += 2 {
			v := int(uint16(line[col])|uint16(line[col+1])<<8) + r
			if v < 0 {
				v = 0
			} else if v > maxInt {
				v = maxInt
			}
			line[col] = byte(v)
			line[col+1] = byte(v >> 8)
		}
	}
}

func addConstantPlaneBlock8PureGo(dst []byte, dstStride int, width int, height int, residual int16) {
	r := int(residual)
	for row := 0; row < height; row++ {
		line := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
		for col, sample := range line {
			line[col] = byte(min(max(int(sample)+r, 0), 0xff))
		}
	}
}

// AddResidualPlaneBlockTrusted adds a residual block to a plane after the
// caller has already validated the destination window, sample width, bit-depth maximum,
// residual extent, and residual stride. dst must start at the block origin and
// contain every row touched by height/stride.
//
// Accesses stay inside the width x height block of each plane or buffer.
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

// AddConstantResidualPlaneBlockTrusted adds the same transform residual to
// every sample in an already-validated destination block.
//
// Accesses stay inside the width x height block of each plane or buffer.
func AddConstantResidualPlaneBlockTrusted(dst []byte, dstStride int, bytesPerSample int, max uint16, width int, height int, residual int16) {
	if bytesPerSample == 1 && max == 0xff && height > 0 {
		addConstantPlaneBlock8Impl(dst, dstStride, width, height, residual)
		return
	}
	switch bytesPerSample {
	case 1:
		maxInt := int(max)
		r := int(residual)
		for row := 0; row < height; row++ {
			line := dst[row*dstStride : row*dstStride+width : row*dstStride+width]
			for col, sample := range line {
				v := int(sample) + r
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				line[col] = byte(v)
			}
		}
	case 2:
		if height > 0 {
			addConstantPlaneBlock16Impl(dst, dstStride, width, height, max, residual)
		}
	}
}

// AddRawTransformPlaneBlockTrusted adds the inverse-transform column-pass
// output directly to an already-predicted destination block. raw contains
// int32 samples before AV1's final round-by-four and int16 saturation; this is
// the fused-add boundary used by dav1d/SVT-style reconstruction paths.
//
// Accesses stay inside the width x height block of each plane or buffer.
func AddRawTransformPlaneBlockTrusted(dst []byte, dstStride int, bytesPerSample int, max uint16, width int, height int, raw []int32, rawStride int) {
	block := planeBlock{
		pix:      dst,
		stride:   dstStride,
		width:    width,
		height:   height,
		rowBytes: width * bytesPerSample,
	}
	addRawTransformPlaneBlockImpl(block, bytesPerSample, max, width, raw, rawStride)
}

func addRawTransformPlaneBlockPureGo(block planeBlock, bytesPerSample int, max uint16, width int, raw []int32, rawStride int) {
	if width <= 0 || block.height <= 0 {
		return
	}
	switch bytesPerSample {
	case 1:
		maxInt := int(max)
		maxU := uint(maxInt)
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+width : row*block.stride+width]
			rawLine := raw[row*rawStride : row*rawStride+width : row*rawStride+width]
			col := 0
			for ; col+3 < width; col += 4 {
				v0 := int(line[col]) + int((int64(rawLine[col])+8)>>4)
				v1 := int(line[col+1]) + int((int64(rawLine[col+1])+8)>>4)
				v2 := int(line[col+2]) + int((int64(rawLine[col+2])+8)>>4)
				v3 := int(line[col+3]) + int((int64(rawLine[col+3])+8)>>4)
				if uint(v0) > maxU {
					if v0 < 0 {
						v0 = 0
					} else {
						v0 = maxInt
					}
				}
				if uint(v1) > maxU {
					if v1 < 0 {
						v1 = 0
					} else {
						v1 = maxInt
					}
				}
				if uint(v2) > maxU {
					if v2 < 0 {
						v2 = 0
					} else {
						v2 = maxInt
					}
				}
				if uint(v3) > maxU {
					if v3 < 0 {
						v3 = 0
					} else {
						v3 = maxInt
					}
				}
				line[col] = byte(v0)
				line[col+1] = byte(v1)
				line[col+2] = byte(v2)
				line[col+3] = byte(v3)
			}
			for ; col < width; col++ {
				v := int(line[col]) + int((int64(rawLine[col])+8)>>4)
				if uint(v) > maxU {
					if v < 0 {
						v = 0
					} else {
						v = maxInt
					}
				}
				line[col] = byte(v)
			}
		}
	case 2:
		maxInt := int(max)
		maxU := uint(maxInt)
		rowBytes := width * 2
		for row := 0; row < block.height; row++ {
			line := block.pix[row*block.stride : row*block.stride+rowBytes : row*block.stride+rowBytes]
			rawLine := raw[row*rawStride : row*rawStride+width : row*rawStride+width]
			col := 0
			for ; col+3 < width; col += 4 {
				i := col * 2
				v0 := int(binary.LittleEndian.Uint16(line[i:i+2])) + rawTransformResidual(rawLine[col])
				v1 := int(binary.LittleEndian.Uint16(line[i+2:i+4])) + rawTransformResidual(rawLine[col+1])
				v2 := int(binary.LittleEndian.Uint16(line[i+4:i+6])) + rawTransformResidual(rawLine[col+2])
				v3 := int(binary.LittleEndian.Uint16(line[i+6:i+8])) + rawTransformResidual(rawLine[col+3])
				if uint(v0|v1|v2|v3) > maxU {
					if v0 < 0 {
						v0 = 0
					} else if v0 > maxInt {
						v0 = maxInt
					}
					if v1 < 0 {
						v1 = 0
					} else if v1 > maxInt {
						v1 = maxInt
					}
					if v2 < 0 {
						v2 = 0
					} else if v2 > maxInt {
						v2 = maxInt
					}
					if v3 < 0 {
						v3 = 0
					} else if v3 > maxInt {
						v3 = maxInt
					}
				}
				binary.LittleEndian.PutUint16(line[i:i+2], uint16(v0))
				binary.LittleEndian.PutUint16(line[i+2:i+4], uint16(v1))
				binary.LittleEndian.PutUint16(line[i+4:i+6], uint16(v2))
				binary.LittleEndian.PutUint16(line[i+6:i+8], uint16(v3))
			}
			for ; col < width; col++ {
				i := col * 2
				v := int(binary.LittleEndian.Uint16(line[i:i+2])) + rawTransformResidual(rawLine[col])
				if uint(v) > maxU {
					if v < 0 {
						v = 0
					} else {
						v = maxInt
					}
				}
				binary.LittleEndian.PutUint16(line[i:i+2], uint16(v))
			}
		}
	}
}

func rawTransformResidual(v int32) int {
	r := int((int64(v) + 8) >> 4)
	if r < -32768 {
		return -32768
	}
	if r > 32767 {
		return 32767
	}
	return r
}

// addResidualPlaneBlockPureGo is the portable reference for the residual-add
// inner loop. Every SIMD variant the dispatcher selects MUST produce bit-exact
// output relative to this function. The caller has already validated the block
// window, the residual extent, and that bytesPerSample is 1 or 2.
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
