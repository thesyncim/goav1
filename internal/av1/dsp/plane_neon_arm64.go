// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

// NEON-accelerated AddResidualPlaneBlock inner loop. The .s file processes the
// leading width&^7 columns of every row eight lanes at a time; this wrapper
// runs the width&7 tail in pure Go, identical to addResidualPlaneBlockPureGo.
//
// The kernel is bit-exact with the reference: prediction and residual are
// widened to s32, added, clamped to [0, max] with smax(0)/smin(max), and
// narrowed back. AddResidualPlaneBlock has already validated every pointer and
// extent, so the wrapper only slices what the reference would slice.

//go:noescape
func addResidual8NEONAsm(dst *byte, dstStride uintptr, res *int16, resStride uintptr, max uint32, groups uintptr, height uintptr)

//go:noescape
func addResidual16NEONAsm(dst *byte, dstStride uintptr, res *int16, resStride uintptr, max uint32, groups uintptr, height uintptr)

//go:noescape
func addRawTransform8NEONAsm(dst *byte, dstStride uintptr, raw *int32, rawStride uintptr, max uint32, groups uintptr, height uintptr)

//go:noescape
func addRawTransform16NEONAsm(dst *byte, dstStride uintptr, raw *int32, rawStride uintptr, max uint32, groups uintptr, height uintptr)

func addResidualPlaneBlockNEON(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
	groups := width >> 3
	if groups == 0 {
		// Narrow blocks (width < 8) have no vector portion; the reference is
		// already cheap here.
		addResidualPlaneBlockPureGo(block, bytesPerSample, max, width, residual, residualStride)
		return
	}
	vecCols := groups << 3
	maxInt := int(max)

	switch bytesPerSample {
	case 1:
		addResidual8NEONAsm(
			&block.pix[0], uintptr(block.stride),
			&residual[0], uintptr(residualStride*2),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		tail := width - vecCols
		for row := 0; row < block.height; row++ {
			base := row*block.stride + vecCols
			resBase := row*residualStride + vecCols
			line := block.pix[base : base+tail : base+tail]
			resLine := residual[resBase : resBase+tail : resBase+tail]
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
		addResidual16NEONAsm(
			&block.pix[0], uintptr(block.stride),
			&residual[0], uintptr(residualStride*2),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		tail := width - vecCols
		for row := 0; row < block.height; row++ {
			base := row*block.stride + vecCols*2
			resBase := row*residualStride + vecCols
			line := block.pix[base : base+2*tail : base+2*tail]
			resLine := residual[resBase : resBase+tail : resBase+tail]
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

func addRawTransformPlaneBlockNEON(block planeBlock, bytesPerSample int, max uint16, width int, raw []int32, rawStride int) {
	groups := width >> 3
	if groups == 0 {
		addRawTransformPlaneBlockPureGo(block, bytesPerSample, max, width, raw, rawStride)
		return
	}
	vecCols := groups << 3
	maxInt := int(max)

	switch bytesPerSample {
	case 1:
		addRawTransform8NEONAsm(
			&block.pix[0], uintptr(block.stride),
			&raw[0], uintptr(rawStride*4),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		tail := width - vecCols
		for row := 0; row < block.height; row++ {
			base := row*block.stride + vecCols
			rawBase := row*rawStride + vecCols
			line := block.pix[base : base+tail : base+tail]
			rawLine := raw[rawBase : rawBase+tail : rawBase+tail]
			for col, sample := range line {
				v := int(sample) + rawTransformResidual(rawLine[col])
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				line[col] = byte(v)
			}
		}
	case 2:
		addRawTransform16NEONAsm(
			&block.pix[0], uintptr(block.stride),
			&raw[0], uintptr(rawStride*4),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		tail := width - vecCols
		for row := 0; row < block.height; row++ {
			base := row*block.stride + vecCols*2
			rawBase := row*rawStride + vecCols
			line := block.pix[base : base+2*tail : base+2*tail]
			rawLine := raw[rawBase : rawBase+tail : rawBase+tail]
			for col, rv := range rawLine {
				i := col * 2
				v := int(uint16(line[i])|uint16(line[i+1])<<8) + rawTransformResidual(rv)
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				line[i] = byte(v)
				line[i+1] = byte(v >> 8)
			}
		}
	}
}
