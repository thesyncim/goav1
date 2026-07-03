// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package dsp

// AVX2-accelerated AddResidualPlaneBlock inner loop. The .s file processes the
// leading width&^7 columns of every row eight lanes at a time; this wrapper
// runs the width&7 tail in pure Go, identical to addResidualPlaneBlockPureGo.
//
// The kernel is bit-exact with the reference: prediction and residual are
// widened to s32 (VPMOVZXBD/VPMOVZXWD for the predicted samples, VPMOVSXWD for
// the signed residual), added (VPADDD), clamped to [0, max] with
// VPMAXSD(0)/VPMINSD(max), then narrowed back (VPACKUSDW, plus VPACKUSWB for
// the 8-bit shape). AddResidualPlaneBlock has already validated every pointer
// and extent, so the wrapper only slices what the reference would slice.

//go:noescape
func addResidual8AVX2Asm(dst *byte, dstStride uintptr, res *int16, resStride uintptr, max uint32, groups uintptr, height uintptr)

//go:noescape
func addResidual16AVX2Asm(dst *byte, dstStride uintptr, res *int16, resStride uintptr, max uint32, groups uintptr, height uintptr)

//go:noescape
func addRawTransform8AVX2Asm(dst *byte, dstStride uintptr, raw *int32, rawStride uintptr, max uint32, groups uintptr, height uintptr)

//go:noescape
func addRawTransform16AVX2Asm(dst *byte, dstStride uintptr, raw *int32, rawStride uintptr, max uint32, groups uintptr, height uintptr)

func addResidualPlaneBlockAVX2(block planeBlock, bytesPerSample int, max uint16, width int, residual []int16, residualStride int) {
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
		addResidual8AVX2Asm(
			&block.pix[0], uintptr(block.stride),
			&residual[0], uintptr(residualStride*2),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		for row := 0; row < block.height; row++ {
			base := row * block.stride
			resBase := row * residualStride
			for col := vecCols; col < width; col++ {
				v := int(block.pix[base+col]) + int(residual[resBase+col])
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				block.pix[base+col] = byte(v)
			}
		}
	case 2:
		addResidual16AVX2Asm(
			&block.pix[0], uintptr(block.stride),
			&residual[0], uintptr(residualStride*2),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		for row := 0; row < block.height; row++ {
			base := row * block.stride
			resBase := row * residualStride
			for col := vecCols; col < width; col++ {
				off := base + col*2
				v := int(uint16(block.pix[off])|uint16(block.pix[off+1])<<8) + int(residual[resBase+col])
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				block.pix[off] = byte(v)
				block.pix[off+1] = byte(v >> 8)
			}
		}
	}
}

// addRawTransformPlaneBlockAVX2 fuses the inverse-transform column-pass output
// into an already-predicted block. The .s file processes the leading width&^7
// columns of every row eight lanes at a time; this wrapper runs the width&7
// tail in pure Go, identical to addRawTransformPlaneBlockPureGo. It is bit-exact
// with the reference (see the .s header for the round/saturate/add/clamp
// sequence). AddRawTransformPlaneBlockTrusted has already validated the block
// window, sample width, and raw extent.
func addRawTransformPlaneBlockAVX2(block planeBlock, bytesPerSample int, max uint16, width int, raw []int32, rawStride int) {
	groups := width >> 3
	if groups == 0 {
		// Narrow blocks (width < 8) have no vector portion.
		addRawTransformPlaneBlockPureGo(block, bytesPerSample, max, width, raw, rawStride)
		return
	}
	vecCols := groups << 3
	maxInt := int(max)

	switch bytesPerSample {
	case 1:
		addRawTransform8AVX2Asm(
			&block.pix[0], uintptr(block.stride),
			&raw[0], uintptr(rawStride*4),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		for row := 0; row < block.height; row++ {
			base := row * block.stride
			rawBase := row * rawStride
			for col := vecCols; col < width; col++ {
				v := int(block.pix[base+col]) + rawTransformResidual(raw[rawBase+col])
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				block.pix[base+col] = byte(v)
			}
		}
	case 2:
		addRawTransform16AVX2Asm(
			&block.pix[0], uintptr(block.stride),
			&raw[0], uintptr(rawStride*4),
			uint32(max), uintptr(groups), uintptr(block.height),
		)
		if vecCols == width {
			return
		}
		for row := 0; row < block.height; row++ {
			base := row * block.stride
			rawBase := row * rawStride
			for col := vecCols; col < width; col++ {
				off := base + col*2
				v := int(uint16(block.pix[off])|uint16(block.pix[off+1])<<8) + rawTransformResidual(raw[rawBase+col])
				if v < 0 {
					v = 0
				} else if v > maxInt {
					v = maxInt
				}
				block.pix[off] = byte(v)
				block.pix[off+1] = byte(v >> 8)
			}
		}
	}
}
