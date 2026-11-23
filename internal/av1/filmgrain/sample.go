// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

func lumaGrainSample(grain []int16, offset uint8, blockCol int, blockRow int, x int, y int) int16 {
	return grain[lumaGrainSampleIndex(offset, blockCol, blockRow, x, y)]
}

// lumaGrainSampleIndex returns the flat grain-template index for the sample
// lumaGrainSample would read. For consecutive x (fixed y, block) the index is
// contiguous, which lets the apply kernel consume grain[idx:idx+n] directly.
func lumaGrainSampleIndex(offset uint8, blockCol int, blockRow int, x int, y int) int {
	col := lumaGrainOffset(int(offset>>4)) + x + LumaBlockSize*blockCol
	row := lumaGrainOffset(int(offset&0x0f)) + y + LumaBlockSize*blockRow
	return row*LumaGrainWidth + col
}

func chromaGrainSample(grain []int16, offset uint8, shiftX int, shiftY int, blockCol int, blockRow int, x int, y int) int16 {
	return grain[chromaGrainSampleIndex(offset, shiftX, shiftY, blockCol, blockRow, x, y)]
}

// chromaGrainSampleIndex returns the flat grain-template index for the sample
// chromaGrainSample would read. For consecutive x the index is contiguous.
func chromaGrainSampleIndex(offset uint8, shiftX int, shiftY int, blockCol int, blockRow int, x int, y int) int {
	blockWidth := LumaBlockSize >> shiftX
	blockHeight := LumaBlockSize >> shiftY
	col := chromaGrainOffset(int(offset>>4), shiftX) + x + blockWidth*blockCol
	row := chromaGrainOffset(int(offset&0x0f), shiftY) + y + blockHeight*blockRow
	return row*ChromaGrainWidth + col
}

// lumaClipBounds returns the [min,max] sample clamp used by applyLumaSample,
// matching its restricted-range handling exactly.
func lumaClipBounds(params LumaRowParams) (int, int) {
	minValue := 0
	maxValue := (256 << (params.BitDepth - 8)) - 1
	if params.ClipToRestrictedRange {
		minValue = LumaLegalMin << (params.BitDepth - 8)
		maxValue = LumaLegalMax << (params.BitDepth - 8)
	}
	return minValue, maxValue
}

func lumaGrainOffset(n int) int {
	return 3 + LumaOverlapSamples*(3+n)
}

// chromaClipBounds returns the [min,max] sample clamp used by
// applyChromaSample, matching its restricted-range handling exactly.
func chromaClipBounds(params ChromaRowParams) (int, int) {
	minValue := 0
	maxValue := (256 << (params.BitDepth - 8)) - 1
	if params.ClipToRestrictedRange {
		minValue = LumaLegalMin << (params.BitDepth - 8)
		maxLegal := ChromaLegalMax
		if params.MatrixIdentity {
			maxLegal = ChromaIdentityMax
		}
		maxValue = maxLegal << (params.BitDepth - 8)
	}
	return minValue, maxValue
}

func chromaGrainOffset(n int, shift int) int {
	return 3 + (LumaOverlapSamples>>shift)*(3+n)
}

func chromaSubsamplingShift(subsampled bool) int {
	if subsampled {
		return 1
	}
	return 0
}
