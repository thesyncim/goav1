// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

// LumaGrainSample returns one luma grain-template sample using AV1's luma
// block offset addressing. blockCol and blockRow select the current, left, top,
// or top-left block position used by overlap blending.
func LumaGrainSample(grain []int16, offset uint8, blockCol int, blockRow int, x int, y int) (int16, error) {
	if len(grain) < LumaGrainSamples ||
		blockCol < 0 || blockCol > 1 ||
		blockRow < 0 || blockRow > 1 ||
		x < 0 || x >= LumaBlockSize ||
		y < 0 || y >= LumaBlockSize {
		return 0, ErrInvalidParams
	}
	col := lumaGrainOffset(int(offset>>4)) + x + LumaBlockSize*blockCol
	row := lumaGrainOffset(int(offset&0x0f)) + y + LumaBlockSize*blockRow
	if col < 0 || col >= LumaGrainWidth || row < 0 || row >= LumaGrainHeight {
		return 0, ErrInvalidParams
	}
	return lumaGrainSample(grain, offset, blockCol, blockRow, x, y), nil
}

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

// ChromaGrainSample returns one chroma grain-template sample using AV1's chroma
// block offset addressing for the selected subsampling mode.
func ChromaGrainSample(grain []int16, offset uint8, subsamplingX bool, subsamplingY bool, blockCol int, blockRow int, x int, y int) (int16, error) {
	shiftX := chromaSubsamplingShift(subsamplingX)
	shiftY := chromaSubsamplingShift(subsamplingY)
	blockWidth := LumaBlockSize >> shiftX
	blockHeight := LumaBlockSize >> shiftY
	if len(grain) < ChromaGrainSamples ||
		blockCol < 0 || blockCol > 1 ||
		blockRow < 0 || blockRow > 1 ||
		x < 0 || x >= blockWidth ||
		y < 0 || y >= blockHeight {
		return 0, ErrInvalidParams
	}
	col := chromaGrainOffset(int(offset>>4), shiftX) + x + blockWidth*blockCol
	row := chromaGrainOffset(int(offset&0x0f), shiftY) + y + blockHeight*blockRow
	width, height := chromaGrainDimensions(subsamplingX, subsamplingY)
	if col < 0 || col >= width || row < 0 || row >= height {
		return 0, ErrInvalidParams
	}
	return chromaGrainSample(grain, offset, shiftX, shiftY, blockCol, blockRow, x, y), nil
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
