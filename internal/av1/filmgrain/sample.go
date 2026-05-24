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
	col := lumaGrainOffset(int(offset>>4)) + x + LumaBlockSize*blockCol
	row := lumaGrainOffset(int(offset&0x0f)) + y + LumaBlockSize*blockRow
	return grain[row*LumaGrainWidth+col]
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
	blockWidth := LumaBlockSize >> shiftX
	blockHeight := LumaBlockSize >> shiftY
	col := chromaGrainOffset(int(offset>>4), shiftX) + x + blockWidth*blockCol
	row := chromaGrainOffset(int(offset&0x0f), shiftY) + y + blockHeight*blockRow
	return grain[row*ChromaGrainWidth+col]
}

func lumaGrainOffset(n int) int {
	return 3 + LumaOverlapSamples*(3+n)
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
