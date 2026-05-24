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

func lumaGrainOffset(n int) int {
	return 3 + LumaOverlapSamples*(3+n)
}
