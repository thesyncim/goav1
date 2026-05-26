package prediction

const smoothWeightLog2Scale = 8

var smoothWeights = [...]uint16{
	255, 149, 85, 64,
	255, 197, 146, 105, 73, 50, 37, 32,
	255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
	255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
	66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
	255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156,
	150, 144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
	65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20, 18, 16, 15,
	13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
}

func predictPaeth(block planeBlock, bytesPerSample int, above []uint16, left []uint16, aboveLeft uint16) {
	for row := 0; row < block.height; row++ {
		for col := 0; col < block.width; col++ {
			setBlockSample(block, bytesPerSample, row, col, paethPredictorSingle(left[row], above[col], aboveLeft))
		}
	}
}

func predictSmoothWithExtent(block planeBlock, bytesPerSample int, predWidth int, predHeight int, above []uint16, left []uint16) error {
	weightsW, err := smoothWeightsForSize(predWidth)
	if err != nil {
		return err
	}
	weightsH, err := smoothWeightsForSize(predHeight)
	if err != nil {
		return err
	}
	belowPred := left[predHeight-1]
	rightPred := above[predWidth-1]
	scale := uint16(1 << smoothWeightLog2Scale)
	for row := 0; row < block.height; row++ {
		for col := 0; col < block.width; col++ {
			pred := uint32(weightsH[row])*uint32(above[col]) +
				uint32(scale-weightsH[row])*uint32(belowPred) +
				uint32(weightsW[col])*uint32(left[row]) +
				uint32(scale-weightsW[col])*uint32(rightPred)
			setBlockSample(block, bytesPerSample, row, col, uint16(divideRound(pred, 1+smoothWeightLog2Scale)))
		}
	}
	return nil
}

func predictSmoothVerticalWithExtent(block planeBlock, bytesPerSample int, predHeight int, above []uint16, left []uint16) error {
	weights, err := smoothWeightsForSize(predHeight)
	if err != nil {
		return err
	}
	belowPred := left[predHeight-1]
	scale := uint16(1 << smoothWeightLog2Scale)
	for row := 0; row < block.height; row++ {
		for col := 0; col < block.width; col++ {
			pred := uint32(weights[row])*uint32(above[col]) +
				uint32(scale-weights[row])*uint32(belowPred)
			setBlockSample(block, bytesPerSample, row, col, uint16(divideRound(pred, smoothWeightLog2Scale)))
		}
	}
	return nil
}

func predictSmoothHorizontalWithExtent(block planeBlock, bytesPerSample int, predWidth int, above []uint16, left []uint16) error {
	weights, err := smoothWeightsForSize(predWidth)
	if err != nil {
		return err
	}
	rightPred := above[predWidth-1]
	scale := uint16(1 << smoothWeightLog2Scale)
	for row := 0; row < block.height; row++ {
		for col := 0; col < block.width; col++ {
			pred := uint32(weights[col])*uint32(left[row]) +
				uint32(scale-weights[col])*uint32(rightPred)
			setBlockSample(block, bytesPerSample, row, col, uint16(divideRound(pred, smoothWeightLog2Scale)))
		}
	}
	return nil
}

func paethPredictorSingle(left uint16, top uint16, topLeft uint16) uint16 {
	base := int(top) + int(left) - int(topLeft)
	pLeft := absInt(base - int(left))
	pTop := absInt(base - int(top))
	pTopLeft := absInt(base - int(topLeft))
	if pLeft <= pTop && pLeft <= pTopLeft {
		return left
	}
	if pTop <= pTopLeft {
		return top
	}
	return topLeft
}

func smoothWeightsForSize(size int) ([]uint16, error) {
	switch size {
	case 4, 8, 16, 32, 64:
		return smoothWeights[size-4 : size-4+size], nil
	default:
		return nil, ErrInvalidPrediction
	}
}

func divideRound(value uint32, bits int) uint32 {
	return (value + (1 << (bits - 1))) >> bits
}
