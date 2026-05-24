package filmgrain

const (
	LumaLegalMin      = 16
	LumaLegalMax      = 235
	ChromaLegalMax    = 240
	ChromaIdentityMax = 235
)

// BlendLumaOverlap blends a previous block's grain with the current block's
// grain for AV1's two-sample luma overlap region.
func BlendLumaOverlap(previous int16, current int16, offset int, bitDepth uint8) (int16, error) {
	if offset < 0 || offset >= LumaOverlapSamples || (bitDepth != 8 && bitDepth != 10 && bitDepth != 12) {
		return 0, ErrInvalidParams
	}
	return blendLumaOverlap(previous, current, offset, bitDepth), nil
}

func blendLumaOverlap(previous int16, current int16, offset int, bitDepth uint8) int16 {
	previousWeight, currentWeight := 27, 17
	if offset == 1 {
		previousWeight, currentWeight = 17, 27
	}
	v := roundPowerOfTwo(int(previous)*previousWeight+int(current)*currentWeight, 5)
	grainMin := -(1 << (bitDepth - 1))
	grainMax := (1 << (bitDepth - 1)) - 1
	return int16(clipInt(v, grainMin, grainMax))
}

// ApplyLumaSample blends one luma grain sample with one decoded luma sample.
func ApplyLumaSample(orig uint16, grain int16, lut []uint8, bitDepth uint8, scalingShift uint8, restrictedRange bool) (uint16, error) {
	if scalingShift < 8 || scalingShift > 11 {
		return 0, ErrInvalidParams
	}
	if len(lut) < ScalingLUTSize || bitDepth < 8 || bitDepth > 12 || int(orig) >= 1<<bitDepth {
		return 0, ErrInvalidParams
	}
	return applyLumaSample(orig, grain, lut, bitDepth, scalingShift, restrictedRange), nil
}

func applyLumaSample(orig uint16, grain int16, lut []uint8, bitDepth uint8, scalingShift uint8, restrictedRange bool) uint16 {
	scale := scaleLUT(lut, int(orig), bitDepth)
	noise := roundPowerOfTwo(int(scale)*int(grain), int(scalingShift))
	minValue := 0
	maxValue := (256 << (bitDepth - 8)) - 1
	if restrictedRange {
		minValue = LumaLegalMin << (bitDepth - 8)
		maxValue = LumaLegalMax << (bitDepth - 8)
	}
	return uint16(clipInt(int(orig)+noise, minValue, maxValue))
}
