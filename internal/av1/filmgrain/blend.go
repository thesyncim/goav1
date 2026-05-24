package filmgrain

const (
	LumaLegalMin = 16
	LumaLegalMax = 235
)

// BlendLumaOverlap blends a previous block's grain with the current block's
// grain for AV1's two-sample luma overlap region.
func BlendLumaOverlap(previous int16, current int16, offset int, bitDepth uint8) (int16, error) {
	if offset < 0 || offset >= LumaOverlapSamples || (bitDepth != 8 && bitDepth != 10 && bitDepth != 12) {
		return 0, ErrInvalidParams
	}
	previousWeight, currentWeight := 27, 17
	if offset == 1 {
		previousWeight, currentWeight = 17, 27
	}
	v := roundPowerOfTwo(int(previous)*previousWeight+int(current)*currentWeight, 5)
	grainMin := -(1 << (bitDepth - 1))
	grainMax := (1 << (bitDepth - 1)) - 1
	return int16(clipInt(v, grainMin, grainMax)), nil
}

// ApplyLumaSample blends one luma grain sample with one decoded luma sample.
func ApplyLumaSample(orig uint16, grain int16, lut []uint8, bitDepth uint8, scalingShift uint8, restrictedRange bool) (uint16, error) {
	if scalingShift < 8 || scalingShift > 11 {
		return 0, ErrInvalidParams
	}
	scale, err := ScaleLUT(lut, int(orig), bitDepth)
	if err != nil {
		return 0, err
	}
	noise := roundPowerOfTwo(int(scale)*int(grain), int(scalingShift))
	minValue := 0
	maxValue := (256 << (bitDepth - 8)) - 1
	if restrictedRange {
		minValue = LumaLegalMin << (bitDepth - 8)
		maxValue = LumaLegalMax << (bitDepth - 8)
	}
	return uint16(clipInt(int(orig)+noise, minValue, maxValue)), nil
}
