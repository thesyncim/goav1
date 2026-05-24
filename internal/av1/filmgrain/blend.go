package filmgrain

const (
	LumaLegalMin = 16
	LumaLegalMax = 235
)

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
