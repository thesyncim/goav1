package filmgrain

// ChromaGrainParams contains the inputs needed to generate one chroma
// film-grain template before it is placed over a decoded chroma plane.
type ChromaGrainParams struct {
	Seed  uint16
	Plane int

	BitDepth        uint8
	NumYPoints      uint8
	GrainScaleShift uint8

	SubsamplingX bool
	SubsamplingY bool

	ARCoeffLag   uint8
	ARCoeffShift uint8
	ARCoeffs     [MaxChromaARCoeffs]int8
}

// GenerateChromaGrain fills dst with one AV1 chroma grain template. The
// template uses an 82-sample stride for every chroma subsampling mode.
func GenerateChromaGrain(dst []int16, luma []int16, params ChromaGrainParams) error {
	if err := validateChromaGrainParams(dst, luma, params); err != nil {
		return err
	}
	grain := dst[:ChromaGrainSamples]
	for i := range grain {
		grain[i] = 0
	}

	rng, err := NewPlaneRandom(params.Seed, params.Plane)
	if err != nil {
		return err
	}
	width, height := chromaGrainDimensions(params.SubsamplingX, params.SubsamplingY)
	gaussianShift := int(12 - params.BitDepth + params.GrainScaleShift)
	for y := range height {
		for x := range width {
			index, err := rng.Number(GaussianBits)
			if err != nil {
				return err
			}
			grain[y*ChromaGrainWidth+x] = int16(roundPowerOfTwo(int(Gaussian(index)), gaussianShift))
		}
	}

	lag := int(params.ARCoeffLag)
	if lag == 0 && params.NumYPoints == 0 {
		return nil
	}
	coeffShift := int(params.ARCoeffShift)
	roundingOffset := 1 << (coeffShift - 1)
	grainMin := -(1 << (params.BitDepth - 1))
	grainMax := (1 << (params.BitDepth - 1)) - 1
	for y := 3; y < height; y++ {
		for x := 3; x < width-3; x++ {
			sum := 0
			pos := 0
			for deltaRow := -lag; deltaRow <= 0; deltaRow++ {
				for deltaCol := -lag; deltaCol <= lag; deltaCol++ {
					if deltaRow == 0 && deltaCol == 0 {
						if params.NumYPoints != 0 {
							sum += chromaLumaAverage(luma, x, y, params.SubsamplingX, params.SubsamplingY) * int(params.ARCoeffs[pos])
						}
						break
					}
					sample := int(grain[(y+deltaRow)*ChromaGrainWidth+x+deltaCol])
					sum += int(params.ARCoeffs[pos]) * sample
					pos++
				}
			}
			v := int(grain[y*ChromaGrainWidth+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[y*ChromaGrainWidth+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
	return nil
}

func validateChromaGrainParams(dst []int16, luma []int16, params ChromaGrainParams) error {
	if len(dst) < ChromaGrainSamples ||
		(params.BitDepth != 8 && params.BitDepth != 10 && params.BitDepth != 12) ||
		params.NumYPoints > MaxLumaScalingPoints ||
		params.GrainScaleShift > 3 ||
		params.ARCoeffLag > 3 {
		return ErrInvalidParams
	}
	if params.Plane != ChromaPlaneCb && params.Plane != ChromaPlaneCr {
		return ErrInvalidParams
	}
	if params.NumYPoints != 0 && len(luma) < LumaGrainSamples {
		return ErrInvalidParams
	}
	if (params.ARCoeffLag != 0 || params.NumYPoints != 0) &&
		(params.ARCoeffShift < 6 || params.ARCoeffShift > 9) {
		return ErrInvalidParams
	}
	return nil
}

func chromaGrainDimensions(subsamplingX bool, subsamplingY bool) (int, int) {
	width := ChromaGrainWidth
	if subsamplingX {
		width = ChromaSubsampledGrainWidth
	}
	height := ChromaGrainHeight
	if subsamplingY {
		height = ChromaSubsampledGrainHeight
	}
	return width, height
}

func chromaLumaAverage(luma []int16, x int, y int, subsamplingX bool, subsamplingY bool) int {
	subX := 0
	if subsamplingX {
		subX = 1
	}
	subY := 0
	if subsamplingY {
		subY = 1
	}
	lumaX := ((x - 3) << subX) + 3
	lumaY := ((y - 3) << subY) + 3
	sum := 0
	for dy := 0; dy <= subY; dy++ {
		for dx := 0; dx <= subX; dx++ {
			sum += int(luma[(lumaY+dy)*LumaGrainWidth+lumaX+dx])
		}
	}
	return roundPowerOfTwo(sum, subX+subY)
}
