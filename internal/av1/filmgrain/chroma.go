// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

// ChromaGrainParams contains the inputs needed to generate one chroma
// film-grain template before it is placed over a decoded chroma plane.
type ChromaGrainParams struct {
	Seed  uint16
	Plane uint8

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

	rng, err := NewPlaneRandom(params.Seed, int(params.Plane))
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
	switch lag {
	case 1:
		generateChromaGrainARLag1(grain, luma, &params, width, height, roundingOffset, coeffShift, grainMin, grainMax)
		return nil
	case 2:
		generateChromaGrainARLag2(grain, luma, &params, width, height, roundingOffset, coeffShift, grainMin, grainMax)
		return nil
	case 3:
		generateChromaGrainARLag3(grain, luma, &params, width, height, roundingOffset, coeffShift, grainMin, grainMax)
		return nil
	}
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

func generateChromaGrainARLag1(grain []int16, luma []int16, params *ChromaGrainParams, width int, height int, roundingOffset int, coeffShift int, grainMin int, grainMax int) {
	hasLuma := params.NumYPoints != 0
	luma444 := hasLuma && !params.SubsamplingX && !params.SubsamplingY
	for y := 3; y < height; y++ {
		row := y * ChromaGrainWidth
		rowM1 := row - ChromaGrainWidth
		for x := 3; x < width-3; x++ {
			sum := int(params.ARCoeffs[0])*int(grain[rowM1+x-1]) +
				int(params.ARCoeffs[1])*int(grain[rowM1+x]) +
				int(params.ARCoeffs[2])*int(grain[rowM1+x+1]) +
				int(params.ARCoeffs[3])*int(grain[row+x-1])
			if luma444 {
				sum += int(luma[y*LumaGrainWidth+x]) * int(params.ARCoeffs[4])
			} else if hasLuma {
				sum += chromaLumaAverage(luma, x, y, params.SubsamplingX, params.SubsamplingY) * int(params.ARCoeffs[4])
			}
			v := int(grain[row+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[row+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
}

func generateChromaGrainARLag2(grain []int16, luma []int16, params *ChromaGrainParams, width int, height int, roundingOffset int, coeffShift int, grainMin int, grainMax int) {
	hasLuma := params.NumYPoints != 0
	luma444 := hasLuma && !params.SubsamplingX && !params.SubsamplingY
	for y := 3; y < height; y++ {
		row := y * ChromaGrainWidth
		rowM1 := row - ChromaGrainWidth
		rowM2 := row - 2*ChromaGrainWidth
		for x := 3; x < width-3; x++ {
			sum := int(params.ARCoeffs[0])*int(grain[rowM2+x-2]) +
				int(params.ARCoeffs[1])*int(grain[rowM2+x-1]) +
				int(params.ARCoeffs[2])*int(grain[rowM2+x]) +
				int(params.ARCoeffs[3])*int(grain[rowM2+x+1]) +
				int(params.ARCoeffs[4])*int(grain[rowM2+x+2]) +
				int(params.ARCoeffs[5])*int(grain[rowM1+x-2]) +
				int(params.ARCoeffs[6])*int(grain[rowM1+x-1]) +
				int(params.ARCoeffs[7])*int(grain[rowM1+x]) +
				int(params.ARCoeffs[8])*int(grain[rowM1+x+1]) +
				int(params.ARCoeffs[9])*int(grain[rowM1+x+2]) +
				int(params.ARCoeffs[10])*int(grain[row+x-2]) +
				int(params.ARCoeffs[11])*int(grain[row+x-1])
			if luma444 {
				sum += int(luma[y*LumaGrainWidth+x]) * int(params.ARCoeffs[12])
			} else if hasLuma {
				sum += chromaLumaAverage(luma, x, y, params.SubsamplingX, params.SubsamplingY) * int(params.ARCoeffs[12])
			}
			v := int(grain[row+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[row+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
}

func generateChromaGrainARLag3(grain []int16, luma []int16, params *ChromaGrainParams, width int, height int, roundingOffset int, coeffShift int, grainMin int, grainMax int) {
	hasLuma := params.NumYPoints != 0
	luma444 := hasLuma && !params.SubsamplingX && !params.SubsamplingY
	for y := 3; y < height; y++ {
		row := y * ChromaGrainWidth
		rowM1 := row - ChromaGrainWidth
		rowM2 := row - 2*ChromaGrainWidth
		rowM3 := row - 3*ChromaGrainWidth
		for x := 3; x < width-3; x++ {
			sum := int(params.ARCoeffs[0])*int(grain[rowM3+x-3]) +
				int(params.ARCoeffs[1])*int(grain[rowM3+x-2]) +
				int(params.ARCoeffs[2])*int(grain[rowM3+x-1]) +
				int(params.ARCoeffs[3])*int(grain[rowM3+x]) +
				int(params.ARCoeffs[4])*int(grain[rowM3+x+1]) +
				int(params.ARCoeffs[5])*int(grain[rowM3+x+2]) +
				int(params.ARCoeffs[6])*int(grain[rowM3+x+3]) +
				int(params.ARCoeffs[7])*int(grain[rowM2+x-3]) +
				int(params.ARCoeffs[8])*int(grain[rowM2+x-2]) +
				int(params.ARCoeffs[9])*int(grain[rowM2+x-1]) +
				int(params.ARCoeffs[10])*int(grain[rowM2+x]) +
				int(params.ARCoeffs[11])*int(grain[rowM2+x+1]) +
				int(params.ARCoeffs[12])*int(grain[rowM2+x+2]) +
				int(params.ARCoeffs[13])*int(grain[rowM2+x+3]) +
				int(params.ARCoeffs[14])*int(grain[rowM1+x-3]) +
				int(params.ARCoeffs[15])*int(grain[rowM1+x-2]) +
				int(params.ARCoeffs[16])*int(grain[rowM1+x-1]) +
				int(params.ARCoeffs[17])*int(grain[rowM1+x]) +
				int(params.ARCoeffs[18])*int(grain[rowM1+x+1]) +
				int(params.ARCoeffs[19])*int(grain[rowM1+x+2]) +
				int(params.ARCoeffs[20])*int(grain[rowM1+x+3]) +
				int(params.ARCoeffs[21])*int(grain[row+x-3]) +
				int(params.ARCoeffs[22])*int(grain[row+x-2]) +
				int(params.ARCoeffs[23])*int(grain[row+x-1])
			if luma444 {
				sum += int(luma[y*LumaGrainWidth+x]) * int(params.ARCoeffs[24])
			} else if hasLuma {
				sum += chromaLumaAverage(luma, x, y, params.SubsamplingX, params.SubsamplingY) * int(params.ARCoeffs[24])
			}
			v := int(grain[row+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[row+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
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
