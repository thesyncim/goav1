// Ported from libaom: av1/decoder/grain_synthesis.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package filmgrain

// LumaGrainParams contains the luma inputs needed to generate the film-grain
// template before it is placed over a decoded frame.
type LumaGrainParams struct {
	Seed uint16

	BitDepth        uint8
	NumYPoints      uint8
	GrainScaleShift uint8

	ARCoeffLag   uint8
	ARCoeffShift uint8
	ARCoeffs     [MaxLumaARCoeffs]int8
}

// GenerateLumaGrain fills dst with AV1's 82x73 luma grain template.
func GenerateLumaGrain(dst []int16, params LumaGrainParams) error {
	if err := validateLumaGrainParams(dst, params); err != nil {
		return err
	}
	grain := dst[:LumaGrainSamples]
	for i := range grain {
		grain[i] = 0
	}
	if params.NumYPoints == 0 {
		return nil
	}

	rng := NewRandom(params.Seed)
	gaussianShift := int(12 - params.BitDepth + params.GrainScaleShift)
	for y := range LumaGrainHeight {
		for x := range LumaGrainWidth {
			index, err := rng.Number(GaussianBits)
			if err != nil {
				return err
			}
			grain[y*LumaGrainWidth+x] = int16(roundPowerOfTwo(int(Gaussian(index)), gaussianShift))
		}
	}

	lag := int(params.ARCoeffLag)
	if lag == 0 {
		return nil
	}
	coeffShift := int(params.ARCoeffShift)
	roundingOffset := 1 << (coeffShift - 1)
	grainMin := -(1 << (params.BitDepth - 1))
	grainMax := (1 << (params.BitDepth - 1)) - 1
	switch lag {
	case 1:
		generateLumaGrainARLag1(grain, &params, roundingOffset, coeffShift, grainMin, grainMax)
		return nil
	case 2:
		generateLumaGrainARLag2(grain, &params, roundingOffset, coeffShift, grainMin, grainMax)
		return nil
	case 3:
		generateLumaGrainARLag3(grain, &params, roundingOffset, coeffShift, grainMin, grainMax)
		return nil
	}
	for y := 3; y < LumaGrainHeight; y++ {
		for x := 3; x < LumaGrainWidth-3; x++ {
			sum := 0
			pos := 0
			for deltaRow := -lag; deltaRow <= 0; deltaRow++ {
				for deltaCol := -lag; deltaCol <= lag; deltaCol++ {
					if deltaRow == 0 && deltaCol == 0 {
						break
					}
					sample := int(grain[(y+deltaRow)*LumaGrainWidth+x+deltaCol])
					sum += int(params.ARCoeffs[pos]) * sample
					pos++
				}
			}
			v := int(grain[y*LumaGrainWidth+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[y*LumaGrainWidth+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
	return nil
}

func generateLumaGrainARLag1(grain []int16, params *LumaGrainParams, roundingOffset int, coeffShift int, grainMin int, grainMax int) {
	for y := 3; y < LumaGrainHeight; y++ {
		row := y * LumaGrainWidth
		rowM1 := row - LumaGrainWidth
		for x := 3; x < LumaGrainWidth-3; x++ {
			sum := int(params.ARCoeffs[0])*int(grain[rowM1+x-1]) +
				int(params.ARCoeffs[1])*int(grain[rowM1+x]) +
				int(params.ARCoeffs[2])*int(grain[rowM1+x+1]) +
				int(params.ARCoeffs[3])*int(grain[row+x-1])
			v := int(grain[row+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[row+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
}

func generateLumaGrainARLag2(grain []int16, params *LumaGrainParams, roundingOffset int, coeffShift int, grainMin int, grainMax int) {
	for y := 3; y < LumaGrainHeight; y++ {
		row := y * LumaGrainWidth
		rowM1 := row - LumaGrainWidth
		rowM2 := row - 2*LumaGrainWidth
		for x := 3; x < LumaGrainWidth-3; x++ {
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
			v := int(grain[row+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[row+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
}

func generateLumaGrainARLag3(grain []int16, params *LumaGrainParams, roundingOffset int, coeffShift int, grainMin int, grainMax int) {
	for y := 3; y < LumaGrainHeight; y++ {
		row := y * LumaGrainWidth
		rowM1 := row - LumaGrainWidth
		rowM2 := row - 2*LumaGrainWidth
		rowM3 := row - 3*LumaGrainWidth
		for x := 3; x < LumaGrainWidth-3; x++ {
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
			v := int(grain[row+x]) + ((sum + roundingOffset) >> coeffShift)
			grain[row+x] = int16(clipInt(v, grainMin, grainMax))
		}
	}
}

func validateLumaGrainParams(dst []int16, params LumaGrainParams) error {
	if len(dst) < LumaGrainSamples ||
		(params.BitDepth != 8 && params.BitDepth != 10 && params.BitDepth != 12) ||
		params.NumYPoints > MaxLumaScalingPoints ||
		params.GrainScaleShift > 3 ||
		params.ARCoeffLag > 3 {
		return ErrInvalidParams
	}
	if params.NumYPoints != 0 && params.ARCoeffLag != 0 &&
		(params.ARCoeffShift < 6 || params.ARCoeffShift > 9) {
		return ErrInvalidParams
	}
	return nil
}

func clipInt(value int, low int, high int) int {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
