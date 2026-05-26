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
