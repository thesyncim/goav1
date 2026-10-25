// Ported from dav1d: src/arm/64/filmgrain.S (sum_lag3_above_neon,
// sum_lag_n_body)
// SPDX-License-Identifier: BSD-2-Clause
// See LICENSE and NOTICE.

//go:build goexperiment.simd && arm64 && !purego

package filmgrain

import "simd/archsimd"

// arLag3Taps holds the 21 above-row lag-3 coefficients (rows y-3..y-1,
// columns x-3..x+3) as int16 lanes, loaded per use: archsimd lowers
// broadcasts to GPR->lane inserts.
type arLag3Taps [21][8]int16

func (t *arLag3Taps) init(coeffs []int8) {
	for i := range t {
		for j := range t[i] {
			t[i][j] = int16(coeffs[i])
		}
	}
}

// arLag3Above is dav1d's sum_lag3_above_neon: the above-row part of the
// lag-3 auto-regression for eight grain samples per step, x = 3..3+n
// (rounded up to whole steps), into above[x-3]. Each tap is a widening
// smull/smull2 into int32, exact for every bit depth. It reads grain rows
// row-3..row-1 only, so the serial left-neighbour pass can follow.
func arLag3Above(above *[80]int32, grain []int16, row int, n int, taps *arLag3Taps) {
	for i := 0; i < n; i += 8 {
		var lo, hi archsimd.Int32x4
		tap := 0
		for r := 3; r >= 1; r-- {
			base := row - r*LumaGrainWidth + i
			g := grain[base : base+14]
			for dc := 0; dc < 7; dc++ {
				v := archsimd.LoadInt16x8Array((*[8]int16)(g[dc : dc+8]))
				c := archsimd.LoadInt16x8Array(&taps[tap])
				lo = lo.Add(v.MulWidenLo(c))
				hi = hi.Add(v.HiToLo().MulWidenLo(c.HiToLo()))
				tap++
			}
		}
		lo.StoreArray((*[4]int32)(above[i : i+4]))
		hi.StoreArray((*[4]int32)(above[i+4 : i+8]))
	}
}

// arLag3 runs the lag-3 auto-regression on rows 3..height-1, columns
// 3..width-4: the above rows through arLag3Above, then the serial
// left-neighbour taps. For chroma (luma != nil) coeffs[24] multiplies the
// co-located luma grain average, as chromaLumaAverage.
func arLag3(grain []int16, luma []int16, coeffs []int8, sx, sy, width, height, roundingOffset, coeffShift, grainMin, grainMax int) {
	var taps arLag3Taps
	taps.init(coeffs)
	c21, c22, c23 := int(coeffs[21]), int(coeffs[22]), int(coeffs[23])
	cl := 0
	if luma != nil {
		cl = int(coeffs[24])
	}
	var above [80]int32
	for y := 3; y < height; y++ {
		row := y * LumaGrainWidth
		arLag3Above(&above, grain, row, width-6, &taps)
		if luma != nil {
			lumaRow := (((y - 3) << sy) + 3) * LumaGrainWidth
			for x := 3; x < width-3; x++ {
				li := lumaRow + ((x - 3) << sx) + 3
				avg := int(luma[li])
				if sx != 0 {
					avg += int(luma[li+1])
				}
				if sy != 0 {
					avg += int(luma[li+LumaGrainWidth])
					if sx != 0 {
						avg += int(luma[li+LumaGrainWidth+1])
					}
				}
				above[x-3] += int32(roundPowerOfTwo(avg, sx+sy) * cl)
			}
		}
		g := grain[row : row+width]
		for x := 3; x < width-3; x++ {
			sum := int(above[x-3]) + c21*int(g[x-3]) + c22*int(g[x-2]) + c23*int(g[x-1])
			g[x] = int16(clipInt(int(g[x])+((sum+roundingOffset)>>coeffShift), grainMin, grainMax))
		}
	}
}

func lumaGrainARLag3(grain []int16, params *LumaGrainParams, roundingOffset, coeffShift, grainMin, grainMax int) {
	arLag3(grain, nil, params.ARCoeffs[:], 0, 0, LumaGrainWidth, LumaGrainHeight, roundingOffset, coeffShift, grainMin, grainMax)
}

func chromaGrainARLag3(grain []int16, luma []int16, params *ChromaGrainParams, width, height, roundingOffset, coeffShift, grainMin, grainMax int) {
	if params.NumYPoints == 0 {
		luma = nil
	}
	arLag3(grain, luma, params.ARCoeffs[:], chromaSubsamplingShift(params.SubsamplingX), chromaSubsamplingShift(params.SubsamplingY), width, height, roundingOffset, coeffShift, grainMin, grainMax)
}
