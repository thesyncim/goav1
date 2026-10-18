// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package restoration

// calculateIntermediateRolling follows dav1d's row-local sum/squared-sum
// construction (src/looprestoration_tmpl.c sgr_box3/5_row_h/v), while retaining
// libaom's exact A/B conversion. Only coefficients in columns [-1,width] and
// the consumed rows are produced. A pair of column-sum rows replaces the full
// sum and squared-sum images; both sums share each source load.
//
// The three-pixel source border contains every full radius-1/2 window used
// here, so no clipped outer box-sum cells are needed. Running sums contain
// the same terms modulo 2^32. The uint32 variance, rounding, lookup and raw
// unscaled B term below are unchanged. Consumers read only the produced
// coefficients (or multiply other rows by exactly zero), so scratch padding
// and skipped rows need no clear. Caller guarantees validSGRBlock and r=1/2.
func calculateIntermediateRolling(dgd []int32, origin, width, height, stride, bitDepth int, params SGRParams, radiusIndex, pass int, aBuf, bBuf []int32, bufStride int) {
	r := int(params.Radius[radiusIndex])
	span := width + 2*r + 2
	var sumStorage, squareStorage [ProcUnitSize + 2*SGRProjBorderHorz]int32
	sums := sumStorage[:span]
	squares := squareStorage[:span]
	first := origin + (-1-r)*stride - 1 - r
	firstRow := dgd[first : first+span]
	for col, v := range firstRow {
		sums[col] = v
		squares[col] = v * v
	}
	for y := 1; y <= 2*r; y++ {
		row := dgd[first+y*stride : first+y*stride+span]
		for col, v := range row {
			sums[col] += v
			squares[col] += v * v
		}
	}
	step := 1
	if pass != 0 {
		step = 2
	}
	n := (2*r + 1) * (2*r + 1)
	scale := uint32(params.S[radiusIndex])
	recip := uint32(oneByX[n-1])
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	for y := -1; y < height+1; y += step {
		sum, square := int32(0), int32(0)
		for col := 0; col <= 2*r; col++ {
			sum += sums[col]
			square += squares[col]
		}
		ar := aBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		br := bBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		for col := 0; col < width+2; col++ {
			a := uint32(roundPowerOfTwo(square, 2*(bitDepth-8)))
			b := uint32(roundPowerOfTwo(sum, bitDepth-8))
			p := uint32(0)
			if a*uint32(n) >= b*b {
				p = a*uint32(n) - b*b
			}
			z := min(roundPowerOfTwoUnsigned(p*scale, SGRProjMTableBits), 255)
			aa := xByXPlus1Value(z)
			ar[col] = aa
			br[col] = int32(roundPowerOfTwoUnsigned(uint32(SGRProjSgr-int(aa))*uint32(sum)*recip, SGRProjRecipBits))
			if col+1 < width+2 {
				sum += sums[col+2*r+1] - sums[col]
				square += squares[col+2*r+1] - squares[col]
			}
		}
		if y+step >= height+1 {
			break
		}
		for advance := 0; advance < step; advance++ {
			outStart := origin + (y-r+advance)*stride - 1 - r
			inStart := origin + (y+r+1+advance)*stride - 1 - r
			out := dgd[outStart : outStart+span]
			in := dgd[inStart : inStart+span]
			for col, v := range in {
				u := out[col]
				sums[col] += v - u
				squares[col] += v*v - u*u
			}
		}
	}
}
