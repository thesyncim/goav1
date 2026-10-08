// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package restoration

// Go-native SIMD self-guided restoration drivers shared by arm64 and amd64. The
// per-pixel blend (sgrBlendRowSIMD) and the lane count (sgrBlendLanes) are
// supplied by the architecture file. The data-dependent LUT gather inside
// calculateIntermediate (xByXPlus1 / oneByX) stays scalar, and the columns beyond
// the last full vector keep the scalar reference, so every output sample is
// bit-identical for 8/10/12-bit input.

// boxsumCell reproduces one scalar box-sum cell for the clamped edge columns. It
// mirrors boxsum exactly.
func boxsumCell(src []int32, srcOrigin, width, srcStride, radius int, squared bool, col, y0, y1 int) int32 {
	sum := int32(0)
	x0 := maxInt(0, col-radius)
	x1 := minInt(width-1, col+radius)
	if squared {
		for y := y0; y <= y1; y++ {
			base := srcOrigin + y*srcStride
			for _, v := range src[base+x0 : base+x1+1] {
				sum += v * v
			}
		}
	} else {
		for y := y0; y <= y1; y++ {
			base := srcOrigin + y*srcStride
			for _, v := range src[base+x0 : base+x1+1] {
				sum += v
			}
		}
	}
	return sum
}

// selfguidedSIMD is the self-guided filter with the SIMD blend, used for the
// radius-1 and radius-2 passes. Its A/B coefficients come from the scalar
// calculateIntermediate with the parameter pass selected by the radius index.
func selfguidedSIMD(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const nb = 5
	const shift = SGRProjSgrBits + nb - SGRProjRstBits
	weights := [9]int32{3, 4, 3, 4, 4, 4, 3, 4, 3}
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		simdCols := width &^ (sgrBlendLanes - 1)
		if simdCols > 0 {
			sgrBlendRowSIMD(dstRow, dgdRow, aBuf[k0-bufStride-1:], aBuf[k0-1:], aBuf[k0+bufStride-1:],
				bBuf[k0-bufStride-1:], bBuf[k0-1:], bBuf[k0+bufStride-1:], &weights, shift, simdCols)
		}
		aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
		aCur := aBuf[k0-1 : k0+width+1]
		aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
		bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
		for col := simdCols; col < width; col++ {
			j := col + 1
			a := (aCur[j]+aCur[j-1]+aCur[j+1]+aPrev[j]+aNext[j])*4 +
				(aPrev[j-1]+aNext[j-1]+aPrev[j+1]+aNext[j+1])*3
			b := (bCur[j]+bCur[j-1]+bCur[j+1]+bPrev[j]+bNext[j])*4 +
				(bPrev[j-1]+bNext[j-1]+bPrev[j+1]+bNext[j+1])*3
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shift)
		}
	}
}

// selfguidedFastSIMD is the radius-2 self-guided filter with the SIMD blend. It
// evaluates the box-sum stencil only on alternating rows (the "fast" subsampled
// variant), so even and odd rows use different weight sets.
func selfguidedFastSIMD(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const shiftEven = SGRProjSgrBits + 5 - SGRProjRstBits
	const shiftOdd = SGRProjSgrBits + 4 - SGRProjRstBits
	evenWeights := [9]int32{5, 6, 5, 0, 0, 0, 5, 6, 5}
	oddWeights := [9]int32{0, 0, 0, 5, 6, 5, 0, 0, 0}
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		simdCols := width &^ (sgrBlendLanes - 1)
		if row&1 == 0 {
			if simdCols > 0 {
				sgrBlendRowSIMD(dstRow, dgdRow, aBuf[k0-bufStride-1:], aBuf[k0-1:], aBuf[k0+bufStride-1:],
					bBuf[k0-bufStride-1:], bBuf[k0-1:], bBuf[k0+bufStride-1:], &evenWeights, shiftEven, simdCols)
			}
			aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
			aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
			bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
			bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
			for col := simdCols; col < width; col++ {
				j := col + 1
				a := (aPrev[j]+aNext[j])*6 +
					(aPrev[j-1]+aPrev[j+1]+aNext[j-1]+aNext[j+1])*5
				b := (bPrev[j]+bNext[j])*6 +
					(bPrev[j-1]+bPrev[j+1]+bNext[j-1]+bNext[j+1])*5
				dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftEven)
			}
			continue
		}
		if simdCols > 0 {
			sgrBlendRowSIMD(dstRow, dgdRow, aBuf[k0-1:], aBuf[k0-1:], aBuf[k0-1:],
				bBuf[k0-1:], bBuf[k0-1:], bBuf[k0-1:], &oddWeights, shiftOdd, simdCols)
		}
		aCur := aBuf[k0-1 : k0+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		for col := simdCols; col < width; col++ {
			j := col + 1
			a := aCur[j]*6 + (aCur[j-1]+aCur[j+1])*5
			b := bCur[j]*6 + (bCur[j-1]+bCur[j+1])*5
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftOdd)
		}
	}
}
