// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

// Frozen ccec1451 driver bodies retain the unfactored general blend calls.
// They independently pin row scheduling, alias batching and complete scratch.

// frozenSGRGeneric is the self-guided filter with the SIMD blend, used for the
// radius-1 and radius-2 passes. Its A/B coefficients come from the scalar
// calculateIntermediate with the parameter pass selected by the radius index.
func frozenSGRGeneric(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
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

// frozenSGRGenericFast is the radius-2 self-guided filter with the SIMD blend. It
// evaluates the box-sum stencil only on alternating rows (the "fast" subsampled
// variant), so even and odd rows use different weight sets.
func frozenSGRGenericFast(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
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
