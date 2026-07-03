// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package restoration

// AVX2-accelerated self-guided restoration per-pixel blend, mirroring the arm64
// NEON port. The 3-row weighted stencil over the A/B buffers, the a*dgd+b
// combine, and the rounding shift run 8 lanes at a time through one general
// AVX2 helper (sgrBlendRowAVX2Asm) parameterised by a 9-tap weight set; the <8
// column tail keeps the scalar reference. The data-dependent LUT gather inside
// calculateIntermediate (xByXPlus1 / oneByX) stays scalar, so every output
// sample is bit-identical for 8/10/12-bit input.
//
// dav1d reference (algorithm shape only): the weighted finish lives in
// third_party/upstream/dav1d/src/x86/looprestoration_avx2.asm
// (sgr_filter_5x5_8bpc / sgr_filter_3x3_8bpc, macro sgr_finish). goav1 keeps
// the libaom integer lattice; the AVX2 here reproduces the exact pure-Go int32
// stencil, not dav1d's.

// sgrBlendAVX2Ctx is the asm calling context for one blended output row. Field
// order and sizes are part of the ABI shared with selfguided_blend_avx2_amd64.s;
// do not reorder. The nine weights map to the 3x3 stencil positions:
//
//	wPL wP wPR   (previous row: left, center, right)
//	wCL wC wCR   (current  row)
//	wNL wN wNR   (next     row)
//
// Each row pointer (aPrev/aCur/aNext and bPrev/bCur/bNext) addresses the
// element one column left of the first output column (index j-1 for col 0), so
// the asm reads three overlapping 8-lane windows (left/center/right) per row.
// Unlike the NEON ctx, shift is the positive VPSRAD count.
type sgrBlendAVX2Ctx struct {
	dst   *int32 // output row base (col 0)
	dgd   *int32 // dgd row base (col 0)
	aPrev *int32 // A previous row, at index (j-1) for col 0
	aCur  *int32 // A current  row
	aNext *int32 // A next     row
	bPrev *int32 // B previous row
	bCur  *int32 // B current  row
	bNext *int32 // B next     row
	cols  uintptr
	wPL   int32
	wP    int32
	wPR   int32
	wCL   int32
	wC    int32
	wCR   int32
	wNL   int32
	wN    int32
	wNR   int32
	bias  int32 // rounding bias 1<<(shift-1)
	shift int32 // positive shift count for VPSRAD (arith >> shift)
}

//go:noescape
func sgrBlendRowAVX2Asm(ctx *sgrBlendAVX2Ctx)

func selfguidedFastAVX2(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const shiftEven = SGRProjSgrBits + 5 - SGRProjRstBits
	const shiftOdd = SGRProjSgrBits + 4 - SGRProjRstBits
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		avx := width &^ 7
		if row&1 == 0 {
			if avx > 0 {
				ctx := sgrBlendAVX2Ctx{
					dst:   &dstRow[0],
					dgd:   &dgdRow[0],
					aPrev: &aBuf[k0-bufStride-1],
					aCur:  &aBuf[k0-1],
					aNext: &aBuf[k0+bufStride-1],
					bPrev: &bBuf[k0-bufStride-1],
					bCur:  &bBuf[k0-1],
					bNext: &bBuf[k0+bufStride-1],
					cols:  uintptr(avx),
					wPL:   5, wP: 6, wPR: 5,
					wCL: 0, wC: 0, wCR: 0,
					wNL: 5, wN: 6, wNR: 5,
					bias: 1 << (shiftEven - 1), shift: shiftEven,
				}
				sgrBlendRowAVX2Asm(&ctx)
			}
			aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
			aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
			bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
			bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
			for col := avx; col < width; col++ {
				j := col + 1
				a := (aPrev[j]+aNext[j])*6 +
					(aPrev[j-1]+aPrev[j+1]+aNext[j-1]+aNext[j+1])*5
				b := (bPrev[j]+bNext[j])*6 +
					(bPrev[j-1]+bPrev[j+1]+bNext[j-1]+bNext[j+1])*5
				dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftEven)
			}
			continue
		}
		if avx > 0 {
			ctx := sgrBlendAVX2Ctx{
				dst:   &dstRow[0],
				dgd:   &dgdRow[0],
				aPrev: &aBuf[k0-1],
				aCur:  &aBuf[k0-1],
				aNext: &aBuf[k0-1],
				bPrev: &bBuf[k0-1],
				bCur:  &bBuf[k0-1],
				bNext: &bBuf[k0-1],
				cols:  uintptr(avx),
				wPL:   0, wP: 0, wPR: 0,
				wCL: 5, wC: 6, wCR: 5,
				wNL: 0, wN: 0, wNR: 0,
				bias: 1 << (shiftOdd - 1), shift: shiftOdd,
			}
			sgrBlendRowAVX2Asm(&ctx)
		}
		aCur := aBuf[k0-1 : k0+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		for col := avx; col < width; col++ {
			j := col + 1
			a := aCur[j]*6 + (aCur[j-1]+aCur[j+1])*5
			b := bCur[j]*6 + (bCur[j-1]+bCur[j+1])*5
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftOdd)
		}
	}
}

func selfguidedAVX2(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const nb = 5
	const shift = SGRProjSgrBits + nb - SGRProjRstBits
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		avx := width &^ 7
		if avx > 0 {
			ctx := sgrBlendAVX2Ctx{
				dst:   &dstRow[0],
				dgd:   &dgdRow[0],
				aPrev: &aBuf[k0-bufStride-1],
				aCur:  &aBuf[k0-1],
				aNext: &aBuf[k0+bufStride-1],
				bPrev: &bBuf[k0-bufStride-1],
				bCur:  &bBuf[k0-1],
				bNext: &bBuf[k0+bufStride-1],
				cols:  uintptr(avx),
				wPL:   3, wP: 4, wPR: 3,
				wCL: 4, wC: 4, wCR: 4,
				wNL: 3, wN: 4, wNR: 3,
				bias: 1 << (shift - 1), shift: shift,
			}
			sgrBlendRowAVX2Asm(&ctx)
		}
		aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
		aCur := aBuf[k0-1 : k0+width+1]
		aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
		bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
		for col := avx; col < width; col++ {
			j := col + 1
			a := (aCur[j]+aCur[j-1]+aCur[j+1]+aPrev[j]+aNext[j])*4 +
				(aPrev[j-1]+aNext[j-1]+aPrev[j+1]+aNext[j+1])*3
			b := (bCur[j]+bCur[j-1]+bCur[j+1]+bPrev[j]+bNext[j])*4 +
				(bPrev[j-1]+bNext[j-1]+bPrev[j+1]+bNext[j+1])*3
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shift)
		}
	}
}
