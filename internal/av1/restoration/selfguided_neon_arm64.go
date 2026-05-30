// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package restoration

// NEON-accelerated self-guided restoration kernels. Two pieces are vectorized:
//
//   - boxsumNEON: the box-filter sums. The interior columns (where the 2r+1
//     horizontal window is fully in bounds) are summed 4 lanes at a time; the r
//     edge columns on each side keep the scalar variable-bound reference.
//   - selfguidedNEON / selfguidedFastNEON: the per-pixel blend. The 3-row
//     weighted stencil over the A/B buffers, the a*dgd+b combine, and the
//     rounding shift run 4 lanes at a time through one general NEON helper
//     (sgrBlendRowNEON) parameterised by a 9-tap weight set; the <4 column tail
//     keeps the scalar reference.
//
// The data-dependent LUT gather inside calculateIntermediate (xByXPlus1 /
// oneByX) stays scalar in this path — only the box sums and the blend are
// vectorized, exactly as the reference computes them in int32, so every output
// sample is bit-identical for 8/10/12-bit input.

// sgrBlendCtx is the asm calling context for one blended output row. Field
// order and sizes are part of the ABI shared with selfguided_neon_arm64.s; do
// not reorder.
//
// The nine weights map to the 3x3 stencil positions:
//
//	wPL wP wPR   (previous row: left, center, right)
//	wCL wC wCR   (current  row)
//	wNL wN wNR   (next     row)
//
// Each row pointer (aPrev/aCur/aNext and bPrev/bCur/bNext) addresses the
// element one column left of the first output column (index j-1 for col 0), so
// the asm reads three contiguous 4-lane windows (left/center/right) per row.
type sgrBlendCtx struct {
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
	shift int32 // -shift (negative => arithmetic right shift via SSHL)
}

//go:noescape
func sgrBlendRowNEON(ctx *sgrBlendCtx)

// boxsumNEON is the dispatch entry for the box sums. The horizontal extent is
// fixed (2r+1) for the interior columns [r, width-1-r]; those run through the
// NEON inner loop. The first r and last r columns have clamped windows and keep
// the scalar reference.
func boxsumNEON(src []int32, srcOrigin int, width int, height int, srcStride int, radius int, squared bool, dst []int32, dstStride int) {
	// The interior horizontal window for column col spans [col-r, col+r], fully
	// in bounds when r <= col <= width-1-r. NEON only helps when that interior
	// band is at least 4 columns wide.
	interiorStart := radius
	interiorEnd := width - 1 - radius // inclusive
	interiorCount := interiorEnd - interiorStart + 1
	if width <= 0 || height <= 0 || radius < 0 || interiorCount < 4 {
		boxsum(src, srcOrigin, width, height, srcStride, radius, squared, dst, dstStride)
		return
	}
	sq := uintptr(0)
	if squared {
		sq = 1
	}
	for row := 0; row < height; row++ {
		y0 := maxInt(0, row-radius)
		y1 := minInt(height-1, row+radius)
		dstRow := dst[row*dstStride : row*dstStride+width]
		// Edge columns [0, r) keep the scalar variable-bound reference.
		for col := 0; col < interiorStart; col++ {
			dstRow[col] = boxsumCell(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
		// Interior columns [r, width-1-r] are summed 4 lanes at a time. For
		// these the horizontal window is exactly [col-r, col+r].
		neonCols := interiorCount &^ 3 // round down to multiple of 4
		if neonCols > 0 {
			// Base of the first lane's window: row y0, column interiorStart-r.
			base := srcOrigin + y0*srcStride + (interiorStart - radius)
			// The asm walks the y0..y1 rows and the 2r+1 horizontal taps
			// internally, summing each interior column's window in registers
			// and storing once.
			boxsumInteriorBandNEON(&src[base], &dstRow[interiorStart], uintptr(neonCols), uintptr(radius), uintptr(y1-y0+1), uintptr(srcStride), sq)
		}
		// Remaining interior columns and the trailing r edge columns: scalar.
		for col := interiorStart + neonCols; col < width; col++ {
			dstRow[col] = boxsumCell(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
	}
}

// boxsumInteriorBandNEON sums the box-filter windows for the interior columns
// of one output row band. For each group of 4 interior columns it accumulates,
// in a 4-lane int32 register, the (squared) source values over all y0..y1 rows
// and all 2r+1 horizontal taps, then stores the four results once. Bit-exact
// with the scalar reference: int32 addition is associative under two's
// complement wraparound, so summing the identical set of (squared) values in a
// different grouping yields the same int32 result.
//
//go:noescape
func boxsumInteriorBandNEON(src *int32, dst *int32, neonCols uintptr, radius uintptr, rows uintptr, srcStride uintptr, squared uintptr)

// boxsumCell reproduces one scalar box-sum cell (used for the clamped edge
// columns). It mirrors boxsum exactly.
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

func selfguidedFastNEON(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const shiftEven = SGRProjSgrBits + 5 - SGRProjRstBits
	const shiftOdd = SGRProjSgrBits + 4 - SGRProjRstBits
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		neon := width &^ 3
		if row&1 == 0 {
			if neon > 0 {
				ctx := sgrBlendCtx{
					dst:   &dstRow[0],
					dgd:   &dgdRow[0],
					aPrev: &aBuf[k0-bufStride-1],
					aCur:  &aBuf[k0-1],
					aNext: &aBuf[k0+bufStride-1],
					bPrev: &bBuf[k0-bufStride-1],
					bCur:  &bBuf[k0-1],
					bNext: &bBuf[k0+bufStride-1],
					cols:  uintptr(neon),
					wPL:   5, wP: 6, wPR: 5,
					wCL: 0, wC: 0, wCR: 0,
					wNL: 5, wN: 6, wNR: 5,
					bias: 1 << (shiftEven - 1), shift: -shiftEven,
				}
				sgrBlendRowNEON(&ctx)
			}
			aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
			aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
			bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
			bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
			for col := neon; col < width; col++ {
				j := col + 1
				a := (aPrev[j]+aNext[j])*6 +
					(aPrev[j-1]+aPrev[j+1]+aNext[j-1]+aNext[j+1])*5
				b := (bPrev[j]+bNext[j])*6 +
					(bPrev[j-1]+bPrev[j+1]+bNext[j-1]+bNext[j+1])*5
				dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftEven)
			}
			continue
		}
		if neon > 0 {
			ctx := sgrBlendCtx{
				dst:   &dstRow[0],
				dgd:   &dgdRow[0],
				aPrev: &aBuf[k0-1],
				aCur:  &aBuf[k0-1],
				aNext: &aBuf[k0-1],
				bPrev: &bBuf[k0-1],
				bCur:  &bBuf[k0-1],
				bNext: &bBuf[k0-1],
				cols:  uintptr(neon),
				wPL:   0, wP: 0, wPR: 0,
				wCL: 5, wC: 6, wCR: 5,
				wNL: 0, wN: 0, wNR: 0,
				bias: 1 << (shiftOdd - 1), shift: -shiftOdd,
			}
			sgrBlendRowNEON(&ctx)
		}
		aCur := aBuf[k0-1 : k0+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		for col := neon; col < width; col++ {
			j := col + 1
			a := aCur[j]*6 + (aCur[j-1]+aCur[j+1])*5
			b := bCur[j]*6 + (bCur[j-1]+bCur[j+1])*5
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shiftOdd)
		}
	}
}

func selfguidedNEON(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediate(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	const nb = 5
	const shift = SGRProjSgrBits + nb - SGRProjRstBits
	for row := range height {
		k0 := aOrigin + row*bufStride
		dgdRow := dgd[dgdOrigin+row*dgdStride : dgdOrigin+row*dgdStride+width]
		dstRow := dst[row*dstStride : row*dstStride+width]
		neon := width &^ 3
		if neon > 0 {
			ctx := sgrBlendCtx{
				dst:   &dstRow[0],
				dgd:   &dgdRow[0],
				aPrev: &aBuf[k0-bufStride-1],
				aCur:  &aBuf[k0-1],
				aNext: &aBuf[k0+bufStride-1],
				bPrev: &bBuf[k0-bufStride-1],
				bCur:  &bBuf[k0-1],
				bNext: &bBuf[k0+bufStride-1],
				cols:  uintptr(neon),
				wPL:   3, wP: 4, wPR: 3,
				wCL: 4, wC: 4, wCR: 4,
				wNL: 3, wN: 4, wNR: 3,
				bias: 1 << (shift - 1), shift: -shift,
			}
			sgrBlendRowNEON(&ctx)
		}
		aPrev := aBuf[k0-bufStride-1 : k0-bufStride+width+1]
		aCur := aBuf[k0-1 : k0+width+1]
		aNext := aBuf[k0+bufStride-1 : k0+bufStride+width+1]
		bPrev := bBuf[k0-bufStride-1 : k0-bufStride+width+1]
		bCur := bBuf[k0-1 : k0+width+1]
		bNext := bBuf[k0+bufStride-1 : k0+bufStride+width+1]
		for col := neon; col < width; col++ {
			j := col + 1
			a := (aCur[j]+aCur[j-1]+aCur[j+1]+aPrev[j]+aNext[j])*4 +
				(aPrev[j-1]+aNext[j-1]+aPrev[j+1]+aNext[j+1])*3
			b := (bCur[j]+bCur[j-1]+bCur[j+1]+bPrev[j]+bNext[j])*4 +
				(bPrev[j-1]+bNext[j-1]+bPrev[j+1]+bNext[j+1])*3
			dstRow[col] = roundPowerOfTwo(a*dgdRow[col]+b, shift)
		}
	}
}
