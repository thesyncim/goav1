// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import (
	"encoding/binary"
	"simd/archsimd"
)

// init binds the Go-native AVX2 self-guided kernels on amd64 when AVX2 is
// available: the box sums and the per-pixel blend run eight lanes at a time. Without
// AVX2 the dispatch keeps the pure-Go reference. The data-dependent LUT gather
// inside calculateIntermediate stays scalar in both paths. The assignment happens
// once, before any decoder goroutine starts, so the steady-state cost is a single
// indirect call.
func init() {
	if archsimd.X86.AVX2() {
		boxsumImpl = boxsumSIMD
		selfguidedImpl = selfguidedSIMD
		selfguidedFastImpl = selfguidedFastSIMD
		return
	}
	boxsumImpl = boxsum
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}

// AVX2 Go-native SIMD self-guided kernels: the box sums, the nine-tap blend, and
// the 8-bit final projection. Eight int32 lanes per vector (one Int32x8) throughout.
// Every operation is the scalar reference's int32 operation, so the results are
// byte-identical; the 8-bit projection narrows with AVX2 packs only (no AVX-512).

// sgrBlendLanes is the number of int32 columns the blend processes per vector.
const sgrBlendLanes = 8

// boxsumSIMD is the Go-native SIMD box sum. The interior columns, whose horizontal
// window [col-r, col+r] is fully in bounds, are summed eight lanes at a time: each
// lane accumulates the (squared) source values over all rows y0..y1 and all 2r+1
// taps. The r clamped edge columns keep the scalar reference. int32 addition wraps
// the same way regardless of grouping, so the sums are bit-identical to boxsum.
func boxsumSIMD(src []int32, srcOrigin int, width int, height int, srcStride int, radius int, squared bool, dst []int32, dstStride int) {
	interiorStart := radius
	interiorEnd := width - 1 - radius // inclusive
	interiorCount := interiorEnd - interiorStart + 1
	if width <= 0 || height <= 0 || radius < 0 || interiorCount < sgrBlendLanes {
		boxsum(src, srcOrigin, width, height, srcStride, radius, squared, dst, dstStride)
		archsimd.ClearAVXUpperBits()
		return
	}
	for row := 0; row < height; row++ {
		y0 := maxInt(0, row-radius)
		y1 := minInt(height-1, row+radius)
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := 0; col < interiorStart; col++ {
			dstRow[col] = boxsumCell(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
		simdCols := interiorCount &^ (sgrBlendLanes - 1)
		for c := 0; c < simdCols; c += sgrBlendLanes {
			acc := archsimd.BroadcastInt32x8(0)
			for y := y0; y <= y1; y++ {
				// Lane L of this group is output column interiorStart+c+L; its tap k
				// reads source column (interiorStart+c+L-radius+k).
				rowBase := srcOrigin + y*srcStride + (interiorStart - radius) + c
				for k := 0; k <= 2*radius; k++ {
					v := archsimd.LoadInt32x8(src[rowBase+k:])
					if squared {
						v = v.Mul(v)
					}
					acc = acc.Add(v)
				}
			}
			acc.Store(dstRow[interiorStart+c:])
		}
		for col := interiorStart + simdCols; col < width; col++ {
			dstRow[col] = boxsumCell(src, srcOrigin, width, srcStride, radius, squared, col, y0, y1)
		}
	}
	archsimd.ClearAVXUpperBits()
}

// sgrBlendRowSIMD computes one blended self-guided output row for cols columns (a
// multiple of sgrBlendLanes). It evaluates dst[c] = roundPowerOfTwo(a[c]*dgd[c] +
// b[c], shift), where a and b are the nine-tap weighted 3x3 stencils over the A/B
// rows (see the arm64 kernel for the tap layout). Every slice is indexed from the
// column one to the left of the first output column.
func sgrBlendRowSIMD(dst []int32, dgd []int32, aPrev []int32, aCur []int32, aNext []int32, bPrev []int32, bCur []int32, bNext []int32, w *[9]int32, shift int, cols int) {
	var wv [9]archsimd.Int32x8
	for i := range wv {
		wv[i] = archsimd.BroadcastInt32x8(w[i])
	}
	biasV := archsimd.BroadcastInt32x8(int32(1) << (shift - 1))
	// vpsravd with a hoisted count: a variable vpsrad count is rematerialized
	// in the loop with a legacy-SSE movq, which stalls on dirty upper state.
	shiftV := archsimd.BroadcastUint32x8(uint32(shift))
	for c := 0; c < cols; c += sgrBlendLanes {
		a := sgrStencil8(aPrev, aCur, aNext, c, &wv)
		b := sgrStencil8(bPrev, bCur, bNext, c, &wv)
		r := a.Mul(loadI32x8(dgd, c)).Add(b).Add(biasV).ShiftRight(shiftV)
		r.Store(dst[c:])
	}
	archsimd.ClearAVXUpperBits()
}

// sgrStencil8 evaluates the nine-tap weighted 3x3 stencil for eight consecutive
// columns starting at c, using the weight vectors in wv.
func sgrStencil8(prev []int32, cur []int32, next []int32, c int, wv *[9]archsimd.Int32x8) archsimd.Int32x8 {
	return loadI32x8(prev, c).Mul(wv[0]).
		Add(loadI32x8(prev, c+1).Mul(wv[1])).
		Add(loadI32x8(prev, c+2).Mul(wv[2])).
		Add(loadI32x8(cur, c).Mul(wv[3])).
		Add(loadI32x8(cur, c+1).Mul(wv[4])).
		Add(loadI32x8(cur, c+2).Mul(wv[5])).
		Add(loadI32x8(next, c).Mul(wv[6])).
		Add(loadI32x8(next, c+1).Mul(wv[7])).
		Add(loadI32x8(next, c+2).Mul(wv[8]))
}

// loadI32x8 loads eight int32 values starting at p[i].
func loadI32x8(p []int32, i int) archsimd.Int32x8 {
	return archsimd.LoadInt32x8(p[i:])
}

// sgrConsts8 holds the loop-invariant vectors of the 8-bit projection.
type sgrConsts8 struct {
	xq0V, xq1V, cuV, bias archsimd.Int32x8
}

// newSGRConsts8 builds the projection constants. cuV = (1<<PrjBits) - xq0 - xq1
// turns u<<PrjBits + xq0*(f0-u) + xq1*(f1-u) into u*cuV + xq0*f0 + xq1*f1 exactly
// in wrapping int32 arithmetic.
func newSGRConsts8(xq0, xq1 int32) sgrConsts8 {
	return sgrConsts8{
		xq0V: archsimd.BroadcastInt32x8(xq0),
		xq1V: archsimd.BroadcastInt32x8(xq1),
		cuV:  archsimd.BroadcastInt32x8((1 << SGRProjPrjBits) - xq0 - xq1),
		bias: archsimd.BroadcastInt32x8(1 << (SGRProjPrjBits + SGRProjRstBits - 1)),
	}
}

// sgrWeightedRowU8SIMD is the Go-native SIMD form of sgrWeightedRowU8. Sixteen
// columns per iteration: one 16-byte load, two 8-lane projections, one 16-byte
// store. The trailing width%16 columns use the scalar reference.
func sgrWeightedRowU8SIMD(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	c := newSGRConsts8(xq0, xq1)
	lowBytes := archsimd.LoadInt8x16Array(&u8LowByteIdx)
	zeroBytes := archsimd.BroadcastUint8x16(0)
	col := 0
	for ; col <= len(dst)-16; col += 16 {
		sv := archsimd.LoadUint8x16(src[col:])
		// zeroBytes.ConcatShiftBytesRight(sv, 8) yields sv[8..15] in the low eight bytes.
		upper := zeroBytes.ConcatShiftBytesRight(sv, 8)
		outA := sgrWeightedU8Project8(sv.ExtendLo8ToUint32(), f0, f1, col, c, lowBytes)
		outB := sgrWeightedU8Project8(upper.ExtendLo8ToUint32(), f0, f1, col+8, c, lowBytes)
		joined := outA.ReshapeToUint64s().InterleaveLo(outB.ReshapeToUint64s()).ReshapeToUint8s()
		joined.StoreArray((*[16]uint8)(dst[col:]))
	}
	if col < width {
		sgrWeightedRowU8(dst[col:], src[col:], f0[col:], f1[col:], xq0, xq1)
	}
	archsimd.ClearAVXUpperBits()
}

// sgrWeightedU8Project8 projects eight samples (zero-extended to 32-bit lanes in s)
// against the flt rows at index i and returns the eight clamped bytes in the low
// 64 bits. The int16 wrap is a sign-extension of the low 16 bits, and the [0,255]
// clamp is a Max/Min pair before the AVX2 narrow.
func sgrWeightedU8Project8(s archsimd.Uint32x8, f0 []int32, f1 []int32, i int, c sgrConsts8, lowBytes archsimd.Int8x16) archsimd.Uint8x16 {
	u := s.ConvertToInt32().ShiftAllLeft(SGRProjRstBits)
	v := c.xq0V.Mul(loadI32x8(f0, i)).Add(c.xq1V.Mul(loadI32x8(f1, i))).
		Add(c.cuV.Mul(u)).Add(c.bias)
	r := v.ShiftAllRight(SGRProjPrjBits + SGRProjRstBits)
	w := r.ShiftAllLeft(16).ShiftAllRight(16)
	w = w.Max(archsimd.BroadcastInt32x8(0)).Min(archsimd.BroadcastInt32x8(255))
	return packClampedInt32x8ToUint16(w).ReshapeToUint8s().PermuteOrZero(lowBytes)
}

// sgrIntermediateAVX2 gates calculateIntermediateAVX2.
var sgrIntermediateAVX2 = archsimd.X86.AVX2()

// calculateIntermediateAVX2 is calculateIntermediate with the rolling
// producer in AVX2 form; the domain split is the scalar one.
func calculateIntermediateAVX2(dgd []int32, origin, width, height, stride, bitDepth, paramsIndex, radiusIndex, pass int, aBuf, bBuf []int32, bufStride int) {
	params := SGRParameterTable[paramsIndex]
	radius := int(params.Radius[radiusIndex])
	if sgrIntermediateAVX2 && validSGRBlock(width, height) && (radius == 1 || radius == 2) &&
		(bitDepth == 8 || bitDepth == 10 || bitDepth == 12) {
		calculateIntermediateRollingAVX2(dgd, origin, width, height, stride, bitDepth, params, radiusIndex, pass, aBuf, bBuf, bufStride)
		return
	}
	calculateIntermediate(dgd, origin, width, height, stride, bitDepth, paramsIndex, radiusIndex, pass, aBuf, bBuf, bufStride)
}

// sgrRollRows adds sign*row to the column sums and sign*row^2 to the column
// squares, eight columns per step (dav1d's sgr_box*_v), over len(sums).
func sgrRollRows(sums, squares, in, out []int32) {
	n := len(sums)
	squares = squares[:n]
	in = in[:n]
	col := 0
	if out == nil {
		for ; col+8 <= n; col += 8 {
			v := archsimd.LoadInt32x8(in[col:])
			archsimd.LoadInt32x8(sums[col:]).Add(v).Store(sums[col:])
			archsimd.LoadInt32x8(squares[col:]).Add(v.Mul(v)).Store(squares[col:])
		}
		for ; col < n; col++ {
			v := in[col]
			sums[col] += v
			squares[col] += v * v
		}
		return
	}
	out = out[:n]
	for ; col+8 <= n; col += 8 {
		v := archsimd.LoadInt32x8(in[col:])
		u := archsimd.LoadInt32x8(out[col:])
		archsimd.LoadInt32x8(sums[col:]).Add(v.Sub(u)).Store(sums[col:])
		archsimd.LoadInt32x8(squares[col:]).Add(v.Mul(v).Sub(u.Mul(u))).Store(squares[col:])
	}
	for ; col < n; col++ {
		v, u := in[col], out[col]
		sums[col] += v - u
		squares[col] += v*v - u*u
	}
}

// calculateIntermediateRollingAVX2 is calculateIntermediateRolling with the
// column-sum updates and the A/B conversion eight columns at a time (dav1d's
// sgr_box3/5 and calc_ab in looprestoration_avx2.asm). The conversion keeps
// the scalar wrapping uint32 arithmetic: p = max(a*n, b*b) - b*b is the
// scalar clamp at zero, and the x/(x+1) table lookup stays scalar (AVX2
// has vpgatherdd, archsimd does not). Columns past the last full vector take the scalar code.
func calculateIntermediateRollingAVX2(dgd []int32, origin, width, height, stride, bitDepth int, params SGRParams, radiusIndex, pass int, aBuf, bBuf []int32, bufStride int) {
	r := int(params.Radius[radiusIndex])
	span := width + 2*r + 2
	var sumStorage, squareStorage [ProcUnitSize + 2*SGRProjBorderHorz]int32
	sums := sumStorage[:span]
	squares := squareStorage[:span]
	first := origin + (-1-r)*stride - 1 - r
	clear(sums)
	clear(squares)
	for y := 0; y <= 2*r; y++ {
		sgrRollRows(sums, squares, dgd[first+y*stride:first+y*stride+span], nil)
	}
	step := 1
	if pass != 0 {
		step = 2
	}
	n := (2*r + 1) * (2*r + 1)
	scale := uint32(params.S[radiusIndex])
	recip := uint32(oneByX[n-1])
	aOrigin := SGRProjBorderVert*bufStride + SGRProjBorderHorz
	vn, vs := archsimd.BroadcastUint32x8(uint32(n)), archsimd.BroadcastUint32x8(scale)
	vr, vSgr := archsimd.BroadcastUint32x8(recip), archsimd.BroadcastUint32x8(SGRProjSgr)
	vzRound := archsimd.BroadcastUint32x8(1 << (SGRProjMTableBits - 1))
	vzMax := archsimd.BroadcastUint32x8(255)
	vbrRound := archsimd.BroadcastUint32x8(1 << (SGRProjRecipBits - 1))
	// A vector starts at col <= vecCols-8; its last input is
	// col+2r+7 <= vecCols+2r-1 <= width+2r+1 == span-1.
	vecCols := (width + 2) &^ 7
	var zBuf [8]uint32
	for y := -1; y < height+1; y += step {
		ar := aBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		br := bBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		for col := 0; col < vecCols; col += 8 {
			sum := archsimd.LoadInt32x8(sums[col:]).Add(archsimd.LoadInt32x8(sums[col+1:])).Add(archsimd.LoadInt32x8(sums[col+2:]))
			square := archsimd.LoadInt32x8(squares[col:]).Add(archsimd.LoadInt32x8(squares[col+1:])).Add(archsimd.LoadInt32x8(squares[col+2:]))
			if r == 2 {
				sum = sum.Add(archsimd.LoadInt32x8(sums[col+3:])).Add(archsimd.LoadInt32x8(sums[col+4:]))
				square = square.Add(archsimd.LoadInt32x8(squares[col+3:])).Add(archsimd.LoadInt32x8(squares[col+4:]))
			}
			a, b := square, sum
			// roundPowerOfTwo by 2*(bitDepth-8) and bitDepth-8, with
			// immediate shift counts.
			switch bitDepth {
			case 10:
				a = a.Add(archsimd.BroadcastInt32x8(1 << 3)).ShiftAllRight(4)
				b = b.Add(archsimd.BroadcastInt32x8(1 << 1)).ShiftAllRight(2)
			case 12:
				a = a.Add(archsimd.BroadcastInt32x8(1 << 7)).ShiftAllRight(8)
				b = b.Add(archsimd.BroadcastInt32x8(1 << 3)).ShiftAllRight(4)
			}
			au, bu := a.AsUint32x8(), b.AsUint32x8()
			bb := bu.Mul(bu)
			p := au.Mul(vn).Max(bb).Sub(bb)
			p.Mul(vs).Add(vzRound).ShiftAllRight(SGRProjMTableBits).Min(vzMax).StoreArray(&zBuf)
			// The lookups go back in with lane inserts: a vector load of
			// eight just-stored scalars would miss store forwarding.
			var lo, hi archsimd.Int32x4
			lo = lo.SetElem(0, int32(xByXPlus1[uint8(zBuf[0])]))
			lo = lo.SetElem(1, int32(xByXPlus1[uint8(zBuf[1])]))
			lo = lo.SetElem(2, int32(xByXPlus1[uint8(zBuf[2])]))
			lo = lo.SetElem(3, int32(xByXPlus1[uint8(zBuf[3])]))
			hi = hi.SetElem(0, int32(xByXPlus1[uint8(zBuf[4])]))
			hi = hi.SetElem(1, int32(xByXPlus1[uint8(zBuf[5])]))
			hi = hi.SetElem(2, int32(xByXPlus1[uint8(zBuf[6])]))
			hi = hi.SetElem(3, int32(xByXPlus1[uint8(zBuf[7])]))
			var aa archsimd.Int32x8
			aa = aa.SetLo(lo).SetHi(hi)
			aa.Store(ar[col:])
			vSgr.Sub(aa.AsUint32x8()).Mul(sum.AsUint32x8()).Mul(vr).Add(vbrRound).ShiftAllRight(SGRProjRecipBits).AsInt32x8().Store(br[col:])
		}
		sum, square := int32(0), int32(0)
		if vecCols < width+2 {
			for col := vecCols; col <= vecCols+2*r; col++ {
				sum += sums[col]
				square += squares[col]
			}
		}
		for col := vecCols; col < width+2; col++ {
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
			sgrRollRows(sums, squares, dgd[inStart:inStart+span], dgd[outStart:outStart+span])
		}
	}
	archsimd.ClearAVXUpperBits()
}

// Go-native SIMD self-guided restoration drivers for amd64. The
// per-pixel blend (sgrBlendRowSIMD) and the lane count (sgrBlendLanes) are
// supplied by the architecture file. The data-dependent LUT gather inside
// calculateIntermediateAVX2 (xByXPlus1 / oneByX) stays scalar, and the columns beyond
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
// radius-1 and radius-2 passes. Its A/B coefficients come from
// calculateIntermediateAVX2 with the parameter pass selected by the radius
// index.
func selfguidedSIMD(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediateAVX2(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
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
	calculateIntermediateAVX2(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
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

// init binds the Go-native AVX2 8-bit-pixel self-guided projection on amd64 when
// AVX2 is available. Without AVX2 the dispatch keeps the pure-Go reference. The
// 8-bit Wiener passes are bound in wiener_dispatch_amd64.go.
func init() {
	if archsimd.X86.AVX2() {
		sgrWeightedRowU8Impl = sgrWeightedRowU8SIMD
		return
	}
	sgrWeightedRowU8Impl = sgrWeightedRowU8
}

// init binds the Go-native SIMD Wiener passes on amd64 when AVX2 is available.
// Without AVX2 the dispatch keeps the pure-Go reference. The assignment happens
// once, before any decoder goroutine starts, so the steady-state cost is a single
// indirect call. Every binding is byte-identical to the pure-Go reference.
func init() {

	if archsimd.X86.AVX2() {
		wienerHorizontalTrustedImpl = wienerHorizontalSIMDTrusted
		wienerVerticalImpl = wienerVerticalSIMD
		wienerHorizontalU8Impl = wienerHorizontalU8SIMDChecked
		wienerVerticalU8Impl = wienerVerticalU8SIMD
		return
	}
	wienerHorizontalTrustedImpl = wienerHorizontalTrusted
	wienerVerticalImpl = wienerVertical
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl = wienerVerticalU8
}

// AVX2 Go-native SIMD horizontal Wiener passes. Eight output columns are
// computed per iteration in 32-bit lanes (one Int32x8), which is exact for every
// bit depth: the widest intermediate is a symmetric pair sum (<= 8190 for the
// 12-bit domain) times a tap, plus the libaom offset, far inside int32.
//
// The seven taps fold into four products exactly as in the arm64 kernel: the
// Wiener filter is symmetric (f0==f6, f1==f5, f2==f4, enforced by
// validWienerInfo), the libaom center reapplication s3<<WienerFilterBits is
// folded into the center tap (f3' = f3 + 128), and
//
//	sum = seed + f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3'*s3
//
// where seed folds the libaom offset and the rounding bias 1<<(round0-1). The
// trailing arithmetic shift and the [0,maxClamp] clamp match
// roundPowerOfTwo + clampInt32 bit for bit.

// wienerHorizontalSIMDTrusted is the uint16 AVX2 horizontal pass without the
// per-sample validity scan, used by the decoder-owned trusted entry.
func wienerHorizontalSIMDTrusted(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	if width < 8 || width%8 != 0 || !wienerFilterSymmetric(filter) {
		wienerHorizontalTrusted(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
		return
	}
	wienerHorizontalSIMDKernel(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
}

// wienerHorizontalSIMDKernel is the uint16 AVX2 horizontal pass for widths that
// are a multiple of 8. Each 8-lane window load is exactly eight samples, so the
// slice bounds checks are the only guard needed; no trailing pad is read.
func wienerHorizontalSIMDKernel(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	limit := int32(1) << (bitDepth + 1 + WienerFilterBits - round0)
	offset := int32(1) << (bitDepth + WienerFilterBits - 1)
	seedV := archsimd.BroadcastInt32x8(offset + roundBias(round0))
	zero := archsimd.BroadcastInt32x8(0)
	maxV := archsimd.BroadcastInt32x8(limit - 1)
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	// The center tap absorbs the s3<<WienerFilterBits center reapplication.
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	// A variable vpsrad count is rematerialized in the loop with a
	// legacy-SSE movq; vpsravd takes a hoisted count vector instead.
	shift := archsimd.BroadcastUint32x8(uint32(round0))
	rows := height + 2*WienerHalfwin
	for row := 0; row < rows; row++ {
		srow := src[srcOrigin+(row-WienerHalfwin)*srcStride-WienerHalfwin:]
		dstRow := temp[row*width : row*width+width]
		for col := 0; col < width; col += 8 {
			s0 := archsimd.LoadUint16x8(srow[col:]).ExtendToUint32().AsInt32x8()
			s1 := archsimd.LoadUint16x8(srow[col+1:]).ExtendToUint32().AsInt32x8()
			s2 := archsimd.LoadUint16x8(srow[col+2:]).ExtendToUint32().AsInt32x8()
			s3 := archsimd.LoadUint16x8(srow[col+3:]).ExtendToUint32().AsInt32x8()
			s4 := archsimd.LoadUint16x8(srow[col+4:]).ExtendToUint32().AsInt32x8()
			s5 := archsimd.LoadUint16x8(srow[col+5:]).ExtendToUint32().AsInt32x8()
			s6 := archsimd.LoadUint16x8(srow[col+6:]).ExtendToUint32().AsInt32x8()
			sum := seedV.Add(s0.Add(s6).Mul(f0)).Add(s1.Add(s5).Mul(f1)).
				Add(s2.Add(s4).Mul(f2)).Add(s3.Mul(f3))
			out := sum.ShiftRight(shift).Max(zero).Min(maxV)
			packClampedInt32x8ToUint16(out).Store(dstRow[col:])
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none, and
	// dirty upper ymm state slows the legacy-SSE code that follows.
	archsimd.ClearAVXUpperBits()
}

// wienerHorizontalU8SIMD is the AVX2 form of wienerHorizontalU8 for 8-bit source
// samples. Each 8-output group loads one 16-byte window and derives the six
// shifted windows in-register with ConcatShiftBytesRight against zero; the
// caller must have checked wienerHorizontalU8SIMDCanRun so the 16-byte load
// stays inside the padded source.
func wienerHorizontalU8SIMD(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if width < 8 || width%8 != 0 || round0 != WienerRound0Bits || !wienerFilterSymmetric(filter) {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	const bitDepth = 8
	limit := int32(1) << (bitDepth + 1 + WienerFilterBits - WienerRound0Bits)
	offset := int32(1) << (bitDepth + WienerFilterBits - 1)
	// The 16-wide columns run first, in their own loop, so no 128-bit vector
	// of the 8-wide tail below is live across the calls.
	w16 := 0
	if wienerU8Wide16 {
		w16 = width &^ 15
		k16 := newWienerU8Taps(filter, offset+roundBias(WienerRound0Bits), limit-1)
		for row := 0; row < height+2*WienerHalfwin; row++ {
			srow := src[srcOrigin+(row-WienerHalfwin)*srcStride-WienerHalfwin:]
			dstRow := temp[row*width : row*width+width]
			for col := 0; col < w16; col += 16 {
				wienerHorizontalU8x16(srow[col:col+24], dstRow[col:col+16], &k16)
			}
		}
	}
	seedV := archsimd.BroadcastInt32x8(offset + roundBias(WienerRound0Bits))
	zero := archsimd.BroadcastInt32x8(0)
	zeroBytes := archsimd.BroadcastUint8x16(0)
	maxV := archsimd.BroadcastInt32x8(limit - 1)
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	shift := uint64(WienerRound0Bits)
	rows := height + 2*WienerHalfwin
	for row := 0; row < rows; row++ {
		srow := src[srcOrigin+(row-WienerHalfwin)*srcStride-WienerHalfwin:]
		dstRow := temp[row*width : row*width+width]
		for col := w16; col < width; col += 8 {
			v0 := archsimd.LoadUint8x16(srow[col:]) // s0..s15 of this group
			// Window k is bytes k..k+7 of v0: shift v0 down by k bytes, zero filling
			// the top (zeroBytes is the high operand, v0 the low one).
			s0 := v0.ExtendLo8ToUint32().AsInt32x8()
			s1 := zeroBytes.ConcatShiftBytesRight(v0, 1).ExtendLo8ToUint32().AsInt32x8()
			s2 := zeroBytes.ConcatShiftBytesRight(v0, 2).ExtendLo8ToUint32().AsInt32x8()
			s3 := zeroBytes.ConcatShiftBytesRight(v0, 3).ExtendLo8ToUint32().AsInt32x8()
			s4 := zeroBytes.ConcatShiftBytesRight(v0, 4).ExtendLo8ToUint32().AsInt32x8()
			s5 := zeroBytes.ConcatShiftBytesRight(v0, 5).ExtendLo8ToUint32().AsInt32x8()
			s6 := zeroBytes.ConcatShiftBytesRight(v0, 6).ExtendLo8ToUint32().AsInt32x8()
			sum := seedV.Add(s0.Add(s6).Mul(f0)).Add(s1.Add(s5).Mul(f1)).
				Add(s2.Add(s4).Mul(f2)).Add(s3.Mul(f3))
			out := sum.ShiftAllRight(shift).Max(zero).Min(maxV)
			packClampedInt32x8ToUint16(out).Store(dstRow[col:])
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none, and
	// dirty upper ymm state slows the legacy-SSE code that follows.
	archsimd.ClearAVXUpperBits()
}

// packClampedInt32x8ToUint16 narrows eight int32 lanes already clamped to
// [0,65535] to uint16 with VPACKUSDW (AVX2). Truncating forms (VPMOVDW) need
// AVX-512, so the pack goes through two 128-bit halves.
func packClampedInt32x8ToUint16(v archsimd.Int32x8) archsimd.Uint16x8 {
	return v.GetLo().SaturateToUint16Concat(v.GetHi())
}

// wienerU8Taps holds the dav1d-style pmaddwd tap pairs of the 16-wide 8-bit
// Wiener kernels.
type wienerU8Taps struct {
	f01, f23, f45, f6 [16]int16 // (f0, f1), (f2, f3'), (f4, f5), (f6, 0) pairs
	seed, max         int32
}

func newWienerU8Taps(filter WienerFilter, seed, max int32) wienerU8Taps {
	k := wienerU8Taps{seed: seed, max: max}
	f3 := int16(int32(filter[3]) + (1 << WienerFilterBits))
	for i := 0; i < 16; i += 2 {
		k.f01[i], k.f01[i+1] = int16(filter[0]), int16(filter[1])
		k.f23[i], k.f23[i+1] = int16(filter[2]), f3
		k.f45[i], k.f45[i+1] = int16(filter[4]), int16(filter[5])
		k.f6[i] = int16(filter[6])
	}
	return k
}

// wienerHorizontalU8x16 is the horizontal pass for sixteen outputs, as
// dav1d's wiener_filter7_8bpc AVX2 forms it: the two 128-bit halves hold
// outputs 0..7 and 8..15, the seven windows come from vpalignr on the
// widened samples s0..s15 and s8..s23, the symmetric pair sums go through
// vpmaddwd against the (f0, f1) and (f2, f3') pairs (exact in int32, every
// operand fits int16), and the clamped result packs back with vpackssdw in
// the order the in-lane unpack produced.
func wienerHorizontalU8x16(s []uint8, dst []uint16, k *wienerU8Taps) {
	a := archsimd.LoadUint8x16(s[0:16]).ExtendToUint16().AsUint8x32()
	b := archsimd.LoadUint8x16(s[8:24]).ExtendToUint16().AsUint8x32()
	w := func(sh uint64) archsimd.Int16x16 { return b.ConcatShiftBytesRightGrouped(a, sh).AsInt16x16() }
	p0 := a.AsInt16x16().Add(w(12))
	p1 := w(2).Add(w(10))
	p2 := w(4).Add(w(8))
	p3 := w(6)
	f01 := archsimd.LoadInt16x16Array(&k.f01)
	f23 := archsimd.LoadInt16x16Array(&k.f23)
	seed := archsimd.BroadcastInt32x8(k.seed)
	zero := archsimd.Int32x8{}
	mx := archsimd.BroadcastInt32x8(k.max)
	lo := p0.InterleaveLoGrouped(p1).DotProductPairs(f01).Add(p2.InterleaveLoGrouped(p3).DotProductPairs(f23)).Add(seed)
	hi := p0.InterleaveHiGrouped(p1).DotProductPairs(f01).Add(p2.InterleaveHiGrouped(p3).DotProductPairs(f23)).Add(seed)
	lo = lo.ShiftAllRight(WienerRound0Bits).Max(zero).Min(mx)
	hi = hi.ShiftAllRight(WienerRound0Bits).Max(zero).Min(mx)
	lo.SaturateToInt16ConcatGrouped(hi).AsUint16x16().Store(dst[:16])
	archsimd.ClearAVXUpperBits()
}

// AVX2 Go-native SIMD vertical Wiener passes. Eight output columns are computed
// per iteration in 32-bit lanes as a plain seven-tap MAC, which is exact for any
// filter (the vertical taps are not assumed symmetric) because every product and
// the sum stay far inside int32 for the whole temp range (samples <= 32767). The
// rounding bias and -offset seed the accumulator so the trailing arithmetic
// shift reproduces roundPowerOfTwo(sum, round1), and the [0,max] clamp matches
// clampInt32 bit for bit.

// wienerVerticalSIMD is the AVX2 form of wienerVertical for the uint16 output
// domain. Widths that are not a multiple of 8 use the scalar reference.
func wienerVerticalSIMD(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	if width < 8 || width%8 != 0 {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
		return
	}
	offset := int32(1) << (bitDepth + round1 - 1)
	seedV := archsimd.BroadcastInt32x8(-offset + roundBias(round1))
	zero := archsimd.BroadcastInt32x8(0)
	maxV := archsimd.BroadcastInt32x8(int32(max))
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	// The center tap absorbs the s3<<WienerFilterBits center reapplication; the
	// vertical taps are not assumed symmetric, so all seven are kept.
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	f4 := archsimd.BroadcastInt32x8(int32(filter[4]))
	f5 := archsimd.BroadcastInt32x8(int32(filter[5]))
	f6 := archsimd.BroadcastInt32x8(int32(filter[6]))
	// vpsravd with a hoisted count vector (see wienerHorizontalSIMDKernel).
	shift := archsimd.BroadcastUint32x8(uint32(round1))
	for row := 0; row < height; row++ {
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := 0; col < width; col += 8 {
			r0 := archsimd.LoadUint16x8(temp[(row+0)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r1 := archsimd.LoadUint16x8(temp[(row+1)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r2 := archsimd.LoadUint16x8(temp[(row+2)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r3 := archsimd.LoadUint16x8(temp[(row+3)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r4 := archsimd.LoadUint16x8(temp[(row+4)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r5 := archsimd.LoadUint16x8(temp[(row+5)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r6 := archsimd.LoadUint16x8(temp[(row+6)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			sum := seedV.Add(r0.Mul(f0)).Add(r1.Mul(f1)).Add(r2.Mul(f2)).Add(r3.Mul(f3)).
				Add(r4.Mul(f4)).Add(r5.Mul(f5)).Add(r6.Mul(f6))
			out := sum.ShiftRight(shift).Max(zero).Min(maxV)
			packClampedInt32x8ToUint16(out).Store(dstRow[col:])
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none, and
	// dirty upper ymm state slows the legacy-SSE code that follows.
	archsimd.ClearAVXUpperBits()
}

// wienerVerticalU8SIMD is the AVX2 form of wienerVerticalU8: the same vertical
// pass with the output clamped to [0,255] and packed to uint8. Eight bytes are
// stored per group; widths that are not a multiple of 8 use the scalar reference.
func wienerVerticalU8SIMD(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	// The 8-bit round1 is the constant wienerRound1U8, so every shift below
	// is an immediate: a variable ymm shift count is rematerialized inside
	// the loop with a legacy-SSE movq, which stalls on dirty upper state.
	if width < 8 || width%8 != 0 || round1 != wienerRound1U8 {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	const bitDepth = 8
	offset := int32(1) << (bitDepth + round1 - 1)
	// All columns run the vpmaddwd kernel (an eight-column tail with its
	// half-width form); the int32 8-wide loop below is kept only as the
	// benchmark baseline (wienerU8Wide16 off). On Zen 3 that loop stalls on a
	// legacy-SSE reload of its 128-bit byte selector every iteration.
	if wienerU8Wide16 {
		k16 := newWienerU8Taps(filter, -offset+roundBias(round1), 255)
		for row := 0; row < height; row++ {
			dstRow := dst[row*dstStride : row*dstStride+width]
			col := 0
			for ; col+16 <= width; col += 16 {
				wienerVerticalU8x16(temp[row*tempStride+col:], tempStride, dstRow[col:col+16], &k16, false)
			}
			if col < width {
				wienerVerticalU8x16(temp[row*tempStride+col:], tempStride, dstRow[col:col+8], &k16, true)
			}
		}
		archsimd.ClearAVXUpperBits()
		return
	}
	const w16 = 0
	seedV := archsimd.BroadcastInt32x8(-offset + roundBias(round1))
	zero := archsimd.BroadcastInt32x8(0)
	maxV := archsimd.BroadcastInt32x8(255)
	// PSHUFB index vector: the low byte of each 16-bit lane. The clamped values
	// are already in [0,255], so the low bytes of the first eight lanes are the
	// exact uint8 results (VPMOVUSWB, the saturating narrow, needs AVX-512).
	lowBytes := archsimd.LoadInt8x16Array(&u8LowByteIdx)
	f0 := archsimd.BroadcastInt32x8(int32(filter[0]))
	f1 := archsimd.BroadcastInt32x8(int32(filter[1]))
	f2 := archsimd.BroadcastInt32x8(int32(filter[2]))
	// The center tap absorbs the s3<<WienerFilterBits center reapplication; the
	// vertical taps are not assumed symmetric, so all seven are kept.
	f3 := archsimd.BroadcastInt32x8(int32(filter[3]) + (1 << WienerFilterBits))
	f4 := archsimd.BroadcastInt32x8(int32(filter[4]))
	f5 := archsimd.BroadcastInt32x8(int32(filter[5]))
	f6 := archsimd.BroadcastInt32x8(int32(filter[6]))
	const shift = wienerRound1U8
	for row := 0; row < height; row++ {
		dstRow := dst[row*dstStride : row*dstStride+width]
		for col := w16; col < width; col += 8 {
			r0 := archsimd.LoadUint16x8(temp[(row+0)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r1 := archsimd.LoadUint16x8(temp[(row+1)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r2 := archsimd.LoadUint16x8(temp[(row+2)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r3 := archsimd.LoadUint16x8(temp[(row+3)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r4 := archsimd.LoadUint16x8(temp[(row+4)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r5 := archsimd.LoadUint16x8(temp[(row+5)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			r6 := archsimd.LoadUint16x8(temp[(row+6)*tempStride+col:]).ExtendToUint32().AsInt32x8()
			sum := seedV.Add(r0.Mul(f0)).Add(r1.Mul(f1)).Add(r2.Mul(f2)).Add(r3.Mul(f3)).
				Add(r4.Mul(f4)).Add(r5.Mul(f5)).Add(r6.Mul(f6))
			out := packClampedInt32x8ToUint16(sum.ShiftAllRight(shift).Max(zero).Min(maxV)).ReshapeToUint8s().PermuteOrZero(lowBytes)
			binary.LittleEndian.PutUint64(dstRow[col:col+8], out.ReshapeToUint64s().GetElem(0))
		}
	}
	// dav1d returns through vzeroupper; the Go compiler emits none, and
	// dirty upper ymm state slows the legacy-SSE code that follows.
	archsimd.ClearAVXUpperBits()
}

// u8LowByteIdx is the PSHUFB selector that gathers the low byte of each of the
// first eight 16-bit lanes into the low eight bytes of the result.
var u8LowByteIdx = [16]int8{0, 2, 4, 6, 8, 10, 12, 14}

// wienerVerticalU8x16 is the vertical pass for sixteen outputs of one row:
// the seven temp rows (<= 8191, so int16) pair up through vpmaddwd against
// the (f0, f1), (f2, f3'), (f4, f5) and (f6, 0) taps, exact in int32; the
// clamped sums pack back in order (in-lane unpack, in-lane vpackssdw) and
// vpshufb plus a 64-bit interleave narrow them to bytes.
func wienerVerticalU8x16(t []uint16, stride int, dst []uint8, k *wienerU8Taps, n8 bool) {
	const shift = wienerRound1U8
	ld := func(i int) archsimd.Int16x16 {
		if n8 {
			return archsimd.Int16x16{}.SetLo(archsimd.LoadUint16x8(t[i*stride : i*stride+8]).AsInt16x8())
		}
		return archsimd.LoadUint16x16(t[i*stride : i*stride+16]).AsInt16x16()
	}
	r0, r1, r2, r3, r4, r5, r6 := ld(0), ld(1), ld(2), ld(3), ld(4), ld(5), ld(6)
	var z archsimd.Int16x16
	f01 := archsimd.LoadInt16x16Array(&k.f01)
	f23 := archsimd.LoadInt16x16Array(&k.f23)
	f45 := archsimd.LoadInt16x16Array(&k.f45)
	f6 := archsimd.LoadInt16x16Array(&k.f6)
	seed := archsimd.BroadcastInt32x8(k.seed)
	lo := r0.InterleaveLoGrouped(r1).DotProductPairs(f01).Add(r2.InterleaveLoGrouped(r3).DotProductPairs(f23)).
		Add(r4.InterleaveLoGrouped(r5).DotProductPairs(f45)).Add(r6.InterleaveLoGrouped(z).DotProductPairs(f6)).Add(seed)
	hi := r0.InterleaveHiGrouped(r1).DotProductPairs(f01).Add(r2.InterleaveHiGrouped(r3).DotProductPairs(f23)).
		Add(r4.InterleaveHiGrouped(r5).DotProductPairs(f45)).Add(r6.InterleaveHiGrouped(z).DotProductPairs(f6)).Add(seed)
	zero := archsimd.Int32x8{}
	mx := archsimd.BroadcastInt32x8(255)
	lo = lo.ShiftAllRight(shift).Max(zero).Min(mx)
	hi = hi.ShiftAllRight(shift).Max(zero).Min(mx)
	v := lo.SaturateToInt16ConcatGrouped(hi).AsUint8x32().PermuteOrZeroGrouped(archsimd.LoadInt8x32Array(&wienerLowBytes32))
	if n8 {
		binary.LittleEndian.PutUint64(dst[:8], v.GetLo().AsUint64x2().GetElem(0))
	} else {
		v.GetLo().AsUint64x2().InterleaveLo(v.GetHi().AsUint64x2()).AsUint8x16().Store(dst[:16])
	}
	archsimd.ClearAVXUpperBits()
}

// wienerLowBytes32 gathers the low byte of each 16-bit lane into the low
// half of its 128-bit lane.
var wienerLowBytes32 = [32]int8{
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
}

// wienerU8Wide16 selects the 16-wide kernels (benchmarks compare both).
var wienerU8Wide16 = true

// wienerRound1U8 is the vertical rounding shift at 8 bits (wienerRounds(8)).
const wienerRound1U8 = 2*WienerFilterBits - WienerRound0Bits
