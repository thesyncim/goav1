// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package restoration

import "simd/archsimd"

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
