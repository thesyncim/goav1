// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"simd/archsimd"
	"unsafe"
)

// calculateIntermediateSIMD retains the scalar dispatch domain. The ARM64
// SIMD drivers alone use this producer; selfguided and selfguidedFast keep
// calculateIntermediate as the independent scalar coefficient route.
func calculateIntermediateSIMD(dgd []int32, origin, width, height, stride, bitDepth, paramsIndex, radiusIndex, pass int, aBuf, bBuf []int32, bufStride int) {
	params := SGRParameterTable[paramsIndex]
	radius := int(params.Radius[radiusIndex])
	if validSGRBlock(width, height) && (radius == 1 || radius == 2) {
		calculateIntermediateRollingSIMD(dgd, origin, width, height, stride, bitDepth, params, radiusIndex, pass, aBuf, bBuf, bufStride)
		return
	}
	calculateIntermediateFull(dgd, origin, width, height, stride, bitDepth, paramsIndex, radiusIndex, pass, aBuf, bBuf, bufStride)
}

// calculateIntermediateRollingSIMD uses the same rolling column sums, row
// schedule and written A/B cells as calculateIntermediateRolling. Four adjacent
// horizontal windows are evaluated together, then converted with wrapping
// int32/uint32 arithmetic. The LUT loads remain scalar and use the live table.
// In particular, B uses the raw sum, not its bit-depth-scaled variance input.
//
// The SIMD drivers are reached after public validation/staging: dgd, A and B
// are disjoint partitions of the caller's int32 workspace. Public src/dst may
// overlap because staging completes before either filter runs. No padding or
// skipped row is written here. The scalar producer remains the reference.
func calculateIntermediateRollingSIMD(dgd []int32, origin, width, height, stride, bitDepth int, params SGRParams, radiusIndex, pass int, aBuf, bBuf []int32, bufStride int) {
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
	shiftA, shiftB := max(2*(bitDepth-8), 0), max(bitDepth-8, 0)
	roundA, roundB := int32(0), int32(0)
	if shiftA > 0 {
		roundA = int32(1 << (shiftA - 1))
	}
	if shiftB > 0 {
		roundB = int32(1 << (shiftB - 1))
	}
	vaRound, vbRound := archsimd.BroadcastInt32x4(roundA), archsimd.BroadcastInt32x4(roundB)
	vaShift, vbShift := archsimd.BroadcastInt32x4(-int32(shiftA)), archsimd.BroadcastInt32x4(-int32(shiftB))
	vn, vs := archsimd.BroadcastUint32x4(uint32(n)), archsimd.BroadcastUint32x4(scale)
	vr, vSgr := archsimd.BroadcastUint32x4(recip), archsimd.BroadcastUint32x4(SGRProjSgr)
	vzRound := archsimd.BroadcastUint32x4(1 << (SGRProjMTableBits - 1))
	vzMax := archsimd.BroadcastUint32x4(255)
	vbrRound := archsimd.BroadcastUint32x4(1 << (SGRProjRecipBits - 1))
	vzShift := archsimd.BroadcastInt32x4(-SGRProjMTableBits)
	vbrShift := archsimd.BroadcastInt32x4(-SGRProjRecipBits)
	vecCols := (width + 2) &^ 3
	sp, qp := unsafe.Pointer(&sumStorage[0]), unsafe.Pointer(&squareStorage[0])
	for y := -1; y < height+1; y += step {
		ar := aBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		br := bBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		// A vector starts at col <= vecCols-4. Its last input is
		// col+2*r+3 <= vecCols+2*r-1 <= width+2*r+1 == span-1.
		// span <= 70 is checked by the slices above; r is 1 or 2.
		// The two stores end at col+3 < vecCols <= len(ar)==len(br).
		// All pointers stay within these live, disjoint int32 arrays; ARM64
		// vector loads permit their ordinary int32 alignment.
		ap, bp := unsafe.Pointer(unsafe.SliceData(ar)), unsafe.Pointer(unsafe.SliceData(br))
		for col := 0; col < vecCols; col += 4 {
			s, q := unsafe.Add(sp, 4*col), unsafe.Add(qp, 4*col)
			sum := archsimd.LoadInt32x4Array((*[4]int32)(s)).Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(s, 4)))).Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(s, 8))))
			square := archsimd.LoadInt32x4Array((*[4]int32)(q)).Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 4)))).Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 8))))
			if r == 2 {
				sum = sum.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(s, 12)))).Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(s, 16))))
				square = square.Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 12)))).Add(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(q, 16))))
			}
			a := square.Add(vaRound).Shift(vaShift).ToBits()
			b := sum.Add(vbRound).Shift(vbShift).ToBits()
			p := a.Mul(vn).SubSaturated(b.Mul(b))
			z := p.Mul(vs).Add(vzRound).Shift(vzShift).Min(vzMax)
			var aa archsimd.Uint32x4
			aa = aa.SetElem(0, uint32(xByXPlus1[uint8(z.GetElem(0))]))
			aa = aa.SetElem(1, uint32(xByXPlus1[uint8(z.GetElem(1))]))
			aa = aa.SetElem(2, uint32(xByXPlus1[uint8(z.GetElem(2))]))
			aa = aa.SetElem(3, uint32(xByXPlus1[uint8(z.GetElem(3))]))
			aa.BitsToInt32().StoreArray((*[4]int32)(unsafe.Add(ap, 4*col)))
			vSgr.Sub(aa).Mul(sum.ToBits()).Mul(vr).Add(vbrRound).Shift(vbrShift).BitsToInt32().StoreArray((*[4]int32)(unsafe.Add(bp, 4*col)))
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
