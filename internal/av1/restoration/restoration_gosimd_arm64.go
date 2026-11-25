// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"encoding/binary"
	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"simd/archsimd"
)

// restorationTruncateInt32PairToInt16 packs the low four lanes from each
// int32 vector into one int16 vector, matching the former XTN/XTN2 pair.
func restorationTruncateInt32PairToInt16(lo, hi archsimd.Int32x4) archsimd.Int16x8 {
	lo64 := lo.TruncToInt16().ToBits().ReshapeToUint64s()
	hi64 := hi.TruncToInt16().ToBits().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s().BitsToInt16()
}

// restorationSaturateInt32PairToUint16 packs two groups of four signed values
// after unsigned saturation, matching the former SQXTUN/SQXTUN2 pair.
func restorationSaturateInt32PairToUint16(lo, hi archsimd.Int32x4) archsimd.Uint16x8 {
	lo64 := lo.SaturateToUint16().ReshapeToUint64s()
	hi64 := hi.SaturateToUint16().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint16s()
}

// restorationSaturateInt16PairToUint8 packs two groups of eight signed values
// after unsigned saturation.
func restorationSaturateInt16PairToUint8(lo, hi archsimd.Int16x8) archsimd.Uint8x16 {
	lo64 := lo.SaturateToUint8().ReshapeToUint64s()
	hi64 := hi.SaturateToUint8().ReshapeToUint64s()
	return lo64.InterleaveLo(hi64).ReshapeToUint8s()
}

// init binds the winning Go-native SIMD high-bit-depth SGR blend under the
// goexperiment.simd build. Measured-losing Wiener and 8-bit SGR candidates
// remain on their NEON dispatchers.
func init() {
	sgrWeightedRowImpl = sgrWeightedRowSIMD
}

// sgrConsts holds the loop-invariant broadcast vectors for sgrProjectHalf.
//
// The shift amounts are pre-broadcast to vectors so each per-pixel shift issues
// a single register VSSHL: archsimd's ShiftAll{Left,Right} always lower to
// variable VSSHL and would otherwise re-materialise the constant with VMOV+VDUP
// on every use inside the loop (a negative Shift count is an arithmetic right
// shift, byte-identical to ShiftAllRight). shRst (=RstBits) is used by the u16
// widen.
//
// bias (the rounding 1<<10) seeds the multiply-accumulate chain and k128
// (=1<<PrjBits) turns u<<PrjBits into a fused u*128 MulAdd, so the projection is
// one Sub + one MulAdd per filter term plus a single MulAdd for the u<<7 term,
// then one arithmetic >>11 — no separate shift for <<7 and no separate add for
// the bias, matching the NEON asm's MLA + rounding-shift structure.
type sgrConsts struct {
	xq0V, xq1V, bias  archsimd.Int32x4
	cuV, shRst, shRnd archsimd.Int32x4
}

func newSGRConsts(xq0, xq1 int32) sgrConsts {
	return sgrConsts{
		xq0V:  archsimd.BroadcastInt32x4(xq0),
		xq1V:  archsimd.BroadcastInt32x4(xq1),
		bias:  archsimd.BroadcastInt32x4(1 << (SGRProjPrjBits + SGRProjRstBits - 1)),
		cuV:   archsimd.BroadcastInt32x4((1 << SGRProjPrjBits) - xq0 - xq1),
		shRst: archsimd.BroadcastInt32x4(SGRProjRstBits),
		shRnd: archsimd.BroadcastInt32x4(-(SGRProjPrjBits + SGRProjRstBits)),
	}
}

// sgrProjectHalf computes roundPowerOfTwo(v, 11) for four columns, where
// v = u<<7 + xq0*(f0-u) + xq1*(f1-u) and u = s<<4 (the caller supplies u
// pre-shifted). The result is the raw int32 projection BEFORE the wrapping
// int16 cast and the [0,max] clip.
//
// All adds/muls are int32 and commute mod 2^32, so seeding the accumulator with
// the rounding bias and adding u<<7 as the last term is bit-identical to the
// scalar order: (f0-u)*xq0 + bias, then +(f1-u)*xq1, then +u*128, equals
// u<<7 + xq0*(f0-u) + xq1*(f1-u) + bias == v + bias, and u*128 == u<<7 in the
// low 32 bits. Each MulAdd is one VMLA, the same fused op the asm uses.
func sgrProjectHalf(u, f0, f1 archsimd.Int32x4, c sgrConsts) archsimd.Int32x4 {
	// Algebraic factor: u<<7 + xq0*(f0-u) + xq1*(f1-u) == u*(128-xq0-xq1) +
	// xq0*f0 + xq1*f1 -- drops both f-u subtracts (6 ops/group -> 4). The round
	// bias seeds the first MulAdd; c.shRnd then arithmetic-shifts right by 11.
	v := c.xq0V.MulAdd(f0, c.bias) // xq0*f0 + bias
	v = c.xq1V.MulAdd(f1, v)       // xq1*f1 + v
	v = c.cuV.MulAdd(u, v)         // (128-xq0-xq1)*u + v
	return v.Shift(c.shRnd)
}

// sgrWeightedRowSIMD is the Go-native-SIMD high-bit-depth (uint16) final
// projection row. The bulk runs 16 pixels per iteration (two 16-byte loads,
// four Int32x4 projections, two 16-byte stores); a trailing 8-wide group and
// the <8 tail keep the scalar path. Byte-identical to sgrWeightedRow.
//
// The widen uses ExtendLo4ToUint32 and HiToLo+ExtendLo4ToUint32. The int32->int16
// pack uses TruncToInt16 and InterleaveLo. Unlike
// the [0,maxI] clamp cannot fold into the narrow (maxI < 65535, so
// SQXTUN's full-range saturation is too wide), so it stays as Max/Min in the
// int16 domain before the uint16 reinterpret.
func sgrWeightedRowSIMD(dst []uint16, src []uint16, f0 []int32, f1 []int32, xq0 int32, xq1 int32, maxI int32) {
	width := len(dst)
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	c := newSGRConsts(xq0, xq1)
	zero16 := archsimd.BroadcastInt16x8(0)
	max16 := archsimd.BroadcastInt16x8(int16(maxI))
	narrow := func(a, b archsimd.Int32x4) archsimd.Uint16x8 {
		return restorationTruncateInt32PairToInt16(a, b).Max(zero16).Min(max16).ConvertToUint16()
	}
	// u16 has no widen-with-shift op, so u = s<<RstBits is a register VSSHL after
	// the UXTL/UXTL2 widen (uSh applies c.shRst).
	uSh := func(v archsimd.Uint32x4) archsimd.Int32x4 { return v.ConvertToInt32().Shift(c.shRst) }
	// dst, src, f0 and f1 share one length, so the i <= len-16 condition
	// proves every window below and the loop carries no bounds checks.
	i := 0
	for ; i <= len(dst)-16; i += 16 {
		d, sv, a, b := (*[16]uint16)(dst[i:]), (*[16]uint16)(src[i:]), (*[16]int32)(f0[i:]), (*[16]int32)(f1[i:])
		sv0 := archsimd.LoadUint16x8Array((*[8]uint16)(sv[0:8]))
		sv1 := archsimd.LoadUint16x8Array((*[8]uint16)(sv[8:16]))
		r0 := sgrProjectHalf(uSh(sv0.ExtendLo4ToUint32()), archsimd.LoadInt32x4Array((*[4]int32)(a[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(b[0:4])), c)
		r1 := sgrProjectHalf(uSh(sv0.HiToLo().ExtendLo4ToUint32()), archsimd.LoadInt32x4Array((*[4]int32)(a[4:8])), archsimd.LoadInt32x4Array((*[4]int32)(b[4:8])), c)
		r2 := sgrProjectHalf(uSh(sv1.ExtendLo4ToUint32()), archsimd.LoadInt32x4Array((*[4]int32)(a[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(b[8:12])), c)
		r3 := sgrProjectHalf(uSh(sv1.HiToLo().ExtendLo4ToUint32()), archsimd.LoadInt32x4Array((*[4]int32)(a[12:16])), archsimd.LoadInt32x4Array((*[4]int32)(b[12:16])), c)
		narrow(r0, r1).StoreArray((*[8]uint16)(d[0:8]))
		narrow(r2, r3).StoreArray((*[8]uint16)(d[8:16]))
	}
	if i <= len(dst)-8 {
		d, sv, a, b := (*[8]uint16)(dst[i:]), (*[8]uint16)(src[i:]), (*[8]int32)(f0[i:]), (*[8]int32)(f1[i:])
		v := archsimd.LoadUint16x8Array(sv)
		rLo := sgrProjectHalf(uSh(v.ExtendLo4ToUint32()), archsimd.LoadInt32x4Array((*[4]int32)(a[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(b[0:4])), c)
		rHi := sgrProjectHalf(uSh(v.HiToLo().ExtendLo4ToUint32()), archsimd.LoadInt32x4Array((*[4]int32)(a[4:8])), archsimd.LoadInt32x4Array((*[4]int32)(b[4:8])), c)
		narrow(rLo, rHi).StoreArray(d)
		i += 8
	}
	if i < width {
		sgrWeightedRow(dst[i:], src[i:], f0[i:], f1[i:], xq0, xq1, maxI)
	}
}

// sgrBlendLanes is the number of int32 columns the blend processes per vector
// (one Int32x4). The composed self-guided driver rounds its SIMD column span down
// to this multiple and runs the scalar reference on the tail.
const sgrBlendLanes = 4

// sgrBlendRowSIMD computes one blended self-guided output row for cols columns
// (a multiple of sgrBlendLanes). It evaluates
//
//	dst[c] = roundPowerOfTwo(a[c]*dgd[c] + b[c], shift)
//
// where a and b are the nine-tap weighted 3x3 stencils over the A/B rows:
//
//	w[0] w[1] w[2]   (previous row: left, center, right)
//	w[3] w[4] w[5]   (current  row)
//	w[6] w[7] w[8]   (next     row)
//
// Every slice is indexed from the column one to the left of the first output
// column, so the three stencil taps of output c are at index c, c+1, c+2. The
// products and sums are int32 and wrap exactly like the scalar reference.
//
// The nine weights are scalar locals broadcast once, and each stencil row is
// resliced once per group so every lane load has a constant, in-range index.
func sgrBlendRowSIMD(dst []int32, dgd []int32, aPrev []int32, aCur []int32, aNext []int32, bPrev []int32, bCur []int32, bNext []int32, w *[9]int32, shift int, cols int) {
	w0 := archsimd.BroadcastInt32x4(w[0])
	w1 := archsimd.BroadcastInt32x4(w[1])
	w2 := archsimd.BroadcastInt32x4(w[2])
	w3 := archsimd.BroadcastInt32x4(w[3])
	w4 := archsimd.BroadcastInt32x4(w[4])
	w5 := archsimd.BroadcastInt32x4(w[5])
	w6 := archsimd.BroadcastInt32x4(w[6])
	w7 := archsimd.BroadcastInt32x4(w[7])
	w8 := archsimd.BroadcastInt32x4(w[8])
	biasV := archsimd.BroadcastInt32x4(int32(1) << (shift - 1))
	shV := archsimd.BroadcastInt32x4(-int32(shift))
	for c := 0; c < cols; c += sgrBlendLanes {
		ap := aPrev[c : c+6 : c+6]
		ac := aCur[c : c+6 : c+6]
		an := aNext[c : c+6 : c+6]
		bp := bPrev[c : c+6 : c+6]
		bc := bCur[c : c+6 : c+6]
		bn := bNext[c : c+6 : c+6]
		a := w0.MulAdd(sgrLane4(ap, 0), w1.MulAdd(sgrLane4(ap, 1), w2.MulAdd(sgrLane4(ap, 2),
			w3.MulAdd(sgrLane4(ac, 0), w4.MulAdd(sgrLane4(ac, 1), w5.MulAdd(sgrLane4(ac, 2),
				w6.MulAdd(sgrLane4(an, 0), w7.MulAdd(sgrLane4(an, 1), w8.Mul(sgrLane4(an, 2))))))))))
		b := w0.MulAdd(sgrLane4(bp, 0), w1.MulAdd(sgrLane4(bp, 1), w2.MulAdd(sgrLane4(bp, 2),
			w3.MulAdd(sgrLane4(bc, 0), w4.MulAdd(sgrLane4(bc, 1), w5.MulAdd(sgrLane4(bc, 2),
				w6.MulAdd(sgrLane4(bn, 0), w7.MulAdd(sgrLane4(bn, 1), w8.Mul(sgrLane4(bn, 2))))))))))
		d := sgrLane4(dgd[c:c+4:c+4], 0)
		r := a.MulAdd(d, b).Add(biasV).Shift(shV)
		r.Store(dst[c : c+4 : c+4])
	}
}

// sgrLane4 loads the four int32 values starting at p[k]. The caller reslices p to
// the exact window, so with a constant k the index is provably in range.
func sgrLane4(p []int32, k int) archsimd.Int32x4 {
	return archsimd.LoadInt32x4Array((*[4]int32)(p[k : k+4]))
}

// init binds the Go-native SIMD self-guided kernels on arm64 (NEON is mandatory on
// every arm64 chip Go runs on). The box sums use boxsumSeparable, which needs no
// SIMD to beat the brute-force window, and the per-pixel blend runs through
// sgrBlendRowSIMD. The data-dependent LUT gather inside calculateIntermediate stays
// scalar in both paths. The assignment happens once, before any decoder goroutine
// starts, so the steady-state cost is a single indirect call.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	boxsumImpl = boxsumSeparable
	if cpu.Detected.NEON {
		selfguidedImpl = selfguidedSIMD
		selfguidedFastImpl = selfguidedFastSIMD
		return
	}
	selfguidedImpl = selfguided
	selfguidedFastImpl = selfguidedFast
}

// selfguidedSIMD is the self-guided filter with the SIMD blend, used for the
// radius-1 and radius-2 passes. Its A/B coefficients come from the rolling SIMD
// calculateIntermediateSIMD with the parameter pass selected by the radius index.
func selfguidedSIMD(dgd []int32, dgdOrigin int, width int, height int, dgdStride int, dst []int32, dstStride int, bitDepth int, paramsIndex int, radiusIndex int, aBuf []int32, bBuf []int32, bufStride int) {
	calculateIntermediateSIMD(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 0, aBuf, bBuf, bufStride)
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
	calculateIntermediateSIMD(dgd, dgdOrigin, width, height, dgdStride, bitDepth, paramsIndex, radiusIndex, 1, aBuf, bBuf, bufStride)
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
	// validSGRBlock bounds width by ProcUnitSize, so vecCols/4 <= 16 and each
	// 8-element box-sum window at col = 4k stays inside the 70-element storage.
	if vecCols > ProcUnitSize+2 {
		panic("restoration: SGR block wider than a processing unit")
	}
	for y := -1; y < height+1; y += step {
		ar := aBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		br := bBuf[aOrigin+y*bufStride-1 : aOrigin+y*bufStride+width+1]
		// A vector starts at col <= vecCols-4. Its last input is
		// col+2*r+3 <= vecCols+2*r-1 <= width+2*r+1 == span-1.
		// span <= 70 is checked by the slices above; r is 1 or 2.
		// The two stores end at col+3 < vecCols <= len(ar)==len(br).
		for k := 0; k < vecCols>>2; k++ {
			col := 4 * k
			sw := (*[8]int32)(sumStorage[col : col+8])
			qw := (*[8]int32)(squareStorage[col : col+8])
			sum := archsimd.LoadInt32x4Array((*[4]int32)(sw[0:4])).Add(archsimd.LoadInt32x4Array((*[4]int32)(sw[1:5]))).Add(archsimd.LoadInt32x4Array((*[4]int32)(sw[2:6])))
			square := archsimd.LoadInt32x4Array((*[4]int32)(qw[0:4])).Add(archsimd.LoadInt32x4Array((*[4]int32)(qw[1:5]))).Add(archsimd.LoadInt32x4Array((*[4]int32)(qw[2:6])))
			if r == 2 {
				sum = sum.Add(archsimd.LoadInt32x4Array((*[4]int32)(sw[3:7]))).Add(archsimd.LoadInt32x4Array((*[4]int32)(sw[4:8])))
				square = square.Add(archsimd.LoadInt32x4Array((*[4]int32)(qw[3:7]))).Add(archsimd.LoadInt32x4Array((*[4]int32)(qw[4:8])))
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
			aa.BitsToInt32().StoreArray((*[4]int32)(ar[col : col+4]))
			vSgr.Sub(aa).Mul(sum.ToBits()).Mul(vr).Add(vbrRound).Shift(vbrShift).BitsToInt32().StoreArray((*[4]int32)(br[col : col+4]))
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

// Go-native SIMD final self-guided projection for 8-bit pixels (the dav1d
// sgr_weighted2_8bpc role with libaom's exact arithmetic). Per eight pixels:
//
//	u  = s << SGRProjRstBits                          (widen + VSSHL #4)
//	r  = roundPowerOfTwo(u*(128-xq0-xq1) + xq0*f0 + xq1*f1, 11)   (sgrProjectHalf)
//	w  = int32(int16(r))                              (VXTN: wrap to 16 bits)
//	d  = uint8(clampInt32(w, 0, 255))                 (SQXTUN: saturate to [0,255])
//
// The identity u<<7 + xq0*(f0-u) + xq1*(f1-u) == u*(128-xq0-xq1) + xq0*f0 +
// xq1*f1 holds in wrapping int32 arithmetic, so the projection is the same one the
// u16 kernel uses. The int16 wrap and the unsigned saturating narrow together equal
// clampInt32(int32(int16(r)), 0, 255) for every r.

// sgrWeightedRowU8SIMD is the Go-native SIMD form of sgrWeightedRowU8. Sixteen
// columns per iteration: one 16-byte load, two 8-lane projections, one 16-byte
// store. The trailing width%16 columns use the scalar reference.
//
// The constants are struct literals built in this function, not returned from a
// helper, so they stay register-resident across the loop; the flt rows are viewed
// as fixed-size array pointers so every lane load has a constant, in-range index.
func sgrWeightedRowU8SIMD(dst []uint8, src []uint8, f0 []int32, f1 []int32, xq0 int32, xq1 int32) {
	width := len(dst)
	src = src[:width]
	f0 = f0[:width]
	f1 = f1[:width]
	c := sgrConsts{
		xq0V:  archsimd.BroadcastInt32x4(xq0),
		xq1V:  archsimd.BroadcastInt32x4(xq1),
		bias:  archsimd.BroadcastInt32x4(1 << (SGRProjPrjBits + SGRProjRstBits - 1)),
		cuV:   archsimd.BroadcastInt32x4((1 << SGRProjPrjBits) - xq0 - xq1),
		shRst: archsimd.BroadcastInt32x4(SGRProjRstBits),
		shRnd: archsimd.BroadcastInt32x4(-(SGRProjPrjBits + SGRProjRstBits)),
	}
	col := 0
	for ; col+16 <= width; col += 16 {
		sv := archsimd.LoadUint8x16Array((*[16]uint8)(src[col:]))
		fa := (*[16]int32)(f0[col:])
		fb := (*[16]int32)(f1[col:])
		lo := sv.ExtendLo8ToUint16()
		hi := sv.HiToLo().ExtendLo8ToUint16()
		// Strip A: samples 0..7 of this group.
		a0 := sgrProjectHalf(lo.ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[0:4])), archsimd.LoadInt32x4Array((*[4]int32)(fb[0:4])), c)
		a1 := sgrProjectHalf(lo.HiToLo().ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[4:8])), archsimd.LoadInt32x4Array((*[4]int32)(fb[4:8])), c)
		outA := restorationTruncateInt32PairToInt16(a0, a1).SaturateToUint8()
		// Strip B: samples 8..15 of this group.
		b0 := sgrProjectHalf(hi.ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[8:12])), archsimd.LoadInt32x4Array((*[4]int32)(fb[8:12])), c)
		b1 := sgrProjectHalf(hi.HiToLo().ExtendLo4ToUint32().ConvertToInt32().Shift(c.shRst),
			archsimd.LoadInt32x4Array((*[4]int32)(fa[12:16])), archsimd.LoadInt32x4Array((*[4]int32)(fb[12:16])), c)
		outB := restorationTruncateInt32PairToInt16(b0, b1).SaturateToUint8()
		// Each strip's eight bytes sit in the low 64 bits; join them in order.
		joined := outA.ReshapeToUint64s().InterleaveLo(outB.ReshapeToUint64s()).ReshapeToUint8s()
		joined.StoreArray((*[16]uint8)(dst[col:]))
	}
	if col < width {
		sgrWeightedRowU8(dst[col:], src[col:], f0[col:], f1[col:], xq0, xq1)
	}
}

// 8-bit self-guided filter passes in dav1d's 8bpc arm64 shape
// (src/arm/64/looprestoration_common.S sgr_box3_vert / sgr_box5_vert and
// looprestoration_tmpl.S sgr_finish_filter1/2): the box sums stay in
// 16-bit (sum) and 32-bit (sum of squares) lanes over the uint8 source, the
// x_by_x lookup runs as dav1d's three-table TBL + popcount form on sixteen
// lanes at once, A is kept as int16 and the 3x3 / fast 5x5 stencils use the
// factored weights
//
//	3x3:      4*cross + 3*diag     == 3*(Hp + Hn) + 4*Hc + Ap + An
//	5x5 even: 6*(Ap + An) + 5*diag == 5*(Hp + Hn) + Ap + An
//	5x5 odd:  6*Ac + 5*(Al + Ar)   == 5*Hc + Ac
//
// where H is the horizontal three-sum of a row of A (or B), computed once per
// row and reused by every output row that reads it. The A/B rows stream
// through a three-row ring, as dav1d's row pointers do, instead of a full
// unit-sized buffer.
//
// Every value is the one calculateIntermediateRolling and the selfguided
// blends produce: dav1d's x is SGRProjSgr - xByXPlus1[z] and its B term is
// x*sum*one_by_x, both computed here in the same uint32 arithmetic, and the
// stencil sums are exact in int16 (at most 32*256) and int32.

// sgrU8Cols is the column capacity of every per-row array: the widest box
// window is ProcUnitSize + 2 + 4 = 70 columns, processed in 16-column chunks
// with lane loads reaching at most 13 columns past the last chunk start.
const sgrU8Cols = 96

type sgrU8State struct {
	colSum [sgrU8Cols]uint16
	colSq  [sgrU8Cols]int32
	a      [3][sgrU8Cols]int16
	b      [3][sgrU8Cols]int32
	ha     [3][sgrU8Cols]int16
	hb     [3][sgrU8Cols]int32
	in     [sgrU8Cols]uint8
	out    [sgrU8Cols]uint8
	dgd    [sgrU8Cols]uint8
	flt    [sgrU8Cols]int32
}

// dav1d's x_by_x_tables: x(z) = RefLo[z] (z < 16) + RangeMins[z>>3] +
// popcount(DiffMasks[z>>3] << (z&7)), with x(z) == SGRProjSgr - xByXPlus1[z].
var (
	sgrXRangeMins = [32]uint8{0, 0, 11, 8, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}
	sgrXDiffMasks = [32]uint8{
		0x00, 0x00, 0xD4, 0x44, 0x42, 0x04, 0x00, 0x00,
		0x00, 0x80, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
	}
	sgrXRefLo = [16]uint8{255, 128, 85, 64, 51, 43, 37, 32, 28, 26, 23, 21, 20, 18, 17, 16}
)

// sgrU8Passes runs the active self-guided passes of params over the bordered
// 8-bit unit (the ApplySelfguidedRestorationU8 layout) into flt0 (radius 2,
// fast) and flt1 (radius 1), each width x height at stride width. It reports
// false, writing nothing, for a parameter set outside the AV1 table shapes.
func sgrU8Passes(src []uint8, srcStride, srcOrigin, width, height int, params SGRParams, flt0, flt1 []int32) bool {
	if (params.Radius[0] != 0 && params.Radius[0] != 2) || (params.Radius[1] != 0 && params.Radius[1] != 1) ||
		width > ProcUnitSize || height > ProcUnitSize || len(flt0) < width*height || len(flt1) < width*height {
		return false
	}
	// Finish rows spill up to seven lanes into the next row; the bound keeps
	// the last row's spill inside the width x height block.
	flt0, flt1 = flt0[:width*height], flt1[:width*height]
	var st sgrU8State
	if params.Radius[0] != 0 {
		sgrU8Pass(&st, src, srcStride, srcOrigin, width, height, 2, uint32(params.S[0]), flt0)
	}
	if params.Radius[1] != 0 {
		if params.Radius[0] != 0 {
			st = sgrU8State{}
		}
		sgrU8Pass(&st, src, srcStride, srcOrigin, width, height, 1, uint32(params.S[1]), flt1)
	}
	return true
}

// sgrU8Pass is one filter pass: r == 1 is the full 3x3 filter, r == 2 the
// fast 5x5 filter whose A/B rows exist only at odd offsets (rows -1, 1, 3...).
func sgrU8Pass(st *sgrU8State, src []uint8, srcStride, srcOrigin, w, h, r int, s uint32, flt []int32) {
	span := w + 2 + 2*r
	left := srcOrigin - 1 - r
	// row returns source row row's box-window span, read in place when the
	// 16-byte chunks covering it stay inside src and copied into buf otherwise.
	reach := (span + 15) &^ 15
	row := func(buf *[sgrU8Cols]uint8, row int) []uint8 {
		o := left + row*srcStride
		if o+reach <= len(src) {
			return src[o : o+reach]
		}
		copy(buf[:span], src[o:o+span])
		return buf[:reach]
	}
	// Column sums over source rows y-r..y+r for A/B row y = -1.
	for y := -1 - r; y <= -1+r; y++ {
		sgrU8Roll(st, row(&st.in, y), nil, reach)
	}
	if r == 1 {
		for y := -1; y <= h; y++ {
			sgrU8RowAB(st, (y+1)%3, w, 9, s, uint32(oneByX[8]))
			if y >= 1 {
				sgrU8Finish3(st, src[srcOrigin+(y-1)*srcStride:], (y-1)%3, y%3, (y+1)%3, w, flt[(y-1)*w:])
			}
			if y < h {
				sgrU8Roll(st, row(&st.in, y+1+r), row(&st.out, y-r), reach)
			}
		}
		return
	}
	k := 0
	for y := -1; y < h+1; y += 2 {
		sgrU8RowAB(st, k%3, w, 25, s, uint32(oneByX[24]))
		if k > 0 {
			even := y - 1
			sgrU8Finish5Even(st, src[srcOrigin+even*srcStride:], (k-1)%3, k%3, w, flt[even*w:])
			if even+1 < h {
				sgrU8Finish5Odd(st, src[srcOrigin+(even+1)*srcStride:], k%3, w, flt[(even+1)*w:])
			}
		}
		if y+2 >= h+1 {
			break
		}
		for adv := 0; adv < 2; adv++ {
			sgrU8Roll(st, row(&st.in, y+r+1+adv), row(&st.out, y-r+adv), reach)
		}
		k++
	}
}

// sgrU8Roll adds in (and, when out is non-nil, removes out) to the column
// sums over reach columns: sum += in - out, sumsq += in*in - out*out ==
// (in-out)*(in+out). Lanes past the span carry unused values.
func sgrU8Roll(st *sgrU8State, inRow, outRow []uint8, reach int) {
	inRow = inRow[:reach]
	sub := outRow != nil
	if sub {
		outRow = outRow[:reach]
	}
	for c := 0; c < reach; c += 16 {
		in := archsimd.LoadUint8x16Array((*[16]uint8)(inRow[c : c+16]))
		var out archsimd.Uint8x16
		if sub {
			out = archsimd.LoadUint8x16Array((*[16]uint8)(outRow[c : c+16]))
		}
		inLo, inHi := in.ExtendLo8ToUint16(), in.HiToLo().ExtendLo8ToUint16()
		outLo, outHi := out.ExtendLo8ToUint16(), out.HiToLo().ExtendLo8ToUint16()
		cs := (*[16]uint16)(st.colSum[c : c+16])
		s0 := archsimd.LoadUint16x8Array((*[8]uint16)(cs[0:8]))
		s1 := archsimd.LoadUint16x8Array((*[8]uint16)(cs[8:16]))
		dLo, dHi := inLo.Sub(outLo), inHi.Sub(outHi)
		s0.Add(dLo).StoreArray((*[8]uint16)(cs[0:8]))
		s1.Add(dHi).StoreArray((*[8]uint16)(cs[8:16]))
		d0 := dLo.BitsToInt16()
		d1 := dHi.BitsToInt16()
		p0 := inLo.Add(outLo).BitsToInt16()
		p1 := inHi.Add(outHi).BitsToInt16()
		cq := (*[16]int32)(st.colSq[c : c+16])
		q0 := archsimd.LoadInt32x4Array((*[4]int32)(cq[0:4]))
		q1 := archsimd.LoadInt32x4Array((*[4]int32)(cq[4:8]))
		q2 := archsimd.LoadInt32x4Array((*[4]int32)(cq[8:12]))
		q3 := archsimd.LoadInt32x4Array((*[4]int32)(cq[12:16]))
		q0.Add(d0.MulWidenLo(p0)).StoreArray((*[4]int32)(cq[0:4]))
		q1.Add(d0.HiToLo().MulWidenLo(p0.HiToLo())).StoreArray((*[4]int32)(cq[4:8]))
		q2.Add(d1.MulWidenLo(p1)).StoreArray((*[4]int32)(cq[8:12]))
		q3.Add(d1.HiToLo().MulWidenLo(p1.HiToLo())).StoreArray((*[4]int32)(cq[12:16]))
	}
}

// sgrU8X is dav1d's x_by_x lookup on sixteen z values (each <= 255).
func sgrU8X(z archsimd.Uint8x16) archsimd.Uint8x16 {
	rmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[0:16]))
	rmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[16:32]))
	dmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[0:16]))
	dmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[16:32]))
	ref := archsimd.LoadUint8x16Array(&sgrXRefLo)
	hi := z.Shift(archsimd.BroadcastInt8x16(-3))
	hi16 := hi.Sub(archsimd.BroadcastUint8x16(16))
	mins := rmLo.LookupOrZero(hi).Or(rmHi.LookupOrZero(hi16))
	diffs := dmLo.LookupOrZero(hi).Or(dmHi.LookupOrZero(hi16))
	lo := z.And(archsimd.BroadcastUint8x16(7)).BitsToInt8()
	return ref.LookupOrZero(z).Add(mins).Add(diffs.Shift(lo).OnesCount())
}

func sgrU8LoadU16(p []uint16, i int) archsimd.Uint16x8 {
	return archsimd.LoadUint16x8Array((*[8]uint16)(p[i : i+8]))
}

func sgrU8LoadI32(p []int32, i int) archsimd.Int32x4 {
	return archsimd.LoadInt32x4Array((*[4]int32)(p[i : i+4]))
}

func sgrU8LoadI16(p []int16, i int) archsimd.Int16x8 {
	return archsimd.LoadInt16x8Array((*[8]int16)(p[i : i+8]))
}

// Box-window sums over a 32-column view at a constant offset o: after
// inlining every lane index is constant and in range of the view.
func sgrU8Sum3(cs *[32]uint16, o int) archsimd.Uint16x8 {
	return sgrU8LoadU16(cs[:], o).Add(sgrU8LoadU16(cs[:], o+1)).Add(sgrU8LoadU16(cs[:], o+2))
}

func sgrU8Sum5(cs *[32]uint16, o int) archsimd.Uint16x8 {
	return sgrU8Sum3(cs, o).Add(sgrU8LoadU16(cs[:], o+3)).Add(sgrU8LoadU16(cs[:], o+4))
}

func sgrU8Sq3(cq *[32]int32, o int) archsimd.Uint32x4 {
	return sgrU8LoadI32(cq[:], o).Add(sgrU8LoadI32(cq[:], o+1)).Add(sgrU8LoadI32(cq[:], o+2)).ToBits()
}

func sgrU8Sq5(cq *[32]int32, o int) archsimd.Uint32x4 {
	return sgrU8LoadI32(cq[:], o).Add(sgrU8LoadI32(cq[:], o+1)).Add(sgrU8LoadI32(cq[:], o+2)).
		Add(sgrU8LoadI32(cq[:], o+3)).Add(sgrU8LoadI32(cq[:], o+4)).ToBits()
}

// sgrU8RowAB writes A (int16), B (int32) and their horizontal three-sums for
// the A/B row whose box windows are the current column sums, into ring slot.
// A/B column index 0 is unit column -1. z = min(round((sumsq*n - sum*sum)*s,
// 20), 255) and B use calculateIntermediateRolling's uint32 arithmetic.
func sgrU8RowAB(st *sgrU8State, slot, w int, n, s, recip uint32) {
	r2 := n == 25
	vn := archsimd.BroadcastUint32x4(n)
	vs := archsimd.BroadcastUint32x4(s)
	vRnd := archsimd.BroadcastUint32x4(1 << (SGRProjMTableBits - 1))
	shl4 := archsimd.BroadcastInt16x8(SGRProjMTableBits - 16)
	vRecip := archsimd.BroadcastUint32x4(recip)
	// B carries sgrU8BBias on top of its rounding: every finish stencil
	// weighs B by 32 (or 16 for the fast filter's odd rows), so the bias
	// lands as exactly the finish rounding term.
	vbRnd := archsimd.BroadcastUint32x4(1<<(SGRProjRecipBits-1) + sgrU8BBias<<SGRProjRecipBits)
	vbSh := archsimd.BroadcastInt32x4(-SGRProjRecipBits)
	v256 := archsimd.BroadcastUint16x8(SGRProjSgr)
	rmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[0:16]))
	rmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXRangeMins[16:32]))
	dmLo := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[0:16]))
	dmHi := archsimd.LoadUint8x16Array((*[16]uint8)(sgrXDiffMasks[16:32]))
	refLo := archsimd.LoadUint8x16Array(&sgrXRefLo)
	shr3 := archsimd.BroadcastInt8x16(-3)
	k16 := archsimd.BroadcastUint8x16(16)
	k7 := archsimd.BroadcastUint8x16(7)
	ar := st.a[slot][:]
	br := st.b[slot][:]
	for j := 0; j < w+2; j += 16 {
		cs := (*[32]uint16)(st.colSum[j : j+32])
		cq := (*[32]int32)(st.colSq[j : j+32])
		var sum0, sum1 archsimd.Uint16x8
		var q0, q1, q2, q3 archsimd.Uint32x4
		if r2 {
			sum0, sum1 = sgrU8Sum5(cs, 0), sgrU8Sum5(cs, 8)
			q0, q1, q2, q3 = sgrU8Sq5(cq, 0), sgrU8Sq5(cq, 4), sgrU8Sq5(cq, 8), sgrU8Sq5(cq, 12)
		} else {
			sum0, sum1 = sgrU8Sum3(cs, 0), sgrU8Sum3(cs, 8)
			q0, q1, q2, q3 = sgrU8Sq3(cq, 0), sgrU8Sq3(cq, 4), sgrU8Sq3(cq, 8), sgrU8Sq3(cq, 12)
		}
		h0, h1 := sum0.HiToLo(), sum1.HiToLo()
		// z = min((p*s + 2^19) >> 20, 255) in uint32 wrap arithmetic, narrowed
		// as dav1d's uqshrn #16 / uqrshrn #4 pair does: the high halves of the
		// biased products (UZP2) are floor(v / 2^16); a saturating << 4 leaves
		// floor(v / 2^20) clamped to 255 in each high byte (UZP2 again).
		z0 := q0.Mul(vn).SubSaturated(sum0.MulWidenLo(sum0)).Mul(vs).Add(vRnd)
		z1 := q1.Mul(vn).SubSaturated(h0.MulWidenLo(h0)).Mul(vs).Add(vRnd)
		z2 := q2.Mul(vn).SubSaturated(sum1.MulWidenLo(sum1)).Mul(vs).Add(vRnd)
		z3 := q3.Mul(vn).SubSaturated(h1.MulWidenLo(h1)).Mul(vs).Add(vRnd)
		zA := z0.ReshapeToUint16s().ConcatOdd(z1.ReshapeToUint16s()).ShiftSaturated(shl4)
		zB := z2.ReshapeToUint16s().ConcatOdd(z3.ReshapeToUint16s()).ShiftSaturated(shl4)
		z := zA.ReshapeToUint8s().ConcatOdd(zB.ReshapeToUint8s())
		// dav1d x_by_x: RefLo[z] + RangeMins[z>>3] + popcount(DiffMasks[z>>3] << (z&7)).
		hi := z.Shift(shr3)
		hi16 := hi.Sub(k16)
		mins := rmLo.LookupOrZero(hi).Or(rmHi.LookupOrZero(hi16))
		diffs := dmLo.LookupOrZero(hi).Or(dmHi.LookupOrZero(hi16))
		x := refLo.LookupOrZero(z).Add(mins).Add(diffs.Shift(z.And(k7).BitsToInt8()).OnesCount())
		x0 := x.ExtendLo8ToUint16()
		x1 := x.HiToLo().ExtendLo8ToUint16()
		ao := (*[16]int16)(ar[j : j+16])
		v256.Sub(x0).BitsToInt16().StoreArray((*[8]int16)(ao[0:8]))
		v256.Sub(x1).BitsToInt16().StoreArray((*[8]int16)(ao[8:16]))
		bo := (*[16]int32)(br[j : j+16])
		x0.MulWidenLo(sum0).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[0:4]))
		x0.HiToLo().MulWidenLo(h0).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[4:8]))
		x1.MulWidenLo(sum1).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[8:12]))
		x1.HiToLo().MulWidenLo(h1).MulAdd(vRecip, vbRnd).Shift(vbSh).BitsToInt32().StoreArray((*[4]int32)(bo[12:16]))
	}
	ha := st.ha[slot][:]
	hb := st.hb[slot][:]
	for c := 0; c < w; c += 8 {
		a := (*[16]int16)(ar[c : c+16])
		b := (*[16]int32)(br[c : c+16])
		sgrU8LoadI16(a[:], 0).Add(sgrU8LoadI16(a[:], 1)).Add(sgrU8LoadI16(a[:], 2)).StoreArray((*[8]int16)(ha[c : c+8]))
		hbo := (*[8]int32)(hb[c : c+8])
		sgrU8LoadI32(b[:], 0).Add(sgrU8LoadI32(b[:], 1)).Add(sgrU8LoadI32(b[:], 2)).StoreArray((*[4]int32)(hbo[0:4]))
		sgrU8LoadI32(b[:], 4).Add(sgrU8LoadI32(b[:], 5)).Add(sgrU8LoadI32(b[:], 6)).StoreArray((*[4]int32)(hbo[4:8]))
	}
}

// sgrU8BBias is added to every stored B value: 8 * 32 == 1 << 8 and
// 8 * 16 == 1 << 7 are the rounding terms of the finish shifts below.
const sgrU8BBias = 8

// sgrU8Out finishes eight output columns: (a*dgd + b) >> shift, the
// rounding bias riding in b.
func sgrU8Out(o *[8]int32, a archsimd.Int16x8, bLo, bHi archsimd.Int32x4, d archsimd.Int16x8, sh archsimd.Int32x4) {
	a.MulWidenLo(d).Add(bLo).Shift(sh).StoreArray((*[4]int32)(o[0:4]))
	a.HiToLo().MulWidenLo(d.HiToLo()).Add(bHi).Shift(sh).StoreArray((*[4]int32)(o[4:8]))
}

// sgrU8Rows prepares a finish row: the source row (read in place when the
// 16-byte loads stay inside it, else copied to st.dgd) and the output row
// (written in place when the 8-lane stores stay inside flt, else staged in
// st.flt and copied by sgrU8Flush). Columns past w spill into the next row,
// which is finished later.
func sgrU8Rows(st *sgrU8State, row []uint8, flt []int32, w int) ([]uint8, []int32) {
	n := (w + 7) &^ 7
	if len(row) < n+8 {
		copy(st.dgd[:w], row[:w])
		row = st.dgd[:]
	}
	if len(flt) < n {
		return row[:n+8], st.flt[:n]
	}
	return row[:n+8], flt[:n]
}

func sgrU8Flush(st *sgrU8State, out, flt []int32, w int) {
	if &out[0] == &st.flt[0] {
		copy(flt[:w], st.flt[:w])
	}
}

func sgrU8Dgd(row []uint8, c int) archsimd.Int16x8 {
	return archsimd.LoadUint8x16Array((*[16]uint8)(row[c : c+16])).ExtendLo8ToUint16().BitsToInt16()
}

// sgrU8Finish3 is the 3x3 output row between ring slots p (above), m and n
// (below); row holds the unit's source row from column 0.
func sgrU8Finish3(st *sgrU8State, row []uint8, p, m, n, w int, flt []int32) {
	row, out := sgrU8Rows(st, row, flt, w)
	sh := archsimd.BroadcastInt32x4(-(SGRProjSgrBits + 5 - SGRProjRstBits))
	three16, three32 := archsimd.BroadcastInt16x8(3), archsimd.BroadcastInt32x4(3)
	haP, haM, haN := st.ha[p][:], st.ha[m][:], st.ha[n][:]
	hbP, hbM, hbN := st.hb[p][:], st.hb[m][:], st.hb[n][:]
	aP, aN := st.a[p][:], st.a[n][:]
	bP, bN := st.b[p][:], st.b[n][:]
	for c := 0; c < len(out); c += 8 {
		// 3*(Hp + Hn) + 4*Hc + Ap + An == 3*(Hp + Hc + Hn) + (Hc + Ap + An).
		haC, hbC0, hbC1 := sgrU8LoadI16(haM, c), sgrU8LoadI32(hbM, c), sgrU8LoadI32(hbM, c+4)
		a := sgrU8LoadI16(haP, c).Add(sgrU8LoadI16(haN, c)).Add(haC).
			MulAdd(three16, haC.Add(sgrU8LoadI16(aP, c+1)).Add(sgrU8LoadI16(aN, c+1)))
		bLo := sgrU8LoadI32(hbP, c).Add(sgrU8LoadI32(hbN, c)).Add(hbC0).
			MulAdd(three32, hbC0.Add(sgrU8LoadI32(bP, c+1)).Add(sgrU8LoadI32(bN, c+1)))
		bHi := sgrU8LoadI32(hbP, c+4).Add(sgrU8LoadI32(hbN, c+4)).Add(hbC1).
			MulAdd(three32, hbC1.Add(sgrU8LoadI32(bP, c+5)).Add(sgrU8LoadI32(bN, c+5)))
		sgrU8Out((*[8]int32)(out[c:c+8]), a, bLo, bHi, sgrU8Dgd(row, c), sh)
	}
	sgrU8Flush(st, out, flt, w)
}

// sgrU8Finish5Even is the fast filter's even output row between the A/B
// rows in slots p (above) and n (below).
func sgrU8Finish5Even(st *sgrU8State, row []uint8, p, n, w int, flt []int32) {
	row, out := sgrU8Rows(st, row, flt, w)
	sh := archsimd.BroadcastInt32x4(-(SGRProjSgrBits + 5 - SGRProjRstBits))
	five16, five32 := archsimd.BroadcastInt16x8(5), archsimd.BroadcastInt32x4(5)
	haP, haN := st.ha[p][:], st.ha[n][:]
	hbP, hbN := st.hb[p][:], st.hb[n][:]
	aP, aN := st.a[p][:], st.a[n][:]
	bP, bN := st.b[p][:], st.b[n][:]
	for c := 0; c < len(out); c += 8 {
		a := sgrU8LoadI16(haP, c).Add(sgrU8LoadI16(haN, c)).
			MulAdd(five16, sgrU8LoadI16(aP, c+1).Add(sgrU8LoadI16(aN, c+1)))
		bLo := sgrU8LoadI32(hbP, c).Add(sgrU8LoadI32(hbN, c)).
			MulAdd(five32, sgrU8LoadI32(bP, c+1).Add(sgrU8LoadI32(bN, c+1)))
		bHi := sgrU8LoadI32(hbP, c+4).Add(sgrU8LoadI32(hbN, c+4)).
			MulAdd(five32, sgrU8LoadI32(bP, c+5).Add(sgrU8LoadI32(bN, c+5)))
		sgrU8Out((*[8]int32)(out[c:c+8]), a, bLo, bHi, sgrU8Dgd(row, c), sh)
	}
	sgrU8Flush(st, out, flt, w)
}

// sgrU8Finish5Odd is the fast filter's odd output row on the A/B row in slot m.
func sgrU8Finish5Odd(st *sgrU8State, row []uint8, m, w int, flt []int32) {
	row, out := sgrU8Rows(st, row, flt, w)
	sh := archsimd.BroadcastInt32x4(-(SGRProjSgrBits + 4 - SGRProjRstBits))
	five16, five32 := archsimd.BroadcastInt16x8(5), archsimd.BroadcastInt32x4(5)
	haM, hbM := st.ha[m][:], st.hb[m][:]
	aM, bM := st.a[m][:], st.b[m][:]
	for c := 0; c < len(out); c += 8 {
		a := sgrU8LoadI16(haM, c).MulAdd(five16, sgrU8LoadI16(aM, c+1))
		bLo := sgrU8LoadI32(hbM, c).MulAdd(five32, sgrU8LoadI32(bM, c+1))
		bHi := sgrU8LoadI32(hbM, c+4).MulAdd(five32, sgrU8LoadI32(bM, c+5))
		sgrU8Out((*[8]int32)(out[c:c+8]), a, bLo, bHi, sgrU8Dgd(row, c), sh)
	}
	sgrU8Flush(st, out, flt, w)
}

// init binds the Go-native SIMD 8-bit-pixel self-guided projection on arm64. NEON
// is mandatory on every arm64 chip Go runs on, so the binding is unconditional.
// The 8-bit Wiener passes are bound in wiener_dispatch_arm64.go.
func init() {
	sgrWeightedRowU8Impl = sgrWeightedRowU8SIMD
}

// init binds the Go-native SIMD Wiener passes on arm64 (NEON is mandatory on
// every arm64 chip Go runs on). The assignment happens once, before any decoder
// goroutine starts, so the steady-state cost is a single indirect call. Every
// binding is byte-identical to the pure-Go reference; the wrappers fall back to
// it for widths that are not 8-aligned.
func init() {
	_ = cpu.Detected // ensure cpu package init runs before this point
	if cpu.Detected.NEON {
		wienerHorizontalTrustedImpl = wienerHorizontalSIMDTrusted
		wienerVerticalImpl = wienerVerticalSIMD
		wienerHorizontalU8Impl = wienerHorizontalU8SIMDTaps
		wienerVerticalU8Impl = wienerVerticalU8SIMDTaps
		return
	}
	wienerHorizontalTrustedImpl = wienerHorizontalTrusted
	wienerVerticalImpl = wienerVertical
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl = wienerVerticalU8
}

// wienerFilterFiveTap reports a symmetric filter whose outer tap pair is zero
// (WienerFilter is [f0 f1 f2 f3 f2 f1 f0 0]): every chroma filter and most luma
// ones, dav1d's wiener_filter5 case.
func wienerFilterFiveTap(filter WienerFilter) bool {
	return filter[0] == 0 && filter[WienerSubpelTaps-2] == 0
}

// wienerHorizontalU8SIMDTaps and wienerVerticalU8SIMDTaps route five-tap
// filters to the kernels that skip the zero outer tap pair.
func wienerHorizontalU8SIMDTaps(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	if wienerFilterFiveTap(filter) {
		if round0 == WienerRound0Bits && wienerFilterSymmetric(filter) && wienerU8Dav1dTapsFit(filter) &&
			wienerHorizontalU8SIMDCanRun(len(src), srcStride, srcOrigin, width, height) &&
			blockFits(len(temp), width, width, height+2*WienerHalfwin) {
			wienerHorizontalU8Dav1d5(src, srcStride, srcOrigin, width, height, filter, temp)
			return
		}
		wienerHorizontalU8SIMD5(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	wienerHorizontalU8SIMDChecked(src, srcStride, srcOrigin, width, height, filter, round0, temp)
}

func wienerVerticalU8SIMDTaps(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	if wienerFilterFiveTap(filter) {
		wienerVerticalU8SIMD5(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	wienerVerticalU8SIMD(temp, tempStride, dst, dstStride, width, height, filter, round1)
}

// wienerVerticalSIMD is the Go-native SIMD uint16 Wiener vertical pass for arm64,
// using the packed-pair form described in wiener_vertical_gosimd_arm64.go: the
// symmetric row pairs are summed in uint16 lanes (a temp sample is at most 32767,
// so a pair never exceeds 65534), multiply-accumulated as packed 32-bit lanes
// with taps pre-scaled by 2^(16-round1), and the high 16 bits of the corrected
// even and odd accumulators are exactly roundPowerOfTwo(sum - offset, round1).
// Those int16 results are clamped to [0,max] with one max and one min.
//
// Sixteen columns per strip; a trailing strip of eight columns is placed at
// width-8 when fewer than sixteen remain, recomputing any overlapped columns with
// identical values. Widths below 8, max above 32767, and filters the packed form
// cannot represent (not mirrored, or taps/rounding outside the int32 bound) take
// the scalar reference.
func wienerVerticalSIMD(temp []uint16, tempStride int, dst []uint16, dstStride int, width int, height int, filter WienerFilter, bitDepth int, round1 int, max uint16) {
	t0, t1, t2, t3, seedA, seedO, ok := wienerVerticalPackedTaps(filter, bitDepth, round1)
	// The stride and height bounds make each row window below one compare.
	if width < 8 || max > 32767 || !ok || height > rowsMax || uint(tempStride) > rowStrideMax || uint(dstStride) > rowStrideMax ||
		!wienerVerticalFootprintFits(len(temp), tempStride, len(dst), dstStride, width, height) {
		wienerVertical(temp, tempStride, dst, dstStride, width, height, filter, bitDepth, round1, max)
		return
	}
	t0V := archsimd.BroadcastUint32x4(t0)
	t1V := archsimd.BroadcastUint32x4(t1)
	t2V := archsimd.BroadcastUint32x4(t2)
	t3V := archsimd.BroadcastUint32x4(t3)
	seedAV := archsimd.BroadcastUint32x4(seedA)
	seedOV := archsimd.BroadcastUint32x4(seedO)
	negK := archsimd.BroadcastUint32x4(0xffff0000) // -65536
	shr16 := archsimd.BroadcastInt32x4(-16)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(int16(max))

	// Each row is read and written through one bounds-checked window.
	col := 0
	for ; col+16 <= width; col += 16 {
		tc, dc := temp[col:], dst[col:]
		w1 := (*[16]uint16)(tc[0*tempStride : 0*tempStride+16])
		a0 := archsimd.LoadUint16x8Array((*[8]uint16)(w1[0:8]))
		b0 := archsimd.LoadUint16x8Array((*[8]uint16)(w1[8:16]))
		w2 := (*[16]uint16)(tc[1*tempStride : 1*tempStride+16])
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(w2[0:8]))
		b1 := archsimd.LoadUint16x8Array((*[8]uint16)(w2[8:16]))
		w3 := (*[16]uint16)(tc[2*tempStride : 2*tempStride+16])
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(w3[0:8]))
		b2 := archsimd.LoadUint16x8Array((*[8]uint16)(w3[8:16]))
		w4 := (*[16]uint16)(tc[3*tempStride : 3*tempStride+16])
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(w4[0:8]))
		b3 := archsimd.LoadUint16x8Array((*[8]uint16)(w4[8:16]))
		w5 := (*[16]uint16)(tc[4*tempStride : 4*tempStride+16])
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(w5[0:8]))
		b4 := archsimd.LoadUint16x8Array((*[8]uint16)(w5[8:16]))
		w6 := (*[16]uint16)(tc[5*tempStride : 5*tempStride+16])
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(w6[0:8]))
		b5 := archsimd.LoadUint16x8Array((*[8]uint16)(w6[8:16]))
		for row := 0; row < height; row++ {
			w7 := (*[16]uint16)(tc[(row+6)*tempStride : (row+6)*tempStride+16])
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(w7[0:8]))
			b6 := archsimd.LoadUint16x8Array((*[8]uint16)(w7[8:16]))

			pa0 := a0.Add(a6).ReshapeToUint32s()
			pa1 := a1.Add(a5).ReshapeToUint32s()
			pa2 := a2.Add(a4).ReshapeToUint32s()
			pa3 := a3.ReshapeToUint32s()
			accA := pa3.MulAdd(t3V, seedAV)
			accA = pa2.MulAdd(t2V, accA)
			accA = pa1.MulAdd(t1V, accA)
			accA = pa0.MulAdd(t0V, accA)
			oddA := pa3.Shift(shr16).MulAdd(t3V, seedOV)
			oddA = pa2.Shift(shr16).MulAdd(t2V, oddA)
			oddA = pa1.Shift(shr16).MulAdd(t1V, oddA)
			oddA = pa0.Shift(shr16).MulAdd(t0V, oddA)
			evenA := oddA.MulAdd(negK, accA)
			va := evenA.ReshapeToUint16s().InterleaveOdd(oddA.ReshapeToUint16s()).BitsToInt16()
			dw := (*[16]uint16)(dc[row*dstStride : row*dstStride+16])
			va.Max(zero).Min(maxV).ToBits().StoreArray((*[8]uint16)(dw[0:8]))

			pb0 := b0.Add(b6).ReshapeToUint32s()
			pb1 := b1.Add(b5).ReshapeToUint32s()
			pb2 := b2.Add(b4).ReshapeToUint32s()
			pb3 := b3.ReshapeToUint32s()
			accB := pb3.MulAdd(t3V, seedAV)
			accB = pb2.MulAdd(t2V, accB)
			accB = pb1.MulAdd(t1V, accB)
			accB = pb0.MulAdd(t0V, accB)
			oddB := pb3.Shift(shr16).MulAdd(t3V, seedOV)
			oddB = pb2.Shift(shr16).MulAdd(t2V, oddB)
			oddB = pb1.Shift(shr16).MulAdd(t1V, oddB)
			oddB = pb0.Shift(shr16).MulAdd(t0V, oddB)
			evenB := oddB.MulAdd(negK, accB)
			vb := evenB.ReshapeToUint16s().InterleaveOdd(oddB.ReshapeToUint16s()).BitsToInt16()
			vb.Max(zero).Min(maxV).ToBits().StoreArray((*[8]uint16)(dw[8:16]))
			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
			b0, b1, b2, b3, b4, b5 = b1, b2, b3, b4, b5, b6
		}
	}
	// Trailing columns: eight-wide strips, the last one shifted left to end at
	// width so it overlaps already-written columns instead of running past them.
	for ; col < width; col += 8 {
		c := min(col, width-8)
		tc, dc := temp[c:], dst[c:]
		a0 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[0*tempStride : 0*tempStride+8]))
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[1*tempStride : 1*tempStride+8]))
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[2*tempStride : 2*tempStride+8]))
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[3*tempStride : 3*tempStride+8]))
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[4*tempStride : 4*tempStride+8]))
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[5*tempStride : 5*tempStride+8]))
		for row := 0; row < height; row++ {
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[(row+6)*tempStride : (row+6)*tempStride+8]))

			p0 := a0.Add(a6).ReshapeToUint32s()
			p1 := a1.Add(a5).ReshapeToUint32s()
			p2 := a2.Add(a4).ReshapeToUint32s()
			p3 := a3.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			acc = p0.MulAdd(t0V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			odd = p0.Shift(shr16).MulAdd(t0V, odd)
			even := odd.MulAdd(negK, acc)
			v := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16()
			v.Max(zero).Min(maxV).ToBits().StoreArray((*[8]uint16)(dc[row*dstStride : row*dstStride+8]))
			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
		}
	}
}

// wienerHorizontalU8Dav1d5 is the five-tap 8-bit horizontal Wiener pass in
// dav1d's wiener_filter5_h_8bpc_neon shape (src/arm/64/looprestoration.S):
// the symmetric tap pairs are summed in uint16, the products accumulate in
// wrapping int16 lanes, and the centre sample's 128*s3 plus the offsets join
// through one saturating add, whose clamp is libaom's [0, 8191] clamp:
//
//	T = 2^14 + 128*s3 + P + 4,  P = f3*s3 + f2*(s2+s4) + f1*(s1+s5)
//	out = clamp(T >> 3, 0, 8191) == (sat16(P + (128*s3 - 16380)) >> 3) + 4096
//
// (T - 32768 is what the saturating add sees; 32768 is a multiple of 8.) The
// identity needs P itself to fit int16, which wienerU8Dav1dTapsFit checks;
// every filter AV1 can code passes. Eight outputs per step from one 16-byte
// load, as bounds-checked row windows.
func wienerHorizontalU8Dav1d5(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, temp []uint16) {
	f1 := archsimd.BroadcastInt16x8(int16(filter[1]))
	f2 := archsimd.BroadcastInt16x8(int16(filter[2]))
	f3 := archsimd.BroadcastInt16x8(int16(filter[3]))
	off := archsimd.BroadcastInt16x8(1<<14 - 1<<2)
	add := archsimd.BroadcastInt16x8(4096)
	shl7 := archsimd.BroadcastInt16x8(7)
	shr3 := archsimd.BroadcastInt16x8(-3)
	rows := height + 2*WienerHalfwin
	for row := range rows {
		base := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		s := src[base : base+width+8]
		d := temp[row*width : row*width+width]
		for col := 0; col+8 <= width; col += 8 {
			a := archsimd.LoadUint8x16Array((*[16]uint8)(s[col : col+16]))
			lo := a.ExtendLo8ToUint16().ReshapeToUint8s()
			hi := a.HiToLo().ExtendLo8ToUint16().ReshapeToUint8s()
			w1 := hi.ConcatShiftBytesRight(lo, 2).ReshapeToUint16s()
			w2 := hi.ConcatShiftBytesRight(lo, 4).ReshapeToUint16s()
			w3 := hi.ConcatShiftBytesRight(lo, 6).ReshapeToUint16s().BitsToInt16()
			w4 := hi.ConcatShiftBytesRight(lo, 8).ReshapeToUint16s()
			w5 := hi.ConcatShiftBytesRight(lo, 10).ReshapeToUint16s()
			p := w3.Mul(f3)
			p = w2.Add(w4).BitsToInt16().MulAdd(f2, p)
			p = w1.Add(w5).BitsToInt16().MulAdd(f1, p)
			c := w3.Shift(shl7).Sub(off)
			p.AddSaturated(c).Shift(shr3).Add(add).ToBits().StoreArray((*[8]uint16)(d[col : col+8]))
		}
	}
}

// wienerU8Dav1dTapsFit reports whether the int16 product sum of
// wienerHorizontalU8Dav1d5 cannot wrap for any 8-bit input.
func wienerU8Dav1dTapsFit(filter WienerFilter) bool {
	pos, neg := 0, 0
	for _, t := range [...]struct{ c, n int }{{int(filter[1]), 2}, {int(filter[2]), 2}, {int(filter[3]), 1}} {
		if t.c > 0 {
			pos += t.c * t.n
		} else {
			neg += t.c * t.n
		}
	}
	return 255*pos <= 1<<15-1 && 255*neg >= -(1<<15)
}

// Go-native SIMD Wiener horizontal passes for arm64.
//
// The Wiener filter is symmetric (f0==f6, f1==f5, f2==f4, enforced by
// validWienerInfo and re-checked here), so with the libaom center reapplication
// s3<<WienerFilterBits folded into the center tap (f3' = f3 + 128) each output is
//
//	sum = seed + f0*(s0+s6) + f1*(s1+s5) + f2*(s2+s4) + f3'*s3
//
// where seed = offset + 1<<(round0-1) also folds the rounding bias. The symmetric
// pair sums fit a positive int16 lane (u8: <= 510; u16 samples <= 4095: <= 8190).
// The uint16 path uses four widening products per half. The uint8 path packs
// adjacent columns into 32-bit lanes and uses scaled taps, as below.
//
// Window construction uses ConcatShiftBytesRight on reshaped lanes. The last
// load of each row reaches two samples past the three-sample border; the
// wrappers' CanRun checks establish that pad before raw pointer loads.

// wienerHorizontalSIMDTrusted is the uint16 SIMD horizontal pass without the
// per-sample validity scan, used by the decoder-owned trusted entry.
func wienerHorizontalSIMDTrusted(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	if !wienerHorizontalU16SIMDCanRun(len(src), srcStride, srcOrigin, width, height, filter) ||
		!blockFits(len(temp), width, width, height+2*WienerHalfwin) {
		wienerHorizontalTrusted(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
		return
	}
	wienerHorizontalSIMDKernel(src, srcStride, srcOrigin, width, height, filter, bitDepth, round0, max, temp)
}

// wienerHorizontalU16SIMDCanRun reports whether the uint16 SIMD horizontal kernel
// may run: a width of whole 8-column groups, a symmetric filter, and the two-sample
// trailing pad that the 16-byte window load reads.
func wienerHorizontalU16SIMDCanRun(srcLen int, srcStride int, srcOrigin int, width int, height int, filter WienerFilter) bool {
	if width < 8 || width%8 != 0 || !wienerFilterSymmetric(filter) {
		return false
	}
	loadWidth, ok := checkedAdd(width, 2)
	return ok && borderedBlockFits(srcLen, srcStride, srcOrigin, loadWidth, height, WienerHalfwin, WienerHalfwin)
}

// wienerHorizontalSIMDKernel is the uint16 SIMD horizontal pass. The caller must
// have checked wienerHorizontalU16SIMDCanRun.
func wienerHorizontalSIMDKernel(src []uint16, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, bitDepth int, round0 int, max uint16, temp []uint16) {
	if t0, t1, t2, t3, seedA, seedO, ok := wienerHorizontalU16PackedTaps(filter, bitDepth, round0, max); ok {
		wienerHorizontalU16PackedKernel(src, srcStride, srcOrigin, width, height, t0, t1, t2, t3, seedA, seedO, bitDepth, round0, temp)
		return
	}
	limit := int32(1) << (bitDepth + 1 + WienerFilterBits - round0)
	offset := int32(1) << (bitDepth + WienerFilterBits - 1)
	seedV := archsimd.BroadcastInt32x4(offset + roundBias(round0))
	shV := archsimd.BroadcastInt32x4(-int32(round0))
	maxV := archsimd.BroadcastUint16x8(uint16(limit - 1))
	g0 := archsimd.BroadcastInt16x8(filter[0])
	g1 := archsimd.BroadcastInt16x8(filter[1])
	g2 := archsimd.BroadcastInt16x8(filter[2])
	// The center tap absorbs the s3<<WienerFilterBits center reapplication.
	g3 := archsimd.BroadcastInt16x8(filter[3] + (1 << WienerFilterBits))
	hg0, hg1 := g0.HiToLo(), g1.HiToLo()
	hg2, hg3 := g2.HiToLo(), g3.HiToLo()

	rows := height + 2*WienerHalfwin
	if uint(width) > 1<<17 {
		panic("restoration: Wiener row wider than any frame")
	}
	// Each row is a window of the checked footprint (width+8 samples from
	// sBase: the 3-sample border each side and the 2-sample load overrun);
	// with width bounded below, each load and store in it is one compare.
	for row := 0; row < rows; row++ {
		sBase := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		dBase := row * width
		srow, trow := src[sBase:sBase+width+8], temp[dBase:dBase+width]
		for col := 0; col < width; col += 8 {
			v0 := archsimd.LoadUint16x8Array((*[8]uint16)(srow[col : col+8]))    // s0..s7
			v1 := archsimd.LoadUint16x8Array((*[8]uint16)(srow[col+8 : col+16])) // s8..s15
			lb := v0.ReshapeToUint8s()
			hb := v1.ReshapeToUint8s()
			w1 := hb.ConcatShiftBytesRight(lb, 2*1).ReshapeToUint16s().BitsToInt16() // s1..s8
			w2 := hb.ConcatShiftBytesRight(lb, 2*2).ReshapeToUint16s().BitsToInt16() // s2..s9
			w3 := hb.ConcatShiftBytesRight(lb, 2*3).ReshapeToUint16s().BitsToInt16() // s3..s10
			w4 := hb.ConcatShiftBytesRight(lb, 2*4).ReshapeToUint16s().BitsToInt16() // s4..s11
			w5 := hb.ConcatShiftBytesRight(lb, 2*5).ReshapeToUint16s().BitsToInt16() // s5..s12
			w6 := hb.ConcatShiftBytesRight(lb, 2*6).ReshapeToUint16s().BitsToInt16() // s6..s13
			a := v0.BitsToInt16().Add(w6)                                            // s0+s6 <= 8190
			b := w1.Add(w5)                                                          // s1+s5
			c := w2.Add(w4)                                                          // s2+s4
			lo := seedV.Add(a.MulWidenLo(g0)).Add(b.MulWidenLo(g1)).
				Add(c.MulWidenLo(g2)).Add(w3.MulWidenLo(g3))
			hi := seedV.Add(a.HiToLo().MulWidenLo(hg0)).Add(b.HiToLo().MulWidenLo(hg1)).
				Add(c.HiToLo().MulWidenLo(hg2)).Add(w3.HiToLo().MulWidenLo(hg3))
			lo = lo.Shift(shV)
			hi = hi.Shift(shV)
			out := restorationSaturateInt32PairToUint16(lo, hi).Min(maxV)
			out.StoreArray((*[8]uint16)(trow[col : col+8]))
		}
	}
}

// wienerHorizontalU16PackedTaps checks the signed 32-bit bound for the packed
// even/odd column form. Extreme filters outside the bound use the widening path.
func wienerHorizontalU16PackedTaps(filter WienerFilter, bitDepth int, round0 int, max uint16) (t0, t1, t2, t3, seedA, seedO uint32, ok bool) {
	if round0 < 1 || round0 > 16 || bitDepth < 0 || bitDepth+WienerFilterBits-1 > 30 || max > 4095 {
		return 0, 0, 0, 0, 0, 0, false
	}
	limitShift := bitDepth + 1 + WienerFilterBits - round0
	if limitShift < 0 || limitShift > 15 {
		return 0, 0, 0, 0, 0, 0, false
	}
	abs := func(x int64) int64 {
		if x < 0 {
			return -x
		}
		return x
	}
	f0, f1, f2 := int64(filter[0]), int64(filter[1]), int64(filter[2])
	f3 := int64(filter[3]) + (1 << WienerFilterBits)
	c := int64(1)<<(bitDepth+WienerFilterBits-1) + int64(roundBias(round0))
	s := int64(1) << (16 - round0)
	if s*(2*int64(max)*(abs(f0)+abs(f1)+abs(f2))+int64(max)*abs(f3)+c) >= 1<<31 {
		return 0, 0, 0, 0, 0, 0, false
	}
	seedO = uint32(s * c)
	seedA = seedO + seedO<<16
	return uint32(s * f0), uint32(s * f1), uint32(s * f2), uint32(s * f3), seedA, seedO, true
}

// wienerHorizontalU16PackedKernel uses one 32-bit MLA per symmetric tap and
// packed column pair. The odd accumulator removes its contribution from the
// even accumulator's high half; TRN2 assembles eight rounded columns.
func wienerHorizontalU16PackedKernel(src []uint16, srcStride int, srcOrigin int, width int, height int, t0, t1, t2, t3, seedA, seedO uint32, bitDepth int, round0 int, temp []uint16) {
	t0V, t1V := archsimd.BroadcastUint32x4(t0), archsimd.BroadcastUint32x4(t1)
	t2V, t3V := archsimd.BroadcastUint32x4(t2), archsimd.BroadcastUint32x4(t3)
	seedAV, seedOV := archsimd.BroadcastUint32x4(seedA), archsimd.BroadcastUint32x4(seedO)
	shr16 := archsimd.BroadcastInt32x4(-16)
	negK := archsimd.BroadcastUint32x4(0xffff0000)
	zero := archsimd.BroadcastInt16x8(0)
	maxV := archsimd.BroadcastInt16x8(int16((1 << (bitDepth + 1 + WienerFilterBits - round0)) - 1))
	// Each row is a window of the footprint the callers checked
	// (wienerHorizontalU16SIMDCanRun, blockFits): width+8 samples from sBase.
	// With width bounded, each load and store in it is one compare.
	if uint(width) > 1<<17 {
		panic("restoration: Wiener row wider than any frame")
	}
	for row := 0; row < height+2*WienerHalfwin; row++ {
		sBase := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		dBase := row * width
		srow, trow := src[sBase:sBase+width+8], temp[dBase:dBase+width]
		v0 := archsimd.LoadUint16x8Array((*[8]uint16)(srow[0:8]))
		for col := 0; col < width; col += 8 {
			v1 := archsimd.LoadUint16x8Array((*[8]uint16)(srow[col+8 : col+16]))
			lb, hb := v0.ReshapeToUint8s(), v1.ReshapeToUint8s()
			w1 := hb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()
			w2 := hb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()
			w3 := hb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()
			w4 := hb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()
			w5 := hb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s()
			w6 := hb.ConcatShiftBytesRight(lb, 12).ReshapeToUint16s()
			p0 := v0.Add(w6).ReshapeToUint32s()
			p1 := w1.Add(w5).ReshapeToUint32s()
			p2 := w2.Add(w4).ReshapeToUint32s()
			p3 := w3.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			acc = p0.MulAdd(t0V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			odd = p0.Shift(shr16).MulAdd(t0V, odd)
			even := odd.MulAdd(negK, acc)
			out := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16().Max(zero).Min(maxV).ToBits()
			out.StoreArray((*[8]uint16)(trow[col : col+8]))
			v0 = v1
		}
	}
}

// wienerHorizontalU8PackedTaps scales the four symmetric taps so the high 16
// bits of each packed accumulator are the rounded output. The bound ensures
// that the even and odd column sums both fit a signed 32-bit lane.
func wienerHorizontalU8PackedTaps(filter WienerFilter) (t0, t1, t2, t3, seedA, seedO uint32, ok bool) {
	f0, f1, f2 := int64(filter[0]), int64(filter[1]), int64(filter[2])
	f3 := int64(filter[3]) + (1 << WienerFilterBits)
	abs := func(x int64) int64 {
		if x < 0 {
			return -x
		}
		return x
	}
	s := int64(1) << (16 - WienerRound0Bits)
	c := int64(1<<(8+WienerFilterBits-1)) + int64(roundBias(WienerRound0Bits))
	if s*(510*(abs(f0)+abs(f1)+abs(f2))+255*abs(f3)+c) >= 1<<31 {
		return 0, 0, 0, 0, 0, 0, false
	}
	seedO = uint32(s * c)
	seedA = seedO + seedO<<16
	return uint32(s * f0), uint32(s * f1), uint32(s * f2), uint32(s * f3), seedA, seedO, true
}

// wienerHorizontalU8SIMD widens the byte window once, then computes adjacent
// columns with packed 32-bit multiply-accumulates. The odd column accumulator
// removes its contribution from the even column's high half. Unrepresentable
// filters and short buffers use the scalar reference.
func wienerHorizontalU8SIMD(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	t0, t1, t2, t3, seedA, seedO, ok := wienerHorizontalU8PackedTaps(filter)
	if round0 != WienerRound0Bits || !wienerFilterSymmetric(filter) || !ok ||
		!wienerHorizontalU8SIMDCanRun(len(src), srcStride, srcOrigin, width, height) ||
		!blockFits(len(temp), width, width, height+2*WienerHalfwin) {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	t0V, t1V := archsimd.BroadcastUint32x4(t0), archsimd.BroadcastUint32x4(t1)
	t2V, t3V := archsimd.BroadcastUint32x4(t2), archsimd.BroadcastUint32x4(t3)
	seedAV, seedOV := archsimd.BroadcastUint32x4(seedA), archsimd.BroadcastUint32x4(seedO)
	negK := archsimd.BroadcastUint32x4(0xffff0000)
	shr16 := archsimd.BroadcastInt32x4(-16)
	zero16 := archsimd.BroadcastInt16x8(0)
	max16 := archsimd.BroadcastInt16x8(8191)

	rows := height + 2*WienerHalfwin
	if uint(width) > 1<<17 {
		panic("restoration: Wiener row wider than any frame")
	}
	w16 := width &^ 15
	// Each row is a window of the checked footprint (width+8 samples from
	// sBase: the 3-sample border each side and the 2-sample load overrun);
	// with width bounded below, each load and store in it is one compare.
	for row := 0; row < rows; row++ {
		sBase := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		dBase := row * width
		srow, trow := src[sBase:sBase+width+8], temp[dBase:dBase+width]
		for col := 0; col < w16; col += 16 {
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(srow[col : col+16]))   // s0..s15
			v1 := archsimd.LoadUint8x16Array((*[16]uint8)(srow[col+8 : col+24])) // s8..s23
			l0 := v0.ExtendLo8ToUint16()                                         // s0..s7  (u16)
			m0 := v0.HiToLo().ExtendLo8ToUint16()                                // s8..s15 (u16)
			h1 := v1.HiToLo().ExtendLo8ToUint16()                                // s16..s23 (u16)
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			hb := h1.ReshapeToUint8s()

			// Group A: outputs col..col+7, windows over s0..s13.
			w1a := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()  // s1..s8
			w2a := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()  // s2..s9
			w3a := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()  // s3..s10
			w4a := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()  // s4..s11
			w5a := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s() // s5..s12
			w6a := mb.ConcatShiftBytesRight(lb, 12).ReshapeToUint16s() // s6..s13
			aA := l0.Add(w6a)
			bA := w1a.Add(w5a)
			cA := w2a.Add(w4a)
			dA := w3a
			pA0 := aA.ReshapeToUint32s()
			pA1 := bA.ReshapeToUint32s()
			pA2 := cA.ReshapeToUint32s()
			pA3 := dA.ReshapeToUint32s()
			accA := pA3.MulAdd(t3V, seedAV)
			accA = pA2.MulAdd(t2V, accA)
			accA = pA1.MulAdd(t1V, accA)
			accA = pA0.MulAdd(t0V, accA)
			oddA := pA3.Shift(shr16).MulAdd(t3V, seedOV)
			oddA = pA2.Shift(shr16).MulAdd(t2V, oddA)
			oddA = pA1.Shift(shr16).MulAdd(t1V, oddA)
			oddA = pA0.Shift(shr16).MulAdd(t0V, oddA)
			evenA := oddA.MulAdd(negK, accA)
			outA := evenA.ReshapeToUint16s().InterleaveOdd(oddA.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			outA.StoreArray((*[8]uint16)(trow[col : col+8]))

			// Group B: outputs col+8..col+15, windows over s8..s21.
			w1b := hb.ConcatShiftBytesRight(mb, 2).ReshapeToUint16s()  // s9..s16
			w2b := hb.ConcatShiftBytesRight(mb, 4).ReshapeToUint16s()  // s10..s17
			w3b := hb.ConcatShiftBytesRight(mb, 6).ReshapeToUint16s()  // s11..s18
			w4b := hb.ConcatShiftBytesRight(mb, 8).ReshapeToUint16s()  // s12..s19
			w5b := hb.ConcatShiftBytesRight(mb, 10).ReshapeToUint16s() // s13..s20
			w6b := hb.ConcatShiftBytesRight(mb, 12).ReshapeToUint16s() // s14..s21
			aB := m0.Add(w6b)
			bB := w1b.Add(w5b)
			cB := w2b.Add(w4b)
			dB := w3b
			pB0 := aB.ReshapeToUint32s()
			pB1 := bB.ReshapeToUint32s()
			pB2 := cB.ReshapeToUint32s()
			pB3 := dB.ReshapeToUint32s()
			accB := pB3.MulAdd(t3V, seedAV)
			accB = pB2.MulAdd(t2V, accB)
			accB = pB1.MulAdd(t1V, accB)
			accB = pB0.MulAdd(t0V, accB)
			oddB := pB3.Shift(shr16).MulAdd(t3V, seedOV)
			oddB = pB2.Shift(shr16).MulAdd(t2V, oddB)
			oddB = pB1.Shift(shr16).MulAdd(t1V, oddB)
			oddB = pB0.Shift(shr16).MulAdd(t0V, oddB)
			evenB := oddB.MulAdd(negK, accB)
			outB := evenB.ReshapeToUint16s().InterleaveOdd(oddB.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			outB.StoreArray((*[8]uint16)(trow[col+8 : col+16]))
		}
		if w16 != width { // trailing width%16 == 8 group
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(srow[w16 : w16+16])) // s0..s15 (needs s0..s13)
			l0 := v0.ExtendLo8ToUint16()
			m0 := v0.HiToLo().ExtendLo8ToUint16()
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			w1 := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()
			w2 := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()
			w3 := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()
			w4 := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()
			w5 := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s()
			w6 := mb.ConcatShiftBytesRight(lb, 12).ReshapeToUint16s()
			a := l0.Add(w6)
			b := w1.Add(w5)
			c := w2.Add(w4)
			d := w3
			p0 := a.ReshapeToUint32s()
			p1 := b.ReshapeToUint32s()
			p2 := c.ReshapeToUint32s()
			p3 := d.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			acc = p0.MulAdd(t0V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			odd = p0.Shift(shr16).MulAdd(t0V, odd)
			even := odd.MulAdd(negK, acc)
			out := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			out.StoreArray((*[8]uint16)(trow[w16 : w16+8]))
		}
	}
}

// wienerHorizontalU8SIMD5 is wienerHorizontalU8SIMD for filters whose outer
// tap is zero (always for chroma, and the common luma case): dav1d's
// wiener_filter5 path. Dropping the zero tap pair removes its window, add,
// and two multiply-accumulates per group; the sums are unchanged.
func wienerHorizontalU8SIMD5(src []uint8, srcStride int, srcOrigin int, width int, height int, filter WienerFilter, round0 int, temp []uint16) {
	_, t1, t2, t3, seedA, seedO, ok := wienerHorizontalU8PackedTaps(filter)
	if round0 != WienerRound0Bits || !wienerFilterSymmetric(filter) || !ok ||
		!wienerHorizontalU8SIMDCanRun(len(src), srcStride, srcOrigin, width, height) ||
		!blockFits(len(temp), width, width, height+2*WienerHalfwin) {
		wienerHorizontalU8(src, srcStride, srcOrigin, width, height, filter, round0, temp)
		return
	}
	t1V := archsimd.BroadcastUint32x4(t1)
	t2V, t3V := archsimd.BroadcastUint32x4(t2), archsimd.BroadcastUint32x4(t3)
	seedAV, seedOV := archsimd.BroadcastUint32x4(seedA), archsimd.BroadcastUint32x4(seedO)
	negK := archsimd.BroadcastUint32x4(0xffff0000)
	shr16 := archsimd.BroadcastInt32x4(-16)
	zero16 := archsimd.BroadcastInt16x8(0)
	max16 := archsimd.BroadcastInt16x8(8191)

	rows := height + 2*WienerHalfwin
	if uint(width) > 1<<17 {
		panic("restoration: Wiener row wider than any frame")
	}
	w16 := width &^ 15
	// Each row is a window of the checked footprint (width+8 samples from
	// sBase: the 3-sample border each side and the 2-sample load overrun);
	// with width bounded below, each load and store in it is one compare.
	for row := 0; row < rows; row++ {
		sBase := srcOrigin + (row-WienerHalfwin)*srcStride - WienerHalfwin
		dBase := row * width
		srow, trow := src[sBase:sBase+width+8], temp[dBase:dBase+width]
		for col := 0; col < w16; col += 16 {
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(srow[col : col+16]))   // s0..s15
			v1 := archsimd.LoadUint8x16Array((*[16]uint8)(srow[col+8 : col+24])) // s8..s23
			l0 := v0.ExtendLo8ToUint16()                                         // s0..s7  (u16)
			m0 := v0.HiToLo().ExtendLo8ToUint16()                                // s8..s15 (u16)
			h1 := v1.HiToLo().ExtendLo8ToUint16()                                // s16..s23 (u16)
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			hb := h1.ReshapeToUint8s()

			// Group A: outputs col..col+7, windows over s0..s13.
			w1a := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()  // s1..s8
			w2a := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()  // s2..s9
			w3a := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()  // s3..s10
			w4a := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()  // s4..s11
			w5a := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s() // s5..s12
			bA := w1a.Add(w5a)
			cA := w2a.Add(w4a)
			dA := w3a
			pA1 := bA.ReshapeToUint32s()
			pA2 := cA.ReshapeToUint32s()
			pA3 := dA.ReshapeToUint32s()
			accA := pA3.MulAdd(t3V, seedAV)
			accA = pA2.MulAdd(t2V, accA)
			accA = pA1.MulAdd(t1V, accA)
			oddA := pA3.Shift(shr16).MulAdd(t3V, seedOV)
			oddA = pA2.Shift(shr16).MulAdd(t2V, oddA)
			oddA = pA1.Shift(shr16).MulAdd(t1V, oddA)
			evenA := oddA.MulAdd(negK, accA)
			outA := evenA.ReshapeToUint16s().InterleaveOdd(oddA.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			outA.StoreArray((*[8]uint16)(trow[col : col+8]))

			// Group B: outputs col+8..col+15, windows over s8..s21.
			w1b := hb.ConcatShiftBytesRight(mb, 2).ReshapeToUint16s()  // s9..s16
			w2b := hb.ConcatShiftBytesRight(mb, 4).ReshapeToUint16s()  // s10..s17
			w3b := hb.ConcatShiftBytesRight(mb, 6).ReshapeToUint16s()  // s11..s18
			w4b := hb.ConcatShiftBytesRight(mb, 8).ReshapeToUint16s()  // s12..s19
			w5b := hb.ConcatShiftBytesRight(mb, 10).ReshapeToUint16s() // s13..s20
			bB := w1b.Add(w5b)
			cB := w2b.Add(w4b)
			dB := w3b
			pB1 := bB.ReshapeToUint32s()
			pB2 := cB.ReshapeToUint32s()
			pB3 := dB.ReshapeToUint32s()
			accB := pB3.MulAdd(t3V, seedAV)
			accB = pB2.MulAdd(t2V, accB)
			accB = pB1.MulAdd(t1V, accB)
			oddB := pB3.Shift(shr16).MulAdd(t3V, seedOV)
			oddB = pB2.Shift(shr16).MulAdd(t2V, oddB)
			oddB = pB1.Shift(shr16).MulAdd(t1V, oddB)
			evenB := oddB.MulAdd(negK, accB)
			outB := evenB.ReshapeToUint16s().InterleaveOdd(oddB.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			outB.StoreArray((*[8]uint16)(trow[col+8 : col+16]))
		}
		if w16 != width { // trailing width%16 == 8 group
			v0 := archsimd.LoadUint8x16Array((*[16]uint8)(srow[w16 : w16+16])) // s0..s15 (needs s0..s13)
			l0 := v0.ExtendLo8ToUint16()
			m0 := v0.HiToLo().ExtendLo8ToUint16()
			lb := l0.ReshapeToUint8s()
			mb := m0.ReshapeToUint8s()
			w1 := mb.ConcatShiftBytesRight(lb, 2).ReshapeToUint16s()
			w2 := mb.ConcatShiftBytesRight(lb, 4).ReshapeToUint16s()
			w3 := mb.ConcatShiftBytesRight(lb, 6).ReshapeToUint16s()
			w4 := mb.ConcatShiftBytesRight(lb, 8).ReshapeToUint16s()
			w5 := mb.ConcatShiftBytesRight(lb, 10).ReshapeToUint16s()
			b := w1.Add(w5)
			c := w2.Add(w4)
			d := w3
			p1 := b.ReshapeToUint32s()
			p2 := c.ReshapeToUint32s()
			p3 := d.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			even := odd.MulAdd(negK, acc)
			out := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16().Max(zero16).Min(max16).ToBits()
			out.StoreArray((*[8]uint16)(trow[w16 : w16+8]))
		}
	}
}

// Packed-pair Wiener vertical passes for arm64.
//
// The official archsimd API has no widening multiply-accumulate (SMLAL) and no
// high-half widening multiply, so the hand asm's widen+MAC cannot be copied op
// for op. Instead the kernels never widen at all:
//
//   - Symmetric row pairs are summed in uint16 lanes. A temp sample is at most
//     32767 (the horizontal pass clamps to [0,maxClamp], maxClamp <= 32767), so
//     a pair sum is at most 65534 and never carries out of its 16-bit lane.
//   - Each 32-bit lane of a pair vector therefore holds even + 65536*odd for an
//     adjacent (even, odd) column pair. One 32-bit multiply-accumulate (MLA)
//     per tap produces A = E + 65536*O (mod 2^32), where E and O are the even and
//     odd column sums: the even sum exact up to a 65536*O pollution term.
//   - The odd column values are extracted with one logical shift (>>16) and
//     multiply-accumulated separately into O. One more MLA, A - 65536*O, cancels
//     the pollution and leaves the exact even sum.
//   - Taps are pre-scaled by s = 2^(16-round1) and the seed folds -offset, the
//     rounding bias 2^(round1-1) and the center reapplication's offset, so the
//     high 16 bits of each 32-bit accumulator are exactly
//     floor((sum - offset + bias) / 2^round1) = roundPowerOfTwo(sum, round1).
//     One TRN2 gathers those high halves of the even and odd accumulators into
//     column order, replacing the asm's shift + two narrows.
//
// Per eight output columns that is 3 pair adds + 4 odd shifts + 9 MLAs + 1 TRN2,
// against the asm's 7 (u16) or 3 (u8) adds and 14 or 8 SMLALs plus its shift and
// narrow tail. A sliding seven-row window loads each temp row once per column
// strip instead of seven times.
//
// wienerVerticalPackedTaps validates that the packed form is exact and returns
// the scaled taps and seeds. Every intermediate value must fit int32 before
// wrapping: s*|sum - offset + bias| <= s*(32767*sum|f| + |offset - bias|) < 2^31,
// which holds for every valid Wiener filter at bit depths 8, 10 and 12. Filters
// that are not mirrored, or whose taps or rounding would break the bound, report
// ok=false and the caller takes the scalar reference.
func wienerVerticalPackedTaps(filter WienerFilter, bitDepth int, round1 int) (t0, t1, t2, t3, seedA, seedO uint32, ok bool) {
	if !wienerFilterSymmetric(filter) || round1 < 1 || round1 > 16 || bitDepth < 0 || bitDepth+round1-1 > 30 {
		return 0, 0, 0, 0, 0, 0, false
	}
	abs := func(v int64) int64 {
		if v < 0 {
			return -v
		}
		return v
	}
	f0 := int64(filter[0])
	f1 := int64(filter[1])
	f2 := int64(filter[2])
	f3 := int64(filter[3]) + (1 << WienerFilterBits)
	c := int64(roundBias(round1)) - int64(1)<<(bitDepth+round1-1)
	s := int64(1) << (16 - round1)
	bound := s * (32767*(2*abs(f0)+2*abs(f1)+2*abs(f2)+abs(f3)) + abs(c))
	if bound >= 1<<31 {
		return 0, 0, 0, 0, 0, 0, false
	}
	seedO = uint32(s * c)
	// A - 65536*O must equal s*(E + c): A carries seedA, O carries seedO.
	seedA = seedO + seedO<<16
	return uint32(s * f0), uint32(s * f1), uint32(s * f2), uint32(s * f3), seedA, seedO, true
}

// wienerVerticalFootprintFits reports whether the vertical pass's whole read
// footprint (height+6 temp rows of width samples) and write footprint (height
// dst rows of width samples) are resident, with overflow-checked arithmetic.
func wienerVerticalFootprintFits(tempLen int, tempStride int, dstLen int, dstStride int, width int, height int) bool {
	tempRows, ok := checkedAdd(height, 2*WienerHalfwin)
	return ok && height > 0 &&
		blockFits(tempLen, tempStride, width, tempRows) &&
		blockFits(dstLen, dstStride, width, height)
}

// wienerVerticalU8SIMD is the Go-native SIMD form of the 8-bit Wiener vertical
// pass over the horizontal-pass intermediate, rounded by round1 and clamped to
// uint8 (see the packed-pair notes above). The high-half results are signed
// int16 values and SQXTUN's unsigned saturation is exactly clampInt32(x, 0, 255).
//
// Sixteen columns per strip (one 16-byte store per row); a trailing strip of
// eight columns is placed at width-8 when fewer than sixteen remain, recomputing
// any overlapped columns with identical values. Widths below 8, footprints that
// do not fit, and filters the packed form cannot represent take the scalar
// reference.
func wienerVerticalU8SIMD(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	t0, t1, t2, t3, seedA, seedO, ok := wienerVerticalPackedTaps(filter, 8, round1)
	// The stride and height bounds make each row window below one compare.
	if width < 8 || !ok || height > rowsMax || uint(tempStride) > rowStrideMax || uint(dstStride) > rowStrideMax ||
		!wienerVerticalFootprintFits(len(temp), tempStride, len(dst), dstStride, width, height) {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	t0V := archsimd.BroadcastUint32x4(t0)
	t1V := archsimd.BroadcastUint32x4(t1)
	t2V := archsimd.BroadcastUint32x4(t2)
	t3V := archsimd.BroadcastUint32x4(t3)
	seedAV := archsimd.BroadcastUint32x4(seedA)
	seedOV := archsimd.BroadcastUint32x4(seedO)
	negK := archsimd.BroadcastUint32x4(0xffff0000) // -65536
	shr16 := archsimd.BroadcastInt32x4(-16)

	// Each row is read and written through one bounds-checked window.

	col := 0
	for ; col+16 <= width; col += 16 {
		tc, dc := temp[col:], dst[col:]
		w1 := (*[16]uint16)(tc[0*tempStride : 0*tempStride+16])
		a0 := archsimd.LoadUint16x8Array((*[8]uint16)(w1[0:8]))
		b0 := archsimd.LoadUint16x8Array((*[8]uint16)(w1[8:16]))
		w2 := (*[16]uint16)(tc[1*tempStride : 1*tempStride+16])
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(w2[0:8]))
		b1 := archsimd.LoadUint16x8Array((*[8]uint16)(w2[8:16]))
		w3 := (*[16]uint16)(tc[2*tempStride : 2*tempStride+16])
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(w3[0:8]))
		b2 := archsimd.LoadUint16x8Array((*[8]uint16)(w3[8:16]))
		w4 := (*[16]uint16)(tc[3*tempStride : 3*tempStride+16])
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(w4[0:8]))
		b3 := archsimd.LoadUint16x8Array((*[8]uint16)(w4[8:16]))
		w5 := (*[16]uint16)(tc[4*tempStride : 4*tempStride+16])
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(w5[0:8]))
		b4 := archsimd.LoadUint16x8Array((*[8]uint16)(w5[8:16]))
		w6 := (*[16]uint16)(tc[5*tempStride : 5*tempStride+16])
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(w6[0:8]))
		b5 := archsimd.LoadUint16x8Array((*[8]uint16)(w6[8:16]))
		for row := 0; row < height; row++ {
			w7 := (*[16]uint16)(tc[(row+6)*tempStride : (row+6)*tempStride+16])
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(w7[0:8]))
			b6 := archsimd.LoadUint16x8Array((*[8]uint16)(w7[8:16]))

			pa0 := a0.Add(a6).ReshapeToUint32s()
			pa1 := a1.Add(a5).ReshapeToUint32s()
			pa2 := a2.Add(a4).ReshapeToUint32s()
			pa3 := a3.ReshapeToUint32s()
			accA := pa3.MulAdd(t3V, seedAV)
			accA = pa2.MulAdd(t2V, accA)
			accA = pa1.MulAdd(t1V, accA)
			accA = pa0.MulAdd(t0V, accA)
			oddA := pa3.Shift(shr16).MulAdd(t3V, seedOV)
			oddA = pa2.Shift(shr16).MulAdd(t2V, oddA)
			oddA = pa1.Shift(shr16).MulAdd(t1V, oddA)
			oddA = pa0.Shift(shr16).MulAdd(t0V, oddA)
			evenA := oddA.MulAdd(negK, accA)
			va := evenA.ReshapeToUint16s().InterleaveOdd(oddA.ReshapeToUint16s()).BitsToInt16()

			pb0 := b0.Add(b6).ReshapeToUint32s()
			pb1 := b1.Add(b5).ReshapeToUint32s()
			pb2 := b2.Add(b4).ReshapeToUint32s()
			pb3 := b3.ReshapeToUint32s()
			accB := pb3.MulAdd(t3V, seedAV)
			accB = pb2.MulAdd(t2V, accB)
			accB = pb1.MulAdd(t1V, accB)
			accB = pb0.MulAdd(t0V, accB)
			oddB := pb3.Shift(shr16).MulAdd(t3V, seedOV)
			oddB = pb2.Shift(shr16).MulAdd(t2V, oddB)
			oddB = pb1.Shift(shr16).MulAdd(t1V, oddB)
			oddB = pb0.Shift(shr16).MulAdd(t0V, oddB)
			evenB := oddB.MulAdd(negK, accB)
			vb := evenB.ReshapeToUint16s().InterleaveOdd(oddB.ReshapeToUint16s()).BitsToInt16()

			restorationSaturateInt16PairToUint8(va, vb).StoreArray((*[16]uint8)(dc[row*dstStride : row*dstStride+16]))
			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
			b0, b1, b2, b3, b4, b5 = b1, b2, b3, b4, b5, b6

		}
	}
	// Trailing columns: eight-wide strips, the last one shifted left to end at
	// width so it overlaps already-written columns instead of running past them.
	for ; col < width; col += 8 {
		c := min(col, width-8)
		tc, dc := temp[c:], dst[c:]
		a0 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[0*tempStride : 0*tempStride+8]))
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[1*tempStride : 1*tempStride+8]))
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[2*tempStride : 2*tempStride+8]))
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[3*tempStride : 3*tempStride+8]))
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[4*tempStride : 4*tempStride+8]))
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[5*tempStride : 5*tempStride+8]))
		for row := 0; row < height; row++ {
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[(row+6)*tempStride : (row+6)*tempStride+8]))

			p0 := a0.Add(a6).ReshapeToUint32s()
			p1 := a1.Add(a5).ReshapeToUint32s()
			p2 := a2.Add(a4).ReshapeToUint32s()
			p3 := a3.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			acc = p0.MulAdd(t0V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			odd = p0.Shift(shr16).MulAdd(t0V, odd)
			even := odd.MulAdd(negK, acc)
			v := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(dc[row*dstStride:row*dstStride+8], v.SaturateToUint8().ReshapeToUint64s().GetElem(0))

			a0, a1, a2, a3, a4, a5 = a1, a2, a3, a4, a5, a6
		}
	}
}

// wienerVerticalU8SIMD5 is wienerVerticalU8SIMD for filters whose outer tap
// is zero (dav1d wiener_filter5): rows 0 and 6 of each window carry no weight.
func wienerVerticalU8SIMD5(temp []uint16, tempStride int, dst []uint8, dstStride int, width int, height int, filter WienerFilter, round1 int) {
	_, t1, t2, t3, seedA, seedO, ok := wienerVerticalPackedTaps(filter, 8, round1)
	// The stride and height bounds make each row window below one compare.
	if width < 8 || !ok || height > rowsMax || uint(tempStride) > rowStrideMax || uint(dstStride) > rowStrideMax ||
		!wienerVerticalFootprintFits(len(temp), tempStride, len(dst), dstStride, width, height) {
		wienerVerticalU8(temp, tempStride, dst, dstStride, width, height, filter, round1)
		return
	}
	t1V := archsimd.BroadcastUint32x4(t1)
	t2V := archsimd.BroadcastUint32x4(t2)
	t3V := archsimd.BroadcastUint32x4(t3)
	seedAV := archsimd.BroadcastUint32x4(seedA)
	seedOV := archsimd.BroadcastUint32x4(seedO)
	negK := archsimd.BroadcastUint32x4(0xffff0000) // -65536
	shr16 := archsimd.BroadcastInt32x4(-16)

	// Each row is read and written through one bounds-checked window.

	col := 0
	for ; col+16 <= width; col += 16 {
		tc, dc := temp[col:], dst[col:]
		w8 := (*[16]uint16)(tc[1*tempStride : 1*tempStride+16])
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(w8[0:8]))
		b1 := archsimd.LoadUint16x8Array((*[8]uint16)(w8[8:16]))
		w9 := (*[16]uint16)(tc[2*tempStride : 2*tempStride+16])
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(w9[0:8]))
		b2 := archsimd.LoadUint16x8Array((*[8]uint16)(w9[8:16]))
		w10 := (*[16]uint16)(tc[3*tempStride : 3*tempStride+16])
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(w10[0:8]))
		b3 := archsimd.LoadUint16x8Array((*[8]uint16)(w10[8:16]))
		w11 := (*[16]uint16)(tc[4*tempStride : 4*tempStride+16])
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(w11[0:8]))
		b4 := archsimd.LoadUint16x8Array((*[8]uint16)(w11[8:16]))
		w12 := (*[16]uint16)(tc[5*tempStride : 5*tempStride+16])
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(w12[0:8]))
		b5 := archsimd.LoadUint16x8Array((*[8]uint16)(w12[8:16]))
		for row := 0; row < height; row++ {
			w13 := (*[16]uint16)(tc[(row+6)*tempStride : (row+6)*tempStride+16])
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(w13[0:8]))
			b6 := archsimd.LoadUint16x8Array((*[8]uint16)(w13[8:16]))

			pa1 := a1.Add(a5).ReshapeToUint32s()
			pa2 := a2.Add(a4).ReshapeToUint32s()
			pa3 := a3.ReshapeToUint32s()
			accA := pa3.MulAdd(t3V, seedAV)
			accA = pa2.MulAdd(t2V, accA)
			accA = pa1.MulAdd(t1V, accA)
			oddA := pa3.Shift(shr16).MulAdd(t3V, seedOV)
			oddA = pa2.Shift(shr16).MulAdd(t2V, oddA)
			oddA = pa1.Shift(shr16).MulAdd(t1V, oddA)
			evenA := oddA.MulAdd(negK, accA)
			va := evenA.ReshapeToUint16s().InterleaveOdd(oddA.ReshapeToUint16s()).BitsToInt16()

			pb1 := b1.Add(b5).ReshapeToUint32s()
			pb2 := b2.Add(b4).ReshapeToUint32s()
			pb3 := b3.ReshapeToUint32s()
			accB := pb3.MulAdd(t3V, seedAV)
			accB = pb2.MulAdd(t2V, accB)
			accB = pb1.MulAdd(t1V, accB)
			oddB := pb3.Shift(shr16).MulAdd(t3V, seedOV)
			oddB = pb2.Shift(shr16).MulAdd(t2V, oddB)
			oddB = pb1.Shift(shr16).MulAdd(t1V, oddB)
			evenB := oddB.MulAdd(negK, accB)
			vb := evenB.ReshapeToUint16s().InterleaveOdd(oddB.ReshapeToUint16s()).BitsToInt16()

			restorationSaturateInt16PairToUint8(va, vb).StoreArray((*[16]uint8)(dc[row*dstStride : row*dstStride+16]))
			a1, a2, a3, a4, a5 = a2, a3, a4, a5, a6
			b1, b2, b3, b4, b5 = b2, b3, b4, b5, b6

		}
	}
	// Trailing columns: eight-wide strips, the last one shifted left to end at
	// width so it overlaps already-written columns instead of running past them.
	for ; col < width; col += 8 {
		c := min(col, width-8)
		tc, dc := temp[c:], dst[c:]
		a1 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[1*tempStride : 1*tempStride+8]))
		a2 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[2*tempStride : 2*tempStride+8]))
		a3 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[3*tempStride : 3*tempStride+8]))
		a4 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[4*tempStride : 4*tempStride+8]))
		a5 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[5*tempStride : 5*tempStride+8]))
		for row := 0; row < height; row++ {
			a6 := archsimd.LoadUint16x8Array((*[8]uint16)(tc[(row+6)*tempStride : (row+6)*tempStride+8]))

			p1 := a1.Add(a5).ReshapeToUint32s()
			p2 := a2.Add(a4).ReshapeToUint32s()
			p3 := a3.ReshapeToUint32s()
			acc := p3.MulAdd(t3V, seedAV)
			acc = p2.MulAdd(t2V, acc)
			acc = p1.MulAdd(t1V, acc)
			odd := p3.Shift(shr16).MulAdd(t3V, seedOV)
			odd = p2.Shift(shr16).MulAdd(t2V, odd)
			odd = p1.Shift(shr16).MulAdd(t1V, odd)
			even := odd.MulAdd(negK, acc)
			v := even.ReshapeToUint16s().InterleaveOdd(odd.ReshapeToUint16s()).BitsToInt16()
			binary.LittleEndian.PutUint64(dc[row*dstStride:row*dstStride+8], v.SaturateToUint8().ReshapeToUint64s().GetElem(0))

			a1, a2, a3, a4, a5 = a2, a3, a4, a5, a6
		}
	}
}
