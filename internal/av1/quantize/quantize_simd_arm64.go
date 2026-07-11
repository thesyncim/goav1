//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"simd/archsimd"
	"unsafe"
)

const quantizeSIMDMaxStep = int32(65535)

func quantizeBlockSIMD(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	if count == 0 || q.AC > quantizeSIMDMaxStep {
		return false
	}
	satAbs := quantizeSIMDSaturationAbs(q.AC, txScale)
	if satAbs <= 0 {
		return false
	}
	recip := int32(0)
	if q.AC != 1 {
		recip = int32((int64(1) << 31) / int64(q.AC))
	}

	i := 0
	for ; i+8 <= count; i += 8 {
		c0 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&coeff[i])))
		c1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&coeff[i+4])))
		if quantizeSIMDHasMinInt32(c0) || quantizeSIMDHasMinInt32(c1) {
			return false
		}
		q0 := quantizeTrunc4SIMD(c0, q.AC, recip, satAbs, txScale)
		q1 := quantizeTrunc4SIMD(c1, q.AC, recip, satAbs, txScale)
		q0.TruncToInt16().TruncToInt16Hi(q1).StoreArray((*[8]int16)(unsafe.Pointer(&qcoeff[i])))
	}
	for ; i < count; i++ {
		if coeff[i] == minInt32 {
			return false
		}
		qcoeff[i] = quantizeScalar(coeff[i], q.AC, txScale)
	}
	qcoeff[0] = quantizeScalar(coeff[0], q.DC, txScale)
	return true
}

func quantizeFPBlockSIMD(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	if count == 0 || q.AC > quantizeSIMDMaxStep || q.DC > quantizeSIMDMaxStep {
		return false
	}
	quantAC := int64(1<<16) / int64(q.AC)
	quantDC := int64(1<<16) / int64(q.DC)
	if quantAC > 1<<14 || quantDC > 1<<14 {
		return false
	}
	roundAC := roundPowerOfTwo((64*int32(q.AC))>>7, txScale)

	// Hoist all constant vectors out of the group loop; the runtime shift
	// (16-txScale) is loop-invariant so precompute its (negative = right) amount
	// vector once and use Shift() instead of ShiftAllRight()'s per-call rebuild.
	maxV := archsimd.BroadcastInt32x4(maxInt16)
	roundV := archsimd.BroadcastInt32x4(roundAC)
	quantV := archsimd.BroadcastInt32x4(int32(quantAC))
	dequantV := archsimd.BroadcastInt32x4(int32(q.AC))
	lshV := archsimd.BroadcastInt32x4(int32(1 + txScale))
	rshV := archsimd.BroadcastInt32x4(-int32(16 - txScale))
	// No per-group overflow guard. The keep mask reproduces the scalar's exact
	// test `abs<<(1+txScale) >= dequant` on the *raw* (unclamped) abs, so it
	// matches the scalar for every int32 coeff — including extremes where that
	// left shift overflows to a negative value and zeroes the coeff (minInt32,
	// maxInt32, anything past maxSafe). Abs() then Min(maxV) bounds the multiply.
	// The NEON asm likewise scans nothing. Walk raw pointers: no bounds check.
	cp := unsafe.Pointer(&coeff[0])
	qp := unsafe.Pointer(&qcoeff[0])
	i := 0
	for ; i+8 <= count; i += 8 {
		c0 := archsimd.LoadInt32x4Array((*[4]int32)(cp))
		c1 := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(cp, 16)))
		raw0 := c0.Abs()
		keep0 := raw0.Shift(lshV).GreaterEqual(dequantV)
		m0 := raw0.Add(roundV).Min(maxV).Mul(quantV).Shift(rshV).Masked(keep0)
		q0 := quantizeApplySignSIMD(m0, c0)
		raw1 := c1.Abs()
		keep1 := raw1.Shift(lshV).GreaterEqual(dequantV)
		m1 := raw1.Add(roundV).Min(maxV).Mul(quantV).Shift(rshV).Masked(keep1)
		q1 := quantizeApplySignSIMD(m1, c1)
		q0.TruncToInt16().TruncToInt16Hi(q1).StoreArray((*[8]int16)(qp))
		cp = unsafe.Add(cp, 32)
		qp = unsafe.Add(qp, 16)
	}
	for ; i < count; i++ {
		if coeff[i] == minInt32 || quantizeScalarFPUnsafeShift(coeff[i], txScale) {
			return false
		}
		qcoeff[i] = quantizeScalarFP(coeff[i], q.AC, quantAC, roundAC, txScale)
	}
	roundDC := roundPowerOfTwo((64*int32(q.DC))>>7, txScale)
	qcoeff[0] = quantizeScalarFP(coeff[0], q.DC, quantDC, roundDC, txScale)
	return true
}

func quantizeFPNoQMatrixSIMD(qcoeff []int32, dqcoeff []int32, coeff []int32, scan []int16, q FPQuantizer) (int, bool) {
	count := len(scan)
	for i, rc := range scan {
		if int(rc) != i {
			return 0, false
		}
	}

	rounding := [2]int32{
		roundPowerOfTwo(int32(q.Round[0]), q.LogScale),
		roundPowerOfTwo(int32(q.Round[1]), q.LogScale),
	}
	eob := 0
	qc, dqc, nonzero := quantizeFPNoQMatrixScalarCoeff(coeff[0], 0, q, rounding)
	qcoeff[0] = qc
	dqcoeff[0] = dqc
	if nonzero {
		eob = 1
	}

	laneIndex := [4]int32{1, 2, 3, 4}
	laneIndexV := archsimd.LoadInt32x4Array(&laneIndex)
	zero := archsimd.BroadcastInt32x4(0)
	i := 1
	for ; i+4 <= count; i += 4 {
		c := archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Pointer(&coeff[i])))
		qv, dqv, mag := quantizeFPNoQMatrix4SIMD(c, int32(q.Quant[1]), int32(q.Dequant[1]), rounding[1], q.LogScale)
		qv.StoreArray((*[4]int32)(unsafe.Pointer(&qcoeff[i])))
		dqv.StoreArray((*[4]int32)(unsafe.Pointer(&dqcoeff[i])))
		if lane := laneIndexV.Masked(mag.NotEqual(zero)).ReduceMax(); lane != 0 {
			eob = i + int(lane)
		}
	}
	for ; i < count; i++ {
		qc, dqc, nonzero = quantizeFPNoQMatrixScalarCoeff(coeff[i], 1, q, rounding)
		qcoeff[i] = qc
		dqcoeff[i] = dqc
		if nonzero {
			eob = i + 1
		}
	}
	return eob, true
}

func quantizeTrunc4SIMD(c archsimd.Int32x4, scale int32, recip int32, satAbs int32, txScale uint8) archsimd.Int32x4 {
	satV := archsimd.BroadcastInt32x4(satAbs)
	abs := c.Max(archsimd.BroadcastInt32x4(-satAbs)).Min(satV).Abs()
	divAbs := abs.Min(archsimd.BroadcastInt32x4(satAbs - 1))
	num := divAbs.ShiftAllLeft(uint64(txScale))
	level := num
	if scale != 1 {
		level = quantizeDiv32SIMD(num, scale, recip)
	}
	level = archsimd.BroadcastInt32x4(maxInt16).IfElse(abs.GreaterEqual(satV), level)
	return quantizeApplySignSIMD(level, c)
}

func quantizeDiv32SIMD(num archsimd.Int32x4, scale int32, recip int32) archsimd.Int32x4 {
	recipV := archsimd.BroadcastInt32x4(recip)
	lo := num.MulWidenLo(recipV).ShiftAllRightConst(31).TruncToInt32()
	hi := num.HiToLo().MulWidenLo(recipV).ShiftAllRightConst(31).TruncToInt32()
	q := quantizePackLo2Int32(lo, hi)
	scaleV := archsimd.BroadcastInt32x4(scale)
	rem := num.Sub(q.Mul(scaleV))
	return q.Add(archsimd.BroadcastInt32x4(1).Masked(rem.GreaterEqual(scaleV)))
}

func quantizeFP4SIMD(c archsimd.Int32x4, dequant int32, quant int32, round int32, txScale uint8) archsimd.Int32x4 {
	return quantizeFPMag4SIMD(c, dequant, quant, round, txScale)
}

func quantizeFPMag4SIMD(c archsimd.Int32x4, dequant int32, quant int32, round int32, txScale uint8) archsimd.Int32x4 {
	abs := c.Max(archsimd.BroadcastInt32x4(-maxInt16)).Min(archsimd.BroadcastInt32x4(maxInt16)).Abs()
	threshold := archsimd.BroadcastInt32x4(quantizeSIMDThreshold(dequant, 1+txScale))
	mag := abs.Add(archsimd.BroadcastInt32x4(round)).Min(archsimd.BroadcastInt32x4(maxInt16))
	mag = mag.Mul(archsimd.BroadcastInt32x4(quant)).ShiftAllRight(uint64(16 - txScale))
	mag = mag.Masked(abs.GreaterEqual(threshold))
	return quantizeApplySignSIMD(mag, c)
}

func quantizeFPNoQMatrix4SIMD(c archsimd.Int32x4, quant int32, dequant int32, round int32, logScale uint8) (archsimd.Int32x4, archsimd.Int32x4, archsimd.Int32x4) {
	abs := c.Max(archsimd.BroadcastInt32x4(-maxInt16)).Min(archsimd.BroadcastInt32x4(maxInt16)).Abs()
	threshold := archsimd.BroadcastInt32x4(quantizeSIMDThreshold(dequant, 1+logScale))
	mag := abs.Add(archsimd.BroadcastInt32x4(round)).Min(archsimd.BroadcastInt32x4(maxInt16))
	mag = mag.Mul(archsimd.BroadcastInt32x4(quant)).ShiftAllRight(uint64(16 - logScale))
	mag = mag.Masked(abs.GreaterEqual(threshold))
	qc := quantizeApplySignSIMD(mag, c)
	dqcMag := mag.Mul(archsimd.BroadcastInt32x4(dequant))
	if logScale != 0 {
		dqcMag = dqcMag.ShiftAllRight(uint64(logScale))
	}
	return qc, quantizeApplySignSIMD(dqcMag, c), mag
}

func quantizeApplySignSIMD(mag archsimd.Int32x4, src archsimd.Int32x4) archsimd.Int32x4 {
	// Branchless conditional negate: signMask is all-ones where src<0, so
	// mag^signMask - signMask is -mag there and mag elsewhere. No zero broadcast,
	// no select — shorter dependency chain than 0.Sub(mag).IfElse(src<0, mag).
	signMask := src.ShiftAllRightConst(31)
	return mag.Xor(signMask).Sub(signMask)
}

func quantizePackLo2Int32(lo archsimd.Int32x4, hi archsimd.Int32x4) archsimd.Int32x4 {
	return lo.ToBits().ReshapeToUint64s().InterleaveLo(hi.ToBits().ReshapeToUint64s()).ReshapeToUint32s().BitsToInt32()
}

func quantizeSIMDHasMinInt32(v archsimd.Int32x4) bool {
	return v.ReduceMin() == minInt32
}

func quantizeSIMDHasFPUnsafeShift(v archsimd.Int32x4, txScale uint8) bool {
	maxSafe := int32(maxInt32 >> (1 + txScale))
	return v.ReduceMax() > maxSafe || v.ReduceMin() < -maxSafe
}

func quantizeScalarFPUnsafeShift(coeff int32, txScale uint8) bool {
	maxSafe := int32(maxInt32 >> (1 + txScale))
	return coeff > maxSafe || coeff < -maxSafe
}

func quantizeSIMDSaturationAbs(scale int32, txScale uint8) int32 {
	num := int64(maxInt16+1) * int64(scale)
	den := int64(1) << txScale
	return int32((num + den - 1) / den)
}

func quantizeSIMDThreshold(dequant int32, shift uint8) int32 {
	den := int32(1) << shift
	return (dequant + den - 1) >> shift
}
