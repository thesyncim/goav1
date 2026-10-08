//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/bits"
	"simd/archsimd"
)

// rdstats_gosimd.go hosts Go-native SIMD (simd/archsimd) forms of the encoder's
// residual extraction, skip-decision RD statistics and coefficient block error.
// They are exact against the scalar references (residualBlockPureGo,
// rdStatsBlockPureGo, blockErrorPureGo): products that can exceed 32 bits widen
// to int64 before they are accumulated, and the int16 narrowing that SVT's
// load_tran_low_to_s16q applies is reproduced with a shift pair on int32 lanes.
// The lane moves and widening squares are arch-specific (gosimd_ops_*.go).

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	residualBlockImpl = residualBlockSIMD
	rdStatsBlockImpl = rdStatsBlockSIMD
	blockErrorImpl = blockErrorSIMD
}

// residualBlockSIMD extracts dst[r*w+c] = src - pred sixteen columns at a time.
// Eight-wide rows and tails are scalar: a 128-bit load would read past them.
func residualBlockSIMD(dst []int16, srcPlane []byte, srcOff, stride int, pred []byte, predStride, w, h int) {
	if w < 8 {
		residualBlockPureGo(dst, srcPlane, srcOff, stride, pred, predStride, w, h)
		return
	}
	for r := range h {
		srow := srcPlane[srcOff+r*stride:]
		prow := pred[r*predStride:]
		drow := dst[r*w : r*w+w]
		c := 0
		for ; c+16 <= w; c += 16 {
			a := archsimd.LoadUint8x16(srow[c : c+16])
			b := archsimd.LoadUint8x16(prow[c : c+16])
			a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16()).Store(drow[c : c+8])
			widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16()).Store(drow[c+8 : c+16])
		}
		for ; c < w; c++ {
			drow[c] = int16(srow[c]) - int16(prow[c])
		}
	}
}

// sext16 sign-extends the low 16 bits of each int32 lane (the int16 narrowing).
func sext16(v archsimd.Int32x4) archsimd.Int32x4 {
	return v.ShiftAllLeft(16).ShiftAllRight(16)
}

// rdStatsBlockSIMD mirrors rdStatsBlockPureGo. For a zero level the AC-step
// dequant is zero, so e = c and the code term is c*c, matching the branch in
// the reference without a branch in the vector body. The 32-bit dequant product
// is exact: |level| <= 2^15 and step < 2^16 at every bit depth. The rate term
// (bit length of each level) stays scalar: archsimd lacks a 16-bit leading-zero
// count on amd64.
func rdStatsBlockSIMD(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (dskip, dcode int64, rate int64, allZero bool) {
	stepV := archsimd.BroadcastInt32x4(step)
	tsU := uint64(ts)
	skipAcc := archsimd.BroadcastInt64x2(0)
	codeAcc := archsimd.BroadcastInt64x2(0)
	orAcc := archsimd.BroadcastInt16x8(0)
	i := 0
	for ; i+8 <= count; i += 8 {
		c0 := archsimd.LoadInt32x4(tran[i : i+4])
		c1 := archsimd.LoadInt32x4(tran[i+4 : i+8])
		q := archsimd.LoadInt16x8(qcoeff[i : i+8])
		skipAcc = sqSumInt32(sqSumInt32(skipAcc, c0), c1)
		dq0 := q.ExtendLo4ToInt32().Mul(stepV).ShiftAllRight(tsU)
		dq1 := hiInt16(q).ExtendLo4ToInt32().Mul(stepV).ShiftAllRight(tsU)
		codeAcc = sqSumInt32(sqSumInt32(codeAcc, c0.Sub(dq0)), c1.Sub(dq1))
		orAcc = orAcc.Or(q)
	}
	tailNZ := false
	for ; i < count; i++ {
		c := int64(tran[i])
		dskip += c * c
		v := qcoeff[i]
		if v == 0 {
			dcode += c * c
			continue
		}
		tailNZ = true
		e := c - (int64(v)*int64(step))>>ts
		dcode += e * e
	}
	for j := range count {
		if v := qcoeff[j]; v != 0 {
			level := v
			if level < 0 {
				level = -level
			}
			rate += int64(2 + bits.Len16(uint16(level)))
		}
	}
	dskip += skipAcc.GetElem(0) + skipAcc.GetElem(1)
	dcode += codeAcc.GetElem(0) + codeAcc.GetElem(1)
	allZero = !tailNZ
	for l := range uint8(8) {
		if orAcc.GetElem(l) != 0 {
			allZero = false
		}
	}
	return dskip, dcode, rate << 9, allZero
}

// blockErrorSIMD mirrors blockErrorPureGo: coefficients narrow to int16, the
// error is the square of the 16-bit difference, and ssz the square of the
// narrowed coefficient. Products are at most 2^30, so they fit an int32 lane
// before widening to the int64 accumulators.
func blockErrorSIMD(coeff []int32, dqcoeff []int32, count int) (err int64, ssz int64) {
	if count <= 0 {
		return 0, 0
	}
	_ = coeff[count-1]
	_ = dqcoeff[count-1]
	errAcc := archsimd.BroadcastInt64x2(0)
	sszAcc := archsimd.BroadcastInt64x2(0)
	i := 0
	for ; i+4 <= count; i += 4 {
		c := sext16(archsimd.LoadInt32x4(coeff[i : i+4]))
		d := sext16(archsimd.LoadInt32x4(dqcoeff[i : i+4]))
		errAcc = sqSumInt32(errAcc, sext16(c.Sub(d)))
		sszAcc = sqSumInt32(sszAcc, c)
	}
	err = errAcc.GetElem(0) + errAcc.GetElem(1)
	ssz = sszAcc.GetElem(0) + sszAcc.GetElem(1)
	for ; i < count; i++ {
		c := int16(coeff[i])
		d := int16(dqcoeff[i])
		diff := int64(int16(c - d))
		err += diff * diff
		cv := int64(c)
		ssz += cv * cv
	}
	return err, ssz
}
