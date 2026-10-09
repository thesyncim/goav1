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
// The lane moves, widening squares and bit lengths are shared helpers with
// arch-specific forms (gosimd_ops_*.go, gosimd_bitlen.go).

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	residualBlockImpl = residualBlockSIMD
	rdStatsBlockImpl = rdStatsBlockSIMD
	blockErrorImpl = blockErrorSIMD
}

// residualBlockSIMD extracts dst[r*w+c] = src - pred. Sixteen-wide rows take 16
// columns per vector; eight-wide rows pack two rows into one vector; anything
// else is scalar.
func residualBlockSIMD(dst []int16, srcPlane []byte, srcOff, stride int, pred []byte, predStride, w, h int) {
	if w < 8 {
		residualBlockPureGo(dst, srcPlane, srcOff, stride, pred, predStride, w, h)
		return
	}
	r := 0
	if w == 8 {
		for ; r+2 <= h; r += 2 {
			a := packRows8(srcPlane, srcOff+r*stride, stride)
			b := packRows8(pred, r*predStride, predStride)
			lo := a.ExtendLo8ToUint16().BitsToInt16().Sub(b.ExtendLo8ToUint16().BitsToInt16())
			hi := widenHi8(a).BitsToInt16().Sub(widenHi8(b).BitsToInt16())
			lo.Store(dst[r*8 : r*8+8])
			hi.Store(dst[(r+1)*8 : (r+1)*8+8])
		}
	}
	for ; r < h; r++ {
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
// per lane is 2 + bits.Len16(|level|) for a non-zero level and 0 otherwise.
func rdStatsBlockSIMD(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (dskip, dcode int64, rate int64, allZero bool) {
	stepV := archsimd.BroadcastInt32x4(step)
	tsU := uint64(ts)
	one := archsimd.BroadcastUint32x4(1)
	skipAcc := archsimd.BroadcastInt64x2(0)
	codeAcc := archsimd.BroadcastInt64x2(0)
	orAcc := archsimd.BroadcastInt16x8(0)
	rateAcc := archsimd.BroadcastInt32x4(0)
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
		// |level| as unsigned 16-bit (|-32768| = 32768), split into 32-bit lanes.
		mag := q.Abs().ToBits()
		lo := mag.ExtendLo4ToUint32()
		hiMag := hiInt16(q.Abs()).ToBits().ExtendLo4ToUint32()
		nzLo := lo.Min(one).BitsToInt32()
		nzHi := hiMag.Min(one).BitsToInt32()
		rateAcc = rateAcc.Add(bitLenFloat32(lo)).Add(bitLenFloat32(hiMag)).
			Add(nzLo.Add(nzLo)).Add(nzHi.Add(nzHi))
	}
	var lanes int64
	allZero = true
	for ; i < count; i++ {
		c := int64(tran[i])
		dskip += c * c
		v := qcoeff[i]
		if v == 0 {
			dcode += c * c
			continue
		}
		e := c - (int64(v)*int64(step))>>ts
		dcode += e * e
		level := v
		if level < 0 {
			level = -level
		}
		lanes += int64(2 + bits.Len16(uint16(level)))
		allZero = false
	}
	for l := range uint8(4) {
		lanes += int64(rateAcc.GetElem(l))
	}
	dskip += skipAcc.GetElem(0) + skipAcc.GetElem(1)
	dcode += codeAcc.GetElem(0) + codeAcc.GetElem(1)
	for l := range uint8(8) {
		if orAcc.GetElem(l) != 0 {
			allZero = false
		}
	}
	return dskip, dcode, lanes << 9, allZero
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
