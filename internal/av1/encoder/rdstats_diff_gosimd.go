//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"math/bits"
	"simd/archsimd"
)

// rdStatsDiffSIMD mirrors rdStatsDiffPureGo eight levels per step. A step
// whose levels are all zero contributes nothing and is skipped; otherwise
// dq*(2c - dq) is exact in int64 lanes (dq is zero on zero levels) and the
// rate lanes follow rdStatsBlockSIMD.
func rdStatsDiffSIMD(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (diff, rate int64, allZero bool) {
	stepV := archsimd.BroadcastInt32x4(step)
	tsU := uint64(ts)
	one := archsimd.BroadcastUint32x4(1)
	diffAcc := archsimd.BroadcastInt64x2(0)
	rateAcc := archsimd.BroadcastInt32x4(0)
	allZero = true
	i := 0
	for ; i+8 <= count; i += 8 {
		q := archsimd.LoadInt16x8(qcoeff[i : i+8])
		u := q.ToBits().ReshapeToUint64s()
		if u.GetElem(0)|u.GetElem(1) == 0 {
			continue
		}
		allZero = false
		c0 := archsimd.LoadInt32x4(tran[i : i+4])
		c1 := archsimd.LoadInt32x4(tran[i+4 : i+8])
		dq0 := q.ExtendLo4ToInt32().Mul(stepV).ShiftAllRight(tsU)
		dq1 := hiInt16(q).ExtendLo4ToInt32().Mul(stepV).ShiftAllRight(tsU)
		diffAcc = mulSumInt32(mulSumInt32(diffAcc, dq0, c0.Add(c0).Sub(dq0)), dq1, c1.Add(c1).Sub(dq1))
		lo := q.Abs().ToBits().ExtendLo4ToUint32()
		hiMag := hiInt16(q.Abs()).ToBits().ExtendLo4ToUint32()
		nzLo := lo.Min(one).BitsToInt32()
		nzHi := hiMag.Min(one).BitsToInt32()
		rateAcc = rateAcc.Add(bitLenFloat32(lo)).Add(bitLenFloat32(hiMag)).
			Add(nzLo.Add(nzLo)).Add(nzHi.Add(nzHi))
	}
	var lanes int64
	for ; i < count; i++ {
		v := qcoeff[i]
		if v == 0 {
			continue
		}
		allZero = false
		c := int64(tran[i])
		dq := (int64(v) * int64(step)) >> ts
		diff += dq * (2*c - dq)
		level := v
		if level < 0 {
			level = -level
		}
		lanes += int64(2 + bits.Len16(uint16(level)))
	}
	for l := range uint8(4) {
		lanes += int64(rateAcc.GetElem(l))
	}
	diff += diffAcc.GetElem(0) + diffAcc.GetElem(1)
	return diff, lanes << 9, allZero
}
