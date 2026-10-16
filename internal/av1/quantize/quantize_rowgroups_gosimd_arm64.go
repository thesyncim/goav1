//go:build goexperiment.simd && arm64 && !purego

package quantize

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

func init() {
	if cpu.Detected.NEON {
		quantizeBRowGroupsImpl = quantizeBRowGroupsSIMD
	}
}

// quantizeBRowGroupsSIMD is quantizeBRowGroupsScalar with the
// quantizeBVectors lane rule. Each column is walked as pairs of four-row
// groups: a fully live pair quantizes eight coefficients into one 16-byte
// store, a half-live pair four (the dead half stored as zero), a dead pair
// stores zeros. The DC coefficient is redone with the scalar rule and DC
// constants, as quantizeBBlockSIMD does.
func quantizeBRowGroupsSIMD(qcoeff []int16, coeff []int32, n int, live uint8, q Quantizer, txScale uint8) bool {
	if q.AC > quantizeVectorMaxStep || q.DC > quantizeVectorMaxStep || q.AC <= 0 || q.DC <= 0 || n&7 != 0 {
		return quantizeBRowGroupsScalar(qcoeff, coeff, n, live, q, txScale)
	}
	l := 31 - int64(bits.LeadingZeros32(uint32(q.AC)))
	if l < int64(txScale) {
		return quantizeBRowGroupsScalar(qcoeff, coeff, n, live, q, txScale)
	}
	_ = qcoeff[n*n-1]
	_ = coeff[n*n-1]
	zbinFactor := int32(80)
	if q.DC < 148 {
		zbinFactor = 84
	}
	acQuant, _ := invertQuant(q.AC)
	quantV := archsimd.BroadcastInt32x4(acQuant)
	roundV := archsimd.BroadcastInt32x4(roundPowerOfTwo((48*q.AC)>>7, txScale))
	zbinU := archsimd.BroadcastInt32x4(roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale)).ToBits()
	maxV := archsimd.BroadcastInt32x4(maxInt16)
	maxU := maxV.ToBits()
	sixteenV := archsimd.BroadcastInt32x4(-16)
	shiftV := archsimd.BroadcastInt32x4(-int32(l - int64(txScale)))
	signShiftV := archsimd.BroadcastInt32x4(-31)
	zero := archsimd.BroadcastInt32x4(0)
	cp := unsafe.Pointer(&coeff[0])
	qp := unsafe.Pointer(&qcoeff[0])
	orAcc := zero
	for col := 0; col < n; col++ {
		for g := 0; g < n/4; g += 2 {
			off := col*n + 4*g
			pair := (live >> g) & 3
			dst := unsafe.Add(qp, 2*off)
			if pair == 0 {
				zero.TruncToInt16().StoreArray((*[8]int16)(dst))
				continue
			}
			lo, hi := zero, zero
			if pair&1 != 0 {
				lo = quantizeBLane4SIMD(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(cp, 4*off))),
					quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
			}
			if pair&2 != 0 {
				hi = quantizeBLane4SIMD(archsimd.LoadInt32x4Array((*[4]int32)(unsafe.Add(cp, 4*off+16))),
					quantV, roundV, zbinU, maxU, maxV, sixteenV, shiftV, signShiftV)
			}
			orAcc = orAcc.Or(lo).Or(hi)
			quantizePackInt16x8SIMD(lo, hi).StoreArray((*[8]int16)(dst))
		}
	}
	nonZero := orAcc.ReduceMax() != 0 || orAcc.ReduceMin() != 0
	if live&1 != 0 {
		// The vector pass quantized DC with the AC rule; redo it with the DC
		// constants and rebuild the non-zero report around it.
		dcQuant, dcShift := invertQuant(q.DC)
		dc := quantBParams{
			zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.DC, 7), txScale),
			round: roundPowerOfTwo((48*q.DC)>>7, txScale),
			quant: dcQuant,
			shift: dcShift,
		}
		acDC := qcoeff[0]
		qcoeff[0] = quantizeScalarB(coeff[0], &dc, txScale)
		if qcoeff[0] != 0 {
			return true
		}
		if acDC != 0 {
			// The AC-rule DC may have been the only non-zero lane.
			for _, v := range qcoeff[:n*n] {
				if v != 0 {
					return true
				}
			}
			return false
		}
	}
	return nonZero
}
