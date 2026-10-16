package encoder

import "math/bits"

// residualBlockImpl extracts the prediction residual (scalar reference unless
// the Go SIMD kernels bind it in rdstats_gosimd.go).
var residualBlockImpl = residualBlockPureGo

// rdStatsBlockImpl accumulates the skip-decision statistics (scalar reference
// unless the Go SIMD kernels bind it in rdstats_gosimd.go).
var rdStatsBlockImpl = rdStatsBlockPureGo

// blockErrorImpl computes the coefficient block error (scalar reference unless
// the Go SIMD kernels bind it in rdstats_gosimd.go).
var blockErrorImpl = blockErrorPureGo

// rdStatsDiffImpl accumulates the skip comparison's terms of a block whose
// zero levels are left out (scalar reference unless the Go SIMD kernels bind
// it in rdstats_gosimd.go).
var rdStatsDiffImpl = rdStatsDiffPureGo

// rdStatsDiffPureGo is rdStatsBlockPureGo reduced to what the skip comparison
// rdDskip<<7 <= rate + rdDcode<<7 reads: a zero level adds the same c*c to
// both distortions, so only non-zero levels contribute, each by
// c*c - (c-dq)*(c-dq) = dq*(2c - dq) to the skip-minus-code distortion
// difference, plus its coarse rate. allZero reports that every level is zero.
func rdStatsDiffPureGo(tran []int32, qcoeff []int16, count int, step int32, ts uint8) (diff, rate int64, allZero bool) {
	allZero = true
	for i := 0; i < count; i++ {
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
		rate += int64(2+bits.Len16(uint16(level))) << 9
	}
	return diff, rate, allZero
}
