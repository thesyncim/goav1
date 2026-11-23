//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

// gosimd_ops_arm64.go and gosimd_ops_amd64.go give the shared Go SIMD kernels
// one vocabulary for lane moves and widening squares. The arm64 forms use the
// NEON zip, extend and widening-multiply ops; the amd64 forms use PSHUFB
// (PermuteOrZero) and PMULDQ (MulWidenEven), which exist in AVX2 without
// AVX-512.

// widenHi8 zero-extends bytes 8..15 of v to 16-bit lanes.
func widenHi8(v archsimd.Uint8x16) archsimd.Uint16x8 {
	return v.HiToLo().ExtendLo8ToUint16()
}

// hiInt16 moves 16-bit lanes 4..7 of v into lanes 0..3.
func hiInt16(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.HiToLo()
}

// sqSumInt32 adds the exact squares of v's four int32 lanes to acc.
func sqSumInt32(acc archsimd.Int64x2, v archsimd.Int32x4) archsimd.Int64x2 {
	hi := v.HiToLo()
	return acc.Add(v.MulWidenLo(v)).Add(hi.MulWidenLo(hi))
}

// mulSumInt32 adds the exact products a[i]*b[i] of the four int32 lane pairs
// to acc.
func mulSumInt32(acc archsimd.Int64x2, a, b archsimd.Int32x4) archsimd.Int64x2 {
	return acc.Add(a.MulWidenLo(b)).Add(a.HiToLo().MulWidenLo(b.HiToLo()))
}
