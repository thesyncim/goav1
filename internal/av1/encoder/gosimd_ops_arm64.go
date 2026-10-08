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
	return v.InterleaveHi(archsimd.BroadcastUint8x16(0)).ReshapeToUint16s()
}

// hiInt16 moves 16-bit lanes 4..7 of v into lanes 0..3.
func hiInt16(v archsimd.Int16x8) archsimd.Int16x8 {
	return v.HiToLo()
}

// hiInt32 moves 32-bit lanes 2..3 of v into lanes 0..1.
func hiInt32(v archsimd.Int32x4) archsimd.Int32x4 {
	return v.HiToLo()
}

// sqSumInt32 adds the exact squares of v's four int32 lanes to acc.
func sqSumInt32(acc archsimd.Int64x2, v archsimd.Int32x4) archsimd.Int64x2 {
	hi := v.HiToLo()
	return acc.Add(v.MulWidenLo(v)).Add(hi.MulWidenLo(hi))
}

// interleaveEven16 and interleaveOdd16 are the VTRN1/VTRN2 transposes of two
// 16-bit vectors: [x0 y0 x2 y2 ...] and [x1 y1 x3 y3 ...].
func interleaveEven16(x, y archsimd.Int16x8) archsimd.Int16x8 { return x.InterleaveEven(y) }
func interleaveOdd16(x, y archsimd.Int16x8) archsimd.Int16x8  { return x.InterleaveOdd(y) }

// interleaveEven32 and interleaveOdd32 are the 32-bit VTRN1/VTRN2 transposes.
func interleaveEven32(x, y archsimd.Int32x4) archsimd.Int32x4 { return x.InterleaveEven(y) }
func interleaveOdd32(x, y archsimd.Int32x4) archsimd.Int32x4  { return x.InterleaveOdd(y) }

// shiftRight16 and shiftRight32 are arithmetic right shifts by the uniform
// negative counts in rsh (SSHL by -n).
func shiftRight16(v, rsh archsimd.Int16x8) archsimd.Int16x8 { return v.Shift(rsh) }
func shiftRight32(v, rsh archsimd.Int32x4) archsimd.Int32x4 { return v.Shift(rsh) }
