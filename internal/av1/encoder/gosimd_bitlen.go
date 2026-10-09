//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import "simd/archsimd"

// bitLenFloat32 computes bits.Len32 for lanes below 2^24 through the float32
// exponent: a value below 2^24 converts exactly, so its biased exponent is
// 127 + floor(log2 v) for v >= 1, and 0 for v = 0 (which the max clamps to 0).
func bitLenFloat32(v archsimd.Uint32x4) archsimd.Int32x4 {
	f := v.BitsToInt32().ConvertToFloat32()
	exp := f.ToBits().ShiftAllRight(23).BitsToInt32()
	return exp.Sub(archsimd.BroadcastInt32x4(126)).Max(archsimd.BroadcastInt32x4(0))
}
