//go:build goexperiment.simd && amd64 && !purego

package transform

import "simd/archsimd"

func itxRoundBias(amount uint8) archsimd.Int32x4 {
	return archsimd.BroadcastInt32x4(1 << (amount - 1))
}

func itxShiftAmount(amount int32) archsimd.Int32x4 {
	return archsimd.BroadcastInt32x4(-amount)
}

func itxMulAcc(accumulator, value, coefficient archsimd.Int32x4) archsimd.Int32x4 {
	return accumulator.Add(value.Mul(coefficient))
}

func itxShr12(value, amount archsimd.Int32x4) archsimd.Int32x4 {
	return value.ShiftAllRight(12)
}
