// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "simd/archsimd"

// arm64 lowering of the four-lane inverse-transform primitives used by the
// generated kernels (dct64_lanes4_gosimd.go). Each is small enough to inline.
//
// NEON has no rounding shift in archsimd, so a rounding shift is a bias added
// into the multiply-accumulate (itxRoundBias) followed by an arithmetic shift.
// The shift goes through SSHL by a register-resident negative amount: a
// constant ShiftAllRight is lowered as an insert+dup+sshl on every use.

func itxRoundBias(n uint8) archsimd.Int32x4 { return archsimd.BroadcastInt32x4(1 << (n - 1)) }

func itxShiftAmount(n int32) archsimd.Int32x4 { return archsimd.BroadcastInt32x4(-n) }

// itxMulAcc returns acc + x*k (MLA).
func itxMulAcc(acc, x, k archsimd.Int32x4) archsimd.Int32x4 { return x.MulAdd(k, acc) }

func itxShr12(v, n12 archsimd.Int32x4) archsimd.Int32x4 { return v.Shift(n12) }
