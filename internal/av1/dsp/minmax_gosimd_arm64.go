// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package dsp

import "simd/archsimd"

// Horizontal reductions for minmax_gosimd.go. NEON reduces a whole vector to
// its minimum or maximum with one across-vector instruction (UMINV/UMAXV).

func minmaxReduceMinU8(v archsimd.Uint8x16) uint8 { return v.ReduceMin() }

func minmaxReduceMaxU8(v archsimd.Uint8x16) uint8 { return v.ReduceMax() }

func minmaxReduceMinU16(v archsimd.Uint16x8) uint16 { return v.ReduceMin() }

func minmaxReduceMaxU16(v archsimd.Uint16x8) uint16 { return v.ReduceMax() }
