// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

// This file replaces sad_dispatch_default.go's wrappers on arm64 under
// GOEXPERIMENT=simd. The lowercase sadNxN wrappers are the surface the
// motion-search hot path (pframe_residual.go) calls, so they call the
// Go-native-SIMD kernels directly rather than through the dispatch variables.

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// sadSIMDBound reports whether init binds the dispatch variables to the
// Go-native-SIMD kernels. Without NEON the portable scalar defaults from sad.go
// remain in place.
var sadSIMDBound = cpu.Detected.NEON

// init binds the dispatch variables to the NEON Go-native-SIMD kernels.
func init() {
	if sadSIMDBound {
		bindSIMDSAD()
	}
}

// --- lowercase wrappers (the hot-path dispatch surface) ---------------------

func sad8x8(src, ref []byte, stride int, limit int) int {
	return sad8x8SIMD(src, stride, ref, stride)
}

func sad16x16(src, ref []byte, stride int) int {
	return sad16x16SIMD(src, stride, ref, stride)
}

func sad32x32(src, ref []byte, stride int) int {
	return sad32x32SIMD(src, stride, ref, stride)
}

// sad64x64 is the direct 64-wide NEON-style leaf, not a four-call composition.
func sad64x64(src, ref []byte, stride int) int {
	return sad64x64SIMD(src, stride, ref, stride)
}

func sad8x8x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad8x8x4Step4SIMD(src, ref, stride)
}

func sad8x8x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x8x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad16x16x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad16x16x4Step4SIMD(src, ref, stride)
}

func sad32x32x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad32x32x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad32x32x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad32x32x4Step4SIMD(src, ref, stride)
}

func sad8x8Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad8x8SIMD(src, srcStride, ref, refStride)
}

func sad16x16Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad16x16SIMD(src, srcStride, ref, refStride)
}

func sad32x32Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad32x32SIMD(src, srcStride, ref, refStride)
}

func sad64x64Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad64x64SIMD(src, srcStride, ref, refStride)
}

func sad8x8CompoundAvgBlock(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	return sad8x8CompoundAvgSIMD(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
}
