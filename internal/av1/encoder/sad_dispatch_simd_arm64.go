// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

// This file replaces sad_dispatch_arm64.go under GOEXPERIMENT=simd. It routes
// the promoted 8x8 four-reference kernel through Go-native SIMD and keeps
// measured-losing and unported shapes on their NEON implementations.
//
// The lowercase sadNxN wrappers are the surface the motion-search hot path
// (pframe_residual.go) actually calls, so they must be defined here (the asm
// dispatch file that defined them is excluded by !goexperiment.simd). Step4,
// dual, compound, and other unported shapes stay on their existing paths.

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init first binds retained NEON kernels, then overrides the promoted 8x8x4
// shape with Go-native SIMD. Other SAD shapes retain their NEON or DOTPROD
// dispatch.
func init() {
	if !cpu.Detected.NEON {
		// No NEON: leave the portable scalar defaults from sad.go in place.
		return
	}
	bindNEONSAD()
	sad8x8x4Impl = sad8x8x4SIMD
}

// --- lowercase wrappers (the hot-path dispatch surface) ---------------------

// sad8x8 stays on NEON.
func sad8x8(src, ref []byte, stride int, limit int) int {
	return sad8x8NEON(src, ref, stride, limit)
}

func sad16x16(src, ref []byte, stride int) int {
	return sad16x16Impl(src, ref, stride)
}

func sad32x32(src, ref []byte, stride int) int {
	return sad32x32Impl(src, ref, stride)
}

// sad64x64 composes the selected 32x32 kernel (no direct 64-wide leaf yet).
func sad64x64(src, ref []byte, stride int) int {
	return sad64x64Composed(src, ref, stride)
}

func sad8x8x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad8x8x4Step4NEON(src, ref, stride)
}

// sad8x8x4 routes to SIMD so the shared source row can be reused across refs.
func sad8x8x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x8x4Impl(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad16x16x4Impl(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	if useDotProdSAD {
		return sad16x16x4Step4DotProd(src, ref, stride)
	}
	return sad16x16x4Step4NEON(src, ref, stride)
}

func sad32x32x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad32x32x4Impl(src, ref0, ref1, ref2, ref3, stride)
}

func sad32x32x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	if useDotProdSAD {
		return sad32x32x4Step4DotProd(src, ref, stride)
	}
	return sad32x32x4Step4NEON(src, ref, stride)
}

func sad8x8Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad8x8DualNEON(src, srcStride, ref, refStride)
}

func sad16x16Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad16x16DualNEON(src, srcStride, ref, refStride)
}

func sad32x32Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad32x32DualNEON(src, srcStride, ref, refStride)
}

func sad64x64Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad64x64DualNEON(src, srcStride, ref, refStride)
}

func sad8x8CompoundAvgBlock(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	return sad8x8CompoundAvgBlockNEON(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
}
