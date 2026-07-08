// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package encoder

// This file replaces sad_dispatch_arm64.go under GOEXPERIMENT=simd. It routes
// the ported single-block and 4-reference SAD shapes through the Go-native
// SIMD kernels (sad_simd_arm64.go) and keeps every other shape on the NEON asm
// kernels (which remain compiled — only sad_dispatch_arm64.go's init is
// excluded). See SIMD_PORT.md.
//
// The lowercase sadNxN wrappers are the surface the motion-search hot path
// (pframe_residual.go) actually calls, so they must be defined here (the asm
// dispatch file that defined them is excluded by !goexperiment.simd). The
// x4/step4/dual/compound shapes stay on the asm bodies for now: they are
// byte-exact and already fast, and this port targets the core single-block +
// x4 shapes named in the task.

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init first binds every dispatch var to NEON (so unported shapes keep asm
// speed), then overrides the ported shapes with the Go-native SIMD kernels.
func init() {
	if !cpu.Detected.NEON {
		// No NEON: leave the portable scalar defaults from sad.go in place.
		return
	}
	bindNEONSAD()
	// Wire the shapes where Go-native SIMD BEATS the NEON asm. 16x16 and 32x32
	// single-block, and the 8x8x4 / 16x16x4 / 32x32x4 four-reference kernels
	// (the motion-search hot path) all beat the asm — the x4 kernels because the
	// loaded/packed src row is reused across all four refs, amortizing its cost.
	// Only 8x8 SINGLE-block stays on asm (left bound to NEON by bindNEONSAD): with
	// no 4x reuse, the 8-byte-row register pack (VDUP+VMOV) costs more than the
	// asm's single VLD1 .8b.
	sad16x16Impl = sad16x16SIMD
	sad32x32Impl = sad32x32SIMD
	sad8x8x4Impl = sad8x8x4SIMD
	sad16x16x4Impl = sad16x16x4SIMD
	sad32x32x4Impl = sad32x32x4SIMD
}

// --- lowercase wrappers (the hot-path dispatch surface) ---------------------

// sad8x8 stays on NEON: the 8-wide Go-SIMD kernel loses to the asm (row-pack
// cost at 8-byte rows). sad8x8SIMD remains available + differential-tested.
func sad8x8(src, ref []byte, stride int, limit int) int {
	return sad8x8NEON(src, ref, stride, limit)
}

func sad16x16(src, ref []byte, stride int) int {
	return sad16x16SIMD(src, ref, stride)
}

func sad32x32(src, ref []byte, stride int) int {
	return sad32x32SIMD(src, ref, stride)
}

// sad64x64 composes the SIMD 32x32 kernel (no direct 64-wide SIMD leaf yet).
func sad64x64(src, ref []byte, stride int) int {
	return sad64x64Composed(src, ref, stride)
}

func sad8x8x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad8x8x4Step4NEON(src, ref, stride)
}

// sad8x8x4 routes to SIMD: reusing the packed src row across the four refs
// amortizes the 8-wide pack cost, so it beats the asm (8.6 vs 10.2 ns).
func sad8x8x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x8x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad16x16x4SIMD(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	if useDotProdSAD {
		return sad16x16x4Step4DotProd(src, ref, stride)
	}
	return sad16x16x4Step4NEON(src, ref, stride)
}

func sad32x32x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad32x32x4SIMD(src, ref0, ref1, ref2, ref3, stride)
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
