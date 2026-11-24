// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

// Go-native-SIMD SAD kernels for motion estimation, shared by arm64 (NEON) and
// amd64 (AVX2). Each kernel composes the per-architecture primitives in
// sad_simd_{arm64,amd64}.go and returns exact integer sums matching the
// sadNxNPureGo references in sad.go.
//
// Every kernel reads its rows through bounds-checked windows of the caller's
// slices.

package encoder

// sad8x8SIMD is the 8x8 SAD with independent source and reference strides.
func sad8x8SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad8x8Rows(src, srcStride, ref, refStride)
}

// sad8x8x4SIMD computes four 8x8 SADs of one source block against four
// reference origins sharing stride.
func sad8x8x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x8x4Rows(src, ref0, ref1, ref2, ref3, stride)
}

// sad8x8x4Step4SIMD computes four 8x8 SADs against reference origins ref+0,
// ref+4, ref+8, and ref+12 (the raster step-4 candidate group).
func sad8x8x4Step4SIMD(src, ref []byte, stride int) (int, int, int, int) {
	return sad8x8x4Rows(src, ref, ref[4:], ref[8:], ref[12:], stride)
}

// sad8x8CompoundAvgSIMD computes SAD(src, round((ref0+ref1)/2)) over one 8x8
// block with independent source and reference strides.
func sad8x8CompoundAvgSIMD(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	return sad8x8CompoundRows(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
}

// sad16x16SIMD is the 16x16 SAD with independent source and reference strides.
func sad16x16SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	if refStride == 16 && len(ref) >= 256 {
		return sad16x16Packed(src, srcStride, (*[256]uint8)(ref[:256]))
	}
	return sad16ColsDual(src, srcStride, ref, refStride, 16)
}

// sad16x16x4SIMD computes four 16x16 SADs against reference origins sharing
// stride.
func sad16x16x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad16ColsX4(src, ref0, ref1, ref2, ref3, stride, 16)
}

// sad16x16x4Step4SIMD computes four 16x16 SADs against ref+0, ref+4, ref+8, and
// ref+12. The last candidate reads 12 bytes past each 16-byte row.
func sad16x16x4Step4SIMD(src, ref []byte, stride int) (int, int, int, int) {
	return sad16ColsX4(src, ref, ref[4:], ref[8:], ref[12:], stride, 16)
}

// sad32x32SIMD is the 32x32 SAD with independent source and reference strides.
// It takes one 32-byte window per row pair.
func sad32x32SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad32Rows(src, srcStride, ref, refStride)
}

// sad32x32x4SIMD computes four 32x32 SADs against reference origins sharing
// stride, as two 16-column strips.
func sad32x32x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	l0, l1, l2, l3 := sad16ColsX4(src, ref0, ref1, ref2, ref3, stride, 32)
	r0, r1, r2, r3 := sad16ColsX4(src[16:], ref0[16:], ref1[16:], ref2[16:], ref3[16:], stride, 32)
	return l0 + r0, l1 + r1, l2 + r2, l3 + r3
}

// sad32x32x4Step4SIMD computes four 32x32 SADs against ref+0, ref+4, ref+8, and
// ref+12, as two 16-column strips. The last candidate reads 12 bytes past each
// 32-byte row.
func sad32x32x4Step4SIMD(src, ref []byte, stride int) (int, int, int, int) {
	l0, l1, l2, l3 := sad16ColsX4(src, ref, ref[4:], ref[8:], ref[12:], stride, 32)
	r0, r1, r2, r3 := sad16ColsX4(src[16:], ref[16:], ref[20:], ref[24:], ref[28:], stride, 32)
	return l0 + r0, l1 + r1, l2 + r2, l3 + r3
}

// sad64x64SIMD is the 64x64 SAD with independent source and reference strides.
// It takes one 64-byte window per row pair.
func sad64x64SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad64Rows(src, srcStride, ref, refStride)
}

// Single-stride adapters match the sadNxNImpl signatures in sad.go.

func sad8x8SingleSIMD(src, ref []byte, stride int, limit int) int {
	return sad8x8Same(src, ref, stride)
}

func sad16x16SingleSIMD(src, ref []byte, stride int) int {
	return sad16x16Same(src, ref, stride)
}

func sad32x32SingleSIMD(src, ref []byte, stride int) int {
	return sad32x32SIMD(src, stride, ref, stride)
}

// bindSIMDSAD points the SAD dispatch variables at the Go-native-SIMD kernels.
// Callers gate it on the CPU features the kernels need.
func bindSIMDSAD() {
	sad8x8Impl = sad8x8SingleSIMD
	sad16x16Impl = sad16x16SingleSIMD
	sad32x32Impl = sad32x32SingleSIMD
	sad8x8x4Step4Impl = sad8x8x4Step4SIMD
	sad8x8x4Impl = sad8x8x4SIMD
	sad16x16x4Impl = sad16x16x4SIMD
	sad16x16x4Step4Impl = sad16x16x4Step4SIMD
	sad32x32x4Impl = sad32x32x4SIMD
	sad32x32x4Step4Impl = sad32x32x4Step4SIMD
	sad8x8DualImpl = sad8x8SIMD
	sad16x16DualImpl = sad16x16SIMD
	sad32x32DualImpl = sad32x32SIMD
	sad8x8CompoundAvgBlockImpl = sad8x8CompoundAvgSIMD
}
