// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

// Go-native-SIMD SAD kernels for motion estimation, shared by arm64 (NEON) and
// amd64 (AVX2). Each kernel composes the per-architecture primitives in
// sad_simd_{arm64,amd64}.go and returns exact integer sums matching the
// sadNxNPureGo references in sad.go.
//
// The single-reference entry points prove the exact bytes they read with one
// bounds check on the last element they touch, so the raw-pointer primitives
// can never read past a caller's slice. The four-reference kernels take
// bounds-checked row windows.

package encoder

import "unsafe"

// sad8x8SIMD is the 8x8 SAD with independent source and reference strides.
func sad8x8SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	_ = src[7*srcStride+7]
	_ = ref[7*refStride+7]
	return sad8x8Ptr(unsafe.Pointer(&src[0]), srcStride, unsafe.Pointer(&ref[0]), refStride)
}

// sad8x8x4SIMD computes four 8x8 SADs of one source block against four
// reference origins sharing stride.
func sad8x8x4SIMD(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	_ = src[7*stride+7]
	_ = ref0[7*stride+7]
	_ = ref1[7*stride+7]
	_ = ref2[7*stride+7]
	_ = ref3[7*stride+7]
	return sad8x8x4Ptr(unsafe.Pointer(&src[0]), unsafe.Pointer(&ref0[0]), unsafe.Pointer(&ref1[0]),
		unsafe.Pointer(&ref2[0]), unsafe.Pointer(&ref3[0]), stride)
}

// sad8x8x4Step4SIMD computes four 8x8 SADs against reference origins ref+0,
// ref+4, ref+8, and ref+12 (the raster step-4 candidate group).
func sad8x8x4Step4SIMD(src, ref []byte, stride int) (int, int, int, int) {
	_ = src[7*stride+7]
	_ = ref[7*stride+19]
	// Strided rows inside the extents checked above; same shape as the SAD rows (checked rows +9-86%).
	rp := unsafe.Pointer(&ref[0])
	return sad8x8x4Ptr(unsafe.Pointer(&src[0]), rp, step(rp, 4), step(rp, 8), step(rp, 12), stride)
}

// sad8x8CompoundAvgSIMD computes SAD(src, round((ref0+ref1)/2)) over one 8x8
// block with independent source and reference strides.
func sad8x8CompoundAvgSIMD(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	_ = src[7*srcStride+7]
	_ = ref0[7*ref0Stride+7]
	_ = ref1[7*ref1Stride+7]
	return sad8x8CompoundPtr(unsafe.Pointer(&src[0]), srcStride, unsafe.Pointer(&ref0[0]), ref0Stride,
		unsafe.Pointer(&ref1[0]), ref1Stride)
}

// sad16x16SIMD is the 16x16 SAD with independent source and reference strides.
func sad16x16SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	_ = src[15*srcStride+15]
	_ = ref[15*refStride+15]
	return sad16ColsDualPtr(unsafe.Pointer(&src[0]), srcStride, unsafe.Pointer(&ref[0]), refStride, 16)
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
// It is two 16-column strips.
func sad32x32SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	_ = src[31*srcStride+31]
	_ = ref[31*refStride+31]
	// Strided rows inside the extents checked above; same shape as the SAD rows (checked rows +9-86%).
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	return sad16ColsDualPtr(sp, srcStride, rp, refStride, 32) +
		sad16ColsDualPtr(step(sp, 16), srcStride, step(rp, 16), refStride, 32)
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
// It is four 16-column strips.
func sad64x64SIMD(src []byte, srcStride int, ref []byte, refStride int) int {
	_ = src[63*srcStride+63]
	_ = ref[63*refStride+63]
	// Strided rows inside the extents checked above; same shape as the SAD rows (checked rows +9-86%).
	sp := unsafe.Pointer(&src[0])
	rp := unsafe.Pointer(&ref[0])
	return sad16ColsDualPtr(sp, srcStride, rp, refStride, 64) +
		sad16ColsDualPtr(step(sp, 16), srcStride, step(rp, 16), refStride, 64) +
		sad16ColsDualPtr(step(sp, 32), srcStride, step(rp, 32), refStride, 64) +
		sad16ColsDualPtr(step(sp, 48), srcStride, step(rp, 48), refStride, 64)
}

// Single-stride adapters match the sadNxNImpl signatures in sad.go.

func sad8x8SingleSIMD(src, ref []byte, stride int, limit int) int {
	return sad8x8SIMD(src, stride, ref, stride)
}

func sad16x16SingleSIMD(src, ref []byte, stride int) int {
	return sad16x16SIMD(src, stride, ref, stride)
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
