//go:build arm64 && !purego

package encoder

func sad8x8(src, ref []byte, stride int, limit int) int {
	return sad8x8NEON(src, ref, stride, limit)
}

func sad16x16(src, ref []byte, stride int) int {
	return sad16x16NEON(src, ref, stride)
}

func sad32x32(src, ref []byte, stride int) int {
	return sad32x32NEON(src, ref, stride)
}

func sad8x8x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad8x8x4Step4NEON(src, ref, stride)
}

func sad8x8x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad8x8x4NEON(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad16x16x4NEON(src, ref0, ref1, ref2, ref3, stride)
}

func sad16x16x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	return sad16x16x4Step4NEON(src, ref, stride)
}

func sad32x32x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	return sad32x32x4NEON(src, ref0, ref1, ref2, ref3, stride)
}

func sad32x32x4Step4(src, ref []byte, stride int) (int, int, int, int) {
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

func sad8x8CompoundAvgBlock(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	return sad8x8CompoundAvgBlockNEON(src, srcStride, ref0, ref0Stride, ref1, ref1Stride)
}
