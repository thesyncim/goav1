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

func sad8x8Dual(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad8x8DualNEON(src, srcStride, ref, refStride)
}
