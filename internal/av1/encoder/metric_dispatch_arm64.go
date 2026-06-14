//go:build arm64 && !purego

package encoder

func pixelStats8x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x8NEON(src, srcStride, ref, refStride)
}

func pixelStats16x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x16NEON(src, srcStride, ref, refStride)
}

func pixelStats32x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x32NEON(src, srcStride, ref, refStride)
}

func satdCoeffs(coeff []int32, count int) int {
	return satdCoeffsNEON(coeff, count)
}

func hadamard8x8(src []int16, srcStride int, coeff []int32) {
	hadamard8x8NEON(src, srcStride, coeff)
}

func hadamard16x16(src []int16, srcStride int, coeff []int32) {
	hadamard16x16NEON(src, srcStride, coeff)
}

func hadamard32x32(src []int16, srcStride int, coeff []int32) {
	hadamard32x32NEON(src, srcStride, coeff)
}
