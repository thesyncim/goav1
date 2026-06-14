//go:build arm64 && !purego

package encoder

func pixelStats8x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x8NEON(src, srcStride, ref, refStride)
}

func pixelStats4x4(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats4x4NEON(src, srcStride, ref, refStride)
}

func pixelStats8x4(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x4NEON(src, srcStride, ref, refStride)
}

func pixelStats4x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats4x8NEON(src, srcStride, ref, refStride)
}

func pixelStats16x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x8NEON(src, srcStride, ref, refStride)
}

func pixelStats8x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x16NEON(src, srcStride, ref, refStride)
}

func pixelStats16x4(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x4NEON(src, srcStride, ref, refStride)
}

func pixelStats4x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats4x16NEON(src, srcStride, ref, refStride)
}

func pixelStats16x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x16NEON(src, srcStride, ref, refStride)
}

func pixelStats32x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x8NEON(src, srcStride, ref, refStride)
}

func pixelStats8x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x32NEON(src, srcStride, ref, refStride)
}

func pixelStats32x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x16NEON(src, srcStride, ref, refStride)
}

func pixelStats16x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x32NEON(src, srcStride, ref, refStride)
}

func pixelStats32x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x32NEON(src, srcStride, ref, refStride)
}

func pixelStats64x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats64x16NEON(src, srcStride, ref, refStride)
}

func pixelStats16x64(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x64NEON(src, srcStride, ref, refStride)
}

func pixelStats64x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats64x32NEON(src, srcStride, ref, refStride)
}

func pixelStats32x64(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x64NEON(src, srcStride, ref, refStride)
}

func satdCoeffs(coeff []int32, count int) int {
	return satdCoeffsNEON(coeff, count)
}

func hadamard4x4(src []int16, srcStride int, coeff []int32) {
	hadamard4x4NEON(src, srcStride, coeff)
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
