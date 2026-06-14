//go:build purego || !arm64

package encoder

func pixelStats8x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x8Impl(src, srcStride, ref, refStride)
}

func pixelStats16x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x8Impl(src, srcStride, ref, refStride)
}

func pixelStats8x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x16Impl(src, srcStride, ref, refStride)
}

func pixelStats16x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x16Impl(src, srcStride, ref, refStride)
}

func pixelStats32x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x8Impl(src, srcStride, ref, refStride)
}

func pixelStats8x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x32Impl(src, srcStride, ref, refStride)
}

func pixelStats32x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x16Impl(src, srcStride, ref, refStride)
}

func pixelStats16x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x32Impl(src, srcStride, ref, refStride)
}

func pixelStats32x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x32Impl(src, srcStride, ref, refStride)
}

func pixelStats64x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats64x16Impl(src, srcStride, ref, refStride)
}

func pixelStats16x64(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x64Impl(src, srcStride, ref, refStride)
}

func pixelStats64x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats64x32Impl(src, srcStride, ref, refStride)
}

func pixelStats32x64(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x64Impl(src, srcStride, ref, refStride)
}

func satdCoeffs(coeff []int32, count int) int {
	return satdCoeffsImpl(coeff, count)
}

func hadamard4x4(src []int16, srcStride int, coeff []int32) {
	hadamard4x4Impl(src, srcStride, coeff)
}

func hadamard8x8(src []int16, srcStride int, coeff []int32) {
	hadamard8x8Impl(src, srcStride, coeff)
}

func hadamard16x16(src []int16, srcStride int, coeff []int32) {
	hadamard16x16Impl(src, srcStride, coeff)
}

func hadamard32x32(src []int16, srcStride int, coeff []int32) {
	hadamard32x32Impl(src, srcStride, coeff)
}
