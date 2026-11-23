package encoder

func pixelStats8x8(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats8x8Impl(src, srcStride, ref, refStride)
}

func pixelStats4x4(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats4x4Impl(src, srcStride, ref, refStride)
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

func pixelStats32x16(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x16Impl(src, srcStride, ref, refStride)
}

func pixelStats16x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats16x32Impl(src, srcStride, ref, refStride)
}

func pixelStats32x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x32Impl(src, srcStride, ref, refStride)
}

func pixelStats64x32(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats64x32Impl(src, srcStride, ref, refStride)
}

func pixelStats32x64(src []byte, srcStride int, ref []byte, refStride int) (uint32, int32) {
	return pixelStats32x64Impl(src, srcStride, ref, refStride)
}
