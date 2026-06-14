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
