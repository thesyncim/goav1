//go:build goexperiment.simd && amd64 && !purego

package encoder

func pixelStatsRows16Arch(src []byte, srcStride int, ref []byte, refStride int, w, h int) (uint32, int32) {
	return pixelStatsRows16Generic(src, srcStride, ref, refStride, w, h)
}
