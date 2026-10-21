//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

func bilinearSubpelSADImpl(src []byte, srcStride int, ref []byte, refStride, n, fx, fy int) int {
	return bilinearSubpelSADScalar(src, srcStride, ref, refStride, n, fx, fy)
}
