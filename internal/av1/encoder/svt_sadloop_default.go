//go:build !(goexperiment.simd && !purego && (arm64 || amd64))

package encoder

func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	a := svtSAD16xhx8Scalar(src, srcStride, ref, refStride, rows)
	for k := range acc {
		acc[k] += a[k]
	}
}
