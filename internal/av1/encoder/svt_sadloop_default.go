//go:build !(goexperiment.simd && !purego && (arm64 || amd64))

package encoder

func svtSAD16xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	a := svtSAD16xhx8Scalar(src, srcStride, ref, refStride, rows)
	for k := range acc {
		acc[k] += a[k]
	}
}

// svtSAD32xhx8 is the 32-wide eight-position step: the two 16-column halves.
func svtSAD32xhx8(acc *[8]uint32, src []byte, srcStride int, ref []byte, refStride, rows int) {
	svtSAD16xhx8(acc, src, srcStride, ref, refStride, rows)
	svtSAD16xhx8(acc, src[16:], srcStride, ref[16:], refStride, rows)
}
