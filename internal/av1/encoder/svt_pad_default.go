//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

func fillBorder(dst []byte, v byte) { fillBorderScalar(dst, v) }
