//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

func memsetBytes(dst []byte, v byte) { memsetBytesScalar(dst, v) }
