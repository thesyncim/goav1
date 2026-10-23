//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

func boxInterpolateRow8(dst, s0, s1 []uint8, f int) { boxInterpolateRow(dst, s0, s1, f, 128) }
