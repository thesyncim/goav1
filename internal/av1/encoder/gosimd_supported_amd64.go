//go:build goexperiment.simd && amd64 && !purego

package encoder

import "simd/archsimd"

// gosimdKernelsSupported reports whether the Go-native SIMD kernels can run on
// this CPU. The amd64 kernels use AVX2-width archsimd operations.
func gosimdKernelsSupported() bool { return archsimd.X86.AVX2() }
