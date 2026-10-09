//go:build goexperiment.simd && arm64 && !purego

package encoder

// gosimdKernelsSupported reports whether the Go-native SIMD kernels can run on
// this CPU. Advanced SIMD is baseline on arm64.
func gosimdKernelsSupported() bool { return true }
