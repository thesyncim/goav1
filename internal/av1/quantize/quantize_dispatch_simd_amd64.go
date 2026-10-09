//go:build goexperiment.simd && amd64 && !purego

package quantize

import "simd/archsimd"

// init binds Go-native SIMD quantizers under GOEXPERIMENT=simd when AVX2 is
// reported by archsimd.X86.
func init() {
	if archsimd.X86.AVX2() {
		quantizeFPBlockImpl = quantizeFPBlockSIMD
		quantizeBBlockImpl = quantizeBBlockSIMD
	}
}
