//go:build goexperiment.simd && amd64 && !purego

package quantize

import "simd/archsimd"

// init binds Go-native SIMD quantizers under GOEXPERIMENT=simd when AVX2 is
// reported by archsimd.X86. Quantize-b is still served by the AVX2 asm kernel
// (quantize_avx2_amd64.go) until it is ported.
func init() {
	if archsimd.X86.AVX2() {
		quantizeFPBlockImpl = quantizeFPBlockSIMD
	}
}
