//go:build arm64 && !purego && !goexperiment.simd

package encoder

// init binds the encoder metric kernels to their NEON implementations. Under
// goexperiment.simd this binding is replaced by metric_simd_arm64.go's init,
// which swaps the SATD/Hadamard kernels for Go-native archsimd ports while
// keeping the pixelStats kernels on NEON assembly (via bindPixelStatsNEON).
func init() {
	bindPixelStatsNEON()
	satdCoeffsImpl = satdCoeffsNEON
	hadamard4x4Impl = hadamard4x4NEON
	hadamard8x8Impl = hadamard8x8NEON
	hadamard16x16Impl = hadamard16x16NEON
	hadamard32x32Impl = hadamard32x32NEON
}
