//go:build arm64 && !purego && !goexperiment.simd

package encoder

// init binds pixel-domain statistics to NEON. SATD and Hadamard keep their
// scalar defaults unless the Go SIMD experiment is enabled.
func init() {
	bindPixelStatsNEON()
}
