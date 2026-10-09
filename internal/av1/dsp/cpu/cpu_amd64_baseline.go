//go:build amd64 && (purego || !goexperiment.simd)

package cpu

// init populates Detected on amd64 builds without the SIMD experiment and on
// purego builds. Only the GOAMD64=v1 SSE2 baseline is reported and no feature
// probing runs, so AVX2 and AVX512 stay false and the dispatch layer keeps its
// pure-Go kernels.
func init() {
	Detected.SSE2 = true
}
