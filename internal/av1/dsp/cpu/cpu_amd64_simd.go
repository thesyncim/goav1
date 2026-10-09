//go:build amd64 && goexperiment.simd && !purego

package cpu

import "simd/archsimd"

// init populates Detected on amd64 SIMD builds from the official archsimd
// CPU feature queries. Those queries already gate AVX state on OSXSAVE and
// XCR0, and AVX-512 on the opmask/ZMM state, so no CPUID or XGETBV probe is
// needed. The GOAMD64=v1 baseline guarantees SSE2 unconditionally.
//
// archsimd exposes no SSE4.1 or SSE4.2 query, so those flags stay false
// rather than being inferred from AVX2.
func init() {
	Detected.SSE2 = true
	Detected.AVX2 = archsimd.X86.AVX2()
	Detected.AVX512 = archsimd.X86.AVX512()
}
