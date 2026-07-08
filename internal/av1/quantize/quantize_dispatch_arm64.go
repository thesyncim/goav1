//go:build arm64 && !purego && !goexperiment.simd

package quantize

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the hand-written NEON quantizers on normal arm64 builds. The
// Go-native SIMD build has its own dispatch file so it can replace selected
// kernels without dropping the NEON ones that have not been ported.
func init() {
	if cpu.Detected.NEON {
		quantizeFPBlockImpl = quantizeFPBlockNEON
		quantizeBBlockImpl = quantizeBBlockNEON
	}
}
