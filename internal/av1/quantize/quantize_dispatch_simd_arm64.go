//go:build goexperiment.simd && arm64 && !purego

package quantize

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds Go-native SIMD quantizers under GOEXPERIMENT=simd. Quantize-b is
// still served by the existing NEON asm kernel so enabling the experiment does
// not regress that path.
func init() {
	if cpu.Detected.NEON {
		quantizeBlockImpl = quantizeBlockSIMD
		quantizeFPBlockImpl = quantizeFPBlockNEON
		quantizeBBlockImpl = quantizeBBlockNEON
		quantizeFPNoQMatrixImpl = quantizeFPNoQMatrixSIMD
	}
}
