//go:build goexperiment.simd && arm64 && !purego

package quantize

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds Go-native SIMD quantizers under GOEXPERIMENT=simd.
func init() {
	if cpu.Detected.NEON {
		quantizeBlockImpl = quantizeBlockSIMD
		quantizeFPBlockImpl = quantizeFPBlockSIMD
		quantizeBBlockImpl = quantizeBBlockSIMD
		quantizeFPNoQMatrixImpl = quantizeFPNoQMatrixSIMD
	}
}
