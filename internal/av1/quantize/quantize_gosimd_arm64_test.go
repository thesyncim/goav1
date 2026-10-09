//go:build goexperiment.simd && arm64 && !purego

package quantize

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

func quantizeSIMDSupported() bool {
	return cpu.Detected.NEON
}
