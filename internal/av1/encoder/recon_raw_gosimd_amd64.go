//go:build goexperiment.simd && amd64 && !purego

package encoder

func init() {
	// Keep the existing amd64 vector kernels until the separate-input fusion
	// has native generated-code and timing coverage on that architecture.
	if gosimdKernelsSupported() {
		addRawResidualToPredReconBlockImpl = addRawResidualToPredReconBlockCopied
	}
}
