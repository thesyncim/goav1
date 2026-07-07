package threading

// OverrideWavefrontMinSBRowsPerWorkerForTest overrides the SB-row threshold
// for focused end-to-end wavefront tests that use small conformance clips.
func OverrideWavefrontMinSBRowsPerWorkerForTest(n int) (restore func()) {
	if n < 1 {
		n = 1
	}
	prev := frameWorkWavefrontMinSBRowsPerWorker
	frameWorkWavefrontMinSBRowsPerWorker = n
	return func() {
		frameWorkWavefrontMinSBRowsPerWorker = prev
	}
}
