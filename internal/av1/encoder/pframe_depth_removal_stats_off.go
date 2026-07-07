//go:build !goav1_encstats

package encoder

const depthRemovalStatsEnabled = false

func noteDepthRemovalSB(dec realtimeDepthRemovalDecision, isBase bool, wouldSplit, proxyFired *[4][4]bool,
	pending32 *[4]bool, pending64, applied32, applied64 bool) {
}

func dumpDepthRemovalStats(frame uint64) {
}
