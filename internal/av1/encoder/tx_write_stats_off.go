//go:build !goav1_encstats

package encoder

import "time"

const txWriteStatsEnabled = false

type txWriteStatsStamp struct{}

func txWriteStatsStart() txWriteStatsStamp {
	return txWriteStatsStamp{}
}

func txWriteStatsFinish(start txWriteStatsStamp) {
}

// TXWriteStats is a no-op stub unless the goav1_encstats build tag is set.
func TXWriteStats() (total time.Duration, passes int64) {
	return 0, 0
}

// ResetTXWriteStats is a no-op stub unless the goav1_encstats build tag is set.
func ResetTXWriteStats() {
}
