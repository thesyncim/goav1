//go:build goav1_encstats

package encoder

import (
	"sync/atomic"
	"time"
)

// txWriteStatsEnabled arms the serial entropy-write timer in goav1_encstats
// builds: the P-frame split pipeline accumulates the wall time of its serial
// write walk so the pack/write stage can be measured per frame.
// On the pipelined wavefront the measured pass overlaps the decision lanes
// and so includes row-wait stalls; on the forced-split serial pipeline it is
// the pure entropy-write time.
const txWriteStatsEnabled = true

var (
	txWriteStatsNS     atomic.Int64
	txWriteStatsPasses atomic.Int64
)

type txWriteStatsStamp struct {
	t time.Time
}

func txWriteStatsStart() txWriteStatsStamp {
	return txWriteStatsStamp{t: time.Now()}
}

func txWriteStatsFinish(start txWriteStatsStamp) {
	txWriteStatsNS.Add(time.Since(start.t).Nanoseconds())
	txWriteStatsPasses.Add(1)
}

// TXWriteStats returns the accumulated serial entropy-write wall time and the
// number of write passes (one per P-frame tile encode on the split pipeline).
func TXWriteStats() (total time.Duration, passes int64) {
	return time.Duration(txWriteStatsNS.Load()), txWriteStatsPasses.Load()
}

// ResetTXWriteStats clears the accumulated serial entropy-write timing.
func ResetTXWriteStats() {
	txWriteStatsNS.Store(0)
	txWriteStatsPasses.Store(0)
}
