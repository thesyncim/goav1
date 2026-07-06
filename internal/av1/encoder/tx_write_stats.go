package encoder

import (
	"os"
	"sync/atomic"
	"time"
)

// txWriteStatsEnabled arms the env-gated serial entropy-write timer
// (GOAV1_TX_WRITE_STATS=1): the P-frame split pipeline accumulates the wall
// time of its serial write walk so the pack/write stage — the wavefront's
// Amdahl bound — can be measured per frame. Zero overhead when unset.
// On the pipelined wavefront the measured pass overlaps the decision lanes
// and so includes row-wait stalls; on the forced-split serial pipeline it is
// the pure entropy-write time.
var txWriteStatsEnabled = os.Getenv("GOAV1_TX_WRITE_STATS") == "1"

var (
	txWriteStatsNS     atomic.Int64
	txWriteStatsPasses atomic.Int64
)

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
