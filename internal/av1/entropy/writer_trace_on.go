//go:build goav1_trace_rng

package entropy

import (
	"fmt"
	"os"
	"sync/atomic"
)

var traceWriteSeq uint64

// traceWriteCDF mirrors traceCDFRead for the write side so encoder symbol
// streams can be diffed against decoder traces. Trace-build only.
func traceWriteCDF(icdf0 uint16, n int) {
	seq := atomic.AddUint64(&traceWriteSeq, 1) - 1
	fmt.Fprintf(os.Stderr, "GOAV1_WRNG seq=%d kind=cdf n=%d cdf0=%d\n", seq, n, icdf0)
}

// traceWriteBool mirrors traceBoolRead for the write side. Trace-build only.
func traceWriteBool(f uint16) {
	seq := atomic.AddUint64(&traceWriteSeq, 1) - 1
	fmt.Fprintf(os.Stderr, "GOAV1_WRNG seq=%d kind=bool f=%d\n", seq, f)
}
