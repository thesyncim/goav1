//go:build goav1_trace_rng

package entropy

import (
	"fmt"
	"os"
	"sync/atomic"
)

const (
	// TraceRNGEnabled is true only for goav1_trace_rng builds.
	TraceRNGEnabled   = true
	traceEntropyReads = true
)

var (
	traceSeq   uint64
	traceFrame uint32
)

// traceCDFRead mirrors libaom's GOAV1_TRACE_RNG cdf line so per-symbol entropy
// state can be diffed against an aomdec trace. Only compiled in when the
// goav1_trace_rng build tag is set.
func traceCDFRead(icdf0 uint16, n int, dif uint64, rng uint32, tell int) {
	seq := atomic.AddUint64(&traceSeq, 1) - 1
	frame := atomic.LoadUint32(&traceFrame)
	fmt.Fprintf(os.Stderr,
		"GOAV1_RNG seq=%d frame=%d kind=cdf n=%d cdf0=%d dif=%016x rng=%08x tell=%d\n",
		seq, frame, n, icdf0, dif, rng, tell)
}

// traceBoolRead mirrors libaom's GOAV1_TRACE_RNG bool line. Only compiled in
// when the goav1_trace_rng build tag is set.
func traceBoolRead(f uint16, dif uint64, rng uint32, tell int) {
	seq := atomic.AddUint64(&traceSeq, 1) - 1
	frame := atomic.LoadUint32(&traceFrame)
	fmt.Fprintf(os.Stderr,
		"GOAV1_RNG seq=%d frame=%d kind=bool f=%d dif=%016x rng=%08x tell=%d\n",
		seq, frame, f, dif, rng, tell)
}

// TraceLabel emits an unconditional marker into the trace stream, so the
// surrounding entropy reads can be tagged with the decoder stage that produced
// them. Only compiled in when the goav1_trace_rng build tag is set.
func TraceLabel(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "GOAV1_MARK ")
	fmt.Fprintf(os.Stderr, format, args...)
	fmt.Fprintln(os.Stderr)
}

// TraceSetFrame records the decoder's logical frame index for subsequent trace
// lines. Only compiled in when the goav1_trace_rng build tag is set.
func TraceSetFrame(frame int) {
	if frame < 0 {
		frame = 0
	}
	atomic.StoreUint32(&traceFrame, uint32(frame))
}

// TraceResetSeq rewinds the global sequence counter to zero. Only compiled in
// when the goav1_trace_rng build tag is set.
func TraceResetSeq() {
	atomic.StoreUint64(&traceSeq, 0)
}
