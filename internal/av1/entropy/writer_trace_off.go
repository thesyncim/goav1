//go:build !goav1_trace_rng

package entropy

func traceWriteCDF(icdf0 uint16, n int) {}

func traceWriteBool(f uint16) {}
