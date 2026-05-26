//go:build !goav1_trace_rng

package entropy

// traceCDFRead is a no-op stub when the goav1_trace_rng build tag is not set.
// Keeping the helper inlineable preserves the entropy reader hot path.
func traceCDFRead(_ uint16, _ int, _ uint32, _ uint32, _ int) {}

// traceBoolRead is a no-op stub when the goav1_trace_rng build tag is not set.
func traceBoolRead(_ uint16, _ uint32, _ uint32, _ int) {}

// TraceLabel is a no-op stub when the goav1_trace_rng build tag is not set.
func TraceLabel(_ string, _ ...any) {}

// TraceSetFrame is a no-op stub when the goav1_trace_rng build tag is not set.
func TraceSetFrame(_ int) {}

// TraceResetSeq is a no-op stub when the goav1_trace_rng build tag is not set.
func TraceResetSeq() {}
