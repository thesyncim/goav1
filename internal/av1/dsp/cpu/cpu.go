// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cpu

// Features describes the SIMD instruction sets the DSP dispatcher may target
// on this process. A field is true only when the underlying CPU advertises
// the feature AND the running Go toolchain is known to emit valid code for
// it on this GOARCH. New architectures default to all-false (pure-Go path).
//
// The struct intentionally lists fields per-architecture so a kernel author
// can write `if cpu.Detected.NEON { ... }` without first checking GOARCH.
// On non-matching architectures the field is simply always false.
type Features struct {
	// AMD64 (x86-64) feature flags. Always false on non-amd64 builds.
	SSE2   bool
	SSE41  bool
	SSE42  bool
	AVX2   bool
	AVX512 bool

	// ARM64 feature flags. Always false on non-arm64 builds.
	NEON    bool
	DOTPROD bool
	I8MM    bool
	SVE     bool
	SVE2    bool
}

// Detected is the result of feature detection, populated exactly once at
// process start via a per-architecture init() function. It must not be
// mutated after init; the dispatch layer reads it from many goroutines.
//
// Tests that need to force the pure-Go path can use OverrideForTest below.
var Detected Features

// OverrideForTest replaces Detected with f and returns a restore function.
// It is intended only for tests that want to exercise a specific dispatch
// branch. Callers are responsible for serialising access; the dispatcher
// resolves its function pointer at init time and is not re-evaluated when
// Detected changes, so this only affects code that consults Detected
// directly.
func OverrideForTest(f Features) (restore func()) {
	prev := Detected
	Detected = f
	return func() { Detected = prev }
}
