// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cpu

import (
	"runtime"
	"testing"
)

// TestDetectedMatchesArch verifies the per-arch init populated Detected the
// way the DSP dispatch layer expects.
func TestDetectedMatchesArch(t *testing.T) {
	switch runtime.GOARCH {
	case "amd64":
		if !Detected.SSE2 {
			t.Fatalf("expected SSE2 on amd64, got %#v", Detected)
		}
		if Detected.NEON || Detected.SVE {
			t.Fatalf("arm64 features unexpectedly set on amd64: %#v", Detected)
		}
	case "arm64":
		if !Detected.NEON {
			t.Fatalf("expected NEON on arm64, got %#v", Detected)
		}
		if Detected.SSE2 || Detected.AVX2 || Detected.AVX512 {
			t.Fatalf("amd64 features unexpectedly set on arm64: %#v", Detected)
		}
	default:
		// Generic platforms keep the all-zero default; the dispatch layer
		// is then guaranteed to take the pure-Go path.
		if Detected != (Features{}) {
			t.Fatalf("expected zero Features on %s, got %#v", runtime.GOARCH, Detected)
		}
	}
}

// TestOverrideForTestRestoresPreviousState confirms callers can scope a
// detection override to a single test without polluting global state.
func TestOverrideForTestRestoresPreviousState(t *testing.T) {
	prev := Detected
	restore := OverrideForTest(Features{})
	if Detected != (Features{}) {
		t.Fatalf("override did not take effect: %#v", Detected)
	}
	restore()
	if Detected != prev {
		t.Fatalf("restore did not bring back previous state: got %#v want %#v", Detected, prev)
	}
}
