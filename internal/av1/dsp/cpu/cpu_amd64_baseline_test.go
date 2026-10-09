//go:build amd64 && (purego || !goexperiment.simd)

package cpu

import "testing"

// TestBaselineAMD64ReportsSSE2Only checks that non-SIMD and purego amd64
// builds run no feature probing and report only the SSE2 baseline.
func TestBaselineAMD64ReportsSSE2Only(t *testing.T) {
	if want := (Features{SSE2: true}); Detected != want {
		t.Fatalf("Detected = %#v, want %#v", Detected, want)
	}
}
