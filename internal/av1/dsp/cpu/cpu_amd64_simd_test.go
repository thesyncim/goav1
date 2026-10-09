//go:build amd64 && goexperiment.simd && !purego

package cpu

import (
	"simd/archsimd"
	"testing"
)

// TestSIMDAMD64MatchesArchsimdQueries checks that the SIMD build reports
// exactly what the official archsimd queries report, with no extra inference.
func TestSIMDAMD64MatchesArchsimdQueries(t *testing.T) {
	if !Detected.SSE2 {
		t.Fatalf("expected SSE2 on amd64, got %#v", Detected)
	}
	if got, want := Detected.AVX2, archsimd.X86.AVX2(); got != want {
		t.Fatalf("Detected.AVX2 = %v, archsimd.X86.AVX2() = %v", got, want)
	}
	if got, want := Detected.AVX512, archsimd.X86.AVX512(); got != want {
		t.Fatalf("Detected.AVX512 = %v, archsimd.X86.AVX512() = %v", got, want)
	}
	if Detected.SSE41 || Detected.SSE42 {
		t.Fatalf("SSE4.1/SSE4.2 have no archsimd query and must stay false: %#v", Detected)
	}
	if Detected.AVX512 && !Detected.AVX2 {
		t.Fatalf("AVX512 reported without AVX2: %#v", Detected)
	}
	if Detected.NEON || Detected.DOTPROD || Detected.I8MM || Detected.SVE || Detected.SVE2 {
		t.Fatalf("arm64 features unexpectedly set on amd64: %#v", Detected)
	}
}
