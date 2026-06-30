package benchenv

import (
	"strings"
	"testing"
)

func TestCanonicalCPUList(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		{in: "0", want: "0"},
		{in: "2,0-1,1", want: "0-2"},
		{in: "0-1,3,5-6", want: "0-1,3,5-6"},
		{in: " 0 - 2 , 4 ", want: "0-2,4"},
	} {
		got, err := CanonicalCPUList(tc.in)
		if err != nil {
			t.Fatalf("CanonicalCPUList(%q): %v", tc.in, err)
		}
		if got != tc.want {
			t.Fatalf("CanonicalCPUList(%q)=%q want %q", tc.in, got, tc.want)
		}
	}

	for _, bad := range []string{"", "x", "2-1", "0,,1"} {
		if got, err := CanonicalCPUList(bad); err == nil {
			t.Fatalf("CanonicalCPUList(%q)=%q, want error", bad, got)
		}
	}
}

func TestCPUListFromAffinityClaim(t *testing.T) {
	for _, tc := range []struct {
		claim string
		want  string
		ok    bool
	}{
		{claim: "taskset 0-3", want: "0-3", ok: true},
		{claim: "cpus=2,0-1", want: "0-2", ok: true},
		{claim: "0,2-3", want: "0,2-3", ok: true},
		{claim: "none", ok: false},
		{claim: "socket 0", ok: false},
	} {
		got, ok := CPUListFromAffinityClaim(tc.claim)
		if ok != tc.ok || got != tc.want {
			t.Fatalf("CPUListFromAffinityClaim(%q)=%q,%v want %q,%v", tc.claim, got, ok, tc.want, tc.ok)
		}
	}
}

func TestValidateCPUAffinityClaim(t *testing.T) {
	restricted := CPUState{
		AffinitySupported:   true,
		AffinityAllowedList: "0-1",
		CPUOnlineList:       "0-3",
	}
	if err := ValidateCPUAffinityClaim("taskset 0-1", restricted); err != nil {
		t.Fatalf("matching explicit claim failed: %v", err)
	}
	if err := ValidateCPUAffinityClaim("none", restricted); err == nil ||
		!strings.Contains(err.Error(), "restricted CPU affinity") {
		t.Fatalf("none claim error=%v, want restricted CPU affinity", err)
	}
	if err := ValidateCPUAffinityClaim("taskset 2-3", restricted); err == nil ||
		!strings.Contains(err.Error(), "allowed CPUs 0-1") {
		t.Fatalf("mismatched explicit claim error=%v", err)
	}

	unrestricted := CPUState{
		AffinitySupported:   true,
		AffinityAllowedList: "0-3",
		CPUOnlineList:       "0-3",
	}
	if err := ValidateCPUAffinityClaim("none", unrestricted); err != nil {
		t.Fatalf("unrestricted none claim failed: %v", err)
	}
	if err := ValidateCPUAffinityClaim("none", CPUState{}); err != nil {
		t.Fatalf("unsupported probe should not fail: %v", err)
	}
	unsupported := CPUState{
		GOOS:               "darwin",
		AffinityProbeError: "cpu affinity probe unsupported on darwin",
	}
	if err := ValidateCPUAffinityClaim("unsupported", unsupported); err != nil {
		t.Fatalf("explicit unsupported claim failed: %v", err)
	}
	if err := ValidateCPUAffinityClaim("taskset 0-1", unsupported); err == nil ||
		!strings.Contains(err.Error(), "cannot be verified") {
		t.Fatalf("unsupported concrete claim error=%v", err)
	}
}

func TestValidateCPUFrequencyPolicyClaim(t *testing.T) {
	observed := CPUState{
		GOOS:              "linux",
		FrequencyGovernor: "performance",
		FrequencyDriver:   "amd-pstate",
	}
	if err := ValidateCPUFrequencyPolicyClaim("governor=performance driver=amd-pstate", observed); err != nil {
		t.Fatalf("matching structured claim failed: %v", err)
	}
	if err := ValidateCPUFrequencyPolicyClaim("performance", observed); err != nil {
		t.Fatalf("matching governor shorthand failed: %v", err)
	}
	if err := ValidateCPUFrequencyPolicyClaim("governor=powersave", observed); err == nil ||
		!strings.Contains(err.Error(), "governor") {
		t.Fatalf("mismatched governor error=%v", err)
	}
	if err := ValidateCPUFrequencyPolicyClaim("unsupported", observed); err == nil ||
		!strings.Contains(err.Error(), "OS reports frequency policy") {
		t.Fatalf("false unsupported error=%v", err)
	}

	unsupported := CPUState{
		GOOS:                "darwin",
		FrequencyProbeError: "cpu frequency probe unsupported on darwin",
	}
	if err := ValidateCPUFrequencyPolicyClaim("macOS automatic", unsupported); err != nil {
		t.Fatalf("automatic unsupported claim failed: %v", err)
	}
	if err := ValidateCPUFrequencyPolicyClaim("unsupported", unsupported); err != nil {
		t.Fatalf("explicit unsupported claim failed: %v", err)
	}
	if err := ValidateCPUFrequencyPolicyClaim("governor=performance", unsupported); err == nil ||
		!strings.Contains(err.Error(), "cannot be verified") {
		t.Fatalf("unsupported concrete claim error=%v", err)
	}
}
