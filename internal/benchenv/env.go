package benchenv

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

var publishBlockedGoEnvVars = []string{
	"GOAMD64",
	"GOARM64",
	"GO386",
	"GOARM",
	"GOMIPS",
	"GOMIPS64",
	"GOPPC64",
	"GOWASM",
	"GOTOOLCHAIN",
	"GOEXPERIMENT",
	"CGO_ENABLED",
	"CC",
	"CXX",
	"GOCACHE",
	"GOMODCACHE",
	"GOPATH",
	"GOTMPDIR",
}

func PublishBlockedGoEnvVars() []string {
	return append([]string(nil), publishBlockedGoEnvVars...)
}

func PublishAmbientGoEnvVars() []string {
	out := []string{"GOFLAGS", "GOMEMLIMIT", "GODEBUG"}
	return append(out, publishBlockedGoEnvVars...)
}

func GoEnvForMetadata() map[string]any {
	return GoEnvForMetadataWithTool("go")
}

func GoEnvForMetadataWithTool(goBin string) map[string]any {
	goBin = strings.TrimSpace(goBin)
	if goBin == "" {
		goBin = "go"
	}
	out, err := exec.Command(goBin, "env", "-json").Output()
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	var env map[string]any
	if err := json.Unmarshal(out, &env); err != nil {
		return map[string]any{"error": err.Error()}
	}
	return env
}

type CPUState struct {
	GOOS                string `json:"goos"`
	AffinitySupported   bool   `json:"affinity_supported"`
	AffinityAllowedList string `json:"affinity_allowed_list,omitempty"`
	CPUOnlineList       string `json:"cpu_online_list,omitempty"`
	AffinityRestricted  bool   `json:"affinity_restricted,omitempty"`
	AffinityProbeError  string `json:"affinity_probe_error,omitempty"`
	FrequencyGovernor   string `json:"frequency_governor,omitempty"`
	FrequencyDriver     string `json:"frequency_driver,omitempty"`
	FrequencyProbeError string `json:"frequency_probe_error,omitempty"`
}

func ObserveCPUState() CPUState {
	state := CPUState{GOOS: runtime.GOOS}
	switch runtime.GOOS {
	case "linux":
		observeLinuxCPUState(&state)
	default:
		state.AffinityProbeError = "cpu affinity probe unsupported on " + runtime.GOOS
		state.FrequencyProbeError = "cpu frequency probe unsupported on " + runtime.GOOS
	}
	return state
}

func ValidateCPUAffinityClaim(claim string, state CPUState) error {
	trimmed := strings.TrimSpace(claim)
	if !state.AffinitySupported || strings.TrimSpace(state.AffinityAllowedList) == "" || strings.TrimSpace(state.CPUOnlineList) == "" {
		if trimmed == "" || isUnpinnedAffinityClaim(trimmed) {
			return nil
		}
		reason := strings.TrimSpace(state.AffinityProbeError)
		if reason == "" {
			if state.GOOS != "" {
				reason = "cpu affinity probe unsupported on " + state.GOOS
			} else {
				reason = "cpu affinity probe unsupported"
			}
		}
		return fmt.Errorf("cpu-affinity claim %q cannot be verified: %s; use none or unsupported when the run is intentionally unpinned", trimmed, reason)
	}
	allowed, err := CanonicalCPUList(state.AffinityAllowedList)
	if err != nil {
		return fmt.Errorf("observed CPU affinity list %q is invalid: %w", state.AffinityAllowedList, err)
	}
	online, err := CanonicalCPUList(state.CPUOnlineList)
	if err != nil {
		return fmt.Errorf("observed online CPU list %q is invalid: %w", state.CPUOnlineList, err)
	}
	restricted := allowed != online
	if isUnpinnedAffinityClaim(trimmed) && restricted {
		return fmt.Errorf("cpu-affinity claims %q but OS reports restricted CPU affinity allowed=%s online=%s", trimmed, allowed, online)
	}
	if claimed, ok := CPUListFromAffinityClaim(trimmed); ok && claimed != allowed {
		return fmt.Errorf("cpu-affinity claims CPUs %s but OS reports allowed CPUs %s", claimed, allowed)
	}
	return nil
}

func observeLinuxCPUState(state *CPUState) {
	onlineRaw, onlineErr := os.ReadFile("/sys/devices/system/cpu/online")
	online := strings.TrimSpace(string(onlineRaw))
	if onlineErr != nil {
		state.AffinityProbeError = onlineErr.Error()
	}
	allowed, err := linuxProcStatusValue("/proc/self/status", "Cpus_allowed_list")
	if err != nil {
		if state.AffinityProbeError != "" {
			state.AffinityProbeError += "; "
		}
		state.AffinityProbeError += err.Error()
	}
	if allowed != "" {
		state.AffinitySupported = true
		state.AffinityAllowedList = allowed
		state.CPUOnlineList = online
		if online != "" {
			allowedCanonical, allowedErr := CanonicalCPUList(allowed)
			onlineCanonical, onlineErr := CanonicalCPUList(online)
			if allowedErr == nil && onlineErr == nil {
				state.AffinityAllowedList = allowedCanonical
				state.CPUOnlineList = onlineCanonical
				state.AffinityRestricted = allowedCanonical != onlineCanonical
			}
		}
	}
	if governor, driver, err := linuxCPUFrequencyPolicy(); err == nil {
		state.FrequencyGovernor = governor
		state.FrequencyDriver = driver
	} else {
		state.FrequencyProbeError = err.Error()
	}
}

func linuxProcStatusValue(path, key string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	prefix := key + ":"
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(line, prefix) {
			return strings.TrimSpace(strings.TrimPrefix(line, prefix)), nil
		}
	}
	return "", fmt.Errorf("%s not found in %s", key, path)
}

func linuxCPUFrequencyPolicy() (governor string, driver string, err error) {
	governorFiles, _ := filepath.Glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")
	driverFiles, _ := filepath.Glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_driver")
	sort.Strings(governorFiles)
	sort.Strings(driverFiles)
	if len(governorFiles) > 0 {
		raw, readErr := os.ReadFile(governorFiles[0])
		if readErr != nil {
			return "", "", readErr
		}
		governor = strings.TrimSpace(string(raw))
	}
	if len(driverFiles) > 0 {
		raw, readErr := os.ReadFile(driverFiles[0])
		if readErr != nil {
			return "", "", readErr
		}
		driver = strings.TrimSpace(string(raw))
	}
	if governor == "" && driver == "" {
		return "", "", fmt.Errorf("cpu frequency policy files not found")
	}
	return governor, driver, nil
}

func isUnpinnedAffinityClaim(claim string) bool {
	normalized := strings.ToLower(strings.TrimSpace(claim))
	switch normalized {
	case "", "none", "no", "no pinning", "no affinity", "unpinned", "unrestricted", "unsupported", "probe unsupported", "unsupported/unpinned", "all", "all cpus", "default", "system default":
		return true
	default:
		return false
	}
}

func CPUListFromAffinityClaim(claim string) (string, bool) {
	trimmed := strings.TrimSpace(strings.ToLower(claim))
	if trimmed == "" {
		return "", false
	}
	hasAffinityKeyword := strings.Contains(trimmed, "cpu") ||
		strings.Contains(trimmed, "affinity") ||
		strings.Contains(trimmed, "taskset") ||
		strings.Contains(trimmed, "cpuset") ||
		strings.Contains(trimmed, "pin")
	for _, token := range strings.FieldsFunc(trimmed, func(r rune) bool {
		return unicode.IsSpace(r) || r == ';' || r == '[' || r == ']' || r == '(' || r == ')'
	}) {
		token = strings.Trim(token, ",")
		candidates := []string{token}
		if _, value, ok := strings.Cut(token, "="); ok {
			candidates = append(candidates, value)
		}
		if _, value, ok := strings.Cut(token, ":"); ok {
			candidates = append(candidates, value)
		}
		for _, candidate := range candidates {
			candidate = strings.Trim(candidate, ",")
			if candidate == "" {
				continue
			}
			if !hasAffinityKeyword && candidate != trimmed && !strings.ContainsAny(candidate, "-,") {
				continue
			}
			canonical, err := CanonicalCPUList(candidate)
			if err == nil {
				return canonical, true
			}
		}
	}
	return "", false
}

func CanonicalCPUList(list string) (string, error) {
	list = strings.ReplaceAll(strings.TrimSpace(list), " ", "")
	if list == "" {
		return "", fmt.Errorf("empty CPU list")
	}
	seen := map[int]bool{}
	for _, part := range strings.Split(list, ",") {
		if part == "" {
			return "", fmt.Errorf("empty CPU range")
		}
		startText, endText, hasRange := strings.Cut(part, "-")
		start, err := strconv.Atoi(startText)
		if err != nil || start < 0 {
			return "", fmt.Errorf("invalid CPU %q", startText)
		}
		end := start
		if hasRange {
			end, err = strconv.Atoi(endText)
			if err != nil || end < start {
				return "", fmt.Errorf("invalid CPU range %q", part)
			}
		}
		for cpu := start; cpu <= end; cpu++ {
			seen[cpu] = true
		}
	}
	cpus := make([]int, 0, len(seen))
	for cpu := range seen {
		cpus = append(cpus, cpu)
	}
	sort.Ints(cpus)
	var ranges []string
	for i := 0; i < len(cpus); {
		start := cpus[i]
		end := start
		for i+1 < len(cpus) && cpus[i+1] == end+1 {
			i++
			end = cpus[i]
		}
		if start == end {
			ranges = append(ranges, strconv.Itoa(start))
		} else {
			ranges = append(ranges, strconv.Itoa(start)+"-"+strconv.Itoa(end))
		}
		i++
	}
	return strings.Join(ranges, ","), nil
}
