// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

// Command gobenchpublish runs a single publish-grade Go benchmark selection and
// writes provenance metadata beside the raw benchmark output.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"

	cpufeatures "github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/benchenv"
)

var observeBenchmarkCPUState = benchenv.ObserveCPUState

type config struct {
	Pkg              string
	Bench            string
	Tags             string
	GoBin            string
	GoSHA256         string
	OutputPath       string
	MetadataPath     string
	EnvironmentNotes string
	CPUAffinity      string
	PowerMode        string
	ThermalState     string
	FrequencyPolicy  string
	BackgroundLoad   string
	GoMaxProcs       int
	CPU              string
	Count            int
	BenchTime        string
	GOGC             string
	Publish          bool
	BenchMem         bool
	ExplicitFlags    map[string]bool
}

type gitMetadata struct {
	Commit string `json:"commit,omitempty"`
	Dirty  bool   `json:"dirty"`
	Error  string `json:"error,omitempty"`
}

type metadata struct {
	GeneratedAtUTC string            `json:"generated_at_utc"`
	Git            gitMetadata       `json:"git"`
	Go             goMetadata        `json:"go"`
	Config         metadataConfig    `json:"config"`
	Environment    environmentConfig `json:"environment"`
	Command        []string          `json:"command"`
	Output         outputMetadata    `json:"output"`
	Status         string            `json:"status"`
	Error          string            `json:"error,omitempty"`
}

type goMetadata struct {
	Version              string            `json:"version"`
	GOOS                 string            `json:"goos"`
	GOARCH               string            `json:"goarch"`
	NumCPU               int               `json:"num_cpu"`
	SIMDTier             string            `json:"simd_tier"`
	SIMDFeatures         []string          `json:"simd_features,omitempty"`
	BuildSettings        map[string]string `json:"build_settings,omitempty"`
	Env                  map[string]any    `json:"env,omitempty"`
	ToolPath             string            `json:"tool_path,omitempty"`
	ToolSHA256           string            `json:"tool_sha256,omitempty"`
	ToolExpectedSHA256   string            `json:"tool_expected_sha256,omitempty"`
	ToolSHA256Verified   bool              `json:"tool_sha256_verified,omitempty"`
	ToolResolutionError  string            `json:"tool_resolution_error,omitempty"`
	ToolFingerprintError string            `json:"tool_fingerprint_error,omitempty"`
}

type metadataConfig struct {
	Package                 string `json:"package"`
	Benchmark               string `json:"benchmark"`
	BenchmarkFunction       string `json:"benchmark_function,omitempty"`
	SubbenchmarkRowsAllowed bool   `json:"subbenchmark_rows_allowed"`
	Tags                    string `json:"tags,omitempty"`
	GoBin                   string `json:"go_bin"`
	GoMaxProcs              int    `json:"gomaxprocs"`
	CPU                     string `json:"cpu"`
	Count                   int    `json:"count"`
	BenchTime               string `json:"benchtime"`
	BenchMem                bool   `json:"benchmem"`
	Publish                 bool   `json:"publish"`
}

type environmentConfig struct {
	GOGC               string            `json:"gogc,omitempty"`
	GOFLAGS            string            `json:"goflags,omitempty"`
	GOMEMLIMIT         string            `json:"gomemlimit,omitempty"`
	GODEBUG            string            `json:"godebug,omitempty"`
	CommandEnvPolicy   string            `json:"benchmark_command_env_policy,omitempty"`
	CommandEnv         map[string]string `json:"benchmark_command_env,omitempty"`
	CommandFilteredEnv []string          `json:"benchmark_command_filtered_env,omitempty"`
	CPUAffinity        string            `json:"cpu_affinity,omitempty"`
	PowerMode          string            `json:"power_mode,omitempty"`
	ThermalState       string            `json:"thermal_state,omitempty"`
	FrequencyPolicy    string            `json:"frequency_policy,omitempty"`
	BackgroundLoad     string            `json:"background_load,omitempty"`
	Notes              string            `json:"notes,omitempty"`
	ObservedCPUState   benchenv.CPUState `json:"observed_cpu_state"`
	EffectiveCommand   string            `json:"effective_command"`
}

type outputMetadata struct {
	Path          string                 `json:"path"`
	Bytes         int64                  `json:"bytes,omitempty"`
	SHA256        string                 `json:"sha256,omitempty"`
	BenchmarkRows []benchmarkRowMetadata `json:"benchmark_rows,omitempty"`
}

type benchmarkRowMetadata struct {
	Name    string `json:"name"`
	Samples int    `json:"samples"`
	CPUs    []int  `json:"cpus,omitempty"`
}

type toolFingerprint struct {
	path             string
	sha256           string
	expectedSHA256   string
	sha256Verified   bool
	resolutionError  string
	fingerprintError string
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	cfg := parseFlags()
	git := currentGitMetadata()
	if err := validateConfig(cfg, git); err != nil {
		return err
	}

	args := goTestArgs(cfg)
	cmd := exec.Command(cfg.GoBin, args...)
	cmd.Env = commandEnv(cfg)
	raw, err := cmd.CombinedOutput()
	if err == nil && cfg.Publish {
		err = validateBenchmarkOutput(cfg, raw)
	}

	if err := writeFileCreatingParents(cfg.OutputPath, raw, 0o644); err != nil {
		return fmt.Errorf("write benchmark output: %w", err)
	}

	meta := buildMetadata(cfg, git, append([]string{cfg.GoBin}, args...), "ok", "")
	if err != nil {
		meta.Status = "error"
		meta.Error = trimCommandError(err, raw)
	}
	if info, hash, statErr := fileInfoAndSHA256(cfg.OutputPath); statErr == nil {
		meta.Output.Bytes = info
		meta.Output.SHA256 = hash
	} else if meta.Error == "" {
		meta.Status = "error"
		meta.Error = statErr.Error()
	}
	meta.Output.BenchmarkRows = parseBenchmarkOutputRows(raw)
	if err := writeJSONCreatingParents(cfg.MetadataPath, meta); err != nil {
		return fmt.Errorf("write metadata: %w", err)
	}
	if err != nil {
		return errors.New(meta.Error)
	}
	return nil
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.Pkg, "pkg", "./internal/av1/tile", "single Go package or package pattern to benchmark")
	flag.StringVar(&cfg.Bench, "bench", ".", "benchmark regexp passed to go test -bench")
	flag.StringVar(&cfg.Tags, "tags", "", "optional build tags passed to go test -tags")
	flag.StringVar(&cfg.GoBin, "go-bin", "go", "Go executable used for the benchmark process")
	flag.StringVar(&cfg.GoSHA256, "go-sha256", "", "expected SHA-256 of -go-bin for publish runs")
	flag.StringVar(&cfg.OutputPath, "out", "/tmp/goav1-go-bench.txt", "raw go test benchmark output path")
	flag.StringVar(&cfg.MetadataPath, "metadata-json", "/tmp/goav1-go-bench-metadata.json", "metadata JSON path")
	flag.StringVar(&cfg.EnvironmentNotes, "environment-notes", "", "power, thermal, and background-load notes")
	flag.StringVar(&cfg.CPUAffinity, "cpu-affinity", "", "CPU affinity/pinning used for publish runs; use none if the process is intentionally unpinned")
	flag.StringVar(&cfg.PowerMode, "power-mode", "", "power source and performance mode used for publish runs")
	flag.StringVar(&cfg.ThermalState, "thermal-state", "", "pre-run thermal state used for publish runs")
	flag.StringVar(&cfg.FrequencyPolicy, "frequency-policy", "", "CPU frequency/governor policy used for publish runs")
	flag.StringVar(&cfg.BackgroundLoad, "background-load", "", "background-load policy used for publish runs")
	flag.IntVar(&cfg.GoMaxProcs, "gomaxprocs", 1, "GOMAXPROCS value for the benchmark process")
	flag.StringVar(&cfg.CPU, "cpu", "1", "go test -cpu value")
	flag.IntVar(&cfg.Count, "count", 7, "go test -count value")
	flag.StringVar(&cfg.BenchTime, "benchtime", "500ms", "go test -benchtime value")
	flag.StringVar(&cfg.GOGC, "gogc", "off", "GOGC value for the benchmark process; empty keeps environment default")
	flag.BoolVar(&cfg.Publish, "publish", false, "require claim-supporting benchmark controls")
	flag.BoolVar(&cfg.BenchMem, "benchmem", true, "include go test -benchmem")
	flag.Parse()
	cfg.ExplicitFlags = explicitFlagSet()
	cfg.GoBin = strings.TrimSpace(cfg.GoBin)
	cfg.GoSHA256 = strings.TrimSpace(cfg.GoSHA256)
	return cfg
}

func explicitFlagSet() map[string]bool {
	set := map[string]bool{}
	flag.Visit(func(f *flag.Flag) {
		set[f.Name] = true
	})
	return set
}

func validateConfig(cfg config, git gitMetadata) error {
	if strings.TrimSpace(cfg.Pkg) == "" {
		return errors.New("missing -pkg")
	}
	if strings.TrimSpace(cfg.Bench) == "" {
		return errors.New("missing -bench")
	}
	if strings.TrimSpace(cfg.OutputPath) == "" {
		return errors.New("missing -out")
	}
	if strings.TrimSpace(cfg.MetadataPath) == "" {
		return errors.New("missing -metadata-json")
	}
	if samePathSetting(cfg.OutputPath, cfg.MetadataPath) {
		return errors.New("-out and -metadata-json must be different paths")
	}
	if cfg.GoMaxProcs <= 0 {
		return errors.New("-gomaxprocs must be > 0")
	}
	if strings.TrimSpace(cfg.CPU) == "" {
		return errors.New("missing -cpu")
	}
	if cfg.Count <= 0 {
		return errors.New("-count must be > 0")
	}
	if strings.TrimSpace(cfg.BenchTime) == "" {
		return errors.New("missing -benchtime")
	}
	if cfg.GoBin == "" || strings.ContainsAny(cfg.GoBin, "\r\n") {
		return fmt.Errorf("invalid -go-bin %q", cfg.GoBin)
	}
	if strings.ContainsAny(cfg.Tags, "\r\n") || strings.ContainsAny(cfg.CPU, "\r\n") ||
		strings.ContainsAny(cfg.Bench, "\r\n") || strings.ContainsAny(cfg.Pkg, "\r\n") ||
		strings.ContainsAny(cfg.BenchTime, "\r\n") || strings.ContainsAny(cfg.GOGC, "\r\n") ||
		strings.ContainsAny(cfg.GoSHA256, "\r\n") {
		return errors.New("benchmark arguments must not contain newlines")
	}
	if cfg.Publish {
		if git.Error != "" {
			return fmt.Errorf("publish requires git metadata: %s", git.Error)
		}
		for _, name := range []string{
			"pkg",
			"bench",
			"go-bin",
			"go-sha256",
			"out",
			"metadata-json",
			"environment-notes",
			"cpu-affinity",
			"power-mode",
			"thermal-state",
			"frequency-policy",
			"background-load",
			"gomaxprocs",
			"cpu",
			"count",
			"benchtime",
			"benchmem",
			"gogc",
		} {
			if err := requireExplicitFlag(cfg, name); err != nil {
				return err
			}
		}
		if err := validatePublishBenchmarkSelection(cfg); err != nil {
			return err
		}
		if err := validatePinnedPublishTool("go", cfg.GoBin, cfg.GoSHA256); err != nil {
			return err
		}
		if git.Dirty {
			return errors.New("publish requires a clean tracked git worktree")
		}
		for _, field := range []struct {
			name  string
			value string
		}{
			{name: "environment-notes", value: cfg.EnvironmentNotes},
			{name: "cpu-affinity", value: cfg.CPUAffinity},
			{name: "power-mode", value: cfg.PowerMode},
			{name: "thermal-state", value: cfg.ThermalState},
			{name: "frequency-policy", value: cfg.FrequencyPolicy},
			{name: "background-load", value: cfg.BackgroundLoad},
		} {
			if err := validateNonEmptyPublishTextFlag(field.name, field.value); err != nil {
				return err
			}
		}
		cpuState := observeBenchmarkCPUState()
		if err := benchenv.ValidateCPUAffinityClaim(cfg.CPUAffinity, cpuState); err != nil {
			return err
		}
		if err := benchenv.ValidateCPUFrequencyPolicyClaim(cfg.FrequencyPolicy, cpuState); err != nil {
			return err
		}
		if cfg.Count < 5 {
			return errors.New("publish requires -count >= 5")
		}
		if !cfg.BenchMem {
			return errors.New("publish requires -benchmem")
		}
		cpu, err := singlePositiveIntFlag("-cpu", cfg.CPU)
		if err != nil {
			return err
		}
		if cpu != cfg.GoMaxProcs {
			return errors.New("publish requires -cpu to match -gomaxprocs; use separate publish runs for CPU sweeps")
		}
		if strings.TrimSpace(cfg.GOGC) == "" {
			return errors.New("publish requires explicit non-empty -gogc")
		}
		if os.Getenv("GOFLAGS") != "" {
			return errors.New("publish requires GOFLAGS unset; use explicit runner flags such as -tags")
		}
		if os.Getenv("GOMEMLIMIT") != "" {
			return errors.New("publish requires GOMEMLIMIT unset")
		}
		if os.Getenv("GODEBUG") != "" {
			return errors.New("publish requires GODEBUG unset")
		}
		for _, name := range benchenv.PublishBlockedGoEnvVars() {
			if os.Getenv(name) != "" {
				return fmt.Errorf("publish requires %s unset; use explicit runner flags or record a separate environment", name)
			}
		}
	}
	return nil
}

func validatePublishBenchmarkSelection(cfg config) error {
	pkg := strings.TrimSpace(cfg.Pkg)
	if pkg == "" {
		return errors.New("publish requires non-empty -pkg")
	}
	if strings.Contains(pkg, "...") || strings.ContainsAny(pkg, " \t") || strings.Contains(pkg, ",") {
		return fmt.Errorf("publish requires -pkg to name one concrete package, got %q", cfg.Pkg)
	}
	bench := strings.TrimSpace(cfg.Bench)
	if !isExactBenchmarkRegexp(bench) {
		return fmt.Errorf("publish requires -bench to select exactly one benchmark function as ^BenchmarkName$; subbenchmark output rows are allowed only below that function, got %q", cfg.Bench)
	}
	return nil
}

func isExactBenchmarkRegexp(bench string) bool {
	if !strings.HasPrefix(bench, "^Benchmark") || !strings.HasSuffix(bench, "$") {
		return false
	}
	name := strings.TrimSuffix(strings.TrimPrefix(bench, "^"), "$")
	if len(name) <= len("Benchmark") {
		return false
	}
	for _, r := range name[len("Benchmark"):] {
		if r >= 'A' && r <= 'Z' {
			continue
		}
		if r >= 'a' && r <= 'z' {
			continue
		}
		if r >= '0' && r <= '9' {
			continue
		}
		if r == '_' {
			continue
		}
		return false
	}
	return true
}

func validateBenchmarkOutput(cfg config, raw []byte) error {
	rows := parseBenchmarkOutputRows(raw)
	if len(rows) == 0 {
		return errors.New("publish requires go test output to contain at least one benchmark row")
	}
	expectedName := exactBenchmarkName(cfg.Bench)
	if expectedName == "" {
		return fmt.Errorf("publish requires exact benchmark function selection, got %q", cfg.Bench)
	}
	expectedCPU, err := singlePositiveIntFlag("-cpu", cfg.CPU)
	if err != nil {
		return err
	}
	for _, row := range rows {
		if !benchmarkRowMatchesSelectedFunction(row.Name, expectedName) {
			return fmt.Errorf("publish benchmark output contains unexpected row %q for -bench %q", row.Name, cfg.Bench)
		}
		if row.Samples != cfg.Count {
			return fmt.Errorf("publish benchmark row %q has %d samples, want -count %d", row.Name, row.Samples, cfg.Count)
		}
		if len(row.CPUs) != 1 || row.CPUs[0] != expectedCPU {
			return fmt.Errorf("publish benchmark row %q used CPU suffixes %v, want [%d]", row.Name, row.CPUs, expectedCPU)
		}
	}
	return nil
}

func benchmarkRowMatchesSelectedFunction(rowName, benchmarkFunction string) bool {
	return rowName == benchmarkFunction || strings.HasPrefix(rowName, benchmarkFunction+"/")
}

func parseBenchmarkOutputRows(raw []byte) []benchmarkRowMetadata {
	rowsByName := map[string]*benchmarkRowMetadata{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		name, cpu, ok := parseBenchmarkLineNameCPU(fields[0])
		if !ok {
			continue
		}
		row := rowsByName[name]
		if row == nil {
			row = &benchmarkRowMetadata{Name: name}
			rowsByName[name] = row
		}
		row.Samples++
		if !containsInt(row.CPUs, cpu) {
			row.CPUs = append(row.CPUs, cpu)
		}
	}
	rows := make([]benchmarkRowMetadata, 0, len(rowsByName))
	for _, row := range rowsByName {
		sort.Ints(row.CPUs)
		rows = append(rows, *row)
	}
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Name < rows[j].Name
	})
	return rows
}

func parseBenchmarkLineNameCPU(token string) (string, int, bool) {
	if !strings.HasPrefix(token, "Benchmark") {
		return "", 0, false
	}
	i := strings.LastIndexByte(token, '-')
	if i <= len("Benchmark") || i == len(token)-1 {
		return "", 0, false
	}
	cpu, err := strconv.Atoi(token[i+1:])
	if err != nil || cpu <= 0 {
		return "", 0, false
	}
	name := token[:i]
	if name == "Benchmark" {
		return "", 0, false
	}
	return name, cpu, true
}

func containsInt(values []int, needle int) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func exactBenchmarkName(bench string) string {
	bench = strings.TrimSpace(bench)
	if !isExactBenchmarkRegexp(bench) {
		return ""
	}
	return strings.TrimSuffix(strings.TrimPrefix(bench, "^"), "$")
}

func validatePinnedPublishTool(name, path, expectedSHA string) error {
	if !filepath.IsAbs(path) {
		return fmt.Errorf("publish requires -%s-bin to be an absolute path, got %q", name, path)
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("publish requires -%s-bin %s: %w", name, path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("publish requires -%s-bin to be a file, got directory %s", name, path)
	}
	expected, err := canonicalSHA256(expectedSHA)
	if err != nil {
		return fmt.Errorf("publish requires valid -%s-sha256: %w", name, err)
	}
	actual, err := sha256File(path)
	if err != nil {
		return fmt.Errorf("publish requires -%s-bin %s hash: %w", name, path, err)
	}
	if actual != expected {
		return fmt.Errorf("publish requires -%s-bin %s to match -%s-sha256 %s, got %s", name, path, name, expected, actual)
	}
	return nil
}

func requireExplicitFlag(cfg config, name string) error {
	if cfg.ExplicitFlags == nil || !cfg.ExplicitFlags[name] {
		return fmt.Errorf("publish requires explicit -%s", name)
	}
	return nil
}

func validateNonEmptyPublishTextFlag(name, value string) error {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fmt.Errorf("publish requires non-empty -%s", name)
	}
	if strings.ContainsAny(trimmed, "\r\n") {
		return fmt.Errorf("publish requires -%s without newlines", name)
	}
	return nil
}

func singlePositiveIntFlag(name, value string) (int, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return 0, fmt.Errorf("missing %s", name)
	}
	if strings.Contains(trimmed, ",") {
		return 0, fmt.Errorf("publish requires %s to be a single positive integer, got %q", name, value)
	}
	n, err := strconv.Atoi(trimmed)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("publish requires %s to be a single positive integer, got %q", name, value)
	}
	return n, nil
}

func samePathSetting(a, b string) bool {
	return pathSettingKey(a) == pathSettingKey(b)
}

func pathSettingKey(path string) string {
	if abs, err := filepath.Abs(path); err == nil {
		return filepath.Clean(abs)
	}
	return filepath.Clean(path)
}

func goTestArgs(cfg config) []string {
	args := []string{"test", "-run", "^$", "-bench", cfg.Bench}
	if cfg.BenchMem {
		args = append(args, "-benchmem")
	}
	args = append(args,
		"-benchtime", cfg.BenchTime,
		"-count", strconv.Itoa(cfg.Count),
		"-cpu", cfg.CPU,
	)
	if cfg.Tags != "" {
		args = append(args, "-tags", cfg.Tags)
	}
	args = append(args, cfg.Pkg)
	return args
}

func commandEnv(cfg config) []string {
	if cfg.Publish {
		return envMapToList(publishBenchmarkCommandEnvMap(cfg))
	}
	env := os.Environ()
	env = setEnv(env, "GOMAXPROCS", strconv.Itoa(cfg.GoMaxProcs))
	if cfg.GOGC != "" {
		env = setEnv(env, "GOGC", cfg.GOGC)
	}
	return env
}

const publishBenchmarkCommandEnvPolicy = "allowlist PATH, HOME, TMPDIR, TEMP, TMP, XDG_RUNTIME_DIR, SystemRoot, WINDIR, ComSpec; force LANG=C, LC_ALL=C, TZ=UTC; set GOMAXPROCS and GOGC from explicit flags"

var publishBenchmarkCommandEnvAllowlist = []string{
	"PATH",
	"HOME",
	"TMPDIR",
	"TEMP",
	"TMP",
	"XDG_RUNTIME_DIR",
	"SystemRoot",
	"WINDIR",
	"ComSpec",
}

var publishBenchmarkCommandEnvReportedFilters = []string{
	"LANG",
	"LC_ALL",
	"LC_CTYPE",
	"TZ",
	"OMP_NUM_THREADS",
	"OMP_DYNAMIC",
	"OMP_PROC_BIND",
	"OMP_PLACES",
	"GOMP_CPU_AFFINITY",
	"KMP_AFFINITY",
	"KMP_HW_SUBSET",
	"KMP_BLOCKTIME",
	"KMP_SETTINGS",
	"MKL_NUM_THREADS",
	"OPENBLAS_NUM_THREADS",
	"VECLIB_MAXIMUM_THREADS",
	"BLIS_NUM_THREADS",
	"NUMEXPR_NUM_THREADS",
	"RAYON_NUM_THREADS",
	"TBB_NUM_THREADS",
	"LD_PRELOAD",
	"LD_LIBRARY_PATH",
	"DYLD_INSERT_LIBRARIES",
	"DYLD_LIBRARY_PATH",
	"DYLD_FRAMEWORK_PATH",
	"DYLD_FALLBACK_LIBRARY_PATH",
	"MallocNanoZone",
}

var publishBenchmarkCommandEnvReportedPrefixes = []string{
	"MALLOC_",
}

func publishBenchmarkCommandEnvMap(cfg config) map[string]string {
	env := make(map[string]string, len(publishBenchmarkCommandEnvAllowlist)+5)
	for _, key := range publishBenchmarkCommandEnvAllowlist {
		if value, ok := os.LookupEnv(key); ok {
			env[key] = value
		}
	}
	env["LANG"] = "C"
	env["LC_ALL"] = "C"
	env["TZ"] = "UTC"
	env["GOMAXPROCS"] = strconv.Itoa(cfg.GoMaxProcs)
	if cfg.GOGC != "" {
		env["GOGC"] = cfg.GOGC
	}
	return env
}

func envMapToList(env map[string]string) []string {
	keys := make([]string, 0, len(env))
	for key := range env {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		out = append(out, key+"="+env[key])
	}
	return out
}

func presentPublishBenchmarkCommandFilteredEnv() []string {
	seen := map[string]bool{}
	var out []string
	for _, key := range publishBenchmarkCommandEnvReportedFilters {
		if value, ok := os.LookupEnv(key); ok && value != "" {
			seen[key] = true
			out = append(out, key)
		}
	}
	for _, item := range os.Environ() {
		key, value, ok := strings.Cut(item, "=")
		if !ok || value == "" || seen[key] {
			continue
		}
		for _, prefix := range publishBenchmarkCommandEnvReportedPrefixes {
			if strings.HasPrefix(key, prefix) {
				seen[key] = true
				out = append(out, key)
				break
			}
		}
	}
	sort.Strings(out)
	return out
}

func setEnv(env []string, key, value string) []string {
	prefix := key + "="
	for i, item := range env {
		if strings.HasPrefix(item, prefix) {
			env[i] = prefix + value
			return env
		}
	}
	return append(env, prefix+value)
}

func buildMetadata(cfg config, git gitMetadata, command []string, status, errText string) metadata {
	goTool := goToolMetadata(cfg.GoBin, cfg.GoSHA256)
	environment := environmentConfig{
		GOGC:             effectiveGOGC(cfg),
		GOFLAGS:          os.Getenv("GOFLAGS"),
		GOMEMLIMIT:       os.Getenv("GOMEMLIMIT"),
		GODEBUG:          os.Getenv("GODEBUG"),
		CPUAffinity:      strings.TrimSpace(cfg.CPUAffinity),
		PowerMode:        strings.TrimSpace(cfg.PowerMode),
		ThermalState:     strings.TrimSpace(cfg.ThermalState),
		FrequencyPolicy:  strings.TrimSpace(cfg.FrequencyPolicy),
		BackgroundLoad:   strings.TrimSpace(cfg.BackgroundLoad),
		Notes:            strings.TrimSpace(cfg.EnvironmentNotes),
		ObservedCPUState: observeBenchmarkCPUState(),
		EffectiveCommand: strings.Join(command, " "),
	}
	if cfg.Publish {
		environment.CommandEnvPolicy = publishBenchmarkCommandEnvPolicy
		environment.CommandEnv = publishBenchmarkCommandEnvMap(cfg)
		environment.CommandFilteredEnv = presentPublishBenchmarkCommandFilteredEnv()
	}
	return metadata{
		GeneratedAtUTC: time.Now().UTC().Format(time.RFC3339),
		Git:            git,
		Go: goMetadata{
			Version:              runtime.Version(),
			GOOS:                 runtime.GOOS,
			GOARCH:               runtime.GOARCH,
			NumCPU:               runtime.NumCPU(),
			SIMDTier:             detectedSIMDTier(),
			SIMDFeatures:         detectedSIMDFeatures(),
			BuildSettings:        goBuildSettings(),
			Env:                  benchenv.GoEnvForMetadataWithTool(cfg.GoBin),
			ToolPath:             goTool.path,
			ToolSHA256:           goTool.sha256,
			ToolExpectedSHA256:   goTool.expectedSHA256,
			ToolSHA256Verified:   goTool.sha256Verified,
			ToolResolutionError:  goTool.resolutionError,
			ToolFingerprintError: goTool.fingerprintError,
		},
		Config: metadataConfig{
			Package:                 cfg.Pkg,
			Benchmark:               cfg.Bench,
			BenchmarkFunction:       exactBenchmarkName(cfg.Bench),
			SubbenchmarkRowsAllowed: true,
			Tags:                    cfg.Tags,
			GoBin:                   cfg.GoBin,
			GoMaxProcs:              cfg.GoMaxProcs,
			CPU:                     cfg.CPU,
			Count:                   cfg.Count,
			BenchTime:               cfg.BenchTime,
			BenchMem:                cfg.BenchMem,
			Publish:                 cfg.Publish,
		},
		Environment: environment,
		Command:     command,
		Output: outputMetadata{
			Path: cfg.OutputPath,
		},
		Status: status,
		Error:  errText,
	}
}

func goBuildSettings() map[string]string {
	info, ok := debug.ReadBuildInfo()
	if !ok || len(info.Settings) == 0 {
		return nil
	}
	out := make(map[string]string, len(info.Settings))
	for _, setting := range info.Settings {
		if setting.Key != "" {
			out[setting.Key] = setting.Value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func detectedSIMDTier() string {
	return simdTierFor(cpufeatures.Detected)
}

func detectedSIMDFeatures() []string {
	return simdFeaturesFor(cpufeatures.Detected)
}

func simdTierFor(f cpufeatures.Features) string {
	switch {
	case f.SVE2:
		return "sve2"
	case f.SVE:
		return "sve"
	case f.I8MM:
		return "neon_i8mm"
	case f.DOTPROD:
		return "neon_dotprod"
	case f.NEON:
		return "neon"
	case f.AVX512:
		return "avx512"
	case f.AVX2:
		return "avx2"
	case f.SSE42:
		return "sse4_2"
	case f.SSE41:
		return "sse4_1"
	case f.SSE2:
		return "sse2"
	default:
		return "purego"
	}
}

func simdFeaturesFor(f cpufeatures.Features) []string {
	var out []string
	if f.SSE2 {
		out = append(out, "sse2")
	}
	if f.SSE41 {
		out = append(out, "sse4_1")
	}
	if f.SSE42 {
		out = append(out, "sse4_2")
	}
	if f.AVX2 {
		out = append(out, "avx2")
	}
	if f.AVX512 {
		out = append(out, "avx512")
	}
	if f.NEON {
		out = append(out, "neon")
	}
	if f.DOTPROD {
		out = append(out, "neon_dotprod")
	}
	if f.I8MM {
		out = append(out, "neon_i8mm")
	}
	if f.SVE {
		out = append(out, "sve")
	}
	if f.SVE2 {
		out = append(out, "sve2")
	}
	return out
}

func effectiveGOGC(cfg config) string {
	if cfg.GOGC != "" {
		return cfg.GOGC
	}
	return os.Getenv("GOGC")
}

func currentGitMetadata() gitMetadata {
	commitRaw, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return gitMetadata{Error: trimCommandError(err, commitRaw)}
	}
	statusRaw, err := exec.Command("git", "status", "--short").Output()
	if err != nil {
		return gitMetadata{Commit: strings.TrimSpace(string(commitRaw)), Error: trimCommandError(err, statusRaw)}
	}
	return gitMetadata{
		Commit: strings.TrimSpace(string(commitRaw)),
		Dirty:  strings.TrimSpace(string(statusRaw)) != "",
	}
}

func writeFileCreatingParents(path string, raw []byte, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, raw, perm)
}

func writeJSONCreatingParents(path string, value any) error {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return writeFileCreatingParents(path, raw, 0o644)
}

func fileInfoAndSHA256(path string) (int64, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, "", err
	}
	sum := sha256.Sum256(raw)
	return int64(len(raw)), hex.EncodeToString(sum[:]), nil
}

func sha256File(path string) (string, error) {
	_, hash, err := fileInfoAndSHA256(path)
	return hash, err
}

func canonicalSHA256(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if len(value) != sha256.Size*2 {
		return "", fmt.Errorf("expected 64-hex SHA-256, got %q", value)
	}
	for _, r := range value {
		if (r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') {
			continue
		}
		return "", fmt.Errorf("expected 64-hex SHA-256, got %q", value)
	}
	return value, nil
}

func goToolMetadata(path, expectedSHA string) toolFingerprint {
	path = strings.TrimSpace(path)
	if path == "" {
		path = "go"
	}
	out := toolFingerprint{path: path}
	if expectedSHA != "" {
		expected, err := canonicalSHA256(expectedSHA)
		if err != nil {
			out.fingerprintError = err.Error()
		} else {
			out.expectedSHA256 = expected
		}
	}
	resolved := path
	if !filepath.IsAbs(resolved) {
		var err error
		resolved, err = exec.LookPath(path)
		if err != nil {
			out.resolutionError = err.Error()
			return out
		}
	}
	out.path = resolved
	hash, err := sha256File(resolved)
	if err != nil {
		out.fingerprintError = err.Error()
		return out
	}
	out.sha256 = hash
	out.sha256Verified = out.expectedSHA256 != "" && out.sha256 == out.expectedSHA256
	return out
}

func trimCommandError(err error, raw []byte) string {
	msg := strings.TrimSpace(string(raw))
	if msg == "" && err != nil {
		msg = err.Error()
	}
	if len(msg) > 600 {
		msg = msg[:600]
	}
	return msg
}
