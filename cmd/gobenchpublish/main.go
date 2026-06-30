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
	"strconv"
	"strings"
	"time"

	cpufeatures "github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/benchenv"
)

type config struct {
	Pkg              string
	Bench            string
	Tags             string
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
	Version       string            `json:"version"`
	GOOS          string            `json:"goos"`
	GOARCH        string            `json:"goarch"`
	NumCPU        int               `json:"num_cpu"`
	SIMDTier      string            `json:"simd_tier"`
	SIMDFeatures  []string          `json:"simd_features,omitempty"`
	BuildSettings map[string]string `json:"build_settings,omitempty"`
	Env           map[string]any    `json:"env,omitempty"`
}

type metadataConfig struct {
	Package    string `json:"package"`
	Benchmark  string `json:"benchmark"`
	Tags       string `json:"tags,omitempty"`
	GoMaxProcs int    `json:"gomaxprocs"`
	CPU        string `json:"cpu"`
	Count      int    `json:"count"`
	BenchTime  string `json:"benchtime"`
	BenchMem   bool   `json:"benchmem"`
	Publish    bool   `json:"publish"`
}

type environmentConfig struct {
	GOGC             string `json:"gogc,omitempty"`
	GOFLAGS          string `json:"goflags,omitempty"`
	GOMEMLIMIT       string `json:"gomemlimit,omitempty"`
	GODEBUG          string `json:"godebug,omitempty"`
	CPUAffinity      string `json:"cpu_affinity,omitempty"`
	PowerMode        string `json:"power_mode,omitempty"`
	ThermalState     string `json:"thermal_state,omitempty"`
	FrequencyPolicy  string `json:"frequency_policy,omitempty"`
	BackgroundLoad   string `json:"background_load,omitempty"`
	Notes            string `json:"notes,omitempty"`
	EffectiveCommand string `json:"effective_command"`
}

type outputMetadata struct {
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes,omitempty"`
	SHA256 string `json:"sha256,omitempty"`
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
	cmd := exec.Command("go", args...)
	cmd.Env = commandEnv(cfg)
	raw, err := cmd.CombinedOutput()

	if err := writeFileCreatingParents(cfg.OutputPath, raw, 0o644); err != nil {
		return fmt.Errorf("write benchmark output: %w", err)
	}

	meta := buildMetadata(cfg, git, append([]string{"go"}, args...), "ok", "")
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
	if strings.ContainsAny(cfg.Tags, "\r\n") || strings.ContainsAny(cfg.CPU, "\r\n") ||
		strings.ContainsAny(cfg.Bench, "\r\n") || strings.ContainsAny(cfg.Pkg, "\r\n") ||
		strings.ContainsAny(cfg.BenchTime, "\r\n") || strings.ContainsAny(cfg.GOGC, "\r\n") {
		return errors.New("benchmark arguments must not contain newlines")
	}
	if cfg.Publish {
		if git.Error != "" {
			return fmt.Errorf("publish requires git metadata: %s", git.Error)
		}
		for _, name := range []string{
			"pkg",
			"bench",
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
	env := os.Environ()
	env = setEnv(env, "GOMAXPROCS", strconv.Itoa(cfg.GoMaxProcs))
	if cfg.GOGC != "" {
		env = setEnv(env, "GOGC", cfg.GOGC)
	}
	return env
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
	return metadata{
		GeneratedAtUTC: time.Now().UTC().Format(time.RFC3339),
		Git:            git,
		Go: goMetadata{
			Version:       runtime.Version(),
			GOOS:          runtime.GOOS,
			GOARCH:        runtime.GOARCH,
			NumCPU:        runtime.NumCPU(),
			SIMDTier:      detectedSIMDTier(),
			SIMDFeatures:  detectedSIMDFeatures(),
			BuildSettings: goBuildSettings(),
			Env:           benchenv.GoEnvForMetadata(),
		},
		Config: metadataConfig{
			Package:    cfg.Pkg,
			Benchmark:  cfg.Bench,
			Tags:       cfg.Tags,
			GoMaxProcs: cfg.GoMaxProcs,
			CPU:        cfg.CPU,
			Count:      cfg.Count,
			BenchTime:  cfg.BenchTime,
			BenchMem:   cfg.BenchMem,
			Publish:    cfg.Publish,
		},
		Environment: environmentConfig{
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
			EffectiveCommand: strings.Join(command, " "),
		},
		Command: command,
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
