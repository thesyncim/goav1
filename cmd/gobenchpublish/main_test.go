// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/thesyncim/goav1/internal/benchenv"
)

func TestValidatePublishConfigRequiresStrictControls(t *testing.T) {
	clearGobenchPublishGoEnv(t)
	stubGobenchPublishCPUState(t, benchenv.CPUState{
		GOOS:                "test",
		AffinitySupported:   true,
		AffinityAllowedList: "0",
		CPUOnlineList:       "0",
	})

	goBin, goSHA := stubGobenchPublishGoTool(t)
	cfg := config{
		Pkg:              "./internal/av1/tile",
		Bench:            "^BenchmarkCoeffCulLevel$",
		GoBin:            goBin,
		GoSHA256:         goSHA,
		OutputPath:       "/tmp/bench.txt",
		MetadataPath:     "/tmp/bench.json",
		EnvironmentNotes: "fixed power mode, idle machine",
		CPUAffinity:      "none",
		PowerMode:        "high power",
		ThermalState:     "cool start",
		FrequencyPolicy:  "automatic",
		BackgroundLoad:   "idle machine",
		GoMaxProcs:       1,
		CPU:              "1",
		Count:            5,
		BenchTime:        "500ms",
		GOGC:             "off",
		Publish:          true,
		BenchMem:         true,
		ExplicitFlags:    gobenchPublishExplicitFlags(),
	}
	if err := validateConfig(cfg, gitMetadata{Commit: "abc"}); err != nil {
		t.Fatalf("valid publish config failed: %v", err)
	}
	oldProbe := observeBenchmarkCPUState
	observeBenchmarkCPUState = func() benchenv.CPUState {
		return benchenv.CPUState{
			GOOS:                "test",
			AffinitySupported:   true,
			AffinityAllowedList: "0",
			CPUOnlineList:       "0-1",
		}
	}
	if err := validateConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "restricted CPU affinity") {
		t.Fatalf("restricted CPU affinity error=%v", err)
	}
	observeBenchmarkCPUState = oldProbe

	dirty := cfg
	if err := validateConfig(dirty, gitMetadata{Commit: "abc", Dirty: true}); err == nil ||
		!strings.Contains(err.Error(), "clean tracked git worktree") {
		t.Fatalf("dirty publish error=%v", err)
	}

	noNotes := cfg
	noNotes.EnvironmentNotes = " "
	if err := validateConfig(noNotes, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "environment-notes") {
		t.Fatalf("missing notes error=%v", err)
	}

	noMachineState := cfg
	noMachineState.CPUAffinity = " "
	if err := validateConfig(noMachineState, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "cpu-affinity") {
		t.Fatalf("missing cpu-affinity error=%v", err)
	}

	tooFew := cfg
	tooFew.Count = 3
	if err := validateConfig(tooFew, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-count >= 5") {
		t.Fatalf("too few count error=%v", err)
	}

	noBenchMem := cfg
	noBenchMem.BenchMem = false
	if err := validateConfig(noBenchMem, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-benchmem") {
		t.Fatalf("missing benchmem error=%v", err)
	}

	implicitBenchMem := cfg
	implicitBenchMem.ExplicitFlags = gobenchPublishExplicitFlags()
	delete(implicitBenchMem.ExplicitFlags, "benchmem")
	if err := validateConfig(implicitBenchMem, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-benchmem") {
		t.Fatalf("implicit benchmem error=%v", err)
	}

	sameOutput := cfg
	sameOutput.MetadataPath = sameOutput.OutputPath
	if err := validateConfig(sameOutput, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "different paths") {
		t.Fatalf("same output path error=%v", err)
	}

	cpuSweep := cfg
	cpuSweep.CPU = "1,4"
	if err := validateConfig(cpuSweep, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "single positive integer") {
		t.Fatalf("cpu sweep error=%v", err)
	}

	cpuMismatch := cfg
	cpuMismatch.CPU = "2"
	if err := validateConfig(cpuMismatch, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "match -gomaxprocs") {
		t.Fatalf("cpu mismatch error=%v", err)
	}

	noGOGC := cfg
	noGOGC.GOGC = " "
	if err := validateConfig(noGOGC, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-gogc") {
		t.Fatalf("missing gogc error=%v", err)
	}

	broadBench := cfg
	broadBench.Bench = "."
	if err := validateConfig(broadBench, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "exactly one benchmark function") {
		t.Fatalf("broad benchmark error=%v", err)
	}

	broadPkg := cfg
	broadPkg.Pkg = "./..."
	if err := validateConfig(broadPkg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "one concrete package") {
		t.Fatalf("broad package error=%v", err)
	}

	badGoHash := cfg
	badGoHash.GoSHA256 = strings.Repeat("0", 64)
	if err := validateConfig(badGoHash, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "go-bin") {
		t.Fatalf("bad go hash error=%v", err)
	}
}

func TestValidatePublishConfigRejectsHiddenGoEnvironment(t *testing.T) {
	clearGobenchPublishGoEnv(t)
	stubGobenchPublishCPUState(t, benchenv.CPUState{
		GOOS:                "test",
		AffinitySupported:   true,
		AffinityAllowedList: "0",
		CPUOnlineList:       "0",
	})
	goBin, goSHA := stubGobenchPublishGoTool(t)
	cfg := config{
		Pkg:              "./internal/av1/tile",
		Bench:            "^BenchmarkCoeffCulLevel$",
		GoBin:            goBin,
		GoSHA256:         goSHA,
		OutputPath:       "/tmp/bench.txt",
		MetadataPath:     "/tmp/bench.json",
		EnvironmentNotes: "fixed power mode, idle machine",
		CPUAffinity:      "none",
		PowerMode:        "high power",
		ThermalState:     "cool start",
		FrequencyPolicy:  "automatic",
		BackgroundLoad:   "idle machine",
		GoMaxProcs:       1,
		CPU:              "1",
		Count:            5,
		BenchTime:        "500ms",
		GOGC:             "off",
		Publish:          true,
		BenchMem:         true,
		ExplicitFlags:    gobenchPublishExplicitFlags(),
	}

	t.Setenv("GOFLAGS", "-race")
	t.Setenv("GOMEMLIMIT", "")
	t.Setenv("GODEBUG", "")
	if err := validateConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GOFLAGS unset") {
		t.Fatalf("hidden GOFLAGS error=%v", err)
	}

	t.Setenv("GOFLAGS", "")
	t.Setenv("GOMEMLIMIT", "512MiB")
	if err := validateConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GOMEMLIMIT unset") {
		t.Fatalf("hidden GOMEMLIMIT error=%v", err)
	}

	t.Setenv("GOMEMLIMIT", "")
	t.Setenv("GODEBUG", "gcstoptheworld=1")
	if err := validateConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GODEBUG unset") {
		t.Fatalf("hidden GODEBUG error=%v", err)
	}

	t.Setenv("GODEBUG", "")
	t.Setenv("GOAMD64", "v4")
	if err := validateConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GOAMD64 unset") {
		t.Fatalf("hidden GOAMD64 error=%v", err)
	}

	t.Setenv("GOAMD64", "")
	t.Setenv("GOCACHE", filepath.Join(t.TempDir(), "cache"))
	if err := validateConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GOCACHE unset") {
		t.Fatalf("hidden GOCACHE error=%v", err)
	}
}

func TestValidateBenchmarkOutputRequiresExactRows(t *testing.T) {
	cfg := config{
		Bench: "^BenchmarkFoo$",
		CPU:   "1",
		Count: 5,
	}
	valid := strings.Join([]string{
		"goos: test",
		"goarch: test",
		"BenchmarkFoo-1 100 10 ns/op 0 B/op 0 allocs/op",
		"BenchmarkFoo-1 100 11 ns/op 0 B/op 0 allocs/op",
		"BenchmarkFoo-1 100 12 ns/op 0 B/op 0 allocs/op",
		"BenchmarkFoo-1 100 13 ns/op 0 B/op 0 allocs/op",
		"BenchmarkFoo-1 100 14 ns/op 0 B/op 0 allocs/op",
		"PASS",
	}, "\n")
	if err := validateBenchmarkOutput(cfg, []byte(valid)); err != nil {
		t.Fatalf("valid rows failed: %v", err)
	}

	noRows := "PASS\nok example 0.001s\n"
	if err := validateBenchmarkOutput(cfg, []byte(noRows)); err == nil ||
		!strings.Contains(err.Error(), "at least one benchmark row") {
		t.Fatalf("no rows error=%v", err)
	}

	partial := strings.Replace(valid, "BenchmarkFoo-1 100 14 ns/op 0 B/op 0 allocs/op\n", "", 1)
	if err := validateBenchmarkOutput(cfg, []byte(partial)); err == nil ||
		!strings.Contains(err.Error(), "has 4 samples") {
		t.Fatalf("partial rows error=%v", err)
	}

	unexpected := strings.Replace(valid, "BenchmarkFoo-1", "BenchmarkBar-1", 1)
	if err := validateBenchmarkOutput(cfg, []byte(unexpected)); err == nil ||
		!strings.Contains(err.Error(), "unexpected row") {
		t.Fatalf("unexpected row error=%v", err)
	}

	wrongCPU := strings.Replace(valid, "BenchmarkFoo-1", "BenchmarkFoo-2", 1)
	if err := validateBenchmarkOutput(cfg, []byte(wrongCPU)); err == nil ||
		!strings.Contains(err.Error(), "CPU suffixes") {
		t.Fatalf("wrong CPU error=%v", err)
	}

	subRows := strings.ReplaceAll(valid, "BenchmarkFoo-1", "BenchmarkFoo/subcase-1")
	if err := validateBenchmarkOutput(cfg, []byte(subRows)); err != nil {
		t.Fatalf("valid subbenchmark rows failed: %v", err)
	}

	if !benchmarkRowMatchesSelectedFunction("BenchmarkFoo/subcase", "BenchmarkFoo") ||
		benchmarkRowMatchesSelectedFunction("BenchmarkFoobar", "BenchmarkFoo") {
		t.Fatalf("benchmark row selector matching is not exact")
	}
}

func TestParseBenchmarkOutputRows(t *testing.T) {
	raw := []byte(strings.Join([]string{
		"BenchmarkFoo-1 100 10 ns/op",
		"BenchmarkFoo-1 100 11 ns/op",
		"BenchmarkFoo/sub-2 100 12 ns/op",
		"PASS",
	}, "\n"))
	rows := parseBenchmarkOutputRows(raw)
	if len(rows) != 2 ||
		rows[0].Name != "BenchmarkFoo" || rows[0].Samples != 2 ||
		len(rows[0].CPUs) != 1 || rows[0].CPUs[0] != 1 ||
		rows[1].Name != "BenchmarkFoo/sub" || rows[1].Samples != 1 ||
		len(rows[1].CPUs) != 1 || rows[1].CPUs[0] != 2 {
		t.Fatalf("rows=%+v", rows)
	}
}

func TestGoTestArgsArePinned(t *testing.T) {
	cfg := config{
		Pkg:       "./internal/av1/tile",
		Bench:     "^BenchmarkCoeffCulLevel$",
		Tags:      "goav1_oracle",
		Count:     7,
		BenchTime: "750ms",
		CPU:       "1",
		BenchMem:  true,
	}
	got := goTestArgs(cfg)
	want := []string{
		"test", "-run", "^$", "-bench", "^BenchmarkCoeffCulLevel$",
		"-benchmem", "-benchtime", "750ms", "-count", "7", "-cpu", "1",
		"-tags", "goav1_oracle", "./internal/av1/tile",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args=%q want %q", got, want)
	}
}

func TestMetadataJSONRecordsOutputHash(t *testing.T) {
	stubGobenchPublishCPUState(t, benchenv.CPUState{
		GOOS:                "test",
		AffinitySupported:   true,
		AffinityAllowedList: "0",
		CPUOnlineList:       "0",
	})
	dir := t.TempDir()
	out := filepath.Join(dir, "bench.txt")
	metaPath := filepath.Join(dir, "bench.json")
	if err := os.WriteFile(out, []byte("BenchmarkX-1 1 2 ns/op 0 B/op 0 allocs/op\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	goBin, goSHA := stubGobenchPublishGoTool(t)
	cfg := config{
		Pkg:              ".",
		Bench:            "^BenchmarkX$",
		GoBin:            goBin,
		GoSHA256:         goSHA,
		OutputPath:       out,
		MetadataPath:     metaPath,
		EnvironmentNotes: "idle",
		CPUAffinity:      "none",
		PowerMode:        "high power",
		ThermalState:     "cool start",
		FrequencyPolicy:  "automatic",
		BackgroundLoad:   "idle machine",
		GoMaxProcs:       1,
		CPU:              "1",
		Count:            5,
		BenchTime:        "500ms",
		GOGC:             "off",
		Publish:          true,
		BenchMem:         true,
		ExplicitFlags:    gobenchPublishExplicitFlags(),
	}
	meta := buildMetadata(cfg, gitMetadata{Commit: "abc"}, []string{"go", "test"}, "ok", "")
	bytes, hash, err := fileInfoAndSHA256(out)
	if err != nil {
		t.Fatal(err)
	}
	meta.Output.Bytes = bytes
	meta.Output.SHA256 = hash
	if err := writeJSONCreatingParents(metaPath, meta); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatal(err)
	}
	var got metadata
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Output.Bytes == 0 || got.Output.SHA256 == "" || got.Config.Count != 5 ||
		got.Config.BenchmarkFunction != "BenchmarkX" || !got.Config.SubbenchmarkRowsAllowed ||
		got.Config.GoMaxProcs != 1 || got.Environment.GOGC != "off" ||
		got.Go.SIMDTier == "" || len(got.Go.Env) == 0 ||
		got.Go.ToolPath != goBin || got.Go.ToolSHA256 != goSHA || !got.Go.ToolSHA256Verified ||
		got.Environment.Notes != "idle" ||
		got.Environment.CPUAffinity != "none" ||
		got.Environment.PowerMode != "high power" ||
		got.Environment.ThermalState != "cool start" ||
		got.Environment.FrequencyPolicy != "automatic" ||
		got.Environment.BackgroundLoad != "idle machine" ||
		got.Environment.ObservedCPUState.AffinityAllowedList != "0" {
		t.Fatalf("metadata=%+v", got)
	}
}

func clearGobenchPublishGoEnv(t *testing.T) {
	t.Helper()
	for _, name := range benchenv.PublishAmbientGoEnvVars() {
		t.Setenv(name, "")
	}
}

func stubGobenchPublishCPUState(t *testing.T, state benchenv.CPUState) {
	t.Helper()
	old := observeBenchmarkCPUState
	observeBenchmarkCPUState = func() benchenv.CPUState { return state }
	t.Cleanup(func() { observeBenchmarkCPUState = old })
}

func stubGobenchPublishGoTool(t *testing.T) (string, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "go")
	if err := os.WriteFile(path, []byte("go tool fixture"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, hash, err := fileInfoAndSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	return path, hash
}

func gobenchPublishExplicitFlags() map[string]bool {
	return map[string]bool{
		"pkg":               true,
		"bench":             true,
		"go-bin":            true,
		"go-sha256":         true,
		"out":               true,
		"metadata-json":     true,
		"environment-notes": true,
		"cpu-affinity":      true,
		"power-mode":        true,
		"thermal-state":     true,
		"frequency-policy":  true,
		"background-load":   true,
		"gomaxprocs":        true,
		"cpu":               true,
		"count":             true,
		"benchtime":         true,
		"benchmem":          true,
		"gogc":              true,
	}
}
