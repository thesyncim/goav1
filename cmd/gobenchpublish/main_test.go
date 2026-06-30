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
)

func TestValidatePublishConfigRequiresStrictControls(t *testing.T) {
	t.Setenv("GOFLAGS", "")
	t.Setenv("GOMEMLIMIT", "")
	t.Setenv("GODEBUG", "")

	cfg := config{
		Pkg:              "./internal/av1/tile",
		Bench:            "^BenchmarkCoeffCulLevel$",
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
}

func TestValidatePublishConfigRejectsHiddenGoEnvironment(t *testing.T) {
	cfg := config{
		Pkg:              "./internal/av1/tile",
		Bench:            "^BenchmarkCoeffCulLevel$",
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
	dir := t.TempDir()
	out := filepath.Join(dir, "bench.txt")
	metaPath := filepath.Join(dir, "bench.json")
	if err := os.WriteFile(out, []byte("BenchmarkX-1 1 2 ns/op 0 B/op 0 allocs/op\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := config{
		Pkg:              ".",
		Bench:            "BenchmarkX",
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
		got.Config.GoMaxProcs != 1 || got.Environment.GOGC != "off" ||
		got.Environment.Notes != "idle" ||
		got.Environment.CPUAffinity != "none" ||
		got.Environment.PowerMode != "high power" ||
		got.Environment.ThermalState != "cool start" ||
		got.Environment.FrequencyPolicy != "automatic" ||
		got.Environment.BackgroundLoad != "idle machine" {
		t.Fatalf("metadata=%+v", got)
	}
}

func gobenchPublishExplicitFlags() map[string]bool {
	return map[string]bool{
		"pkg":               true,
		"bench":             true,
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
