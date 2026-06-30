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
	cfg := config{
		Pkg:              "./internal/av1/tile",
		Bench:            "^BenchmarkCoeffCulLevel$",
		OutputPath:       "/tmp/bench.txt",
		MetadataPath:     "/tmp/bench.json",
		EnvironmentNotes: "fixed power mode, idle machine",
		GoMaxProcs:       1,
		CPU:              "1",
		Count:            5,
		BenchTime:        "500ms",
		GOGC:             "off",
		Publish:          true,
		BenchMem:         true,
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
		GoMaxProcs:       1,
		CPU:              "1",
		Count:            5,
		BenchTime:        "500ms",
		GOGC:             "off",
		Publish:          true,
		BenchMem:         true,
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
		got.Environment.Notes != "idle" {
		t.Fatalf("metadata=%+v", got)
	}
}
