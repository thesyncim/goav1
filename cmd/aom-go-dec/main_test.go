package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunDecodesBundledIVF is the end-to-end smoke test for the aom-go-dec
// CLI. It drives the same exported run() entry point the binary uses on a
// bundled libaom IVF test vector and verifies the raw YUV byte count matches
// the expected I420 layout (Y + U + V at 4:2:0 subsampling for two 352x288
// 8-bit frames).
func TestRunDecodesBundledIVF(t *testing.T) {
	const (
		width      = 352
		height     = 288
		frameCount = 2
		// I420: one byte per Y sample plus 0.5 bytes per chroma sample across
		// the U and V planes. Equivalent to width*height*3/2 per frame.
		bytesPerFrame = width * height * 3 / 2
		expectedBytes = bytesPerFrame * frameCount
	)

	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-00-quantizer-00.ivf")
	if _, err := os.Stat(ivfPath); err != nil {
		t.Fatalf("bundled IVF missing: %v", err)
	}

	outDir := t.TempDir()
	outPath := filepath.Join(outDir, "out.yuv")

	var stderr bytes.Buffer
	if err := run([]string{"-o", outPath, "-quiet", ivfPath}, io.Discard, &stderr); err != nil {
		t.Fatalf("run err=%v\nstderr=%s", err, stderr.String())
	}

	info, err := os.Stat(outPath)
	if err != nil {
		t.Fatalf("stat output: %v", err)
	}
	if got := info.Size(); got != int64(expectedBytes) {
		t.Fatalf("output size=%d want %d (2 * 352*288*1.5 = %d)",
			got, expectedBytes, expectedBytes)
	}

	stderrText := stderr.String()
	if !strings.Contains(stderrText, "352x288") {
		t.Errorf("stderr missing 352x288 header line: %q", stderrText)
	}
	if !strings.Contains(stderrText, "decoded 2 frames") {
		t.Errorf("stderr missing decoded summary line: %q", stderrText)
	}
}

// TestRunDecodesSuperResProfileClip keeps the CLI on the same super-res-capable
// public path as av1.NewDecoderFromIVF. The clip is 4:4:4 8-bit, so each
// displayed frame writes Y, U, and V at full 160x128 resolution.
func TestRunDecodesSuperResProfileClip(t *testing.T) {
	const (
		width         = 160
		height        = 128
		frameCount    = 8
		bytesPerFrame = width * height * 3
		expectedBytes = bytesPerFrame * frameCount
	)

	ivfPath := repoRelativePath(t, "internal", "av1", "testvector", "testdata",
		"profiles", "profile1-444-8bit-superres-inter-160x128.ivf")

	outDir := t.TempDir()
	outPath := filepath.Join(outDir, "out.yuv")

	var stderr bytes.Buffer
	if err := run([]string{"-o", outPath, "-quiet", ivfPath}, io.Discard, &stderr); err != nil {
		t.Fatalf("run err=%v\nstderr=%s", err, stderr.String())
	}

	info, err := os.Stat(outPath)
	if err != nil {
		t.Fatalf("stat output: %v", err)
	}
	if got := info.Size(); got != int64(expectedBytes) {
		t.Fatalf("output size=%d want %d (8 * 160*128*3 = %d)",
			got, expectedBytes, expectedBytes)
	}
	if !strings.Contains(stderr.String(), "decoded 8 frames") {
		t.Errorf("stderr missing decoded summary line: %q", stderr.String())
	}
}

// TestRunWritesStdoutWhenNoOutputFlag confirms the CLI streams YUV to the
// writer supplied as stdout when -o is omitted, leaving stderr free for
// diagnostics.
func TestRunWritesStdoutWhenNoOutputFlag(t *testing.T) {
	const expectedBytes = 352 * 288 * 3 / 2 * 2

	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-00-quantizer-00.ivf")

	var stdout, stderr bytes.Buffer
	if err := run([]string{"-quiet", ivfPath}, &stdout, &stderr); err != nil {
		t.Fatalf("run err=%v\nstderr=%s", err, stderr.String())
	}
	if got := stdout.Len(); got != expectedBytes {
		t.Fatalf("stdout bytes=%d want %d", got, expectedBytes)
	}
}

// TestRunMissingInputReturnsError exercises the argv validation branch so
// the CLI's non-zero exit path stays covered.
func TestRunMissingInputReturnsError(t *testing.T) {
	err := run([]string{}, io.Discard, io.Discard)
	if err == nil {
		t.Fatal("expected error for missing positional argument, got nil")
	}
}

// TestRunUnreadableFileReturnsError confirms file IO errors surface as a
// non-nil error from run() so the binary exits non-zero.
func TestRunUnreadableFileReturnsError(t *testing.T) {
	bogus := filepath.Join(t.TempDir(), "does-not-exist.ivf")
	err := run([]string{bogus}, io.Discard, io.Discard)
	if err == nil {
		t.Fatal("expected error for missing input file, got nil")
	}
}

// repoRelativePath joins parts onto the repository root regardless of the
// directory the test is invoked from. `go test ./cmd/aom-go-dec` runs with
// the cwd set to the package directory, so the testdata path has to be
// resolved relative to two levels up.
func repoRelativePath(t *testing.T, parts ...string) string {
	t.Helper()
	// The test file lives at <repo>/cmd/aom-go-dec/main_test.go. Walk two
	// directories up to anchor on the repo root, then append parts.
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	repoRoot := filepath.Clean(filepath.Join(cwd, "..", ".."))
	return filepath.Join(append([]string{repoRoot}, parts...)...)
}
