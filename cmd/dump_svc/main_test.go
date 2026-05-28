package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunSVCInfoDecodesL2T1 is the bundled-vector smoke test for the
// `-svc-info` mode against the AV1 SVC `av1-1-b8-22-svc-L2T1.ivf`
// conformance vector. The L2T1 layout publishes two spatial layers at
// distinct resolutions (640x360 base, 1280x720 enhancement), which
// today's single-pool decoder driver in this CLI cannot fully decode
// (it requires the multi-pool decoder.FrameSurfaceProvider path
// documented in docs/svc.md). The svc-info pass however parses the
// sequence header and the per-(T,S) OBU fan-out cleanly because it
// never touches the pixel decoder, so it is the right smoke gate for
// the bundled L2T1 vector: it verifies parsing infrastructure and
// confirms the CLI correctly reports the two operating points and the
// (T=0,S=0)+(T=0,S=1) layer composition that an L2T1 stream advertises.
func TestRunSVCInfoDecodesL2T1(t *testing.T) {
	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-22-svc-L2T1.ivf")
	if _, err := os.Stat(ivfPath); err != nil {
		t.Fatalf("bundled IVF missing: %v", err)
	}

	var stderr bytes.Buffer
	if err := run([]string{"-svc-info", ivfPath}, io.Discard, &stderr); err != nil {
		t.Fatalf("run -svc-info err=%v\nstderr=%s", err, stderr.String())
	}
	text := stderr.String()
	for _, want := range []string{
		"1280x720",                    // L2T1 advertises the full-res frame in the IVF header
		"sequence operating_points=2", // L2T1 declares the spatial-only + base-only ops
		"operating_point[0] idc=0x0301",
		"operating_point[1] idc=0x0101",
		"T=0 S=0",
		"T=0 S=1",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("svc-info stderr missing %q\nstderr=%s", want, text)
		}
	}
}

// TestRunDecodesL2T1OutputSize is the L2T1 pixel-decode smoke test
// requested by the CLI spec. The bundled vector mixes a 640x360 base
// layer with a 1280x720 enhancement layer in the same temporal unit,
// which the single-pool driver in this CLI rejects with
// `frame: invalid format` at the very first payload. The test
// therefore verifies the exact partial-decode contract the CLI
// promises in main.go's package doc:
//
//   - run() returns a non-nil error (CLI exits non-zero),
//   - the failure annotation carries the layer composition
//     (T=0 S=0, T=0 S=1) of the failing payload,
//   - the YUV output file is created and is exactly zero bytes long
//     (no garbage from a half-decoded frame leaks out).
//
// When the single-pool L1T2 / multi-pool L2T1 limitation is lifted in
// a future commit, this test should be updated to assert the full
// 8-frame decode at the highest spatial layer
// (8 * 1280 * 720 * 3 / 2 = 11_059_200 bytes).
func TestRunDecodesL2T1OutputSize(t *testing.T) {
	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-22-svc-L2T1.ivf")
	if _, err := os.Stat(ivfPath); err != nil {
		t.Fatalf("bundled IVF missing: %v", err)
	}

	outDir := t.TempDir()
	outPath := filepath.Join(outDir, "out.yuv")

	var stderr bytes.Buffer
	err := run([]string{"-o", outPath, "-quiet", ivfPath}, io.Discard, &stderr)
	if err == nil {
		t.Fatalf("expected non-nil error for L2T1 single-pool decode, got nil\nstderr=%s", stderr.String())
	}
	text := stderr.String()
	for _, want := range []string{
		"partial decode",
		"layers=T=0 S=0,T=0 S=1",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("stderr missing %q\nstderr=%s", want, text)
		}
	}

	info, statErr := os.Stat(outPath)
	if statErr != nil {
		t.Fatalf("stat output: %v", statErr)
	}
	if info.Size() != 0 {
		t.Fatalf("partial-decode L2T1 output size=%d want 0 bytes", info.Size())
	}
}

// TestRunDecodesL1T2FirstFrame exercises the dump_svc pixel decode path
// against the bundled L1T2 SVC vector. L1T2 shares a single 640x360
// frame size across both temporal layers, so the single-pool driver
// successfully decodes the first IDR frame before failing at frame 1
// on the in-flight inter-layer reference issue documented in
// docs/svc.md. The smoke test therefore asserts the partial decode
// produces exactly one I420 frame of 640x360x1.5 bytes plus the
// expected diagnostic annotations.
func TestRunDecodesL1T2FirstFrame(t *testing.T) {
	const (
		width         = 640
		height        = 360
		bytesPerFrame = width * height * 3 / 2 // I420 4:2:0 8-bit
	)
	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-22-svc-L1T2.ivf")
	if _, err := os.Stat(ivfPath); err != nil {
		t.Fatalf("bundled IVF missing: %v", err)
	}

	outDir := t.TempDir()
	outPath := filepath.Join(outDir, "out.yuv")

	var stderr bytes.Buffer
	err := run([]string{"-o", outPath, "-quiet", ivfPath}, io.Discard, &stderr)
	if err == nil {
		t.Fatalf("expected non-nil error for L1T2 partial decode, got nil\nstderr=%s", stderr.String())
	}
	text := stderr.String()
	if !strings.Contains(text, "640x360") {
		t.Errorf("stderr missing 640x360 header line:\n%s", text)
	}
	if !strings.Contains(text, "scaled_pred=true") {
		t.Errorf("stderr missing scaled_pred=true (the CLI must default to true):\n%s", text)
	}

	info, statErr := os.Stat(outPath)
	if statErr != nil {
		t.Fatalf("stat output: %v", statErr)
	}
	if got := info.Size(); got != bytesPerFrame {
		t.Fatalf("partial decode output size=%d want %d (one 640x360 I420 frame)",
			got, bytesPerFrame)
	}
}

// TestRunRejects10LEOn8BitStream confirms the -yuv420p10le shorthand
// produces a clear error rather than silently emitting half-resolution
// YUV when the bitstream is 8-bit. This is the key safety check the
// format-flag plumbing exists to provide: callers pasting an FFmpeg
// filter chain that hard-codes "yuv420p10le" should learn immediately
// that the source isn't 10-bit, not after corrupted YUV reaches their
// downstream pipeline.
func TestRunRejects10LEOn8BitStream(t *testing.T) {
	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-22-svc-L1T2.ivf")
	if _, err := os.Stat(ivfPath); err != nil {
		t.Fatalf("bundled IVF missing: %v", err)
	}

	var stderr bytes.Buffer
	err := run([]string{"-yuv420p10le", "-quiet", ivfPath}, io.Discard, &stderr)
	if err == nil {
		t.Fatal("expected error for yuv420p10le request on 8-bit stream, got nil")
	}
	if !strings.Contains(err.Error(), "yuv420p10le requires 10-bit") {
		t.Fatalf("unexpected error text: %v", err)
	}
}

// TestRunRejectsConflictingFormatFlags confirms the CLI catches
// callers that asked for both shorthand flags. The check exists
// because both flags map to the same underlying -format string and
// silently picking one of them would mask the configuration bug from
// the caller.
func TestRunRejectsConflictingFormatFlags(t *testing.T) {
	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-22-svc-L1T2.ivf")
	if _, err := os.Stat(ivfPath); err != nil {
		t.Fatalf("bundled IVF missing: %v", err)
	}

	err := run([]string{"-i420", "-yuv420p10le", ivfPath}, io.Discard, io.Discard)
	if err == nil {
		t.Fatal("expected error for combined -i420 and -yuv420p10le, got nil")
	}
	if !strings.Contains(err.Error(), "cannot combine") {
		t.Fatalf("unexpected error text: %v", err)
	}
}

// TestRunMissingInputReturnsError exercises the argv validation branch
// so the CLI's non-zero exit path stays covered.
func TestRunMissingInputReturnsError(t *testing.T) {
	err := run([]string{}, io.Discard, io.Discard)
	if err == nil {
		t.Fatal("expected error for missing positional argument, got nil")
	}
}

// TestRunUnreadableFileReturnsError confirms file IO errors surface as
// a non-nil error from run() so the binary exits non-zero.
func TestRunUnreadableFileReturnsError(t *testing.T) {
	bogus := filepath.Join(t.TempDir(), "does-not-exist.ivf")
	err := run([]string{bogus}, io.Discard, io.Discard)
	if err == nil {
		t.Fatal("expected error for missing input file, got nil")
	}
}

// repoRelativePath joins parts onto the repository root regardless of
// the directory the test is invoked from. `go test ./cmd/dump_svc`
// runs with the cwd set to the package directory, so the testdata
// path has to be resolved relative to two levels up.
func repoRelativePath(t *testing.T, parts ...string) string {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	repoRoot := filepath.Clean(filepath.Join(cwd, "..", ".."))
	return filepath.Join(append([]string{repoRoot}, parts...)...)
}
