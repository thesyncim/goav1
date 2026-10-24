package main

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunSVCInfoDecodesL2T1 is the bundled-vector smoke test for the
// `-svc-info` mode against the AV1 SVC `av1-1-b8-22-svc-L2T1.ivf`
// conformance vector. The L2T1 layout publishes two spatial layers at
// distinct resolutions (640x360 base, 1280x720 enhancement). The
// svc-info pass parses the sequence header and the per-(T,S) OBU
// fan-out without touching the pixel decoder: it verifies parsing
// infrastructure and confirms the CLI correctly reports the two
// operating points and the (T=0,S=0)+(T=0,S=1) layer composition that
// an L2T1 stream advertises.
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

// TestRunDecodesSVCVectorsMatchLibaom decodes the bundled libaom SVC
// vectors through the CLI's single-pool driver and checks every emitted
// frame against the libaom per-frame MD5 sidecar (aomdec and dav1d
// --alllayers 0 agree with it): one frame per temporal unit at the
// highest spatial layer. L2T1 and L2T2 mix a 640x360 base layer with a
// 1280x720 enhancement layer that predicts from it through a scaled
// reference; L2T2 and L1T2 add a non-reference upper temporal layer
// whose shown surfaces must go back to the pool. A pixel-free size
// check here once hid that the driver emitted frames without running
// the post-filters, so the gate is per-frame MD5 at 1 and 8 workers.
func TestRunDecodesSVCVectorsMatchLibaom(t *testing.T) {
	// Per-frame MD5s from the libaom .md5 sidecars (aomdec and dav1d agree).
	// CI fetches the vectors but not the sidecars, so they are pinned here.
	for _, tc := range []struct {
		name          string
		bytesPerFrame int
		want          []string
	}{
		{"L1T2", 640 * 360 * 3 / 2, []string{
			"f3874ea6c93fa8e345532e7273a00c78",
			"9fa72a857e695607338354107695176a",
			"a2eda89c577c97cc342d0ebc79f223c2",
			"abe1f2c18b65422df79f7e3fa3e62f2c",
			"d349496e56e18763cc43139d16822329",
			"4edcc8be524cddff3787f87af8b6a40c",
			"450f1a8a26c719f26fe74287974f64b0",
			"a3ac8d3056e8e1deee9c02cccdf2a96a",
		}},
		{"L2T1", 1280 * 720 * 3 / 2, []string{
			"d059f1dbc0f7dcf26a8f96e3175a83cf",
			"b9edb56e9a351536ca9748bd45a355f4",
			"176447941753f50218d4e2ce2ba266cf",
			"7d9836bbc0e98d378961e1f02ffa86ac",
			"706f8beec0e909caab7457fb99d31840",
			"1197272fbcaeb550a7951dbe1cf33047",
			"fb4655c6f9f5bdc86ec634f5da8368df",
			"57e548bcf058a36b3841da3bc3a9080b",
		}},
		{"L2T2", 1280 * 720 * 3 / 2, []string{
			"d059f1dbc0f7dcf26a8f96e3175a83cf",
			"dff9dbe5ad9f9638abfa08440c9cb038",
			"6fe06bb57ca517b0a1da8317eb6c7954",
			"96c732514b66f80d4d6fbf0225684670",
			"af9840249331b2a5ff638bc813bbfa39",
			"b925ea23d949e76008375415ec1ff184",
			"1989756cd5d51f60c358b37329fa9d45",
			"339d0793ec430b0eb35be20918bc9ade",
		}},
	} {
		ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-22-svc-"+tc.name+".ivf")
		if _, err := os.Stat(ivfPath); err != nil {
			t.Fatalf("bundled IVF missing: %v", err)
		}
		want := tc.want
		for _, workers := range []int{1, 8} {
			t.Run(fmt.Sprintf("%s/workers-%d", tc.name, workers), func(t *testing.T) {
				outPath := filepath.Join(t.TempDir(), "out.yuv")
				var stderr bytes.Buffer
				err := run([]string{"-o", outPath, "-quiet", "-workers", fmt.Sprint(workers), ivfPath}, io.Discard, &stderr)
				if err != nil {
					t.Fatalf("run %s decode err=%v\nstderr=%s", tc.name, err, stderr.String())
				}
				got, err := os.ReadFile(outPath)
				if err != nil {
					t.Fatalf("read output: %v", err)
				}
				if len(got) != len(want)*tc.bytesPerFrame {
					t.Fatalf("output size=%d want %d (%d frames of %d bytes)",
						len(got), len(want)*tc.bytesPerFrame, len(want), tc.bytesPerFrame)
				}
				for i, w := range want {
					sum := md5.Sum(got[i*tc.bytesPerFrame : (i+1)*tc.bytesPerFrame])
					if g := hex.EncodeToString(sum[:]); g != w {
						t.Fatalf("frame %d md5=%s want %s (libaom sidecar)", i, g, w)
					}
				}
			})
		}
	}
}

// TestRunDecodesL1T2AllFrames exercises the dump_svc pixel decode path
// against the bundled L1T2 SVC vector. L1T2 shares a single 640x360
// frame size across both temporal layers, so the single-pool driver
// decodes every temporal unit. The decode previously aborted at frame 1
// because the per-frame CDEF index-map scratch was validated against the
// prior frame's decoded cdef_idx values before being cleared: a keyframe
// with more cdef_bits left an index that exceeded the next frame's smaller
// cdef_bits limit. With that clear-before-bind fixed the clip now decodes
// all eight temporal units. The smoke test therefore asserts the full
// 8-frame I420 output (8 * 640 * 360 * 1.5 bytes) plus the expected
// diagnostic annotations.
func TestRunDecodesL1T2AllFrames(t *testing.T) {
	const (
		width         = 640
		height        = 360
		bytesPerFrame = width * height * 3 / 2 // I420 4:2:0 8-bit
		frames        = 8
	)
	ivfPath := repoRelativePath(t, "internal", "av1", "testdata", "libaom", "av1-1-b8-22-svc-L1T2.ivf")
	if _, err := os.Stat(ivfPath); err != nil {
		t.Fatalf("bundled IVF missing: %v", err)
	}

	outDir := t.TempDir()
	outPath := filepath.Join(outDir, "out.yuv")

	var stderr bytes.Buffer
	err := run([]string{"-o", outPath, "-quiet", ivfPath}, io.Discard, &stderr)
	if err != nil {
		t.Fatalf("run L1T2 decode err=%v\nstderr=%s", err, stderr.String())
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
	if got := info.Size(); got != frames*bytesPerFrame {
		t.Fatalf("L1T2 decode output size=%d want %d (%d 640x360 I420 frames)",
			got, frames*bytesPerFrame, frames)
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
