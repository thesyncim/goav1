package main

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"strconv"
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

// TestRunDecodesPostFilteredProfileClips keeps the CLI on the same public
// post-filter path as av1.NewDecoderFromIVF. These small 4:4:4 8-bit clips
// exercise CDEF/restoration, film grain, and inter super-res, then compare
// the written YUV frames against libaom MD5 goldens.
func TestRunDecodesPostFilteredProfileClips(t *testing.T) {
	tests := []struct {
		name          string
		file          string
		width, height int
		frameMD5Hex   []string
	}{
		{
			name:   "cdef-restoration",
			file:   "profile1-444-8bit-cdef-restoration-160x128.ivf",
			width:  160,
			height: 128,
			frameMD5Hex: []string{
				"c7001ffcb04bce20728f246f5494e58a",
				"f1a3ce1e10e2e84e1e61c46e735201ad",
				"1aa1aa327a503776ff58af351198037a",
				"7cfb09c54857fc6dd3994ef09e502fae",
			},
		},
		{
			name:   "film-grain",
			file:   "profile1-444-8bit-filmgrain-96x96.ivf",
			width:  96,
			height: 96,
			frameMD5Hex: []string{
				"db1f78af3fbd46eec482cfcc02c59157",
				"f0abb8e97190db604aba5ffae2dd6b47",
				"706e3f8b36da82f717a1496cbbddf0c9",
			},
		},
		{
			name:   "superres-inter",
			file:   "profile1-444-8bit-superres-inter-160x128.ivf",
			width:  160,
			height: 128,
			frameMD5Hex: []string{
				"7057484f2d2048053692a9bc41ce197b",
				"6b136fd484722654825fe6abc2e1773a",
				"cddb59775cd003ed4624b782fce4eca4",
				"32e3240e78dd8f53e7b673c9120fbe86",
				"5054adcd48bad5490911bb07248fea75",
				"150acaa0e5a682d454d7538b15cc4f2f",
				"5d0ab2de3f3e9ce25036226b11375e49",
				"78c4f82e984b44a272930333264bc764",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ivfPath := repoRelativePath(t, "internal", "av1", "testvector", "testdata",
				"profiles", tc.file)
			outPath := filepath.Join(t.TempDir(), "out.yuv")

			var stderr bytes.Buffer
			if err := run([]string{"-o", outPath, "-quiet", ivfPath}, io.Discard, &stderr); err != nil {
				t.Fatalf("run err=%v\nstderr=%s", err, stderr.String())
			}

			assertYUVFrameMD5s(t, outPath, tc.width*tc.height*3, tc.frameMD5Hex)
			wantSummary := "decoded " + strconv.Itoa(len(tc.frameMD5Hex)) + " frames"
			if !strings.Contains(stderr.String(), wantSummary) {
				t.Errorf("stderr missing decoded summary line %q: %q", wantSummary, stderr.String())
			}
		})
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

func assertYUVFrameMD5s(t *testing.T, path string, bytesPerFrame int, want []string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	wantBytes := bytesPerFrame * len(want)
	if len(data) != wantBytes {
		t.Fatalf("output size=%d want %d (%d frames * %d bytes)",
			len(data), wantBytes, len(want), bytesPerFrame)
	}
	for i, expected := range want {
		sum := md5.Sum(data[i*bytesPerFrame : (i+1)*bytesPerFrame])
		if got := hex.EncodeToString(sum[:]); got != expected {
			t.Fatalf("frame %d md5=%s want %s", i, got, expected)
		}
	}
}
