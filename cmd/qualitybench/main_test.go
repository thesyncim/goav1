package main

import (
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	goav1 "github.com/thesyncim/goav1"
	cpufeatures "github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/benchenv"
)

func TestParsePositiveList(t *testing.T) {
	got, err := parsePositiveList("100, 200,300")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 100 || got[1] != 200 || got[2] != 300 {
		t.Fatalf("got %v", got)
	}
	if _, err := parsePositiveList("100,0"); err == nil {
		t.Fatal("zero bitrate accepted")
	}
}

func TestParseMetricList(t *testing.T) {
	got, err := parseMetricList("psnr_avg, ssim_all, xpsnr_y, vmaf,psnr")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"psnr", "ssim", "xpsnr", "vmaf"}
	if len(got) != len(want) {
		t.Fatalf("metrics=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("metrics=%v want %v", got, want)
		}
	}
	if _, err := parseMetricList("butteraugli"); err == nil {
		t.Fatal("unknown metric accepted")
	}
}

func TestCanonicalSVTASMName(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{in: "neon", want: "neon"},
		{in: "dotprod", want: "neon_dotprod"},
		{in: "i8mm", want: "neon_i8mm"},
		{in: "SVE2", want: "sve2"},
		{in: "max", want: "max"},
	}
	for _, tc := range tests {
		got, ok := canonicalSVTASMName(tc.in)
		if !ok || got != tc.want {
			t.Fatalf("canonicalSVTASMName(%q)=%q,%v want %q,true", tc.in, got, ok, tc.want)
		}
	}
	if got, ok := canonicalSVTASMName("avx2"); ok {
		t.Fatalf("canonicalSVTASMName(avx2)=%q,true want invalid", got)
	}
}

func TestParseEncoderListCanonicalizesAliases(t *testing.T) {
	got := parseEncoderList("goav1, libaom, svt, custom")
	want := []string{"goav1", "aomenc", "svt-av1", "custom"}
	if len(got) != len(want) {
		t.Fatalf("encoders=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("encoders=%v want %v", got, want)
		}
	}
}

func TestParseRequiredEncoderList(t *testing.T) {
	selected := []string{"goav1", "aomenc", "svt-av1"}
	got, err := parseRequiredEncoderList("libaom, svt", selected)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"aomenc", "svt-av1"}
	if len(got) != len(want) {
		t.Fatalf("required=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("required=%v want %v", got, want)
		}
	}
	if _, err := parseRequiredEncoderList("rav1e", selected); err == nil {
		t.Fatal("unknown required encoder accepted")
	}
	if _, err := parseRequiredEncoderList("aomenc", []string{"goav1"}); err == nil {
		t.Fatal("unselected required encoder accepted")
	}
}

func TestParseRequiredEncoderListAll(t *testing.T) {
	got, err := parseRequiredEncoderList("all", []string{"goav1", "libaom", "svt", "svt-av1"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"goav1", "aomenc", "svt-av1"}
	if len(got) != len(want) {
		t.Fatalf("required=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("required=%v want %v", got, want)
		}
	}
	if _, err := parseRequiredEncoderList("all", []string{"goav1", "custom"}); err == nil {
		t.Fatal("all accepted an unknown selected encoder")
	}
}

func TestValidateRequiredMetrics(t *testing.T) {
	filters := map[string]bool{
		"psnr":    true,
		"ssim":    true,
		"xpsnr":   true,
		"libvmaf": true,
	}
	if err := validateRequiredMetrics(filters, []string{"psnr", "vmaf"}); err != nil {
		t.Fatal(err)
	}
	delete(filters, "libvmaf")
	if err := validateRequiredMetrics(filters, []string{"vmaf"}); err == nil {
		t.Fatal("missing libvmaf accepted")
	}
}

func TestValidateFrameMetricFilters(t *testing.T) {
	if err := validateFrameMetricFilters(map[string]bool{"psnr": true}); err != nil {
		t.Fatalf("psnr frame metric filter failed: %v", err)
	}
	if err := validateFrameMetricFilters(map[string]bool{"ssim": true}); err != nil {
		t.Fatalf("ssim frame metric filter failed: %v", err)
	}
	if err := validateFrameMetricFilters(map[string]bool{"xpsnr": true}); err == nil ||
		!strings.Contains(err.Error(), "psnr or ssim") {
		t.Fatalf("missing frame metric filters error=%v", err)
	}
}

func TestParseFFmpegAV1Decoders(t *testing.T) {
	raw := []byte(strings.Join([]string{
		" V..... libdav1d             dav1d AV1 decoder by VideoLAN (codec av1)",
		" V....D av1                  Alliance for Open Media AV1",
		" A....D wmav1                Windows Media Audio 1",
	}, "\n"))
	got := parseFFmpegAV1Decoders(raw)
	if !got["libdav1d"] || !got["av1"] || got["wmav1"] {
		t.Fatalf("av1 decoders=%v", got)
	}
	if err := validateFFmpegAV1Decoder("libdav1d", got); err != nil {
		t.Fatalf("valid decoder failed: %v", err)
	}
	if err := validateFFmpegAV1Decoder("wmav1", got); err == nil ||
		!strings.Contains(err.Error(), "available AV1 decoders") {
		t.Fatalf("invalid decoder error=%v", err)
	}
}

func TestRequiredEncoderError(t *testing.T) {
	required := requiredEncoderSet([]string{"aomenc"})
	if err := requiredEncoderError(required, "clip", encodeResult{
		encoder:   "aomenc",
		targetBPS: 100000,
		status:    "skipped",
		errText:   "aomenc not found",
	}, 100000); err == nil {
		t.Fatal("required skipped encoder accepted")
	}
	if err := requiredEncoderError(required, "clip", encodeResult{
		encoder: "svt-av1",
		status:  "skipped",
	}, 100000); err != nil {
		t.Fatalf("non-required skipped encoder failed: %v", err)
	}
	if err := requiredEncoderError(required, "clip", encodeResult{
		encoder: "aomenc",
		status:  "ok",
	}, 100000); err != nil {
		t.Fatalf("required ok encoder failed: %v", err)
	}
}

func TestValidateRequiredEncoderTools(t *testing.T) {
	cfg := benchConfig{requiredEncoders: []string{"goav1", "aomenc"}}
	if err := validateRequiredEncoderTools(cfg, func(name string) (string, error) {
		if name != "aomenc" {
			t.Fatalf("unexpected tool lookup %q", name)
		}
		return "/bin/aomenc", nil
	}); err != nil {
		t.Fatalf("valid tools failed: %v", err)
	}

	cfg.requiredEncoders = []string{"svt-av1"}
	if err := validateRequiredEncoderTools(cfg, func(name string) (string, error) {
		return "", os.ErrNotExist
	}); err == nil || !strings.Contains(err.Error(), "required encoder svt-av1 unavailable") {
		t.Fatalf("missing required SVT error=%v", err)
	}

	cfg.requiredEncoders = nil
	if err := validateRequiredEncoderTools(cfg, func(name string) (string, error) {
		t.Fatalf("optional encoder should not be looked up")
		return "", os.ErrNotExist
	}); err != nil {
		t.Fatalf("optional lookup failed: %v", err)
	}
}

func TestReadClipManifest(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "clips.csv")
	manifestCSV := strings.Join([]string{
		"clip,input,width,height,frames,fps,pix_fmt,bit_depth,chroma,sha256,source_id,source_url,source_license,category",
		"Talking Head,clips/head.yuv,1920,1080,120,60,i420,8,4:2:0,0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef,lab-head,https://example.invalid/head,CC-BY-4.0,talking-head",
		"Synthetic,,320,180,30,,,,,,,,,",
	}, "\n") + "\n"
	if err := os.WriteFile(manifest, []byte(manifestCSV), 0o644); err != nil {
		t.Fatal(err)
	}
	clips, err := readClipManifest(manifest, benchConfig{fps: 30})
	if err != nil {
		t.Fatal(err)
	}
	if len(clips) != 2 {
		t.Fatalf("clips=%d", len(clips))
	}
	if clips[0].Name != "Talking Head" ||
		clips[0].Input != filepath.Join(dir, "clips/head.yuv") ||
		clips[0].Width != 1920 || clips[0].Height != 1080 ||
		clips[0].Frames != 120 || clips[0].FPS != 60 || !clips[0].FPSPresent ||
		clips[0].PixFmt != "i420" || clips[0].BitDepth != 8 ||
		clips[0].Chroma != "4:2:0" || clips[0].SourceID != "lab-head" ||
		clips[0].SourceURL != "https://example.invalid/head" ||
		clips[0].SourceLicense != "CC-BY-4.0" ||
		clips[0].Category != "talking-head" {
		t.Fatalf("clip[0]=%+v", clips[0])
	}
	if clips[1].Name != "Synthetic" || clips[1].Input != "" || clips[1].FPS != 30 || clips[1].FPSPresent {
		t.Fatalf("clip[1]=%+v", clips[1])
	}
}

func TestReadClipManifestRejectsBadGeometry(t *testing.T) {
	manifest := filepath.Join(t.TempDir(), "clips.csv")
	if err := os.WriteFile(manifest, []byte("input,width,height,frames\nclip.yuv,15,64,10\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readClipManifest(manifest, benchConfig{fps: 30}); err == nil {
		t.Fatal("bad geometry accepted")
	}
}

func TestReadClipManifestRejectsDuplicateClipIdentity(t *testing.T) {
	tests := []struct {
		name    string
		rows    []string
		wantErr string
	}{
		{
			name: "duplicate-name",
			rows: []string{
				"same,first.yuv,64,64,2,30",
				"same,second.yuv,64,64,2,30",
			},
			wantErr: "duplicate clip name",
		},
		{
			name: "safe-dir-collision",
			rows: []string{
				"talking/head,first.yuv,64,64,2,30",
				"talking:head,second.yuv,64,64,2,30",
			},
			wantErr: "collides",
		},
		{
			name: "case-folded-safe-dir-collision",
			rows: []string{
				"ClipA,first.yuv,64,64,2,30",
				"clipa,second.yuv,64,64,2,30",
			},
			wantErr: "collides",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			manifest := filepath.Join(t.TempDir(), "clips.csv")
			body := "clip,input,width,height,frames,fps\n" + strings.Join(tc.rows, "\n") + "\n"
			if err := os.WriteFile(manifest, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := readClipManifest(manifest, benchConfig{fps: 30}); err == nil ||
				!strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("manifest error=%v want %q", err, tc.wantErr)
			}
		})
	}
}

func TestValidateRequiredCorpus(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "first.yuv")
	second := filepath.Join(dir, "second.yuv")
	if err := os.WriteFile(first, []byte{0}, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(second, []byte{0}, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := benchConfig{
		manifestPath:  filepath.Join(dir, "clips.csv"),
		requireCorpus: true,
		minClips:      2,
		minSourceIDs:  2,
		minCategories: 2,
	}
	clips := []clipSpec{
		{Name: "first", Input: first, SourceID: "camera-a", Category: "talking-head"},
		{Name: "second", Input: second, SourceID: "camera-b", Category: "screen-content"},
	}
	if err := validateRequiredCorpus(cfg, clips); err != nil {
		t.Fatalf("valid corpus failed: %v", err)
	}

	sameSource := append([]clipSpec(nil), clips...)
	sameSource[1].SourceID = "Camera-A"
	if err := validateRequiredCorpus(cfg, sameSource); err == nil ||
		!strings.Contains(err.Error(), "source_id") {
		t.Fatalf("homogeneous source_id error=%v", err)
	}

	sameCategory := append([]clipSpec(nil), clips...)
	sameCategory[1].Category = "Talking-Head"
	if err := validateRequiredCorpus(cfg, sameCategory); err == nil ||
		!strings.Contains(err.Error(), "category") {
		t.Fatalf("homogeneous category error=%v", err)
	}

	cfg.minClips = 3
	if err := validateRequiredCorpus(cfg, clips); err == nil {
		t.Fatal("undersized corpus accepted")
	}

	cfg.minClips = 2
	cfg.manifestPath = ""
	if err := validateRequiredCorpus(cfg, clips); err == nil {
		t.Fatal("non-manifest corpus accepted")
	}

	cfg.manifestPath = filepath.Join(dir, "clips.csv")
	clips[1].Input = ""
	if err := validateRequiredCorpus(cfg, clips); err == nil {
		t.Fatal("synthetic corpus row accepted")
	}

	clips[1].Input = filepath.Join(dir, "missing.yuv")
	if err := validateRequiredCorpus(cfg, clips); err == nil {
		t.Fatal("missing corpus input accepted")
	}
}

func TestValidateClipManifestExactness(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "clip.yuv")
	if err := os.WriteFile(input, make([]byte, expectedRawI420Bytes(16, 16, 2)), 0o644); err != nil {
		t.Fatal(err)
	}
	hash, err := sha256File(input)
	if err != nil {
		t.Fatal(err)
	}
	valid := []clipSpec{{
		Name:          "clip",
		Input:         input,
		Width:         16,
		Height:        16,
		Frames:        2,
		FPS:           30,
		FPSPresent:    true,
		PixFmt:        "i420",
		BitDepth:      8,
		Chroma:        "4:2:0",
		DeclaredHash:  strings.ToUpper(hash),
		SourceID:      "lab-clip",
		SourceURL:     "https://example.invalid/clip",
		SourceLicense: "CC-BY-4.0",
		Category:      "talking-head",
	}}
	if err := validateClipManifestExactness(benchConfig{publish: true}, valid); err != nil {
		t.Fatalf("valid publish manifest failed: %v", err)
	}
	clone := func() []clipSpec {
		return append([]clipSpec(nil), valid...)
	}

	wrongHash := clone()
	wrongHash[0].DeclaredHash = strings.Repeat("0", 64)
	if err := validateClipManifestExactness(benchConfig{}, wrongHash); err == nil ||
		!strings.Contains(err.Error(), "does not match actual") {
		t.Fatalf("wrong hash error=%v", err)
	}

	missingFormat := clone()
	missingFormat[0].PixFmt = ""
	if err := validateClipManifestExactness(benchConfig{publish: true}, missingFormat); err == nil ||
		!strings.Contains(err.Error(), "pix_fmt=i420") {
		t.Fatalf("missing format error=%v", err)
	}

	missingFPS := clone()
	missingFPS[0].FPSPresent = false
	if err := validateClipManifestExactness(benchConfig{publish: true}, missingFPS); err == nil ||
		!strings.Contains(err.Error(), "manifest fps") {
		t.Fatalf("missing fps error=%v", err)
	}

	shortExternal := clone()
	if err := validateClipManifestExactness(benchConfig{publish: true, encoders: []string{"goav1", "aomenc"}}, shortExternal); err == nil ||
		!strings.Contains(err.Error(), "at least 2s") {
		t.Fatalf("short external clip error=%v", err)
	}

	longExternal := clone()
	longExternal[0].Frames = 60
	longExternal[0].FPS = 30
	if err := validateClipManifestExactness(benchConfig{publish: true, encoders: []string{"goav1", "aomenc"}}, longExternal); err != nil {
		t.Fatalf("long external clip failed: %v", err)
	}

	missingProvenance := clone()
	missingProvenance[0].Category = ""
	if err := validateClipManifestExactness(benchConfig{publish: true}, missingProvenance); err == nil ||
		!strings.Contains(err.Error(), "category") {
		t.Fatalf("missing provenance error=%v", err)
	}
}

func TestSafeClipDir(t *testing.T) {
	if got, want := safeClipDir("Talking Head/Low Light"), "Talking_Head_Low_Light"; got != want {
		t.Fatalf("safeClipDir=%q want %q", got, want)
	}
}

func TestIVFPayloadBytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.ivf")
	data := ivf.AppendFileHeader(nil, 16, 16, 30, 1, 2)
	data = ivf.AppendFrame(data, []byte{1, 2, 3}, 0)
	data = ivf.AppendFrame(data, []byte{4, 5, 6, 7, 8}, 1)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ivfPayloadBytes(path, 16, 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got != 8 {
		t.Fatalf("payload bytes=%d want 8", got)
	}
}

func TestIVFPayloadBytesRequiresExactContainer(t *testing.T) {
	dir := t.TempDir()
	valid := ivf.AppendFileHeader(nil, 16, 16, 30, 1, 1)
	valid = ivf.AppendFrame(valid, []byte{1, 2, 3}, 0)
	validPath := filepath.Join(dir, "valid.ivf")
	if err := os.WriteFile(validPath, valid, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ivfPayloadBytes(validPath, 32, 16, 1); err == nil || !strings.Contains(err.Error(), "IVF size") {
		t.Fatalf("wrong size error=%v", err)
	}
	if _, err := ivfPayloadBytes(validPath, 16, 16, 2); err == nil || !strings.Contains(err.Error(), "frame count") {
		t.Fatalf("wrong frame count error=%v", err)
	}

	empty := ivf.AppendFileHeader(nil, 16, 16, 30, 1, 1)
	empty = ivf.AppendFrame(empty, nil, 0)
	emptyPath := filepath.Join(dir, "empty.ivf")
	if err := os.WriteFile(emptyPath, empty, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ivfPayloadBytes(emptyPath, 16, 16, 1); err == nil || !strings.Contains(err.Error(), "empty payload") {
		t.Fatalf("empty payload error=%v", err)
	}

	truncatedPath := filepath.Join(dir, "truncated.ivf")
	if err := os.WriteFile(truncatedPath, valid[:len(valid)-1], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ivfPayloadBytes(truncatedPath, 16, 16, 1); err == nil {
		t.Fatal("truncated IVF accepted")
	}
}

func TestWriteLengthPrefixedPayload(t *testing.T) {
	var buf bytes.Buffer
	if err := writeLengthPrefixedPayload(&buf, []byte{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := writeLengthPrefixedPayload(&buf, []byte{4, 5}); err != nil {
		t.Fatal(err)
	}
	want := []byte{3, 0, 0, 0, 1, 2, 3, 2, 0, 0, 0, 4, 5}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("stream=%v want %v", buf.Bytes(), want)
	}
}

func TestEncodeGoAV1PersistsAndDecodesLengthPrefixedPayloadStream(t *testing.T) {
	cfg := benchConfig{
		width:      64,
		height:     64,
		frames:     2,
		fps:        30,
		workdir:    t.TempDir(),
		layers:     1,
		timingMode: timingModeEndToEnd,
	}
	result := encodeGoAV1(cfg, syntheticFrames(cfg.frames, cfg.width, cfg.height), 100000)
	if result.status != "ok" {
		t.Fatalf("encode status=%s err=%s", result.status, result.errText)
	}
	if result.encodedPath == "" || result.encodedContainer != "goav1-length-prefixed-low-overhead-stream" {
		t.Fatalf("encoded artifact path=%q container=%q", result.encodedPath, result.encodedContainer)
	}
	encodedBytes, encodedHash, err := fileBytesAndSHA256(result.encodedPath)
	if err != nil {
		t.Fatal(err)
	}
	if result.encodedBytes != encodedBytes || result.encodedSHA256 != encodedHash {
		t.Fatalf("encoded metadata bytes/hash=%d/%s want %d/%s", result.encodedBytes, result.encodedSHA256, encodedBytes, encodedHash)
	}
	if result.settings["payload_sha256"] == "" {
		t.Fatalf("settings missing payload hash: %+v", result.settings)
	}
	if result.settings["input_timing_scope"] != "synthetic-frame-generation" {
		t.Fatalf("input timing scope=%q", result.settings["input_timing_scope"])
	}
	if result.settings["scene_cut"] != "false" {
		t.Fatalf("scene-cut setting=%q want false", result.settings["scene_cut"])
	}
	if result.settings["metric_decode"] != "goav1-public-decoder" {
		t.Fatalf("metric decode=%q want public decoder", result.settings["metric_decode"])
	}
	raw, err := os.ReadFile(result.encodedPath)
	if err != nil {
		t.Fatal(err)
	}
	var payloadBytes int64
	frames := 0
	for off := 0; off < len(raw); {
		if len(raw)-off < 4 {
			t.Fatalf("short length prefix at offset %d", off)
		}
		n := int(binary.LittleEndian.Uint32(raw[off : off+4]))
		off += 4
		if n < 0 || n > len(raw)-off {
			t.Fatalf("bad payload length %d at offset %d stream=%d", n, off-4, len(raw))
		}
		payloadBytes += int64(n)
		off += n
		frames++
	}
	if frames != cfg.frames || payloadBytes != result.bytes || result.encodedBytes != result.bytes+int64(4*cfg.frames) {
		t.Fatalf("artifact frames=%d payload=%d encoded=%d result_payload=%d", frames, payloadBytes, result.encodedBytes, result.bytes)
	}
	decodedBytes, decodedHash, err := fileBytesAndSHA256(result.decodedYUV)
	if err != nil {
		t.Fatal(err)
	}
	if result.decodedBytes != expectedRawI420Bytes(cfg.width, cfg.height, cfg.frames) ||
		result.decodedBytes != decodedBytes ||
		result.decodedSHA256 != decodedHash {
		t.Fatalf("decoded metadata bytes/hash=%d/%s want %d/%s", result.decodedBytes, result.decodedSHA256, decodedBytes, decodedHash)
	}
}

func TestEncodeGoAV1EndToEndReloadsRawInputInsideTiming(t *testing.T) {
	cfg := benchConfig{
		width:      64,
		height:     64,
		frames:     2,
		fps:        30,
		workdir:    t.TempDir(),
		input:      filepath.Join(t.TempDir(), "bad.yuv"),
		layers:     1,
		timingMode: timingModeEndToEnd,
	}
	if err := os.WriteFile(cfg.input, []byte("too short"), 0o644); err != nil {
		t.Fatal(err)
	}
	result := encodeGoAV1(cfg, syntheticFrames(cfg.frames, cfg.width, cfg.height), 100000)
	if result.status != "error" || !strings.Contains(result.errText, "want exact raw I420 size") {
		t.Fatalf("encode status=%s err=%s", result.status, result.errText)
	}
}

func TestRunEncoderGoAV1EndToEndReadsSharedReferencePath(t *testing.T) {
	dir := t.TempDir()
	cfg := benchConfig{
		width:      64,
		height:     64,
		frames:     2,
		fps:        30,
		workdir:    dir,
		input:      filepath.Join(dir, "bad-original.yuv"),
		layers:     1,
		timingMode: timingModeEndToEnd,
	}
	if err := os.WriteFile(cfg.input, []byte("bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	frames := syntheticFrames(cfg.frames, cfg.width, cfg.height)
	refPath := filepath.Join(dir, "source.yuv")
	if err := writeFrames(refPath, frames, cfg.width, cfg.height); err != nil {
		t.Fatal(err)
	}
	result := runEncoder(cfg, frames, refPath, "goav1", 100000)
	if result.status != "ok" {
		t.Fatalf("encode status=%s err=%s", result.status, result.errText)
	}
	if result.settings["input_timing_scope"] != "raw-file-read-and-frame-construction" {
		t.Fatalf("input timing scope=%q", result.settings["input_timing_scope"])
	}
}

func TestGoAV1EncodeHelperWritesArtifactResult(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "source.yuv")
	frames := syntheticFrames(2, 64, 64)
	if err := writeFrames(input, frames, 64, 64); err != nil {
		t.Fatal(err)
	}
	resultPath := filepath.Join(dir, "helper.json")
	args := []string{
		"-width", "64",
		"-height", "64",
		"-frames", "2",
		"-fps", "30",
		"-input", input,
		"-workdir", dir,
		"-bitrate", "100000",
		"-layers", "1",
		"-result-json", resultPath,
	}
	if err := runGoAV1EncodeHelper(args); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(resultPath)
	if err != nil {
		t.Fatal(err)
	}
	var helper goAV1EncodeHelperResult
	if err := json.Unmarshal(raw, &helper); err != nil {
		t.Fatal(err)
	}
	if helper.Status != "ok" || helper.EncodedPath == "" || helper.EncodedSHA256 == "" || helper.Bytes <= 0 {
		t.Fatalf("helper result=%+v", helper)
	}
	if helper.Settings["timing_mode"] != timingModeEndToEnd ||
		helper.Settings["input_timing_scope"] != "raw-file-read-and-frame-construction" {
		t.Fatalf("helper settings=%+v", helper.Settings)
	}
	if _, err := os.Stat(helper.EncodedPath); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(helper.DecodedYUV); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("helper should not decode inside timing, decodedYUV=%q err=%v", helper.DecodedYUV, err)
	}
}

func TestEncodeGoAV1ExternalBaselineMetricsUseFFmpegDecode(t *testing.T) {
	dir := t.TempDir()
	ffmpegPath := filepath.Join(dir, "ffmpeg")
	argsPath := filepath.Join(dir, "ffmpeg.args")
	yuvBytes := expectedRawI420Bytes(64, 64, 2)
	script := fmt.Sprintf(`#!/bin/sh
: > %q
for arg in "$@"; do
	printf '%%s\n' "$arg" >> %q
done
out=""
for arg in "$@"; do
	out="$arg"
done
dd if=/dev/zero of="$out" bs=%d count=1 2>/dev/null
`, argsPath, argsPath, yuvBytes)
	if err := os.WriteFile(ffmpegPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := benchConfig{
		width:            64,
		height:           64,
		frames:           2,
		fps:              30,
		workdir:          dir,
		layers:           1,
		timingMode:       timingModeEndToEnd,
		encoders:         []string{"goav1", "aomenc"},
		ffmpegBin:        ffmpegPath,
		ffmpegAV1Decoder: "libdav1d",
	}

	result := encodeGoAV1(cfg, syntheticFrames(cfg.frames, cfg.width, cfg.height), 100000)
	if result.status != "ok" {
		t.Fatalf("encode status=%s err=%s", result.status, result.errText)
	}
	if result.settings["metric_decode"] != "ffmpeg" ||
		result.settings["ffmpeg_av1_decoder"] != "libdav1d" ||
		result.settings["metric_bitstream_container"] != "ivf" {
		t.Fatalf("metric settings=%+v", result.settings)
	}
	metricPath := result.settings["metric_bitstream_path"]
	if metricPath == "" || filepath.Ext(metricPath) != ".ivf" {
		t.Fatalf("metric bitstream path=%q", metricPath)
	}
	metricBytes, metricSHA, err := fileBytesAndSHA256(metricPath)
	if err != nil {
		t.Fatal(err)
	}
	if result.settings["metric_bitstream_bytes"] != strconv.FormatInt(metricBytes, 10) ||
		result.settings["metric_bitstream_sha256"] != metricSHA {
		t.Fatalf("metric artifact settings=%+v want bytes=%d sha=%s", result.settings, metricBytes, metricSHA)
	}
	ivfData, err := os.ReadFile(metricPath)
	if err != nil {
		t.Fatal(err)
	}
	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatal(err)
	}
	header := it.Header()
	if header.Width != uint16(cfg.width) || header.Height != uint16(cfg.height) ||
		header.TimebaseNum != uint32(cfg.fps) || header.TimebaseDen != 1 ||
		header.FrameCount != uint32(cfg.frames) {
		t.Fatalf("ivf header=%+v", header)
	}
	for i := 0; i < cfg.frames; i++ {
		frame, ok, err := it.Next()
		if err != nil || !ok {
			t.Fatalf("ivf frame %d ok=%v err=%v", i, ok, err)
		}
		if frame.Timestamp != uint64(i) || len(frame.Payload) == 0 {
			t.Fatalf("ivf frame %d=%+v", i, frame)
		}
	}
	if _, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("unexpected extra ivf frame ok=%v err=%v", ok, err)
	}
	args, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatal(err)
	}
	argText := string(args)
	for _, want := range []string{"-c:v\nlibdav1d\n", "-i\n" + metricPath + "\n", "-f\nrawvideo\n" + result.decodedYUV + "\n"} {
		if !strings.Contains(argText, want) {
			t.Fatalf("ffmpeg args missing %q in:\n%s", want, argText)
		}
	}
	if result.decodedBytes != expectedRawI420Bytes(cfg.width, cfg.height, cfg.frames) {
		t.Fatalf("decoded bytes=%d", result.decodedBytes)
	}
}

func TestLoadFramesRequiresExactRawInputSize(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "clip.yuv")
	exact := expectedRawI420Bytes(16, 16, 2)
	if err := os.WriteFile(input, make([]byte, exact+1), 0o644); err != nil {
		t.Fatal(err)
	}
	_, _, err := loadFrames(benchConfig{
		input:  input,
		width:  16,
		height: 16,
		frames: 2,
	})
	if err == nil || !strings.Contains(err.Error(), "exact raw I420") {
		t.Fatalf("trailing input error=%v", err)
	}

	if err := os.WriteFile(input, make([]byte, exact), 0o644); err != nil {
		t.Fatal(err)
	}
	if frames, _, err := loadFrames(benchConfig{input: input, width: 16, height: 16, frames: 2}); err != nil || len(frames) != 2 {
		t.Fatalf("exact input frames=%d err=%v", len(frames), err)
	}
}

func TestParseVMAFMean(t *testing.T) {
	got, err := parseVMAFMean([]byte(`{"frames":[{},{}],"pooled_metrics":{"vmaf":{"mean":91.25}}}`), 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-91.25) > 1e-9 {
		t.Fatalf("vmaf=%f", got)
	}
	if _, err := parseVMAFMean([]byte(`{"frames":[{}],"pooled_metrics":{"vmaf":{"mean":91.25}}}`), 2); err == nil ||
		!strings.Contains(err.Error(), "exact frame count") {
		t.Fatalf("vmaf frame count error=%v", err)
	}
	if _, err := parseVMAFMean([]byte(`{"frames":[{}],"pooled_metrics":{}}`), 1); err == nil {
		t.Fatal("missing vmaf accepted")
	}
}

func TestParseFrameMetricValues(t *testing.T) {
	raw := []byte(`[Parsed_psnr_0 @ 0x1] n:1 mse_avg:10.0 psnr_avg:38.123456
[Parsed_psnr_0 @ 0x1] n:2 mse_avg:0.0 psnr_avg:inf
`)
	got, err := parseFrameMetricValues(raw, `psnr_avg:([0-9.]+|inf|Inf|INF)`, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].FrameIndex != 0 || got[0].Value != "38.1235" ||
		got[1].FrameIndex != 1 || got[1].Value != "inf" {
		t.Fatalf("values=%+v", got)
	}

	raw = []byte(`[Parsed_ssim_0 @ 0x1] n:7 Y:0.900000 U:0.950000 V:0.960000 All:0.912345 (10.1)
`)
	got, err = parseFrameMetricValues(raw, `All:([0-9.]+|inf|Inf|INF)`, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].FrameIndex != 6 || got[0].Value != "0.9123" {
		t.Fatalf("values=%+v", got)
	}

	if _, err := parseFrameMetricValues([]byte("summary only"), `All:([0-9.]+)`, 1); err == nil {
		t.Fatal("missing frame values accepted")
	}
	if _, err := parseFrameMetricValues([]byte(`[Parsed_psnr_0 @ 0x1] n:1 psnr_avg:38.0
`), `psnr_avg:([0-9.]+)`, 2); err == nil ||
		!strings.Contains(err.Error(), "exact frame count") {
		t.Fatalf("short frame metrics error=%v", err)
	}
}

func TestParseScalarMetricOutputAcceptsInfinity(t *testing.T) {
	for _, tc := range []struct {
		name    string
		raw     string
		pattern string
	}{
		{name: "psnr", raw: "n:1 average:inf min:inf max:inf", pattern: `average:([0-9.]+|inf|Inf|INF)`},
		{name: "ssim", raw: "All:Inf (inf)", pattern: `All:([0-9.]+|inf|Inf|INF)`},
		{name: "xpsnr", raw: "XPSNR y: INF", pattern: `XPSNR\s+y:\s*([0-9]+(?:\.[0-9]+)?|inf|Inf|INF)`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseScalarMetricOutput([]byte(tc.raw), tc.name, tc.pattern)
			if err != nil {
				t.Fatal(err)
			}
			if !math.IsInf(got, 1) || formatMetric(got) != "inf" {
				t.Fatalf("metric=%v formatted=%q, want +Inf/inf", got, formatMetric(got))
			}
		})
	}
}

func TestMetricArgsUseDecodedAsMainAndReferenceSecond(t *testing.T) {
	cfg := benchConfig{width: 64, height: 32, frames: 2, fps: 30}
	scalar := rawMetricArgs(cfg, "source.yuv", "decoded.yuv", "psnr")
	if got := metricInputOrder(scalar); strings.Join(got, ",") != "decoded.yuv,source.yuv" {
		t.Fatalf("scalar metric inputs=%v args=%v", got, scalar)
	}
	if !strings.Contains(strings.Join(scalar, "\x00"), "[0:v][1:v]psnr") {
		t.Fatalf("scalar metric filter args=%v", scalar)
	}

	vmaf := vmafArgs(cfg, "source.yuv", "decoded.yuv", "vmaf.json")
	if got := metricInputOrder(vmaf); strings.Join(got, ",") != "decoded.yuv,source.yuv" {
		t.Fatalf("vmaf metric inputs=%v args=%v", got, vmaf)
	}
	if !strings.Contains(strings.Join(vmaf, "\x00"), "[0:v][1:v]libvmaf") {
		t.Fatalf("vmaf filter args=%v", vmaf)
	}
	modelCfg := cfg
	modelCfg.vmafModel = "version=vmaf_v0.6.1:name=default"
	vmaf = vmafArgs(modelCfg, "source.yuv", "decoded.yuv", "vmaf.json")
	if !strings.Contains(strings.Join(vmaf, "\x00"), `model=version=vmaf_v0.6.1\:name=default`) {
		t.Fatalf("vmaf model filter args=%v", vmaf)
	}
}

func metricInputOrder(args []string) []string {
	var inputs []string
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "-i" {
			inputs = append(inputs, args[i+1])
		}
	}
	return inputs
}

func TestBDRatePercent(t *testing.T) {
	anchor := []rdPoint{
		{Metric: 30, Rate: 100_000},
		{Metric: 35, Rate: 200_000},
		{Metric: 40, Rate: 400_000},
		{Metric: 45, Rate: 800_000},
	}
	same, _, _, err := bdRatePercent(anchor, anchor)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(same) > 1e-9 {
		t.Fatalf("same curve bd-rate=%f want 0", same)
	}

	doubleRate := []rdPoint{
		{Metric: 30, Rate: 200_000},
		{Metric: 35, Rate: 400_000},
		{Metric: 40, Rate: 800_000},
		{Metric: 45, Rate: 1_600_000},
	}
	got, lo, hi, err := bdRatePercent(anchor, doubleRate)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-100) > 1e-6 || lo != 30 || hi != 45 {
		t.Fatalf("double-rate bd-rate=%f overlap=[%f,%f]", got, lo, hi)
	}

	halfRate := []rdPoint{
		{Metric: 30, Rate: 50_000},
		{Metric: 35, Rate: 100_000},
		{Metric: 40, Rate: 200_000},
		{Metric: 45, Rate: 400_000},
	}
	got, _, _, err = bdRatePercent(anchor, halfRate)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got+50) > 1e-6 {
		t.Fatalf("half-rate bd-rate=%f want -50", got)
	}
}

func TestBDRatePercentRejectsInsufficientPoints(t *testing.T) {
	_, _, _, err := bdRatePercent(
		[]rdPoint{{Metric: 30, Rate: 100}, {Metric: 31, Rate: 120}, {Metric: 32, Rate: 140}},
		[]rdPoint{{Metric: 30, Rate: 100}, {Metric: 31, Rate: 120}, {Metric: 32, Rate: 140}},
	)
	if err == nil {
		t.Fatal("insufficient points accepted")
	}
}

func TestBDRatePercentRejectsDuplicateMetricValues(t *testing.T) {
	anchor := []rdPoint{
		{Metric: 30, Rate: 100_000},
		{Metric: 35, Rate: 200_000},
		{Metric: 35, Rate: 250_000},
		{Metric: 40, Rate: 400_000},
		{Metric: 45, Rate: 800_000},
	}
	candidate := []rdPoint{
		{Metric: 30, Rate: 200_000},
		{Metric: 35, Rate: 400_000},
		{Metric: 40, Rate: 800_000},
		{Metric: 45, Rate: 1_600_000},
	}
	_, _, _, err := bdRatePercent(anchor, candidate)
	if err == nil || !strings.Contains(err.Error(), "anchor: duplicate RD metric point 35") {
		t.Fatalf("duplicate metric error=%v", err)
	}
}

func TestRDPointsForUsesFullPrecisionMetricValues(t *testing.T) {
	const want = 40.00004
	rows := []benchRow{
		{
			clip:      "clip",
			encoder:   "goav1",
			actualBPS: 100_000,
			metrics: metrics{
				psnr:      "40.0000",
				psnrValue: want,
				psnrValid: true,
			},
			status: "ok",
		},
	}
	points := rdPointsFor(rows, "clip", "goav1", metrics.psnrRDMetric)
	if len(points) != 1 {
		t.Fatalf("points=%v", points)
	}
	if points[0].Metric != want {
		t.Fatalf("metric=%0.8f want full precision %0.8f", points[0].Metric, want)
	}
}

func TestSummarizeBDRateUsesActualBitrate(t *testing.T) {
	rows := []benchRow{
		{clip: "clip", encoder: "anchor", actualBPS: 100_000, metrics: metrics{psnr: "30"}, status: "ok"},
		{clip: "clip", encoder: "anchor", actualBPS: 200_000, metrics: metrics{psnr: "35"}, status: "ok"},
		{clip: "clip", encoder: "anchor", actualBPS: 400_000, metrics: metrics{psnr: "40"}, status: "ok"},
		{clip: "clip", encoder: "anchor", actualBPS: 800_000, metrics: metrics{psnr: "45"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 200_000, metrics: metrics{psnr: "30"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 400_000, metrics: metrics{psnr: "35"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 800_000, metrics: metrics{psnr: "40"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 1_600_000, metrics: metrics{psnr: "45"}, status: "ok"},
	}
	summaries := summarizeBDRate("anchor", rows)
	var psnr summaryRow
	for _, summary := range summaries {
		if summary.Encoder == "candidate" && summary.Metric == "psnr_avg" {
			psnr = summary
			break
		}
	}
	if psnr.Status != "ok" {
		t.Fatalf("psnr summary=%+v", psnr)
	}
	if math.Abs(psnr.BDRatePct-100) > 1e-6 {
		t.Fatalf("bd-rate=%f want 100", psnr.BDRatePct)
	}
}

func TestValidateRequiredSummaries(t *testing.T) {
	rows := []benchRow{
		{clip: "clip", encoder: "anchor", actualBPS: 100_000, metrics: metrics{psnr: "30"}, status: "ok"},
		{clip: "clip", encoder: "anchor", actualBPS: 200_000, metrics: metrics{psnr: "35"}, status: "ok"},
		{clip: "clip", encoder: "anchor", actualBPS: 400_000, metrics: metrics{psnr: "40"}, status: "ok"},
		{clip: "clip", encoder: "anchor", actualBPS: 800_000, metrics: metrics{psnr: "45"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 200_000, metrics: metrics{psnr: "30"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 400_000, metrics: metrics{psnr: "35"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 800_000, metrics: metrics{psnr: "40"}, status: "ok"},
		{clip: "clip", encoder: "candidate", actualBPS: 1_600_000, metrics: metrics{psnr: "45"}, status: "ok"},
	}
	cfg := benchConfig{
		anchorEncoder:   "anchor",
		encoders:        []string{"anchor", "candidate"},
		requiredMetrics: []string{"psnr"},
	}
	summaries := summarizeBDRate("anchor", rows)
	if err := validateRequiredSummaries(cfg, rows, summaries); err != nil {
		t.Fatalf("valid required summary failed: %v", err)
	}

	cfg.requiredMetrics = []string{"vmaf"}
	if err := validateRequiredSummaries(cfg, rows, summaries); err == nil {
		t.Fatal("missing required metric summary accepted")
	}

	cfg.requiredMetrics = []string{"psnr"}
	cfg.encoders = []string{"anchor"}
	if err := validateRequiredSummaries(cfg, rows, summaries); err == nil {
		t.Fatal("summary without candidate accepted")
	}
}

func TestValidateRequiredSummariesRejectsErrorRows(t *testing.T) {
	rows := []benchRow{{clip: "clip", encoder: "anchor", status: "ok"}}
	cfg := benchConfig{
		anchorEncoder:   "anchor",
		encoders:        []string{"anchor", "candidate"},
		requiredMetrics: []string{"psnr"},
	}
	summaries := []summaryRow{{
		Clip:    "clip",
		Anchor:  "anchor",
		Encoder: "candidate",
		Metric:  "psnr_avg",
		Status:  "error",
		ErrText: "need at least 4 points",
	}}
	if err := validateRequiredSummaries(cfg, rows, summaries); err == nil {
		t.Fatal("error summary row accepted")
	}
}

func TestWriteStatsRow(t *testing.T) {
	var buf bytes.Buffer
	writer := csv.NewWriter(&buf)
	if err := writeStatsHeader(writer); err != nil {
		t.Fatal(err)
	}
	stats := goav1.EncoderDecisionStats{
		Frames:             2,
		Keyframes:          1,
		InterFrames:        1,
		Tiles:              2,
		PartitionDecisions: 12,
		Blocks:             8,
		InterBlocks:        4,
		IntraBlocks:        4,
		SkipBlocks:         1,
		CodedBlocks:        7,
		LumaTXBs:           7,
	}
	stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceLast] = 3
	stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceGolden] = 1
	stats.BlockSizes[goav1.EncoderDecisionBlockSize8x8] = 6
	stats.TXTypes[goav1.EncoderDecisionTransformDCTDCT] = 6
	stats.TXTypes[goav1.EncoderDecisionTransformADSTADST] = 1
	stats.NonDCTTXBs = 1
	row := benchRow{
		clip:      "clip",
		width:     64,
		height:    64,
		frames:    2,
		fps:       30,
		encoder:   "goav1",
		targetBPS: 100000,
		actualBPS: 96000,
		status:    "ok",
	}
	if err := writeStatsRow(writer, row, stats); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	records, err := csv.NewReader(bytes.NewReader(buf.Bytes())).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 {
		t.Fatalf("records=%d", len(records))
	}
	header := map[string]int{}
	for i, name := range records[0] {
		header[name] = i
	}
	if records[1][header["encoded_frames"]] != "2" ||
		records[1][header["primary_golden"]] != "1" ||
		records[1][header["tx_adst_adst"]] != "1" ||
		records[1][header["non_dct_txbs"]] != "1" {
		t.Fatalf("stats row=%v", records[1])
	}
}

func TestWriteFrameStatsRow(t *testing.T) {
	var buf bytes.Buffer
	writer := csv.NewWriter(&buf)
	if err := writeFrameStatsHeader(writer); err != nil {
		t.Fatal(err)
	}
	stats := goav1.EncoderDecisionStats{
		Frames:             1,
		InterFrames:        1,
		Tiles:              2,
		PartitionDecisions: 8,
		Blocks:             4,
		InterBlocks:        4,
		CodedBlocks:        3,
		SkipBlocks:         1,
		CompoundBlocks:     2,
		NonDCTTXBs:         1,
	}
	stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceLast] = 2
	stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceGolden] = 2
	row := benchRow{
		clip:      "clip",
		width:     64,
		height:    64,
		fps:       30,
		encoder:   "goav1",
		targetBPS: 100000,
		status:    "ok",
	}
	frame := goAV1FrameStats{
		FrameIndex:      3,
		TemporalID:      2,
		QIndexBefore:    90,
		QIndexAfter:     96,
		Bytes:           123,
		CumulativeBytes: 456,
		Duration:        1500 * time.Microsecond,
		Stats:           stats,
	}
	if err := writeFrameStatsRow(writer, row, frame); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	records, err := csv.NewReader(bytes.NewReader(buf.Bytes())).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 {
		t.Fatalf("records=%d", len(records))
	}
	header := map[string]int{}
	for i, name := range records[0] {
		header[name] = i
	}
	if records[1][header["frame_index"]] != "3" ||
		records[1][header["temporal_id"]] != "2" ||
		records[1][header["qindex_after"]] != "96" ||
		records[1][header["frame_bits"]] != "984" ||
		records[1][header["encode_ms"]] != "1.500" ||
		records[1][header["primary_golden"]] != "2" ||
		records[1][header["tx_non_dct"]] != "1" {
		t.Fatalf("frame stats row=%v", records[1])
	}
}

func TestWriteFrameMetricRow(t *testing.T) {
	var buf bytes.Buffer
	writer := csv.NewWriter(&buf)
	if err := writeFrameMetricsHeader(writer); err != nil {
		t.Fatal(err)
	}
	if err := writeFrameMetricRow(writer, "clip", "goav1", 100000, 2, "psnr_avg", "41.2500"); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	records, err := csv.NewReader(bytes.NewReader(buf.Bytes())).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || records[1][0] != "clip" || records[1][3] != "2" ||
		records[1][4] != "psnr_avg" || records[1][5] != "41.2500" {
		t.Fatalf("records=%v", records)
	}
}

func TestDiffDecisionStats(t *testing.T) {
	before := goav1.EncoderDecisionStats{Frames: 4, Blocks: 10}
	before.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceLast] = 3
	after := goav1.EncoderDecisionStats{Frames: 5, Blocks: 16, NonDCTTXBs: 2}
	after.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceLast] = 5
	got := diffDecisionStats(after, before)
	if got.Frames != 1 || got.Blocks != 6 || got.NonDCTTXBs != 2 ||
		got.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceLast] != 2 {
		t.Fatalf("diff=%+v", got)
	}
	if subU64(1, 2) != 0 {
		t.Fatal("subU64 underflowed")
	}
}

func TestWriteBenchRowIncludesCPUBudgetColumns(t *testing.T) {
	var buf bytes.Buffer
	writer := csv.NewWriter(&buf)
	row := benchRow{
		clip:      "clip",
		width:     64,
		height:    64,
		frames:    30,
		fps:       30,
		encoder:   "svt-av1",
		targetBPS: 100000,
		actualBPS: 96000,
		encodeFPS: "120.00",
		duration:  250 * time.Millisecond,
		cpuUser:   400 * time.Millisecond,
		cpuSystem: 100 * time.Millisecond,
		cpuOK:     true,
		bytes:     400,
		metrics:   metrics{psnr: "40.0", ssim: "0.99", xpsnr: "NA", vmaf: "NA"},
		status:    "ok",
	}
	if err := writeBenchRow(writer, row); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	records, err := csv.NewReader(&buf).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	got := records[0]
	if got[9] != "0.250" || got[10] != "0.400" || got[11] != "0.100" ||
		got[12] != "0.500" || got[13] != "2.00" || got[14] != "400" {
		t.Fatalf("cpu columns=%v", got)
	}
}

func TestMetadataConfigCopiesSlices(t *testing.T) {
	cfg := benchConfig{
		width:              64,
		height:             64,
		frames:             4,
		fps:                30,
		encoders:           []string{"goav1"},
		bitrates:           []int{100000},
		requiredMetrics:    []string{"psnr"},
		requiredEncoders:   []string{"goav1"},
		requireSummary:     true,
		requireCorpus:      true,
		minClips:           6,
		minSourceIDs:       2,
		minCategories:      2,
		anchorEncoder:      "goav1",
		goMaxProcs:         4,
		goGC:               "off",
		goav1MaxThreads:    4,
		goav1Effort:        int(goav1.EncoderWebRTCMinEffortLevel),
		goav1SceneCut:      false,
		goav1ProcessTiming: true,
		aomThreads:         1,
		aomRowMT:           0,
		aomCPUUsed:         8,
		svtLP:              5,
		svtPreset:          13,
		ffmpegBin:          "/tools/ffmpeg",
		ffmpegSHA256:       strings.Repeat("1", 64),
		ffmpegAV1Decoder:   "libdav1d",
		vmafModel:          "version=vmaf_v0.6.1",
		aomencBin:          "/tools/aomenc",
		aomencSHA256:       strings.Repeat("2", 64),
		svtBin:             "/tools/SvtAv1EncApp",
		svtSHA256:          strings.Repeat("3", 64),
		timingMode:         timingModeEndToEnd,
		runOrder:           runOrderShuffle,
		shuffleSeed:        42,
		runs:               5,
		warmupRuns:         1,
		publish:            true,
	}
	got, err := metadataConfigFor(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg.encoders[0] = "mutated"
	cfg.bitrates[0] = 1
	cfg.requiredMetrics[0] = "vmaf"
	cfg.requiredEncoders[0] = "aomenc"
	if got.Encoders[0] != "goav1" || got.Bitrates[0] != 100000 ||
		got.RequiredMetrics[0] != "psnr" || got.RequiredEncoders[0] != "goav1" ||
		!got.RequireSummary || !got.RequireCorpus || got.MinClips != 6 ||
		got.MinSourceIDs != 2 || got.MinCategories != 2 ||
		got.GoMaxProcs != 4 || got.GoGC != "off" || got.GoAV1MaxThreads != 4 ||
		got.GoAV1Effort != int(goav1.EncoderWebRTCMinEffortLevel) ||
		got.GoAV1SceneCut || !got.GoAV1ProcessTime ||
		got.TileColumnsLog2 != 0 || got.TileSemantics != "tile-columns-log2" ||
		got.AOMThreads != 1 || got.AOMRowMT != 0 ||
		got.AOMCPUUsed != 8 || got.SVTLP != 5 || got.SVTPreset != 13 ||
		got.FFmpegBin != "/tools/ffmpeg" || got.FFmpegSHA256 != strings.Repeat("1", 64) ||
		got.FFmpegAV1Decoder != "libdav1d" ||
		got.VMAFModel != "version=vmaf_v0.6.1" ||
		got.AOMEncBin != "/tools/aomenc" || got.AOMEncSHA256 != strings.Repeat("2", 64) ||
		got.SVTBin != "/tools/SvtAv1EncApp" || got.SVTSHA256 != strings.Repeat("3", 64) ||
		got.TimingMode != timingModeEndToEnd ||
		got.RunOrder != runOrderShuffle || got.ShuffleSeed != 42 ||
		got.SampleOrder != "interleaved-by-sample-pass" ||
		got.Runs != 5 || got.WarmupRuns != 1 || !got.Publish {
		t.Fatalf("metadata config aliases inputs: %+v", got)
	}
}

func TestFairnessNotesDocumentSVTLP(t *testing.T) {
	notes := fairnessNotes(benchConfig{encoders: []string{"goav1", "aomenc", "svt-av1"}, svtLP: 0, timingMode: timingModeEndToEnd, publish: true})
	joined := strings.Join(notes, "\n")
	if !strings.Contains(joined, "not a target processor or thread count") ||
		!strings.Contains(joined, "observed_parallelism") ||
		!strings.Contains(joined, "timing_mode") ||
		!strings.Contains(joined, "raw input loading") ||
		!strings.Contains(joined, "run_order") ||
		!strings.Contains(joined, "explicit seed") ||
		!strings.Contains(joined, "sample passes") ||
		!strings.Contains(joined, "median wall-time") ||
		!strings.Contains(joined, "artifact hashes") ||
		!strings.Contains(joined, "sweep --lp 0..6") ||
		!strings.Contains(joined, "-goav1-max-threads") ||
		!strings.Contains(joined, "-goav1-effort") ||
		!strings.Contains(joined, "goav1_process_timing") ||
		!strings.Contains(joined, "-goav1-process-timing=true") ||
		!strings.Contains(joined, "-goav1-scene-cut=false") ||
		!strings.Contains(joined, "tile-column log2") ||
		!strings.Contains(joined, "-ffmpeg-av1-decoder") ||
		!strings.Contains(joined, "1000/500/600 ms") ||
		!strings.Contains(joined, "-aom-cpu-used") ||
		!strings.Contains(joined, "-aom-threads") ||
		!strings.Contains(joined, "-aom-row-mt") ||
		!strings.Contains(joined, "-svt-preset") ||
		!strings.Contains(joined, "simd_tier") ||
		!strings.Contains(joined, "svt_asm") ||
		!strings.Contains(joined, "--lp 0") ||
		!strings.Contains(joined, "-layers 1") ||
		!strings.Contains(joined, "Publish mode") {
		t.Fatalf("fairness notes=%q", joined)
	}
}

func TestExternalBaselineSettingsRecordPinnedKnobs(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	cfg := benchConfig{
		workdir:          t.TempDir(),
		width:            64,
		height:           64,
		fps:              60,
		frames:           31,
		tiles:            2,
		keyInterval:      60,
		aomThreads:       3,
		aomRowMT:         1,
		aomCPUUsed:       8,
		svtLP:            4,
		svtPreset:        13,
		svtASM:           "neon",
		ffmpegAV1Decoder: "libdav1d",
	}

	aom := encodeAOM(cfg, filepath.Join(cfg.workdir, "input.yuv"), 1_200_000)
	if aom.status != "skipped" {
		t.Fatalf("aom status=%q want skipped", aom.status)
	}
	assertSettings(t, aom.settings, map[string]string{
		"profile":            "0",
		"bit_depth":          "8",
		"input_bit_depth":    "8",
		"color_format":       "i420",
		"quiet":              "1",
		"deadline":           "rt",
		"end_usage":          "cbr",
		"target_kbps":        "1200",
		"fps":                "60/1",
		"cpu_used":           "8",
		"aom_threads":        "3",
		"aom_row_mt":         "1",
		"lag_in_frames":      "0",
		"auto_alt_ref":       "0",
		"enable_fwd_kf":      "0",
		"drop_frame":         "0",
		"buf_sz_ms":          "1000",
		"buf_initial_ms":     "500",
		"buf_optimal_ms":     "600",
		"limit_frames":       "31",
		"kf_min_dist":        "60",
		"kf_max_dist":        "60",
		"tile_columns":       "2",
		"tile_columns_log2":  "2",
		"tile_semantics":     "tile-columns-log2",
		"ffmpeg_av1_decoder": "libdav1d",
	})

	svt := encodeSVT(cfg, filepath.Join(cfg.workdir, "input.yuv"), 1_200_000)
	if svt.status != "skipped" {
		t.Fatalf("svt status=%q want skipped", svt.status)
	}
	assertSettings(t, svt.settings, map[string]string{
		"preset":             "13",
		"profile":            "0",
		"level":              "0",
		"input_depth":        "8",
		"color_format":       "1",
		"fps_num":            "60",
		"fps_denom":          "1",
		"frames":             "31",
		"rate_control":       "cbr",
		"target_kbps":        "1200",
		"buf_sz_ms":          "1000",
		"buf_initial_ms":     "500",
		"buf_optimal_ms":     "600",
		"lookahead":          "0",
		"pred_struct":        "1",
		"rtc":                "1",
		"scd":                "0",
		"tf":                 "0",
		"irefresh_type":      "2",
		"keyint":             "60",
		"progress":           "0",
		"tile_columns":       "2",
		"tile_columns_log2":  "2",
		"tile_semantics":     "tile-columns-log2",
		"ffmpeg_av1_decoder": "libdav1d",
		"svt_lp":             "4",
		"svt_asm":            "neon",
	})
}

func assertSettings(t *testing.T, got map[string]string, want map[string]string) {
	t.Helper()
	for key, wantValue := range want {
		if gotValue := got[key]; gotValue != wantValue {
			t.Fatalf("settings[%q]=%q want %q in %+v", key, gotValue, wantValue, got)
		}
	}
}

func TestTimeCommandCapturesFailureWithoutRerun(t *testing.T) {
	dir := t.TempDir()
	countPath := filepath.Join(dir, "count")
	scriptPath := filepath.Join(dir, "fail-once.sh")
	script := fmt.Sprintf(`#!/bin/sh
n=$(cat %q 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > %q
echo "stdout-run-$n"
echo "stderr-run-$n" >&2
exit 7
`, countPath, countPath)
	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	var result encodeResult
	_ = timeCommand(defaultCommandTimeout, scriptPath, nil, &result)
	rawCount, err := os.ReadFile(countPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(rawCount)) != "1" {
		t.Fatalf("command ran %s times, want once", strings.TrimSpace(string(rawCount)))
	}
	if result.status != "error" ||
		!strings.Contains(result.errText, "stdout-run-1") ||
		!strings.Contains(result.errText, "stderr-run-1") {
		t.Fatalf("result status=%q err=%q", result.status, result.errText)
	}
}

func TestExternalCommandEnvSanitizesAmbientControls(t *testing.T) {
	dir := t.TempDir()
	envPath := filepath.Join(dir, "env.txt")
	scriptPath := filepath.Join(dir, "env.sh")
	script := fmt.Sprintf(`#!/bin/sh
{
printf 'omp=%%s\n' "$OMP_NUM_THREADS"
printf 'dyld=%%s\n' "$DYLD_INSERT_LIBRARIES"
printf 'lc=%%s\n' "$LC_ALL"
printf 'tz=%%s\n' "$TZ"
printf 'path=%%s\n' "$PATH"
} > %q
`, envPath)
	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OMP_NUM_THREADS", "99")
	t.Setenv("DYLD_INSERT_LIBRARIES", "/tmp/not-real.dylib")
	t.Setenv("LC_ALL", "fr_FR.UTF-8")
	t.Setenv("TZ", "Europe/Lisbon")

	var result encodeResult
	_ = timeCommand(defaultCommandTimeout, scriptPath, nil, &result)
	if result.status != "" {
		t.Fatalf("command status=%q err=%q", result.status, result.errText)
	}
	raw, err := os.ReadFile(envPath)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, want := range []string{"omp=\n", "dyld=\n", "lc=C\n", "tz=UTC\n", "path="} {
		if !strings.Contains(text, want) {
			t.Fatalf("sanitized env missing %q in:\n%s", want, text)
		}
	}
}

func TestValidatePublishConfigRequiresExplicitControls(t *testing.T) {
	clearQualitybenchPublishGoEnv(t)
	stubQualitybenchCPUState(t, benchenv.CPUState{
		GOOS:                "test",
		AffinitySupported:   true,
		AffinityAllowedList: "0-3",
		CPUOnlineList:       "0-3",
	})
	ffmpegBin, ffmpegHash := writeTestExecutableWithSHA256(t, "ffmpeg")
	goBin, goHash := writeTestExecutableWithSHA256(t, "go")
	aomencBin, aomencHash := writeTestExecutableWithSHA256(t, "aomenc")
	svtBin, svtHash := writeTestExecutableWithSHA256(t, "SvtAv1EncApp")
	vmafModelPath := filepath.Join(t.TempDir(), "vmaf.json")
	if err := os.WriteFile(vmafModelPath, []byte(`{"model":"fixture"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := benchConfig{
		workdir:             "/tmp/work",
		csvPath:             "/tmp/quality.csv",
		metadataPath:        "/tmp/meta.json",
		manifestPath:        "/tmp/clips.csv",
		summaryCSVPath:      "/tmp/summary.csv",
		frameMetricsCSVPath: "/tmp/frame-metrics.csv",
		anchorEncoder:       "aomenc",
		requiredEncodersRaw: "all",
		encoders:            []string{"goav1", "aomenc", "svt-av1"},
		bitrates:            []int{3000000, 6000000, 9000000, 12000000},
		requiredMetrics:     []string{"psnr", "vmaf"},
		requireCorpus:       true,
		minClips:            2,
		minSourceIDs:        2,
		minCategories:       2,
		requireSummary:      true,
		fps:                 60,
		goMaxProcs:          4,
		goBin:               goBin,
		goSHA256:            goHash,
		goav1MaxThreads:     4,
		goav1Effort:         0,
		goav1SceneCut:       false,
		goav1ProcessTiming:  true,
		layers:              1,
		tiles:               0,
		goldenInterval:      0,
		keyInterval:         60,
		aomThreads:          4,
		aomRowMT:            1,
		aomCPUUsed:          8,
		svtLP:               4,
		svtPreset:           13,
		svtASM:              "neon",
		ffmpegAV1Decoder:    "libdav1d",
		vmafModel:           "version=vmaf_v0.6.1",
		timingMode:          timingModeEndToEnd,
		runOrder:            runOrderShuffle,
		shuffleSeed:         7,
		runs:                3,
		warmupRuns:          1,
		commandTimeout:      30 * time.Minute,
		environmentNotes:    "fixed power mode, idle machine",
		cpuAffinity:         "none",
		powerMode:           "high power",
		thermalState:        "cool start",
		frequencyPolicy:     "automatic",
		backgroundLoad:      "idle machine",
		goGC:                "off",
		ffmpegBin:           ffmpegBin,
		ffmpegSHA256:        ffmpegHash,
		aomencBin:           aomencBin,
		aomencSHA256:        aomencHash,
		svtBin:              svtBin,
		svtSHA256:           svtHash,
		explicitFlags: map[string]bool{
			"bitrates":             true,
			"encoders":             true,
			"workdir":              true,
			"csv":                  true,
			"metadata-json":        true,
			"manifest":             true,
			"require-corpus":       true,
			"min-clips":            true,
			"min-source-ids":       true,
			"min-categories":       true,
			"require-encoders":     true,
			"require-metrics":      true,
			"summary-csv":          true,
			"frame-metrics-csv":    true,
			"require-summary":      true,
			"gomaxprocs":           true,
			"go-bin":               true,
			"go-sha256":            true,
			"gogc":                 true,
			"fps":                  true,
			"layers":               true,
			"tiles":                true,
			"golden":               true,
			"keyint":               true,
			"anchor":               true,
			"timing-mode":          true,
			"run-order":            true,
			"shuffle-seed":         true,
			"runs":                 true,
			"warmup-runs":          true,
			"command-timeout":      true,
			"goav1-max-threads":    true,
			"goav1-effort":         true,
			"goav1-scene-cut":      true,
			"goav1-process-timing": true,
			"environment-notes":    true,
			"cpu-affinity":         true,
			"power-mode":           true,
			"thermal-state":        true,
			"frequency-policy":     true,
			"background-load":      true,
			"ffmpeg-bin":           true,
			"ffmpeg-sha256":        true,
			"aom-cpu-used":         true,
			"aom-threads":          true,
			"aom-row-mt":           true,
			"aomenc-bin":           true,
			"aomenc-sha256":        true,
			"svt-preset":           true,
			"svt-lp":               true,
			"svt-asm":              true,
			"svt-bin":              true,
			"svt-sha256":           true,
			"ffmpeg-av1-decoder":   true,
			"vmaf-model":           true,
		},
	}
	if err := validatePublishConfig(cfg, gitMetadata{Commit: "abc"}); err != nil {
		t.Fatalf("valid publish config failed: %v", err)
	}
	missingGoPin := cfg
	missingGoPin.explicitFlags = copyStringBoolMap(cfg.explicitFlags)
	delete(missingGoPin.explicitFlags, "go-sha256")
	if err := validatePublishConfig(missingGoPin, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "go-sha256") {
		t.Fatalf("missing go-sha256 error=%v", err)
	}
	badGoPin := cfg
	badGoPin.explicitFlags = copyStringBoolMap(cfg.explicitFlags)
	badGoPin.goSHA256 = strings.Repeat("0", 64)
	if err := validatePublishConfig(badGoPin, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "go-bin") {
		t.Fatalf("bad go hash error=%v", err)
	}
	oldProbe := observeBenchmarkCPUState
	observeBenchmarkCPUState = func() benchenv.CPUState {
		return benchenv.CPUState{
			GOOS:                "test",
			AffinitySupported:   true,
			AffinityAllowedList: "0-1",
			CPUOnlineList:       "0-3",
		}
	}
	if err := validatePublishConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "restricted CPU affinity") {
		t.Fatalf("restricted CPU affinity error=%v", err)
	}
	observeBenchmarkCPUState = oldProbe

	pathModel := cfg
	pathModel.vmafModel = "path=" + vmafModelPath
	if err := validatePublishConfig(pathModel, gitMetadata{Commit: "abc"}); err != nil {
		t.Fatalf("absolute VMAF path model failed: %v", err)
	}

	relativePathModel := cfg
	relativePathModel.vmafModel = "path=relative-model.json"
	if err := validatePublishConfig(relativePathModel, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "absolute") {
		t.Fatalf("relative VMAF path model error=%v", err)
	}

	ambiguousModel := cfg
	ambiguousModel.vmafModel = "vmaf_v0.6.1"
	if err := validatePublishConfig(ambiguousModel, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "version=") {
		t.Fatalf("ambiguous VMAF model error=%v", err)
	}

	missingEnvironmentNotes := cfg
	missingEnvironmentNotes.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingEnvironmentNotes.explicitFlags[k] = v
	}
	delete(missingEnvironmentNotes.explicitFlags, "environment-notes")
	if err := validatePublishConfig(missingEnvironmentNotes, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-environment-notes") {
		t.Fatalf("missing environment notes error=%v", err)
	}

	emptyEnvironmentNotes := cfg
	emptyEnvironmentNotes.environmentNotes = " "
	if err := validatePublishConfig(emptyEnvironmentNotes, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "non-empty -environment-notes") {
		t.Fatalf("empty environment notes error=%v", err)
	}

	missingMachineState := cfg
	missingMachineState.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingMachineState.explicitFlags[k] = v
	}
	delete(missingMachineState.explicitFlags, "thermal-state")
	if err := validatePublishConfig(missingMachineState, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-thermal-state") {
		t.Fatalf("missing thermal-state error=%v", err)
	}

	emptyMachineState := cfg
	emptyMachineState.backgroundLoad = " "
	if err := validatePublishConfig(emptyMachineState, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "non-empty -background-load") {
		t.Fatalf("empty background-load error=%v", err)
	}

	missingGoAV1Threads := cfg
	missingGoAV1Threads.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingGoAV1Threads.explicitFlags[k] = v
	}
	delete(missingGoAV1Threads.explicitFlags, "goav1-max-threads")
	if err := validatePublishConfig(missingGoAV1Threads, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-goav1-max-threads") {
		t.Fatalf("missing explicit goav1 max threads error=%v", err)
	}

	missingGoAV1Effort := cfg
	missingGoAV1Effort.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingGoAV1Effort.explicitFlags[k] = v
	}
	delete(missingGoAV1Effort.explicitFlags, "goav1-effort")
	if err := validatePublishConfig(missingGoAV1Effort, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-goav1-effort") {
		t.Fatalf("missing explicit goav1 effort error=%v", err)
	}

	missingGoAV1SceneCut := cfg
	missingGoAV1SceneCut.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingGoAV1SceneCut.explicitFlags[k] = v
	}
	delete(missingGoAV1SceneCut.explicitFlags, "goav1-scene-cut")
	if err := validatePublishConfig(missingGoAV1SceneCut, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-goav1-scene-cut") {
		t.Fatalf("missing explicit goav1 scene-cut error=%v", err)
	}

	missingGoGC := cfg
	missingGoGC.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingGoGC.explicitFlags[k] = v
	}
	delete(missingGoGC.explicitFlags, "gogc")
	if err := validatePublishConfig(missingGoGC, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-gogc") {
		t.Fatalf("missing explicit gogc error=%v", err)
	}

	missingFrameMetrics := cfg
	missingFrameMetrics.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingFrameMetrics.explicitFlags[k] = v
	}
	delete(missingFrameMetrics.explicitFlags, "frame-metrics-csv")
	if err := validatePublishConfig(missingFrameMetrics, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-frame-metrics-csv") {
		t.Fatalf("missing frame metrics error=%v", err)
	}

	emptyFrameMetrics := cfg
	emptyFrameMetrics.frameMetricsCSVPath = ""
	if err := validatePublishConfig(emptyFrameMetrics, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-frame-metrics-csv") {
		t.Fatalf("empty frame metrics error=%v", err)
	}

	relativeFFmpeg := cfg
	relativeFFmpeg.ffmpegBin = "ffmpeg"
	if err := validatePublishConfig(relativeFFmpeg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "absolute path") {
		t.Fatalf("relative ffmpeg error=%v", err)
	}

	badFFmpegHash := cfg
	badFFmpegHash.ffmpegSHA256 = strings.Repeat("0", 64)
	if err := validatePublishConfig(badFFmpegHash, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "to match") {
		t.Fatalf("bad ffmpeg hash error=%v", err)
	}

	t.Setenv("GOFLAGS", "-race")
	if err := validatePublishConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GOFLAGS unset") {
		t.Fatalf("hidden GOFLAGS error=%v", err)
	}
	t.Setenv("GOFLAGS", "")

	t.Setenv("GOAMD64", "v4")
	if err := validatePublishConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GOAMD64 unset") {
		t.Fatalf("hidden GOAMD64 error=%v", err)
	}
	t.Setenv("GOAMD64", "")

	t.Setenv("GOCACHE", filepath.Join(t.TempDir(), "cache"))
	if err := validatePublishConfig(cfg, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "GOCACHE unset") {
		t.Fatalf("hidden GOCACHE error=%v", err)
	}
	t.Setenv("GOCACHE", "")

	missing := cfg
	missing.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missing.explicitFlags[k] = v
	}
	delete(missing.explicitFlags, "aom-row-mt")
	if err := validatePublishConfig(missing, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-aom-row-mt") {
		t.Fatalf("missing explicit aom row-mt error=%v", err)
	}

	missingAOMSpeed := cfg
	missingAOMSpeed.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingAOMSpeed.explicitFlags[k] = v
	}
	delete(missingAOMSpeed.explicitFlags, "aom-cpu-used")
	if err := validatePublishConfig(missingAOMSpeed, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-aom-cpu-used") {
		t.Fatalf("missing explicit aom speed error=%v", err)
	}

	missingSVTPreset := cfg
	missingSVTPreset.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingSVTPreset.explicitFlags[k] = v
	}
	delete(missingSVTPreset.explicitFlags, "svt-preset")
	if err := validatePublishConfig(missingSVTPreset, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-svt-preset") {
		t.Fatalf("missing explicit svt preset error=%v", err)
	}

	missingLayers := cfg
	missingLayers.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingLayers.explicitFlags[k] = v
	}
	delete(missingLayers.explicitFlags, "layers")
	if err := validatePublishConfig(missingLayers, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-layers") {
		t.Fatalf("missing explicit layers error=%v", err)
	}

	layeredExternal := cfg
	layeredExternal.layers = 3
	if err := validatePublishConfig(layeredExternal, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-layers 1") {
		t.Fatalf("layered external publish error=%v", err)
	}

	layeredGoAV1Only := cfg
	layeredGoAV1Only.encoders = []string{"goav1"}
	layeredGoAV1Only.layers = 3
	if err := validatePublishConfig(layeredGoAV1Only, gitMetadata{Commit: "abc"}); err != nil {
		t.Fatalf("goav1-only layered publish config failed: %v", err)
	}

	sceneCutExternal := cfg
	sceneCutExternal.goav1SceneCut = true
	if err := validatePublishConfig(sceneCutExternal, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-goav1-scene-cut=false") {
		t.Fatalf("scene-cut external publish error=%v", err)
	}

	sceneCutGoAV1Only := cfg
	sceneCutGoAV1Only.encoders = []string{"goav1"}
	sceneCutGoAV1Only.goav1SceneCut = true
	if err := validatePublishConfig(sceneCutGoAV1Only, gitMetadata{Commit: "abc"}); err != nil {
		t.Fatalf("goav1-only scene-cut publish config failed: %v", err)
	}

	missingGoAV1ProcessTiming := cfg
	missingGoAV1ProcessTiming.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingGoAV1ProcessTiming.explicitFlags[k] = v
	}
	delete(missingGoAV1ProcessTiming.explicitFlags, "goav1-process-timing")
	if err := validatePublishConfig(missingGoAV1ProcessTiming, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-goav1-process-timing") {
		t.Fatalf("missing goav1 process timing error=%v", err)
	}

	inProcessGoAV1External := cfg
	inProcessGoAV1External.goav1ProcessTiming = false
	if err := validatePublishConfig(inProcessGoAV1External, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-goav1-process-timing=true") {
		t.Fatalf("in-process goav1 external publish error=%v", err)
	}

	inProcessGoAV1Only := cfg
	inProcessGoAV1Only.encoders = []string{"goav1"}
	inProcessGoAV1Only.goav1ProcessTiming = false
	if err := validatePublishConfig(inProcessGoAV1Only, gitMetadata{Commit: "abc"}); err != nil {
		t.Fatalf("goav1-only in-process publish config failed: %v", err)
	}

	missingDecoder := cfg
	missingDecoder.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingDecoder.explicitFlags[k] = v
	}
	delete(missingDecoder.explicitFlags, "ffmpeg-av1-decoder")
	if err := validatePublishConfig(missingDecoder, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-ffmpeg-av1-decoder") {
		t.Fatalf("missing explicit ffmpeg decoder error=%v", err)
	}

	emptyDecoder := cfg
	emptyDecoder.ffmpegAV1Decoder = ""
	if err := validatePublishConfig(emptyDecoder, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "non-empty -ffmpeg-av1-decoder") {
		t.Fatalf("empty ffmpeg decoder error=%v", err)
	}

	missingVMAFModel := cfg
	missingVMAFModel.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingVMAFModel.explicitFlags[k] = v
	}
	delete(missingVMAFModel.explicitFlags, "vmaf-model")
	if err := validatePublishConfig(missingVMAFModel, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-vmaf-model") {
		t.Fatalf("missing explicit vmaf model error=%v", err)
	}

	emptyVMAFModel := cfg
	emptyVMAFModel.vmafModel = ""
	if err := validatePublishConfig(emptyVMAFModel, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "non-empty -vmaf-model") {
		t.Fatalf("empty vmaf model error=%v", err)
	}

	coreTiming := cfg
	coreTiming.timingMode = timingModeCore
	if err := validatePublishConfig(coreTiming, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-timing-mode e2e") {
		t.Fatalf("core timing publish error=%v", err)
	}

	fixedOrder := cfg
	fixedOrder.runOrder = runOrderBitrateEncoder
	if err := validatePublishConfig(fixedOrder, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-run-order shuffle") {
		t.Fatalf("fixed-order publish error=%v", err)
	}

	shuffled := cfg
	shuffled.runOrder = runOrderShuffle
	shuffled.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		shuffled.explicitFlags[k] = v
	}
	delete(shuffled.explicitFlags, "shuffle-seed")
	if err := validatePublishConfig(shuffled, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-shuffle-seed") {
		t.Fatalf("implicit shuffle seed error=%v", err)
	}

	tooFewRuns := cfg
	tooFewRuns.runs = 2
	if err := validatePublishConfig(tooFewRuns, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-runs >= 3") {
		t.Fatalf("too few runs error=%v", err)
	}

	noWarmup := cfg
	noWarmup.warmupRuns = 0
	if err := validatePublishConfig(noWarmup, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-warmup-runs >= 1") {
		t.Fatalf("missing warmup error=%v", err)
	}

	missingMinSourceIDs := cfg
	missingMinSourceIDs.explicitFlags = map[string]bool{}
	for k, v := range cfg.explicitFlags {
		missingMinSourceIDs.explicitFlags[k] = v
	}
	delete(missingMinSourceIDs.explicitFlags, "min-source-ids")
	if err := validatePublishConfig(missingMinSourceIDs, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-min-source-ids") {
		t.Fatalf("missing min-source-ids error=%v", err)
	}

	tooFewSourceIDs := cfg
	tooFewSourceIDs.minSourceIDs = 1
	if err := validatePublishConfig(tooFewSourceIDs, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-min-source-ids >= 2") {
		t.Fatalf("too few source ids error=%v", err)
	}

	tooFewCategories := cfg
	tooFewCategories.minCategories = 1
	if err := validatePublishConfig(tooFewCategories, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-min-categories >= 2") {
		t.Fatalf("too few categories error=%v", err)
	}

	dirty := cfg
	if err := validatePublishConfig(dirty, gitMetadata{Commit: "abc", Dirty: true}); err == nil ||
		!strings.Contains(err.Error(), "clean") {
		t.Fatalf("dirty git error=%v", err)
	}

	notAll := cfg
	notAll.requiredEncodersRaw = "goav1,aomenc,svt-av1"
	if err := validatePublishConfig(notAll, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "-require-encoders all") {
		t.Fatalf("non-all required encoders error=%v", err)
	}

	tooFewBitrates := cfg
	tooFewBitrates.bitrates = []int{3000000, 6000000, 9000000}
	if err := validatePublishConfig(tooFewBitrates, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "four distinct -bitrates") {
		t.Fatalf("too few bitrates error=%v", err)
	}

	duplicateBitrates := cfg
	duplicateBitrates.bitrates = []int{3000000, 6000000, 6000000, 12000000}
	if err := validatePublishConfig(duplicateBitrates, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "each -bitrates entry") {
		t.Fatalf("duplicate bitrates error=%v", err)
	}

	duplicateEncoders := cfg
	duplicateEncoders.encoders = []string{"goav1", "aomenc", "aomenc"}
	if err := validatePublishConfig(duplicateEncoders, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "each -encoders entry") {
		t.Fatalf("duplicate encoders error=%v", err)
	}

	collidingMetadata := cfg
	collidingMetadata.metadataPath = collidingMetadata.csvPath
	if err := validatePublishConfig(collidingMetadata, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "different paths") {
		t.Fatalf("colliding metadata path error=%v", err)
	}

	collidingDiagnostics := cfg
	collidingDiagnostics.statsCSVPath = collidingDiagnostics.summaryCSVPath
	if err := validatePublishConfig(collidingDiagnostics, gitMetadata{Commit: "abc"}); err == nil ||
		!strings.Contains(err.Error(), "different paths") {
		t.Fatalf("colliding diagnostics path error=%v", err)
	}
}

func TestEncodeJobsRunOrder(t *testing.T) {
	cfg := benchConfig{
		encoders: []string{"goav1", "aomenc"},
		bitrates: []int{100, 200},
		runOrder: runOrderBitrateEncoder,
	}
	if got := encodeJobLabels(encodeJobsForConfig(cfg)); got != "100:goav1,100:aomenc,200:goav1,200:aomenc" {
		t.Fatalf("bitrate order=%s", got)
	}
	cfg.runOrder = runOrderEncoderBitrate
	if got := encodeJobLabels(encodeJobsForConfig(cfg)); got != "100:goav1,200:goav1,100:aomenc,200:aomenc" {
		t.Fatalf("encoder order=%s", got)
	}
	cfg.bitrates = []int{100, 200, 300}
	cfg.runOrder = runOrderShuffle
	cfg.shuffleSeed = 9
	first := encodeJobLabels(encodeJobsForConfig(cfg))
	second := encodeJobLabels(encodeJobsForConfig(cfg))
	if first != second {
		t.Fatalf("shuffle not deterministic: %s vs %s", first, second)
	}
	cfg.shuffleSeed = 10
	if third := encodeJobLabels(encodeJobsForConfig(cfg)); third == first {
		t.Fatalf("different shuffle seed produced same order: %s", third)
	}
}

func TestRunEncoderJobsMeasuredInterleavesSamplesByPass(t *testing.T) {
	cfg := benchConfig{
		workdir:    t.TempDir(),
		runs:       2,
		warmupRuns: 1,
	}
	jobs := []encodeJob{
		{bitrate: 100, encoder: "goav1"},
		{bitrate: 200, encoder: "aomenc"},
	}
	var calls []string
	results := runEncoderJobsMeasuredWithRunner(cfg, nil, "source.yuv", jobs, func(sampleCfg benchConfig, _ []goav1.I420Frame, refPath, encoderName string, bitrate int) encodeResult {
		if refPath != "source.yuv" {
			t.Fatalf("refPath=%q", refPath)
		}
		calls = append(calls, filepath.Base(sampleCfg.workdir))
		return encodeResult{
			encoder:       encoderName,
			targetBPS:     bitrate,
			duration:      time.Duration(len(calls)) * time.Millisecond,
			bytes:         int64(bitrate),
			encodedBytes:  int64(bitrate + 1),
			decodedBytes:  int64(bitrate + 2),
			encodedSHA256: strconv.Itoa(bitrate),
			decodedSHA256: strconv.Itoa(bitrate + 10),
			status:        "ok",
		}
	})
	wantCalls := []string{
		"goav1_100_warmup_01",
		"aomenc_200_warmup_01",
		"goav1_100_run_01",
		"aomenc_200_run_01",
		"goav1_100_run_02",
		"aomenc_200_run_02",
	}
	if strings.Join(calls, ",") != strings.Join(wantCalls, ",") {
		t.Fatalf("calls=%v want %v", calls, wantCalls)
	}
	if len(results) != 2 {
		t.Fatalf("results=%d", len(results))
	}
	for _, result := range results {
		if result.status != "ok" || result.runs != 2 || result.warmupRuns != 1 || len(result.samples) != 2 {
			t.Fatalf("result=%+v", result)
		}
	}
}

func TestRunEncoderJobsMeasuredPublishRequiresDeterministicArtifacts(t *testing.T) {
	cfg := benchConfig{
		workdir: t.TempDir(),
		runs:    3,
		publish: true,
	}
	jobs := []encodeJob{{bitrate: 100, encoder: "goav1"}}
	call := 0
	results := runEncoderJobsMeasuredWithRunner(cfg, nil, "source.yuv", jobs, func(sampleCfg benchConfig, _ []goav1.I420Frame, refPath, encoderName string, bitrate int) encodeResult {
		call++
		if refPath != "source.yuv" {
			t.Fatalf("refPath=%q", refPath)
		}
		encodedHash := "encoded-same"
		if call == 2 {
			encodedHash = "encoded-drift"
		}
		return encodeResult{
			encoder:       encoderName,
			targetBPS:     bitrate,
			duration:      time.Duration(call) * time.Millisecond,
			bytes:         10,
			encodedBytes:  20,
			decodedBytes:  30,
			encodedSHA256: encodedHash,
			decodedSHA256: "decoded-same",
			status:        "ok",
			encodedPath:   filepath.Join(sampleCfg.workdir, "encoded.ivf"),
			decodedYUV:    filepath.Join(sampleCfg.workdir, "decoded.yuv"),
			cpuAvailable:  true,
			cpuUser:       time.Duration(call) * time.Millisecond,
		}
	})
	if len(results) != 1 {
		t.Fatalf("results=%d", len(results))
	}
	got := results[0]
	if got.status != "error" || !strings.Contains(got.errText, "deterministic") ||
		!strings.Contains(got.errText, "encoded artifact sha256") {
		t.Fatalf("publish drift result=%+v", got)
	}
	if got.runs != 3 || len(got.samples) != 3 {
		t.Fatalf("summary runs=%d samples=%d", got.runs, len(got.samples))
	}
}

func TestRunEncoderJobsMeasuredPublishRequiresCPUBudgetEvidence(t *testing.T) {
	cfg := benchConfig{
		workdir: t.TempDir(),
		runs:    3,
		publish: true,
	}
	jobs := []encodeJob{{bitrate: 100, encoder: "goav1"}}
	call := 0
	results := runEncoderJobsMeasuredWithRunner(cfg, nil, "source.yuv", jobs, func(sampleCfg benchConfig, _ []goav1.I420Frame, refPath, encoderName string, bitrate int) encodeResult {
		call++
		if refPath != "source.yuv" {
			t.Fatalf("refPath=%q", refPath)
		}
		return encodeResult{
			encoder:       encoderName,
			targetBPS:     bitrate,
			duration:      time.Duration(call) * time.Millisecond,
			bytes:         10,
			encodedBytes:  20,
			decodedBytes:  30,
			encodedSHA256: "encoded-same",
			decodedSHA256: "decoded-same",
			status:        "ok",
			encodedPath:   filepath.Join(sampleCfg.workdir, "encoded.ivf"),
			decodedYUV:    filepath.Join(sampleCfg.workdir, "decoded.yuv"),
		}
	})
	if len(results) != 1 {
		t.Fatalf("results=%d", len(results))
	}
	got := results[0]
	if got.status != "error" || !strings.Contains(got.errText, "process CPU timing") {
		t.Fatalf("missing CPU evidence result=%+v", got)
	}
	if got.runs != 3 || len(got.samples) != 3 {
		t.Fatalf("summary runs=%d samples=%d", got.runs, len(got.samples))
	}
}

func writeTestExecutableWithSHA256(t *testing.T, name string) (string, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	hash, err := sha256File(path)
	if err != nil {
		t.Fatal(err)
	}
	return path, hash
}

func clearQualitybenchPublishGoEnv(t *testing.T) {
	t.Helper()
	for _, name := range benchenv.PublishAmbientGoEnvVars() {
		t.Setenv(name, "")
	}
}

func stubQualitybenchCPUState(t *testing.T, state benchenv.CPUState) {
	t.Helper()
	old := observeBenchmarkCPUState
	observeBenchmarkCPUState = func() benchenv.CPUState { return state }
	t.Cleanup(func() { observeBenchmarkCPUState = old })
}

func encodeJobLabels(jobs []encodeJob) string {
	labels := make([]string, len(jobs))
	for i, job := range jobs {
		labels[i] = strconv.Itoa(job.bitrate) + ":" + job.encoder
	}
	return strings.Join(labels, ",")
}

func TestAttachEncodeRunSummaryUsesMedianWallSample(t *testing.T) {
	results := []encodeResult{
		{encoder: "goav1", selectedRun: 1, duration: 300 * time.Millisecond, bytes: 30, status: "ok"},
		{encoder: "goav1", selectedRun: 2, duration: 100 * time.Millisecond, bytes: 10, status: "ok"},
		{encoder: "goav1", selectedRun: 3, duration: 200 * time.Millisecond, bytes: 20, status: "ok"},
	}
	selected := attachEncodeRunSummary(medianEncodeResult(results), results, 1, 3)
	if selected.selectedRun != 3 || selected.duration != 200*time.Millisecond {
		t.Fatalf("selected run=%d duration=%s", selected.selectedRun, selected.duration)
	}
	if selected.warmupRuns != 1 || selected.runs != 3 ||
		selected.minWall != 100*time.Millisecond ||
		selected.medianWall != 200*time.Millisecond ||
		selected.maxWall != 300*time.Millisecond ||
		selected.iqrWall != 200*time.Millisecond {
		t.Fatalf("summary warmup=%d runs=%d min=%s median=%s max=%s iqr=%s",
			selected.warmupRuns, selected.runs, selected.minWall, selected.medianWall, selected.maxWall, selected.iqrWall)
	}
	if len(selected.samples) != 3 || selected.samples[0].Run != 1 ||
		selected.samples[1].EncodeWallSecs != 0.1 ||
		selected.samples[2].CompressedBytes != 20 {
		t.Fatalf("samples=%+v", selected.samples)
	}
}

func copyStringBoolMap(src map[string]bool) map[string]bool {
	dst := make(map[string]bool, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func TestValidatePublishClipInputsRequiresExactSize(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.yuv")
	if err := os.WriteFile(good, make([]byte, expectedRawI420Bytes(16, 16, 2)), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := validatePublishClipInputs([]clipSpec{{
		Name:   "good",
		Input:  good,
		Width:  16,
		Height: 16,
		Frames: 2,
	}}); err != nil {
		t.Fatalf("valid clip failed: %v", err)
	}

	trailing := filepath.Join(dir, "trailing.yuv")
	if err := os.WriteFile(trailing, make([]byte, expectedRawI420Bytes(16, 16, 2)+1), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := validatePublishClipInputs([]clipSpec{{
		Name:   "trailing",
		Input:  trailing,
		Width:  16,
		Height: 16,
		Frames: 2,
	}}); err == nil || !strings.Contains(err.Error(), "exact raw I420") {
		t.Fatalf("trailing bytes error=%v", err)
	}
}

func TestValidatePublishWorkdirRequiresEmptyDirectory(t *testing.T) {
	base := t.TempDir()
	missing := filepath.Join(base, "missing")
	if err := validatePublishWorkdir(missing); err != nil {
		t.Fatalf("missing workdir should be creatable: %v", err)
	}
	filePath := filepath.Join(base, "file")
	if err := os.WriteFile(filePath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := validatePublishWorkdir(filePath); err == nil ||
		!strings.Contains(err.Error(), "not a directory") {
		t.Fatalf("file workdir error=%v", err)
	}
	nonEmpty := filepath.Join(base, "non-empty")
	if err := os.Mkdir(nonEmpty, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nonEmpty, "old.csv"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := validatePublishWorkdir(nonEmpty); err == nil ||
		!strings.Contains(err.Error(), "empty before timing") {
		t.Fatalf("non-empty workdir error=%v", err)
	}
	empty := filepath.Join(base, "empty")
	if err := os.Mkdir(empty, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := validatePublishWorkdir(empty); err != nil {
		t.Fatalf("empty workdir failed: %v", err)
	}
}

func TestSIMDMetadataFor(t *testing.T) {
	got := simdFeaturesFor(cpufeatures.Features{SSE2: true, SSE41: true, AVX2: true})
	if strings.Join(got, ",") != "sse2,sse4_1,avx2" {
		t.Fatalf("x86 simd features=%v", got)
	}
	if tier := simdTierFor(cpufeatures.Features{NEON: true, DOTPROD: true, I8MM: true}); tier != "neon_i8mm" {
		t.Fatalf("arm simd tier=%q", tier)
	}
	if tier := simdTierFor(cpufeatures.Features{}); tier != "purego" {
		t.Fatalf("purego tier=%q", tier)
	}
}

func TestGitDirtyFromStatusCountsUntracked(t *testing.T) {
	if gitDirtyFromStatus(nil) {
		t.Fatal("empty status reported dirty")
	}
	if !gitDirtyFromStatus([]byte(" M cmd/qualitybench/main.go\n")) {
		t.Fatal("modified tracked file reported clean")
	}
	if !gitDirtyFromStatus([]byte("?? local_experiment.go\n")) {
		t.Fatal("untracked source file reported clean")
	}
}

func TestClipMetadataForHashesInputs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "clip.yuv")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := clipMetadataFor([]clipSpec{{
		Name:   "clip",
		Input:  path,
		Width:  2,
		Height: 2,
		Frames: 1,
		FPS:    30,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("metadata rows=%d", len(got))
	}
	if got[0].Synthetic || got[0].InputBytes != 3 ||
		got[0].ExpectedBytes != 6 ||
		got[0].SHA256 != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
		t.Fatalf("clip metadata=%+v", got[0])
	}

	got, err = clipMetadataFor([]clipSpec{{Name: "synthetic", Width: 16, Height: 16, Frames: 2, FPS: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !got[0].Synthetic || got[0].SHA256 != "" || got[0].ExpectedBytes != 768 {
		t.Fatalf("synthetic metadata=%+v", got)
	}
}

func TestFileBytesAndSHA256(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.bin")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	bytes, hash, err := fileBytesAndSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != 3 || hash != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" {
		t.Fatalf("bytes=%d hash=%s", bytes, hash)
	}
}

func TestDecodedYUVMetadataRequiresExactSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "decoded.yuv")
	exact := expectedRawI420Bytes(16, 16, 2)
	if err := os.WriteFile(path, make([]byte, exact), 0o644); err != nil {
		t.Fatal(err)
	}
	bytes, hash, err := decodedYUVMetadata(path, 16, 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != exact || hash == "" {
		t.Fatalf("metadata bytes=%d hash=%q", bytes, hash)
	}
	if err := os.WriteFile(path, make([]byte, exact-1), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := decodedYUVMetadata(path, 16, 16, 2); err == nil || !strings.Contains(err.Error(), "exact raw I420") {
		t.Fatalf("short decoded size error=%v", err)
	}
}

func TestCommandMetadataRecordsBinaryHash(t *testing.T) {
	meta := commandMetadata("go", []string{"version"})
	if !meta.Found || meta.Path == "" || meta.SHA256 == "" || meta.Version == "" {
		t.Fatalf("go metadata=%+v", meta)
	}
}

func TestWriteMetadataJSON(t *testing.T) {
	stubQualitybenchCPUState(t, benchenv.CPUState{
		GOOS:                "test",
		AffinitySupported:   true,
		AffinityAllowedList: "0-3",
		CPUOnlineList:       "0-3",
	})
	dir := t.TempDir()
	path := filepath.Join(dir, "metadata.json")
	input := filepath.Join(dir, "clip.yuv")
	if err := os.WriteFile(input, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(dir, "clips.csv")
	if err := os.WriteFile(manifest, []byte("clip,input,width,height,frames,fps\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	manifestSHA, err := sha256File(manifest)
	if err != nil {
		t.Fatal(err)
	}
	vmafModel := filepath.Join(dir, "vmaf.json")
	if err := os.WriteFile(vmafModel, []byte(`{"model":"fixture"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	vmafModelSHA, err := sha256File(vmafModel)
	if err != nil {
		t.Fatal(err)
	}
	cfg := benchConfig{
		width:            64,
		height:           64,
		frames:           2,
		fps:              30,
		manifestPath:     manifest,
		metadataPath:     path,
		encoders:         []string{"goav1"},
		bitrates:         []int{100000},
		requiredMetrics:  []string{"psnr"},
		requiredEncoders: []string{"goav1"},
		requireSummary:   true,
		requireCorpus:    true,
		minClips:         6,
		anchorEncoder:    "goav1",
		layers:           1,
		environmentNotes: "fixed power mode",
		cpuAffinity:      "none",
		powerMode:        "high power",
		thermalState:     "cool start",
		frequencyPolicy:  "automatic",
		backgroundLoad:   "idle machine",
		commandTimeout:   45 * time.Second,
		vmafModel:        "path=" + vmafModel,
	}
	invocations := []encoderInvocationMetadata{{
		Clip:             "clip",
		Width:            64,
		Height:           64,
		Frames:           2,
		FPS:              30,
		Encoder:          "goav1",
		TargetBPS:        100000,
		ActualBPS:        96000,
		CompressedBytes:  800,
		EncodedContainer: "goav1-payload-stream",
		EncodedBytes:     800,
		EncodedSHA256:    "encoded",
		DecodedPath:      filepath.Join(dir, "decoded.yuv"),
		DecodedBytes:     12288,
		DecodedSHA256:    "decoded",
		Status:           "ok",
		Settings:         map[string]string{"target_bitrate": "100000"},
	}}
	clips := []clipSpec{{
		Name:   "clip",
		Input:  input,
		Width:  64,
		Height: 64,
		Frames: 2,
		FPS:    30,
	}}
	git := gitMetadata{Commit: "snapshot", Dirty: false}
	t.Setenv("OMP_NUM_THREADS", "99")
	if err := writeMetadataJSON(cfg, map[string]bool{"psnr": true, "ssim": false}, git, clips, invocations); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc qualitybenchMetadata
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.GeneratedAtUTC == "" || doc.Go.Version == "" || doc.Config.Width != 64 {
		t.Fatalf("metadata header=%+v", doc)
	}
	if doc.Git.Commit != "snapshot" || doc.Git.Dirty {
		t.Fatalf("git snapshot=%+v", doc.Git)
	}
	if doc.Go.SIMDTier == "" {
		t.Fatalf("missing simd metadata: %+v", doc.Go)
	}
	if doc.Go.ToolPath == "" || doc.Go.ToolSHA256 == "" || doc.Tools["go"].SHA256 == "" {
		t.Fatalf("missing go tool metadata: go=%+v tools=%+v", doc.Go, doc.Tools)
	}
	if doc.Environment.GOMAXPROCS <= 0 || doc.Environment.NumCPU <= 0 {
		t.Fatalf("environment metadata=%+v", doc.Environment)
	}
	if doc.Environment.ObservedCPUState.AffinityAllowedList != "0-3" ||
		doc.Environment.ObservedCPUState.CPUOnlineList != "0-3" {
		t.Fatalf("observed cpu state=%+v", doc.Environment.ObservedCPUState)
	}
	if doc.Environment.ExternalCommandEnvPolicy == "" ||
		doc.Environment.ExternalCommandEnv["LC_ALL"] != "C" ||
		doc.Environment.ExternalCommandEnv["TZ"] != "UTC" {
		t.Fatalf("external command env metadata=%+v", doc.Environment)
	}
	foundFiltered := false
	for _, name := range doc.Environment.ExternalCommandFilteredEnv {
		if name == "OMP_NUM_THREADS" {
			foundFiltered = true
			break
		}
	}
	if !foundFiltered {
		t.Fatalf("filtered external env metadata=%+v", doc.Environment.ExternalCommandFilteredEnv)
	}
	if doc.Environment.Notes != "fixed power mode" ||
		doc.Environment.CPUAffinity != "none" ||
		doc.Environment.PowerMode != "high power" ||
		doc.Environment.ThermalState != "cool start" ||
		doc.Environment.FrequencyPolicy != "automatic" ||
		doc.Environment.BackgroundLoad != "idle machine" {
		t.Fatalf("environment metadata=%+v", doc.Environment)
	}
	if doc.Config.ManifestSHA256 != manifestSHA {
		t.Fatalf("manifest sha=%q want %q", doc.Config.ManifestSHA256, manifestSHA)
	}
	if doc.Config.SampleOrder != "interleaved-by-sample-pass" {
		t.Fatalf("sample order=%q", doc.Config.SampleOrder)
	}
	if doc.Config.CommandTimeout != "45s" {
		t.Fatalf("command timeout=%q", doc.Config.CommandTimeout)
	}
	if doc.Config.VMAFModelPath != vmafModel || doc.Config.VMAFModelSHA256 != vmafModelSHA {
		t.Fatalf("vmaf model path=%q sha=%q want %q %q",
			doc.Config.VMAFModelPath, doc.Config.VMAFModelSHA256, vmafModel, vmafModelSHA)
	}
	if len(doc.Config.RequiredEncoders) != 1 || doc.Config.RequiredEncoders[0] != "goav1" {
		t.Fatalf("required encoders=%+v", doc.Config.RequiredEncoders)
	}
	if !doc.Config.RequireSummary {
		t.Fatalf("require summary=%v", doc.Config.RequireSummary)
	}
	if !doc.Config.RequireCorpus || doc.Config.MinClips != 6 {
		t.Fatalf("corpus config require=%v min=%d", doc.Config.RequireCorpus, doc.Config.MinClips)
	}
	if len(doc.Clips) != 1 || doc.Clips[0].Name != "clip" || doc.Clips[0].SHA256 == "" ||
		doc.Clips[0].ExpectedBytes != 12288 || doc.Clips[0].InputBytes != 3 {
		t.Fatalf("clips=%+v", doc.Clips)
	}
	if !doc.MetricFilters["psnr"] || doc.MetricFilters["libvmaf"] {
		t.Fatalf("metric filters=%+v", doc.MetricFilters)
	}
	if len(doc.Encodes) != 1 || doc.Encodes[0].Encoder != "goav1" ||
		doc.Encodes[0].Settings["target_bitrate"] != "100000" ||
		doc.Encodes[0].CompressedBytes != 800 ||
		doc.Encodes[0].EncodedContainer != "goav1-payload-stream" ||
		doc.Encodes[0].EncodedSHA256 != "encoded" ||
		doc.Encodes[0].DecodedSHA256 != "decoded" {
		t.Fatalf("encodes=%+v", doc.Encodes)
	}
	if _, ok := doc.Tools["ffmpeg"]; !ok {
		t.Fatalf("tools=%+v", doc.Tools)
	}
}

func TestCommandTimeoutReportsDeadline(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "sleepy")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nsleep 2\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	var result encodeResult
	elapsed := timeCommand(10*time.Millisecond, bin, nil, &result)
	if result.status != "error" || !strings.Contains(result.errText, "command timed out after 10ms") {
		t.Fatalf("timeout result=%+v", result)
	}
	if elapsed > time.Second {
		t.Fatalf("timeout elapsed=%s, want prompt cancellation", elapsed)
	}

	out, err := combinedOutputWithTimeout(10*time.Millisecond, bin)
	if err == nil || !strings.Contains(err.Error(), "command timed out after 10ms") {
		t.Fatalf("combined timeout err=%v out=%q", err, out)
	}
}
