package main

import (
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	goav1 "github.com/thesyncim/goav1"
	cpufeatures "github.com/thesyncim/goav1/internal/av1/dsp/cpu"
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
		clips[0].Frames != 120 || clips[0].FPS != 60 ||
		clips[0].PixFmt != "i420" || clips[0].BitDepth != 8 ||
		clips[0].Chroma != "4:2:0" || clips[0].SourceID != "lab-head" ||
		clips[0].SourceURL != "https://example.invalid/head" ||
		clips[0].SourceLicense != "CC-BY-4.0" ||
		clips[0].Category != "talking-head" {
		t.Fatalf("clip[0]=%+v", clips[0])
	}
	if clips[1].Name != "Synthetic" || clips[1].Input != "" || clips[1].FPS != 30 {
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
	}
	clips := []clipSpec{
		{Name: "first", Input: first},
		{Name: "second", Input: second},
	}
	if err := validateRequiredCorpus(cfg, clips); err != nil {
		t.Fatalf("valid corpus failed: %v", err)
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
	var data []byte
	header := make([]byte, 32)
	copy(header[:4], "DKIF")
	data = append(data, header...)
	for _, size := range []uint32{3, 5} {
		frame := make([]byte, 12)
		binary.LittleEndian.PutUint32(frame[:4], size)
		data = append(data, frame...)
		data = append(data, make([]byte, size)...)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ivfPayloadBytes(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != 8 {
		t.Fatalf("payload bytes=%d want 8", got)
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
	got, err := parseVMAFMean([]byte(`{"pooled_metrics":{"vmaf":{"mean":91.25}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-91.25) > 1e-9 {
		t.Fatalf("vmaf=%f", got)
	}
	if _, err := parseVMAFMean([]byte(`{"pooled_metrics":{}}`)); err == nil {
		t.Fatal("missing vmaf accepted")
	}
}

func TestParseFrameMetricValues(t *testing.T) {
	raw := []byte(`[Parsed_psnr_0 @ 0x1] n:1 mse_avg:10.0 psnr_avg:38.123456
[Parsed_psnr_0 @ 0x1] n:2 mse_avg:0.0 psnr_avg:inf
`)
	got, err := parseFrameMetricValues(raw, `psnr_avg:([0-9.]+|inf|Inf|INF)`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].FrameIndex != 0 || got[0].Value != "38.1235" ||
		got[1].FrameIndex != 1 || got[1].Value != "inf" {
		t.Fatalf("values=%+v", got)
	}

	raw = []byte(`[Parsed_ssim_0 @ 0x1] n:7 Y:0.900000 U:0.950000 V:0.960000 All:0.912345 (10.1)
`)
	got, err = parseFrameMetricValues(raw, `All:([0-9.]+|inf|Inf|INF)`)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].FrameIndex != 6 || got[0].Value != "0.9123" {
		t.Fatalf("values=%+v", got)
	}

	if _, err := parseFrameMetricValues([]byte("summary only"), `All:([0-9.]+)`); err == nil {
		t.Fatal("missing frame values accepted")
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
		width:            64,
		height:           64,
		frames:           4,
		fps:              30,
		encoders:         []string{"goav1"},
		bitrates:         []int{100000},
		requiredMetrics:  []string{"psnr"},
		requiredEncoders: []string{"goav1"},
		requireSummary:   true,
		requireCorpus:    true,
		minClips:         6,
		anchorEncoder:    "goav1",
		goMaxProcs:       4,
		goav1MaxThreads:  4,
		goav1Effort:      int(goav1.EncoderWebRTCMinEffortLevel),
		aomThreads:       1,
		aomRowMT:         0,
		aomCPUUsed:       8,
		svtLP:            5,
		svtPreset:        13,
		timingMode:       timingModeEndToEnd,
		runOrder:         runOrderShuffle,
		shuffleSeed:      42,
		runs:             5,
		warmupRuns:       1,
		publish:          true,
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
		got.GoMaxProcs != 4 || got.GoAV1MaxThreads != 4 ||
		got.GoAV1Effort != int(goav1.EncoderWebRTCMinEffortLevel) ||
		got.AOMThreads != 1 || got.AOMRowMT != 0 ||
		got.AOMCPUUsed != 8 || got.SVTLP != 5 || got.SVTPreset != 13 ||
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
		!strings.Contains(joined, "run_order") ||
		!strings.Contains(joined, "explicit seed") ||
		!strings.Contains(joined, "sample passes") ||
		!strings.Contains(joined, "median wall-time") ||
		!strings.Contains(joined, "sweep --lp 0..6") ||
		!strings.Contains(joined, "-goav1-max-threads") ||
		!strings.Contains(joined, "-goav1-effort") ||
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

func TestValidatePublishConfigRequiresExplicitControls(t *testing.T) {
	cfg := benchConfig{
		workdir:             "/tmp/work",
		csvPath:             "/tmp/quality.csv",
		metadataPath:        "/tmp/meta.json",
		manifestPath:        "/tmp/clips.csv",
		summaryCSVPath:      "/tmp/summary.csv",
		anchorEncoder:       "aomenc",
		requiredEncodersRaw: "all",
		encoders:            []string{"goav1", "aomenc", "svt-av1"},
		bitrates:            []int{3000000, 6000000, 9000000, 12000000},
		requiredMetrics:     []string{"psnr", "ssim"},
		requireCorpus:       true,
		minClips:            2,
		requireSummary:      true,
		fps:                 60,
		goMaxProcs:          4,
		goav1MaxThreads:     4,
		goav1Effort:         0,
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
		timingMode:          timingModeEndToEnd,
		runOrder:            runOrderShuffle,
		shuffleSeed:         7,
		runs:                3,
		warmupRuns:          1,
		explicitFlags: map[string]bool{
			"bitrates":          true,
			"encoders":          true,
			"workdir":           true,
			"csv":               true,
			"metadata-json":     true,
			"manifest":          true,
			"require-corpus":    true,
			"min-clips":         true,
			"require-encoders":  true,
			"require-metrics":   true,
			"summary-csv":       true,
			"require-summary":   true,
			"gomaxprocs":        true,
			"fps":               true,
			"layers":            true,
			"tiles":             true,
			"golden":            true,
			"keyint":            true,
			"anchor":            true,
			"timing-mode":       true,
			"run-order":         true,
			"shuffle-seed":      true,
			"runs":              true,
			"warmup-runs":       true,
			"goav1-max-threads": true,
			"goav1-effort":      true,
			"aom-cpu-used":      true,
			"aom-threads":       true,
			"aom-row-mt":        true,
			"svt-preset":        true,
			"svt-lp":            true,
			"svt-asm":           true,
		},
	}
	if err := validatePublishConfig(cfg, gitMetadata{Commit: "abc"}); err != nil {
		t.Fatalf("valid publish config failed: %v", err)
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
	if doc.Environment.GOMAXPROCS <= 0 || doc.Environment.NumCPU <= 0 {
		t.Fatalf("environment metadata=%+v", doc.Environment)
	}
	if doc.Config.ManifestSHA256 != manifestSHA {
		t.Fatalf("manifest sha=%q want %q", doc.Config.ManifestSHA256, manifestSHA)
	}
	if doc.Config.SampleOrder != "interleaved-by-sample-pass" {
		t.Fatalf("sample order=%q", doc.Config.SampleOrder)
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
