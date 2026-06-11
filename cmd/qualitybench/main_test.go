package main

import (
	"bytes"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	goav1 "github.com/thesyncim/goav1"
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

func TestReadClipManifest(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "clips.csv")
	if err := os.WriteFile(manifest, []byte("clip,input,width,height,frames,fps\nTalking Head,clips/head.yuv,1920,1080,120,60\nSynthetic,,320,180,30,\n"), 0o644); err != nil {
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
		clips[0].Frames != 120 || clips[0].FPS != 60 {
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

func TestMetadataConfigCopiesSlices(t *testing.T) {
	cfg := benchConfig{
		width:           64,
		height:          64,
		frames:          4,
		fps:             30,
		encoders:        []string{"goav1"},
		bitrates:        []int{100000},
		requiredMetrics: []string{"psnr"},
		anchorEncoder:   "goav1",
	}
	got := metadataConfigFor(cfg)
	cfg.encoders[0] = "mutated"
	cfg.bitrates[0] = 1
	cfg.requiredMetrics[0] = "vmaf"
	if got.Encoders[0] != "goav1" || got.Bitrates[0] != 100000 || got.RequiredMetrics[0] != "psnr" {
		t.Fatalf("metadata config aliases inputs: %+v", got)
	}
}

func TestWriteMetadataJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "metadata.json")
	cfg := benchConfig{
		width:           64,
		height:          64,
		frames:          2,
		fps:             30,
		metadataPath:    path,
		encoders:        []string{"goav1"},
		bitrates:        []int{100000},
		requiredMetrics: []string{"psnr"},
		anchorEncoder:   "goav1",
		layers:          1,
	}
	invocations := []encoderInvocationMetadata{{
		Clip:      "clip",
		Width:     64,
		Height:    64,
		Frames:    2,
		FPS:       30,
		Encoder:   "goav1",
		TargetBPS: 100000,
		ActualBPS: 96000,
		Status:    "ok",
		Settings:  map[string]string{"target_bitrate": "100000"},
	}}
	if err := writeMetadataJSON(cfg, map[string]bool{"psnr": true, "ssim": false}, invocations); err != nil {
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
	if !doc.MetricFilters["psnr"] || doc.MetricFilters["libvmaf"] {
		t.Fatalf("metric filters=%+v", doc.MetricFilters)
	}
	if len(doc.Encodes) != 1 || doc.Encodes[0].Encoder != "goav1" ||
		doc.Encodes[0].Settings["target_bitrate"] != "100000" {
		t.Fatalf("encodes=%+v", doc.Encodes)
	}
	if _, ok := doc.Tools["ffmpeg"]; !ok {
		t.Fatalf("tools=%+v", doc.Tools)
	}
}
