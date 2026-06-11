package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
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
