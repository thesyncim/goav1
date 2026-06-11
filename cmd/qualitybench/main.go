// Command qualitybench compares goav1 against installed realtime AV1 baselines
// on the same raw I420 source frames, then measures decoded output with FFmpeg
// objective metrics.
package main

import (
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	goav1 "github.com/thesyncim/goav1"
)

const (
	defaultWidth  = 1920
	defaultHeight = 1080
	defaultFrames = 120
	defaultFPS    = 60
)

type benchConfig struct {
	width          int
	height         int
	frames         int
	fps            int
	input          string
	workdir        string
	csvPath        string
	summaryCSVPath string
	anchorEncoder  string
	encoders       []string
	bitrates       []int
	layers         int
	tiles          int
	goldenInterval int
	keyInterval    int
	keep           bool
}

type encodeResult struct {
	encoder    string
	targetBPS  int
	bytes      int64
	duration   time.Duration
	decodedYUV string
	status     string
	errText    string
}

type metrics struct {
	psnr  string
	ssim  string
	xpsnr string
	vmaf  string
}

type benchRow struct {
	clip      string
	width     int
	height    int
	frames    int
	fps       int
	encoder   string
	targetBPS int
	actualBPS int64
	encodeFPS string
	bytes     int64
	metrics   metrics
	status    string
	errText   string
}

type rdPoint struct {
	Metric float64
	Rate   float64
}

type cubicFit struct {
	Coeff  [4]float64
	Center float64
	Scale  float64
}

type summaryRow struct {
	Clip          string
	Anchor        string
	Encoder       string
	Metric        string
	AnchorPoints  int
	EncoderPoints int
	OverlapMin    float64
	OverlapMax    float64
	BDRatePct     float64
	Status        string
	ErrText       string
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := parseFlags()
	if err != nil {
		return err
	}
	cleanup := false
	if cfg.workdir == "" {
		cfg.workdir, err = os.MkdirTemp("", "goav1-qualitybench-")
		if err != nil {
			return err
		}
		cleanup = !cfg.keep
	} else if err := os.MkdirAll(cfg.workdir, 0o755); err != nil {
		return err
	}
	if cleanup {
		defer os.RemoveAll(cfg.workdir)
	}

	frames, clipName, err := loadFrames(cfg)
	if err != nil {
		return err
	}
	refPath := filepath.Join(cfg.workdir, "source.yuv")
	if err := writeFrames(refPath, frames, cfg.width, cfg.height); err != nil {
		return err
	}

	filters := ffmpegFilters()
	var out io.Writer = os.Stdout
	var csvFile *os.File
	if cfg.csvPath != "" {
		csvFile, err = os.Create(cfg.csvPath)
		if err != nil {
			return err
		}
		defer csvFile.Close()
		out = csvFile
	}
	writer := csv.NewWriter(out)
	defer writer.Flush()
	if err := writer.Write([]string{
		"clip", "width", "height", "frames", "fps", "encoder", "target_bps",
		"actual_bps", "encode_fps", "bytes", "psnr_avg", "ssim_all",
		"xpsnr_y", "vmaf", "status", "error",
	}); err != nil {
		return err
	}

	var rows []benchRow
	for _, bitrate := range cfg.bitrates {
		for _, encoderName := range cfg.encoders {
			result := runEncoder(cfg, frames, refPath, encoderName, bitrate)
			m := metrics{psnr: "NA", ssim: "NA", xpsnr: "NA", vmaf: "NA"}
			if result.status == "ok" {
				m = measureDecoded(cfg, filters, refPath, result.decodedYUV, encoderName, bitrate)
			}
			actualBPS := int64(0)
			if result.bytes > 0 && cfg.frames > 0 {
				actualBPS = result.bytes * 8 * int64(cfg.fps) / int64(cfg.frames)
			}
			encodeFPS := ""
			if result.duration > 0 {
				encodeFPS = strconv.FormatFloat(float64(cfg.frames)/result.duration.Seconds(), 'f', 2, 64)
			}
			row := benchRow{
				clip:      clipName,
				width:     cfg.width,
				height:    cfg.height,
				frames:    cfg.frames,
				fps:       cfg.fps,
				encoder:   result.encoder,
				targetBPS: result.targetBPS,
				actualBPS: actualBPS,
				encodeFPS: encodeFPS,
				bytes:     result.bytes,
				metrics:   m,
				status:    result.status,
				errText:   result.errText,
			}
			rows = append(rows, row)
			if err := writeBenchRow(writer, row); err != nil {
				return err
			}
			writer.Flush()
		}
	}
	if cfg.summaryCSVPath != "" {
		if err := writeSummaryCSV(cfg, rows); err != nil {
			return err
		}
	}
	if cfg.csvPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.csvPath)
	}
	if cfg.summaryCSVPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.summaryCSVPath)
	}
	if cfg.keep || cfg.workdir != "" && !cleanup {
		fmt.Fprintf(os.Stderr, "qualitybench workdir: %s\n", cfg.workdir)
	}
	return writer.Error()
}

func parseFlags() (benchConfig, error) {
	var cfg benchConfig
	bitrates := flag.String("bitrates", "3000000,6000000,9000000,12000000", "comma-separated target bitrates in bits per second")
	encoders := flag.String("encoders", "goav1,aomenc,svt-av1", "comma-separated encoders: goav1,aomenc,svt-av1")
	flag.StringVar(&cfg.input, "input", "", "raw I420 input file; omit to use the deterministic synthetic scene")
	flag.IntVar(&cfg.width, "width", defaultWidth, "frame width in pixels")
	flag.IntVar(&cfg.height, "height", defaultHeight, "frame height in pixels")
	flag.IntVar(&cfg.frames, "frames", defaultFrames, "frames to encode")
	flag.IntVar(&cfg.fps, "fps", defaultFPS, "input frame rate and bitrate timebase")
	flag.IntVar(&cfg.layers, "layers", 1, "goav1 temporal layers (1, 2, or 3)")
	flag.IntVar(&cfg.tiles, "tiles", 0, "tile-column log2 override for encoders that expose one")
	flag.IntVar(&cfg.goldenInterval, "golden", 0, "goav1 golden refresh interval (0 = default, negative = disabled)")
	flag.IntVar(&cfg.keyInterval, "keyint", 0, "force periodic keyframes every N frames after frame 0 (0 = only initial key)")
	flag.StringVar(&cfg.workdir, "workdir", "", "directory for raw, decoded, and encoded intermediates")
	flag.StringVar(&cfg.csvPath, "csv", "", "write CSV to this path instead of stdout")
	flag.StringVar(&cfg.summaryCSVPath, "summary-csv", "", "write BD-rate summary CSV to this path")
	flag.StringVar(&cfg.anchorEncoder, "anchor", "", "encoder name to use as BD-rate anchor (default: first -encoders entry)")
	flag.BoolVar(&cfg.keep, "keep", false, "keep the temporary workdir when -workdir is not set")
	flag.Parse()

	var err error
	cfg.bitrates, err = parsePositiveList(*bitrates)
	if err != nil {
		return benchConfig{}, fmt.Errorf("bitrates: %w", err)
	}
	cfg.encoders = parseNameList(*encoders)
	if len(cfg.encoders) == 0 {
		return benchConfig{}, errors.New("no encoders selected")
	}
	if cfg.anchorEncoder == "" {
		cfg.anchorEncoder = cfg.encoders[0]
	} else {
		cfg.anchorEncoder = strings.TrimSpace(strings.ToLower(cfg.anchorEncoder))
	}
	if cfg.width < 16 || cfg.height < 16 || cfg.width%2 != 0 || cfg.height%2 != 0 {
		return benchConfig{}, fmt.Errorf("invalid frame size %dx%d: need even dimensions >= 16", cfg.width, cfg.height)
	}
	if cfg.frames <= 0 {
		return benchConfig{}, fmt.Errorf("invalid frame count %d", cfg.frames)
	}
	if cfg.fps <= 0 {
		return benchConfig{}, fmt.Errorf("invalid fps %d", cfg.fps)
	}
	if cfg.layers < 1 || cfg.layers > 3 {
		return benchConfig{}, fmt.Errorf("invalid temporal layers %d", cfg.layers)
	}
	if cfg.keyInterval < 0 {
		return benchConfig{}, fmt.Errorf("invalid key interval %d", cfg.keyInterval)
	}
	return cfg, nil
}

func parsePositiveList(s string) ([]int, error) {
	fields := strings.Split(s, ",")
	out := make([]int, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field == "" {
			continue
		}
		n, err := strconv.Atoi(field)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("invalid positive integer %q", field)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, errors.New("empty list")
	}
	return out, nil
}

func parseNameList(s string) []string {
	fields := strings.Split(s, ",")
	out := make([]string, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(strings.ToLower(field))
		if field != "" {
			out = append(out, field)
		}
	}
	return out
}

func writeBenchRow(writer *csv.Writer, row benchRow) error {
	return writer.Write([]string{
		row.clip,
		strconv.Itoa(row.width),
		strconv.Itoa(row.height),
		strconv.Itoa(row.frames),
		strconv.Itoa(row.fps),
		row.encoder,
		strconv.Itoa(row.targetBPS),
		strconv.FormatInt(row.actualBPS, 10),
		row.encodeFPS,
		strconv.FormatInt(row.bytes, 10),
		row.metrics.psnr,
		row.metrics.ssim,
		row.metrics.xpsnr,
		row.metrics.vmaf,
		row.status,
		row.errText,
	})
}

func writeSummaryCSV(cfg benchConfig, rows []benchRow) error {
	file, err := os.Create(cfg.summaryCSVPath)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	defer writer.Flush()
	if err := writer.Write([]string{
		"clip", "anchor", "encoder", "metric", "anchor_points", "encoder_points",
		"overlap_min", "overlap_max", "bd_rate_pct", "status", "error",
	}); err != nil {
		return err
	}
	for _, summary := range summarizeBDRate(cfg.anchorEncoder, rows) {
		overlapMin, overlapMax, bdRate := "", "", ""
		if summary.Status == "ok" {
			overlapMin = formatMetric(summary.OverlapMin)
			overlapMax = formatMetric(summary.OverlapMax)
			bdRate = formatMetric(summary.BDRatePct)
		}
		if err := writer.Write([]string{
			summary.Clip,
			summary.Anchor,
			summary.Encoder,
			summary.Metric,
			strconv.Itoa(summary.AnchorPoints),
			strconv.Itoa(summary.EncoderPoints),
			overlapMin,
			overlapMax,
			bdRate,
			summary.Status,
			summary.ErrText,
		}); err != nil {
			return err
		}
	}
	return writer.Error()
}

func summarizeBDRate(anchor string, rows []benchRow) []summaryRow {
	anchor = strings.ToLower(anchor)
	metricsByName := [...]struct {
		name string
		get  func(metrics) string
	}{
		{name: "psnr_avg", get: func(m metrics) string { return m.psnr }},
		{name: "ssim_all", get: func(m metrics) string { return m.ssim }},
		{name: "xpsnr_y", get: func(m metrics) string { return m.xpsnr }},
		{name: "vmaf", get: func(m metrics) string { return m.vmaf }},
	}

	seenClips := map[string]bool{}
	var clips []string
	seenEncoders := map[string]bool{}
	var encoders []string
	for _, row := range rows {
		if !seenClips[row.clip] {
			seenClips[row.clip] = true
			clips = append(clips, row.clip)
		}
		if row.status == "ok" && row.encoder != "" && !seenEncoders[row.encoder] {
			seenEncoders[row.encoder] = true
			encoders = append(encoders, row.encoder)
		}
	}

	var summaries []summaryRow
	for _, clip := range clips {
		for _, encoder := range encoders {
			if strings.ToLower(encoder) == anchor {
				continue
			}
			for _, metric := range metricsByName {
				anchorPoints := rdPointsFor(rows, clip, anchor, metric.get)
				encoderPoints := rdPointsFor(rows, clip, strings.ToLower(encoder), metric.get)
				summary := summaryRow{
					Clip:          clip,
					Anchor:        anchor,
					Encoder:       encoder,
					Metric:        metric.name,
					AnchorPoints:  len(anchorPoints),
					EncoderPoints: len(encoderPoints),
					Status:        "ok",
				}
				bdRate, lo, hi, err := bdRatePercent(anchorPoints, encoderPoints)
				if err != nil {
					summary.Status = "error"
					summary.ErrText = err.Error()
				} else {
					summary.BDRatePct = bdRate
					summary.OverlapMin = lo
					summary.OverlapMax = hi
				}
				summaries = append(summaries, summary)
			}
		}
	}
	return summaries
}

func rdPointsFor(rows []benchRow, clip string, encoder string, getMetric func(metrics) string) []rdPoint {
	var points []rdPoint
	for _, row := range rows {
		if row.status != "ok" || row.clip != clip || strings.ToLower(row.encoder) != encoder || row.actualBPS <= 0 {
			continue
		}
		metric, err := strconv.ParseFloat(getMetric(row.metrics), 64)
		if err != nil || math.IsNaN(metric) || math.IsInf(metric, 0) {
			continue
		}
		points = append(points, rdPoint{
			Metric: metric,
			Rate:   float64(row.actualBPS),
		})
	}
	return points
}

func loadFrames(cfg benchConfig) ([]goav1.I420Frame, string, error) {
	if cfg.input == "" {
		return syntheticFrames(cfg.frames, cfg.width, cfg.height), "synthetic", nil
	}
	raw, err := os.ReadFile(cfg.input)
	if err != nil {
		return nil, "", err
	}
	cw, ch := cfg.width/2, cfg.height/2
	frameLen := cfg.width*cfg.height + 2*cw*ch
	if len(raw) < cfg.frames*frameLen {
		return nil, "", fmt.Errorf("input holds %d frames, need %d", len(raw)/frameLen, cfg.frames)
	}
	frames := make([]goav1.I420Frame, cfg.frames)
	for n := range cfg.frames {
		base := n * frameLen
		frames[n] = goav1.I420Frame{
			Y:            raw[base : base+cfg.width*cfg.height],
			U:            raw[base+cfg.width*cfg.height : base+cfg.width*cfg.height+cw*ch],
			V:            raw[base+cfg.width*cfg.height+cw*ch : base+frameLen],
			YStride:      cfg.width,
			ChromaStride: cw,
			Width:        cfg.width,
			Height:       cfg.height,
		}
	}
	return frames, filepath.Base(cfg.input), nil
}

func syntheticFrames(count, width, height int) []goav1.I420Frame {
	lumaPixels := width * height
	rng := rand.New(rand.NewSource(9))
	bg := make([]byte, lumaPixels)
	for i := range bg {
		bg[i] = uint8(50 + rng.Intn(90))
	}
	blurred := make([]byte, lumaPixels)
	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			sum := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					sum += int(bg[(y+dy)*width+x+dx])
				}
			}
			blurred[y*width+x] = uint8(sum / 9)
		}
	}
	copy(bg, blurred)
	frames := make([]goav1.I420Frame, count)
	for n := range count {
		frames[n] = makeFrame(bg, n, width, height)
	}
	return frames
}

func makeFrame(bg []byte, n, width, height int) goav1.I420Frame {
	cw, ch := width/2, height/2
	f := goav1.I420Frame{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	dx := (n * 2) % 16
	for y := range height {
		copy(f.Y[y*width:y*width+width-dx], bg[y*width+dx:y*width+width])
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 130
	}
	scaleX := func(v int) int { return v * width / defaultWidth }
	scaleY := func(v int) int { return v * height / defaultHeight }
	max1 := func(v int) int {
		if v < 1 {
			return 1
		}
		return v
	}
	for _, obj := range [2][3]int{
		{scaleX(200) + n*max1(width/160), scaleY(300), max1(scaleX(96))},
		{scaleX(1300) - n*max1(width/213), scaleY(700), max1(scaleX(64))},
	} {
		ox, oy, sz := obj[0], obj[1], obj[2]
		for y := oy; y < oy+sz && y < height; y++ {
			for x := ox; x < ox+sz && x < width; x++ {
				if x >= 0 {
					f.Y[y*width+x] = 215
				}
			}
		}
	}
	return f
}

func writeFrames(path string, frames []goav1.I420Frame, width, height int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, frame := range frames {
		if err := writeFrame(f, frame, width, height); err != nil {
			return err
		}
	}
	return nil
}

func writeFrame(w io.Writer, frame goav1.I420Frame, width, height int) error {
	if len(frame.Y) < (height-1)*frame.YStride+width {
		return errors.New("short Y plane")
	}
	cw, ch := width/2, height/2
	if len(frame.U) < (ch-1)*frame.ChromaStride+cw || len(frame.V) < (ch-1)*frame.ChromaStride+cw {
		return errors.New("short chroma plane")
	}
	for y := 0; y < height; y++ {
		if _, err := w.Write(frame.Y[y*frame.YStride : y*frame.YStride+width]); err != nil {
			return err
		}
	}
	for y := 0; y < ch; y++ {
		if _, err := w.Write(frame.U[y*frame.ChromaStride : y*frame.ChromaStride+cw]); err != nil {
			return err
		}
	}
	for y := 0; y < ch; y++ {
		if _, err := w.Write(frame.V[y*frame.ChromaStride : y*frame.ChromaStride+cw]); err != nil {
			return err
		}
	}
	return nil
}

func runEncoder(cfg benchConfig, frames []goav1.I420Frame, refPath string, encoderName string, bitrate int) encodeResult {
	switch encoderName {
	case "goav1":
		return encodeGoAV1(cfg, frames, bitrate)
	case "aomenc", "libaom":
		return encodeAOM(cfg, refPath, bitrate)
	case "svt-av1", "svtav1", "svt":
		return encodeSVT(cfg, refPath, bitrate)
	default:
		return encodeResult{encoder: encoderName, targetBPS: bitrate, status: "skipped", errText: "unknown encoder"}
	}
}

func encodeGoAV1(cfg benchConfig, frames []goav1.I420Frame, bitrate int) encodeResult {
	result := encodeResult{
		encoder:    "goav1",
		targetBPS:  bitrate,
		decodedYUV: filepath.Join(cfg.workdir, fmt.Sprintf("goav1_%d.yuv", bitrate)),
	}
	out, err := os.Create(result.decodedYUV)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	defer out.Close()

	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width:          cfg.width,
		Height:         cfg.height,
		TargetBitrate:  bitrate,
		Framerate:      cfg.fps,
		TemporalLayers: cfg.layers,
		TileColumns:    cfg.tiles,
		GoldenInterval: cfg.goldenInterval,
	})
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	var encodeDuration time.Duration
	for i, frame := range frames {
		forceKey := cfg.keyInterval > 0 && i > 0 && i%cfg.keyInterval == 0
		frameStart := time.Now()
		encoded, err := enc.Encode(frame, forceKey)
		encodeDuration += time.Since(frameStart)
		if err != nil {
			result.status, result.errText = "error", err.Error()
			return result
		}
		result.bytes += int64(len(encoded.Data))
		if err := writeFrame(out, enc.Reconstruction(), cfg.width, cfg.height); err != nil {
			result.status, result.errText = "error", err.Error()
			return result
		}
	}
	result.duration = encodeDuration
	result.status = "ok"
	return result
}

func encodeAOM(cfg benchConfig, refPath string, bitrate int) encodeResult {
	result := encodeResult{
		encoder:    "aomenc",
		targetBPS:  bitrate,
		decodedYUV: filepath.Join(cfg.workdir, fmt.Sprintf("aomenc_%d.yuv", bitrate)),
	}
	if _, err := exec.LookPath("aomenc"); err != nil {
		result.status, result.errText = "skipped", "aomenc not found"
		return result
	}
	ivfPath := filepath.Join(cfg.workdir, fmt.Sprintf("aomenc_%d.ivf", bitrate))
	args := []string{
		"--ivf",
		"--codec=av1",
		"--rt",
		"--cpu-used=8",
		"--end-usage=cbr",
		fmt.Sprintf("--target-bitrate=%d", kbps(bitrate)),
		fmt.Sprintf("--fps=%d/1", cfg.fps),
		fmt.Sprintf("--width=%d", cfg.width),
		fmt.Sprintf("--height=%d", cfg.height),
		"--i420",
		"--threads=4",
		"--lag-in-frames=0",
		"--auto-alt-ref=0",
		"--enable-fwd-kf=0",
		"--drop-frame=0",
		"--buf-sz=1000",
		"--buf-initial-sz=500",
		"--buf-optimal-sz=600",
		fmt.Sprintf("--limit=%d", cfg.frames),
	}
	if cfg.tiles > 0 {
		args = append(args, fmt.Sprintf("--tile-columns=%d", cfg.tiles))
	}
	if cfg.keyInterval > 0 {
		args = append(args,
			fmt.Sprintf("--kf-min-dist=%d", cfg.keyInterval),
			fmt.Sprintf("--kf-max-dist=%d", cfg.keyInterval),
		)
	} else {
		args = append(args, "--kf-min-dist=9999", "--kf-max-dist=9999")
	}
	args = append(args, "-o", ivfPath, refPath)
	result.duration = timeCommand("aomenc", args, &result)
	if result.status != "" {
		return result
	}
	payloadBytes, err := ivfPayloadBytes(ivfPath)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.bytes = payloadBytes
	if err := decodeIVFWithFFmpeg(ivfPath, result.decodedYUV, cfg.frames); err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.status = "ok"
	return result
}

func encodeSVT(cfg benchConfig, refPath string, bitrate int) encodeResult {
	result := encodeResult{
		encoder:    "svt-av1",
		targetBPS:  bitrate,
		decodedYUV: filepath.Join(cfg.workdir, fmt.Sprintf("svtav1_%d.yuv", bitrate)),
	}
	if _, err := exec.LookPath("SvtAv1EncApp"); err != nil {
		result.status, result.errText = "skipped", "SvtAv1EncApp not found"
		return result
	}
	ivfPath := filepath.Join(cfg.workdir, fmt.Sprintf("svtav1_%d.ivf", bitrate))
	keyint := "-1"
	if cfg.keyInterval > 0 {
		keyint = strconv.Itoa(cfg.keyInterval)
	}
	args := []string{
		"-i", refPath,
		"-b", ivfPath,
		"-w", strconv.Itoa(cfg.width),
		"-h", strconv.Itoa(cfg.height),
		"--fps-num", strconv.Itoa(cfg.fps),
		"--fps-denom", "1",
		"--frames", strconv.Itoa(cfg.frames),
		"--input-depth", "8",
		"--color-format", "1",
		"--preset", "13",
		"--rc", "2",
		"--tbr", strconv.Itoa(kbps(bitrate)),
		"--lookahead", "0",
		"--pred-struct", "1",
		"--rtc", "1",
		"--scd", "0",
		"--enable-tf", "0",
		"--irefresh-type", "2",
		"--keyint", keyint,
		"--progress", "0",
		"--lp", "4",
	}
	if cfg.tiles > 0 {
		args = append(args, "--tile-columns", strconv.Itoa(cfg.tiles))
	}
	result.duration = timeCommand("SvtAv1EncApp", args, &result)
	if result.status != "" {
		return result
	}
	payloadBytes, err := ivfPayloadBytes(ivfPath)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.bytes = payloadBytes
	if err := decodeIVFWithFFmpeg(ivfPath, result.decodedYUV, cfg.frames); err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.status = "ok"
	return result
}

func kbps(bps int) int {
	if bps < 1000 {
		return 1
	}
	return (bps + 500) / 1000
}

func timeCommand(name string, args []string, result *encodeResult) time.Duration {
	start := time.Now()
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	elapsed := time.Since(start)
	if err != nil {
		result.status = "error"
		result.errText = trimCommandOutput(err, out)
	}
	return elapsed
}

func trimCommandOutput(err error, out []byte) string {
	msg := strings.TrimSpace(string(out))
	if msg == "" {
		return err.Error()
	}
	msg = strings.ReplaceAll(msg, "\n", " | ")
	if len(msg) > 600 {
		msg = msg[:600]
	}
	return msg
}

func decodeIVFWithFFmpeg(ivfPath, yuvPath string, frames int) error {
	args := []string{
		"-y",
		"-hide_banner",
		"-loglevel", "error",
		"-i", ivfPath,
		"-pix_fmt", "yuv420p",
		"-frames:v", strconv.Itoa(frames),
		"-f", "rawvideo",
		yuvPath,
	}
	cmd := exec.Command("ffmpeg", args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg decode: %s", trimCommandOutput(err, out))
	}
	return nil
}

func ivfPayloadBytes(path string) (int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	var header [32]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return 0, err
	}
	if string(header[:4]) != "DKIF" {
		return 0, errors.New("not an IVF file")
	}
	var total int64
	for {
		var frameHeader [12]byte
		_, err := io.ReadFull(f, frameHeader[:])
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			break
		}
		if err != nil {
			return 0, err
		}
		size := int64(binary.LittleEndian.Uint32(frameHeader[:4]))
		total += size
		if _, err := f.Seek(size, io.SeekCurrent); err != nil {
			return 0, err
		}
	}
	return total, nil
}

func ffmpegFilters() map[string]bool {
	out := map[string]bool{}
	cmd := exec.Command("ffmpeg", "-hide_banner", "-filters")
	raw, err := cmd.Output()
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			out[fields[1]] = true
		}
	}
	return out
}

func measureDecoded(cfg benchConfig, filters map[string]bool, refPath, decodedPath, encoderName string, bitrate int) metrics {
	m := metrics{psnr: "NA", ssim: "NA", xpsnr: "NA", vmaf: "NA"}
	if filters["psnr"] {
		if v, err := runScalarMetric(cfg, refPath, decodedPath, "psnr", `average:([0-9.]+)`); err == nil {
			m.psnr = formatMetric(v)
		}
	}
	if filters["ssim"] {
		if v, err := runScalarMetric(cfg, refPath, decodedPath, "ssim", `All:([0-9.]+)`); err == nil {
			m.ssim = formatMetric(v)
		}
	}
	if filters["xpsnr"] {
		if v, err := runScalarMetric(cfg, refPath, decodedPath, "xpsnr", `XPSNR\s+y:\s*([0-9]+(?:\.[0-9]+)?)`); err == nil {
			m.xpsnr = formatMetric(v)
		}
	}
	if filters["libvmaf"] {
		if v, err := runVMAF(cfg, refPath, decodedPath, encoderName, bitrate); err == nil {
			m.vmaf = formatMetric(v)
		}
	}
	return m
}

func runScalarMetric(cfg benchConfig, refPath, decodedPath, filterName, pattern string) (float64, error) {
	args := rawMetricArgs(cfg, refPath, decodedPath, filterName)
	cmd := exec.Command("ffmpeg", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("%s: %s", filterName, trimCommandOutput(err, out))
	}
	re := regexp.MustCompile(pattern)
	m := re.FindStringSubmatch(string(out))
	if len(m) < 2 {
		return 0, fmt.Errorf("%s metric not found in ffmpeg output", filterName)
	}
	return strconv.ParseFloat(m[1], 64)
}

func rawMetricArgs(cfg benchConfig, refPath, decodedPath, filterName string) []string {
	size := fmt.Sprintf("%dx%d", cfg.width, cfg.height)
	return []string{
		"-hide_banner",
		"-nostats",
		"-f", "rawvideo",
		"-pixel_format", "yuv420p",
		"-video_size", size,
		"-framerate", strconv.Itoa(cfg.fps),
		"-i", refPath,
		"-f", "rawvideo",
		"-pixel_format", "yuv420p",
		"-video_size", size,
		"-framerate", strconv.Itoa(cfg.fps),
		"-i", decodedPath,
		"-lavfi", fmt.Sprintf("[0:v][1:v]%s", filterName),
		"-frames:v", strconv.Itoa(cfg.frames),
		"-f", "null",
		"-",
	}
}

func runVMAF(cfg benchConfig, refPath, decodedPath, encoderName string, bitrate int) (float64, error) {
	logPath := filepath.Join(cfg.workdir, fmt.Sprintf("%s_%d_vmaf.json", encoderName, bitrate))
	filter := fmt.Sprintf("[0:v][1:v]libvmaf=log_path=%s:log_fmt=json", escapeFilterPath(logPath))
	size := fmt.Sprintf("%dx%d", cfg.width, cfg.height)
	args := []string{
		"-hide_banner",
		"-nostats",
		"-f", "rawvideo",
		"-pixel_format", "yuv420p",
		"-video_size", size,
		"-framerate", strconv.Itoa(cfg.fps),
		"-i", refPath,
		"-f", "rawvideo",
		"-pixel_format", "yuv420p",
		"-video_size", size,
		"-framerate", strconv.Itoa(cfg.fps),
		"-i", decodedPath,
		"-lavfi", filter,
		"-frames:v", strconv.Itoa(cfg.frames),
		"-f", "null",
		"-",
	}
	cmd := exec.Command("ffmpeg", args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return 0, fmt.Errorf("libvmaf: %s", trimCommandOutput(err, out))
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		return 0, err
	}
	return parseVMAFMean(raw)
}

func parseVMAFMean(raw []byte) (float64, error) {
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return 0, err
	}
	pooled, _ := doc["pooled_metrics"].(map[string]any)
	vmaf, _ := pooled["vmaf"].(map[string]any)
	mean, ok := vmaf["mean"].(float64)
	if !ok {
		return 0, errors.New("vmaf mean not found")
	}
	return mean, nil
}

func bdRatePercent(anchor []rdPoint, candidate []rdPoint) (pct float64, overlapMin float64, overlapMax float64, err error) {
	anchor, err = normalizedRDPoints(anchor)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("anchor: %w", err)
	}
	candidate, err = normalizedRDPoints(candidate)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("candidate: %w", err)
	}
	overlapMin = math.Max(anchor[0].Metric, candidate[0].Metric)
	overlapMax = math.Min(anchor[len(anchor)-1].Metric, candidate[len(candidate)-1].Metric)
	if !(overlapMax > overlapMin) {
		return 0, overlapMin, overlapMax, errors.New("no overlapping metric range")
	}
	anchorPoly, err := fitCubicLogRate(anchor)
	if err != nil {
		return 0, overlapMin, overlapMax, fmt.Errorf("anchor fit: %w", err)
	}
	candidatePoly, err := fitCubicLogRate(candidate)
	if err != nil {
		return 0, overlapMin, overlapMax, fmt.Errorf("candidate fit: %w", err)
	}
	span := overlapMax - overlapMin
	anchorAvg := integrateCubicFit(anchorPoly, overlapMin, overlapMax) / span
	candidateAvg := integrateCubicFit(candidatePoly, overlapMin, overlapMax) / span
	return (math.Exp(candidateAvg-anchorAvg) - 1) * 100, overlapMin, overlapMax, nil
}

func normalizedRDPoints(points []rdPoint) ([]rdPoint, error) {
	if len(points) < 4 {
		return nil, fmt.Errorf("need at least 4 RD points, got %d", len(points))
	}
	out := append([]rdPoint(nil), points...)
	sort.Slice(out, func(i, j int) bool {
		return out[i].Metric < out[j].Metric
	})
	write := 0
	for _, point := range out {
		if point.Rate <= 0 || math.IsNaN(point.Rate) || math.IsInf(point.Rate, 0) ||
			math.IsNaN(point.Metric) || math.IsInf(point.Metric, 0) {
			return nil, errors.New("invalid RD point")
		}
		if write > 0 && math.Abs(point.Metric-out[write-1].Metric) < 1e-9 {
			out[write-1].Rate = point.Rate
			continue
		}
		out[write] = point
		write++
	}
	out = out[:write]
	if len(out) < 4 {
		return nil, fmt.Errorf("need at least 4 unique metric points, got %d", len(out))
	}
	return out, nil
}

func fitCubicLogRate(points []rdPoint) (cubicFit, error) {
	center := (points[0].Metric + points[len(points)-1].Metric) / 2
	scale := (points[len(points)-1].Metric - points[0].Metric) / 2
	if !(scale > 0) {
		return cubicFit{}, errors.New("zero metric span")
	}
	var normal [4][5]float64
	for _, point := range points {
		x := (point.Metric - center) / scale
		y := math.Log(point.Rate)
		powers := [7]float64{1}
		for i := 1; i < len(powers); i++ {
			powers[i] = powers[i-1] * x
		}
		for row := 0; row < 4; row++ {
			for col := 0; col < 4; col++ {
				normal[row][col] += powers[row+col]
			}
			normal[row][4] += y * powers[row]
		}
	}
	coeff, err := solve4x4(normal)
	if err != nil {
		return cubicFit{}, err
	}
	return cubicFit{Coeff: coeff, Center: center, Scale: scale}, nil
}

func solve4x4(a [4][5]float64) ([4]float64, error) {
	for col := 0; col < 4; col++ {
		pivot := col
		for row := col + 1; row < 4; row++ {
			if math.Abs(a[row][col]) > math.Abs(a[pivot][col]) {
				pivot = row
			}
		}
		if math.Abs(a[pivot][col]) < 1e-12 {
			return [4]float64{}, errors.New("singular cubic fit")
		}
		if pivot != col {
			a[col], a[pivot] = a[pivot], a[col]
		}
		scale := a[col][col]
		for j := col; j < 5; j++ {
			a[col][j] /= scale
		}
		for row := 0; row < 4; row++ {
			if row == col {
				continue
			}
			factor := a[row][col]
			for j := col; j < 5; j++ {
				a[row][j] -= factor * a[col][j]
			}
		}
	}
	return [4]float64{a[0][4], a[1][4], a[2][4], a[3][4]}, nil
}

func integrateCubic(c [4]float64, lo float64, hi float64) float64 {
	primitive := func(x float64) float64 {
		x2 := x * x
		x3 := x2 * x
		x4 := x3 * x
		return c[0]*x + c[1]*x2/2 + c[2]*x3/3 + c[3]*x4/4
	}
	return primitive(hi) - primitive(lo)
}

func integrateCubicFit(fit cubicFit, lo float64, hi float64) float64 {
	tLo := (lo - fit.Center) / fit.Scale
	tHi := (hi - fit.Center) / fit.Scale
	return integrateCubic(fit.Coeff, tLo, tHi) * fit.Scale
}

func escapeFilterPath(path string) string {
	return strings.ReplaceAll(path, ":", `\:`)
}

func formatMetric(v float64) string {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return "NA"
	}
	return strconv.FormatFloat(v, 'f', 4, 64)
}
