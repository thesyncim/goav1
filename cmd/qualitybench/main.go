// Command qualitybench compares goav1 against installed realtime AV1 baselines
// on the same raw I420 source frames, then measures decoded output with FFmpeg
// objective metrics.
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/csv"
	"encoding/hex"
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
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	goav1 "github.com/thesyncim/goav1"
	cpufeatures "github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

const (
	defaultWidth  = 1920
	defaultHeight = 1080
	defaultFrames = 120
	defaultFPS    = 60

	timingModeCore     = "core"
	timingModeEndToEnd = "e2e"

	runOrderBitrateEncoder = "bitrate-encoder"
	runOrderEncoderBitrate = "encoder-bitrate"
	runOrderShuffle        = "shuffle"
)

type benchConfig struct {
	width               int
	height              int
	frames              int
	fps                 int
	input               string
	manifestPath        string
	workdir             string
	csvPath             string
	summaryCSVPath      string
	statsCSVPath        string
	frameStatsCSVPath   string
	frameMetricsCSVPath string
	metadataPath        string
	anchorEncoder       string
	requiredEncodersRaw string
	timingMode          string
	runOrder            string
	shuffleSeed         int64
	encoders            []string
	bitrates            []int
	requiredMetrics     []string
	requiredEncoders    []string
	requireSummary      bool
	requireCorpus       bool
	minClips            int
	layers              int
	tiles               int
	goldenInterval      int
	keyInterval         int
	goMaxProcs          int
	aomThreads          int
	aomRowMT            int
	svtLP               int
	svtASM              string
	publish             bool
	keep                bool
	explicitFlags       map[string]bool
}

type encodeResult struct {
	encoder          string
	targetBPS        int
	bytes            int64
	duration         time.Duration
	cpuUser          time.Duration
	cpuSystem        time.Duration
	cpuAvailable     bool
	encodedPath      string
	encodedContainer string
	encodedBytes     int64
	encodedSHA256    string
	decodedYUV       string
	decodedBytes     int64
	decodedSHA256    string
	status           string
	errText          string
	stats            goav1.EncoderDecisionStats
	frameStats       []goAV1FrameStats
	command          []string
	settings         map[string]string
}

type metrics struct {
	psnr  string
	ssim  string
	xpsnr string
	vmaf  string
}

type clipSpec struct {
	Name   string
	Input  string
	Width  int
	Height int
	Frames int
	FPS    int
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
	duration  time.Duration
	cpuUser   time.Duration
	cpuSystem time.Duration
	cpuOK     bool
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

type goAV1FrameStats struct {
	FrameIndex      int
	Keyframe        bool
	TemporalID      uint8
	QIndexBefore    uint8
	QIndexAfter     uint8
	Bytes           int64
	Duration        time.Duration
	CumulativeBytes int64
	Stats           goav1.EncoderDecisionStats
}

type frameMetricValue struct {
	FrameIndex int
	Value      string
}

type qualitybenchMetadata struct {
	GeneratedAtUTC string                      `json:"generated_at_utc"`
	Go             runtimeMetadata             `json:"go"`
	Git            gitMetadata                 `json:"git"`
	Config         metadataConfig              `json:"config"`
	FairnessNotes  []string                    `json:"fairness_notes,omitempty"`
	MetricFilters  map[string]bool             `json:"metric_filters"`
	Tools          map[string]toolMetadata     `json:"tools"`
	Clips          []clipMetadata              `json:"clips"`
	Encodes        []encoderInvocationMetadata `json:"encodes"`
}

type runtimeMetadata struct {
	Version      string   `json:"version"`
	GOOS         string   `json:"goos"`
	GOARCH       string   `json:"goarch"`
	SIMDTier     string   `json:"simd_tier"`
	SIMDFeatures []string `json:"simd_features,omitempty"`
}

type gitMetadata struct {
	Commit string `json:"commit"`
	Dirty  bool   `json:"dirty"`
	Error  string `json:"error,omitempty"`
}

type metadataConfig struct {
	Width            int      `json:"width"`
	Height           int      `json:"height"`
	Frames           int      `json:"frames"`
	FPS              int      `json:"fps"`
	Input            string   `json:"input,omitempty"`
	Manifest         string   `json:"manifest,omitempty"`
	Encoders         []string `json:"encoders"`
	Bitrates         []int    `json:"bitrates"`
	RequiredMetrics  []string `json:"required_metrics,omitempty"`
	RequiredEncoders []string `json:"required_encoders,omitempty"`
	RequireSummary   bool     `json:"require_summary,omitempty"`
	RequireCorpus    bool     `json:"require_corpus,omitempty"`
	MinClips         int      `json:"min_clips,omitempty"`
	Anchor           string   `json:"anchor"`
	Layers           int      `json:"layers"`
	Tiles            int      `json:"tiles"`
	GoldenInterval   int      `json:"golden_interval"`
	KeyInterval      int      `json:"key_interval"`
	GoMaxProcs       int      `json:"gomaxprocs"`
	AOMThreads       int      `json:"aom_threads"`
	AOMRowMT         int      `json:"aom_row_mt"`
	SVTLP            int      `json:"svt_lp"`
	SVTASM           string   `json:"svt_asm,omitempty"`
	TimingMode       string   `json:"timing_mode"`
	RunOrder         string   `json:"run_order"`
	ShuffleSeed      int64    `json:"shuffle_seed,omitempty"`
	Publish          bool     `json:"publish,omitempty"`
}

type toolMetadata struct {
	Path         string `json:"path,omitempty"`
	Found        bool   `json:"found"`
	SHA256       string `json:"sha256,omitempty"`
	Version      string `json:"version,omitempty"`
	VersionError string `json:"version_error,omitempty"`
}

type clipMetadata struct {
	Name          string `json:"name"`
	Input         string `json:"input,omitempty"`
	Synthetic     bool   `json:"synthetic,omitempty"`
	Width         int    `json:"width"`
	Height        int    `json:"height"`
	Frames        int    `json:"frames"`
	FPS           int    `json:"fps"`
	ExpectedBytes int64  `json:"expected_bytes"`
	InputBytes    int64  `json:"input_bytes,omitempty"`
	SHA256        string `json:"sha256,omitempty"`
}

type encoderInvocationMetadata struct {
	Clip             string            `json:"clip"`
	Width            int               `json:"width"`
	Height           int               `json:"height"`
	Frames           int               `json:"frames"`
	FPS              int               `json:"fps"`
	Encoder          string            `json:"encoder"`
	TargetBPS        int               `json:"target_bps"`
	ActualBPS        int64             `json:"actual_bps"`
	CompressedBytes  int64             `json:"compressed_bytes,omitempty"`
	EncodedPath      string            `json:"encoded_path,omitempty"`
	EncodedContainer string            `json:"encoded_container,omitempty"`
	EncodedBytes     int64             `json:"encoded_bytes,omitempty"`
	EncodedSHA256    string            `json:"encoded_sha256,omitempty"`
	DecodedPath      string            `json:"decoded_path,omitempty"`
	DecodedBytes     int64             `json:"decoded_bytes,omitempty"`
	DecodedSHA256    string            `json:"decoded_sha256,omitempty"`
	EncodeWallSecs   float64           `json:"encode_wall_seconds,omitempty"`
	CPUAvailable     bool              `json:"cpu_available,omitempty"`
	CPUUserSecs      float64           `json:"cpu_user_seconds,omitempty"`
	CPUSystemSecs    float64           `json:"cpu_system_seconds,omitempty"`
	CPUTotalSecs     float64           `json:"cpu_total_seconds,omitempty"`
	ObservedParallel float64           `json:"observed_parallelism,omitempty"`
	Status           string            `json:"status"`
	Error            string            `json:"error,omitempty"`
	Command          []string          `json:"command,omitempty"`
	Settings         map[string]string `json:"settings,omitempty"`
}

type processCPUTimes struct {
	user   time.Duration
	system time.Duration
}

type encodeJob struct {
	bitrate int
	encoder string
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
	if cfg.publish {
		if err := validatePublishConfig(cfg, currentGitMetadata()); err != nil {
			return err
		}
	}
	if cfg.goMaxProcs > 0 {
		runtime.GOMAXPROCS(cfg.goMaxProcs)
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

	filters := ffmpegFilters()
	if err := validateRequiredMetrics(filters, cfg.requiredMetrics); err != nil {
		return err
	}
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
		"actual_bps", "encode_fps", "encode_wall_sec", "cpu_user_sec",
		"cpu_system_sec", "cpu_total_sec", "observed_parallelism", "bytes", "psnr_avg", "ssim_all",
		"xpsnr_y", "vmaf", "status", "error",
	}); err != nil {
		return err
	}
	var statsWriter *csv.Writer
	var statsFile *os.File
	if cfg.statsCSVPath != "" {
		statsFile, err = os.Create(cfg.statsCSVPath)
		if err != nil {
			return err
		}
		defer statsFile.Close()
		statsWriter = csv.NewWriter(statsFile)
		defer statsWriter.Flush()
		if err := writeStatsHeader(statsWriter); err != nil {
			return err
		}
	}
	var frameStatsWriter *csv.Writer
	var frameStatsFile *os.File
	if cfg.frameStatsCSVPath != "" {
		frameStatsFile, err = os.Create(cfg.frameStatsCSVPath)
		if err != nil {
			return err
		}
		defer frameStatsFile.Close()
		frameStatsWriter = csv.NewWriter(frameStatsFile)
		defer frameStatsWriter.Flush()
		if err := writeFrameStatsHeader(frameStatsWriter); err != nil {
			return err
		}
	}
	var frameMetricsWriter *csv.Writer
	var frameMetricsFile *os.File
	if cfg.frameMetricsCSVPath != "" {
		frameMetricsFile, err = os.Create(cfg.frameMetricsCSVPath)
		if err != nil {
			return err
		}
		defer frameMetricsFile.Close()
		frameMetricsWriter = csv.NewWriter(frameMetricsFile)
		defer frameMetricsWriter.Flush()
		if err := writeFrameMetricsHeader(frameMetricsWriter); err != nil {
			return err
		}
	}

	var rows []benchRow
	var invocations []encoderInvocationMetadata
	clips, err := clipSpecsForConfig(cfg)
	if err != nil {
		return err
	}
	if err := validateRequiredCorpus(cfg, clips); err != nil {
		return err
	}
	if cfg.publish {
		if err := validatePublishClipInputs(clips); err != nil {
			return err
		}
	}
	for _, clip := range clips {
		clipRows, clipInvocations, err := runClip(cfg, clip, filters, writer, statsWriter, frameStatsWriter, frameMetricsWriter)
		if err != nil {
			return err
		}
		rows = append(rows, clipRows...)
		invocations = append(invocations, clipInvocations...)
	}
	var summaries []summaryRow
	if cfg.summaryCSVPath != "" || cfg.requireSummary {
		summaries = summarizeBDRate(cfg.anchorEncoder, rows)
	}
	if cfg.summaryCSVPath != "" {
		if err := writeSummaryCSV(cfg.summaryCSVPath, summaries); err != nil {
			return err
		}
	}
	var summaryErr error
	if cfg.requireSummary {
		summaryErr = validateRequiredSummaries(cfg, rows, summaries)
	}
	if cfg.metadataPath != "" {
		if err := writeMetadataJSON(cfg, filters, clips, invocations); err != nil {
			return err
		}
	}
	if cfg.csvPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.csvPath)
	}
	if cfg.summaryCSVPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.summaryCSVPath)
	}
	if cfg.statsCSVPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.statsCSVPath)
	}
	if cfg.frameStatsCSVPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.frameStatsCSVPath)
	}
	if cfg.frameMetricsCSVPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.frameMetricsCSVPath)
	}
	if cfg.metadataPath != "" {
		fmt.Fprintf(os.Stderr, "qualitybench wrote %s\n", cfg.metadataPath)
	}
	if cfg.keep || cfg.workdir != "" && !cleanup {
		fmt.Fprintf(os.Stderr, "qualitybench workdir: %s\n", cfg.workdir)
	}
	if err := writer.Error(); err != nil {
		return err
	}
	if statsWriter != nil {
		if err := statsWriter.Error(); err != nil {
			return err
		}
	}
	if frameStatsWriter != nil {
		if err := frameStatsWriter.Error(); err != nil {
			return err
		}
	}
	if frameMetricsWriter != nil {
		if err := frameMetricsWriter.Error(); err != nil {
			return err
		}
	}
	return summaryErr
}

func parseFlags() (benchConfig, error) {
	var cfg benchConfig
	bitrates := flag.String("bitrates", "3000000,6000000,9000000,12000000", "comma-separated target bitrates in bits per second")
	encoders := flag.String("encoders", "goav1,aomenc,svt-av1", "comma-separated encoders: goav1,aomenc,svt-av1")
	requiredMetrics := flag.String("require-metrics", "", "comma-separated metrics that must be available and computed for each successful encode: psnr,ssim,xpsnr,vmaf")
	requiredEncoders := flag.String("require-encoders", "", "comma-separated encoders that must produce ok rows, or all")
	flag.StringVar(&cfg.input, "input", "", "raw I420 input file; omit to use the deterministic synthetic scene")
	flag.StringVar(&cfg.manifestPath, "manifest", "", "CSV corpus manifest with clip,input,width,height,frames,fps columns")
	flag.IntVar(&cfg.width, "width", defaultWidth, "frame width in pixels")
	flag.IntVar(&cfg.height, "height", defaultHeight, "frame height in pixels")
	flag.IntVar(&cfg.frames, "frames", defaultFrames, "frames to encode")
	flag.IntVar(&cfg.fps, "fps", defaultFPS, "input frame rate and bitrate timebase")
	flag.IntVar(&cfg.minClips, "min-clips", 0, "minimum number of clips required before encoding")
	flag.IntVar(&cfg.layers, "layers", 1, "goav1 temporal layers (1, 2, or 3)")
	flag.IntVar(&cfg.tiles, "tiles", 0, "tile-column log2 override for encoders that expose one")
	flag.IntVar(&cfg.goldenInterval, "golden", 0, "goav1 golden refresh interval (0 = default, negative = disabled)")
	flag.IntVar(&cfg.keyInterval, "keyint", 0, "force periodic keyframes every N frames after frame 0 (0 = only initial key)")
	flag.IntVar(&cfg.goMaxProcs, "gomaxprocs", 0, "set Go GOMAXPROCS for in-process goav1 encodes (0 = keep environment/runtime default)")
	flag.IntVar(&cfg.aomThreads, "aom-threads", 4, "aomenc --threads value for libaom rows")
	flag.IntVar(&cfg.aomRowMT, "aom-row-mt", 1, "aomenc --row-mt value for libaom rows (0 = off, 1 = on)")
	flag.IntVar(&cfg.svtLP, "svt-lp", 0, "SVT --lp parallelism level, not a thread count (0 = SVT auto, valid range 0..6)")
	flag.StringVar(&cfg.svtASM, "svt-asm", "", "limit SVT --asm instruction set (empty = SVT default max; e.g. c,neon,neon_dotprod,neon_i8mm,sve,sve2)")
	flag.StringVar(&cfg.workdir, "workdir", "", "directory for raw, decoded, and encoded intermediates")
	flag.StringVar(&cfg.csvPath, "csv", "", "write CSV to this path instead of stdout")
	flag.StringVar(&cfg.summaryCSVPath, "summary-csv", "", "write BD-rate summary CSV to this path")
	flag.StringVar(&cfg.statsCSVPath, "stats-csv", "", "write goav1 encoder decision diagnostics CSV to this path")
	flag.StringVar(&cfg.frameStatsCSVPath, "frame-stats-csv", "", "write per-frame goav1 rate-control and decision diagnostics CSV to this path")
	flag.StringVar(&cfg.frameMetricsCSVPath, "frame-metrics-csv", "", "write per-frame decoded PSNR/SSIM diagnostics CSV to this path")
	flag.StringVar(&cfg.metadataPath, "metadata-json", "", "write reproducibility metadata JSON to this path")
	flag.StringVar(&cfg.anchorEncoder, "anchor", "", "encoder name to use as BD-rate anchor (default: first -encoders entry)")
	flag.StringVar(&cfg.timingMode, "timing-mode", timingModeCore, "encode timing mode: core or e2e")
	flag.StringVar(&cfg.runOrder, "run-order", runOrderBitrateEncoder, "encode tuple order: bitrate-encoder, encoder-bitrate, or shuffle")
	flag.Int64Var(&cfg.shuffleSeed, "shuffle-seed", 1, "deterministic seed used when -run-order=shuffle")
	flag.BoolVar(&cfg.requireSummary, "require-summary", false, "fail if required BD-rate summary rows are missing or invalid")
	flag.BoolVar(&cfg.requireCorpus, "require-corpus", false, "require a manifest-backed real clip corpus before encoding")
	flag.BoolVar(&cfg.publish, "publish", false, "require strict reproducibility controls for published benchmark tables")
	flag.BoolVar(&cfg.keep, "keep", false, "keep the temporary workdir when -workdir is not set")
	flag.Parse()
	cfg.explicitFlags = explicitFlagSet()
	cfg.requiredEncodersRaw = *requiredEncoders

	var err error
	cfg.bitrates, err = parsePositiveList(*bitrates)
	if err != nil {
		return benchConfig{}, fmt.Errorf("bitrates: %w", err)
	}
	cfg.encoders = parseEncoderList(*encoders)
	if len(cfg.encoders) == 0 {
		return benchConfig{}, errors.New("no encoders selected")
	}
	cfg.requiredMetrics, err = parseMetricList(*requiredMetrics)
	if err != nil {
		return benchConfig{}, fmt.Errorf("require-metrics: %w", err)
	}
	cfg.requiredEncoders, err = parseRequiredEncoderList(*requiredEncoders, cfg.encoders)
	if err != nil {
		return benchConfig{}, fmt.Errorf("require-encoders: %w", err)
	}
	if cfg.requireSummary {
		if cfg.summaryCSVPath == "" {
			return benchConfig{}, errors.New("require-summary requires -summary-csv")
		}
		if len(cfg.requiredMetrics) == 0 {
			return benchConfig{}, errors.New("require-summary requires -require-metrics")
		}
	}
	if cfg.minClips < 0 {
		return benchConfig{}, fmt.Errorf("invalid minimum clip count %d", cfg.minClips)
	}
	if cfg.requireCorpus && cfg.minClips < 2 {
		return benchConfig{}, errors.New("require-corpus requires -min-clips >= 2")
	}
	if cfg.anchorEncoder == "" {
		cfg.anchorEncoder = cfg.encoders[0]
	} else {
		cfg.anchorEncoder = canonicalEncoderName(cfg.anchorEncoder)
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
	if cfg.goMaxProcs < 0 {
		return benchConfig{}, fmt.Errorf("invalid GOMAXPROCS %d", cfg.goMaxProcs)
	}
	if cfg.aomThreads <= 0 {
		return benchConfig{}, fmt.Errorf("invalid aomenc --threads value %d", cfg.aomThreads)
	}
	if cfg.aomRowMT != 0 && cfg.aomRowMT != 1 {
		return benchConfig{}, fmt.Errorf("invalid aomenc --row-mt value %d: valid values are 0 or 1", cfg.aomRowMT)
	}
	if cfg.timingMode != timingModeCore && cfg.timingMode != timingModeEndToEnd {
		return benchConfig{}, fmt.Errorf("invalid timing mode %q: valid values are core or e2e", cfg.timingMode)
	}
	cfg.runOrder = strings.TrimSpace(strings.ToLower(cfg.runOrder))
	if !validRunOrder(cfg.runOrder) {
		return benchConfig{}, fmt.Errorf("invalid run order %q: valid values are %s, %s, or %s", cfg.runOrder, runOrderBitrateEncoder, runOrderEncoderBitrate, runOrderShuffle)
	}
	if cfg.svtLP < 0 || cfg.svtLP > 6 {
		return benchConfig{}, fmt.Errorf("invalid SVT --lp level %d: valid range is 0..6; --lp is a parallelism level, not a thread count", cfg.svtLP)
	}
	if cfg.svtASM != "" {
		var ok bool
		cfg.svtASM, ok = canonicalSVTASMName(cfg.svtASM)
		if !ok {
			return benchConfig{}, fmt.Errorf("invalid SVT --asm value %q: valid values are c, neon, crc32, neon_dotprod, neon_i8mm, sve, sve2, max", cfg.svtASM)
		}
	}
	return cfg, nil
}

func explicitFlagSet() map[string]bool {
	set := map[string]bool{}
	flag.Visit(func(f *flag.Flag) {
		set[f.Name] = true
	})
	return set
}

func validatePublishConfig(cfg benchConfig, git gitMetadata) error {
	if git.Error != "" {
		return fmt.Errorf("publish requires git metadata: %s", git.Error)
	}
	if git.Dirty {
		return errors.New("publish requires a clean tracked git worktree")
	}
	required := []string{
		"workdir",
		"csv",
		"metadata-json",
		"manifest",
		"require-corpus",
		"min-clips",
		"require-encoders",
		"require-metrics",
		"summary-csv",
		"require-summary",
		"gomaxprocs",
		"timing-mode",
		"run-order",
	}
	for _, name := range required {
		if err := requireExplicitFlag(cfg, name); err != nil {
			return err
		}
	}
	if strings.TrimSpace(strings.ToLower(cfg.requiredEncodersRaw)) != "all" {
		return errors.New("publish requires -require-encoders all")
	}
	if cfg.goMaxProcs <= 0 {
		return errors.New("publish requires -gomaxprocs > 0")
	}
	if cfg.timingMode != timingModeEndToEnd {
		return errors.New("publish requires -timing-mode e2e")
	}
	if cfg.runOrder == runOrderShuffle {
		if err := requireExplicitFlag(cfg, "shuffle-seed"); err != nil {
			return err
		}
	}
	if !cfg.requireCorpus || cfg.manifestPath == "" || cfg.minClips < 2 {
		return errors.New("publish requires -require-corpus with -manifest and -min-clips >= 2")
	}
	if cfg.csvPath == "" || cfg.metadataPath == "" || cfg.summaryCSVPath == "" || !cfg.requireSummary {
		return errors.New("publish requires -csv, -metadata-json, -summary-csv, and -require-summary")
	}
	if len(cfg.requiredMetrics) == 0 {
		return errors.New("publish requires -require-metrics")
	}
	if encoderSelected(cfg, "aomenc") {
		if err := requireExplicitFlag(cfg, "aom-threads"); err != nil {
			return err
		}
		if err := requireExplicitFlag(cfg, "aom-row-mt"); err != nil {
			return err
		}
	}
	if encoderSelected(cfg, "svt-av1") {
		if err := requireExplicitFlag(cfg, "svt-lp"); err != nil {
			return err
		}
		if err := requireExplicitFlag(cfg, "svt-asm"); err != nil {
			return err
		}
	}
	return nil
}

func requireExplicitFlag(cfg benchConfig, name string) error {
	if !cfg.explicitFlags[name] {
		return fmt.Errorf("publish requires explicit -%s", name)
	}
	return nil
}

func encoderSelected(cfg benchConfig, name string) bool {
	for _, selected := range cfg.encoders {
		if selected == name {
			return true
		}
	}
	return false
}

func validatePublishClipInputs(clips []clipSpec) error {
	for _, clip := range clips {
		if clip.Input == "" {
			return fmt.Errorf("publish requires %s to use a manifest input file", clip.Name)
		}
		info, err := os.Stat(clip.Input)
		if err != nil {
			return fmt.Errorf("%s input: %w", clip.Name, err)
		}
		expected := expectedRawI420Bytes(clip.Width, clip.Height, clip.Frames)
		if info.Size() != expected {
			return fmt.Errorf("%s input size=%d, want exact raw I420 size %d", clip.Name, info.Size(), expected)
		}
	}
	return nil
}

func validRunOrder(order string) bool {
	return order == runOrderBitrateEncoder || order == runOrderEncoderBitrate || order == runOrderShuffle
}

func encodeJobsForConfig(cfg benchConfig) []encodeJob {
	jobs := make([]encodeJob, 0, len(cfg.bitrates)*len(cfg.encoders))
	switch cfg.runOrder {
	case runOrderEncoderBitrate:
		for _, encoderName := range cfg.encoders {
			for _, bitrate := range cfg.bitrates {
				jobs = append(jobs, encodeJob{bitrate: bitrate, encoder: encoderName})
			}
		}
	default:
		for _, bitrate := range cfg.bitrates {
			for _, encoderName := range cfg.encoders {
				jobs = append(jobs, encodeJob{bitrate: bitrate, encoder: encoderName})
			}
		}
	}
	if cfg.runOrder == runOrderShuffle {
		rng := rand.New(rand.NewSource(cfg.shuffleSeed))
		rng.Shuffle(len(jobs), func(i, j int) {
			jobs[i], jobs[j] = jobs[j], jobs[i]
		})
	}
	return jobs
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

func parseEncoderList(s string) []string {
	names := parseNameList(s)
	for i, name := range names {
		names[i] = canonicalEncoderName(name)
	}
	return names
}

func parseMetricList(s string) ([]string, error) {
	names := parseNameList(s)
	out := make([]string, 0, len(names))
	seen := map[string]bool{}
	for _, name := range names {
		switch name {
		case "psnr", "psnr_avg":
			name = "psnr"
		case "ssim", "ssim_all":
			name = "ssim"
		case "xpsnr", "xpsnr_y":
			name = "xpsnr"
		case "vmaf":
		default:
			return nil, fmt.Errorf("unknown metric %q", name)
		}
		if !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	return out, nil
}

func canonicalSVTASMName(name string) (string, bool) {
	switch strings.TrimSpace(strings.ToLower(name)) {
	case "c":
		return "c", true
	case "neon":
		return "neon", true
	case "crc32":
		return "crc32", true
	case "neon_dotprod", "dotprod":
		return "neon_dotprod", true
	case "neon_i8mm", "i8mm":
		return "neon_i8mm", true
	case "sve":
		return "sve", true
	case "sve2":
		return "sve2", true
	case "max":
		return "max", true
	default:
		return "", false
	}
}

func parseRequiredEncoderList(s string, selected []string) ([]string, error) {
	names := parseNameList(s)
	if len(names) == 0 {
		return nil, nil
	}
	if len(names) == 1 && names[0] == "all" {
		return requireAllSelectedEncoders(selected)
	}
	selectedSet := requiredEncoderSet(selected)
	out := make([]string, 0, len(names))
	seen := map[string]bool{}
	for _, name := range names {
		if name == "all" {
			return nil, errors.New("all cannot be mixed with explicit encoder names")
		}
		canonical, ok := canonicalKnownEncoderName(name)
		if !ok {
			return nil, fmt.Errorf("unknown encoder %q", name)
		}
		if !selectedSet[canonical] {
			return nil, fmt.Errorf("required encoder %s is not selected by -encoders", canonical)
		}
		if !seen[canonical] {
			seen[canonical] = true
			out = append(out, canonical)
		}
	}
	return out, nil
}

func requireAllSelectedEncoders(selected []string) ([]string, error) {
	out := make([]string, 0, len(selected))
	seen := map[string]bool{}
	for _, name := range selected {
		canonical, ok := canonicalKnownEncoderName(name)
		if !ok {
			return nil, fmt.Errorf("unknown selected encoder %q", name)
		}
		if !seen[canonical] {
			seen[canonical] = true
			out = append(out, canonical)
		}
	}
	return out, nil
}

func canonicalEncoderName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	if canonical, ok := canonicalKnownEncoderName(name); ok {
		return canonical
	}
	return name
}

func canonicalKnownEncoderName(name string) (string, bool) {
	switch strings.TrimSpace(strings.ToLower(name)) {
	case "goav1":
		return "goav1", true
	case "aomenc", "libaom", "libaom-av1":
		return "aomenc", true
	case "svt-av1", "svtav1", "svt", "svtav1encapp", "svt-av1encapp":
		return "svt-av1", true
	default:
		return "", false
	}
}

func requiredMetricSet(metrics []string) map[string]bool {
	out := make(map[string]bool, len(metrics))
	for _, metric := range metrics {
		out[metric] = true
	}
	return out
}

func requiredEncoderSet(encoders []string) map[string]bool {
	out := make(map[string]bool, len(encoders))
	for _, encoder := range encoders {
		out[canonicalEncoderName(encoder)] = true
	}
	return out
}

func validateRequiredMetrics(filters map[string]bool, metrics []string) error {
	for _, metric := range metrics {
		filter := ffmpegFilterForMetric(metric)
		if !filters[filter] {
			return fmt.Errorf("required metric %s unavailable: ffmpeg filter %s not found", metric, filter)
		}
	}
	return nil
}

func ffmpegFilterForMetric(metric string) string {
	if metric == "vmaf" {
		return "libvmaf"
	}
	return metric
}

func clipSpecsForConfig(cfg benchConfig) ([]clipSpec, error) {
	if cfg.manifestPath != "" {
		return readClipManifest(cfg.manifestPath, cfg)
	}
	name := "synthetic"
	if cfg.input != "" {
		name = filepath.Base(cfg.input)
	}
	return []clipSpec{{
		Name:   name,
		Input:  cfg.input,
		Width:  cfg.width,
		Height: cfg.height,
		Frames: cfg.frames,
		FPS:    cfg.fps,
	}}, nil
}

func readClipManifest(path string, defaults benchConfig) ([]clipSpec, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, errors.New("manifest needs a header and at least one clip")
	}
	header := map[string]int{}
	for i, field := range records[0] {
		header[strings.ToLower(strings.TrimSpace(field))] = i
	}
	inputCol, ok := header["input"]
	if !ok {
		return nil, errors.New("manifest missing input column")
	}
	widthCol, ok := header["width"]
	if !ok {
		return nil, errors.New("manifest missing width column")
	}
	heightCol, ok := header["height"]
	if !ok {
		return nil, errors.New("manifest missing height column")
	}
	framesCol, ok := header["frames"]
	if !ok {
		return nil, errors.New("manifest missing frames column")
	}
	clipCol, haveClip := header["clip"]
	fpsCol, haveFPS := header["fps"]

	clips := make([]clipSpec, 0, len(records)-1)
	for rowIndex, record := range records[1:] {
		rowNum := rowIndex + 2
		input := manifestField(record, inputCol)
		if input != "" && !filepath.IsAbs(input) {
			input = filepath.Join(filepath.Dir(path), input)
		}
		width, err := parseManifestPositiveInt(record, widthCol, "width", rowNum)
		if err != nil {
			return nil, err
		}
		height, err := parseManifestPositiveInt(record, heightCol, "height", rowNum)
		if err != nil {
			return nil, err
		}
		frames, err := parseManifestPositiveInt(record, framesCol, "frames", rowNum)
		if err != nil {
			return nil, err
		}
		fps := defaults.fps
		if haveFPS && strings.TrimSpace(manifestField(record, fpsCol)) != "" {
			fps, err = parseManifestPositiveInt(record, fpsCol, "fps", rowNum)
			if err != nil {
				return nil, err
			}
		}
		name := ""
		if haveClip {
			name = strings.TrimSpace(manifestField(record, clipCol))
		}
		if name == "" {
			if input != "" {
				name = filepath.Base(input)
			} else {
				name = fmt.Sprintf("clip%d", len(clips)+1)
			}
		}
		if width < 16 || height < 16 || width%2 != 0 || height%2 != 0 {
			return nil, fmt.Errorf("manifest row %d invalid frame size %dx%d: need even dimensions >= 16", rowNum, width, height)
		}
		clips = append(clips, clipSpec{
			Name:   name,
			Input:  input,
			Width:  width,
			Height: height,
			Frames: frames,
			FPS:    fps,
		})
	}
	return clips, nil
}

func manifestField(record []string, col int) string {
	if col < 0 || col >= len(record) {
		return ""
	}
	return strings.TrimSpace(record[col])
}

func parseManifestPositiveInt(record []string, col int, name string, row int) (int, error) {
	raw := manifestField(record, col)
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("manifest row %d invalid %s %q", row, name, raw)
	}
	return n, nil
}

func validateRequiredCorpus(cfg benchConfig, clips []clipSpec) error {
	if cfg.requireCorpus && cfg.manifestPath == "" {
		return errors.New("require-corpus requires -manifest")
	}
	if cfg.minClips > 0 && len(clips) < cfg.minClips {
		return fmt.Errorf("clip corpus requires at least %d clips, got %d", cfg.minClips, len(clips))
	}
	if !cfg.requireCorpus {
		return nil
	}
	if len(clips) == 0 {
		return errors.New("required corpus has no clips")
	}
	for _, clip := range clips {
		if strings.TrimSpace(clip.Input) == "" {
			return fmt.Errorf("%s: required corpus clip has no input path; synthetic clips are not allowed", clip.Name)
		}
		info, err := os.Stat(clip.Input)
		if err != nil {
			return fmt.Errorf("%s: required corpus input %s: %w", clip.Name, clip.Input, err)
		}
		if info.IsDir() {
			return fmt.Errorf("%s: required corpus input %s is a directory", clip.Name, clip.Input)
		}
	}
	return nil
}

func runClip(cfg benchConfig, clip clipSpec, filters map[string]bool, writer *csv.Writer, statsWriter *csv.Writer, frameStatsWriter *csv.Writer, frameMetricsWriter *csv.Writer) ([]benchRow, []encoderInvocationMetadata, error) {
	clipCfg := cfg
	clipCfg.input = clip.Input
	clipCfg.width = clip.Width
	clipCfg.height = clip.Height
	clipCfg.frames = clip.Frames
	clipCfg.fps = clip.FPS
	if cfg.manifestPath != "" {
		clipCfg.workdir = filepath.Join(cfg.workdir, safeClipDir(clip.Name))
		if err := os.MkdirAll(clipCfg.workdir, 0o755); err != nil {
			return nil, nil, err
		}
	}

	frames, _, err := loadFrames(clipCfg)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", clip.Name, err)
	}
	refPath := filepath.Join(clipCfg.workdir, "source.yuv")
	if err := writeFrames(refPath, frames, clip.Width, clip.Height); err != nil {
		return nil, nil, fmt.Errorf("%s source: %w", clip.Name, err)
	}

	var rows []benchRow
	var invocations []encoderInvocationMetadata
	required := requiredMetricSet(cfg.requiredMetrics)
	requiredEncoders := requiredEncoderSet(cfg.requiredEncoders)
	for _, job := range encodeJobsForConfig(cfg) {
		bitrate, encoderName := job.bitrate, job.encoder
		result := runEncoder(clipCfg, frames, refPath, encoderName, bitrate)
		m := metrics{psnr: "NA", ssim: "NA", xpsnr: "NA", vmaf: "NA"}
		var metricErr error
		if result.status == "ok" {
			m, metricErr = measureDecoded(clipCfg, filters, required, refPath, result.decodedYUV, encoderName, bitrate)
			if metricErr != nil {
				result.status, result.errText = "error", metricErr.Error()
			}
			if metricErr == nil && frameMetricsWriter != nil {
				if err := writeFrameMetricRows(clipCfg, filters, frameMetricsWriter, clip.Name, result.encoder, bitrate, refPath, result.decodedYUV); err != nil {
					return rows, invocations, err
				}
				frameMetricsWriter.Flush()
			}
		}
		actualBPS := int64(0)
		if result.bytes > 0 && clip.Frames > 0 {
			actualBPS = result.bytes * 8 * int64(clip.FPS) / int64(clip.Frames)
		}
		encodeFPS := ""
		if result.duration > 0 {
			encodeFPS = strconv.FormatFloat(float64(clip.Frames)/result.duration.Seconds(), 'f', 2, 64)
		}
		row := benchRow{
			clip:      clip.Name,
			width:     clip.Width,
			height:    clip.Height,
			frames:    clip.Frames,
			fps:       clip.FPS,
			encoder:   result.encoder,
			targetBPS: result.targetBPS,
			actualBPS: actualBPS,
			duration:  result.duration,
			cpuUser:   result.cpuUser,
			cpuSystem: result.cpuSystem,
			cpuOK:     result.cpuAvailable,
			encodeFPS: encodeFPS,
			bytes:     result.bytes,
			metrics:   m,
			status:    result.status,
			errText:   result.errText,
		}
		rows = append(rows, row)
		encodedPath, encodedContainer := result.encodedPath, result.encodedContainer
		if result.encodedSHA256 == "" {
			encodedPath, encodedContainer = "", ""
		}
		decodedPath := result.decodedYUV
		if result.decodedSHA256 == "" {
			decodedPath = ""
		}
		invocations = append(invocations, encoderInvocationMetadata{
			Clip:             clip.Name,
			Width:            clip.Width,
			Height:           clip.Height,
			Frames:           clip.Frames,
			FPS:              clip.FPS,
			Encoder:          result.encoder,
			TargetBPS:        result.targetBPS,
			ActualBPS:        actualBPS,
			CompressedBytes:  result.bytes,
			EncodedPath:      encodedPath,
			EncodedContainer: encodedContainer,
			EncodedBytes:     result.encodedBytes,
			EncodedSHA256:    result.encodedSHA256,
			DecodedPath:      decodedPath,
			DecodedBytes:     result.decodedBytes,
			DecodedSHA256:    result.decodedSHA256,
			EncodeWallSecs:   durationSeconds(result.duration),
			CPUAvailable:     result.cpuAvailable,
			CPUUserSecs:      durationSeconds(result.cpuUser),
			CPUSystemSecs:    durationSeconds(result.cpuSystem),
			CPUTotalSecs:     durationSeconds(totalCPU(result.cpuUser, result.cpuSystem)),
			ObservedParallel: observedParallelism(result.duration, result.cpuUser, result.cpuSystem, result.cpuAvailable),
			Status:           result.status,
			Error:            result.errText,
			Command:          result.command,
			Settings:         result.settings,
		})
		if err := writeBenchRow(writer, row); err != nil {
			return nil, nil, err
		}
		if statsWriter != nil && result.encoder == "goav1" {
			if err := writeStatsRow(statsWriter, row, result.stats); err != nil {
				return nil, nil, err
			}
			statsWriter.Flush()
		}
		if frameStatsWriter != nil && result.encoder == "goav1" {
			for _, frameStats := range result.frameStats {
				if err := writeFrameStatsRow(frameStatsWriter, row, frameStats); err != nil {
					return nil, nil, err
				}
			}
			frameStatsWriter.Flush()
		}
		writer.Flush()
		if err := requiredEncoderError(requiredEncoders, clip.Name, result, bitrate); err != nil {
			return rows, invocations, err
		}
		if metricErr != nil {
			return rows, invocations, fmt.Errorf("%s %s %d bps: %w", clip.Name, result.encoder, bitrate, metricErr)
		}
	}
	return rows, invocations, nil
}

func requiredEncoderError(required map[string]bool, clipName string, result encodeResult, bitrate int) error {
	if !required[result.encoder] || result.status == "ok" {
		return nil
	}
	if result.errText != "" {
		return fmt.Errorf("%s %s %d bps: required encoder status %s: %s", clipName, result.encoder, bitrate, result.status, result.errText)
	}
	return fmt.Errorf("%s %s %d bps: required encoder status %s", clipName, result.encoder, bitrate, result.status)
}

func safeClipDir(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "clip"
	}
	return b.String()
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
		formatSeconds(row.duration),
		formatCPUSeconds(row.cpuUser, row.cpuOK),
		formatCPUSeconds(row.cpuSystem, row.cpuOK),
		formatCPUSeconds(totalCPU(row.cpuUser, row.cpuSystem), row.cpuOK),
		formatObservedParallelism(row.duration, row.cpuUser, row.cpuSystem, row.cpuOK),
		strconv.FormatInt(row.bytes, 10),
		row.metrics.psnr,
		row.metrics.ssim,
		row.metrics.xpsnr,
		row.metrics.vmaf,
		row.status,
		row.errText,
	})
}

func totalCPU(user, system time.Duration) time.Duration {
	return user + system
}

func durationSeconds(d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return d.Seconds()
}

func observedParallelism(wall, user, system time.Duration, cpuOK bool) float64 {
	if !cpuOK || wall <= 0 {
		return 0
	}
	return totalCPU(user, system).Seconds() / wall.Seconds()
}

func formatSeconds(d time.Duration) string {
	if d <= 0 {
		return ""
	}
	return strconv.FormatFloat(d.Seconds(), 'f', 3, 64)
}

func formatCPUSeconds(d time.Duration, ok bool) string {
	if !ok {
		return ""
	}
	return strconv.FormatFloat(d.Seconds(), 'f', 3, 64)
}

func formatObservedParallelism(wall, user, system time.Duration, cpuOK bool) string {
	v := observedParallelism(wall, user, system, cpuOK)
	if v == 0 {
		return ""
	}
	return strconv.FormatFloat(v, 'f', 2, 64)
}

func writeStatsHeader(writer *csv.Writer) error {
	return writer.Write([]string{
		"clip", "width", "height", "frames", "fps", "encoder", "target_bps",
		"actual_bps", "encoded_frames", "keyframes", "inter_frames", "tiles",
		"partition_decisions", "blocks", "inter_blocks", "intra_blocks",
		"skip_blocks", "coded_blocks", "compound_blocks", "split_tx_blocks",
		"single_tx_blocks", "luma_txbs", "primary_last", "primary_golden",
		"block_8x8", "block_16x16", "block_32x32", "block_64x64",
		"block_16x8", "block_8x16", "block_32x16", "block_16x32",
		"tx_dct_dct", "tx_adst_dct", "tx_dct_adst", "tx_adst_adst",
		"tx_idtx", "non_dct_txbs", "status", "error",
	})
}

func writeStatsRow(writer *csv.Writer, row benchRow, stats goav1.EncoderDecisionStats) error {
	return writer.Write([]string{
		row.clip,
		strconv.Itoa(row.width),
		strconv.Itoa(row.height),
		strconv.Itoa(row.frames),
		strconv.Itoa(row.fps),
		row.encoder,
		strconv.Itoa(row.targetBPS),
		strconv.FormatInt(row.actualBPS, 10),
		formatU64(stats.Frames),
		formatU64(stats.Keyframes),
		formatU64(stats.InterFrames),
		formatU64(stats.Tiles),
		formatU64(stats.PartitionDecisions),
		formatU64(stats.Blocks),
		formatU64(stats.InterBlocks),
		formatU64(stats.IntraBlocks),
		formatU64(stats.SkipBlocks),
		formatU64(stats.CodedBlocks),
		formatU64(stats.CompoundBlocks),
		formatU64(stats.SplitTXBlocks),
		formatU64(stats.SingleTXBlocks),
		formatU64(stats.LumaTXBs),
		formatU64(stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceLast]),
		formatU64(stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceGolden]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize8x8]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize16x16]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize32x32]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize64x64]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize16x8]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize8x16]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize32x16]),
		formatU64(stats.BlockSizes[goav1.EncoderDecisionBlockSize16x32]),
		formatU64(stats.TXTypes[goav1.EncoderDecisionTransformDCTDCT]),
		formatU64(stats.TXTypes[goav1.EncoderDecisionTransformADSTDCT]),
		formatU64(stats.TXTypes[goav1.EncoderDecisionTransformDCTADST]),
		formatU64(stats.TXTypes[goav1.EncoderDecisionTransformADSTADST]),
		formatU64(stats.TXTypes[goav1.EncoderDecisionTransformIDTX]),
		formatU64(stats.NonDCTTXBs),
		row.status,
		row.errText,
	})
}

func writeFrameStatsHeader(writer *csv.Writer) error {
	return writer.Write([]string{
		"clip", "width", "height", "fps", "encoder", "target_bps",
		"frame_index", "keyframe", "temporal_id", "qindex_before", "qindex_after",
		"frame_bytes", "frame_bits", "cumulative_bytes", "encode_ms",
		"encoded_frames", "keyframes", "inter_frames", "tiles",
		"partition_decisions", "blocks", "inter_blocks", "intra_blocks",
		"skip_blocks", "coded_blocks", "compound_blocks", "primary_last",
		"primary_golden", "tx_non_dct", "status", "error",
	})
}

func writeFrameStatsRow(writer *csv.Writer, row benchRow, frame goAV1FrameStats) error {
	stats := frame.Stats
	return writer.Write([]string{
		row.clip,
		strconv.Itoa(row.width),
		strconv.Itoa(row.height),
		strconv.Itoa(row.fps),
		row.encoder,
		strconv.Itoa(row.targetBPS),
		strconv.Itoa(frame.FrameIndex),
		strconv.FormatBool(frame.Keyframe),
		strconv.Itoa(int(frame.TemporalID)),
		strconv.Itoa(int(frame.QIndexBefore)),
		strconv.Itoa(int(frame.QIndexAfter)),
		strconv.FormatInt(frame.Bytes, 10),
		strconv.FormatInt(frame.Bytes*8, 10),
		strconv.FormatInt(frame.CumulativeBytes, 10),
		strconv.FormatFloat(float64(frame.Duration)/float64(time.Millisecond), 'f', 3, 64),
		formatU64(stats.Frames),
		formatU64(stats.Keyframes),
		formatU64(stats.InterFrames),
		formatU64(stats.Tiles),
		formatU64(stats.PartitionDecisions),
		formatU64(stats.Blocks),
		formatU64(stats.InterBlocks),
		formatU64(stats.IntraBlocks),
		formatU64(stats.SkipBlocks),
		formatU64(stats.CodedBlocks),
		formatU64(stats.CompoundBlocks),
		formatU64(stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceLast]),
		formatU64(stats.PrimaryReferenceBlocks[goav1.EncoderDecisionReferenceGolden]),
		formatU64(stats.NonDCTTXBs),
		row.status,
		row.errText,
	})
}

func writeFrameMetricsHeader(writer *csv.Writer) error {
	return writer.Write([]string{
		"clip", "encoder", "target_bps", "frame_index", "metric", "value",
	})
}

func writeFrameMetricRows(cfg benchConfig, filters map[string]bool, writer *csv.Writer, clipName, encoderName string, bitrate int, refPath, decodedPath string) error {
	if filters["psnr"] {
		values, err := runFrameMetric(cfg, refPath, decodedPath, encoderName, bitrate, "psnr", `psnr_avg:([0-9.]+|inf|Inf|INF)`)
		if err != nil {
			return err
		}
		for _, value := range values {
			if err := writeFrameMetricRow(writer, clipName, encoderName, bitrate, value.FrameIndex, "psnr_avg", value.Value); err != nil {
				return err
			}
		}
	}
	if filters["ssim"] {
		values, err := runFrameMetric(cfg, refPath, decodedPath, encoderName, bitrate, "ssim", `All:([0-9.]+|inf|Inf|INF)`)
		if err != nil {
			return err
		}
		for _, value := range values {
			if err := writeFrameMetricRow(writer, clipName, encoderName, bitrate, value.FrameIndex, "ssim_all", value.Value); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeFrameMetricRow(writer *csv.Writer, clipName, encoderName string, bitrate int, frameIndex int, metricName, value string) error {
	return writer.Write([]string{
		clipName,
		encoderName,
		strconv.Itoa(bitrate),
		strconv.Itoa(frameIndex),
		metricName,
		value,
	})
}

func goAV1DiagnosticsEnabled(cfg benchConfig) bool {
	return cfg.statsCSVPath != "" || cfg.frameStatsCSVPath != ""
}

func diffDecisionStats(after, before goav1.EncoderDecisionStats) goav1.EncoderDecisionStats {
	var out goav1.EncoderDecisionStats
	out.Frames = subU64(after.Frames, before.Frames)
	out.Keyframes = subU64(after.Keyframes, before.Keyframes)
	out.InterFrames = subU64(after.InterFrames, before.InterFrames)
	out.Tiles = subU64(after.Tiles, before.Tiles)
	out.PartitionDecisions = subU64(after.PartitionDecisions, before.PartitionDecisions)
	for i := range out.Partitions {
		out.Partitions[i] = subU64(after.Partitions[i], before.Partitions[i])
	}
	for i := range out.PartitionsByLevel {
		for j := range out.PartitionsByLevel[i] {
			out.PartitionsByLevel[i][j] = subU64(after.PartitionsByLevel[i][j], before.PartitionsByLevel[i][j])
		}
	}
	out.Blocks = subU64(after.Blocks, before.Blocks)
	out.InterBlocks = subU64(after.InterBlocks, before.InterBlocks)
	out.IntraBlocks = subU64(after.IntraBlocks, before.IntraBlocks)
	out.SkipBlocks = subU64(after.SkipBlocks, before.SkipBlocks)
	out.CodedBlocks = subU64(after.CodedBlocks, before.CodedBlocks)
	out.CompoundBlocks = subU64(after.CompoundBlocks, before.CompoundBlocks)
	for i := range out.BlockSizes {
		out.BlockSizes[i] = subU64(after.BlockSizes[i], before.BlockSizes[i])
	}
	for i := range out.PrimaryReferenceBlocks {
		out.PrimaryReferenceBlocks[i] = subU64(after.PrimaryReferenceBlocks[i], before.PrimaryReferenceBlocks[i])
		out.ReferenceUses[i] = subU64(after.ReferenceUses[i], before.ReferenceUses[i])
	}
	for i := range out.InterModes {
		out.InterModes[i] = subU64(after.InterModes[i], before.InterModes[i])
	}
	for i := range out.CompoundInterModes {
		out.CompoundInterModes[i] = subU64(after.CompoundInterModes[i], before.CompoundInterModes[i])
	}
	out.SplitTXBlocks = subU64(after.SplitTXBlocks, before.SplitTXBlocks)
	out.SingleTXBlocks = subU64(after.SingleTXBlocks, before.SingleTXBlocks)
	out.LumaTXBs = subU64(after.LumaTXBs, before.LumaTXBs)
	for i := range out.TXTypes {
		out.TXTypes[i] = subU64(after.TXTypes[i], before.TXTypes[i])
	}
	out.NonDCTTXBs = subU64(after.NonDCTTXBs, before.NonDCTTXBs)
	return out
}

func subU64(after, before uint64) uint64 {
	if after < before {
		return 0
	}
	return after - before
}

func formatU64(n uint64) string {
	return strconv.FormatUint(n, 10)
}

func writeMetadataJSON(cfg benchConfig, filters map[string]bool, clips []clipSpec, invocations []encoderInvocationMetadata) error {
	clipMetadata, err := clipMetadataFor(clips)
	if err != nil {
		return err
	}
	doc := qualitybenchMetadata{
		GeneratedAtUTC: time.Now().UTC().Format(time.RFC3339Nano),
		Go: runtimeMetadata{
			Version:      runtime.Version(),
			GOOS:         runtime.GOOS,
			GOARCH:       runtime.GOARCH,
			SIMDTier:     detectedSIMDTier(),
			SIMDFeatures: detectedSIMDFeatures(),
		},
		Git:           currentGitMetadata(),
		Config:        metadataConfigFor(cfg),
		FairnessNotes: fairnessNotes(cfg),
		MetricFilters: metricFilterAvailability(filters),
		Tools:         toolMetadataForRun(),
		Clips:         clipMetadata,
		Encodes:       invocations,
	}
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return os.WriteFile(cfg.metadataPath, raw, 0o644)
}

func clipMetadataFor(clips []clipSpec) ([]clipMetadata, error) {
	out := make([]clipMetadata, 0, len(clips))
	for _, clip := range clips {
		meta := clipMetadata{
			Name:          clip.Name,
			Input:         clip.Input,
			Synthetic:     clip.Input == "",
			Width:         clip.Width,
			Height:        clip.Height,
			Frames:        clip.Frames,
			FPS:           clip.FPS,
			ExpectedBytes: expectedRawI420Bytes(clip.Width, clip.Height, clip.Frames),
		}
		if clip.Input != "" {
			info, err := os.Stat(clip.Input)
			if err != nil {
				return nil, fmt.Errorf("%s input metadata: %w", clip.Name, err)
			}
			meta.InputBytes = info.Size()
			hash, err := sha256File(clip.Input)
			if err != nil {
				return nil, fmt.Errorf("%s input metadata: %w", clip.Name, err)
			}
			meta.SHA256 = hash
		}
		out = append(out, meta)
	}
	return out, nil
}

func expectedRawI420Bytes(width, height, frames int) int64 {
	return int64(frames) * int64(width*height+2*(width/2)*(height/2))
}

func sha256File(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func fileBytesAndSHA256(path string) (int64, string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, "", err
	}
	if info.IsDir() {
		return 0, "", fmt.Errorf("%s is a directory", path)
	}
	hash, err := sha256File(path)
	if err != nil {
		return 0, "", err
	}
	return info.Size(), hash, nil
}

func metadataConfigFor(cfg benchConfig) metadataConfig {
	encoders := append([]string(nil), cfg.encoders...)
	bitrates := append([]int(nil), cfg.bitrates...)
	required := append([]string(nil), cfg.requiredMetrics...)
	requiredEncoders := append([]string(nil), cfg.requiredEncoders...)
	return metadataConfig{
		Width:            cfg.width,
		Height:           cfg.height,
		Frames:           cfg.frames,
		FPS:              cfg.fps,
		Input:            cfg.input,
		Manifest:         cfg.manifestPath,
		Encoders:         encoders,
		Bitrates:         bitrates,
		RequiredMetrics:  required,
		RequiredEncoders: requiredEncoders,
		RequireSummary:   cfg.requireSummary,
		RequireCorpus:    cfg.requireCorpus,
		MinClips:         cfg.minClips,
		Anchor:           cfg.anchorEncoder,
		Layers:           cfg.layers,
		Tiles:            cfg.tiles,
		GoldenInterval:   cfg.goldenInterval,
		KeyInterval:      cfg.keyInterval,
		GoMaxProcs:       cfg.goMaxProcs,
		AOMThreads:       cfg.aomThreads,
		AOMRowMT:         cfg.aomRowMT,
		SVTLP:            cfg.svtLP,
		SVTASM:           cfg.svtASM,
		TimingMode:       cfg.timingMode,
		RunOrder:         cfg.runOrder,
		ShuffleSeed:      cfg.shuffleSeed,
		Publish:          cfg.publish,
	}
}

func detectedSIMDTier() string {
	return simdTierFor(cpufeatures.Detected)
}

func detectedSIMDFeatures() []string {
	return simdFeaturesFor(cpufeatures.Detected)
}

func simdTierFor(f cpufeatures.Features) string {
	switch {
	case f.SVE2:
		return "sve2"
	case f.SVE:
		return "sve"
	case f.I8MM:
		return "neon_i8mm"
	case f.DOTPROD:
		return "neon_dotprod"
	case f.NEON:
		return "neon"
	case f.AVX512:
		return "avx512"
	case f.AVX2:
		return "avx2"
	case f.SSE42:
		return "sse4_2"
	case f.SSE41:
		return "sse4_1"
	case f.SSE2:
		return "sse2"
	default:
		return "purego"
	}
}

func simdFeaturesFor(f cpufeatures.Features) []string {
	var out []string
	if f.SSE2 {
		out = append(out, "sse2")
	}
	if f.SSE41 {
		out = append(out, "sse4_1")
	}
	if f.SSE42 {
		out = append(out, "sse4_2")
	}
	if f.AVX2 {
		out = append(out, "avx2")
	}
	if f.AVX512 {
		out = append(out, "avx512")
	}
	if f.NEON {
		out = append(out, "neon")
	}
	if f.DOTPROD {
		out = append(out, "neon_dotprod")
	}
	if f.I8MM {
		out = append(out, "neon_i8mm")
	}
	if f.SVE {
		out = append(out, "sve")
	}
	if f.SVE2 {
		out = append(out, "sve2")
	}
	return out
}

func fairnessNotes(cfg benchConfig) []string {
	notes := []string{
		"SVT-AV1 --lp is a documented parallelism level in the range 0..6, not a target processor or thread count; numeric equality with GOMAXPROCS is not treated as equivalent concurrency.",
		"CSV and metadata include wall seconds, CPU seconds, and observed_parallelism=cpu_total_seconds/encode_wall_seconds so comparisons can be checked against observed CPU budget.",
		"qualitybench records timing_mode; core mode keeps the historical goav1 per-frame Encode timer, while e2e mode times goav1 setup, encode calls, and decoded-output writes for fairer CLI comparisons.",
		"qualitybench records run_order and shuffle_seed so encoder/bitrate order effects can be reproduced or randomized deterministically.",
		"For fair SVT comparisons, keep GOMAXPROCS explicit for goav1 and either leave SVT at --lp 0 or sweep --lp 0..6, then report the SVT level whose observed_parallelism is closest to goav1 rather than matching knob values.",
		"For fair libaom comparisons, set -aom-threads and -aom-row-mt explicitly and report both; qualitybench forwards them to aomenc --threads and --row-mt and records them in metadata.",
		"SVT-AV1 --asm defaults to max and may use CPU-specific kernels such as neon_dotprod or neon_i8mm; use -svt-asm to pin the assembly tier when comparing against goav1's current SIMD coverage.",
		"goav1 metadata records detected simd_tier and simd_features; compare those against SVT's recorded svt_asm setting instead of assuming --asm max and goav1 cover the same kernels.",
	}
	if cfg.publish {
		notes = append(notes, "Publish mode required a clean tracked git worktree, explicit artifact paths, manifest-backed corpus, exact raw input sizes, explicit concurrency controls, required encoders, required metrics, and required BD-rate summary rows.")
	}
	if cfg.svtLP == 0 {
		notes = append(notes, "SVT-AV1 is run with --lp 0 by default, letting SVT choose its parallelism level from the machine rather than forcing a misleading numeric match.")
	}
	if cfg.svtASM == "" {
		notes = append(notes, "SVT-AV1 is run with its default --asm max setting; report this as a best-SVT row, not as baseline-NEON-equivalent SIMD coverage.")
	} else {
		notes = append(notes, "SVT-AV1 is run with --asm "+cfg.svtASM+" so the comparison records an explicit assembly tier.")
	}
	return notes
}

func metricFilterAvailability(filters map[string]bool) map[string]bool {
	return map[string]bool{
		"psnr":    filters["psnr"],
		"ssim":    filters["ssim"],
		"xpsnr":   filters["xpsnr"],
		"libvmaf": filters["libvmaf"],
	}
}

func toolMetadataForRun() map[string]toolMetadata {
	return map[string]toolMetadata{
		"ffmpeg":       commandMetadata("ffmpeg", []string{"-hide_banner", "-version"}),
		"aomenc":       commandMetadata("aomenc", []string{"--version"}, []string{"--help"}),
		"SvtAv1EncApp": commandMetadata("SvtAv1EncApp", []string{"--version"}),
	}
}

func commandMetadata(name string, versionArgSets ...[]string) toolMetadata {
	path, err := exec.LookPath(name)
	if err != nil {
		return toolMetadata{Found: false, VersionError: err.Error()}
	}
	meta := toolMetadata{Found: true, Path: path}
	if hash, err := sha256File(path); err == nil {
		meta.SHA256 = hash
	}
	for _, args := range versionArgSets {
		out, err := exec.Command(path, args...).CombinedOutput()
		line := firstNonEmptyLine(string(out))
		if err != nil {
			meta.VersionError = trimCommandOutput(err, out)
			continue
		}
		meta.VersionError = ""
		if line != "" {
			meta.Version = line
		}
		return meta
	}
	return meta
}

func firstNonEmptyLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			return line
		}
	}
	return ""
}

func currentGitMetadata() gitMetadata {
	var meta gitMetadata
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		meta.Error = err.Error()
		return meta
	}
	meta.Commit = strings.TrimSpace(string(out))
	status, err := exec.Command("git", "status", "--short", "--untracked-files=no").Output()
	if err != nil {
		meta.Error = err.Error()
		return meta
	}
	meta.Dirty = strings.TrimSpace(string(status)) != ""
	return meta
}

func writeSummaryCSV(path string, summaries []summaryRow) error {
	file, err := os.Create(path)
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
	for _, summary := range summaries {
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

func validateRequiredSummaries(cfg benchConfig, rows []benchRow, summaries []summaryRow) error {
	clips := clipNamesFromRows(rows)
	if len(clips) == 0 {
		return errors.New("required BD-rate summary unavailable: no benchmark rows")
	}
	anchor := canonicalEncoderName(cfg.anchorEncoder)
	byKey := make(map[string]summaryRow, len(summaries))
	for _, summary := range summaries {
		byKey[summaryKey(summary.Clip, summary.Encoder, summary.Metric)] = summary
	}
	for _, clip := range clips {
		expected := 0
		for _, encoder := range cfg.encoders {
			encoder = canonicalEncoderName(encoder)
			if encoder == anchor {
				continue
			}
			for _, metric := range cfg.requiredMetrics {
				expected++
				metricName := summaryMetricName(metric)
				summary, ok := byKey[summaryKey(clip, encoder, metricName)]
				if !ok {
					return fmt.Errorf("%s %s %s: required BD-rate summary missing", clip, encoder, metricName)
				}
				if summary.Status != "ok" {
					if summary.ErrText != "" {
						return fmt.Errorf("%s %s %s: required BD-rate summary status %s: %s", clip, encoder, metricName, summary.Status, summary.ErrText)
					}
					return fmt.Errorf("%s %s %s: required BD-rate summary status %s", clip, encoder, metricName, summary.Status)
				}
			}
		}
		if expected == 0 {
			return fmt.Errorf("%s: required BD-rate summary unavailable: no candidate encoders", clip)
		}
	}
	return nil
}

func clipNamesFromRows(rows []benchRow) []string {
	seen := map[string]bool{}
	var clips []string
	for _, row := range rows {
		if row.clip == "" || seen[row.clip] {
			continue
		}
		seen[row.clip] = true
		clips = append(clips, row.clip)
	}
	return clips
}

func summaryMetricName(metric string) string {
	switch metric {
	case "psnr":
		return "psnr_avg"
	case "ssim":
		return "ssim_all"
	case "xpsnr":
		return "xpsnr_y"
	default:
		return metric
	}
}

func summaryKey(clip, encoder, metric string) string {
	return clip + "\x00" + canonicalEncoderName(encoder) + "\x00" + metric
}

func summarizeBDRate(anchor string, rows []benchRow) []summaryRow {
	anchor = canonicalEncoderName(anchor)
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

func writeLengthPrefixedPayload(w io.Writer, payload []byte) error {
	if uint64(len(payload)) > uint64(^uint32(0)) {
		return fmt.Errorf("payload too large: %d bytes", len(payload))
	}
	var length [4]byte
	binary.LittleEndian.PutUint32(length[:], uint32(len(payload)))
	if _, err := w.Write(length[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func runEncoder(cfg benchConfig, frames []goav1.I420Frame, refPath string, encoderName string, bitrate int) encodeResult {
	encoderName = canonicalEncoderName(encoderName)
	switch encoderName {
	case "goav1":
		return encodeGoAV1(cfg, frames, bitrate)
	case "aomenc":
		return encodeAOM(cfg, refPath, bitrate)
	case "svt-av1":
		return encodeSVT(cfg, refPath, bitrate)
	default:
		return encodeResult{encoder: encoderName, targetBPS: bitrate, status: "skipped", errText: "unknown encoder"}
	}
}

func encodeGoAV1(cfg benchConfig, frames []goav1.I420Frame, bitrate int) encodeResult {
	result := encodeResult{
		encoder:          "goav1",
		targetBPS:        bitrate,
		encodedContainer: "goav1-length-prefixed-low-overhead-stream",
		encodedPath:      filepath.Join(cfg.workdir, fmt.Sprintf("goav1_%d.obus", bitrate)),
		decodedYUV:       filepath.Join(cfg.workdir, fmt.Sprintf("goav1_%d.yuv", bitrate)),
		settings: map[string]string{
			"width":           strconv.Itoa(cfg.width),
			"height":          strconv.Itoa(cfg.height),
			"target_bitrate":  strconv.Itoa(bitrate),
			"framerate":       strconv.Itoa(cfg.fps),
			"temporal_layers": strconv.Itoa(cfg.layers),
			"tile_columns":    strconv.Itoa(cfg.tiles),
			"golden_interval": strconv.Itoa(cfg.goldenInterval),
			"key_interval":    strconv.Itoa(cfg.keyInterval),
			"gomaxprocs":      strconv.Itoa(runtime.GOMAXPROCS(0)),
			"num_cpu":         strconv.Itoa(runtime.NumCPU()),
			"simd_tier":       detectedSIMDTier(),
			"simd_features":   strings.Join(detectedSIMDFeatures(), ","),
			"timing_mode":     cfg.timingMode,
		},
	}
	var endToEndStart time.Time
	var endToEndCPUBefore processCPUTimes
	var endToEndCPUOK bool
	if cfg.timingMode == timingModeEndToEnd {
		endToEndStart = time.Now()
		endToEndCPUBefore, endToEndCPUOK = currentProcessCPUTimes()
	}
	encodedOut, err := os.Create(result.encodedPath)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	encodedClosed := false
	defer func() {
		if !encodedClosed {
			_ = encodedOut.Close()
		}
	}()
	out, err := os.Create(result.decodedYUV)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	closed := false
	defer func() {
		if !closed {
			_ = out.Close()
		}
	}()

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
	statsEnabled := goAV1DiagnosticsEnabled(cfg)
	if statsEnabled {
		enc.ResetDecisionStats()
		enc.SetDecisionStatsEnabled(true)
	}
	var encodeDuration time.Duration
	encodedHash := sha256.New()
	for i, frame := range frames {
		qBefore := enc.QIndex()
		statsBefore := goav1.EncoderDecisionStats{}
		if statsEnabled {
			statsBefore = enc.DecisionStats()
		}
		forceKey := cfg.keyInterval > 0 && i > 0 && i%cfg.keyInterval == 0
		cpuBefore, cpuOK := currentProcessCPUTimes()
		frameStart := time.Now()
		encoded, err := enc.Encode(frame, forceKey)
		frameDuration := time.Since(frameStart)
		encodeDuration += frameDuration
		if cfg.timingMode == timingModeCore && cpuOK {
			if cpuAfter, ok := currentProcessCPUTimes(); ok {
				result.cpuUser += nonNegativeDuration(cpuAfter.user - cpuBefore.user)
				result.cpuSystem += nonNegativeDuration(cpuAfter.system - cpuBefore.system)
				result.cpuAvailable = true
			}
		}
		if err != nil {
			result.status, result.errText = "error", err.Error()
			return result
		}
		frameBytes := int64(len(encoded.Data))
		result.bytes += frameBytes
		_, _ = encodedHash.Write(encoded.Data)
		if err := writeLengthPrefixedPayload(encodedOut, encoded.Data); err != nil {
			result.status, result.errText = "error", err.Error()
			return result
		}
		if err := writeFrame(out, enc.Reconstruction(), cfg.width, cfg.height); err != nil {
			result.status, result.errText = "error", err.Error()
			return result
		}
		if cfg.frameStatsCSVPath != "" {
			result.frameStats = append(result.frameStats, goAV1FrameStats{
				FrameIndex:      i,
				Keyframe:        encoded.Keyframe,
				TemporalID:      encoded.TemporalID,
				QIndexBefore:    qBefore,
				QIndexAfter:     enc.QIndex(),
				Bytes:           frameBytes,
				Duration:        frameDuration,
				CumulativeBytes: result.bytes,
				Stats:           diffDecisionStats(enc.DecisionStats(), statsBefore),
			})
		}
	}
	if err := out.Close(); err != nil {
		closed = true
		result.status, result.errText = "error", err.Error()
		return result
	}
	closed = true
	if err := encodedOut.Close(); err != nil {
		encodedClosed = true
		result.status, result.errText = "error", err.Error()
		return result
	}
	encodedClosed = true
	if cfg.timingMode == timingModeEndToEnd {
		result.duration = time.Since(endToEndStart)
		if endToEndCPUOK {
			if cpuAfter, ok := currentProcessCPUTimes(); ok {
				result.cpuUser = nonNegativeDuration(cpuAfter.user - endToEndCPUBefore.user)
				result.cpuSystem = nonNegativeDuration(cpuAfter.system - endToEndCPUBefore.system)
				result.cpuAvailable = true
			}
		}
	} else {
		result.duration = encodeDuration
	}
	result.settings["payload_sha256"] = hex.EncodeToString(encodedHash.Sum(nil))
	encodedBytes, encodedFileHash, err := fileBytesAndSHA256(result.encodedPath)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.encodedBytes = encodedBytes
	result.encodedSHA256 = encodedFileHash
	decodedBytes, decodedHash, err := fileBytesAndSHA256(result.decodedYUV)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.decodedBytes = decodedBytes
	result.decodedSHA256 = decodedHash
	if statsEnabled {
		result.stats = enc.DecisionStats()
	}
	result.status = "ok"
	return result
}

func encodeAOM(cfg benchConfig, refPath string, bitrate int) encodeResult {
	result := encodeResult{
		encoder:          "aomenc",
		targetBPS:        bitrate,
		encodedContainer: "ivf",
		decodedYUV:       filepath.Join(cfg.workdir, fmt.Sprintf("aomenc_%d.yuv", bitrate)),
		settings: map[string]string{
			"aom_threads": strconv.Itoa(cfg.aomThreads),
			"aom_row_mt":  strconv.Itoa(cfg.aomRowMT),
		},
	}
	if _, err := exec.LookPath("aomenc"); err != nil {
		result.status, result.errText = "skipped", "aomenc not found"
		return result
	}
	ivfPath := filepath.Join(cfg.workdir, fmt.Sprintf("aomenc_%d.ivf", bitrate))
	result.encodedPath = ivfPath
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
		fmt.Sprintf("--threads=%d", cfg.aomThreads),
		fmt.Sprintf("--row-mt=%d", cfg.aomRowMT),
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
	result.command = commandLine("aomenc", args)
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
	encodedBytes, encodedHash, err := fileBytesAndSHA256(ivfPath)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.encodedBytes = encodedBytes
	result.encodedSHA256 = encodedHash
	if err := decodeIVFWithFFmpeg(ivfPath, result.decodedYUV, cfg.frames); err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	decodedBytes, decodedHash, err := fileBytesAndSHA256(result.decodedYUV)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.decodedBytes = decodedBytes
	result.decodedSHA256 = decodedHash
	result.status = "ok"
	return result
}

func encodeSVT(cfg benchConfig, refPath string, bitrate int) encodeResult {
	result := encodeResult{
		encoder:          "svt-av1",
		targetBPS:        bitrate,
		encodedContainer: "ivf",
		decodedYUV:       filepath.Join(cfg.workdir, fmt.Sprintf("svtav1_%d.yuv", bitrate)),
		settings: map[string]string{
			"preset":       "13",
			"rate_control": "cbr",
			"target_kbps":  strconv.Itoa(kbps(bitrate)),
			"lookahead":    "0",
			"pred_struct":  "1",
			"rtc":          "1",
			"scd":          "0",
			"tf":           "0",
			"svt_lp":       strconv.Itoa(cfg.svtLP),
			"svt_lp_note":  "parallelism level 0..6, not a processor/thread count",
		},
	}
	if cfg.svtASM == "" {
		result.settings["svt_asm"] = "default"
		result.settings["svt_asm_note"] = "SVT default, currently max unless overridden by SVT"
	} else {
		result.settings["svt_asm"] = cfg.svtASM
	}
	if _, err := exec.LookPath("SvtAv1EncApp"); err != nil {
		result.status, result.errText = "skipped", "SvtAv1EncApp not found"
		return result
	}
	ivfPath := filepath.Join(cfg.workdir, fmt.Sprintf("svtav1_%d.ivf", bitrate))
	result.encodedPath = ivfPath
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
		"--lp", strconv.Itoa(cfg.svtLP),
	}
	if cfg.svtASM != "" {
		args = append(args, "--asm", cfg.svtASM)
	}
	if cfg.tiles > 0 {
		args = append(args, "--tile-columns", strconv.Itoa(cfg.tiles))
	}
	result.command = commandLine("SvtAv1EncApp", args)
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
	encodedBytes, encodedHash, err := fileBytesAndSHA256(ivfPath)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.encodedBytes = encodedBytes
	result.encodedSHA256 = encodedHash
	if err := decodeIVFWithFFmpeg(ivfPath, result.decodedYUV, cfg.frames); err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	decodedBytes, decodedHash, err := fileBytesAndSHA256(result.decodedYUV)
	if err != nil {
		result.status, result.errText = "error", err.Error()
		return result
	}
	result.decodedBytes = decodedBytes
	result.decodedSHA256 = decodedHash
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
	if cmd.ProcessState != nil {
		result.cpuUser = cmd.ProcessState.UserTime()
		result.cpuSystem = cmd.ProcessState.SystemTime()
		result.cpuAvailable = true
	}
	if err != nil {
		result.status = "error"
		result.errText = trimCommandOutput(err, out)
	}
	return elapsed
}

func nonNegativeDuration(d time.Duration) time.Duration {
	if d < 0 {
		return 0
	}
	return d
}

func commandLine(name string, args []string) []string {
	out := make([]string, 0, len(args)+1)
	out = append(out, name)
	out = append(out, args...)
	return out
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

func measureDecoded(cfg benchConfig, filters map[string]bool, required map[string]bool, refPath, decodedPath, encoderName string, bitrate int) (metrics, error) {
	m := metrics{psnr: "NA", ssim: "NA", xpsnr: "NA", vmaf: "NA"}
	if filters["psnr"] {
		if v, err := runScalarMetric(cfg, refPath, decodedPath, "psnr", `average:([0-9.]+)`); err == nil {
			m.psnr = formatMetric(v)
		} else if required["psnr"] {
			return m, fmt.Errorf("required metric psnr: %w", err)
		}
	} else if required["psnr"] {
		return m, errors.New("required metric psnr unavailable: ffmpeg filter psnr not found")
	}
	if filters["ssim"] {
		if v, err := runScalarMetric(cfg, refPath, decodedPath, "ssim", `All:([0-9.]+)`); err == nil {
			m.ssim = formatMetric(v)
		} else if required["ssim"] {
			return m, fmt.Errorf("required metric ssim: %w", err)
		}
	} else if required["ssim"] {
		return m, errors.New("required metric ssim unavailable: ffmpeg filter ssim not found")
	}
	if filters["xpsnr"] {
		if v, err := runScalarMetric(cfg, refPath, decodedPath, "xpsnr", `XPSNR\s+y:\s*([0-9]+(?:\.[0-9]+)?)`); err == nil {
			m.xpsnr = formatMetric(v)
		} else if required["xpsnr"] {
			return m, fmt.Errorf("required metric xpsnr: %w", err)
		}
	} else if required["xpsnr"] {
		return m, errors.New("required metric xpsnr unavailable: ffmpeg filter xpsnr not found")
	}
	if filters["libvmaf"] {
		if v, err := runVMAF(cfg, refPath, decodedPath, encoderName, bitrate); err == nil {
			m.vmaf = formatMetric(v)
		} else if required["vmaf"] {
			return m, fmt.Errorf("required metric vmaf: %w", err)
		}
	} else if required["vmaf"] {
		return m, errors.New("required metric vmaf unavailable: ffmpeg filter libvmaf not found")
	}
	return m, nil
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

func runFrameMetric(cfg benchConfig, refPath, decodedPath, encoderName string, bitrate int, filterName, valuePattern string) ([]frameMetricValue, error) {
	logPath := filepath.Join(cfg.workdir, fmt.Sprintf("%s_%d_%s_frames.log", safeClipDir(encoderName), bitrate, filterName))
	_ = os.Remove(logPath)
	filterSpec := fmt.Sprintf("%s=stats_file=%s", filterName, escapeFilterPath(logPath))
	args := rawMetricArgs(cfg, refPath, decodedPath, filterSpec)
	cmd := exec.Command("ffmpeg", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s frame metrics: %s", filterName, trimCommandOutput(err, out))
	}
	raw, err := os.ReadFile(logPath)
	if err != nil {
		return nil, fmt.Errorf("%s frame metrics: %w", filterName, err)
	}
	values, err := parseFrameMetricValues(raw, valuePattern)
	if err != nil {
		return nil, fmt.Errorf("%s frame metrics: %w", filterName, err)
	}
	return values, nil
}

func parseFrameMetricValues(raw []byte, valuePattern string) ([]frameMetricValue, error) {
	lineRe := regexp.MustCompile(`\bn:\s*([0-9]+)`)
	valueRe := regexp.MustCompile(valuePattern)
	var values []frameMetricValue
	for _, line := range strings.Split(string(raw), "\n") {
		nMatch := lineRe.FindStringSubmatch(line)
		if len(nMatch) < 2 {
			continue
		}
		valueMatch := valueRe.FindStringSubmatch(line)
		if len(valueMatch) < 2 {
			continue
		}
		frameIndex, err := strconv.Atoi(nMatch[1])
		if err != nil {
			return nil, err
		}
		if frameIndex > 0 {
			frameIndex--
		}
		value, err := normalizeFrameMetricValue(valueMatch[1])
		if err != nil {
			return nil, err
		}
		values = append(values, frameMetricValue{FrameIndex: frameIndex, Value: value})
	}
	if len(values) == 0 {
		return nil, errors.New("no per-frame values found")
	}
	return values, nil
}

func normalizeFrameMetricValue(raw string) (string, error) {
	if strings.EqualFold(raw, "inf") {
		return "inf", nil
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return "", err
	}
	return formatMetric(v), nil
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
