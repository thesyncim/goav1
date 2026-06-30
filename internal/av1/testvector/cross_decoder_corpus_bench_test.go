//go:build goav1_oracle

package testvector

// Multi-config cross-decoder throughput benchmark (the "corpus" benchmark).
//
// WHY THIS EXISTS (vs cross_decoder_bench_test.go)
//
// TestCrossDecoderThroughput times goav1 vs the C decoders on the bundled
// libaom conformance vectors. Those clips are tiny (a couple of frames each),
// so per-process startup dominates the external decoders' wall-clock and
// steady-state decode is barely measured. TestCrossDecoderCorpus closes that
// gap: it runs against a locally-generated corpus of ~30-60 frame clips that
// span resolutions (256x144 / 640x360 / 1280x720), rate points (cq 20/32/55),
// coding tools (all-intra vs inter GOP, single vs multi tile-column), bit
// depths (8-bit primary plus 10/12-bit profile coverage), and chroma sampling
// (4:2:0 primary plus a profile-2 4:2:2 probe). At that length steady-state
// decode dominates startup, so the goav1-vs-dav1d-vs-aomdec ratios are a
// cleaner requested-single-thread throughput comparison. SVT rows, when
// present, are requested with SvtAv1DecApp --lp 1; SVT documents --lp as a
// parallelism level rather than a processor/thread count.
//
// The corpus is NOT committed (it is large binary video). Regenerate it with
// scripts/gen_bench_corpus.sh, which scales/length-extends a small source y4m
// with ffmpeg and encodes the matrix with aomenc. The benchmark skips
// gracefully when the corpus directory is absent, so a plain `go test` (or this
// test without GOAV1_BENCH_CORPUS=1) is unaffected.
//
// Set GOAV1_BENCH_CORPUS_PUBLISH=1 for any row copied into a performance
// table. Publish mode requires every registered external reference decoder
// path and SHA-256 to be pinned explicitly, so a missing or PATH-drifted
// dav1d/aomdec/SVT binary cannot silently change a competitor column. It also
// requires GOAV1_BENCH_CORPUS_REPORT_JSON so the exact git/env/tool/corpus/
// timing provenance is persisted alongside the human log.
// GOAV1_BENCH_CORPUS_REQUIRE_DECODERS can require a named comma-separated
// subset (or "all") for local audits.
//
// METHODOLOGY (mirrors cross_decoder_bench_test.go)
//
//   - SAME WORK. goav1 runs the FULL decode INCLUDING the post-filter chain
//     (loop-filter / CDEF / loop-restoration / super-res / film-grain) for
//     every visible frame, via the exact FrameWork plumbing the oracle
//     conformance harness uses. Timed runs discard output just like the C
//     decoders; MD5 verification is performed once while loading the corpus so
//     hashing does not pollute the throughput number.
//   - CORRECTNESS GATE / CONFORMANCE PROBE. For every clip goav1 accumulates a
//     stream MD5 over the concatenated visible-frame planes (the libaom
//     test/md5_helper.h layout: visible Y rows, then U, then V, no stride
//     padding) and compares it byte-for-byte against a stream-MD5 or per-frame
//     MD5 sidecar produced by reference decoders. A clip whose goav1 digest
//     does not match, or that fails to decode, is a CONFORMANCE BUG: it is
//     excluded from the timing aggregate and reported prominently. This bench
//     therefore doubles as a conformance probe on real content, not just a perf
//     tool.
//   - UNIFORM, FAIR TIMING. goav1, aomdec, and dav1d are single-threaded
//     (goav1 worker pool = 1; aomdec --threads=1; dav1d --threads 1). SVT is
//     requested with SvtAv1DecApp --lp 1, which is a parallelism level rather
//     than a verified thread-count knob. All decoders are decode-only with
//     output discarded. Each (decoder, clip) is warmed up once then run
//     N times; the median measured sample is the reported table value, with min
//     and IQR retained in JSON for noise audits.
//   - IN-PROCESS vs SUBPROCESS. goav1 is timed in-process (no exec/startup);
//     the C decoders are subprocesses whose raw wall-clock includes process
//     startup. We measure each external decoder's startup baseline and report
//     BOTH raw and startup-adjusted numbers, exactly like the existing bench.
//     At ~48 frames/clip the startup share is small, so raw and adjusted are
//     close -- that convergence is the point of the longer clips.
//
// Reuses externalDecoder / runExternal / fpsOf / truncate / crossBenchRuns from
// cross_decoder_bench_test.go (same package + build tag).

import (
	"bytes"
	"crypto/md5"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/decoder"
	cpufeatures "github.com/thesyncim/goav1/internal/av1/dsp/cpu"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/benchenv"
)

// corpusClip is one benchmark/corpus clip resolved on disk together with its
// expected MD5 oracle and decode metadata.
type corpusClip struct {
	name          string // file stem, e.g. "p360_inter_q32"
	ivfPath       string
	ivfData       []byte
	wantMD5       MD5   // expected stream digest, when the sidecar is stream-MD5
	wantFrameMD5s []MD5 // expected visible-frame digests, when the sidecar is per-frame MD5
	oracleKind    corpusOracleKind
	oraclePath    string
	frames        int   // visible frames goav1 emitted (== external frame count)
	width         int   // coded width of the first frame
	height        int   // coded height of the first frame
	bitDepth      uint8 // 8, 10, or 12
	chroma        string
	tileCols      uint8 // tile columns of the first frame
	allIntra      bool  // every frame is a keyframe
}

type corpusClipCandidate struct {
	name    string
	ivfPath string
}

type corpusPublishManifest struct {
	path           string
	formatVersion  int
	sourceID       string
	sourceSHA256   string
	sourceURL      string
	sourceLicense  string
	sourceCategory string
	sources        []corpusPublishManifestSource
	toolVersions   map[string]string
	toolSHA256     map[string]string
	expectedClips  int
	rows           map[string]corpusPublishManifestRow
}

type corpusPublishManifestSource struct {
	id       string
	sha256   string
	url      string
	license  string
	category string
	clips    int
}

type corpusPublishManifestRow struct {
	name           string
	width          int
	height         int
	frames         int
	cq             int
	bitDepth       uint8
	chroma         string
	profile        int
	ivfBytes       int64
	ivfSHA256      string
	md5            MD5
	md5SHA256      string
	dav1dCheck     string
	aomencArgs     string
	sourceID       string
	sourceSHA256   string
	sourceURL      string
	sourceLicense  string
	sourceCategory string
}

type corpusOracleKind uint8

const (
	corpusOracleStreamMD5 corpusOracleKind = iota + 1
	corpusOracleFrameMD5
)

const (
	envBenchCorpus                  = "GOAV1_BENCH_CORPUS"
	envBenchCorpusPublish           = "GOAV1_BENCH_CORPUS_PUBLISH"
	envBenchCorpusAllowUnmanifested = "GOAV1_BENCH_CORPUS_ALLOW_UNMANIFESTED"
	envBenchCorpusRequireDecoders   = "GOAV1_BENCH_CORPUS_REQUIRE_DECODERS"
	envBenchCorpusReportJSON        = "GOAV1_BENCH_CORPUS_REPORT_JSON"
	envBenchCorpusGoBin             = "GOAV1_BENCH_CORPUS_GO_BIN"
	envBenchCorpusGoSHA256          = "GOAV1_BENCH_CORPUS_GO_SHA256"
	envBenchCorpusMinSources        = "GOAV1_BENCH_CORPUS_MIN_SOURCES"
	envBenchCorpusMinCategories     = "GOAV1_BENCH_CORPUS_MIN_CATEGORIES"
	envBenchCorpusEnvironmentNotes  = "GOAV1_BENCH_CORPUS_ENVIRONMENT_NOTES"
	envBenchCorpusCPUAffinity       = "GOAV1_BENCH_CORPUS_CPU_AFFINITY"
	envBenchCorpusPowerMode         = "GOAV1_BENCH_CORPUS_POWER_MODE"
	envBenchCorpusThermalState      = "GOAV1_BENCH_CORPUS_THERMAL_STATE"
	envBenchCorpusFrequencyPolicy   = "GOAV1_BENCH_CORPUS_FREQUENCY_POLICY"
	envBenchCorpusBackgroundLoad    = "GOAV1_BENCH_CORPUS_BACKGROUND_LOAD"
	envBenchCorpusDecoderPrefix     = "GOAV1_BENCH_CORPUS_"
)

const (
	corpusManifestFile      = "manifest.tsv"
	corpusManifestMagicV1   = "# goav1_bench_corpus_manifest_v1"
	corpusManifestMagicV2   = "# goav1_bench_corpus_manifest_v2"
	corpusManifestColumnsV1 = "name\twidth\theight\tframes\tcq\tdepth\tchroma\tprofile\tivf_bytes\tivf_sha256\tmd5\tmd5_sha256\tdav1d_check\taomenc_args"
	corpusManifestColumnsV2 = corpusManifestColumnsV1 + "\tsource_id\tsource_sha256\tsource_url\tsource_license\tsource_category"
)

type corpusOracleSidecar struct {
	path      string
	kind      corpusOracleKind
	streamMD5 MD5
	frameMD5s []MD5
}

// corpusDir resolves the benchmark corpus directory. It honors
// GOAV1_BENCH_CORPUS_DIR, then falls back to testdata/benchcorpus under the
// repo root (../../.. from this package). The boolean reports whether the
// directory exists and is non-empty.
func corpusDir(t *testing.T) (string, bool) {
	t.Helper()
	if dir := os.Getenv("GOAV1_BENCH_CORPUS_DIR"); dir != "" {
		return dir, dirHasIVF(dir)
	}
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..", "testdata", "benchcorpus"))
	return dir, dirHasIVF(dir)
}

func dirHasIVF(dir string) bool {
	matches, _ := filepath.Glob(filepath.Join(dir, "*.ivf"))
	return len(matches) > 0
}

// loadCorpusClips discovers every *.ivf with a supported MD5 sidecar in dir,
// parses the expected digest(s), and decodes each clip ONCE through goav1 to
// fill in frame count / dimensions / tile / bit-depth metadata AND to verify
// the decoded output matches the sidecar. Clips that fail to decode or mismatch
// are returned in failed (a conformance bug) and omitted from clips.
func loadCorpusClips(t *testing.T, dir string) (clips []corpusClip, failed []corpusFailure) {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "*.ivf"))
	if err != nil {
		t.Fatalf("glob corpus: %v", err)
	}
	sort.Strings(paths)
	candidates := make([]corpusClipCandidate, 0, len(paths))
	for _, ivfPath := range paths {
		candidates = append(candidates, corpusClipCandidate{
			name:    strings.TrimSuffix(filepath.Base(ivfPath), filepath.Ext(ivfPath)),
			ivfPath: ivfPath,
		})
	}
	return loadCorpusClipCandidates(t, candidates)
}

func loadCorpusClipCandidates(t *testing.T, candidates []corpusClipCandidate) (clips []corpusClip, failed []corpusFailure) {
	t.Helper()
	for _, candidate := range candidates {
		name := candidate.name
		ivfPath := candidate.ivfPath
		oracle, err := loadCorpusOracleSidecar(ivfPath)
		if err != nil {
			t.Logf("corpus: skipping %s (MD5 sidecar: %v)", name, err)
			continue
		}
		ivfData, err := os.ReadFile(ivfPath)
		if err != nil {
			t.Logf("corpus: skipping %s (read ivf: %v)", name, err)
			continue
		}

		clip := corpusClip{
			name:          name,
			ivfPath:       ivfPath,
			ivfData:       ivfData,
			wantMD5:       oracle.streamMD5,
			wantFrameMD5s: oracle.frameMD5s,
			oracleKind:    oracle.kind,
			oraclePath:    oracle.path,
		}
		res, derr := decodeCorpusClip(ivfData)
		if derr != nil {
			failed = append(failed, corpusFailure{name: name, reason: fmt.Sprintf("goav1 decode error: %v", derr)})
			continue
		}
		clip.frames = res.frames
		clip.width = res.width
		clip.height = res.height
		clip.bitDepth = res.bitDepth
		clip.chroma = res.chroma
		clip.tileCols = res.tileCols
		clip.allIntra = res.allIntra
		if err := verifyCorpusOracle(res, oracle); err != nil {
			failed = append(failed, corpusFailure{name: name, reason: err.Error()})
			continue
		}
		clips = append(clips, clip)
	}
	return clips, failed
}

func loadCorpusOracleSidecar(ivfPath string) (corpusOracleSidecar, error) {
	for _, path := range corpusOracleSidecarPaths(ivfPath) {
		raw, err := os.ReadFile(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return corpusOracleSidecar{}, fmt.Errorf("read %s: %w", path, err)
		}
		oracle, err := parseCorpusOracleSidecar(path, raw)
		if err != nil {
			return corpusOracleSidecar{}, fmt.Errorf("parse %s: %w", path, err)
		}
		return oracle, nil
	}
	return corpusOracleSidecar{}, fmt.Errorf("no supported MD5 sidecar found (tried %s)", strings.Join(corpusOracleSidecarPaths(ivfPath), ", "))
}

func corpusOracleSidecarPaths(ivfPath string) []string {
	stem := strings.TrimSuffix(ivfPath, filepath.Ext(ivfPath))
	return []string{
		stem + ".md5",
		ivfPath + ".md5",
		stem + ".framemd5",
		ivfPath + ".framemd5",
	}
}

func corpusOracleSidecarExists(ivfPath string) (bool, error) {
	for _, path := range corpusOracleSidecarPaths(ivfPath) {
		if _, err := os.Stat(path); err == nil {
			return true, nil
		} else if !os.IsNotExist(err) {
			return false, fmt.Errorf("stat MD5 sidecar %s: %w", path, err)
		}
	}
	return false, nil
}

func parseCorpusOracleSidecar(path string, src []byte) (corpusOracleSidecar, error) {
	tokens, err := parseCorpusMD5Tokens(src)
	if err != nil {
		return corpusOracleSidecar{}, err
	}
	oracle := corpusOracleSidecar{path: path}
	if strings.EqualFold(filepath.Ext(path), ".framemd5") || len(tokens) > 1 {
		oracle.kind = corpusOracleFrameMD5
		oracle.frameMD5s = tokens
		return oracle, nil
	}
	oracle.kind = corpusOracleStreamMD5
	oracle.streamMD5 = tokens[0]
	return oracle, nil
}

func parseCorpusMD5Tokens(src []byte) ([]MD5, error) {
	var out []MD5
	for len(src) > 0 {
		line := src
		if i := bytes.IndexByte(src, '\n'); i >= 0 {
			line = src[:i]
			src = src[i+1:]
		} else {
			src = nil
		}
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) == 0 || trimmed[0] == '#' {
			continue
		}
		for i := 0; i+32 <= len(line); i++ {
			if i > 0 {
				if _, ok := hexNibble(line[i-1]); ok {
					continue
				}
			}
			if i+32 < len(line) {
				if _, ok := hexNibble(line[i+32]); ok {
					continue
				}
			}
			md5, err := ParseMD5Hex(line[i : i+32])
			if err == nil {
				out = append(out, md5)
				i += 31
			}
		}
	}
	if len(out) == 0 {
		return nil, ErrInvalidMD5
	}
	return out, nil
}

func verifyCorpusOracle(res corpusDecodeResult, oracle corpusOracleSidecar) error {
	switch oracle.kind {
	case corpusOracleStreamMD5:
		if res.streamMD5 == oracle.streamMD5 {
			return nil
		}
		return fmt.Errorf("MD5 MISMATCH: goav1=%x want=%x (%dx%d %d-bit frames=%d sidecar=%s)",
			res.streamMD5, oracle.streamMD5, res.width, res.height, res.bitDepth, res.frames, oracle.path)
	case corpusOracleFrameMD5:
		if len(res.frameMD5s) != len(oracle.frameMD5s) {
			return fmt.Errorf("FRAME MD5 COUNT MISMATCH: goav1_frames=%d want_frames=%d (%dx%d %d-bit sidecar=%s)",
				len(res.frameMD5s), len(oracle.frameMD5s), res.width, res.height, res.bitDepth, oracle.path)
		}
		for i := range oracle.frameMD5s {
			if res.frameMD5s[i] != oracle.frameMD5s[i] {
				return fmt.Errorf("FRAME MD5 MISMATCH: frame=%d goav1=%x want=%x (%dx%d %d-bit sidecar=%s)",
					i, res.frameMD5s[i], oracle.frameMD5s[i], res.width, res.height, res.bitDepth, oracle.path)
			}
		}
		return nil
	default:
		return fmt.Errorf("unknown MD5 sidecar kind %d", oracle.kind)
	}
}

func corpusClipOracleLog(clip corpusClip) string {
	sidecar := filepath.Base(clip.oraclePath)
	if sidecar == "." || sidecar == string(filepath.Separator) {
		sidecar = ""
	}
	switch clip.oracleKind {
	case corpusOracleFrameMD5:
		if sidecar == "" {
			return fmt.Sprintf("frame_md5s=%d", len(clip.wantFrameMD5s))
		}
		return fmt.Sprintf("frame_md5s=%d sidecar=%s", len(clip.wantFrameMD5s), sidecar)
	case corpusOracleStreamMD5:
		if sidecar == "" {
			return fmt.Sprintf("md5=%x", clip.wantMD5)
		}
		return fmt.Sprintf("md5=%x sidecar=%s", clip.wantMD5, sidecar)
	default:
		return "md5=unknown"
	}
}

// corpusFailure records a clip that failed the goav1 conformance check.
type corpusFailure struct {
	name   string
	reason string
}

// corpusDecodeResult is the outcome of a single in-process goav1 decode.
type corpusDecodeResult struct {
	streamMD5 MD5
	frameMD5s []MD5
	frames    int
	width     int
	height    int
	bitDepth  uint8
	chroma    string
	tileCols  uint8
	allIntra  bool
}

// decodeCorpusClip runs the full goav1 FrameWork decode (residual + prediction
// + post-filter) single-threaded over every IVF frame, accumulating the libaom
// stream MD5 across all visible output frames while also keeping each visible
// frame MD5. It is a RemoteVector-free sibling of runLibaomFrameWorkDryRun that
// returns digests and metadata so it works on arbitrary IVFs.
//
// The corpus is plain single-layer (non-SVC) AV1, so a single frame pool and
// motion store suffice; we still honor show_frame / show_existing_frame so the
// emitted-frame set matches aomdec/dav1d exactly.
func decodeCorpusClip(ivfData []byte) (corpusDecodeResult, error) {
	return decodeCorpusClipWithMode(ivfData, true)
}

func decodeCorpusClipDiscard(ivfData []byte) (corpusDecodeResult, error) {
	return decodeCorpusClipWithMode(ivfData, false)
}

func decodeCorpusClipWithMode(ivfData []byte, verify bool) (corpusDecodeResult, error) {
	var res corpusDecodeResult
	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		return res, fmt.Errorf("ivf iterator: %w", err)
	}
	workerPool, err := threading.NewPool(1)
	if err != nil {
		return res, err
	}
	defer workerPool.Close()

	state := &corpusDecodeState{
		verify:     verify,
		workerPool: workerPool,
		layers:     newCorpusLayers(),
		allIntra:   true,
	}
	if verify {
		state.streamHash = md5.New()
	}
	defer state.layers.keepAlive()

	var stream decoder.Stream
	var events [16]decoder.Event
	for {
		ivfFrame, ok, err := it.Next()
		if err != nil {
			return res, fmt.Errorf("ivf frame %d: %w", state.ivfFrames, err)
		}
		if !ok {
			break
		}
		count, err := stream.PushLowOverhead(ivfFrame.Payload, events[:])
		if err != nil {
			return res, fmt.Errorf("ivf frame %d push: %w", ivfFrame.Index, err)
		}
		if err := state.runEvents(events[:count]); err != nil {
			return res, fmt.Errorf("ivf frame %d: %w", ivfFrame.Index, err)
		}
		state.ivfFrames++
	}

	var sum MD5
	if verify {
		state.streamHash.Sum(sum[:0])
	}
	res.streamMD5 = sum
	res.frameMD5s = append(res.frameMD5s, state.frameMD5s...)
	res.frames = state.visibleFrames
	res.width = state.width
	res.height = state.height
	res.bitDepth = state.bitDepth
	res.chroma = state.chroma
	res.tileCols = state.tileCols
	res.allIntra = state.allIntra && state.visibleFrames > 0
	if state.visibleFrames == 0 {
		return res, fmt.Errorf("no visible frames decoded")
	}
	return res, nil
}

// corpusDecodeState carries the running decode across IVF frames: the shared
// per-layer pools, the streaming MD5 accumulator, and observed metadata.
type corpusDecodeState struct {
	verify      bool
	streamHash  hash.Hash
	workerPool  *threading.Pool
	layers      *corpusSpatialLayers
	sideData    corpusFrameWorkSideDataRunner
	postScratch corpusPostFilterScratch

	ivfFrames     int
	visibleFrames int
	frameMD5s     []MD5
	width         int
	height        int
	bitDepth      uint8
	chroma        string
	tileCols      uint8
	allIntra      bool
}

// runEvents executes frame work + post-filter for every frame-bearing event in
// one IVF frame and hashes each visible output into the stream digest. It is a
// trimmed, digest-accumulating version of the loop in runLibaomFrameWorkDryRun.
func (s *corpusDecodeState) runEvents(events []decoder.Event) error {
	var (
		referenceSurfaces [parser.InterRefsPerFrame]int
		referenceFrames   [parser.InterRefsPerFrame]*frame.Frame
		spans             [parser.MaxTiles]parser.TileSpan
		jobs              [parser.MaxTiles]tile.Job
		batches           [parser.MaxTiles]threading.Batch
		releases          [parser.RefFrames]int
	)
	layers := s.layers
	for i := range events {
		event := events[i]
		if !eventRunsFrameWork(event) {
			continue
		}
		layer := layers.layer(event)

		var (
			postOutput       *frame.Frame
			postRan          bool
			currentMVSurface = -1
		)
		spatialID := event.SpatialID
		globalSurface := func(local int) int { return libaomGlobalSurfaceID(spatialID, local) }
		result, err := layer.state.RunEventWithContextAndExternalReferences(
			&layers.sharedRefs, &layer.pool, event.SequenceHeader, event, 32,
			referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:],
			s.workerPool, layers, globalSurface, layers, &layers.sharedFrameContexts,
			&s.sideData,
			libaomFrameWorkBatchRunner(func(ctx decoder.FrameWorkBatch) error {
				return corpusRunTileWork(ctx, layer, layers, referenceSurfaces[:], &currentMVSurface)
			}),
			libaomFrameWorkPostFilterRunner(func(ctx decoder.FrameWorkPostFilterContext) error {
				post := decoder.FrameWorkBoundSupportedPostFilterRunner{}
				size, err := libaomSupportedPostFilterScratchLen(ctx)
				if err != nil {
					return fmt.Errorf("supported postfilter scratch: %w", err)
				}
				post.Scratch = s.postScratch.bind(size)
				if err := post.Apply(ctx); err != nil {
					return fmt.Errorf("apply postfilters: %w", err)
				}
				if currentMVSurface >= 0 {
					if err := decoder.PublishTemporalMotionReference(post.Context.Event, currentMVSurface, &layer.mvFrames[currentMVSurface], layer.mvStore); err != nil {
						return err
					}
				}
				postOutput = post.Context.Output
				if post.DisplayOutput != nil {
					postOutput = post.DisplayOutput
				}
				postRan = true
				return nil
			}),
		)
		if err != nil {
			return fmt.Errorf("spatial=%d run event: %w", event.SpatialID, err)
		}

		// Observe metadata from the first frame seen.
		if s.width == 0 {
			s.width = int(event.FrameSize.CodedWidth)
			s.height = int(event.FrameSize.Height)
			cc := event.SequenceHeader.ColorConfig
			s.bitDepth = cc.BitDepth
			s.chroma = corpusChromaName(cc.MonoChrome, cc.SubsamplingX, cc.SubsamplingY)
			s.tileCols = event.TileInfo.Cols
		}

		if result.Run.CompletedFrame {
			if !postRan || postOutput == nil {
				return fmt.Errorf("spatial=%d completed without postfilter output", event.SpatialID)
			}
			if event.FrameHeader.FrameType != parser.FrameTypeKey {
				s.allIntra = false
			}
			// Only show_frame=true frames are emitted by aomdec/dav1d.
			if event.FrameHeader.ShowFrame {
				if err := s.emitVisibleFrame(*postOutput); err != nil {
					return err
				}
			}
		} else if event.Kind == decoder.EventExistingFrame && result.Step.Kind == decoder.FrameWorkStepShowExisting {
			// show_existing_frame re-displays a retained reference surface; the
			// emitted pixels are the cached surface, so re-hash its plane bytes.
			if !s.verify {
				s.visibleFrames++
				continue
			}
			surface := result.Step.ShowExisting.Surface
			f, err := layer.pool.Frame(surface)
			if err != nil {
				return fmt.Errorf("show_existing surface=%d: %w", surface, err)
			}
			if err := s.emitVisibleFrame(*f); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *corpusDecodeState) emitVisibleFrame(f frame.Frame) error {
	if !s.verify {
		s.visibleFrames++
		return nil
	}
	return s.hashVisibleFrame(f)
}

type corpusFrameWorkSideDataRunner struct {
	scratch decoder.FrameWorkSideDataScratch
}

func (r *corpusFrameWorkSideDataRunner) BindFrameWorkSideData(state *decoder.FrameWorkState, ctx decoder.FrameWorkBatch) error {
	size, err := decoder.FrameWorkSideDataScratchLen(ctx)
	if err != nil {
		return err
	}
	r.ensure(size)
	side, err := size.BindRunner(r.scratch)
	if err != nil {
		return err
	}
	return side.BindFrameWorkSideData(state, ctx)
}

func (r *corpusFrameWorkSideDataRunner) ensure(size decoder.FrameWorkSideDataScratchSize) {
	if len(r.scratch.CDEFIndex) < size.CDEF {
		r.scratch.CDEFIndex = make([]uint8, size.CDEF)
	}
	if len(r.scratch.CDEFRead) < size.CDEF {
		r.scratch.CDEFRead = make([]bool, size.CDEF)
	}
	if len(r.scratch.LoopFilterRecords) < size.LoopFilterRecords {
		r.scratch.LoopFilterRecords = make([]threading.FrameWorkLoopFilterBlockRecord, size.LoopFilterRecords)
	}
	if len(r.scratch.RestorationRecords) < size.RestorationRecords {
		r.scratch.RestorationRecords = make([]tile.RestorationUnitRecord, size.RestorationRecords)
	}
	if len(r.scratch.RestorationAbove) < size.RestorationBoundary {
		r.scratch.RestorationAbove = make([]uint16, size.RestorationBoundary)
	}
	if len(r.scratch.RestorationBelow) < size.RestorationBoundary {
		r.scratch.RestorationBelow = make([]uint16, size.RestorationBoundary)
	}
}

type corpusPostFilterScratch struct {
	scratch decoder.FrameWorkPostFilterScratch
}

func (s *corpusPostFilterScratch) bind(size decoder.FrameWorkPostFilterScratchSize) decoder.FrameWorkPostFilterScratch {
	s.ensure(size)
	return decoder.FrameWorkPostFilterScratch{
		LoopFilterEdges:    s.scratch.LoopFilterEdges[:libaomMaxInt(size.LoopFilter.Edges, 0)],
		LoopFilterSchedule: s.scratch.LoopFilterSchedule[:libaomMaxInt(size.LoopFilter.Schedule, 0)],

		CDEFSamples:       corpusUint16PlaneScratch(s.scratch.CDEFSamples, size.CDEF.Samples),
		CDEFDst:           corpusUint16PlaneScratch(s.scratch.CDEFDst, size.CDEF.Dst),
		CDEFDirectionGrid: s.scratch.CDEFDirectionGrid[:libaomMaxInt(size.CDEF.DirectionGrid, 0)],
		CDEFVarianceGrid:  s.scratch.CDEFVarianceGrid[:libaomMaxInt(size.CDEF.VarianceGrid, 0)],
		CDEFInput:         s.scratch.CDEFInput[:libaomMaxInt(size.CDEF.Input, 0)],
		CDEFUnitDst:       s.scratch.CDEFUnitDst[:libaomMaxInt(size.CDEF.UnitDst, 0)],

		SuperResOutputFrame: s.scratch.SuperResOutputFrame[:libaomMaxInt(size.SuperRes.OutputFrame, 0)],
		SuperResCoded:       corpusUint16PlaneScratch(s.scratch.SuperResCoded, size.SuperRes.CodedSamples),
		SuperResOutput:      corpusUint16PlaneScratch(s.scratch.SuperResOutput, size.SuperRes.OutputSamples),

		RestorationData:   s.scratch.RestorationData[:libaomMaxInt(size.Restoration.Samples.DataLen, 0)],
		RestorationDst:    s.scratch.RestorationDst[:libaomMaxInt(size.Restoration.Samples.DstLen, 0)],
		RestorationWiener: s.scratch.RestorationWiener[:libaomMaxInt(size.Restoration.Apply.Unit.Wiener, 0)],
		RestorationSGR:    s.scratch.RestorationSGR[:libaomMaxInt(size.Restoration.Apply.Unit.SGRProj, 0)],
		RestorationAbove:  s.scratch.RestorationAbove[:libaomMaxInt(size.Restoration.Apply.Boundary.Above, 0)],
		RestorationBelow:  s.scratch.RestorationBelow[:libaomMaxInt(size.Restoration.Apply.Boundary.Below, 0)],

		FilmGrainOutputFrame:   s.scratch.FilmGrainOutputFrame[:libaomMaxInt(size.FilmGrain.OutputFrame, 0)],
		FilmGrainLumaGrain:     s.scratch.FilmGrainLumaGrain[:libaomMaxInt(size.FilmGrain.LumaGrain, 0)],
		FilmGrainChromaGrain:   corpusInt16ChromaScratch(s.scratch.FilmGrainChromaGrain, size.FilmGrain.ChromaGrain),
		FilmGrainLumaSamples:   s.scratch.FilmGrainLumaSamples[:libaomMaxInt(size.FilmGrain.LumaSamples, 0)],
		FilmGrainChromaSamples: corpusUint16ChromaScratch(s.scratch.FilmGrainChromaSamples, size.FilmGrain.ChromaSamples),
	}
}

func (s *corpusPostFilterScratch) ensure(size decoder.FrameWorkPostFilterScratchSize) {
	if len(s.scratch.LoopFilterEdges) < size.LoopFilter.Edges {
		s.scratch.LoopFilterEdges = make([]decoder.FrameWorkLoopFilterPostFilterEdge, size.LoopFilter.Edges)
	}
	if len(s.scratch.LoopFilterSchedule) < size.LoopFilter.Schedule {
		s.scratch.LoopFilterSchedule = make([]uint32, size.LoopFilter.Schedule)
	}
	for plane := 0; plane < 3; plane++ {
		if len(s.scratch.CDEFSamples[plane]) < size.CDEF.Samples[plane] {
			s.scratch.CDEFSamples[plane] = make([]uint16, size.CDEF.Samples[plane])
		}
		if len(s.scratch.CDEFDst[plane]) < size.CDEF.Dst[plane] {
			s.scratch.CDEFDst[plane] = make([]uint16, size.CDEF.Dst[plane])
		}
		if len(s.scratch.SuperResCoded[plane]) < size.SuperRes.CodedSamples[plane] {
			s.scratch.SuperResCoded[plane] = make([]uint16, size.SuperRes.CodedSamples[plane])
		}
		if len(s.scratch.SuperResOutput[plane]) < size.SuperRes.OutputSamples[plane] {
			s.scratch.SuperResOutput[plane] = make([]uint16, size.SuperRes.OutputSamples[plane])
		}
	}
	if len(s.scratch.CDEFDirectionGrid) < size.CDEF.DirectionGrid {
		s.scratch.CDEFDirectionGrid = make([]cdef.DirectionGrid, size.CDEF.DirectionGrid)
	}
	if len(s.scratch.CDEFVarianceGrid) < size.CDEF.VarianceGrid {
		s.scratch.CDEFVarianceGrid = make([]cdef.VarianceGrid, size.CDEF.VarianceGrid)
	}
	if len(s.scratch.CDEFInput) < size.CDEF.Input {
		s.scratch.CDEFInput = make([]uint16, size.CDEF.Input)
	}
	if len(s.scratch.CDEFUnitDst) < size.CDEF.UnitDst {
		s.scratch.CDEFUnitDst = make([]uint16, size.CDEF.UnitDst)
	}
	if len(s.scratch.SuperResOutputFrame) < size.SuperRes.OutputFrame {
		s.scratch.SuperResOutputFrame = make([]byte, size.SuperRes.OutputFrame)
	}
	if len(s.scratch.RestorationData) < size.Restoration.Samples.DataLen {
		s.scratch.RestorationData = make([]uint16, size.Restoration.Samples.DataLen)
	}
	if len(s.scratch.RestorationDst) < size.Restoration.Samples.DstLen {
		s.scratch.RestorationDst = make([]uint16, size.Restoration.Samples.DstLen)
	}
	if len(s.scratch.RestorationWiener) < size.Restoration.Apply.Unit.Wiener {
		s.scratch.RestorationWiener = make([]uint16, size.Restoration.Apply.Unit.Wiener)
	}
	if len(s.scratch.RestorationSGR) < size.Restoration.Apply.Unit.SGRProj {
		s.scratch.RestorationSGR = make([]int32, size.Restoration.Apply.Unit.SGRProj)
	}
	if len(s.scratch.RestorationAbove) < size.Restoration.Apply.Boundary.Above {
		s.scratch.RestorationAbove = make([]uint16, size.Restoration.Apply.Boundary.Above)
	}
	if len(s.scratch.RestorationBelow) < size.Restoration.Apply.Boundary.Below {
		s.scratch.RestorationBelow = make([]uint16, size.Restoration.Apply.Boundary.Below)
	}
	if len(s.scratch.FilmGrainOutputFrame) < size.FilmGrain.OutputFrame {
		s.scratch.FilmGrainOutputFrame = make([]byte, size.FilmGrain.OutputFrame)
	}
	if len(s.scratch.FilmGrainLumaGrain) < size.FilmGrain.LumaGrain {
		s.scratch.FilmGrainLumaGrain = make([]int16, size.FilmGrain.LumaGrain)
	}
	for plane := 0; plane < 2; plane++ {
		if len(s.scratch.FilmGrainChromaGrain[plane]) < size.FilmGrain.ChromaGrain[plane] {
			s.scratch.FilmGrainChromaGrain[plane] = make([]int16, size.FilmGrain.ChromaGrain[plane])
		}
		if len(s.scratch.FilmGrainChromaSamples[plane]) < size.FilmGrain.ChromaSamples[plane] {
			s.scratch.FilmGrainChromaSamples[plane] = make([]uint16, size.FilmGrain.ChromaSamples[plane])
		}
	}
	if len(s.scratch.FilmGrainLumaSamples) < size.FilmGrain.LumaSamples {
		s.scratch.FilmGrainLumaSamples = make([]uint16, size.FilmGrain.LumaSamples)
	}
}

func corpusUint16PlaneScratch(scratch [3][]uint16, size [3]int) [3][]uint16 {
	return [3][]uint16{
		scratch[0][:libaomMaxInt(size[0], 0)],
		scratch[1][:libaomMaxInt(size[1], 0)],
		scratch[2][:libaomMaxInt(size[2], 0)],
	}
}

func corpusInt16ChromaScratch(scratch [2][]int16, size [2]int) [2][]int16 {
	return [2][]int16{
		scratch[0][:libaomMaxInt(size[0], 0)],
		scratch[1][:libaomMaxInt(size[1], 0)],
	}
}

func corpusUint16ChromaScratch(scratch [2][]uint16, size [2]int) [2][]uint16 {
	return [2][]uint16{
		scratch[0][:libaomMaxInt(size[0], 0)],
		scratch[1][:libaomMaxInt(size[1], 0)],
	}
}

func corpusChromaName(monochrome, subsamplingX, subsamplingY bool) string {
	switch {
	case monochrome:
		return "400"
	case subsamplingX && subsamplingY:
		return "420"
	case subsamplingX:
		return "422"
	case subsamplingY:
		return "440"
	default:
		return "444"
	}
}

// hashVisibleFrame folds one emitted frame's visible plane bytes into the
// running stream digest, matching FrameMD5's plane walk (Y, then U, then V;
// monochrome substitutes a neutral chroma plane).
func (s *corpusDecodeState) hashVisibleFrame(f frame.Frame) error {
	digest, err := FrameMD5(f)
	if err != nil {
		return err
	}
	if err := addFrameToStreamMD5(s.streamHash, f); err != nil {
		return err
	}
	s.frameMD5s = append(s.frameMD5s, digest)
	s.visibleFrames++
	return nil
}

// addFrameToStreamMD5 writes one frame's libaom-layout plane bytes into h. It
// reuses the exact per-plane writers FrameMD5 uses (addFramePlaneMD5 /
// addMonochromeNeutralChromaMD5) so the accumulated stream digest matches
// aomdec's --md5 / dav1d's md5 muxer (which hash every frame into one digest).
func addFrameToStreamMD5(h hash.Hash, f frame.Frame) error {
	if f.Layout.BytesPerSample != 1 && f.Layout.BytesPerSample != 2 {
		return frame.ErrInvalidFormat
	}
	if err := addFramePlaneMD5(h, f.Y, f.Layout.BytesPerSample); err != nil {
		return err
	}
	if f.Format.MonoChrome {
		return addMonochromeNeutralChromaMD5(h, f.Format, f.Layout.BytesPerSample)
	}
	if err := addFramePlaneMD5(h, f.U, f.Layout.BytesPerSample); err != nil {
		return err
	}
	return addFramePlaneMD5(h, f.V, f.Layout.BytesPerSample)
}

// corpusRunTileWork decodes and reconstructs every tile job for one frame-work
// batch (full residual + prediction). It mirrors the batch runner body in
// runLibaomFrameWorkDryRun, minus the per-path statistics counters the
// conformance test asserts on.
func corpusRunTileWork(ctx decoder.FrameWorkBatch, layer *corpusSpatialLayerState, layers *corpusSpatialLayers, referenceSurfaces []int, currentMVSurface *int) error {
	surface, err := ctx.Surface()
	if err != nil {
		return err
	}
	if surface >= len(layer.mvFrames) || layer.mvLength == 0 {
		return decoder.ErrInvalidSurfaceReference
	}
	if *currentMVSurface != surface || layer.mvFrames[surface].Entries == nil {
		first := surface * layer.mvLength
		currentMVFrame, err := ctx.BindReferenceMVFrame(layer.mvEntryBacking[first : first+layer.mvLength])
		if err != nil {
			return err
		}
		layer.mvFrames[surface] = currentMVFrame
		*currentMVSurface = surface
	}
	ctx.CurrentMVFrame = &layer.mvFrames[surface]
	if ctx.TileInfo.UseRefFrameMVS {
		temporalMVs, err := ctx.BindTemporalMotionField(layer.temporalEntryBacking)
		if err != nil {
			return err
		}
		ctx.TemporalMVs = &temporalMVs
		resolved, err := decoder.ResolveTemporalMotionReferencesWithProvider(layers, referenceSurfaces[:len(ctx.References)], ctx.ReferenceMVs[:])
		if err != nil {
			return err
		}
		if resolved != len(ctx.References) {
			return decoder.ErrInvalidSurfaceReference
		}
		if _, err := ctx.SetupTemporalMotionField(); err != nil {
			return err
		}
	}
	var restorationReq *threading.FrameWorkTileRestorationRequest
	if ctx.RestorationFrameBuffers != nil {
		restoration := threading.FrameWorkTileRestorationRequest{Buffers: *ctx.RestorationFrameBuffers}
		if err := restoration.InitReferences(); err != nil {
			return err
		}
		restorationReq = &restoration
	}
	for j := 0; j < len(ctx.Jobs); j++ {
		var decodeState tile.DecodeState
		if err := ctx.JobDecodeState(j, &decodeState); err != nil {
			return err
		}
		var storage threading.FrameWorkTileResidualCDFStorage
		if err := ctx.InitTileResidualCDFStorage(&storage); err != nil {
			return err
		}
		var scratch threading.FrameWorkTileResidualScratch
		rootCols, err := ctx.JobBlockLoopContextRootColumns(j)
		if err != nil {
			return err
		}
		scratch.LoopContext.Above = make([]tile.BlockLoopRootAboveContext, rootCols)
		loopReq, err := ctx.JobBlockLoopRequest(j, nil, nil, 0)
		if err != nil {
			return err
		}
		loopReq.DecodePredictionModes = true
		loopReq.DecodeInterModes = true
		loopReq.DecodeMotionVectors = true
		loopReq.DecodeInterIntra = true
		loopReq.DecodeMotionModes = true
		loopReq.DecodeCompoundBlend = true
		int32Scratch, residualScratch, err := libaomResidualScratch(ctx)
		if err != nil {
			return err
		}
		var interScratch threading.FrameWorkInterPredictionScratch
		predictionScratch := threading.FrameWorkPredictionScratch{Inter: &interScratch}
		if _, err := ctx.DecodeAndReconstructJobResiduals(j, &decodeState, storage.CDFs(), &scratch, threading.FrameWorkTileResidualRequest{
			Loop:          loopReq,
			TransformMode: ctx.TransformRef.TransformMode,
			CDEFIndexMap:  ctx.CDEFIndexMap,
			LoopFilterMap: ctx.LoopFilterMap,
			Restoration:   restorationReq,
			Transforms: func(visit tile.BlockLoopVisit) (threading.FrameWorkBlockTransforms, error) {
				if visit.Prediction.Valid && !visit.Prediction.Intra {
					return ctx.ReadInterBlockTransforms(&decodeState, visit)
				}
				return ctx.ReadIntraBlockTransforms(&decodeState, visit)
			},
			Int32Scratch:      int32Scratch,
			ResidualScratch:   residualScratch,
			PredictionScratch: &predictionScratch,
		}); err != nil {
			return fmt.Errorf("decode/reconstruct job %d: %w", j, err)
		}
		if err := ctx.RetainTileResidualCDFStorage(j, &decodeState, &storage); err != nil {
			return err
		}
		if _, err := ctx.JobOutputPlane(j, threading.FrameWorkPlaneY); err != nil {
			return err
		}
		if _, err := ctx.LoopRestorationPlan(false); err != nil {
			return err
		}
	}
	return nil
}

// ---- per-layer state (single-layer corpus, but the shared-refs plumbing is
// the same as the oracle harness so the FrameWork API is satisfied) ----------

type corpusSpatialLayerState struct {
	spatialID uint8
	format    frame.Format

	pool                 frame.Pool
	state                decoder.FrameWorkState
	backing              []byte
	frameSlots           []frame.Frame
	free                 []int
	used                 []bool
	mvEntryBacking       []tile.ReferenceMVEntry
	temporalEntryBacking []tile.TemporalMotionEntry
	mvFrames             []tile.ReferenceMVFrame
	mvStore              []tile.TemporalMotionReferenceFrame
	mvLength             int
}

type corpusSpatialLayers struct {
	byID                map[uint8]*corpusSpatialLayerState
	sharedRefs          decoder.SurfaceReferences
	sharedFrameContexts decoder.SharedFrameContextStore
}

func newCorpusLayers() *corpusSpatialLayers {
	return &corpusSpatialLayers{byID: make(map[uint8]*corpusSpatialLayerState)}
}

func (s *corpusSpatialLayers) FrameSurface(id int) (*frame.Frame, error) {
	spatialID, local, ok := libaomDecodeGlobalSurfaceID(id)
	if !ok {
		return nil, decoder.ErrInvalidSurfaceReference
	}
	layer, ok := s.byID[spatialID]
	if !ok || layer == nil {
		return nil, decoder.ErrInvalidSurfaceReference
	}
	return layer.pool.Frame(local)
}

func (s *corpusSpatialLayers) ReleaseFrameSurfaces(ids []int) error {
	for _, id := range ids {
		spatialID, local, ok := libaomDecodeGlobalSurfaceID(id)
		if !ok {
			return decoder.ErrInvalidSurfaceReference
		}
		layer, ok := s.byID[spatialID]
		if !ok || layer == nil {
			return decoder.ErrInvalidSurfaceReference
		}
		if err := layer.pool.Release(local); err != nil {
			return err
		}
	}
	return nil
}

func (s *corpusSpatialLayers) TemporalMotionReference(id int) (tile.TemporalMotionReferenceFrame, error) {
	spatialID, local, ok := libaomDecodeGlobalSurfaceID(id)
	if !ok {
		return tile.TemporalMotionReferenceFrame{}, decoder.ErrInvalidSurfaceReference
	}
	layer, ok := s.byID[spatialID]
	if !ok || layer == nil {
		return tile.TemporalMotionReferenceFrame{}, decoder.ErrInvalidSurfaceReference
	}
	if local < 0 || local >= len(layer.mvStore) {
		return tile.TemporalMotionReferenceFrame{}, decoder.ErrInvalidSurfaceReference
	}
	return layer.mvStore[local], nil
}

func (s *corpusSpatialLayers) layer(event decoder.Event) *corpusSpatialLayerState {
	id := event.SpatialID
	if layer, ok := s.byID[id]; ok {
		return layer
	}
	layer := &corpusSpatialLayerState{spatialID: id}
	layer.pool, layer.format, layer.backing, layer.frameSlots, layer.free, layer.used = bindCorpusFramePool(event, 16)
	layer.mvEntryBacking, layer.temporalEntryBacking, layer.mvFrames, layer.mvStore, layer.mvLength = bindCorpusMotionStore(event, len(layer.frameSlots))
	s.byID[id] = layer
	return layer
}

func (s *corpusSpatialLayers) keepAlive() {
	for _, layer := range s.byID {
		runtime.KeepAlive(layer.backing)
		runtime.KeepAlive(layer.frameSlots)
		runtime.KeepAlive(layer.free)
		runtime.KeepAlive(layer.used)
		runtime.KeepAlive(layer.mvEntryBacking)
		runtime.KeepAlive(layer.temporalEntryBacking)
		runtime.KeepAlive(layer.mvFrames)
		runtime.KeepAlive(layer.mvStore)
	}
}

func bindCorpusFramePool(event decoder.Event, count int) (frame.Pool, frame.Format, []byte, []frame.Frame, []int, []bool) {
	format := frameFormatFromEvent(event)
	layout, err := frame.RequiredSize(format)
	if err != nil {
		panic(fmt.Sprintf("corpus: RequiredSize: %v", err))
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		panic(fmt.Sprintf("corpus: BindPool: %v", err))
	}
	return pool, format, backing, frames, free, used
}

func bindCorpusMotionStore(event decoder.Event, count int) ([]tile.ReferenceMVEntry, []tile.TemporalMotionEntry, []tile.ReferenceMVFrame, []tile.TemporalMotionReferenceFrame, int) {
	miCols := libaomVectorMIExtent(event.FrameSize.CodedWidth)
	miRows := libaomVectorMIExtent(event.FrameSize.Height)
	length, err := tile.ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		panic(fmt.Sprintf("corpus: ReferenceMVFrameEntries: %v", err))
	}
	return make([]tile.ReferenceMVEntry, count*length),
		make([]tile.TemporalMotionEntry, length),
		make([]tile.ReferenceMVFrame, count),
		make([]tile.TemporalMotionReferenceFrame, count),
		length
}

func TestParseCorpusOracleSidecar(t *testing.T) {
	want, err := ParseMD5Hex([]byte("0123456789abcdeffedcba9876543210"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := ParseMD5Hex([]byte("fedcba98765432100123456789abcdef"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name      string
		path      string
		input     string
		kind      corpusOracleKind
		streamMD5 MD5
		frameMD5s []MD5
	}{
		{
			name:      "plain stream digest",
			path:      "clip.md5",
			input:     "0123456789abcdeffedcba9876543210\n",
			kind:      corpusOracleStreamMD5,
			streamMD5: want,
		},
		{
			name:      "aomdec stream digest",
			path:      "clip.md5",
			input:     "MD5 (clip.ivf) = 0123456789abcdeffedcba9876543210\n",
			kind:      corpusOracleStreamMD5,
			streamMD5: want,
		},
		{
			name:      "multi-line md5 sidecar",
			path:      "clip.ivf.md5",
			input:     "0123456789abcdeffedcba9876543210  frame0.yuv\nfedcba98765432100123456789abcdef  frame1.yuv\n",
			kind:      corpusOracleFrameMD5,
			frameMD5s: []MD5{want, second},
		},
		{
			name:      "fate framemd5",
			path:      "clip.framemd5",
			input:     "#format: frame checksums\n0, 0, 0, 1, 6144, 0123456789abcdeffedcba9876543210\n0, 1, 1, 1, 6144, fedcba98765432100123456789abcdef\n",
			kind:      corpusOracleFrameMD5,
			frameMD5s: []MD5{want, second},
		},
		{
			name:      "single framemd5 token remains frame oracle",
			path:      "clip.ivf.framemd5",
			input:     "0123456789abcdeffedcba9876543210\n",
			kind:      corpusOracleFrameMD5,
			frameMD5s: []MD5{want},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseCorpusOracleSidecar(tc.path, []byte(tc.input))
			if err != nil {
				t.Fatalf("parseCorpusOracleSidecar(%q): %v", tc.input, err)
			}
			if got.kind != tc.kind {
				t.Fatalf("kind=%d want %d", got.kind, tc.kind)
			}
			if got.streamMD5 != tc.streamMD5 {
				t.Fatalf("streamMD5=%x want %x", got.streamMD5, tc.streamMD5)
			}
			if len(got.frameMD5s) != len(tc.frameMD5s) {
				t.Fatalf("frameMD5s len=%d want %d", len(got.frameMD5s), len(tc.frameMD5s))
			}
			for i := range tc.frameMD5s {
				if got.frameMD5s[i] != tc.frameMD5s[i] {
					t.Fatalf("frameMD5s[%d]=%x want %x", i, got.frameMD5s[i], tc.frameMD5s[i])
				}
			}
		})
	}
	if _, err := parseCorpusMD5Tokens([]byte("0123456789abcdeffedcba987654321000")); err == nil {
		t.Fatal("parseCorpusMD5Tokens accepted a longer unbounded hex token")
	}
}

func TestCorpusRequiredExternalDecoderNames(t *testing.T) {
	decoders := []externalDecoder{
		{name: "aomdec"},
		{name: "dav1d"},
		{name: "SvtAv1DecApp"},
	}
	for _, tc := range []struct {
		name    string
		publish bool
		raw     string
		want    []string
		wantErr bool
	}{
		{
			name: "local run requires none",
		},
		{
			name:    "publish requires all",
			publish: true,
			want:    []string{"aomdec", "dav1d", "SvtAv1DecApp"},
		},
		{
			name:    "publish cannot be narrowed",
			publish: true,
			raw:     "dav1d",
			want:    []string{"aomdec", "dav1d", "SvtAv1DecApp"},
		},
		{
			name: "explicit subset is case insensitive",
			raw:  "DAV1D, svtav1decapp",
			want: []string{"dav1d", "SvtAv1DecApp"},
		},
		{
			name: "explicit all",
			raw:  "all",
			want: []string{"aomdec", "dav1d", "SvtAv1DecApp"},
		},
		{
			name:    "unknown decoder fails",
			raw:     "dav1d,missingdec",
			wantErr: true,
		},
		{
			name:    "empty explicit list fails",
			raw:     " , ; ",
			wantErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := corpusRequiredExternalDecoderNames(decoders, tc.publish, tc.raw)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("corpusRequiredExternalDecoderNames: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("required len=%d want %d (%v)", len(got), len(tc.want), got)
			}
			for _, name := range tc.want {
				if !got[name] {
					t.Fatalf("required[%q]=false want true (%v)", name, got)
				}
			}
		})
	}
}

func TestCorpusPublishReportToolsUsesVersionArgs(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "version-fixture")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\necho version args: \"$@\"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	tools := corpusPublishReportTools([]corpusTimingDecoder{{
		name: "fixturedec",
		external: externalDecoder{
			name:        "fixturedec",
			startupArgs: func(string) []string { return []string{"--startup"} },
			versionArgs: func(string) []string { return []string{"--real-version"} },
		},
		bin: bin,
	}})
	if len(tools) != 1 {
		t.Fatalf("tools len=%d want 1", len(tools))
	}
	if tools[0].Version != "version args: --real-version" {
		t.Fatalf("version=%q want version args from versionArgs", tools[0].Version)
	}
	if tools[0].VersionError != "" {
		t.Fatalf("version error=%q want empty", tools[0].VersionError)
	}
}

func TestResolveCorpusExternalDecodersRequiresMissing(t *testing.T) {
	decoders := []externalDecoder{
		{name: "aomdec"},
		{name: "dav1d"},
	}
	required := map[string]bool{"aomdec": true}
	resolved, missing := resolveCorpusExternalDecoders(decoders, required, func(dec externalDecoder) (string, bool) {
		if dec.name == "dav1d" {
			return "/reference/dav1d", true
		}
		return "", false
	})
	if len(missing) != 1 || missing[0] != "aomdec" {
		t.Fatalf("missing=%v want [aomdec]", missing)
	}
	if len(resolved) != 1 || resolved[0].decoder.name != "dav1d" || resolved[0].bin != "/reference/dav1d" {
		t.Fatalf("resolved=%v want dav1d", resolved)
	}
}

func TestResolveCorpusPublishExternalDecodersRequiresPinnedTools(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "dav1d")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	sha, _, err := corpusFileSHA256(bin)
	if err != nil {
		t.Fatal(err)
	}
	decoders := []externalDecoder{{name: "dav1d"}}
	required := map[string]bool{"dav1d": true}
	binEnv, shaEnv := corpusPublishExternalDecoderPinEnvNames("dav1d")

	t.Run("valid", func(t *testing.T) {
		t.Setenv(binEnv, bin)
		t.Setenv(shaEnv, sha)
		resolved, err := resolveCorpusPublishExternalDecoders(decoders, required)
		if err != nil {
			t.Fatalf("resolveCorpusPublishExternalDecoders: %v", err)
		}
		if len(resolved) != 1 || resolved[0].bin != bin || resolved[0].decoder.name != "dav1d" {
			t.Fatalf("resolved=%+v", resolved)
		}
	})

	t.Run("missing path", func(t *testing.T) {
		t.Setenv(binEnv, "")
		t.Setenv(shaEnv, sha)
		if _, err := resolveCorpusPublishExternalDecoders(decoders, required); err == nil ||
			!strings.Contains(err.Error(), binEnv) {
			t.Fatalf("missing path error=%v", err)
		}
	})

	t.Run("relative path", func(t *testing.T) {
		t.Setenv(binEnv, "dav1d")
		t.Setenv(shaEnv, sha)
		if _, err := resolveCorpusPublishExternalDecoders(decoders, required); err == nil ||
			!strings.Contains(err.Error(), "absolute path") {
			t.Fatalf("relative path error=%v", err)
		}
	})

	t.Run("not executable", func(t *testing.T) {
		plain := filepath.Join(t.TempDir(), "dav1d")
		if err := os.WriteFile(plain, []byte("fixture"), 0o644); err != nil {
			t.Fatal(err)
		}
		plainSHA, _, err := corpusFileSHA256(plain)
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv(binEnv, plain)
		t.Setenv(shaEnv, plainSHA)
		if _, err := resolveCorpusPublishExternalDecoders(decoders, required); err == nil ||
			!strings.Contains(err.Error(), "not an executable") {
			t.Fatalf("not executable error=%v", err)
		}
	})

	t.Run("bad hash", func(t *testing.T) {
		t.Setenv(binEnv, bin)
		t.Setenv(shaEnv, strings.Repeat("0", 64))
		if _, err := resolveCorpusPublishExternalDecoders(decoders, required); err == nil ||
			!strings.Contains(err.Error(), "sha256") {
			t.Fatalf("bad hash error=%v", err)
		}
	})
}

func TestValidateCorpusPublishDecoderManifestHashes(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "dav1d")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	sha, _, err := corpusFileSHA256(bin)
	if err != nil {
		t.Fatal(err)
	}
	resolved := []resolvedCorpusExternalDecoder{{
		decoder: externalDecoder{name: "dav1d"},
		bin:     bin,
	}}
	manifest := corpusPublishManifest{
		path:       filepath.Join(t.TempDir(), corpusManifestFile),
		toolSHA256: map[string]string{"dav1d": sha},
	}
	if err := validateCorpusPublishDecoderManifestHashes(manifest, resolved); err != nil {
		t.Fatalf("matching decoder hash failed: %v", err)
	}

	manifest.toolSHA256["dav1d"] = strings.Repeat("0", 64)
	if err := validateCorpusPublishDecoderManifestHashes(manifest, resolved); err == nil ||
		!strings.Contains(err.Error(), "does not match manifest dav1d_sha256") {
		t.Fatalf("mismatched decoder hash error=%v", err)
	}

	delete(manifest.toolSHA256, "dav1d")
	if err := validateCorpusPublishDecoderManifestHashes(manifest, resolved); err == nil ||
		!strings.Contains(err.Error(), "manifest dav1d_sha256") {
		t.Fatalf("missing decoder hash error=%v", err)
	}

	if err := validateCorpusPublishDecoderManifestHashes(manifest, []resolvedCorpusExternalDecoder{{
		decoder: externalDecoder{name: "SvtAv1DecApp"},
		bin:     bin,
	}}); err != nil {
		t.Fatalf("SVT should not require a generator manifest hash today: %v", err)
	}
}

func TestCorpusInterleavedTimingJobsRotateDecoders(t *testing.T) {
	jobs := corpusInterleavedTimingJobs(3, 4)
	var got []string
	for _, job := range jobs {
		got = append(got, fmt.Sprintf("c%d:d%d", job.clipIndex, job.decoderIndex))
	}
	want := strings.Join([]string{
		"c0:d0", "c0:d1", "c0:d2", "c0:d3",
		"c1:d1", "c1:d2", "c1:d3", "c1:d0",
		"c2:d2", "c2:d3", "c2:d0", "c2:d1",
	}, ",")
	if strings.Join(got, ",") != want {
		t.Fatalf("jobs=%s want %s", strings.Join(got, ","), want)
	}
	if jobs := corpusInterleavedTimingJobs(0, 4); len(jobs) != 0 {
		t.Fatalf("empty clip jobs=%v", jobs)
	}
	if jobs := corpusInterleavedTimingJobs(3, 0); len(jobs) != 0 {
		t.Fatalf("empty decoder jobs=%v", jobs)
	}
}

func TestCorpusClipCoverageSummaryDerivesLoadedAxes(t *testing.T) {
	clips := []corpusClip{
		{width: 256, height: 144, bitDepth: 8, chroma: "420", tileCols: 0, allIntra: true},
		{width: 512, height: 288, bitDepth: 10, chroma: "444", tileCols: 1, allIntra: false},
	}
	got := corpusClipCoverageSummary(clips)
	for _, want := range []string{"2 generated", "256x144", "512x288", "8/10", "420/444", "0/1", "inter/intra"} {
		if !strings.Contains(got, want) {
			t.Fatalf("summary %q missing %q", got, want)
		}
	}
}

func TestLoadCorpusPublishManifestValidatesFiles(t *testing.T) {
	dir := t.TempDir()
	md5Hex := "0123456789abcdeffedcba9876543210"
	md5, err := ParseMD5Hex([]byte(md5Hex))
	if err != nil {
		t.Fatal(err)
	}
	writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 1)

	manifest, err := loadCorpusPublishManifest(dir)
	if err != nil {
		t.Fatalf("loadCorpusPublishManifest: %v", err)
	}
	row := manifest.rows["clip"]
	if manifest.expectedClips != 1 || row.width != 2 || row.height != 4 || row.frames != 3 ||
		row.cq != 32 || row.bitDepth != 8 || row.chroma != "420" || row.profile != 0 ||
		row.md5 != md5 || row.dav1dCheck != "dav1d=OK" || row.aomencArgs != "args" {
		t.Fatalf("manifest=%+v row=%+v", manifest, row)
	}
	if manifest.toolSHA256["aomenc"] != testSHA256([]byte("aomenc fixture")) ||
		manifest.toolSHA256["aomdec"] != testSHA256([]byte("aomdec fixture")) ||
		manifest.toolSHA256["ffmpeg"] != testSHA256([]byte("ffmpeg fixture")) {
		t.Fatalf("tool hashes=%+v", manifest.toolSHA256)
	}
	clip := corpusClip{name: "clip", width: 2, height: 4, frames: 3, bitDepth: 8, chroma: "420", oracleKind: corpusOracleStreamMD5, wantMD5: md5}
	if err := validateCorpusPublishLoadedClips(manifest, []corpusClip{clip}); err != nil {
		t.Fatalf("validateCorpusPublishLoadedClips: %v", err)
	}
	clip.width = 8
	if err := validateCorpusPublishLoadedClips(manifest, []corpusClip{clip}); err == nil {
		t.Fatal("validateCorpusPublishLoadedClips accepted mismatched metadata")
	}
}

func TestLoadCorpusPublishManifestV2RecordsSourceDiversity(t *testing.T) {
	t.Setenv(envBenchCorpusMinSources, "2")
	t.Setenv(envBenchCorpusMinCategories, "2")
	dir := t.TempDir()
	md5Hex := "0123456789abcdeffedcba9876543210"
	writeCorpusManifestV2Fixture(t, dir, []corpusManifestV2FixtureClip{
		{name: "camera_pan_p144_inter_q32", sourceID: "camera-pan", sourceCategory: "camera-motion", sourceData: []byte("source-a")},
		{name: "screen_share_p144_inter_q32", sourceID: "screen-share", sourceCategory: "screen-content", sourceData: []byte("source-b")},
	}, md5Hex)

	manifest, err := loadCorpusPublishManifest(dir)
	if err != nil {
		t.Fatalf("loadCorpusPublishManifest: %v", err)
	}
	if manifest.formatVersion != 2 || len(manifest.sources) != 2 || manifest.sourceID != "multiple" ||
		corpusManifestSourceCategoryCount(manifest.sources) != 2 {
		t.Fatalf("manifest source summary=%+v sources=%+v", manifest, manifest.sources)
	}
	if err := validateCorpusPublishSourceCoverage(manifest); err != nil {
		t.Fatalf("validateCorpusPublishSourceCoverage: %v", err)
	}
	row := manifest.rows["screen_share_p144_inter_q32"]
	if row.sourceID != "screen-share" || row.sourceCategory != "screen-content" || row.sourceSHA256 != testSHA256([]byte("source-b")) {
		t.Fatalf("row source provenance=%+v", row)
	}
	clips := []corpusClip{
		{name: "camera_pan_p144_inter_q32", ivfPath: filepath.Join(dir, "camera_pan_p144_inter_q32.ivf"), frames: 3, width: 2, height: 4, bitDepth: 8, chroma: "420"},
		{name: "screen_share_p144_inter_q32", ivfPath: filepath.Join(dir, "screen_share_p144_inter_q32.ivf"), frames: 3, width: 2, height: 4, bitDepth: 8, chroma: "420"},
	}
	reportClips := corpusPublishReportClips(clips, manifest)
	if len(reportClips) != 2 || reportClips[1].SourceID != "screen-share" ||
		reportClips[1].SourceSHA256 != testSHA256([]byte("source-b")) ||
		reportClips[1].SourceCategory != "screen-content" {
		t.Fatalf("report clips=%+v", reportClips)
	}
}

func TestLoadCorpusBenchmarkManifestAllowsOnlyExploratoryMissingManifest(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(envBenchCorpusAllowUnmanifested, "1")
	if manifest, ok := loadCorpusBenchmarkManifest(t, dir, false); ok || manifest.path != "" {
		t.Fatalf("missing exploratory manifest ok=%v manifest=%+v", ok, manifest)
	}
	t.Setenv(envBenchCorpusAllowUnmanifested, "")
	if allowMissingCorpusBenchmarkManifest() {
		t.Fatal("missing manifest allow gate accepted empty value")
	}

	md5Hex := "0123456789abcdeffedcba9876543210"
	writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 1)
	manifest, ok := loadCorpusBenchmarkManifest(t, dir, false)
	if !ok || manifest.expectedClips != 1 || manifest.path == "" {
		t.Fatalf("valid exploratory manifest ok=%v manifest=%+v", ok, manifest)
	}
}

func TestWriteCorpusPublishReport(t *testing.T) {
	t.Setenv(envBenchCorpusEnvironmentNotes, "fixed power mode")
	t.Setenv(envBenchCorpusCPUAffinity, "none")
	t.Setenv(envBenchCorpusPowerMode, "high power")
	t.Setenv(envBenchCorpusThermalState, "cool start")
	t.Setenv(envBenchCorpusFrequencyPolicy, "automatic")
	t.Setenv(envBenchCorpusBackgroundLoad, "idle machine")
	dir := t.TempDir()
	md5Hex := "0123456789abcdeffedcba9876543210"
	md5, err := ParseMD5Hex([]byte(md5Hex))
	if err != nil {
		t.Fatal(err)
	}
	ivfPath := filepath.Join(dir, "clip.ivf")
	writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 1)
	manifest, err := loadCorpusPublishManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	manifestSHA, _, err := corpusFileSHA256(manifest.path)
	if err != nil {
		t.Fatal(err)
	}
	toolBin := os.Args[0]
	clip := corpusClip{
		name:       "clip",
		ivfPath:    ivfPath,
		wantMD5:    md5,
		oracleKind: corpusOracleStreamMD5,
		frames:     3,
		width:      2,
		height:     4,
		bitDepth:   8,
		chroma:     "420",
		tileCols:   1,
	}
	goav1Samples := summarizeCorpusDurations([]time.Duration{40 * time.Millisecond, 30 * time.Millisecond, 32 * time.Millisecond})
	dav1dStartupSamples := summarizeCorpusDurations([]time.Duration{2 * time.Millisecond, time.Millisecond, 3 * time.Millisecond})
	dav1dSamples := summarizeCorpusDurations([]time.Duration{16 * time.Millisecond, 15 * time.Millisecond, 18 * time.Millisecond})
	results := []decoderResult{
		{
			name:        "goav1",
			inProcess:   true,
			totalRaw:    corpusSelectedDuration(goav1Samples),
			totalFrames: 3,
			perVector:   map[string]time.Duration{ivfPath: corpusSelectedDuration(goav1Samples)},
			perVectorSamples: map[string]corpusDurationSamples{
				ivfPath: goav1Samples,
			},
		},
		{
			name:           "dav1d",
			startup:        corpusSelectedDuration(dav1dStartupSamples),
			startupSamples: dav1dStartupSamples,
			totalRaw:       corpusSelectedDuration(dav1dSamples),
			totalFrames:    3,
			perVector:      map[string]time.Duration{ivfPath: corpusSelectedDuration(dav1dSamples)},
			perVectorSamples: map[string]corpusDurationSamples{
				ivfPath: dav1dSamples,
			},
		},
	}
	timers := []corpusTimingDecoder{
		{name: "goav1", inProcess: true, resultSlot: 0},
		{
			name: "dav1d",
			external: externalDecoder{
				name:        "dav1d",
				startupArgs: func(string) []string { return []string{"-test.run=^$"} },
			},
			bin:        toolBin,
			resultSlot: 1,
		},
	}
	reportPath := filepath.Join(dir, "report", "corpus.json")
	if err := writeCorpusPublishReport(reportPath, dir, manifest, []corpusClip{clip}, results, timers); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	var report corpusPublishReport
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatal(err)
	}
	if report.Corpus.ManifestSHA256 != manifestSHA || report.Corpus.ExpectedClips != 1 || report.Corpus.LoadedClips != 1 || report.Corpus.TotalFrames != 3 {
		t.Fatalf("corpus report=%+v want manifest hash/clip counts", report.Corpus)
	}
	if report.Corpus.SourceID != "fixture-source" ||
		report.Corpus.SourceSHA256 != testSHA256([]byte("source fixture")) ||
		report.Corpus.SourceURL != "https://example.invalid/source" ||
		report.Corpus.SourceLicense != "fixture-license" ||
		report.Corpus.SourceCategory != "fixture-category" ||
		report.Corpus.SourceCount != 1 ||
		report.Corpus.SourceCategories != 1 ||
		len(report.Corpus.Sources) != 1 ||
		report.Corpus.Sources[0].ID != "fixture-source" ||
		report.Corpus.ToolVersions["aomenc"] != "aomenc fixture" ||
		report.Corpus.ToolVersions["aomdec"] != "aomdec fixture" ||
		report.Corpus.ToolVersions["ffmpeg"] != "ffmpeg fixture" {
		t.Fatalf("corpus provenance=%+v", report.Corpus)
	}
	if report.Corpus.ToolSHA256["aomenc"] != testSHA256([]byte("aomenc fixture")) ||
		report.Corpus.ToolSHA256["aomdec"] != testSHA256([]byte("aomdec fixture")) ||
		report.Corpus.ToolSHA256["ffmpeg"] != testSHA256([]byte("ffmpeg fixture")) {
		t.Fatalf("corpus tool hashes=%+v", report.Corpus.ToolSHA256)
	}
	if report.Environment.Notes != "fixed power mode" ||
		report.Environment.CPUAffinity != "none" ||
		report.Environment.PowerMode != "high power" ||
		report.Environment.ThermalState != "cool start" ||
		report.Environment.FrequencyPolicy != "automatic" ||
		report.Environment.BackgroundLoad != "idle machine" ||
		report.Environment.GOMAXPROCS <= 0 ||
		report.Environment.NumCPU <= 0 || report.Environment.PATH == "" {
		t.Fatalf("environment=%+v", report.Environment)
	}
	if !strings.Contains(report.Timing.Statistic, "median measured") ||
		report.Timing.ExternalCommandTimeout != externalDecoderCommandTimeout.String() {
		t.Fatalf("timing=%+v", report.Timing)
	}
	if len(report.Clips) != 1 || report.Clips[0].Name != "clip" || report.Clips[0].CQ != 32 ||
		report.Clips[0].BitDepth != 8 || report.Clips[0].Profile != 0 ||
		report.Clips[0].SourceID != "fixture-source" ||
		report.Clips[0].SourceSHA256 != testSHA256([]byte("source fixture")) ||
		report.Clips[0].DAV1DCheck != "dav1d=OK" || report.Clips[0].AOMEncArgs != "args" {
		t.Fatalf("clips=%+v", report.Clips)
	}
	if len(report.Tools) != 2 || !report.Tools[0].InProcess || report.Tools[1].SHA256 == "" {
		t.Fatalf("tools=%+v", report.Tools)
	}
	if len(report.Decoders) != 2 || report.Decoders[1].AdjustedMS != 14 || report.Decoders[1].VsGoAV1Raw != 2 ||
		report.Decoders[1].StartupMedianMS != 2 || report.Decoders[1].StartupIQRMS != 2 ||
		len(report.Decoders[1].StartupSamplesMS) != 3 {
		t.Fatalf("decoders=%+v", report.Decoders)
	}
	if len(report.Decoders[1].PerClip) != 1 || report.Decoders[1].PerClip[0].AdjustedMS != 14 ||
		report.Decoders[1].PerClip[0].MedianMS != 16 || report.Decoders[1].PerClip[0].IQRMS != 3 ||
		len(report.Decoders[1].PerClip[0].SamplesMS) != 3 {
		t.Fatalf("per-clip=%+v", report.Decoders[1].PerClip)
	}
}

func TestValidateCorpusPublishGitClean(t *testing.T) {
	if err := validateCorpusPublishGitClean(corpusPublishGit{Commit: "abc"}); err != nil {
		t.Fatalf("clean git failed: %v", err)
	}
	if err := validateCorpusPublishGitClean(corpusPublishGit{Commit: "abc", Dirty: true}); err == nil ||
		!strings.Contains(err.Error(), "clean git worktree") {
		t.Fatalf("dirty git error=%v", err)
	}
	if err := validateCorpusPublishGitClean(corpusPublishGit{Error: "no git"}); err == nil ||
		!strings.Contains(err.Error(), "metadata unavailable") {
		t.Fatalf("missing git error=%v", err)
	}
}

func TestValidateCorpusPublishEnvironmentNotes(t *testing.T) {
	if err := validateCorpusPublishEnvironmentNotes("fixed power mode"); err != nil {
		t.Fatalf("valid notes failed: %v", err)
	}
	if err := validateCorpusPublishEnvironmentNotes(" "); err == nil ||
		!strings.Contains(err.Error(), envBenchCorpusEnvironmentNotes) {
		t.Fatalf("empty notes error=%v", err)
	}
}

func TestValidateCorpusPublishEnvironmentRejectsAmbientGoDrift(t *testing.T) {
	goPath, err := filepath.Abs(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	goSHA, _, err := corpusFileSHA256(goPath)
	if err != nil {
		t.Fatal(err)
	}
	valid := corpusPublishEnvironment{
		GOMAXPROCS:           1,
		GOMAXPROCSEnv:        "1",
		GoToolPath:           goPath,
		GoToolSHA256:         goSHA,
		GoToolExpectedSHA256: goSHA,
		GoToolSHA256Verified: true,
		GOGC:                 "off",
		CPUAffinity:          "none",
		PowerMode:            "high power",
		ThermalState:         "cool start",
		FrequencyPolicy:      "automatic",
		BackgroundLoad:       "idle machine",
	}
	if err := validateCorpusPublishEnvironment(valid); err != nil {
		t.Fatalf("valid environment failed: %v", err)
	}

	for _, tc := range []struct {
		name string
		mut  func(*corpusPublishEnvironment)
		want string
	}{
		{
			name: "missing gomaxprocs",
			mut:  func(env *corpusPublishEnvironment) { env.GOMAXPROCSEnv = "" },
			want: "GOMAXPROCS",
		},
		{
			name: "bad gomaxprocs",
			mut:  func(env *corpusPublishEnvironment) { env.GOMAXPROCSEnv = "1,2" },
			want: "GOMAXPROCS",
		},
		{
			name: "runtime mismatch",
			mut:  func(env *corpusPublishEnvironment) { env.GOMAXPROCS = 2 },
			want: "does not match",
		},
		{
			name: "missing gogc",
			mut:  func(env *corpusPublishEnvironment) { env.GOGC = "" },
			want: "GOGC",
		},
		{
			name: "goflags",
			mut:  func(env *corpusPublishEnvironment) { env.GOFLAGS = "-race" },
			want: "GOFLAGS unset",
		},
		{
			name: "gomemlimit",
			mut:  func(env *corpusPublishEnvironment) { env.GOMEMLIMIT = "512MiB" },
			want: "GOMEMLIMIT unset",
		},
		{
			name: "godebug",
			mut:  func(env *corpusPublishEnvironment) { env.GODEBUG = "gcstoptheworld=1" },
			want: "GODEBUG unset",
		},
		{
			name: "restricted cpu affinity",
			mut: func(env *corpusPublishEnvironment) {
				env.ObservedCPUState = benchenv.CPUState{
					AffinitySupported:   true,
					AffinityAllowedList: "0-1",
					CPUOnlineList:       "0-3",
				}
			},
			want: "restricted CPU affinity",
		},
		{
			name: "goamd64",
			mut: func(env *corpusPublishEnvironment) {
				env.BlockedGoEnv = map[string]string{"GOAMD64": "v4"}
			},
			want: "GOAMD64 unset",
		},
		{
			name: "gocache",
			mut: func(env *corpusPublishEnvironment) {
				env.BlockedGoEnv = map[string]string{"GOCACHE": filepath.Join(t.TempDir(), "cache")}
			},
			want: "GOCACHE unset",
		},
		{
			name: "missing cpu affinity",
			mut:  func(env *corpusPublishEnvironment) { env.CPUAffinity = " " },
			want: envBenchCorpusCPUAffinity,
		},
		{
			name: "missing power mode",
			mut:  func(env *corpusPublishEnvironment) { env.PowerMode = "" },
			want: envBenchCorpusPowerMode,
		},
		{
			name: "missing thermal state",
			mut:  func(env *corpusPublishEnvironment) { env.ThermalState = "" },
			want: envBenchCorpusThermalState,
		},
		{
			name: "missing frequency policy",
			mut:  func(env *corpusPublishEnvironment) { env.FrequencyPolicy = "" },
			want: envBenchCorpusFrequencyPolicy,
		},
		{
			name: "missing background load",
			mut:  func(env *corpusPublishEnvironment) { env.BackgroundLoad = "" },
			want: envBenchCorpusBackgroundLoad,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env := valid
			tc.mut(&env)
			if err := validateCorpusPublishEnvironment(env); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestLoadCorpusPublishManifestRejectsStaleCorpus(t *testing.T) {
	md5Hex := "0123456789abcdeffedcba9876543210"
	for _, tc := range []struct {
		name  string
		setup func(t *testing.T, dir string)
	}{
		{
			name: "missing manifest",
		},
		{
			name: "expected count mismatch",
			setup: func(t *testing.T, dir string) {
				writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 2)
			},
		},
		{
			name: "missing source provenance",
			setup: func(t *testing.T, dir string) {
				writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 1)
				path := filepath.Join(dir, corpusManifestFile)
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				data = []byte(strings.Replace(string(data), "# source_url=https://example.invalid/source\n", "", 1))
				if err := os.WriteFile(path, data, 0o644); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "missing tool sha256",
			setup: func(t *testing.T, dir string) {
				writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 1)
				path := filepath.Join(dir, corpusManifestFile)
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				data = []byte(strings.Replace(string(data), "# aomenc_sha256="+testSHA256([]byte("aomenc fixture"))+"\n", "", 1))
				if err := os.WriteFile(path, data, 0o644); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "ivf hash mismatch",
			setup: func(t *testing.T, dir string) {
				writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 1)
				path := filepath.Join(dir, corpusManifestFile)
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				badHash := strings.Repeat("0", 64)
				data = []byte(strings.Replace(string(data), testSHA256([]byte("ivf-data")), badHash, 1))
				if err := os.WriteFile(path, data, 0o644); err != nil {
					t.Fatal(err)
				}
			},
		},
		{
			name: "extra ivf",
			setup: func(t *testing.T, dir string) {
				writeCorpusManifestFixture(t, dir, "clip", []byte("ivf-data"), []byte(md5Hex+"\n"), md5Hex, 2, 4, 3, 8, "420", 1)
				if err := os.WriteFile(filepath.Join(dir, "extra.ivf"), []byte("extra"), 0o644); err != nil {
					t.Fatal(err)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.setup != nil {
				tc.setup(t, dir)
			}
			if _, err := loadCorpusPublishManifest(dir); err == nil {
				t.Fatal("loadCorpusPublishManifest accepted stale corpus")
			}
		})
	}
}

func TestParseCorpusPublishManifestRowRequiresDAV1DOKForCheckedRows(t *testing.T) {
	md5Hex := "0123456789abcdeffedcba9876543210"
	sha := strings.Repeat("1", 64)
	row := func(depth int, chroma string, dav1d string) string {
		return fmt.Sprintf("clip\t64\t64\t3\t32\t%d\t%s\t0\t123\t%s\t%s\t%s\t%s\targs",
			depth, chroma, sha, md5Hex, sha, dav1d)
	}
	if _, err := parseCorpusPublishManifestRow(row(8, "420", "dav1d=MISMATCH(deadbeef)")); err == nil ||
		!strings.Contains(err.Error(), "dav1d_check") {
		t.Fatalf("8-bit 4:2:0 mismatch error=%v", err)
	}
	if _, err := parseCorpusPublishManifestRow(row(8, "420", "dav1d=OK")); err != nil {
		t.Fatalf("8-bit 4:2:0 dav1d OK failed: %v", err)
	}
	if _, err := parseCorpusPublishManifestRow(row(10, "420", "dav1d skipped")); err != nil {
		t.Fatalf("10-bit row should not require dav1d OK: %v", err)
	}
	if _, err := parseCorpusPublishManifestRow(row(8, "444", "dav1d skipped")); err != nil {
		t.Fatalf("4:4:4 row should not require dav1d OK: %v", err)
	}
}

func writeCorpusManifestFixture(t *testing.T, dir, name string, ivfData, md5Data []byte, md5Hex string, width, height, frames, depth int, chroma string, expectedClips int) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name+".ivf"), ivfData, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name+".md5"), md5Data, 0o644); err != nil {
		t.Fatal(err)
	}
	row := fmt.Sprintf("%s\t%d\t%d\t%d\t32\t%d\t%s\t0\t%d\t%s\t%s\t%s\tdav1d=OK\targs",
		name, width, height, frames, depth, chroma, len(ivfData), testSHA256(ivfData), md5Hex, testSHA256(md5Data))
	text := strings.Join([]string{
		corpusManifestMagicV1,
		"# source_id=fixture-source",
		"# source_sha256=" + testSHA256([]byte("source fixture")),
		"# source_url=https://example.invalid/source",
		"# source_license=fixture-license",
		"# source_category=fixture-category",
		"# aomenc_sha256=" + testSHA256([]byte("aomenc fixture")),
		"# aomenc_version=aomenc fixture",
		"# aomdec_sha256=" + testSHA256([]byte("aomdec fixture")),
		"# aomdec_version=aomdec fixture",
		"# ffmpeg_sha256=" + testSHA256([]byte("ffmpeg fixture")),
		"# ffmpeg_version=ffmpeg fixture",
		fmt.Sprintf("# expected_clips=%d", expectedClips),
		corpusManifestColumnsV1,
		row,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(dir, corpusManifestFile), []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

type corpusManifestV2FixtureClip struct {
	name           string
	sourceID       string
	sourceCategory string
	sourceData     []byte
}

func writeCorpusManifestV2Fixture(t *testing.T, dir string, clips []corpusManifestV2FixtureClip, md5Hex string) {
	t.Helper()
	md5Data := []byte(md5Hex + "\n")
	rows := make([]string, 0, len(clips))
	for i, clip := range clips {
		ivfData := []byte(fmt.Sprintf("ivf-data-%d", i))
		if err := os.WriteFile(filepath.Join(dir, clip.name+".ivf"), ivfData, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, clip.name+".md5"), md5Data, 0o644); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, fmt.Sprintf("%s\t2\t4\t3\t32\t8\t420\t0\t%d\t%s\t%s\t%s\tdav1d=OK\targs\t%s\t%s\thttps://example.invalid/%s\tfixture-license\t%s",
			clip.name, len(ivfData), testSHA256(ivfData), md5Hex, testSHA256(md5Data),
			clip.sourceID, testSHA256(clip.sourceData), clip.sourceID, clip.sourceCategory))
	}
	lines := []string{
		corpusManifestMagicV2,
		fmt.Sprintf("# source_count=%d", len(clips)),
		"# aomenc_sha256=" + testSHA256([]byte("aomenc fixture")),
		"# aomenc_version=aomenc fixture",
		"# aomdec_sha256=" + testSHA256([]byte("aomdec fixture")),
		"# aomdec_version=aomdec fixture",
		"# ffmpeg_sha256=" + testSHA256([]byte("ffmpeg fixture")),
		"# ffmpeg_version=ffmpeg fixture",
		fmt.Sprintf("# expected_clips=%d", len(clips)),
		corpusManifestColumnsV2,
	}
	lines = append(lines, rows...)
	text := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(dir, corpusManifestFile), []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func testSHA256(data []byte) string {
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum[:])
}

func TestExternalCorpusFrameMD5SidecarSmoke(t *testing.T) {
	src := filepath.Join("testdata", "profiles", "profile1-444-8bit-64x64.ivf")
	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read profile smoke clip: %v", err)
	}
	dir := t.TempDir()
	ivfPath := filepath.Join(dir, "clip.ivf")
	if err := os.WriteFile(ivfPath, data, 0o644); err != nil {
		t.Fatalf("write profile smoke clip: %v", err)
	}
	md5Path := ivfPath + ".md5"
	if err := os.WriteFile(md5Path, []byte(strings.Join([]string{
		"00211cdc8f799c808849c955a318a0f5",
		"397ff01920ff514bc611ab49d76371c1",
		"f8fbfb25a42da47a7adb71510de9b178",
	}, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write frame-md5 sidecar: %v", err)
	}

	candidates, skippedNoMD5 := discoverExternalCorpusCandidates(t, []string{dir})
	if skippedNoMD5 != 0 {
		t.Fatalf("skipped_no_md5=%d want 0", skippedNoMD5)
	}
	if len(candidates) != 1 {
		t.Fatalf("candidates=%d want 1", len(candidates))
	}
	clips, failed := loadCorpusClipCandidates(t, candidates)
	if len(failed) != 0 {
		t.Fatalf("failed=%v", failed)
	}
	if len(clips) != 1 {
		t.Fatalf("clips=%d want 1", len(clips))
	}
	clip := clips[0]
	if clip.oracleKind != corpusOracleFrameMD5 {
		t.Fatalf("oracleKind=%d want frame MD5", clip.oracleKind)
	}
	if clip.oraclePath != md5Path {
		t.Fatalf("oraclePath=%q want %q", clip.oraclePath, md5Path)
	}
	if clip.frames != 3 || clip.width != 64 || clip.height != 64 || clip.bitDepth != 8 || clip.chroma != "444" {
		t.Fatalf("metadata=%dx%d %d-bit %s frames=%d, want 64x64 8-bit 444 frames=3",
			clip.width, clip.height, clip.bitDepth, clip.chroma, clip.frames)
	}
	if got := corpusClipOracleLog(clip); !strings.Contains(got, "frame_md5s=3") || !strings.Contains(got, "clip.ivf.md5") {
		t.Fatalf("oracle log %q missing frame count/sidecar", got)
	}
}

// TestGeneratedCorpusConformance verifies every generated corpus clip once,
// without the repeated timing loop from TestCrossDecoderCorpus. It is an
// opt-in broad real-content conformance gate for locally materialized clips.
func TestGeneratedCorpusConformance(t *testing.T) {
	if os.Getenv("GOAV1_CORPUS_CONFORMANCE") != "1" {
		t.Skip("set GOAV1_CORPUS_CONFORMANCE=1 (with the generated corpus) to run generated-corpus conformance")
	}
	dir, ok := corpusDir(t)
	if !ok {
		t.Skipf("generated-corpus: no clips in %s (regenerate with scripts/gen_bench_corpus.sh)", dir)
	}
	t.Logf("generated-corpus: corpus dir = %s", dir)

	clips, failed := loadCorpusClips(t, dir)
	if len(failed) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "\n!!! CONFORMANCE BUGS: %d generated corpus clip(s) FAILED goav1 byte-exact decode !!!\n", len(failed))
		for _, f := range failed {
			fmt.Fprintf(&b, "    - %-26s %s\n", f.name, f.reason)
		}
		t.Fatal(b.String())
	}
	if len(clips) == 0 {
		t.Skip("generated-corpus: no usable clips")
	}

	totalFrames := 0
	for _, clip := range clips {
		totalFrames += clip.frames
		t.Logf("generated-corpus: %-30s %dx%d %d-bit %s frames=%d tiles=%d %s",
			clip.name, clip.width, clip.height, clip.bitDepth, clip.chroma, clip.frames, clip.tileCols, corpusClipOracleLog(clip))
	}
	t.Logf("generated-corpus: %d clips / %d visible frames passed MD5 conformance", len(clips), totalFrames)
}

func TestExternalCorpusConformance(t *testing.T) {
	if os.Getenv("GOAV1_EXTERNAL_CORPUS") != "1" {
		t.Skip("set GOAV1_EXTERNAL_CORPUS=1 with GOAV1_EXTERNAL_CORPUS_DIR(S) to run external-corpus conformance")
	}
	dirs := externalCorpusDirsFromEnv()
	if len(dirs) == 0 {
		t.Skip("external-corpus: set GOAV1_EXTERNAL_CORPUS_DIR or GOAV1_EXTERNAL_CORPUS_DIRS")
	}

	candidates, skippedNoMD5 := discoverExternalCorpusCandidates(t, dirs)
	if len(candidates) == 0 {
		t.Skipf("external-corpus: no .ivf clips with supported MD5 sidecars in %s (skipped %d .ivf without sidecars)",
			strings.Join(dirs, string(os.PathListSeparator)), skippedNoMD5)
	}
	t.Logf("external-corpus: roots=%s usable_ivf=%d skipped_no_md5=%d",
		strings.Join(dirs, string(os.PathListSeparator)), len(candidates), skippedNoMD5)

	clips, failed := loadCorpusClipCandidates(t, candidates)
	if len(failed) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "\n!!! CONFORMANCE BUGS: %d external corpus clip(s) FAILED goav1 byte-exact decode !!!\n", len(failed))
		for _, f := range failed {
			fmt.Fprintf(&b, "    - %-50s %s\n", f.name, f.reason)
		}
		t.Fatal(b.String())
	}
	if len(clips) == 0 {
		t.Skip("external-corpus: no usable clips")
	}

	totalFrames := 0
	for _, clip := range clips {
		totalFrames += clip.frames
		t.Logf("external-corpus: %-50s %dx%d %d-bit %s frames=%d tiles=%d %s",
			clip.name, clip.width, clip.height, clip.bitDepth, clip.chroma, clip.frames, clip.tileCols, corpusClipOracleLog(clip))
	}
	t.Logf("external-corpus: %d clips / %d visible frames passed MD5 conformance", len(clips), totalFrames)
}

func externalCorpusDirsFromEnv() []string {
	raw := os.Getenv("GOAV1_EXTERNAL_CORPUS_DIRS")
	if raw == "" {
		raw = os.Getenv("GOAV1_EXTERNAL_CORPUS_DIR")
	}
	var dirs []string
	seen := map[string]bool{}
	for _, group := range strings.Split(raw, ",") {
		for _, dir := range filepath.SplitList(group) {
			dir = strings.TrimSpace(dir)
			if dir == "" {
				continue
			}
			dir = filepath.Clean(dir)
			if seen[dir] {
				continue
			}
			seen[dir] = true
			dirs = append(dirs, dir)
		}
	}
	return dirs
}

func discoverExternalCorpusCandidates(t *testing.T, dirs []string) (candidates []corpusClipCandidate, skippedNoMD5 int) {
	t.Helper()
	for _, root := range dirs {
		info, err := os.Stat(root)
		if err != nil {
			t.Fatalf("external-corpus: stat %s: %v", root, err)
		}
		if !info.IsDir() {
			t.Fatalf("external-corpus: %s is not a directory", root)
		}
		err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() || !strings.EqualFold(filepath.Ext(path), ".ivf") {
				return nil
			}
			ok, err := corpusOracleSidecarExists(path)
			if err != nil {
				return err
			}
			if !ok {
				skippedNoMD5++
				return nil
			}
			candidates = append(candidates, corpusClipCandidate{
				name:    externalCorpusClipName(root, path),
				ivfPath: path,
			})
			return nil
		})
		if err != nil {
			t.Fatalf("external-corpus: walk %s: %v", root, err)
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].ivfPath < candidates[j].ivfPath
	})
	return candidates, skippedNoMD5
}

func externalCorpusClipName(root, path string) string {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		rel = filepath.Base(path)
	}
	rel = filepath.ToSlash(strings.TrimSuffix(rel, filepath.Ext(rel)))
	prefix := filepath.Base(filepath.Clean(root))
	if prefix == "." || prefix == string(filepath.Separator) {
		return rel
	}
	return prefix + "/" + rel
}

func loadCorpusPublishManifest(dir string) (corpusPublishManifest, error) {
	path := filepath.Join(dir, corpusManifestFile)
	data, err := os.ReadFile(path)
	if err != nil {
		return corpusPublishManifest{}, fmt.Errorf("read %s: %w", path, err)
	}
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return corpusPublishManifest{}, fmt.Errorf("%s: missing manifest header", path)
	}
	formatVersion := 0
	columns := ""
	switch strings.TrimSpace(lines[0]) {
	case corpusManifestMagicV1:
		formatVersion = 1
		columns = corpusManifestColumnsV1
	case corpusManifestMagicV2:
		formatVersion = 2
		columns = corpusManifestColumnsV2
	default:
		return corpusPublishManifest{}, fmt.Errorf("%s: missing %q or %q header", path, corpusManifestMagicV1, corpusManifestMagicV2)
	}

	headers := map[string]string{}
	manifest := corpusPublishManifest{
		path:          path,
		formatVersion: formatVersion,
		toolVersions:  map[string]string{},
		toolSHA256:    map[string]string{},
		rows:          map[string]corpusPublishManifestRow{},
	}
	sawColumns := false
	for i, line := range lines[1:] {
		lineNo := i + 2
		if strings.TrimSpace(line) == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if sawColumns {
				return corpusPublishManifest{}, fmt.Errorf("%s:%d: comment after column header", path, lineNo)
			}
			key, value, ok := strings.Cut(strings.TrimSpace(strings.TrimPrefix(line, "#")), "=")
			if ok {
				headers[strings.TrimSpace(key)] = strings.TrimSpace(value)
			}
			continue
		}
		if !sawColumns {
			if line != columns {
				return corpusPublishManifest{}, fmt.Errorf("%s:%d: unexpected columns %q", path, lineNo, line)
			}
			sawColumns = true
			continue
		}
		row, err := parseCorpusPublishManifestRowVersion(line, formatVersion)
		if err != nil {
			return corpusPublishManifest{}, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if _, exists := manifest.rows[row.name]; exists {
			return corpusPublishManifest{}, fmt.Errorf("%s:%d: duplicate clip %q", path, lineNo, row.name)
		}
		manifest.rows[row.name] = row
	}
	if !sawColumns {
		return corpusPublishManifest{}, fmt.Errorf("%s: missing column header", path)
	}
	switch formatVersion {
	case 1:
		if manifest.sourceID, err = requiredCorpusManifestHeader(path, headers, "source_id"); err != nil {
			return corpusPublishManifest{}, err
		}
		if manifest.sourceSHA256, err = requiredCorpusManifestSHA256Header(path, headers, "source_sha256"); err != nil {
			return corpusPublishManifest{}, err
		}
		if manifest.sourceURL, err = requiredCorpusManifestHeader(path, headers, "source_url"); err != nil {
			return corpusPublishManifest{}, err
		}
		if manifest.sourceLicense, err = requiredCorpusManifestHeader(path, headers, "source_license"); err != nil {
			return corpusPublishManifest{}, err
		}
		if manifest.sourceCategory, err = requiredCorpusManifestHeader(path, headers, "source_category"); err != nil {
			return corpusPublishManifest{}, err
		}
		manifest.sources = []corpusPublishManifestSource{{
			id:       manifest.sourceID,
			sha256:   manifest.sourceSHA256,
			url:      manifest.sourceURL,
			license:  manifest.sourceLicense,
			category: manifest.sourceCategory,
			clips:    len(manifest.rows),
		}}
		for name, row := range manifest.rows {
			row.sourceID = manifest.sourceID
			row.sourceSHA256 = manifest.sourceSHA256
			row.sourceURL = manifest.sourceURL
			row.sourceLicense = manifest.sourceLicense
			row.sourceCategory = manifest.sourceCategory
			manifest.rows[name] = row
		}
	case 2:
		expectedSources, err := requiredPositiveCorpusManifestHeader(path, headers, "source_count")
		if err != nil {
			return corpusPublishManifest{}, err
		}
		manifest.sources, err = corpusPublishManifestSources(manifest.rows)
		if err != nil {
			return corpusPublishManifest{}, fmt.Errorf("%s: %w", path, err)
		}
		if len(manifest.sources) != expectedSources {
			return corpusPublishManifest{}, fmt.Errorf("%s: source_count=%d but rows contain %d source(s)", path, expectedSources, len(manifest.sources))
		}
		manifest.assignLegacySourceSummary()
	}
	for _, tool := range []string{"aomenc", "aomdec", "ffmpeg"} {
		version, err := requiredCorpusManifestHeader(path, headers, tool+"_version")
		if err != nil {
			return corpusPublishManifest{}, err
		}
		sha, err := requiredCorpusManifestSHA256Header(path, headers, tool+"_sha256")
		if err != nil {
			return corpusPublishManifest{}, err
		}
		manifest.toolVersions[tool] = version
		manifest.toolSHA256[tool] = sha
	}
	if err := loadOptionalCorpusToolManifestHeader(path, headers, "dav1d", manifest.toolVersions, manifest.toolSHA256); err != nil {
		return corpusPublishManifest{}, err
	}

	expectedRaw, ok := headers["expected_clips"]
	if !ok {
		return corpusPublishManifest{}, fmt.Errorf("%s: missing expected_clips header", path)
	}
	expected, err := strconv.Atoi(expectedRaw)
	if err != nil || expected <= 0 {
		return corpusPublishManifest{}, fmt.Errorf("%s: invalid expected_clips=%q", path, expectedRaw)
	}
	manifest.expectedClips = expected
	if len(manifest.rows) != expected {
		return corpusPublishManifest{}, fmt.Errorf("%s: manifest rows=%d, expected_clips=%d", path, len(manifest.rows), expected)
	}
	if err := validateCorpusPublishManifestFiles(dir, manifest); err != nil {
		return corpusPublishManifest{}, err
	}
	return manifest, nil
}

func requiredCorpusManifestHeader(path string, headers map[string]string, name string) (string, error) {
	value := strings.TrimSpace(headers[name])
	if value == "" {
		return "", fmt.Errorf("%s: missing %s header", path, name)
	}
	return value, nil
}

func requiredCorpusManifestSHA256Header(path string, headers map[string]string, name string) (string, error) {
	value, err := requiredCorpusManifestHeader(path, headers, name)
	if err != nil {
		return "", err
	}
	if err := validateCorpusManifestSHA256(value, name); err != nil {
		return "", fmt.Errorf("%s: %w", path, err)
	}
	return strings.ToLower(strings.TrimSpace(value)), nil
}

func requiredPositiveCorpusManifestHeader(path string, headers map[string]string, name string) (int, error) {
	value := strings.TrimSpace(headers[name])
	if value == "" {
		return 0, fmt.Errorf("%s: missing %s header", path, name)
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return 0, fmt.Errorf("%s: invalid %s=%q", path, name, value)
	}
	return parsed, nil
}

func loadOptionalCorpusToolManifestHeader(path string, headers map[string]string, tool string, versions, hashes map[string]string) error {
	version := strings.TrimSpace(headers[tool+"_version"])
	sha := strings.TrimSpace(headers[tool+"_sha256"])
	if version == "" && sha == "" {
		return nil
	}
	if version == "" {
		return fmt.Errorf("%s: missing %s_version header for present %s_sha256", path, tool, tool)
	}
	if sha == "" {
		return fmt.Errorf("%s: missing %s_sha256 header for present %s_version", path, tool, tool)
	}
	if err := validateCorpusManifestSHA256(sha, tool+"_sha256"); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	versions[tool] = version
	hashes[tool] = strings.ToLower(sha)
	return nil
}

func parseCorpusPublishManifestRow(line string) (corpusPublishManifestRow, error) {
	return parseCorpusPublishManifestRowVersion(line, 1)
}

func parseCorpusPublishManifestRowVersion(line string, formatVersion int) (corpusPublishManifestRow, error) {
	fields := strings.Split(line, "\t")
	wantFields := 14
	if formatVersion == 2 {
		wantFields = 19
	}
	if len(fields) != wantFields {
		return corpusPublishManifestRow{}, fmt.Errorf("fields=%d want %d", len(fields), wantFields)
	}
	name := strings.TrimSpace(fields[0])
	if name == "" || strings.ContainsAny(name, `/\`) || name == "." || name == ".." {
		return corpusPublishManifestRow{}, fmt.Errorf("invalid clip name %q", fields[0])
	}
	width, err := parsePositiveCorpusManifestInt(fields[1], "width")
	if err != nil {
		return corpusPublishManifestRow{}, err
	}
	height, err := parsePositiveCorpusManifestInt(fields[2], "height")
	if err != nil {
		return corpusPublishManifestRow{}, err
	}
	frames, err := parsePositiveCorpusManifestInt(fields[3], "frames")
	if err != nil {
		return corpusPublishManifestRow{}, err
	}
	cq, err := parsePositiveCorpusManifestInt(fields[4], "cq")
	if err != nil {
		return corpusPublishManifestRow{}, err
	}
	depth, err := parsePositiveCorpusManifestInt(fields[5], "depth")
	if err != nil {
		return corpusPublishManifestRow{}, err
	}
	if depth > 255 {
		return corpusPublishManifestRow{}, fmt.Errorf("depth=%d out of range", depth)
	}
	profile, err := parseCorpusManifestProfile(fields[7])
	if err != nil {
		return corpusPublishManifestRow{}, err
	}
	ivfBytes, err := strconv.ParseInt(fields[8], 10, 64)
	if err != nil || ivfBytes <= 0 {
		return corpusPublishManifestRow{}, fmt.Errorf("invalid ivf_bytes=%q", fields[8])
	}
	if err := validateCorpusManifestSHA256(fields[9], "ivf_sha256"); err != nil {
		return corpusPublishManifestRow{}, err
	}
	md5, err := ParseMD5Hex([]byte(strings.TrimSpace(fields[10])))
	if err != nil {
		return corpusPublishManifestRow{}, fmt.Errorf("invalid md5=%q", fields[10])
	}
	if err := validateCorpusManifestSHA256(fields[11], "md5_sha256"); err != nil {
		return corpusPublishManifestRow{}, err
	}
	chroma := strings.TrimSpace(fields[6])
	switch chroma {
	case "420", "422", "444":
	default:
		return corpusPublishManifestRow{}, fmt.Errorf("invalid chroma=%q", chroma)
	}
	dav1dCheck := strings.TrimSpace(fields[12])
	if dav1dCheck == "" {
		return corpusPublishManifestRow{}, errors.New("empty dav1d_check")
	}
	if depth == 8 && chroma == "420" && dav1dCheck != "dav1d=OK" {
		return corpusPublishManifestRow{}, fmt.Errorf("8-bit 4:2:0 dav1d_check=%q, want dav1d=OK", dav1dCheck)
	}
	aomencArgs := strings.TrimSpace(fields[13])
	if aomencArgs == "" {
		return corpusPublishManifestRow{}, errors.New("empty aomenc_args")
	}
	row := corpusPublishManifestRow{
		name:       name,
		width:      width,
		height:     height,
		frames:     frames,
		cq:         cq,
		bitDepth:   uint8(depth),
		chroma:     chroma,
		profile:    profile,
		ivfBytes:   ivfBytes,
		ivfSHA256:  strings.ToLower(strings.TrimSpace(fields[9])),
		md5:        md5,
		md5SHA256:  strings.ToLower(strings.TrimSpace(fields[11])),
		dav1dCheck: dav1dCheck,
		aomencArgs: aomencArgs,
	}
	if formatVersion == 2 {
		if row.sourceID, err = parseRequiredCorpusManifestField(fields[14], "source_id"); err != nil {
			return corpusPublishManifestRow{}, err
		}
		if err := validateCorpusManifestSHA256(fields[15], "source_sha256"); err != nil {
			return corpusPublishManifestRow{}, err
		}
		row.sourceSHA256 = strings.ToLower(strings.TrimSpace(fields[15]))
		if row.sourceURL, err = parseRequiredCorpusManifestField(fields[16], "source_url"); err != nil {
			return corpusPublishManifestRow{}, err
		}
		if row.sourceLicense, err = parseRequiredCorpusManifestField(fields[17], "source_license"); err != nil {
			return corpusPublishManifestRow{}, err
		}
		if row.sourceCategory, err = parseRequiredCorpusManifestField(fields[18], "source_category"); err != nil {
			return corpusPublishManifestRow{}, err
		}
	}
	return row, nil
}

func parseRequiredCorpusManifestField(raw, name string) (string, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return "", fmt.Errorf("empty %s", name)
	}
	if strings.ContainsAny(value, "\r\n\t") {
		return "", fmt.Errorf("invalid %s contains control whitespace", name)
	}
	return value, nil
}

func corpusPublishManifestSources(rows map[string]corpusPublishManifestRow) ([]corpusPublishManifestSource, error) {
	sourcesByID := make(map[string]corpusPublishManifestSource)
	for _, row := range rows {
		source := corpusPublishManifestSource{
			id:       row.sourceID,
			sha256:   row.sourceSHA256,
			url:      row.sourceURL,
			license:  row.sourceLicense,
			category: row.sourceCategory,
			clips:    1,
		}
		if existing, ok := sourcesByID[source.id]; ok {
			if existing.sha256 != source.sha256 ||
				existing.url != source.url ||
				existing.license != source.license ||
				existing.category != source.category {
				return nil, fmt.Errorf("source_id %q has conflicting provenance", source.id)
			}
			existing.clips++
			sourcesByID[source.id] = existing
			continue
		}
		sourcesByID[source.id] = source
	}
	sources := make([]corpusPublishManifestSource, 0, len(sourcesByID))
	for _, source := range sourcesByID {
		sources = append(sources, source)
	}
	sort.Slice(sources, func(i, j int) bool { return sources[i].id < sources[j].id })
	return sources, nil
}

func (manifest *corpusPublishManifest) assignLegacySourceSummary() {
	if len(manifest.sources) == 0 {
		return
	}
	if len(manifest.sources) == 1 {
		source := manifest.sources[0]
		manifest.sourceID = source.id
		manifest.sourceSHA256 = source.sha256
		manifest.sourceURL = source.url
		manifest.sourceLicense = source.license
		manifest.sourceCategory = source.category
		return
	}
	manifest.sourceID = "multiple"
	manifest.sourceSHA256 = "multiple"
	manifest.sourceURL = "multiple"
	manifest.sourceLicense = "multiple"
	manifest.sourceCategory = "multiple"
}

func parseCorpusManifestProfile(raw string) (int, error) {
	profile, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || profile < 0 || profile > 2 {
		return 0, fmt.Errorf("invalid profile=%q", raw)
	}
	return profile, nil
}

func parsePositiveCorpusManifestInt(raw, name string) (int, error) {
	v, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("invalid %s=%q", name, raw)
	}
	return v, nil
}

func validateCorpusManifestSHA256(raw, name string) error {
	raw = strings.TrimSpace(raw)
	if len(raw) != 64 {
		return fmt.Errorf("invalid %s length=%d", name, len(raw))
	}
	for _, c := range raw {
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') {
			continue
		}
		return fmt.Errorf("invalid %s=%q", name, raw)
	}
	return nil
}

func validateCorpusPublishManifestFiles(dir string, manifest corpusPublishManifest) error {
	paths, err := filepath.Glob(filepath.Join(dir, "*.ivf"))
	if err != nil {
		return fmt.Errorf("glob corpus IVF: %w", err)
	}
	for _, path := range paths {
		name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		if _, ok := manifest.rows[name]; !ok {
			return fmt.Errorf("%s: extra IVF not listed in manifest: %s", manifest.path, filepath.Base(path))
		}
	}

	names := make([]string, 0, len(manifest.rows))
	for name := range manifest.rows {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		row := manifest.rows[name]
		ivfPath := filepath.Join(dir, row.name+".ivf")
		ivfSHA, ivfBytes, err := corpusFileSHA256(ivfPath)
		if err != nil {
			return fmt.Errorf("%s: %w", manifest.path, err)
		}
		if ivfBytes != row.ivfBytes {
			return fmt.Errorf("%s: %s.ivf bytes=%d want %d", manifest.path, row.name, ivfBytes, row.ivfBytes)
		}
		if !strings.EqualFold(ivfSHA, row.ivfSHA256) {
			return fmt.Errorf("%s: %s.ivf sha256=%s want %s", manifest.path, row.name, ivfSHA, row.ivfSHA256)
		}

		md5Path := filepath.Join(dir, row.name+".md5")
		md5SHA, _, err := corpusFileSHA256(md5Path)
		if err != nil {
			return fmt.Errorf("%s: %w", manifest.path, err)
		}
		if !strings.EqualFold(md5SHA, row.md5SHA256) {
			return fmt.Errorf("%s: %s.md5 sha256=%s want %s", manifest.path, row.name, md5SHA, row.md5SHA256)
		}
		oracle, err := loadCorpusOracleSidecar(ivfPath)
		if err != nil {
			return fmt.Errorf("%s: %s.md5: %w", manifest.path, row.name, err)
		}
		if oracle.kind != corpusOracleStreamMD5 {
			return fmt.Errorf("%s: %s.md5 is not a stream-MD5 sidecar", manifest.path, row.name)
		}
		if oracle.streamMD5 != row.md5 {
			return fmt.Errorf("%s: %s.md5=%x want %x", manifest.path, row.name, oracle.streamMD5, row.md5)
		}
	}
	return nil
}

func corpusFileSHA256(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), n, nil
}

func validateCorpusPublishLoadedClips(manifest corpusPublishManifest, clips []corpusClip) error {
	seen := make(map[string]bool, len(clips))
	for _, clip := range clips {
		row, ok := manifest.rows[clip.name]
		if !ok {
			return fmt.Errorf("%s: loaded clip %q is not listed in manifest", manifest.path, clip.name)
		}
		seen[clip.name] = true
		if clip.width != row.width || clip.height != row.height || clip.frames != row.frames || clip.bitDepth != row.bitDepth || clip.chroma != row.chroma {
			return fmt.Errorf("%s: %s metadata=%dx%d frames=%d depth=%d chroma=%s, want %dx%d frames=%d depth=%d chroma=%s",
				manifest.path, clip.name, clip.width, clip.height, clip.frames, clip.bitDepth, clip.chroma,
				row.width, row.height, row.frames, row.bitDepth, row.chroma)
		}
		if clip.oracleKind != corpusOracleStreamMD5 || clip.wantMD5 != row.md5 {
			return fmt.Errorf("%s: %s oracle md5=%x kind=%d, want stream md5=%x",
				manifest.path, clip.name, clip.wantMD5, clip.oracleKind, row.md5)
		}
	}
	if len(seen) != len(manifest.rows) {
		names := make([]string, 0, len(manifest.rows))
		for name := range manifest.rows {
			if !seen[name] {
				names = append(names, name)
			}
		}
		sort.Strings(names)
		return fmt.Errorf("%s: manifest clip(s) not loaded: %s", manifest.path, strings.Join(names, ", "))
	}
	return nil
}

func validateCorpusPublishSourceCoverage(manifest corpusPublishManifest) error {
	minSources, err := corpusPublishMinimum(envBenchCorpusMinSources, 2)
	if err != nil {
		return err
	}
	minCategories, err := corpusPublishMinimum(envBenchCorpusMinCategories, 2)
	if err != nil {
		return err
	}
	if len(manifest.sources) < minSources {
		return fmt.Errorf("%s: publish corpus has %d source(s), want at least %d; regenerate with GOAV1_BENCH_SOURCES_TSV",
			manifest.path, len(manifest.sources), minSources)
	}
	categoryCount := corpusManifestSourceCategoryCount(manifest.sources)
	if categoryCount < minCategories {
		return fmt.Errorf("%s: publish corpus has %d source category/categories, want at least %d; regenerate with GOAV1_BENCH_SOURCES_TSV",
			manifest.path, categoryCount, minCategories)
	}
	if manifest.formatVersion < 2 {
		return fmt.Errorf("%s: publish corpus uses v%d manifest; regenerate with GOAV1_BENCH_SOURCES_TSV for row-level source provenance",
			manifest.path, manifest.formatVersion)
	}
	return nil
}

func corpusPublishMinimum(envName string, fallback int) (int, error) {
	raw := strings.TrimSpace(os.Getenv(envName))
	if raw == "" {
		return fallback, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer, got %q", envName, raw)
	}
	return value, nil
}

func loadCorpusBenchmarkManifest(t *testing.T, dir string, publish bool) (corpusPublishManifest, bool) {
	t.Helper()
	manifest, err := loadCorpusPublishManifest(dir)
	if err != nil {
		if publish {
			t.Fatalf("cross-corpus publish: %v", err)
		}
		if !allowMissingCorpusBenchmarkManifest() {
			t.Fatalf("cross-corpus: no valid %s in %s (%v); set %s=1 only for exploratory timing on intentionally unmanifested local data",
				corpusManifestFile, dir, err, envBenchCorpusAllowUnmanifested)
		}
		t.Logf("cross-corpus: WARNING: no valid %s in %s (%v); exploratory timing is not publishable and may use stale or partial ignored corpus data",
			corpusManifestFile, dir, err)
		return corpusPublishManifest{}, false
	}
	label := "cross-corpus"
	if publish {
		label = "cross-corpus publish"
	}
	t.Logf("%s: manifest=%s expected_clips=%d", label, manifest.path, manifest.expectedClips)
	return manifest, true
}

func allowMissingCorpusBenchmarkManifest() bool {
	return os.Getenv(envBenchCorpusAllowUnmanifested) == "1"
}

type resolvedCorpusExternalDecoder struct {
	decoder externalDecoder
	bin     string
}

type corpusTimingDecoder struct {
	name       string
	inProcess  bool
	external   externalDecoder
	bin        string
	resultSlot int
}

type corpusPublishReport struct {
	GeneratedAtUTC string                       `json:"generated_at_utc"`
	Git            corpusPublishGit             `json:"git"`
	Environment    corpusPublishEnvironment     `json:"environment"`
	Corpus         corpusPublishReportCorpus    `json:"corpus"`
	Timing         corpusPublishReportTiming    `json:"timing"`
	Tools          []corpusPublishReportTool    `json:"tools"`
	Clips          []corpusPublishReportClip    `json:"clips"`
	Decoders       []corpusPublishReportDecoder `json:"decoders"`
}

type corpusPublishGit struct {
	Commit string `json:"commit"`
	Dirty  bool   `json:"dirty"`
	Error  string `json:"error,omitempty"`
}

type corpusPublishEnvironment struct {
	GoVersion            string            `json:"go_version"`
	GOOS                 string            `json:"goos"`
	GOARCH               string            `json:"goarch"`
	GOMAXPROCS           int               `json:"gomaxprocs"`
	GOMAXPROCSEnv        string            `json:"gomaxprocs_env,omitempty"`
	NumCPU               int               `json:"num_cpu"`
	SIMDTier             string            `json:"simd_tier"`
	SIMDFeatures         []string          `json:"simd_features,omitempty"`
	BuildSettings        map[string]string `json:"build_settings,omitempty"`
	GoEnv                map[string]any    `json:"go_env,omitempty"`
	GoToolPath           string            `json:"go_tool_path,omitempty"`
	GoToolSHA256         string            `json:"go_tool_sha256,omitempty"`
	GoToolExpectedSHA256 string            `json:"go_tool_expected_sha256,omitempty"`
	GoToolSHA256Verified bool              `json:"go_tool_sha256_verified,omitempty"`
	GoToolError          string            `json:"go_tool_error,omitempty"`
	BlockedGoEnv         map[string]string `json:"blocked_go_env,omitempty"`
	ObservedCPUState     benchenv.CPUState `json:"observed_cpu_state"`

	CPUModel        string `json:"cpu_model,omitempty"`
	Hostname        string `json:"hostname,omitempty"`
	OSVersion       string `json:"os_version,omitempty"`
	KernelVersion   string `json:"kernel_version,omitempty"`
	PATH            string `json:"path,omitempty"`
	GOFLAGS         string `json:"goflags,omitempty"`
	GOGC            string `json:"gogc,omitempty"`
	GOMEMLIMIT      string `json:"gomemlimit,omitempty"`
	GODEBUG         string `json:"godebug,omitempty"`
	CPUAffinity     string `json:"cpu_affinity,omitempty"`
	PowerMode       string `json:"power_mode,omitempty"`
	ThermalState    string `json:"thermal_state,omitempty"`
	FrequencyPolicy string `json:"frequency_policy,omitempty"`
	BackgroundLoad  string `json:"background_load,omitempty"`
	Notes           string `json:"notes,omitempty"`
}

type corpusPublishReportCorpus struct {
	Dir              string                      `json:"dir"`
	Manifest         string                      `json:"manifest"`
	ManifestSHA256   string                      `json:"manifest_sha256"`
	SourceID         string                      `json:"source_id"`
	SourceSHA256     string                      `json:"source_sha256"`
	SourceURL        string                      `json:"source_url"`
	SourceLicense    string                      `json:"source_license"`
	SourceCategory   string                      `json:"source_category"`
	SourceCount      int                         `json:"source_count"`
	SourceCategories int                         `json:"source_categories"`
	Sources          []corpusPublishReportSource `json:"sources,omitempty"`
	ToolVersions     map[string]string           `json:"tool_versions,omitempty"`
	ToolSHA256       map[string]string           `json:"tool_sha256,omitempty"`
	ExpectedClips    int                         `json:"expected_clips"`
	LoadedClips      int                         `json:"loaded_clips"`
	TotalFrames      int                         `json:"total_frames"`
	TimingOrder      string                      `json:"timing_order"`
	RequiredDecoders string                      `json:"required_decoders,omitempty"`
}

type corpusPublishReportSource struct {
	ID       string `json:"id"`
	SHA256   string `json:"sha256"`
	URL      string `json:"url"`
	License  string `json:"license"`
	Category string `json:"category"`
	Clips    int    `json:"clips"`
}

type corpusPublishReportTiming struct {
	Runs                   int    `json:"runs"`
	WarmupRuns             int    `json:"warmup_runs"`
	Statistic              string `json:"statistic"`
	ConcurrencyModel       string `json:"concurrency_model"`
	InProcessGoAV1         bool   `json:"in_process_goav1"`
	ExternalStartupModel   string `json:"external_startup_model"`
	ExternalCommandTimeout string `json:"external_command_timeout"`
}

type corpusPublishReportTool struct {
	Decoder      string `json:"decoder"`
	InProcess    bool   `json:"in_process,omitempty"`
	Path         string `json:"path,omitempty"`
	SHA256       string `json:"sha256,omitempty"`
	Version      string `json:"version,omitempty"`
	VersionError string `json:"version_error,omitempty"`
}

type corpusPublishReportClip struct {
	Name           string `json:"name"`
	Path           string `json:"path"`
	Frames         int    `json:"frames"`
	Width          int    `json:"width"`
	Height         int    `json:"height"`
	CQ             int    `json:"cq,omitempty"`
	BitDepth       uint8  `json:"bit_depth"`
	Chroma         string `json:"chroma"`
	Profile        int    `json:"profile"`
	TileCols       uint8  `json:"tile_cols"`
	AllIntra       bool   `json:"all_intra"`
	SourceID       string `json:"source_id,omitempty"`
	SourceSHA256   string `json:"source_sha256,omitempty"`
	SourceURL      string `json:"source_url,omitempty"`
	SourceLicense  string `json:"source_license,omitempty"`
	SourceCategory string `json:"source_category,omitempty"`
	DAV1DCheck     string `json:"dav1d_check,omitempty"`
	AOMEncArgs     string `json:"aomenc_args,omitempty"`
}

type corpusPublishReportDecoder struct {
	Name             string                          `json:"name"`
	InProcess        bool                            `json:"in_process,omitempty"`
	Frames           int                             `json:"frames"`
	RawMS            float64                         `json:"raw_ms"`
	RawFPS           float64                         `json:"raw_fps"`
	StartupMS        float64                         `json:"startup_ms,omitempty"`
	StartupSamplesMS []float64                       `json:"startup_samples_ms,omitempty"`
	StartupMedianMS  float64                         `json:"startup_median_ms,omitempty"`
	StartupIQRMS     float64                         `json:"startup_iqr_ms,omitempty"`
	AdjustedMS       float64                         `json:"adjusted_ms,omitempty"`
	AdjustedFPS      float64                         `json:"adjusted_fps,omitempty"`
	VsGoAV1Raw       float64                         `json:"vs_goav1_raw,omitempty"`
	PerClip          []corpusPublishReportClipTiming `json:"per_clip"`
}

type corpusPublishReportClipTiming struct {
	Clip        string    `json:"clip"`
	RawMS       float64   `json:"raw_ms"`
	SamplesMS   []float64 `json:"samples_ms,omitempty"`
	MedianMS    float64   `json:"median_ms,omitempty"`
	IQRMS       float64   `json:"iqr_ms,omitempty"`
	AdjustedMS  float64   `json:"adjusted_ms,omitempty"`
	FPS         float64   `json:"fps"`
	AdjustedFPS float64   `json:"adjusted_fps,omitempty"`
}

type corpusDurationSamples struct {
	Runs   []time.Duration
	Min    time.Duration
	Median time.Duration
	Max    time.Duration
	IQR    time.Duration
}

type corpusTimingJob struct {
	clipIndex    int
	decoderIndex int
}

func corpusInterleavedTimingJobs(clipCount, decoderCount int) []corpusTimingJob {
	if clipCount <= 0 || decoderCount <= 0 {
		return nil
	}
	jobs := make([]corpusTimingJob, 0, clipCount*decoderCount)
	for clipIndex := 0; clipIndex < clipCount; clipIndex++ {
		for offset := 0; offset < decoderCount; offset++ {
			jobs = append(jobs, corpusTimingJob{
				clipIndex:    clipIndex,
				decoderIndex: (clipIndex + offset) % decoderCount,
			})
		}
	}
	return jobs
}

func measureCorpusDurations(warmup int, runs int, fn func() error) (corpusDurationSamples, error) {
	for i := 0; i < warmup; i++ {
		if err := fn(); err != nil {
			return corpusDurationSamples{}, err
		}
	}
	samples := make([]time.Duration, 0, runs)
	for i := 0; i < runs; i++ {
		start := time.Now()
		if err := fn(); err != nil {
			return corpusDurationSamples{}, err
		}
		samples = append(samples, time.Since(start))
	}
	return summarizeCorpusDurations(samples), nil
}

func summarizeCorpusDurations(samples []time.Duration) corpusDurationSamples {
	if len(samples) == 0 {
		return corpusDurationSamples{}
	}
	ordered := append([]time.Duration(nil), samples...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	return corpusDurationSamples{
		Runs:   append([]time.Duration(nil), samples...),
		Min:    ordered[0],
		Median: ordered[len(ordered)/2],
		Max:    ordered[len(ordered)-1],
		IQR:    ordered[(3*len(ordered))/4] - ordered[len(ordered)/4],
	}
}

func corpusSelectedDuration(samples corpusDurationSamples) time.Duration {
	return samples.Median
}

func corpusRequiredExternalDecoderNames(decoders []externalDecoder, publish bool, raw string) (map[string]bool, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" && !publish {
		return nil, nil
	}

	names := make([]string, 0, len(decoders))
	known := make(map[string]string, len(decoders))
	for _, dec := range decoders {
		names = append(names, dec.name)
		known[strings.ToLower(dec.name)] = dec.name
	}
	sortedNames := append([]string(nil), names...)
	sort.Strings(sortedNames)

	required := make(map[string]bool, len(decoders))
	requireAll := publish
	for _, token := range strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ';' || r == ' ' || r == '\t' || r == '\n' || r == '\r'
	}) {
		token = strings.TrimSpace(token)
		if token == "" {
			continue
		}
		if strings.EqualFold(token, "all") {
			requireAll = true
			continue
		}
		name, ok := known[strings.ToLower(token)]
		if !ok {
			return nil, fmt.Errorf("unknown external decoder %q in %s (known: %s, all)",
				token, envBenchCorpusRequireDecoders, strings.Join(sortedNames, ", "))
		}
		required[name] = true
	}
	if requireAll {
		for _, name := range names {
			required[name] = true
		}
	}
	if raw != "" && len(required) == 0 {
		return nil, fmt.Errorf("%s did not name any external decoders", envBenchCorpusRequireDecoders)
	}
	return required, nil
}

func resolveCorpusExternalDecoders(decoders []externalDecoder, required map[string]bool, resolve func(externalDecoder) (string, bool)) (resolved []resolvedCorpusExternalDecoder, missing []string) {
	for _, dec := range decoders {
		bin, ok := resolve(dec)
		if !ok {
			if required[dec.name] {
				missing = append(missing, dec.name)
			}
			continue
		}
		resolved = append(resolved, resolvedCorpusExternalDecoder{decoder: dec, bin: bin})
	}
	return resolved, missing
}

func resolveCorpusPublishExternalDecoders(decoders []externalDecoder, required map[string]bool) ([]resolvedCorpusExternalDecoder, error) {
	resolved := make([]resolvedCorpusExternalDecoder, 0, len(required))
	for _, dec := range decoders {
		if !required[dec.name] {
			continue
		}
		binEnv, shaEnv := corpusPublishExternalDecoderPinEnvNames(dec.name)
		bin := strings.TrimSpace(os.Getenv(binEnv))
		if bin == "" {
			return nil, fmt.Errorf("set %s to the absolute %s decoder path for corpus publish", binEnv, dec.name)
		}
		if !filepath.IsAbs(bin) {
			return nil, fmt.Errorf("%s must be an absolute path, got %q", binEnv, bin)
		}
		info, err := os.Stat(bin)
		if err != nil {
			return nil, fmt.Errorf("%s %q: %w", binEnv, bin, err)
		}
		if info.IsDir() || info.Mode()&0o111 == 0 {
			return nil, fmt.Errorf("%s %q is not an executable file", binEnv, bin)
		}
		wantSHA := strings.ToLower(strings.TrimSpace(os.Getenv(shaEnv)))
		if err := validateCorpusManifestSHA256(wantSHA, shaEnv); err != nil {
			return nil, fmt.Errorf("%s: %w", shaEnv, err)
		}
		gotSHA, _, err := corpusFileSHA256(bin)
		if err != nil {
			return nil, fmt.Errorf("%s %q sha256: %w", binEnv, bin, err)
		}
		if !strings.EqualFold(gotSHA, wantSHA) {
			return nil, fmt.Errorf("%s %q sha256=%s want %s", binEnv, bin, gotSHA, wantSHA)
		}
		resolved = append(resolved, resolvedCorpusExternalDecoder{decoder: dec, bin: bin})
	}
	return resolved, nil
}

func validateCorpusPublishDecoderManifestHashes(manifest corpusPublishManifest, resolved []resolvedCorpusExternalDecoder) error {
	for _, decoder := range resolved {
		tool, ok := corpusManifestDecoderToolName(decoder.decoder.name)
		if !ok {
			continue
		}
		want := strings.TrimSpace(manifest.toolSHA256[tool])
		if want == "" {
			return fmt.Errorf("%s: publish timing decoder %s requires manifest %s_sha256 header", manifest.path, decoder.decoder.name, tool)
		}
		got, _, err := corpusFileSHA256(decoder.bin)
		if err != nil {
			return fmt.Errorf("%s timing binary %s sha256: %w", decoder.decoder.name, decoder.bin, err)
		}
		if !strings.EqualFold(got, want) {
			return fmt.Errorf("%s: publish timing decoder %s sha256=%s does not match manifest %s_sha256=%s",
				manifest.path, decoder.decoder.name, got, tool, strings.ToLower(want))
		}
	}
	return nil
}

func corpusManifestDecoderToolName(decoderName string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(decoderName)) {
	case "aomdec":
		return "aomdec", true
	case "dav1d":
		return "dav1d", true
	default:
		return "", false
	}
}

func corpusPublishExternalDecoderPinEnvNames(decoderName string) (binEnv string, shaEnv string) {
	var suffix strings.Builder
	for _, r := range decoderName {
		switch {
		case r >= 'a' && r <= 'z':
			suffix.WriteRune(r - 'a' + 'A')
		case r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			suffix.WriteRune(r)
		}
	}
	base := envBenchCorpusDecoderPrefix + suffix.String()
	return base + "_BIN", base + "_SHA256"
}

func corpusExternalDecoderLookupSummary(decoders []externalDecoder, names []string) string {
	if len(names) == 0 {
		return ""
	}
	need := make(map[string]bool, len(names))
	for _, name := range names {
		need[name] = true
	}
	var parts []string
	for _, dec := range decoders {
		if !need[dec.name] {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s=[%s]", dec.name, strings.Join(dec.lookups, ", ")))
	}
	return strings.Join(parts, "; ")
}

// TestCrossDecoderCorpus is the multi-config steady-state throughput benchmark.
// It requires the goav1_oracle build tag AND GOAV1_BENCH_CORPUS=1, and the
// generated corpus on disk (scripts/gen_bench_corpus.sh). It is skipped
// otherwise so plain test runs are unaffected.
func TestCrossDecoderCorpus(t *testing.T) {
	if os.Getenv(envBenchCorpus) != "1" {
		t.Skip("set GOAV1_BENCH_CORPUS=1 (with the generated corpus) to run the multi-config cross-decoder throughput benchmark")
	}
	publish := os.Getenv(envBenchCorpusPublish) == "1"
	reportPath := strings.TrimSpace(os.Getenv(envBenchCorpusReportJSON))
	if publish && reportPath == "" {
		t.Fatalf("cross-corpus publish: set %s to write the machine-readable benchmark sidecar", envBenchCorpusReportJSON)
	}
	if publish {
		if err := validateCorpusPublishGitClean(currentCorpusPublishGit()); err != nil {
			t.Fatalf("cross-corpus publish: %v", err)
		}
		if err := validateCorpusPublishEnvironmentNotes(os.Getenv(envBenchCorpusEnvironmentNotes)); err != nil {
			t.Fatalf("cross-corpus publish: %v", err)
		}
		if err := validateCorpusPublishEnvironment(currentCorpusPublishEnvironment()); err != nil {
			t.Fatalf("cross-corpus publish: %v", err)
		}
	}
	dir, ok := corpusDir(t)
	if !ok {
		if publish {
			t.Fatalf("cross-corpus publish: no clips in %s (regenerate with scripts/gen_bench_corpus.sh)", dir)
		}
		t.Skipf("cross-corpus: no clips in %s (regenerate with scripts/gen_bench_corpus.sh)", dir)
	}
	t.Logf("cross-corpus: corpus dir = %s", dir)

	manifest, haveManifest := loadCorpusBenchmarkManifest(t, dir, publish)
	if publish {
		if err := validateCorpusPublishSourceCoverage(manifest); err != nil {
			t.Fatalf("cross-corpus publish: %v", err)
		}
	}

	clips, failed := loadCorpusClips(t, dir)

	// Report conformance failures prominently regardless of timing.
	if len(failed) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "\n!!! CONFORMANCE BUGS: %d corpus clip(s) FAILED goav1 byte-exact decode !!!\n", len(failed))
		fmt.Fprintf(&b, "    (not timed; throughput requires a fully passing corpus)\n")
		for _, f := range failed {
			fmt.Fprintf(&b, "    - %-26s %s\n", f.name, f.reason)
		}
		t.Log(b.String())
		t.Fatalf("cross-corpus: %d corpus clip(s) failed goav1 byte-exact decode", len(failed))
	}
	if len(clips) == 0 {
		if publish {
			t.Fatal("cross-corpus publish: no usable clips")
		}
		t.Skip("cross-corpus: no usable clips")
	}
	if haveManifest {
		if err := validateCorpusPublishLoadedClips(manifest, clips); err != nil {
			if publish {
				t.Fatalf("cross-corpus publish: %v", err)
			}
			t.Fatalf("cross-corpus: valid manifest does not match loaded clips: %v", err)
		}
		t.Logf("cross-corpus: loaded %d/%d manifest clips", len(clips), manifest.expectedClips)
	}

	decoders := crossBenchExternalDecoders()
	requiredDecoders, err := corpusRequiredExternalDecoderNames(decoders, publish, os.Getenv(envBenchCorpusRequireDecoders))
	if err != nil {
		t.Fatalf("cross-corpus: %v", err)
	}
	var resolvedExternal []resolvedCorpusExternalDecoder
	if publish {
		resolvedExternal, err = resolveCorpusPublishExternalDecoders(decoders, requiredDecoders)
		if err != nil {
			t.Fatalf("cross-corpus publish: %v", err)
		}
		if err := validateCorpusPublishDecoderManifestHashes(manifest, resolvedExternal); err != nil {
			t.Fatalf("cross-corpus publish: %v", err)
		}
	} else {
		var missingExternal []string
		resolvedExternal, missingExternal = resolveCorpusExternalDecoders(decoders, requiredDecoders, func(dec externalDecoder) (string, bool) {
			return dec.resolveBinary()
		})
		if len(missingExternal) > 0 {
			t.Fatalf("cross-corpus: required external decoder(s) not found on PATH: %s (lookups: %s)",
				strings.Join(missingExternal, ", "), corpusExternalDecoderLookupSummary(decoders, missingExternal))
		}
	}
	resolvedNames := make(map[string]bool, len(resolvedExternal))
	for _, resolved := range resolvedExternal {
		resolvedNames[resolved.decoder.name] = true
	}
	for _, dec := range decoders {
		if !resolvedNames[dec.name] && !requiredDecoders[dec.name] {
			t.Logf("cross-corpus: %s not found on PATH -- skipping", dec.name)
		}
	}

	// Build decoder result slots first, then time decode jobs in a deterministic
	// clip-rotated order so thermal/load drift cannot always favor the same
	// decoder column.
	results := []decoderResult{{
		name:             "goav1",
		inProcess:        true,
		perVector:        map[string]time.Duration{},
		perVectorSamples: map[string]corpusDurationSamples{},
	}}
	timers := []corpusTimingDecoder{{
		name:       "goav1",
		inProcess:  true,
		resultSlot: 0,
	}}

	// External startup baselines are measured before the interleaved decode
	// pass; the report still prints both raw and startup-adjusted timings.
	for _, resolved := range resolvedExternal {
		dec := resolved.decoder
		bin := resolved.bin
		t.Logf("cross-corpus: %s resolved to %s", dec.name, bin)

		startupSamples, err := measureCorpusDurations(1, crossBenchRuns, func() error {
			_ = runExternal(bin, dec.startupArgs(bin))
			return nil
		})
		startup := corpusSelectedDuration(startupSamples)
		if err != nil {
			startup = 0
		}

		results = append(results, decoderResult{
			name:             dec.name,
			startup:          startup,
			startupSamples:   startupSamples,
			perVector:        map[string]time.Duration{},
			perVectorSamples: map[string]corpusDurationSamples{},
		})
		timers = append(timers, corpusTimingDecoder{
			name:       dec.name,
			external:   dec,
			bin:        bin,
			resultSlot: len(results) - 1,
		})
	}

	usable := make([]bool, len(timers))
	for i := range usable {
		usable[i] = true
	}
	for _, job := range corpusInterleavedTimingJobs(len(clips), len(timers)) {
		if !usable[job.decoderIndex] {
			continue
		}
		clip := clips[job.clipIndex]
		timer := timers[job.decoderIndex]
		var samples corpusDurationSamples
		var err error
		if timer.inProcess {
			samples, err = measureCorpusDurations(1, crossBenchRuns, func() error {
				result, err := decodeCorpusClipDiscard(clip.ivfData)
				if err == nil && result.frames != clip.frames {
					err = fmt.Errorf("decoded %d visible frames, want %d", result.frames, clip.frames)
				}
				return err
			})
			if err != nil {
				t.Fatalf("cross-corpus: goav1 timed decode error on %s (%v)", clip.name, err)
			}
		} else {
			dec := timer.external
			samples, err = measureCorpusDurations(1, crossBenchRuns, func() error {
				return runExternal(timer.bin, dec.decodeArgs(timer.bin, clip.ivfPath))
			})
			if err != nil {
				if requiredDecoders[dec.name] {
					t.Fatalf("cross-corpus: required decoder %s failed to decode %s (%v)", dec.name, clip.name, err)
				}
				t.Logf("cross-corpus: %s failed to decode %s (%v) -- excluding decoder", dec.name, clip.name, err)
				usable[job.decoderIndex] = false
				continue
			}
		}
		res := &results[timer.resultSlot]
		selected := corpusSelectedDuration(samples)
		res.perVector[clip.ivfPath] = selected
		res.perVectorSamples[clip.ivfPath] = samples
		res.totalRaw += selected
		res.totalFrames += clip.frames
	}

	filteredResults := results[:0]
	filteredTimers := timers[:0]
	for i, timer := range timers {
		if usable[i] {
			filteredResults = append(filteredResults, results[timer.resultSlot])
			filteredTimers = append(filteredTimers, timer)
		}
	}

	if publish {
		if err := writeCorpusPublishReport(reportPath, dir, manifest, clips, filteredResults, filteredTimers); err != nil {
			t.Fatalf("cross-corpus: write %s: %v", reportPath, err)
		}
		t.Logf("cross-corpus: wrote report JSON %s", reportPath)
	}

	printCorpusReport(t, clips, filteredResults)
}

func writeCorpusPublishReport(path, dir string, manifest corpusPublishManifest, clips []corpusClip, results []decoderResult, timers []corpusTimingDecoder) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("%s is empty", envBenchCorpusReportJSON)
	}
	manifestSHA := ""
	if manifest.path != "" {
		sha, _, err := corpusFileSHA256(manifest.path)
		if err != nil {
			return fmt.Errorf("manifest sha256: %w", err)
		}
		manifestSHA = sha
	}
	report := corpusPublishReport{
		GeneratedAtUTC: time.Now().UTC().Format(time.RFC3339Nano),
		Git:            currentCorpusPublishGit(),
		Environment:    currentCorpusPublishEnvironment(),
		Corpus: corpusPublishReportCorpus{
			Dir:              dir,
			Manifest:         manifest.path,
			ManifestSHA256:   manifestSHA,
			SourceID:         manifest.sourceID,
			SourceSHA256:     manifest.sourceSHA256,
			SourceURL:        manifest.sourceURL,
			SourceLicense:    manifest.sourceLicense,
			SourceCategory:   manifest.sourceCategory,
			SourceCount:      len(manifest.sources),
			SourceCategories: corpusManifestSourceCategoryCount(manifest.sources),
			Sources:          corpusPublishReportSources(manifest.sources),
			ToolVersions:     copyCorpusNonEmptyStringMap(manifest.toolVersions),
			ToolSHA256:       copyCorpusNonEmptyStringMap(manifest.toolSHA256),
			ExpectedClips:    manifest.expectedClips,
			LoadedClips:      len(clips),
			TotalFrames:      corpusTotalFrames(clips),
			TimingOrder:      "deterministic clip-rotated decoder interleave",
			RequiredDecoders: os.Getenv(envBenchCorpusRequireDecoders),
		},
		Timing: corpusPublishReportTiming{
			Runs:                   crossBenchRuns,
			WarmupRuns:             1,
			Statistic:              "median measured wall-clock selected; JSON stores every measured sample plus min, max, and IQR",
			ConcurrencyModel:       "goav1 worker pool = 1; aomdec --threads=1; dav1d --threads 1; SvtAv1DecApp --lp 1 requests SVT parallelism level 1, not a verified thread count",
			InProcessGoAV1:         true,
			ExternalStartupModel:   "raw includes subprocess startup; adjusted subtracts one measured startup baseline per clip",
			ExternalCommandTimeout: externalDecoderCommandTimeout.String(),
		},
		Tools:    corpusPublishReportTools(timers),
		Clips:    corpusPublishReportClips(clips, manifest),
		Decoders: corpusPublishReportDecoders(clips, results),
	}
	raw, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

func copyCorpusNonEmptyStringMap(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]string, len(src))
	for name, version := range src {
		if strings.TrimSpace(version) != "" {
			dst[name] = version
		}
	}
	return dst
}

func currentCorpusPublishGit() corpusPublishGit {
	var meta corpusPublishGit
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		meta.Error = err.Error()
		return meta
	}
	meta.Commit = strings.TrimSpace(string(out))
	status, err := exec.Command("git", "status", "--short").Output()
	if err != nil {
		meta.Error = err.Error()
		return meta
	}
	meta.Dirty = strings.TrimSpace(string(status)) != ""
	return meta
}

func validateCorpusPublishGitClean(git corpusPublishGit) error {
	if git.Error != "" {
		return fmt.Errorf("git metadata unavailable: %s", git.Error)
	}
	if git.Dirty {
		return errors.New("requires a clean git worktree")
	}
	return nil
}

func validateCorpusPublishEnvironmentNotes(notes string) error {
	return validateCorpusPublishTextEnvironment(envBenchCorpusEnvironmentNotes, notes)
}

func validateCorpusPublishTextEnvironment(name, value string) error {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return fmt.Errorf("set %s for corpus publish", name)
	}
	if strings.ContainsAny(trimmed, "\r\n") {
		return fmt.Errorf("set %s without newlines for corpus publish", name)
	}
	return nil
}

func validateCorpusPublishEnvironment(env corpusPublishEnvironment) error {
	setting := strings.TrimSpace(env.GOMAXPROCSEnv)
	if setting == "" {
		return errors.New("set GOMAXPROCS to a single positive integer for corpus publish")
	}
	gomaxprocs, err := strconv.Atoi(setting)
	if err != nil || gomaxprocs <= 0 {
		return fmt.Errorf("set GOMAXPROCS to a single positive integer for corpus publish, got %q", env.GOMAXPROCSEnv)
	}
	if env.GOMAXPROCS != gomaxprocs {
		return fmt.Errorf("GOMAXPROCS environment value %d does not match runtime GOMAXPROCS %d", gomaxprocs, env.GOMAXPROCS)
	}
	if strings.TrimSpace(env.GoToolError) != "" {
		return fmt.Errorf("corpus publish Go tool pin: %s", env.GoToolError)
	}
	if strings.TrimSpace(env.GoToolPath) == "" {
		return fmt.Errorf("set %s to the absolute Go executable path for corpus publish", envBenchCorpusGoBin)
	}
	if !filepath.IsAbs(env.GoToolPath) {
		return fmt.Errorf("%s must be an absolute path, got %q", envBenchCorpusGoBin, env.GoToolPath)
	}
	if strings.TrimSpace(env.GoToolExpectedSHA256) == "" {
		return fmt.Errorf("set %s to the SHA-256 of %s for corpus publish", envBenchCorpusGoSHA256, envBenchCorpusGoBin)
	}
	if !env.GoToolSHA256Verified {
		return fmt.Errorf("corpus publish Go tool sha256=%s does not match %s=%s",
			env.GoToolSHA256, envBenchCorpusGoSHA256, env.GoToolExpectedSHA256)
	}
	if strings.TrimSpace(env.GOGC) == "" {
		return errors.New("set GOGC explicitly for corpus publish")
	}
	if strings.TrimSpace(env.GOFLAGS) != "" {
		return errors.New("corpus publish requires GOFLAGS unset; use explicit go test flags")
	}
	if strings.TrimSpace(env.GOMEMLIMIT) != "" {
		return errors.New("corpus publish requires GOMEMLIMIT unset")
	}
	if strings.TrimSpace(env.GODEBUG) != "" {
		return errors.New("corpus publish requires GODEBUG unset")
	}
	if err := benchenv.ValidateCPUAffinityClaim(env.CPUAffinity, env.ObservedCPUState); err != nil {
		return err
	}
	for _, name := range benchenv.PublishBlockedGoEnvVars() {
		if strings.TrimSpace(env.BlockedGoEnv[name]) != "" {
			return fmt.Errorf("corpus publish requires %s unset; use explicit go test flags or record a separate environment", name)
		}
	}
	for _, field := range []struct {
		name  string
		value string
	}{
		{name: envBenchCorpusCPUAffinity, value: env.CPUAffinity},
		{name: envBenchCorpusPowerMode, value: env.PowerMode},
		{name: envBenchCorpusThermalState, value: env.ThermalState},
		{name: envBenchCorpusFrequencyPolicy, value: env.FrequencyPolicy},
		{name: envBenchCorpusBackgroundLoad, value: env.BackgroundLoad},
	} {
		if err := validateCorpusPublishTextEnvironment(field.name, field.value); err != nil {
			return err
		}
	}
	return nil
}

func corpusPublishGoToolMetadata() (path, expectedSHA, actualSHA string, verified bool, errText string) {
	path = strings.TrimSpace(os.Getenv(envBenchCorpusGoBin))
	expectedSHA = strings.ToLower(strings.TrimSpace(os.Getenv(envBenchCorpusGoSHA256)))
	if path == "" {
		return "", expectedSHA, "", false, fmt.Sprintf("set %s to the absolute Go executable path for corpus publish", envBenchCorpusGoBin)
	}
	if !filepath.IsAbs(path) {
		return path, expectedSHA, "", false, fmt.Sprintf("%s must be an absolute path, got %q", envBenchCorpusGoBin, path)
	}
	if err := validateCorpusManifestSHA256(expectedSHA, envBenchCorpusGoSHA256); err != nil {
		return path, expectedSHA, "", false, err.Error()
	}
	info, err := os.Stat(path)
	if err != nil {
		return path, expectedSHA, "", false, fmt.Sprintf("%s %q: %v", envBenchCorpusGoBin, path, err)
	}
	if info.IsDir() || info.Mode()&0o111 == 0 {
		return path, expectedSHA, "", false, fmt.Sprintf("%s %q is not an executable file", envBenchCorpusGoBin, path)
	}
	actualSHA, _, err = corpusFileSHA256(path)
	if err != nil {
		return path, expectedSHA, "", false, fmt.Sprintf("%s %q sha256: %v", envBenchCorpusGoBin, path, err)
	}
	return path, expectedSHA, actualSHA, actualSHA == expectedSHA, ""
}

func currentCorpusPublishEnvironment() corpusPublishEnvironment {
	hostname, _ := os.Hostname()
	goToolPath, goExpectedSHA, goSHA, goVerified, goErr := corpusPublishGoToolMetadata()
	return corpusPublishEnvironment{
		GoVersion:            runtime.Version(),
		GOOS:                 runtime.GOOS,
		GOARCH:               runtime.GOARCH,
		GOMAXPROCS:           runtime.GOMAXPROCS(0),
		GOMAXPROCSEnv:        os.Getenv("GOMAXPROCS"),
		NumCPU:               runtime.NumCPU(),
		SIMDTier:             corpusSIMDTier(),
		SIMDFeatures:         corpusSIMDFeatures(),
		BuildSettings:        corpusGoBuildSettings(),
		GoEnv:                benchenv.GoEnvForMetadataWithTool(goToolPath),
		GoToolPath:           goToolPath,
		GoToolSHA256:         goSHA,
		GoToolExpectedSHA256: goExpectedSHA,
		GoToolSHA256Verified: goVerified,
		GoToolError:          goErr,
		BlockedGoEnv:         corpusBlockedGoEnv(),
		ObservedCPUState:     benchenv.ObserveCPUState(),
		CPUModel:             detectCorpusCPUModel(),
		Hostname:             hostname,
		OSVersion:            detectCorpusOSVersion(),
		KernelVersion:        detectCorpusKernelVersion(),
		PATH:                 os.Getenv("PATH"),
		GOFLAGS:              os.Getenv("GOFLAGS"),
		GOGC:                 os.Getenv("GOGC"),
		GOMEMLIMIT:           os.Getenv("GOMEMLIMIT"),
		GODEBUG:              os.Getenv("GODEBUG"),
		CPUAffinity:          strings.TrimSpace(os.Getenv(envBenchCorpusCPUAffinity)),
		PowerMode:            strings.TrimSpace(os.Getenv(envBenchCorpusPowerMode)),
		ThermalState:         strings.TrimSpace(os.Getenv(envBenchCorpusThermalState)),
		FrequencyPolicy:      strings.TrimSpace(os.Getenv(envBenchCorpusFrequencyPolicy)),
		BackgroundLoad:       strings.TrimSpace(os.Getenv(envBenchCorpusBackgroundLoad)),
		Notes:                strings.TrimSpace(os.Getenv(envBenchCorpusEnvironmentNotes)),
	}
}

func corpusBlockedGoEnv() map[string]string {
	out := make(map[string]string)
	for _, name := range benchenv.PublishBlockedGoEnvVars() {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			out[name] = value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func corpusGoBuildSettings() map[string]string {
	info, ok := debug.ReadBuildInfo()
	if !ok || len(info.Settings) == 0 {
		return nil
	}
	out := make(map[string]string, len(info.Settings))
	for _, setting := range info.Settings {
		if setting.Key != "" {
			out[setting.Key] = setting.Value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func corpusSIMDTier() string {
	return simdTierForCorpus(cpufeatures.Detected)
}

func corpusSIMDFeatures() []string {
	return simdFeaturesForCorpus(cpufeatures.Detected)
}

func simdTierForCorpus(f cpufeatures.Features) string {
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

func simdFeaturesForCorpus(f cpufeatures.Features) []string {
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

func detectCorpusCPUModel() string {
	switch runtime.GOOS {
	case "darwin":
		return corpusFirstCommandLine("sysctl", "-n", "machdep.cpu.brand_string")
	case "linux":
		data, err := os.ReadFile("/proc/cpuinfo")
		if err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				key, value, ok := strings.Cut(line, ":")
				if !ok {
					continue
				}
				key = strings.TrimSpace(strings.ToLower(key))
				if key == "model name" || key == "hardware" {
					if value = strings.TrimSpace(value); value != "" {
						return value
					}
				}
			}
		}
	}
	return ""
}

func detectCorpusOSVersion() string {
	switch runtime.GOOS {
	case "darwin":
		return corpusFirstCommandLine("sw_vers", "-productVersion")
	case "linux":
		data, err := os.ReadFile("/etc/os-release")
		if err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				key, value, ok := strings.Cut(line, "=")
				if ok && key == "PRETTY_NAME" {
					return strings.Trim(strings.TrimSpace(value), `"`)
				}
			}
		}
	}
	return ""
}

func detectCorpusKernelVersion() string {
	return corpusFirstCommandLine("uname", "-srvm")
}

func corpusFirstCommandLine(name string, args ...string) string {
	out, err := exec.Command(name, args...).Output()
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(out), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return ""
}

func corpusTotalFrames(clips []corpusClip) int {
	total := 0
	for _, clip := range clips {
		total += clip.frames
	}
	return total
}

func corpusManifestSourceCategoryCount(sources []corpusPublishManifestSource) int {
	if len(sources) == 0 {
		return 0
	}
	categories := make(map[string]bool, len(sources))
	for _, source := range sources {
		categories[source.category] = true
	}
	return len(categories)
}

func corpusPublishReportSources(sources []corpusPublishManifestSource) []corpusPublishReportSource {
	if len(sources) == 0 {
		return nil
	}
	out := make([]corpusPublishReportSource, 0, len(sources))
	for _, source := range sources {
		out = append(out, corpusPublishReportSource{
			ID:       source.id,
			SHA256:   source.sha256,
			URL:      source.url,
			License:  source.license,
			Category: source.category,
			Clips:    source.clips,
		})
	}
	return out
}

func corpusPublishReportTools(timers []corpusTimingDecoder) []corpusPublishReportTool {
	tools := make([]corpusPublishReportTool, 0, len(timers))
	for _, timer := range timers {
		tool := corpusPublishReportTool{
			Decoder:   timer.name,
			InProcess: timer.inProcess,
			Path:      timer.bin,
		}
		if timer.inProcess {
			tools = append(tools, tool)
			continue
		}
		if sha, _, err := corpusFileSHA256(timer.bin); err == nil {
			tool.SHA256 = sha
		}
		line, versionErr := corpusCommandVersionLine(timer.bin, timer.external.versionArgsFor(timer.bin))
		tool.Version = line
		tool.VersionError = versionErr
		tools = append(tools, tool)
	}
	return tools
}

func corpusCommandVersionLine(bin string, args []string) (string, string) {
	out, err := exec.Command(bin, args...).CombinedOutput()
	line := firstCorpusNonEmptyLine(string(out))
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg == "" {
			msg = err.Error()
		}
		return line, msg
	}
	return line, ""
}

func firstCorpusNonEmptyLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			return line
		}
	}
	return ""
}

func corpusPublishReportClips(clips []corpusClip, manifest corpusPublishManifest) []corpusPublishReportClip {
	out := make([]corpusPublishReportClip, 0, len(clips))
	for _, clip := range clips {
		row, haveRow := manifest.rows[clip.name]
		cq, profile := 0, 0
		dav1dCheck, aomencArgs := "", ""
		sourceID, sourceSHA256, sourceURL, sourceLicense, sourceCategory := "", "", "", "", ""
		if haveRow {
			cq = row.cq
			profile = row.profile
			dav1dCheck = row.dav1dCheck
			aomencArgs = row.aomencArgs
			sourceID = row.sourceID
			sourceSHA256 = row.sourceSHA256
			sourceURL = row.sourceURL
			sourceLicense = row.sourceLicense
			sourceCategory = row.sourceCategory
		}
		out = append(out, corpusPublishReportClip{
			Name:           clip.name,
			Path:           clip.ivfPath,
			Frames:         clip.frames,
			Width:          clip.width,
			Height:         clip.height,
			CQ:             cq,
			BitDepth:       clip.bitDepth,
			Chroma:         clip.chroma,
			Profile:        profile,
			TileCols:       clip.tileCols,
			AllIntra:       clip.allIntra,
			SourceID:       sourceID,
			SourceSHA256:   sourceSHA256,
			SourceURL:      sourceURL,
			SourceLicense:  sourceLicense,
			SourceCategory: sourceCategory,
			DAV1DCheck:     dav1dCheck,
			AOMEncArgs:     aomencArgs,
		})
	}
	return out
}

func corpusPublishReportDecoders(clips []corpusClip, results []decoderResult) []corpusPublishReportDecoder {
	baseRaw := time.Duration(0)
	for _, result := range results {
		if result.inProcess {
			baseRaw = result.totalRaw
			break
		}
	}
	out := make([]corpusPublishReportDecoder, 0, len(results))
	for _, result := range results {
		adjusted := result.totalRaw - result.startup*time.Duration(len(clips))
		if result.inProcess {
			adjusted = result.totalRaw
		}
		if adjusted < 0 {
			adjusted = 0
		}
		row := corpusPublishReportDecoder{
			Name:             result.name,
			InProcess:        result.inProcess,
			Frames:           result.totalFrames,
			RawMS:            durationMilliseconds(result.totalRaw),
			RawFPS:           fpsOf(result.totalFrames, result.totalRaw),
			StartupMS:        durationMilliseconds(result.startup),
			StartupSamplesMS: durationListMilliseconds(result.startupSamples.Runs),
			StartupMedianMS:  durationMilliseconds(result.startupSamples.Median),
			StartupIQRMS:     durationMilliseconds(result.startupSamples.IQR),
			AdjustedMS:       durationMilliseconds(adjusted),
			AdjustedFPS:      fpsOf(result.totalFrames, adjusted),
			PerClip:          corpusPublishReportPerClipTimings(clips, result),
		}
		if baseRaw > 0 && result.totalRaw > 0 {
			row.VsGoAV1Raw = float64(baseRaw) / float64(result.totalRaw)
		}
		out = append(out, row)
	}
	return out
}

func corpusPublishReportPerClipTimings(clips []corpusClip, result decoderResult) []corpusPublishReportClipTiming {
	out := make([]corpusPublishReportClipTiming, 0, len(clips))
	for _, clip := range clips {
		raw := result.perVector[clip.ivfPath]
		samples := result.perVectorSamples[clip.ivfPath]
		adjusted := raw
		if !result.inProcess {
			adjusted = raw - result.startup
			if adjusted < 0 {
				adjusted = 0
			}
		}
		out = append(out, corpusPublishReportClipTiming{
			Clip:        clip.name,
			RawMS:       durationMilliseconds(raw),
			SamplesMS:   durationListMilliseconds(samples.Runs),
			MedianMS:    durationMilliseconds(samples.Median),
			IQRMS:       durationMilliseconds(samples.IQR),
			AdjustedMS:  durationMilliseconds(adjusted),
			FPS:         fpsOf(clip.frames, raw),
			AdjustedFPS: fpsOf(clip.frames, adjusted),
		})
	}
	return out
}

func durationMilliseconds(d time.Duration) float64 {
	return float64(d.Nanoseconds()) / 1e6
}

func durationListMilliseconds(in []time.Duration) []float64 {
	if len(in) == 0 {
		return nil
	}
	out := make([]float64, 0, len(in))
	for _, d := range in {
		out = append(out, durationMilliseconds(d))
	}
	return out
}

// printCorpusReport renders the per-clip and aggregate throughput tables.
func printCorpusReport(t *testing.T, clips []corpusClip, results []decoderResult) {
	t.Helper()
	var b strings.Builder
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "==================================================================================\n")
	fmt.Fprintf(&b, " goav1 multi-config cross-decoder throughput  (steady-state; PERF TRACKING)\n")
	fmt.Fprintf(&b, "==================================================================================\n")
	fmt.Fprintf(&b, " %s\n", corpusClipCoverageSummary(clips))
	fmt.Fprintf(&b, " median-of-%d measured samples; requested single-thread controls; full decode + post-filter; output discarded.\n", crossBenchRuns)
	fmt.Fprintf(&b, " timing order: deterministic clip-rotated decoder interleave to reduce thermal/load column bias.\n")
	fmt.Fprintf(&b, " goav1: IN-PROCESS, byte-exact verified once while loading corpus; timed path discards output.\n")
	fmt.Fprintf(&b, " others: SUBPROCESS, decode-only, output discarded; raw includes process startup,\n")
	fmt.Fprintf(&b, "         adj subtracts one measured startup baseline per invocation. At ~48 frames/clip\n")
	fmt.Fprintf(&b, "         the startup share is small, so raw≈adj (that's the point of the longer clips).\n")
	fmt.Fprintf(&b, " SVT note: SvtAv1DecApp uses --lp 1 here; --lp is SVT's parallelism level, not a verified thread count.\n")
	fmt.Fprintf(&b, "==================================================================================\n\n")

	// ---- clip manifest (config detail) ----
	fmt.Fprintf(&b, "CLIP MANIFEST\n")
	fmt.Fprintf(&b, "%-30s %-10s %6s %5s %-6s %5s %-7s\n", "clip", "res", "frames", "bits", "chroma", "tiles", "type")
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 76))
	for _, c := range clips {
		typ := "inter"
		if c.allIntra {
			typ = "intra"
		}
		fmt.Fprintf(&b, "%-30s %-10s %6d %5d %-6s %5d %-7s\n",
			truncate(c.name, 30), fmt.Sprintf("%dx%d", c.width, c.height), c.frames, c.bitDepth, c.chroma, c.tileCols, typ)
	}
	fmt.Fprintf(&b, "\n")

	var base decoderResult
	for _, r := range results {
		if r.inProcess {
			base = r
		}
	}

	// ---- aggregate ----
	fmt.Fprintf(&b, "AGGREGATE (all clips combined)\n")
	fmt.Fprintf(&b, "%-14s %7s %12s %10s %12s %10s %10s\n",
		"decoder", "frames", "raw_ms", "raw_fps", "adj_ms", "adj_fps", "vs_goav1")
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 82))
	for _, r := range results {
		rawMS := float64(r.totalRaw.Nanoseconds()) / 1e6
		rawFPS := fpsOf(r.totalFrames, r.totalRaw)
		adj := r.totalRaw - r.startup*time.Duration(len(clips))
		if adj < 0 {
			adj = 0
		}
		adjMS := float64(adj.Nanoseconds()) / 1e6
		adjFPS := fpsOf(r.totalFrames, adj)
		var vs string
		if r.inProcess {
			vs = "1.00x"
		} else if base.totalRaw > 0 && r.totalRaw > 0 {
			vs = fmt.Sprintf("%.2fx", float64(base.totalRaw)/float64(r.totalRaw))
		} else {
			vs = "n/a"
		}
		if r.inProcess {
			fmt.Fprintf(&b, "%-14s %7d %12.3f %10.1f %12s %10s %10s\n",
				r.name+"*", r.totalFrames, rawMS, rawFPS, "(in-proc)", "(in-proc)", vs)
		} else {
			fmt.Fprintf(&b, "%-14s %7d %12.3f %10.1f %12.3f %10.1f %10s\n",
				r.name, r.totalFrames, rawMS, rawFPS, adjMS, adjFPS, vs)
		}
	}
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 82))
	fmt.Fprintf(&b, "* goav1 in-process; raw == adjusted. vs_goav1 = goav1_raw / decoder_raw\n")
	fmt.Fprintf(&b, "  (>1 means that external decoder is faster than goav1 by that factor).\n")
	for _, r := range results {
		if r.inProcess {
			continue
		}
		fmt.Fprintf(&b, "  %-14s startup baseline = %.3f ms/process\n", r.name, float64(r.startup.Nanoseconds())/1e6)
	}
	fmt.Fprintf(&b, "\n")

	// ---- per-clip adjusted fps (steady-state estimate) ----
	cols := append([]decoderResult(nil), results...)
	sort.SliceStable(cols, func(i, j int) bool {
		if cols[i].inProcess != cols[j].inProcess {
			return cols[i].inProcess
		}
		return cols[i].name < cols[j].name
	})
	fmt.Fprintf(&b, "PER-CLIP fps  [external = startup-adjusted steady-state estimate]\n")
	fmt.Fprintf(&b, "%-26s %6s", "clip", "frames")
	for _, c := range cols {
		fmt.Fprintf(&b, " %12s", c.name)
	}
	fmt.Fprintf(&b, "  %12s\n", "goav1/dav1d")
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 26+7+13*len(cols)+14))
	for _, clip := range clips {
		fmt.Fprintf(&b, "%-26s %6d", truncate(clip.name, 26), clip.frames)
		var goFPS, davFPS float64
		for _, c := range cols {
			d, ok := c.perVector[clip.ivfPath]
			if !ok {
				fmt.Fprintf(&b, " %12s", "-")
				continue
			}
			eff := d
			if !c.inProcess {
				eff = d - c.startup
				if eff < 0 {
					eff = 0
				}
			}
			f := fpsOf(clip.frames, eff)
			if c.inProcess {
				goFPS = f
			}
			if c.name == "dav1d" {
				davFPS = f
			}
			fmt.Fprintf(&b, " %12.1f", f)
		}
		ratio := "-"
		if davFPS > 0 {
			ratio = fmt.Sprintf("%.2fx", goFPS/davFPS)
		}
		fmt.Fprintf(&b, "  %12s\n", ratio)
	}
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "goav1/dav1d < 1 means goav1 is slower than dav1d by that factor (the honest gap).\n")
	t.Log(b.String())
}

func corpusClipCoverageSummary(clips []corpusClip) string {
	resolutions := make(map[string]bool, len(clips))
	bitDepths := make(map[int]bool, 3)
	chromas := make(map[string]bool, 3)
	tileCols := make(map[int]bool, 2)
	types := make(map[string]bool, 2)
	for _, clip := range clips {
		resolutions[fmt.Sprintf("%dx%d", clip.width, clip.height)] = true
		bitDepths[int(clip.bitDepth)] = true
		chromas[clip.chroma] = true
		tileCols[int(clip.tileCols)] = true
		if clip.allIntra {
			types["intra"] = true
		} else {
			types["inter"] = true
		}
	}
	return fmt.Sprintf(
		"clips: %d generated (resolutions %s; bit-depths %s; chroma %s; tile-columns %s; types %s)",
		len(clips),
		joinSortedSet(resolutions),
		joinSortedIntSet(bitDepths),
		joinSortedSet(chromas),
		joinSortedIntSet(tileCols),
		joinSortedSet(types),
	)
}

func joinSortedSet(values map[string]bool) string {
	if len(values) == 0 {
		return "none"
	}
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return strings.Join(out, "/")
}

func joinSortedIntSet(values map[int]bool) string {
	if len(values) == 0 {
		return "none"
	}
	out := make([]int, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Ints(out)
	parts := make([]string, len(out))
	for i, value := range out {
		parts[i] = strconv.Itoa(value)
	}
	return strings.Join(parts, "/")
}
