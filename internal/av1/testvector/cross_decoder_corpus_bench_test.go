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
// coding tools (all-intra vs inter GOP, single vs multi tile-column), and bit
// depths (8-bit primary plus one 10-bit). At that length steady-state decode
// dominates startup, so the goav1-vs-dav1d-vs-aomdec ratios are an honest
// single-thread throughput comparison.
//
// The corpus is NOT committed (it is large binary video). Regenerate it with
// scripts/gen_bench_corpus.sh, which scales/length-extends a small source y4m
// with ffmpeg and encodes the matrix with aomenc. The benchmark skips
// gracefully when the corpus directory is absent, so a plain `go test` (or this
// test without GOAV1_BENCH_CORPUS=1) is unaffected.
//
// METHODOLOGY (mirrors cross_decoder_bench_test.go)
//
//   - SAME WORK. goav1 runs the FULL decode INCLUDING the post-filter chain
//     (loop-filter / CDEF / loop-restoration / super-res / film-grain) for
//     every visible frame, via the exact FrameWork plumbing the oracle
//     conformance harness uses (RunEventWithContextAndExternalReferences plus
//     the libaomFrameWork* runners/scratch helpers in oracle_enabled.go).
//   - CORRECTNESS GATE / CONFORMANCE PROBE. For every clip goav1 accumulates a
//     stream MD5 over the concatenated visible-frame planes (the libaom
//     test/md5_helper.h layout: visible Y rows, then U, then V, no stride
//     padding) and compares it byte-for-byte against the sidecar .md5 produced
//     by aomdec (and cross-checked against dav1d during generation). A clip
//     whose goav1 digest does not match, or that fails to decode, is a
//     CONFORMANCE BUG: it is excluded from the timing aggregate and reported
//     prominently. This bench therefore doubles as a conformance probe on real
//     content, not just a perf tool.
//   - UNIFORM, FAIR TIMING. Every decoder is single-threaded
//     (goav1 worker pool = 1; aomdec --threads=1; dav1d --threads 1) and
//     decode-only with output discarded. Each (decoder, clip) is warmed up once
//     then run best-of-N (min wall-clock) to reject scheduler/IO noise.
//   - IN-PROCESS vs SUBPROCESS. goav1 is timed in-process (no exec/startup);
//     the C decoders are subprocesses whose raw wall-clock includes process
//     startup. We measure each external decoder's startup baseline and report
//     BOTH raw and startup-adjusted numbers, exactly like the existing bench.
//     At ~48 frames/clip the startup share is small, so raw and adjusted are
//     close -- that convergence is the point of the longer clips.
//
// Reuses externalDecoder / runExternal / minDuration / fpsOf / truncate /
// crossBenchRuns from cross_decoder_bench_test.go (same package + build tag).

import (
	"crypto/md5"
	"fmt"
	"hash"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// corpusClip is one generated benchmark clip resolved on disk together with its
// expected libaom stream MD5 and decode metadata.
type corpusClip struct {
	name     string // file stem, e.g. "p360_inter_q32"
	ivfPath  string
	ivfData  []byte
	wantMD5  MD5   // expected stream digest (from the .md5 sidecar)
	frames   int   // visible frames goav1 emitted (== external frame count)
	width    int   // coded width of the first frame
	height   int   // coded height of the first frame
	bitDepth uint8 // 8 or 10
	tileCols uint8 // tile columns of the first frame
	allIntra bool  // every frame is a keyframe
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

// loadCorpusClips discovers every *.ivf with a sibling *.md5 in dir, parses the
// expected digest, and decodes each clip ONCE through goav1 to fill in the
// frame count / dimensions / tile / bit-depth metadata AND to verify the goav1
// stream digest matches the sidecar. Clips that fail to decode or mismatch are
// returned in failed (a conformance bug) and omitted from clips.
func loadCorpusClips(t *testing.T, dir string) (clips []corpusClip, failed []corpusFailure) {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "*.ivf"))
	if err != nil {
		t.Fatalf("glob corpus: %v", err)
	}
	sort.Strings(paths)
	for _, ivfPath := range paths {
		name := strings.TrimSuffix(filepath.Base(ivfPath), ".ivf")
		md5Path := strings.TrimSuffix(ivfPath, ".ivf") + ".md5"

		md5Raw, err := os.ReadFile(md5Path)
		if err != nil {
			t.Logf("corpus: skipping %s (no .md5 sidecar: %v)", name, err)
			continue
		}
		want, err := ParseMD5Hex([]byte(strings.TrimSpace(string(md5Raw))))
		if err != nil {
			t.Logf("corpus: skipping %s (bad .md5 sidecar: %v)", name, err)
			continue
		}
		ivfData, err := os.ReadFile(ivfPath)
		if err != nil {
			t.Logf("corpus: skipping %s (read ivf: %v)", name, err)
			continue
		}

		clip := corpusClip{name: name, ivfPath: ivfPath, ivfData: ivfData, wantMD5: want}
		res, derr := decodeCorpusClip(ivfData)
		if derr != nil {
			failed = append(failed, corpusFailure{name: name, reason: fmt.Sprintf("goav1 decode error: %v", derr)})
			continue
		}
		clip.frames = res.frames
		clip.width = res.width
		clip.height = res.height
		clip.bitDepth = res.bitDepth
		clip.tileCols = res.tileCols
		clip.allIntra = res.allIntra
		if res.streamMD5 != want {
			failed = append(failed, corpusFailure{
				name:   name,
				reason: fmt.Sprintf("MD5 MISMATCH: goav1=%x want=%x (%dx%d %d-bit frames=%d)", res.streamMD5, want, res.width, res.height, res.bitDepth, res.frames),
			})
			continue
		}
		clips = append(clips, clip)
	}
	return clips, failed
}

// corpusFailure records a clip that failed the goav1 conformance check.
type corpusFailure struct {
	name   string
	reason string
}

// corpusDecodeResult is the outcome of a single in-process goav1 decode.
type corpusDecodeResult struct {
	streamMD5 MD5
	frames    int
	width     int
	height    int
	bitDepth  uint8
	tileCols  uint8
	allIntra  bool
}

// decodeCorpusClip runs the full goav1 FrameWork decode (residual + prediction
// + post-filter) single-threaded over every IVF frame, accumulating the libaom
// stream MD5 across all visible output frames. It is a RemoteVector-free
// sibling of runLibaomFrameWorkDryRun that returns the digest and metadata
// instead of comparing per-frame digests, so it works on arbitrary IVFs.
//
// The corpus is plain single-layer (non-SVC) AV1, so a single frame pool and
// motion store suffice; we still honor show_frame / show_existing_frame so the
// emitted-frame set matches aomdec/dav1d exactly.
func decodeCorpusClip(ivfData []byte) (corpusDecodeResult, error) {
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
		streamHash: md5.New(),
		workerPool: workerPool,
		layers:     newCorpusLayers(),
		allIntra:   true,
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
	state.streamHash.Sum(sum[:0])
	res.streamMD5 = sum
	res.frames = state.visibleFrames
	res.width = state.width
	res.height = state.height
	res.bitDepth = state.bitDepth
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
	streamHash hash.Hash
	workerPool *threading.Pool
	layers     *corpusSpatialLayers

	ivfFrames     int
	visibleFrames int
	width         int
	height        int
	bitDepth      uint8
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
			libaomFrameWorkSideDataRunner{},
			libaomFrameWorkBatchRunner(func(ctx decoder.FrameWorkBatch) error {
				return corpusRunTileWork(ctx, layer, layers, referenceSurfaces[:], &currentMVSurface)
			}),
			libaomFrameWorkPostFilterRunner(func(ctx decoder.FrameWorkPostFilterContext) error {
				post := decoder.FrameWorkBoundSupportedPostFilterRunner{}
				size, err := libaomSupportedPostFilterScratchLen(ctx)
				if err != nil {
					return fmt.Errorf("supported postfilter scratch: %w", err)
				}
				post.Scratch = libaomPostFilterScratchStorage(size)
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
			s.bitDepth = event.SequenceHeader.ColorConfig.BitDepth
			s.tileCols = event.TileInfo.Cols
		}

		if result.Run.CompletedFrame {
			if !postRan || postOutput == nil {
				return fmt.Errorf("spatial=%d completed without postfilter output", event.SpatialID)
			}
			if event.FrameHeader.FrameType != parser.FrameTypeKey {
				s.allIntra = false
			}
			// Cache the post-filtered surface MD5 for show_existing_frame reuse.
			surface := -1
			switch result.Step.Kind {
			case decoder.FrameWorkStepBegin:
				surface = result.Step.Begin.Surface
			case decoder.FrameWorkStepTile:
				surface = result.Step.Tile.Surface
			}
			if surface >= 0 && surface < len(layer.md5BySurface) {
				digest, err := FrameMD5(*postOutput)
				if err != nil {
					return err
				}
				layer.md5BySurface[surface] = digest
				layer.md5Valid[surface] = true
			}
			// Only show_frame=true frames are emitted by aomdec/dav1d.
			if event.FrameHeader.ShowFrame {
				if err := s.hashVisibleFrame(*postOutput); err != nil {
					return err
				}
			}
		} else if event.Kind == decoder.EventExistingFrame && result.Step.Kind == decoder.FrameWorkStepShowExisting {
			// show_existing_frame re-displays a retained reference surface; the
			// emitted pixels are the cached surface, so re-hash its plane bytes.
			surface := result.Step.ShowExisting.Surface
			f, err := layer.pool.Frame(surface)
			if err != nil {
				return fmt.Errorf("show_existing surface=%d: %w", surface, err)
			}
			if err := s.hashVisibleFrame(*f); err != nil {
				return err
			}
		}
	}
	return nil
}

// hashVisibleFrame folds one emitted frame's visible plane bytes into the
// running stream digest, matching FrameMD5's plane walk (Y, then U, then V;
// monochrome substitutes a neutral chroma plane).
func (s *corpusDecodeState) hashVisibleFrame(f frame.Frame) error {
	if err := addFrameToStreamMD5(s.streamHash, f); err != nil {
		return err
	}
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

	md5BySurface []MD5
	md5Valid     []bool
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
	layer.md5BySurface = make([]MD5, len(layer.frameSlots))
	layer.md5Valid = make([]bool, len(layer.frameSlots))
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

// TestGeneratedCorpusConformance verifies every generated corpus clip once,
// without the best-of-N timing loop from TestCrossDecoderCorpus. It is an
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
		t.Logf("generated-corpus: %-26s %dx%d %d-bit frames=%d tiles=%d md5=%x",
			clip.name, clip.width, clip.height, clip.bitDepth, clip.frames, clip.tileCols, clip.wantMD5)
	}
	t.Logf("generated-corpus: %d clips / %d visible frames passed stream-MD5 conformance", len(clips), totalFrames)
}

// TestCrossDecoderCorpus is the multi-config steady-state throughput benchmark.
// It requires the goav1_oracle build tag AND GOAV1_BENCH_CORPUS=1, and the
// generated corpus on disk (scripts/gen_bench_corpus.sh). It is skipped
// otherwise so plain test runs are unaffected.
func TestCrossDecoderCorpus(t *testing.T) {
	if os.Getenv("GOAV1_BENCH_CORPUS") != "1" {
		t.Skip("set GOAV1_BENCH_CORPUS=1 (with the generated corpus) to run the multi-config cross-decoder throughput benchmark")
	}
	dir, ok := corpusDir(t)
	if !ok {
		t.Skipf("cross-corpus: no clips in %s (regenerate with scripts/gen_bench_corpus.sh)", dir)
	}
	t.Logf("cross-corpus: corpus dir = %s", dir)

	clips, failed := loadCorpusClips(t, dir)

	// Report conformance failures prominently regardless of timing.
	if len(failed) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "\n!!! CONFORMANCE BUGS: %d corpus clip(s) FAILED goav1 byte-exact decode !!!\n", len(failed))
		fmt.Fprintf(&b, "    (excluded from timing; these are real decoder bugs on the noted config)\n")
		for _, f := range failed {
			fmt.Fprintf(&b, "    - %-26s %s\n", f.name, f.reason)
		}
		t.Log(b.String())
	}
	if len(clips) == 0 {
		if len(failed) > 0 {
			t.Fatalf("cross-corpus: every clip failed goav1 decode (%d failures) -- see conformance report above", len(failed))
		}
		t.Skip("cross-corpus: no usable clips")
	}

	// ----- goav1 (in-process, full decode + post-filter, MD5-verified) -----
	goav1 := decoderResult{name: "goav1", inProcess: true, perVector: map[string]time.Duration{}}
	for _, clip := range clips {
		clip := clip
		best, err := minDuration(1, crossBenchRuns, func() error {
			_, err := decodeCorpusClip(clip.ivfData)
			return err
		})
		if err != nil {
			// Should not happen (loadCorpusClips already decoded it), but treat
			// as a conformance failure rather than crashing the bench.
			failed = append(failed, corpusFailure{name: clip.name, reason: fmt.Sprintf("goav1 timed decode error: %v", err)})
			continue
		}
		goav1.perVector[clip.ivfPath] = best
		goav1.totalRaw += best
		goav1.totalFrames += clip.frames
	}
	results := []decoderResult{goav1}

	// ----- external reference decoders -----
	for _, dec := range crossBenchExternalDecoders() {
		bin, ok := dec.resolveBinary()
		if !ok {
			t.Logf("cross-corpus: %s not found on PATH — skipping", dec.name)
			continue
		}
		t.Logf("cross-corpus: %s resolved to %s", dec.name, bin)

		startup, err := minDuration(1, crossBenchRuns, func() error {
			_ = runExternal(bin, dec.startupArgs(bin))
			return nil
		})
		if err != nil {
			startup = 0
		}

		res := decoderResult{name: dec.name, startup: startup, perVector: map[string]time.Duration{}}
		usable := true
		for _, clip := range clips {
			clip := clip
			best, err := minDuration(1, crossBenchRuns, func() error {
				return runExternal(bin, dec.decodeArgs(bin, clip.ivfPath))
			})
			if err != nil {
				t.Logf("cross-corpus: %s failed to decode %s (%v) — excluding decoder", dec.name, clip.name, err)
				usable = false
				break
			}
			res.perVector[clip.ivfPath] = best
			res.totalRaw += best
			res.totalFrames += clip.frames
		}
		if usable {
			results = append(results, res)
		}
	}

	printCorpusReport(t, clips, results)
}

// printCorpusReport renders the per-clip and aggregate throughput tables.
func printCorpusReport(t *testing.T, clips []corpusClip, results []decoderResult) {
	t.Helper()
	var b strings.Builder
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "==================================================================================\n")
	fmt.Fprintf(&b, " goav1 multi-config cross-decoder throughput  (steady-state; PERF TRACKING)\n")
	fmt.Fprintf(&b, "==================================================================================\n")
	fmt.Fprintf(&b, " clips: %d generated (256x144/640x360/1280x720; cq 20/32/55; intra/inter; tiles; 8/10-bit)\n", len(clips))
	fmt.Fprintf(&b, " best-of-%d (min wall-clock); single-thread; full decode + post-filter.\n", crossBenchRuns)
	fmt.Fprintf(&b, " goav1: IN-PROCESS, byte-exact verified (stream MD5 == aomdec/dav1d).\n")
	fmt.Fprintf(&b, " others: SUBPROCESS, decode-only, output discarded; raw includes process startup,\n")
	fmt.Fprintf(&b, "         adj subtracts one measured startup baseline per invocation. At ~48 frames/clip\n")
	fmt.Fprintf(&b, "         the startup share is small, so raw≈adj (that's the point of the longer clips).\n")
	fmt.Fprintf(&b, "==================================================================================\n\n")

	// ---- clip manifest (config detail) ----
	fmt.Fprintf(&b, "CLIP MANIFEST\n")
	fmt.Fprintf(&b, "%-26s %-10s %6s %5s %5s %-7s\n", "clip", "res", "frames", "bits", "tiles", "type")
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 64))
	for _, c := range clips {
		typ := "inter"
		if c.allIntra {
			typ = "intra"
		}
		fmt.Fprintf(&b, "%-26s %-10s %6d %5d %5d %-7s\n",
			truncate(c.name, 26), fmt.Sprintf("%dx%d", c.width, c.height), c.frames, c.bitDepth, c.tileCols, typ)
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
