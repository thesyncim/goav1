//go:build goav1_oracle

package testvector

import (
	"context"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"testing"
	"unsafe"
)

// TestLibaomExtendedFrameWorkDryRun exercises the SuiteLevelExtended cohort
// through the framework dry-run pipeline. The cohort is strictly opt-in: this
// test only runs when GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1. It is never
// part of the default fast gate or the testvectors workflow, and is intended
// to surface latent decoder gaps in:
//
//   - 10-bit streams across the full quantizer range (q=32, q=63);
//   - sub-superblock-aligned (34x34, 64x64, 66x66) and multi-superblock
//     (208x208, 226x226) frame sizes;
//   - additional SVC permutations (L2T1, L2T2) which also exercise
//     the multi-tile path (Cols=3 base, Cols=4 enhancement) - libaom
//     v3.14.0 ships no dedicated av1-1-b8-XX-tiles.ivf, so the SVC
//     vectors are the only committed multi-tile bitstreams. See
//     VectorLabelMultiTile in remote.go and the SVC L1T2 entry in
//     libaom_manifest.go for the non-SVC tile-path probe at
//     SuiteLevelRelevant.
//
// Each vector executes runLibaomFrameWorkDryRun, which gates strictly on
// frame 0's MD5 by default and logs a per-frame progress line for later
// frames. Set GOAV1_STRICT_MD5=1 to upgrade the check to every frame.
func TestLibaomExtendedFrameWorkDryRun(t *testing.T) {
	if os.Getenv("GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN") != "1" {
		t.Skip("set GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 to run the opt-in extended-vector framework dry-run")
	}
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelExtended, 0, nil)
	if len(selected) == 0 {
		t.Fatal("extended cohort selected no vectors")
	}
	for _, vector := range selected {
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
}

// TestLibaomFullFrameWorkDryRun exercises every checksum-pinned vector in the
// committed libaom manifest through the framework dry-run pipeline.
func TestLibaomFullFrameWorkDryRun(t *testing.T) {
	if os.Getenv("GOAV1_FULL_LIBAOM_FRAMEWORK_DRYRUN") != "1" {
		t.Skip("set GOAV1_FULL_LIBAOM_FRAMEWORK_DRYRUN=1 to run the opt-in full-vector framework dry-run")
	}
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelFull, 0, nil)
	if len(selected) == 0 {
		t.Fatal("full cohort selected no vectors")
	}
	ran := 0
	for _, vector := range selected {
		ran++
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
	t.Logf("full_vectors=%d", ran)
	if ran == 0 {
		t.Fatal("full cohort ran no vectors")
	}
}

type libaomFirstLumaTXB struct {
	Visit      tile.BlockVisit
	Prediction tile.BlockPredictionModeResult
	Tree       tile.TransformTreeResult

	Final [16]byte
}

func TestLibaomQuantizer00FirstLumaTXBMatchesOracle(t *testing.T) {
	got := decodeLibaomQuantizer00FirstLumaTXB(t)
	if got.Visit.Size != tile.BlockSize4x16 || got.Tree.Y != tile.TransformSize4x4 {
		t.Fatalf("first block visit=%+v tx=%+v want 4x16/4x4", got.Visit, got.Tree)
	}
	if !got.Prediction.Valid || !got.Prediction.Intra || got.Prediction.LumaMode != tile.IntraModeDC || got.Prediction.FilterIntraValid {
		t.Fatalf("first prediction=%+v want DC intra without filter-intra", got.Prediction)
	}
	wantFinal := []byte{86, 88, 90, 89, 86, 85, 88, 89, 89, 87, 87, 89, 90, 90, 91, 91}
	if !slices.Equal(got.Final[:], wantFinal) {
		t.Fatalf("final=%v want %v", got.Final, wantFinal)
	}
}

func decodeLibaomQuantizer00FirstLumaTXB(t *testing.T) libaomFirstLumaTXB {
	t.Helper()
	vector := mustLibaomRemoteVector(t, TagDecoderLibaomQuantizer00)
	ivfData := readLibaomRemoteFile(t, vector.Stream)
	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	ivfFrame, ok, err := it.Next()
	if err != nil || !ok {
		t.Fatalf("first ivf frame ok=%v err=%v", ok, err)
	}
	var stream decoder.Stream
	var events [16]decoder.Event
	count, err := stream.PushLowOverhead(ivfFrame.Payload, events[:])
	if err != nil {
		t.Fatal(err)
	}

	var refs decoder.SurfaceReferences
	var state decoder.FrameWorkState
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [parser.MaxTiles]parser.TileSpan
	var jobs [parser.MaxTiles]tile.Job
	var batches [parser.MaxTiles]threading.Batch
	var releases [parser.RefFrames]int
	var got libaomFirstLumaTXB
	seen := false

	for i := 0; i < count && !seen; i++ {
		event := events[i]
		if !eventRunsFrameWork(event) {
			continue
		}
		pool, _, backing, frameSlots, free, used := bindLibaomVectorFramePool(t, event, 8)
		_, err := state.RunEventWithContextAndPostFilter(&refs, &pool, event.SequenceHeader, &event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx decoder.FrameWorkBatch) error {
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
				afterBlock := func(visit tile.BlockLoopVisit) error {
					if seen || visit.Block.MICol != 0 || visit.Block.MIRow != 0 {
						return nil
					}
					seen = true
					plane, err := ctx.JobOutputPlane(j, threading.FrameWorkPlaneY)
					if err != nil {
						return err
					}
					got.Visit = visit.Block
					got.Prediction = visit.Prediction
					got.Tree = visit.Coefficients.Tree
					planeX := int(plane.X)
					planeY := int(plane.Y)
					planeStride := int(plane.Stride)
					for yy := 0; yy < 4; yy++ {
						row := (yy - planeY) * planeStride
						copy(got.Final[yy*4:yy*4+4], plane.Pix[row-planeX:row-planeX+4])
					}
					return nil
				}
				int32Scratch, residualScratch, err := libaomResidualScratch(ctx)
				if err != nil {
					return err
				}
				var interScratch threading.FrameWorkInterPredictionScratch
				predictionScratch := threading.FrameWorkPredictionScratch{Inter: &interScratch}
				_, err = ctx.DecodeAndReconstructJobResiduals(j, &decodeState, storage.CDFs(), &scratch, threading.FrameWorkTileResidualRequest{
					Loop:              loopReq,
					TransformMode:     ctx.TransformRef.TransformMode,
					Int32Scratch:      int32Scratch,
					ResidualScratch:   residualScratch,
					PredictionScratch: &predictionScratch,
					AfterBlock:        afterBlock,
				})
				if err != nil {
					return err
				}
				if err := ctx.RetainTileResidualCDFStorage(j, &decodeState, &storage); err != nil {
					return err
				}
			}
			return nil
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		_ = backing
		_ = frameSlots
		_ = free
		_ = used
	}
	if !seen {
		t.Fatal("first luma TXB not found")
	}
	return got
}

// libaomPostFilterParallel, when set via GOAV1_POSTFILTER_PARALLEL=<N>, drives
// the oracle dry-run's supported post-filter chain through the parallel band
// fan-out with N workers. It lets the strict-MD5 gate prove the parallel path
// matches libaom byte-for-byte on the whole corpus. It is a single shared,
// reused instance (the dry-run decodes frames serially), so it stays
// allocation-free after warm-up just like the production decoder.
var libaomPostFilterParallel = func() *decoder.FrameWorkPostFilterParallel {
	v := os.Getenv("GOAV1_POSTFILTER_PARALLEL")
	if v == "" {
		return nil
	}
	n := 0
	for _, c := range v {
		if c < '0' || c > '9' {
			return nil
		}
		n = n*10 + int(c-'0')
	}
	if n <= 1 {
		return nil
	}
	return &decoder.FrameWorkPostFilterParallel{Workers: n}
}()

func TestLibaomQuantizer00OfficialMD5Manifest(t *testing.T) {
	vector := mustLibaomRemoteVector(t, TagDecoderLibaomQuantizer00)
	md5Data := readLibaomRemoteFile(t, vector.MD5)
	digests := parseLibaomMD5Digests(t, vector.Tag, md5Data)
	if err := (Manifest{Digests: digests}).Validate(); err != nil {
		t.Fatal(err)
	}
	if len(digests) != 2 {
		t.Fatalf("digest count=%d want 2", len(digests))
	}

	oracle := NewOracle(Manifest{Digests: digests})
	for _, digest := range digests {
		if err := oracle.CheckMD5(digest.Tag, digest.FrameIndex, digest.MD5); err != nil {
			t.Fatalf("official md5 oracle frame=%d err=%v", digest.FrameIndex, err)
		}
	}
}

func TestLibaomRemoteSuiteFullDownloads(t *testing.T) {
	if os.Getenv("GOAV1_FULL_LIBAOM_VECTORS") != "1" {
		t.Skip("set GOAV1_FULL_LIBAOM_VECTORS=1 to download the full libaom vector manifest")
	}
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelFull, 0, nil)
	if len(selected) == 0 {
		t.Fatal("full libaom manifest selected no vectors")
	}
	suite, err := LibaomRemoteSuite(context.Background(), libaomCacheDir(t), SuiteLevelFull, 0)
	if err != nil {
		t.Fatal(err)
	}
	if suite.Name != manifest.Name {
		t.Fatalf("suite name=%q want %q", suite.Name, manifest.Name)
	}
	if err := suite.Manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(suite.Manifest.Vectors) != len(selected) {
		t.Fatalf("vectors=%d want %d", len(suite.Manifest.Vectors), len(selected))
	}
	if len(suite.Manifest.Digests) < len(selected) {
		t.Fatalf("digests=%d want at least %d", len(suite.Manifest.Digests), len(selected))
	}
	for _, remote := range selected {
		vector, ok := suite.Manifest.Find(remote.Tag)
		if !ok {
			t.Fatalf("missing downloaded vector tag=%x", remote.Tag)
		}
		if vector.Name != remote.Name || len(vector.Input) == 0 {
			t.Fatalf("vector tag=%x got name=%q bytes=%d", remote.Tag, vector.Name, len(vector.Input))
		}
		if _, ok := suite.Manifest.FindDigest(remote.Tag, 0); !ok {
			t.Fatalf("missing frame0 digest for tag=%x", remote.Tag)
		}
	}
}

func TestLibaomQuantizer00StreamEvents(t *testing.T) {
	vector := mustLibaomRemoteVector(t, TagDecoderLibaomQuantizer00)
	ivfData := readLibaomRemoteFile(t, vector.Stream)
	md5Data := readLibaomRemoteFile(t, vector.MD5)
	digests := parseLibaomMD5Digests(t, vector.Tag, md5Data)
	if err := (Manifest{Digests: digests}).Validate(); err != nil {
		t.Fatal(err)
	}

	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}

	var stream decoder.Stream
	var events [16]decoder.Event
	frameCount := 0
	oracle := NewOracle(Manifest{Digests: digests})
	for {
		frame, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf frame %d: %v", frameCount, err)
		}
		if !ok {
			break
		}

		count, err := stream.PushLowOverhead(frame.Payload, events[:])
		if err != nil {
			t.Fatalf("frame %d PushLowOverhead: %v", frame.Index, err)
		}
		if count == 0 {
			t.Fatalf("frame %d produced no decoder events", frame.Index)
		}

		final := false
		for i := 0; i < count; i++ {
			event := events[i]
			if !eventCompletesDecodedFrame(event) {
				continue
			}
			digestIndex := int(frame.Index)
			if digestIndex >= len(digests) {
				t.Fatalf("frame %d has no official md5 digest", frame.Index)
			}
			if digests[digestIndex].FrameIndex != frame.Index {
				t.Fatalf("frame digest index=%d want %d", digests[digestIndex].FrameIndex, frame.Index)
			}
			final = true
			if event.FrameSize.CodedWidth != 352 || event.FrameSize.Height != 288 {
				t.Fatalf("frame %d size=%dx%d want 352x288", frame.Index, event.FrameSize.CodedWidth, event.FrameSize.Height)
			}
			if event.TileGroup.TileCount == 0 || event.TileGroup.DataSize == 0 {
				t.Fatalf("frame %d tile group=%+v", frame.Index, event.TileGroup)
			}
			if err := oracle.CheckMD5(vector.Tag, frame.Index, digests[digestIndex].MD5); err != nil {
				t.Fatalf("frame %d official md5 err=%v", frame.Index, err)
			}
		}
		if !final {
			t.Fatalf("frame %d did not produce a final frame event", frame.Index)
		}
		frameCount++
	}
	if frameCount != len(digests) {
		t.Fatalf("frame count=%d want %d", frameCount, len(digests))
	}
}

func TestLibaomQuantizer00FrameWorkDryRun(t *testing.T) {
	vector := mustLibaomRemoteVector(t, TagDecoderLibaomQuantizer00)
	runLibaomFrameWorkDryRun(t, vector)
}

func TestLibaomFastFrameWorkDryRun(t *testing.T) {
	if os.Getenv("GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN") != "1" {
		t.Skip("set GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 to run the fast-vector framework dry-run")
	}
	for _, vector := range LibaomRemoteManifest().SelectRemote(SuiteLevelFast, 0, nil) {
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
}

// runLibaomFrameWorkDryRun executes the libaom framework dry-run for a single
// vector. It supports two MD5 verification modes:
//
//   - Lenient mode (default): only the first emitted output's MD5 must match
//     the official digest; subsequent outputs merely log a progress line
//     indicating whether they match. This is the behaviour required by
//     `make testvectors-fast` and the `dryrun-fast` CI workflow, which gate
//     on the currently-passing fast cohort. A vector with a correct first
//     output but broken later outputs still "passes" in this mode — the
//     per-output log makes the silent mismatch visible without failing CI.
//
//   - Strict mode (GOAV1_STRICT_MD5=1): every emitted output's MD5 must
//     match the official digest. Any mismatch fails the subtest. This mode
//     is intended for diagnostic snapshots — it surfaces vectors whose
//     later outputs are silently wrong while the first happens to be
//     correct.
//
// Frame-emit ordering. libaom's aomdec with output_all_layers=0 emits one
// MD5 line per *visible* output frame, in IVF/temporal order. A frame is
// visible when either:
//
//   - The frame OBU has show_frame=1 (regular shown frame), or
//   - A show_existing_frame OBU re-displays a previously-decoded retained
//     reference (typically an ALTREF backup that was decoded with
//     show_frame=0 / showable_frame=1).
//
// Hidden frames (show_frame=0, showable_frame=1) are decoded for reference
// but produce *no* MD5 line in libaom's manifest. They are emitted later
// via a show_existing_frame event whose MD5 matches the pixel content of
// the hidden frame at the time it was first decoded.
//
// The dry-run mirrors this by:
//
//   - Running the postfilter for every completed frame (regardless of
//     show_frame) and recording its MD5 keyed by the layer-local frame
//     surface index. This gives us a per-surface MD5 cache.
//   - For each completion event with FrameHeader.ShowFrame=true, recording
//     a per-IVF-frame "emit" for that frame's MD5.
//   - For each EventExistingFrame (show_existing_frame), recording an
//     "emit" using the cached MD5 of the referenced surface (read from
//     result.Step.ShowExisting.Surface).
//   - SVC handling preserved: if multiple spatial layers are observed
//     within one IVF frame, only the emit at the highest SpatialID is
//     surfaced (matching aomdec's output_all_layers=0 default).
//
// SVC handling: libaom's aomdec with the default output_all_layers=0 emits
// one MD5 per temporal unit at the highest-spatial-layer output. The
// dry-run mirrors that:
//
//   - Each spatial layer gets its own frame pool and frame-work state. SVC
//     streams with mixed-size spatial layers (e.g. L2T1 base 640x360 +
//     enhancement 1280x720) cannot share one frame.Pool because
//     Pool.AcquireFormat is strict about width/height. Reference slots are
//     stream-global per the AV1 spec, so a single shared SurfaceReferences
//     spans all layers; the decoder's external-references path routes
//     cross-pool reference frames through the layer aggregator's
//     FrameSurface provider, and temporal-MV metadata through the shared
//     frame-context store.
//
// In both modes, a trailing summary line is logged:
//
//	vector=NAME temporal_units=N md5_matches=M first_mismatch=F
//
// where first_mismatch is -1 if every emitted output matched.
// libaomDryRunWorkers reports the worker-lane count for the framework dry-run.
// It defaults to 1 (the historical single-lane behaviour, byte-for-byte) and can
// be raised via GOAV1_DRYRUN_WORKERS to exercise worker-count invariance of the
// parallel postfilter/reconstruction at the same strict per-frame MD5 gate.
func libaomDryRunWorkers(t *testing.T) int {
	t.Helper()
	raw := os.Getenv("GOAV1_DRYRUN_WORKERS")
	if raw == "" {
		return 1
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		t.Fatalf("GOAV1_DRYRUN_WORKERS=%q: want a positive integer", raw)
	}
	return n
}

func runLibaomFrameWorkDryRun(t *testing.T, vector RemoteVector) {
	t.Helper()
	strictMD5 := os.Getenv("GOAV1_STRICT_MD5") == "1"
	workers := libaomDryRunWorkers(t)
	ivfData := readLibaomRemoteFile(t, vector.Stream)
	md5Data := readLibaomRemoteFile(t, vector.MD5)
	digests := parseLibaomMD5Digests(t, vector.Tag, md5Data)

	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	workerPool, err := threading.NewPool(workers)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	var stream decoder.Stream
	var sideData frameWorkSideData
	layers := newLibaomSpatialLayers()
	defer layers.keepAlive()

	var events [16]decoder.Event
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [parser.MaxTiles]parser.TileSpan
	var jobs [parser.MaxTiles]tile.Job
	var batches [parser.MaxTiles]threading.Batch
	var releases [parser.RefFrames]int

	frameCount := 0
	temporalUnits := 0
	md5Matches := 0
	firstMismatch := -1
	stats := libaomFrameWorkStats{}
	for {
		ivfFrame, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf frame %d: %v", frameCount, err)
		}
		if !ok {
			break
		}
		count, err := stream.PushLowOverhead(ivfFrame.Payload, events[:])
		if err != nil {
			t.Fatalf("ivf frame %d PushLowOverhead: %v", ivfFrame.Index, err)
		}

		// Per-IVF-frame (temporal-unit) emit tracking. aomdec with the
		// default output_all_layers=0 emits one MD5 line per *visible*
		// output frame. Hidden frames (show_frame=0, showable_frame=1)
		// are decoded for reference but produce no MD5; they are
		// emitted later via a show_existing_frame event that re-displays
		// the retained reference surface.
		//
		// Within one IVF frame the stream may contain (a) several
		// decoded frames where only some have show_frame=1, and/or (b)
		// a show_existing_frame OBU that re-displays a previously
		// decoded retained reference. Each show_frame=true completion
		// and each show_existing_frame event contributes one entry to
		// ivfEmit. After all events are processed, if multiple spatial
		// layers were observed (SVC) we surface only the emit at the
		// highest SpatialID (matching aomdec output_all_layers=0).
		var ivfEmitMD5s [32]MD5
		var ivfEmitSpatial [32]int
		ivfEmitCount := 0
		maxSpatial := uint8(0)

		for i := 0; i < count; i++ {
			event := events[i]
			if !eventRunsFrameWork(event) {
				continue
			}
			layer := layers.layer(t, event)

			var (
				postMD5 MD5
				postRan bool
			)
			spatialID := event.SpatialID
			globalSurface := func(local int) int { return libaomGlobalSurfaceID(spatialID, local) }
			result, err := layer.state.RunEventWithContextAndExternalReferences(&layers.sharedRefs, &layer.pool, event.SequenceHeader, &event, 32, referenceSurfaces[:], referenceFrames[:], workers, spans[:], jobs[:], batches[:], releases[:], workerPool, layers, globalSurface, layers, &layers.sharedFrameContexts, &sideData, libaomFrameWorkBatchRunner(func(ctx decoder.FrameWorkBatch) error {
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
					jobStats, err := ctx.DecodeAndReconstructJobResiduals(j, &decodeState, storage.CDFs(), &scratch, threading.FrameWorkTileResidualRequest{
						Loop:              loopReq,
						TransformMode:     ctx.TransformRef.TransformMode,
						CDEFIndexMap:      ctx.CDEFIndexMap,
						LoopFilterMap:     ctx.LoopFilterMap,
						Restoration:       restorationReq,
						Int32Scratch:      int32Scratch,
						ResidualScratch:   residualScratch,
						PredictionScratch: &predictionScratch,
					})
					if err != nil {
						return fmt.Errorf("decode/reconstruct job %d stats=%+v: %w", j, jobStats, err)
					}
					if err := ctx.RetainTileResidualCDFStorage(j, &decodeState, &storage); err != nil {
						return err
					}
					stats.partitionReads += int(jobStats.Loop.PartitionReads)
					stats.blockDeltaPaths += int(jobStats.Loop.Blocks)
					stats.residualTXBs += int(jobStats.TXBs)
					stats.residuals += int(jobStats.Residuals)
					stats.predictions += int(jobStats.Predictions)
					if _, err := ctx.JobOutputPlane(j, threading.FrameWorkPlaneY); err != nil {
						return err
					}
					if _, err := ctx.LoopRestorationPlan(false); err != nil {
						return err
					}
					stats.tileJobs++
					if decodeState.RetainFrameContext {
						stats.retainedContexts++
					}
					if updates, err := ctx.JobUpdatesFrameContext(j); err != nil {
						return err
					} else if updates {
						stats.contextUpdateTiles++
					}
				}
				if ctx.CDEFIndexMap != nil {
					for _, read := range ctx.CDEFIndexMap.Read {
						if read {
							stats.cdefUnitsRead++
						}
					}
				}
				return nil
			}), libaomFrameWorkPostFilterRunner(func(ctx decoder.FrameWorkPostFilterContext) error {
				// When exercising the parallel post-filter path (GOAV1_POSTFILTER_PARALLEL),
				// fan the independent row bands across goroutines. The strict-MD5 gate then
				// proves the parallel path matches libaom on the whole corpus.
				if libaomPostFilterParallel != nil {
					ctx.Parallel = libaomPostFilterParallel
				}
				output, err := applySupportedPostFilters(ctx, libaomPostFilterScratchStorage)
				if err != nil {
					t.Logf("postfilter event=%s active=%v has_cdef=%v has_lf=%v has_restoration=%v",
						vector.Name, ctx.ActivePostFilters(), ctx.CDEFIndexMap != nil, ctx.LoopFilterMap != nil, ctx.RestorationFrameBuffers != nil)
					return fmt.Errorf("apply supported postfilters: %w", err)
				}
				got, err := FrameMD5(*output)
				if err != nil {
					return err
				}
				postMD5 = got
				postRan = true
				return nil
			}))
			if err != nil {
				t.Fatalf("ivf frame %d spatial=%d RunEventWithContextAndSideDataAndPostFilterRunners: %v", ivfFrame.Index, event.SpatialID, err)
			}
			if event.SpatialID > maxSpatial {
				maxSpatial = event.SpatialID
			}
			if result.Run.CompletedFrame {
				if !postRan {
					t.Fatalf("ivf frame %d spatial=%d completed without postfilter", ivfFrame.Index, event.SpatialID)
				}
				// Cache the postfilter MD5 keyed by the layer-local
				// pool surface index. This lets a subsequent
				// show_existing_frame event recover the MD5 of the
				// retained reference surface even when the original
				// frame was hidden (show_frame=0).
				surface := -1
				switch result.Step.Kind {
				case decoder.FrameWorkStepBegin:
					surface = result.Step.Begin.Surface
				case decoder.FrameWorkStepTile:
					surface = result.Step.Tile.Surface
				}
				if surface >= 0 && surface < len(layer.md5BySurface) {
					layer.md5BySurface[surface] = postMD5
					layer.md5Valid[surface] = true
				}
				// Only show_frame=true completions appear as MD5
				// lines in aomdec's manifest. Hidden frames are
				// emitted later via show_existing_frame.
				if event.FrameHeader.ShowFrame {
					if ivfEmitCount >= len(ivfEmitMD5s) {
						t.Fatalf("ivf frame %d emit buffer overflow (cap=%d)", ivfFrame.Index, len(ivfEmitMD5s))
					}
					ivfEmitMD5s[ivfEmitCount] = postMD5
					ivfEmitSpatial[ivfEmitCount] = int(event.SpatialID)
					ivfEmitCount++
				}
			} else if event.Kind == decoder.EventExistingFrame && result.Step.Kind == decoder.FrameWorkStepShowExisting {
				// show_existing_frame re-displays a retained
				// reference surface. Look up its cached MD5 from
				// when the frame was originally decoded.
				surface := result.Step.ShowExisting.Surface
				if surface < 0 || surface >= len(layer.md5BySurface) || !layer.md5Valid[surface] {
					t.Fatalf("ivf frame %d show_existing_frame surface=%d has no cached md5 (spatial=%d)", ivfFrame.Index, surface, event.SpatialID)
				}
				if ivfEmitCount >= len(ivfEmitMD5s) {
					t.Fatalf("ivf frame %d emit buffer overflow (cap=%d)", ivfFrame.Index, len(ivfEmitMD5s))
				}
				ivfEmitMD5s[ivfEmitCount] = layer.md5BySurface[surface]
				// Grain is applied only to output pictures, as dav1d's
				// output_image -> dav1d_apply_grain: the hidden frame's
				// postfilter skipped it, so the shown frame gets it now
				// from its stored parameters, seed included.
				if event.FilmGrain.Apply {
					got, err := libaomShowExistingFilmGrainMD5(&layer.pool, &event, result.Step, surface)
					if err != nil {
						t.Fatalf("ivf frame %d show_existing_frame surface=%d film grain: %v", ivfFrame.Index, surface, err)
					}
					ivfEmitMD5s[ivfEmitCount] = got
				}
				ivfEmitSpatial[ivfEmitCount] = int(event.SpatialID)
				ivfEmitCount++
			}
		}

		// Emit digest checks for this IVF frame. Non-SVC (maxSpatial==0):
		// one digest per visible emit (show_frame=true completion or
		// show_existing_frame event) in source order. SVC (maxSpatial>0):
		// surface only the emit at the highest SpatialID, matching libaom
		// aomdec with output_all_layers=0.
		var emitMD5s [32]MD5
		var emitSpatial [32]int
		emitCount := 0
		if ivfEmitCount > 0 {
			if maxSpatial == 0 {
				for i := 0; i < ivfEmitCount; i++ {
					emitMD5s[emitCount] = ivfEmitMD5s[i]
					emitSpatial[emitCount] = ivfEmitSpatial[i]
					emitCount++
				}
			} else {
				// Find the LAST emit at the highest spatial layer.
				lastIdx := -1
				for i := 0; i < ivfEmitCount; i++ {
					if uint8(ivfEmitSpatial[i]) == maxSpatial {
						lastIdx = i
					}
				}
				if lastIdx >= 0 {
					emitMD5s[emitCount] = ivfEmitMD5s[lastIdx]
					emitSpatial[emitCount] = ivfEmitSpatial[lastIdx]
					emitCount++
				}
			}
		}

		for k := 0; k < emitCount; k++ {
			if temporalUnits >= len(digests) {
				t.Fatalf("ivf frame %d temporal_unit=%d has no official digest (have %d)", ivfFrame.Index, temporalUnits, len(digests))
			}
			expected := digests[temporalUnits].MD5
			match := emitMD5s[k] == expected
			if temporalUnits == 0 && !match {
				t.Fatalf("ivf frame %d temporal_unit=0 spatial=%d md5 got=%x official=%x", ivfFrame.Index, emitSpatial[k], emitMD5s[k], expected)
			}
			if strictMD5 && !match {
				t.Fatalf("ivf frame %d temporal_unit=%d spatial=%d md5 got=%x official=%x (strict mode)", ivfFrame.Index, temporalUnits, emitSpatial[k], emitMD5s[k], expected)
			}
			if match {
				md5Matches++
			} else if firstMismatch < 0 {
				firstMismatch = temporalUnits
			}
			t.Logf("ivf=%d temporal_unit=%d spatial=%d md5 progress got=%x official=%x txbs=%d residuals=%d cdef_units=%d",
				ivfFrame.Index, temporalUnits, emitSpatial[k], emitMD5s[k], expected, stats.residualTXBs, stats.residuals, stats.cdefUnitsRead)
			temporalUnits++
		}
		frameCount++
	}
	t.Logf("vector=%s temporal_units=%d md5_matches=%d first_mismatch=%d", vector.Name, temporalUnits, md5Matches, firstMismatch)
	if temporalUnits != len(digests) {
		t.Fatalf("temporal_units=%d want %d (ivf_frames=%d)", temporalUnits, len(digests), frameCount)
	}
	if stats.tileJobs == 0 {
		t.Fatal("no tile jobs ran")
	}
	// A retained frame context requires the stream to (a) designate a
	// context_update_tile_id whose adapted CDFs refresh the frame context and
	// (b) leave CDF end-update enabled for that frame. Low-latency SVC encodes
	// (e.g. av1-1-b8-22-svc-L2T1) disable the frame-end CDF update on every
	// frame, so no tile is ever retained even though decoding is byte-exact.
	// Only assert the retain path ran when the stream actually requested a
	// frame-context refresh.
	if stats.contextUpdateTiles > 0 && stats.retainedContexts == 0 {
		t.Fatal("a context-update tile was requested but none was retained")
	}
	if stats.partitionReads == 0 {
		t.Fatal("no partition syntax was read")
	}
	if stats.blockDeltaPaths == 0 {
		t.Fatal("no block delta syntax path ran")
	}
	if stats.residualTXBs == 0 {
		t.Fatal("no residual TXBs were decoded")
	}
	if stats.residuals == 0 {
		t.Fatal("no residual blocks were reconstructed")
	}
	if stats.predictions == 0 {
		t.Fatal("no prediction blocks were written")
	}
}

// libaomFrameWorkStats accumulates the per-vector dry-run counters used to
// assert that the framework exercised tile, prediction, residual, and CDEF
// code paths at least once.
type libaomFrameWorkStats struct {
	tileJobs           int
	retainedContexts   int
	contextUpdateTiles int
	partitionReads     int
	blockDeltaPaths    int
	residualTXBs       int
	residuals          int
	predictions        int
	cdefUnitsRead      int
}

// libaomSpatialLayerState owns the dry-run frame pool and frame-work state for
// one spatial layer (SpatialID). SVC streams with mixed
// spatial-layer dimensions (e.g. L2T1/L2T2: 640x360 + 1280x720) require a
// per-layer pool because frame.Pool stores a single Format and rejects
// AcquireFormat calls whose dimensions differ from the bound format.
// Reference slots are stream-global per the AV1 spec and live on the
// containing libaomSpatialLayers' sharedRefs; cross-layer reference frames
// route through the layers FrameSurface provider and temporal-MV metadata
// through sharedFrameContexts.
type libaomSpatialLayerState struct {
	spatialID uint8
	format    frame.Format

	pool       frame.Pool
	state      decoder.FrameWorkState
	backing    []byte
	frameSlots []frame.Frame
	free       []int
	used       []bool

	// md5BySurface caches the postfilter MD5 of every completed frame
	// keyed by its layer-local pool surface index. It is updated when a
	// frame is first decoded (whether show=true or hidden) and consulted
	// by show_existing_frame events to recover the MD5 of the retained
	// reference surface they re-display. md5Valid[surface] tracks whether
	// the cached MD5 is meaningful.
	md5BySurface []MD5
	md5Valid     []bool
}

// libaomSpatialLayers indexes per-spatial-layer dry-run state by SpatialID.
// The zero value is ready to use; layers are bound lazily on first use.
//
// Reference slots are stream-global per the AV1 spec, so sharedRefs maps the
// AV1 slot index to a caller-owned "global surface ID" that encodes
// (layerID, local pool index). SVC enhancement-layer frames reference
// surfaces in lower-spatial-layer pools through this shared map, with the
// decoder consulting FrameSurface (and ReleaseFrameSurfaces) to route by
// layer.
type libaomSpatialLayers struct {
	byID       map[uint8]*libaomSpatialLayerState
	sharedRefs decoder.SurfaceReferences
	// sharedFrameContexts is the stream-global, surface-keyed entropy (CDF)
	// frame-context store. It is shared across spatial layers exactly like
	// sharedRefs because AV1 frame contexts live in the shared RefCntBuffer
	// pool, so an enhancement layer inheriting from a base-layer reference
	// (primary_ref_frame) must read the base layer's saved context.
	sharedFrameContexts decoder.SharedFrameContextStore
}

// libaomGlobalSurfaceStride bounds the per-layer pool index range in the
// shared global surface ID namespace. Pools are sized to 16 slots (see
// bindLibaomVectorFramePool), so 256 is more than enough headroom.
const libaomGlobalSurfaceStride = 256

func libaomGlobalSurfaceID(spatialID uint8, local int) int {
	if local < 0 {
		return -1
	}
	return int(spatialID)*libaomGlobalSurfaceStride + local
}

func libaomDecodeGlobalSurfaceID(id int) (uint8, int, bool) {
	if id < 0 {
		return 0, 0, false
	}
	return uint8(id / libaomGlobalSurfaceStride), id % libaomGlobalSurfaceStride, true
}

func newLibaomSpatialLayers() *libaomSpatialLayers {
	return &libaomSpatialLayers{byID: make(map[uint8]*libaomSpatialLayerState)}
}

// FrameSurface implements decoder.FrameSurfaceProvider. The shared refs map
// stores global surface IDs encoding (spatialID, local pool index); this
// method decodes the ID and pulls the *frame.Frame from the right layer's
// pool.
func (s *libaomSpatialLayers) FrameSurface(id int) (*frame.Frame, error) {
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

// ReleaseFrameSurfaces implements decoder.FrameSurfaceReleaser. Releases are
// dispatched to the pool that owns the released surface, matching the global
// ID encoding.
func (s *libaomSpatialLayers) ReleaseFrameSurfaces(ids []int) error {
	if len(ids) == 0 {
		return nil
	}
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

// layer returns the per-spatial-layer state for event.SpatialID, binding it
// (pool + MD5 cache) on first observation. The pool is sized to 16 slots
// at the format implied by the first event seen for that spatial layer,
// which is the layer's native CodedWidth x Height — matching how libaom
// configures one decoder instance per spatial layer.
func (s *libaomSpatialLayers) layer(t *testing.T, event decoder.Event) *libaomSpatialLayerState {
	t.Helper()
	id := event.SpatialID
	layer, ok := s.byID[id]
	if !ok {
		layer = &libaomSpatialLayerState{spatialID: id}
		layer.pool, layer.format, layer.backing, layer.frameSlots, layer.free, layer.used = bindLibaomVectorFramePool(t, event, 16)
		layer.md5BySurface = make([]MD5, len(layer.frameSlots))
		layer.md5Valid = make([]bool, len(layer.frameSlots))
		s.byID[id] = layer
		return layer
	}
	if got := frameFormatFromEvent(event); got != layer.format {
		t.Fatalf("spatial layer %d format=%+v want %+v (per-layer pool was bound to first-seen size)", id, got, layer.format)
	}
	return layer
}

// keepAlive defers GC of every layer's backing storage until after the
// test completes, matching the runtime.KeepAlive guarantees the original
// single-pool path provided.
func (s *libaomSpatialLayers) keepAlive() {
	for _, layer := range s.byID {
		runtime.KeepAlive(layer.backing)
		runtime.KeepAlive(layer.frameSlots)
		runtime.KeepAlive(layer.free)
		runtime.KeepAlive(layer.used)
	}
}

func eventCompletesDecodedFrame(event decoder.Event) bool {
	return (event.Kind == decoder.EventFrame || event.Kind == decoder.EventTileGroup) && event.TileGroup.Final
}

func eventRunsFrameWork(event decoder.Event) bool {
	switch event.Kind {
	case decoder.EventFrameHeader, decoder.EventFrame, decoder.EventTileGroup, decoder.EventExistingFrame:
		return true
	default:
		return false
	}
}

func libaomResidualScratch(ctx decoder.FrameWorkBatch) ([]int32, []int16, error) {
	if _, _, err := ctx.BlockQuantizer(ctx.Quantization.BaseQIdx, 0, threading.FrameWorkPlaneY); err != nil {
		return nil, nil, err
	}
	// Allocate for the largest supported residual scratch.
	int32Len, int16Len, err := reconstruct.ScratchLen(reconstruct.Block{
		Size:      transform.Size{Width: 64, Height: 64},
		Transform: transform.TypeDCTDCT,
	})
	if err != nil {
		return nil, nil, err
	}
	return make([]int32, int32Len), make([]int16, int16Len), nil
}

type libaomFrameWorkBatchRunner func(decoder.FrameWorkBatch) error

func (fn libaomFrameWorkBatchRunner) Run(ctx decoder.FrameWorkBatch) error {
	return fn(ctx)
}

type libaomFrameWorkPostFilterRunner func(decoder.FrameWorkPostFilterContext) error

func (fn libaomFrameWorkPostFilterRunner) Apply(ctx decoder.FrameWorkPostFilterContext) error {
	return fn(ctx)
}

// frameWorkSideData binds the per-frame CDEF index, loop-filter, and
// loop-restoration side data of the frame's active post-filter stages, reusing
// its buffers across frames.
type frameWorkSideData struct {
	cdefIndex          []uint8
	cdefRead           []bool
	loopFilterRecords  []threading.FrameWorkLoopFilterBlockRecord
	lfMasks            []lfmask.FilterMask
	lfLevelCache       []uint8
	restorationRecords []tile.RestorationUnitRecord
	restorationAbove   []uint16
	restorationBelow   []uint16
}

func (r *frameWorkSideData) BindFrameWorkSideData(s *decoder.FrameWorkState, b decoder.FrameWorkBatch) error {
	lf := b.LoopFilter
	lfActive := lf.LevelY[0] != 0 || lf.LevelY[1] != 0 || lf.LevelU != 0 || lf.LevelV != 0
	cdefActive := b.CDEF.Bits != 0 || b.CDEF.YStrength[0] != 0 || b.CDEF.UVStrength[0] != 0
	if cdefActive {
		_, _, n, err := b.CDEFIndexMapShape()
		if err != nil {
			return err
		}
		r.cdefIndex = sideDataBuffer(r.cdefIndex, n)
		r.cdefRead = sideDataBuffer(r.cdefRead, n)
		clear(r.cdefIndex)
		clear(r.cdefRead)
		cdefMap, err := b.BindCDEFIndexMap(r.cdefIndex, r.cdefRead)
		if err != nil {
			return err
		}
		if err := s.SetCDEFIndexMap(cdefMap); err != nil {
			return err
		}
	}
	if lfActive || cdefActive {
		cols, rows, n, err := b.LoopFilterMapShape()
		if err != nil {
			return err
		}
		r.loopFilterRecords = sideDataBuffer(r.loopFilterRecords, n)
		lfMap := threading.FrameWorkLoopFilterMap{Records: r.loopFilterRecords, Stride: uint16(cols), Rows: uint16(rows)}
		if err := s.SetLoopFilterMap(lfMap); err != nil {
			return err
		}
		if lfActive {
			sb128w, sb128h, maskCount := threading.FrameWorkLoopFilterMaskShape(cols, rows)
			r.lfMasks = sideDataBuffer(r.lfMasks, maskCount)
			r.lfLevelCache = sideDataBuffer(r.lfLevelCache, 4*threading.FrameWorkLoopFilterLevelCacheLen(cols, rows))
			levelCache := threading.FrameWorkLoopFilterLevelCacheBind(r.lfLevelCache, cols, rows)
			clear(r.lfMasks)
			clear(levelCache)
			color := b.Sequence.ColorConfig
			layout := lfmask.Layout{Mono: color.MonoChrome}
			if color.SubsamplingX {
				layout.SSHor = 1
			}
			if color.SubsamplingY {
				layout.SSVer = 1
			}
			if err := s.SetLoopFilterMasks(threading.FrameWorkLoopFilterMasks{
				Masks:      r.lfMasks,
				LevelCache: levelCache,
				Cols:       cols,
				Rows:       rows,
				SB128W:     sb128w,
				SB128H:     sb128h,
				Layout:     layout,
				HasChroma:  !color.MonoChrome,
			}); err != nil {
				return err
			}
		}
	}
	if b.Restoration.Type[0] != parser.RestorationNone || b.Restoration.Type[1] != parser.RestorationNone || b.Restoration.Type[2] != parser.RestorationNone {
		plan, err := b.RestorationFramePlan()
		if err != nil {
			return err
		}
		r.restorationRecords = sideDataBuffer(r.restorationRecords, plan.UnitRecordLen())
		r.restorationAbove = sideDataBuffer(r.restorationAbove, plan.BoundaryBufferLen())
		r.restorationBelow = sideDataBuffer(r.restorationBelow, plan.BoundaryBufferLen())
		buffers, err := b.BindRestorationFrameBuffers(r.restorationRecords, r.restorationAbove, r.restorationBelow)
		if err != nil {
			return err
		}
		if err := s.SetRestorationFrameBuffers(buffers); err != nil {
			return err
		}
	}
	return nil
}

func sideDataBuffer[T any](buf []T, n int) []T {
	if cap(buf) < n {
		return make([]T, n)
	}
	return buf[:n]
}

// applySupportedPostFilters runs the frame's remaining supported post-filter
// stages over the side data bound in ctx, with scratch from storage, and
// returns the display output.
func applySupportedPostFilters(ctx decoder.FrameWorkPostFilterContext, storage func(decoder.FrameWorkPostFilterScratchSize) decoder.FrameWorkPostFilterScratch) (*frame.Frame, error) {
	var options decoder.FrameWorkPostFilterBindOptions
	var probe decoder.FrameWorkPostFilterRequest
	if ctx.LoopFilterMap != nil {
		options.LoopFilterMap = *ctx.LoopFilterMap
		options.LoopFilterTrustedCoverage = true
		probe.LoopFilter.Map = options.LoopFilterMap
	}
	if ctx.CDEFIndexMap != nil {
		options.CDEFIndexMap = *ctx.CDEFIndexMap
		probe.CDEF.IndexMap = options.CDEFIndexMap
	}
	if ctx.RestorationFrameBuffers != nil {
		options.RestorationRecords = ctx.RestorationFrameBuffers.Records
		options.RestorationBoundaries = ctx.RestorationFrameBuffers.Boundaries
		probe.Restoration.Records = options.RestorationRecords
	}
	size, err := ctx.SupportedPostFilterScratchLen(probe)
	if err != nil {
		return nil, fmt.Errorf("supported postfilter scratch: %w", err)
	}
	req, err := size.BindRequest(options, storage(size))
	if err != nil {
		return nil, err
	}
	next, display, _, err := ctx.ApplySupportedPostFiltersForPublication(req)
	if err != nil {
		return nil, err
	}
	if err := next.RequirePublishablePostFilterOutput(); err != nil {
		return nil, err
	}
	return display, nil
}

// libaomShowExistingFilmGrainMD5 runs the film grain stage alone on a shown
// existing surface and returns the MD5 of the grained display picture. The
// in-loop filters already ran when the frame was decoded, and the retained
// reference stays untouched.
func libaomShowExistingFilmGrainMD5(pool *frame.Pool, event *decoder.Event, step decoder.FrameWorkStep, surface int) (MD5, error) {
	output, err := pool.Frame(surface)
	if err != nil {
		return MD5{}, err
	}
	ctx := decoder.FrameWorkPostFilterContext{Event: *event, Step: step, Output: output}
	ctx = ctx.WithCompletedPostFilters(decoder.FrameWorkPostFilterLoopFilter | decoder.FrameWorkPostFilterCDEF |
		decoder.FrameWorkPostFilterSuperRes | decoder.FrameWorkPostFilterLoopRestoration)
	display, err := applySupportedPostFilters(ctx, libaomPostFilterScratchStorage)
	if err != nil {
		return MD5{}, err
	}
	if display == output {
		return MD5{}, fmt.Errorf("film grain produced no display output")
	}
	return FrameMD5(*display)
}

func libaomPostFilterScratchStorage(size decoder.FrameWorkPostFilterScratchSize) decoder.FrameWorkPostFilterScratch {
	scratch := decoder.FrameWorkPostFilterScratch{
		LoopFilterEdges:    make([]decoder.FrameWorkLoopFilterPostFilterEdge, libaomMaxInt(size.LoopFilter.Edges, 0)),
		LoopFilterSchedule: make([]uint32, libaomMaxInt(size.LoopFilter.Schedule, 0)),

		CDEFDirectionGrid: make([]cdef.DirectionGrid, libaomMaxInt(size.CDEF.DirectionGrid, 0)),
		CDEFVarianceGrid:  make([]cdef.VarianceGrid, libaomMaxInt(size.CDEF.VarianceGrid, 0)),
		CDEFInput:         make([]uint16, libaomMaxInt(size.CDEF.Input, 0)),
		CDEFUnitDst:       make([]uint16, libaomMaxInt(size.CDEF.UnitDst, 0)),

		SuperResOutputFrame: make([]byte, libaomMaxInt(size.SuperRes.OutputFrame, 0)),

		RestorationData:   make([]uint16, libaomMaxInt(size.Restoration.Samples.DataLen, 0)),
		RestorationDst:    make([]uint16, libaomMaxInt(size.Restoration.Samples.DstLen, 0)),
		RestorationWiener: make([]uint16, libaomMaxInt(size.Restoration.Apply.Unit.Wiener, 0)),
		RestorationSGR:    make([]int32, libaomMaxInt(size.Restoration.Apply.Unit.SGRProj, 0)),
		RestorationAbove:  make([]uint16, libaomMaxInt(size.Restoration.Apply.Boundary.Above, 0)),
		RestorationBelow:  make([]uint16, libaomMaxInt(size.Restoration.Apply.Boundary.Below, 0)),

		RestorationPoolData:   make([]uint16, libaomMaxInt(size.Restoration.Pool.TotalData(), 0)),
		RestorationPoolWiener: make([]uint16, libaomMaxInt(size.Restoration.Pool.TotalWiener(), 0)),
		RestorationPoolSGR:    make([]int32, libaomMaxInt(size.Restoration.Pool.TotalSGR(), 0)),

		FilmGrainOutputFrame: make([]byte, libaomMaxInt(size.FilmGrain.OutputFrame, 0)),
		FilmGrainLumaGrain:   make([]int16, libaomMaxInt(size.FilmGrain.LumaGrain, 0)),
		FilmGrainLumaSamples: make([]uint16, libaomMaxInt(size.FilmGrain.LumaSamples, 0)),
	}
	for plane := 0; plane < 3; plane++ {
		scratch.CDEFSamples[plane] = make([]uint16, libaomMaxInt(size.CDEF.Samples[plane], 0))
		scratch.CDEFSampleBytes[plane] = make([]byte, libaomMaxInt(size.CDEF.SampleBytes[plane], 0))
		scratch.CDEFDst[plane] = make([]uint16, libaomMaxInt(size.CDEF.Dst[plane], 0))
		scratch.SuperResCoded[plane] = make([]uint16, libaomMaxInt(size.SuperRes.CodedSamples[plane], 0))
		scratch.SuperResOutput[plane] = make([]uint16, libaomMaxInt(size.SuperRes.OutputSamples[plane], 0))
	}
	for plane := 0; plane < 2; plane++ {
		scratch.FilmGrainChromaGrain[plane] = make([]int16, libaomMaxInt(size.FilmGrain.ChromaGrain[plane], 0))
		scratch.FilmGrainChromaSamples[plane] = make([]uint16, libaomMaxInt(size.FilmGrain.ChromaSamples[plane], 0))
	}
	return scratch
}

func libaomMaxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func bindLibaomVectorFramePool(t *testing.T, event decoder.Event, count int) (frame.Pool, frame.Format, []byte, []frame.Frame, []int, []bool) {
	t.Helper()
	format := frameFormatFromEvent(event)
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*count)
	frames := make([]frame.Frame, count)
	free := make([]int, count)
	used := make([]bool, count)
	pool, err := frame.BindPool(backing, format, frames, free, used)
	if err != nil {
		t.Fatal(err)
	}
	return pool, format, backing, frames, free, used
}

func frameFormatFromEvent(event decoder.Event) frame.Format {
	sbSizeLog2 := uint8(6)
	if event.SequenceHeader.Use128x128Superblock {
		sbSizeLog2 = 7
	}
	return frame.Format{
		Width:        int(event.FrameSize.CodedWidth),
		Height:       int(event.FrameSize.Height),
		BitDepth:     event.SequenceHeader.ColorConfig.BitDepth,
		MonoChrome:   event.SequenceHeader.ColorConfig.MonoChrome,
		SubsamplingX: event.SequenceHeader.ColorConfig.SubsamplingX,
		SubsamplingY: event.SequenceHeader.ColorConfig.SubsamplingY,
		SBSizeLog2:   sbSizeLog2,
		Align:        32,
	}
}

func parseLibaomMD5Digests(t *testing.T, tag Tag, src []byte) []FrameDigest {
	t.Helper()
	digests, err := ParseLibaomMD5Digests(tag, src)
	if err != nil {
		t.Fatal(err)
	}
	return digests
}

func mustLibaomRemoteVector(t *testing.T, tag Tag) RemoteVector {
	t.Helper()
	vector, ok := LibaomRemoteVector(tag)
	if !ok {
		t.Fatalf("missing libaom vector tag=%x", tag)
	}
	return vector
}

func readLibaomRemoteFile(t *testing.T, file RemoteFile) []byte {
	t.Helper()
	manifest := LibaomRemoteManifest()
	path, err := EnsureRemoteFile(context.Background(), libaomCacheDir(t), manifest.BaseURL, file)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func libaomCacheDir(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "testdata", "libaom"))
}

// TestLibaomRelevantSupportedFrameWorkDryRun exercises the SuiteLevelRelevant
// vectors whose active postfilters are reference-safe in the publishable output
// path.
func TestLibaomRelevantSupportedFrameWorkDryRun(t *testing.T) {
	if os.Getenv("GOAV1_RELEVANT_SUPPORTED_LIBAOM_FRAMEWORK_DRYRUN") != "1" {
		t.Skip("set GOAV1_RELEVANT_SUPPORTED_LIBAOM_FRAMEWORK_DRYRUN=1 to run the opt-in relevant-vector framework dry-run")
	}
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelRelevant, 0, nil)
	if len(selected) == 0 {
		t.Fatal("relevant cohort selected no vectors")
	}
	ran := 0
	for _, vector := range selected {
		ran++
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
	t.Logf("relevant_supported_vectors=%d", ran)
	if ran == 0 {
		t.Fatal("relevant supported cohort ran no vectors")
	}
}

func TestOracleEnabledBuildChecksManifest(t *testing.T) {
	if !OracleEnabled {
		t.Fatal("OracleEnabled=false in oracle build")
	}
	if size := unsafe.Sizeof(Oracle{}); size == 0 {
		t.Fatal("enabled oracle unexpectedly has zero size")
	}
	oracle := NewOracle(Manifest{Vectors: []Vector{{
		Tag:  TagRTPPayloadSingleOBU,
		Kind: KindRTP,
		Want: []byte{0xaa, 0xbb},
	}}, Digests: []FrameDigest{{
		Tag:        TagIVFSingleFrameAV1,
		FrameIndex: 2,
		MD5:        coreIVFSingleFrameMD5,
	}}})
	if err := oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xaa, 0xbb}); err != nil {
		t.Fatalf("CheckBytes err=%v", err)
	}
	if err := oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xaa}); !errors.Is(err, ErrMismatchedBytes) {
		t.Fatalf("mismatch err=%v want %v", err, ErrMismatchedBytes)
	}
	if err := oracle.CheckBytes(TagRTPPayloadFragmentedOBU, nil); !errors.Is(err, ErrMissingVector) {
		t.Fatalf("missing err=%v want %v", err, ErrMissingVector)
	}
	if err := oracle.CheckMD5(TagIVFSingleFrameAV1, 2, coreIVFSingleFrameMD5); err != nil {
		t.Fatalf("CheckMD5 err=%v", err)
	}
	if err := oracle.CheckMD5(TagIVFSingleFrameAV1, 2, MD5{}); !errors.Is(err, ErrMismatchedMD5) {
		t.Fatalf("digest mismatch err=%v want %v", err, ErrMismatchedMD5)
	}
	if err := oracle.CheckMD5(TagIVFSingleFrameAV1, 3, MD5{}); !errors.Is(err, ErrMissingDigest) {
		t.Fatalf("missing digest err=%v want %v", err, ErrMissingDigest)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if OracleEnabled {
			_ = oracle.CheckBytes(TagRTPPayloadSingleOBU, []byte{0xaa, 0xbb})
			_ = oracle.CheckMD5(TagIVFSingleFrameAV1, 2, coreIVFSingleFrameMD5)
		}
	})
	if allocs != 0 {
		t.Fatalf("enabled oracle check allocated: %f", allocs)
	}
}

func TestOracleEnabledBuildChecksCoreSuite(t *testing.T) {
	suite := CoreSuite()
	oracle := NewOracle(suite.Manifest)
	for _, vector := range suite.Manifest.Vectors {
		if err := oracle.CheckBytes(vector.Tag, vector.Want); err != nil {
			t.Fatalf("%s oracle err=%v", vector.Name, err)
		}
	}
	for _, digest := range suite.Manifest.Digests {
		if err := oracle.CheckMD5(digest.Tag, digest.FrameIndex, digest.MD5); err != nil {
			t.Fatalf("digest oracle err=%v", err)
		}
	}
}
