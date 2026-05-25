//go:build goav1_oracle

package testvector

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

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
		t.Skip("set GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 to run the in-progress fast-vector framework dry-run")
	}
	for _, vector := range LibaomRemoteManifest().SelectRemote(SuiteLevelFast, 0, nil) {
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
}

func runLibaomFrameWorkDryRun(t *testing.T, vector RemoteVector) {
	t.Helper()
	ivfData := readLibaomRemoteFile(t, vector.Stream)
	md5Data := readLibaomRemoteFile(t, vector.MD5)
	digests := parseLibaomMD5Digests(t, vector.Tag, md5Data)

	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}
	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	var stream decoder.Stream
	var refs decoder.SurfaceReferences
	var state decoder.FrameWorkState
	var pool frame.Pool
	var poolFormat frame.Format
	var havePool bool
	var backing []byte
	var frameSlots []frame.Frame
	var free []int
	var used []bool
	var mvEntryBacking []tile.ReferenceMVEntry
	var temporalEntryBacking []tile.TemporalMotionEntry
	var mvFrames []tile.ReferenceMVFrame
	var mvStore []tile.TemporalMotionReferenceFrame
	var mvLength int

	var events [16]decoder.Event
	var referenceSurfaces [parser.InterRefsPerFrame]int
	var referenceFrames [parser.InterRefsPerFrame]*frame.Frame
	var spans [parser.MaxTiles]parser.TileSpan
	var jobs [parser.MaxTiles]tile.Job
	var batches [parser.MaxTiles]threading.Batch
	var releases [parser.RefFrames]int

	frameCount := 0
	completed := 0
	tileJobs := 0
	retainedContexts := 0
	partitionReads := 0
	blockPrefixReads := 0
	blockDeltaPaths := 0
	predictionModeReads := 0
	intraModeReads := 0
	residualTXBs := 0
	residuals := 0
	predictions := 0
	cdefUnitsRead := 0
	temporalReferenceResolves := 0
	temporalMotionProjections := 0
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
			t.Fatalf("frame %d PushLowOverhead: %v", ivfFrame.Index, err)
		}
		for i := 0; i < count; i++ {
			event := events[i]
			if !eventRunsFrameWork(event) {
				continue
			}
			if !havePool {
				pool, poolFormat, backing, frameSlots, free, used = bindLibaomVectorFramePool(t, event, 8)
				mvEntryBacking, temporalEntryBacking, mvFrames, mvStore, mvLength = bindLibaomVectorMotionStore(t, event, len(frameSlots))
				havePool = true
			} else {
				gotFormat := frameFormatFromEvent(event)
				if gotFormat != poolFormat {
					t.Fatalf("frame %d format=%+v want %+v", ivfFrame.Index, gotFormat, poolFormat)
				}
			}

			var postMD5 MD5
			postRan := false
			currentMVSurface := -1
			result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, libaomFrameWorkSideDataRunner{}, libaomFrameWorkBatchRunner(func(ctx decoder.FrameWorkBatch) error {
				surface, err := ctx.Surface()
				if err != nil {
					return err
				}
				if surface >= len(mvFrames) || mvLength == 0 {
					return decoder.ErrInvalidSurfaceReference
				}
				if currentMVSurface != surface || mvFrames[surface].Entries == nil {
					first := surface * mvLength
					currentMVFrame, err := ctx.BindReferenceMVFrame(mvEntryBacking[first : first+mvLength])
					if err != nil {
						return err
					}
					mvFrames[surface] = currentMVFrame
					currentMVSurface = surface
				}
				ctx.CurrentMVFrame = &mvFrames[surface]
				if ctx.TileInfo.UseRefFrameMVS {
					temporalMVs, err := ctx.BindTemporalMotionField(temporalEntryBacking)
					if err != nil {
						return err
					}
					ctx.TemporalMVs = &temporalMVs
					resolved, err := decoder.ResolveTemporalMotionReferences(referenceSurfaces[:len(ctx.References)], mvStore, ctx.ReferenceMVs[:])
					if err != nil {
						return err
					}
					if resolved != len(ctx.References) {
						return decoder.ErrInvalidSurfaceReference
					}
					setupStats, err := ctx.SetupTemporalMotionField()
					if err != nil {
						return err
					}
					temporalReferenceResolves += resolved
					temporalMotionProjections += setupStats.Projections
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
					stats, err := ctx.DecodeAndReconstructJobResiduals(j, &decodeState, storage.CDFs(), &scratch, threading.FrameWorkTileResidualRequest{
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
					})
					if err != nil {
						return fmt.Errorf("decode/reconstruct job %d stats=%+v: %w", j, stats, err)
					}
					if err := ctx.RetainTileResidualCDFStorage(j, &decodeState, &storage); err != nil {
						return err
					}
					partitionReads += stats.Loop.PartitionReads
					blockPrefixReads += stats.Loop.Prefixes
					blockDeltaPaths += stats.Loop.Blocks
					predictionModeReads += stats.Loop.PredictionModes
					intraModeReads += stats.Loop.IntraModes
					residualTXBs += stats.TXBs
					residuals += stats.Residuals
					predictions += stats.Predictions
					if _, err := ctx.JobOutputPlane(j, threading.FrameWorkPlaneY); err != nil {
						return err
					}
					if _, err := ctx.LoopRestorationPlan(false); err != nil {
						return err
					}
					tileJobs++
					if decodeState.RetainFrameContext {
						retainedContexts++
					}
				}
				if ctx.CDEFIndexMap != nil {
					for _, read := range ctx.CDEFIndexMap.Read {
						if read {
							cdefUnitsRead++
						}
					}
				}
				return nil
			}), libaomFrameWorkPostFilterRunner(func(ctx decoder.FrameWorkPostFilterContext) error {
				post := decoder.FrameWorkBoundSupportedPostFilterRunner{}
				size, err := libaomSupportedPostFilterScratchLen(ctx)
				if err != nil {
					return fmt.Errorf("supported postfilter scratch: %w", err)
				}
				post.Scratch = libaomPostFilterScratchStorage(size)
				if err := post.Apply(ctx); err != nil {
					t.Logf("postfilter event=%s active=%v supported_size=%+v has_cdef=%v has_lf=%v has_restoration=%v",
						vector.Name, ctx.ActivePostFilters(), size, ctx.CDEFIndexMap != nil, ctx.LoopFilterMap != nil, ctx.RestorationFrameBuffers != nil)
					return fmt.Errorf("apply supported postfilters: %w", err)
				}
				if currentMVSurface >= 0 {
					if err := decoder.PublishTemporalMotionReference(post.Context.Event, currentMVSurface, &mvFrames[currentMVSurface], mvStore); err != nil {
						return err
					}
				}
				digestIndex := int(ivfFrame.Index)
				if digestIndex >= len(digests) || digests[digestIndex].FrameIndex != ivfFrame.Index {
					t.Fatalf("frame %d missing official digest", ivfFrame.Index)
				}
				got, err := FrameMD5(*post.Context.Output)
				if err != nil {
					return err
				}
				if ivfFrame.Index == 0 && got != digests[digestIndex].MD5 {
					return fmt.Errorf("frame 0 md5 got=%x official=%x", got, digests[digestIndex].MD5)
				}
				postMD5 = got
				postRan = true
				return nil
			}))
			if err != nil {
				t.Fatalf("frame %d RunEventWithContextAndPostFilter: %v", ivfFrame.Index, err)
			}
			if result.Run.CompletedFrame {
				if !postRan {
					t.Fatalf("frame %d completed without postfilter", ivfFrame.Index)
				}
				digestIndex := int(ivfFrame.Index)
				if digestIndex >= len(digests) {
					t.Fatalf("frame %d missing official digest", ivfFrame.Index)
				}
				t.Logf("frame %d md5 progress got=%x official=%x txbs=%d residuals=%d cdef_units=%d mfmv_refs=%d mfmv_projections=%d",
					ivfFrame.Index, postMD5, digests[digestIndex].MD5, residualTXBs, residuals, cdefUnitsRead, temporalReferenceResolves, temporalMotionProjections)
				completed++
			}
		}
		frameCount++
	}
	if frameCount != len(digests) || completed != len(digests) {
		t.Fatalf("frames=%d completed=%d want %d", frameCount, completed, len(digests))
	}
	if tileJobs == 0 {
		t.Fatal("no tile jobs ran")
	}
	if retainedContexts == 0 {
		t.Fatal("no context-update tile was retained")
	}
	if partitionReads == 0 {
		t.Fatal("no partition syntax was read")
	}
	if blockPrefixReads == 0 {
		t.Fatal("no block prefix syntax was read")
	}
	if blockDeltaPaths == 0 {
		t.Fatal("no block delta syntax path ran")
	}
	if predictionModeReads == 0 {
		t.Fatal("no prediction mode syntax was read")
	}
	if intraModeReads == 0 {
		t.Fatal("no intra mode syntax was read")
	}
	if residualTXBs == 0 {
		t.Fatal("no residual TXBs were decoded")
	}
	if residuals == 0 {
		t.Fatal("no residual blocks were reconstructed")
	}
	if predictions == 0 {
		t.Fatal("no prediction blocks were written")
	}
	runtime.KeepAlive(backing)
	runtime.KeepAlive(frameSlots)
	runtime.KeepAlive(free)
	runtime.KeepAlive(used)
	runtime.KeepAlive(mvEntryBacking)
	runtime.KeepAlive(temporalEntryBacking)
	runtime.KeepAlive(mvFrames)
	runtime.KeepAlive(mvStore)
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
	q, _, err := ctx.BlockQuantizer(ctx.Quantization.BaseQIdx, 0, threading.FrameWorkPlaneY)
	if err != nil {
		return nil, nil, err
	}
	// Allocate for the largest supported residual scratch; each decoded TXB
	// still carries its own lossless decision through ReconstructBlockCoeff.
	int32Len, int16Len, err := reconstruct.ScratchLen(reconstruct.Block{
		Size:      transform.Size{Width: 64, Height: 64},
		Transform: transform.TypeDCTDCT,
		Quantizer: q,
		Lossless:  false,
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

type libaomFrameWorkSideDataRunner struct{}

func (libaomFrameWorkSideDataRunner) BindFrameWorkSideData(state *decoder.FrameWorkState, ctx decoder.FrameWorkBatch) error {
	size, err := decoder.FrameWorkSideDataScratchLen(ctx)
	if err != nil {
		return err
	}
	side, err := size.BindRunner(libaomFrameWorkSideDataScratch(size))
	if err != nil {
		return err
	}
	return side.BindFrameWorkSideData(state, ctx)
}

func libaomFrameWorkSideDataScratch(size decoder.FrameWorkSideDataScratchSize) decoder.FrameWorkSideDataScratch {
	return decoder.FrameWorkSideDataScratch{
		CDEFIndex:          make([]uint8, libaomMaxInt(size.CDEF, 0)),
		CDEFRead:           make([]bool, libaomMaxInt(size.CDEF, 0)),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, libaomMaxInt(size.LoopFilterRecords, 0)),
		RestorationRecords: make([]tile.RestorationUnitRecord, libaomMaxInt(size.RestorationRecords, 0)),
		RestorationAbove:   make([]uint16, libaomMaxInt(size.RestorationBoundary, 0)),
		RestorationBelow:   make([]uint16, libaomMaxInt(size.RestorationBoundary, 0)),
	}
}

func libaomSupportedPostFilterScratchLen(ctx decoder.FrameWorkPostFilterContext) (decoder.FrameWorkPostFilterScratchSize, error) {
	var probe decoder.FrameWorkPostFilterRequest
	if ctx.LoopFilterMap != nil {
		probe.LoopFilter.Map = *ctx.LoopFilterMap
	}
	if ctx.CDEFIndexMap != nil {
		probe.CDEF.IndexMap = *ctx.CDEFIndexMap
	}
	if ctx.RestorationFrameBuffers != nil {
		probe.Restoration.Records = ctx.RestorationFrameBuffers.Records
	}
	return ctx.SupportedPostFilterScratchLen(probe)
}

func libaomPostFilterScratchStorage(size decoder.FrameWorkPostFilterScratchSize) decoder.FrameWorkPostFilterScratch {
	scratch := decoder.FrameWorkPostFilterScratch{
		LoopFilterEdges: make([]decoder.FrameWorkLoopFilterPostFilterEdge, libaomMaxInt(size.LoopFilter.Edges, 0)),

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

		FilmGrainLumaGrain:   make([]int16, libaomMaxInt(size.FilmGrain.LumaGrain, 0)),
		FilmGrainLumaSamples: make([]uint16, libaomMaxInt(size.FilmGrain.LumaSamples, 0)),
	}
	for plane := 0; plane < 3; plane++ {
		scratch.CDEFSamples[plane] = make([]uint16, libaomMaxInt(size.CDEF.Samples[plane], 0))
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

func bindLibaomVectorMotionStore(t *testing.T, event decoder.Event, count int) ([]tile.ReferenceMVEntry, []tile.TemporalMotionEntry, []tile.ReferenceMVFrame, []tile.TemporalMotionReferenceFrame, int) {
	t.Helper()
	miCols := libaomVectorMIExtent(event.FrameSize.CodedWidth)
	miRows := libaomVectorMIExtent(event.FrameSize.Height)
	length, err := tile.ReferenceMVFrameEntries(miRows, miCols)
	if err != nil {
		t.Fatal(err)
	}
	return make([]tile.ReferenceMVEntry, count*length),
		make([]tile.TemporalMotionEntry, length),
		make([]tile.ReferenceMVFrame, count),
		make([]tile.TemporalMotionReferenceFrame, count),
		length
}

func libaomVectorMIExtent(pixels uint32) uint32 {
	return ((pixels + 7) >> 3) << 1
}

func frameFormatFromEvent(event decoder.Event) frame.Format {
	return frame.Format{
		Width:        int(event.FrameSize.CodedWidth),
		Height:       int(event.FrameSize.Height),
		BitDepth:     event.SequenceHeader.ColorConfig.BitDepth,
		MonoChrome:   event.SequenceHeader.ColorConfig.MonoChrome,
		SubsamplingX: event.SequenceHeader.ColorConfig.SubsamplingX,
		SubsamplingY: event.SequenceHeader.ColorConfig.SubsamplingY,
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
