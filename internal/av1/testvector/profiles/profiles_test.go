//go:build goav1_oracle

// Package profiles holds the libaom profile-conformance regression test in its
// OWN test binary, fully isolated from the internal/av1/testvector oracle
// suite. Because it compiles into a different package (and therefore a
// different test binary) than internal/av1/testvector, it CANNOT share process
// state with TestLibaomFastFrameWorkDryRun or any other oracle test, so a
// decode here can never starve a fast-suite subtest of a reference surface.
//
// Each clip is decoded through a brand-new, fully local decoder: its own frame
// pool, frame-work state, motion store, surface-reference map, and scratch.
// Nothing is package-level and nothing is reused between clips, so the test is
// self-isolating even within its own binary.
//
// The decode path mirrors the byte-exact framework dry-run harness in
// internal/av1/testvector (runLibaomFrameWorkDryRun): residual decode +
// reconstruction + the full supported post-filter chain, with the per-block
// loop request configured to read intra/inter modes, motion vectors, motion
// modes, inter-intra, and compound blend. This is the path proven byte-exact
// against libaom. (The throughput-oriented public stream-runner path used by
// cmd/aom-go-dec is NOT yet byte-exact and is unsuitable for a conformance
// gate.)
//
// The clips are 64x64 all-intra (kf-max-dist=1) 3-frame profile-conformance
// bitstreams committed to git under internal/av1/testvector/testdata/profiles/.
// libaom's published AV1 test-data ships no 4:4:4 (profile 1), 4:2:2
// (profile 2) or 12-bit (profile 2) vectors, so these clips guard those intra
// decode paths.
//
// Regen recipe (run from a libaom build tree, e.g. /tmp/aom-build):
//
//	# 4:4:4 8-bit:
//	aomenc --i444 --width=64 --height=64 --limit=3 --ivf --profile=1 \
//	  --cpu-used=1 --end-usage=q --cq-level=32 --kf-max-dist=1 \
//	  --lag-in-frames=0 -o profile1-444-8bit-64x64.ivf src444.yuv
//	# 4:2:2 8-bit:
//	aomenc --i422 ... --profile=2 ... -o profile2-422-8bit-64x64.ivf src422.yuv
//	# 4:2:0 12-bit:
//	aomenc --i420 ... --profile=2 --bit-depth=12 --input-bit-depth=12 \
//	  --cq-level=40 ... -o profile2-420-12bit-64x64.ivf src420_12.yuv
//	# Golden per-frame MD5s come from `aomdec --rawvideo` (libaom
//	# test/md5_helper.h layout, which testvector.FrameMD5 reproduces).
package profiles

import (
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
	"github.com/thesyncim/goav1/internal/av1/testvector"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

type profileClip struct {
	name string
	file string

	frameMD5Hex []string

	wantBitDepth     uint8
	wantSubsamplingX bool
	wantSubsamplingY bool
}

var profileClips = []profileClip{
	{
		// Profile 1: 4:4:4 8-bit. SubsamplingX=false SubsamplingY=false.
		name: "profile1-444-8bit-64x64",
		file: "profile1-444-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"00211cdc8f799c808849c955a318a0f5",
			"397ff01920ff514bc611ab49d76371c1",
			"f8fbfb25a42da47a7adb71510de9b178",
		},
		wantBitDepth:     8,
		wantSubsamplingX: false,
		wantSubsamplingY: false,
	},
	{
		// Profile 2: 4:2:2 8-bit. SubsamplingX=true SubsamplingY=false.
		name: "profile2-422-8bit-64x64",
		file: "profile2-422-8bit-64x64.ivf",
		frameMD5Hex: []string{
			"dabd492413632a810adeaf4e5d0c6d97",
			"eb6cf8d1d4d644686cf03c513acd5978",
			"84983e98afeef6692448e98fc5980431",
		},
		wantBitDepth:     8,
		wantSubsamplingX: true,
		wantSubsamplingY: false,
	},
	{
		// Profile 2: 4:2:0 12-bit. 2 bytes/sample, BitDepth=12.
		name: "profile2-420-12bit-64x64",
		file: "profile2-420-12bit-64x64.ivf",
		frameMD5Hex: []string{
			"e714741e4ad4fce5a4469c79705f132c",
			"447103c7d7358e4cbb6f5b98ce4e1be1",
			"dcd86cbbf80f81d9699eaf6c6e879e72",
		},
		wantBitDepth:     12,
		wantSubsamplingX: true,
		wantSubsamplingY: true,
	},
}

// TestProfileVendoredClips decodes each vendored profile-conformance clip
// through the byte-exact framework pipeline and asserts every visible frame
// matches its libaom aomdec golden MD5. Each clip gets a brand-new decoder
// with fully local state.
func TestProfileVendoredClips(t *testing.T) {
	for _, clip := range profileClips {
		clip := clip
		t.Run(clip.name, func(t *testing.T) {
			runProfileClip(t, clip)
		})
	}
}

func runProfileClip(t *testing.T, clip profileClip) {
	t.Helper()

	wantDigests := make([]testvector.MD5, len(clip.frameMD5Hex))
	for i, hexStr := range clip.frameMD5Hex {
		md5, err := testvector.ParseMD5Hex([]byte(hexStr))
		if err != nil {
			t.Fatalf("parse golden md5[%d]=%q: %v", i, hexStr, err)
		}
		wantDigests[i] = md5
	}

	ivfData := readProfileClip(t, clip.file)
	it, err := ivf.NewIterator(ivfData)
	if err != nil {
		t.Fatalf("NewIterator: %v", err)
	}

	workerPool, err := threading.NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	defer workerPool.Close()

	// All decoder state below is local to this call; nothing is shared with
	// any other clip, test, or package.
	var stream decoder.Stream
	var refs decoder.SurfaceReferences
	var state decoder.FrameWorkState
	var pool frame.Pool
	var backing []byte
	var frameSlots []frame.Frame
	var free []int
	var used []bool
	var havePool bool
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

	// keepAlive prevents the backing arenas from being GC'd while the pool's
	// *frame.Frame pointers are live across iterations.
	defer func() {
		runtime.KeepAlive(backing)
		runtime.KeepAlive(frameSlots)
		runtime.KeepAlive(free)
		runtime.KeepAlive(used)
		runtime.KeepAlive(mvEntryBacking)
		runtime.KeepAlive(temporalEntryBacking)
		runtime.KeepAlive(mvFrames)
		runtime.KeepAlive(mvStore)
	}()

	emitted := 0
	checkedColorConfig := false
	for {
		ivfFrame, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf frame %d: %v", emitted, err)
		}
		if !ok {
			break
		}
		count, err := stream.PushLowOverhead(ivfFrame.Payload, events[:])
		if err != nil {
			t.Fatalf("ivf frame %d PushLowOverhead: %v", ivfFrame.Index, err)
		}
		for i := 0; i < count; i++ {
			event := events[i]
			if !eventRunsFrameWork(event) {
				continue
			}
			if !checkedColorConfig {
				cc := event.SequenceHeader.ColorConfig
				if cc.BitDepth != clip.wantBitDepth {
					t.Fatalf("BitDepth=%d want %d", cc.BitDepth, clip.wantBitDepth)
				}
				if cc.MonoChrome {
					t.Fatalf("MonoChrome=true want false")
				}
				if cc.SubsamplingX != clip.wantSubsamplingX || cc.SubsamplingY != clip.wantSubsamplingY {
					t.Fatalf("subsampling=(%v,%v) want (%v,%v)",
						cc.SubsamplingX, cc.SubsamplingY, clip.wantSubsamplingX, clip.wantSubsamplingY)
				}
				checkedColorConfig = true
			}
			if !havePool {
				pool, backing, frameSlots, free, used = bindFramePool(t, event, 8)
				mvEntryBacking, temporalEntryBacking, mvFrames, mvStore, mvLength = bindMotionStore(t, event, 8)
				havePool = true
			}

			var postMD5 testvector.MD5
			havePost := false
			currentMVSurface := -1
			result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, sideDataRunner{}, batchRunner(func(ctx decoder.FrameWorkBatch) error {
				surface, err := ctx.Surface()
				if err != nil {
					return err
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
					int32Scratch, residualScratch, err := residualScratch(ctx)
					if err != nil {
						return err
					}
					var interScratch threading.FrameWorkInterPredictionScratch
					predictionScratch := threading.FrameWorkPredictionScratch{Inter: &interScratch}
					_, err = ctx.DecodeAndReconstructJobResiduals(j, &decodeState, storage.CDFs(), &scratch, threading.FrameWorkTileResidualRequest{
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
						return fmt.Errorf("decode/reconstruct job %d: %w", j, err)
					}
					if err := ctx.RetainTileResidualCDFStorage(j, &decodeState, &storage); err != nil {
						return err
					}
				}
				return nil
			}), postFilterRunner(func(ctx decoder.FrameWorkPostFilterContext) error {
				post := decoder.FrameWorkBoundSupportedPostFilterRunner{}
				size, err := supportedPostFilterScratchLen(ctx)
				if err != nil {
					return err
				}
				post.Scratch = postFilterScratchStorage(size)
				if err := post.Apply(ctx); err != nil {
					return err
				}
				if currentMVSurface >= 0 {
					if err := decoder.PublishTemporalMotionReference(post.Context.Event, currentMVSurface, &mvFrames[currentMVSurface], mvStore); err != nil {
						return err
					}
				}
				got, err := testvector.FrameMD5(*post.Context.Output)
				if err != nil {
					return err
				}
				postMD5 = got
				havePost = true
				return nil
			}))
			if err != nil {
				t.Fatalf("ivf frame %d RunEventWithContextAndSideDataAndPostFilterRunners: %v", ivfFrame.Index, err)
			}
			if !result.Run.CompletedFrame {
				continue
			}
			if !event.FrameHeader.ShowFrame {
				continue
			}
			if !havePost {
				t.Fatalf("ivf frame %d completed without postfilter", ivfFrame.Index)
			}
			if emitted >= len(wantDigests) {
				t.Fatalf("ivf frame %d: more visible frames than golden digests (%d)", ivfFrame.Index, len(wantDigests))
			}
			if postMD5 != wantDigests[emitted] {
				t.Fatalf("frame %d md5 got=%x want=%x (libaom golden)", emitted, postMD5, wantDigests[emitted])
			}
			emitted++
		}
	}
	if emitted != len(wantDigests) {
		t.Fatalf("emitted %d visible frames, want %d golden digests", emitted, len(wantDigests))
	}
}

func readProfileClip(t *testing.T, name string) []byte {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "testdata", "profiles", name))
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read clip %s: %v", path, err)
	}
	return data
}

func eventRunsFrameWork(event decoder.Event) bool {
	switch event.Kind {
	case decoder.EventFrameHeader, decoder.EventFrame, decoder.EventTileGroup, decoder.EventExistingFrame:
		return true
	default:
		return false
	}
}

func bindFramePool(t *testing.T, event decoder.Event, count int) (frame.Pool, []byte, []frame.Frame, []int, []bool) {
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
	return pool, backing, frames, free, used
}

func bindMotionStore(t *testing.T, event decoder.Event, count int) ([]tile.ReferenceMVEntry, []tile.TemporalMotionEntry, []tile.ReferenceMVFrame, []tile.TemporalMotionReferenceFrame, int) {
	t.Helper()
	miCols := miExtent(event.FrameSize.CodedWidth)
	miRows := miExtent(event.FrameSize.Height)
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

func miExtent(pixels uint32) uint32 {
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

func residualScratch(ctx decoder.FrameWorkBatch) ([]int32, []int16, error) {
	q, _, err := ctx.BlockQuantizer(ctx.Quantization.BaseQIdx, 0, threading.FrameWorkPlaneY)
	if err != nil {
		return nil, nil, err
	}
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

type batchRunner func(decoder.FrameWorkBatch) error

func (fn batchRunner) Run(ctx decoder.FrameWorkBatch) error { return fn(ctx) }

type postFilterRunner func(decoder.FrameWorkPostFilterContext) error

func (fn postFilterRunner) Apply(ctx decoder.FrameWorkPostFilterContext) error { return fn(ctx) }

type sideDataRunner struct{}

func (sideDataRunner) BindFrameWorkSideData(state *decoder.FrameWorkState, ctx decoder.FrameWorkBatch) error {
	size, err := decoder.FrameWorkSideDataScratchLen(ctx)
	if err != nil {
		return err
	}
	side, err := size.BindRunner(sideDataScratch(size))
	if err != nil {
		return err
	}
	return side.BindFrameWorkSideData(state, ctx)
}

func sideDataScratch(size decoder.FrameWorkSideDataScratchSize) decoder.FrameWorkSideDataScratch {
	return decoder.FrameWorkSideDataScratch{
		CDEFIndex:          make([]uint8, maxInt(size.CDEF, 0)),
		CDEFRead:           make([]bool, maxInt(size.CDEF, 0)),
		LoopFilterRecords:  make([]threading.FrameWorkLoopFilterBlockRecord, maxInt(size.LoopFilterRecords, 0)),
		RestorationRecords: make([]tile.RestorationUnitRecord, maxInt(size.RestorationRecords, 0)),
		RestorationAbove:   make([]uint16, maxInt(size.RestorationBoundary, 0)),
		RestorationBelow:   make([]uint16, maxInt(size.RestorationBoundary, 0)),
	}
}

func supportedPostFilterScratchLen(ctx decoder.FrameWorkPostFilterContext) (decoder.FrameWorkPostFilterScratchSize, error) {
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

func postFilterScratchStorage(size decoder.FrameWorkPostFilterScratchSize) decoder.FrameWorkPostFilterScratch {
	scratch := decoder.FrameWorkPostFilterScratch{
		LoopFilterEdges: make([]decoder.FrameWorkLoopFilterPostFilterEdge, maxInt(size.LoopFilter.Edges, 0)),

		CDEFDirectionGrid: make([]cdef.DirectionGrid, maxInt(size.CDEF.DirectionGrid, 0)),
		CDEFVarianceGrid:  make([]cdef.VarianceGrid, maxInt(size.CDEF.VarianceGrid, 0)),
		CDEFInput:         make([]uint16, maxInt(size.CDEF.Input, 0)),
		CDEFUnitDst:       make([]uint16, maxInt(size.CDEF.UnitDst, 0)),

		SuperResOutputFrame: make([]byte, maxInt(size.SuperRes.OutputFrame, 0)),

		RestorationData:   make([]uint16, maxInt(size.Restoration.Samples.DataLen, 0)),
		RestorationDst:    make([]uint16, maxInt(size.Restoration.Samples.DstLen, 0)),
		RestorationWiener: make([]uint16, maxInt(size.Restoration.Apply.Unit.Wiener, 0)),
		RestorationSGR:    make([]int32, maxInt(size.Restoration.Apply.Unit.SGRProj, 0)),
		RestorationAbove:  make([]uint16, maxInt(size.Restoration.Apply.Boundary.Above, 0)),
		RestorationBelow:  make([]uint16, maxInt(size.Restoration.Apply.Boundary.Below, 0)),

		FilmGrainLumaGrain:   make([]int16, maxInt(size.FilmGrain.LumaGrain, 0)),
		FilmGrainLumaSamples: make([]uint16, maxInt(size.FilmGrain.LumaSamples, 0)),
	}
	for plane := 0; plane < 3; plane++ {
		scratch.CDEFSamples[plane] = make([]uint16, maxInt(size.CDEF.Samples[plane], 0))
		scratch.CDEFDst[plane] = make([]uint16, maxInt(size.CDEF.Dst[plane], 0))
		scratch.SuperResCoded[plane] = make([]uint16, maxInt(size.SuperRes.CodedSamples[plane], 0))
		scratch.SuperResOutput[plane] = make([]uint16, maxInt(size.SuperRes.OutputSamples[plane], 0))
	}
	for plane := 0; plane < 2; plane++ {
		scratch.FilmGrainChromaGrain[plane] = make([]int16, maxInt(size.FilmGrain.ChromaGrain[plane], 0))
		scratch.FilmGrainChromaSamples[plane] = make([]uint16, maxInt(size.FilmGrain.ChromaSamples[plane], 0))
	}
	return scratch
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
