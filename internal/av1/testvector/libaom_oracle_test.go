//go:build goav1_oracle

package testvector

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

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
				havePool = true
			} else {
				gotFormat := frameFormatFromEvent(event)
				if gotFormat != poolFormat {
					t.Fatalf("frame %d format=%+v want %+v", ivfFrame.Index, gotFormat, poolFormat)
				}
			}

			var postMD5 MD5
			postRan := false
			result, err := state.RunEventWithContextAndPostFilter(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx decoder.FrameWorkBatch) error {
				_, _, cdefLen, err := ctx.CDEFIndexMapShape()
				if err != nil {
					return err
				}
				cdefIndex := make([]uint8, cdefLen)
				cdefRead := make([]bool, cdefLen)
				cdefMap, err := ctx.BindCDEFIndexMap(cdefIndex, cdefRead)
				if err != nil {
					return err
				}
				for j := 0; j < len(ctx.Jobs); j++ {
					var decodeState tile.DecodeState
					if err := ctx.JobDecodeState(j, &decodeState); err != nil {
						return err
					}
					var storage threading.FrameWorkTileResidualCDFStorage
					if err := storage.InitDefault(ctx.Quantization.BaseQIdx); err != nil {
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
						CDEFIndexMap:  &cdefMap,
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
				for _, read := range cdefMap.Read {
					if read {
						cdefUnitsRead++
					}
				}
				return nil
			}, func(ctx decoder.FrameWorkPostFilterContext) error {
				if err := ctx.RequireNoActivePostFilters(); err != nil {
					return err
				}
				digestIndex := int(ivfFrame.Index)
				if digestIndex >= len(digests) || digests[digestIndex].FrameIndex != ivfFrame.Index {
					t.Fatalf("frame %d missing official digest", ivfFrame.Index)
				}
				got, err := FrameMD5(*ctx.Output)
				if err != nil {
					return err
				}
				if ivfFrame.Index == 0 && got != digests[digestIndex].MD5 {
					return fmt.Errorf("frame 0 md5 got=%x official=%x", got, digests[digestIndex].MD5)
				}
				postMD5 = got
				postRan = true
				return nil
			})
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
				t.Logf("frame %d md5 progress got=%x official=%x txbs=%d residuals=%d cdef_units=%d",
					ivfFrame.Index, postMD5, digests[digestIndex].MD5, residualTXBs, residuals, cdefUnitsRead)
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
