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
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
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
	blockDeltaReads := 0
	intraEntryReads := 0
	transformPathReads := 0
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
				for j := 0; j < len(ctx.Jobs); j++ {
					var decodeState tile.DecodeState
					if err := ctx.JobDecodeState(j, &decodeState); err != nil {
						return err
					}
					region, err := ctx.JobRegion(j)
					if err != nil {
						return err
					}
					var partitionCDFs tile.PartitionCDFs
					if err := partitionCDFs.InitDefault(); err != nil {
						return err
					}
					root := tile.RootBlockLevel(ctx.Sequence.Use128x128Superblock)
					var partitionCtx tile.PartitionContext
					blockSize, reads, err := readFirstLibaomLeafBlock(&decodeState, &partitionCDFs, &partitionCtx, root, region)
					if err != nil {
						return err
					}
					partitionReads += reads
					if !ctx.Segmentation.Enabled {
						var modeCDFs tile.BlockModeCDFs
						if err := modeCDFs.InitDefault(); err != nil {
							return err
						}
						var modeCtx tile.BlockModeContext
						prefix, err := decodeState.ReadBlockModePrefix(&modeCDFs, &modeCtx, tile.BlockModeRequest{
							Size:     blockSize,
							SkipMode: ctx.SkipMode,
							CDEF:     ctx.CDEF,
							Segment:  parser.SegmentData{RefFrame: -1},
						})
						if err != nil {
							return err
						}
						blockPrefixReads++
						if region.MIColStart == 0 && region.MIRowStart == 0 {
							dims, ok := blockSize.Dimensions()
							if !ok {
								return fmt.Errorf("invalid block size %d", blockSize)
							}
							block, err := ctx.JobBlockDeltaContext(j, region.MIColStart, region.MIRowStart,
								dims.W4 == ctx.Sequence.SBSizeMIB && dims.H4 == ctx.Sequence.SBSizeMIB,
								prefix.SkipTransform)
							if err != nil {
								return err
							}
							deltaCDFs, err := initLibaomDeltaCDFs()
							if err != nil {
								return err
							}
							if err := decodeState.ReadBlockDeltas(ctx.Delta, block, deltaCDFs); err != nil {
								return err
							}
							blockDeltaReads++

							var intraCDFs tile.IntraModeCDFs
							if err := intraCDFs.InitDefault(); err != nil {
								return err
							}
							intra, err := decodeState.ReadIntraFlag(&intraCDFs, &modeCtx, tile.IntraFlagRequest{
								FrameType:           ctx.FrameHeader.FrameType,
								AllowIntrabc:        ctx.FrameSize.AllowIntrabc,
								SkipMode:            prefix.SkipMode,
								SegmentationEnabled: ctx.Segmentation.Enabled,
								Segment:             parser.SegmentData{RefFrame: -1},
							})
							if err != nil {
								return err
							}
							intraEntryReads++
							if intra {
								mode, err := decodeState.ReadLumaIntraMode(&intraCDFs, &modeCtx, tile.LumaIntraModeRequest{
									FrameType: ctx.FrameHeader.FrameType,
									Size:      blockSize,
								})
								if err != nil {
									return err
								}
								if err := modeCtx.MarkIntra(blockSize, 0, 0, true, mode); err != nil {
									return err
								}
								if ctx.TransformRef.TransformMode != parser.TransformModeSwitchable || ctx.Segmentation.Lossless[0] {
									var txCDFs tile.TransformCDFs
									if err := txCDFs.InitDefault(); err != nil {
										return err
									}
									tx, err := decodeState.ReadSelectedTransformSize(&txCDFs, &modeCtx, tile.SelectedTransformRequest{
										Size:          blockSize,
										TransformMode: ctx.TransformRef.TransformMode,
										Lossless:      ctx.Segmentation.Lossless[0],
									})
									if err != nil {
										return err
									}
									if err := modeCtx.MarkTransform(blockSize, 0, 0, tx, true); err != nil {
										return err
									}
									transformPathReads++
								}
							}
						}
					}
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
				return nil
			}, func(ctx decoder.FrameWorkPostFilterContext) error {
				digestIndex := int(ivfFrame.Index)
				if digestIndex >= len(digests) || digests[digestIndex].FrameIndex != ivfFrame.Index {
					t.Fatalf("frame %d missing official digest", ivfFrame.Index)
				}
				got, err := FrameMD5(*ctx.Output)
				if err != nil {
					return err
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
				if postMD5 == digests[digestIndex].MD5 {
					t.Fatalf("frame %d unexpectedly matched official md5 before reconstruction is wired", ivfFrame.Index)
				}
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
	if blockDeltaReads == 0 {
		t.Fatal("no block delta syntax path ran")
	}
	if intraEntryReads == 0 {
		t.Fatal("no intra entry syntax was read")
	}
	if transformPathReads == 0 {
		t.Fatal("no transform size syntax path ran")
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

func readFirstLibaomLeafBlock(state *tile.DecodeState, cdfs *tile.PartitionCDFs, ctx *tile.PartitionContext, root tile.BlockLevel, region threading.FrameWorkJobRegion) (tile.BlockSize, int, error) {
	level := root
	x4 := uint32(0)
	y4 := uint32(0)
	reads := 0
	for {
		partitionCtx, err := ctx.Context(level, int(y4), int(x4))
		if err != nil {
			return 0, reads, err
		}
		half := uint32(level.HalfSize4x4())
		partition, err := state.ReadPartition(cdfs, level, partitionCtx,
			region.MIRowEnd > region.MIRowStart+y4+half,
			region.MIColEnd > region.MIColStart+x4+half)
		if err != nil {
			return 0, reads, err
		}
		reads++
		if !partition.ValidForLevel(level) {
			return 0, reads, fmt.Errorf("partition %d invalid for level %d", partition, level)
		}
		if partition == tile.PartitionSplit && level != tile.BlockLevel8x8 {
			level++
			continue
		}
		blockSize, _, ok := partition.BlockSizes(level)
		if !ok {
			return 0, reads, fmt.Errorf("partition %d has no block size at level %d", partition, level)
		}
		return blockSize, reads, nil
	}
}

func initLibaomDeltaCDFs() (tile.DeltaCDFs, error) {
	var q entropy.CDF
	var lf entropy.CDF
	var multi [tile.FrameLoopFilterCount]entropy.CDF
	if err := q.InitDefaultDelta(); err != nil {
		return tile.DeltaCDFs{}, err
	}
	if err := lf.InitDefaultDelta(); err != nil {
		return tile.DeltaCDFs{}, err
	}
	cdfs := tile.DeltaCDFs{Q: &q, LF: &lf}
	for i := 0; i < tile.FrameLoopFilterCount; i++ {
		if err := multi[i].InitDefaultDelta(); err != nil {
			return tile.DeltaCDFs{}, err
		}
		cdfs.LFMulti[i] = &multi[i]
	}
	return cdfs, nil
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
