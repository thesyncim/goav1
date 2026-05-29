//go:build goav1_oracle

package testvector

import (
	_ "embed"
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// Vendored profile-conformance clips. libaom's published AV1 test-data
// (test/test-data.sha1) ships no 4:4:4 (profile 1), 4:2:2 (profile 2), or
// 12-bit (profile 2) bitstreams, so these tiny clips are generated with
// libaom's aomenc and committed directly to git. Each is a 64x64 all-intra
// (kf-max-dist=1) 3-frame clip; the golden per-frame MD5s below were produced
// by decoding each clip with libaom's `aomdec --rawvideo` and hashing the
// visible Y/U/V planes (libaom test/md5_helper.h layout, which FrameMD5
// reproduces exactly). See TestLibaomProfileVendoredClips for the regen recipe.
//
// These guard the profile-1 (4:4:4 8-bit), profile-2 (4:2:2 8-bit) and
// profile-2 (12-bit) intra decode paths, which were proven byte-exact against
// aomdec but were previously only covered by throwaway ad-hoc tests.

//go:embed testdata/profiles/profile1-444-8bit-64x64.ivf
var profile444_8bitIVF []byte

//go:embed testdata/profiles/profile2-422-8bit-64x64.ivf
var profile422_8bitIVF []byte

//go:embed testdata/profiles/profile2-420-12bit-64x64.ivf
var profile420_12bitIVF []byte

// libaomProfileClip is a vendored profile-conformance clip plus its golden
// per-frame MD5s. The MD5 hex strings are the exact libaom aomdec --rawvideo
// digests (see the regen recipe in TestLibaomProfileVendoredClips).
type libaomProfileClip struct {
	name        string
	ivf         []byte
	frameMD5Hex []string

	wantBitDepth     uint8
	wantSubsamplingX bool
	wantSubsamplingY bool
}

var libaomProfileClips = []libaomProfileClip{
	{
		// Profile 1: 4:4:4 8-bit. SubsamplingX=false SubsamplingY=false.
		name: "profile1-444-8bit-64x64",
		ivf:  profile444_8bitIVF,
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
		ivf:  profile422_8bitIVF,
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
		ivf:  profile420_12bitIVF,
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

// TestLibaomProfileVendoredClips decodes the vendored profile-conformance
// clips through the framework pipeline and asserts every frame is byte-exact
// against its libaom aomdec golden MD5. Unlike the remote-manifest dry-runs,
// these clips are committed to git (testdata/profiles/), so the test runs
// fully offline and deterministically.
//
// Regen recipe (run from a libaom build tree):
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
//	# Golden MD5s (per visible frame, Y then U then V rows, little-endian):
//	aomdec --rawvideo -o out.raw clip.ivf  # then md5 each frame slice
func TestLibaomProfileVendoredClips(t *testing.T) {
	for _, clip := range libaomProfileClips {
		clip := clip
		t.Run(clip.name, func(t *testing.T) {
			runLibaomProfileClip(t, clip)
		})
	}
}

func runLibaomProfileClip(t *testing.T, clip libaomProfileClip) {
	t.Helper()
	wantDigests := make([]MD5, len(clip.frameMD5Hex))
	for i, hexStr := range clip.frameMD5Hex {
		md5, err := ParseMD5Hex([]byte(hexStr))
		if err != nil {
			t.Fatalf("parse golden md5[%d]=%q: %v", i, hexStr, err)
		}
		wantDigests[i] = md5
	}

	it, err := ivf.NewIterator(clip.ivf)
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
				pool, _, _, _, _, _ = bindLibaomVectorFramePool(t, event, 8)
				mvEntryBacking, temporalEntryBacking, mvFrames, mvStore, mvLength = bindLibaomVectorMotionStore(t, event, 8)
				havePool = true
			}

			var postMD5 MD5
			havePost := false
			currentMVSurface := -1
			result, err := state.RunEventWithContextAndSideDataAndPostFilterRunners(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, libaomFrameWorkSideDataRunner{}, libaomFrameWorkBatchRunner(func(ctx decoder.FrameWorkBatch) error {
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
					int32Scratch, residualScratch, err := libaomResidualScratch(ctx)
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
			}), libaomFrameWorkPostFilterRunner(func(ctx decoder.FrameWorkPostFilterContext) error {
				post := decoder.FrameWorkBoundSupportedPostFilterRunner{}
				size, err := libaomSupportedPostFilterScratchLen(ctx)
				if err != nil {
					return err
				}
				post.Scratch = libaomPostFilterScratchStorage(size)
				if err := post.Apply(ctx); err != nil {
					return err
				}
				if currentMVSurface >= 0 {
					if err := decoder.PublishTemporalMotionReference(post.Context.Event, currentMVSurface, &mvFrames[currentMVSurface], mvStore); err != nil {
						return err
					}
				}
				got, err := FrameMD5(*post.Context.Output)
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
