//go:build goav1_oracle

package testvector

import (
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

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
