//go:build goav1_oracle

package testvector

import (
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

type libaomFirstLumaTXB struct {
	Visit      tile.BlockVisit
	Prediction tile.BlockPredictionModeResult
	Block      tile.TransformBlock
	Transform  transform.Type
	EOB        int

	Coeffs   [16]int16
	Residual [16]int16
	Final    [16]byte
}

func TestLibaomQuantizer00FirstLumaTXBMatchesOracle(t *testing.T) {
	got := decodeLibaomQuantizer00FirstLumaTXB(t)
	if got.Visit.Size != tile.BlockSize4x16 || got.Block.Size != tile.TransformSize4x4 {
		t.Fatalf("first block visit=%+v tx=%+v want 4x16/4x4", got.Visit, got.Block)
	}
	if !got.Prediction.Valid || !got.Prediction.Intra || got.Prediction.LumaMode != tile.IntraModeDC || got.Prediction.FilterIntraValid {
		t.Fatalf("first prediction=%+v want DC intra without filter-intra", got.Prediction)
	}
	if got.Transform != transform.TypeDCTDCT || got.EOB != 16 {
		t.Fatalf("first tx transform=%v eob=%d want DCT_DCT/eob16", got.Transform, got.EOB)
	}
	wantCoeffs := []int16{-158, -3, 4, -1, -3, -3, -1, 0, 1, -2, -3, -1, 0, -1, -1, -1}
	wantResidual := []int16{-42, -40, -38, -39, -42, -43, -40, -39, -39, -41, -41, -39, -38, -38, -37, -37}
	wantFinal := []byte{86, 88, 90, 89, 86, 85, 88, 89, 89, 87, 87, 89, 90, 90, 91, 91}
	if !slices.Equal(got.Coeffs[:], wantCoeffs) {
		t.Fatalf("coeffs=%v want %v", got.Coeffs, wantCoeffs)
	}
	if !slices.Equal(got.Residual[:], wantResidual) {
		t.Fatalf("residual=%v want %v", got.Residual, wantResidual)
	}
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
		_, err := state.RunEventWithContextAndPostFilter(&refs, &pool, event.SequenceHeader, event, 32, referenceSurfaces[:], referenceFrames[:], 1, spans[:], jobs[:], batches[:], releases[:], workerPool, func(ctx decoder.FrameWorkBatch) error {
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
				loopReq.CoeffVisitor = func(visit tile.BlockLoopVisit, block tile.BlockCoeffBlock) error {
					if seen || block.Plane != 0 || block.Block.X4 != 0 || block.Block.Y4 != 0 {
						return nil
					}
					seen = true
					residual, err := libaomResidualForBlock(ctx, decodeState.CurrentBaseQIdx, visit, block)
					if err != nil {
						return err
					}
					plane, err := ctx.JobOutputPlane(j, threading.FrameWorkPlaneY)
					if err != nil {
						return err
					}
					got.Visit = visit.Block
					got.Prediction = visit.Prediction
					got.Block = block.Block
					got.Transform = block.Transform
					got.EOB = int(block.Result.EOB)
					copy(got.Coeffs[:], block.Coeffs[:16])
					copy(got.Residual[:], residual[:16])
					blockX := int(block.Block.X4) * 4
					blockY := int(block.Block.Y4) * 4
					for yy := 0; yy < 4; yy++ {
						row := (blockY + yy - plane.Y) * plane.Stride
						col := blockX - plane.X
						copy(got.Final[yy*4:yy*4+4], plane.Pix[row+col:row+col+4])
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
					Loop:          loopReq,
					TransformMode: ctx.TransformRef.TransformMode,
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

func libaomResidualForBlock(ctx decoder.FrameWorkBatch, qIndex uint8, visit tile.BlockLoopVisit, block tile.BlockCoeffBlock) ([]int16, error) {
	q, lossless, err := ctx.BlockQuantizer(qIndex, visit.SegmentID, threading.FrameWorkPlaneY)
	if err != nil {
		return nil, err
	}
	size, err := block.Block.Size.TransformSize()
	if err != nil {
		return nil, err
	}
	scanSize, err := transform.ScanSize(size)
	if err != nil {
		return nil, err
	}
	dequant := make([]int32, scanSize.Width*scanSize.Height)
	txScale, err := quantize.TransformScale(size.Width, size.Height)
	if err != nil {
		return nil, err
	}
	if err := quantize.DequantizeBlockScaled(dequant, scanSize.Height, block.Coeffs, scanSize.Height, scanSize.Width, scanSize.Height, q, txScale); err != nil {
		return nil, err
	}
	residual := make([]int16, size.Width*size.Height)
	if lossless {
		if err := transform.InverseWHT4x4Block(residual, size.Width, dequant, scanSize.Height, int(block.Result.EOB)); err != nil {
			return nil, err
		}
		return residual, nil
	}
	scratchLen, _, err := reconstruct.ScratchLen(reconstruct.Block{Size: size, Transform: block.Transform, Quantizer: q})
	if err != nil {
		return nil, err
	}
	scratch := make([]int32, scratchLen)
	if err := transform.InverseBlock(residual, size.Width, dequant, scanSize.Height, scratch, size, block.Transform); err != nil {
		return nil, err
	}
	return residual, nil
}
