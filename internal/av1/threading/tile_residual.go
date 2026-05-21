package threading

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// FrameWorkTileResidualCDFs groups the caller-owned entropy states needed to
// walk block syntax, decode transform trees, and decode residual coefficients.
type FrameWorkTileResidualCDFs struct {
	Loop  tile.BlockLoopCDFs
	Coeff tile.BlockCoeffCDFs
}

// FrameWorkTileResidualScratch is caller-owned state reused while decoding one
// tile job's block loop and coefficient contexts.
type FrameWorkTileResidualScratch struct {
	Loop         tile.BlockLoopScratch
	Coeff        tile.BlockCoeffScratch
	CoeffContext tile.CoeffEntropyContext
}

// FrameWorkBlockTransforms carries the transform policy already determined by
// the block-mode layer. The residual driver only consumes it; it does not guess
// mode-dependent transform types.
type FrameWorkBlockTransforms struct {
	Inter           bool
	Luma            transform.Type
	Chroma          [2]transform.Type
	EOBMultiContext [3]int
}

// FrameWorkBlockTransformSelector returns transform syntax decisions for one
// block-loop visit.
type FrameWorkBlockTransformSelector func(tile.BlockLoopVisit) (FrameWorkBlockTransforms, error)

// FrameWorkBlockPredictor prepares prediction pixels for one block before the
// residual, if any, is added to the output frame.
type FrameWorkBlockPredictor func(tile.BlockLoopVisit) error

// FrameWorkTileResidualRequest describes one tile job residual decode pass.
type FrameWorkTileResidualRequest struct {
	Loop          tile.BlockLoopRequest
	TransformMode parser.TransformMode

	Predict    FrameWorkBlockPredictor
	Transforms FrameWorkBlockTransformSelector

	Int32Scratch    []int32
	ResidualScratch []int16
}

// FrameWorkTileResidualStats summarizes the composed block-loop/coefficient
// decode and reconstruction work for one tile job.
type FrameWorkTileResidualStats struct {
	Loop tile.BlockLoopStats

	CoefficientBlocks int
	SkippedBlocks     int

	TXBs        int
	NonZero     int
	AllZero     int
	EOBTotal    int
	Residuals   int
	Predictions int
}

// JobBlockLoopRequest derives the block-loop request for Jobs[index] from the
// frame context carried by this batch.
func (b FrameWorkBatch) JobBlockLoopRequest(index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride int) (tile.BlockLoopRequest, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return tile.BlockLoopRequest{}, err
	}
	return tile.BlockLoopRequest{
		Walk: tile.BlockWalkRequest{
			Root:       tile.RootBlockLevel(b.Sequence.Use128x128Superblock),
			MIColStart: region.MIColStart,
			MIRowStart: region.MIRowStart,
			MIColEnd:   region.MIColEnd,
			MIRowEnd:   region.MIRowEnd,
		},
		SkipMode:           b.SkipMode,
		CDEF:               b.CDEF,
		Segmentation:       b.Segmentation,
		Delta:              b.Delta,
		SBSizeMIB:          b.Sequence.SBSizeMIB,
		Monochrome:         b.Sequence.ColorConfig.MonoChrome,
		CurrentSegmentMap:  currentSegmentMap,
		PreviousSegmentMap: previousSegmentMap,
		SegmentMapStride:   segmentMapStride,
	}, nil
}

// DecodeAndReconstructJobResiduals walks one tile job's blocks, invokes the
// caller's prediction hook, decodes residual TXBs, and reconstructs each decoded
// TXB into the batch output frame.
func (b FrameWorkBatch) DecodeAndReconstructJobResiduals(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest) (FrameWorkTileResidualStats, error) {
	if state == nil || scratch == nil || req.Transforms == nil {
		return FrameWorkTileResidualStats{}, ErrInvalidBatch
	}
	var stats FrameWorkTileResidualStats
	loopStats, err := state.DecodeBlockLoop(cdfs.Loop, &scratch.Loop, req.Loop, func(visit tile.BlockLoopVisit) error {
		if req.Predict != nil {
			if err := req.Predict(visit); err != nil {
				return err
			}
			stats.Predictions++
		}

		transforms, err := req.Transforms(visit)
		if err != nil {
			return err
		}
		qIndex := state.CurrentBaseQIdx
		_, lossless, err := b.BlockQIndex(qIndex, visit.SegmentID)
		if err != nil {
			return err
		}
		coeffReq := tile.BlockCoeffRequest{
			Transform: tile.TransformTreeRequest{
				Size:          visit.Block.Size,
				X4:            visit.Block.X4,
				Y4:            visit.Block.Y4,
				VisibleW4:     visit.Block.VisibleW4,
				VisibleH4:     visit.Block.VisibleH4,
				Color:         b.Sequence.ColorConfig,
				TransformMode: req.TransformMode,
				Inter:         transforms.Inter,
				SkipTransform: visit.Prefix.SkipTransform,
				Lossless:      lossless,
			},
			LumaType:        transforms.Luma,
			ChromaType:      transforms.Chroma,
			EOBMultiContext: transforms.EOBMultiContext,
		}

		if visit.Prefix.SkipTransform {
			stats.SkippedBlocks++
		} else {
			stats.CoefficientBlocks++
		}
		result, err := state.DecodeBlockCoefficients(cdfs.Coeff, &scratch.Loop.Mode, &scratch.CoeffContext, &scratch.Coeff, coeffReq, func(block tile.BlockCoeffBlock) error {
			typ, err := frameWorkBlockTransformForPlane(transforms, block.Plane)
			if err != nil {
				return err
			}
			if err := b.ReconstructBlockCoeff(index, FrameWorkBlockCoeffReconstruction{
				Visit:           visit.Block,
				Block:           block,
				Transform:       typ,
				CurrentQIndex:   qIndex,
				SegmentID:       visit.SegmentID,
				Int32Scratch:    req.Int32Scratch,
				ResidualScratch: req.ResidualScratch,
			}); err != nil {
				return err
			}
			stats.Residuals++
			return nil
		})
		if err != nil {
			return err
		}
		frameWorkAccumulateResidualStats(&stats, result.TotalStats())
		return nil
	})
	stats.Loop = loopStats
	if err != nil {
		return stats, err
	}
	return stats, nil
}

func frameWorkBlockTransformForPlane(transforms FrameWorkBlockTransforms, plane int) (transform.Type, error) {
	switch plane {
	case 0:
		return transforms.Luma, nil
	case 1:
		return transforms.Chroma[0], nil
	case 2:
		return transforms.Chroma[1], nil
	default:
		return 0, ErrInvalidBatch
	}
}

func frameWorkAccumulateResidualStats(stats *FrameWorkTileResidualStats, coeff tile.LumaCoeffStats) {
	stats.TXBs += coeff.TXBs
	stats.NonZero += coeff.NonZero
	stats.AllZero += coeff.AllZero
	stats.EOBTotal += coeff.EOBTotal
}
