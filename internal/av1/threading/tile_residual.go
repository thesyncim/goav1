package threading

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// FrameWorkTileResidualCDFs groups the caller-owned entropy states needed to
// walk block syntax, decode transform trees, and decode residual coefficients.
type FrameWorkTileResidualCDFs struct {
	Loop          tile.BlockLoopCDFs
	Coeff         tile.BlockCoeffCDFs
	TransformType *tile.TransformTypeCDFs
}

// FrameWorkTileResidualScratch is caller-owned state reused while decoding one
// tile job's block loop and coefficient contexts.
type FrameWorkTileResidualScratch struct {
	Loop         tile.BlockLoopScratch
	Coeff        tile.BlockCoeffScratch
	CoeffContext tile.CoeffEntropyContext
	InterTX      tile.InterCoeffTransformSelector

	controller frameWorkTileResidualLoopController
	stats      FrameWorkTileResidualStats
}

// FrameWorkBlockTransforms carries the transform policy already determined by
// the block-mode layer. The residual driver only consumes it; it does not guess
// mode-dependent transform types.
type FrameWorkBlockTransforms struct {
	Inter           bool
	Luma            transform.Type
	Chroma          [2]transform.Type
	TransformSelect tile.CoeffTransformSelector
	ReadInterTX     bool
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
		FrameType:          b.FrameHeader.FrameType,
		AllowIntrabc:       b.FrameSize.AllowIntrabc,
		ReferenceMode:      b.TransformRef.ReferenceMode,
		SkipModeRefs: [2]tile.ReferenceFrame{
			tile.ReferenceFrame(b.SkipMode.RefFrameIdx[0]),
			tile.ReferenceFrame(b.SkipMode.RefFrameIdx[1]),
		},
	}, nil
}

// DecodeAndReconstructJobResiduals walks one tile job's blocks, invokes the
// caller's prediction hook, decodes residual TXBs, and reconstructs each decoded
// TXB into the batch output frame.
func (b FrameWorkBatch) DecodeAndReconstructJobResiduals(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest) (FrameWorkTileResidualStats, error) {
	if state == nil || scratch == nil || req.Transforms == nil {
		return FrameWorkTileResidualStats{}, ErrInvalidBatch
	}
	scratch.stats = FrameWorkTileResidualStats{}

	loopCDFs := cdfs.Loop
	if loopCDFs.Transform == nil {
		loopCDFs.Transform = cdfs.Coeff.Transform
	}
	if loopCDFs.Coeff == nil {
		loopCDFs.Coeff = cdfs.Coeff.Coeff
	}
	loopReq := req.Loop
	loopReq.DecodeCoefficients = true
	scratch.controller = frameWorkTileResidualLoopController{
		batch:                  b,
		index:                  index,
		state:                  state,
		cdfs:                   cdfs,
		scratch:                scratch,
		req:                    req,
		stats:                  &scratch.stats,
		userBeforeCoefficients: loopReq.BeforeCoefficients,
		userCoeffVisitor:       loopReq.CoeffVisitor,
	}

	loopStats, err := tile.DecodeBlockLoopWithCoeffController(state, loopCDFs, &scratch.Loop, loopReq, &scratch.controller, func(visit tile.BlockLoopVisit) error {
		if !visit.CoefficientsValid {
			return ErrInvalidBatch
		}
		frameWorkAccumulateResidualStats(&scratch.stats, visit.Coefficients.TotalStats())
		return nil
	})
	scratch.stats.Loop = loopStats
	if err != nil {
		return scratch.stats, err
	}
	return scratch.stats, nil
}

type frameWorkTileResidualLoopController struct {
	batch   FrameWorkBatch
	index   int
	state   *tile.DecodeState
	cdfs    FrameWorkTileResidualCDFs
	scratch *FrameWorkTileResidualScratch
	req     FrameWorkTileResidualRequest
	stats   *FrameWorkTileResidualStats

	userBeforeCoefficients tile.BlockLoopVisitor
	userCoeffVisitor       tile.BlockLoopCoeffVisitor
}

func (c *frameWorkTileResidualLoopController) BeforeBlockCoefficients(visit tile.BlockLoopVisit) error {
	if c.userBeforeCoefficients != nil {
		if err := c.userBeforeCoefficients(visit); err != nil {
			return err
		}
	}
	if c.req.Predict != nil {
		if err := c.req.Predict(visit); err != nil {
			return err
		}
		c.stats.Predictions++
	}
	if visit.Prefix.SkipTransform {
		c.stats.SkippedBlocks++
	} else {
		c.stats.CoefficientBlocks++
	}
	return nil
}

func (c *frameWorkTileResidualLoopController) SelectBlockCoeffRequest(visit tile.BlockLoopVisit) (tile.BlockCoeffRequest, error) {
	transforms, err := c.req.Transforms(visit)
	if err != nil {
		return tile.BlockCoeffRequest{}, err
	}
	qIndex := c.state.CurrentBaseQIdx
	_, lossless, err := c.batch.BlockQIndex(qIndex, visit.SegmentID)
	if err != nil {
		return tile.BlockCoeffRequest{}, err
	}
	transformSelect := transforms.TransformSelect
	if transforms.ReadInterTX {
		c.scratch.InterTX.Reset(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, visit.Prefix.SkipTransform, lossless)
		transformSelect = &c.scratch.InterTX
	}
	return tile.BlockCoeffRequest{
		Transform: tile.TransformTreeRequest{
			Size:          visit.Block.Size,
			X4:            visit.Block.X4,
			Y4:            visit.Block.Y4,
			VisibleW4:     visit.Block.VisibleW4,
			VisibleH4:     visit.Block.VisibleH4,
			Color:         c.batch.Sequence.ColorConfig,
			TransformMode: c.req.TransformMode,
			Inter:         transforms.Inter,
			SkipTransform: visit.Prefix.SkipTransform,
			Lossless:      lossless,
		},
		LumaType:        transforms.Luma,
		ChromaType:      transforms.Chroma,
		TransformSelect: transformSelect,
		EOBMultiContext: transforms.EOBMultiContext,
	}, nil
}

func (c *frameWorkTileResidualLoopController) VisitBlockCoeff(visit tile.BlockLoopVisit, block tile.BlockCoeffBlock) error {
	if err := c.batch.ReconstructBlockCoeff(c.index, FrameWorkBlockCoeffReconstruction{
		Visit:           visit.Block,
		Block:           block,
		Transform:       block.Transform,
		CurrentQIndex:   c.state.CurrentBaseQIdx,
		SegmentID:       visit.SegmentID,
		Int32Scratch:    c.req.Int32Scratch,
		ResidualScratch: c.req.ResidualScratch,
	}); err != nil {
		return err
	}
	c.stats.Residuals++
	if c.userCoeffVisitor != nil {
		return c.userCoeffVisitor(visit, block)
	}
	return nil
}

func (b FrameWorkBatch) ReadInterBlockTransforms(state *tile.DecodeState, visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	if state == nil {
		return FrameWorkBlockTransforms{}, ErrInvalidBatch
	}
	if _, _, err := b.BlockQIndex(state.CurrentBaseQIdx, visit.SegmentID); err != nil {
		return FrameWorkBlockTransforms{}, err
	}
	return FrameWorkBlockTransforms{
		Inter:       true,
		Luma:        transform.TypeDCTDCT,
		Chroma:      [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT},
		ReadInterTX: true,
	}, nil
}

func frameWorkAccumulateResidualStats(stats *FrameWorkTileResidualStats, coeff tile.LumaCoeffStats) {
	stats.TXBs += coeff.TXBs
	stats.NonZero += coeff.NonZero
	stats.AllZero += coeff.AllZero
	stats.EOBTotal += coeff.EOBTotal
}
