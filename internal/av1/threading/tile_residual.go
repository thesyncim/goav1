package threading

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
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
	Restoration   tile.RestorationCDFs
}

// FrameWorkTileResidualCDFStorage owns the default entropy states needed for a
// full block-loop, transform, and coefficient pass. Callers can keep one per
// worker/tile context and pass CDFs() into DecodeAndReconstructJobResiduals.
type FrameWorkTileResidualCDFStorage struct {
	Partition             tile.PartitionCDFs
	Mode                  tile.BlockModeCDFs
	Intra                 tile.IntraModeCDFs
	InterRef              tile.InterRefCDFs
	InterMode             tile.InterModeCDFs
	MV                    tile.MVCDFs
	Interp                tile.InterpFilterCDFs
	Motion                tile.MotionModeCDFs
	Blend                 tile.CompoundBlendCDFs
	Transform             tile.TransformCDFs
	TransformType         tile.TransformTypeCDFs
	Coeff                 tile.CoeffCDFs
	DeltaQ                entropy.CDF
	DeltaLF               entropy.CDF
	DeltaLFMulti          [tile.FrameLoopFilterCount]entropy.CDF
	RestorationSwitchable entropy.CDF
	RestorationWiener     entropy.CDF
	RestorationSGRProj    entropy.CDF
}

// InitDefault seeds the storage with the CDF defaults ported from libaom/dav1d.
func (s *FrameWorkTileResidualCDFStorage) InitDefault(baseQIndex uint8) error {
	if s == nil {
		return ErrInvalidBatch
	}
	var next FrameWorkTileResidualCDFStorage
	if err := next.Partition.InitDefault(); err != nil {
		return err
	}
	if err := next.Mode.InitDefault(); err != nil {
		return err
	}
	if err := next.Intra.InitDefault(); err != nil {
		return err
	}
	if err := next.InterRef.InitDefault(); err != nil {
		return err
	}
	if err := next.InterMode.InitDefault(); err != nil {
		return err
	}
	if err := next.MV.InitDefault(); err != nil {
		return err
	}
	if err := next.Interp.InitDefault(); err != nil {
		return err
	}
	if err := next.Motion.InitDefault(); err != nil {
		return err
	}
	if err := next.Blend.InitDefault(); err != nil {
		return err
	}
	if err := next.Transform.InitDefault(); err != nil {
		return err
	}
	if err := next.TransformType.InitDefault(); err != nil {
		return err
	}
	if err := next.Coeff.InitDefault(baseQIndex); err != nil {
		return err
	}
	if err := next.DeltaQ.InitDefaultDelta(); err != nil {
		return err
	}
	if err := next.DeltaLF.InitDefaultDelta(); err != nil {
		return err
	}
	for i := 0; i < tile.FrameLoopFilterCount; i++ {
		if err := next.DeltaLFMulti[i].InitDefaultDelta(); err != nil {
			return err
		}
	}
	if err := tile.InitDefaultRestorationCDFs(tile.RestorationCDFs{
		Switchable: &next.RestorationSwitchable,
		Wiener:     &next.RestorationWiener,
		SGRProj:    &next.RestorationSGRProj,
	}); err != nil {
		return err
	}
	*s = next
	return nil
}

// CDFs returns the pointer view consumed by the residual driver.
func (s *FrameWorkTileResidualCDFStorage) CDFs() FrameWorkTileResidualCDFs {
	if s == nil {
		return FrameWorkTileResidualCDFs{}
	}
	delta := tile.DeltaCDFs{
		Q:  &s.DeltaQ,
		LF: &s.DeltaLF,
	}
	for i := 0; i < tile.FrameLoopFilterCount; i++ {
		delta.LFMulti[i] = &s.DeltaLFMulti[i]
	}
	return FrameWorkTileResidualCDFs{
		Loop: tile.BlockLoopCDFs{
			Partition: &s.Partition,
			Mode:      &s.Mode,
			Intra:     &s.Intra,
			InterRef:  &s.InterRef,
			InterMode: &s.InterMode,
			MV:        &s.MV,
			Interp:    &s.Interp,
			Motion:    &s.Motion,
			Blend:     &s.Blend,
			Transform: &s.Transform,
			Coeff:     &s.Coeff,
			Delta:     delta,
		},
		Coeff: tile.BlockCoeffCDFs{
			Transform: &s.Transform,
			Coeff:     &s.Coeff,
		},
		TransformType: &s.TransformType,
		Restoration: tile.RestorationCDFs{
			Switchable: &s.RestorationSwitchable,
			Wiener:     &s.RestorationWiener,
			SGRProj:    &s.RestorationSGRProj,
		},
	}
}

// InitTileResidualCDFStorage seeds storage from the frame context selected for
// this tile group, falling back to AV1 defaults when the batch is used outside
// decoder-managed frame work.
func (b FrameWorkBatch) InitTileResidualCDFStorage(storage *FrameWorkTileResidualCDFStorage) error {
	if storage == nil {
		return ErrInvalidBatch
	}
	if b.InitialTileResidualCDFs != nil {
		*storage = *b.InitialTileResidualCDFs
		return nil
	}
	return storage.InitDefault(b.Quantization.BaseQIdx)
}

// RetainTileResidualCDFStorage publishes storage as the adapted frame context
// only for the context_update_tile_id job when CDF updates are enabled.
func (b FrameWorkBatch) RetainTileResidualCDFStorage(index int, state *tile.DecodeState, storage *FrameWorkTileResidualCDFStorage) error {
	if index < 0 || index >= len(b.Jobs) || state == nil || storage == nil {
		return ErrInvalidBatch
	}
	if !state.RetainFrameContext {
		return nil
	}
	updates, err := b.JobUpdatesFrameContext(index)
	if err != nil {
		return err
	}
	if !updates {
		return nil
	}
	if b.RetainedTileResidualCDFs == nil || b.RetainedTileResidualCDFsValid == nil {
		return ErrInvalidBatch
	}
	*b.RetainedTileResidualCDFs = *storage
	*b.RetainedTileResidualCDFsValid = true
	return nil
}

// FrameWorkTileResidualScratch is caller-owned state reused while decoding one
// tile job's block loop and coefficient contexts.
type FrameWorkTileResidualScratch struct {
	Loop         tile.BlockLoopScratch
	LoopContext  tile.BlockLoopContextCarrier
	Coeff        tile.BlockCoeffScratch
	CoeffContext tile.CoeffEntropyContext
	IntraTX      tile.IntraCoeffTransformSelector
	InterTX      tile.InterCoeffTransformSelector
	CFL          FrameWorkCFLPredictionScratch

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
	ReadIntraTX     bool
	ReadInterTX     bool
	EOBMultiContext [3]int
}

// FrameWorkBlockTransformSelector returns transform syntax decisions for one
// block-loop visit.
type FrameWorkBlockTransformSelector func(tile.BlockLoopVisit) (FrameWorkBlockTransforms, error)

// FrameWorkBlockPredictor prepares prediction pixels for one block before the
// residual, if any, is added to the output frame.
type FrameWorkBlockPredictor func(tile.BlockLoopVisit) error

// FrameWorkTileRestorationRequest carries frame-wide restoration unit storage
// for one residual decode pass. References are copied into the worker-local
// controller so Wiener/SGR ref-subexp state advances without mutating the
// caller's request while the job runs.
type FrameWorkTileRestorationRequest struct {
	Buffers    FrameWorkRestorationFrameBuffers
	References [3]tile.RestorationReferences
}

// InitReferences seeds the per-plane restoration reference state used by AV1's
// restoration unit syntax.
func (r *FrameWorkTileRestorationRequest) InitReferences() error {
	if r == nil {
		return ErrInvalidBatch
	}
	for plane := range r.References {
		r.References[plane] = tile.DefaultRestorationReferences()
	}
	return nil
}

// FrameWorkTileResidualRequest describes one tile job residual decode pass.
type FrameWorkTileResidualRequest struct {
	Loop          tile.BlockLoopRequest
	TransformMode parser.TransformMode

	Predict           FrameWorkBlockPredictor
	PredictionScratch *FrameWorkPredictionScratch
	CDEFIndexMap      *FrameWorkCDEFIndexMap
	LoopFilterMap     *FrameWorkLoopFilterMap
	Restoration       *FrameWorkTileRestorationRequest
	// UseDefaultTransforms derives transform decisions from the frame-work
	// block mode state when Transforms is nil.
	UseDefaultTransforms bool
	Transforms           FrameWorkBlockTransformSelector
	AfterBlock           tile.BlockLoopVisitor

	Int32Scratch    []int32
	ResidualScratch []int16
}

// FrameWorkTileResidualStats summarizes the composed block-loop/coefficient
// decode and reconstruction work for one tile job.
type FrameWorkTileResidualStats struct {
	Loop tile.BlockLoopStats

	CoefficientBlocks int
	SkippedBlocks     int

	TXBs             int
	NonZero          int
	AllZero          int
	EOBTotal         int
	Residuals        int
	Predictions      int
	RestorationUnits int
}

// JobBlockLoopRequest derives the block-loop request for Jobs[index] from the
// frame context carried by this batch.
func (b FrameWorkBatch) JobBlockLoopRequest(index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride int) (tile.BlockLoopRequest, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return tile.BlockLoopRequest{}, err
	}
	globalMVs, globalTypes := frameWorkBlockLoopGlobalMotion(b.GlobalMotion, b.TileInfo.AllowHighPrecisionMV, b.FrameHeader.ForceIntegerMV)
	refSignBias, err := frameWorkBlockLoopRefSignBias(b.Sequence.EnableOrderHint, b.Sequence.OrderHintBits, b.FrameHeader.OrderHint, b.ReferenceOrderHints)
	if err != nil {
		return tile.BlockLoopRequest{}, err
	}
	refFrameSide, err := frameWorkBlockLoopRefFrameSide(b.Sequence.EnableOrderHint, b.Sequence.OrderHintBits, b.FrameHeader.OrderHint, b.ReferenceOrderHints)
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
		SkipMode:                    b.SkipMode,
		CDEF:                        b.CDEF,
		Segmentation:                b.Segmentation,
		Delta:                       b.Delta,
		SBSizeMIB:                   b.Sequence.SBSizeMIB,
		Monochrome:                  b.Sequence.ColorConfig.MonoChrome,
		Color:                       b.Sequence.ColorConfig,
		Lossless:                    b.Segmentation.AllLossless,
		CurrentSegmentMap:           currentSegmentMap,
		PreviousSegmentMap:          previousSegmentMap,
		SegmentMapStride:            segmentMapStride,
		FrameType:                   b.FrameHeader.FrameType,
		AllowIntrabc:                b.FrameSize.AllowIntrabc,
		ReferenceMode:               b.TransformRef.ReferenceMode,
		GlobalMVs:                   globalMVs,
		GlobalMotion:                b.GlobalMotion.Ref,
		GlobalMotionTypes:           globalTypes,
		RefSignBias:                 refSignBias,
		RefFrameSide:                refFrameSide,
		ReferenceOrderHints:         b.ReferenceOrderHints,
		InterpolationFilter:         b.TileInfo.InterpolationFilter,
		EnableDualFilter:            b.Sequence.EnableDualFilter,
		EnableFilterIntra:           b.Sequence.EnableFilterIntra,
		AllowHighPrecisionMV:        b.TileInfo.AllowHighPrecisionMV,
		ForceIntegerMV:              b.FrameHeader.ForceIntegerMV,
		EnableInterIntraCompound:    b.Sequence.EnableInterIntraCompound,
		SwitchableMotionMode:        b.TileInfo.SwitchableMotionMode,
		AllowWarpedMotion:           b.FrameMode.AllowWarpedMotion,
		EnableMaskedCompound:        b.Sequence.EnableMaskedCompound,
		EnableDistWtdCompound:       b.Sequence.EnableJNTComp,
		EnableOrderHint:             b.Sequence.EnableOrderHint,
		UseRefFrameMVS:              b.TileInfo.UseRefFrameMVS,
		TemporalMVSampleUnavailable: b.TileInfo.UseRefFrameMVS,
		OrderHintBits:               b.Sequence.OrderHintBits,
		CurrentOrderHint:            b.FrameHeader.OrderHint,
		TemporalMVs:                 b.TemporalMVs,
		CurrentMVFrame:              b.CurrentMVFrame,
		SkipModeRefs: [2]tile.ReferenceFrame{
			tile.ReferenceFrame(b.SkipMode.RefFrameIdx[0]),
			tile.ReferenceFrame(b.SkipMode.RefFrameIdx[1]),
		},
	}, nil
}

func frameWorkBlockLoopGlobalMotion(params parser.GlobalMotionParams, allowHighPrecisionMV bool, forceIntegerMV bool) ([parser.InterRefsPerFrame]motion.Vector, [parser.InterRefsPerFrame]parser.GlobalMotionType) {
	var mvs [parser.InterRefsPerFrame]motion.Vector
	var types [parser.InterRefsPerFrame]parser.GlobalMotionType
	for i, ref := range params.Ref {
		types[i] = ref.Type
		if ref.Type == parser.GlobalMotionTranslation {
			mv := motion.Vector{
				Row: ref.Matrix[0] >> 13,
				Col: ref.Matrix[1] >> 13,
			}
			if forceIntegerMV {
				mv.Row = (mv.Row >> 3) << 3
				mv.Col = (mv.Col >> 3) << 3
			} else if !allowHighPrecisionMV {
				mv.Row = (mv.Row >> 1) << 1
				mv.Col = (mv.Col >> 1) << 1
			}
			mvs[i] = mv
		}
	}
	return mvs, types
}

func frameWorkBlockLoopRefSignBias(enabled bool, bits uint8, current uint32, refs [parser.InterRefsPerFrame]uint32) ([parser.InterRefsPerFrame]bool, error) {
	var bias [parser.InterRefsPerFrame]bool
	if !enabled {
		return bias, nil
	}
	for i, ref := range refs {
		distance, err := frameWorkRelativeOrderHint(bits, ref, current)
		if err != nil {
			return bias, err
		}
		bias[i] = distance > 0
	}
	return bias, nil
}

func frameWorkBlockLoopRefFrameSide(enabled bool, bits uint8, current uint32, refs [parser.InterRefsPerFrame]uint32) ([parser.InterRefsPerFrame]int8, error) {
	var side [parser.InterRefsPerFrame]int8
	if !enabled {
		return side, nil
	}
	for i, ref := range refs {
		distance, err := frameWorkRelativeOrderHint(bits, ref, current)
		if err != nil {
			return side, err
		}
		if distance > 0 {
			side[i] = 1
		} else if ref == current {
			side[i] = -1
		}
	}
	return side, nil
}

// JobBlockLoopContextRootColumns returns the number of caller-owned above-edge
// carrier slots needed by Jobs[index]'s block-loop request.
func (b FrameWorkBatch) JobBlockLoopContextRootColumns(index int) (int, error) {
	region, err := b.JobRegion(index)
	if err != nil {
		return 0, err
	}
	root := tile.RootBlockLevel(b.Sequence.Use128x128Superblock)
	rootSize := uint32(root.Size4x4())
	if rootSize == 0 || region.MIColEnd <= region.MIColStart {
		return 0, ErrInvalidBatch
	}
	return int((region.MIColEnd - region.MIColStart + rootSize - 1) / rootSize), nil
}

// DecodeAndReconstructJobResiduals walks one tile job's blocks, invokes the
// caller's prediction hook, decodes residual TXBs, and reconstructs each decoded
// TXB into the batch output frame.
func (b FrameWorkBatch) DecodeAndReconstructJobResiduals(index int, state *tile.DecodeState, cdfs FrameWorkTileResidualCDFs, scratch *FrameWorkTileResidualScratch, req FrameWorkTileResidualRequest) (FrameWorkTileResidualStats, error) {
	if state == nil || scratch == nil || (req.Transforms == nil && !req.UseDefaultTransforms) {
		return FrameWorkTileResidualStats{}, ErrInvalidBatch
	}
	scratch.stats = FrameWorkTileResidualStats{}
	if req.CDEFIndexMap == nil {
		req.CDEFIndexMap = b.CDEFIndexMap
	}
	if req.LoopFilterMap == nil {
		req.LoopFilterMap = b.LoopFilterMap
	}

	loopCDFs := cdfs.Loop
	if loopCDFs.Transform == nil {
		loopCDFs.Transform = cdfs.Coeff.Transform
	}
	if loopCDFs.Coeff == nil {
		loopCDFs.Coeff = cdfs.Coeff.Coeff
	}
	loopReq := req.Loop
	if loopReq.ContextCarrier == nil && scratch.LoopContext.Above != nil {
		loopReq.ContextCarrier = &scratch.LoopContext
	}
	loopReq.DecodeCoefficients = true
	var restoration FrameWorkTileRestorationRequest
	readRestoration := false
	if req.Restoration != nil {
		restoration = *req.Restoration
		readRestoration = true
	}
	scratch.controller = frameWorkTileResidualLoopController{
		batch:                  b,
		index:                  index,
		state:                  state,
		cdfs:                   cdfs,
		scratch:                scratch,
		req:                    req,
		stats:                  &scratch.stats,
		userBeforeSuperblock:   loopReq.BeforeSuperblock,
		userBeforeCoefficients: loopReq.BeforeCoefficients,
		userCoeffVisitor:       loopReq.CoeffVisitor,
		restoration:            restoration,
		readRestoration:        readRestoration,
	}
	if loopReq.BeforeSuperblock != nil || readRestoration {
		loopReq.BeforeSuperblock = scratch.controller.BeforeSuperblock
	}

	loopStats, err := tile.DecodeBlockLoopWithCoeffController(state, loopCDFs, &scratch.Loop, loopReq, &scratch.controller, func(visit tile.BlockLoopVisit) error {
		if !visit.CoefficientsValid {
			return ErrInvalidBatch
		}
		frameWorkAccumulateResidualStats(&scratch.stats, visit.Coefficients.TotalStats())
		if req.LoopFilterMap != nil {
			if err := req.LoopFilterMap.MarkBlock(visit, state); err != nil {
				return err
			}
		}
		if req.AfterBlock != nil {
			return req.AfterBlock(visit)
		}
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
	userBeforeSuperblock   tile.BlockLoopSuperblockVisitor
	restoration            FrameWorkTileRestorationRequest
	readRestoration        bool

	pendingCFLPrediction bool
	cflPredictionDone    bool
	cflVisit             tile.BlockLoopVisit
}

func (c *frameWorkTileResidualLoopController) BeforeSuperblock(visit tile.BlockLoopSuperblockVisit) error {
	if c.userBeforeSuperblock != nil {
		if err := c.userBeforeSuperblock(visit); err != nil {
			return err
		}
	}
	if !c.readRestoration {
		return nil
	}
	n, err := c.batch.ReadRestorationUnitsForSuperblock(c.index, c.state, c.cdfs.Restoration, &c.restoration, visit)
	if err != nil {
		return err
	}
	c.stats.RestorationUnits += n
	return nil
}

func (c *frameWorkTileResidualLoopController) BeforeBlockCoefficients(visit tile.BlockLoopVisit) error {
	if c.userBeforeCoefficients != nil {
		if err := c.userBeforeCoefficients(visit); err != nil {
			return err
		}
	}
	if c.req.CDEFIndexMap != nil {
		if err := c.req.CDEFIndexMap.MarkBlock(c.batch.CDEF, visit); err != nil {
			return err
		}
	}
	c.pendingCFLPrediction = false
	c.cflPredictionDone = false
	c.cflVisit = tile.BlockLoopVisit{}
	if err := c.predictBeforeCoefficients(visit); err != nil {
		return err
	}
	if visit.Prefix.SkipTransform {
		if err := c.predictDeferredCFLChroma(); err != nil {
			return err
		}
	}
	if visit.Prefix.SkipTransform {
		c.stats.SkippedBlocks++
	} else {
		c.stats.CoefficientBlocks++
	}
	return nil
}

func (c *frameWorkTileResidualLoopController) predictBeforeCoefficients(visit tile.BlockLoopVisit) error {
	if c.req.Predict != nil {
		if err := c.req.Predict(visit); err != nil {
			return fmt.Errorf("predict callback block=%+v: %w", visit.Block, err)
		}
		c.stats.Predictions++
		return nil
	}
	if c.req.PredictionScratch == nil {
		return nil
	}
	if visit.Prediction.Valid && visit.Prediction.Intra && !visit.Prefix.SkipTransform {
		if frameWorkVisitUsesCFL(visit) {
			c.pendingCFLPrediction = true
			c.cflVisit = visit
		}
		return nil
	}
	if frameWorkVisitUsesCFL(visit) {
		if err := c.batch.PredictBlockLumaIntra(c.index, visit, &c.req.PredictionScratch.Intra); err != nil {
			return fmt.Errorf("predict cfl luma block=%+v: %w", visit.Block, err)
		}
		c.pendingCFLPrediction = true
		c.cflVisit = visit
		c.stats.Predictions++
		return nil
	}
	if err := c.batch.PredictBlock(c.index, visit, c.req.PredictionScratch); err != nil {
		return fmt.Errorf("predict block=%+v prediction=%+v prefix=%+v: %w", visit.Block, visit.Prediction, visit.Prefix, err)
	}
	c.stats.Predictions++
	return nil
}

func (c *frameWorkTileResidualLoopController) predictDeferredCFLChroma() error {
	if !c.pendingCFLPrediction || c.cflPredictionDone {
		return nil
	}
	if err := c.batch.PredictBlockChromaCFL(c.index, c.cflVisit, &c.scratch.CFL); err != nil {
		return fmt.Errorf("predict cfl chroma block=%+v: %w", c.cflVisit.Block, err)
	}
	c.cflPredictionDone = true
	return nil
}

func (c *frameWorkTileResidualLoopController) SelectBlockCoeffRequest(visit tile.BlockLoopVisit) (tile.BlockCoeffRequest, error) {
	transforms, err := c.selectBlockTransforms(visit)
	if err != nil {
		return tile.BlockCoeffRequest{}, err
	}
	qIndex := c.state.CurrentBaseQIdx
	_, lossless, err := c.batch.BlockQIndex(qIndex, visit.SegmentID)
	if err != nil {
		return tile.BlockCoeffRequest{}, err
	}
	transformSelect := transforms.TransformSelect
	if transforms.ReadIntraTX {
		if !visit.Prediction.Valid || !visit.Prediction.Intra {
			return tile.BlockCoeffRequest{}, ErrInvalidBatch
		}
		chromaMode := visit.Prediction.ChromaMode
		if !chromaMode.Valid() {
			chromaMode = tile.ChromaIntraModeDC
		}
		c.scratch.IntraTX.Reset(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, visit.Prefix.SkipTransform, lossless, qIndex, visit.Prediction.LumaMode, chromaMode)
		transformSelect = &c.scratch.IntraTX
	} else if transforms.ReadInterTX {
		c.scratch.InterTX.ResetForColor(c.state, c.cdfs.TransformType, c.batch.FrameMode.ReducedTxSet, visit.Prefix.SkipTransform, lossless, c.batch.Sequence.ColorConfig)
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

func (c *frameWorkTileResidualLoopController) selectBlockTransforms(visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	if c.req.Transforms != nil {
		return c.req.Transforms(visit)
	}
	if c.req.UseDefaultTransforms {
		return c.batch.ReadBlockTransforms(c.state, visit)
	}
	return FrameWorkBlockTransforms{}, ErrInvalidBatch
}

func (c *frameWorkTileResidualLoopController) VisitBlockCoeff(visit tile.BlockLoopVisit, block tile.BlockCoeffBlock) error {
	if c.req.Predict == nil && c.req.PredictionScratch != nil && visit.Prediction.Valid && visit.Prediction.Intra && !visit.Prefix.SkipTransform {
		if block.Plane != 0 && frameWorkVisitUsesCFL(visit) {
			if err := c.predictDeferredCFLChroma(); err != nil {
				return err
			}
		} else {
			if err := c.batch.PredictBlockIntraCoeff(c.index, visit, block, &c.req.PredictionScratch.Intra); err != nil {
				return fmt.Errorf("predict intra txb plane=%d block=%+v: %w", block.Plane, block.Block, err)
			}
			c.stats.Predictions++
		}
	} else if block.Plane != 0 {
		if err := c.predictDeferredCFLChroma(); err != nil {
			return err
		}
	}
	if err := c.batch.ReconstructBlockCoeff(c.index, FrameWorkBlockCoeffReconstruction{
		Visit:           visit.Block,
		Block:           block,
		Transform:       block.Transform,
		CurrentQIndex:   c.state.CurrentBaseQIdx,
		SegmentID:       visit.SegmentID,
		Int32Scratch:    c.req.Int32Scratch,
		ResidualScratch: c.req.ResidualScratch,
	}); err != nil {
		return fmt.Errorf("reconstruct plane=%d block=%+v tx=%d: %w", block.Plane, block.Block, block.Transform, err)
	}
	c.stats.Residuals++
	if c.userCoeffVisitor != nil {
		return c.userCoeffVisitor(visit, block)
	}
	return nil
}

func frameWorkVisitUsesCFL(visit tile.BlockLoopVisit) bool {
	return visit.Prediction.Valid &&
		visit.Prediction.Intra &&
		visit.Prediction.ChromaModeValid &&
		visit.Prediction.ChromaMode == tile.ChromaIntraModeCFL
}

func (b FrameWorkBatch) ReadBlockTransforms(state *tile.DecodeState, visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	if visit.Prediction.Valid && visit.Prediction.Intra {
		return b.ReadIntraBlockTransforms(state, visit)
	}
	return b.ReadInterBlockTransforms(state, visit)
}

func (b FrameWorkBatch) ReadIntraBlockTransforms(state *tile.DecodeState, visit tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) {
	if state == nil || !visit.Prediction.Valid || !visit.Prediction.Intra {
		return FrameWorkBlockTransforms{}, ErrInvalidBatch
	}
	if _, _, err := b.BlockQIndex(state.CurrentBaseQIdx, visit.SegmentID); err != nil {
		return FrameWorkBlockTransforms{}, err
	}
	return FrameWorkBlockTransforms{
		Inter:       false,
		Luma:        transform.TypeDCTDCT,
		Chroma:      [2]transform.Type{transform.TypeDCTDCT, transform.TypeDCTDCT},
		ReadIntraTX: true,
	}, nil
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
