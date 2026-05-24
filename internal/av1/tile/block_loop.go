package tile

import (
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// BlockLoopCDFs groups the caller-owned entropy state used by the block syntax
// loop.
type BlockLoopCDFs struct {
	Partition *PartitionCDFs
	Mode      *BlockModeCDFs
	Intra     *IntraModeCDFs
	InterRef  *InterRefCDFs
	InterMode *InterModeCDFs
	MV        *MVCDFs
	Interp    *InterpFilterCDFs
	Motion    *MotionModeCDFs
	Blend     *CompoundBlendCDFs
	Transform *TransformCDFs
	Coeff     *CoeffCDFs
	Delta     DeltaCDFs
}

// BlockLoopScratch is per-superblock scratch for recursive partition and mode
// context traversal.
type BlockLoopScratch struct {
	Partition PartitionContext
	Mode      BlockModeContext
	CDEF      CDEFIndexContext
	Coeff     BlockCoeffScratch
	CoeffCtx  CoeffEntropyContext
}

// BlockLoopContextCarrier holds caller-owned edge contexts when a block-loop
// request walks more than one root block. Above must contain one slot per root
// column in the request; Left carries the previous root in the current root row.
// Keeping this carrier optional preserves the single-root zero-value path.
type BlockLoopContextCarrier struct {
	Above []BlockLoopRootAboveContext
	Left  BlockLoopRootLeftContext
}

type BlockLoopRootAboveContext struct {
	Partition [MaxPartitionSlots]uint8
	mode      blockModeAboveContext
	Coeff     [3][MaxBlockModeSlots]uint8
}

type BlockLoopRootLeftContext struct {
	Partition [MaxPartitionSlots]uint8
	mode      blockModeLeftContext
	Coeff     [3][MaxBlockModeSlots]uint8
}

type blockModeAboveContext struct {
	Skip        [MaxBlockModeSlots]uint8
	SkipMode    [MaxBlockModeSlots]uint8
	SegmentPred [MaxBlockModeSlots]uint8
	Intra       [MaxBlockModeSlots]uint8
	Mode        [MaxBlockModeSlots]IntraMode
	TxIntra     [MaxBlockModeSlots]uint8
	Tx          [MaxBlockModeSlots]uint8
	Ref         [2][MaxBlockModeSlots]ReferenceFrame
	Compound    [MaxBlockModeSlots]uint8
	CompGroup   [MaxBlockModeSlots]uint8
	CompIndex   [MaxBlockModeSlots]uint8
	InterMotion [MaxBlockModeSlots]InterMotionResult
	MotionValid [MaxBlockModeSlots]uint8
	Interp      [MaxBlockModeSlots]motion.InterpFilters
	InterpValid [MaxBlockModeSlots]uint8
	BlockSize   [MaxBlockModeSlots]BlockSize
}

type blockModeLeftContext struct {
	Skip        [MaxBlockModeSlots]uint8
	SkipMode    [MaxBlockModeSlots]uint8
	SegmentPred [MaxBlockModeSlots]uint8
	Intra       [MaxBlockModeSlots]uint8
	Mode        [MaxBlockModeSlots]IntraMode
	TxIntra     [MaxBlockModeSlots]uint8
	Tx          [MaxBlockModeSlots]uint8
	Ref         [2][MaxBlockModeSlots]ReferenceFrame
	Compound    [MaxBlockModeSlots]uint8
	CompGroup   [MaxBlockModeSlots]uint8
	CompIndex   [MaxBlockModeSlots]uint8
	InterMotion [MaxBlockModeSlots]InterMotionResult
	MotionValid [MaxBlockModeSlots]uint8
	Interp      [MaxBlockModeSlots]motion.InterpFilters
	InterpValid [MaxBlockModeSlots]uint8
	BlockSize   [MaxBlockModeSlots]BlockSize
}

// BlockLoopRequest carries frame and tile state needed by the syntax loop.
type BlockLoopRequest struct {
	Walk BlockWalkRequest

	ContextCarrier *BlockLoopContextCarrier

	SkipMode     parser.SkipModeParams
	CDEF         parser.CDEFParams
	Segmentation parser.SegmentationParams
	Delta        parser.DeltaParams

	SBSizeMIB  uint8
	Monochrome bool

	CurrentSegmentMap  []uint8
	PreviousSegmentMap []uint8
	SegmentMapStride   int

	FrameType             parser.FrameType
	AllowIntrabc          bool
	ReferenceMode         parser.ReferenceMode
	SkipModeRefs          [2]ReferenceFrame
	DecodePredictionModes bool
	DecodeInterModes      bool
	DecodeMotionVectors   bool
	DecodeInterIntra      bool
	DecodeMotionModes     bool
	DecodeCompoundBlend   bool
	DecodeCoefficients    bool
	EnableFilterIntra     bool

	GlobalMVs            [referenceFrameCount]motion.Vector
	GlobalMotion         [referenceFrameCount]parser.WarpedMotionParams
	GlobalMotionTypes    [referenceFrameCount]parser.GlobalMotionType
	RefSignBias          [referenceFrameCount]bool
	ReferenceOrderHints  [referenceFrameCount]uint32
	ScaledReferences     [referenceFrameCount]bool
	InterpolationFilter  parser.InterpolationFilter
	EnableDualFilter     bool
	AllowHighPrecisionMV bool
	ForceIntegerMV       bool

	EnableInterIntraCompound bool
	SwitchableMotionMode     bool
	AllowWarpedMotion        bool
	OverlappableNeighbors    int
	NumProjRef               int

	EnableMaskedCompound  bool
	EnableDistWtdCompound bool
	EnableOrderHint       bool
	OrderHintBits         uint8
	CurrentOrderHint      uint32

	Color               parser.ColorConfig
	TransformMode       parser.TransformMode
	Lossless            bool
	LumaTransformType   transform.Type
	ChromaTransformType [2]transform.Type
	TransformSelect     CoeffTransformSelector
	EOBMultiContext     [3]int
	CoeffController     BlockLoopCoeffController
	CoeffRequest        BlockLoopCoeffRequestSelector
	BeforeCoefficients  BlockLoopVisitor
	CoeffVisitor        BlockLoopCoeffVisitor
}

// BlockLoopVisit is reported after partition, segmentation, prefix, and delta
// syntax have been decoded for one leaf block.
type BlockLoopVisit struct {
	Block BlockVisit

	SegmentID         uint8
	Segment           parser.SegmentData
	SegmentPredicted  bool
	Prefix            BlockModeResult
	Prediction        BlockPredictionModeResult
	Coefficients      BlockCoeffResult
	CoefficientsValid bool
	Delta             BlockDeltaContext
}

type BlockLoopStats struct {
	PartitionReads      int
	Blocks              int
	SegmentPredictions  int
	SegmentIDs          int
	Prefixes            int
	PredictionModes     int
	IntraModes          int
	InterEntries        int
	InterReferences     int
	InterModes          int
	RefMVStacks         int
	DRLIndices          int
	InterMVReferences   int
	MotionVectors       int
	MVResiduals         int
	InterpFilters       int
	InterIntras         int
	MotionModes         int
	CompoundBlends      int
	CoefficientBlocks   int
	CoefficientTXBs     int
	CoefficientNonZero  int
	CoefficientAllZero  int
	CoefficientEOBTotal int
	DeltaReads          int
}

type BlockLoopVisitor func(BlockLoopVisit) error

type BlockLoopCoeffController interface {
	BeforeBlockCoefficients(BlockLoopVisit) error
	SelectBlockCoeffRequest(BlockLoopVisit) (BlockCoeffRequest, error)
	VisitBlockCoeff(BlockLoopVisit, BlockCoeffBlock) error
}

type BlockLoopCoeffRequestSelector func(BlockLoopVisit) (BlockCoeffRequest, error)

type BlockLoopCoeffVisitor func(BlockLoopVisit, BlockCoeffBlock) error

type noBlockLoopCoeffController struct{}

func (noBlockLoopCoeffController) BeforeBlockCoefficients(BlockLoopVisit) error {
	return ErrInvalidDecodeState
}

func (noBlockLoopCoeffController) SelectBlockCoeffRequest(BlockLoopVisit) (BlockCoeffRequest, error) {
	return BlockCoeffRequest{}, ErrInvalidDecodeState
}

func (noBlockLoopCoeffController) VisitBlockCoeff(BlockLoopVisit, BlockCoeffBlock) error {
	return ErrInvalidDecodeState
}

// DecodeBlockLoop walks every root block in req and decodes the shared
// per-block syntax prefix needed before intra/inter and transform decode.
func (s *DecodeState) DecodeBlockLoop(cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, visit BlockLoopVisitor) (BlockLoopStats, error) {
	if req.CoeffController != nil {
		return decodeBlockLoopWithCoeffController(s, cdfs, scratch, req, req.CoeffController, true, visit)
	}
	return decodeBlockLoopWithCoeffController(s, cdfs, scratch, req, noBlockLoopCoeffController{}, false, visit)
}

// DecodeBlockLoopWithCoeffController is DecodeBlockLoop with a call-scoped
// coefficient controller. Passing the controller as an argument lets hot callers
// keep controller state on the stack instead of storing it in BlockLoopRequest.
func DecodeBlockLoopWithCoeffController[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, visit BlockLoopVisitor) (BlockLoopStats, error) {
	return decodeBlockLoopWithCoeffController(s, cdfs, scratch, req, coeffController, true, visit)
}

func decodeBlockLoopWithCoeffController[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, hasCoeffController bool, visit BlockLoopVisitor) (BlockLoopStats, error) {
	if s == nil || scratch == nil || cdfs.Partition == nil || cdfs.Mode == nil || visit == nil {
		return BlockLoopStats{}, ErrInvalidDecodeState
	}
	if err := validateBlockLoopRequest(req, hasCoeffController); err != nil {
		return BlockLoopStats{}, err
	}

	var stats BlockLoopStats
	rootSize := uint32(req.Walk.Root.Size4x4())
	for miRow := req.Walk.MIRowStart; miRow < req.Walk.MIRowEnd; miRow += rootSize {
		for miCol := req.Walk.MIColStart; miCol < req.Walk.MIColEnd; miCol += rootSize {
			rootColIndex := int((miCol - req.Walk.MIColStart) / rootSize)
			if err := blockLoopLoadRootContext(scratch, req.ContextCarrier, rootColIndex, miRow > req.Walk.neighborMIRowStart(), miCol > req.Walk.neighborMIColStart()); err != nil {
				return stats, err
			}
			rootReq := BlockWalkRequest{
				Root:               req.Walk.Root,
				MIColStart:         miCol,
				MIRowStart:         miRow,
				MIColEnd:           minUint32(req.Walk.MIColEnd, miCol+rootSize),
				MIRowEnd:           minUint32(req.Walk.MIRowEnd, miRow+rootSize),
				UseNeighborBounds:  true,
				NeighborMIColStart: req.Walk.neighborMIColStart(),
				NeighborMIRowStart: req.Walk.neighborMIRowStart(),
			}
			walkStats, err := walkBlocks(&scratch.Partition, rootReq, func(level BlockLevel, context int, haveRight bool, haveBottom bool) (Partition, error) {
				return s.ReadPartition(cdfs.Partition, level, context, haveRight, haveBottom)
			}, func(block BlockVisit) error {
				visitInfo, err := decodeBlockLoopVisitWithCoeffController(s, cdfs, scratch, req, coeffController, hasCoeffController, block)
				if err != nil {
					return err
				}
				stats.Blocks++
				stats.Prefixes++
				if visitInfo.SegmentPredicted {
					stats.SegmentPredictions++
				}
				if req.Segmentation.Enabled && req.Segmentation.UpdateMap {
					stats.SegmentIDs++
				}
				if visitInfo.Prediction.Valid {
					stats.PredictionModes++
					if visitInfo.Prediction.Intra {
						stats.IntraModes++
					} else {
						stats.InterEntries++
						if visitInfo.Prediction.InterReferencesValid {
							stats.InterReferences++
						}
						if visitInfo.Prediction.InterModeValid {
							stats.InterModes++
						}
						if visitInfo.Prediction.ReferenceMVStackValid {
							stats.RefMVStacks++
						}
						if visitInfo.Prediction.DRLIndexValid {
							stats.DRLIndices++
						}
						if visitInfo.Prediction.InterMVReferencesValid {
							stats.InterMVReferences++
						}
						if visitInfo.Prediction.InterMotionValid {
							stats.MotionVectors++
							for _, valid := range visitInfo.Prediction.MVResidualValid {
								if valid {
									stats.MVResiduals++
								}
							}
							stats.InterpFilters += visitInfo.Prediction.InterpFilterReads
							if visitInfo.Prediction.InterIntraValid {
								stats.InterIntras++
							}
							if visitInfo.Prediction.MotionModeValid {
								stats.MotionModes++
							}
							if visitInfo.Prediction.CompoundBlendValid {
								stats.CompoundBlends++
							}
						}
					}
				}
				if visitInfo.CoefficientsValid {
					stats.CoefficientBlocks++
					total := visitInfo.Coefficients.TotalStats()
					stats.CoefficientTXBs += total.TXBs
					stats.CoefficientNonZero += total.NonZero
					stats.CoefficientAllZero += total.AllZero
					stats.CoefficientEOBTotal += total.EOBTotal
				}
				readDelta, err := shouldReadBlockDelta(visitInfo.Delta)
				if err != nil {
					return err
				}
				if readDelta && req.Delta.DeltaQPresent {
					stats.DeltaReads++
				}
				return visit(visitInfo)
			})
			stats.PartitionReads += walkStats.PartitionReads
			if err != nil {
				return stats, err
			}
			if err := blockLoopStoreRootContext(scratch, req.ContextCarrier, rootColIndex); err != nil {
				return stats, err
			}
		}
	}
	return stats, nil
}

func blockLoopLoadRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int, haveTop bool, haveLeft bool) error {
	scratch.Partition = PartitionContext{}
	scratch.Mode = BlockModeContext{}
	scratch.CDEF.Reset()
	scratch.CoeffCtx = CoeffEntropyContext{}
	if carrier == nil {
		return nil
	}
	if rootColIndex < 0 || rootColIndex >= len(carrier.Above) {
		return ErrInvalidDecodeState
	}
	if haveTop {
		above := carrier.Above[rootColIndex]
		scratch.Partition.Above = above.Partition
		scratch.Mode.AboveSkip = above.mode.Skip
		scratch.Mode.AboveSkipMode = above.mode.SkipMode
		scratch.Mode.AboveSegmentPred = above.mode.SegmentPred
		scratch.Mode.AboveIntra = above.mode.Intra
		scratch.Mode.AboveMode = above.mode.Mode
		scratch.Mode.AboveTxIntra = above.mode.TxIntra
		scratch.Mode.AboveTx = above.mode.Tx
		scratch.Mode.AboveRef = above.mode.Ref
		scratch.Mode.AboveCompound = above.mode.Compound
		scratch.Mode.AboveCompGroup = above.mode.CompGroup
		scratch.Mode.AboveCompIndex = above.mode.CompIndex
		scratch.Mode.AboveInterMotion = above.mode.InterMotion
		scratch.Mode.AboveMotionValid = above.mode.MotionValid
		scratch.Mode.AboveInterp = above.mode.Interp
		scratch.Mode.AboveInterpValid = above.mode.InterpValid
		scratch.Mode.AboveBlockSize = above.mode.BlockSize
		for plane := 0; plane < 3; plane++ {
			scratch.CoeffCtx.Above[plane] = above.Coeff[plane]
		}
	}
	if haveLeft {
		left := carrier.Left
		scratch.Partition.Left = left.Partition
		scratch.Mode.LeftSkip = left.mode.Skip
		scratch.Mode.LeftSkipMode = left.mode.SkipMode
		scratch.Mode.LeftSegmentPred = left.mode.SegmentPred
		scratch.Mode.LeftIntra = left.mode.Intra
		scratch.Mode.LeftMode = left.mode.Mode
		scratch.Mode.LeftTxIntra = left.mode.TxIntra
		scratch.Mode.LeftTx = left.mode.Tx
		scratch.Mode.LeftRef = left.mode.Ref
		scratch.Mode.LeftCompound = left.mode.Compound
		scratch.Mode.LeftCompGroup = left.mode.CompGroup
		scratch.Mode.LeftCompIndex = left.mode.CompIndex
		scratch.Mode.LeftInterMotion = left.mode.InterMotion
		scratch.Mode.LeftMotionValid = left.mode.MotionValid
		scratch.Mode.LeftInterp = left.mode.Interp
		scratch.Mode.LeftInterpValid = left.mode.InterpValid
		scratch.Mode.LeftBlockSize = left.mode.BlockSize
		for plane := 0; plane < 3; plane++ {
			scratch.CoeffCtx.Left[plane] = left.Coeff[plane]
		}
	}
	return nil
}

func blockLoopStoreRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int) error {
	if carrier == nil {
		return nil
	}
	if rootColIndex < 0 || rootColIndex >= len(carrier.Above) {
		return ErrInvalidDecodeState
	}
	above := &carrier.Above[rootColIndex]
	above.Partition = scratch.Partition.Above
	above.mode.Skip = scratch.Mode.AboveSkip
	above.mode.SkipMode = scratch.Mode.AboveSkipMode
	above.mode.SegmentPred = scratch.Mode.AboveSegmentPred
	above.mode.Intra = scratch.Mode.AboveIntra
	above.mode.Mode = scratch.Mode.AboveMode
	above.mode.TxIntra = scratch.Mode.AboveTxIntra
	above.mode.Tx = scratch.Mode.AboveTx
	above.mode.Ref = scratch.Mode.AboveRef
	above.mode.Compound = scratch.Mode.AboveCompound
	above.mode.CompGroup = scratch.Mode.AboveCompGroup
	above.mode.CompIndex = scratch.Mode.AboveCompIndex
	above.mode.InterMotion = scratch.Mode.AboveInterMotion
	above.mode.MotionValid = scratch.Mode.AboveMotionValid
	above.mode.Interp = scratch.Mode.AboveInterp
	above.mode.InterpValid = scratch.Mode.AboveInterpValid
	above.mode.BlockSize = scratch.Mode.AboveBlockSize
	for plane := 0; plane < 3; plane++ {
		above.Coeff[plane] = scratch.CoeffCtx.Above[plane]
	}

	left := &carrier.Left
	left.Partition = scratch.Partition.Left
	left.mode.Skip = scratch.Mode.LeftSkip
	left.mode.SkipMode = scratch.Mode.LeftSkipMode
	left.mode.SegmentPred = scratch.Mode.LeftSegmentPred
	left.mode.Intra = scratch.Mode.LeftIntra
	left.mode.Mode = scratch.Mode.LeftMode
	left.mode.TxIntra = scratch.Mode.LeftTxIntra
	left.mode.Tx = scratch.Mode.LeftTx
	left.mode.Ref = scratch.Mode.LeftRef
	left.mode.Compound = scratch.Mode.LeftCompound
	left.mode.CompGroup = scratch.Mode.LeftCompGroup
	left.mode.CompIndex = scratch.Mode.LeftCompIndex
	left.mode.InterMotion = scratch.Mode.LeftInterMotion
	left.mode.MotionValid = scratch.Mode.LeftMotionValid
	left.mode.Interp = scratch.Mode.LeftInterp
	left.mode.InterpValid = scratch.Mode.LeftInterpValid
	left.mode.BlockSize = scratch.Mode.LeftBlockSize
	for plane := 0; plane < 3; plane++ {
		left.Coeff[plane] = scratch.CoeffCtx.Left[plane]
	}
	return nil
}

func decodeBlockLoopVisitWithCoeffController[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, hasCoeffController bool, block BlockVisit) (BlockLoopVisit, error) {
	ctx := &scratch.Mode
	cdef := &scratch.CDEF
	segmentID := uint8(0)
	segment := defaultSegmentData()
	segmentPredicted := false
	var err error
	if req.Segmentation.Enabled && (!req.Segmentation.UpdateMap || req.Segmentation.Data.Preskip) {
		segmentID, segmentPredicted, segment, err = s.decodeBlockSegment(cdfs.Mode, ctx, req, block, false)
		if err != nil {
			return BlockLoopVisit{}, err
		}
	}

	prefixReq := BlockModeRequest{
		Size:                block.Size,
		SkipMode:            req.SkipMode,
		CDEF:                req.CDEF,
		SegmentationEnabled: req.Segmentation.Enabled,
		Segment:             segment,
		X4:                  block.X4,
		Y4:                  block.Y4,
	}
	prefix, err := s.readBlockModePrefix(cdfs.Mode, ctx, cdef, prefixReq, segmentPredicted)
	if err != nil {
		return BlockLoopVisit{}, err
	}

	if req.Segmentation.Enabled && req.Segmentation.UpdateMap && !req.Segmentation.Data.Preskip {
		segmentID, segmentPredicted, segment, err = s.decodeBlockSegment(cdfs.Mode, ctx, req, block, prefix.SkipTransform)
		if err != nil {
			return BlockLoopVisit{}, err
		}
		if segmentPredicted {
			prefix.SegmentPredicted = true
			if err := ctx.Mark(block.Size, block.X4, block.Y4, prefix); err != nil {
				return BlockLoopVisit{}, err
			}
		}
	}

	delta := BlockDeltaContext{
		MICol:          block.MICol,
		MIRow:          block.MIRow,
		SBSizeMIB:      req.SBSizeMIB,
		FullSuperblock: block.VisibleW4 == req.SBSizeMIB && block.VisibleH4 == req.SBSizeMIB,
		SkipTransform:  prefix.SkipTransform,
		Monochrome:     req.Monochrome,
	}
	if err := s.ReadBlockDeltas(req.Delta, delta, cdfs.Delta); err != nil {
		return BlockLoopVisit{}, err
	}

	var prediction BlockPredictionModeResult
	if req.DecodePredictionModes {
		prediction, err = s.decodeBlockPredictionMode(cdfs, ctx, req, block, prefix, segmentID, segment)
		if err != nil {
			return BlockLoopVisit{}, err
		}
	}

	visit := BlockLoopVisit{
		Block:            block,
		SegmentID:        segmentID,
		Segment:          segment,
		SegmentPredicted: segmentPredicted,
		Prefix:           prefix,
		Prediction:       prediction,
		Delta:            delta,
	}
	if req.DecodeCoefficients {
		if hasCoeffController {
			if err := coeffController.BeforeBlockCoefficients(visit); err != nil {
				return BlockLoopVisit{}, err
			}
		} else if req.BeforeCoefficients != nil {
			if err := req.BeforeCoefficients(visit); err != nil {
				return BlockLoopVisit{}, err
			}
		}
		coeffReq, err := blockLoopCoeffRequest(req, coeffController, hasCoeffController, visit)
		if err != nil {
			return BlockLoopVisit{}, err
		}
		coeffVisit := req.CoeffVisitor
		if coeffVisit == nil {
			coeffVisit = discardBlockLoopCoeff
		}
		coefficients, err := s.DecodeBlockCoefficients(BlockCoeffCDFs{
			Transform: cdfs.Transform,
			Coeff:     cdfs.Coeff,
		}, ctx, &scratch.CoeffCtx, &scratch.Coeff, coeffReq, func(block BlockCoeffBlock) error {
			if hasCoeffController {
				return coeffController.VisitBlockCoeff(visit, block)
			}
			return coeffVisit(visit, block)
		})
		if err != nil {
			return BlockLoopVisit{}, err
		}
		visit.Coefficients = coefficients
		visit.CoefficientsValid = true
	}
	return visit, nil
}

func (s *DecodeState) decodeBlockPredictionMode(cdfs BlockLoopCDFs, ctx *BlockModeContext, req BlockLoopRequest, block BlockVisit, prefix BlockModeResult, segmentID uint8, segment parser.SegmentData) (BlockPredictionModeResult, error) {
	intra, err := s.ReadIntraFlag(cdfs.Intra, ctx, IntraFlagRequest{
		FrameType:           req.FrameType,
		AllowIntrabc:        req.AllowIntrabc,
		SkipMode:            prefix.SkipMode,
		SegmentationEnabled: req.Segmentation.Enabled,
		Segment:             segment,
		X4:                  block.X4,
		Y4:                  block.Y4,
		HaveTop:             block.HaveTop,
		HaveLeft:            block.HaveLeft,
	})
	if err != nil {
		return BlockPredictionModeResult{}, err
	}

	result := BlockPredictionModeResult{Valid: true, Intra: intra, LumaMode: IntraModeDC}
	if !intra {
		refs, err := s.ReadInterReferences(cdfs.InterRef, ctx, InterReferenceRequest{
			Size:                block.Size,
			ReferenceMode:       req.ReferenceMode,
			SkipMode:            prefix.SkipMode,
			SkipModeRefs:        req.SkipModeRefs,
			SegmentationEnabled: req.Segmentation.Enabled,
			Segment:             segment,
			X4:                  block.X4,
			Y4:                  block.Y4,
			HaveTop:             block.HaveTop,
			HaveLeft:            block.HaveLeft,
		})
		if err != nil {
			return BlockPredictionModeResult{}, err
		}
		result.InterReferences = refs
		result.InterReferencesValid = true
		globalMVs := blockReferenceGlobalMVsForBlock(refs, req.GlobalMVs, req.GlobalMotion, req.AllowHighPrecisionMV, req.ForceIntegerMV, block)
		if req.DecodeInterModes {
			stack, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
				Size:        block.Size,
				References:  refs,
				X4:          block.X4,
				Y4:          block.Y4,
				HaveTop:     block.HaveTop,
				HaveLeft:    block.HaveLeft,
				GlobalMVs:   globalMVs,
				RefSignBias: req.RefSignBias,
			})
			if err != nil {
				return BlockPredictionModeResult{}, err
			}
			mode, err := s.ReadBlockInterMode(cdfs.InterMode, InterModeRequest{
				Compound:            refs.Compound,
				SkipMode:            prefix.SkipMode,
				SegmentationEnabled: req.Segmentation.Enabled,
				Segment:             segment,
				ModeContext:         stack.ModeContext,
			})
			if err != nil {
				return BlockPredictionModeResult{}, err
			}
			drlReq, err := stack.Stack.DRLRequestForMode(mode)
			if err != nil {
				return BlockPredictionModeResult{}, err
			}
			drlIndex, err := s.ReadDRLIndex(cdfs.InterMode, drlReq)
			if err != nil {
				return BlockPredictionModeResult{}, err
			}
			result.InterMode = mode
			result.InterModeValid = true
			result.ReferenceMVStack = stack
			result.ReferenceMVStackValid = true
			result.DRLIndex = drlIndex
			result.DRLIndexValid = drlReq.usesNewMV() || drlReq.usesNearMV()
			if !interModeUsesGlobalOnly(mode) {
				mvRefs, err := stack.Stack.ResolveInterMVReferences(mode, drlIndex, req.AllowHighPrecisionMV, req.ForceIntegerMV)
				if err != nil {
					return BlockPredictionModeResult{}, err
				}
				result.InterMVReferences = mvRefs
				result.InterMVReferencesValid = true
			}
			if req.DecodeMotionVectors {
				motionResult, err := s.ReadInterMotion(cdfs.MV, InterMotionRequest{
					References:   refs,
					Mode:         mode,
					ReferenceMVs: result.InterMVReferences,
					GlobalMVs:    globalMVs,
					Precision:    MVPrecision(req.AllowHighPrecisionMV, req.ForceIntegerMV),
				})
				if err != nil {
					return BlockPredictionModeResult{}, err
				}
				result.InterMotion = motionResult.Motion
				result.InterMotionValid = true
				result.MVResiduals = motionResult.Residuals
				result.MVResidualValid = motionResult.ResidualValid

				motionMode := MotionModeTranslation
				if req.DecodeInterIntra && !refs.Compound {
					interIntra, err := s.ReadInterIntra(cdfs.Blend, InterIntraRequest{
						Size:                     block.Size,
						Mode:                     mode.Mode,
						EnableInterIntraCompound: req.EnableInterIntraCompound,
						SkipMode:                 prefix.SkipMode,
						Compound:                 refs.Compound,
					})
					if err != nil {
						return BlockPredictionModeResult{}, err
					}
					result.InterIntra = interIntra
					result.InterIntraValid = true
				}
				if req.DecodeMotionModes {
					motionMode, err = s.ReadMotionMode(cdfs.Motion, MotionModeRequest{
						Size:                  block.Size,
						Mode:                  mode.Mode,
						Compound:              refs.Compound,
						SkipMode:              prefix.SkipMode,
						InterIntra:            result.InterIntra.Enabled,
						SwitchableMotionMode:  req.SwitchableMotionMode,
						AllowWarpedMotion:     req.AllowWarpedMotion,
						ForceIntegerMV:        req.ForceIntegerMV,
						GlobalMotionType:      blockReferenceGlobalMotionType(refs, req.GlobalMotionTypes),
						ScaledReference:       blockReferenceScaled(refs, req.ScaledReferences),
						OverlappableNeighbors: req.OverlappableNeighbors,
						NumProjRef:            req.NumProjRef,
					})
					if err != nil {
						return BlockPredictionModeResult{}, err
					}
					result.MotionMode = motionMode
					result.MotionModeValid = true
				}
				if req.DecodeCompoundBlend && refs.Compound {
					blend, err := s.ReadCompoundBlend(cdfs.Blend, ctx, CompoundBlendRequest{
						Size:                  block.Size,
						Compound:              refs.Compound,
						SkipMode:              prefix.SkipMode,
						MotionMode:            motionMode,
						EnableMaskedCompound:  req.EnableMaskedCompound,
						EnableDistWtdCompound: req.EnableDistWtdCompound,
						EnableOrderHint:       req.EnableOrderHint,
						OrderHintBits:         req.OrderHintBits,
						CurrentOrderHint:      req.CurrentOrderHint,
						RefOrderHint:          blockReferenceOrderHints(refs, req.ReferenceOrderHints),
						X4:                    block.X4,
						Y4:                    block.Y4,
						HaveTop:               block.HaveTop,
						HaveLeft:              block.HaveLeft,
					})
					if err != nil {
						return BlockPredictionModeResult{}, err
					}
					result.CompoundBlend = blend
					result.CompoundBlendValid = true
				}
				filters, filterReads, err := s.ReadInterpFilters(cdfs.Interp, ctx, InterpFilterRequest{
					FrameFilter:      req.InterpolationFilter,
					EnableDualFilter: req.EnableDualFilter,
					Size:             block.Size,
					References:       refs,
					Mode:             mode,
					MotionMode:       motionMode,
					GlobalTypes:      blockReferenceGlobalMotionTypes(refs, req.GlobalMotionTypes),
					SkipMode:         prefix.SkipMode,
					X4:               block.X4,
					Y4:               block.Y4,
					HaveTop:          block.HaveTop,
					HaveLeft:         block.HaveLeft,
				})
				if err != nil {
					return BlockPredictionModeResult{}, err
				}
				result.InterpFilters = filters
				result.InterpFiltersValid = true
				result.InterpFilterReads = filterReads
				if err := ctx.MarkInterMotion(block.Size, block.X4, block.Y4, motionResult.Motion); err != nil {
					return BlockPredictionModeResult{}, err
				}
				if err := ctx.MarkInterFilters(block.Size, block.X4, block.Y4, refs, filters); err != nil {
					return BlockPredictionModeResult{}, err
				}
				if result.CompoundBlendValid {
					if err := ctx.MarkCompoundBlend(block.Size, block.X4, block.Y4, result.CompoundBlend); err != nil {
						return BlockPredictionModeResult{}, err
					}
				}
				return result, nil
			}
		}
		if err := ctx.MarkInter(block.Size, block.X4, block.Y4, refs); err != nil {
			return BlockPredictionModeResult{}, err
		}
		return result, nil
	}

	mode, err := s.ReadLumaIntraMode(cdfs.Intra, ctx, LumaIntraModeRequest{
		FrameType: req.FrameType,
		Size:      block.Size,
		X4:        block.X4,
		Y4:        block.Y4,
	})
	if err != nil {
		return BlockPredictionModeResult{}, err
	}
	if err := ctx.MarkIntra(block.Size, block.X4, block.Y4, true, mode); err != nil {
		return BlockPredictionModeResult{}, err
	}
	result.LumaMode = mode
	lumaAngleDelta, err := s.ReadIntraAngleDelta(cdfs.Intra, IntraAngleDeltaRequest{
		Size: block.Size,
		Mode: mode,
	})
	if err != nil {
		return BlockPredictionModeResult{}, err
	}
	result.LumaAngleDelta = lumaAngleDelta
	if HasChromaBlock(TransformTreeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}, req.Color) {
		lossless := req.Lossless || req.Segmentation.Lossless[segmentID]
		cflAllowed, err := ChromaIntraCFLAllowed(block.Size, req.Color, lossless)
		if err != nil {
			return BlockPredictionModeResult{}, err
		}
		chromaMode, cflAlpha, err := s.ReadChromaIntraMode(cdfs.Intra, ChromaIntraModeRequest{
			Size:       block.Size,
			LumaMode:   mode,
			CFLAllowed: cflAllowed,
		})
		if err != nil {
			return BlockPredictionModeResult{}, err
		}
		result.ChromaMode = chromaMode
		result.ChromaModeValid = true
		if chromaMode == ChromaIntraModeCFL {
			result.CFLAlpha = cflAlpha
			result.CFLAlphaValid = true
		} else {
			chromaLumaMode, err := chromaMode.LumaMode()
			if err != nil {
				return BlockPredictionModeResult{}, err
			}
			chromaAngleDelta, err := s.ReadIntraAngleDelta(cdfs.Intra, IntraAngleDeltaRequest{
				Size: block.Size,
				Mode: chromaLumaMode,
			})
			if err != nil {
				return BlockPredictionModeResult{}, err
			}
			result.ChromaAngleDelta = chromaAngleDelta
		}
	}
	filterMode, filterValid, err := s.ReadFilterIntraMode(cdfs.Intra, FilterIntraRequest{
		EnableFilterIntra: req.EnableFilterIntra,
		Size:              block.Size,
		LumaMode:          mode,
		PaletteYSize:      0,
	})
	if err != nil {
		return BlockPredictionModeResult{}, err
	}
	result.FilterIntraMode = filterMode
	result.FilterIntraValid = filterValid
	return result, nil
}

func (s *DecodeState) decodeBlockSegment(cdfs *BlockModeCDFs, ctx *BlockModeContext, req BlockLoopRequest, block BlockVisit, skip bool) (uint8, bool, parser.SegmentData, error) {
	if !req.Segmentation.Enabled {
		return 0, false, defaultSegmentData(), nil
	}
	if req.Segmentation.Data.LastActiveID < 0 {
		if err := fillBlockSegmentID(req, block, 0); err != nil {
			return 0, false, parser.SegmentData{}, err
		}
		return 0, false, req.Segmentation.Data.Segments[0], nil
	}
	if !req.Segmentation.UpdateMap {
		id, err := previousBlockSegmentID(req, block)
		if err != nil {
			return 0, false, parser.SegmentData{}, err
		}
		if err := fillBlockSegmentID(req, block, id); err != nil {
			return 0, false, parser.SegmentData{}, err
		}
		return id, false, req.Segmentation.Data.Segments[id], nil
	}

	predicted := false
	if req.Segmentation.TemporalUpdate && !skip {
		var err error
		predicted, err = s.ReadSegmentPrediction(cdfs, ctx, block.X4, block.Y4)
		if err != nil {
			return 0, false, parser.SegmentData{}, err
		}
	}
	if predicted {
		id, err := previousBlockSegmentID(req, block)
		if err != nil {
			return 0, false, parser.SegmentData{}, err
		}
		if err := fillBlockSegmentID(req, block, id); err != nil {
			return 0, false, parser.SegmentData{}, err
		}
		return id, true, req.Segmentation.Data.Segments[id], nil
	}

	if len(req.CurrentSegmentMap) == 0 {
		return 0, false, parser.SegmentData{}, ErrInvalidDecodeState
	}
	pred, context, err := PredictCurrentSegmentID(req.CurrentSegmentMap, req.SegmentMapStride,
		int(block.MICol), int(block.MIRow), block.HaveTop, block.HaveLeft)
	if err != nil {
		return 0, false, parser.SegmentData{}, err
	}
	id, err := s.ReadSegmentID(cdfs, pred, context, req.Segmentation.Data.LastActiveID, skip)
	if err != nil {
		return 0, false, parser.SegmentData{}, err
	}
	if err := fillBlockSegmentID(req, block, id); err != nil {
		return 0, false, parser.SegmentData{}, err
	}
	return id, false, req.Segmentation.Data.Segments[id], nil
}

func (s *DecodeState) readBlockModePrefix(cdfs *BlockModeCDFs, ctx *BlockModeContext, cdef *CDEFIndexContext, req BlockModeRequest, segmentPredicted bool) (BlockModeResult, error) {
	skipMode, err := s.ReadSkipMode(cdfs, ctx, req)
	if err != nil {
		return BlockModeResult{}, err
	}
	skip, err := s.ReadSkipTransform(cdfs, ctx, req, skipMode)
	if err != nil {
		return BlockModeResult{}, err
	}
	cdefIndex, err := s.ReadCDEFIndexForBlock(req.CDEF, cdef, req.Size, req.X4, req.Y4, skip)
	if err != nil {
		return BlockModeResult{}, err
	}
	result := BlockModeResult{
		SegmentPredicted: segmentPredicted,
		SkipMode:         skipMode,
		SkipTransform:    skip,
		CDEFIndex:        cdefIndex,
	}
	if err := ctx.Mark(req.Size, req.X4, req.Y4, result); err != nil {
		return BlockModeResult{}, err
	}
	return result, nil
}

func validateBlockLoopRequest(req BlockLoopRequest, hasCoeffController bool) error {
	if !req.Walk.Root.Valid() || req.SBSizeMIB == 0 ||
		req.Walk.MIColEnd <= req.Walk.MIColStart ||
		req.Walk.MIRowEnd <= req.Walk.MIRowStart {
		return ErrInvalidDecodeState
	}
	if req.DecodeInterModes && !req.DecodePredictionModes {
		return ErrInvalidDecodeState
	}
	if req.DecodeMotionVectors && !req.DecodeInterModes {
		return ErrInvalidDecodeState
	}
	if (req.DecodeInterIntra || req.DecodeMotionModes || req.DecodeCompoundBlend) && !req.DecodeMotionVectors {
		return ErrInvalidDecodeState
	}
	if req.DecodeCoefficients && !req.DecodePredictionModes && !hasCoeffController && req.CoeffRequest == nil {
		return ErrInvalidDecodeState
	}
	rootSize := uint32(req.Walk.Root.Size4x4())
	if rootSize == 0 || req.Walk.MIColStart%rootSize != 0 || req.Walk.MIRowStart%rootSize != 0 {
		return ErrInvalidDecodeState
	}
	if req.ContextCarrier != nil {
		rootCols := blockLoopRootColumns(req.Walk, rootSize)
		if rootCols <= 0 || len(req.ContextCarrier.Above) < rootCols {
			return ErrInvalidDecodeState
		}
	}
	if req.Segmentation.Enabled && (req.Segmentation.UpdateMap || len(req.PreviousSegmentMap) != 0) {
		if req.SegmentMapStride <= 0 {
			return ErrInvalidDecodeState
		}
	}
	return nil
}

func blockLoopRootColumns(req BlockWalkRequest, rootSize uint32) int {
	if rootSize == 0 || req.MIColEnd <= req.MIColStart {
		return 0
	}
	return int((req.MIColEnd - req.MIColStart + rootSize - 1) / rootSize)
}

func previousBlockSegmentID(req BlockLoopRequest, block BlockVisit) (uint8, error) {
	if len(req.PreviousSegmentMap) == 0 {
		return 0, nil
	}
	return MinPreviousSegmentID(req.PreviousSegmentMap, req.SegmentMapStride,
		int(block.MICol), int(block.MIRow), int(block.VisibleW4), int(block.VisibleH4))
}

func fillBlockSegmentID(req BlockLoopRequest, block BlockVisit, segmentID uint8) error {
	if len(req.CurrentSegmentMap) == 0 {
		return nil
	}
	return FillSegmentID(req.CurrentSegmentMap, req.SegmentMapStride,
		int(block.MICol), int(block.MIRow), int(block.VisibleW4), int(block.VisibleH4), segmentID)
}

func defaultSegmentData() parser.SegmentData {
	return parser.SegmentData{RefFrame: -1}
}

func blockLoopCoeffRequest[T BlockLoopCoeffController](req BlockLoopRequest, coeffController T, hasCoeffController bool, visit BlockLoopVisit) (BlockCoeffRequest, error) {
	if hasCoeffController {
		return coeffController.SelectBlockCoeffRequest(visit)
	}
	if req.CoeffRequest != nil {
		return req.CoeffRequest(visit)
	}
	if !visit.Prediction.Valid {
		return BlockCoeffRequest{}, ErrInvalidDecodeState
	}
	block := visit.Block
	return BlockCoeffRequest{
		Transform: TransformTreeRequest{
			Size:          block.Size,
			X4:            block.X4,
			Y4:            block.Y4,
			VisibleW4:     block.VisibleW4,
			VisibleH4:     block.VisibleH4,
			Color:         req.Color,
			TransformMode: req.TransformMode,
			Inter:         !visit.Prediction.Intra,
			SkipTransform: visit.Prefix.SkipTransform,
			Lossless:      req.Lossless,
		},
		LumaType:        req.LumaTransformType,
		ChromaType:      req.ChromaTransformType,
		TransformSelect: req.TransformSelect,
		EOBMultiContext: req.EOBMultiContext,
	}, nil
}

func discardBlockLoopCoeff(BlockLoopVisit, BlockCoeffBlock) error {
	return nil
}

func blockReferenceGlobalMVs(refs InterReferencesResult, global [referenceFrameCount]motion.Vector) [2]motion.Vector {
	var out [2]motion.Vector
	if refs.Ref[0].Valid() {
		out[0] = global[refs.Ref[0]]
	}
	if refs.Compound && refs.Ref[1].Valid() {
		out[1] = global[refs.Ref[1]]
	}
	return out
}

func blockReferenceGlobalMVsForBlock(refs InterReferencesResult, fallback [referenceFrameCount]motion.Vector, global [referenceFrameCount]parser.WarpedMotionParams, allowHighPrecisionMV bool, forceIntegerMV bool, block BlockVisit) [2]motion.Vector {
	out := blockReferenceGlobalMVs(refs, fallback)
	if refs.Ref[0].Valid() {
		if mv, ok := globalMotionVector(global[refs.Ref[0]], allowHighPrecisionMV, forceIntegerMV, block); ok {
			out[0] = mv
		}
	}
	if refs.Compound && refs.Ref[1].Valid() {
		if mv, ok := globalMotionVector(global[refs.Ref[1]], allowHighPrecisionMV, forceIntegerMV, block); ok {
			out[1] = mv
		}
	}
	return out
}

func globalMotionVector(params parser.WarpedMotionParams, allowHighPrecisionMV bool, forceIntegerMV bool, block BlockVisit) (motion.Vector, bool) {
	if !globalMotionParamsInitialized(params) {
		return motion.Vector{}, false
	}
	switch params.Type {
	case parser.GlobalMotionIdentity:
		return motion.Vector{}, true
	case parser.GlobalMotionTranslation:
		mv := motion.Vector{
			Row: params.Matrix[0] >> globalMotionTransOnlyPrecDiff,
			Col: params.Matrix[1] >> globalMotionTransOnlyPrecDiff,
		}
		if forceIntegerMV {
			mv.Row = globalMotionIntegerMVPrecision(mv.Row)
			mv.Col = globalMotionIntegerMVPrecision(mv.Col)
		}
		return mv, true
	case parser.GlobalMotionRotZoom, parser.GlobalMotionAffine:
		dims, ok := block.Size.Dimensions()
		if !ok {
			return motion.Vector{}, false
		}
		x := int64(block.MICol)*4 + int64(dims.W4)*2 - 1
		y := int64(block.MIRow)*4 + int64(dims.H4)*2 - 1
		mat := params.Matrix
		xc := int64(mat[2]-(1<<globalMotionWarpedModelPrecBits))*x + int64(mat[3])*y + int64(mat[0])
		yc := int64(mat[4])*x + int64(mat[5]-(1<<globalMotionWarpedModelPrecBits))*y + int64(mat[1])
		mv := motion.Vector{
			Row: globalMotionConvertToTransPrec(yc, allowHighPrecisionMV),
			Col: globalMotionConvertToTransPrec(xc, allowHighPrecisionMV),
		}
		if forceIntegerMV {
			mv.Row = globalMotionIntegerMVPrecision(mv.Row)
			mv.Col = globalMotionIntegerMVPrecision(mv.Col)
		}
		return mv, true
	default:
		return motion.Vector{}, false
	}
}

const (
	globalMotionWarpedModelPrecBits = 16
	globalMotionTransOnlyPrecDiff   = globalMotionWarpedModelPrecBits - 3
)

func globalMotionParamsInitialized(params parser.WarpedMotionParams) bool {
	if params.Type != parser.GlobalMotionIdentity {
		return true
	}
	return params.Matrix[2] == 1<<globalMotionWarpedModelPrecBits ||
		params.Matrix[5] == 1<<globalMotionWarpedModelPrecBits ||
		params.Matrix[0] != 0 ||
		params.Matrix[1] != 0 ||
		params.Matrix[3] != 0 ||
		params.Matrix[4] != 0
}

func globalMotionConvertToTransPrec(value int64, allowHighPrecisionMV bool) int32 {
	if allowHighPrecisionMV {
		return int32(globalMotionRoundPowerOfTwoSigned(value, globalMotionWarpedModelPrecBits-3))
	}
	return int32(globalMotionRoundPowerOfTwoSigned(value, globalMotionWarpedModelPrecBits-2)) * 2
}

func globalMotionRoundPowerOfTwoSigned(value int64, bits uint) int64 {
	if value < 0 {
		return -((-value + (1 << (bits - 1))) >> bits)
	}
	return (value + (1 << (bits - 1))) >> bits
}

func globalMotionIntegerMVPrecision(v int32) int32 {
	mod := v % 8
	if mod == 0 {
		return v
	}
	v -= mod
	if mod < 0 {
		if -mod > 4 {
			v -= 8
		}
		return v
	}
	if mod > 4 {
		v += 8
	}
	return v
}

func blockReferenceGlobalMotionType(refs InterReferencesResult, global [referenceFrameCount]parser.GlobalMotionType) parser.GlobalMotionType {
	if refs.Ref[0].Valid() {
		return global[refs.Ref[0]]
	}
	return parser.GlobalMotionIdentity
}

func blockReferenceGlobalMotionTypes(refs InterReferencesResult, global [referenceFrameCount]parser.GlobalMotionType) [2]parser.GlobalMotionType {
	var out [2]parser.GlobalMotionType
	if refs.Ref[0].Valid() {
		out[0] = global[refs.Ref[0]]
	}
	if refs.Compound && refs.Ref[1].Valid() {
		out[1] = global[refs.Ref[1]]
	}
	return out
}

func blockReferenceScaled(refs InterReferencesResult, scaled [referenceFrameCount]bool) bool {
	if refs.Ref[0].Valid() {
		return scaled[refs.Ref[0]]
	}
	return false
}

func blockReferenceOrderHints(refs InterReferencesResult, orderHints [referenceFrameCount]uint32) [2]uint32 {
	var out [2]uint32
	if refs.Ref[0].Valid() {
		out[0] = orderHints[refs.Ref[0]]
	}
	if refs.Compound && refs.Ref[1].Valid() {
		out[1] = orderHints[refs.Ref[1]]
	}
	return out
}

func interModeUsesGlobalOnly(mode InterModeResult) bool {
	if mode.Compound {
		return mode.CompoundMode == CompoundInterModeGlobalGlobal
	}
	return mode.Mode == InterModeGlobalMV
}
