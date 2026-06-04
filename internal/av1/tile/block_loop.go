package tile

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/entropy"
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
	Palette   PaletteModeScratch

	visit BlockLoopVisit
}

// BlockLoopContextCarrier holds caller-owned edge contexts when a block-loop
// request walks more than one root block. Above must contain one slot per root
// column in the request; Left carries the previous root in the current root row.
// Keeping this carrier optional preserves the single-root zero-value path.
type BlockLoopContextCarrier struct {
	Above           []BlockLoopRootAboveContext
	Left            BlockLoopRootLeftContext
	Diagonal        []diagonalCornerSlot
	PendingDiagonal []diagonalCornerSlot
}

type diagonalCornerSlot struct {
	InterMotion      [intrabcCrossSBHistory][intrabcCrossSBHistory]InterMotionResult
	MotionValid      [intrabcCrossSBHistory][intrabcCrossSBHistory]uint8
	BlockSize        [intrabcCrossSBHistory][intrabcCrossSBHistory]BlockSize
	BlockSizeVisited [intrabcCrossSBHistory][intrabcCrossSBHistory]uint8
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
	ChromaIntra [MaxBlockModeSlots]uint8
	ChromaMode  [MaxBlockModeSlots]ChromaIntraMode
	TxIntra     [MaxBlockModeSlots]uint8
	Tx          [MaxBlockModeSlots]uint8
	Ref         [2][MaxBlockModeSlots]ReferenceFrame
	Compound    [MaxBlockModeSlots]uint8
	CompGroup   [MaxBlockModeSlots]uint8
	CompIndex   [MaxBlockModeSlots]uint8
	// InterIntra carries the per-MI inter-intra status across superblock rows
	// so a warped-motion block at an SB top edge sees the SB-above neighbor's
	// inter-intra flag. libaom's frame-wide mi grid exposes ref_frame[1] =
	// INTRA_FRAME there, which av1_findSamples rejects as a warp sample source;
	// without this field the restored above context defaulted to 0 and goav1
	// over-collected the neighbor.
	InterIntra  [MaxBlockModeSlots]uint8
	InterMotion [MaxBlockModeSlots]InterMotionResult
	MotionValid [MaxBlockModeSlots]uint8
	Interp      [MaxBlockModeSlots]motion.InterpFilters
	InterpValid [MaxBlockModeSlots]uint8
	BlockSize   [MaxBlockModeSlots]BlockSize
	PaletteY    [MaxBlockModeSlots]paletteContext
	PaletteUV   [MaxBlockModeSlots]paletteContext

	// InterMotionHistory captures the bottom intrabcCrossSBHistory rows of the
	// prior SB's grid in row-major order: depth 0 is the bottom row of the
	// prior SB (= row immediately above the current SB), depth d is d+1 rows
	// above. Mirrors what libaom's frame-wide mi grid would expose for the IBC
	// outer mv-ref row scan reaching back into the prior SB.
	InterMotionHistory      [intrabcCrossSBHistory][MaxBlockModeSlots]InterMotionResult
	MotionValidHistory      [intrabcCrossSBHistory][MaxBlockModeSlots]uint8
	BlockSizeHistory        [intrabcCrossSBHistory][MaxBlockModeSlots]BlockSize
	BlockSizeVisitedHistory [intrabcCrossSBHistory][MaxBlockModeSlots]uint8
}

type blockModeLeftContext struct {
	Skip        [MaxBlockModeSlots]uint8
	SkipMode    [MaxBlockModeSlots]uint8
	SegmentPred [MaxBlockModeSlots]uint8
	Intra       [MaxBlockModeSlots]uint8
	Mode        [MaxBlockModeSlots]IntraMode
	ChromaIntra [MaxBlockModeSlots]uint8
	ChromaMode  [MaxBlockModeSlots]ChromaIntraMode
	TxIntra     [MaxBlockModeSlots]uint8
	Tx          [MaxBlockModeSlots]uint8
	Ref         [2][MaxBlockModeSlots]ReferenceFrame
	Compound    [MaxBlockModeSlots]uint8
	CompGroup   [MaxBlockModeSlots]uint8
	CompIndex   [MaxBlockModeSlots]uint8
	// InterIntra carries the per-MI inter-intra status across superblock columns
	// (see blockModeAboveContext.InterIntra for the rationale).
	InterIntra  [MaxBlockModeSlots]uint8
	InterMotion [MaxBlockModeSlots]InterMotionResult
	MotionValid [MaxBlockModeSlots]uint8
	Interp      [MaxBlockModeSlots]motion.InterpFilters
	InterpValid [MaxBlockModeSlots]uint8
	BlockSize   [MaxBlockModeSlots]BlockSize
	PaletteY    [MaxBlockModeSlots]paletteContext
	PaletteUV   [MaxBlockModeSlots]paletteContext

	// InterMotionHistory captures the rightmost intrabcCrossSBHistory columns
	// of the prior SB-to-left's grid: depth 0 is the rightmost column (=
	// column immediately to the left of the current SB), depth d is d+1
	// columns to the left. Mirrors libaom's frame mi grid for the IBC outer
	// mv-ref column scan reaching back into the prior SB.
	InterMotionHistory      [intrabcCrossSBHistory][MaxBlockModeSlots]InterMotionResult
	MotionValidHistory      [intrabcCrossSBHistory][MaxBlockModeSlots]uint8
	BlockSizeHistory        [intrabcCrossSBHistory][MaxBlockModeSlots]BlockSize
	BlockSizeVisitedHistory [intrabcCrossSBHistory][MaxBlockModeSlots]uint8
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

	FrameType               parser.FrameType
	AllowIntrabc            bool
	ReferenceMode           parser.ReferenceMode
	SkipModeRefs            [2]ReferenceFrame
	DecodePredictionModes   bool
	DecodeInterModes        bool
	DecodeMotionVectors     bool
	DecodeInterIntra        bool
	DecodeMotionModes       bool
	DecodeCompoundBlend     bool
	DecodeCoefficients      bool
	EnableFilterIntra       bool
	AllowScreenContentTools bool

	GlobalMVs            [referenceFrameCount]motion.Vector
	GlobalMotion         [referenceFrameCount]parser.WarpedMotionParams
	GlobalMotionTypes    [referenceFrameCount]parser.GlobalMotionType
	RefSignBias          [referenceFrameCount]bool
	RefFrameSide         [referenceFrameCount]int8
	ReferenceOrderHints  [referenceFrameCount]uint8
	ScaledReferences     [referenceFrameCount]bool
	InterpolationFilter  parser.InterpolationFilter
	EnableDualFilter     bool
	AllowHighPrecisionMV bool
	ForceIntegerMV       bool

	EnableInterIntraCompound bool
	SwitchableMotionMode     bool
	AllowWarpedMotion        bool

	EnableMaskedCompound  bool
	EnableDistWtdCompound bool
	EnableOrderHint       bool
	UseRefFrameMVS        bool
	// TemporalMVSampleUnavailable carries libaom's add_tpl_ref_mv() miss for
	// the current motion-field port stage. Full MFMV projection will replace
	// this frame-level unavailable marker with per-block temporal samples.
	TemporalMVSampleUnavailable bool
	OrderHintBits               uint8
	CurrentOrderHint            uint8
	TemporalMVs                 *TemporalMotionField
	CurrentMVFrame              *ReferenceMVFrame

	// FrameMIRows and FrameMICols are the frame's MI grid extent
	// (ALIGN_POWER_OF_TWO(dim, 3) >> MI_SIZE_LOG2), used to clamp ref-MV stack
	// candidates to the frame boundary exactly as libaom's clamp_mv_ref.
	FrameMIRows uint32
	FrameMICols uint32

	Color               parser.ColorConfig
	TransformMode       parser.TransformMode
	Lossless            bool
	LumaTransformType   transform.Type
	ChromaTransformType [2]transform.Type
	TransformSelect     CoeffTransformSelector
	EOBMultiContext     [3]uint8
	CoeffController     BlockLoopCoeffController
	CoeffRequest        BlockLoopCoeffRequestSelector
	BeforeSuperblock    BlockLoopSuperblockVisitor
	BeforeCoefficients  BlockLoopVisitor
	CoeffVisitor        BlockLoopCoeffVisitor
}

// BlockLoopSuperblockVisit is reported before partition syntax is decoded for
// one root superblock in the block loop.
type BlockLoopSuperblockVisit struct {
	MICol     uint32
	MIRow     uint32
	SBSizeMIB uint8
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
	PartitionReads      int32
	Blocks              int32
	SegmentPredictions  int32
	SegmentIDs          int32
	Prefixes            int32
	PredictionModes     int32
	IntraModes          int32
	InterEntries        int32
	InterReferences     int32
	InterModes          int32
	RefMVStacks         int32
	DRLIndices          int32
	InterMVReferences   int32
	MotionVectors       int32
	MVResiduals         int32
	InterpFilters       int32
	InterIntras         int32
	MotionModes         int32
	CompoundBlends      int32
	CoefficientBlocks   int32
	CoefficientTXBs     int32
	CoefficientNonZero  int32
	CoefficientAllZero  int32
	CoefficientEOBTotal int32
	DeltaReads          int32
}

type BlockLoopSuperblockVisitor func(BlockLoopSuperblockVisit) error

type BlockLoopVisitor func(BlockLoopVisit) error
type BlockLoopPointerVisitor func(*BlockLoopVisit) error

type BlockLoopCoeffController interface {
	BeforeBlockCoefficients(BlockLoopVisit) error
	SelectBlockCoeffRequest(BlockLoopVisit) (BlockCoeffRequest, error)
	VisitBlockCoeff(BlockLoopVisit, BlockCoeffBlock) error
	BeforeBlockCoefficientsPtr(*BlockLoopVisit) error
	SelectBlockCoeffRequestPtr(*BlockLoopVisit) (BlockCoeffRequest, error)
	VisitBlockCoeffPtr(*BlockLoopVisit, *BlockCoeffBlock) error
}

type BlockLoopCoeffRequestSelector func(BlockLoopVisit) (BlockCoeffRequest, error)

type BlockLoopCoeffVisitor func(BlockLoopVisit, BlockCoeffBlock) error

type noBlockLoopCoeffController struct{}

func (noBlockLoopCoeffController) BeforeBlockCoefficients(BlockLoopVisit) error {
	return ErrInvalidDecodeState
}

func (noBlockLoopCoeffController) BeforeBlockCoefficientsPtr(*BlockLoopVisit) error {
	return ErrInvalidDecodeState
}

func (noBlockLoopCoeffController) SelectBlockCoeffRequest(BlockLoopVisit) (BlockCoeffRequest, error) {
	return BlockCoeffRequest{}, ErrInvalidDecodeState
}

func (noBlockLoopCoeffController) SelectBlockCoeffRequestPtr(*BlockLoopVisit) (BlockCoeffRequest, error) {
	return BlockCoeffRequest{}, ErrInvalidDecodeState
}

func (noBlockLoopCoeffController) VisitBlockCoeff(BlockLoopVisit, BlockCoeffBlock) error {
	return ErrInvalidDecodeState
}

func (noBlockLoopCoeffController) VisitBlockCoeffPtr(*BlockLoopVisit, *BlockCoeffBlock) error {
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
	if visit == nil {
		return BlockLoopStats{}, ErrInvalidDecodeState
	}
	return decodeBlockLoopWithCoeffControllerPtr(s, cdfs, scratch, req, coeffController, true, func(block *BlockLoopVisit) error {
		return visit(*block)
	})
}

func decodeBlockLoopWithCoeffController[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, hasCoeffController bool, visit BlockLoopVisitor) (BlockLoopStats, error) {
	if visit == nil {
		return BlockLoopStats{}, ErrInvalidDecodeState
	}
	return decodeBlockLoopWithCoeffControllerPtr(s, cdfs, scratch, req, coeffController, hasCoeffController, func(block *BlockLoopVisit) error {
		return visit(*block)
	})
}

// DecodeBlockLoopWithCoeffControllerPtr is DecodeBlockLoopWithCoeffController
// with a pointer visitor for hot callers that only need to inspect the visit
// during the callback. BlockLoopVisit carries large prediction/coefficient
// side data, so the pointer path avoids one full visit copy per decoded block.
func DecodeBlockLoopWithCoeffControllerPtr[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, visit BlockLoopPointerVisitor) (BlockLoopStats, error) {
	return decodeBlockLoopWithCoeffControllerPtr(s, cdfs, scratch, req, coeffController, true, visit)
}

func decodeBlockLoopWithCoeffControllerPtr[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, hasCoeffController bool, visit BlockLoopPointerVisitor) (BlockLoopStats, error) {
	if s == nil || scratch == nil || cdfs.Partition == nil || cdfs.Mode == nil || visit == nil {
		return BlockLoopStats{}, ErrInvalidDecodeState
	}
	if err := validateBlockLoopRequest(req, hasCoeffController); err != nil {
		return BlockLoopStats{}, err
	}

	var stats BlockLoopStats
	rootSize := uint32(req.Walk.Root.Size4x4())
	ensureIntrabcDiagonalCarriers(req.ContextCarrier)
	for miRow := req.Walk.MIRowStart; miRow < req.Walk.MIRowEnd; miRow += rootSize {
		promotePendingDiagonalCarriers(req.ContextCarrier)
		for miCol := req.Walk.MIColStart; miCol < req.Walk.MIColEnd; miCol += rootSize {
			rootColIndex := int((miCol - req.Walk.MIColStart) / rootSize)
			if err := blockLoopLoadRootContext(scratch, req.ContextCarrier, rootColIndex, miRow > req.Walk.neighborMIRowStart(), miCol > req.Walk.neighborMIColStart(), req.SBSizeMIB); err != nil {
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
			if req.BeforeSuperblock != nil {
				if err := req.BeforeSuperblock(BlockLoopSuperblockVisit{
					MICol:     miCol,
					MIRow:     miRow,
					SBSizeMIB: req.SBSizeMIB,
				}); err != nil {
					return stats, err
				}
			}
			walkStats, err := walkBlocks(&scratch.Partition, rootReq, func(level BlockLevel, context int, haveRight bool, haveBottom bool) (Partition, error) {
				return s.ReadPartition(cdfs.Partition, level, context, haveRight, haveBottom)
			}, func(block BlockVisit) error {
				visitInfo, err := decodeBlockLoopVisitWithCoeffControllerPtr(s, cdfs, scratch, req, coeffController, hasCoeffController, block)
				if err != nil {
					return fmt.Errorf("decode block=%+v: %w", block, err)
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
							stats.InterpFilters += int32(visitInfo.Prediction.InterpFilterReads)
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
					stats.CoefficientTXBs += int32(total.TXBs)
					stats.CoefficientNonZero += int32(total.NonZero)
					stats.CoefficientAllZero += int32(total.AllZero)
					stats.CoefficientEOBTotal += int32(total.EOBTotal)
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
			stats.PartitionReads += int32(walkStats.PartitionReads)
			if err != nil {
				return stats, fmt.Errorf("walk root col=%d row=%d: %w", rootColIndex, miRow, err)
			}
			if err := blockLoopStoreRootContext(scratch, req.ContextCarrier, rootColIndex, req.SBSizeMIB); err != nil {
				return stats, fmt.Errorf("store root context col=%d row=%d: %w", rootColIndex, miRow, err)
			}
			captureDiagonalCornerToPending(req.ContextCarrier, rootColIndex+1, &scratch.Mode, req.SBSizeMIB)
		}
	}
	return stats, nil
}

func blockLoopLoadRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int, haveTop bool, haveLeft bool, sbSizeMIB uint8) error {
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
		above := &carrier.Above[rootColIndex]
		scratch.Partition.Above = above.Partition
		scratch.Mode.AboveSkip = above.mode.Skip
		scratch.Mode.AboveSkipMode = above.mode.SkipMode
		scratch.Mode.AboveSegmentPred = above.mode.SegmentPred
		scratch.Mode.AboveIntra = above.mode.Intra
		scratch.Mode.AboveMode = above.mode.Mode
		scratch.Mode.AboveChromaIntra = above.mode.ChromaIntra
		scratch.Mode.AboveChromaMode = above.mode.ChromaMode
		scratch.Mode.AboveTxIntra = above.mode.TxIntra
		scratch.Mode.AboveTx = above.mode.Tx
		scratch.Mode.AboveRef = above.mode.Ref
		scratch.Mode.AboveCompound = above.mode.Compound
		scratch.Mode.AboveCompGroup = above.mode.CompGroup
		scratch.Mode.AboveCompIndex = above.mode.CompIndex
		scratch.Mode.AboveInterIntra = above.mode.InterIntra
		scratch.Mode.AboveInterMotion = above.mode.InterMotion
		scratch.Mode.AboveMotionValid = above.mode.MotionValid
		scratch.Mode.SBTopInterMotion = above.mode.InterMotion
		scratch.Mode.SBTopMotionValid = above.mode.MotionValid
		scratch.Mode.SBTopBlockSize = above.mode.BlockSize
		scratch.Mode.SBTopInterMotionGrid = above.mode.InterMotionHistory
		scratch.Mode.SBTopMotionValidGrid = above.mode.MotionValidHistory
		scratch.Mode.SBTopBlockSizeGrid = above.mode.BlockSizeHistory
		scratch.Mode.SBTopBlockSizeVisitedGrid = above.mode.BlockSizeVisitedHistory
		scratch.Mode.AboveInterp = above.mode.Interp
		scratch.Mode.AboveInterpValid = above.mode.InterpValid
		scratch.Mode.AboveBlockSize = above.mode.BlockSize
		scratch.Mode.AbovePaletteY = above.mode.PaletteY
		scratch.Mode.AbovePaletteUV = above.mode.PaletteUV
		for plane := range 3 {
			scratch.CoeffCtx.Above[plane] = above.Coeff[plane]
		}
		// Fill cells past the current SB column (slots SBSizeMIB..) with the
		// neighboring SB-above-and-right's left-edge data so cross-SB lookups
		// such as topright (which reads frame mi col >= mi_col + W4) and outer
		// row scans extending past column SBSizeMIB-1 see libaom's frame-wide
		// mi grid content rather than zeroed slots from a smaller SB layout.
		extendAboveContextFromRightCarrier(&scratch.Mode, carrier, rootColIndex+1, sbSizeMIB)
	}
	if haveLeft {
		left := &carrier.Left
		scratch.Partition.Left = left.Partition
		scratch.Mode.LeftSkip = left.mode.Skip
		scratch.Mode.LeftSkipMode = left.mode.SkipMode
		scratch.Mode.LeftSegmentPred = left.mode.SegmentPred
		scratch.Mode.LeftIntra = left.mode.Intra
		scratch.Mode.LeftMode = left.mode.Mode
		scratch.Mode.LeftChromaIntra = left.mode.ChromaIntra
		scratch.Mode.LeftChromaMode = left.mode.ChromaMode
		scratch.Mode.LeftTxIntra = left.mode.TxIntra
		scratch.Mode.LeftTx = left.mode.Tx
		scratch.Mode.LeftRef = left.mode.Ref
		scratch.Mode.LeftCompound = left.mode.Compound
		scratch.Mode.LeftCompGroup = left.mode.CompGroup
		scratch.Mode.LeftCompIndex = left.mode.CompIndex
		scratch.Mode.LeftInterIntra = left.mode.InterIntra
		scratch.Mode.LeftInterMotion = left.mode.InterMotion
		scratch.Mode.LeftMotionValid = left.mode.MotionValid
		scratch.Mode.SBLeftInterMotion = left.mode.InterMotion
		scratch.Mode.SBLeftMotionValid = left.mode.MotionValid
		scratch.Mode.SBLeftBlockSize = left.mode.BlockSize
		scratch.Mode.SBLeftInterMotionGrid = left.mode.InterMotionHistory
		scratch.Mode.SBLeftMotionValidGrid = left.mode.MotionValidHistory
		scratch.Mode.SBLeftBlockSizeGrid = left.mode.BlockSizeHistory
		scratch.Mode.SBLeftBlockSizeVisitedGrid = left.mode.BlockSizeVisitedHistory
		scratch.Mode.LeftInterp = left.mode.Interp
		scratch.Mode.LeftInterpValid = left.mode.InterpValid
		scratch.Mode.LeftBlockSize = left.mode.BlockSize
		scratch.Mode.LeftPaletteY = left.mode.PaletteY
		scratch.Mode.LeftPaletteUV = left.mode.PaletteUV
		for plane := range 3 {
			scratch.CoeffCtx.Left[plane] = left.Coeff[plane]
		}
	}
	if haveTop && haveLeft && rootColIndex >= 0 && rootColIndex < len(carrier.Diagonal) {
		scratch.Mode.SBDiagonalInterMotionGrid = carrier.Diagonal[rootColIndex].InterMotion
		scratch.Mode.SBDiagonalMotionValidGrid = carrier.Diagonal[rootColIndex].MotionValid
		scratch.Mode.SBDiagonalBlockSizeGrid = carrier.Diagonal[rootColIndex].BlockSize
		scratch.Mode.SBDiagonalBlockSizeVisitedGrid = carrier.Diagonal[rootColIndex].BlockSizeVisited
	}
	return nil
}

func blockLoopStoreRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int, sbSizeMIB uint8) error {
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
	above.mode.ChromaIntra = scratch.Mode.AboveChromaIntra
	above.mode.ChromaMode = scratch.Mode.AboveChromaMode
	above.mode.TxIntra = scratch.Mode.AboveTxIntra
	above.mode.Tx = scratch.Mode.AboveTx
	above.mode.Ref = scratch.Mode.AboveRef
	above.mode.Compound = scratch.Mode.AboveCompound
	above.mode.CompGroup = scratch.Mode.AboveCompGroup
	above.mode.CompIndex = scratch.Mode.AboveCompIndex
	above.mode.InterIntra = scratch.Mode.AboveInterIntra
	above.mode.InterMotion = scratch.Mode.AboveInterMotion
	above.mode.MotionValid = scratch.Mode.AboveMotionValid
	above.mode.Interp = scratch.Mode.AboveInterp
	above.mode.InterpValid = scratch.Mode.AboveInterpValid
	above.mode.BlockSize = scratch.Mode.AboveBlockSize
	above.mode.PaletteY = scratch.Mode.AbovePaletteY
	above.mode.PaletteUV = scratch.Mode.AbovePaletteUV
	captureAboveCrossSBHistory(&above.mode, &scratch.Mode, sbSizeMIB)
	for plane := range 3 {
		above.Coeff[plane] = scratch.CoeffCtx.Above[plane]
	}

	left := &carrier.Left
	left.Partition = scratch.Partition.Left
	left.mode.Skip = scratch.Mode.LeftSkip
	left.mode.SkipMode = scratch.Mode.LeftSkipMode
	left.mode.SegmentPred = scratch.Mode.LeftSegmentPred
	left.mode.Intra = scratch.Mode.LeftIntra
	left.mode.Mode = scratch.Mode.LeftMode
	left.mode.ChromaIntra = scratch.Mode.LeftChromaIntra
	left.mode.ChromaMode = scratch.Mode.LeftChromaMode
	left.mode.TxIntra = scratch.Mode.LeftTxIntra
	left.mode.Tx = scratch.Mode.LeftTx
	left.mode.Ref = scratch.Mode.LeftRef
	left.mode.Compound = scratch.Mode.LeftCompound
	left.mode.CompGroup = scratch.Mode.LeftCompGroup
	left.mode.CompIndex = scratch.Mode.LeftCompIndex
	left.mode.InterIntra = scratch.Mode.LeftInterIntra
	left.mode.InterMotion = scratch.Mode.LeftInterMotion
	left.mode.MotionValid = scratch.Mode.LeftMotionValid
	left.mode.Interp = scratch.Mode.LeftInterp
	left.mode.InterpValid = scratch.Mode.LeftInterpValid
	left.mode.BlockSize = scratch.Mode.LeftBlockSize
	left.mode.PaletteY = scratch.Mode.LeftPaletteY
	left.mode.PaletteUV = scratch.Mode.LeftPaletteUV
	captureLeftCrossSBHistory(&left.mode, &scratch.Mode, sbSizeMIB)
	for plane := range 3 {
		left.Coeff[plane] = scratch.CoeffCtx.Left[plane]
	}
	return nil
}

// ensureIntrabcDiagonalCarriers lazily allocates Diagonal/PendingDiagonal so
// callers may leave them nil. Allocation matches len(Above).
func ensureIntrabcDiagonalCarriers(carrier *BlockLoopContextCarrier) {
	if carrier == nil {
		return
	}
	PreallocBlockLoopContextCarrierScratch(carrier, len(carrier.Above))
}

// PreallocBlockLoopContextCarrierScratch primes the diagonal edge carriers used
// while walking multiple root blocks so callers can keep block-loop execution
// allocation-free.
func PreallocBlockLoopContextCarrierScratch(carrier *BlockLoopContextCarrier, rootColumns int) {
	if carrier == nil || rootColumns <= 0 {
		return
	}
	if len(carrier.Diagonal) < rootColumns {
		carrier.Diagonal = make([]diagonalCornerSlot, rootColumns)
	}
	if len(carrier.PendingDiagonal) < rootColumns {
		carrier.PendingDiagonal = make([]diagonalCornerSlot, rootColumns)
	}
}

// promotePendingDiagonalCarriers swaps the previous row's PendingDiagonal
// entries into the live Diagonal slots consumed at root-col load time. Called
// at the start of each root row so the diagonal-up-left corner stored by the
// row that just finished is the one each new-row SB reads.
func promotePendingDiagonalCarriers(carrier *BlockLoopContextCarrier) {
	if carrier == nil {
		return
	}
	n := len(carrier.PendingDiagonal)
	if n == 0 {
		return
	}
	if len(carrier.Diagonal) < n {
		carrier.Diagonal = make([]diagonalCornerSlot, n)
	}
	for i := range n {
		carrier.Diagonal[i] = carrier.PendingDiagonal[i]
		carrier.PendingDiagonal[i] = diagonalCornerSlot{}
	}
}

// captureDiagonalCornerToPending snapshots the just-finished SB's bottom-right
// corner cells into PendingDiagonal[nextColIndex]. The next row's SB at that
// column will read this as its diagonal-up-left corner after the row-boundary
// promotion. Writing to nextColIndex (not the current column) avoids
// clobbering the live Diagonal slot the current row's right-neighbor still
// needs to consume.
func captureDiagonalCornerToPending(carrier *BlockLoopContextCarrier, nextColIndex int, mode *BlockModeContext, sbSizeMIB uint8) {
	if carrier == nil || mode == nil || sbSizeMIB == 0 {
		return
	}
	if nextColIndex < 0 || nextColIndex >= len(carrier.PendingDiagonal) {
		return
	}
	dst := &carrier.PendingDiagonal[nextColIndex]
	*dst = diagonalCornerSlot{}
	right := int(sbSizeMIB) - 1
	bottom := int(sbSizeMIB) - 1
	if right >= MaxBlockModeSlots {
		right = MaxBlockModeSlots - 1
	}
	if bottom >= MaxBlockModeSlots {
		bottom = MaxBlockModeSlots - 1
	}
	for d := range intrabcCrossSBHistory {
		row := bottom - d
		if row < 0 {
			break
		}
		for e := range intrabcCrossSBHistory {
			col := right - e
			if col < 0 {
				break
			}
			dst.InterMotion[d][e] = mode.GridInterMotion[row][col]
			dst.MotionValid[d][e] = mode.GridMotionValid[row][col]
			dst.BlockSize[d][e] = mode.GridBlockSize[row][col]
			dst.BlockSizeVisited[d][e] = mode.GridBlockSizeVisited[row][col]
		}
	}
}

// extendAboveContextFromRightCarrier fills the per-SB AboveInterMotion and
// SBTopInterMotion(Grid) slots past the current SB column with the left edge
// of the SB above-and-to-the-right. libaom's frame-wide mi grid lets top-right
// (mi_col+width, mi_row-1) and far-right outer row scans read straight into
// the neighboring SB's interior; goav1's per-SB carrier needs an explicit
// merge of carrier.Above[col_idx+1] to mirror that.
func extendAboveContextFromRightCarrier(mode *BlockModeContext, carrier *BlockLoopContextCarrier, rightColIndex int, sbSizeMIB uint8) {
	if mode == nil || carrier == nil || sbSizeMIB == 0 {
		return
	}
	if rightColIndex <= 0 || rightColIndex >= len(carrier.Above) {
		return
	}
	right := &carrier.Above[rightColIndex]
	start := int(sbSizeMIB)
	for off := 0; off < int(sbSizeMIB); off++ {
		dst := start + off
		if dst >= MaxBlockModeSlots {
			break
		}
		if off >= MaxBlockModeSlots {
			break
		}
		mode.AboveInterMotion[dst] = right.mode.InterMotion[off]
		mode.AboveMotionValid[dst] = right.mode.MotionValid[off]
		mode.AboveBlockSize[dst] = right.mode.BlockSize[off]
		mode.AboveIntra[dst] = right.mode.Intra[off]
		mode.AboveMode[dst] = right.mode.Mode[off]
		mode.AboveChromaIntra[dst] = right.mode.ChromaIntra[off]
		mode.AboveChromaMode[dst] = right.mode.ChromaMode[off]
		mode.AboveRef[0][dst] = right.mode.Ref[0][off]
		mode.AboveRef[1][dst] = right.mode.Ref[1][off]
		mode.AboveCompound[dst] = right.mode.Compound[off]
		mode.AboveCompGroup[dst] = right.mode.CompGroup[off]
		mode.AboveCompIndex[dst] = right.mode.CompIndex[off]
		mode.AboveInterIntra[dst] = right.mode.InterIntra[off]
		mode.SBTopInterMotion[dst] = right.mode.InterMotion[off]
		mode.SBTopMotionValid[dst] = right.mode.MotionValid[off]
		mode.SBTopBlockSize[dst] = right.mode.BlockSize[off]
	}
	// SBTopRight* carries the right-SB-above bottom row indexed by offset
	// within the right SB. topRightInterMotion consults this when X4+W4 >=
	// MaxBlockModeSlots — the case sbSizeMIB == MaxBlockModeSlots (SB128x128)
	// where extendAboveContextFromRightCarrier's [sbSizeMIB..2*sbSizeMIB) write
	// range is fully out of slot range. Mirrors libaom's frame-wide mi grid
	// lookup xd->mi[-stride + xd->width] for cells past the current SB's right
	// edge. Loading the full MaxBlockModeSlots row works for any sbSizeMIB <=
	// MaxBlockModeSlots since the right-SB-above contexts use the same array
	// dimensions.
	for off := range MaxBlockModeSlots {
		mode.SBTopRightInterMotion[off] = right.mode.InterMotion[off]
		mode.SBTopRightMotionValid[off] = right.mode.MotionValid[off]
		mode.SBTopRightBlockSize[off] = right.mode.BlockSize[off]
		mode.SBTopRightIntra[off] = right.mode.Intra[off]
	}
	mode.SBTopRightValid = true
	// Depth-0 of the history grid mirrors SBTopInterMotion (= the bottom row
	// of the prior SB above us). The deeper history rows come from
	// right.mode.InterMotionHistory[d-1] et al., which capture the bottom
	// rows-of-the-right-SB and extend the cross-SB row scan beyond column
	// SBSizeMIB-1.
	for off := 0; off < int(sbSizeMIB); off++ {
		dst := start + off
		if dst >= MaxBlockModeSlots {
			break
		}
		if off >= MaxBlockModeSlots {
			break
		}
		mode.SBTopInterMotionGrid[0][dst] = right.mode.InterMotion[off]
		mode.SBTopMotionValidGrid[0][dst] = right.mode.MotionValid[off]
		mode.SBTopBlockSizeGrid[0][dst] = right.mode.BlockSize[off]
		for d := 1; d < intrabcCrossSBHistory; d++ {
			mode.SBTopInterMotionGrid[d][dst] = right.mode.InterMotionHistory[d-1][off]
			mode.SBTopMotionValidGrid[d][dst] = right.mode.MotionValidHistory[d-1][off]
			mode.SBTopBlockSizeGrid[d][dst] = right.mode.BlockSizeHistory[d-1][off]
		}
	}
}

// captureAboveCrossSBHistory copies the bottom intrabcCrossSBHistory rows of
// the just-finished superblock's grid into the carrier's above slot so the
// next superblock (the one immediately below) can serve IBC outer mv-ref row
// scans (rowOffset -3, -5) that read into the prior SB's interior. Depth 0 is
// the bottom row of the prior SB (= row immediately above the next SB).
func captureAboveCrossSBHistory(dst *blockModeAboveContext, mode *BlockModeContext, sbSizeMIB uint8) {
	if dst == nil || mode == nil {
		return
	}
	dst.InterMotionHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]InterMotionResult{}
	dst.MotionValidHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]uint8{}
	dst.BlockSizeHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]BlockSize{}
	dst.BlockSizeVisitedHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]uint8{}
	if sbSizeMIB == 0 {
		return
	}
	bottom := int(sbSizeMIB) - 1
	if bottom < 0 {
		return
	}
	if bottom >= MaxBlockModeSlots {
		bottom = MaxBlockModeSlots - 1
	}
	for d := range intrabcCrossSBHistory {
		row := bottom - d
		if row < 0 {
			break
		}
		dst.InterMotionHistory[d] = mode.GridInterMotion[row]
		dst.MotionValidHistory[d] = mode.GridMotionValid[row]
		dst.BlockSizeHistory[d] = mode.GridBlockSize[row]
		dst.BlockSizeVisitedHistory[d] = mode.GridBlockSizeVisited[row]
	}
}

// captureLeftCrossSBHistory copies the rightmost intrabcCrossSBHistory columns
// of the just-finished superblock's grid into the carrier's left slot so the
// next superblock (the one immediately to the right) can serve IBC outer
// mv-ref column scans (colOffset -3, -5) that read into the prior SB's
// interior. Depth 0 is the rightmost column of the prior SB (= column
// immediately to the left of the next SB).
func captureLeftCrossSBHistory(dst *blockModeLeftContext, mode *BlockModeContext, sbSizeMIB uint8) {
	if dst == nil || mode == nil {
		return
	}
	dst.InterMotionHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]InterMotionResult{}
	dst.MotionValidHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]uint8{}
	dst.BlockSizeHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]BlockSize{}
	dst.BlockSizeVisitedHistory = [intrabcCrossSBHistory][MaxBlockModeSlots]uint8{}
	if sbSizeMIB == 0 {
		return
	}
	right := int(sbSizeMIB) - 1
	if right < 0 {
		return
	}
	if right >= MaxBlockModeSlots {
		right = MaxBlockModeSlots - 1
	}
	for d := range intrabcCrossSBHistory {
		col := right - d
		if col < 0 {
			break
		}
		for y := range MaxBlockModeSlots {
			dst.InterMotionHistory[d][y] = mode.GridInterMotion[y][col]
			dst.MotionValidHistory[d][y] = mode.GridMotionValid[y][col]
			dst.BlockSizeHistory[d][y] = mode.GridBlockSize[y][col]
			dst.BlockSizeVisitedHistory[d][y] = mode.GridBlockSizeVisited[y][col]
		}
	}
}

func decodeBlockLoopVisitWithCoeffController[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, hasCoeffController bool, block BlockVisit) (BlockLoopVisit, error) {
	visit, err := decodeBlockLoopVisitWithCoeffControllerPtr(s, cdfs, scratch, req, coeffController, hasCoeffController, block)
	if err != nil {
		return BlockLoopVisit{}, err
	}
	return *visit, nil
}

func decodeBlockLoopVisitWithCoeffControllerPtr[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, hasCoeffController bool, block BlockVisit) (*BlockLoopVisit, error) {
	if entropy.TraceRNGEnabled {
		entropy.TraceLabel("BLOCK mi_row=%d mi_col=%d size=%d", block.MIRow, block.MICol, int(block.Size))
	}
	ctx := &scratch.Mode
	cdef := &scratch.CDEF
	blockX4 := int(block.X4)
	blockY4 := int(block.Y4)
	segmentID := uint8(0)
	segment := defaultSegmentData()
	segmentPredicted := false
	var err error
	if req.Segmentation.Enabled && (!req.Segmentation.UpdateMap || req.Segmentation.Data.Preskip) {
		segmentID, segmentPredicted, segment, err = s.decodeBlockSegment(cdfs.Mode, ctx, req, block, false)
		if err != nil {
			return nil, err
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
	prefix, err := s.readBlockModePrefixSyntax(cdfs.Mode, ctx, prefixReq, segmentPredicted)
	if err != nil {
		return nil, fmt.Errorf("read prefix: %w", err)
	}

	if req.Segmentation.Enabled && req.Segmentation.UpdateMap && !req.Segmentation.Data.Preskip {
		segmentID, segmentPredicted, segment, err = s.decodeBlockSegment(cdfs.Mode, ctx, req, block, prefix.SkipTransform)
		if err != nil {
			return nil, err
		}
		if segmentPredicted {
			prefix.SegmentPredicted = true
		}
	}
	cdefIndex, err := s.ReadCDEFIndexForBlock(prefixReq.CDEF, cdef, prefixReq.Size, int(prefixReq.X4), int(prefixReq.Y4), prefix.SkipTransform)
	if err != nil {
		return nil, err
	}
	prefix.CDEFIndex = cdefIndex
	if err := ctx.Mark(block.Size, blockX4, blockY4, prefix); err != nil {
		return nil, err
	}

	// libaom's read_delta_qindex / read_delta_lf gate the delta read on
	// (bsize != sb_size || !skip), where bsize is the block's *coded* size
	// (mbmi->bsize) — never the frame-edge-clipped visible extent. A skip block
	// whose coded size equals the superblock but whose visible W4/H4 shrinks at
	// the bottom/right frame edge must still be treated as a full superblock, or
	// the delta_q/delta_lf symbol is read when libaom skips it, advancing the
	// entropy decoder by an extra symbol and desyncing every subsequent block.
	fullSuperblock := false
	if dims, ok := block.Size.Dimensions(); ok {
		fullSuperblock = dims.W4 == req.SBSizeMIB && dims.H4 == req.SBSizeMIB
	}
	delta := BlockDeltaContext{
		MICol:          block.MICol,
		MIRow:          block.MIRow,
		SBSizeMIB:      req.SBSizeMIB,
		FullSuperblock: fullSuperblock,
		SkipTransform:  prefix.SkipTransform,
		Monochrome:     req.Monochrome,
	}
	if err := s.ReadBlockDeltas(req.Delta, delta, cdfs.Delta); err != nil {
		return nil, err
	}

	// Capture the above/left neighbor state at slot (X4, Y4) before
	// decodeBlockPredictionMode invokes MarkIntra / MarkIntrabcMotion. The
	// intra tx_size context consumes these because Go's slot-shared
	// AboveIntra/LeftIntra/AboveBlockSize/LeftBlockSize would otherwise be
	// the current block's just-written flags, while libaom's get_tx_size_context
	// reads xd->above_mbmi / xd->left_mbmi (pointers to the actual neighbors,
	// unaffected by writes to xd->mi[0]).
	//
	// The snapshot fires for every block: intra-only frames with intrabc
	// enabled, inter frames with mixed intra/inter blocks, and key frames.
	// Without the snapshot, get_tx_size_context() reads the just-written
	// current block's flags and selects the wrong category/context for
	// `selected_tx_size`, which propagates into a diverging adapted CDF
	// across frame 0 → frame 1.
	ctx.TxNeighborValid = false
	if blockX4 < MaxBlockModeSlots && blockY4 < MaxBlockModeSlots {
		ctx.TxNeighborValid = true
		ctx.TxAboveNeighborIntra = ctx.AboveIntra[blockX4]
		ctx.TxAboveNeighborBlockSize = ctx.AboveBlockSize[blockX4]
		ctx.TxLeftNeighborIntra = ctx.LeftIntra[blockY4]
		ctx.TxLeftNeighborBlockSize = ctx.LeftBlockSize[blockY4]
	}

	var prediction BlockPredictionModeResult
	if req.DecodePredictionModes {
		prediction, err = s.decodeBlockPredictionMode(cdfs, ctx, req, block, prefix, segmentID, segment, &scratch.Palette)
		if err != nil {
			return nil, fmt.Errorf("decode prediction: %w", err)
		}
	}

	visit := &scratch.visit
	visit.Block = block
	visit.SegmentID = segmentID
	visit.Segment = segment
	visit.SegmentPredicted = segmentPredicted
	visit.Prefix = prefix
	visit.Prediction = prediction
	visit.CoefficientsValid = false
	visit.Delta = delta
	if req.CurrentMVFrame != nil {
		if err := req.CurrentMVFrame.MarkBlockPtr(block.MICol, block.MIRow, block.VisibleW4, block.VisibleH4, &visit.Prediction, req.RefFrameSide); err != nil {
			return nil, err
		}
	}
	if req.DecodeCoefficients {
		if hasCoeffController {
			if err := coeffController.BeforeBlockCoefficientsPtr(visit); err != nil {
				return nil, fmt.Errorf("before coefficients: %w", err)
			}
			coeffReq, err := coeffController.SelectBlockCoeffRequestPtr(visit)
			if err != nil {
				return nil, fmt.Errorf("select coeff request: %w", err)
			}
			coefficients, err := s.decodeBlockCoefficientsPtr(BlockCoeffCDFs{
				Transform: cdfs.Transform,
				Coeff:     cdfs.Coeff,
			}, ctx, &scratch.CoeffCtx, &scratch.Coeff, coeffReq, func(block *BlockCoeffBlock) error {
				return coeffController.VisitBlockCoeffPtr(visit, block)
			})
			if err != nil {
				return nil, fmt.Errorf("decode coefficients: %w", err)
			}
			visit.Coefficients = coefficients
			visit.CoefficientsValid = true
		} else {
			if req.BeforeCoefficients != nil {
				if err := req.BeforeCoefficients(*visit); err != nil {
					return nil, fmt.Errorf("before coefficients callback: %w", err)
				}
			}
			coeffReq, err := blockLoopCoeffRequest(req, coeffController, hasCoeffController, *visit)
			if err != nil {
				return nil, fmt.Errorf("select coeff request: %w", err)
			}
			coeffVisit := req.CoeffVisitor
			if coeffVisit == nil {
				coeffVisit = discardBlockLoopCoeff
			}
			coefficients, err := s.DecodeBlockCoefficients(BlockCoeffCDFs{
				Transform: cdfs.Transform,
				Coeff:     cdfs.Coeff,
			}, ctx, &scratch.CoeffCtx, &scratch.Coeff, coeffReq, func(block BlockCoeffBlock) error {
				return coeffVisit(*visit, block)
			})
			if err != nil {
				return nil, fmt.Errorf("decode coefficients: %w", err)
			}
			visit.Coefficients = coefficients
			visit.CoefficientsValid = true
		}
	}
	return visit, nil
}

func (s *DecodeState) decodeBlockPredictionMode(cdfs BlockLoopCDFs, ctx *BlockModeContext, req BlockLoopRequest, block BlockVisit, prefix BlockModeResult, segmentID uint8, segment parser.SegmentData, paletteMap *PaletteModeScratch) (BlockPredictionModeResult, error) {
	blockX4 := int(block.X4)
	blockY4 := int(block.Y4)
	intraFlag, err := s.ReadIntraFlagResult(cdfs.Intra, ctx, IntraFlagRequest{
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

	result := BlockPredictionModeResult{
		Valid:        true,
		Intra:        intraFlag.Intra,
		Intrabc:      intraFlag.Intrabc,
		IntrabcValid: intraFlag.IntrabcValid,
		LumaMode:     IntraModeDC,
	}
	if intraFlag.Intrabc {
		result.ChromaMode = ChromaIntraModeDC
		result.ChromaModeValid = true
		result.MotionMode = MotionModeTranslation
		result.MotionModeValid = true
		result.InterMode = InterModeResult{Mode: InterModeNewMV}
		result.InterModeValid = true
		if req.DecodeMotionVectors {
			motionResult, err := s.readIntrabcMotion(cdfs.MV, ctx, req, block)
			if err != nil {
				return BlockPredictionModeResult{}, err
			}
			result.InterMotion = motionResult.Motion
			result.InterMotionValid = true
			result.MVResiduals = motionResult.Residuals
			result.MVResidualValid = motionResult.ResidualValid
			if err := ctx.MarkIntrabcMotion(block.Size, blockX4, blockY4, result.InterMotion, hasChromaForBlock(block.Size, blockX4, blockY4, req.Color)); err != nil {
				return BlockPredictionModeResult{}, err
			}
		}
		return result, nil
	}
	if !intraFlag.Intra {
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
			return BlockPredictionModeResult{}, fmt.Errorf("read inter references: %w", err)
		}
		result.InterReferences = refs
		result.InterReferencesValid = true
		globalMVs := blockReferenceGlobalMVsForBlock(refs, req.GlobalMVs, req.GlobalMotion, req.AllowHighPrecisionMV, req.ForceIntegerMV, block)
		globalMotionTypes := blockReferenceGlobalMotionTypes(refs, req.GlobalMotionTypes)
		if req.DecodeInterModes {
			stack, err := ctx.BuildReferenceMVStack(ReferenceMVStackRequest{
				Size:             block.Size,
				References:       refs,
				X4:               blockX4,
				Y4:               blockY4,
				HaveTop:          block.HaveTop,
				HaveLeft:         block.HaveLeft,
				HaveTopRight:     blockHasTopRight(req.SBSizeMIB, block),
				GlobalMVs:        globalMVs,
				GlobalMotionType: globalMotionTypes,
				RefSignBias:      req.RefSignBias,

				MICol:          block.MICol,
				MIRow:          block.MIRow,
				TileMIColStart: req.Walk.MIColStart,
				TileMIRowStart: req.Walk.MIRowStart,
				TileMIColEnd:   req.Walk.MIColEnd,
				TileMIRowEnd:   req.Walk.MIRowEnd,
				FrameMIRows:    req.FrameMIRows,
				FrameMICols:    req.FrameMICols,

				TemporalMVs:          req.TemporalMVs,
				OrderHintBits:        req.OrderHintBits,
				CurrentOrderHint:     req.CurrentOrderHint,
				ReferenceOrderHints:  req.ReferenceOrderHints,
				AllowHighPrecisionMV: req.AllowHighPrecisionMV,
				ForceIntegerMV:       req.ForceIntegerMV,

				UseRefFrameMVS:              req.UseRefFrameMVS,
				TemporalMVSampleUnavailable: req.TemporalMVSampleUnavailable,
			})
			if err != nil {
				return BlockPredictionModeResult{}, fmt.Errorf("build ref mv stack: %w", err)
			}
			mode, err := s.ReadBlockInterMode(cdfs.InterMode, InterModeRequest{
				Compound:            refs.Compound,
				SkipMode:            prefix.SkipMode,
				SegmentationEnabled: req.Segmentation.Enabled,
				Segment:             segment,
				ModeContext:         stack.ModeContext,
			})
			if err != nil {
				return BlockPredictionModeResult{}, fmt.Errorf("read inter mode: %w", err)
			}
			drlReq, err := stack.Stack.DRLRequestForMode(mode)
			if err != nil {
				return BlockPredictionModeResult{}, fmt.Errorf("select drl request: %w", err)
			}
			drlIndex, err := s.ReadDRLIndex(cdfs.InterMode, drlReq)
			if err != nil {
				return BlockPredictionModeResult{}, fmt.Errorf("read drl index: %w", err)
			}
			result.InterMode = mode
			result.InterModeValid = true
			result.ReferenceMVStack = stack
			result.ReferenceMVStackValid = true
			result.DRLIndex = uint8(drlIndex)
			result.DRLIndexValid = drlReq.usesNewMV() || drlReq.usesNearMV()
			if !interModeUsesGlobalOnly(mode) {
				mvRefs, err := stack.Stack.ResolveInterMVReferences(mode, drlIndex, req.AllowHighPrecisionMV, req.ForceIntegerMV)
				if err != nil {
					return BlockPredictionModeResult{}, fmt.Errorf("resolve inter mv references mode=%+v drl=%d stack_count=%d nearest=%d: %w", mode, drlIndex, stack.Stack.Count, stack.NearestCount, err)
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
					return BlockPredictionModeResult{}, fmt.Errorf("read inter motion: %w", err)
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
						return BlockPredictionModeResult{}, fmt.Errorf("read inter-intra: %w", err)
					}
					result.InterIntra = interIntra
					result.InterIntraValid = true
					if interIntra.Enabled {
						result.InterMotion.InterIntra = true
					}
				}
				if req.DecodeMotionModes {
					overlappableNeighbors, err := ctx.CollectOverlappableNeighbors(OverlappableNeighborRequest{
						Size:         block.Size,
						X4:           block.X4,
						Y4:           block.Y4,
						VisibleW4:    block.VisibleW4,
						VisibleH4:    block.VisibleH4,
						HaveTop:      block.HaveTop,
						HaveLeft:     block.HaveLeft,
						HaveTopRight: blockHasTopRight(req.SBSizeMIB, block),
					})
					if err != nil {
						return BlockPredictionModeResult{}, fmt.Errorf("collect overlappable neighbors: %w", err)
					}
					// libaom's av1_findSamples scans the full above row and
					// left column past the OBMC neighbor cap (commit 0e64960
					// landed the wider scan as WarpProjectionWithContext for the
					// actual warp projection). Use the same wider scan here so
					// num_proj_ref matches libaom — the OBMC-capped count
					// undershoots for blocks with many short left neighbors,
					// which steers ReadMotionMode into the OBMC CDF when libaom
					// reads the 3-symbol WARP CDF.
					// WarpSampleCountWithContext now resolves top-left/top-right
					// corner neighbors across the superblock boundary (via
					// warpSampleGrid -> crossSBInterGridInterMotion), matching
					// libaom's frame-wide av1_findSamples scan, so no separate
					// SB-diagonal corner augmentation is needed here.
					numProjRef, err := ctx.WarpSampleCountWithContext(block, refs.Ref[0], req.SBSizeMIB)
					if err != nil {
						return BlockPredictionModeResult{}, fmt.Errorf("count warp samples: %w", err)
					}
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
						OverlappableNeighbors: uint8(overlappableNeighbors.MotionModeCount()),
						NumProjRef:            uint8(numProjRef),
					})
					if err != nil {
						return BlockPredictionModeResult{}, fmt.Errorf("read motion mode: %w", err)
					}
					result.MotionMode = motionMode
					result.MotionModeValid = true
					if motionMode == MotionModeWarp {
						model, invalid, err := ctx.WarpProjectionWithContext(block, refs.Ref[0], motionResult.Motion.MV[0], req.SBSizeMIB)
						if err != nil {
							return BlockPredictionModeResult{}, fmt.Errorf("project warped motion: %w", err)
						}
						result.WarpedMotionInvalid = invalid
						if !invalid {
							result.WarpedMotion = model
							result.WarpedMotionValid = true
						}
					}
					if motionMode == MotionModeOBMC || motionMode == MotionModeWarp {
						result.OverlappableNeighbors = overlappableNeighbors
						result.OverlappableNeighborsValid = true
					}
					// libaom promotes GLOBALMV/GLOBAL_GLOBALMV blocks with a
					// non-translational frame-level warp model and a block
					// min-side of at least 8 luma samples to WARP_PRED (see
					// av1_init_warp_params). The motion_mode signaling
					// (SIMPLE_TRANSLATION here) is independent of the
					// underlying prediction path: the warp params are taken
					// from the frame-level global motion array, not from the
					// local OverlappableNeighbor projection.
					if motionMode == MotionModeTranslation &&
						!refs.Compound &&
						refs.Ref[0].Valid() &&
						isGlobalMVBlock(mode, block.Size, blockReferenceGlobalMotionType(refs, req.GlobalMotionTypes)) &&
						!blockReferenceScaled(refs, req.ScaledReferences) {
						params := req.GlobalMotion[refs.Ref[0]]
						if model, ok := warpShearParams(WarpedMotionModel{Params: params}); ok {
							result.GlobalWarpedMotion = model
							result.GlobalWarpedMotionValid = true
						}
					}
					// COMPOUND GLOBAL_GLOBALMV: libaom warps each reference
					// independently via its own global motion params. Mirror the
					// single-ref promotion above for both refs.
					if motionMode == MotionModeTranslation &&
						refs.Compound &&
						!blockReferenceScaled(refs, req.ScaledReferences) {
						for r := range 2 {
							if !refs.Ref[r].Valid() {
								continue
							}
							if !isGlobalMVBlock(mode, block.Size, req.GlobalMotionTypes[refs.Ref[r]]) {
								continue
							}
							params := req.GlobalMotion[refs.Ref[r]]
							if model, ok := warpShearParams(WarpedMotionModel{Params: params}); ok {
								result.GlobalWarpedMotionCompound[r] = model
								result.GlobalWarpedMotionCompoundValid[r] = true
							}
						}
					}
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
						return BlockPredictionModeResult{}, fmt.Errorf("read compound blend: %w", err)
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
					return BlockPredictionModeResult{}, fmt.Errorf("read interp filters: %w", err)
				}
				result.InterpFilters = filters
				result.InterpFiltersValid = true
				result.InterpFilterReads = uint8(filterReads)
				// libaom's build_inter_predictors_sub8x8 path runs at the
				// chroma anchor of any inter block whose luma block has
				// width or height 4 under chroma subsampling. Collect
				// the per-sub-block MVs before MarkInterMotion writes
				// the anchor's own MV to the grid; the helper records
				// the anchor cell with the just-decoded filters and the
				// caller-supplied anchor MV/reference (the neighbor
				// cells come straight off the grid).
				if hasChroma := hasChromaForBlock(block.Size, blockX4, blockY4, req.Color); hasChroma {
					if sub, ok := ctx.CollectSubChromaInterCells(block.Size, blockX4, blockY4, req.Color.SubsamplingX, req.Color.SubsamplingY, filters); ok {
						selfCell := &sub.Cells[sub.Count-1]
						selfCell.MV = result.InterMotion.MV[0]
						selfCell.Reference = refs.Ref[0]
						selfCell.InterpFilters = filters
						result.SubChromaInter = sub
						result.SubChromaInterValid = true
					}
				}
				if err := ctx.MarkInterMotion(block.Size, blockX4, blockY4, result.InterMotion, hasChromaForBlock(block.Size, blockX4, blockY4, req.Color)); err != nil {
					return BlockPredictionModeResult{}, fmt.Errorf("mark inter motion: %w", err)
				}
				if err := ctx.MarkInterFilters(block.Size, blockX4, blockY4, refs, filters); err != nil {
					return BlockPredictionModeResult{}, fmt.Errorf("mark inter filters: %w", err)
				}
				if result.InterIntraValid && result.InterIntra.Enabled {
					// MarkInterMotion -> MarkInter clears AboveInterIntra /
					// LeftInterIntra; record the inter-intra status here so
					// the warp findSamples scan rejects this neighbor for
					// later blocks (matches libaom's ref_frame[1] =
					// INTRA_FRAME bookkeeping for inter-intra blocks). The
					// fallback path below also calls MarkInterIntra after
					// its MarkInter.
					if err := ctx.MarkInterIntra(block.Size, blockX4, blockY4); err != nil {
						return BlockPredictionModeResult{}, fmt.Errorf("mark inter-intra neighbors: %w", err)
					}
				}
				if result.CompoundBlendValid {
					if err := ctx.MarkCompoundBlend(block.Size, blockX4, blockY4, result.CompoundBlend); err != nil {
						return BlockPredictionModeResult{}, fmt.Errorf("mark compound blend: %w", err)
					}
				}
				return result, nil
			}
		}
		if err := ctx.MarkInter(block.Size, blockX4, blockY4, refs, hasChromaForBlock(block.Size, blockX4, blockY4, req.Color)); err != nil {
			return BlockPredictionModeResult{}, fmt.Errorf("mark inter references: %w", err)
		}
		if result.InterIntraValid && result.InterIntra.Enabled {
			if err := ctx.MarkInterIntra(block.Size, blockX4, blockY4); err != nil {
				return BlockPredictionModeResult{}, fmt.Errorf("mark inter-intra neighbors: %w", err)
			}
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
	smoothNeighbor, err := ctx.IntraEdgeSmoothNeighbor(blockX4, blockY4, block.HaveTop, block.HaveLeft)
	if err != nil {
		return BlockPredictionModeResult{}, err
	}
	result.IntraEdgeSmoothNeighbor = smoothNeighbor
	if err := ctx.MarkIntra(block.Size, blockX4, blockY4, true, mode); err != nil {
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
	hasChroma := hasChromaForBlock(block.Size, blockX4, blockY4, req.Color)
	if hasChroma {
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
		chromaHaveTop, chromaHaveLeft, err := blockChromaNeighborAvailability(req, block)
		if err != nil {
			return BlockPredictionModeResult{}, err
		}
		chromaSmoothNeighbor, err := ctx.ChromaIntraEdgeSmoothNeighbor(blockX4, blockY4, chromaHaveTop, chromaHaveLeft, req.Color.SubsamplingX, req.Color.SubsamplingY)
		if err != nil {
			return BlockPredictionModeResult{}, err
		}
		result.ChromaIntraEdgeSmoothNeighbor = chromaSmoothNeighbor
		if err := ctx.MarkChromaIntra(block.Size, blockX4, blockY4, true, chromaMode); err != nil {
			return BlockPredictionModeResult{}, err
		}
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
	if err := s.ReadPaletteMode(cdfs.Intra, ctx, PaletteModeRequest{
		AllowScreenContentTools: req.AllowScreenContentTools,
		Size:                    block.Size,
		LumaMode:                mode,
		X4:                      block.X4,
		Y4:                      block.Y4,
		HaveTop:                 block.HaveTop,
		HaveLeft:                block.HaveLeft,
		BitDepth:                req.Color.BitDepth,
		Color:                   req.Color,
		ChromaMode:              result.ChromaMode,
		ChromaModeValid:         result.ChromaModeValid,
		HasChroma:               hasChroma,
	}, &result.Palette, paletteMap); err != nil {
		return BlockPredictionModeResult{}, err
	}
	if err := ctx.MarkPaletteY(block.Size, blockX4, blockY4, result.Palette); err != nil {
		return BlockPredictionModeResult{}, err
	}
	if err := ctx.MarkPaletteUV(block.Size, blockX4, blockY4, result.Palette); err != nil {
		return BlockPredictionModeResult{}, err
	}
	filterMode, filterValid, err := s.ReadFilterIntraMode(cdfs.Intra, FilterIntraRequest{
		EnableFilterIntra: req.EnableFilterIntra,
		Size:              block.Size,
		LumaMode:          mode,
		PaletteYSize:      result.Palette.YSize,
	})
	if err != nil {
		return BlockPredictionModeResult{}, err
	}
	result.FilterIntraMode = filterMode
	result.FilterIntraValid = filterValid
	if err := s.ReadPaletteTokens(cdfs.Intra, PaletteModeRequest{
		AllowScreenContentTools: req.AllowScreenContentTools,
		Size:                    block.Size,
		LumaMode:                mode,
		X4:                      block.X4,
		Y4:                      block.Y4,
		HaveTop:                 block.HaveTop,
		HaveLeft:                block.HaveLeft,
		BitDepth:                req.Color.BitDepth,
		Color:                   req.Color,
		ChromaMode:              result.ChromaMode,
		ChromaModeValid:         result.ChromaModeValid,
		HasChroma:               hasChroma,
	}, &result.Palette, paletteMap); err != nil {
		return BlockPredictionModeResult{}, err
	}
	return result, nil
}

func blockChromaNeighborAvailability(req BlockLoopRequest, block BlockVisit) (haveTop bool, haveLeft bool, err error) {
	dims, ok := block.Size.Dimensions()
	if !ok {
		return false, false, ErrInvalidDecodeState
	}
	haveTop = block.HaveTop
	haveLeft = block.HaveLeft
	if req.Color.SubsamplingY && dims.H4 < 2 {
		start := req.Walk.neighborMIRowStart()
		haveTop = block.MIRow > 0 && block.MIRow-1 > start
	}
	if req.Color.SubsamplingX && dims.W4 < 2 {
		start := req.Walk.neighborMIColStart()
		haveLeft = block.MICol > 0 && block.MICol-1 > start
	}
	return haveTop, haveLeft, nil
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
		predicted, err = s.ReadSegmentPrediction(cdfs, ctx, int(block.X4), int(block.Y4))
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

func (s *DecodeState) readBlockModePrefixSyntax(cdfs *BlockModeCDFs, ctx *BlockModeContext, req BlockModeRequest, segmentPredicted bool) (BlockModeResult, error) {
	skipMode, err := s.ReadSkipMode(cdfs, ctx, req)
	if err != nil {
		return BlockModeResult{}, err
	}
	skip, err := s.ReadSkipTransform(cdfs, ctx, req, skipMode)
	if err != nil {
		return BlockModeResult{}, err
	}
	return BlockModeResult{
		SegmentPredicted: segmentPredicted,
		SkipMode:         skipMode,
		SkipTransform:    skip,
	}, nil
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
	if req.TemporalMVSampleUnavailable && !req.UseRefFrameMVS {
		return ErrInvalidDecodeState
	}
	if req.TemporalMVs != nil && !req.UseRefFrameMVS {
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
			HaveTop:       block.HaveTop,
			HaveLeft:      block.HaveLeft,
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

func (s *DecodeState) readIntrabcMotion(cdfs *MVCDFs, ctx *BlockModeContext, req BlockLoopRequest, block BlockVisit) (InterMotionDecodeResult, error) {
	if s == nil {
		return InterMotionDecodeResult{}, ErrInvalidDecodeState
	}
	pred, err := intrabcPredictedMV(ctx, req, block)
	if err != nil {
		return InterMotionDecodeResult{}, err
	}
	mv, residual, err := s.ReadMotionVector(cdfs, pred, MVSubpelNone)
	if err != nil {
		return InterMotionDecodeResult{}, err
	}
	if !intrabcDVTileValid(mv, req, block) {
		if altPred, ok := intrabcAlternateFallbackMV(req, block, pred); ok {
			altMV, ok := motionVectorFromInt32(
				int32(altPred.Row)+int32(residual.Diff.Row),
				int32(altPred.Col)+int32(residual.Diff.Col),
			)
			if ok && intrabcDVTileValid(altMV, req, block) {
				mv = altMV
			}
		}
	}
	if !motionVectorValid(mv) {
		return InterMotionDecodeResult{}, ErrInvalidDecodeState
	}
	return InterMotionDecodeResult{
		Motion: InterMotionResult{
			Mode: InterModeResult{Mode: InterModeNewMV},
			MV:   [2]motion.Vector{mv},
		},
		Residuals:     [2]MVResidualResult{residual},
		ResidualValid: [2]bool{true},
	}, nil
}

func intrabcPredictedMV(ctx *BlockModeContext, req BlockLoopRequest, block BlockVisit) (motion.Vector, error) {
	if ctx == nil {
		return motion.Vector{}, ErrInvalidDecodeState
	}
	stack, err := ctx.IntrabcReferenceDVStack(ReferenceMVStackRequest{
		Size:         block.Size,
		X4:           int(block.X4),
		Y4:           int(block.Y4),
		HaveTop:      block.HaveTop,
		HaveLeft:     block.HaveLeft,
		HaveTopRight: blockHasTopRight(req.SBSizeMIB, block),
		MICol:        block.MICol,
		MIRow:        block.MIRow,

		TileMIColStart: req.Walk.MIColStart,
		TileMIRowStart: req.Walk.MIRowStart,
		TileMIColEnd:   req.Walk.MIColEnd,
		TileMIRowEnd:   req.Walk.MIRowEnd,
	})
	if err != nil {
		return motion.Vector{}, err
	}
	for i := 0; i < MaxMVRefCandidates && i < int(stack.Count); i++ {
		if mv := stack.Candidates[i].This; mv != (motion.Vector{}) && intrabcDVValid(mv, req, block) {
			return mv, nil
		}
	}
	return intrabcFallbackMV(req, block)
}

func intrabcFallbackMV(req BlockLoopRequest, block BlockVisit) (motion.Vector, error) {
	if req.SBSizeMIB == 0 {
		return motion.Vector{}, ErrInvalidDecodeState
	}
	sbSize4 := int64(req.SBSizeMIB)
	if int64(block.MIRow)-sbSize4 < int64(req.Walk.MIRowStart) {
		mv, ok := motionVectorFromInt32(0, -int32((sbSize4*4+intrabcDelayPixels)*8))
		if !ok {
			return motion.Vector{}, ErrInvalidDecodeState
		}
		return mv, nil
	}
	mv, ok := motionVectorFromInt32(-int32(sbSize4*4*8), 0)
	if !ok {
		return motion.Vector{}, ErrInvalidDecodeState
	}
	return mv, nil
}

func intrabcAlternateFallbackMV(req BlockLoopRequest, block BlockVisit, pred motion.Vector) (motion.Vector, bool) {
	fallback, err := intrabcFallbackMV(req, block)
	if err != nil || pred != fallback {
		return motion.Vector{}, false
	}
	sbSize4 := int64(req.SBSizeMIB)
	horizontal, ok := motionVectorFromInt32(0, -int32((sbSize4*4+intrabcDelayPixels)*8))
	if !ok {
		return motion.Vector{}, false
	}
	vertical, ok := motionVectorFromInt32(-int32(sbSize4*4*8), 0)
	if !ok {
		return motion.Vector{}, false
	}
	if pred == horizontal {
		return vertical, true
	}
	return horizontal, true
}

func intrabcDVTileValid(dv motion.Vector, req BlockLoopRequest, block BlockVisit) bool {
	if dv.Row&7 != 0 || dv.Col&7 != 0 {
		return false
	}
	dims, ok := block.Size.Dimensions()
	if !ok {
		return false
	}
	blockW := int64(dims.W4) * 4
	blockH := int64(dims.H4) * 4
	srcTop := int64(block.MIRow)*4*8 + int64(dv.Row)
	srcLeft := int64(block.MICol)*4*8 + int64(dv.Col)
	tileTop := int64(req.Walk.MIRowStart) * 4 * 8
	tileLeft := int64(req.Walk.MIColStart) * 4 * 8
	if srcTop < tileTop || srcLeft < tileLeft {
		return false
	}
	srcBottom := (int64(block.MIRow)*4+blockH)*8 + int64(dv.Row)
	srcRight := (int64(block.MICol)*4+blockW)*8 + int64(dv.Col)
	tileBottom := int64(req.Walk.MIRowEnd) * 4 * 8
	tileRight := int64(req.Walk.MIColEnd) * 4 * 8
	if srcBottom > tileBottom || srcRight > tileRight {
		return false
	}
	if !req.Monochrome {
		if blockW < 8 && req.Color.SubsamplingX && srcLeft < tileLeft+4*8 {
			return false
		}
		if blockH < 8 && req.Color.SubsamplingY && srcTop < tileTop+4*8 {
			return false
		}
	}
	return true
}

func intrabcDVValid(dv motion.Vector, req BlockLoopRequest, block BlockVisit) bool {
	if dv.Row&7 != 0 || dv.Col&7 != 0 || req.SBSizeMIB == 0 {
		return false
	}
	dims, ok := block.Size.Dimensions()
	if !ok {
		return false
	}
	blockW := int64(dims.W4) * 4
	blockH := int64(dims.H4) * 4
	srcTop := int64(block.MIRow)*4*8 + int64(dv.Row)
	srcLeft := int64(block.MICol)*4*8 + int64(dv.Col)
	tileTop := int64(req.Walk.MIRowStart) * 4 * 8
	tileLeft := int64(req.Walk.MIColStart) * 4 * 8
	if srcTop < tileTop || srcLeft < tileLeft {
		return false
	}
	srcBottom := (int64(block.MIRow)*4+blockH)*8 + int64(dv.Row)
	srcRight := (int64(block.MICol)*4+blockW)*8 + int64(dv.Col)
	tileBottom := int64(req.Walk.MIRowEnd) * 4 * 8
	tileRight := int64(req.Walk.MIColEnd) * 4 * 8
	if srcBottom > tileBottom || srcRight > tileRight {
		return false
	}
	if !req.Monochrome {
		if blockW < 8 && req.Color.SubsamplingX && srcLeft < tileLeft+4*8 {
			return false
		}
		if blockH < 8 && req.Color.SubsamplingY && srcTop < tileTop+4*8 {
			return false
		}
	}

	mibSizeLog2 := 4
	if req.SBSizeMIB > 16 {
		mibSizeLog2 = 5
	}
	maxMIBSize := int64(1 << mibSizeLog2)
	sbSize := maxMIBSize * 4
	activeSBRow := int64(block.MIRow) >> mibSizeLog2
	activeSB64Col := (int64(block.MICol) * 4) >> 6
	srcSBRow := ((srcBottom >> 3) - 1) / sbSize
	srcSB64Col := ((srcRight >> 3) - 1) >> 6
	totalSB64PerRow := ((int64(req.Walk.MIColEnd) - int64(req.Walk.MIColStart) - 1) >> 4) + 1
	activeSB64 := activeSBRow*totalSB64PerRow + activeSB64Col
	srcSB64 := srcSBRow*totalSB64PerRow + srcSB64Col
	if srcSB64 >= activeSB64-intrabcDelaySB64 {
		return false
	}
	gradient := int64(1 + intrabcDelaySB64)
	if sbSize > 64 {
		gradient++
	}
	wfOffset := gradient * (activeSBRow - srcSBRow)
	return srcSBRow <= activeSBRow &&
		srcSB64Col < activeSB64Col-intrabcDelaySB64+wfOffset
}

func globalMotionVector(params parser.WarpedMotionParams, allowHighPrecisionMV bool, forceIntegerMV bool, block BlockVisit) (motion.Vector, bool) {
	if !globalMotionParamsInitialized(params) {
		return motion.Vector{}, false
	}
	switch params.Type {
	case parser.GlobalMotionIdentity:
		return motion.Vector{}, true
	case parser.GlobalMotionTranslation:
		row := params.Matrix[0] >> globalMotionTransOnlyPrecDiff
		col := params.Matrix[1] >> globalMotionTransOnlyPrecDiff
		if forceIntegerMV {
			row = globalMotionIntegerMVPrecision(row)
			col = globalMotionIntegerMVPrecision(col)
		}
		return motionVectorFromInt32(row, col)
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
		row := globalMotionConvertToTransPrec(yc, allowHighPrecisionMV)
		col := globalMotionConvertToTransPrec(xc, allowHighPrecisionMV)
		if forceIntegerMV {
			row = globalMotionIntegerMVPrecision(row)
			col = globalMotionIntegerMVPrecision(col)
		}
		return motionVectorFromInt32(row, col)
	default:
		return motion.Vector{}, false
	}
}

const (
	globalMotionWarpedModelPrecBits = 16
	globalMotionTransOnlyPrecDiff   = globalMotionWarpedModelPrecBits - 3
	intrabcDelayPixels              = 256
	intrabcDelaySB64                = intrabcDelayPixels / 64
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

func blockReferenceOrderHints(refs InterReferencesResult, orderHints [referenceFrameCount]uint8) [2]uint8 {
	var out [2]uint8
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

// warpSampleDiagonalCorner reports the contribution of the bottom-right cell of
// the superblock diagonally up-and-left of the current SB to libaom's
// av1_findSamples top-left scan. WarpSampleCountWithContext addresses the
// top-left neighbor via the per-SB grid at (block.X4-1, block.Y4-1), which
// has no valid entry for negative coordinates, so the cross-SB diagonal cell
// is dropped when the current block is at the SB top-left corner. The
// caller-owned SBDiagonal* grids carry the snapshot of the prior diagonal
// superblock's bottom-right corner (see captureDiagonalCornerToPending), so
// this helper looks the cell up there and returns 1 when it matches libaom's
// findSamples acceptance test (same ref0, no second reference, no
// inter-intra, valid motion). Any other block position returns 0 because the
// existing scan already covers it.
func warpSampleDiagonalCorner(block BlockVisit, ctx *BlockModeContext, ref ReferenceFrame) int {
	if ctx == nil || !ref.Valid() {
		return 0
	}
	if block.X4 != 0 || block.Y4 != 0 {
		return 0
	}
	if !block.HaveTop || !block.HaveLeft {
		return 0
	}
	if ctx.SBDiagonalMotionValidGrid[0][0] == 0 {
		return 0
	}
	motionResult := ctx.SBDiagonalInterMotionGrid[0][0]
	if motionResult.References.Compound {
		return 0
	}
	if motionResult.References.Ref[0] != ref {
		return 0
	}
	if motionResult.References.Ref[1] != ReferenceFrameNone {
		return 0
	}
	if motionResult.InterIntra {
		return 0
	}
	return 1
}
