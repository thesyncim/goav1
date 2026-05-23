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

// BlockLoopRequest carries frame and tile state needed by the syntax loop.
type BlockLoopRequest struct {
	Walk BlockWalkRequest

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

	GlobalMVs            [referenceFrameCount]motion.Vector
	GlobalMotionTypes    [referenceFrameCount]parser.GlobalMotionType
	RefSignBias          [referenceFrameCount]bool
	ReferenceOrderHints  [referenceFrameCount]uint32
	ScaledReferences     [referenceFrameCount]bool
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

type BlockLoopCoeffVisitor func(BlockLoopVisit, BlockCoeffBlock) error

// DecodeBlockLoop walks every root block in req and decodes the shared
// per-block syntax prefix needed before intra/inter and transform decode.
func (s *DecodeState) DecodeBlockLoop(cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, visit BlockLoopVisitor) (BlockLoopStats, error) {
	if s == nil || scratch == nil || cdfs.Partition == nil || cdfs.Mode == nil || visit == nil {
		return BlockLoopStats{}, ErrInvalidDecodeState
	}
	if err := validateBlockLoopRequest(req); err != nil {
		return BlockLoopStats{}, err
	}

	var stats BlockLoopStats
	rootSize := uint32(req.Walk.Root.Size4x4())
	for miRow := req.Walk.MIRowStart; miRow < req.Walk.MIRowEnd; miRow += rootSize {
		for miCol := req.Walk.MIColStart; miCol < req.Walk.MIColEnd; miCol += rootSize {
			scratch.Partition = PartitionContext{}
			scratch.Mode = BlockModeContext{}
			scratch.CDEF.Reset()
			scratch.CoeffCtx = CoeffEntropyContext{}
			rootReq := BlockWalkRequest{
				Root:       req.Walk.Root,
				MIColStart: miCol,
				MIRowStart: miRow,
				MIColEnd:   minUint32(req.Walk.MIColEnd, miCol+rootSize),
				MIRowEnd:   minUint32(req.Walk.MIRowEnd, miRow+rootSize),
			}
			walkStats, err := walkBlocks(&scratch.Partition, rootReq, func(level BlockLevel, context int, haveRight bool, haveBottom bool) (Partition, error) {
				return s.ReadPartition(cdfs.Partition, level, context, haveRight, haveBottom)
			}, func(block BlockVisit) error {
				visitInfo, err := s.decodeBlockLoopVisit(cdfs, scratch, req, block)
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
		}
	}
	return stats, nil
}

func (s *DecodeState) decodeBlockLoopVisit(cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, block BlockVisit) (BlockLoopVisit, error) {
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
		prediction, err = s.decodeBlockPredictionMode(cdfs, ctx, req, block, prefix, segment)
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
		if !prediction.Valid {
			return BlockLoopVisit{}, ErrInvalidDecodeState
		}
		coeffVisit := req.CoeffVisitor
		if coeffVisit == nil {
			coeffVisit = discardBlockLoopCoeff
		}
		coefficients, err := s.DecodeBlockCoefficients(BlockCoeffCDFs{
			Transform: cdfs.Transform,
			Coeff:     cdfs.Coeff,
		}, ctx, &scratch.CoeffCtx, &scratch.Coeff, BlockCoeffRequest{
			Transform: TransformTreeRequest{
				Size:          block.Size,
				X4:            block.X4,
				Y4:            block.Y4,
				VisibleW4:     block.VisibleW4,
				VisibleH4:     block.VisibleH4,
				Color:         req.Color,
				TransformMode: req.TransformMode,
				Inter:         !prediction.Intra,
				SkipTransform: prefix.SkipTransform,
				Lossless:      req.Lossless,
			},
			LumaType:        req.LumaTransformType,
			ChromaType:      req.ChromaTransformType,
			TransformSelect: req.TransformSelect,
			EOBMultiContext: req.EOBMultiContext,
		}, func(block BlockCoeffBlock) error {
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

func (s *DecodeState) decodeBlockPredictionMode(cdfs BlockLoopCDFs, ctx *BlockModeContext, req BlockLoopRequest, block BlockVisit, prefix BlockModeResult, segment parser.SegmentData) (BlockPredictionModeResult, error) {
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
		globalMVs := blockReferenceGlobalMVs(refs, req.GlobalMVs)
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
				if err := ctx.MarkInterMotion(block.Size, block.X4, block.Y4, motionResult.Motion); err != nil {
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

func validateBlockLoopRequest(req BlockLoopRequest) error {
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
	if req.DecodeCoefficients && !req.DecodePredictionModes {
		return ErrInvalidDecodeState
	}
	rootSize := uint32(req.Walk.Root.Size4x4())
	if rootSize == 0 || req.Walk.MIColStart%rootSize != 0 || req.Walk.MIRowStart%rootSize != 0 {
		return ErrInvalidDecodeState
	}
	if req.Segmentation.Enabled && (req.Segmentation.UpdateMap || len(req.PreviousSegmentMap) != 0) {
		if req.SegmentMapStride <= 0 {
			return ErrInvalidDecodeState
		}
	}
	return nil
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

func blockReferenceGlobalMotionType(refs InterReferencesResult, global [referenceFrameCount]parser.GlobalMotionType) parser.GlobalMotionType {
	if refs.Ref[0].Valid() {
		return global[refs.Ref[0]]
	}
	return parser.GlobalMotionIdentity
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
