package tile

import "github.com/thesyncim/goav1/internal/av1/parser"

// BlockLoopCDFs groups the caller-owned entropy state used by the block syntax
// loop.
type BlockLoopCDFs struct {
	Partition *PartitionCDFs
	Mode      *BlockModeCDFs
	Intra     *IntraModeCDFs
	Delta     DeltaCDFs
}

// BlockLoopScratch is per-superblock scratch for recursive partition and mode
// context traversal.
type BlockLoopScratch struct {
	Partition PartitionContext
	Mode      BlockModeContext
	CDEF      CDEFIndexContext
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
	DecodePredictionModes bool
}

// BlockLoopVisit is reported after partition, segmentation, prefix, and delta
// syntax have been decoded for one leaf block.
type BlockLoopVisit struct {
	Block BlockVisit

	SegmentID        uint8
	Segment          parser.SegmentData
	SegmentPredicted bool
	Prefix           BlockModeResult
	Prediction       BlockPredictionModeResult
	Delta            BlockDeltaContext
}

type BlockLoopStats struct {
	PartitionReads     int
	Blocks             int
	SegmentPredictions int
	SegmentIDs         int
	Prefixes           int
	PredictionModes    int
	IntraModes         int
	InterEntries       int
	DeltaReads         int
}

type BlockLoopVisitor func(BlockLoopVisit) error

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
				visitInfo, err := s.decodeBlockLoopVisit(cdfs, &scratch.Mode, &scratch.CDEF, req, block)
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
					}
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

func (s *DecodeState) decodeBlockLoopVisit(cdfs BlockLoopCDFs, ctx *BlockModeContext, cdef *CDEFIndexContext, req BlockLoopRequest, block BlockVisit) (BlockLoopVisit, error) {
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
		prediction, err = s.decodeBlockPredictionMode(cdfs.Intra, ctx, req, block, prefix, segment)
		if err != nil {
			return BlockLoopVisit{}, err
		}
	}

	return BlockLoopVisit{
		Block:            block,
		SegmentID:        segmentID,
		Segment:          segment,
		SegmentPredicted: segmentPredicted,
		Prefix:           prefix,
		Prediction:       prediction,
		Delta:            delta,
	}, nil
}

func (s *DecodeState) decodeBlockPredictionMode(cdfs *IntraModeCDFs, ctx *BlockModeContext, req BlockLoopRequest, block BlockVisit, prefix BlockModeResult, segment parser.SegmentData) (BlockPredictionModeResult, error) {
	intra, err := s.ReadIntraFlag(cdfs, ctx, IntraFlagRequest{
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
		if err := ctx.MarkIntraEntry(block.Size, block.X4, block.Y4, false, IntraModeDC); err != nil {
			return BlockPredictionModeResult{}, err
		}
		return result, nil
	}

	mode, err := s.ReadLumaIntraMode(cdfs, ctx, LumaIntraModeRequest{
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
