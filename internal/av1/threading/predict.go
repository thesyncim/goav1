package threading

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

const (
	frameWorkIntraPredictionMaxEdgeSamples = 128
	// 128x128 directional intra needs origin+width+height samples on each
	// edge (= 257 for the worst case). Sized to 257 so 128x128 intra blocks
	// pass the directional edge-filter range check.
	frameWorkIntraEdgeScratchSamples       = 257
	frameWorkDirectionalEdgeOrigin         = 128
	frameWorkDirectionalEdgeSamples        = 512
	frameWorkMaxMIBSizeLog2                = 5
	frameWorkInterPredictionMaxBlockBytes  = 128 * 128 * 2
	frameWorkInterPredictionMaxMaskSamples = 128 * 128
	frameWorkMaxFrameDistance              = 31
	frameWorkDistPrecisionBits             = 4
	frameWorkDiffWtdMaskBase               = 38
	frameWorkDiffWtdFactor                 = 16
	frameWorkBlendA64MaxAlpha              = 64
)

var frameWorkQuantDistWeight = [4][2]int{
	{2, 3},
	{2, 5},
	{2, 7},
	{1, frameWorkMaxFrameDistance},
}

var frameWorkQuantDistLookup = [4][2]int{
	{9, 7},
	{11, 5},
	{12, 4},
	{13, 3},
}

var frameWorkOBMCMask1 = [1]uint8{64}
var frameWorkOBMCMask2 = [2]uint8{45, 64}
var frameWorkOBMCMask4 = [4]uint8{39, 50, 59, 64}
var frameWorkOBMCMask8 = [8]uint8{36, 42, 48, 53, 57, 61, 64, 64}
var frameWorkOBMCMask16 = [16]uint8{34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64}
var frameWorkOBMCMask32 = [32]uint8{33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60, 61, 62, 64, 64, 64, 64, 64, 64, 64, 64}
var frameWorkOBMCMask64 = [64]uint8{
	33, 34, 35, 35, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 44, 44,
	45, 46, 47, 47, 48, 49, 50, 51, 51, 51, 52, 52, 53, 54, 55, 56,
	56, 56, 57, 57, 58, 58, 59, 60, 60, 60, 60, 60, 61, 62, 62, 62,
	62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
}
var frameWorkInterIntraWeights = [128]uint8{
	60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35,
	34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 22, 21, 20,
	19, 19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12,
	11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7,
	6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4,
	4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
}

// FrameWorkIntraPredictionScratch carries caller-owned edge buffers for luma
// and chroma intra prediction. Keep it outside FrameWorkTileResidualScratch so
// callers that do not use built-in prediction pay no per-worker storage cost.
type FrameWorkIntraPredictionScratch struct {
	Above [frameWorkDirectionalEdgeSamples]uint16
	Left  [frameWorkDirectionalEdgeSamples]uint16
	Edge  [frameWorkIntraEdgeScratchSamples]uint16
}

// FrameWorkCFLPredictionScratch carries caller-owned CfL luma AC buffers.
// Callers only need this scratch when scheduling chroma CfL after luma
// reconstruction.
type FrameWorkCFLPredictionScratch struct {
	Intra   FrameWorkIntraPredictionScratch
	ReconQ3 [prediction.CFLBufSquare]uint16
	ACQ3    [prediction.CFLBufSquare]int16
}

// FrameWorkInterPredictionScratch carries caller-owned temporary buffers for
// compound inter prediction. It stays separate from intra/residual scratch so
// callers that do not predict compound blocks pay no storage cost.
type FrameWorkInterPredictionScratch struct {
	First  [frameWorkInterPredictionMaxBlockBytes]byte
	Second [frameWorkInterPredictionMaxBlockBytes]byte
	Mask   [frameWorkInterPredictionMaxMaskSamples]byte
	Intra  FrameWorkIntraPredictionScratch
	// CONV_BUF intermediate predictors for the compound inter path, which
	// blends at the un-rounded 16-bit precision libaom keeps in
	// av1_dist_wtd_convolve_*.
	Conv0 motion.CompoundConvBuf
	Conv1 motion.CompoundConvBuf
}

// FrameWorkPredictionScratch groups caller-owned prediction scratch. Keeping it
// separate from residual scratch lets callers that do not use built-in
// prediction avoid carrying these buffers.
type FrameWorkPredictionScratch struct {
	Intra FrameWorkIntraPredictionScratch
	Inter *FrameWorkInterPredictionScratch
}

// PredictBlock writes prediction pixels for one decoded block-loop visit where
// all present planes are supported by the current prediction implementation.
// Inter blocks cover every present plane; intra blocks cover luma and non-CfL
// chroma predictors. CfL needs luma-first reconstruction scheduling and remains
// an explicit unsupported path here until that pipeline is wired.
func (b FrameWorkBatch) PredictBlock(index int, visit tile.BlockLoopVisit, scratch *FrameWorkPredictionScratch) error {
	if !visit.Prediction.Valid {
		return ErrInvalidBatch
	}
	if visit.Prediction.Intra {
		if scratch == nil {
			return ErrInvalidBatch
		}
		return b.PredictBlockIntra(index, visit, &scratch.Intra)
	}
	var interScratch *FrameWorkInterPredictionScratch
	if scratch != nil {
		interScratch = scratch.Inter
	}
	return b.PredictBlockInter(index, visit, interScratch)
}

// PredictBlockLuma dispatches luma prediction for one decoded block-loop visit.
// It covers intra and translational inter/compound luma modes.
func (b FrameWorkBatch) PredictBlockLuma(index int, visit tile.BlockLoopVisit, scratch *FrameWorkPredictionScratch) error {
	if !visit.Prediction.Valid {
		return ErrInvalidBatch
	}
	if frameWorkPredictionIsIntrabc(visit.Prediction) {
		return b.predictBlockIntrabcPlane(index, visit, FrameWorkPlaneY)
	}
	if visit.Prediction.Intra {
		if scratch == nil {
			return ErrInvalidBatch
		}
		return b.PredictBlockLumaIntra(index, visit, &scratch.Intra)
	}
	if visit.Prediction.InterMotion.References.Compound {
		if scratch == nil || scratch.Inter == nil {
			return ErrInvalidBatch
		}
		return b.PredictBlockLumaInterCompound(index, visit, scratch.Inter)
	}
	if visit.Prediction.MotionModeValid && visit.Prediction.MotionMode == tile.MotionModeOBMC {
		if scratch == nil || scratch.Inter == nil {
			return ErrInvalidBatch
		}
		return b.PredictBlockLumaInterOBMC(index, visit, scratch.Inter)
	}
	return b.PredictBlockLumaInter(index, visit)
}

// PredictBlockInter writes inter prediction pixels for every present plane of
// one decoded inter block. Single-reference translation/OBMC/warp and
// average/dist-wtd/wedge/diff-wtd compound translation, and non-wedge
// inter-intra prediction are supported; scaled references are rejected until
// their prediction paths are integrated.
func (b FrameWorkBatch) PredictBlockInter(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch) error {
	filters, err := frameWorkVisitMotionFilters(b.TileInfo, visit.Prediction)
	if err != nil {
		return err
	}
	return b.PredictBlockInterWithFilters(index, visit, scratch, filters)
}

// PredictBlockInterWithFilters is PredictBlockInter with explicit interpolation
// filters, matching callers that have already decoded switchable filter syntax.
func (b FrameWorkBatch) PredictBlockInterWithFilters(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if !visit.Prediction.Valid ||
		visit.Prediction.Intra ||
		!visit.Prediction.InterMotionValid {
		return ErrInvalidBatch
	}
	if frameWorkPredictionIsIntrabc(visit.Prediction) {
		for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
			if err := b.predictBlockIntrabcPlane(index, visit, plane); err != nil {
				return fmt.Errorf("intrabc plane %d: %w", plane, err)
			}
		}
		return nil
	}
	if visit.Prediction.InterMotion.References.Compound {
		if scratch == nil {
			return ErrInvalidBatch
		}
		for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
			if err := b.predictBlockInterCompoundPlaneWithFilters(index, visit, plane, scratch, filters); err != nil {
				return err
			}
		}
		return nil
	}
	if visit.Prediction.MotionModeValid && visit.Prediction.MotionMode == tile.MotionModeOBMC {
		return b.PredictBlockInterOBMCWithFilters(index, visit, scratch, filters)
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		if scratch == nil {
			return ErrInvalidBatch
		}
		for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
			if err := b.predictBlockInterIntraPlaneWithFilters(index, visit, plane, scratch, filters); err != nil {
				return err
			}
		}
		return nil
	}
	for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
		if err := b.predictBlockInterPlaneWithFilters(index, visit, plane, filters); err != nil {
			return err
		}
	}
	return nil
}

// PredictBlockInterOBMC writes single-reference inter prediction for every
// present plane and blends the above/left OBMC neighbor predictors selected by
// motion_mode.
func (b FrameWorkBatch) PredictBlockInterOBMC(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch) error {
	filters, err := frameWorkVisitMotionFilters(b.TileInfo, visit.Prediction)
	if err != nil {
		return err
	}
	return b.PredictBlockInterOBMCWithFilters(index, visit, scratch, filters)
}

// PredictBlockInterOBMCWithFilters is PredictBlockInterOBMC with explicit
// interpolation filters for the current block's base predictor.
func (b FrameWorkBatch) PredictBlockInterOBMCWithFilters(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if scratch == nil {
		return ErrInvalidBatch
	}
	for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
		if err := b.predictBlockInterOBMCPlaneWithFilters(index, visit, plane, scratch, filters); err != nil {
			return err
		}
	}
	return nil
}

// PredictBlockIntra writes intra prediction pixels for every present plane of
// one decoded intra block. Chroma DC, vertical, horizontal, directional, Paeth,
// and smooth modes are supported. CfL callers should schedule
// PredictBlockLumaIntra, luma residual reconstruction, then PredictBlockChromaCFL.
func (b FrameWorkBatch) PredictBlockIntra(index int, visit tile.BlockLoopVisit, scratch *FrameWorkIntraPredictionScratch) error {
	if scratch == nil || !visit.Prediction.Valid || !visit.Prediction.Intra {
		return ErrInvalidBatch
	}
	if err := b.PredictBlockLumaIntra(index, visit, scratch); err != nil {
		return err
	}
	return b.PredictBlockChromaIntra(index, visit, scratch)
}

// PredictBlockChromaIntra writes non-CfL chroma intra prediction for one
// decoded block-loop visit. It is split from PredictBlockIntra so the decode
// loop can run luma prediction/reconstruction before CfL chroma.
func (b FrameWorkBatch) PredictBlockChromaIntra(index int, visit tile.BlockLoopVisit, scratch *FrameWorkIntraPredictionScratch) error {
	if scratch == nil || !visit.Prediction.Valid || !visit.Prediction.Intra {
		return ErrInvalidBatch
	}
	if b.Sequence.ColorConfig.MonoChrome {
		return nil
	}
	if !tile.HasChromaBlock(tile.TransformTreeRequest{Size: visit.Block.Size, X4: visit.Block.X4, Y4: visit.Block.Y4}, b.Sequence.ColorConfig) {
		return nil
	}
	if !visit.Prediction.ChromaModeValid || visit.Prediction.ChromaMode == tile.ChromaIntraModeCFL || visit.Prediction.CFLAlphaValid {
		return ErrInvalidBatch
	}
	for plane := FrameWorkPlaneU; plane <= FrameWorkPlaneV; plane++ {
		if err := b.predictBlockChromaIntraPlane(index, visit, plane, scratch); err != nil {
			return err
		}
	}
	return nil
}

// PredictBlockChromaCFL writes CfL chroma prediction for one decoded block-loop
// visit. The luma block in the output frame must already contain reconstructed
// samples, matching libaom's luma-first CfL scheduling.
func (b FrameWorkBatch) PredictBlockChromaCFL(index int, visit tile.BlockLoopVisit, scratch *FrameWorkCFLPredictionScratch) error {
	if scratch == nil || !visit.Prediction.Valid || !visit.Prediction.Intra ||
		!visit.Prediction.ChromaModeValid || visit.Prediction.ChromaMode != tile.ChromaIntraModeCFL ||
		!visit.Prediction.CFLAlphaValid {
		return ErrInvalidBatch
	}
	if b.Sequence.ColorConfig.MonoChrome ||
		!tile.HasChromaBlock(tile.TransformTreeRequest{Size: visit.Block.Size, X4: visit.Block.X4, Y4: visit.Block.Y4}, b.Sequence.ColorConfig) {
		return ErrInvalidBatch
	}
	if err := b.predictBlockChromaCFLPlane(index, visit, FrameWorkPlaneU, scratch); err != nil {
		return err
	}
	return b.predictBlockChromaCFLPlane(index, visit, FrameWorkPlaneV, scratch)
}

// PredictBlockIntraCoeff writes intra prediction for one transform block from a
// decoded block. Non-skip intra reconstruction uses this TXB-granular path so
// top/left dependencies inside large blocks follow libaom's
// predict-and-reconstruct order.
func (b FrameWorkBatch) PredictBlockIntraCoeff(index int, visit tile.BlockLoopVisit, block tile.BlockCoeffBlock, scratch *FrameWorkIntraPredictionScratch) error {
	if scratch == nil || !visit.Prediction.Valid || !visit.Prediction.Intra {
		return ErrInvalidBatch
	}
	switch block.Plane {
	case 0:
		return b.predictBlockLumaIntraTransform(index, visit, block.Block, scratch)
	case 1, 2:
		if !visit.Prediction.ChromaModeValid || visit.Prediction.ChromaMode == tile.ChromaIntraModeCFL || visit.Prediction.CFLAlphaValid {
			return ErrInvalidBatch
		}
		return b.predictBlockChromaIntraTransform(index, visit, FrameWorkPlane(block.Plane), block.Block, scratch)
	default:
		return ErrInvalidBatch
	}
}

// PredictBlockLumaIntra writes luma intra prediction pixels for one decoded
// block-loop visit into Jobs[index]'s output window. It covers luma DC,
// vertical, horizontal, directional, filter-intra, Paeth, and smooth modes.
func (b FrameWorkBatch) PredictBlockLumaIntra(index int, visit tile.BlockLoopVisit, scratch *FrameWorkIntraPredictionScratch) error {
	if scratch == nil || !visit.Prediction.Valid || !visit.Prediction.Intra {
		return ErrInvalidBatch
	}
	width, height, ok := frameWorkBlockVisiblePixels(visit.Block)
	if !ok ||
		width > frameWorkIntraPredictionMaxEdgeSamples ||
		height > frameWorkIntraPredictionMaxEdgeSamples {
		return ErrInvalidBatch
	}
	predWidth, predHeight, err := frameWorkBlockLumaPredictionExtentPixels(visit.Block)
	if err != nil {
		return err
	}

	region, err := b.JobRegion(index)
	if err != nil {
		return err
	}
	if !frameWorkBlockWithinJobRegion(region, visit.Block) {
		return ErrInvalidBatch
	}
	window, err := b.JobOutputPlane(index, FrameWorkPlaneY)
	if err != nil {
		return err
	}
	x, y, err := frameWorkBlockLumaPosition(visit.Block)
	if err != nil {
		return err
	}
	width, height, ok = frameWorkClipVisiblePixelsToWindow(window, x, y, width, height)
	if !ok {
		if frameWorkPlaneBlockStartsBeyondOutput(b.Output, FrameWorkPlaneY, x, y) {
			return nil
		}
		return ErrInvalidBatch
	}
	dst := frameWorkPlaneFromWindow(window)
	absX := x
	absY := y
	x -= window.X
	y -= window.Y

	if visit.Prediction.Palette.YSize > 0 {
		if err := frameWorkPredictLumaPalette(dst, window.BytesPerSample, x, y, width, height, visit.Block, visit.Prediction.Palette, 0, 0); err != nil {
			return err
		}
		return nil
	}

	if visit.Prediction.FilterIntraValid {
		mode, ok := frameWorkFilterIntraPredictionMode(visit.Prediction.FilterIntraMode)
		if !ok {
			return ErrInvalidBatch
		}
		readBoundX, readBoundY := frameWorkWindowEdgeReadBound(window)
		edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, readBoundX, readBoundY, visit.Block, scratch, true)
		if err != nil {
			return err
		}
		if err := prediction.PredictFilterIntraPlaneBlockWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, mode, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	if angle, ok := frameWorkLumaIntraDirectionalAngle(visit.Prediction.LumaMode, visit.Prediction.LumaAngleDelta); ok {
		readBoundX, readBoundY := frameWorkWindowEdgeReadBound(window)
		allowTopRight, allowBottomLeft := frameWorkLumaDirectionalExtendedEdges(visit.Block, b.Sequence.SBSizeMIB, region.MIColEnd, region.MIRowEnd, absX, absY, predWidth, predHeight)
		edges, err := frameWorkDirectionalPredictionEdges(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, predWidth, predHeight, angle, visit.Block, scratch, b.Sequence.EnableIntraEdgeFilter, visit.Prediction.IntraEdgeSmoothNeighbor, allowTopRight, allowBottomLeft, readBoundX, readBoundY)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlockWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	mode, ok := frameWorkLumaIntraPredictionMode(visit.Prediction.LumaMode)
	if !ok {
		return ErrInvalidBatch
	}
	readBoundX, readBoundY := frameWorkWindowEdgeReadBound(window)
	edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, readBoundX, readBoundY, visit.Block, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockLumaIntraTransform(index int, visit tile.BlockLoopVisit, tx tile.TransformBlock, scratch *FrameWorkIntraPredictionScratch) error {
	_, _, predWidth, predHeight, err := frameWorkTransformVisibleAndExtentPixels(tx)
	if err != nil {
		return err
	}
	region, err := b.JobRegion(index)
	if err != nil {
		return err
	}
	window, err := b.JobOutputPlane(index, FrameWorkPlaneY)
	if err != nil {
		return err
	}
	absX, absY, err := frameWorkBlockLumaTransformPosition(visit.Block, tx)
	if err != nil {
		return err
	}
	// libaom predicts the WHOLE transform (predWidth x predHeight) into the
	// frame buffer and adds the full-transform residual; the trailing
	// past-cropped-edge rows/cols spill into the superblock-aligned padding and
	// are read by later blocks (e.g. CfL luma subsample). Write the full
	// transform extent clamped to the window's writable (superblock-aligned,
	// allocation-bounded) extent. The predWidth/predHeight passed to the
	// predictor math below stay un-clipped so edge weighting matches libaom.
	width, height, ok := frameWorkClipVisiblePixelsToWindow(window, absX, absY, predWidth, predHeight)
	if !ok {
		if frameWorkPlaneBlockStartsBeyondOutput(b.Output, FrameWorkPlaneY, absX, absY) {
			return nil
		}
		return ErrInvalidBatch
	}
	dst := frameWorkPlaneFromWindow(window)
	x := absX - window.X
	y := absY - window.Y
	edgeBlock := frameWorkPredictionTransformEdgeBlock(visit.Block, visit.Block.X4, visit.Block.Y4, tx.X4, tx.Y4)
	edgeBlock = frameWorkPredictionEdgeBlockForWindow(edgeBlock, absX, absY, window)
	if visit.Prediction.Palette.YSize > 0 {
		baseX, baseY, err := frameWorkBlockLumaPosition(visit.Block)
		if err != nil {
			return err
		}
		if err := frameWorkPredictLumaPalette(dst, window.BytesPerSample, x, y, width, height, visit.Block, visit.Prediction.Palette, absX-baseX, absY-baseY); err != nil {
			return err
		}
		return nil
	}

	if visit.Prediction.FilterIntraValid {
		mode, ok := frameWorkFilterIntraPredictionMode(visit.Prediction.FilterIntraMode)
		if !ok {
			return ErrInvalidBatch
		}
		readBoundX, readBoundY := frameWorkWindowEdgeReadBound(window)
		edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, readBoundX, readBoundY, edgeBlock, scratch, true)
		if err != nil {
			return err
		}
		if err := prediction.PredictFilterIntraPlaneBlockWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, mode, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	if angle, ok := frameWorkLumaIntraDirectionalAngle(visit.Prediction.LumaMode, visit.Prediction.LumaAngleDelta); ok {
		readBoundX, readBoundY := frameWorkWindowEdgeReadBound(window)
		allowTopRight, allowBottomLeft := frameWorkLumaDirectionalExtendedEdges(edgeBlock, b.Sequence.SBSizeMIB, region.MIColEnd, region.MIRowEnd, absX, absY, predWidth, predHeight)
		edges, err := frameWorkDirectionalPredictionEdges(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, predWidth, predHeight, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, visit.Prediction.IntraEdgeSmoothNeighbor, allowTopRight, allowBottomLeft, readBoundX, readBoundY)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlockWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	mode, ok := frameWorkLumaIntraPredictionMode(visit.Prediction.LumaMode)
	if !ok {
		return ErrInvalidBatch
	}
	readBoundX, readBoundY := frameWorkWindowEdgeReadBound(window)
	edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, readBoundX, readBoundY, edgeBlock, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockChromaIntraPlane(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, scratch *FrameWorkIntraPredictionScratch) error {
	geom, present, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !present {
		return err
	}
	region, err := b.JobRegion(index)
	if err != nil {
		return err
	}
	predWidth, predHeight, err := frameWorkBlockPlanePredictionExtentPixels(visit.Block, b.Sequence.ColorConfig, plane)
	if err != nil {
		return err
	}
	// libaom predicts the whole block (predWidth x predHeight) into the frame
	// buffer; for a block straddling the cropped edge the trailing rows/cols
	// spill into the superblock-aligned padding and feed later reads (CfL luma
	// subsample, CDEF/restoration bottom-edge input). Write the full extent
	// clamped to the window's writable extent rather than the cropped-visible
	// geom.Width/Height. predWidth/predHeight stay un-clipped for edge weighting.
	writeWidth, writeHeight, ok := frameWorkClipVisiblePixelsToWindow(geom.Window, geom.X, geom.Y, predWidth, predHeight)
	if !ok {
		writeWidth, writeHeight = geom.Width, geom.Height
	}
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	if visit.Prediction.Palette.UVSize > 0 {
		if err := frameWorkPredictChromaPalette(geom.Output, geom.BytesPerSample, geom.X, geom.Y, geom.Width, geom.Height, visit.Block, b.Sequence.ColorConfig, plane, visit.Prediction.Palette, 0, 0); err != nil {
			return err
		}
		return nil
	}
	readBoundX, readBoundY := frameWorkWindowEdgeReadBoundAbsolute(geom.Window)
	if angle, ok := frameWorkChromaIntraDirectionalAngle(visit.Prediction.ChromaMode, visit.Prediction.ChromaAngleDelta); ok {
		allowTopRight, allowBottomLeft := frameWorkChromaDirectionalExtendedEdges(edgeBlock, b.Sequence.SBSizeMIB, region.MIColEnd, region.MIRowEnd, geom.X, geom.Y, geom.X, geom.Y, geom.Width, geom.Height, geom.SubsamplingX, geom.SubsamplingY)
		edges, err := frameWorkDirectionalPredictionEdges(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, visit.Prediction.ChromaIntraEdgeSmoothNeighbor, allowTopRight, allowBottomLeft, readBoundX, readBoundY)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlockWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, geom.Width, geom.Height, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	mode, ok := frameWorkChromaIntraPredictionMode(visit.Prediction.ChromaMode)
	if !ok {
		return ErrInvalidBatch
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, writeWidth, writeHeight, predWidth, predHeight, readBoundX, readBoundY, edgeBlock, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, writeWidth, writeHeight, predWidth, predHeight, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockChromaIntraTransform(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, tx tile.TransformBlock, scratch *FrameWorkIntraPredictionScratch) error {
	geom, present, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !present {
		return err
	}
	region, err := b.JobRegion(index)
	if err != nil {
		return err
	}
	if !frameWorkBlockWithinJobRegion(region, visit.Block) {
		return ErrInvalidBatch
	}
	_, _, predWidth, predHeight, err := frameWorkTransformVisibleAndExtentPixels(tx)
	if err != nil {
		return err
	}
	baseX4 := visit.Block.X4 >> int(frameWorkSubsampleShift(geom.SubsamplingX))
	baseY4 := visit.Block.Y4 >> int(frameWorkSubsampleShift(geom.SubsamplingY))
	offX4 := tx.X4 - baseX4
	offY4 := tx.Y4 - baseY4
	if offX4 < 0 || offY4 < 0 {
		return ErrInvalidBatch
	}
	absX := geom.X + offX4*4
	absY := geom.Y + offY4*4
	// Write the full transform extent (clamped to the superblock-aligned
	// writable window), matching libaom predicting the whole tx_size into the
	// frame buffer; the predWidth/predHeight passed to the predictor stay
	// un-clipped for libaom-faithful edge weighting.
	width, height, ok := frameWorkClipVisiblePixelsToWindow(geom.Window, absX, absY, predWidth, predHeight)
	if !ok {
		if frameWorkPlaneBlockStartsBeyondOutput(b.Output, plane, absX, absY) {
			return nil
		}
		return ErrInvalidBatch
	}
	x := absX
	y := absY
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	edgeBlock = frameWorkPredictionTransformEdgeBlock(edgeBlock, baseX4, baseY4, tx.X4, tx.Y4)
	edgeBlock = frameWorkPredictionEdgeBlockForWindow(edgeBlock, absX, absY, geom.Window)

	if visit.Prediction.Palette.UVSize > 0 {
		if err := frameWorkPredictChromaPalette(geom.Output, geom.BytesPerSample, x, y, width, height, visit.Block, b.Sequence.ColorConfig, plane, visit.Prediction.Palette, absX-geom.X, absY-geom.Y); err != nil {
			return err
		}
		return nil
	}

	readBoundX, readBoundY := frameWorkWindowEdgeReadBoundAbsolute(geom.Window)
	if angle, ok := frameWorkChromaIntraDirectionalAngle(visit.Prediction.ChromaMode, visit.Prediction.ChromaAngleDelta); ok {
		allowTopRight, allowBottomLeft := frameWorkChromaDirectionalExtendedEdges(edgeBlock, b.Sequence.SBSizeMIB, region.MIColEnd, region.MIRowEnd, geom.X, geom.Y, absX, absY, predWidth, predHeight, geom.SubsamplingX, geom.SubsamplingY)
		edges, err := frameWorkDirectionalPredictionEdges(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, predWidth, predHeight, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, visit.Prediction.ChromaIntraEdgeSmoothNeighbor, allowTopRight, allowBottomLeft, readBoundX, readBoundY)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlockWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	mode, ok := frameWorkChromaIntraPredictionMode(visit.Prediction.ChromaMode)
	if !ok {
		return ErrInvalidBatch
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, readBoundX, readBoundY, edgeBlock, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockChromaCFLPlane(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, scratch *FrameWorkCFLPredictionScratch) error {
	geom, present, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !present {
		return err
	}
	if b.Output == nil {
		return ErrInvalidBatch
	}
	// libaom reconstructs whole transform blocks out to the MI-aligned frame
	// grid, so the luma samples a CfL block reads exist past the cropped
	// luma width/height. Extend the luma plane view to its MI-aligned writable
	// extent (matching frameWorkExtendPlaneToClip for prediction writes) so the
	// subsample read can address those reconstructed past-crop samples.
	lumaWindow, err := b.JobOutputPlane(index, FrameWorkPlaneY)
	if err != nil {
		return err
	}
	luma := frameWorkExtendPlaneToClip(b.Output.Y, lumaWindow, b.Output.Layout.BytesPerSample)
	lumaX := geom.X
	lumaY := geom.Y
	fullWidth, fullHeight, err := frameWorkBlockPlanePredictionExtentPixels(visit.Block, b.Sequence.ColorConfig, plane)
	if err != nil {
		return err
	}
	// libaom's cfl_store_tx subsamples the FULL reconstructed transform of the
	// luma block (it has no boundary clamp; the trailing rows/cols of a
	// transform crossing the cropped edge were genuinely reconstructed into the
	// superblock-aligned padding). Subsample the full chroma extent
	// (fullWidth x fullHeight) rather than the cropped-visible geom.Width/Height
	// so the CfL AC matches libaom exactly. cfl_pad (PadCFLReconQ3) only
	// replicates when a stored dimension is short of the full extent, which now
	// happens solely on genuine overrun past the underlying allocation.
	lumaW := fullWidth
	lumaH := fullHeight
	bufWidth := fullWidth
	bufHeight := fullHeight
	if geom.SubsamplingX {
		lumaX <<= 1
		lumaW <<= 1
	}
	if geom.SubsamplingY {
		lumaY <<= 1
		lumaH <<= 1
	}
	// Clamp the luma read to the available (superblock-aligned, allocation-
	// bounded) luma extent. Only a transform that overruns the actual
	// allocation (never, for an in-grid block, since the allocation is
	// superblock-aligned) falls back to cfl_pad replication for the missing
	// chroma column/row.
	if lumaX+lumaW > luma.Width {
		availLumaW := luma.Width - lumaX
		if geom.SubsamplingX {
			availLumaW &^= 1
		}
		if availLumaW < lumaW {
			lumaW = availLumaW
			if geom.SubsamplingX {
				bufWidth = lumaW >> 1
			} else {
				bufWidth = lumaW
			}
		}
	}
	if lumaY+lumaH > luma.Height {
		availLumaH := luma.Height - lumaY
		if geom.SubsamplingY {
			availLumaH &^= 1
		}
		if availLumaH < lumaH {
			lumaH = availLumaH
			if geom.SubsamplingY {
				bufHeight = lumaH >> 1
			} else {
				bufHeight = lumaH
			}
		}
	}
	if lumaW <= 0 || lumaH <= 0 || bufWidth <= 0 || bufHeight <= 0 {
		return ErrInvalidBatch
	}
	if err := frameWorkSubsampleLumaCFLQ3(scratch.ReconQ3[:], luma, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, lumaX, lumaY, lumaW, lumaH, geom.SubsamplingX, geom.SubsamplingY); err != nil {
		return err
	}
	if _, _, err := prediction.PadCFLReconQ3(scratch.ReconQ3[:], bufWidth, bufHeight, fullWidth, fullHeight); err != nil {
		return ErrInvalidBatch
	}
	if err := prediction.SubtractCFLAverage(scratch.ReconQ3[:], scratch.ACQ3[:], fullWidth, fullHeight); err != nil {
		return ErrInvalidBatch
	}
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	readBoundX, readBoundY := frameWorkWindowEdgeReadBoundAbsolute(geom.Window)
	// libaom computes the CfL DC predictor and applies CfL over the chroma
	// TRANSFORM size (av1_cfl_predict_block(xd, ..., tx_size, plane)), writing
	// the whole transform into the frame buffer; the trailing past-cropped-edge
	// rows/cols land in the superblock-aligned padding (and feed later
	// neighbor/filter reads). Using the frame-edge-clipped visible extent here
	// would average a different number of DC neighbor samples (e.g. only the 4
	// visible left rows of a bottom-edge TX_16X8 instead of all 8), shifting the
	// DC base and therefore every CfL sample by a constant. Predict over the
	// full transform extent (fullWidth x fullHeight) clamped only to the
	// superblock-aligned writable window, matching libaom.
	writeWidth, writeHeight, ok := frameWorkClipVisiblePixelsToWindow(geom.Window, geom.X, geom.Y, fullWidth, fullHeight)
	if !ok {
		writeWidth, writeHeight = geom.Width, geom.Height
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, writeWidth, writeHeight, fullWidth, fullHeight, readBoundX, readBoundY, edgeBlock, &scratch.Intra, false)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, writeWidth, writeHeight, fullWidth, fullHeight, prediction.IntraModeDC, edges); err != nil {
		return ErrInvalidBatch
	}
	predType := prediction.CFLPredU
	if plane == FrameWorkPlaneV {
		predType = prediction.CFLPredV
	}
	alphaQ3, err := prediction.CFLAlphaQ3(visit.Prediction.CFLAlpha.Index, visit.Prediction.CFLAlpha.JointSign, predType)
	if err != nil {
		return ErrInvalidBatch
	}
	if err := prediction.PredictCFLPlaneBlockVisible(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, writeWidth, writeHeight, fullWidth, fullHeight, scratch.ACQ3[:], alphaQ3); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

// PredictBlockLumaInter writes single-reference translational luma inter
// prediction for one decoded block-loop visit. Switchable filters, compound
// blending, scaled references, warped/global refinement, and chroma prediction
// are rejected or handled by later inter-prediction stages.
func (b FrameWorkBatch) PredictBlockLumaInter(index int, visit tile.BlockLoopVisit) error {
	filters, err := frameWorkVisitMotionFilters(b.TileInfo, visit.Prediction)
	if err != nil {
		return err
	}
	return b.PredictBlockLumaInterWithFilters(index, visit, filters)
}

// PredictBlockLumaInterWithFilters is PredictBlockLumaInter with explicit
// interpolation filters. It is useful for blocks whose switchable filter syntax
// has already been decoded by a caller.
func (b FrameWorkBatch) PredictBlockLumaInterWithFilters(index int, visit tile.BlockLoopVisit, filters motion.InterpFilters) error {
	if !visit.Prediction.Valid ||
		visit.Prediction.Intra ||
		frameWorkPredictionIsIntrabc(visit.Prediction) ||
		!visit.Prediction.InterMotionValid {
		return ErrInvalidBatch
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		motionResult.References.Ref[1] != tile.ReferenceFrameNone {
		return ErrInvalidBatch
	}
	return b.predictBlockInterPlaneWithFilters(index, visit, FrameWorkPlaneY, filters)
}

// PredictBlockLumaInterOBMC writes single-reference luma inter prediction and
// blends the above/left OBMC neighbor predictors selected by motion_mode.
func (b FrameWorkBatch) PredictBlockLumaInterOBMC(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch) error {
	filters, err := frameWorkVisitMotionFilters(b.TileInfo, visit.Prediction)
	if err != nil {
		return err
	}
	return b.PredictBlockLumaInterOBMCWithFilters(index, visit, scratch, filters)
}

// PredictBlockLumaInterOBMCWithFilters is PredictBlockLumaInterOBMC with
// explicit interpolation filters for the current block's base predictor.
func (b FrameWorkBatch) PredictBlockLumaInterOBMCWithFilters(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	return b.predictBlockInterOBMCPlaneWithFilters(index, visit, FrameWorkPlaneY, scratch, filters)
}

// PredictBlockLumaInterCompoundAverage writes average compound luma inter
// prediction for one decoded block-loop visit. Inter-intra, scaled references,
// and warped/global refinement are rejected until those paths are integrated.
func (b FrameWorkBatch) PredictBlockLumaInterCompoundAverage(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch) error {
	filters, err := frameWorkVisitMotionFilters(b.TileInfo, visit.Prediction)
	if err != nil {
		return err
	}
	return b.PredictBlockLumaInterCompoundAverageWithFilters(index, visit, scratch, filters)
}

// PredictBlockLumaInterCompoundAverageWithFilters is
// PredictBlockLumaInterCompoundAverage with explicit interpolation filters. It
// only accepts decoded CompoundTypeAverage blocks.
func (b FrameWorkBatch) PredictBlockLumaInterCompoundAverageWithFilters(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if visit.Prediction.CompoundBlend.Type != tile.CompoundTypeAverage {
		return ErrInvalidBatch
	}
	return b.PredictBlockLumaInterCompoundWithFilters(index, visit, scratch, filters)
}

// PredictBlockLumaInterCompound writes compound luma inter prediction for one
// decoded block-loop visit. It currently covers average, distance-weighted,
// wedge, and difference-weighted compound. Inter-intra, scaled references, and
// warped/global refinement are rejected until those paths are integrated.
func (b FrameWorkBatch) PredictBlockLumaInterCompound(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch) error {
	filters, err := frameWorkVisitMotionFilters(b.TileInfo, visit.Prediction)
	if err != nil {
		return err
	}
	return b.PredictBlockLumaInterCompoundWithFilters(index, visit, scratch, filters)
}

// PredictBlockLumaInterCompoundWithFilters is PredictBlockLumaInterCompound
// with explicit interpolation filters.
func (b FrameWorkBatch) PredictBlockLumaInterCompoundWithFilters(index int, visit tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if scratch == nil ||
		!visit.Prediction.Valid ||
		visit.Prediction.Intra ||
		frameWorkPredictionIsIntrabc(visit.Prediction) ||
		!visit.Prediction.InterMotionValid ||
		!visit.Prediction.CompoundBlendValid {
		return ErrInvalidBatch
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	if visit.Prediction.MotionModeValid && visit.Prediction.MotionMode != tile.MotionModeTranslation {
		return ErrInvalidBatch
	}
	blend := visit.Prediction.CompoundBlend
	if blend.Type != tile.CompoundTypeAverage &&
		blend.Type != tile.CompoundTypeDistWtd &&
		blend.Type != tile.CompoundTypeWedge &&
		blend.Type != tile.CompoundTypeDiffWtd {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if !motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		!motionResult.References.Ref[1].Valid() {
		return ErrInvalidBatch
	}
	return b.predictBlockInterCompoundPlaneWithFilters(index, visit, FrameWorkPlaneY, scratch, filters)
}

func (b FrameWorkBatch) predictBlockInterPlaneWithFilters(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, filters motion.InterpFilters) error {
	if frameWorkPredictionIsIntrabc(visit.Prediction) {
		return b.predictBlockIntrabcPlane(index, visit, plane)
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	if visit.Prediction.MotionModeValid && visit.Prediction.MotionMode == tile.MotionModeWarp && !visit.Prediction.WarpedMotionInvalid {
		_, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		// libaom's av1_init_warp_params() gates WARP_PRED on the *full* plane
		// block dimensions (xd->plane[plane].width/height), not the
		// frame-edge-clipped visible extent: a chroma block whose visible
		// height shrinks below 8 at the bottom partial superblock still warps
		// when its un-clipped plane block is >= 8 on both sides. Using the
		// clipped geom here instead falls back to TRANSLATION_PRED and diverges
		// from libaom on the bottom partial-SB chroma rows. The plane-present
		// (!ok) check stays above this call: the plane-block-size lookup errors
		// for absent planes (monochrome chroma), which must short-circuit to nil.
		warpable, err := frameWorkBlockPlaneWarpAllowed(visit.Block, b.Sequence.ColorConfig, plane)
		if err != nil {
			return err
		}
		if warpable {
			return b.predictBlockInterWarpPlaneWithFilters(index, visit, plane, filters)
		}
		return b.predictBlockInterReferencePlaneToOutput(index, visit.Block, plane, visit.Prediction.InterMotion.References.Ref[0], visit.Prediction.InterMotion.MV[0], filters)
	}
	if visit.Prediction.MotionModeValid && !frameWorkPredictionUsesTranslation(visit.Prediction) {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		motionResult.References.Ref[1] != tile.ReferenceFrameNone {
		return ErrInvalidBatch
	}
	// libaom promotes GLOBALMV blocks with a non-translational frame-level
	// warp model and a block min-side of at least 8 luma samples to
	// WARP_PRED through av1_init_warp_params(). The block-level motion_mode
	// stays SIMPLE_TRANSLATION; the warp uses the frame-level params.
	if visit.Prediction.GlobalWarpedMotionValid {
		_, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		// As with local warp, libaom gates the global-warp WARP_PRED promotion
		// on the un-clipped plane block dimensions, not the visible extent. The
		// plane-present (!ok) check stays above this call (monochrome chroma).
		warpable, err := frameWorkBlockPlaneWarpAllowed(visit.Block, b.Sequence.ColorConfig, plane)
		if err != nil {
			return err
		}
		if warpable {
			return b.predictBlockInterGlobalWarpPlaneWithFilters(index, visit, plane, filters)
		}
	}
	// libaom's build_inter_predictors_sub8x8 splits the chroma block of an
	// inter block whose luma width or height is 4 (under chroma
	// subsampling) into per-luma-cell sub-blocks, each predicted with
	// its own neighbor's MV. The tile decoder pre-computed those cells in
	// visit.Prediction.SubChromaInter; we drive them here for U/V only,
	// keeping luma on the standard single-MV path.
	if visit.Prediction.SubChromaInterValid && (plane == FrameWorkPlaneU || plane == FrameWorkPlaneV) {
		return b.predictBlockInterSubChromaPlane(index, visit, plane)
	}
	return b.predictBlockInterReferencePlaneToOutput(index, visit.Block, plane, motionResult.References.Ref[0], motionResult.MV[0], filters)
}

// predictBlockInterSubChromaPlane drives libaom's sub8x8 chroma prediction
// for one chroma plane. It iterates over the chroma sub-blocks recorded by
// the tile decoder (see tile.CollectSubChromaInterCells) and dispatches a
// translational predictor per cell using that cell's own MV.
func (b FrameWorkBatch) predictBlockInterSubChromaPlane(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane) error {
	if plane != FrameWorkPlaneU && plane != FrameWorkPlaneV {
		return ErrInvalidBatch
	}
	if !visit.Prediction.SubChromaInterValid {
		return ErrInvalidBatch
	}
	geom, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}
	sub := visit.Prediction.SubChromaInter
	if sub.Count <= 0 {
		return ErrInvalidBatch
	}
	for i := 0; i < sub.Count; i++ {
		cell := sub.Cells[i]
		cellX := geom.X + cell.OffsetX
		cellY := geom.Y + cell.OffsetY
		width := cell.Width
		height := cell.Height
		if cellX < geom.X || cellY < geom.Y {
			return ErrInvalidBatch
		}
		if cellX+width > geom.X+geom.Width {
			width = geom.X + geom.Width - cellX
		}
		if cellY+height > geom.Y+geom.Height {
			height = geom.Y + geom.Height - cellY
		}
		if width <= 0 || height <= 0 {
			continue
		}
		reference, refOk := frameWorkReferenceFromTile(cell.Reference)
		if !refOk {
			return ErrInvalidBatch
		}
		refWindow, err := b.ReferencePlane(reference, plane)
		if err != nil {
			return err
		}
		ref := frame.Plane{
			Pix:    refWindow.Pix,
			Stride: refWindow.Stride,
			Width:  refWindow.Width,
			Height: refWindow.Height,
		}
		sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
		if err != nil {
			return err
		}
		if !sameSize {
			if err := frameWorkPredictScaledReferencePlane(geom, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth,
				cellX, cellY, cellX, cellY, width, height, cell.MV, geom.SubsamplingX, geom.SubsamplingY, cell.InterpFilters); err != nil {
				return err
			}
			continue
		}
		refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(cellX, cellY, cell.MV, geom.SubsamplingX, geom.SubsamplingY)
		if err != nil {
			return ErrInvalidBatch
		}
		if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(geom.Output, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, cellX, cellY, refX, refY, width, height, subX, subY, cell.InterpFilters); err != nil {
			return ErrInvalidBatch
		}
	}
	return nil
}

// predictBlockInterGlobalWarpPlaneWithFilters drives libaom's WARP_PRED dispatch
// with the translational interpolation filters that libaom would use when the
// scaled-reference gate downgrades WARP_PRED to TRANSLATION_PRED (see
// allow_warp() in av1/common/reconinter.c, which returns 0 whenever
// av1_is_scaled(sf) is true).
func (b FrameWorkBatch) predictBlockInterGlobalWarpPlaneWithFilters(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, filters motion.InterpFilters) error {
	if !visit.Prediction.Valid ||
		visit.Prediction.Intra ||
		frameWorkPredictionIsIntrabc(visit.Prediction) ||
		!visit.Prediction.InterMotionValid ||
		!visit.Prediction.GlobalWarpedMotionValid {
		return ErrInvalidBatch
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		motionResult.References.Ref[1] != tile.ReferenceFrameNone {
		return ErrInvalidBatch
	}
	geom, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !ok {
		return err
	}
	reference, ok := frameWorkReferenceFromTile(motionResult.References.Ref[0])
	if !ok {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return err
	}
	if !sameSize {
		// libaom: av1_is_scaled(sf) makes allow_warp() return 0, so the
		// mode stays TRANSLATION_PRED and av1_make_inter_predictor()
		// runs the scaled 8-tap convolver on the block-level MV instead
		// of the global warp matrix.
		return frameWorkPredictScaledReferencePlane(geom, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth,
			geom.X, geom.Y, geom.X, geom.Y, geom.Width, geom.Height, motionResult.MV[0], geom.SubsamplingX, geom.SubsamplingY, filters)
	}
	model := visit.Prediction.GlobalWarpedMotion
	if err := motion.PredictWarpedPlaneBlockBitDepth(geom.Output, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, model.Params.Matrix, model.Alpha, model.Beta, model.Gamma, model.Delta, geom.SubsamplingX, geom.SubsamplingY); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockIntrabcPlane(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane) error {
	if !visit.Prediction.Valid ||
		visit.Prediction.Intra ||
		!frameWorkPredictionIsIntrabc(visit.Prediction) ||
		!visit.Prediction.InterMotionValid {
		return ErrInvalidBatch
	}
	x, y, width, height, subsamplingX, subsamplingY, present, err := frameWorkBlockPlanePosition(visit.Block, b.Sequence.ColorConfig, plane)
	if err != nil || !present {
		return err
	}
	if frameWorkPlaneBlockStartsBeyondOutput(b.Output, plane, x, y) {
		return nil
	}
	window, err := b.JobOutputPlane(index, plane)
	if err != nil {
		return err
	}
	width, height, ok := frameWorkClipVisiblePixelsToWindow(window, x, y, width, height)
	if !ok {
		return ErrInvalidBatch
	}
	if b.Output == nil {
		return ErrInvalidBatch
	}
	output, outputSubX, outputSubY, ok := frameWorkFramePlane(b.Output, plane)
	if !ok || b.Output.Layout.BytesPerSample <= 0 {
		return ErrInvalidBatch
	}
	if outputSubX != subsamplingX || outputSubY != subsamplingY {
		return ErrInvalidBatch
	}
	geom := frameWorkPredictionPlaneGeometry{
		Output:         output,
		Window:         window,
		X:              x,
		Y:              y,
		Width:          width,
		Height:         height,
		SubsamplingX:   subsamplingX,
		SubsamplingY:   subsamplingY,
		BytesPerSample: b.Output.Layout.BytesPerSample,
	}
	mv := visit.Prediction.InterMotion.MV[0]
	if mv.Row%8 != 0 || mv.Col%8 != 0 {
		return ErrInvalidBatch
	}
	rowOffset := int(mv.Row / 8)
	colOffset := int(mv.Col / 8)
	if geom.SubsamplingY {
		rowOffset >>= 1
	}
	if geom.SubsamplingX {
		colOffset >>= 1
	}
	srcX := geom.X + colOffset
	srcY := geom.Y + rowOffset
	if srcX < 0 || srcY < 0 || srcX+geom.Width > geom.Output.Width || srcY+geom.Height > geom.Output.Height {
		return ErrInvalidBatch
	}
	rowBytes, ok := frameWorkCheckedMul(geom.Width, geom.BytesPerSample)
	if !ok {
		return ErrInvalidBatch
	}
	for row := 0; row < geom.Height; row++ {
		srcOff, ok := frameWorkPlaneSampleOffset(geom.Output, geom.BytesPerSample, srcX, srcY+row)
		if !ok {
			return ErrInvalidBatch
		}
		dstOff, ok := frameWorkPlaneSampleOffset(geom.Output, geom.BytesPerSample, geom.X, geom.Y+row)
		if !ok {
			return ErrInvalidBatch
		}
		copy(geom.Output.Pix[dstOff:dstOff+rowBytes], geom.Output.Pix[srcOff:srcOff+rowBytes])
	}
	return nil
}

func (b FrameWorkBatch) predictBlockInterIntraPlaneWithFilters(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if scratch == nil ||
		frameWorkPredictionIsIntrabc(visit.Prediction) ||
		!visit.Prediction.InterIntraValid ||
		!visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	if visit.Prediction.MotionModeValid && !frameWorkPredictionUsesTranslation(visit.Prediction) {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		motionResult.References.Ref[1] != tile.ReferenceFrameNone {
		return ErrInvalidBatch
	}
	geom, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !ok {
		return err
	}
	inter, err := frameWorkInterScratchPlane(scratch.First[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		return err
	}
	intra, err := frameWorkInterScratchPlane(scratch.Second[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		return err
	}
	// libaom applies global warp to the inter part of an inter-intra block when
	// is_global_mv_block holds (GLOBALMV + non-translational frame model + luma
	// min-side >= 8); inter-intra forces SIMPLE_TRANSLATION motion_mode but does
	// not disable the global-warp predictor. Stage the warp in the inter scratch,
	// matching the non-inter-intra global-warp path; chroma planes < 8 and scaled
	// references fall back to the translational predictor as libaom does.
	if visit.Prediction.GlobalWarpedMotionValid && geom.Width >= 8 && geom.Height >= 8 {
		if err := b.predictBlockInterGlobalWarpToScratch(inter, plane, motionResult.References.Ref[0], visit.Prediction.GlobalWarpedMotion, motionResult.MV[0], geom, filters); err != nil {
			return err
		}
	} else if err := b.predictBlockInterReferencePlaneToScratch(inter, visit.Block, plane, motionResult.References.Ref[0], motionResult.MV[0], geom, filters); err != nil {
		return err
	}
	mode, ok := frameWorkInterIntraPredictionMode(visit.Prediction.InterIntra.Mode)
	if !ok {
		return ErrInvalidBatch
	}
	// libaom av1_build_intra_predictors_for_interintra() builds the intra part at
	// the full plane-block dimensions (pd->width/height, plane_bsize). When the
	// block straddles the right/bottom frame edge the visible extent
	// (geom.Width/Height) is not a valid intra block size, so the smooth/DC
	// weight tables and edge lengths must be selected from the full plane-block
	// extent while only the visible sub-rectangle is written into the intra
	// scratch (the blend below combines just that visible extent).
	predWidth, predHeight, err := frameWorkBlockPlanePredictionExtentPixels(visit.Block, b.Sequence.ColorConfig, plane)
	if err != nil {
		return err
	}
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	readBoundX, readBoundY := frameWorkWindowEdgeReadBoundAbsolute(geom.Window)
	edges, err := frameWorkIntraPredictionEdgesWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, predWidth, predHeight, readBoundX, readBoundY, edgeBlock, &scratch.Intra, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtent(intra, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, 0, 0, geom.Width, geom.Height, predWidth, predHeight, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	mask := scratch.Mask[:]
	maskStride := geom.Width
	maskSubX := false
	maskSubY := false
	if visit.Prediction.InterIntra.UseWedge {
		// libaom combine_interintra() builds the wedge mask at the full luma
		// block resolution and blends with mask_stride = block_size_wide[bsize]
		// (the full luma block width), sub-sampling the mask for chroma. The
		// blend then runs over the (clamped-to-visible) plane extent. Using the
		// full luma width as the stride -- rather than the visible width -- keeps
		// the mask addressable when the block straddles the right/bottom frame
		// edge (visible < full block).
		lumaWidth, lumaHeight, err := frameWorkBlockLumaPredictionExtentPixels(visit.Block)
		if err != nil {
			return err
		}
		maskStride = lumaWidth
		maskSubX = geom.SubsamplingX
		maskSubY = geom.SubsamplingY
		if err := frameWorkBuildWedgeMask(mask, maskStride, visit.Block.Size, visit.Prediction.InterIntra.WedgeIndex, false); err != nil {
			return err
		}
		mask = mask[:lumaWidth*lumaHeight]
	} else {
		// libaom combine_interintra() builds the smooth/DC inter-intra mask at the
		// full plane-block dimensions (build_smooth_interintra_mask over bw x bh)
		// with mask_stride = bw and no sub-sampling. The mask weights depend on the
		// full plane-block size, so build at predWidth x predHeight (with stride
		// predWidth) and blend over the visible sub-rectangle. This keeps the mask
		// addressable and bit-exact when the block straddles the frame edge.
		maskStride = predWidth
		if err := frameWorkBuildInterIntraMask(mask, maskStride, predWidth, predHeight, visit.Prediction.InterIntra.Mode); err != nil {
			return err
		}
		mask = mask[:predWidth*predHeight]
	}
	return frameWorkBlendInterIntraBlock(geom.Output, inter, intra, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, mask, maskStride, maskSubX, maskSubY)
}

func frameWorkPredictionUsesTranslation(pred tile.BlockPredictionModeResult) bool {
	if pred.MotionMode == tile.MotionModeTranslation {
		return true
	}
	return pred.MotionMode == tile.MotionModeWarp && pred.WarpedMotionInvalid
}

func frameWorkPredictionIsIntrabc(pred tile.BlockPredictionModeResult) bool {
	return pred.IntrabcValid && pred.Intrabc
}

// predictBlockInterWarpPlaneWithFilters drives libaom's WARP_PRED dispatch with
// the translational interpolation filters that libaom would use when the
// scaled-reference gate downgrades WARP_PRED to TRANSLATION_PRED (see
// allow_warp() in av1/common/reconinter.c, which returns 0 whenever
// av1_is_scaled(sf) is true).
func (b FrameWorkBatch) predictBlockInterWarpPlaneWithFilters(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, filters motion.InterpFilters) error {
	if !visit.Prediction.Valid ||
		visit.Prediction.Intra ||
		frameWorkPredictionIsIntrabc(visit.Prediction) ||
		!visit.Prediction.InterMotionValid ||
		!visit.Prediction.WarpedMotionValid {
		return ErrInvalidBatch
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		motionResult.References.Ref[1] != tile.ReferenceFrameNone {
		return ErrInvalidBatch
	}
	geom, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !ok {
		return err
	}
	reference, ok := frameWorkReferenceFromTile(motionResult.References.Ref[0])
	if !ok {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return err
	}
	if !sameSize {
		// libaom: av1_is_scaled(sf) makes allow_warp() return 0, so the
		// mode stays TRANSLATION_PRED and av1_make_inter_predictor()
		// runs the scaled 8-tap convolver on the block-level MV instead
		// of the local warp matrix.
		return frameWorkPredictScaledReferencePlane(geom, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth,
			geom.X, geom.Y, geom.X, geom.Y, geom.Width, geom.Height, motionResult.MV[0], geom.SubsamplingX, geom.SubsamplingY, filters)
	}
	model := visit.Prediction.WarpedMotion
	if err := motion.PredictWarpedPlaneBlockBitDepth(geom.Output, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, model.Params.Matrix, model.Alpha, model.Beta, model.Gamma, model.Delta, geom.SubsamplingX, geom.SubsamplingY); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockInterOBMCPlaneWithFilters(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if scratch == nil ||
		!visit.Prediction.Valid ||
		visit.Prediction.Intra ||
		frameWorkPredictionIsIntrabc(visit.Prediction) ||
		!visit.Prediction.InterMotionValid ||
		!visit.Prediction.MotionModeValid ||
		visit.Prediction.MotionMode != tile.MotionModeOBMC ||
		!visit.Prediction.OverlappableNeighborsValid {
		return ErrInvalidBatch
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		motionResult.References.Ref[1] != tile.ReferenceFrameNone {
		return ErrInvalidBatch
	}
	geom, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !ok {
		return err
	}
	if err := b.predictBlockInterReferencePlaneToOutput(index, visit.Block, plane, motionResult.References.Ref[0], motionResult.MV[0], filters); err != nil {
		return err
	}
	tmp, err := frameWorkInterScratchPlane(scratch.First[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		return err
	}
	neighbors := visit.Prediction.OverlappableNeighbors
	// libaom av1_skip_u4x4_pred_in_obmc(): the above-row OBMC prediction is
	// skipped for planes whose (subsampled) block size is 4x4, 8x4 or 4x8.
	// In 4:2:0 an 8x8 luma OBMC block has a 4x4 chroma plane, so its chroma
	// above-blend is dropped while the luma above-blend (never <8x8 for OBMC)
	// is kept. The left column is always predicted (dir==1 is never skipped).
	skipAbove := (geom.Width == 4 && geom.Height == 4) ||
		(geom.Width == 8 && geom.Height == 4) ||
		(geom.Width == 4 && geom.Height == 8)
	if !skipAbove {
		for i := 0; i < neighbors.AboveCount; i++ {
			if err := b.predictAndBlendOBMCAbove(plane, geom, tmp, visit.Block, neighbors.Above[i]); err != nil {
				return err
			}
		}
	}
	for i := 0; i < neighbors.LeftCount; i++ {
		if err := b.predictAndBlendOBMCLeft(plane, geom, tmp, visit.Block, neighbors.Left[i]); err != nil {
			return err
		}
	}
	return nil
}

func (b FrameWorkBatch) predictBlockInterCompoundPlaneWithFilters(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if scratch == nil {
		return ErrInvalidBatch
	}
	if frameWorkPredictionIsIntrabc(visit.Prediction) {
		return ErrInvalidBatch
	}
	if visit.Prediction.InterIntraValid && visit.Prediction.InterIntra.Enabled {
		return ErrInvalidBatch
	}
	if visit.Prediction.MotionModeValid && visit.Prediction.MotionMode != tile.MotionModeTranslation {
		return ErrInvalidBatch
	}
	blend := visit.Prediction.CompoundBlend
	if blend.Type != tile.CompoundTypeAverage &&
		blend.Type != tile.CompoundTypeDistWtd &&
		blend.Type != tile.CompoundTypeWedge &&
		blend.Type != tile.CompoundTypeDiffWtd {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if !motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		!motionResult.References.Ref[1].Valid() {
		return ErrInvalidBatch
	}
	geom, ok, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !ok {
		return err
	}
	usesWarp0 := visit.Prediction.GlobalWarpedMotionCompoundValid[0] && geom.Width >= 8 && geom.Height >= 8
	usesWarp1 := visit.Prediction.GlobalWarpedMotionCompoundValid[1] && geom.Width >= 8 && geom.Height >= 8
	// libaom keeps each compound reference predictor at the un-rounded 16-bit
	// CONV_BUF precision (av1_dist_wtd_convolve_* / av1_warp_affine_c with
	// is_compound, round_1 = COMPOUND_ROUND1_BITS) and only rounds to a pixel
	// after the blend. Blending two already-rounded 8-bit predictors loses
	// precision (off-by-1..3 across every compound block). Both translational and
	// global-warp references produce the CONV_BUF directly; only scaled references
	// keep the legacy 8-bit blend until the scaled convolver emits CONV_BUF
	// precision.
	scaled0, err := b.frameWorkCompoundRefScaled(motionResult.References.Ref[0], plane, geom)
	if err != nil {
		return err
	}
	scaled1, err := b.frameWorkCompoundRefScaled(motionResult.References.Ref[1], plane, geom)
	if err != nil {
		return err
	}
	if !scaled0 && !scaled1 {
		return b.predictBlockInterCompoundConvBuf(index, visit, plane, scratch, geom, blend, filters)
	}
	first, err := frameWorkInterScratchPlane(scratch.First[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		return err
	}
	second, err := frameWorkInterScratchPlane(scratch.Second[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		return err
	}
	// libaom warps each reference of a compound GLOBAL_GLOBALMV block with its
	// own global motion params (av1_init_warp_params per ref); chroma planes
	// below 8x8 and scaled refs fall back to translation, matching the
	// single-ref dispatch and av1_init_warp_params' block_width/height < 8 gate.
	if usesWarp0 {
		if err := b.predictBlockInterGlobalWarpToScratch(first, plane, motionResult.References.Ref[0], visit.Prediction.GlobalWarpedMotionCompound[0], motionResult.MV[0], geom, filters); err != nil {
			return err
		}
	} else if err := b.predictBlockInterReferencePlaneToScratch(first, visit.Block, plane, motionResult.References.Ref[0], motionResult.MV[0], geom, filters); err != nil {
		return err
	}
	if usesWarp1 {
		if err := b.predictBlockInterGlobalWarpToScratch(second, plane, motionResult.References.Ref[1], visit.Prediction.GlobalWarpedMotionCompound[1], motionResult.MV[1], geom, filters); err != nil {
			return err
		}
	} else if err := b.predictBlockInterReferencePlaneToScratch(second, visit.Block, plane, motionResult.References.Ref[1], motionResult.MV[1], geom, filters); err != nil {
		return err
	}
	switch blend.Type {
	case tile.CompoundTypeAverage, tile.CompoundTypeDistWtd:
		fwdOffset, bckOffset, err := b.frameWorkCompoundOffsets(motionResult.References, blend)
		if err != nil {
			return err
		}
		if err := frameWorkBlendCompoundBlock(geom.Output, first, second, geom.BytesPerSample, geom.X, geom.Y, geom.Width, geom.Height, fwdOffset, bckOffset); err != nil {
			return err
		}
	case tile.CompoundTypeWedge:
		lumaWidth, lumaHeight, ok := frameWorkBlockVisiblePixels(visit.Block)
		if !ok {
			return ErrInvalidBatch
		}
		maskStride := lumaWidth
		if err := frameWorkBuildWedgeMask(scratch.Mask[:], maskStride, visit.Block.Size, blend.WedgeIndex, blend.WedgeSign); err != nil {
			return err
		}
		if err := frameWorkBlendMaskedCompoundBlock(geom.Output, first, second, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, scratch.Mask[:lumaWidth*lumaHeight], maskStride, geom.SubsamplingX, geom.SubsamplingY); err != nil {
			return err
		}
	case tile.CompoundTypeDiffWtd:
		lumaWidth, lumaHeight, ok := frameWorkBlockVisiblePixels(visit.Block)
		if !ok {
			return ErrInvalidBatch
		}
		maskStride := lumaWidth
		if plane == FrameWorkPlaneY {
			if err := frameWorkBuildDiffWtdMask(scratch.Mask[:], maskStride, first, second, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.Width, geom.Height, blend.MaskType); err != nil {
				return err
			}
		}
		if err := frameWorkBlendMaskedCompoundBlock(geom.Output, first, second, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, scratch.Mask[:lumaWidth*lumaHeight], maskStride, geom.SubsamplingX, geom.SubsamplingY); err != nil {
			return err
		}
	default:
		return ErrInvalidBatch
	}
	return nil
}

// predictBlockInterCompoundConvBuf implements the translational compound inter
// prediction path at libaom's un-rounded 16-bit CONV_BUF precision: each
// reference is convolved into a CONV_BUF (av1_dist_wtd_convolve_*) and the two
// buffers are blended (distance-weighted average or A64 soft mask) with a single
// final rounding, matching the bitstream-exact compound output.
func (b FrameWorkBatch) predictBlockInterCompoundConvBuf(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, scratch *FrameWorkInterPredictionScratch, geom frameWorkPredictionPlaneGeometry, blend tile.CompoundBlendResult, filters motion.InterpFilters) error {
	motionResult := visit.Prediction.InterMotion
	bitDepth := b.Sequence.ColorConfig.BitDepth
	warp := geom.Width >= 8 && geom.Height >= 8
	warp0 := warp && visit.Prediction.GlobalWarpedMotionCompoundValid[0]
	warp1 := warp && visit.Prediction.GlobalWarpedMotionCompoundValid[1]
	if err := b.predictBlockInterCompoundRefToConvBuf(&scratch.Conv0, plane, motionResult.References.Ref[0], motionResult.MV[0], geom, filters, warp0, visit.Prediction.GlobalWarpedMotionCompound[0]); err != nil {
		return err
	}
	if err := b.predictBlockInterCompoundRefToConvBuf(&scratch.Conv1, plane, motionResult.References.Ref[1], motionResult.MV[1], geom, filters, warp1, visit.Prediction.GlobalWarpedMotionCompound[1]); err != nil {
		return err
	}
	switch blend.Type {
	case tile.CompoundTypeAverage, tile.CompoundTypeDistWtd:
		fwdOffset, bckOffset, err := b.frameWorkCompoundOffsets(motionResult.References, blend)
		if err != nil {
			return err
		}
		if err := motion.BlendCompoundAvg(geom.Output, &scratch.Conv0, &scratch.Conv1, geom.BytesPerSample, bitDepth, geom.X, geom.Y, geom.Width, geom.Height, fwdOffset, bckOffset); err != nil {
			return ErrInvalidBatch
		}
	case tile.CompoundTypeWedge:
		lumaWidth, lumaHeight, ok := frameWorkBlockVisiblePixels(visit.Block)
		if !ok {
			return ErrInvalidBatch
		}
		maskStride := lumaWidth
		if err := frameWorkBuildWedgeMask(scratch.Mask[:], maskStride, visit.Block.Size, blend.WedgeIndex, blend.WedgeSign); err != nil {
			return err
		}
		if err := motion.BlendCompoundMaskD16(geom.Output, &scratch.Conv0, &scratch.Conv1, geom.BytesPerSample, bitDepth, geom.X, geom.Y, geom.Width, geom.Height, scratch.Mask[:lumaWidth*lumaHeight], maskStride, geom.SubsamplingX, geom.SubsamplingY); err != nil {
			return ErrInvalidBatch
		}
	case tile.CompoundTypeDiffWtd:
		if !blend.MaskType.Valid() {
			return ErrInvalidBatch
		}
		lumaWidth, lumaHeight, ok := frameWorkBlockVisiblePixels(visit.Block)
		if !ok {
			return ErrInvalidBatch
		}
		maskStride := lumaWidth
		if plane == FrameWorkPlaneY {
			if err := motion.BuildDiffWtdMaskD16(scratch.Mask[:], maskStride, &scratch.Conv0, &scratch.Conv1, bitDepth, geom.Width, geom.Height, blend.MaskType == tile.DiffWtdMaskType38Inv); err != nil {
				return ErrInvalidBatch
			}
		}
		if err := motion.BlendCompoundMaskD16(geom.Output, &scratch.Conv0, &scratch.Conv1, geom.BytesPerSample, bitDepth, geom.X, geom.Y, geom.Width, geom.Height, scratch.Mask[:lumaWidth*lumaHeight], maskStride, geom.SubsamplingX, geom.SubsamplingY); err != nil {
			return ErrInvalidBatch
		}
	default:
		return ErrInvalidBatch
	}
	return nil
}

// frameWorkCompoundRefScaled reports whether a compound reference plane is
// scaled relative to the current frame. Scaled compound refs keep the legacy
// 8-bit blend path until the scaled convolver emits CONV_BUF precision.
func (b FrameWorkBatch) frameWorkCompoundRefScaled(refFrame tile.ReferenceFrame, plane FrameWorkPlane, geom frameWorkPredictionPlaneGeometry) (bool, error) {
	reference, ok := frameWorkReferenceFromTile(refFrame)
	if !ok {
		return false, ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return false, err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return false, err
	}
	return !sameSize, nil
}

// predictBlockInterCompoundRefToConvBuf fills a CONV_BUF with one translational
// reference predictor at compound precision, mirroring the origin/subpel
// derivation of predictBlockInterReferencePlaneToScratch.
func (b FrameWorkBatch) predictBlockInterCompoundRefToConvBuf(buf *motion.CompoundConvBuf, plane FrameWorkPlane, refFrame tile.ReferenceFrame, mv motion.Vector, geom frameWorkPredictionPlaneGeometry, filters motion.InterpFilters, useWarp bool, model tile.WarpedMotionModel) error {
	reference, ok := frameWorkReferenceFromTile(refFrame)
	if !ok {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return err
	}
	if !sameSize {
		// Scaled-reference compound is rare and not yet ported to the CONV_BUF
		// path; the caller never reaches here for scaled refs in the fast suite.
		return ErrInvalidBatch
	}
	if useWarp {
		if err := motion.PredictWarpedCompoundToConvBuf(buf, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, model.Params.Matrix, model.Alpha, model.Beta, model.Gamma, model.Delta, geom.SubsamplingX, geom.SubsamplingY); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(geom.X, geom.Y, mv, geom.SubsamplingX, geom.SubsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	if err := motion.PredictInterCompoundRefToConvBuf(buf, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, refX, refY, geom.Width, geom.Height, subX, subY, filters); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) frameWorkCompoundOffsets(refs tile.InterReferencesResult, blend tile.CompoundBlendResult) (int, int, error) {
	switch blend.Type {
	case tile.CompoundTypeAverage:
		return 8, 8, nil
	case tile.CompoundTypeDistWtd:
		if blend.CompoundIndex != 0 || !b.Sequence.EnableOrderHint || b.Sequence.OrderHintBits == 0 || b.Sequence.OrderHintBits > 31 {
			return 0, 0, ErrInvalidBatch
		}
		if !refs.Compound || !refs.Ref[0].Valid() || !refs.Ref[1].Valid() {
			return 0, 0, ErrInvalidBatch
		}
		ref0 := b.ReferenceOrderHints[refs.Ref[0]]
		ref1 := b.ReferenceOrderHints[refs.Ref[1]]
		return frameWorkDistanceWeightedCompoundOffsets(b.Sequence.OrderHintBits, b.FrameHeader.OrderHint, ref0, ref1)
	default:
		return 0, 0, ErrInvalidBatch
	}
}

func frameWorkDistanceWeightedCompoundOffsets(orderHintBits uint8, currentOrderHint uint32, ref0OrderHint uint32, ref1OrderHint uint32) (int, int, error) {
	d0, err := frameWorkRelativeOrderHint(orderHintBits, ref1OrderHint, currentOrderHint)
	if err != nil {
		return 0, 0, ErrInvalidBatch
	}
	d1, err := frameWorkRelativeOrderHint(orderHintBits, currentOrderHint, ref0OrderHint)
	if err != nil {
		return 0, 0, ErrInvalidBatch
	}
	if d0 < 0 {
		d0 = -d0
	}
	if d1 < 0 {
		d1 = -d1
	}
	if d0 > frameWorkMaxFrameDistance {
		d0 = frameWorkMaxFrameDistance
	}
	if d1 > frameWorkMaxFrameDistance {
		d1 = frameWorkMaxFrameDistance
	}
	order := 0
	if d0 <= d1 {
		order = 1
	}
	if d0 == 0 || d1 == 0 {
		return frameWorkQuantDistLookup[3][order], frameWorkQuantDistLookup[3][1-order], nil
	}
	i := 0
	for ; i < 3; i++ {
		c0 := frameWorkQuantDistWeight[i][order]
		c1 := frameWorkQuantDistWeight[i][1-order]
		d0c0 := d0 * c0
		d1c1 := d1 * c1
		if (d0 > d1 && d0c0 < d1c1) || (d0 <= d1 && d0c0 > d1c1) {
			break
		}
	}
	return frameWorkQuantDistLookup[i][order], frameWorkQuantDistLookup[i][1-order], nil
}

func frameWorkRelativeOrderHint(bits uint8, a uint32, b uint32) (int, error) {
	if bits == 0 || bits > 31 {
		return 0, ErrInvalidBatch
	}
	limit := uint32(1) << bits
	if a >= limit || b >= limit {
		return 0, ErrInvalidBatch
	}
	mask := int32(1 << (bits - 1))
	diff := int32(a) - int32(b)
	return int((diff & (mask - 1)) - (diff & mask)), nil
}

type frameWorkPredictionPlaneGeometry struct {
	Output frame.Plane
	Window FrameWorkPlaneRegion

	X      int
	Y      int
	Width  int
	Height int

	// CodedWidth / CodedHeight are the current frame's coded (cropped) plane
	// dimensions. They drive the reference same-size / scale-factor decision
	// (libaom compares the reference frame's dimensions to the current coded
	// frame dimensions, not to the MI-aligned write extent). Output.Width /
	// Output.Height may exceed these because the predictor's writable plane is
	// extended to the MI-aligned padding (frameWorkExtendPlaneToClip).
	CodedWidth  int
	CodedHeight int

	SubsamplingX bool
	SubsamplingY bool

	BytesPerSample int
}

func (b FrameWorkBatch) predictBlockInterReferencePlaneToOutput(index int, block tile.BlockVisit, plane FrameWorkPlane, refFrame tile.ReferenceFrame, mv motion.Vector, filters motion.InterpFilters) error {
	geom, ok, err := b.blockPredictionPlaneGeometry(index, block, plane)
	if err != nil || !ok {
		return err
	}
	reference, ok := frameWorkReferenceFromTile(refFrame)
	if !ok {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return err
	}
	if !sameSize {
		return frameWorkPredictScaledReferencePlane(geom, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth,
			geom.X, geom.Y, geom.X, geom.Y, geom.Width, geom.Height, mv, geom.SubsamplingX, geom.SubsamplingY, filters)
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(geom.X, geom.Y, mv, geom.SubsamplingX, geom.SubsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	// Select the interpolation-filter kernel from the un-clipped plane block
	// dimensions, not the frame-edge-clipped output extent. libaom's
	// av1_get_interp_filter_params_with_block_size() picks the narrow 4-tap
	// filter only when block_size_wide/high[plane_bsize] (the un-clipped plane
	// block side) is <= 4. A chroma block straddling the right/bottom frame
	// edge can have its visible extent shrink to <= 4 while its plane block
	// stays wider, and libaom keeps the 8-tap filter; clipping the filter
	// block size to the visible extent would wrongly switch to the 4-tap
	// filter and diverge by +-1 on the edge chroma samples. For luma and
	// interior chroma the un-clipped extent equals geom.Width/Height (no-op).
	filterW, filterH, err := frameWorkBlockPlanePredictionExtentPixels(block, b.Sequence.ColorConfig, plane)
	if err != nil {
		return ErrInvalidBatch
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSize(geom.Output, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, refX, refY, geom.Width, geom.Height, filterW, filterH, subX, subY, filters); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockInterReferencePlaneToScratch(dst frame.Plane, block tile.BlockVisit, plane FrameWorkPlane, refFrame tile.ReferenceFrame, mv motion.Vector, geom frameWorkPredictionPlaneGeometry, filters motion.InterpFilters) error {
	reference, ok := frameWorkReferenceFromTile(refFrame)
	if !ok {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return err
	}
	if !sameSize {
		// libaom: av1_make_inter_predictor() in av1/common/reconinter.c
		// looks up scale factors via xd->block_ref_scale_factors, which
		// are derived per-frame by av1_setup_scale_factors_for_frame()
		// from the output frame size — not from the staging buffer used
		// for inter-intra / masked compound. The scratch plane here is
		// block-sized, so we anchor the Q14 ratios to geom.Output.
		return frameWorkPredictScaledReferencePlaneToBuffer(dst, ref, geom, b.Sequence.ColorConfig.BitDepth,
			0, 0, geom.X, geom.Y, mv, filters)
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(geom.X, geom.Y, mv, geom.SubsamplingX, geom.SubsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	// Select the interpolation-filter kernel from the un-clipped plane block
	// dimensions, not the frame-edge-clipped visible extent — identical to the
	// predictBlockInterReferencePlaneToOutput path. libaom builds the inter part
	// of an inter-intra / masked-compound block over the full plane block
	// (pd->width/height) and av1_get_interp_filter_params_with_block_size()
	// keeps the 8-tap filter as long as the un-clipped plane side is > 4. A
	// chroma block straddling the bottom/right frame edge whose visible extent
	// shrinks to <= 4 must still use the wide filter, or the staged inter
	// predictor diverges by +-1 on the edge chroma samples (the surviving
	// Class-B 10-bit reconstruction gap).
	filterW, filterH, err := frameWorkBlockPlanePredictionExtentPixels(block, b.Sequence.ColorConfig, plane)
	if err != nil {
		return ErrInvalidBatch
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSize(dst, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, 0, 0, refX, refY, geom.Width, geom.Height, filterW, filterH, subX, subY, filters); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

// predictBlockInterGlobalWarpToScratch stages a global-warp inter prediction in
// a block-sized scratch plane (for inter-intra / masked blending). It mirrors
// predictBlockInterGlobalWarpPlaneWithFilters but samples into dst at (0,0);
// scaled references fall back to the translational predictor as libaom does
// (av1_is_scaled makes allow_warp() return 0).
func (b FrameWorkBatch) predictBlockInterGlobalWarpToScratch(dst frame.Plane, plane FrameWorkPlane, refFrame tile.ReferenceFrame, model tile.WarpedMotionModel, mv motion.Vector, geom frameWorkPredictionPlaneGeometry, filters motion.InterpFilters) error {
	reference, ok := frameWorkReferenceFromTile(refFrame)
	if !ok {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return err
	}
	if !sameSize {
		return frameWorkPredictScaledReferencePlaneToBuffer(dst, ref, geom, b.Sequence.ColorConfig.BitDepth,
			0, 0, geom.X, geom.Y, mv, filters)
	}
	return motion.PredictWarpedPlaneBlockToScratchBitDepth(dst, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth,
		geom.X, geom.Y, geom.Width, geom.Height, model.Params.Matrix, model.Alpha, model.Beta, model.Gamma, model.Delta, geom.SubsamplingX, geom.SubsamplingY)
}

func (b FrameWorkBatch) predictAndBlendOBMCAbove(plane FrameWorkPlane, geom frameWorkPredictionPlaneGeometry, tmp frame.Plane, block tile.BlockVisit, neighbor tile.OverlappableNeighbor) error {
	if !neighbor.InterpFiltersValid {
		return ErrInvalidBatch
	}
	relX, ok := frameWorkOBMCPlaneOffset(neighbor.RelX4, geom.SubsamplingX)
	if !ok {
		return ErrInvalidBatch
	}
	width, ok := frameWorkOBMCPlaneSpan(neighbor.Span4, geom.SubsamplingX)
	if !ok {
		return ErrInvalidBatch
	}
	height, err := frameWorkOBMCAboveHeight(block.Size, geom)
	if err != nil {
		return err
	}
	// Select the OBMC mask from the full overlap height (libaom
	// av1_get_obmc_mask), then clip the blend write to the plane extent.
	mask, ok := frameWorkOBMCMask(height)
	if !ok {
		return ErrInvalidBatch
	}
	// libaom's dec_build_prediction_by_above_pred predicts the neighbor over
	// bw = (op_mi_size*MI_SIZE)>>ss_x where op_mi_size = AOMMIN(xd->width,
	// mi_size_wide[neighbor]) is keyed off the UN-clipped current block MI width,
	// and bh = clamp(block_high>>(ss_y+1), 4, ...). The interp-filter kernel is
	// selected from those un-clipped dimensions. neighbor.Span4 is computed
	// against the frame-clipped visible block width, so for a block straddling
	// the right frame edge it shrinks below libaom's op_mi_size; recompute the
	// un-clipped overlap width here for kernel selection while keeping the
	// frame-clipped span for the blend write.
	filterH := height
	filterW, err := frameWorkOBMCAboveFilterWidth(block, neighbor, geom, b.Sequence.ColorConfig, plane, relX, width)
	if err != nil {
		return err
	}
	if height > geom.Height {
		height = geom.Height
	}
	if width > geom.Width-relX {
		width = geom.Width - relX
	}
	if width <= 0 || height <= 0 {
		return ErrInvalidBatch
	}
	if err := b.predictOBMCNeighborToScratch(tmp, plane, neighbor, geom, relX, 0, geom.X+relX, geom.Y, width, height, filterW, filterH); err != nil {
		return err
	}
	return frameWorkBlendOBMCV(geom.Output, tmp, geom.BytesPerSample, geom.X+relX, geom.Y, relX, 0, width, height, mask)
}

func (b FrameWorkBatch) predictAndBlendOBMCLeft(plane FrameWorkPlane, geom frameWorkPredictionPlaneGeometry, tmp frame.Plane, block tile.BlockVisit, neighbor tile.OverlappableNeighbor) error {
	if !neighbor.InterpFiltersValid {
		return ErrInvalidBatch
	}
	relY, ok := frameWorkOBMCPlaneOffset(neighbor.RelY4, geom.SubsamplingY)
	if !ok {
		return ErrInvalidBatch
	}
	height, ok := frameWorkOBMCPlaneSpan(neighbor.Span4, geom.SubsamplingY)
	if !ok {
		return ErrInvalidBatch
	}
	width, err := frameWorkOBMCLeftWidth(block.Size, geom)
	if err != nil {
		return err
	}
	// Select the OBMC mask from the full overlap width, then clip the blend
	// write to the plane extent.
	mask, ok := frameWorkOBMCMask(width)
	if !ok {
		return ErrInvalidBatch
	}
	// libaom's dec_build_prediction_by_left_pred predicts the neighbor over
	// bw = clamp(block_wide>>(ss_x+1), 4, ...), bh = (op_mi_size*MI_SIZE)>>ss_y
	// where op_mi_size = AOMMIN(xd->height, mi_size_high[neighbor]) is keyed off
	// the UN-clipped current block MI height (xd->height), not the frame-edge
	// visible height. The interp-filter kernel is selected from those un-clipped
	// dimensions (av1_get_interp_filter_params_with_block_size). neighbor.Span4
	// is computed in the tile decoder against the frame-clipped visible block
	// height, so for a block straddling the bottom frame edge it shrinks below
	// libaom's op_mi_size and goav1 would wrongly switch to the 4-tap filter.
	// Recompute the un-clipped overlap height here for kernel selection from the
	// un-clipped plane extent and the neighbor's own size, mirroring libaom,
	// while keeping the frame-clipped span for the actual blend write.
	filterW := width
	filterH, err := frameWorkOBMCLeftFilterHeight(block, neighbor, geom, b.Sequence.ColorConfig, plane, relY, height)
	if err != nil {
		return err
	}
	if width > geom.Width {
		width = geom.Width
	}
	if height > geom.Height-relY {
		height = geom.Height - relY
	}
	if width <= 0 || height <= 0 {
		return ErrInvalidBatch
	}
	if err := b.predictOBMCNeighborToScratch(tmp, plane, neighbor, geom, 0, relY, geom.X, geom.Y+relY, width, height, filterW, filterH); err != nil {
		return err
	}
	return frameWorkBlendOBMCH(geom.Output, tmp, geom.BytesPerSample, geom.X, geom.Y+relY, 0, relY, width, height, mask)
}

func (b FrameWorkBatch) predictOBMCNeighborToScratch(dst frame.Plane, plane FrameWorkPlane, neighbor tile.OverlappableNeighbor, geom frameWorkPredictionPlaneGeometry, dstX int, dstY int, absX int, absY int, width int, height int, filterW int, filterH int) error {
	motionResult := neighbor.Motion
	if !motionResult.References.Ref[0].Valid() {
		return ErrInvalidBatch
	}
	return b.predictInterReferenceAreaToScratch(dst, plane, motionResult.References.Ref[0], motionResult.MV[0], geom, dstX, dstY, absX, absY, width, height, filterW, filterH, neighbor.InterpFilters)
}

func (b FrameWorkBatch) predictInterReferenceAreaToScratch(dst frame.Plane, plane FrameWorkPlane, refFrame tile.ReferenceFrame, mv motion.Vector, geom frameWorkPredictionPlaneGeometry, dstX int, dstY int, absX int, absY int, width int, height int, filterW int, filterH int, filters motion.InterpFilters) error {
	if !frameWorkPlaneBlockAddressable(dst, geom.BytesPerSample, dstX, dstY, width, height) {
		return ErrInvalidBatch
	}
	reference, ok := frameWorkReferenceFromTile(refFrame)
	if !ok {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, plane)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	// OBMC / sub-block area predictions may straddle a scaled reference. The
	// regular same-size convolver below assumes ref.Width / ref.Height equal
	// geom.Output.Width / geom.Output.Height; for SVC L2T1 spatial=1 the
	// enhancement layer references the half-size spatial=0 base, so the
	// area must run through the scaled 8-tap convolver instead. libaom
	// mirror: av1_make_inter_predictor() routes through
	// av1_convolve_2d_scale_c whenever av1_is_scaled(sf).
	sameSize, err := frameWorkSameOrScaledReferencePlane(geom, ref)
	if err != nil {
		return err
	}
	if !sameSize {
		curWidth, curHeight := frameWorkScaledReferenceCurrentDims(geom)
		return frameWorkPredictScaledReferencePlaneWithDims(dst, ref, curWidth, curHeight,
			geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, dstX, dstY, absX, absY, width, height, mv,
			geom.SubsamplingX, geom.SubsamplingY, filters)
	}
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(absX, absY, mv, geom.SubsamplingX, geom.SubsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	// Select the interpolation-filter kernel from the un-clipped neighbor
	// prediction dimensions (filterW/filterH), not the frame-edge-clipped write
	// extent (width/height). libaom's OBMC neighbor predictor keys the
	// 4-tap-vs-8-tap choice off the un-clipped bw/bh; a neighbor straddling the
	// bottom/right frame edge whose visible extent shrinks to <= 4 must still use
	// the wide filter or the blended OBMC samples diverge by +-1.
	if filterW <= 0 {
		filterW = width
	}
	if filterH <= 0 {
		filterH = height
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepthFilterSize(dst, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, dstX, dstY, refX, refY, width, height, filterW, filterH, subX, subY, filters); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) blockPredictionPlaneGeometry(index int, block tile.BlockVisit, plane FrameWorkPlane) (frameWorkPredictionPlaneGeometry, bool, error) {
	x, y, width, height, subsamplingX, subsamplingY, ok, err := frameWorkBlockPlanePosition(block, b.Sequence.ColorConfig, plane)
	if err != nil || !ok {
		return frameWorkPredictionPlaneGeometry{}, ok, err
	}
	window, err := b.JobOutputPlane(index, plane)
	if err != nil {
		return frameWorkPredictionPlaneGeometry{}, false, err
	}
	if b.Output == nil {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	// A block can land entirely beyond the visible output plane when the
	// bitstream's MI grid was rounded up past the coded frame dimensions
	// (libaom clips prediction writes to the visible area rather than
	// rejecting the block). Treat that case as a silently-skipped prediction;
	// genuinely malformed callers still hit the !ok path below.
	if frameWorkPlaneBlockStartsBeyondOutput(b.Output, plane, x, y) {
		return frameWorkPredictionPlaneGeometry{}, false, nil
	}
	width, height, ok = frameWorkClipVisiblePixelsToWindow(window, x, y, width, height)
	if !ok {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	output, outputSubX, outputSubY, ok := frameWorkFramePlane(b.Output, plane)
	if !ok || b.Output.Layout.BytesPerSample <= 0 {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	if outputSubX != subsamplingX || outputSubY != subsamplingY {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	// Extend the predictor's plane bound to the MI-aligned writable extent so
	// prediction writes past the visible edge land in the underlying buffer's
	// past-visible stride padding instead of failing the planeBlockWindow
	// bounds check (libaom writes whole transform blocks regardless of where
	// the visible boundary lands; later blocks read those samples as
	// predictor neighbors).
	codedWidth := output.Width
	codedHeight := output.Height
	output = frameWorkExtendPlaneToClip(output, window, b.Output.Layout.BytesPerSample)
	return frameWorkPredictionPlaneGeometry{
		Output:         output,
		Window:         window,
		X:              x,
		Y:              y,
		Width:          width,
		Height:         height,
		CodedWidth:     codedWidth,
		CodedHeight:    codedHeight,
		SubsamplingX:   subsamplingX,
		SubsamplingY:   subsamplingY,
		BytesPerSample: b.Output.Layout.BytesPerSample,
	}, true, nil
}

// frameWorkExtendPlaneToClip returns a frame.Plane view whose Width/Height
// span the window's MI-aligned writable extent. The plane.Pix slice is widened
// to (clipHeight-1)*Stride + clipWidth*BytesPerSample so prediction writes
// past the visible edge stay within the underlying frame buffer. When the
// window has no recorded clip extent the plane is returned unchanged.
func frameWorkExtendPlaneToClip(plane frame.Plane, window FrameWorkPlaneRegion, bytesPerSample int) frame.Plane {
	clipWidth := window.ClipWidth
	if clipWidth <= 0 {
		clipWidth = window.Width
	}
	clipHeight := window.ClipHeight
	if clipHeight <= 0 {
		clipHeight = window.Height
	}
	// Translate window-relative clip extent (which starts at window.X /
	// window.Y) into plane-relative bounds.
	planeWidth := window.X + clipWidth
	planeHeight := window.Y + clipHeight
	if planeWidth <= plane.Width && planeHeight <= plane.Height {
		return plane
	}
	if planeWidth > plane.Stride/bytesPerSample {
		planeWidth = plane.Stride / bytesPerSample
	}
	if planeWidth < plane.Width {
		planeWidth = plane.Width
	}
	if planeHeight < plane.Height {
		planeHeight = plane.Height
	}
	// Recompute Pix length to span (planeHeight-1)*Stride + planeWidth*BPS.
	// The original Pix already spans the full plane buffer; we just need to
	// ensure callers can read/write up to the new bounds. Don't extend Pix
	// beyond the existing slice length; that would be a logic bug.
	newRowBytes := planeWidth * bytesPerSample
	newLen := min((planeHeight-1)*plane.Stride+newRowBytes, len(plane.Pix))
	return frame.Plane{
		Pix:    plane.Pix[:newLen],
		Stride: plane.Stride,
		Width:  planeWidth,
		Height: planeHeight,
	}
}

func frameWorkPredictionPlaneEdgeBlock(block tile.BlockVisit, geom frameWorkPredictionPlaneGeometry) tile.BlockVisit {
	return frameWorkPredictionEdgeBlockForWindow(block, geom.X, geom.Y, geom.Window)
}

func frameWorkPredictionEdgeBlockForWindow(block tile.BlockVisit, x int, y int, window FrameWorkPlaneRegion) tile.BlockVisit {
	if x <= window.X {
		block.HaveLeft = false
	}
	if y <= window.Y {
		block.HaveTop = false
	}
	return block
}

func frameWorkPredictionTransformEdgeBlock(block tile.BlockVisit, baseX4 int, baseY4 int, txX4 int, txY4 int) tile.BlockVisit {
	if txX4 > baseX4 {
		block.HaveLeft = true
	}
	if txY4 > baseY4 {
		block.HaveTop = true
	}
	return block
}

func frameWorkLumaDirectionalExtendedEdges(block tile.BlockVisit, sbSizeMIB uint8, miColEnd uint32, miRowEnd uint32, absX int, absY int, width int, height int) (allowTopRight bool, allowBottomLeft bool) {
	blockX := int(block.MICol) * 4
	blockY := int(block.MIRow) * 4
	return frameWorkDirectionalExtendedEdges(block, block.Size, sbSizeMIB, miColEnd, miRowEnd, blockX, blockY, absX, absY, width, height, 0, 0)
}

func frameWorkChromaDirectionalExtendedEdges(block tile.BlockVisit, sbSizeMIB uint8, miColEnd uint32, miRowEnd uint32, originX int, originY int, absX int, absY int, width int, height int, subsamplingX bool, subsamplingY bool) (allowTopRight bool, allowBottomLeft bool) {
	ssX := int(frameWorkSubsampleShift(subsamplingX))
	ssY := int(frameWorkSubsampleShift(subsamplingY))
	size := frameWorkChromaAvailabilityBlockSize(block.Size, subsamplingX, subsamplingY)
	return frameWorkDirectionalExtendedEdges(block, size, sbSizeMIB, miColEnd, miRowEnd, originX, originY, absX, absY, width, height, ssX, ssY)
}

func frameWorkChromaAvailabilityBlockSize(size tile.BlockSize, subsamplingX bool, subsamplingY bool) tile.BlockSize {
	switch size {
	case tile.BlockSize4x4:
		if subsamplingX && subsamplingY {
			return tile.BlockSize8x8
		}
		if subsamplingX {
			return tile.BlockSize8x4
		}
		if subsamplingY {
			return tile.BlockSize4x8
		}
	case tile.BlockSize4x8:
		if subsamplingX {
			return tile.BlockSize8x8
		}
	case tile.BlockSize8x4:
		if subsamplingY {
			return tile.BlockSize8x8
		}
	case tile.BlockSize4x16:
		if subsamplingX {
			return tile.BlockSize8x16
		}
	case tile.BlockSize16x4:
		if subsamplingY {
			return tile.BlockSize16x8
		}
	}
	return size
}

func frameWorkDirectionalExtendedEdges(block tile.BlockVisit, size tile.BlockSize, sbSizeMIB uint8, miColEnd uint32, miRowEnd uint32, originX int, originY int, absX int, absY int, width int, height int, ssX int, ssY int) (allowTopRight bool, allowBottomLeft bool) {
	colOff := absX - originX
	rowOff := absY - originY
	allowTopRight = frameWorkHasTopRight(block, size, sbSizeMIB, miColEnd, colOff, rowOff, width, ssX, ssY)
	allowBottomLeft = frameWorkHasBottomLeft(block, size, sbSizeMIB, miRowEnd, colOff, rowOff, height, ssX, ssY)
	return allowTopRight, allowBottomLeft
}

func frameWorkHasTopRight(block tile.BlockVisit, size tile.BlockSize, sbSizeMIB uint8, miColEnd uint32, colOffPx int, rowOffPx int, width int, ssX int, ssY int) bool {
	if !block.HaveTop || sbSizeMIB == 0 || colOffPx < 0 || rowOffPx < 0 || width <= 0 ||
		colOffPx%4 != 0 || rowOffPx%4 != 0 || width%4 != 0 {
		return false
	}
	dims, ok := size.Dimensions()
	if !ok {
		return false
	}
	colOff := colOffPx >> 2
	rowOff := rowOffPx >> 2
	txW := width >> 2
	if ssX < 0 || ssY < 0 || ssX > 1 || ssY > 1 ||
		block.MICol+uint32((colOff+txW)<<ssX) >= miColEnd {
		return false
	}
	blockW := max(int(dims.W4)>>ssX, 1)
	if rowOff > 0 {
		if int(dims.W4) > 16 {
			block64 := max(16>>ssX, 1)
			if rowOff == 16>>ssY && colOff+txW == block64 {
				return true
			}
			colOff64 := colOff % block64
			return colOff64+txW < block64
		}
		return colOff+txW < blockW
	}
	if colOff+txW < blockW {
		return true
	}
	bwLog2 := int(dims.Log2W)
	bhLog2 := int(dims.Log2H)
	sb := int(sbSizeMIB)
	if sb <= 0 {
		return false
	}
	blkRowInSB := int(block.MIRow&uint32(sb-1)) >> bhLog2
	blkColInSB := int(block.MICol&uint32(sb-1)) >> bwLog2
	if blkRowInSB == 0 {
		return true
	}
	if ((blkColInSB + 1) << bwLog2) >= sb {
		return false
	}
	table := frameWorkTopRightAvailabilityTable(block.Partition, size)
	if len(table) == 0 {
		return false
	}
	thisBlockIndex := blkRowInSB<<(frameWorkMaxMIBSizeLog2-bwLog2) + blkColInSB
	idx1 := thisBlockIndex >> 3
	idx2 := thisBlockIndex & 7
	return idx1 >= 0 && idx1 < len(table) && ((table[idx1]>>idx2)&1) != 0
}

func frameWorkHasBottomLeft(block tile.BlockVisit, size tile.BlockSize, sbSizeMIB uint8, miRowEnd uint32, colOffPx int, rowOffPx int, height int, ssX int, ssY int) bool {
	if !block.HaveLeft || sbSizeMIB == 0 || colOffPx < 0 || rowOffPx < 0 || height <= 0 ||
		colOffPx%4 != 0 || rowOffPx%4 != 0 || height%4 != 0 {
		return false
	}
	dims, ok := size.Dimensions()
	if !ok {
		return false
	}
	colOff := colOffPx >> 2
	rowOff := rowOffPx >> 2
	txH := height >> 2
	if ssX < 0 || ssY < 0 || ssX > 1 || ssY > 1 ||
		block.MIRow+uint32((rowOff+txH)<<ssY) >= miRowEnd {
		return false
	}
	if int(dims.W4) > 16 && colOff > 0 {
		blockW64 := max(16>>ssX, 1)
		colOff64 := colOff % blockW64
		if colOff64 == 0 {
			planeBlockH64 := max(16>>ssY, 1)
			rowOff64 := rowOff % planeBlockH64
			blockH := max(int(dims.H4)>>ssY, 1)
			if blockH > planeBlockH64 {
				blockH = planeBlockH64
			}
			return rowOff64+txH < blockH
		}
	}
	if colOff > 0 {
		return false
	}
	blockH := max(int(dims.H4)>>ssY, 1)
	if rowOff+txH < blockH {
		return true
	}
	bwLog2 := int(dims.Log2W)
	bhLog2 := int(dims.Log2H)
	sb := int(sbSizeMIB)
	if sb <= 0 {
		return false
	}
	blkRowInSB := int(block.MIRow&uint32(sb-1)) >> bhLog2
	blkColInSB := int(block.MICol&uint32(sb-1)) >> bwLog2
	if blkColInSB == 0 {
		rowOffInSB := (blkRowInSB << bhLog2 >> ssY) + rowOff
		return rowOffInSB+txH < sb>>ssY
	}
	if ((blkRowInSB + 1) << bhLog2) >= sb {
		return false
	}
	table := frameWorkBottomLeftAvailabilityTable(block.Partition, size)
	if len(table) == 0 {
		return false
	}
	thisBlockIndex := blkRowInSB<<(frameWorkMaxMIBSizeLog2-bwLog2) + blkColInSB
	idx1 := thisBlockIndex >> 3
	idx2 := thisBlockIndex & 7
	return idx1 >= 0 && idx1 < len(table) && ((table[idx1]>>idx2)&1) != 0
}

func frameWorkTopRightAvailabilityTable(partition tile.Partition, size tile.BlockSize) []uint8 {
	if frameWorkPartitionUsesVerticalOrder(partition) {
		switch size {
		case tile.BlockSize4x8:
			return frameWorkHasTopRight4x8[:]
		case tile.BlockSize8x8:
			return frameWorkHasTopRightVert8x8[:]
		case tile.BlockSize8x16:
			return frameWorkHasTopRight8x16[:]
		case tile.BlockSize16x16:
			return frameWorkHasTopRightVert16x16[:]
		case tile.BlockSize16x32:
			return frameWorkHasTopRight16x32[:]
		case tile.BlockSize32x32:
			return frameWorkHasTopRightVert32x32[:]
		case tile.BlockSize32x64:
			return frameWorkHasTopRight32x64[:]
		case tile.BlockSize64x64:
			return frameWorkHasTopRightVert64x64[:]
		case tile.BlockSize64x128:
			return frameWorkHasTopRight64x128[:]
		case tile.BlockSize128x128:
			return frameWorkHasTopRight128x128[:]
		default:
			return nil
		}
	}
	switch size {
	case tile.BlockSize4x4:
		return frameWorkHasTopRight4x4[:]
	case tile.BlockSize4x8:
		return frameWorkHasTopRight4x8[:]
	case tile.BlockSize8x4:
		return frameWorkHasTopRight8x4[:]
	case tile.BlockSize8x8:
		return frameWorkHasTopRight8x8[:]
	case tile.BlockSize8x16:
		return frameWorkHasTopRight8x16[:]
	case tile.BlockSize16x8:
		return frameWorkHasTopRight16x8[:]
	case tile.BlockSize16x16:
		return frameWorkHasTopRight16x16[:]
	case tile.BlockSize16x32:
		return frameWorkHasTopRight16x32[:]
	case tile.BlockSize32x16:
		return frameWorkHasTopRight32x16[:]
	case tile.BlockSize32x32:
		return frameWorkHasTopRight32x32[:]
	case tile.BlockSize32x64:
		return frameWorkHasTopRight32x64[:]
	case tile.BlockSize64x32:
		return frameWorkHasTopRight64x32[:]
	case tile.BlockSize64x64:
		return frameWorkHasTopRight64x64[:]
	case tile.BlockSize64x128:
		return frameWorkHasTopRight64x128[:]
	case tile.BlockSize128x64:
		return frameWorkHasTopRight128x64[:]
	case tile.BlockSize128x128:
		return frameWorkHasTopRight128x128[:]
	case tile.BlockSize4x16:
		return frameWorkHasTopRight4x16[:]
	case tile.BlockSize16x4:
		return frameWorkHasTopRight16x4[:]
	case tile.BlockSize8x32:
		return frameWorkHasTopRight8x32[:]
	case tile.BlockSize32x8:
		return frameWorkHasTopRight32x8[:]
	case tile.BlockSize16x64:
		return frameWorkHasTopRight16x64[:]
	case tile.BlockSize64x16:
		return frameWorkHasTopRight64x16[:]
	default:
		return nil
	}
}

func frameWorkBottomLeftAvailabilityTable(partition tile.Partition, size tile.BlockSize) []uint8 {
	if frameWorkPartitionUsesVerticalOrder(partition) {
		switch size {
		case tile.BlockSize4x8:
			return frameWorkHasBottomLeft4x8[:]
		case tile.BlockSize8x8:
			return frameWorkHasBottomLeftVert8x8[:]
		case tile.BlockSize8x16:
			return frameWorkHasBottomLeft8x16[:]
		case tile.BlockSize16x16:
			return frameWorkHasBottomLeftVert16x16[:]
		case tile.BlockSize16x32:
			return frameWorkHasBottomLeft16x32[:]
		case tile.BlockSize32x32:
			return frameWorkHasBottomLeftVert32x32[:]
		case tile.BlockSize32x64:
			return frameWorkHasBottomLeft32x64[:]
		case tile.BlockSize64x64:
			return frameWorkHasBottomLeftVert64x64[:]
		case tile.BlockSize64x128:
			return frameWorkHasBottomLeft64x128[:]
		case tile.BlockSize128x128:
			return frameWorkHasBottomLeft128x128[:]
		default:
			return nil
		}
	}
	switch size {
	case tile.BlockSize4x4:
		return frameWorkHasBottomLeft4x4[:]
	case tile.BlockSize4x8:
		return frameWorkHasBottomLeft4x8[:]
	case tile.BlockSize8x4:
		return frameWorkHasBottomLeft8x4[:]
	case tile.BlockSize8x8:
		return frameWorkHasBottomLeft8x8[:]
	case tile.BlockSize8x16:
		return frameWorkHasBottomLeft8x16[:]
	case tile.BlockSize16x8:
		return frameWorkHasBottomLeft16x8[:]
	case tile.BlockSize16x16:
		return frameWorkHasBottomLeft16x16[:]
	case tile.BlockSize16x32:
		return frameWorkHasBottomLeft16x32[:]
	case tile.BlockSize32x16:
		return frameWorkHasBottomLeft32x16[:]
	case tile.BlockSize32x32:
		return frameWorkHasBottomLeft32x32[:]
	case tile.BlockSize32x64:
		return frameWorkHasBottomLeft32x64[:]
	case tile.BlockSize64x32:
		return frameWorkHasBottomLeft64x32[:]
	case tile.BlockSize64x64:
		return frameWorkHasBottomLeft64x64[:]
	case tile.BlockSize64x128:
		return frameWorkHasBottomLeft64x128[:]
	case tile.BlockSize128x64:
		return frameWorkHasBottomLeft128x64[:]
	case tile.BlockSize128x128:
		return frameWorkHasBottomLeft128x128[:]
	case tile.BlockSize4x16:
		return frameWorkHasBottomLeft4x16[:]
	case tile.BlockSize16x4:
		return frameWorkHasBottomLeft16x4[:]
	case tile.BlockSize8x32:
		return frameWorkHasBottomLeft8x32[:]
	case tile.BlockSize32x8:
		return frameWorkHasBottomLeft32x8[:]
	case tile.BlockSize16x64:
		return frameWorkHasBottomLeft16x64[:]
	case tile.BlockSize64x16:
		return frameWorkHasBottomLeft64x16[:]
	default:
		return nil
	}
}

func frameWorkPartitionUsesVerticalOrder(partition tile.Partition) bool {
	return partition == tile.PartitionTLeftSplit || partition == tile.PartitionTRightSplit
}

var (
	frameWorkHasTopRight4x4 = [...]uint8{
		255, 255, 255, 255, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
		127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
		255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
		127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
		255, 255, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
		127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
		255, 127, 255, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
		127, 127, 127, 127, 85, 85, 85, 85, 119, 119, 119, 119, 85, 85, 85, 85,
	}
	frameWorkHasTopRight4x8 = [...]uint8{
		255, 255, 255, 255, 119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119,
		255, 127, 255, 127, 119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119,
		255, 255, 255, 127, 119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119,
		255, 127, 255, 127, 119, 119, 119, 119, 127, 127, 127, 127, 119, 119, 119, 119,
	}
	frameWorkHasTopRight8x4 = [...]uint8{
		255, 255, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
		127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
		255, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
		127, 127, 0, 0, 85, 85, 0, 0, 119, 119, 0, 0, 85, 85, 0, 0,
	}
	frameWorkHasTopRight8x8 = [...]uint8{
		255, 255, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
		255, 127, 85, 85, 119, 119, 85, 85, 127, 127, 85, 85, 119, 119, 85, 85,
	}
	frameWorkHasTopRight8x16 = [...]uint8{
		255, 255, 119, 119, 127, 127, 119, 119, 255, 127, 119, 119, 127, 127, 119, 119,
	}
	frameWorkHasTopRight16x8 = [...]uint8{
		255, 0, 85, 0, 119, 0, 85, 0, 127, 0, 85, 0, 119, 0, 85, 0,
	}
	frameWorkHasTopRight16x16 = [...]uint8{
		255, 85, 119, 85, 127, 85, 119, 85,
	}
	frameWorkHasTopRight16x32 = [...]uint8{
		255, 119, 127, 119,
	}
	frameWorkHasTopRight32x16 = [...]uint8{
		15, 5, 7, 5,
	}
	frameWorkHasTopRight32x32 = [...]uint8{
		95, 87,
	}
	frameWorkHasTopRight32x64 = [...]uint8{
		127,
	}
	frameWorkHasTopRight64x32 = [...]uint8{
		19,
	}
	frameWorkHasTopRight64x64 = [...]uint8{
		7,
	}
	frameWorkHasTopRight64x128 = [...]uint8{
		3,
	}
	frameWorkHasTopRight128x64 = [...]uint8{
		1,
	}
	frameWorkHasTopRight128x128 = [...]uint8{
		1,
	}
	frameWorkHasTopRight4x16 = [...]uint8{
		255, 255, 255, 255, 127, 127, 127, 127, 255, 127, 255, 127, 127, 127, 127, 127,
		255, 255, 255, 127, 127, 127, 127, 127, 255, 127, 255, 127, 127, 127, 127, 127,
	}
	frameWorkHasTopRight16x4 = [...]uint8{
		255, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
		127, 0, 0, 0, 85, 0, 0, 0, 119, 0, 0, 0, 85, 0, 0, 0,
	}
	frameWorkHasTopRight8x32 = [...]uint8{
		255, 255, 127, 127, 255, 127, 127, 127,
	}
	frameWorkHasTopRight32x8 = [...]uint8{
		15, 0, 5, 0, 7, 0, 5, 0,
	}
	frameWorkHasTopRight16x64 = [...]uint8{
		255, 127,
	}
	frameWorkHasTopRight64x16 = [...]uint8{
		3, 1,
	}
	frameWorkHasTopRightVert8x8 = [...]uint8{
		255, 255, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
		255, 127, 0, 0, 119, 119, 0, 0, 127, 127, 0, 0, 119, 119, 0, 0,
	}
	frameWorkHasTopRightVert16x16 = [...]uint8{
		255, 0, 119, 0, 127, 0, 119, 0,
	}
	frameWorkHasTopRightVert32x32 = [...]uint8{
		15, 7,
	}
	frameWorkHasTopRightVert64x64 = [...]uint8{
		3,
	}

	frameWorkHasBottomLeft4x4 = [...]uint8{
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1,
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 1, 0,
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1,
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 0, 0,
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1,
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 1, 0,
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 1, 1, 1,
		84, 85, 85, 85, 16, 17, 17, 17, 84, 85, 85, 85, 0, 0, 0, 0,
	}
	frameWorkHasBottomLeft4x8 = [...]uint8{
		16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0,
		16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 0, 0,
		16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 1, 0,
		16, 17, 17, 17, 0, 1, 1, 1, 16, 17, 17, 17, 0, 0, 0, 0,
	}
	frameWorkHasBottomLeft8x4 = [...]uint8{
		254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 1,
		254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0,
		254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 1,
		254, 255, 84, 85, 254, 255, 16, 17, 254, 255, 84, 85, 254, 255, 0, 0,
	}
	frameWorkHasBottomLeft8x8 = [...]uint8{
		84, 85, 16, 17, 84, 85, 0, 1, 84, 85, 16, 17, 84, 85, 0, 0,
		84, 85, 16, 17, 84, 85, 0, 1, 84, 85, 16, 17, 84, 85, 0, 0,
	}
	frameWorkHasBottomLeft8x16 = [...]uint8{
		16, 17, 0, 1, 16, 17, 0, 0, 16, 17, 0, 1, 16, 17, 0, 0,
	}
	frameWorkHasBottomLeft16x8 = [...]uint8{
		254, 84, 254, 16, 254, 84, 254, 0, 254, 84, 254, 16, 254, 84, 254, 0,
	}
	frameWorkHasBottomLeft16x16 = [...]uint8{
		84, 16, 84, 0, 84, 16, 84, 0,
	}
	frameWorkHasBottomLeft16x32 = [...]uint8{
		16, 0, 16, 0,
	}
	frameWorkHasBottomLeft32x16 = [...]uint8{
		78, 14, 78, 14,
	}
	frameWorkHasBottomLeft32x32 = [...]uint8{
		4, 4,
	}
	frameWorkHasBottomLeft32x64 = [...]uint8{
		0,
	}
	frameWorkHasBottomLeft64x32 = [...]uint8{
		34,
	}
	frameWorkHasBottomLeft64x64 = [...]uint8{
		0,
	}
	frameWorkHasBottomLeft64x128 = [...]uint8{
		0,
	}
	frameWorkHasBottomLeft128x64 = [...]uint8{
		0,
	}
	frameWorkHasBottomLeft128x128 = [...]uint8{
		0,
	}
	frameWorkHasBottomLeft4x16 = [...]uint8{
		0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
		0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
	}
	frameWorkHasBottomLeft16x4 = [...]uint8{
		254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
		254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
	}
	frameWorkHasBottomLeft8x32 = [...]uint8{
		0, 1, 0, 0, 0, 1, 0, 0,
	}
	frameWorkHasBottomLeft32x8 = [...]uint8{
		238, 78, 238, 14, 238, 78, 238, 14,
	}
	frameWorkHasBottomLeft16x64 = [...]uint8{
		0, 0,
	}
	frameWorkHasBottomLeft64x16 = [...]uint8{
		42, 42,
	}
	frameWorkHasBottomLeftVert8x8 = [...]uint8{
		254, 255, 16, 17, 254, 255, 0, 1, 254, 255, 16, 17, 254, 255, 0, 0,
		254, 255, 16, 17, 254, 255, 0, 1, 254, 255, 16, 17, 254, 255, 0, 0,
	}
	frameWorkHasBottomLeftVert16x16 = [...]uint8{
		254, 16, 254, 0, 254, 16, 254, 0,
	}
	frameWorkHasBottomLeftVert32x32 = [...]uint8{
		14, 14,
	}
	frameWorkHasBottomLeftVert64x64 = [...]uint8{
		2,
	}
)

func frameWorkTransformVisibleAndExtentPixels(tx tile.TransformBlock) (width int, height int, predWidth int, predHeight int, err error) {
	dims, ok := tx.Size.Dimensions()
	if !ok || tx.VisibleW4 == 0 || tx.VisibleH4 == 0 ||
		tx.VisibleW4 > dims.W4 || tx.VisibleH4 > dims.H4 {
		return 0, 0, 0, 0, ErrInvalidBatch
	}
	width, ok = frameWorkInt64Mul4(int64(tx.VisibleW4))
	if !ok {
		return 0, 0, 0, 0, ErrInvalidBatch
	}
	height, ok = frameWorkInt64Mul4(int64(tx.VisibleH4))
	if !ok {
		return 0, 0, 0, 0, ErrInvalidBatch
	}
	predWidth, ok = frameWorkInt64Mul4(int64(dims.W4))
	if !ok {
		return 0, 0, 0, 0, ErrInvalidBatch
	}
	predHeight, ok = frameWorkInt64Mul4(int64(dims.H4))
	if !ok {
		return 0, 0, 0, 0, ErrInvalidBatch
	}
	return width, height, predWidth, predHeight, nil
}

func frameWorkInterScratchPlane(buf []byte, bytesPerSample int, width int, height int) (frame.Plane, error) {
	if width <= 0 || height <= 0 || width > 128 || height > 128 || (bytesPerSample != 1 && bytesPerSample != 2) {
		return frame.Plane{}, ErrInvalidBatch
	}
	stride, ok := frameWorkCheckedMul(width, bytesPerSample)
	if !ok {
		return frame.Plane{}, ErrInvalidBatch
	}
	size, ok := frameWorkCheckedMul(stride, height)
	if !ok || size > len(buf) {
		return frame.Plane{}, ErrInvalidBatch
	}
	return frame.Plane{
		Pix:    buf[:size],
		Stride: stride,
		Width:  width,
		Height: height,
	}, nil
}

func frameWorkOBMCAboveHeight(size tile.BlockSize, geom frameWorkPredictionPlaneGeometry) (int, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return 0, ErrInvalidBatch
	}
	overlap := min(int(dims.H4)*4, 64)
	overlap >>= 1
	if geom.SubsamplingY {
		overlap >>= 1
	}
	if overlap < 1 {
		overlap = 1
	}
	// Return the full overlap (libaom min(block,64)>>ss>>1): it drives OBMC
	// mask selection; the blend write is clipped to the plane by the caller.
	return overlap, nil
}

func frameWorkOBMCLeftWidth(size tile.BlockSize, geom frameWorkPredictionPlaneGeometry) (int, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return 0, ErrInvalidBatch
	}
	overlap := min(int(dims.W4)*4, 64)
	overlap >>= 1
	if geom.SubsamplingX {
		overlap >>= 1
	}
	if overlap < 1 {
		overlap = 1
	}
	// Full overlap drives OBMC mask selection; the blend write is clipped to
	// the plane by the caller.
	return overlap, nil
}

// frameWorkOBMCLeftFilterHeight returns the OBMC left-neighbor prediction height
// (in plane samples) used to select the interpolation-filter kernel. libaom keys
// the kernel off bh = (op_mi_size * MI_SIZE) >> ss_y where
// op_mi_size = AOMMIN(xd->height, mi_size_high[neighbor]) uses the UN-clipped
// current block MI height (xd->height). neighbor.Span4 (clippedSpan, in plane
// samples) is instead derived from the frame-clipped visible block height by the
// tile decoder, so it agrees with libaom EXCEPT when the block straddles the
// bottom frame edge: there clippedSpan shrinks below libaom's op_mi_size and
// would wrongly switch goav1 to the 4-tap filter. To keep the common (non-edge)
// path byte-identical, only recompute when the plane is frame-clipped vertically
// (geom.Height below the un-clipped plane extent); otherwise return clippedSpan
// unchanged. The result drives kernel selection only; the blend write stays
// clipped to the visible plane by the caller.
func frameWorkOBMCLeftFilterHeight(block tile.BlockVisit, neighbor tile.OverlappableNeighbor, geom frameWorkPredictionPlaneGeometry, color parser.ColorConfig, plane FrameWorkPlane, relY int, clippedSpan int) (int, error) {
	_, extentH, err := frameWorkBlockPlanePredictionExtentPixels(block, color, plane)
	if err != nil {
		return 0, err
	}
	if geom.Height >= extentH {
		// Not bottom-clipped: clippedSpan already matches libaom's op_mi_size.
		return clippedSpan, nil
	}
	neighborDims, ok := neighbor.Size.Dimensions()
	if !ok {
		return 0, ErrInvalidBatch
	}
	blockDims, ok := block.Size.Dimensions()
	if !ok {
		return 0, ErrInvalidBatch
	}
	blockH4 := int(blockDims.H4)
	neighborH4 := min(int(neighborDims.H4), 16)
	opMI := min(neighborH4, blockH4)
	bh := opMI * 4
	if geom.SubsamplingY {
		bh >>= 1
	}
	if bh <= 0 {
		return clippedSpan, nil
	}
	return bh, nil
}

// frameWorkOBMCAboveFilterWidth is the symmetric helper for the OBMC above
// neighbor: libaom keys the kernel off bw = (op_mi_size * MI_SIZE) >> ss_x with
// op_mi_size = AOMMIN(xd->width, mi_size_wide[neighbor]), the UN-clipped current
// block MI width. Only recompute when the plane is frame-clipped horizontally so
// the non-edge path stays byte-identical to neighbor.Span4.
func frameWorkOBMCAboveFilterWidth(block tile.BlockVisit, neighbor tile.OverlappableNeighbor, geom frameWorkPredictionPlaneGeometry, color parser.ColorConfig, plane FrameWorkPlane, relX int, clippedSpan int) (int, error) {
	extentW, _, err := frameWorkBlockPlanePredictionExtentPixels(block, color, plane)
	if err != nil {
		return 0, err
	}
	if geom.Width >= extentW {
		return clippedSpan, nil
	}
	blockDims, ok := block.Size.Dimensions()
	if !ok {
		return 0, ErrInvalidBatch
	}
	neighborDims, ok := neighbor.Size.Dimensions()
	if !ok {
		return 0, ErrInvalidBatch
	}
	blockW4 := int(blockDims.W4)
	neighborW4 := min(int(neighborDims.W4), 16)
	opMI := min(neighborW4, blockW4)
	bw := opMI * 4
	if geom.SubsamplingX {
		bw >>= 1
	}
	if bw <= 0 {
		return clippedSpan, nil
	}
	return bw, nil
}

func frameWorkOBMCPlaneOffset(rel4 int, subsampled bool) (int, bool) {
	offset := rel4 * 4
	if subsampled {
		if offset&1 != 0 {
			return 0, false
		}
		offset >>= 1
	}
	return offset, offset >= 0
}

func frameWorkOBMCPlaneSpan(span4 uint8, subsampled bool) (int, bool) {
	span := int(span4) * 4
	if subsampled {
		span = (span + 1) >> 1
	}
	return span, span > 0
}

func frameWorkOBMCMask(length int) ([]uint8, bool) {
	switch length {
	case 1:
		return frameWorkOBMCMask1[:], true
	case 2:
		return frameWorkOBMCMask2[:], true
	case 4:
		return frameWorkOBMCMask4[:], true
	case 8:
		return frameWorkOBMCMask8[:], true
	case 16:
		return frameWorkOBMCMask16[:], true
	case 32:
		return frameWorkOBMCMask32[:], true
	case 64:
		return frameWorkOBMCMask64[:], true
	default:
		return nil, false
	}
}

func frameWorkBlendOBMCV(dst frame.Plane, tmp frame.Plane, bytesPerSample int, dstX int, dstY int, tmpX int, tmpY int, width int, height int, mask []uint8) error {
	if len(mask) < height ||
		!frameWorkPlaneBlockAddressable(dst, bytesPerSample, dstX, dstY, width, height) ||
		!frameWorkPlaneBlockAddressable(tmp, bytesPerSample, tmpX, tmpY, width, height) {
		return ErrInvalidBatch
	}
	for row := range height {
		m := uint16(mask[row])
		for col := range width {
			a, ok := frameWorkLoadSample(dst, bytesPerSample, dstX+col, dstY+row)
			if !ok {
				return ErrInvalidBatch
			}
			b, ok := frameWorkLoadSample(tmp, bytesPerSample, tmpX+col, tmpY+row)
			if !ok {
				return ErrInvalidBatch
			}
			if !frameWorkStoreSample(dst, bytesPerSample, dstX+col, dstY+row, frameWorkBlendA64(m, a, b)) {
				return ErrInvalidBatch
			}
		}
	}
	return nil
}

func frameWorkBlendOBMCH(dst frame.Plane, tmp frame.Plane, bytesPerSample int, dstX int, dstY int, tmpX int, tmpY int, width int, height int, mask []uint8) error {
	if len(mask) < width ||
		!frameWorkPlaneBlockAddressable(dst, bytesPerSample, dstX, dstY, width, height) ||
		!frameWorkPlaneBlockAddressable(tmp, bytesPerSample, tmpX, tmpY, width, height) {
		return ErrInvalidBatch
	}
	for row := range height {
		for col := range width {
			m := uint16(mask[col])
			a, ok := frameWorkLoadSample(dst, bytesPerSample, dstX+col, dstY+row)
			if !ok {
				return ErrInvalidBatch
			}
			b, ok := frameWorkLoadSample(tmp, bytesPerSample, tmpX+col, tmpY+row)
			if !ok {
				return ErrInvalidBatch
			}
			if !frameWorkStoreSample(dst, bytesPerSample, dstX+col, dstY+row, frameWorkBlendA64(m, a, b)) {
				return ErrInvalidBatch
			}
		}
	}
	return nil
}

func frameWorkBlendCompoundBlock(dst frame.Plane, first frame.Plane, second frame.Plane, bytesPerSample int, dstX int, dstY int, width int, height int, fwdOffset int, bckOffset int) error {
	if !frameWorkPlaneBlockAddressable(dst, bytesPerSample, dstX, dstY, width, height) ||
		!frameWorkPlaneBlockAddressable(first, bytesPerSample, 0, 0, width, height) ||
		!frameWorkPlaneBlockAddressable(second, bytesPerSample, 0, 0, width, height) {
		return ErrInvalidBatch
	}
	if fwdOffset < 0 || bckOffset < 0 || fwdOffset+bckOffset != 1<<frameWorkDistPrecisionBits {
		return ErrInvalidBatch
	}
	switch bytesPerSample {
	case 1:
		for row := range height {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX : (dstY+row)*dst.Stride+dstX+width]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width]
			for col := range width {
				out := (uint32(firstLine[col])*uint32(fwdOffset) + uint32(secondLine[col])*uint32(bckOffset) + 1<<(frameWorkDistPrecisionBits-1)) >> frameWorkDistPrecisionBits
				dstLine[col] = byte(out)
			}
		}
	case 2:
		for row := range height {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX*2 : (dstY+row)*dst.Stride+dstX*2+width*2]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width*2]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width*2]
			for col := range width {
				i := col * 2
				a := uint16(firstLine[i]) | uint16(firstLine[i+1])<<8
				b := uint16(secondLine[i]) | uint16(secondLine[i+1])<<8
				out := uint16((uint32(a)*uint32(fwdOffset) + uint32(b)*uint32(bckOffset) + 1<<(frameWorkDistPrecisionBits-1)) >> frameWorkDistPrecisionBits)
				dstLine[i] = byte(out)
				dstLine[i+1] = byte(out >> 8)
			}
		}
	default:
		return ErrInvalidBatch
	}
	return nil
}

func frameWorkBuildDiffWtdMask(mask []byte, maskStride int, first frame.Plane, second frame.Plane, bytesPerSample int, bitDepth uint8, width int, height int, maskType tile.DiffWtdMaskType) error {
	if !maskType.Valid() ||
		!frameWorkPlaneBlockAddressable(first, bytesPerSample, 0, 0, width, height) ||
		!frameWorkPlaneBlockAddressable(second, bytesPerSample, 0, 0, width, height) ||
		!frameWorkMaskBlockFits(len(mask), maskStride, width, height) {
		return ErrInvalidBatch
	}
	shift := uint8(0)
	if bitDepth > 8 {
		shift = bitDepth - 8
	} else if bitDepth != 8 {
		return ErrInvalidBatch
	}
	invert := maskType == tile.DiffWtdMaskType38Inv
	for row := range height {
		for col := range width {
			a, ok := frameWorkLoadSample(first, bytesPerSample, col, row)
			if !ok {
				return ErrInvalidBatch
			}
			b, ok := frameWorkLoadSample(second, bytesPerSample, col, row)
			if !ok {
				return ErrInvalidBatch
			}
			diff := int(a)
			if b > a {
				diff = int(b - a)
			} else {
				diff -= int(b)
			}
			if shift != 0 {
				diff >>= shift
			}
			m := min(frameWorkDiffWtdMaskBase+diff/frameWorkDiffWtdFactor, frameWorkBlendA64MaxAlpha)
			if invert {
				m = frameWorkBlendA64MaxAlpha - m
			}
			mask[row*maskStride+col] = byte(m)
		}
	}
	return nil
}

func frameWorkBuildInterIntraMask(mask []byte, maskStride int, width int, height int, mode tile.InterIntraMode) error {
	if !mode.Valid() ||
		!frameWorkMaskBlockFits(len(mask), maskStride, width, height) {
		return ErrInvalidBatch
	}
	scaleBase := max(height, width)
	if scaleBase <= 0 || scaleBase > len(frameWorkInterIntraWeights) {
		return ErrInvalidBatch
	}
	sizeScale := len(frameWorkInterIntraWeights) / scaleBase
	if sizeScale <= 0 {
		return ErrInvalidBatch
	}
	for row := range height {
		for col := range width {
			index := 0
			switch mode {
			case tile.InterIntraModeVertical:
				index = row * sizeScale
			case tile.InterIntraModeHorizontal:
				index = col * sizeScale
			case tile.InterIntraModeSmooth:
				index = min(col, row)
				index *= sizeScale
			case tile.InterIntraModeDC:
				mask[row*maskStride+col] = frameWorkBlendA64MaxAlpha / 2
				continue
			default:
				return ErrInvalidBatch
			}
			if index < 0 || index >= len(frameWorkInterIntraWeights) {
				return ErrInvalidBatch
			}
			mask[row*maskStride+col] = frameWorkInterIntraWeights[index]
		}
	}
	return nil
}

func frameWorkBlendMaskedCompoundBlock(dst frame.Plane, first frame.Plane, second frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mask []byte, maskStride int, subX bool, subY bool) error {
	max, ok := frameWorkSampleMax(bitDepth)
	if !ok ||
		!frameWorkPlaneBlockAddressable(dst, bytesPerSample, dstX, dstY, width, height) ||
		!frameWorkPlaneBlockAddressable(first, bytesPerSample, 0, 0, width, height) ||
		!frameWorkPlaneBlockAddressable(second, bytesPerSample, 0, 0, width, height) ||
		!frameWorkMaskBlockFits(len(mask), maskStride, frameWorkMaskWidth(width, subX), frameWorkMaskHeight(height, subY)) {
		return ErrInvalidBatch
	}
	switch bytesPerSample {
	case 1:
		for row := range height {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX : (dstY+row)*dst.Stride+dstX+width]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width]
			for col := range width {
				m, ok := frameWorkBlendMaskSample(mask, maskStride, row, col, subX, subY)
				if !ok {
					return ErrInvalidBatch
				}
				a := uint16(firstLine[col])
				b := uint16(secondLine[col])
				if a > max || b > max {
					return ErrInvalidBatch
				}
				dstLine[col] = byte(frameWorkBlendA64(uint16(m), a, b))
			}
		}
	case 2:
		for row := range height {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX*2 : (dstY+row)*dst.Stride+dstX*2+width*2]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width*2]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width*2]
			for col := range width {
				i := col * 2
				a := uint16(firstLine[i]) | uint16(firstLine[i+1])<<8
				b := uint16(secondLine[i]) | uint16(secondLine[i+1])<<8
				m, ok := frameWorkBlendMaskSample(mask, maskStride, row, col, subX, subY)
				if !ok {
					return ErrInvalidBatch
				}
				if a > max || b > max {
					return ErrInvalidBatch
				}
				out := frameWorkBlendA64(uint16(m), a, b)
				dstLine[i] = byte(out)
				dstLine[i+1] = byte(out >> 8)
			}
		}
	default:
		return ErrInvalidBatch
	}
	return nil
}

func frameWorkBlendInterIntraBlock(dst frame.Plane, inter frame.Plane, intra frame.Plane, bytesPerSample int, bitDepth uint8, dstX int, dstY int, width int, height int, mask []byte, maskStride int, subX bool, subY bool) error {
	return frameWorkBlendMaskedCompoundBlock(dst, intra, inter, bytesPerSample, bitDepth, dstX, dstY, width, height, mask, maskStride, subX, subY)
}

func frameWorkBlendA64(mask uint16, first uint16, second uint16) uint16 {
	return uint16((uint32(mask)*uint32(first) + uint32(frameWorkBlendA64MaxAlpha-mask)*uint32(second) + 32) >> 6)
}

func frameWorkBlendMaskSample(mask []byte, stride int, row int, col int, subX bool, subY bool) (uint8, bool) {
	var out uint8
	switch {
	case !subX && !subY:
		out = mask[row*stride+col]
	case subX && subY:
		sum := int(mask[(2*row)*stride+2*col]) +
			int(mask[(2*row+1)*stride+2*col]) +
			int(mask[(2*row)*stride+2*col+1]) +
			int(mask[(2*row+1)*stride+2*col+1])
		out = uint8((sum + 2) >> 2)
	case subX:
		sum := int(mask[row*stride+2*col]) + int(mask[row*stride+2*col+1])
		out = uint8((sum + 1) >> 1)
	default:
		sum := int(mask[(2*row)*stride+col]) + int(mask[(2*row+1)*stride+col])
		out = uint8((sum + 1) >> 1)
	}
	return out, out <= frameWorkBlendA64MaxAlpha
}

func frameWorkMaskBlockFits(length int, stride int, width int, height int) bool {
	if width <= 0 || height <= 0 || stride < width {
		return false
	}
	lastRow, ok := frameWorkCheckedMul(height-1, stride)
	if !ok {
		return false
	}
	needed, ok := frameWorkCheckedAdd(lastRow, width)
	return ok && needed <= length
}

func frameWorkMaskWidth(width int, subX bool) int {
	if subX {
		return width << 1
	}
	return width
}

func frameWorkMaskHeight(height int, subY bool) int {
	if subY {
		return height << 1
	}
	return height
}

func frameWorkSampleMax(bitDepth uint8) (uint16, bool) {
	switch bitDepth {
	case 8, 10, 12:
		return uint16((1 << bitDepth) - 1), true
	default:
		return 0, false
	}
}

func frameWorkPlaneBlockAddressable(plane frame.Plane, bytesPerSample int, x int, y int, width int, height int) bool {
	if (bytesPerSample != 1 && bytesPerSample != 2) ||
		x < 0 || y < 0 ||
		width <= 0 || height <= 0 ||
		x > plane.Width-width ||
		y > plane.Height-height {
		return false
	}
	rowBytes, ok := frameWorkCheckedMul(width, bytesPerSample)
	if !ok || rowBytes <= 0 || rowBytes > plane.Stride {
		return false
	}
	rowOffset, ok := frameWorkCheckedMul(y, plane.Stride)
	if !ok {
		return false
	}
	colOffset, ok := frameWorkCheckedMul(x, bytesPerSample)
	if !ok {
		return false
	}
	offset, ok := frameWorkCheckedAdd(rowOffset, colOffset)
	if !ok || offset < 0 {
		return false
	}
	lastRowOffset, ok := frameWorkCheckedMul(height-1, plane.Stride)
	if !ok {
		return false
	}
	windowLen, ok := frameWorkCheckedAdd(lastRowOffset, rowBytes)
	if !ok {
		return false
	}
	end, ok := frameWorkCheckedAdd(offset, windowLen)
	return ok && end <= len(plane.Pix)
}

func frameWorkMotionFilters(info parser.TileInfo) (motion.InterpFilters, error) {
	var filter motion.InterpFilter
	switch info.InterpolationFilter {
	case parser.InterpolationEightTap:
		filter = motion.InterpEightTapRegular
	case parser.InterpolationSmooth:
		filter = motion.InterpEightTapSmooth
	case parser.InterpolationSharp:
		filter = motion.InterpMultiTapSharp
	case parser.InterpolationBilinear:
		filter = motion.InterpBilinear
	default:
		return motion.InterpFilters{}, ErrInvalidBatch
	}
	return motion.InterpFilters{X: filter, Y: filter}, nil
}

func frameWorkVisitMotionFilters(info parser.TileInfo, prediction tile.BlockPredictionModeResult) (motion.InterpFilters, error) {
	if info.InterpolationFilter == parser.InterpolationSwitchable {
		if !prediction.InterpFiltersValid || !prediction.InterpFilters.X.Valid() || !prediction.InterpFilters.Y.Valid() {
			return motion.InterpFilters{}, ErrInvalidBatch
		}
		return prediction.InterpFilters, nil
	}
	return frameWorkMotionFilters(info)
}

func frameWorkReferenceFromTile(ref tile.ReferenceFrame) (FrameWorkReference, bool) {
	switch ref {
	case tile.ReferenceFrameLast:
		return FrameWorkReferenceLast, true
	case tile.ReferenceFrameLast2:
		return FrameWorkReferenceLast2, true
	case tile.ReferenceFrameLast3:
		return FrameWorkReferenceLast3, true
	case tile.ReferenceFrameGolden:
		return FrameWorkReferenceGolden, true
	case tile.ReferenceFrameBWD:
		return FrameWorkReferenceBwd, true
	case tile.ReferenceFrameAltref2:
		return FrameWorkReferenceAltRef2, true
	case tile.ReferenceFrameAltref:
		return FrameWorkReferenceAltRef, true
	default:
		return 0, false
	}
}

func frameWorkLumaIntraPredictionMode(mode tile.IntraMode) (prediction.IntraMode, bool) {
	switch mode {
	case tile.IntraModeDC:
		return prediction.IntraModeDC, true
	case tile.IntraModeVertical:
		return prediction.IntraModeVertical, true
	case tile.IntraModeHorizontal:
		return prediction.IntraModeHorizontal, true
	case tile.IntraModeSmooth:
		return prediction.IntraModeSmooth, true
	case tile.IntraModeSmoothVertical:
		return prediction.IntraModeSmoothVertical, true
	case tile.IntraModeSmoothHorizontal:
		return prediction.IntraModeSmoothHorizontal, true
	case tile.IntraModePaeth:
		return prediction.IntraModePaeth, true
	default:
		return 0, false
	}
}

func frameWorkFilterIntraPredictionMode(mode tile.FilterIntraMode) (prediction.FilterIntraMode, bool) {
	switch mode {
	case tile.FilterIntraModeDC:
		return prediction.FilterIntraModeDC, true
	case tile.FilterIntraModeVertical:
		return prediction.FilterIntraModeVertical, true
	case tile.FilterIntraModeHorizontal:
		return prediction.FilterIntraModeHorizontal, true
	case tile.FilterIntraModeD157:
		return prediction.FilterIntraModeD157, true
	case tile.FilterIntraModePaeth:
		return prediction.FilterIntraModePaeth, true
	default:
		return 0, false
	}
}

func frameWorkLumaIntraDirectionalAngle(mode tile.IntraMode, delta int8) (int, bool) {
	base := 0
	switch mode {
	case tile.IntraModeVertical:
		base = 90
	case tile.IntraModeHorizontal:
		base = 180
	case tile.IntraModeD45:
		base = 45
	case tile.IntraModeD135:
		base = 135
	case tile.IntraModeD113:
		base = 113
	case tile.IntraModeD157:
		base = 157
	case tile.IntraModeD203:
		base = 203
	case tile.IntraModeD67:
		base = 67
	default:
		return 0, false
	}
	angle := base + int(delta)*tile.AngleDeltaStep
	if angle == 90 && mode == tile.IntraModeVertical && delta == 0 {
		return 0, false
	}
	if angle == 180 && mode == tile.IntraModeHorizontal && delta == 0 {
		return 0, false
	}
	return angle, true
}

func frameWorkChromaIntraPredictionMode(mode tile.ChromaIntraMode) (prediction.IntraMode, bool) {
	lumaMode, err := mode.LumaMode()
	if err != nil {
		return 0, false
	}
	return frameWorkLumaIntraPredictionMode(lumaMode)
}

func frameWorkInterIntraPredictionMode(mode tile.InterIntraMode) (prediction.IntraMode, bool) {
	switch mode {
	case tile.InterIntraModeDC:
		return prediction.IntraModeDC, true
	case tile.InterIntraModeVertical:
		return prediction.IntraModeVertical, true
	case tile.InterIntraModeHorizontal:
		return prediction.IntraModeHorizontal, true
	case tile.InterIntraModeSmooth:
		return prediction.IntraModeSmooth, true
	default:
		return 0, false
	}
}

func frameWorkChromaIntraDirectionalAngle(mode tile.ChromaIntraMode, delta int8) (int, bool) {
	lumaMode, err := mode.LumaMode()
	if err != nil {
		return 0, false
	}
	return frameWorkLumaIntraDirectionalAngle(lumaMode, delta)
}

func frameWorkBlockVisiblePixels(block tile.BlockVisit) (int, int, bool) {
	if block.VisibleW4 == 0 || block.VisibleH4 == 0 {
		return 0, 0, false
	}
	width, ok := frameWorkInt64Mul4(int64(block.VisibleW4))
	if !ok {
		return 0, 0, false
	}
	height, ok := frameWorkInt64Mul4(int64(block.VisibleH4))
	if !ok {
		return 0, 0, false
	}
	return width, height, true
}

func frameWorkBlockLumaPredictionExtentPixels(block tile.BlockVisit) (int, int, error) {
	dims, ok := block.Size.Dimensions()
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	return int(dims.W4) * 4, int(dims.H4) * 4, nil
}

// frameWorkPlaneFromWindow returns a frame.Plane view that spans the window's
// writable (MI-aligned) extent. Width/Height match ClipWidth/ClipHeight when
// set (falling back to Width/Height); this is the extent prediction and
// residual writes may legitimately reach when blocks straddle the visible
// edge.
func frameWorkPlaneFromWindow(window FrameWorkPlaneRegion) frame.Plane {
	width := window.ClipWidth
	if width <= 0 {
		width = window.Width
	}
	height := window.ClipHeight
	if height <= 0 {
		height = window.Height
	}
	return frame.Plane{
		Pix:    window.Pix,
		Stride: window.Stride,
		Width:  width,
		Height: height,
	}
}

// frameWorkClipVisiblePixelsToWindow trims a block's pixel rectangle to the
// caller's plane window. The window's pixel-grid origin is (window.X,
// window.Y); writes are permitted up to the MI-aligned trailing edge
// (window.ClipWidth x window.ClipHeight, falling back to window.Width x
// window.Height when ClipWidth/ClipHeight are zero). Callers pass the block's
// absolute plane-grid coordinates plus its MI-aligned visible width and
// height; the helper returns the rectangle clipped to the window's writable
// (MI-aligned) edge.
//
// AV1 prediction and residual writes may legitimately land past the visible
// coded-frame edge: libaom writes whole transform blocks regardless of the
// visible boundary and later blocks read those past-visible samples as
// predictor neighbors. The writable extent is the MI-aligned region
// (xd->mi_params.mi_cols * MI_SIZE in libaom; region.MIColEnd*4 in goav1),
// not the visible width. Clipping to the visible width drops past-visible
// stores and causes downstream blocks to read zeros where libaom has the
// previously-written samples. Use frameWorkPlaneBlockStartsBeyondOutput
// to detect blocks whose origin lies entirely past the coded frame (the
// MI grid can round up beyond the visible region; those blocks are
// silently skipped).
//
// The clamp returns ok=false when the block's origin lands outside the
// writable region (negative coordinates or x/y at or past the writable end).
// When ok is true the returned (width, height) is the largest sub-rectangle
// that fits inside the writable region. Callers must use the returned
// width/height for the downstream writeback and the input predWidth/predHeight
// for libaom's edge/DC/Smooth sample weighting.
func frameWorkClipVisiblePixelsToWindow(window FrameWorkPlaneRegion, x int, y int, width int, height int) (int, int, bool) {
	if width <= 0 || height <= 0 || x < window.X || y < window.Y {
		return 0, 0, false
	}
	clipWidth := window.ClipWidth
	if clipWidth <= 0 {
		clipWidth = window.Width
	}
	clipHeight := window.ClipHeight
	if clipHeight <= 0 {
		clipHeight = window.Height
	}
	windowXEnd, ok := frameWorkCheckedAdd(window.X, clipWidth)
	if !ok {
		return 0, 0, false
	}
	windowYEnd, ok := frameWorkCheckedAdd(window.Y, clipHeight)
	if !ok {
		return 0, 0, false
	}
	if x >= windowXEnd || y >= windowYEnd {
		return 0, 0, false
	}
	if x+width > windowXEnd {
		width = windowXEnd - x
	}
	if y+height > windowYEnd {
		height = windowYEnd - y
	}
	return width, height, width > 0 && height > 0
}

// frameWorkPlaneBlockStartsBeyondOutput reports whether a block's plane-grid
// origin lies entirely past the coded-frame extent of output. AV1 bitstreams
// can address blocks whose MI grid was rounded up past the coded width/height
// (e.g. a 34x34 frame rounds to a 40x40 MI grid, so partition walks emit
// blocks at MI col 9 starting at luma pixel 36). Callers use this short-circuit
// to skip prediction/reconstruction silently when the block has no visible
// samples to write, distinct from the genuine clip-failure path. Negative
// coordinates are rejected so they hit the caller's invalid-batch path.
func frameWorkPlaneBlockStartsBeyondOutput(output *frame.Frame, plane FrameWorkPlane, x int, y int) bool {
	if output == nil || x < 0 || y < 0 {
		return false
	}
	dst, _, _, ok := frameWorkFramePlane(output, plane)
	if !ok {
		return false
	}
	// libaom reconstructs the bottom/right partial superblock into the
	// MI-aligned padding rows/cols of the YV12 buffer, so a block whose origin
	// lands past the cropped Width/Height but inside the MI-aligned allocation
	// is NOT beyond the output: it must still be reconstructed (its samples
	// feed intra neighbor context and CDEF edges for the visible blocks). The
	// allocation spans len(Pix)/Stride rows of Stride/BytesPerSample samples;
	// only origins past that aligned extent are genuinely beyond the frame.
	bytesPerSample := output.Layout.BytesPerSample
	allocWidth := dst.Width
	allocHeight := dst.Height
	if bytesPerSample > 0 && dst.Stride > 0 {
		if w := dst.Stride / bytesPerSample; w > allocWidth {
			allocWidth = w
		}
		if h := len(dst.Pix) / dst.Stride; h > allocHeight {
			allocHeight = h
		}
	}
	return x >= allocWidth || y >= allocHeight
}

func frameWorkBlockWithinJobRegion(region FrameWorkJobRegion, block tile.BlockVisit) bool {
	return block.MICol >= region.MIColStart &&
		block.MIRow >= region.MIRowStart &&
		block.MIColEnd <= region.MIColEnd &&
		block.MIRowEnd <= region.MIRowEnd &&
		block.MIColEnd > block.MICol &&
		block.MIRowEnd > block.MIRow
}

func frameWorkBlockPlanePosition(block tile.BlockVisit, color parser.ColorConfig, plane FrameWorkPlane) (x int, y int, width int, height int, subsamplingX bool, subsamplingY bool, present bool, err error) {
	if plane == FrameWorkPlaneY {
		width, height, present = frameWorkBlockVisiblePixels(block)
		if !present {
			return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
		}
		x, y, err = frameWorkBlockLumaPosition(block)
		return x, y, width, height, false, false, true, err
	}
	if plane != FrameWorkPlaneU && plane != FrameWorkPlaneV {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	if color.MonoChrome {
		return 0, 0, 0, 0, false, false, false, nil
	}
	req := tile.TransformTreeRequest{
		Size:      block.Size,
		X4:        block.X4,
		Y4:        block.Y4,
		VisibleW4: block.VisibleW4,
		VisibleH4: block.VisibleH4,
	}
	if !tile.HasChromaBlock(req, color) {
		return 0, 0, 0, 0, false, false, false, nil
	}
	if _, err := tile.PlaneBlockSize(block.Size, color, int(plane)); err != nil {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	dims, ok := block.Size.Dimensions()
	if !ok || block.VisibleW4 == 0 || block.VisibleH4 == 0 || block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	ssX := int(frameWorkSubsampleShift(color.SubsamplingX))
	ssY := int(frameWorkSubsampleShift(color.SubsamplingY))
	miCol := block.MICol
	miRow := block.MIRow
	if ssX != 0 && miCol&1 != 0 && dims.W4 == 1 {
		miCol--
	}
	if ssY != 0 && miRow&1 != 0 && dims.H4 == 1 {
		miRow--
	}
	x, ok = frameWorkInt64Mul4(int64(miCol))
	if !ok {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	y, ok = frameWorkInt64Mul4(int64(miRow))
	if !ok {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	x >>= ssX
	y >>= ssY
	visibleW4 := ((block.X4 + int(block.VisibleW4) + ssX) >> ssX) - (block.X4 >> ssX)
	visibleH4 := ((block.Y4 + int(block.VisibleH4) + ssY) >> ssY) - (block.Y4 >> ssY)
	if visibleW4 <= 0 || visibleH4 <= 0 {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	width, ok = frameWorkInt64Mul4(int64(visibleW4))
	if !ok {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	height, ok = frameWorkInt64Mul4(int64(visibleH4))
	if !ok {
		return 0, 0, 0, 0, false, false, false, ErrInvalidBatch
	}
	return x, y, width, height, color.SubsamplingX, color.SubsamplingY, true, nil
}

// frameWorkBlockPlaneWarpAllowed reports whether warped motion is permitted for
// the given plane of a block, mirroring libaom's av1_init_warp_params() guard
// "if (inter_pred_params->block_height < 8 || block_width < 8) return;". The
// block_width/block_height there are xd->plane[plane].width/height, i.e. the
// full (un-clipped) plane block dimensions, so this uses the prediction extent
// rather than the frame-edge-clipped visible extent.
func frameWorkBlockPlaneWarpAllowed(block tile.BlockVisit, color parser.ColorConfig, plane FrameWorkPlane) (bool, error) {
	width, height, err := frameWorkBlockPlanePredictionExtentPixels(block, color, plane)
	if err != nil {
		return false, err
	}
	return width >= 8 && height >= 8, nil
}

func frameWorkBlockPlanePredictionExtentPixels(block tile.BlockVisit, color parser.ColorConfig, plane FrameWorkPlane) (int, int, error) {
	if plane == FrameWorkPlaneY {
		dims, ok := block.Size.Dimensions()
		if !ok {
			return 0, 0, ErrInvalidBatch
		}
		return int(dims.W4) * 4, int(dims.H4) * 4, nil
	}
	if plane != FrameWorkPlaneU && plane != FrameWorkPlaneV {
		return 0, 0, ErrInvalidBatch
	}
	planeSize, err := tile.PlaneBlockSize(block.Size, color, int(plane))
	if err != nil {
		return 0, 0, ErrInvalidBatch
	}
	dims, ok := planeSize.Dimensions()
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	return int(dims.W4) * 4, int(dims.H4) * 4, nil
}

func frameWorkSubsampleLumaCFLQ3(dst []uint16, plane frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, subX bool, subY bool) error {
	if !frameWorkPlaneBlockAddressable(plane, bytesPerSample, x, y, width, height) {
		return ErrInvalidBatch
	}
	outW := width
	outH := height
	if subX {
		if width&1 != 0 {
			return ErrInvalidBatch
		}
		outW >>= 1
	}
	if subY {
		if height&1 != 0 {
			return ErrInvalidBatch
		}
		outH >>= 1
	}
	if outW <= 0 || outH <= 0 ||
		width > prediction.CFLBufLine || height > prediction.CFLBufLine ||
		!frameWorkMaskBlockFits(len(dst), prediction.CFLBufLine, outW, outH) {
		return ErrInvalidBatch
	}
	if bytesPerSample == 1 {
		if bitDepth != 8 {
			return ErrInvalidBatch
		}
		rowOffset, ok := frameWorkCheckedMul(y, plane.Stride)
		if !ok {
			return ErrInvalidBatch
		}
		offset, ok := frameWorkCheckedAdd(rowOffset, x)
		if !ok || offset < 0 || offset >= len(plane.Pix) {
			return ErrInvalidBatch
		}
		if err := prediction.SubsampleLuma8ToQ3(dst, plane.Pix[offset:], plane.Stride, width, height, subX, subY); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	if bytesPerSample != 2 {
		return ErrInvalidBatch
	}
	max, ok := frameWorkSampleMax(bitDepth)
	if !ok || bitDepth <= 8 {
		return ErrInvalidBatch
	}
	switch {
	case subX && subY:
		for row := 0; row < height; row += 2 {
			outRow := row >> 1
			for col := 0; col < width; col += 2 {
				p0, ok := frameWorkLoadCFLSourceSample(plane, bytesPerSample, max, x+col, y+row)
				if !ok {
					return ErrInvalidBatch
				}
				p1, ok := frameWorkLoadCFLSourceSample(plane, bytesPerSample, max, x+col+1, y+row)
				if !ok {
					return ErrInvalidBatch
				}
				p2, ok := frameWorkLoadCFLSourceSample(plane, bytesPerSample, max, x+col, y+row+1)
				if !ok {
					return ErrInvalidBatch
				}
				p3, ok := frameWorkLoadCFLSourceSample(plane, bytesPerSample, max, x+col+1, y+row+1)
				if !ok {
					return ErrInvalidBatch
				}
				dst[outRow*prediction.CFLBufLine+(col>>1)] = (p0 + p1 + p2 + p3) << 1
			}
		}
	case subX:
		for row := 0; row < outH; row++ {
			for col := 0; col < width; col += 2 {
				p0, ok := frameWorkLoadCFLSourceSample(plane, bytesPerSample, max, x+col, y+row)
				if !ok {
					return ErrInvalidBatch
				}
				p1, ok := frameWorkLoadCFLSourceSample(plane, bytesPerSample, max, x+col+1, y+row)
				if !ok {
					return ErrInvalidBatch
				}
				dst[row*prediction.CFLBufLine+(col>>1)] = (p0 + p1) << 2
			}
		}
	default:
		for row := 0; row < outH; row++ {
			for col := 0; col < outW; col++ {
				p, ok := frameWorkLoadCFLSourceSample(plane, bytesPerSample, max, x+col, y+row)
				if !ok {
					return ErrInvalidBatch
				}
				dst[row*prediction.CFLBufLine+col] = p << 3
			}
		}
	}
	return nil
}

func frameWorkLoadCFLSourceSample(plane frame.Plane, bytesPerSample int, max uint16, x int, y int) (uint16, bool) {
	sample, ok := frameWorkLoadSample(plane, bytesPerSample, x, y)
	return sample, ok && sample <= max
}

func frameWorkBlockLumaPosition(block tile.BlockVisit) (int, int, error) {
	if block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow {
		return 0, 0, ErrInvalidBatch
	}
	x, ok := frameWorkInt64Mul4(int64(block.MICol))
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	y, ok := frameWorkInt64Mul4(int64(block.MIRow))
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	return x, y, nil
}

func frameWorkBlockLumaTransformPosition(block tile.BlockVisit, tx tile.TransformBlock) (int, int, error) {
	if tx.X4 < block.X4 || tx.Y4 < block.Y4 ||
		tx.X4+int(tx.VisibleW4) > block.X4+int(block.VisibleW4) ||
		tx.Y4+int(tx.VisibleH4) > block.Y4+int(block.VisibleH4) {
		return 0, 0, ErrInvalidBatch
	}
	baseX, baseY, err := frameWorkBlockLumaPosition(block)
	if err != nil {
		return 0, 0, err
	}
	offX, ok := frameWorkInt64Mul4(int64(tx.X4 - block.X4))
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	offY, ok := frameWorkInt64Mul4(int64(tx.Y4 - block.Y4))
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	x, ok := frameWorkCheckedAdd(baseX, offX)
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	y, ok := frameWorkCheckedAdd(baseY, offY)
	if !ok {
		return 0, 0, ErrInvalidBatch
	}
	return x, y, nil
}

func frameWorkIntraPredictionEdges(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, fillMissing bool) (prediction.IntraEdges, error) {
	return frameWorkIntraPredictionEdgesWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, width, height, 0, 0, block, scratch, fillMissing)
}

// frameWorkWindowEdgeReadBound returns the past-end neighbor-read bound (in the
// window-local coordinate system used by dst) that matches libaom's intra
// predictor n_top_px / n_left_px truncation. libaom caps real neighbor samples
// at xr = (mb_to_right_edge >> 3) + ... and yd = (mb_to_bottom_edge >> 3) + ...,
// both of which are derived from xd->mi_params.mi_{cols,rows} * MI_SIZE — the
// MI-aligned decode-buffer extent, NOT the cropped display dimension. The MI-
// aligned writable extent is window.ClipWidth / window.ClipHeight (zero means
// "visible == aligned", i.e. an SB-aligned plane where Width already equals the
// MI grid). Returning the aligned extent lets a block in the bottom/right
// partial superblock read the in-grid reconstructed neighbor samples libaom
// reads (e.g. a 34px-wide frame has a 40px MI-grid width, so an 8x8 transform
// at x=32 sees n_top_px=8, not the cropped n_top_px=2).
func frameWorkWindowEdgeReadBound(window FrameWorkPlaneRegion) (int, int) {
	// Neighbor reads cap at the MI-aligned extent (libaom's n_top_px/n_left_px
	// from mi_{cols,rows} * MI_SIZE), NOT the superblock-aligned writable extent.
	// Reading to the wider writable edge would let blocks straddling the MI grid
	// edge see past-MI samples libaom never reads, corrupting interior edge
	// blocks.
	w := window.ReadWidth
	if w <= 0 {
		w = window.ClipWidth
	}
	if w <= 0 {
		w = window.Width
	}
	h := window.ReadHeight
	if h <= 0 {
		h = window.ClipHeight
	}
	if h <= 0 {
		h = window.Height
	}
	return w, h
}

// frameWorkWindowEdgeReadBoundAbsolute returns the same MI-aligned neighbor-read
// past-end as frameWorkWindowEdgeReadBound but in absolute plane coordinates
// (for geom-absolute callers that index geom.Output directly). The aligned
// trailing edge is window.X + ClipWidth (window.Y + ClipHeight), matching
// libaom's xr / yd derived from xd->mi_params.mi_{cols,rows} * MI_SIZE.
func frameWorkWindowEdgeReadBoundAbsolute(window FrameWorkPlaneRegion) (int, int) {
	w, h := frameWorkWindowEdgeReadBound(window)
	return window.X + w, window.Y + h
}

// frameWorkIntraPredictionEdgesWithExtent builds the top/left/top-left intra
// predictor edge samples for one block. visibleW / visibleH are the
// past-visible-end positions in the same coordinate system as x / dst (so for
// window-local callers they pass window.Width / window.Height, and for
// geom-absolute callers they pass geom.Window.X + geom.Window.Width and
// geom.Window.Y + geom.Window.Height). When visibleW or visibleH is <= 0 the
// function falls back to dst.Width / dst.Height (legacy "no visible bound"
// behavior). The visible bound is required to match libaom's
// n_top_px = min(txwpx, xr+txwpx) / n_left_px = min(txhpx, yd+txhpx) edge
// truncation: blocks straddling the right or bottom visible edge must read
// only the in-frame neighbor samples and replicate the last visible sample
// for the past-visible slots, never the MI-aligned past-visible neighbors
// that libaom does not see.
func frameWorkIntraPredictionEdgesWithExtent(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, edgeWidth int, edgeHeight int, visibleW int, visibleH int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, fillMissing bool) (prediction.IntraEdges, error) {
	if scratch == nil {
		return prediction.IntraEdges{}, ErrInvalidBatch
	}
	if edgeWidth < width || edgeHeight < height ||
		edgeWidth > frameWorkIntraPredictionMaxEdgeSamples ||
		edgeHeight > frameWorkIntraPredictionMaxEdgeSamples {
		return prediction.IntraEdges{}, ErrInvalidBatch
	}
	readBoundX := dst.Width
	if visibleW > 0 && visibleW < readBoundX {
		readBoundX = visibleW
	}
	readBoundY := dst.Height
	if visibleH > 0 && visibleH < readBoundY {
		readBoundY = visibleH
	}
	var edges prediction.IntraEdges
	// Track libaom's n_top_px / n_left_px so the AboveLeft selection below
	// can match libaom when the block straddles the visible edge:
	// nTopPx == 0 && nLeftPx > 0 -> AboveLeft = left_ref[0] = load(x-1, y)
	// nTopPx > 0 && nLeftPx == 0 -> AboveLeft = above_ref[0] = load(x, y-1)
	// otherwise (both > 0)        -> AboveLeft = load(x-1, y-1)
	nTopPx := 0
	nLeftPx := 0
	if block.HaveTop {
		if y <= 0 {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
		available := edgeWidth
		if x+available > readBoundX {
			available = readBoundX - x
		}
		if available > 0 {
			nTopPx = available
			for col := 0; col < available; col++ {
				sample, ok := frameWorkLoadSample(dst, bytesPerSample, x+col, y-1)
				if !ok {
					return prediction.IntraEdges{}, ErrInvalidBatch
				}
				scratch.Above[col] = sample
			}
			for col := available; col < edgeWidth; col++ {
				scratch.Above[col] = scratch.Above[available-1]
			}
			edges.Above = scratch.Above[:edgeWidth]
			edges.AboveAvailable = true
		} else {
			// libaom n_top_px=0 fallback: when the block has HaveTop but the
			// entire top neighbor lies past the visible right edge, fill
			// above_row with left_ref[0] when n_left_px>0 (HaveLeft and not
			// past the bottom visible edge) or with base+1 (=129 for 8-bit)
			// otherwise. frameWorkMissingAboveSample implements that fallback.
			sample, err := frameWorkMissingAboveSample(dst, bytesPerSample, bitDepth, x, y, block)
			if err != nil {
				return prediction.IntraEdges{}, err
			}
			for col := range edgeWidth {
				scratch.Above[col] = sample
			}
			edges.Above = scratch.Above[:edgeWidth]
			edges.AboveAvailable = true
		}
	} else if fillMissing {
		sample, err := frameWorkMissingAboveSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return prediction.IntraEdges{}, err
		}
		for col := range edgeWidth {
			scratch.Above[col] = sample
		}
		edges.Above = scratch.Above[:edgeWidth]
		edges.AboveAvailable = true
	}
	if block.HaveLeft {
		if x <= 0 {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
		available := edgeHeight
		if y+available > readBoundY {
			available = readBoundY - y
		}
		if available > 0 {
			nLeftPx = available
			for row := 0; row < available; row++ {
				sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y+row)
				if !ok {
					return prediction.IntraEdges{}, ErrInvalidBatch
				}
				scratch.Left[row] = sample
			}
			for row := available; row < edgeHeight; row++ {
				scratch.Left[row] = scratch.Left[available-1]
			}
			edges.Left = scratch.Left[:edgeHeight]
			edges.LeftAvailable = true
		} else {
			// libaom n_left_px=0 fallback: mirror n_top_px=0 — when the block
			// starts past the bottom visible edge, fill left with the top
			// neighbor (if HaveTop) or with base+1 otherwise.
			sample, err := frameWorkMissingLeftSample(dst, bytesPerSample, bitDepth, x, y, block)
			if err != nil {
				return prediction.IntraEdges{}, err
			}
			for row := range edgeHeight {
				scratch.Left[row] = sample
			}
			edges.Left = scratch.Left[:edgeHeight]
			edges.LeftAvailable = true
		}
	} else if fillMissing {
		sample, err := frameWorkMissingLeftSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return prediction.IntraEdges{}, err
		}
		for row := range edgeHeight {
			scratch.Left[row] = sample
		}
		edges.Left = scratch.Left[:edgeHeight]
		edges.LeftAvailable = true
	}
	if block.HaveTop && block.HaveLeft {
		var sample uint16
		var ok bool
		switch {
		case nTopPx > 0 && nLeftPx > 0:
			sample, ok = frameWorkLoadSample(dst, bytesPerSample, x-1, y-1)
		case nTopPx > 0:
			// n_left_px=0: AboveLeft = above_ref[0] = load(x, y-1)
			sample, ok = frameWorkLoadSample(dst, bytesPerSample, x, y-1)
		case nLeftPx > 0:
			// n_top_px=0: AboveLeft = left_ref[0] = load(x-1, y)
			sample, ok = frameWorkLoadSample(dst, bytesPerSample, x-1, y)
		default:
			// both 0: AboveLeft = base (=128 for 8-bit)
			sample, ok = frameWorkIntraBoundaryDefault(bitDepth, 0)
		}
		if !ok {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
		edges.AboveLeft = sample
		edges.AboveLeftAvailable = true
	} else if fillMissing {
		sample, err := frameWorkMissingTopLeftSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return prediction.IntraEdges{}, err
		}
		edges.AboveLeft = sample
		edges.AboveLeftAvailable = true
	}
	return edges, nil
}

func frameWorkDirectionalPredictionEdges(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, angle int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, enableIntraEdgeFilter bool, smoothNeighbor bool, allowTopRight bool, allowBottomLeft bool, visibleW int, visibleH int) (prediction.DirectionalEdges, error) {
	if scratch == nil || angle <= 0 || angle >= 270 {
		return prediction.DirectionalEdges{}, ErrInvalidBatch
	}
	edges := prediction.DirectionalEdges{
		Above:       scratch.Above[:],
		Left:        scratch.Left[:],
		AboveOrigin: frameWorkDirectionalEdgeOrigin,
		LeftOrigin:  frameWorkDirectionalEdgeOrigin,
	}

	switch {
	case angle < 90:
		if err := frameWorkFillDirectionalAbove(dst, bytesPerSample, bitDepth, x, y, 0, width+height-1, width, allowTopRight, block, scratch, visibleW); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	case angle < 180:
		if err := frameWorkFillDirectionalAbove(dst, bytesPerSample, bitDepth, x, y, -height, width+height, width, false, block, scratch, visibleW); err != nil {
			return prediction.DirectionalEdges{}, err
		}
		if err := frameWorkFillDirectionalLeft(dst, bytesPerSample, bitDepth, x, y, -width, width+height, height, false, block, scratch, visibleH); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	default:
		if err := frameWorkFillDirectionalLeft(dst, bytesPerSample, bitDepth, x, y, 0, width+height-1, height, allowBottomLeft, block, scratch, visibleH); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	}
	if angle != 90 && angle != 180 {
		topLeft, err := frameWorkDirectionalAboveLeftSample(dst, bytesPerSample, bitDepth, x, y, block, visibleW, visibleH, width, height)
		if err != nil {
			return prediction.DirectionalEdges{}, err
		}
		scratch.Above[frameWorkDirectionalEdgeOrigin-1] = topLeft
		scratch.Left[frameWorkDirectionalEdgeOrigin-1] = topLeft
	}
	if enableIntraEdgeFilter {
		if err := frameWorkApplyDirectionalIntraEdgeFilter(bitDepth, width, height, angle, block, smoothNeighbor, &edges, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	}
	return edges, nil
}

func frameWorkApplyDirectionalIntraEdgeFilter(bitDepth uint8, width int, height int, angle int, block tile.BlockVisit, smoothNeighbor bool, edges *prediction.DirectionalEdges, scratch *FrameWorkIntraPredictionScratch) error {
	needAbove := angle < 180
	needLeft := angle > 90
	needRight := angle < 90
	needBottom := angle > 180
	nTop := 0
	if block.HaveTop {
		nTop = width
	}
	nLeft := 0
	if block.HaveLeft {
		nLeft = height
	}
	if angle != 90 && angle != 180 {
		if needAbove && needLeft && width+height >= 24 {
			if err := prediction.FilterIntraEdgeCorner(edges.Above, edges.AboveOrigin, edges.Left, edges.LeftOrigin, bitDepth); err != nil {
				return ErrInvalidBatch
			}
		}
		if needAbove && nTop > 0 {
			strength := prediction.IntraEdgeFilterStrength(width, height, angle-90, smoothNeighbor)
			npx := nTop + 1
			if needRight {
				npx += height
			}
			if !frameWorkDirectionalEdgeFilterRangeFits(edges.AboveOrigin, npx) {
				return ErrInvalidBatch
			}
			if err := prediction.FilterIntraEdge(edges.Above[edges.AboveOrigin-1:edges.AboveOrigin-1+npx], scratch.Edge[:], strength, bitDepth); err != nil {
				return ErrInvalidBatch
			}
		}
		if needLeft && nLeft > 0 {
			strength := prediction.IntraEdgeFilterStrength(height, width, angle-180, smoothNeighbor)
			npx := nLeft + 1
			if needBottom {
				npx += width
			}
			if !frameWorkDirectionalEdgeFilterRangeFits(edges.LeftOrigin, npx) {
				return ErrInvalidBatch
			}
			if err := prediction.FilterIntraEdge(edges.Left[edges.LeftOrigin-1:edges.LeftOrigin-1+npx], scratch.Edge[:], strength, bitDepth); err != nil {
				return ErrInvalidBatch
			}
		}
	}
	upsampleAbove := prediction.UseIntraEdgeUpsample(width, height, angle-90, smoothNeighbor)
	if needAbove && upsampleAbove {
		npx := width
		if needRight {
			npx += height
		}
		if err := prediction.UpsampleIntraEdge(edges.Above, edges.AboveOrigin, npx, scratch.Edge[:], bitDepth); err != nil {
			return ErrInvalidBatch
		}
		edges.UpsampleAbove = true
	}
	upsampleLeft := prediction.UseIntraEdgeUpsample(height, width, angle-180, smoothNeighbor)
	if needLeft && upsampleLeft {
		npx := height
		if needBottom {
			npx += width
		}
		if err := prediction.UpsampleIntraEdge(edges.Left, edges.LeftOrigin, npx, scratch.Edge[:], bitDepth); err != nil {
			return ErrInvalidBatch
		}
		edges.UpsampleLeft = true
	}
	return nil
}

func frameWorkDirectionalEdgeFilterRangeFits(origin int, npx int) bool {
	return npx > 0 &&
		origin-1 >= 0 &&
		origin-1+npx <= frameWorkDirectionalEdgeSamples &&
		npx <= frameWorkIntraEdgeScratchSamples
}

// frameWorkDirectionalAboveLeftSample mirrors libaom's above_row[-1] /
// left_col[-1] derivation. The visibleW / visibleH arguments are the
// past-visible-end positions in the same coordinate system as x / y / dst;
// width / height are the directional-predictor primary extents (txwpx /
// txhpx). When visibleW or visibleH is <= 0, the routine reverts to the
// pre-edge-truncation behavior (no past-visible accounting).
func frameWorkDirectionalAboveLeftSample(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, block tile.BlockVisit, visibleW int, visibleH int, width int, height int) (uint16, error) {
	if block.HaveTop && block.HaveLeft {
		nTopPx := width
		if visibleW > 0 {
			nTopPx = max(visibleW-x, 0)
			if nTopPx > width {
				nTopPx = width
			}
		}
		nLeftPx := height
		if visibleH > 0 {
			nLeftPx = max(visibleH-y, 0)
			if nLeftPx > height {
				nLeftPx = height
			}
		}
		switch {
		case nTopPx > 0 && nLeftPx > 0:
			sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y-1)
			if !ok {
				return 0, ErrInvalidBatch
			}
			return sample, nil
		case nTopPx > 0:
			sample, ok := frameWorkLoadSample(dst, bytesPerSample, x, y-1)
			if !ok {
				return 0, ErrInvalidBatch
			}
			return sample, nil
		case nLeftPx > 0:
			sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y)
			if !ok {
				return 0, ErrInvalidBatch
			}
			return sample, nil
		default:
			sample, ok := frameWorkIntraBoundaryDefault(bitDepth, 0)
			if !ok {
				return 0, ErrInvalidBatch
			}
			return sample, nil
		}
	}
	if block.HaveTop {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x, y-1)
		if !ok {
			return 0, ErrInvalidBatch
		}
		return sample, nil
	}
	if block.HaveLeft {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y)
		if !ok {
			return 0, ErrInvalidBatch
		}
		return sample, nil
	}
	sample, ok := frameWorkIntraBoundaryDefault(bitDepth, 0)
	if !ok {
		return 0, ErrInvalidBatch
	}
	return sample, nil
}

// frameWorkFillDirectionalAbove fills the above directional-predictor edge.
// visibleW is the past-visible-end column in the same coordinate system as x
// / dst (so window-local callers pass window.Width and geom-absolute callers
// pass geom.Window.X + geom.Window.Width). When visibleW <= 0 the function
// falls back to dst.Width (legacy behavior). The visible bound is required
// to mirror libaom's n_top_px = min(txwpx, xr + txwpx) /
// n_topright_px = min(txwpx, xr) caps: when a transform block straddles the
// visible right edge, libaom clamps the real-sample range to the visible
// extent and replicates the last visible sample for the past-visible slots.
func frameWorkFillDirectionalAbove(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, minIndex int, maxIndex int, primaryWidth int, allowTopRight bool, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, visibleW int) error {
	if !frameWorkDirectionalRangeFits(minIndex, maxIndex) {
		return ErrInvalidBatch
	}
	if !block.HaveTop {
		sample, err := frameWorkMissingAboveSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return err
		}
		for i := minIndex; i <= maxIndex; i++ {
			scratch.Above[frameWorkDirectionalEdgeOrigin+i] = sample
		}
		return nil
	}
	if y <= 0 {
		return ErrInvalidBatch
	}
	readBoundX := dst.Width
	if visibleW > 0 && visibleW < readBoundX {
		readBoundX = visibleW
	}
	// libaom caps n_top_px = min(txwpx, xr + txwpx) and
	// n_topright_px = min(txwpx, xr). For a block at x with primary
	// transform width primaryWidth and a visible plane extent of
	// readBoundX, that means real samples come from cols
	// x..min(x+primaryWidth, readBoundX)-1 with the rest replicated, and
	// when allowTopRight is set the top-right extension may extend at most
	// to readBoundX-1 (a further primaryWidth cols, capped).
	primaryLimit := max(readBoundX-x, 0)
	if primaryLimit > primaryWidth {
		primaryLimit = primaryWidth
	}
	if primaryLimit == 0 {
		// libaom n_top_px == 0 fallback: fill the entire above buffer with
		// left_ref[0] (load(x-1, y)) when HaveLeft, else base+1=129.
		// frameWorkMissingAboveSample implements exactly that contract.
		sample, err := frameWorkMissingAboveSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return err
		}
		for i := minIndex; i <= maxIndex; i++ {
			scratch.Above[frameWorkDirectionalEdgeOrigin+i] = sample
		}
		return nil
	}
	topRightLimit := max(readBoundX-x-primaryWidth, 0)
	if topRightLimit > primaryWidth {
		topRightLimit = primaryWidth
	}
	// libaom stores the above_row buffer in three logical zones:
	//   zone A: i in [0, primaryWidth-1]
	//     - i < primaryLimit:      real sample (load from above)
	//     - i >= primaryLimit:     replicate above_row[primaryLimit-1]
	//   zone B: i in [primaryWidth, primaryWidth + topRightLimit - 1] when allowTopRight
	//     - real sample (load from above)
	//   zone C: rest (i >= primaryWidth + topRightLimit, or no allowTopRight)
	//     - replicate the last filled sample, which is either above_row[primaryWidth+topRightLimit-1]
	//       (when allowTopRight && topRightLimit > 0) or above_row[primaryLimit-1] otherwise.
	maxRealEnd := primaryLimit
	if allowTopRight && topRightLimit > 0 {
		maxRealEnd = primaryWidth + topRightLimit
	}
	for i := minIndex; i <= maxIndex; i++ {
		var sampleX int
		switch {
		case i < primaryLimit:
			sampleX = x + i
		case allowTopRight && i >= primaryWidth && i < primaryWidth+topRightLimit:
			sampleX = x + i
		default:
			sampleX = x + maxRealEnd - 1
		}
		if sampleX < 0 {
			sampleX = 0
		} else if sampleX >= dst.Width {
			sampleX = dst.Width - 1
		}
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, sampleX, y-1)
		if !ok {
			return ErrInvalidBatch
		}
		scratch.Above[frameWorkDirectionalEdgeOrigin+i] = sample
	}
	return nil
}

// frameWorkFillDirectionalLeft fills the left directional-predictor edge.
// visibleH is the past-visible-end row in the same coordinate system as y
// / dst. See frameWorkFillDirectionalAbove for the n_left_px /
// n_bottomleft_px contract this mirrors.
func frameWorkFillDirectionalLeft(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, minIndex int, maxIndex int, primaryHeight int, allowBottomLeft bool, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, visibleH int) error {
	if !frameWorkDirectionalRangeFits(minIndex, maxIndex) {
		return ErrInvalidBatch
	}
	if !block.HaveLeft {
		sample, err := frameWorkMissingLeftSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return err
		}
		for i := minIndex; i <= maxIndex; i++ {
			scratch.Left[frameWorkDirectionalEdgeOrigin+i] = sample
		}
		return nil
	}
	if x <= 0 {
		return ErrInvalidBatch
	}
	readBoundY := dst.Height
	if visibleH > 0 && visibleH < readBoundY {
		readBoundY = visibleH
	}
	// Mirror libaom's n_left_px = min(txhpx, yd + txhpx) and
	// n_bottomleft_px = min(txhpx, yd) caps.
	primaryLimit := max(readBoundY-y, 0)
	if primaryLimit > primaryHeight {
		primaryLimit = primaryHeight
	}
	if primaryLimit == 0 {
		// libaom n_left_px == 0 fallback: fill with above_ref[0] (load(x, y-1))
		// when HaveTop, else base-1=127. frameWorkMissingLeftSample implements
		// that contract.
		sample, err := frameWorkMissingLeftSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return err
		}
		for i := minIndex; i <= maxIndex; i++ {
			scratch.Left[frameWorkDirectionalEdgeOrigin+i] = sample
		}
		return nil
	}
	bottomLeftLimit := max(readBoundY-y-primaryHeight, 0)
	if bottomLeftLimit > primaryHeight {
		bottomLeftLimit = primaryHeight
	}
	// Mirror frameWorkFillDirectionalAbove's libaom three-zone layout:
	//   zone A: i < primaryLimit -> real sample (load from x-1, y+i)
	//   zone B: allowBottomLeft && i in [primaryHeight, primaryHeight+bottomLeftLimit) -> real
	//   zone C: otherwise -> replicate the last filled sample (maxRealEnd-1).
	maxRealEnd := primaryLimit
	if allowBottomLeft && bottomLeftLimit > 0 {
		maxRealEnd = primaryHeight + bottomLeftLimit
	}
	for i := minIndex; i <= maxIndex; i++ {
		var sampleY int
		switch {
		case i < primaryLimit:
			sampleY = y + i
		case allowBottomLeft && i >= primaryHeight && i < primaryHeight+bottomLeftLimit:
			sampleY = y + i
		default:
			sampleY = y + maxRealEnd - 1
		}
		if sampleY < 0 {
			sampleY = 0
		} else if sampleY >= dst.Height {
			sampleY = dst.Height - 1
		}
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, sampleY)
		if !ok {
			return ErrInvalidBatch
		}
		scratch.Left[frameWorkDirectionalEdgeOrigin+i] = sample
	}
	return nil
}

func frameWorkMissingAboveSample(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, block tile.BlockVisit) (uint16, error) {
	if block.HaveLeft {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y)
		if !ok {
			return 0, ErrInvalidBatch
		}
		return sample, nil
	}
	sample, ok := frameWorkIntraBoundaryDefault(bitDepth, -1)
	if !ok {
		return 0, ErrInvalidBatch
	}
	return sample, nil
}

func frameWorkMissingLeftSample(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, block tile.BlockVisit) (uint16, error) {
	if block.HaveTop {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x, y-1)
		if !ok {
			return 0, ErrInvalidBatch
		}
		return sample, nil
	}
	sample, ok := frameWorkIntraBoundaryDefault(bitDepth, 1)
	if !ok {
		return 0, ErrInvalidBatch
	}
	return sample, nil
}

func frameWorkMissingTopLeftSample(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, block tile.BlockVisit) (uint16, error) {
	if block.HaveLeft {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y)
		if !ok {
			return 0, ErrInvalidBatch
		}
		return sample, nil
	}
	if block.HaveTop {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x, y-1)
		if !ok {
			return 0, ErrInvalidBatch
		}
		return sample, nil
	}
	sample, ok := frameWorkIntraBoundaryDefault(bitDepth, 0)
	if !ok {
		return 0, ErrInvalidBatch
	}
	return sample, nil
}

func frameWorkIntraBoundaryDefault(bitDepth uint8, offset int) (uint16, bool) {
	max, ok := frameWorkSampleMax(bitDepth)
	if !ok {
		return 0, false
	}
	mid := int(max+1) >> 1
	value := mid + offset
	if value < 0 || value > int(max) {
		return 0, false
	}
	return uint16(value), true
}

func frameWorkDirectionalRangeFits(minIndex int, maxIndex int) bool {
	if minIndex > maxIndex {
		return false
	}
	return frameWorkDirectionalEdgeOrigin+minIndex >= 0 &&
		frameWorkDirectionalEdgeOrigin+maxIndex < frameWorkDirectionalEdgeSamples
}

func frameWorkLoadSample(plane frame.Plane, bytesPerSample int, x int, y int) (uint16, bool) {
	offset, ok := frameWorkPlaneSampleOffset(plane, bytesPerSample, x, y)
	if !ok {
		return 0, false
	}
	switch bytesPerSample {
	case 1:
		if offset >= len(plane.Pix) {
			return 0, false
		}
		return uint16(plane.Pix[offset]), true
	case 2:
		if offset+1 >= len(plane.Pix) {
			return 0, false
		}
		return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8, true
	default:
		return 0, false
	}
}

func frameWorkPlaneSampleOffset(plane frame.Plane, bytesPerSample int, x int, y int) (int, bool) {
	if x < 0 || y < 0 || x >= plane.Width || y >= plane.Height {
		return 0, false
	}
	rowOffset, ok := frameWorkCheckedMul(y, plane.Stride)
	if !ok {
		return 0, false
	}
	colOffset, ok := frameWorkCheckedMul(x, bytesPerSample)
	if !ok {
		return 0, false
	}
	offset, ok := frameWorkCheckedAdd(rowOffset, colOffset)
	if !ok || offset < 0 {
		return 0, false
	}
	return offset, true
}

func frameWorkStoreSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) bool {
	if x < 0 || y < 0 || x >= plane.Width || y >= plane.Height {
		return false
	}
	rowOffset, ok := frameWorkCheckedMul(y, plane.Stride)
	if !ok {
		return false
	}
	colOffset, ok := frameWorkCheckedMul(x, bytesPerSample)
	if !ok {
		return false
	}
	offset, ok := frameWorkCheckedAdd(rowOffset, colOffset)
	if !ok || offset < 0 {
		return false
	}
	switch bytesPerSample {
	case 1:
		if value > 0xff || offset >= len(plane.Pix) {
			return false
		}
		plane.Pix[offset] = byte(value)
	case 2:
		if offset > len(plane.Pix)-2 {
			return false
		}
		plane.Pix[offset] = byte(value)
		plane.Pix[offset+1] = byte(value >> 8)
	default:
		return false
	}
	return true
}

func frameWorkPredictChromaPalette(dst frame.Plane, bytesPerSample int, x int, y int, width int, height int, block tile.BlockVisit, color parser.ColorConfig, plane FrameWorkPlane, palette tile.PaletteModeResult, mapX int, mapY int) error {
	if palette.UVSize == 0 || palette.UVSize > tile.PaletteMaxSize || palette.UVMap == nil || (plane != FrameWorkPlaneU && plane != FrameWorkPlaneV) {
		return ErrInvalidBatch
	}
	planeSize, err := tile.PlaneBlockSize(block.Size, color, int(plane))
	if err != nil {
		return ErrInvalidBatch
	}
	dims, ok := planeSize.Dimensions()
	if !ok {
		return ErrInvalidBatch
	}
	mapStride := int(dims.W4) * 4
	mapHeight := int(dims.H4) * 4
	if mapX < 0 || mapY < 0 || mapX+width > mapStride || mapY+height > mapHeight {
		return ErrInvalidBatch
	}
	colors := palette.UColors
	if plane == FrameWorkPlaneV {
		colors = palette.VColors
	}
	for row := range height {
		for col := range width {
			index := palette.UVMap[(mapY+row)*mapStride+mapX+col]
			if index >= palette.UVSize {
				return ErrInvalidBatch
			}
			if !frameWorkStoreSample(dst, bytesPerSample, x+col, y+row, colors[index]) {
				return ErrInvalidBatch
			}
		}
	}
	return nil
}

func frameWorkPredictLumaPalette(dst frame.Plane, bytesPerSample int, x int, y int, width int, height int, block tile.BlockVisit, palette tile.PaletteModeResult, mapX int, mapY int) error {
	if palette.YSize == 0 || palette.YSize > tile.PaletteMaxSize || palette.YMap == nil {
		return ErrInvalidBatch
	}
	dims, ok := block.Size.Dimensions()
	if !ok {
		return ErrInvalidBatch
	}
	mapStride := int(dims.W4) * 4
	mapHeight := int(dims.H4) * 4
	if mapX < 0 || mapY < 0 || mapX+width > mapStride || mapY+height > mapHeight {
		return ErrInvalidBatch
	}
	for row := range height {
		for col := range width {
			index := palette.YMap[(mapY+row)*mapStride+mapX+col]
			if index >= palette.YSize {
				return ErrInvalidBatch
			}
			if !frameWorkStoreSample(dst, bytesPerSample, x+col, y+row, palette.YColors[index]) {
				return ErrInvalidBatch
			}
		}
	}
	return nil
}
