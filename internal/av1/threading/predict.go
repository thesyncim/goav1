package threading

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

const (
	frameWorkIntraPredictionMaxEdgeSamples = 128
	frameWorkIntraEdgeScratchSamples       = 129
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
	return b.PredictBlockLumaInter(index, visit)
}

// PredictBlockInter writes inter prediction pixels for every present plane of
// one decoded inter block. Single-reference and average/dist-wtd/wedge
// compound translation are supported; inter-intra, warped/global refinement,
// scaled references, and intrabc are handled by later stages.
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
	if !visit.Prediction.Valid || visit.Prediction.Intra || !visit.Prediction.InterMotionValid {
		return ErrInvalidBatch
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
	for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
		if err := b.predictBlockInterPlaneWithFilters(index, visit, plane, filters); err != nil {
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

	window, err := b.JobOutputPlane(index, FrameWorkPlaneY)
	if err != nil {
		return err
	}
	x, y, err := frameWorkBlockLumaPosition(visit.Block)
	if err != nil {
		return err
	}
	if !frameWorkPlaneBlockFits(window, x, y, width, height) {
		return ErrInvalidBatch
	}
	dst := frame.Plane{
		Pix:    window.Pix,
		Stride: window.Stride,
		Width:  window.Width,
		Height: window.Height,
	}
	x -= window.X
	y -= window.Y

	if visit.Prediction.FilterIntraValid {
		mode, ok := frameWorkFilterIntraPredictionMode(visit.Prediction.FilterIntraMode)
		if !ok {
			return ErrInvalidBatch
		}
		edges, err := frameWorkIntraPredictionEdges(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, visit.Block, scratch, true)
		if err != nil {
			return err
		}
		if err := prediction.PredictFilterIntraPlaneBlock(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, mode, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	if angle, ok := frameWorkLumaIntraDirectionalAngle(visit.Prediction.LumaMode, visit.Prediction.LumaAngleDelta); ok {
		edges, err := frameWorkDirectionalPredictionEdges(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, angle, visit.Block, scratch, b.Sequence.EnableIntraEdgeFilter, visit.Prediction.IntraEdgeSmoothNeighbor, true, true)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlock(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	mode, ok := frameWorkLumaIntraPredictionMode(visit.Prediction.LumaMode)
	if !ok {
		return ErrInvalidBatch
	}
	edges, err := frameWorkIntraPredictionEdges(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, visit.Block, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlock(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockLumaIntraTransform(index int, visit tile.BlockLoopVisit, tx tile.TransformBlock, scratch *FrameWorkIntraPredictionScratch) error {
	width, height, predWidth, predHeight, err := frameWorkTransformVisibleAndExtentPixels(tx)
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
	if !frameWorkPlaneBlockFits(window, absX, absY, width, height) {
		return ErrInvalidBatch
	}
	dst := frame.Plane{
		Pix:    window.Pix,
		Stride: window.Stride,
		Width:  window.Width,
		Height: window.Height,
	}
	x := absX - window.X
	y := absY - window.Y
	edgeBlock := frameWorkPredictionTransformEdgeBlock(visit.Block, visit.Block.X4, visit.Block.Y4, tx.X4, tx.Y4)
	edgeBlock = frameWorkPredictionEdgeBlockForWindow(edgeBlock, absX, absY, window)

	if visit.Prediction.FilterIntraValid {
		mode, ok := frameWorkFilterIntraPredictionMode(visit.Prediction.FilterIntraMode)
		if !ok {
			return ErrInvalidBatch
		}
		edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, edgeBlock, scratch, true)
		if err != nil {
			return err
		}
		if err := prediction.PredictFilterIntraPlaneBlockWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, mode, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	if angle, ok := frameWorkLumaIntraDirectionalAngle(visit.Prediction.LumaMode, visit.Prediction.LumaAngleDelta); ok {
		allowTopRight, allowBottomLeft := frameWorkLumaTransformDirectionalExtendedEdges(visit.Block, b.Sequence.SBSizeMIB, absX, absY, predWidth, predHeight)
		edges, err := frameWorkDirectionalPredictionEdges(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, predWidth, predHeight, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, visit.Prediction.IntraEdgeSmoothNeighbor, allowTopRight, allowBottomLeft)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlock(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}

	mode, ok := frameWorkLumaIntraPredictionMode(visit.Prediction.LumaMode)
	if !ok {
		return ErrInvalidBatch
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, edgeBlock, scratch, mode != prediction.IntraModeDC)
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
	predWidth, predHeight, err := frameWorkBlockPlanePredictionExtentPixels(visit.Block, b.Sequence.ColorConfig, plane)
	if err != nil {
		return err
	}
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	if angle, ok := frameWorkChromaIntraDirectionalAngle(visit.Prediction.ChromaMode, visit.Prediction.ChromaAngleDelta); ok {
		edges, err := frameWorkDirectionalPredictionEdges(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, false, true, true)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlock(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	mode, ok := frameWorkChromaIntraPredictionMode(visit.Prediction.ChromaMode)
	if !ok {
		return ErrInvalidBatch
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, predWidth, predHeight, edgeBlock, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, predWidth, predHeight, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockChromaIntraTransform(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, tx tile.TransformBlock, scratch *FrameWorkIntraPredictionScratch) error {
	geom, present, err := b.blockPredictionPlaneGeometry(index, visit.Block, plane)
	if err != nil || !present {
		return err
	}
	width, height, predWidth, predHeight, err := frameWorkTransformVisibleAndExtentPixels(tx)
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
	if !frameWorkPlaneBlockFits(geom.Window, absX, absY, width, height) {
		return ErrInvalidBatch
	}
	x := absX
	y := absY
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	edgeBlock = frameWorkPredictionTransformEdgeBlock(edgeBlock, baseX4, baseY4, tx.X4, tx.Y4)
	edgeBlock = frameWorkPredictionEdgeBlockForWindow(edgeBlock, absX, absY, geom.Window)

	if angle, ok := frameWorkChromaIntraDirectionalAngle(visit.Prediction.ChromaMode, visit.Prediction.ChromaAngleDelta); ok {
		edges, err := frameWorkDirectionalPredictionEdges(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, predWidth, predHeight, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, false, true, true)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlock(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, angle, edges); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	mode, ok := frameWorkChromaIntraPredictionMode(visit.Prediction.ChromaMode)
	if !ok {
		return ErrInvalidBatch
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, predWidth, predHeight, edgeBlock, scratch, mode != prediction.IntraModeDC)
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
	luma := b.Output.Y
	lumaX := geom.X
	lumaY := geom.Y
	lumaW := geom.Width
	lumaH := geom.Height
	if geom.SubsamplingX {
		lumaX <<= 1
		lumaW <<= 1
	}
	if geom.SubsamplingY {
		lumaY <<= 1
		lumaH <<= 1
	}
	if err := frameWorkSubsampleLumaCFLQ3(scratch.ReconQ3[:], luma, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, lumaX, lumaY, lumaW, lumaH, geom.SubsamplingX, geom.SubsamplingY); err != nil {
		return err
	}
	if err := prediction.SubtractCFLAverage(scratch.ReconQ3[:], scratch.ACQ3[:], geom.Width, geom.Height); err != nil {
		return ErrInvalidBatch
	}
	edgeBlock := frameWorkPredictionPlaneEdgeBlock(visit.Block, geom)
	edges, err := frameWorkIntraPredictionEdges(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, edgeBlock, &scratch.Intra, false)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlock(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, prediction.IntraModeDC, edges); err != nil {
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
	if err := prediction.PredictCFLPlaneBlock(geom.Output, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, geom.Width, geom.Height, scratch.ACQ3[:], alphaQ3); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

// PredictBlockLumaInter writes single-reference translational luma inter
// prediction for one decoded block-loop visit. Switchable filters, compound
// blending, scaled references, warped/global refinement, and chroma prediction
// are handled by later inter-prediction stages.
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
	if !visit.Prediction.Valid || visit.Prediction.Intra || !visit.Prediction.InterMotionValid {
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

// PredictBlockLumaInterCompoundAverage writes average compound luma inter
// prediction for one decoded block-loop visit. Inter-intra, scaled references,
// and warped/global refinement are handled by later inter-prediction stages.
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
// warped/global refinement are handled by later stages.
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
	if visit.Prediction.MotionModeValid && visit.Prediction.MotionMode != tile.MotionModeTranslation {
		return ErrInvalidBatch
	}
	motionResult := visit.Prediction.InterMotion
	if motionResult.References.Compound ||
		!motionResult.References.Ref[0].Valid() ||
		motionResult.References.Ref[1] != tile.ReferenceFrameNone {
		return ErrInvalidBatch
	}
	return b.predictBlockInterReferencePlaneToOutput(index, visit.Block, plane, motionResult.References.Ref[0], motionResult.MV[0], filters)
}

func (b FrameWorkBatch) predictBlockInterCompoundPlaneWithFilters(index int, visit tile.BlockLoopVisit, plane FrameWorkPlane, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) error {
	if scratch == nil {
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
	first, err := frameWorkInterScratchPlane(scratch.First[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		return err
	}
	second, err := frameWorkInterScratchPlane(scratch.Second[:], geom.BytesPerSample, geom.Width, geom.Height)
	if err != nil {
		return err
	}
	if err := b.predictBlockInterReferencePlaneToScratch(first, plane, motionResult.References.Ref[0], motionResult.MV[0], geom, filters); err != nil {
		return err
	}
	if err := b.predictBlockInterReferencePlaneToScratch(second, plane, motionResult.References.Ref[1], motionResult.MV[1], geom, filters); err != nil {
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
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(geom.X, geom.Y, mv, geom.SubsamplingX, geom.SubsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(geom.Output, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, geom.X, geom.Y, refX, refY, geom.Width, geom.Height, subX, subY, filters); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

func (b FrameWorkBatch) predictBlockInterReferencePlaneToScratch(dst frame.Plane, plane FrameWorkPlane, refFrame tile.ReferenceFrame, mv motion.Vector, geom frameWorkPredictionPlaneGeometry, filters motion.InterpFilters) error {
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
	refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(geom.X, geom.Y, mv, geom.SubsamplingX, geom.SubsamplingY)
	if err != nil {
		return ErrInvalidBatch
	}
	if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dst, ref, geom.BytesPerSample, b.Sequence.ColorConfig.BitDepth, 0, 0, refX, refY, geom.Width, geom.Height, subX, subY, filters); err != nil {
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
	if !frameWorkPlaneBlockFits(window, x, y, width, height) {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	if b.Output == nil {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	output, outputSubX, outputSubY, ok := frameWorkFramePlane(b.Output, plane)
	if !ok || b.Output.Layout.BytesPerSample <= 0 {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	if outputSubX != subsamplingX || outputSubY != subsamplingY {
		return frameWorkPredictionPlaneGeometry{}, false, ErrInvalidBatch
	}
	return frameWorkPredictionPlaneGeometry{
		Output:         output,
		Window:         window,
		X:              x,
		Y:              y,
		Width:          width,
		Height:         height,
		SubsamplingX:   subsamplingX,
		SubsamplingY:   subsamplingY,
		BytesPerSample: b.Output.Layout.BytesPerSample,
	}, true, nil
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

func frameWorkLumaTransformDirectionalExtendedEdges(block tile.BlockVisit, sbSizeMIB uint8, absX int, absY int, width int, height int) (allowTopRight bool, allowBottomLeft bool) {
	blockX := int(block.MICol) * 4
	blockY := int(block.MIRow) * 4
	blockW := int(block.VisibleW4) * 4
	colOff := absX - blockX
	rowOff := absY - blockY
	allowTopRight = colOff+width < blockW
	allowBottomLeft = frameWorkLumaTransformHasBottomLeft(block, sbSizeMIB, colOff, rowOff, height)
	return allowTopRight, allowBottomLeft
}

func frameWorkLumaTransformHasBottomLeft(block tile.BlockVisit, sbSizeMIB uint8, colOffPx int, rowOffPx int, height int) bool {
	if !block.HaveLeft || sbSizeMIB == 0 || colOffPx < 0 || rowOffPx < 0 || height <= 0 ||
		colOffPx%4 != 0 || rowOffPx%4 != 0 || height%4 != 0 {
		return false
	}
	dims, ok := block.Size.Dimensions()
	if !ok {
		return false
	}
	colOff := colOffPx >> 2
	rowOff := rowOffPx >> 2
	txH := height >> 2
	if colOff > 0 {
		return false
	}
	blockH := int(dims.H4)
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
		rowOffInSB := (blkRowInSB << bhLog2) + rowOff
		return rowOffInSB+txH < sb
	}
	if ((blkRowInSB + 1) << bhLog2) >= sb {
		return false
	}
	table := frameWorkBottomLeftAvailabilityTable(block.Size)
	if len(table) == 0 {
		return false
	}
	thisBlockIndex := blkRowInSB<<(frameWorkMaxMIBSizeLog2-bwLog2) + blkColInSB
	idx1 := thisBlockIndex >> 3
	idx2 := thisBlockIndex & 7
	return idx1 >= 0 && idx1 < len(table) && ((table[idx1]>>idx2)&1) != 0
}

func frameWorkBottomLeftAvailabilityTable(size tile.BlockSize) []uint8 {
	switch size {
	case tile.BlockSize4x4:
		return frameWorkHasBottomLeft4x4[:]
	case tile.BlockSize4x8:
		return frameWorkHasBottomLeft4x8[:]
	case tile.BlockSize8x4:
		return frameWorkHasBottomLeft8x4[:]
	case tile.BlockSize8x8:
		return frameWorkHasBottomLeft8x8[:]
	case tile.BlockSize4x16:
		return frameWorkHasBottomLeft4x16[:]
	case tile.BlockSize16x4:
		return frameWorkHasBottomLeft16x4[:]
	default:
		return nil
	}
}

var (
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
	frameWorkHasBottomLeft4x16 = [...]uint8{
		0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
		0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
	}
	frameWorkHasBottomLeft16x4 = [...]uint8{
		254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
		254, 254, 254, 84, 254, 254, 254, 16, 254, 254, 254, 84, 254, 254, 254, 0,
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

func frameWorkAverageCompoundBlock(dst frame.Plane, first frame.Plane, second frame.Plane, bytesPerSample int, dstX int, dstY int, width int, height int) error {
	return frameWorkBlendCompoundBlock(dst, first, second, bytesPerSample, dstX, dstY, width, height, 8, 8)
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
		for row := 0; row < height; row++ {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX : (dstY+row)*dst.Stride+dstX+width]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width]
			for col := 0; col < width; col++ {
				out := (uint32(firstLine[col])*uint32(fwdOffset) + uint32(secondLine[col])*uint32(bckOffset) + 1<<(frameWorkDistPrecisionBits-1)) >> frameWorkDistPrecisionBits
				dstLine[col] = byte(out)
			}
		}
	case 2:
		for row := 0; row < height; row++ {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX*2 : (dstY+row)*dst.Stride+dstX*2+width*2]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width*2]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width*2]
			for col := 0; col < width; col++ {
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
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
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
			m := frameWorkDiffWtdMaskBase + diff/frameWorkDiffWtdFactor
			if m > frameWorkBlendA64MaxAlpha {
				m = frameWorkBlendA64MaxAlpha
			}
			if invert {
				m = frameWorkBlendA64MaxAlpha - m
			}
			mask[row*maskStride+col] = byte(m)
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
		for row := 0; row < height; row++ {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX : (dstY+row)*dst.Stride+dstX+width]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width]
			for col := 0; col < width; col++ {
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
		for row := 0; row < height; row++ {
			dstLine := dst.Pix[(dstY+row)*dst.Stride+dstX*2 : (dstY+row)*dst.Stride+dstX*2+width*2]
			firstLine := first.Pix[row*first.Stride : row*first.Stride+width*2]
			secondLine := second.Pix[row*second.Stride : row*second.Stride+width*2]
			for col := 0; col < width; col++ {
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
	return frameWorkIntraPredictionEdgesWithExtent(dst, bytesPerSample, bitDepth, x, y, width, height, width, height, block, scratch, fillMissing)
}

func frameWorkIntraPredictionEdgesWithExtent(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, edgeWidth int, edgeHeight int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, fillMissing bool) (prediction.IntraEdges, error) {
	if scratch == nil {
		return prediction.IntraEdges{}, ErrInvalidBatch
	}
	if edgeWidth < width || edgeHeight < height ||
		edgeWidth > frameWorkIntraPredictionMaxEdgeSamples ||
		edgeHeight > frameWorkIntraPredictionMaxEdgeSamples {
		return prediction.IntraEdges{}, ErrInvalidBatch
	}
	var edges prediction.IntraEdges
	if block.HaveTop {
		if y <= 0 {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
		available := edgeWidth
		if x+available > dst.Width {
			available = dst.Width - x
		}
		if available <= 0 {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
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
	} else if fillMissing {
		sample, err := frameWorkMissingAboveSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return prediction.IntraEdges{}, err
		}
		for col := 0; col < edgeWidth; col++ {
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
		if y+available > dst.Height {
			available = dst.Height - y
		}
		if available <= 0 {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
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
	} else if fillMissing {
		sample, err := frameWorkMissingLeftSample(dst, bytesPerSample, bitDepth, x, y, block)
		if err != nil {
			return prediction.IntraEdges{}, err
		}
		for row := 0; row < edgeHeight; row++ {
			scratch.Left[row] = sample
		}
		edges.Left = scratch.Left[:edgeHeight]
		edges.LeftAvailable = true
	}
	if block.HaveTop && block.HaveLeft {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y-1)
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

func frameWorkDirectionalPredictionEdges(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, width int, height int, angle int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, enableIntraEdgeFilter bool, smoothNeighbor bool, allowTopRight bool, allowBottomLeft bool) (prediction.DirectionalEdges, error) {
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
		if err := frameWorkFillDirectionalAbove(dst, bytesPerSample, bitDepth, x, y, 0, width+height-1, width, allowTopRight, block, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	case angle < 180:
		if err := frameWorkFillDirectionalAbove(dst, bytesPerSample, bitDepth, x, y, -height, width+height, width, false, block, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
		if err := frameWorkFillDirectionalLeft(dst, bytesPerSample, bitDepth, x, y, -width, width+height, height, false, block, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	default:
		if err := frameWorkFillDirectionalLeft(dst, bytesPerSample, bitDepth, x, y, 0, width+height-1, height, allowBottomLeft, block, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	}
	if angle != 90 && angle != 180 {
		topLeft, err := frameWorkDirectionalAboveLeftSample(dst, bytesPerSample, bitDepth, x, y, block)
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

func frameWorkDirectionalAboveLeftSample(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, block tile.BlockVisit) (uint16, error) {
	if block.HaveTop && block.HaveLeft {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y-1)
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

func frameWorkFillDirectionalAbove(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, minIndex int, maxIndex int, primaryWidth int, allowTopRight bool, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch) error {
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
	for i := minIndex; i <= maxIndex; i++ {
		sampleX := x + i
		if !allowTopRight && i >= primaryWidth {
			sampleX = x + primaryWidth - 1
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

func frameWorkFillDirectionalLeft(dst frame.Plane, bytesPerSample int, bitDepth uint8, x int, y int, minIndex int, maxIndex int, primaryHeight int, allowBottomLeft bool, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch) error {
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
	for i := minIndex; i <= maxIndex; i++ {
		sampleY := y + i
		if !allowBottomLeft && i >= primaryHeight {
			sampleY = y + primaryHeight - 1
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
	switch bytesPerSample {
	case 1:
		if offset >= len(plane.Pix) {
			return 0, false
		}
		return uint16(plane.Pix[offset]), true
	case 2:
		if offset > len(plane.Pix)-2 {
			return 0, false
		}
		return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8, true
	default:
		return 0, false
	}
}
