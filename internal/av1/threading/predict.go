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
	frameWorkDirectionalEdgeOrigin         = 128
	frameWorkDirectionalEdgeSamples        = 512
)

// FrameWorkIntraPredictionScratch carries caller-owned edge buffers for luma
// intra prediction. Keep it outside FrameWorkTileResidualScratch so callers
// that do not use built-in prediction pay no per-worker storage cost.
type FrameWorkIntraPredictionScratch struct {
	Above [frameWorkDirectionalEdgeSamples]uint16
	Left  [frameWorkDirectionalEdgeSamples]uint16
}

// FrameWorkPredictionScratch groups caller-owned prediction scratch. Keeping it
// separate from residual scratch lets callers that do not use built-in
// prediction avoid carrying these buffers.
type FrameWorkPredictionScratch struct {
	Intra FrameWorkIntraPredictionScratch
}

// PredictBlockLuma dispatches luma prediction for one decoded block-loop visit.
// It currently covers intra and single-reference translational inter modes.
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
	return b.PredictBlockLumaInter(index, visit)
}

// PredictBlockLumaIntra writes luma intra prediction pixels for one decoded
// block-loop visit into Jobs[index]'s output window. It covers luma DC,
// vertical, horizontal, directional, Paeth, and smooth modes; filter intra and
// chroma predictors are wired separately because they use different syntax and
// plane geometry.
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

	if angle, ok := frameWorkLumaIntraDirectionalAngle(visit.Prediction.LumaMode); ok {
		edges, err := frameWorkDirectionalPredictionEdges(dst, window.BytesPerSample, x, y, width, height, angle, visit.Block, scratch)
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
	edges, err := frameWorkIntraPredictionEdges(dst, window.BytesPerSample, x, y, width, height, visit.Block, scratch)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlock(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

// PredictBlockLumaInter writes single-reference translational luma inter
// prediction for one decoded block-loop visit. Switchable filters, compound
// blending, scaled references, warped/global refinement, and chroma prediction
// are handled by later inter-prediction stages.
func (b FrameWorkBatch) PredictBlockLumaInter(index int, visit tile.BlockLoopVisit) error {
	filters, err := frameWorkMotionFilters(b.TileInfo)
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
	reference, ok := frameWorkReferenceFromTile(motionResult.References.Ref[0])
	if !ok {
		return ErrInvalidBatch
	}
	width, height, ok := frameWorkBlockVisiblePixels(visit.Block)
	if !ok {
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
	if b.Output == nil {
		return ErrInvalidBatch
	}
	output, _, _, ok := frameWorkFramePlane(b.Output, FrameWorkPlaneY)
	if !ok || b.Output.Layout.BytesPerSample <= 0 {
		return ErrInvalidBatch
	}
	refWindow, err := b.ReferencePlane(reference, FrameWorkPlaneY)
	if err != nil {
		return err
	}
	ref := frame.Plane{
		Pix:    refWindow.Pix,
		Stride: refWindow.Stride,
		Width:  refWindow.Width,
		Height: refWindow.Height,
	}
	if err := motion.PredictInterPlaneBlockWithFilterBitDepth(output, ref, b.Output.Layout.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, motionResult.MV[0], filters); err != nil {
		return ErrInvalidBatch
	}
	return nil
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

func frameWorkLumaIntraDirectionalAngle(mode tile.IntraMode) (int, bool) {
	switch mode {
	case tile.IntraModeD45:
		return 45, true
	case tile.IntraModeD135:
		return 135, true
	case tile.IntraModeD113:
		return 113, true
	case tile.IntraModeD157:
		return 157, true
	case tile.IntraModeD203:
		return 203, true
	case tile.IntraModeD67:
		return 67, true
	default:
		return 0, false
	}
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

func frameWorkIntraPredictionEdges(dst frame.Plane, bytesPerSample int, x int, y int, width int, height int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch) (prediction.IntraEdges, error) {
	var edges prediction.IntraEdges
	if block.HaveTop {
		if y <= 0 {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
		for col := 0; col < width; col++ {
			sample, ok := frameWorkLoadSample(dst, bytesPerSample, x+col, y-1)
			if !ok {
				return prediction.IntraEdges{}, ErrInvalidBatch
			}
			scratch.Above[col] = sample
		}
		edges.Above = scratch.Above[:width]
		edges.AboveAvailable = true
	}
	if block.HaveLeft {
		if x <= 0 {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
		for row := 0; row < height; row++ {
			sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y+row)
			if !ok {
				return prediction.IntraEdges{}, ErrInvalidBatch
			}
			scratch.Left[row] = sample
		}
		edges.Left = scratch.Left[:height]
		edges.LeftAvailable = true
	}
	if block.HaveTop && block.HaveLeft {
		sample, ok := frameWorkLoadSample(dst, bytesPerSample, x-1, y-1)
		if !ok {
			return prediction.IntraEdges{}, ErrInvalidBatch
		}
		edges.AboveLeft = sample
		edges.AboveLeftAvailable = true
	}
	return edges, nil
}

func frameWorkDirectionalPredictionEdges(dst frame.Plane, bytesPerSample int, x int, y int, width int, height int, angle int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch) (prediction.DirectionalEdges, error) {
	if angle <= 0 || angle >= 270 {
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
		if !block.HaveTop {
			return prediction.DirectionalEdges{}, ErrInvalidBatch
		}
		if err := frameWorkFillDirectionalAbove(dst, bytesPerSample, x, y, 0, width+height-1, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	case angle < 180:
		if !block.HaveTop || !block.HaveLeft {
			return prediction.DirectionalEdges{}, ErrInvalidBatch
		}
		if err := frameWorkFillDirectionalAbove(dst, bytesPerSample, x, y, -height, width+height, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
		if err := frameWorkFillDirectionalLeft(dst, bytesPerSample, x, y, -width, width+height, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	default:
		if !block.HaveLeft {
			return prediction.DirectionalEdges{}, ErrInvalidBatch
		}
		if err := frameWorkFillDirectionalLeft(dst, bytesPerSample, x, y, 0, width+height-1, scratch); err != nil {
			return prediction.DirectionalEdges{}, err
		}
	}
	return edges, nil
}

func frameWorkFillDirectionalAbove(dst frame.Plane, bytesPerSample int, x int, y int, minIndex int, maxIndex int, scratch *FrameWorkIntraPredictionScratch) error {
	if y <= 0 || !frameWorkDirectionalRangeFits(minIndex, maxIndex) {
		return ErrInvalidBatch
	}
	for i := minIndex; i <= maxIndex; i++ {
		sampleX := x + i
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

func frameWorkFillDirectionalLeft(dst frame.Plane, bytesPerSample int, x int, y int, minIndex int, maxIndex int, scratch *FrameWorkIntraPredictionScratch) error {
	if x <= 0 || !frameWorkDirectionalRangeFits(minIndex, maxIndex) {
		return ErrInvalidBatch
	}
	for i := minIndex; i <= maxIndex; i++ {
		sampleY := y + i
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
