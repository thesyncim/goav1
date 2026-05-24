package threading

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

const frameWorkIntraPredictionMaxEdgeSamples = 128

// FrameWorkIntraPredictionScratch carries caller-owned edge buffers for luma
// intra prediction. Keep it outside FrameWorkTileResidualScratch so callers
// that do not use built-in prediction pay no per-worker storage cost.
type FrameWorkIntraPredictionScratch struct {
	Above [frameWorkIntraPredictionMaxEdgeSamples]uint16
	Left  [frameWorkIntraPredictionMaxEdgeSamples]uint16
}

// PredictBlockLumaIntra writes luma intra prediction pixels for one decoded
// block-loop visit into Jobs[index]'s output window. Directional intra, filter
// intra, and chroma predictors require their own edge preparation and are left
// for the caller until those paths are wired.
func (b FrameWorkBatch) PredictBlockLumaIntra(index int, visit tile.BlockLoopVisit, scratch *FrameWorkIntraPredictionScratch) error {
	if scratch == nil || !visit.Prediction.Valid || !visit.Prediction.Intra {
		return ErrInvalidBatch
	}
	mode, ok := frameWorkLumaIntraPredictionMode(visit.Prediction.LumaMode)
	if !ok {
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

	edges, err := frameWorkIntraPredictionEdges(dst, window.BytesPerSample, x, y, width, height, visit.Block, scratch)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlock(dst, window.BytesPerSample, b.Sequence.ColorConfig.BitDepth, x, y, width, height, mode, edges); err != nil {
		return ErrInvalidBatch
	}
	return nil
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
