// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// cachedJobRegionPtr is cachedJobRegionTrusted returning the cached region in
// place, or nil when the job geometry cache does not hold it.
func (b *FrameWorkBatch) cachedJobRegionPtr(index int) *FrameWorkJobRegion {
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	c := b.geomCache
	if c == nil || !ok ||
		c.validMask&frameWorkJobGeometryRegionValid == 0 ||
		c.regionIndex != cacheIndex {
		return nil
	}
	return &c.region
}

// predictBlockChromaIntraTransformPtr predicts one chroma intra transform
// block, as dav1d's recon_b_intra inner loop: the plane origin comes from the
// job's bound output plane and the window/region are read in place, so the
// per-transform work is the position, the edge flags, the edge gather and the
// predictor. It computes exactly what predictBlockChromaIntraTransformGeneric
// computes, which still serves the uncached geometry.
func (b *FrameWorkBatch) predictBlockChromaIntraTransformPtr(index int, visit *tile.BlockLoopVisit, plane FrameWorkPlane, tx tile.TransformBlock, scratch *FrameWorkIntraPredictionScratch) error {
	window := b.cachedJobOutputPlanePtr(index, plane)
	region := b.cachedJobRegionPtr(index)
	if window == nil || region == nil {
		return b.predictBlockChromaIntraTransformGeneric(index, visit, plane, tx, scratch)
	}
	blk := &visit.Block
	var t frameWorkInterTarget
	output, originX, originY, ssX, ssY, bps, ok := b.chromaIntraOrigin(index, blk, plane)
	if !ok {
		present, err := b.interTarget(index, blk, plane, &t)
		if err != nil || !present {
			return err
		}
		output, originX, originY, ssX, ssY, bps = &t.output, t.x, t.y, t.ssX, t.ssY, t.bps
	}
	if blk.MICol < region.MIColStart || blk.MIRow < region.MIRowStart ||
		blk.MIColEnd > region.MIColEnd || blk.MIRowEnd > region.MIRowEnd ||
		blk.MIColEnd <= blk.MICol || blk.MIRowEnd <= blk.MIRow {
		return ErrInvalidBatch
	}
	_, _, predWidth, predHeight, err := frameWorkTransformVisibleAndExtentPixels(tx)
	if err != nil {
		return err
	}
	baseX4 := int(blk.X4) >> int(frameWorkSubsampleShift(ssX))
	baseY4 := int(blk.Y4) >> int(frameWorkSubsampleShift(ssY))
	txX4 := int(tx.X4)
	txY4 := int(tx.Y4)
	offX4 := txX4 - baseX4
	offY4 := txY4 - baseY4
	if offX4 < 0 || offY4 < 0 {
		return ErrInvalidBatch
	}
	absX := originX + offX4*4
	absY := originY + offY4*4
	width, height, ok := frameWorkClipVisiblePixelsToWindowPtr(window, absX, absY, predWidth, predHeight)
	if !ok {
		if frameWorkPlaneBlockStartsBeyondOutput(b.Output, plane, absX, absY) {
			return nil
		}
		return ErrInvalidBatch
	}
	// frameWorkPredictionPlaneEdgeBlock, frameWorkPredictionTransformEdgeBlock
	// and frameWorkPredictionEdgeBlockForWindow, applied to the flags only.
	winX, winY := int(window.X), int(window.Y)
	haveLeft := blk.HaveLeft && originX > winX
	haveTop := blk.HaveTop && originY > winY
	if txX4 > baseX4 {
		haveLeft = true
	}
	if txY4 > baseY4 {
		haveTop = true
	}
	if absX <= winX {
		haveLeft = false
	}
	if absY <= winY {
		haveTop = false
	}
	edgeBlock := *blk
	edgeBlock.HaveLeft = haveLeft
	edgeBlock.HaveTop = haveTop

	pred := &visit.Prediction
	bitDepth := b.Sequence.ColorConfig.BitDepth
	bounded := bps == 1 && bitDepth == 8
	if pred.Palette.UVSize > 0 {
		return frameWorkPredictChromaPalette(*output, bps, absX, absY, width, height, *blk, b.Sequence.ColorConfig, plane, pred.Palette, absX-originX, absY-originY)
	}
	readW, readH := frameWorkWindowEdgeReadBoundPtr(window)
	readBoundX, readBoundY := winX+readW, winY+readH
	lumaMode, err := pred.ChromaMode.LumaMode()
	if err != nil {
		return ErrInvalidBatch
	}
	if angle, ok := frameWorkLumaIntraDirectionalAngle(lumaMode, pred.ChromaAngleDelta); ok {
		allowTopRight, allowBottomLeft := frameWorkChromaDirectionalExtendedEdges(edgeBlock, b.Sequence.SBSizeMIB, uint32(region.MIColEnd), uint32(region.MIRowEnd), originX, originY, absX, absY, predWidth, predHeight, ssX, ssY)
		edges, err := frameWorkDirectionalPredictionEdges(*output, bps, bitDepth, absX, absY, predWidth, predHeight, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, pred.ChromaIntraEdgeSmoothNeighbor, allowTopRight, allowBottomLeft, readBoundX, readBoundY)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlockWithExtentBounded(*output, bps, bitDepth, absX, absY, width, height, predWidth, predHeight, angle, edges, bounded); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	mode, ok := frameWorkLumaIntraPredictionMode(lumaMode)
	if !ok {
		return ErrInvalidBatch
	}
	if bounded && width == predWidth && height == predHeight {
		if done, err := frameWorkPredictIntra8(output, absX, absY, width, height, readBoundX, readBoundY, haveTop, haveLeft, mode, scratch); done {
			return err
		}
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(*output, bps, bitDepth, absX, absY, width, height, predWidth, predHeight, readBoundX, readBoundY, edgeBlock, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtentBounded(*output, bps, bitDepth, absX, absY, width, height, predWidth, predHeight, mode, edges, bounded); err != nil {
		return ErrInvalidBatch
	}
	return nil
}

// chromaIntraOrigin is interTarget reduced to what the chroma intra transform
// predictor reads, the job's bound output plane and the block's plane origin,
// as dav1d's recon_b_intra offsets the chroma dst from f->cur. ok is false
// when the binding is not primed or the block is not a plain in-plane chroma
// block; callers then take interTarget, which derives every rejection.
func (b *FrameWorkBatch) chromaIntraOrigin(index int, blk *tile.BlockVisit, plane FrameWorkPlane) (output *frame.Plane, x int, y int, ssX bool, ssY bool, bps int, ok bool) {
	bind := b.cachedInterBinding(index, plane, b.Output)
	color := &b.Sequence.ColorConfig
	if bind == nil || color.MonoChrome || bind.ssX != color.SubsamplingX || bind.ssY != color.SubsamplingY {
		return nil, 0, 0, false, false, 0, false
	}
	shX, shY := frameWorkSubsampleShift(bind.ssX), frameWorkSubsampleShift(bind.ssY)
	chroma, has, valid := tile.ChromaPlaneBlock(blk.Size, int(blk.X4), int(blk.Y4), uint8(shX), uint8(shY))
	dims, dimsOK := blk.Size.Dimensions()
	chromaDims, chromaOK := chroma.Dimensions()
	if !has || !valid || !dimsOK || !chromaOK ||
		!frameWorkPredictionExtentValid(int(chromaDims.W4)*4) || !frameWorkPredictionExtentValid(int(chromaDims.H4)*4) ||
		blk.VisibleW4 == 0 || blk.VisibleH4 == 0 || blk.MIColEnd <= blk.MICol || blk.MIRowEnd <= blk.MIRow {
		return nil, 0, 0, false, false, 0, false
	}
	miCol, miRow := int(blk.MICol), int(blk.MIRow)
	if shX != 0 && miCol&1 != 0 && dims.W4 == 1 {
		miCol--
	}
	if shY != 0 && miRow&1 != 0 && dims.H4 == 1 {
		miRow--
	}
	x, y = miCol*4>>shX, miRow*4>>shY
	if frameWorkPlaneBlockStartsBeyondPlane(bind.raw, bind.bps, x, y) {
		return nil, 0, 0, false, false, 0, false
	}
	return &bind.output, x, y, bind.ssX, bind.ssY, bind.bps, true
}
