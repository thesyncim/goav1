// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
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
	var t frameWorkInterTarget
	present, err := b.interTarget(index, &visit.Block, plane, &t)
	if err != nil || !present {
		return err
	}
	blk := &visit.Block
	if blk.MICol < region.MIColStart || blk.MIRow < region.MIRowStart ||
		blk.MIColEnd > region.MIColEnd || blk.MIRowEnd > region.MIRowEnd ||
		blk.MIColEnd <= blk.MICol || blk.MIRowEnd <= blk.MIRow {
		return ErrInvalidBatch
	}
	_, _, predWidth, predHeight, err := frameWorkTransformVisibleAndExtentPixels(tx)
	if err != nil {
		return err
	}
	baseX4 := int(blk.X4) >> int(frameWorkSubsampleShift(t.ssX))
	baseY4 := int(blk.Y4) >> int(frameWorkSubsampleShift(t.ssY))
	txX4 := int(tx.X4)
	txY4 := int(tx.Y4)
	offX4 := txX4 - baseX4
	offY4 := txY4 - baseY4
	if offX4 < 0 || offY4 < 0 {
		return ErrInvalidBatch
	}
	originX, originY := t.x, t.y
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
	bps := t.bps
	bitDepth := b.Sequence.ColorConfig.BitDepth
	bounded := bps == 1 && bitDepth == 8
	if pred.Palette.UVSize > 0 {
		return frameWorkPredictChromaPalette(t.output, bps, absX, absY, width, height, *blk, b.Sequence.ColorConfig, plane, pred.Palette, absX-originX, absY-originY)
	}
	readW, readH := frameWorkWindowEdgeReadBoundPtr(window)
	readBoundX, readBoundY := winX+readW, winY+readH
	if angle, ok := frameWorkChromaIntraDirectionalAngle(pred.ChromaMode, pred.ChromaAngleDelta); ok {
		allowTopRight, allowBottomLeft := frameWorkChromaDirectionalExtendedEdges(edgeBlock, b.Sequence.SBSizeMIB, uint32(region.MIColEnd), uint32(region.MIRowEnd), originX, originY, absX, absY, predWidth, predHeight, t.ssX, t.ssY)
		edges, err := frameWorkDirectionalPredictionEdges(t.output, bps, bitDepth, absX, absY, predWidth, predHeight, angle, edgeBlock, scratch, b.Sequence.EnableIntraEdgeFilter, pred.ChromaIntraEdgeSmoothNeighbor, allowTopRight, allowBottomLeft, readBoundX, readBoundY)
		if err != nil {
			return err
		}
		if err := prediction.PredictDirectionalIntraPlaneBlockWithExtentBounded(t.output, bps, bitDepth, absX, absY, width, height, predWidth, predHeight, angle, edges, bounded); err != nil {
			return ErrInvalidBatch
		}
		return nil
	}
	mode, ok := frameWorkChromaIntraPredictionMode(pred.ChromaMode)
	if !ok {
		return ErrInvalidBatch
	}
	edges, err := frameWorkIntraPredictionEdgesWithExtent(t.output, bps, bitDepth, absX, absY, width, height, predWidth, predHeight, readBoundX, readBoundY, edgeBlock, scratch, mode != prediction.IntraModeDC)
	if err != nil {
		return err
	}
	if err := prediction.PredictIntraPlaneBlockWithExtentBounded(t.output, bps, bitDepth, absX, absY, width, height, predWidth, predHeight, mode, edges, bounded); err != nil {
		return ErrInvalidBatch
	}
	return nil
}
