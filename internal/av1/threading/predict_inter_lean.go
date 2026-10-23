// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// frameWorkInterTarget is the subset of frameWorkPredictionPlaneGeometry that
// the same-size translational predictors read. The hot single-reference and
// compound paths fill it in place instead of building, returning and copying
// the full geometry struct (and re-deriving the packed filter extent and coded
// dimensions from it) for every plane of every block.
type frameWorkInterTarget struct {
	output frame.Plane

	x      int
	y      int
	width  int // frame-edge-clipped visible extent (geom.Width/Height)
	height int
	writeW int // clipped full plane-block extent (geom.WriteWidth/Height)
	writeH int
	// filterW / filterH are the un-clipped plane block dimensions that select
	// the interpolation kernel (geom.filterWidth/filterHeight).
	filterW int
	filterH int
	codedW  int
	codedH  int
	bps     int
	ssX     bool
	ssY     bool
}

// interTarget derives exactly what blockPredictionPlaneGeometry derives for
// (index, block, plane), with the same (ok, err) results: ok is false with a
// nil error for an absent plane or a block that starts beyond the output. The
// job window is read in place from the job geometry cache when it is primed.
func (b *FrameWorkBatch) interTarget(index int, block *tile.BlockVisit, plane FrameWorkPlane, t *frameWorkInterTarget) (bool, error) {
	color := &b.Sequence.ColorConfig
	x, y, width, height, ssX, ssY, ok, err := frameWorkBlockPlanePositionPtr(block, color, plane)
	if err != nil || !ok {
		return ok, err
	}
	return b.interTargetAt(index, block, plane, x, y, width, height, ssX, ssY, t)
}

// interTargetAt is interTarget after the plane position step, so the two
// chroma planes of a block can share one position, as dav1d's recon_b_inter
// derives the chroma block geometry once for both.
func (b *FrameWorkBatch) interTargetAt(index int, block *tile.BlockVisit, plane FrameWorkPlane, x, y, width, height int, ssX, ssY bool, t *frameWorkInterTarget) (bool, error) {
	color := &b.Sequence.ColorConfig
	window := b.cachedJobOutputPlanePtr(index, plane)
	if window == nil {
		w, err := b.JobOutputPlane(index, plane)
		if err != nil {
			return false, err
		}
		window = &w
	}
	out := b.Output
	if out == nil {
		return false, ErrInvalidBatch
	}
	if bind := b.cachedInterBinding(index, plane, out); bind != nil {
		return b.interTargetBound(block, plane, window, bind, x, y, width, height, ssX, ssY, t)
	}
	output, outputSubX, outputSubY, planeOK := frameWorkFramePlane(out, plane)
	bps := out.Layout.BytesPerSample
	if planeOK && frameWorkPlaneBlockStartsBeyondPlane(output, bps, x, y) {
		return false, nil
	}
	width, height, ok := frameWorkClipVisiblePixelsToWindowPtr(window, x, y, width, height)
	if !ok {
		return false, ErrInvalidBatch
	}
	writeW, writeH := width, height
	fullW, fullH, err := frameWorkBlockPlanePredictionExtentPixelsPtr(block, color, plane)
	if err != nil {
		return false, err
	}
	if cw, ch, ok := frameWorkClipVisiblePixelsToWindowPtr(window, x, y, fullW, fullH); ok {
		writeW, writeH = cw, ch
	}
	if !frameWorkPredictionExtentValid(fullW) || !frameWorkPredictionExtentValid(fullH) {
		return false, ErrInvalidBatch
	}
	if !planeOK || bps <= 0 || bps > int(^uint8(0)) || outputSubX != ssX || outputSubY != ssY {
		return false, ErrInvalidBatch
	}
	if _, ok := frameWorkPredictionIntToUint32(output.Width); !ok {
		return false, ErrInvalidBatch
	}
	if _, ok := frameWorkPredictionIntToUint32(output.Height); !ok {
		return false, ErrInvalidBatch
	}
	if width <= 0 || width > 128 || height <= 0 || height > 128 ||
		writeW <= 0 || writeW > 128 || writeH <= 0 || writeH > 128 {
		return false, ErrInvalidBatch
	}
	t.codedW = output.Width
	t.codedH = output.Height
	t.output = frameWorkExtendPlaneToClipPtr(output, window, bps)
	b.storeInterBinding(index, plane, out, output, t.output, bps, ssX, ssY)
	t.x = x
	t.y = y
	t.width = width
	t.height = height
	t.writeW = writeW
	t.writeH = writeH
	t.filterW = fullW
	t.filterH = fullH
	t.bps = bps
	t.ssX = ssX
	t.ssY = ssY
	return true, nil
}

// cachedJobOutputPlanePtr is cachedJobOutputPlaneTrusted returning the cached
// window in place.
func (b *FrameWorkBatch) cachedJobOutputPlanePtr(index int, plane FrameWorkPlane) *FrameWorkPlaneRegion {
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	c := b.geomCache
	if c == nil || !ok || plane > FrameWorkPlaneV {
		return nil
	}
	if c.validMask&frameWorkJobGeometryPlaneMask(plane) == 0 || c.planeIndex[plane] != cacheIndex {
		return nil
	}
	return &c.plane[plane]
}

// frameWorkPlaneBlockStartsBeyondPlane is frameWorkPlaneBlockStartsBeyondOutput
// for an already-selected plane, without the two divisions: x is past the
// aligned width max(W, Stride/bps) exactly when x >= W and bps*(x+1) > Stride,
// and y is past max(H, len/Stride) exactly when y >= H and Stride*(y+1) > len.
func frameWorkPlaneBlockStartsBeyondPlane(dst frame.Plane, bytesPerSample int, x int, y int) bool {
	if x < 0 || y < 0 {
		return false
	}
	if bytesPerSample <= 0 || dst.Stride <= 0 {
		return x >= dst.Width || y >= dst.Height
	}
	return (x >= dst.Width && bytesPerSample*(x+1) > dst.Stride) ||
		(y >= dst.Height && dst.Stride*(y+1) > len(dst.Pix))
}

// frameWorkPredictionExtentValid reports whether frameWorkPredictionBlockExtentCode
// accepts v.
func frameWorkPredictionExtentValid(v int) bool {
	return v >= 4 && v <= 128 && v&(v-1) == 0
}

// referencePlaneViewInto is referencePlaneView for the lean inter routes: the
// same validation and the same trimmed view, written in place and reported as
// a bool (the routes fall back to the general path, which owns the errors),
// with overflow checks that avoid integer division.
func (b *FrameWorkBatch) referencePlaneViewInto(reference FrameWorkReference, plane FrameWorkPlane, ref *frame.Plane) bool {
	index := int(reference)
	if index >= parser.InterRefsPerFrame || index >= len(b.References) {
		return false
	}
	refFrame := b.References[index]
	if refFrame == nil {
		return false
	}
	var p *frame.Plane
	switch plane {
	case FrameWorkPlaneY:
		p = &refFrame.Y
	case FrameWorkPlaneU:
		if refFrame.Format.MonoChrome {
			return false
		}
		p = &refFrame.U
	case FrameWorkPlaneV:
		if refFrame.Format.MonoChrome {
			return false
		}
		p = &refFrame.V
	default:
		return false
	}
	bytesPerSample := refFrame.Layout.BytesPerSample
	width, height, stride := p.Width, p.Height, p.Stride
	if bytesPerSample <= 0 || bytesPerSample > int(^uint8(0)) ||
		stride <= 0 || uint64(stride) > uint64(^uint32(0)) ||
		width <= 0 || height <= 0 ||
		uint64(width) > uint64(^uint32(0)) || uint64(height) > uint64(^uint32(0)) {
		return false
	}
	// width < 2^32 and bytesPerSample < 2^8: no overflow.
	rowBytes := width * bytesPerSample
	if rowBytes > stride {
		return false
	}
	hi, lo := bits.Mul64(uint64(height-1), uint64(stride))
	if hi != 0 || lo > uint64(^uint(0)>>1)-uint64(rowBytes) {
		return false
	}
	end := int(lo) + rowBytes
	if end > len(p.Pix) {
		return false
	}
	*ref = frame.Plane{Pix: p.Pix[:end], Stride: stride, Width: width, Height: height}
	return true
}

// cachedInterBinding returns the job's bound output plane for plane when the
// geometry cache holds one for index and out, or nil.
func (b *FrameWorkBatch) cachedInterBinding(index int, plane FrameWorkPlane, out *frame.Frame) *frameWorkInterBinding {
	c := b.geomCache
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	if c == nil || !ok || plane > FrameWorkPlaneV || c.interOut != out ||
		c.interMask&(1<<uint8(plane)) == 0 || c.planeIndex[plane] != cacheIndex ||
		c.validMask&frameWorkJobGeometryPlaneMask(plane) == 0 {
		return nil
	}
	return &c.inter[plane]
}

// storeInterBinding records the job-invariant output binding derived by a
// full interTarget, so later blocks of the job skip deriving it again.
func (b *FrameWorkBatch) storeInterBinding(index int, plane FrameWorkPlane, out *frame.Frame, raw, output frame.Plane, bps int, ssX, ssY bool) {
	c := b.geomCache
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	if c == nil || !ok || plane > FrameWorkPlaneV ||
		c.validMask&frameWorkJobGeometryPlaneMask(plane) == 0 || c.planeIndex[plane] != cacheIndex {
		return
	}
	if c.interOut != out {
		c.interOut = out
		c.interMask = 0
	}
	c.inter[plane] = frameWorkInterBinding{raw: raw, output: output, codedW: raw.Width, codedH: raw.Height, bps: bps, ssX: ssX, ssY: ssY}
	c.interMask |= 1 << uint8(plane)
}

// interTargetBound is interTarget for a block of a job whose output binding is
// already validated: it repeats only the per-block position, clip and extent
// steps.
func (b *FrameWorkBatch) interTargetBound(block *tile.BlockVisit, plane FrameWorkPlane, window *FrameWorkPlaneRegion, bind *frameWorkInterBinding, x, y, width, height int, ssX, ssY bool, t *frameWorkInterTarget) (bool, error) {
	if bind.ssX != ssX || bind.ssY != ssY {
		return false, ErrInvalidBatch
	}
	if frameWorkPlaneBlockStartsBeyondPlane(bind.raw, bind.bps, x, y) {
		return false, nil
	}
	fullW, fullH, err := frameWorkBlockPlanePredictionExtentPixelsPtr(block, &b.Sequence.ColorConfig, plane)
	if err != nil {
		return false, err
	}
	// dav1d's mc() writes the whole block at its plane position; a block
	// whose full extent lies inside the window's writable extent needs no
	// clipping, so its visible and write extents are known directly.
	clipW, clipH := int(window.ClipWidth), int(window.ClipHeight)
	if clipW <= 0 {
		clipW = int(window.Width)
	}
	if clipH <= 0 {
		clipH = int(window.Height)
	}
	if rx, ry := x-int(window.X), y-int(window.Y); rx >= 0 && ry >= 0 && rx+fullW <= clipW && ry+fullH <= clipH &&
		width > 0 && height > 0 && width <= fullW && height <= fullH &&
		frameWorkPredictionExtentValid(fullW) && frameWorkPredictionExtentValid(fullH) {
		*t = frameWorkInterTarget{
			output: bind.output,
			x:      x, y: y,
			width: width, height: height,
			writeW: fullW, writeH: fullH,
			filterW: fullW, filterH: fullH,
			codedW: bind.codedW, codedH: bind.codedH,
			bps: bind.bps, ssX: ssX, ssY: ssY,
		}
		return true, nil
	}
	width, height, ok := frameWorkClipVisiblePixelsToWindowPtr(window, x, y, width, height)
	if !ok {
		return false, ErrInvalidBatch
	}
	writeW, writeH := width, height
	if cw, ch, ok := frameWorkClipVisiblePixelsToWindowPtr(window, x, y, fullW, fullH); ok {
		writeW, writeH = cw, ch
	}
	if !frameWorkPredictionExtentValid(fullW) || !frameWorkPredictionExtentValid(fullH) ||
		width > 128 || height > 128 || writeW <= 0 || writeW > 128 || writeH <= 0 || writeH > 128 {
		return false, ErrInvalidBatch
	}
	*t = frameWorkInterTarget{
		output: bind.output,
		x:      x, y: y,
		width: width, height: height,
		writeW: writeW, writeH: writeH,
		filterW: fullW, filterH: fullH,
		codedW: bind.codedW, codedH: bind.codedH,
		bps: bind.bps, ssX: ssX, ssY: ssY,
	}
	return true, nil
}

// chromaIntraGeometry returns the plane geometry the chroma intra transform
// predictor reads (Output, Window, X, Y, BytesPerSample and subsampling). It
// derives them from the job's bound output plane, as dav1d's recon_b_intra
// offsets from f->cur, instead of recomputing the full validated geometry for
// every transform. Without a cached job window it takes the full path.
func (b *FrameWorkBatch) chromaIntraGeometry(index int, block *tile.BlockVisit, plane FrameWorkPlane) (frameWorkPredictionPlaneGeometry, bool, error) {
	window := b.cachedJobOutputPlanePtr(index, plane)
	if window == nil {
		return b.blockPredictionPlaneGeometry(index, *block, plane)
	}
	var t frameWorkInterTarget
	present, err := b.interTarget(index, block, plane, &t)
	if err != nil || !present {
		return frameWorkPredictionPlaneGeometry{}, present, err
	}
	return frameWorkPredictionPlaneGeometry{
		Output:         t.output,
		Window:         *window,
		X:              t.x,
		Y:              t.y,
		BytesPerSample: uint8(t.bps),
		SubsamplingX:   t.ssX,
		SubsamplingY:   t.ssY,
	}, true, nil
}

// boundRefPlane is dav1d's f->refp[refidx].p binding: the validated view
// referencePlaneViewInto derives for (reference, plane), computed once per job
// and then read in place. It returns nil when the view is invalid or no job
// geometry cache is installed; callers then take referencePlaneViewInto.
func (b *FrameWorkBatch) boundRefPlane(reference FrameWorkReference, plane FrameWorkPlane) *frame.Plane {
	c := b.geomCache
	i := uint(reference)*3 + uint(plane)
	if c == nil || plane > FrameWorkPlaneV || i >= uint(len(c.ref)) {
		return nil
	}
	bit := uint32(1) << i
	if c.refMask&bit != 0 {
		return &c.ref[i]
	}
	if !b.referencePlaneViewInto(reference, plane, &c.ref[i]) {
		return nil
	}
	c.refMask |= bit
	return &c.ref[i]
}

// sameSizeRefPlane returns the bound reference view when it has the output
// plane's coded size (the lean unscaled routes), else nil.
func (b *FrameWorkBatch) sameSizeRefPlane(reference FrameWorkReference, plane FrameWorkPlane, codedW, codedH int, tmp *frame.Plane) *frame.Plane {
	ref := b.boundRefPlane(reference, plane)
	if ref == nil {
		if !b.referencePlaneViewInto(reference, plane, tmp) {
			return nil
		}
		ref = tmp
	}
	if ref.Width != codedW || ref.Height != codedH || !b.referenceSameSize(reference) {
		// dav1d takes the unscaled mc() path only for a reference of the
		// current luma frame size; odd sizes can match in chroma alone.
		return nil
	}
	return ref
}

// predictBlockInterLean8 is dav1d's recon_b_inter prediction (src/recon_tmpl.c)
// for the common 8-bit translational block, single-reference or compound: the
// block-level mode checks run once, the luma plane predicts through mc(), and
// U and V share one chroma position as dav1d's cbw4/cbh4/cbx4/cby4. Any plane
// the lean target cannot take (scaled reference, high bit depth layout) goes
// through the general per-plane route, which owns those semantics. It reports
// false, with nothing written, for a block outside the translational shapes.
func (b *FrameWorkBatch) predictBlockInterLean8(index int, visit *tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) (bool, error) {
	pred := &visit.Prediction
	refs := &pred.InterMotion.References
	if b.Sequence.ColorConfig.BitDepth != 8 ||
		(pred.InterIntraValid && pred.InterIntra.Enabled) ||
		(pred.MotionModeValid && pred.MotionMode != tile.MotionModeTranslation) {
		return false, nil
	}
	if refs.Compound {
		blend := pred.CompoundBlend
		if scratch == nil || !refs.Ref[0].Valid() || !refs.Ref[1].Valid() ||
			(blend.Type != tile.CompoundTypeAverage && blend.Type != tile.CompoundTypeDistWtd &&
				blend.Type != tile.CompoundTypeWedge && blend.Type != tile.CompoundTypeDiffWtd) {
			return false, nil
		}
		return true, b.predictBlockInterLeanPlanes(index, visit, scratch, filters, true)
	}
	if pred.GlobalWarpedMotionValid || !refs.Ref[0].Valid() || refs.Ref[1] != tile.ReferenceFrameNone {
		return false, nil
	}
	if _, ok := frameWorkReferenceFromTile(refs.Ref[0]); !ok {
		return false, nil
	}
	return true, b.predictBlockInterLeanPlanes(index, visit, scratch, filters, false)
}

func (b *FrameWorkBatch) predictBlockInterLeanPlanes(index int, visit *tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters, compound bool) error {
	block := &visit.Block
	color := &b.Sequence.ColorConfig
	var cx, cy, cw, ch int
	var cssX, cssY bool
	for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
		if !compound && plane != FrameWorkPlaneY && visit.Prediction.SubChromaInterValid {
			if err := b.predictBlockInterSubChromaPlanePtr(index, visit, plane, scratch); err != nil {
				return err
			}
			continue
		}
		var x, y, w, h int
		var ssX, ssY bool
		switch plane {
		case FrameWorkPlaneY, FrameWorkPlaneU:
			var ok bool
			var err error
			x, y, w, h, ssX, ssY, ok, err = frameWorkBlockPlanePositionPtr(block, color, plane)
			if err != nil {
				return err
			}
			if !ok {
				if plane == FrameWorkPlaneU {
					// No chroma for this block (or monochrome): V has none either.
					return nil
				}
				continue
			}
			if plane == FrameWorkPlaneU {
				cx, cy, cw, ch, cssX, cssY = x, y, w, h, ssX, ssY
			}
		default:
			x, y, w, h, ssX, ssY = cx, cy, cw, ch, cssX, cssY
		}
		var t frameWorkInterTarget
		ok, err := b.interTargetAt(index, block, plane, x, y, w, h, ssX, ssY, &t)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		if compound {
			handled, err := b.predictBlockInterCompoundLean(visit, plane, scratch, &t, visit.Prediction.CompoundBlend, filters)
			if !handled {
				err = b.predictBlockInterCompoundPlaneWithFiltersPtr(index, visit, plane, scratch, filters)
			}
			if err != nil {
				return err
			}
			continue
		}
		var tmp frame.Plane
		var ref *frame.Plane
		if t.bps == 1 {
			reference, _ := frameWorkReferenceFromTile(visit.Prediction.InterMotion.References.Ref[0])
			ref = b.sameSizeRefPlane(reference, plane, t.codedW, t.codedH, &tmp)
		}
		if ref == nil {
			if err := b.predictBlockInterPlaneWithFiltersPtr(index, visit, plane, filters, scratch); err != nil {
				return err
			}
			continue
		}
		if err := motion.PredictTranslation8(&t.output, ref, t.x, t.y, t.writeW, t.writeH, t.filterW, t.filterH, visit.Prediction.InterMotion.MV[0], t.ssX, t.ssY, filters, frameWorkConvolveScratch(scratch)); err != nil {
			return ErrInvalidBatch
		}
	}
	return nil
}
