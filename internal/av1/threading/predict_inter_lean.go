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
		c.leanOK = false
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
		// Field stores, not a composite literal: the literal is built in a
		// temporary and then block-copied into *t.
		t.output = bind.output
		t.x, t.y = x, y
		t.width, t.height = width, height
		t.writeW, t.writeH = fullW, fullH
		t.filterW, t.filterH = fullW, fullH
		t.codedW, t.codedH = bind.codedW, bind.codedH
		t.bps, t.ssX, t.ssY = bind.bps, ssX, ssY
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
	t.output = bind.output
	t.x, t.y = x, y
	t.width, t.height = width, height
	t.writeW, t.writeH = writeW, writeH
	t.filterW, t.filterH = fullW, fullH
	t.codedW, t.codedH = bind.codedW, bind.codedH
	t.bps, t.ssX, t.ssY = bind.bps, ssX, ssY
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

// predictBlockInterLean is dav1d's recon_b_inter prediction (src/recon_tmpl.c,
// templated over the bit depth) for the common translational block,
// single-reference or compound: the block-level mode checks run once, the luma
// plane predicts through mc(), and U and V share one chroma position as
// dav1d's cbw4/cbh4/cbx4/cby4. Any plane the lean target cannot take (scaled
// reference, a sample width that does not match the bit depth) goes through
// the general per-plane route, which owns those semantics. It reports false,
// with nothing written, for a block outside the translational shapes.
func (b *FrameWorkBatch) predictBlockInterLean(index int, visit *tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters) (bool, error) {
	pred := &visit.Prediction
	refs := &pred.InterMotion.References
	if (pred.InterIntraValid && pred.InterIntra.Enabled) ||
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
	if lean := b.leanInter(index); lean != nil {
		if handled, err := b.reconBInter(index, lean, visit, scratch, &filters); handled {
			return err
		}
	}
	block := &visit.Block
	color := &b.Sequence.ColorConfig
	var cx, cy, cw, ch int
	var cssX, cssY bool
	// interTargetAt writes every field of t whenever it reports ok, so one
	// target serves all three planes without re-zeroing it per plane.
	var t frameWorkInterTarget
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
		if b.leanSampleWidth(t.bps) {
			reference, _ := frameWorkReferenceFromTile(visit.Prediction.InterMotion.References.Ref[0])
			ref = b.sameSizeRefPlane(reference, plane, t.codedW, t.codedH, &tmp)
		}
		if ref == nil {
			if err := b.predictBlockInterPlaneWithFiltersPtr(index, visit, plane, filters, scratch); err != nil {
				return err
			}
			continue
		}
		if t.bps == 1 {
			err = motion.PredictTranslation8(&t.output, ref, t.x, t.y, t.writeW, t.writeH, t.filterW, t.filterH, visit.Prediction.InterMotion.MV[0], t.ssX, t.ssY, filters, frameWorkConvolveScratch(scratch))
		} else {
			err = b.predictTranslation16(&t, ref, visit.Prediction.InterMotion.MV[0], filters, scratch)
		}
		if err != nil {
			return ErrInvalidBatch
		}
	}
	return nil
}

// leanSampleWidth reports whether a lean target's sample width is the one the
// stream's bit depth implies (1 byte at 8 bits, 2 above), the only layouts the
// lean translational predictors take.
func (b *FrameWorkBatch) leanSampleWidth(bps int) bool {
	if b.Sequence.ColorConfig.BitDepth == 8 {
		return bps == 1
	}
	return bps == 2
}

// predictTranslation16 is dav1d's mc() for a same-size reference above 8
// bits: the 16bpc put kernels at the lean target. (The 8-bit call sites run
// motion.PredictTranslation8 inline.)
func (b *FrameWorkBatch) predictTranslation16(t *frameWorkInterTarget, ref *frame.Plane, mv motion.Vector, filters motion.InterpFilters, scratch *FrameWorkInterPredictionScratch) error {
	return motion.PredictTranslation16(&t.output, ref, b.Sequence.ColorConfig.BitDepth, t.x, t.y, t.writeW, t.writeH, t.filterW, t.filterH, mv, t.ssX, t.ssY, filters, frameWorkConvolveScratch(scratch))
}

// prepTranslation16 is the compound counterpart: one reference's CONV_BUF.
func (b *FrameWorkBatch) prepTranslation16(buf *motion.CompoundConvBuf, t *frameWorkInterTarget, ref *frame.Plane, mv motion.Vector, filters motion.InterpFilters, scratch *FrameWorkInterPredictionScratch) error {
	return motion.PrepTranslation16(buf, ref, b.Sequence.ColorConfig.BitDepth, t.x, t.y, t.width, t.height, t.filterW, t.filterH, mv, t.ssX, t.ssY, filters, &scratch.Compound)
}

// frameWorkLeanInterPlane is one plane of the job's bound output as dav1d's
// recon_b_inter reads f->cur: the output plane extended to the writable clip
// window, the window origin and clip extent, and the raw plane extent.
type frameWorkLeanInterPlane struct {
	output    frame.Plane
	winX      int
	winY      int
	clipW     int
	clipH     int
	codedW    int
	codedH    int
	rawStride int
	rawLen    int
}

// frameWorkLeanInter is the job-invariant state of the bound inter path: every
// present plane's output binding, the chroma subsampling and the sample width,
// plus dav1d's f->refp same-size reference views, resolved once per reference
// plane (refBound marks a resolved view, refSame a same-size one).
type frameWorkLeanInter struct {
	plane    [3]frameWorkLeanInterPlane
	chroma   bool
	ssX      uint8
	ssY      uint8
	bps      int
	refBound uint32
	refSame  uint32
	ref      [3 * parser.InterRefsPerFrame]*frame.Plane
	// filters is the frame's interpolation filter pair (dav1d's
	// f->frame_hdr->subpel_filter_mode) when switchable is false; filtersOK
	// is false for a frame filter the predictors do not take.
	filters    motion.InterpFilters
	switchable bool
	filtersOK  bool
}

// leanInter returns the job's bound inter state, deriving it once every
// present plane has an inter binding (the first blocks of a job take
// interTargetAt, which binds them). It returns nil without a geometry cache
// or while a plane is unbound.
func (b *FrameWorkBatch) leanInter(index int) *frameWorkLeanInter {
	c := b.geomCache
	if c == nil {
		return nil
	}
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	if !ok {
		return nil
	}
	if c.leanOK && c.leanIndex == cacheIndex && c.interOut == b.Output {
		return &c.lean
	}
	out := b.Output
	if out == nil {
		return nil
	}
	color := &b.Sequence.ColorConfig
	last := FrameWorkPlaneV
	if color.MonoChrome {
		last = FrameWorkPlaneY
	}
	l := &c.lean
	for plane := FrameWorkPlaneY; plane <= last; plane++ {
		window := b.cachedJobOutputPlanePtr(index, plane)
		bind := b.cachedInterBinding(index, plane, out)
		if window == nil || bind == nil || !b.leanSampleWidth(bind.bps) ||
			bind.raw.Stride <= 0 || (plane != FrameWorkPlaneY && (bind.ssX != color.SubsamplingX || bind.ssY != color.SubsamplingY)) ||
			(plane == FrameWorkPlaneY && (bind.ssX || bind.ssY)) {
			return nil
		}
		clipW, clipH := int(window.ClipWidth), int(window.ClipHeight)
		if clipW <= 0 {
			clipW = int(window.Width)
		}
		if clipH <= 0 {
			clipH = int(window.Height)
		}
		p := &l.plane[plane]
		p.output = bind.output
		p.winX, p.winY = int(window.X), int(window.Y)
		p.clipW, p.clipH = clipW, clipH
		p.codedW, p.codedH = bind.codedW, bind.codedH
		p.rawStride, p.rawLen = bind.raw.Stride, len(bind.raw.Pix)
		if plane == FrameWorkPlaneY {
			l.bps = bind.bps
		} else if bind.bps != l.bps {
			return nil
		}
	}
	l.chroma = last == FrameWorkPlaneV
	l.ssX = uint8(frameWorkSubsampleShift(color.SubsamplingX))
	l.ssY = uint8(frameWorkSubsampleShift(color.SubsamplingY))
	l.refBound, l.refSame = 0, 0
	l.switchable = b.TileInfo.InterpolationFilter == parser.InterpolationSwitchable
	filters, err := frameWorkMotionFilters(b.TileInfo)
	l.filters, l.filtersOK = filters, err == nil
	c.leanIndex = cacheIndex
	c.leanOK = true
	return l
}

// bindSameRefs resolves the bound state's same-size views of every plane of
// reference (sameSizeRefPlane's result, dav1d's f->refp[refidx].p), once per
// job: refBound marks a resolved view and refSame a same-size one. The bound
// state lives in the job's geometry cache, so boundRefPlane fails only for an
// invalid view, where sameSizeRefPlane fails too.
func (b *FrameWorkBatch) bindSameRefs(l *frameWorkLeanInter, reference FrameWorkReference) {
	for plane := FrameWorkPlaneY; plane <= FrameWorkPlaneV; plane++ {
		i := uint(reference)*3 + uint(plane)
		if i >= uint(len(l.ref)) {
			return
		}
		bit := uint32(1) << i
		if l.refBound&bit != 0 {
			continue
		}
		p := &l.plane[plane]
		ref := b.boundRefPlane(reference, plane)
		if ref != nil && (ref.Width != p.codedW || ref.Height != p.codedH || !b.referenceSameSize(reference)) {
			ref = nil
		}
		l.ref[i] = ref
		l.refBound |= bit
		if ref != nil {
			l.refSame |= bit
		}
	}
}

// boundLeanInter returns the job's bound inter state when it is already
// derived for index and the current output, else nil (the general chain then
// derives it).
func (b *FrameWorkBatch) boundLeanInter(index int) *frameWorkLeanInter {
	c := b.geomCache
	cacheIndex, ok := frameWorkJobCacheIndex(index)
	if c == nil || !ok || !c.leanOK || c.leanIndex != cacheIndex || c.interOut != b.Output {
		return nil
	}
	return &c.lean
}

// reconBInter is dav1d's recon_b_inter (src/recon_tmpl.c) on the job's bound
// state for a translational block, single-reference or compound (no
// inter-intra, OBMC or warp, no single-reference global motion); it reports
// false, with nothing written, for any other block, which takes the general
// prediction chain. The filters are explicitFilters when given, else the
// block's (dav1d b->filter2d). Within the block: the luma position and extents
// come straight from the block, U and V share one chroma position
// (cbx4/cby4/cbw4/cbh4), and each plane goes to its mc() kernel
// (motion.PredictTranslation8/16, or PrepTranslation8/16 plus the compound
// blend) from the bound output and reference planes. A plane whose block is
// not wholly inside the writable window takes interTargetAt; a plane the bound
// views cannot take (scaled reference, global-warp compound) takes the general
// per-plane route, which owns those semantics.
func (b *FrameWorkBatch) reconBInter(index int, l *frameWorkLeanInter, visit *tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, explicitFilters *motion.InterpFilters) (bool, error) {
	pred := &visit.Prediction
	refs := &pred.InterMotion.References
	if !pred.Valid || pred.Intra || !pred.InterMotionValid || (pred.IntrabcValid && pred.Intrabc) ||
		(pred.InterIntraValid && pred.InterIntra.Enabled) ||
		(pred.MotionModeValid && pred.MotionMode != tile.MotionModeTranslation) {
		return false, nil
	}
	compound := refs.Compound
	if compound {
		blend := pred.CompoundBlend.Type
		if scratch == nil || !refs.Ref[0].Valid() || !refs.Ref[1].Valid() ||
			(blend != tile.CompoundTypeAverage && blend != tile.CompoundTypeDistWtd &&
				blend != tile.CompoundTypeWedge && blend != tile.CompoundTypeDiffWtd) {
			return false, nil
		}
	} else if pred.GlobalWarpedMotionValid || !refs.Ref[0].Valid() || refs.Ref[1] != tile.ReferenceFrameNone {
		return false, nil
	}
	// dav1d's b->filter2d: the block's switchable pair, or the frame's.
	var filters motion.InterpFilters
	switch {
	case explicitFilters != nil:
		if !explicitFilters.X.Valid() || !explicitFilters.Y.Valid() {
			return false, nil
		}
		filters = *explicitFilters
	case l.switchable:
		if !pred.InterpFiltersValid || !pred.InterpFilters.X.Valid() || !pred.InterpFilters.Y.Valid() {
			return false, nil
		}
		filters = pred.InterpFilters
	case l.filtersOK:
		filters = l.filters
	default:
		return false, nil
	}
	block := &visit.Block
	dims, ok := block.Size.Dimensions()
	if !ok || block.VisibleW4 == 0 || block.VisibleH4 == 0 || block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow {
		return true, ErrInvalidBatch
	}
	mv := &pred.InterMotion.MV
	reference, _ := frameWorkReferenceFromTile(refs.Ref[0])
	reference1, _ := frameWorkReferenceFromTile(refs.Ref[1])
	// The references' plane views, bound once per job; slot ref*3+plane.
	slot0, slot1 := uint(reference)*3, uint(reference1)*3
	if l.refBound>>slot0&7 != 7 {
		b.bindSameRefs(l, reference)
	}
	if compound && l.refBound>>slot1&7 != 7 {
		b.bindSameRefs(l, reference1)
	}
	// Luma first; the chroma geometry is derived once for both planes below.
	plane := FrameWorkPlaneY
	x, y := int(block.MICol)*4, int(block.MIRow)*4
	width, height := int(block.VisibleW4)*4, int(block.VisibleH4)*4
	fullW, fullH := int(dims.W4)*4, int(dims.H4)*4
	ssX, ssY := false, false
	var t frameWorkInterTarget
	for {
		// The block's target in the bound output plane: a block wholly inside
		// the writable window writes its full extent there (dav1d mc() on
		// f->cur); any other block takes interTargetAt, which clips it.
		p := &l.plane[plane]
		out := &p.output
		px, py := x, y
		w, h := width, height
		writeW, writeH := fullW, fullH
		filterW, filterH := fullW, fullH
		codedW, codedH := p.codedW, p.codedH
		bps := l.bps
		if rx, ry := x-p.winX, y-p.winY; rx < 0 || ry < 0 || rx+fullW > p.clipW || ry+fullH > p.clipH ||
			width > fullW || height > fullH ||
			!frameWorkPredictionExtentValid(fullW) || !frameWorkPredictionExtentValid(fullH) ||
			(x >= p.codedW && l.bps*(x+1) > p.rawStride) || (y >= p.codedH && p.rawStride*(y+1) > p.rawLen) {
			present, err := b.interTargetAt(index, block, plane, x, y, width, height, ssX, ssY, &t)
			if err != nil {
				return true, err
			}
			if !present {
				goto next
			}
			out = &t.output
			px, py = t.x, t.y
			w, h = t.width, t.height
			writeW, writeH = t.writeW, t.writeH
			filterW, filterH = t.filterW, t.filterH
			codedW, codedH = t.codedW, t.codedH
			bps = t.bps
		}
		if compound {
			// dav1d's two prep mc() calls into the CONV_BUFs, then the blend.
			var ref0, ref1 *frame.Plane
			if (w < 8 || h < 8 || (!pred.GlobalWarpedMotionCompoundValid[0] && !pred.GlobalWarpedMotionCompoundValid[1])) &&
				bps == l.bps && codedW == p.codedW && codedH == p.codedH {
				if i0, i1 := slot0+uint(plane), slot1+uint(plane); (l.refSame>>i0)&(l.refSame>>i1)&1 != 0 {
					ref0, ref1 = l.ref[i0], l.ref[i1]
				}
			}
			if ref0 == nil || ref1 == nil {
				if err := b.predictBlockInterCompoundPlaneWithFiltersPtr(index, visit, plane, scratch, filters); err != nil {
					return true, err
				}
				goto next
			}
			if l.bps == 1 {
				if motion.PrepTranslation8(&scratch.Conv0, ref0, px, py, w, h, filterW, filterH, mv[0], ssX, ssY, filters, &scratch.Compound) != nil ||
					motion.PrepTranslation8(&scratch.Conv1, ref1, px, py, w, h, filterW, filterH, mv[1], ssX, ssY, filters, &scratch.Compound) != nil {
					return true, ErrInvalidBatch
				}
			} else {
				bd := b.Sequence.ColorConfig.BitDepth
				if motion.PrepTranslation16(&scratch.Conv0, ref0, bd, px, py, w, h, filterW, filterH, mv[0], ssX, ssY, filters, &scratch.Compound) != nil ||
					motion.PrepTranslation16(&scratch.Conv1, ref1, bd, px, py, w, h, filterW, filterH, mv[1], ssX, ssY, filters, &scratch.Compound) != nil {
					return true, ErrInvalidBatch
				}
			}
			if err := b.blendCompoundConvBufs(visit, plane, scratch, pred.CompoundBlend, *out, bps, px, py, w, h, ssX, ssY); err != nil {
				return true, err
			}
		} else {
			var ref *frame.Plane
			if bps == l.bps && codedW == p.codedW && codedH == p.codedH {
				if i := slot0 + uint(plane); l.refSame>>i&1 != 0 {
					ref = l.ref[i]
				}
			}
			if ref == nil {
				if err := b.predictBlockInterPlaneWithFiltersPtr(index, visit, plane, filters, scratch); err != nil {
					return true, err
				}
				goto next
			}
			var err error
			if l.bps == 1 {
				// dav1d's mc(): the put kernel straight from the reference
				// origin; a shape the kernels decline takes the general path.
				refX, refY, subX, subY := motion.TranslationOrigin(px, py, mv[0], ssX, ssY)
				if !motion.Put8(out, ref, px, py, refX, refY, writeW, writeH, filterW, filterH, subX, subY, filters, frameWorkConvolveScratch(scratch)) {
					err = motion.PredictTranslation8(out, ref, px, py, writeW, writeH, filterW, filterH, mv[0], ssX, ssY, filters, frameWorkConvolveScratch(scratch))
				}
			} else {
				err = motion.PredictTranslation16(out, ref, b.Sequence.ColorConfig.BitDepth, px, py, writeW, writeH, filterW, filterH, mv[0], ssX, ssY, filters, frameWorkConvolveScratch(scratch))
			}
			if err != nil {
				return true, ErrInvalidBatch
			}
		}
	next:
		if plane == FrameWorkPlaneV || !l.chroma {
			return true, nil
		}
		if plane == FrameWorkPlaneU {
			plane = FrameWorkPlaneV
			continue
		}
		// The chroma block of dav1d's recon_b_inter (cbw4/cbh4/cbx4/cby4),
		// shared by U and V.
		chroma, has, ok := tile.ChromaPlaneBlock(block.Size, int(block.X4), int(block.Y4), l.ssX, l.ssY)
		if !ok {
			return true, ErrInvalidBatch
		}
		if !has {
			return true, nil
		}
		cdims, ok := chroma.Dimensions()
		if !ok {
			return true, ErrInvalidBatch
		}
		if !compound && pred.SubChromaInterValid {
			if err := b.predictBlockInterSubChromaPlanePtr(index, visit, FrameWorkPlaneU, scratch); err != nil {
				return true, err
			}
			return true, b.predictBlockInterSubChromaPlanePtr(index, visit, FrameWorkPlaneV, scratch)
		}
		sx, sy := int(l.ssX), int(l.ssY)
		miCol, miRow := int(block.MICol), int(block.MIRow)
		if sx != 0 && miCol&1 != 0 && dims.W4 == 1 {
			miCol--
		}
		if sy != 0 && miRow&1 != 0 && dims.H4 == 1 {
			miRow--
		}
		blockX4, blockY4 := int(block.X4), int(block.Y4)
		width = (((blockX4 + int(block.VisibleW4) + sx) >> sx) - (blockX4 >> sx)) * 4
		height = (((blockY4 + int(block.VisibleH4) + sy) >> sy) - (blockY4 >> sy)) * 4
		if width <= 0 || height <= 0 {
			return true, ErrInvalidBatch
		}
		x, y = (miCol*4)>>sx, (miRow*4)>>sy
		fullW, fullH = int(cdims.W4)*4, int(cdims.H4)*4
		ssX, ssY = sx != 0, sy != 0
		plane = FrameWorkPlaneU
	}
}
