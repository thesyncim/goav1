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
		return b.predictBlockInterBound(index, lean, visit, scratch, filters, compound)
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
	c.leanIndex = cacheIndex
	c.leanOK = true
	return l
}

// target is interTargetBound's unclipped case on the bound state: a block
// whose full extent lies inside the job's writable window. It reports false,
// with t untouched, for any other block, which takes interTargetAt.
func (l *frameWorkLeanInter) target(plane FrameWorkPlane, x, y, width, height, fullW, fullH int, t *frameWorkInterTarget) bool {
	p := &l.plane[plane]
	rx, ry := x-p.winX, y-p.winY
	if rx < 0 || ry < 0 || rx+fullW > p.clipW || ry+fullH > p.clipH ||
		width <= 0 || height <= 0 || width > fullW || height > fullH ||
		!frameWorkPredictionExtentValid(fullW) || !frameWorkPredictionExtentValid(fullH) {
		return false
	}
	// frameWorkPlaneBlockStartsBeyondPlane: such a block is not predicted.
	if (x >= p.codedW && l.bps*(x+1) > p.rawStride) || (y >= p.codedH && p.rawStride*(y+1) > p.rawLen) {
		return false
	}
	t.output = p.output
	t.x, t.y = x, y
	t.width, t.height = width, height
	t.writeW, t.writeH = fullW, fullH
	t.filterW, t.filterH = fullW, fullH
	t.codedW, t.codedH = p.codedW, p.codedH
	t.bps = l.bps
	t.ssX, t.ssY = plane != FrameWorkPlaneY && l.ssX != 0, plane != FrameWorkPlaneY && l.ssY != 0
	return true
}

// sameRef is sameSizeRefPlane on the bound state: the view is resolved once
// per job and reference plane.
func (b *FrameWorkBatch) sameRef(l *frameWorkLeanInter, reference FrameWorkReference, plane FrameWorkPlane) *frame.Plane {
	i := uint(reference)*3 + uint(plane)
	if i >= uint(len(l.ref)) {
		return nil
	}
	bit := uint32(1) << i
	if l.refBound&bit == 0 {
		// The bound state lives in the job's geometry cache, so boundRefPlane
		// fails only for an invalid view, where sameSizeRefPlane fails too.
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
	if l.refSame&bit == 0 {
		return nil
	}
	return l.ref[i]
}

// predictBlockInterBound is predictBlockInterLeanPlanes on the job's bound
// state, dav1d's recon_b_inter: the luma position and extents come straight
// from the block, and U and V share one chroma position (cbx4/cby4/cbw4/cbh4).
// A plane whose block is not wholly inside the writable window takes
// interTargetAt.
func (b *FrameWorkBatch) predictBlockInterBound(index int, l *frameWorkLeanInter, visit *tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters, compound bool) error {
	block := &visit.Block
	dims, ok := block.Size.Dimensions()
	if !ok || block.VisibleW4 == 0 || block.VisibleH4 == 0 || block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow {
		return ErrInvalidBatch
	}
	var t frameWorkInterTarget
	x, y := int(block.MICol)*4, int(block.MIRow)*4
	if err := b.predictBoundPlane(index, l, visit, scratch, filters, compound, FrameWorkPlaneY,
		x, y, int(block.VisibleW4)*4, int(block.VisibleH4)*4, int(dims.W4)*4, int(dims.H4)*4, false, false, &t); err != nil {
		return err
	}
	if !l.chroma {
		return nil
	}
	ssX, ssY := int(l.ssX), int(l.ssY)
	chroma, has, ok := tile.ChromaPlaneBlock(block.Size, int(block.X4), int(block.Y4), l.ssX, l.ssY)
	if !ok {
		return ErrInvalidBatch
	}
	if !has {
		return nil
	}
	cdims, ok := chroma.Dimensions()
	if !ok {
		return ErrInvalidBatch
	}
	miCol, miRow := int(block.MICol), int(block.MIRow)
	if ssX != 0 && miCol&1 != 0 && dims.W4 == 1 {
		miCol--
	}
	if ssY != 0 && miRow&1 != 0 && dims.H4 == 1 {
		miRow--
	}
	blockX4, blockY4 := int(block.X4), int(block.Y4)
	cw := (((blockX4 + int(block.VisibleW4) + ssX) >> ssX) - (blockX4 >> ssX)) * 4
	ch := (((blockY4 + int(block.VisibleH4) + ssY) >> ssY) - (blockY4 >> ssY)) * 4
	if cw <= 0 || ch <= 0 {
		return ErrInvalidBatch
	}
	cx, cy := (miCol*4)>>ssX, (miRow*4)>>ssY
	for plane := FrameWorkPlaneU; plane <= FrameWorkPlaneV; plane++ {
		if !compound && visit.Prediction.SubChromaInterValid {
			if err := b.predictBlockInterSubChromaPlanePtr(index, visit, plane, scratch); err != nil {
				return err
			}
			continue
		}
		if err := b.predictBoundPlane(index, l, visit, scratch, filters, compound, plane,
			cx, cy, cw, ch, int(cdims.W4)*4, int(cdims.H4)*4, ssX != 0, ssY != 0, &t); err != nil {
			return err
		}
	}
	return nil
}

// predictBoundPlane predicts one plane of a bound translational block at
// (x, y) with visible (width, height) and full (fullW, fullH) extents.
func (b *FrameWorkBatch) predictBoundPlane(index int, l *frameWorkLeanInter, visit *tile.BlockLoopVisit, scratch *FrameWorkInterPredictionScratch, filters motion.InterpFilters, compound bool, plane FrameWorkPlane, x, y, width, height, fullW, fullH int, ssX, ssY bool, t *frameWorkInterTarget) error {
	if !l.target(plane, x, y, width, height, fullW, fullH, t) {
		ok, err := b.interTargetAt(index, &visit.Block, plane, x, y, width, height, ssX, ssY, t)
		if err != nil || !ok {
			return err
		}
	}
	if compound {
		handled, err := b.predictBlockInterCompoundLean(visit, plane, scratch, t, visit.Prediction.CompoundBlend, filters)
		if !handled {
			err = b.predictBlockInterCompoundPlaneWithFiltersPtr(index, visit, plane, scratch, filters)
		}
		return err
	}
	var ref *frame.Plane
	if b.leanSampleWidth(t.bps) {
		reference, _ := frameWorkReferenceFromTile(visit.Prediction.InterMotion.References.Ref[0])
		ref = b.sameRef(l, reference, plane)
	}
	if ref == nil {
		return b.predictBlockInterPlaneWithFiltersPtr(index, visit, plane, filters, scratch)
	}
	var err error
	if t.bps == 1 {
		err = motion.PredictTranslation8(&t.output, ref, t.x, t.y, t.writeW, t.writeH, t.filterW, t.filterH, visit.Prediction.InterMotion.MV[0], t.ssX, t.ssY, filters, frameWorkConvolveScratch(scratch))
	} else {
		err = b.predictTranslation16(t, ref, visit.Prediction.InterMotion.MV[0], filters, scratch)
	}
	if err != nil {
		return ErrInvalidBatch
	}
	return nil
}
