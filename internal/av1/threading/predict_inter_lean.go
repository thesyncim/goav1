// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/frame"
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
	output, outputSubX, outputSubY, planeOK := frameWorkFramePlane(out, plane)
	bps := out.Layout.BytesPerSample
	if planeOK && frameWorkPlaneBlockStartsBeyondPlane(output, bps, x, y) {
		return false, nil
	}
	width, height, ok = frameWorkClipVisiblePixelsToWindowPtr(window, x, y, width, height)
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
