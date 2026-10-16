// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
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
// nil error for an absent plane or a block that starts beyond the output.
func (b *FrameWorkBatch) interTarget(index int, block *tile.BlockVisit, plane FrameWorkPlane, t *frameWorkInterTarget) (bool, error) {
	color := b.Sequence.ColorConfig
	x, y, width, height, ssX, ssY, ok, err := frameWorkBlockPlanePosition(*block, color, plane)
	if err != nil || !ok {
		return ok, err
	}
	window, windowOK := b.cachedJobOutputPlaneTrusted(index, plane)
	if !windowOK {
		window, err = b.JobOutputPlane(index, plane)
		if err != nil {
			return false, err
		}
	}
	out := b.Output
	if out == nil {
		return false, ErrInvalidBatch
	}
	if frameWorkPlaneBlockStartsBeyondOutput(out, plane, x, y) {
		return false, nil
	}
	width, height, ok = frameWorkClipVisiblePixelsToWindow(window, x, y, width, height)
	if !ok {
		return false, ErrInvalidBatch
	}
	writeW, writeH := width, height
	fullW, fullH, err := frameWorkBlockPlanePredictionExtentPixels(*block, color, plane)
	if err != nil {
		return false, err
	}
	if cw, ch, ok := frameWorkClipVisiblePixelsToWindow(window, x, y, fullW, fullH); ok {
		writeW, writeH = cw, ch
	}
	if _, ok := frameWorkPredictionFilterExtentPacked(fullW, fullH); !ok {
		return false, ErrInvalidBatch
	}
	output, outputSubX, outputSubY, ok := frameWorkFramePlane(out, plane)
	bps := out.Layout.BytesPerSample
	if !ok || bps <= 0 || bps > int(^uint8(0)) || outputSubX != ssX || outputSubY != ssY {
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
	t.output = frameWorkExtendPlaneToClip(output, window, bps)
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
