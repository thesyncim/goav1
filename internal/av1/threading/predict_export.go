package threading

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// predict_export.go exposes the frame-work intra edge construction to the
// encoder, which must predict candidate blocks exactly the way the decoder
// will reconstruct them. These are thin wrappers over the same helpers the
// decode pipeline calls; no logic lives here.

// BuildLumaDirectionalEdges derives the extended-edge availability for a luma
// block the way the reconstruction pipeline does and builds the directional
// prediction edges over dst. The caller owns the scratch.
func BuildLumaDirectionalEdges(dst frame.Plane, bitDepth uint8, x, y, width, height, angle int, block tile.BlockVisit, scratch *FrameWorkIntraPredictionScratch, sbSizeMIB uint8, miColEnd, miRowEnd uint32, enableIntraEdgeFilter, smoothNeighbor bool, visibleW, visibleH int) (prediction.DirectionalEdges, error) {
	allowTopRight, allowBottomLeft := frameWorkLumaDirectionalExtendedEdges(block, sbSizeMIB, miColEnd, miRowEnd, x, y, width, height)
	return frameWorkDirectionalPredictionEdges(dst, 1, bitDepth, x, y, width, height, angle, block, scratch, enableIntraEdgeFilter, smoothNeighbor, allowTopRight, allowBottomLeft, visibleW, visibleH)
}
