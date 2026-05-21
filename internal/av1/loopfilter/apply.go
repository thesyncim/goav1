package loopfilter

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// DeltaState is the block-local delta-lf state carried by tile decode.
type DeltaState struct {
	FromBase int8
	Multi    [DeltaCount]int8
}

// Filter4Request describes one narrow deblocking edge in plane coordinates.
type Filter4Request struct {
	LevelRequest

	X      int
	Y      int
	Length int
}

// FilterResult reports the resolved setup used for one edge filter call.
type FilterResult struct {
	Level      uint8
	Thresholds Thresholds
	Applied    bool
}

// ResolveBlockLevel selects the tile-local delta-lf value and resolves the
// final AV1 loop-filter level for one block edge.
func ResolveBlockLevel(params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req LevelRequest) (uint8, error) {
	deltaLF, err := SelectDelta(delta, state.FromBase, state.Multi, req.Plane, req.Edge)
	if err != nil {
		return 0, err
	}
	req.DeltaLF = deltaLF
	return ResolveLevel(params, seg, req)
}

// Filter4BlockEdge resolves loop-filter level and thresholds, then applies
// AV1's narrow four-sample deblocking filter when the resolved level is nonzero.
func Filter4BlockEdge(dst frame.Plane, bytesPerSample int, bitDepth uint8, params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, state DeltaState, req Filter4Request) (FilterResult, error) {
	if params.Sharpness > MaxSharpness {
		return FilterResult{}, ErrInvalidFilter
	}
	level, err := ResolveBlockLevel(params, seg, delta, state, req.LevelRequest)
	if err != nil {
		return FilterResult{}, err
	}
	if level == 0 {
		return FilterResult{}, nil
	}
	thresholds, err := ThresholdsForLevel(level, params.Sharpness)
	if err != nil {
		return FilterResult{}, err
	}
	if err := Filter4Edge(dst, bytesPerSample, bitDepth, req.Edge, req.X, req.Y, req.Length, thresholds); err != nil {
		return FilterResult{}, err
	}
	return FilterResult{
		Level:      level,
		Thresholds: thresholds,
		Applied:    true,
	}, nil
}
