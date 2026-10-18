// Ported from libaom: av1/common/av1_loopfilter.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package loopfilter

import "github.com/thesyncim/goav1/internal/av1/parser"

const (
	MaxLevel     = 63
	MaxSharpness = 7
	DeltaCount   = 4
)

// Plane identifies the AV1 plane whose loop-filter level is being resolved.
type Plane uint8

const (
	PlaneY Plane = iota
	PlaneU
	PlaneV
)

// Edge identifies the edge direction for AV1's luma loop-filter levels.
type Edge uint8

const (
	EdgeVertical Edge = iota
	EdgeHorizontal
)

// ModeDeltaClass selects one of AV1's two loop-filter mode-delta entries.
type ModeDeltaClass uint8

const (
	ModeDeltaClassZero ModeDeltaClass = iota
	ModeDeltaClassMotion
)

// LevelRequest describes the block-local state needed to resolve one
// frame-level loop-filter level into the value used by filtering masks.
type LevelRequest struct {
	Plane Plane
	Edge  Edge

	SegmentID uint8
	RefFrame  uint8
	Mode      ModeDeltaClass

	DeltaLF int8
}

// PreparedLevelResolver is an immutable frame-level snapshot used when many
// blocks resolve loop-filter levels during one decode. It copies only the
// numeric inputs needed by the resolver, so callers do not retain pointers to
// event or parser scratch that may be reused for the next frame.
//
// The shape intentionally mirrors ResolveLevel's operation order: clamp the
// base plus delta-lf, clamp the segmentation delta, then apply scaled
// reference/mode deltas and clamp once more.
type PreparedLevelResolver struct {
	base          [3][2]uint8
	deltaPresent  bool
	deltaMulti    bool
	segmentEnable bool
	modeRefEnable bool
	lumaZero      bool
	monoChrome    bool
	refDelta      [parser.RefFrames]int8
	modeDelta     [2]int8
	segmentDelta  [parser.MaxSegments]preparedSegmentDelta
}

type preparedSegmentDelta struct {
	yVert int16
	yHorz int16
	u     int16
	v     int16
}

// PrepareLevelResolver snapshots the frame-header and segmentation values
// needed by ResolveBlockLevels. It is intended to be called once per frame or
// tile job, then reused for every decoded block in that frame.
func PrepareLevelResolver(params parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams, monoChrome bool) PreparedLevelResolver {
	r := PreparedLevelResolver{
		base: [3][2]uint8{
			{ClampLevel(int(params.LevelY[0])), ClampLevel(int(params.LevelY[1]))},
			{ClampLevel(int(params.LevelU)), ClampLevel(int(params.LevelU))},
			{ClampLevel(int(params.LevelV)), ClampLevel(int(params.LevelV))},
		},
		deltaPresent:  delta.DeltaLFPresent,
		deltaMulti:    delta.DeltaLFMulti,
		segmentEnable: seg.Enabled,
		modeRefEnable: params.ModeRefDeltaEnabled,
		lumaZero:      params.LevelY[0] == 0 && params.LevelY[1] == 0,
		monoChrome:    monoChrome,
		refDelta:      params.Deltas.Ref,
		modeDelta:     params.Deltas.Mode,
	}
	if seg.Enabled {
		for i := range r.segmentDelta {
			data := seg.Data.Segments[i]
			r.segmentDelta[i] = preparedSegmentDelta{
				yVert: data.DeltaLFYV,
				yHorz: data.DeltaLFYH,
				u:     data.DeltaLFU,
				v:     data.DeltaLFV,
			}
		}
	}
	return r
}

// ResolveBlockLevels resolves all plane/edge levels for one decoded block.
// Validation and early-return ordering match the optimized decoder record
// resolver: ref/mode are validated before the luma-zero fast path, and the
// segment ID is only checked when enabled segmentation is actually consulted.
func (r *PreparedLevelResolver) ResolveBlockLevels(segmentID, refFrame uint8, mode ModeDeltaClass, deltaFromBase int8, multi [DeltaCount]int8) ([3][2]uint8, error) {
	if r == nil {
		return [3][2]uint8{}, ErrInvalidFilter
	}
	var levels [3][2]uint8
	if refFrame >= parser.RefFrames || !validMode(mode) {
		return levels, ErrInvalidFilter
	}
	if r.lumaZero {
		return levels, nil
	}
	if !r.deltaPresent && !r.segmentEnable && !r.modeRefEnable {
		return r.base, nil
	}
	planeCount := 3
	if r.monoChrome {
		planeCount = 1
	}
	for plane := 0; plane < planeCount; plane++ {
		for edge := 0; edge < 2; edge++ {
			base := r.base[plane][edge]
			if plane != int(PlaneY) && base == 0 {
				continue
			}
			deltaLF := int8(0)
			if r.deltaPresent {
				if !r.deltaMulti {
					deltaLF = deltaFromBase
				} else {
					idx := plane*2 + edge
					if plane != int(PlaneY) {
						idx = plane + 1
					}
					deltaLF = multi[idx]
				}
			}
			segDelta := int16(0)
			if r.segmentEnable {
				if segmentID >= parser.MaxSegments {
					return levels, ErrInvalidFilter
				}
				seg := r.segmentDelta[segmentID]
				switch plane {
				case int(PlaneY):
					if edge == int(EdgeVertical) {
						segDelta = seg.yVert
					} else {
						segDelta = seg.yHorz
					}
				case int(PlaneU):
					segDelta = seg.u
				case int(PlaneV):
					segDelta = seg.v
				}
			}
			level := clampLevelInt(int(base) + int(deltaLF))
			level = clampLevelInt(level + int(segDelta))
			if r.modeRefEnable {
				scale := 1 << (level >> 5)
				delta := int(r.refDelta[refFrame])
				if refFrame > 0 {
					delta += int(r.modeDelta[mode])
				}
				level = clampLevelInt(level + delta*scale)
			}
			levels[plane][edge] = uint8(level)
		}
	}
	return levels, nil
}

// ClampLevel clamps v to AV1's legal loop-filter level range.
func ClampLevel(v int) uint8 {
	return uint8(clampLevelInt(v))
}

// BaseLevel returns the frame-header loop-filter level for a plane and edge.
func BaseLevel(params parser.LoopFilterParams, plane Plane, edge Edge) (uint8, error) {
	if !validEdge(edge) {
		return 0, ErrInvalidFilter
	}
	switch plane {
	case PlaneY:
		if edge == EdgeVertical {
			return ClampLevel(int(params.LevelY[0])), nil
		}
		return ClampLevel(int(params.LevelY[1])), nil
	case PlaneU:
		return ClampLevel(int(params.LevelU)), nil
	case PlaneV:
		return ClampLevel(int(params.LevelV)), nil
	default:
		return 0, ErrInvalidFilter
	}
}

// DeltaIndex returns AV1's delta_lf_multi index for a plane and edge.
func DeltaIndex(plane Plane, edge Edge) (int, error) {
	if !validEdge(edge) {
		return 0, ErrInvalidFilter
	}
	switch plane {
	case PlaneY:
		if edge == EdgeVertical {
			return 0, nil
		}
		return 1, nil
	case PlaneU:
		return 2, nil
	case PlaneV:
		return 3, nil
	default:
		return 0, ErrInvalidFilter
	}
}

// SelectDelta returns the block-local delta-lf value for a plane and edge.
func SelectDelta(params parser.DeltaParams, fromBase int8, multi [DeltaCount]int8, plane Plane, edge Edge) (int8, error) {
	if !params.DeltaLFPresent {
		return 0, nil
	}
	if !params.DeltaLFMulti {
		return fromBase, nil
	}
	idx, err := DeltaIndex(plane, edge)
	if err != nil {
		return 0, err
	}
	return multi[idx], nil
}

// SegmentDelta returns the segmentation loop-filter delta for a plane and edge.
func SegmentDelta(seg parser.SegmentationParams, segmentID uint8, plane Plane, edge Edge) (int16, error) {
	if segmentID >= parser.MaxSegments || !validEdge(edge) {
		return 0, ErrInvalidFilter
	}
	if !seg.Enabled {
		return 0, nil
	}
	data := seg.Data.Segments[segmentID]
	switch plane {
	case PlaneY:
		if edge == EdgeVertical {
			return data.DeltaLFYV, nil
		}
		return data.DeltaLFYH, nil
	case PlaneU:
		return data.DeltaLFU, nil
	case PlaneV:
		return data.DeltaLFV, nil
	default:
		return 0, ErrInvalidFilter
	}
}

// ResolveLevel applies block delta-lf, segmentation, and mode/ref deltas to
// produce the AV1 loop-filter level used by edge-mask construction.
func ResolveLevel(params parser.LoopFilterParams, seg parser.SegmentationParams, req LevelRequest) (uint8, error) {
	if req.SegmentID >= parser.MaxSegments || req.RefFrame >= parser.RefFrames || !validMode(req.Mode) {
		return 0, ErrInvalidFilter
	}

	base, err := BaseLevel(params, req.Plane, req.Edge)
	if err != nil {
		return 0, err
	}
	if params.LevelY[0] == 0 && params.LevelY[1] == 0 {
		return 0, nil
	}
	if req.Plane != PlaneY && base == 0 {
		return 0, nil
	}

	segDelta, err := SegmentDelta(seg, req.SegmentID, req.Plane, req.Edge)
	if err != nil {
		return 0, err
	}
	// libaom (av1/common/av1_loopfilter.c:get_filter_level) clamps the
	// intermediate base+delta_lf result before folding in the per-segment
	// delta; otherwise the saturating clamp before scale = 1 << (lvl >> 5)
	// disagrees and ref/mode deltas get the wrong scale.
	level := clampLevelInt(int(base) + int(req.DeltaLF))
	level = clampLevelInt(level + int(segDelta))
	if params.ModeRefDeltaEnabled {
		scale := 1 << (level >> 5)
		delta := int(params.Deltas.Ref[req.RefFrame])
		if req.RefFrame > 0 {
			delta += int(params.Deltas.Mode[req.Mode])
		}
		level = clampLevelInt(level + delta*scale)
	}
	return uint8(level), nil
}

func validEdge(edge Edge) bool {
	return edge == EdgeVertical || edge == EdgeHorizontal
}

func validMode(mode ModeDeltaClass) bool {
	return mode == ModeDeltaClassZero || mode == ModeDeltaClassMotion
}

func clampLevelInt(v int) int {
	if v < 0 {
		return 0
	}
	if v > MaxLevel {
		return MaxLevel
	}
	return v
}
