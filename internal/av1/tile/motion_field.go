package tile

import "github.com/thesyncim/goav1/internal/av1/motion"

const (
	motionFieldMaxFrameDistance = 31
	motionFieldMFMVStackSize    = 3
	motionFieldMVUpper          = 1 << 14
	motionFieldMVLower          = -(1 << 14)
	motionFieldMaxOffsetWidth   = 64
	motionFieldMaxOffsetHeight  = 0
)

var motionFieldDivMult = [32]int16{
	0, 16384, 8192, 5461, 4096, 3276, 2730, 2340,
	2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
	1024, 963, 910, 862, 819, 780, 744, 712,
	682, 655, 630, 606, 585, 564, 546, 528,
}

// TemporalMotionField stores libaom's current-frame tpl_mvs side data at the
// same half-MI granularity as ReferenceMVFrame.
type TemporalMotionField struct {
	Rows    uint16
	Cols    uint16
	Stride  uint16
	Entries []TemporalMotionEntry
}

// TemporalMotionEntry mirrors the TPL_MV_REF payload needed by add_tpl_ref_mv.
type TemporalMotionEntry struct {
	MV             motion.Vector
	RefFrameOffset uint8
	Valid          bool
}

// TemporalMotionReferenceFrame describes one resolved AV1 reference frame's
// MV_REF side data and order-hint metadata.
type TemporalMotionReferenceFrame struct {
	Frame *ReferenceMVFrame

	OrderHint     uint8
	RefOrderHints [referenceFrameCount]uint8

	IntraOnly bool
}

// TemporalMotionSetupRequest describes libaom's av1_setup_motion_field() pass
// for the current frame.
type TemporalMotionSetupRequest struct {
	EnableOrderHint  bool
	OrderHintBits    uint8
	CurrentOrderHint uint8

	References [referenceFrameCount]TemporalMotionReferenceFrame
}

// TemporalMotionSetupStats reports which setup projections were attempted.
type TemporalMotionSetupStats struct {
	Projections uint8
	LastOverlay bool
	RefStamp    int8
}

// TemporalMotionProjectionRequest describes one libaom motion_field_projection
// pass from a decoded reference frame's MV_REF grid into the current frame.
type TemporalMotionProjectionRequest struct {
	StartFrame *ReferenceMVFrame

	OrderHintBits      uint8
	CurrentOrderHint   uint8
	StartOrderHint     uint8
	StartRefOrderHints [referenceFrameCount]uint8

	// Backward matches libaom's dir == 2 path used for LAST/LAST2 projection.
	Backward bool
}

// Init attaches caller-owned storage and clears it to invalid temporal samples.
func (f *TemporalMotionField) Init(miRows uint32, miCols uint32, entries []TemporalMotionEntry) error {
	if f == nil {
		return ErrInvalidDecodeState
	}
	rows, cols, need, err := referenceMVHalfMIGridShape(miRows, miCols)
	if err != nil {
		return err
	}
	if len(entries) < need {
		return ErrInvalidDecodeState
	}
	f.Rows = rows
	f.Cols = cols
	f.Stride = cols
	f.Entries = entries[:need]
	f.Clear()
	return nil
}

// Clear invalidates all temporal samples while preserving caller-owned storage.
func (f *TemporalMotionField) Clear() {
	if f == nil {
		return
	}
	for i := range f.Entries {
		f.Entries[i] = TemporalMotionEntry{}
	}
}

// Setup ports libaom's av1_setup_motion_field() projection order. The field is
// cleared only when order hints are enabled, matching libaom's early return.
func (f *TemporalMotionField) Setup(req TemporalMotionSetupRequest) (TemporalMotionSetupStats, error) {
	if err := f.validate(); err != nil {
		return TemporalMotionSetupStats{}, err
	}
	stats := TemporalMotionSetupStats{RefStamp: motionFieldMFMVStackSize - 1}
	if !req.EnableOrderHint {
		return stats, nil
	}
	if _, err := motionFieldRelativeOrderHint(req.OrderHintBits, 0, req.CurrentOrderHint); err != nil {
		return TemporalMotionSetupStats{}, err
	}
	f.Clear()

	refOrderHints := temporalReferenceOrderHints(req.References)
	if req.References[ReferenceFrameLast].Frame != nil {
		stats.LastOverlay = req.References[ReferenceFrameLast].RefOrderHints[ReferenceFrameAltref] == refOrderHints[ReferenceFrameGolden]
		if !stats.LastOverlay {
			projected, err := f.projectSetupReference(req, ReferenceFrameLast, true)
			if err != nil {
				return TemporalMotionSetupStats{}, err
			}
			if projected {
				stats.Projections++
			}
		}
		stats.RefStamp--
	}

	for _, ref := range [...]ReferenceFrame{ReferenceFrameBWD, ReferenceFrameAltref2} {
		future, err := temporalReferenceFuture(req.OrderHintBits, refOrderHints[ref], req.CurrentOrderHint)
		if err != nil {
			return TemporalMotionSetupStats{}, err
		}
		if !future {
			continue
		}
		projected, err := f.projectSetupReference(req, ref, false)
		if err != nil {
			return TemporalMotionSetupStats{}, err
		}
		if projected {
			stats.RefStamp--
			stats.Projections++
		}
	}

	future, err := temporalReferenceFuture(req.OrderHintBits, refOrderHints[ReferenceFrameAltref], req.CurrentOrderHint)
	if err != nil {
		return TemporalMotionSetupStats{}, err
	}
	if future && stats.RefStamp >= 0 {
		projected, err := f.projectSetupReference(req, ReferenceFrameAltref, false)
		if err != nil {
			return TemporalMotionSetupStats{}, err
		}
		if projected {
			stats.RefStamp--
			stats.Projections++
		}
	}

	if stats.RefStamp >= 0 {
		projected, err := f.projectSetupReference(req, ReferenceFrameLast2, true)
		if err != nil {
			return TemporalMotionSetupStats{}, err
		}
		if projected {
			stats.Projections++
		}
	}
	return stats, nil
}

// ProjectReferenceFrame ports libaom's motion_field_projection() inner pass.
// The field is not cleared; callers should Clear or Init once before applying
// the ordered set of reference-frame projections for the current frame.
func (f *TemporalMotionField) ProjectReferenceFrame(req TemporalMotionProjectionRequest) (bool, error) {
	if err := f.validate(); err != nil {
		return false, err
	}
	start := req.StartFrame
	if start == nil {
		return false, nil
	}
	if err := validateReferenceMVFrame(start); err != nil {
		return false, err
	}
	if start.Rows != f.Rows || start.Cols != f.Cols {
		return false, nil
	}

	startToCurrent, err := motionFieldRelativeOrderHint(req.OrderHintBits, req.StartOrderHint, req.CurrentOrderHint)
	if err != nil {
		return false, err
	}
	if req.Backward {
		startToCurrent = -startToCurrent
	}
	refOffsets, err := motionFieldRefOffsets(req.OrderHintBits, req.StartOrderHint, req.StartRefOrderHints)
	if err != nil {
		return false, err
	}
	projectionRefs := motionFieldProjectionReferences(startToCurrent, refOffsets)

	startRows := int(start.Rows)
	startCols := int(start.Cols)
	startStride := int(start.Stride)
	rows := int(f.Rows)
	cols := int(f.Cols)
	stride := int(f.Stride)
	for blkRow := 0; blkRow < startRows; blkRow++ {
		baseBlkRow := (blkRow >> 3) << 3
		rowMin := max(0, baseBlkRow-(motionFieldMaxOffsetHeight>>3))
		rowMax := min(rows, baseBlkRow+8+(motionFieldMaxOffsetHeight>>3))
		sourceRowBase := blkRow * startStride
		sourceRow := start.Entries[sourceRowBase : sourceRowBase+startCols]
		destinationOriginRow := blkRow * stride
		for blkCol, mvRef := range sourceRow {
			if !mvRef.Valid || !mvRef.Ref.Valid() {
				continue
			}
			projection := projectionRefs[mvRef.Ref]
			if !projection.valid {
				continue
			}
			if mvRef.MV.Row == 0 && mvRef.MV.Col == 0 {
				// With matching grids, the source position is always admissible.
				// A zero vector projects back to that origin for every valid scale.
				f.Entries[destinationOriginRow+blkCol] = TemporalMotionEntry{
					MV:             mvRef.MV,
					RefFrameOffset: projection.refFrameOffset,
					Valid:          true,
				}
				continue
			}

			projectedRow := motionFieldProjectComponent(mvRef.MV.Row, projection.scale)
			rowOffset := motionFieldBlockOffset(projectedRow)
			row := blkRow + rowOffset
			if req.Backward {
				row = blkRow - rowOffset
			}
			if row < rowMin || row >= rowMax {
				continue
			}

			projectedCol := motionFieldProjectComponent(mvRef.MV.Col, projection.scale)
			colOffset := motionFieldBlockOffset(projectedCol)
			baseBlkCol := (blkCol >> 3) << 3
			colMin := max(0, baseBlkCol-(motionFieldMaxOffsetWidth>>3))
			colMax := min(cols, baseBlkCol+8+(motionFieldMaxOffsetWidth>>3))
			col := blkCol + colOffset
			if req.Backward {
				col = blkCol - colOffset
			}
			if col < colMin || col >= colMax {
				continue
			}

			f.Entries[row*stride+col] = TemporalMotionEntry{
				MV:             mvRef.MV,
				RefFrameOffset: projection.refFrameOffset,
				Valid:          true,
			}
		}
	}
	return true, nil
}

func (f *TemporalMotionField) projectSetupReference(req TemporalMotionSetupRequest, ref ReferenceFrame, backward bool) (bool, error) {
	start := req.References[ref]
	if start.Frame == nil || start.IntraOnly {
		return false, nil
	}
	return f.ProjectReferenceFrame(TemporalMotionProjectionRequest{
		StartFrame:         start.Frame,
		OrderHintBits:      req.OrderHintBits,
		CurrentOrderHint:   req.CurrentOrderHint,
		StartOrderHint:     start.OrderHint,
		StartRefOrderHints: start.RefOrderHints,
		Backward:           backward,
	})
}

func (f *TemporalMotionField) validate() error {
	if f == nil || f.Rows <= 0 || f.Cols <= 0 || f.Stride < f.Cols ||
		len(f.Entries) < (int(f.Rows)-1)*int(f.Stride)+int(f.Cols) {
		return ErrInvalidDecodeState
	}
	return nil
}

func validateReferenceMVFrame(f *ReferenceMVFrame) error {
	return f.Validate()
}

func temporalReferenceOrderHints(refs [referenceFrameCount]TemporalMotionReferenceFrame) [referenceFrameCount]uint8 {
	var hints [referenceFrameCount]uint8
	for ref := range referenceFrameCount {
		hints[ref] = refs[ref].OrderHint
	}
	return hints
}

func temporalReferenceFuture(bits uint8, ref uint8, current uint8) (bool, error) {
	distance, err := motionFieldRelativeOrderHint(bits, ref, current)
	if err != nil {
		return false, err
	}
	return distance > 0, nil
}

func motionFieldRefOffsets(bits uint8, start uint8, refs [referenceFrameCount]uint8) ([referenceFrameCount]int, error) {
	var offsets [referenceFrameCount]int
	for ref := range referenceFrameCount {
		offset, err := motionFieldRelativeOrderHint(bits, start, refs[ref])
		if err != nil {
			return offsets, err
		}
		offsets[ref] = offset
	}
	return offsets, nil
}

type motionFieldProjectionReference struct {
	scale          int64
	refFrameOffset uint8
	valid          bool
}

func motionFieldProjectionReferences(startToCurrent int, refOffsets [referenceFrameCount]int) [referenceFrameCount]motionFieldProjectionReference {
	var refs [referenceFrameCount]motionFieldProjectionReference
	if absInt(startToCurrent) > motionFieldMaxFrameDistance {
		return refs
	}
	for ref := range referenceFrameCount {
		offset := refOffsets[ref]
		if offset <= 0 || offset > motionFieldMaxFrameDistance {
			continue
		}
		refs[ref] = motionFieldProjectionReference{
			scale:          int64(startToCurrent) * int64(motionFieldDivMult[offset]),
			refFrameOffset: uint8(offset),
			valid:          true,
		}
	}
	return refs
}

func motionFieldRelativeOrderHint(bits uint8, a uint8, b uint8) (int, error) {
	if bits == 0 || bits > 8 {
		return 0, ErrInvalidDecodeState
	}
	limit := uint32(1) << bits
	ua := uint32(a)
	ub := uint32(b)
	if ua >= limit || ub >= limit {
		return 0, ErrInvalidDecodeState
	}
	mask := int32(1 << (bits - 1))
	diff := int32(ua) - int32(ub)
	return int((diff & (mask - 1)) - (diff & mask)), nil
}

func motionFieldProjectMV(ref motion.Vector, num int, den int) (motion.Vector, error) {
	if den <= 0 {
		return motion.Vector{}, ErrInvalidDecodeState
	}
	if den > motionFieldMaxFrameDistance {
		den = motionFieldMaxFrameDistance
	}
	if num > motionFieldMaxFrameDistance {
		num = motionFieldMaxFrameDistance
	} else if num < -motionFieldMaxFrameDistance {
		num = -motionFieldMaxFrameDistance
	}
	scale := int64(num) * int64(motionFieldDivMult[den])
	return motionFieldProjectMVScale(ref, scale), nil
}

func motionFieldProjectMVScale(ref motion.Vector, scale int64) motion.Vector {
	return motion.Vector{
		Row: motionFieldProjectComponent(ref.Row, scale),
		Col: motionFieldProjectComponent(ref.Col, scale),
	}
}

func motionFieldProjectComponent(v int16, scale int64) int16 {
	projected := roundPowerOfTwoSigned(int64(v)*scale, 14)
	return int16(clampInt64(projected, motionFieldMVLower+1, motionFieldMVUpper-1))
}

func motionFieldBlockPosition(rows int, cols int, blkRow int, blkCol int, mv motion.Vector, backward bool) (int, int, bool) {
	baseBlkRow := (blkRow >> 3) << 3
	baseBlkCol := (blkCol >> 3) << 3
	rowOffset := motionFieldBlockOffset(mv.Row)
	colOffset := motionFieldBlockOffset(mv.Col)

	row := blkRow + rowOffset
	col := blkCol + colOffset
	if backward {
		row = blkRow - rowOffset
		col = blkCol - colOffset
	}

	if row < 0 || row >= rows || col < 0 || col >= cols {
		return 0, 0, false
	}
	if row < baseBlkRow-(motionFieldMaxOffsetHeight>>3) ||
		row >= baseBlkRow+8+(motionFieldMaxOffsetHeight>>3) ||
		col < baseBlkCol-(motionFieldMaxOffsetWidth>>3) ||
		col >= baseBlkCol+8+(motionFieldMaxOffsetWidth>>3) {
		return 0, 0, false
	}
	return row, col, true
}

func motionFieldBlockOffset(v int16) int {
	if v < 0 {
		return -int((-int64(v)) >> (4 + 2))
	}
	return int(v >> (4 + 2))
}

func roundPowerOfTwoSigned(value int64, bits uint) int64 {
	if value < 0 {
		return -((-value + (1 << (bits - 1))) >> bits)
	}
	return (value + (1 << (bits - 1))) >> bits
}

func clampInt64(v int64, lo int64, hi int64) int64 {
	return min(max(v, lo), hi)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
