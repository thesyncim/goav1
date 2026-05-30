package threading

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

const frameWorkCDEFUnitMI = 16

// FrameWorkCDEFIndexMap stores decoded cdef_idx values in frame-level 64x64
// CDEF-unit order. Index and Read are caller-owned so tile workers can fill the
// map without allocating in the block loop.
type FrameWorkCDEFIndexMap struct {
	Index  []uint8
	Read   []bool
	Stride int
	Rows   int
}

// Reset clears all decoded CDEF-unit state while preserving caller-owned
// storage and shape.
func (m FrameWorkCDEFIndexMap) Reset() error {
	if err := m.validate(); err != nil {
		return err
	}
	clear(m.Index[:m.Stride*m.Rows])
	clear(m.Read[:m.Stride*m.Rows])
	return nil
}

// CDEFIndexMapShape returns the frame-level 64x64 CDEF-unit grid dimensions.
func (b *FrameWorkBatch) CDEFIndexMapShape() (cols int, rows int, length int, err error) {
	if !b.Sequence.Valid() || b.FrameSize.CodedWidth == 0 || b.FrameSize.Height == 0 {
		return 0, 0, 0, ErrInvalidBatch
	}
	miCols, ok := frameWorkMIExtent(b.FrameSize.CodedWidth)
	if !ok {
		return 0, 0, 0, ErrInvalidBatch
	}
	miRows, ok := frameWorkMIExtent(b.FrameSize.Height)
	if !ok {
		return 0, 0, 0, ErrInvalidBatch
	}
	cdefCols := (miCols + frameWorkCDEFUnitMI - 1) / frameWorkCDEFUnitMI
	cdefRows := (miRows + frameWorkCDEFUnitMI - 1) / frameWorkCDEFUnitMI
	length64 := uint64(cdefCols) * uint64(cdefRows)
	maxInt := uint64(^uint(0) >> 1)
	if cdefCols == 0 || cdefRows == 0 || length64 > maxInt {
		return 0, 0, 0, ErrInvalidBatch
	}
	return int(cdefCols), int(cdefRows), int(length64), nil
}

// BindCDEFIndexMap validates and slices caller-owned storage for this frame.
func (b *FrameWorkBatch) BindCDEFIndexMap(index []uint8, read []bool) (FrameWorkCDEFIndexMap, error) {
	cols, rows, length, err := b.CDEFIndexMapShape()
	if err != nil {
		return FrameWorkCDEFIndexMap{}, err
	}
	limit, ok := frameWorkCDEFIndexLimit(b.CDEF)
	if !ok || len(index) < length || len(read) < length {
		return FrameWorkCDEFIndexMap{}, ErrInvalidBatch
	}
	index = index[:length]
	read = read[:length]
	for i, valid := range read {
		if valid && index[i] >= limit {
			return FrameWorkCDEFIndexMap{}, ErrInvalidBatch
		}
	}
	return FrameWorkCDEFIndexMap{
		Index:  index,
		Read:   read,
		Stride: cols,
		Rows:   rows,
	}, nil
}

// MarkBlock records the decoded cdef_idx for every 64x64 CDEF unit covered by
// visit. Skipped blocks leave the map untouched. Single-strength frames do not
// code cdef_idx syntax, but non-skipped units are still valid with index zero.
func (m FrameWorkCDEFIndexMap) MarkBlock(params parser.CDEFParams, visit tile.BlockLoopVisit) error {
	if err := m.validate(); err != nil {
		return err
	}
	limit, ok := frameWorkCDEFIndexLimit(params)
	if !ok {
		return ErrInvalidBatch
	}
	if visit.Prefix.SkipTransform {
		return nil
	}
	index := uint8(0)
	if params.Bits != 0 {
		index = visit.Prefix.CDEFIndex
		if index >= limit {
			return ErrInvalidBatch
		}
	}
	block := visit.Block
	if block.MIColEnd <= block.MICol || block.MIRowEnd <= block.MIRow {
		return ErrInvalidBatch
	}
	unitColStart := block.MICol / frameWorkCDEFUnitMI
	unitRowStart := block.MIRow / frameWorkCDEFUnitMI
	unitColEnd := (block.MIColEnd - 1) / frameWorkCDEFUnitMI
	unitRowEnd := (block.MIRowEnd - 1) / frameWorkCDEFUnitMI
	if unitColEnd >= uint32(m.Stride) || unitRowEnd >= uint32(m.Rows) {
		return ErrInvalidBatch
	}
	for unitRow := unitRowStart; unitRow <= unitRowEnd; unitRow++ {
		row := int(unitRow) * m.Stride
		for unitCol := unitColStart; unitCol <= unitColEnd; unitCol++ {
			offset := row + int(unitCol)
			m.Index[offset] = index
			m.Read[offset] = true
		}
	}
	return nil
}

func (m FrameWorkCDEFIndexMap) validate() error {
	if m.Stride <= 0 || m.Rows <= 0 {
		return ErrInvalidBatch
	}
	maxInt := int(^uint(0) >> 1)
	if m.Stride > maxInt/m.Rows ||
		uint64(m.Stride) > uint64(^uint32(0)) ||
		uint64(m.Rows) > uint64(^uint32(0)) {
		return ErrInvalidBatch
	}
	length := m.Stride * m.Rows
	if len(m.Index) < length || len(m.Read) < length {
		return ErrInvalidBatch
	}
	return nil
}

func frameWorkCDEFIndexLimit(params parser.CDEFParams) (uint8, bool) {
	if params.Bits > 3 {
		return 0, false
	}
	limit := uint8(1) << params.Bits
	if limit > parser.MaxCDEFStrengths {
		return 0, false
	}
	return limit, true
}
