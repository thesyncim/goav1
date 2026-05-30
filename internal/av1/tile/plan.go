package tile

import "github.com/thesyncim/goav1/internal/av1/parser"

// Job is one deterministic tile decode work item. Coordinates are in
// superblocks so reconstruction can map them directly to row/column workers.
type Job struct {
	Tile uint16
	Row  uint8
	Col  uint8

	SBX    uint16
	SBY    uint16
	SBCols uint16
	SBRows uint16

	Offset int
	Size   int

	LastInRow bool
	LastRow   bool

	// UpdatesFrameContext marks the context_update_tile_id tile when
	// refresh_context is enabled. Decode workers can use this to retain only the
	// designated tile's adapted entropy state after all tiles finish.
	UpdatesFrameContext bool
}

// BuildJobs converts tile payload spans into deterministic work items. dst is
// caller-owned so the decode path can reuse one buffer across frames.
func BuildJobs(dst []Job, tiles parser.TileInfo, spans []parser.TileSpan) (int, error) {
	count := len(spans)
	if count == 0 {
		return 0, nil
	}
	if len(dst) < count {
		return 0, ErrJobBufferTooSmall
	}
	if tiles.Cols == 0 || tiles.Rows == 0 {
		return 0, ErrInvalidPlan
	}

	numTiles := uint16(tiles.Cols) * uint16(tiles.Rows)
	if tiles.RefreshContext && tiles.ContextUpdateTileID >= numTiles {
		return 0, ErrInvalidPlan
	}
	var previous uint16
	for i := range count {
		span := spans[i]
		if span.Tile >= numTiles || span.Offset < 0 || span.Size < 0 {
			return 0, ErrInvalidPlan
		}
		if i != 0 && span.Tile != previous+1 {
			return 0, ErrInvalidPlan
		}

		row := uint8(span.Tile / uint16(tiles.Cols))
		col := uint8(span.Tile % uint16(tiles.Cols))
		if span.Row != row || span.Col != col || row >= tiles.Rows || col >= tiles.Cols {
			return 0, ErrInvalidPlan
		}

		sbX := tiles.ColStartSB[col]
		sbY := tiles.RowStartSB[row]
		sbXEnd := tiles.ColStartSB[col+1]
		sbYEnd := tiles.RowStartSB[row+1]
		if sbXEnd <= sbX || sbYEnd <= sbY || sbXEnd > tiles.SBCols || sbYEnd > tiles.SBRows {
			return 0, ErrInvalidPlan
		}

		dst[i] = Job{
			Tile:      span.Tile,
			Row:       row,
			Col:       col,
			SBX:       sbX,
			SBY:       sbY,
			SBCols:    sbXEnd - sbX,
			SBRows:    sbYEnd - sbY,
			Offset:    span.Offset,
			Size:      span.Size,
			LastInRow: col == tiles.Cols-1,
			LastRow:   row == tiles.Rows-1,

			UpdatesFrameContext: tiles.RefreshContext && span.Tile == tiles.ContextUpdateTileID,
		}
		previous = span.Tile
	}
	return count, nil
}
