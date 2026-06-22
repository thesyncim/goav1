// Ported from libaom:
//   av1/decoder/obu.c (read_and_decode_one_tile_list)
//   AV1 specification section 5.11.1 (tile_list_obu)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "errors"

// Tile list OBU constants per AV1 spec section 5.11.1.
const (
	// TileListHeaderBytes is the fixed-size tile_list_obu() header preceding
	// the per-tile entries: output_frame_width_in_tiles_minus_1 (u8),
	// output_frame_height_in_tiles_minus_1 (u8), and tile_count_minus_1 (u16).
	TileListHeaderBytes = 4

	// TileListEntryHeaderBytes is the fixed-size per-tile prefix:
	// anchor_frame_idx (u8), anchor_tile_row (u8), anchor_tile_col (u8),
	// tile_data_size_minus_1 (u16). The encoded tile payload follows.
	TileListEntryHeaderBytes = 5

	// TileListMaxExternalReferences mirrors MAX_EXTERNAL_REFERENCES in libaom.
	// anchor_frame_idx values at or above this bound are invalid.
	TileListMaxExternalReferences = 128

	// TileListMaxTiles mirrors MAX_TILES in libaom and bounds the number of
	// entries a single tile_list_obu() may carry.
	TileListMaxTiles = 512
)

// Tile list OBU parse errors.
var (
	ErrTileListShortHeader        = errors.New("parser: tile list short header")
	ErrTileListShortEntry         = errors.New("parser: tile list short tile entry")
	ErrTileListShortTileData      = errors.New("parser: tile list short tile data")
	ErrTileListTrailingBytes      = errors.New("parser: tile list trailing bytes")
	ErrTileListTooManyTiles       = errors.New("parser: tile list tile_count exceeds output frame")
	ErrTileListInvalidTileCount   = errors.New("parser: tile list tile_count out of range")
	ErrTileListInvalidAnchorIndex = errors.New("parser: tile list anchor_frame_idx out of range")
	ErrTileListInvalidAnchorTile  = errors.New("parser: tile list anchor tile out of reference frame")
	ErrTileListNonUniformTileSize = errors.New("parser: tile list requires uniform tile size")
)

// TileListEntry describes one entry of a tile_list_obu(). TileData aliases the
// payload bytes supplied to ParseTileListOBU.
type TileListEntry struct {
	AnchorFrameIdx     uint8
	AnchorTileRow      uint8
	AnchorTileCol      uint8
	TileDataSizeMinus1 uint16
	TileData           []byte
}

// TileDataSize reports the decoded size of the per-tile payload
// (tile_data_size_minus_1 + 1).
func (e TileListEntry) TileDataSize() int {
	return int(e.TileDataSizeMinus1) + 1
}

// TileList is the parsed tile_list_obu() per AV1 spec section 5.11.1.
//
// Entries aliases the caller-provided payload buffer and never allocates on
// the steady-state path of ParseTileListOBU.
type TileList struct {
	OutputFrameWidthInTilesMinus1  uint8
	OutputFrameHeightInTilesMinus1 uint8
	TileCountMinus1                uint16
	Entries                        []TileListEntry
}

// OutputFrameWidthInTiles returns the decoded output_frame_width_in_tiles
// value (output_frame_width_in_tiles_minus_1 + 1).
func (l TileList) OutputFrameWidthInTiles() int {
	return int(l.OutputFrameWidthInTilesMinus1) + 1
}

// OutputFrameHeightInTiles returns the decoded output_frame_height_in_tiles
// value (output_frame_height_in_tiles_minus_1 + 1).
func (l TileList) OutputFrameHeightInTiles() int {
	return int(l.OutputFrameHeightInTilesMinus1) + 1
}

// TileCount returns the decoded tile_count value (tile_count_minus_1 + 1).
func (l TileList) TileCount() int {
	return int(l.TileCountMinus1) + 1
}

// ParseTileListOBU parses an OBU_TILE_LIST payload per AV1 spec section
// 5.11.1. payload is the slice carried in obu.Unit.Payload (the OBU header and
// obu_size prefix already stripped).
//
// entries is a caller-provided scratch slice used to materialise the per-tile
// entries. If it has insufficient capacity for tile_count_minus_1+1 entries,
// ParseTileListOBU allocates the result slice; otherwise parsing is
// zero-allocation. The returned TileList.Entries aliases entries (or the
// allocated slice). The TileData fields alias payload.
func ParseTileListOBU(payload []byte, entries []TileListEntry) (TileList, error) {
	var list TileList
	if len(payload) < TileListHeaderBytes {
		return list, ErrTileListShortHeader
	}

	list.OutputFrameWidthInTilesMinus1 = payload[0]
	list.OutputFrameHeightInTilesMinus1 = payload[1]
	list.TileCountMinus1 = uint16(payload[2])<<8 | uint16(payload[3])

	tileCount := int(list.TileCountMinus1) + 1
	if tileCount > TileListMaxTiles {
		return list, ErrTileListInvalidTileCount
	}
	outputCapacity := (int(list.OutputFrameWidthInTilesMinus1) + 1) *
		(int(list.OutputFrameHeightInTilesMinus1) + 1)
	if tileCount > outputCapacity {
		return list, ErrTileListTooManyTiles
	}

	if cap(entries) >= tileCount {
		entries = entries[:tileCount]
	} else {
		entries = make([]TileListEntry, tileCount)
	}

	off := TileListHeaderBytes
	for i := range tileCount {
		if off+TileListEntryHeaderBytes > len(payload) {
			return list, ErrTileListShortEntry
		}
		anchorIdx := payload[off]
		if anchorIdx >= TileListMaxExternalReferences {
			return list, ErrTileListInvalidAnchorIndex
		}
		entry := TileListEntry{
			AnchorFrameIdx:     anchorIdx,
			AnchorTileRow:      payload[off+1],
			AnchorTileCol:      payload[off+2],
			TileDataSizeMinus1: uint16(payload[off+3])<<8 | uint16(payload[off+4]),
		}
		off += TileListEntryHeaderBytes
		dataSize := int(entry.TileDataSizeMinus1) + 1
		end := off + dataSize
		if end < off || end > len(payload) {
			return list, ErrTileListShortTileData
		}
		entry.TileData = payload[off:end]
		entries[i] = entry
		off = end
	}

	if off != len(payload) {
		return list, ErrTileListTrailingBytes
	}

	list.Entries = entries
	return list, nil
}

// ValidateTileListAnchors checks tile_list_obu() anchor_tile_row/col fields
// against the tile grid of the active external reference frame.
//
// The tile-list output grid is only the destination mosaic. Libaom's
// read_and_decode_one_tile_list validates anchors against cm->tiles.rows/cols
// from the selected external reference frame, so callers should use this after
// the reference frame header has provided its TileInfo.
func ValidateTileListAnchors(list TileList, tiles TileInfo) error {
	if tiles.Cols == 0 || tiles.Rows == 0 {
		return ErrTileListInvalidAnchorTile
	}
	if list.TileCount() != len(list.Entries) {
		return ErrTileListInvalidTileCount
	}
	for _, entry := range list.Entries {
		if entry.AnchorTileRow >= tiles.Rows || entry.AnchorTileCol >= tiles.Cols {
			return ErrTileListInvalidAnchorTile
		}
	}
	return nil
}

// TileListUniformTileSize returns the reference frame tile width and height in
// superblocks when the tile grid can be used for tile-list output. It mirrors
// libaom's av1_get_uniform_tile_size(): uniform-spacing grids are accepted as
// encoded, while explicit grids must have equal widths and equal heights.
func TileListUniformTileSize(tiles TileInfo) (widthSB uint16, heightSB uint16, ok bool) {
	cols := int(tiles.Cols)
	rows := int(tiles.Rows)
	if cols <= 0 || rows <= 0 || cols > MaxTileCols || rows > MaxTileRows {
		return 0, 0, false
	}
	if tiles.ColStartSB[0] != 0 || tiles.RowStartSB[0] != 0 {
		return 0, 0, false
	}

	widthSB = tiles.ColStartSB[1] - tiles.ColStartSB[0]
	heightSB = tiles.RowStartSB[1] - tiles.RowStartSB[0]
	if widthSB == 0 || heightSB == 0 {
		return 0, 0, false
	}
	if tiles.UniformSpacing {
		return widthSB, heightSB, true
	}

	for i := 0; i < cols; i++ {
		start, end := tiles.ColStartSB[i], tiles.ColStartSB[i+1]
		if end <= start || end-start != widthSB {
			return 0, 0, false
		}
	}
	for i := 0; i < rows; i++ {
		start, end := tiles.RowStartSB[i], tiles.RowStartSB[i+1]
		if end <= start || end-start != heightSB {
			return 0, 0, false
		}
	}
	return widthSB, heightSB, true
}

// ValidateTileListDecodeLayout validates the source-shaped tile-list
// prerequisites known before reconstruction and returns the uniform reference
// tile size in superblocks. The caller still needs to decode each listed tile
// and blit it into a tile-list output frame.
func ValidateTileListDecodeLayout(list TileList, tiles TileInfo) (widthSB uint16, heightSB uint16, err error) {
	if err := ValidateTileListAnchors(list, tiles); err != nil {
		return 0, 0, err
	}
	widthSB, heightSB, ok := TileListUniformTileSize(tiles)
	if !ok {
		return 0, 0, ErrTileListNonUniformTileSize
	}
	return widthSB, heightSB, nil
}

// AppendTileListOBU serialises list into dst as the bytes that would appear in
// the OBU_TILE_LIST payload (the obu_size prefix is not emitted; callers wrap
// the result with obu.PutHeader / a leb128 size when constructing a full OBU).
//
// AppendTileListOBU returns dst extended in-place so callers can pre-size with
// AppendLEB128-style usage. The function performs no allocation when dst has
// sufficient capacity.
func AppendTileListOBU(dst []byte, list TileList) []byte {
	dst = append(dst,
		list.OutputFrameWidthInTilesMinus1,
		list.OutputFrameHeightInTilesMinus1,
		byte(list.TileCountMinus1>>8),
		byte(list.TileCountMinus1),
	)
	for _, entry := range list.Entries {
		dst = append(dst,
			entry.AnchorFrameIdx,
			entry.AnchorTileRow,
			entry.AnchorTileCol,
			byte(entry.TileDataSizeMinus1>>8),
			byte(entry.TileDataSizeMinus1),
		)
		dst = append(dst, entry.TileData...)
	}
	return dst
}
