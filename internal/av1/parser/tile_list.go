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
	for i := 0; i < tileCount; i++ {
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
