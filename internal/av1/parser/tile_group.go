// Ported from libaom:
//   av1/decoder/decodeframe.c
//   av1/decoder/obu.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const MaxTiles = MaxTileRows * MaxTileCols

// TileGroup is the tile_group_obu() header state. DataOffset points at the
// first tile payload byte inside the OBU payload passed to ParseTileGroupHeader.
type TileGroup struct {
	StartTile              uint16
	EndTile                uint16
	TileCount              uint16
	NextTile               uint16
	TileStartAndEndPresent bool
	Final                  bool
	HeaderBits             uint8
	BitsRead               uint16
	DataOffset             int
	DataSize               int
}

// TileSpan identifies one tile payload inside a tile group. Callers provide the
// output slice so packet ingestion and decode scheduling do not allocate.
type TileSpan struct {
	Tile   uint16
	Row    uint8
	Col    uint8
	Offset int
	Size   int
}

// ParseTileGroupHeader parses the header and byte-alignment bits of
// tile_group_obu(). startBits is zero for OBU_TILE_GROUP and the frame header
// bit count for OBU_FRAME.
func ParseTileGroupHeader(payload []byte, tiles TileInfo, startBits int, expectedStart uint16, frameOBU bool) (TileGroup, error) {
	if tiles.Cols == 0 || tiles.Rows == 0 {
		return TileGroup{}, ErrInvalidTileGroup
	}
	numTiles := uint16(tiles.Cols) * uint16(tiles.Rows)
	if numTiles == 0 || numTiles > MaxTiles {
		return TileGroup{}, ErrInvalidTileGroup
	}

	// In an OBU_FRAME, frame_obu() performs a byte_alignment() between
	// frame_header_obu() and tile_group_obu() (spec 5.10.1). startBits is the
	// raw frame-header bit count, which is not necessarily byte aligned (e.g.
	// asymmetric tile layouts), so round it up to the next byte boundary before
	// the tile group begins. Standalone OBU_TILE_GROUP payloads always start
	// byte aligned (startBits == 0) so this is a no-op there.
	if frameOBU {
		startBits = (startBits + 7) &^ 7
	}

	r := bitstream.NewReader(payload)
	if err := r.SkipBits(startBits); err != nil {
		return TileGroup{}, err
	}
	headerStart := r.BitsRead()

	var group TileGroup
	if numTiles > 1 {
		present, err := r.ReadBool()
		if err != nil {
			return TileGroup{}, err
		}
		group.TileStartAndEndPresent = present
		if frameOBU && present {
			return TileGroup{}, ErrInvalidTileGroup
		}
	}

	if numTiles == 1 || !group.TileStartAndEndPresent {
		group.StartTile = 0
		group.EndTile = numTiles - 1
	} else {
		tileBits := tiles.Log2Cols + tiles.Log2Rows
		if tileBits == 0 {
			return TileGroup{}, ErrInvalidTileGroup
		}
		v, err := r.ReadBits(tileBits)
		if err != nil {
			return TileGroup{}, err
		}
		group.StartTile = uint16(v)
		v, err = r.ReadBits(tileBits)
		if err != nil {
			return TileGroup{}, err
		}
		group.EndTile = uint16(v)
	}

	if group.StartTile != expectedStart || group.StartTile > group.EndTile || group.EndTile >= numTiles {
		return TileGroup{}, ErrInvalidTileGroup
	}
	if err := readByteAlignment(&r); err != nil {
		return TileGroup{}, err
	}

	group.TileCount = group.EndTile - group.StartTile + 1
	group.Final = group.EndTile == numTiles-1
	if group.Final {
		group.NextTile = 0
	} else {
		group.NextTile = group.EndTile + 1
	}
	group.HeaderBits = uint8(r.BitsRead() - headerStart)
	bitsRead, err := readerBitsRead(&r)
	if err != nil {
		return TileGroup{}, err
	}
	group.BitsRead = bitsRead
	group.DataOffset = int(group.BitsRead) >> 3
	group.DataSize = len(payload) - group.DataOffset
	if group.DataSize < 0 {
		return TileGroup{}, ErrInvalidTileGroup
	}
	return group, nil
}

// SplitTileGroup writes tile payload spans for group into spans and returns the
// number of spans written.
func SplitTileGroup(payload []byte, tiles TileInfo, group TileGroup, spans []TileSpan) (int, error) {
	count := int(group.TileCount)
	if count == 0 || len(spans) < count || group.DataOffset < 0 || group.DataOffset > len(payload) {
		return 0, ErrInvalidTileGroup
	}
	offset := group.DataOffset
	remaining := len(payload) - offset
	for i := range count {
		tile := group.StartTile + uint16(i)
		size := remaining
		if i != count-1 {
			if tiles.TileSizeBytes == 0 || remaining < int(tiles.TileSizeBytes) {
				return 0, ErrInvalidTileGroup
			}
			v := uint32(0)
			for j := uint8(0); j < tiles.TileSizeBytes; j++ {
				v |= uint32(payload[offset+int(j)]) << (8 * uint(j))
			}
			offset += int(tiles.TileSizeBytes)
			remaining -= int(tiles.TileSizeBytes)
			size = int(v) + 1
			if size > remaining {
				return 0, ErrInvalidTileGroup
			}
		}
		spans[i] = TileSpan{
			Tile:   tile,
			Row:    uint8(tile / uint16(tiles.Cols)),
			Col:    uint8(tile % uint16(tiles.Cols)),
			Offset: offset,
			Size:   size,
		}
		offset += size
		remaining -= size
	}
	if remaining != 0 {
		return 0, ErrInvalidTileGroup
	}
	return count, nil
}

func readByteAlignment(r *bitstream.Reader) error {
	for !r.ByteAligned() {
		b, err := r.ReadBit()
		if err != nil {
			return err
		}
		if b != 0 {
			return ErrInvalidTileGroup
		}
	}
	return nil
}
