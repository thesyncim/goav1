// Ported from libaom: av1/common/enums.h
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package obu

// Type is the AV1 obu_type field.
type Type uint8

const (
	TypeReserved             Type = 0
	TypeSequenceHeader       Type = 1
	TypeTemporalDelimiter    Type = 2
	TypeFrameHeader          Type = 3
	TypeTileGroup            Type = 4
	TypeMetadata             Type = 5
	TypeFrame                Type = 6
	TypeRedundantFrameHeader Type = 7
	TypeTileList             Type = 8
	TypePadding              Type = 15
)

func (t Type) String() string {
	switch t {
	case TypeSequenceHeader:
		return "OBU_SEQUENCE_HEADER"
	case TypeTemporalDelimiter:
		return "OBU_TEMPORAL_DELIMITER"
	case TypeFrameHeader:
		return "OBU_FRAME_HEADER"
	case TypeTileGroup:
		return "OBU_TILE_GROUP"
	case TypeMetadata:
		return "OBU_METADATA"
	case TypeFrame:
		return "OBU_FRAME"
	case TypeRedundantFrameHeader:
		return "OBU_REDUNDANT_FRAME_HEADER"
	case TypeTileList:
		return "OBU_TILE_LIST"
	case TypePadding:
		return "OBU_PADDING"
	default:
		return "OBU_RESERVED"
	}
}

// Reserved reports whether t is reserved by the AV1 bitstream specification.
func (t Type) Reserved() bool {
	return t == TypeReserved || (t >= 9 && t <= 14)
}
