package goav1

import (
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
)

// OBUType identifies one of the AV1 Open Bitstream Unit types defined in
// section 5.3.1 of the AV1 specification.
type OBUType = internalobu.Type

// AV1 OBU type constants. They correspond one-to-one to the obu_type values
// in the AV1 specification.
const (
	OBUReserved             OBUType = internalobu.TypeReserved
	OBUSequenceHeader       OBUType = internalobu.TypeSequenceHeader
	OBUTemporalDelimiter    OBUType = internalobu.TypeTemporalDelimiter
	OBUFrameHeader          OBUType = internalobu.TypeFrameHeader
	OBUTileGroup            OBUType = internalobu.TypeTileGroup
	OBUMetadata             OBUType = internalobu.TypeMetadata
	OBUFrame                OBUType = internalobu.TypeFrame
	OBURedundantFrameHeader OBUType = internalobu.TypeRedundantFrameHeader
	OBUTileList             OBUType = internalobu.TypeTileList
	OBUPadding              OBUType = internalobu.TypePadding
)

// OBUHeader holds the parsed obu_header() syntax (type, extension, has_size
// flag, and optional temporal/spatial IDs).
type OBUHeader = internalobu.Header

// OBUUnit pairs an OBUHeader with the OBU payload bytes. The Payload slice
// aliases the buffer supplied to the parser; callers must not retain it
// beyond that buffer's lifetime.
type OBUUnit = internalobu.Unit

// LowOverheadIterator walks a contiguous low-overhead OBU stream (with obu_size
// fields) one unit at a time without allocating.
type LowOverheadIterator = internalobu.LowOverheadIterator

// TemporalUnit groups the OBU bytes that belong to a single temporal unit:
// the temporal-delimiter OBU plus the following OBUs up to the next delimiter.
type TemporalUnit = internalobu.TemporalUnit

// TemporalUnitIterator walks an AV1 Section 5 stream temporal unit by temporal
// unit. It performs no allocations and returns slices that alias the input
// buffer.
type TemporalUnitIterator = internalobu.TemporalUnitIterator

// AnnexBUnit is one entry produced by AnnexBIterator: an OBU element with its
// length-prefixed Annex B framing already stripped.
type AnnexBUnit = internalobu.AnnexBUnit

// AnnexBIterator walks an AV1 Annex B length-delimited stream, yielding OBU
// units with their Annex B size prefixes consumed. It does not allocate.
type AnnexBIterator = internalobu.AnnexBIterator

// OBU parse errors. The ErrOBU* values are returned by the OBU parsers and
// iterators when the input is truncated, contains forbidden/reserved bits, or
// violates the Annex B / low-overhead framing rules.
var (
	ErrOBUShortHeader              = internalobu.ErrShortHeader
	ErrOBUForbiddenBit             = internalobu.ErrForbiddenBit
	ErrOBUReservedBit              = internalobu.ErrReservedBit
	ErrOBUInvalidType              = internalobu.ErrInvalidType
	ErrOBUInvalidAnnexB            = internalobu.ErrInvalidAnnexB
	ErrOBUMissingSizeField         = internalobu.ErrMissingSizeField
	ErrOBUMissingTemporalDelimiter = internalobu.ErrMissingTemporalDelimiter
	ErrOBUSizeMismatch             = internalobu.ErrSizeMismatch
	ErrOBUShortPayload             = internalobu.ErrShortPayload
)

// ParseOBUHeader parses an obu_header() at the start of src. It returns the
// parsed header and the number of bytes consumed.
func ParseOBUHeader(src []byte) (OBUHeader, int, error) {
	return internalobu.ParseHeader(src)
}

// PutOBUHeader serializes header into dst and returns the number of bytes
// written. The caller is responsible for sizing dst (at most 2 bytes).
func PutOBUHeader(dst []byte, header OBUHeader) (int, error) {
	return internalobu.PutHeader(dst, header)
}

// ParseOBUElement parses a single OBU whose size matches the length of
// element. It is the variant used for RTP elements which carry an OBU without
// an obu_size field.
func ParseOBUElement(element []byte) (OBUUnit, error) {
	return internalobu.ParseElement(element)
}

// ParseLowOverheadOBU parses one OBU with an obu_size field from the front of
// src and returns the parsed unit together with the number of bytes consumed.
func ParseLowOverheadOBU(src []byte) (OBUUnit, int, error) {
	return internalobu.ParseLowOverhead(src)
}

// NormalizeLowOverheadOBU rewrites raw (which may use the WebRTC-style
// obu_has_size_field=0 framing) into dst with obu_size fields restored, so the
// result can be consumed by the low-overhead parsers. It returns the number
// of bytes written.
func NormalizeLowOverheadOBU(dst []byte, raw []byte) (int, error) {
	return internalobu.NormalizeLowOverhead(dst, raw)
}

// NewLowOverheadIterator returns an iterator over a low-overhead OBU stream
// (one with obu_size fields). The iterator aliases src.
func NewLowOverheadIterator(src []byte) LowOverheadIterator {
	return internalobu.NewLowOverheadIterator(src)
}

// NewTemporalUnitIterator returns an iterator over an AV1 Section 5 stream.
// Each TemporalUnit it yields aliases src.
func NewTemporalUnitIterator(src []byte) TemporalUnitIterator {
	return internalobu.NewTemporalUnitIterator(src)
}

// ParseAnnexBElement parses one length-prefixed Annex B element from the front
// of src and returns the parsed OBU together with the number of bytes
// consumed (including the length prefix).
func ParseAnnexBElement(src []byte) (OBUUnit, int, error) {
	return internalobu.ParseAnnexBElement(src)
}

// NewAnnexBIterator returns an iterator over an AV1 Annex B stream. The
// iterator aliases src.
func NewAnnexBIterator(src []byte) AnnexBIterator {
	return internalobu.NewAnnexBIterator(src)
}

// Metadata OBU public types and constants. These mirror the parsed payload
// variants defined by AV1 spec section 5.8.
type (
	// MetadataType is the AV1 metadata_type leb128 field.
	MetadataType = internalobu.MetadataType

	// Metadata is the parsed AV1 Metadata OBU payload. Type selects which of
	// the embedded variant fields is populated.
	Metadata = internalobu.Metadata

	// MetadataITUTT35 is the parsed ITU-T T.35 metadata payload, used for
	// closed captions, HDR10+, Dolby Vision and other registered payloads.
	MetadataITUTT35 = internalobu.MetadataITUTT35

	// MetadataHDRCLL is the parsed Content Light Level Info payload.
	MetadataHDRCLL = internalobu.MetadataHDRCLL

	// MetadataHDRMDCV is the parsed Mastering Display Color Volume payload.
	MetadataHDRMDCV = internalobu.MetadataHDRMDCV

	// MetadataScalability is the parsed scalability descriptor payload.
	MetadataScalability = internalobu.MetadataScalability

	// MetadataScalabilityStructure carries the optional
	// scalability_structure() body present when ModeIDC equals
	// MetadataScalabilityModeSS.
	MetadataScalabilityStructure = internalobu.ScalabilityStructure

	// MetadataScalabilityTemporalGroupEntry is one entry of the
	// scalability_structure temporal-group description.
	MetadataScalabilityTemporalGroupEntry = internalobu.ScalabilityTemporalGroupEntry

	// MetadataTimecode is the parsed metadata_timecode() payload.
	MetadataTimecode = internalobu.MetadataTimecode
)

// AV1 metadata_type values per spec section 6.7.1.
const (
	MetadataTypeReserved0   MetadataType = internalobu.MetadataTypeReserved0
	MetadataTypeITUTT35     MetadataType = internalobu.MetadataTypeITUTT35
	MetadataTypeHDRCLL      MetadataType = internalobu.MetadataTypeHDRCLL
	MetadataTypeHDRMDCV     MetadataType = internalobu.MetadataTypeHDRMDCV
	MetadataTypeScalability MetadataType = internalobu.MetadataTypeScalability
	MetadataTypeTimecode    MetadataType = internalobu.MetadataTypeTimecode

	// MetadataScalabilityModeSS is the scalability_mode_idc value that
	// signals an explicit scalability_structure() in the payload.
	MetadataScalabilityModeSS = internalobu.ScalabilityModeSS
)

// Metadata OBU parse errors. ParseMetadataOBU returns these for syntactically
// invalid Metadata OBU payloads; callers should use errors.Is.
var (
	ErrMetadataShortPayload     = internalobu.ErrMetadataShortPayload
	ErrMetadataMissingTrailing  = internalobu.ErrMetadataMissingTrailing
	ErrMetadataInvalidTrailing  = internalobu.ErrMetadataInvalidTrailing
	ErrMetadataMissingCountry   = internalobu.ErrMetadataMissingCountry
	ErrMetadataMissingCountryEx = internalobu.ErrMetadataMissingCountryEx
	ErrMetadataInvalidReserved  = internalobu.ErrMetadataInvalidReserved
)

// ParseMetadataOBU parses the payload of an AV1 Metadata OBU (the bytes after
// the OBU header and obu_size leb128) per spec section 5.8. The returned
// Metadata aliases payload when the variant exposes byte slices (ITU-T T.35);
// parsing performs no allocations.
func ParseMetadataOBU(payload []byte) (Metadata, error) {
	return internalobu.ParseMetadata(payload)
}

// AnnexBFrameSpan describes one frame_unit boundary inside the source buffer
// passed to LowOverheadToAnnexB. Exported so callers can pre-allocate the
// frame-span scratch and avoid auxiliary allocations on the conversion path.
type AnnexBFrameSpan = internalobu.AnnexBFrameSpan

// AppendAnnexBOBU appends a single OBU element to dst using AV1 Annex B
// framing (a leb128 obu_unit_size prefix followed by the raw OBU bytes).
// dst is returned extended in place. Use AppendAnnexBFrameUnit /
// AppendAnnexBTemporalUnit for nested frame_unit / temporal_unit framing.
func AppendAnnexBOBU(dst []byte, raw []byte) []byte {
	return internalobu.AppendAnnexBOBU(dst, raw)
}

// AppendAnnexBFrameUnit appends one frame_unit to dst (a leb128 frame_unit_size
// prefix followed by one obu_unit_size-prefixed OBU per element of obus).
func AppendAnnexBFrameUnit(dst []byte, obus ...[]byte) []byte {
	return internalobu.AppendAnnexBFrameUnit(dst, obus...)
}

// AppendAnnexBTemporalUnit appends a temporal_unit (a slice of frame_units,
// each a slice of OBU raw byte slices) to dst using the Annex B framing per
// spec section 5.2.
func AppendAnnexBTemporalUnit(dst []byte, frameUnits [][][]byte) []byte {
	return internalobu.AppendAnnexBTemporalUnit(dst, frameUnits)
}

// LowOverheadToAnnexB converts a low-overhead OBU stream (the kind emitted by
// IVF / Section 5 temporal-unit framing) into Annex B framing per spec section
// 5.2. scratch is an optional caller-owned slice used for frame-span
// bookkeeping; when its capacity covers the number of frame_units in src, no
// auxiliary allocations occur.
func LowOverheadToAnnexB(dst []byte, src []byte, scratch []AnnexBFrameSpan) ([]byte, error) {
	return internalobu.LowOverheadToAnnexB(dst, src, scratch)
}

// AnnexBToLowOverhead converts an Annex B stream into the concatenated
// low-overhead OBU stream consumable by NewLowOverheadIterator. Each emitted
// OBU preserves the raw bytes carried in the source stream; only the Annex B
// size prefixes are stripped.
func AnnexBToLowOverhead(dst []byte, src []byte) ([]byte, error) {
	return internalobu.AnnexBToLowOverhead(dst, src)
}

// TileList OBU public types and constants per AV1 spec section 5.11.1.
type (
	// TileList is the parsed OBU_TILE_LIST payload. Entries aliases the
	// payload bytes supplied to ParseTileListOBU.
	TileList = internalparser.TileList

	// TileListEntry describes one entry of a tile_list_obu(). TileData
	// aliases the payload bytes supplied to ParseTileListOBU.
	TileListEntry = internalparser.TileListEntry
)

// Tile list OBU constants per AV1 spec section 5.11.1.
const (
	// TileListHeaderBytes is the fixed-size tile_list_obu() header preceding
	// per-tile entries.
	TileListHeaderBytes = internalparser.TileListHeaderBytes

	// TileListEntryHeaderBytes is the fixed-size per-tile prefix preceding
	// each tile_list_obu() entry's coded payload.
	TileListEntryHeaderBytes = internalparser.TileListEntryHeaderBytes

	// TileListMaxExternalReferences mirrors MAX_EXTERNAL_REFERENCES in libaom.
	TileListMaxExternalReferences = internalparser.TileListMaxExternalReferences

	// TileListMaxTiles mirrors MAX_TILES in libaom and bounds the number of
	// entries one tile_list_obu() may carry.
	TileListMaxTiles = internalparser.TileListMaxTiles
)

// Tile list OBU parse errors. ParseTileListOBU returns these for invalid
// OBU_TILE_LIST payloads; callers should use errors.Is.
var (
	ErrTileListShortHeader        = internalparser.ErrTileListShortHeader
	ErrTileListShortEntry         = internalparser.ErrTileListShortEntry
	ErrTileListShortTileData      = internalparser.ErrTileListShortTileData
	ErrTileListTrailingBytes      = internalparser.ErrTileListTrailingBytes
	ErrTileListTooManyTiles       = internalparser.ErrTileListTooManyTiles
	ErrTileListInvalidTileCount   = internalparser.ErrTileListInvalidTileCount
	ErrTileListInvalidAnchorIndex = internalparser.ErrTileListInvalidAnchorIndex
	ErrTileListInvalidAnchorTile  = internalparser.ErrTileListInvalidAnchorTile
)

// ParseTileListOBU parses an OBU_TILE_LIST payload per AV1 spec section
// 5.11.1. payload is the slice carried in OBUUnit.Payload (the OBU header and
// obu_size prefix already stripped).
//
// entries is a caller-provided scratch slice used to materialise the per-tile
// entries. If its capacity covers tile_count_minus_1+1, parsing is
// zero-allocation; otherwise ParseTileListOBU allocates the result slice. The
// returned TileList.Entries aliases entries (or the allocated slice); the
// TileData fields alias payload.
func ParseTileListOBU(payload []byte, entries []TileListEntry) (TileList, error) {
	return internalparser.ParseTileListOBU(payload, entries)
}

// ValidateTileListAnchors checks tile_list_obu() anchor_tile_row/col fields
// against the tile grid of the active external reference frame. The tile-list
// output dimensions describe the destination mosaic; they are not the bound
// for anchor tile coordinates.
func ValidateTileListAnchors(list TileList, tiles TileInfo) error {
	return internalparser.ValidateTileListAnchors(list, tiles)
}

// AppendTileListOBU serialises list into dst as the bytes that would appear
// in the OBU_TILE_LIST payload (the obu_size prefix is not emitted). It is
// the inverse of ParseTileListOBU and the standard helper callers use when
// constructing a tile_list_obu() from a fan-out of decoded tiles.
func AppendTileListOBU(dst []byte, list TileList) []byte {
	return internalparser.AppendTileListOBU(dst, list)
}
