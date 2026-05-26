package goav1

import internalobu "github.com/thesyncim/goav1/internal/av1/obu"

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
