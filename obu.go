package goav1

import internalobu "github.com/thesyncim/goav1/internal/av1/obu"

type OBUType = internalobu.Type

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

type OBUHeader = internalobu.Header
type OBUUnit = internalobu.Unit
type LowOverheadIterator = internalobu.LowOverheadIterator

var (
	ErrOBUShortHeader      = internalobu.ErrShortHeader
	ErrOBUForbiddenBit     = internalobu.ErrForbiddenBit
	ErrOBUReservedBit      = internalobu.ErrReservedBit
	ErrOBUMissingSizeField = internalobu.ErrMissingSizeField
	ErrOBUShortPayload     = internalobu.ErrShortPayload
)

func ParseOBUHeader(src []byte) (OBUHeader, int, error) {
	return internalobu.ParseHeader(src)
}

func PutOBUHeader(dst []byte, header OBUHeader) (int, error) {
	return internalobu.PutHeader(dst, header)
}

func ParseOBUElement(element []byte) (OBUUnit, error) {
	return internalobu.ParseElement(element)
}

func ParseLowOverheadOBU(src []byte) (OBUUnit, int, error) {
	return internalobu.ParseLowOverhead(src)
}

func NewLowOverheadIterator(src []byte) LowOverheadIterator {
	return internalobu.NewLowOverheadIterator(src)
}
