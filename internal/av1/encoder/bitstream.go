package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// OBU describes one encoder-emitted low-overhead OBU. Payload aliases caller
// memory and is copied into the output by AppendLowOverheadOBU.
type OBU struct {
	Type       obu.Type
	TemporalID uint8
	SpatialID  uint8
	Payload    []byte
}

// LowOverheadOBUSize returns the exact number of bytes required to emit unit as
// one low-overhead OBU with obu_has_size_field set.
func LowOverheadOBUSize(unit OBU) (int, error) {
	if err := validateOBU(unit); err != nil {
		return 0, err
	}
	return lowOverheadOBUSizeUnchecked(unit), nil
}

// AppendLowOverheadOBU appends unit as one low-overhead OBU with
// obu_has_size_field set. The function never grows dst: callers must reserve
// capacity first or use LowOverheadOBUSize to size a reusable buffer.
func AppendLowOverheadOBU(dst []byte, unit OBU) ([]byte, error) {
	if err := validateOBU(unit); err != nil {
		return dst, err
	}
	size := lowOverheadOBUSizeUnchecked(unit)
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         unit.Type,
		Extension:    unit.TemporalID != 0 || unit.SpatialID != 0,
		HasSizeField: true,
		TemporalID:   unit.TemporalID,
		SpatialID:    unit.SpatialID,
	})
	if err != nil {
		return dst, err
	}
	off += n
	n, err = bitstream.PutLEB128(out[off:], uint32(len(unit.Payload)))
	if err != nil {
		return dst, err
	}
	off += n
	copy(out[off:], unit.Payload)
	return out, nil
}

// LowOverheadTemporalUnitSize returns the exact number of bytes required to emit
// a low-overhead temporal unit: one temporal-delimiter OBU followed by obus.
func LowOverheadTemporalUnitSize(obus []OBU) (int, error) {
	if len(obus) == 0 {
		return 0, ErrInvalidFrame
	}
	size := lowOverheadOBUSizeUnchecked(OBU{Type: obu.TypeTemporalDelimiter})
	for i := range obus {
		unit := obus[i]
		if unit.Type == obu.TypeTemporalDelimiter {
			return 0, ErrInvalidFrame
		}
		if err := validateOBU(unit); err != nil {
			return 0, err
		}
		size += lowOverheadOBUSizeUnchecked(unit)
	}
	return size, nil
}

// AppendLowOverheadTemporalUnit appends one temporal-delimiter OBU followed by
// obus. The function never grows dst and validates the full unit before writing,
// so errors leave dst length unchanged.
func AppendLowOverheadTemporalUnit(dst []byte, obus []OBU) ([]byte, error) {
	size, err := LowOverheadTemporalUnitSize(obus)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}
	out, err := AppendLowOverheadOBU(dst, OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		return dst, err
	}
	for i := range obus {
		out, err = AppendLowOverheadOBU(out, obus[i])
		if err != nil {
			return dst, err
		}
	}
	return out, nil
}

func validateOBU(unit OBU) error {
	if unit.Type > 15 || unit.Type.Reserved() {
		return ErrInvalidFrame
	}
	if unit.TemporalID > 7 || unit.SpatialID > 3 {
		return ErrInvalidFrame
	}
	if uint64(len(unit.Payload)) > bitstream.MaxLEB128Value {
		return bitstream.ErrLEB128Overflow
	}
	return nil
}

func lowOverheadOBUSizeUnchecked(unit OBU) int {
	headerLen := 1
	if unit.TemporalID != 0 || unit.SpatialID != 0 {
		headerLen = 2
	}
	return headerLen + bitstream.LEB128Len(uint32(len(unit.Payload))) + len(unit.Payload)
}
