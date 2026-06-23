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

// WebRTCScalabilityMetadataPayloadSize returns the metadata_obu() payload size
// for mode's predefined AV1 scalability_mode_idc. ok is false when the WebRTC
// mode has no predefined AV1 IDC and requires an explicit SS structure instead.
func WebRTCScalabilityMetadataPayloadSize(mode ScalabilityMode) (size int, ok bool) {
	if _, ok := WebRTCScalabilityModeIDC(mode); !ok {
		return 0, false
	}
	return bitstream.LEB128Len(uint32(obu.MetadataTypeScalability)) + 1 + 1, true
}

// AppendWebRTCScalabilityMetadataPayload appends metadata_obu() payload bytes
// for mode's predefined AV1 scalability_mode_idc without growing dst.
func AppendWebRTCScalabilityMetadataPayload(dst []byte, mode ScalabilityMode) ([]byte, bool, error) {
	idc, ok := WebRTCScalabilityModeIDC(mode)
	if !ok {
		return dst, false, nil
	}
	size, _ := WebRTCScalabilityMetadataPayloadSize(mode)
	if cap(dst)-len(dst) < size {
		return dst, true, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	n, err := bitstream.PutLEB128(out[off:], uint32(obu.MetadataTypeScalability))
	if err != nil {
		return dst, true, err
	}
	off += n
	out[off] = idc
	off++
	out[off] = 0x80
	return out, true, nil
}

// LowOverheadWebRTCScalabilityMetadataOBUSize returns the OBU size for mode's
// predefined AV1 scalability metadata. ok is false when no predefined IDC
// exists for mode.
func LowOverheadWebRTCScalabilityMetadataOBUSize(mode ScalabilityMode) (size int, ok bool, err error) {
	payloadSize, ok := WebRTCScalabilityMetadataPayloadSize(mode)
	if !ok {
		return 0, false, nil
	}
	return 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize, true, nil
}

// AppendLowOverheadWebRTCScalabilityMetadataOBU appends one low-overhead
// METADATA_TYPE_SCALABILITY OBU for mode's predefined AV1 IDC.
func AppendLowOverheadWebRTCScalabilityMetadataOBU(dst []byte, mode ScalabilityMode) ([]byte, bool, error) {
	payloadSize, ok := WebRTCScalabilityMetadataPayloadSize(mode)
	if !ok {
		return dst, false, nil
	}
	obuSize := 1 + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize
	if cap(dst)-len(dst) < obuSize {
		return dst, true, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+obuSize]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         obu.TypeMetadata,
		HasSizeField: true,
	})
	if err != nil {
		return dst, true, err
	}
	off += n
	n, err = bitstream.PutLEB128(out[off:], uint32(payloadSize))
	if err != nil {
		return dst, true, err
	}
	off += n
	payload, _, err := AppendWebRTCScalabilityMetadataPayload(out[:off], mode)
	if err != nil {
		return dst, true, err
	}
	if len(payload) != len(out) {
		return dst, true, ErrInvalidFrame
	}
	return out, true, nil
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
