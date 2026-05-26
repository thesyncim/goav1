// Ported from libaom:
//   av1/common/obu_util.c
//   av1/decoder/obu.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package obu

// Header is the parsed AV1 OBU header and optional extension header.
type Header struct {
	Type         Type
	Extension    bool
	HasSizeField bool
	TemporalID   uint8
	SpatialID    uint8
}

// HeaderLen returns the encoded OBU header length.
func (h Header) HeaderLen() int {
	if h.Extension {
		return 2
	}
	return 1
}

// ParseHeader parses an OBU header and optional extension header.
func ParseHeader(src []byte) (Header, int, error) {
	if len(src) < 1 {
		return Header{}, 0, ErrShortHeader
	}

	b := src[0]
	if b&0x80 != 0 {
		return Header{}, 0, ErrForbiddenBit
	}
	if b&0x01 != 0 {
		return Header{}, 0, ErrReservedBit
	}

	h := Header{
		Type:         Type((b >> 3) & 0x0f),
		Extension:    b&0x04 != 0,
		HasSizeField: b&0x02 != 0,
	}

	if h.Extension {
		if len(src) < 2 {
			return Header{}, 0, ErrShortHeader
		}
		ext := src[1]
		if ext&0x07 != 0 {
			return Header{}, 0, ErrReservedBit
		}
		h.TemporalID = (ext >> 5) & 0x07
		h.SpatialID = (ext >> 3) & 0x03
		return h, 2, nil
	}

	return h, 1, nil
}

// PutHeader writes h into dst.
func PutHeader(dst []byte, h Header) (int, error) {
	if len(dst) < h.HeaderLen() {
		return 0, ErrShortHeader
	}
	if h.Type > 15 || h.TemporalID > 7 || h.SpatialID > 3 {
		return 0, ErrInvalidType
	}

	b := byte(h.Type) << 3
	if h.Extension {
		b |= 0x04
	}
	if h.HasSizeField {
		b |= 0x02
	}
	dst[0] = b

	if !h.Extension {
		return 1, nil
	}

	dst[1] = (h.TemporalID << 5) | (h.SpatialID << 3)
	return 2, nil
}
