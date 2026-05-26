// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package obu

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// NormalizeLowOverhead writes raw as one low-overhead OBU with obu_size present.
//
// RTP payloads normally clear obu_has_size_field. WebRTC's AV1 depacketizer
// reconstructs decoder input by setting the size bit, writing leb128 payload
// size, and copying the OBU payload. This function is the same byte-level
// primitive over one complete OBU element.
func NormalizeLowOverhead(dst []byte, raw []byte) (int, error) {
	unit, err := ParseElement(raw)
	if err != nil {
		return 0, err
	}
	if len(unit.Raw) != len(raw) {
		return 0, ErrSizeMismatch
	}

	header := unit.Header
	header.HasSizeField = true

	off, err := PutHeader(dst, header)
	if err != nil {
		return 0, err
	}
	n, err := bitstream.PutLEB128(dst[off:], uint32(len(unit.Payload)))
	if err != nil {
		return 0, err
	}
	off += n
	if len(dst)-off < len(unit.Payload) {
		return 0, bitstream.ErrShortBuffer
	}
	off += copy(dst[off:], unit.Payload)
	return off, nil
}
