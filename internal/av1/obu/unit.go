// Ported from libaom: av1/common/obu_util.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package obu

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// Unit is a parsed OBU framing view.
//
// Raw and Payload alias the caller-provided input bytes. Unit does not own the
// memory and parsing does not allocate.
type Unit struct {
	Header    Header
	HeaderLen uint8
	SizeLen   uint8
	Size      uint32
	Payload   []byte
	Raw       []byte
}

// ParseElement parses one RTP OBU element or otherwise length-delimited OBU.
//
// If the OBU has an internal size field, the size must fit inside element. If
// it has no size field, the payload is inferred from element length.
func ParseElement(element []byte) (Unit, error) {
	h, headerLen, err := ParseHeader(element)
	if err != nil {
		return Unit{}, err
	}

	if !h.HasSizeField {
		payload := element[headerLen:]
		return Unit{
			Header:    h,
			HeaderLen: uint8(headerLen),
			Size:      uint32(len(payload)),
			Payload:   payload,
			Raw:       element,
		}, nil
	}

	size, sizeLen, err := bitstream.ReadLEB128(element[headerLen:])
	if err != nil {
		return Unit{}, err
	}

	start := headerLen + sizeLen
	end := start + int(size)
	if end < start || end > len(element) {
		return Unit{}, ErrShortPayload
	}

	return Unit{
		Header:    h,
		HeaderLen: uint8(headerLen),
		SizeLen:   uint8(sizeLen),
		Size:      size,
		Payload:   element[start:end],
		Raw:       element[:end],
	}, nil
}

// ParseLowOverhead parses the next OBU from an AV1 low-overhead bitstream.
func ParseLowOverhead(src []byte) (Unit, int, error) {
	h, headerLen, err := ParseHeader(src)
	if err != nil {
		return Unit{}, 0, err
	}
	if !h.HasSizeField {
		return Unit{}, 0, ErrMissingSizeField
	}

	size, sizeLen, err := bitstream.ReadLEB128(src[headerLen:])
	if err != nil {
		return Unit{}, 0, err
	}

	start := headerLen + sizeLen
	end := start + int(size)
	if end < start || end > len(src) {
		return Unit{}, 0, ErrShortPayload
	}

	return Unit{
		Header:    h,
		HeaderLen: uint8(headerLen),
		SizeLen:   uint8(sizeLen),
		Size:      size,
		Payload:   src[start:end],
		Raw:       src[:end],
	}, end, nil
}

// LowOverheadIterator iterates OBUs in AV1 low-overhead bitstream format.
type LowOverheadIterator struct {
	src []byte
	off int
}

func NewLowOverheadIterator(src []byte) LowOverheadIterator {
	return LowOverheadIterator{src: src}
}

func (it *LowOverheadIterator) Next() (Unit, bool, error) {
	if it.off == len(it.src) {
		return Unit{}, false, nil
	}
	if it.off > len(it.src) {
		return Unit{}, false, ErrShortPayload
	}

	unit, consumed, err := ParseLowOverhead(it.src[it.off:])
	if err != nil {
		return Unit{}, false, err
	}
	it.off += consumed
	return unit, true, nil
}
