// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const MaxCDEFStrengths = 8

// CDEFParams is cdef_params() from uncompressed_header().
type CDEFParams struct {
	Damping       uint8
	Bits          uint8
	StrengthCount uint8
	YStrength     [MaxCDEFStrengths]uint8
	UVStrength    [MaxCDEFStrengths]uint8
	BitsRead      uint16
}

func ParseCDEFParams(payload []byte, seq SequenceHeader, size FrameSize, seg SegmentationParams, lf LoopFilterParams) (CDEFParams, error) {
	r := bitstream.NewReader(payload)
	if err := skipBitsRead(&r, lf.BitsRead); err != nil {
		return CDEFParams{}, err
	}

	var cdef CDEFParams
	if seg.AllLossless || !seq.EnableCDEF || size.AllowIntrabc {
		bitsRead, err := readerBitsRead(&r)
		if err != nil {
			return CDEFParams{}, err
		}
		cdef.BitsRead = bitsRead
		return cdef, nil
	}

	v, err := r.ReadBits(2)
	if err != nil {
		return CDEFParams{}, err
	}
	cdef.Damping = uint8(v + 3)
	v, err = r.ReadBits(2)
	if err != nil {
		return CDEFParams{}, err
	}
	cdef.Bits = uint8(v)
	cdef.StrengthCount = 1 << cdef.Bits
	for i := uint8(0); i < cdef.StrengthCount; i++ {
		v, err = r.ReadBits(6)
		if err != nil {
			return CDEFParams{}, err
		}
		cdef.YStrength[i] = uint8(v)
		if !seq.ColorConfig.MonoChrome {
			v, err = r.ReadBits(6)
			if err != nil {
				return CDEFParams{}, err
			}
			cdef.UVStrength[i] = uint8(v)
		}
	}

	bitsRead, err := readerBitsRead(&r)
	if err != nil {
		return CDEFParams{}, err
	}
	cdef.BitsRead = bitsRead
	return cdef, nil
}
