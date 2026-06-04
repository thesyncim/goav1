// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// DeltaParams is delta_q_params() plus delta_lf_params().
type DeltaParams struct {
	DeltaQPresent  bool
	DeltaQResLog2  uint8
	DeltaLFPresent bool
	DeltaLFResLog2 uint8
	DeltaLFMulti   bool
	BitsRead       uint16
}

func ParseDeltaParams(payload []byte, size FrameSize, quant QuantizationParams, seg SegmentationParams) (DeltaParams, error) {
	r := bitstream.NewReader(payload)
	if err := skipBitsRead(&r, seg.BitsRead); err != nil {
		return DeltaParams{}, err
	}

	var delta DeltaParams
	if quant.BaseQIdx != 0 {
		present, err := r.ReadBool()
		if err != nil {
			return DeltaParams{}, err
		}
		delta.DeltaQPresent = present
		if delta.DeltaQPresent {
			v, err := r.ReadBits(2)
			if err != nil {
				return DeltaParams{}, err
			}
			delta.DeltaQResLog2 = uint8(v)
			if !size.AllowIntrabc {
				if delta.DeltaLFPresent, err = r.ReadBool(); err != nil {
					return DeltaParams{}, err
				}
			}
			if delta.DeltaLFPresent {
				v, err = r.ReadBits(2)
				if err != nil {
					return DeltaParams{}, err
				}
				delta.DeltaLFResLog2 = uint8(v)
				if delta.DeltaLFMulti, err = r.ReadBool(); err != nil {
					return DeltaParams{}, err
				}
			}
		}
	}

	bitsRead, err := readerBitsRead(&r)
	if err != nil {
		return DeltaParams{}, err
	}
	delta.BitsRead = bitsRead
	return delta, nil
}
