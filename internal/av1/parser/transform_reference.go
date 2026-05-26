// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

type TransformMode uint8

const (
	TransformMode4x4Only TransformMode = iota
	TransformModeLargest
	TransformModeSwitchable
)

type ReferenceMode uint8

const (
	ReferenceModeSingle ReferenceMode = iota
	ReferenceModeCompound
	ReferenceModeSelect
)

// TransformReferenceParams is the transform mode and frame reference mode
// syntax immediately following loop restoration params.
type TransformReferenceParams struct {
	TransformMode TransformMode
	ReferenceMode ReferenceMode
	BitsRead      int
}

func ParseTransformReferenceParams(payload []byte, prefix FrameHeaderPrefix, seg SegmentationParams, restoration RestorationParams) (TransformReferenceParams, error) {
	r := bitstream.NewReader(payload)
	if err := r.SkipBits(restoration.BitsRead); err != nil {
		return TransformReferenceParams{}, err
	}

	params := TransformReferenceParams{
		TransformMode: TransformMode4x4Only,
		ReferenceMode: ReferenceModeSingle,
	}
	if !seg.AllLossless {
		selectable, err := r.ReadBool()
		if err != nil {
			return TransformReferenceParams{}, err
		}
		if selectable {
			params.TransformMode = TransformModeSwitchable
		} else {
			params.TransformMode = TransformModeLargest
		}
	}

	if frameTypeIsInterOrSwitch(prefix.FrameType) {
		selectable, err := r.ReadBool()
		if err != nil {
			return TransformReferenceParams{}, err
		}
		if selectable {
			params.ReferenceMode = ReferenceModeSelect
		}
	}

	params.BitsRead = r.BitsRead()
	return params, nil
}
