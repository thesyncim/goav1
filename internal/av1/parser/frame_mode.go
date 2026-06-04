// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// FrameModeParams contains frame-level motion/transform feature bits after
// skip_mode_params().
type FrameModeParams struct {
	AllowWarpedMotion bool
	ReducedTxSet      bool
	BitsRead          uint16
}

func ParseFrameModeParams(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, skip SkipModeParams) (FrameModeParams, error) {
	r := bitstream.NewReader(payload)
	if err := skipBitsRead(&r, skip.BitsRead); err != nil {
		return FrameModeParams{}, err
	}

	var params FrameModeParams
	if !prefix.ErrorResilientMode && frameTypeIsInterOrSwitch(prefix.FrameType) && seq.EnableWarpedMotion {
		allow, err := r.ReadBool()
		if err != nil {
			return FrameModeParams{}, err
		}
		params.AllowWarpedMotion = allow
	}

	reduced, err := r.ReadBool()
	if err != nil {
		return FrameModeParams{}, err
	}
	params.ReducedTxSet = reduced
	bitsRead, err := readerBitsRead(&r)
	if err != nil {
		return FrameModeParams{}, err
	}
	params.BitsRead = bitsRead
	return params, nil
}
