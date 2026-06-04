// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// SkipModeParams is the frame-level skip_mode_params() state after reference
// mode selection.
type SkipModeParams struct {
	Allowed     bool
	Enabled     bool
	RefFrameIdx [2]uint8
	BitsRead    int
}

func ParseSkipModeParams(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, refs *ReferenceState, transformRef TransformReferenceParams) (SkipModeParams, error) {
	r := bitstream.NewReader(payload)
	if err := r.SkipBits(transformRef.BitsRead); err != nil {
		return SkipModeParams{}, err
	}

	params := SkipModeParams{BitsRead: r.BitsRead()}
	if !seq.EnableOrderHint || !frameTypeIsInterOrSwitch(prefix.FrameType) || transformRef.ReferenceMode == ReferenceModeSingle {
		return params, nil
	}
	if refs == nil {
		return SkipModeParams{}, ErrReferenceFrameNeeded
	}
	allowed, err := deriveSkipModeRefs(seq, prefix, size, refs, &params)
	if err != nil {
		return SkipModeParams{}, err
	}
	if !allowed {
		return params, nil
	}

	enabled, err := r.ReadBool()
	if err != nil {
		return SkipModeParams{}, err
	}
	params.Enabled = enabled
	params.BitsRead = r.BitsRead()
	return params, nil
}

func deriveSkipModeRefs(seq SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, refs *ReferenceState, params *SkipModeParams) (bool, error) {
	beforeIdx := -1
	afterIdx := -1
	beforeDiff := int16(0)
	afterDiff := int16(0)

	for i := range InterRefsPerFrame {
		slot := size.RefFrameIdx[i]
		if slot >= RefFrames {
			return false, ErrInvalidFrameHeader
		}
		ref := refs.Frames[slot]
		if !ref.Valid {
			return false, ErrReferenceFrameNeeded
		}
		diff := relativeOrderHint(seq.OrderHintBits, ref.OrderHint, prefix.OrderHint)
		if diff > 0 {
			if afterIdx < 0 || diff < afterDiff {
				afterIdx = i
				afterDiff = diff
			}
			continue
		}
		if diff < 0 && (beforeIdx < 0 || diff > beforeDiff) {
			beforeIdx = i
			beforeDiff = diff
		}
	}

	if beforeIdx >= 0 && afterIdx >= 0 {
		setSkipModeRefs(params, beforeIdx, afterIdx)
		return true, nil
	}
	if beforeIdx < 0 {
		return false, nil
	}

	before2Idx := -1
	before2Diff := int16(0)
	for i := range InterRefsPerFrame {
		ref := refs.Frames[size.RefFrameIdx[i]]
		diff := relativeOrderHint(seq.OrderHintBits, ref.OrderHint, prefix.OrderHint)
		if diff >= beforeDiff {
			continue
		}
		if before2Idx < 0 || diff > before2Diff {
			before2Idx = i
			before2Diff = diff
		}
	}
	if before2Idx < 0 {
		return false, nil
	}

	setSkipModeRefs(params, beforeIdx, before2Idx)
	return true, nil
}

func setSkipModeRefs(params *SkipModeParams, a int, b int) {
	params.Allowed = true
	if a < b {
		params.RefFrameIdx[0] = uint8(a)
		params.RefFrameIdx[1] = uint8(b)
		return
	}
	params.RefFrameIdx[0] = uint8(b)
	params.RefFrameIdx[1] = uint8(a)
}
