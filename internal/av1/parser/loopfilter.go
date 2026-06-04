// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const LoopFilterModeDeltas = 2

// LoopFilterDeltas is the reference-copyable mode/ref delta table.
type LoopFilterDeltas struct {
	Ref  [RefFrames]int8
	Mode [LoopFilterModeDeltas]int8
}

// LoopFilterParams is loop_filter_params() from uncompressed_header().
type LoopFilterParams struct {
	LevelY [2]uint8
	LevelU uint8
	LevelV uint8

	Sharpness uint8

	ModeRefDeltaEnabled bool
	ModeRefDeltaUpdate  bool
	Deltas              LoopFilterDeltas

	BitsRead uint16
}

func ParseLoopFilterParams(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, seg SegmentationParams, delta DeltaParams, previous *LoopFilterDeltas) (LoopFilterParams, error) {
	r := bitstream.NewReader(payload)
	if err := skipBitsRead(&r, delta.BitsRead); err != nil {
		return LoopFilterParams{}, err
	}

	var lf LoopFilterParams
	if seg.AllLossless || size.AllowIntrabc {
		lf.ModeRefDeltaEnabled = true
		lf.ModeRefDeltaUpdate = true
		lf.Deltas = defaultLoopFilterDeltas()
		bitsRead, err := readerBitsRead(&r)
		if err != nil {
			return LoopFilterParams{}, err
		}
		lf.BitsRead = bitsRead
		return lf, nil
	}

	if prefix.PrimaryRefFrame == PrimaryRefNone {
		lf.Deltas = defaultLoopFilterDeltas()
	} else {
		if previous == nil {
			return LoopFilterParams{}, ErrReferenceFrameNeeded
		}
		lf.Deltas = *previous
	}

	v, err := r.ReadBits(6)
	if err != nil {
		return LoopFilterParams{}, err
	}
	lf.LevelY[0] = uint8(v)
	v, err = r.ReadBits(6)
	if err != nil {
		return LoopFilterParams{}, err
	}
	lf.LevelY[1] = uint8(v)
	if !seq.ColorConfig.MonoChrome && (lf.LevelY[0] != 0 || lf.LevelY[1] != 0) {
		v, err = r.ReadBits(6)
		if err != nil {
			return LoopFilterParams{}, err
		}
		lf.LevelU = uint8(v)
		v, err = r.ReadBits(6)
		if err != nil {
			return LoopFilterParams{}, err
		}
		lf.LevelV = uint8(v)
	}
	v, err = r.ReadBits(3)
	if err != nil {
		return LoopFilterParams{}, err
	}
	lf.Sharpness = uint8(v)

	if lf.ModeRefDeltaEnabled, err = r.ReadBool(); err != nil {
		return LoopFilterParams{}, err
	}
	if lf.ModeRefDeltaEnabled {
		if lf.ModeRefDeltaUpdate, err = r.ReadBool(); err != nil {
			return LoopFilterParams{}, err
		}
		if lf.ModeRefDeltaUpdate {
			if err := readLoopFilterDeltaUpdates(&r, &lf.Deltas); err != nil {
				return LoopFilterParams{}, err
			}
		}
	}

	bitsRead, err := readerBitsRead(&r)
	if err != nil {
		return LoopFilterParams{}, err
	}
	lf.BitsRead = bitsRead
	return lf, nil
}

func readLoopFilterDeltaUpdates(r *bitstream.Reader, deltas *LoopFilterDeltas) error {
	for i := range RefFrames {
		update, err := r.ReadBool()
		if err != nil {
			return err
		}
		if !update {
			continue
		}
		v, err := readSignedBits(r, 7)
		if err != nil {
			return err
		}
		deltas.Ref[i] = int8(v)
	}
	for i := range LoopFilterModeDeltas {
		update, err := r.ReadBool()
		if err != nil {
			return err
		}
		if !update {
			continue
		}
		v, err := readSignedBits(r, 7)
		if err != nil {
			return err
		}
		deltas.Mode[i] = int8(v)
	}
	return nil
}

func defaultLoopFilterDeltas() LoopFilterDeltas {
	return LoopFilterDeltas{
		Ref:  [RefFrames]int8{1, 0, 0, 0, -1, 0, -1, -1},
		Mode: [LoopFilterModeDeltas]int8{0, 0},
	}
}
