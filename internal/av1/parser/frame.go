// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const (
	PrimaryRefNone = 7
)

type FrameType uint8

const (
	FrameTypeKey FrameType = iota
	FrameTypeInter
	FrameTypeIntraOnly
	FrameTypeSwitch
)

// FrameHeaderPrefix is the early, decoder-routing portion of
// uncompressed_header().
//
// Later parser stages continue from BitsRead to decode frame size, reference
// selection, tiling, quantization, segmentation, loop filter, CDEF, restoration,
// and tile data boundaries.
type FrameHeaderPrefix struct {
	ShowExistingFrame bool
	ExistingFrameIdx  uint8

	FrameType     FrameType
	ShowFrame     bool
	ShowableFrame bool

	ErrorResilientMode bool
	DisableCDFUpdate   bool

	AllowScreenContentTools bool
	ForceIntegerMV          bool

	FramePresentationDelay uint32
	FrameID                uint32
	FrameSizeOverride      bool
	OrderHint              uint8
	PrimaryRefFrame        uint8

	BitsRead int
}

// ParseFrameHeaderPrefix parses the first control fields of AV1
// uncompressed_header() from a frame_header_obu or frame_obu payload.
//
// The function intentionally stops after primary_ref_frame. It does not consume
// AV1 trailing bits because real frame headers continue with frame size,
// reference, tile, quantizer, filter, and restoration syntax.
func ParseFrameHeaderPrefix(payload []byte, seq SequenceHeader) (FrameHeaderPrefix, error) {
	r := bitstream.NewReader(payload)
	var hdr FrameHeaderPrefix

	hdr.PrimaryRefFrame = PrimaryRefNone
	if !seq.ReducedStillPictureHeader {
		showExisting, err := r.ReadBool()
		if err != nil {
			return FrameHeaderPrefix{}, err
		}
		hdr.ShowExistingFrame = showExisting
	}

	if hdr.ShowExistingFrame {
		if err := parseShowExistingFrameHeader(&r, seq, &hdr); err != nil {
			return FrameHeaderPrefix{}, err
		}
		hdr.BitsRead = r.BitsRead()
		return hdr, nil
	}

	if seq.ReducedStillPictureHeader {
		hdr.FrameType = FrameTypeKey
		hdr.ShowFrame = true
		hdr.ShowableFrame = false
		hdr.ErrorResilientMode = true
	} else if err := parseFrameTypeAndVisibility(&r, seq, &hdr); err != nil {
		return FrameHeaderPrefix{}, err
	}

	if err := parseFrameHeaderFeatureFlags(&r, seq, &hdr); err != nil {
		return FrameHeaderPrefix{}, err
	}
	if err := parseFrameHeaderReferencePrefix(&r, seq, &hdr); err != nil {
		return FrameHeaderPrefix{}, err
	}

	hdr.BitsRead = r.BitsRead()
	return hdr, nil
}

func parseShowExistingFrameHeader(r *bitstream.Reader, seq SequenceHeader, hdr *FrameHeaderPrefix) error {
	v, err := r.ReadBits(3)
	if err != nil {
		return err
	}
	hdr.ExistingFrameIdx = uint8(v)
	if err := readFramePresentationDelay(r, seq, hdr); err != nil {
		return err
	}
	if seq.FrameIDNumbersPresent {
		v, err = r.ReadBits(seq.FrameIDBits())
		if err != nil {
			return err
		}
		hdr.FrameID = uint32(v)
	}
	return nil
}

func parseFrameTypeAndVisibility(r *bitstream.Reader, seq SequenceHeader, hdr *FrameHeaderPrefix) error {
	v, err := r.ReadBits(2)
	if err != nil {
		return err
	}
	hdr.FrameType = FrameType(v)
	if hdr.ShowFrame, err = r.ReadBool(); err != nil {
		return err
	}
	if seq.StillPicture && (hdr.FrameType != FrameTypeKey || !hdr.ShowFrame) {
		return ErrInvalidFrameHeader
	}

	if hdr.ShowFrame {
		if err := readFramePresentationDelay(r, seq, hdr); err != nil {
			return err
		}
		hdr.ShowableFrame = hdr.FrameType != FrameTypeKey
	} else if hdr.ShowableFrame, err = r.ReadBool(); err != nil {
		return err
	}

	if hdr.FrameType == FrameTypeSwitch || (hdr.FrameType == FrameTypeKey && hdr.ShowFrame) {
		hdr.ErrorResilientMode = true
		return nil
	}
	hdr.ErrorResilientMode, err = r.ReadBool()
	return err
}

func parseFrameHeaderFeatureFlags(r *bitstream.Reader, seq SequenceHeader, hdr *FrameHeaderPrefix) error {
	var err error
	if hdr.DisableCDFUpdate, err = r.ReadBool(); err != nil {
		return err
	}

	if seq.SeqForceScreenContentTools == SelectScreenContentTools {
		if hdr.AllowScreenContentTools, err = r.ReadBool(); err != nil {
			return err
		}
	} else {
		hdr.AllowScreenContentTools = seq.SeqForceScreenContentTools != 0
	}

	if hdr.AllowScreenContentTools {
		if seq.SeqForceIntegerMV == SelectIntegerMV {
			if hdr.ForceIntegerMV, err = r.ReadBool(); err != nil {
				return err
			}
		} else {
			hdr.ForceIntegerMV = seq.SeqForceIntegerMV != 0
		}
	}
	if frameTypeIsKeyOrIntra(hdr.FrameType) {
		hdr.ForceIntegerMV = true
	}
	return nil
}

func parseFrameHeaderReferencePrefix(r *bitstream.Reader, seq SequenceHeader, hdr *FrameHeaderPrefix) error {
	if seq.ReducedStillPictureHeader {
		return nil
	}

	if seq.FrameIDNumbersPresent {
		v, err := r.ReadBits(seq.FrameIDBits())
		if err != nil {
			return err
		}
		hdr.FrameID = uint32(v)
	}

	if hdr.FrameType == FrameTypeSwitch {
		hdr.FrameSizeOverride = true
	} else {
		override, err := r.ReadBool()
		if err != nil {
			return err
		}
		hdr.FrameSizeOverride = override
	}

	if seq.EnableOrderHint {
		v, err := r.ReadBits(seq.OrderHintBits)
		if err != nil {
			return err
		}
		hdr.OrderHint = uint8(v)
	}

	if !hdr.ErrorResilientMode && frameTypeIsInterOrSwitch(hdr.FrameType) {
		v, err := r.ReadBits(3)
		if err != nil {
			return err
		}
		hdr.PrimaryRefFrame = uint8(v)
	}
	return nil
}

func readFramePresentationDelay(r *bitstream.Reader, seq SequenceHeader, hdr *FrameHeaderPrefix) error {
	if !seq.DecoderModelInfoPresent || seq.TimingInfo.EqualPictureInterval {
		return nil
	}
	v, err := r.ReadBits(seq.DecoderModelInfo.FramePresentationTimeLength)
	if err != nil {
		return err
	}
	hdr.FramePresentationDelay = uint32(v)
	return nil
}

func (s SequenceHeader) FrameIDBits() uint8 {
	if !s.FrameIDNumbersPresent {
		return 0
	}
	return s.DeltaFrameIDLength + s.AdditionalFrameIDLength
}

func frameTypeIsKeyOrIntra(t FrameType) bool {
	return t == FrameTypeKey || t == FrameTypeIntraOnly
}

func frameTypeIsInterOrSwitch(t FrameType) bool {
	return t == FrameTypeInter || t == FrameTypeSwitch
}

func (h FrameHeaderPrefix) UsesIntraFrameSizePath() bool {
	if h.ShowExistingFrame {
		return false
	}
	return frameTypeIsKeyOrIntra(h.FrameType)
}
