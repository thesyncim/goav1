// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

const (
	RefFrames         = refFrames
	InterRefsPerFrame = 7
)

// ReferenceFrame is the parser-visible metadata retained for one AV1 reference
// slot.
type ReferenceFrame struct {
	Valid bool

	FrameID   uint16
	OrderHint uint8
	FrameType FrameType

	ShowableFrame bool
	Size          FrameSize

	Segmentation     SegmentationData
	LoopFilterDeltas LoopFilterDeltas
	GlobalMotion     GlobalMotionParams
	FilmGrain        FilmGrainParams
}

// ReferenceState is caller-owned decoder reference metadata used while parsing
// inter-frame headers.
type ReferenceState struct {
	Frames [RefFrames]ReferenceFrame
}

func (s *ReferenceState) Reset() {
	*s = ReferenceState{}
}

func (s *ReferenceState) Update(prefix FrameHeaderPrefix, size FrameSize) {
	s.UpdateWithGlobalMotion(prefix, size, DefaultGlobalMotionParams())
}

func (s *ReferenceState) UpdateWithGlobalMotion(prefix FrameHeaderPrefix, size FrameSize, globalMotion GlobalMotionParams) {
	s.UpdateWithFrameState(prefix, size, globalMotion, FilmGrainParams{})
}

func (s *ReferenceState) UpdateWithFrameState(prefix FrameHeaderPrefix, size FrameSize, globalMotion GlobalMotionParams, filmGrain FilmGrainParams) {
	s.UpdateWithDecodeState(prefix, size, defaultSegmentationData(), defaultLoopFilterDeltas(), globalMotion, filmGrain)
}

func (s *ReferenceState) UpdateWithDecodeState(prefix FrameHeaderPrefix, size FrameSize, segmentation SegmentationData, loopFilter LoopFilterDeltas, globalMotion GlobalMotionParams, filmGrain FilmGrainParams) {
	if prefix.ShowExistingFrame {
		return
	}
	ref := ReferenceFrame{
		Valid:            true,
		FrameID:          prefix.FrameID,
		OrderHint:        prefix.OrderHint,
		FrameType:        prefix.FrameType,
		ShowableFrame:    prefix.ShowableFrame,
		Size:             size,
		Segmentation:     segmentation,
		LoopFilterDeltas: loopFilter,
		GlobalMotion:     globalMotion,
		FilmGrain:        filmGrain,
	}
	for i := range RefFrames {
		if (size.RefreshFrameFlags & (1 << uint(i))) != 0 {
			s.Frames[i] = ref
		}
	}
}

func defaultSegmentationData() SegmentationData {
	var data SegmentationData
	clearSegmentationRefs(&data)
	return data
}

// ValidateReferenceScale ports the reference check in dav1d_submit_frame()
// (decode.c): every reference an inter or switch frame names must hold a
// picture whose size is within the 2x-downscale / 16x-upscale prediction
// range of the frame (the spec's conformance requirement
// 2 * FrameWidth >= RefUpscaledWidth, 2 * FrameHeight >= RefFrameHeight,
// FrameWidth <= 16 * RefUpscaledWidth, FrameHeight <= 16 * RefFrameHeight).
// Intra frames read no references.
// dav1d also compares pixel layout and bit depth, which cannot differ here:
// a sequence header that changes them resets the reference state.
func ValidateReferenceScale(prefix FrameHeaderPrefix, size FrameSize, refs *ReferenceState) error {
	if !frameTypeIsInterOrSwitch(prefix.FrameType) {
		return nil
	}
	if refs == nil {
		return ErrReferenceFrameNeeded
	}
	w, h := uint64(size.CodedWidth), uint64(size.Height)
	for _, slot := range size.RefFrameIdx {
		if slot >= RefFrames || !refs.Frames[slot].Valid {
			return ErrReferenceFrameNeeded
		}
		ref := &refs.Frames[slot].Size
		rw, rh := uint64(ref.UpscaledWidth), uint64(ref.Height)
		if 2*w < rw || 2*h < rh || w > 16*rw || h > 16*rh {
			return ErrInvalidReferenceScale
		}
	}
	return nil
}
