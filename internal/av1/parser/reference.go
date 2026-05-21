package parser

const (
	RefFrames         = refFrames
	InterRefsPerFrame = 7
)

// ReferenceFrame is the parser-visible metadata retained for one AV1 reference
// slot.
type ReferenceFrame struct {
	Valid bool

	FrameID   uint32
	OrderHint uint32
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
	for i := 0; i < RefFrames; i++ {
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
