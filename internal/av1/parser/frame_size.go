package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const (
	refFrames = 8
)

// FrameSize is the frame-size portion of uncompressed_header() for key and
// intra-only frames.
//
// UpscaledWidth is AV1 FrameWidth. CodedWidth is the super-res scaled width
// used by reconstruction buffers. Height is shared by both.
type FrameSize struct {
	CodedWidth    uint32
	UpscaledWidth uint32
	Height        uint32
	RenderWidth   uint32
	RenderHeight  uint32

	SuperResEnabled     bool
	SuperResDenominator uint8
	HaveRenderSize      bool
	AllowIntrabc        bool

	RefreshFrameFlags        uint8
	BufferRemovalTimePresent bool
	BufferRemovalTimes       [32]uint32
	RefOrderHints            [refFrames]uint32

	FrameRefsShortSignaling bool
	LastFrameIdx            uint8
	GoldenFrameIdx          uint8
	RefFrameIdx             [InterRefsPerFrame]uint8
	DeltaFrameIDMinus1      [InterRefsPerFrame]uint32
	UsedReferenceFrameSize  bool
	ReferenceFrameSizeIdx   uint8

	BitsRead int
}

// ParseIntraFrameSize parses the frame-size path used by key and intra-only
// frames. Inter frames may derive dimensions from references and are handled by
// a later parser stage that has reference-frame state.
func ParseIntraFrameSize(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, temporalID uint8, spatialID uint8) (FrameSize, error) {
	if !prefix.UsesIntraFrameSizePath() {
		return FrameSize{}, ErrReferenceFrameNeeded
	}
	return parseFrameSize(payload, seq, prefix, nil, temporalID, spatialID)
}

// ParseFrameSize parses frame-size and reference routing syntax after
// ParseFrameHeaderPrefix. refs is required for inter and switch frames.
func ParseFrameSize(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, refs *ReferenceState, temporalID uint8, spatialID uint8) (FrameSize, error) {
	if prefix.ShowExistingFrame {
		return FrameSize{}, ErrInvalidFrameHeader
	}
	return parseFrameSize(payload, seq, prefix, refs, temporalID, spatialID)
}

func parseFrameSize(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, refs *ReferenceState, temporalID uint8, spatialID uint8) (FrameSize, error) {
	r := bitstream.NewReader(payload)
	if err := r.SkipBits(prefix.BitsRead); err != nil {
		return FrameSize{}, err
	}

	var size FrameSize
	if err := parseBufferRemovalTimes(&r, seq, temporalID, spatialID, &size); err != nil {
		return FrameSize{}, err
	}
	if err := parseRefreshFrameFlags(&r, seq, prefix, &size); err != nil {
		return FrameSize{}, err
	}

	if frameTypeIsInterOrSwitch(prefix.FrameType) {
		if refs == nil {
			return FrameSize{}, ErrReferenceFrameNeeded
		}
		if err := parseInterReferences(&r, seq, prefix, refs, &size); err != nil {
			return FrameSize{}, err
		}
		if !prefix.ErrorResilientMode && prefix.FrameSizeOverride {
			if err := parseFrameDimensionsWithRefs(&r, seq, prefix, refs, &size); err != nil {
				return FrameSize{}, err
			}
		} else if err := parseFrameDimensions(&r, seq, prefix, &size); err != nil {
			return FrameSize{}, err
		}
		size.BitsRead = r.BitsRead()
		return size, nil
	}

	if err := parseFrameDimensions(&r, seq, prefix, &size); err != nil {
		return FrameSize{}, err
	}
	if prefix.AllowScreenContentTools && !size.SuperResEnabled {
		allow, err := r.ReadBool()
		if err != nil {
			return FrameSize{}, err
		}
		size.AllowIntrabc = allow
	}

	size.BitsRead = r.BitsRead()
	return size, nil
}

func parseBufferRemovalTimes(r *bitstream.Reader, seq SequenceHeader, temporalID uint8, spatialID uint8, size *FrameSize) error {
	if !seq.DecoderModelInfoPresent {
		return nil
	}
	present, err := r.ReadBool()
	if err != nil {
		return err
	}
	size.BufferRemovalTimePresent = present
	if !present {
		return nil
	}

	bits := seq.DecoderModelInfo.BufferRemovalTimeLength
	for i := uint8(0); i < seq.OperatingPointsCount; i++ {
		op := seq.OperatingPoints[i]
		if !op.DecoderModelPresent {
			continue
		}
		if op.IDC != 0 {
			inTemporalLayer := ((op.IDC >> temporalID) & 1) != 0
			inSpatialLayer := ((op.IDC >> (spatialID + 8)) & 1) != 0
			if !inTemporalLayer || !inSpatialLayer {
				continue
			}
		}
		v, err := r.ReadBits(bits)
		if err != nil {
			return err
		}
		size.BufferRemovalTimes[i] = uint32(v)
	}
	return nil
}

func parseRefreshFrameFlags(r *bitstream.Reader, seq SequenceHeader, prefix FrameHeaderPrefix, size *FrameSize) error {
	if (prefix.FrameType == FrameTypeKey && prefix.ShowFrame) || prefix.FrameType == FrameTypeSwitch {
		size.RefreshFrameFlags = 0xff
	} else {
		v, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		size.RefreshFrameFlags = uint8(v)
		if prefix.FrameType == FrameTypeIntraOnly && size.RefreshFrameFlags == 0xff {
			return ErrInvalidFrameHeader
		}
	}

	readOrderHints := prefix.ErrorResilientMode && seq.EnableOrderHint &&
		(frameTypeIsInterOrSwitch(prefix.FrameType) || size.RefreshFrameFlags != 0xff)
	if readOrderHints {
		for i := 0; i < refFrames; i++ {
			v, err := r.ReadBits(seq.OrderHintBits)
			if err != nil {
				return err
			}
			size.RefOrderHints[i] = uint32(v)
		}
	}
	return nil
}

func parseInterReferences(r *bitstream.Reader, seq SequenceHeader, prefix FrameHeaderPrefix, refs *ReferenceState, size *FrameSize) error {
	if seq.EnableOrderHint {
		short, err := r.ReadBool()
		if err != nil {
			return err
		}
		size.FrameRefsShortSignaling = short
		if short {
			if err := parseShortFrameReferences(r, seq, prefix, refs, size); err != nil {
				return err
			}
		}
	}

	for i := 0; i < InterRefsPerFrame; i++ {
		if !size.FrameRefsShortSignaling {
			v, err := r.ReadBits(3)
			if err != nil {
				return err
			}
			size.RefFrameIdx[i] = uint8(v)
		}
		ref := refs.Frames[size.RefFrameIdx[i]]
		if !ref.Valid {
			return ErrReferenceFrameNeeded
		}
		if seq.FrameIDNumbersPresent {
			v, err := r.ReadBits(seq.DeltaFrameIDLength)
			if err != nil {
				return err
			}
			size.DeltaFrameIDMinus1[i] = uint32(v)
			if ref.FrameID != expectedReferenceFrameID(prefix.FrameID, uint32(v)+1, seq.FrameIDBits()) {
				return ErrInvalidFrameHeader
			}
		}
	}
	return nil
}

func parseShortFrameReferences(r *bitstream.Reader, seq SequenceHeader, prefix FrameHeaderPrefix, refs *ReferenceState, size *FrameSize) error {
	last, err := r.ReadBits(3)
	if err != nil {
		return err
	}
	golden, err := r.ReadBits(3)
	if err != nil {
		return err
	}
	size.LastFrameIdx = uint8(last)
	size.GoldenFrameIdx = uint8(golden)
	return setShortFrameReferences(seq, prefix, refs, int(last), int(golden), size)
}

type shortRefInfo struct {
	mapIdx  int
	sortIdx int
}

func setShortFrameReferences(seq SequenceHeader, prefix FrameHeaderPrefix, refs *ReferenceState, last int, golden int, size *FrameSize) error {
	if last < 0 || last >= RefFrames || golden < 0 || golden >= RefFrames {
		return ErrInvalidFrameHeader
	}
	curSortIdx := 1 << uint(seq.OrderHintBits-1)
	info := [RefFrames]shortRefInfo{}
	lastSortIdx := -1
	goldenSortIdx := -1
	for i := 0; i < RefFrames; i++ {
		info[i] = shortRefInfo{mapIdx: i, sortIdx: -1}
		ref := refs.Frames[i]
		if !ref.Valid {
			return ErrReferenceFrameNeeded
		}
		info[i].sortIdx = curSortIdx + relativeOrderHint(seq.OrderHintBits, ref.OrderHint, prefix.OrderHint)
		if i == last {
			lastSortIdx = info[i].sortIdx
		}
		if i == golden {
			goldenSortIdx = info[i].sortIdx
		}
	}
	if lastSortIdx == -1 || lastSortIdx >= curSortIdx ||
		goldenSortIdx == -1 || goldenSortIdx >= curSortIdx {
		return ErrInvalidFrameHeader
	}

	sortShortRefInfo(info[:])
	fwdStart := 0
	fwdEnd := RefFrames - 1
	for i := 0; i < RefFrames; i++ {
		if info[i].sortIdx == -1 {
			fwdStart++
			continue
		}
		if info[i].sortIdx >= curSortIdx {
			fwdEnd = i - 1
			break
		}
	}
	bwdStart := fwdEnd + 1
	bwdEnd := RefFrames - 1

	set := [InterRefsPerFrame]bool{}
	if bwdStart <= bwdEnd {
		assignRef(size, &set, 6, info[bwdEnd].mapIdx)
		bwdEnd--
	}
	if bwdStart <= bwdEnd {
		assignRef(size, &set, 4, info[bwdStart].mapIdx)
		bwdStart++
	}
	if bwdStart <= bwdEnd {
		assignRef(size, &set, 5, info[bwdStart].mapIdx)
	}

	for i := fwdStart; i <= fwdEnd; i++ {
		if info[i].mapIdx == last {
			assignRef(size, &set, 0, info[i].mapIdx)
		}
		if info[i].mapIdx == golden {
			assignRef(size, &set, 3, info[i].mapIdx)
		}
	}
	if !set[0] || !set[3] {
		return ErrInvalidFrameHeader
	}

	refFrameList := [InterRefsPerFrame - 2]int{1, 2, 4, 5, 6}
	refIdx := 0
	for ; refIdx < len(refFrameList); refIdx++ {
		slot := refFrameList[refIdx]
		if set[slot] {
			continue
		}
		for fwdStart <= fwdEnd && (info[fwdEnd].mapIdx == last || info[fwdEnd].mapIdx == golden) {
			fwdEnd--
		}
		if fwdStart > fwdEnd {
			break
		}
		assignRef(size, &set, slot, info[fwdEnd].mapIdx)
		fwdEnd--
	}
	if fwdStart > fwdEnd {
		return ErrInvalidFrameHeader
	}
	for ; refIdx < len(refFrameList); refIdx++ {
		slot := refFrameList[refIdx]
		if set[slot] {
			continue
		}
		assignRef(size, &set, slot, info[fwdStart].mapIdx)
	}
	for i := 0; i < InterRefsPerFrame; i++ {
		if !set[i] {
			return ErrInvalidFrameHeader
		}
	}
	return nil
}

func assignRef(size *FrameSize, set *[InterRefsPerFrame]bool, slot int, mapIdx int) {
	size.RefFrameIdx[slot] = uint8(mapIdx)
	set[slot] = true
}

func sortShortRefInfo(info []shortRefInfo) {
	for i := 1; i < len(info); i++ {
		v := info[i]
		j := i - 1
		for j >= 0 && (info[j].sortIdx > v.sortIdx || (info[j].sortIdx == v.sortIdx && info[j].mapIdx > v.mapIdx)) {
			info[j+1] = info[j]
			j--
		}
		info[j+1] = v
	}
}

func relativeOrderHint(bits uint8, a uint32, b uint32) int {
	if bits == 0 {
		return 0
	}
	mask := int32(1 << uint(bits-1))
	diff := int32(a) - int32(b)
	return int((diff & (mask - 1)) - (diff & mask))
}

func expectedReferenceFrameID(current uint32, delta uint32, bits uint8) uint32 {
	mod := uint64(1) << uint(bits)
	return uint32((uint64(current) + mod - uint64(delta)) & (mod - 1))
}

func parseFrameDimensions(r *bitstream.Reader, seq SequenceHeader, prefix FrameHeaderPrefix, size *FrameSize) error {
	if prefix.FrameSizeOverride {
		w, err := r.ReadBits(seq.FrameWidthBits)
		if err != nil {
			return err
		}
		h, err := r.ReadBits(seq.FrameHeightBits)
		if err != nil {
			return err
		}
		size.UpscaledWidth = uint32(w + 1)
		size.Height = uint32(h + 1)
		if size.UpscaledWidth > seq.MaxFrameWidth || size.Height > seq.MaxFrameHeight {
			return ErrInvalidFrameHeader
		}
	} else {
		size.UpscaledWidth = seq.MaxFrameWidth
		size.Height = seq.MaxFrameHeight
	}

	if err := parseSuperRes(r, seq, size); err != nil {
		return err
	}

	haveRenderSize, err := r.ReadBool()
	if err != nil {
		return err
	}
	size.HaveRenderSize = haveRenderSize
	if haveRenderSize {
		w, err := r.ReadBits(16)
		if err != nil {
			return err
		}
		h, err := r.ReadBits(16)
		if err != nil {
			return err
		}
		size.RenderWidth = uint32(w + 1)
		size.RenderHeight = uint32(h + 1)
	} else {
		size.RenderWidth = size.UpscaledWidth
		size.RenderHeight = size.Height
	}
	return nil
}

func parseSuperRes(r *bitstream.Reader, seq SequenceHeader, size *FrameSize) error {
	size.SuperResEnabled = false
	size.SuperResDenominator = 8
	if seq.EnableSuperRes {
		enabled, err := r.ReadBool()
		if err != nil {
			return err
		}
		size.SuperResEnabled = enabled
		if enabled {
			v, err := r.ReadBits(3)
			if err != nil {
				return err
			}
			size.SuperResDenominator = uint8(v + 9)
		}
	}

	if size.SuperResEnabled {
		size.CodedWidth = superResCodedWidth(size.UpscaledWidth, size.SuperResDenominator)
	} else {
		size.CodedWidth = size.UpscaledWidth
	}
	return nil
}

func parseFrameDimensionsWithRefs(r *bitstream.Reader, seq SequenceHeader, prefix FrameHeaderPrefix, refs *ReferenceState, size *FrameSize) error {
	for i := 0; i < InterRefsPerFrame; i++ {
		use, err := r.ReadBool()
		if err != nil {
			return err
		}
		if !use {
			continue
		}
		ref := refs.Frames[size.RefFrameIdx[i]]
		if !ref.Valid || ref.Size.UpscaledWidth == 0 || ref.Size.Height == 0 {
			return ErrReferenceFrameNeeded
		}
		size.CodedWidth = ref.Size.CodedWidth
		size.UpscaledWidth = ref.Size.UpscaledWidth
		size.Height = ref.Size.Height
		size.RenderWidth = ref.Size.RenderWidth
		size.RenderHeight = ref.Size.RenderHeight
		size.HaveRenderSize = ref.Size.HaveRenderSize
		size.SuperResDenominator = 8
		if err := parseSuperRes(r, seq, size); err != nil {
			return err
		}
		size.UsedReferenceFrameSize = true
		size.ReferenceFrameSizeIdx = uint8(i)
		return nil
	}
	return parseFrameDimensions(r, seq, prefix, size)
}

func superResCodedWidth(upscaledWidth uint32, denominator uint8) uint32 {
	w := (upscaledWidth*8 + uint32(denominator>>1)) / uint32(denominator)
	minWidth := upscaledWidth
	if minWidth > 16 {
		minWidth = 16
	}
	if w < minWidth {
		return minWidth
	}
	return w
}
