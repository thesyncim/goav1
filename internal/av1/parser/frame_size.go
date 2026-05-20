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

	BitsRead int
}

// ParseIntraFrameSize parses the frame-size path used by key and intra-only
// frames. Inter frames may derive dimensions from references and are handled by
// a later parser stage that has reference-frame state.
func ParseIntraFrameSize(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, temporalID uint8, spatialID uint8) (FrameSize, error) {
	if !prefix.UsesIntraFrameSizePath() {
		return FrameSize{}, ErrReferenceFrameNeeded
	}

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
	if prefix.FrameType == FrameTypeKey && prefix.ShowFrame {
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

	if size.RefreshFrameFlags != 0xff && prefix.ErrorResilientMode && seq.EnableOrderHint {
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
