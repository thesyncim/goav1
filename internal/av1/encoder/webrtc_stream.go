package encoder

import "fmt"

// webrtc_stream.go joins the streaming video encoder to the WebRTC packaging
// layer: every encoded frame carries the generic frame info and a serialized
// RTP dependency descriptor (AV1 RTP payload spec), so the temporal units can
// go straight into RTP packetization with congestion-controlled forwarding.

// WebRTCEncodedFrame is one encoded frame with its WebRTC metadata.
type WebRTCEncodedFrame struct {
	// TU is the low-overhead temporal unit (the RTP payload content).
	TU []byte
	// Keyframe reports whether the frame resets the decode chain.
	Keyframe bool
	// Info is the generic frame info behind the dependency descriptor.
	Info WebRTCGenericFrameInfo
	// Descriptor is the serialized RTP dependency descriptor for a
	// single-packet frame; keyframes attach the dependency structure.
	Descriptor []byte
}

// WebRTCStream encodes an L1T1 or L1T2 stream with per-frame dependency
// descriptors.
type WebRTCStream struct {
	enc            *VideoEncoder
	structure      WebRTCFrameDependencyStructure
	idState        FrameIDBufferState
	frameID        uint64
	temporalLayers uint8
}

// NewWebRTCStream creates an L1T1 WebRTC stream under CBR rate control.
func NewWebRTCStream(width, height int, rc RateControlConfig) (*WebRTCStream, error) {
	return NewWebRTCStreamLayers(width, height, rc, 1)
}

// NewWebRTCStreamLayers creates a WebRTC stream with the given temporal layer
// count (1 = L1T1, 2 = L1T2 with droppable odd frames).
func NewWebRTCStreamLayers(width, height int, rc RateControlConfig, temporalLayers int) (*WebRTCStream, error) {
	enc, err := NewVideoEncoderCBR(width, height, rc)
	if err != nil {
		return nil, err
	}
	mode := ScalabilityModeL1T1
	switch temporalLayers {
	case 1:
	case 2:
		mode = ScalabilityModeL1T2
		if err := enc.SetTemporalLayers(2); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("encoder: unsupported temporal layer count %d", temporalLayers)
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: int32(width), Height: int32(height)},
		Scalability: mode,
	})
	if err != nil {
		return nil, fmt.Errorf("dependency structure: %w", err)
	}
	return &WebRTCStream{enc: enc, structure: structure, temporalLayers: uint8(temporalLayers)}, nil
}

// Encode encodes one frame and returns it with WebRTC packaging metadata.
func (s *WebRTCStream) Encode(src SourceFrame420, forceKey bool) (WebRTCEncodedFrame, error) {
	tid := s.enc.TemporalID()
	tu, key, err := s.enc.Encode(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	settings := FrameEncodeSettings{
		TemporalID: tid,
		Output:     true,
	}
	if key {
		settings.Type = FrameTypeKey
		settings.UpdateBuffer = 0
		settings.UpdateBufferSet = true
		tid = 0
	} else {
		settings.Type = FrameTypeDelta
		settings.ReferenceBuffers[0] = 0
		settings.ReferenceCount = 1
		if tid == 0 {
			// Layer-0 frames refresh the reference buffer; layer-1 frames are
			// droppable and update nothing.
			settings.UpdateBuffer = 0
			settings.UpdateBufferSet = true
		}
	}
	layers := s.temporalLayers
	if layers == 0 {
		layers = 1
	}
	info, next, err := WebRTCGenericFrameInfoForFrame(settings, s.frameID, s.idState, 1, layers)
	if err != nil {
		return WebRTCEncodedFrame{}, fmt.Errorf("frame info: %w", err)
	}
	descSize, err := WebRTCDependencyDescriptorSize(s.structure, info, key)
	if err != nil {
		return WebRTCEncodedFrame{}, fmt.Errorf("dependency descriptor size: %w", err)
	}
	descriptor, err := AppendWebRTCDependencyDescriptor(make([]byte, 0, descSize), s.structure, info, true, true, key)
	if err != nil {
		return WebRTCEncodedFrame{}, fmt.Errorf("dependency descriptor: %w", err)
	}
	s.idState = next
	s.frameID++
	return WebRTCEncodedFrame{TU: tu, Keyframe: key, Info: info, Descriptor: descriptor}, nil
}
