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

// WebRTCStream encodes an L1T1 stream with per-frame dependency descriptors.
type WebRTCStream struct {
	enc       *VideoEncoder
	structure WebRTCFrameDependencyStructure
	idState   FrameIDBufferState
	frameID   uint64
}

// NewWebRTCStream creates an L1T1 WebRTC stream under CBR rate control.
func NewWebRTCStream(width, height int, rc RateControlConfig) (*WebRTCStream, error) {
	enc, err := NewVideoEncoderCBR(width, height, rc)
	if err != nil {
		return nil, err
	}
	structure, err := WebRTCFrameDependencyStructureForConfig(Config{
		Resolution:  Resolution{Width: int32(width), Height: int32(height)},
		Scalability: ScalabilityModeL1T1,
	})
	if err != nil {
		return nil, fmt.Errorf("dependency structure: %w", err)
	}
	return &WebRTCStream{enc: enc, structure: structure}, nil
}

// Encode encodes one frame and returns it with WebRTC packaging metadata.
func (s *WebRTCStream) Encode(src SourceFrame420, forceKey bool) (WebRTCEncodedFrame, error) {
	tu, key, err := s.enc.Encode(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	settings := FrameEncodeSettings{
		UpdateBuffer:    0,
		UpdateBufferSet: true,
		Output:          true,
	}
	if key {
		settings.Type = FrameTypeKey
	} else {
		settings.Type = FrameTypeDelta
		settings.ReferenceBuffers[0] = 0
		settings.ReferenceCount = 1
	}
	info, next, err := WebRTCGenericFrameInfoForFrame(settings, s.frameID, s.idState, 1, 1)
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
