package encoder

// LayerRecon returns the reconstruction of the last frame the stream coded
// for spatial layer i.
func (s *WebRTCStream) LayerRecon(i uint8) SourceFrame420 {
	return s.encoders[i].Recon()
}

// EncodeKeyWithSequenceMax codes src as a keyframe whose sequence header
// advertises maxWidth x maxHeight, as upper spatial layers' encoders do.
func (e *VideoEncoder) EncodeKeyWithSequenceMax(src SourceFrame420, maxWidth, maxHeight int) ([]byte, error) {
	return e.encodeKeyWithSequenceMax(src, maxWidth, maxHeight)
}

// NewHighBitDepth420VideoEncoderWithColor builds the high-bit-depth 4:2:0
// encoder a WebRTC stream uses for its layers.
var NewHighBitDepth420VideoEncoderWithColor = newHighBitDepth420VideoEncoderWithColor
