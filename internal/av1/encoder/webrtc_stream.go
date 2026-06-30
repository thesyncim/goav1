package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// webrtc_stream.go joins the streaming video encoder to the WebRTC packaging
// layer: every encoded frame carries the generic frame info and a serialized
// RTP dependency descriptor (AV1 RTP payload spec), so the temporal units can
// go straight into RTP packetization with congestion-controlled forwarding.

// WebRTCEncodedFrame is one encoded frame with its WebRTC metadata.
type WebRTCEncodedFrame struct {
	// TU is the low-overhead temporal unit (the RTP payload content).
	TU []byte
	// Keyframe reports whether this frame belongs to a key picture.
	Keyframe bool
	// CodedKeyframe reports whether this frame is an AV1 keyframe. In
	// multi-spatial SVC key pictures, upper spatial layers may belong to the
	// key picture while still being coded as inter frames.
	CodedKeyframe bool
	// LastFrameInPicture reports whether this is the final frame in the WebRTC
	// picture. RTP packetization uses this to place the marker bit only once
	// per picture.
	LastFrameInPicture bool
	// Info is the generic frame info behind the dependency descriptor.
	Info WebRTCGenericFrameInfo
	// Descriptor is the serialized RTP dependency descriptor for a
	// single-packet frame; keyframes attach the dependency structure.
	Descriptor []byte
	// Structure is the dependency structure used to serialize descriptors for
	// this frame. Callers that fragment TU into multiple RTP payloads use it
	// together with Info to attach per-packet dependency descriptors.
	Structure WebRTCFrameDependencyStructure
	// AttachDependencyStructure reports whether the first packet descriptor for
	// this frame should carry Structure.
	AttachDependencyStructure bool
}

// WebRTCEncodedPicture is one WebRTC picture. Multi-spatial scalability emits
// one RTP-frame-ready output per active spatial layer.
type WebRTCEncodedPicture struct {
	Frames   [WebRTCMaxSpatialLayers]WebRTCEncodedFrame
	FrameNum uint8
	Keyframe bool
	Unit     WebRTCPictureTemporalUnit
}

// WebRTCStream encodes a WebRTC AV1 stream with per-frame dependency
// descriptors. Single-spatial streams produce one output frame per Encode call;
// spatial SVC/simulcast streams produce one output frame per active spatial
// layer through EncodePicture.
type WebRTCStream struct {
	config Config
	state  WebRTCEncoderState

	rcMinQ, rcMaxQ uint8

	tileColumns       int
	tileColumnsSet    bool
	goldenInterval    int
	goldenIntervalSet bool

	encoders          [WebRTCMaxSpatialLayers]*VideoEncoder
	monoEncoders      [WebRTCMaxSpatialLayers]*MonochromeVideoEncoder
	mono16Encoders    [WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder
	color16Encoders   [WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder
	scaledFrames      [WebRTCMaxSpatialLayers]SourceFrame420
	scaledMono        [WebRTCMaxSpatialLayers]SourceFrameMono
	scaledMono16      [WebRTCMaxSpatialLayers]SourceFrameMono16
	scaledFrames16    [WebRTCMaxSpatialLayers]SourceFrame42016
	layerScratch      [WebRTCMaxSpatialLayers][]byte
	descriptorScratch [WebRTCMaxSpatialLayers][]byte

	referenceFrames        [WebRTCReferenceBuffers]SourceFrame420
	monoReferenceFrames    [WebRTCReferenceBuffers]SourceFrameMono
	mono16ReferenceFrames  [WebRTCReferenceBuffers]SourceFrameMono16
	color16ReferenceFrames [WebRTCReferenceBuffers]SourceFrame42016
}

// NewWebRTCStream creates an L1T1 WebRTC stream under CBR rate control.
func NewWebRTCStream(width, height int, rc RateControlConfig) (*WebRTCStream, error) {
	return NewWebRTCStreamLayers(width, height, rc, 1)
}

// NewWebRTCStreamLayers creates a WebRTC stream with the given temporal layer
// count (1 = L1T1, 2 = L1T2, 3 = L1T3).
func NewWebRTCStreamLayers(width, height int, rc RateControlConfig, temporalLayers int) (*WebRTCStream, error) {
	if temporalLayers < 1 || temporalLayers > int(WebRTCMaxTemporalLayers) {
		return nil, fmt.Errorf("encoder: unsupported temporal layer count %d", temporalLayers)
	}
	targetKbps := int32((rc.TargetBitsPerSecond + 999) / 1000)
	if targetKbps <= 0 {
		targetKbps = 1
	}
	cfg := Config{
		Resolution:        Resolution{Width: int32(width), Height: int32(height)},
		MaxFramerate:      Rational{Num: int32(rc.FramesPerSecond), Den: 1},
		MinBitrateKbps:    targetKbps,
		MaxBitrateKbps:    targetKbps,
		TargetBitrateKbps: targetKbps,
		RateControl:       RateControlCBR,
	}
	if temporalLayers > 1 {
		var ok bool
		cfg.Scalability, ok = DefaultScalabilityMode(uint8(temporalLayers), 1)
		if !ok {
			return nil, fmt.Errorf("encoder: unsupported temporal layer count %d", temporalLayers)
		}
	}
	stream, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		return nil, err
	}
	stream.setRateControlQRange(rc.MinQIndex, rc.MaxQIndex)
	return stream, nil
}

// NewWebRTCStreamConfig creates a WebRTC stream from the lower-level WebRTC
// encoder config. The pixel encoder accepts profile-valid 8/10/12-bit I420,
// I422, I444, and explicit monochrome streams under CBR or CQP. Multi-spatial
// pixel output is supported for WebRTC SVC and simulcast modes, including key
// and key-shift schedules.
func NewWebRTCStreamConfig(config Config) (*WebRTCStream, error) {
	normalized, fps, err := normalizeWebRTCStreamConfig(config)
	if err != nil {
		return nil, err
	}
	var stream WebRTCStream
	stream.config = normalized
	stream.rcMinQ, stream.rcMaxQ = 20, 200
	if webRTCStreamHighBitDepthColor(normalized) {
		for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
			enc, err := newWebRTCStreamLayerColor16Encoder(normalized, i, fps, stream.rcMinQ, stream.rcMaxQ)
			if err != nil {
				_ = stream.closeEncoders()
				return nil, err
			}
			stream.color16Encoders[i] = enc
		}
	} else if webRTCStreamHighBitDepthMonochrome(normalized) {
		for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
			enc, err := newWebRTCStreamLayerMono16Encoder(normalized, i, fps, stream.rcMinQ, stream.rcMaxQ)
			if err != nil {
				_ = stream.closeEncoders()
				return nil, err
			}
			stream.mono16Encoders[i] = enc
		}
	} else if webRTCStreamMonochrome(normalized) {
		for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
			enc, err := newWebRTCStreamLayerMonoEncoder(normalized, i, fps, stream.rcMinQ, stream.rcMaxQ)
			if err != nil {
				_ = stream.closeEncoders()
				return nil, err
			}
			stream.monoEncoders[i] = enc
		}
	} else {
		for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
			enc, err := newWebRTCStreamLayerEncoder(normalized, i, fps, stream.rcMinQ, stream.rcMaxQ)
			if err != nil {
				_ = stream.closeEncoders()
				return nil, err
			}
			stream.encoders[i] = enc
		}
	}
	stream.applyConfiguredTileColumns(normalized)
	return &stream, nil
}

func NormalizeWebRTCStreamConfig(config Config) (Config, error) {
	normalized, _, err := normalizeWebRTCStreamConfig(config)
	return normalized, err
}

func normalizeWebRTCStreamConfig(config Config) (Config, int, error) {
	normalized, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return Config{}, 0, err
	}
	if !webRTCPixelScalabilitySupported(normalized) {
		return Config{}, 0, ErrUnsupported
	}
	if !webRTCStreamPixelFormatSupported(normalized) {
		return Config{}, 0, ErrUnsupported
	}
	fps := webRTCStreamFramesPerSecond(normalized.MaxFramerate)
	if fps <= 0 {
		return Config{}, 0, ErrInvalidConfig
	}
	switch normalized.RateControl {
	case RateControlCBR:
		for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
			if webRTCStreamLayerTargetKbps(normalized, i) <= 0 {
				return Config{}, 0, ErrInvalidConfig
			}
		}
	case RateControlCQP:
		if normalized.Quantizer == 0 {
			return Config{}, 0, ErrUnsupported
		}
	default:
		return Config{}, 0, ErrUnsupported
	}
	return normalized, fps, nil
}

func webRTCPixelScalabilitySupported(config Config) bool {
	if config.SpatialLayerCount == 0 || config.SpatialLayerCount > WebRTCMaxSpatialLayers {
		return false
	}
	return true
}

func webRTCStreamPixelColorConfigSupported(color SequenceColorConfig) bool {
	return videoColorSupported(color)
}

func webRTCStreamPixelFormatSupported(config Config) bool {
	if webRTCStreamHighBitDepthColor(config) {
		return config.ColorConfig.BitDepth == config.BitDepth &&
			!config.ColorConfig.MonoChrome &&
			highBitDepthVideoColorSupported(config.Profile, config.ColorConfig)
	}
	if webRTCStreamMonochrome(config) {
		switch config.ColorConfig.BitDepth {
		case 8, 10:
			return config.Profile == Profile0
		case 12:
			return config.Profile == Profile2
		default:
			return false
		}
	}
	if config.BitDepth != 8 || config.ColorConfig.BitDepth != 8 ||
		!webRTCStreamPixelColorConfigSupported(config.ColorConfig) {
		return false
	}
	switch config.Profile {
	case Profile0:
		return config.ColorConfig.SubsamplingX && config.ColorConfig.SubsamplingY
	case Profile1:
		return !config.ColorConfig.SubsamplingX && !config.ColorConfig.SubsamplingY
	case Profile2:
		return config.ColorConfig.SubsamplingX && !config.ColorConfig.SubsamplingY
	default:
		return false
	}
}

func webRTCStreamMonochrome(config Config) bool {
	return config.ColorConfig.MonoChrome
}

func webRTCStreamHighBitDepthMonochrome(config Config) bool {
	return config.ColorConfig.MonoChrome && config.ColorConfig.BitDepth > 8
}

func webRTCStreamHighBitDepth420(config Config) bool {
	return !config.ColorConfig.MonoChrome &&
		config.ColorConfig.BitDepth > 8 &&
		config.ColorConfig.SubsamplingX &&
		config.ColorConfig.SubsamplingY
}

func webRTCStreamHighBitDepthColor(config Config) bool {
	return !config.ColorConfig.MonoChrome && config.ColorConfig.BitDepth > 8
}

func newWebRTCStreamLayerEncoder(config Config, layerIndex uint8, fps int, minQ uint8, maxQ uint8) (*VideoEncoder, error) {
	layer := config.SpatialLayers[layerIndex]
	var enc *VideoEncoder
	var err error
	switch config.RateControl {
	case RateControlCBR:
		targetKbps := webRTCStreamLayerTargetKbps(config, layerIndex)
		enc, err = newVideoEncoderCBRWithColor(int(layer.Resolution.Width), int(layer.Resolution.Height), RateControlConfig{
			TargetBitsPerSecond: int(targetKbps) * 1000,
			FramesPerSecond:     fps,
			MinQIndex:           minQ,
			MaxQIndex:           maxQ,
		}, config.ColorConfig)
	case RateControlCQP:
		enc, err = newVideoEncoderWithColor(int(layer.Resolution.Width), int(layer.Resolution.Height), config.Quantizer, config.ColorConfig)
	default:
		return nil, ErrUnsupported
	}
	if err != nil {
		return nil, err
	}
	enc.SetScreenContentSelection(true)
	if err := enc.SetContentHint(config.Content); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetEffortLevel(config.Speed); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
		_ = enc.Close()
		return nil, err
	}
	enc.SetSceneCutKeyframes(false)
	return enc, nil
}

func newWebRTCStreamLayerMonoEncoder(config Config, layerIndex uint8, fps int, minQ uint8, maxQ uint8) (*MonochromeVideoEncoder, error) {
	layer := config.SpatialLayers[layerIndex]
	var enc *MonochromeVideoEncoder
	var err error
	switch config.RateControl {
	case RateControlCBR:
		targetKbps := webRTCStreamLayerTargetKbps(config, layerIndex)
		enc, err = NewMonochromeVideoEncoderCBR(int(layer.Resolution.Width), int(layer.Resolution.Height), RateControlConfig{
			TargetBitsPerSecond: int(targetKbps) * 1000,
			FramesPerSecond:     fps,
			MinQIndex:           minQ,
			MaxQIndex:           maxQ,
		})
	case RateControlCQP:
		enc, err = NewMonochromeVideoEncoder(int(layer.Resolution.Width), int(layer.Resolution.Height), config.Quantizer)
	default:
		return nil, ErrUnsupported
	}
	if err != nil {
		return nil, err
	}
	enc.SetScreenContentSelection(true)
	if err := enc.SetContentHint(config.Content); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetEffortLevel(config.Speed); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
		_ = enc.Close()
		return nil, err
	}
	return enc, nil
}

func newWebRTCStreamLayerMono16Encoder(config Config, layerIndex uint8, fps int, minQ uint8, maxQ uint8) (*HighBitDepthMonochromeVideoEncoder, error) {
	layer := config.SpatialLayers[layerIndex]
	var enc *HighBitDepthMonochromeVideoEncoder
	var err error
	switch config.RateControl {
	case RateControlCBR:
		targetKbps := webRTCStreamLayerTargetKbps(config, layerIndex)
		enc, err = NewHighBitDepthMonochromeVideoEncoderCBR(int(layer.Resolution.Width), int(layer.Resolution.Height), config.ColorConfig.BitDepth, RateControlConfig{
			TargetBitsPerSecond: int(targetKbps) * 1000,
			FramesPerSecond:     fps,
			MinQIndex:           minQ,
			MaxQIndex:           maxQ,
		})
	case RateControlCQP:
		enc, err = NewHighBitDepthMonochromeVideoEncoder(int(layer.Resolution.Width), int(layer.Resolution.Height), config.ColorConfig.BitDepth, config.Quantizer)
	default:
		return nil, ErrUnsupported
	}
	if err != nil {
		return nil, err
	}
	enc.SetScreenContentSelection(true)
	if err := enc.SetContentHint(config.Content); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetEffortLevel(config.Speed); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
		_ = enc.Close()
		return nil, err
	}
	return enc, nil
}

func newWebRTCStreamLayerColor16Encoder(config Config, layerIndex uint8, fps int, minQ uint8, maxQ uint8) (*HighBitDepth420VideoEncoder, error) {
	layer := config.SpatialLayers[layerIndex]
	var enc *HighBitDepth420VideoEncoder
	var err error
	switch config.RateControl {
	case RateControlCBR:
		targetKbps := webRTCStreamLayerTargetKbps(config, layerIndex)
		enc, err = newHighBitDepth420VideoEncoderCBRWithColor(int(layer.Resolution.Width), int(layer.Resolution.Height), config.Profile, config.ColorConfig, RateControlConfig{
			TargetBitsPerSecond: int(targetKbps) * 1000,
			FramesPerSecond:     fps,
			MinQIndex:           minQ,
			MaxQIndex:           maxQ,
		})
	case RateControlCQP:
		enc, err = newHighBitDepth420VideoEncoderWithColor(int(layer.Resolution.Width), int(layer.Resolution.Height), config.Profile, config.ColorConfig, config.Quantizer)
	default:
		return nil, ErrUnsupported
	}
	if err != nil {
		return nil, err
	}
	enc.SetScreenContentSelection(true)
	if err := enc.SetContentHint(config.Content); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetEffortLevel(config.Speed); err != nil {
		_ = enc.Close()
		return nil, err
	}
	if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
		_ = enc.Close()
		return nil, err
	}
	return enc, nil
}

func webRTCStreamLayerTargetKbps(config Config, layerIndex uint8) int32 {
	if layerIndex >= config.SpatialLayerCount {
		return 0
	}
	targetKbps := config.SpatialLayers[layerIndex].TargetBitrateKbps
	if targetKbps <= 0 {
		targetKbps = config.TargetBitrateKbps
	}
	return targetKbps
}

func (s *WebRTCStream) setRateControlQRange(minQ uint8, maxQ uint8) {
	if minQ == 0 {
		minQ = 20
	}
	if maxQ == 0 || maxQ <= minQ {
		maxQ = 200
	}
	s.rcMinQ, s.rcMaxQ = minQ, maxQ
	for i := uint8(0); i < s.config.SpatialLayerCount; i++ {
		enc := s.encoders[i]
		if enc != nil {
			enc.rcMinQ = minQ
			enc.rcMaxQ = maxQ
			enc.qIndex = minQ/2 + maxQ/2
		}
		mono := s.monoEncoders[i]
		if mono != nil {
			mono.rcMinQ = minQ
			mono.rcMaxQ = maxQ
			mono.qIndex = minQ/2 + maxQ/2
		}
		mono16 := s.mono16Encoders[i]
		if mono16 != nil {
			mono16.rcMinQ = minQ
			mono16.rcMaxQ = maxQ
			mono16.qIndex = minQ/2 + maxQ/2
		}
		color16 := s.color16Encoders[i]
		if color16 != nil {
			color16.rcMinQ = minQ
			color16.rcMaxQ = maxQ
			color16.qIndex = minQ/2 + maxQ/2
		}
	}
}

// SetGoldenInterval forwards the golden-reference refresh policy to every
// underlying spatial encoder. Zero disables golden references.
func (s *WebRTCStream) SetGoldenInterval(n int) {
	if s != nil {
		s.goldenInterval = n
		s.goldenIntervalSet = true
	}
	for i := uint8(0); s != nil && i < s.config.SpatialLayerCount; i++ {
		if s.encoders[i] != nil {
			s.encoders[i].SetGoldenInterval(n)
		}
		if s.monoEncoders[i] != nil {
			s.monoEncoders[i].SetGoldenInterval(n)
		}
		if s.mono16Encoders[i] != nil {
			s.mono16Encoders[i].SetGoldenInterval(n)
		}
		if s.color16Encoders[i] != nil {
			s.color16Encoders[i].SetGoldenInterval(n)
		}
	}
}

// SetTileColumns forwards the tile-column override to every underlying spatial
// encoder.
func (s *WebRTCStream) SetTileColumns(cols int) {
	if s != nil {
		s.tileColumns = cols
		s.tileColumnsSet = true
	}
	for i := uint8(0); s != nil && i < s.config.SpatialLayerCount; i++ {
		if s.encoders[i] != nil {
			s.encoders[i].SetTileColumns(cols)
		}
		if s.monoEncoders[i] != nil {
			s.monoEncoders[i].SetTileColumns(cols)
		}
		if s.mono16Encoders[i] != nil {
			s.mono16Encoders[i].SetTileColumns(cols)
		}
		if s.color16Encoders[i] != nil {
			s.color16Encoders[i].SetTileColumns(cols)
		}
	}
}

func (s *WebRTCStream) SetMaxThreads(n int) {
	if s == nil {
		return
	}
	if n < 0 {
		n = 0
	}
	s.config.MaxThreads = int32(n)
	s.applyConfiguredTileColumns(s.config)
}

func (s *WebRTCStream) configuredTileColumns(config Config) (int, bool) {
	if config.MaxThreads > 0 {
		return int(config.MaxThreads), true
	}
	if s != nil && s.tileColumnsSet {
		return s.tileColumns, true
	}
	return 0, false
}

func (s *WebRTCStream) applyConfiguredTileColumns(config Config) {
	for i := uint8(0); s != nil && i < config.SpatialLayerCount; i++ {
		s.applyConfiguredTileColumnsToVideo(config, s.encoders[i])
		s.applyConfiguredTileColumnsToMono(config, s.monoEncoders[i])
		s.applyConfiguredTileColumnsToMono16(config, s.mono16Encoders[i])
		s.applyConfiguredTileColumnsToColor16(config, s.color16Encoders[i])
	}
}

func (s *WebRTCStream) applyConfiguredTileColumnsToVideo(config Config, enc *VideoEncoder) {
	if enc == nil {
		return
	}
	if config.MaxThreads > 0 {
		enc.SetMaxThreads(int(config.MaxThreads))
		return
	}
	enc.singleThread = false
	if cols, ok := s.configuredTileColumns(config); ok {
		enc.SetTileColumns(cols)
		return
	}
	enc.setDefaultTileColumns()
}

func (s *WebRTCStream) applyConfiguredTileColumnsToMono(config Config, enc *MonochromeVideoEncoder) {
	if enc == nil {
		return
	}
	if config.MaxThreads > 0 {
		enc.SetMaxThreads(int(config.MaxThreads))
		return
	}
	if cols, ok := s.configuredTileColumns(config); ok {
		enc.SetTileColumns(cols)
		return
	}
	enc.setDefaultTileColumns()
}

func (s *WebRTCStream) applyConfiguredTileColumnsToMono16(config Config, enc *HighBitDepthMonochromeVideoEncoder) {
	if enc == nil {
		return
	}
	if config.MaxThreads > 0 {
		enc.SetMaxThreads(int(config.MaxThreads))
		return
	}
	if cols, ok := s.configuredTileColumns(config); ok {
		enc.SetTileColumns(cols)
		return
	}
	enc.setDefaultTileColumns()
}

func (s *WebRTCStream) applyConfiguredTileColumnsToColor16(config Config, enc *HighBitDepth420VideoEncoder) {
	if enc == nil {
		return
	}
	if config.MaxThreads > 0 {
		enc.SetMaxThreads(int(config.MaxThreads))
		return
	}
	if cols, ok := s.configuredTileColumns(config); ok {
		enc.SetTileColumns(cols)
		return
	}
	enc.setDefaultTileColumns()
}

// Config returns the normalized WebRTC stream config.
func (s *WebRTCStream) Config() Config {
	if s == nil {
		return Config{}
	}
	return s.config
}

// Close waits for background work in every active spatial encoder to finish
// and releases their persistent workers. It is safe to call more than once.
func (s *WebRTCStream) Close() error {
	if s == nil {
		return nil
	}
	return s.closeEncoders()
}

// SetConfig atomically updates bitrate, framerate, rate-control mode, fixed
// quantizer, and supported scalability settings. Changes that alter layer
// geometry or dependency structure make the next encoded picture a key picture
// while preserving frame IDs.
func (s *WebRTCStream) SetConfig(config Config) error {
	if s == nil {
		return ErrInvalidConfig
	}
	normalized, fps, err := normalizeWebRTCStreamConfig(config)
	if err != nil {
		return err
	}
	oldStructure, oldStructureErr := WebRTCFrameDependencyStructureForConfig(s.config)
	newStructure, err := WebRTCFrameDependencyStructureForConfig(normalized)
	if err != nil {
		return err
	}
	sameStructure := oldStructureErr == nil && oldStructure == newStructure
	sameGeometry := webRTCStreamLayerGeometryEqual(s.config, normalized)
	samePixelFormat := webRTCStreamPixelFormatEqual(s.config, normalized)
	if sameGeometry && samePixelFormat {
		if err := s.updateLayerControls(normalized, fps); err != nil {
			return err
		}
		s.applyConfiguredTileColumns(normalized)
		s.config = normalized
		if !sameStructure {
			s.state = webRTCEncoderStateForNextKey(s.state)
			s.referenceFrames = [WebRTCReferenceBuffers]SourceFrame420{}
			s.monoReferenceFrames = [WebRTCReferenceBuffers]SourceFrameMono{}
			s.mono16ReferenceFrames = [WebRTCReferenceBuffers]SourceFrameMono16{}
			s.color16ReferenceFrames = [WebRTCReferenceBuffers]SourceFrame42016{}
		}
		return nil
	}

	var nextEncoders [WebRTCMaxSpatialLayers]*VideoEncoder
	var nextMonoEncoders [WebRTCMaxSpatialLayers]*MonochromeVideoEncoder
	var nextMono16Encoders [WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder
	var nextColor16Encoders [WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder
	if webRTCStreamHighBitDepthColor(normalized) {
		var err error
		nextColor16Encoders, err = s.buildReplacementColor16LayerEncoders(normalized, fps)
		if err != nil {
			return err
		}
		if err := s.closeEncoders(); err != nil {
			closeWebRTCStreamColor16Encoders(nextColor16Encoders)
			return err
		}
	} else if webRTCStreamHighBitDepthMonochrome(normalized) {
		var err error
		nextMono16Encoders, err = s.buildReplacementMono16LayerEncoders(normalized, fps)
		if err != nil {
			return err
		}
		if err := s.closeEncoders(); err != nil {
			closeWebRTCStreamMono16Encoders(nextMono16Encoders)
			return err
		}
	} else if webRTCStreamMonochrome(normalized) {
		var err error
		nextMonoEncoders, err = s.buildReplacementMonoLayerEncoders(normalized, fps)
		if err != nil {
			return err
		}
		if err := s.closeEncoders(); err != nil {
			closeWebRTCStreamMonoEncoders(nextMonoEncoders)
			return err
		}
	} else {
		var err error
		nextEncoders, err = s.buildReplacementLayerEncoders(normalized, fps)
		if err != nil {
			return err
		}
		if err := s.closeEncoders(); err != nil {
			closeWebRTCStreamEncoders(nextEncoders)
			return err
		}
	}
	s.encoders = nextEncoders
	s.monoEncoders = nextMonoEncoders
	s.mono16Encoders = nextMono16Encoders
	s.color16Encoders = nextColor16Encoders
	s.config = normalized
	s.scaledFrames = [WebRTCMaxSpatialLayers]SourceFrame420{}
	s.scaledMono = [WebRTCMaxSpatialLayers]SourceFrameMono{}
	s.scaledMono16 = [WebRTCMaxSpatialLayers]SourceFrameMono16{}
	s.scaledFrames16 = [WebRTCMaxSpatialLayers]SourceFrame42016{}
	s.layerScratch = [WebRTCMaxSpatialLayers][]byte{}
	s.descriptorScratch = [WebRTCMaxSpatialLayers][]byte{}
	s.referenceFrames = [WebRTCReferenceBuffers]SourceFrame420{}
	s.monoReferenceFrames = [WebRTCReferenceBuffers]SourceFrameMono{}
	s.mono16ReferenceFrames = [WebRTCReferenceBuffers]SourceFrameMono16{}
	s.color16ReferenceFrames = [WebRTCReferenceBuffers]SourceFrame42016{}
	s.state = webRTCEncoderStateForNextKey(s.state)
	return nil
}

func (s *WebRTCStream) updateLayerControls(config Config, fps int) error {
	if webRTCStreamHighBitDepthColor(config) {
		return s.updateColor16LayerControls(config, fps)
	}
	if webRTCStreamHighBitDepthMonochrome(config) {
		return s.updateMono16LayerControls(config, fps)
	}
	if webRTCStreamMonochrome(config) {
		return s.updateMonoLayerControls(config, fps)
	}
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc := s.encoders[i]
		if enc == nil {
			return ErrInvalidConfig
		}
		if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
			return err
		}
		enc.SetScreenContentSelection(true)
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.SetRateControlConfig(RateControlConfig{
				TargetBitsPerSecond: int(targetKbps) * 1000,
				FramesPerSecond:     fps,
				MinQIndex:           s.rcMinQ,
				MaxQIndex:           s.rcMaxQ,
			}); err != nil {
				return err
			}
		case RateControlCQP:
			if err := enc.SetQIndex(config.Quantizer); err != nil {
				return err
			}
		default:
			return ErrUnsupported
		}
	}
	return nil
}

func (s *WebRTCStream) updateMonoLayerControls(config Config, fps int) error {
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc := s.monoEncoders[i]
		if enc == nil {
			return ErrInvalidConfig
		}
		if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
			return err
		}
		enc.SetScreenContentSelection(true)
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.SetRateControlConfig(RateControlConfig{
				TargetBitsPerSecond: int(targetKbps) * 1000,
				FramesPerSecond:     fps,
				MinQIndex:           s.rcMinQ,
				MaxQIndex:           s.rcMaxQ,
			}); err != nil {
				return err
			}
		case RateControlCQP:
			if err := enc.SetQIndex(config.Quantizer); err != nil {
				return err
			}
		default:
			return ErrUnsupported
		}
	}
	return nil
}

func (s *WebRTCStream) updateMono16LayerControls(config Config, fps int) error {
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc := s.mono16Encoders[i]
		if enc == nil {
			return ErrInvalidConfig
		}
		if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
			return err
		}
		enc.SetScreenContentSelection(true)
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.SetRateControlConfig(RateControlConfig{
				TargetBitsPerSecond: int(targetKbps) * 1000,
				FramesPerSecond:     fps,
				MinQIndex:           s.rcMinQ,
				MaxQIndex:           s.rcMaxQ,
			}); err != nil {
				return err
			}
		case RateControlCQP:
			if err := enc.SetQIndex(config.Quantizer); err != nil {
				return err
			}
		default:
			return ErrUnsupported
		}
	}
	return nil
}

func (s *WebRTCStream) updateColor16LayerControls(config Config, fps int) error {
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc := s.color16Encoders[i]
		if enc == nil {
			return ErrInvalidConfig
		}
		if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
			return err
		}
		enc.SetScreenContentSelection(true)
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.SetRateControlConfig(RateControlConfig{
				TargetBitsPerSecond: int(targetKbps) * 1000,
				FramesPerSecond:     fps,
				MinQIndex:           s.rcMinQ,
				MaxQIndex:           s.rcMaxQ,
			}); err != nil {
				return err
			}
		case RateControlCQP:
			if err := enc.SetQIndex(config.Quantizer); err != nil {
				return err
			}
		default:
			return ErrUnsupported
		}
	}
	return nil
}

func (s *WebRTCStream) buildReplacementLayerEncoders(config Config, fps int) ([WebRTCMaxSpatialLayers]*VideoEncoder, error) {
	var encoders [WebRTCMaxSpatialLayers]*VideoEncoder
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc, err := newWebRTCStreamLayerEncoder(config, i, fps, s.rcMinQ, s.rcMaxQ)
		if err != nil {
			closeWebRTCStreamEncoders(encoders)
			return [WebRTCMaxSpatialLayers]*VideoEncoder{}, err
		}
		if s.tileColumns > 0 {
			enc.SetTileColumns(s.tileColumns)
		}
		s.applyConfiguredTileColumnsToVideo(config, enc)
		if s.goldenIntervalSet {
			enc.SetGoldenInterval(s.goldenInterval)
		}
		if err := enc.Prewarm(); err != nil {
			closeWebRTCStreamEncoders(encoders)
			_ = enc.Close()
			return [WebRTCMaxSpatialLayers]*VideoEncoder{}, err
		}
		encoders[i] = enc
	}
	return encoders, nil
}

func (s *WebRTCStream) buildReplacementMonoLayerEncoders(config Config, fps int) ([WebRTCMaxSpatialLayers]*MonochromeVideoEncoder, error) {
	var encoders [WebRTCMaxSpatialLayers]*MonochromeVideoEncoder
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc, err := newWebRTCStreamLayerMonoEncoder(config, i, fps, s.rcMinQ, s.rcMaxQ)
		if err != nil {
			closeWebRTCStreamMonoEncoders(encoders)
			return [WebRTCMaxSpatialLayers]*MonochromeVideoEncoder{}, err
		}
		if s.tileColumns > 0 {
			enc.SetTileColumns(s.tileColumns)
		}
		s.applyConfiguredTileColumnsToMono(config, enc)
		if s.goldenIntervalSet {
			enc.SetGoldenInterval(s.goldenInterval)
		}
		if err := enc.Prewarm(); err != nil {
			closeWebRTCStreamMonoEncoders(encoders)
			_ = enc.Close()
			return [WebRTCMaxSpatialLayers]*MonochromeVideoEncoder{}, err
		}
		encoders[i] = enc
	}
	return encoders, nil
}

func (s *WebRTCStream) buildReplacementMono16LayerEncoders(config Config, fps int) ([WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder, error) {
	var encoders [WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc, err := newWebRTCStreamLayerMono16Encoder(config, i, fps, s.rcMinQ, s.rcMaxQ)
		if err != nil {
			closeWebRTCStreamMono16Encoders(encoders)
			return [WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder{}, err
		}
		if s.tileColumns > 0 {
			enc.SetTileColumns(s.tileColumns)
		}
		s.applyConfiguredTileColumnsToMono16(config, enc)
		if s.goldenIntervalSet {
			enc.SetGoldenInterval(s.goldenInterval)
		}
		if err := enc.Prewarm(); err != nil {
			closeWebRTCStreamMono16Encoders(encoders)
			_ = enc.Close()
			return [WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder{}, err
		}
		encoders[i] = enc
	}
	return encoders, nil
}

func (s *WebRTCStream) buildReplacementColor16LayerEncoders(config Config, fps int) ([WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder, error) {
	var encoders [WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc, err := newWebRTCStreamLayerColor16Encoder(config, i, fps, s.rcMinQ, s.rcMaxQ)
		if err != nil {
			closeWebRTCStreamColor16Encoders(encoders)
			return [WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder{}, err
		}
		if s.tileColumns > 0 {
			enc.SetTileColumns(s.tileColumns)
		}
		s.applyConfiguredTileColumnsToColor16(config, enc)
		if s.goldenIntervalSet {
			enc.SetGoldenInterval(s.goldenInterval)
		}
		if err := enc.Prewarm(); err != nil {
			closeWebRTCStreamColor16Encoders(encoders)
			_ = enc.Close()
			return [WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder{}, err
		}
		encoders[i] = enc
	}
	return encoders, nil
}

func webRTCStreamLayerGeometryEqual(a Config, b Config) bool {
	if a.SpatialLayerCount != b.SpatialLayerCount {
		return false
	}
	for i := uint8(0); i < a.SpatialLayerCount; i++ {
		if a.SpatialLayers[i].Resolution != b.SpatialLayers[i].Resolution {
			return false
		}
	}
	return true
}

func webRTCStreamPixelFormatEqual(a Config, b Config) bool {
	return a.Profile == b.Profile &&
		a.BitDepth == b.BitDepth &&
		a.ColorConfig == b.ColorConfig
}

func (s *WebRTCStream) closeEncoders() error {
	var firstErr error
	for i := uint8(0); i < WebRTCMaxSpatialLayers; i++ {
		if s.encoders[i] != nil {
			if err := s.encoders[i].Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		if s.monoEncoders[i] != nil {
			if err := s.monoEncoders[i].Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		if s.mono16Encoders[i] != nil {
			if err := s.mono16Encoders[i].Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		if s.color16Encoders[i] != nil {
			if err := s.color16Encoders[i].Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}

func closeWebRTCStreamEncoders(encoders [WebRTCMaxSpatialLayers]*VideoEncoder) {
	for i := range encoders {
		if encoders[i] != nil {
			_ = encoders[i].Close()
		}
	}
}

func closeWebRTCStreamMonoEncoders(encoders [WebRTCMaxSpatialLayers]*MonochromeVideoEncoder) {
	for i := range encoders {
		if encoders[i] != nil {
			_ = encoders[i].Close()
		}
	}
}

func closeWebRTCStreamMono16Encoders(encoders [WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder) {
	for i := range encoders {
		if encoders[i] != nil {
			_ = encoders[i].Close()
		}
	}
}

func closeWebRTCStreamColor16Encoders(encoders [WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder) {
	for i := range encoders {
		if encoders[i] != nil {
			_ = encoders[i].Close()
		}
	}
}

func webRTCEncoderStateForNextKey(state WebRTCEncoderState) WebRTCEncoderState {
	return WebRTCEncoderState{
		NextOrderHint: state.NextOrderHint,
		NextFrameID:   state.NextFrameID,
	}
}

// Prewarm sizes the underlying encoders' reusable buffers without advancing
// the WebRTC frame metadata.
func (s *WebRTCStream) Prewarm() error {
	if s == nil {
		return nil
	}
	maxLayerBytes := 0
	for i := uint8(0); i < s.config.SpatialLayerCount; i++ {
		if layerBytes := webRTCStreamLayerPrewarmBytes(s.config, i); layerBytes > maxLayerBytes {
			maxLayerBytes = layerBytes
		}
	}
	descriptorBytes, err := webRTCStreamDescriptorPrewarmBytes(s.config)
	if err != nil {
		return err
	}
	for i := uint8(0); i < s.config.SpatialLayerCount; i++ {
		if s.encoders[i] != nil {
			if err := s.encoders[i].Prewarm(); err != nil {
				return err
			}
		} else if s.monoEncoders[i] != nil {
			if err := s.monoEncoders[i].Prewarm(); err != nil {
				return err
			}
		} else if s.mono16Encoders[i] != nil {
			if err := s.mono16Encoders[i].Prewarm(); err != nil {
				return err
			}
		} else if s.color16Encoders[i] != nil {
			if err := s.color16Encoders[i].Prewarm(); err != nil {
				return err
			}
		}
		if cap(s.layerScratch[i]) < maxLayerBytes {
			s.layerScratch[i] = make([]byte, 0, maxLayerBytes)
		}
		if cap(s.descriptorScratch[i]) < descriptorBytes {
			s.descriptorScratch[i] = make([]byte, 0, descriptorBytes)
		}
	}
	return nil
}

func webRTCStreamLayerPrewarmBytes(config Config, spatialID uint8) int {
	if spatialID >= config.SpatialLayerCount {
		return 0
	}
	resolution := config.SpatialLayers[spatialID].Resolution
	pixels := int(resolution.Width) * int(resolution.Height)
	if pixels < 1024 {
		pixels = 1024
	}
	return pixels + pixels/16 + 4096
}

func webRTCStreamDescriptorPrewarmBytes(config Config) (int, error) {
	state := WebRTCEncoderState{}
	steps := int(config.TemporalLayerCount)*2 + 2
	if steps < 4 {
		steps = 4
	}
	maxBytes := 0
	for step := 0; step < steps; step++ {
		unit, next, err := WebRTCNextTemporalUnitForState(config, state, step == 0)
		if err != nil {
			return 0, err
		}
		frameNum := webRTCPictureTemporalUnitFrameNum(unit)
		for i := uint8(0); i < frameNum; i++ {
			control, structure, err := webRTCPictureTemporalUnitFrameControl(unit, state, i)
			if err != nil {
				return 0, err
			}
			size, err := WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, control.AttachDependencyStructure)
			if err != nil {
				return 0, err
			}
			if size > maxBytes {
				maxBytes = size
			}
		}
		state = next
	}
	if maxBytes == 0 {
		return 0, ErrInvalidFrame
	}
	return maxBytes, nil
}

// Encode encodes one single-spatial frame and returns it with WebRTC packaging
// metadata.
func (s *WebRTCStream) Encode(src SourceFrame420, forceKey bool) (WebRTCEncodedFrame, error) {
	if s == nil || s.config.SpatialLayerCount != 1 {
		return WebRTCEncodedFrame{}, ErrUnsupported
	}
	if webRTCStreamMonochrome(s.config) {
		return WebRTCEncodedFrame{}, ErrUnsupported
	}
	picture, err := s.EncodePicture(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	return picture.Frames[0], nil
}

// EncodeMonochrome encodes one single-spatial native monochrome frame and
// returns it with WebRTC packaging metadata.
func (s *WebRTCStream) EncodeMonochrome(src SourceFrameMono, forceKey bool) (WebRTCEncodedFrame, error) {
	if s == nil || s.config.SpatialLayerCount != 1 || !webRTCStreamMonochrome(s.config) || webRTCStreamHighBitDepthMonochrome(s.config) {
		return WebRTCEncodedFrame{}, ErrUnsupported
	}
	picture, err := s.EncodeMonochromePicture(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	return picture.Frames[0], nil
}

// EncodeHighBitDepthMonochrome encodes one single-spatial native 10/12-bit
// monochrome frame and returns it with WebRTC packaging metadata.
func (s *WebRTCStream) EncodeHighBitDepthMonochrome(src SourceFrameMono16, forceKey bool) (WebRTCEncodedFrame, error) {
	if s == nil || s.config.SpatialLayerCount != 1 || !webRTCStreamHighBitDepthMonochrome(s.config) {
		return WebRTCEncodedFrame{}, ErrUnsupported
	}
	picture, err := s.EncodeHighBitDepthMonochromePicture(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	return picture.Frames[0], nil
}

// EncodeHighBitDepth420 encodes one single-spatial native 10/12-bit 4:2:0
// frame and returns it with WebRTC packaging metadata.
func (s *WebRTCStream) EncodeHighBitDepth420(src SourceFrame42016, forceKey bool) (WebRTCEncodedFrame, error) {
	if s == nil || s.config.SpatialLayerCount != 1 || !webRTCStreamHighBitDepth420(s.config) {
		return WebRTCEncodedFrame{}, ErrUnsupported
	}
	picture, err := s.EncodeHighBitDepth420Picture(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	return picture.Frames[0], nil
}

// EncodeHighBitDepthColor encodes one single-spatial native 10/12-bit color
// frame and returns it with WebRTC packaging metadata.
func (s *WebRTCStream) EncodeHighBitDepthColor(src SourceFrame42016, forceKey bool) (WebRTCEncodedFrame, error) {
	if s == nil || s.config.SpatialLayerCount != 1 || !webRTCStreamHighBitDepthColor(s.config) {
		return WebRTCEncodedFrame{}, ErrUnsupported
	}
	picture, err := s.EncodeHighBitDepthColorPicture(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	return picture.Frames[0], nil
}

// EncodePicture encodes one picture and returns one frame per active spatial
// layer with dependency descriptors matching the configured scalability mode.
func (s *WebRTCStream) EncodePicture(src SourceFrame420, forceKey bool) (WebRTCEncodedPicture, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return WebRTCEncodedPicture{}, ErrInvalidConfig
	}
	if webRTCStreamMonochrome(s.config) {
		return WebRTCEncodedPicture{}, ErrUnsupported
	}
	if src.Width != int(s.config.Resolution.Width) || src.Height != int(s.config.Resolution.Height) {
		return WebRTCEncodedPicture{}, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, s.config.Resolution.Width, s.config.Resolution.Height)
	}
	unit, next, err := WebRTCNextTemporalUnitForState(s.config, s.state, forceKey)
	if err != nil {
		return WebRTCEncodedPicture{}, err
	}
	frameNum := webRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || frameNum > s.config.SpatialLayerCount {
		return WebRTCEncodedPicture{}, ErrInvalidFrame
	}

	var picture WebRTCEncodedPicture
	picture.FrameNum = frameNum
	picture.Keyframe = unit.Key
	picture.Unit = unit
	metadataDims := webRTCScalabilityMetadataDimensionsForConfig(s.config)
	for i := uint8(0); i < frameNum; i++ {
		settings, ok := webRTCPictureUnitFrameSettings(unit, i)
		if !ok {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		layerSrc, err := s.sourceForLayer(src, settings.SpatialID, settings.Resolution)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		enc := s.encoders[settings.SpatialID]
		if enc == nil {
			return WebRTCEncodedPicture{}, ErrInvalidConfig
		}
		tu, key, err := s.encodePictureLayer(enc, layerSrc, settings, unit.Key)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if key != webRTCStreamExpectedCodedKey(unit.Key, settings, s.config.Scalability) {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		includeScalabilityMetadata := unit.Key && settings.SpatialID == 0
		layerSize, err := webRTCLayerTemporalUnitSize(tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if cap(s.layerScratch[i]) < layerSize {
			s.layerScratch[i] = make([]byte, 0, layerSize)
		}
		layerTU, err := appendWebRTCLayerTemporalUnit(s.layerScratch[i][:0], tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		s.layerScratch[i] = layerTU
		control, structure, err := webRTCPictureTemporalUnitFrameControl(unit, s.state, i)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		descriptor, err := s.appendPictureDependencyDescriptor(i, structure, control.GenericFrameInfo, control.AttachDependencyStructure)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		picture.Frames[i] = WebRTCEncodedFrame{
			TU:                        layerTU,
			Keyframe:                  unit.Key,
			CodedKeyframe:             key,
			LastFrameInPicture:        i+1 == frameNum,
			Info:                      control.GenericFrameInfo,
			Descriptor:                descriptor,
			Structure:                 structure,
			AttachDependencyStructure: control.AttachDependencyStructure,
		}
		if err := s.updateReferenceFrame(settings, enc); err != nil {
			return WebRTCEncodedPicture{}, err
		}
	}
	s.state = next
	return picture, nil
}

// EncodeMonochromePicture encodes one native monochrome WebRTC picture.
func (s *WebRTCStream) EncodeMonochromePicture(src SourceFrameMono, forceKey bool) (WebRTCEncodedPicture, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return WebRTCEncodedPicture{}, ErrInvalidConfig
	}
	if !webRTCStreamMonochrome(s.config) || webRTCStreamHighBitDepthMonochrome(s.config) {
		return WebRTCEncodedPicture{}, ErrUnsupported
	}
	if src.Width != int(s.config.Resolution.Width) || src.Height != int(s.config.Resolution.Height) {
		return WebRTCEncodedPicture{}, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, s.config.Resolution.Width, s.config.Resolution.Height)
	}
	unit, next, err := WebRTCNextTemporalUnitForState(s.config, s.state, forceKey)
	if err != nil {
		return WebRTCEncodedPicture{}, err
	}
	frameNum := webRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || frameNum > s.config.SpatialLayerCount {
		return WebRTCEncodedPicture{}, ErrInvalidFrame
	}

	var picture WebRTCEncodedPicture
	picture.FrameNum = frameNum
	picture.Keyframe = unit.Key
	picture.Unit = unit
	metadataDims := webRTCScalabilityMetadataDimensionsForConfig(s.config)
	for i := uint8(0); i < frameNum; i++ {
		settings, ok := webRTCPictureUnitFrameSettings(unit, i)
		if !ok {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		layerSrc, err := s.sourceForMonoLayer(src, settings.SpatialID, settings.Resolution)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		enc := s.monoEncoders[settings.SpatialID]
		if enc == nil {
			return WebRTCEncodedPicture{}, ErrInvalidConfig
		}
		tu, key, err := s.encodeMonoPictureLayer(enc, layerSrc, settings, unit.Key)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if key != webRTCStreamExpectedCodedKey(unit.Key, settings, s.config.Scalability) {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		includeScalabilityMetadata := unit.Key && settings.SpatialID == 0
		layerSize, err := webRTCLayerTemporalUnitSize(tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if cap(s.layerScratch[i]) < layerSize {
			s.layerScratch[i] = make([]byte, 0, layerSize)
		}
		layerTU, err := appendWebRTCLayerTemporalUnit(s.layerScratch[i][:0], tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		s.layerScratch[i] = layerTU
		control, structure, err := webRTCPictureTemporalUnitFrameControl(unit, s.state, i)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		descriptor, err := s.appendPictureDependencyDescriptor(i, structure, control.GenericFrameInfo, control.AttachDependencyStructure)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		picture.Frames[i] = WebRTCEncodedFrame{
			TU:                        layerTU,
			Keyframe:                  unit.Key,
			CodedKeyframe:             key,
			LastFrameInPicture:        i+1 == frameNum,
			Info:                      control.GenericFrameInfo,
			Descriptor:                descriptor,
			Structure:                 structure,
			AttachDependencyStructure: control.AttachDependencyStructure,
		}
		if err := s.updateMonoReferenceFrame(settings, enc); err != nil {
			return WebRTCEncodedPicture{}, err
		}
	}
	s.state = next
	return picture, nil
}

// EncodeHighBitDepthMonochromePicture encodes one native 10/12-bit monochrome
// WebRTC picture.
func (s *WebRTCStream) EncodeHighBitDepthMonochromePicture(src SourceFrameMono16, forceKey bool) (WebRTCEncodedPicture, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return WebRTCEncodedPicture{}, ErrInvalidConfig
	}
	if !webRTCStreamHighBitDepthMonochrome(s.config) {
		return WebRTCEncodedPicture{}, ErrUnsupported
	}
	if src.Width != int(s.config.Resolution.Width) || src.Height != int(s.config.Resolution.Height) {
		return WebRTCEncodedPicture{}, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, s.config.Resolution.Width, s.config.Resolution.Height)
	}
	if src.BitDepth != s.config.ColorConfig.BitDepth {
		return WebRTCEncodedPicture{}, fmt.Errorf("encoder: source bit depth %d does not match stream bit depth %d", src.BitDepth, s.config.ColorConfig.BitDepth)
	}
	unit, next, err := WebRTCNextTemporalUnitForState(s.config, s.state, forceKey)
	if err != nil {
		return WebRTCEncodedPicture{}, err
	}
	frameNum := webRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || frameNum > s.config.SpatialLayerCount {
		return WebRTCEncodedPicture{}, ErrInvalidFrame
	}

	var picture WebRTCEncodedPicture
	picture.FrameNum = frameNum
	picture.Keyframe = unit.Key
	picture.Unit = unit
	metadataDims := webRTCScalabilityMetadataDimensionsForConfig(s.config)
	for i := uint8(0); i < frameNum; i++ {
		settings, ok := webRTCPictureUnitFrameSettings(unit, i)
		if !ok {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		layerSrc, err := s.sourceForMono16Layer(src, settings.SpatialID, settings.Resolution)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		enc := s.mono16Encoders[settings.SpatialID]
		if enc == nil {
			return WebRTCEncodedPicture{}, ErrInvalidConfig
		}
		tu, key, err := s.encodeMono16PictureLayer(enc, layerSrc, settings, unit.Key)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if key != webRTCStreamExpectedCodedKey(unit.Key, settings, s.config.Scalability) {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		includeScalabilityMetadata := unit.Key && settings.SpatialID == 0
		layerSize, err := webRTCLayerTemporalUnitSize(tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if cap(s.layerScratch[i]) < layerSize {
			s.layerScratch[i] = make([]byte, 0, layerSize)
		}
		layerTU, err := appendWebRTCLayerTemporalUnit(s.layerScratch[i][:0], tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		s.layerScratch[i] = layerTU
		control, structure, err := webRTCPictureTemporalUnitFrameControl(unit, s.state, i)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		descriptor, err := s.appendPictureDependencyDescriptor(i, structure, control.GenericFrameInfo, control.AttachDependencyStructure)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		picture.Frames[i] = WebRTCEncodedFrame{
			TU:                        layerTU,
			Keyframe:                  unit.Key,
			CodedKeyframe:             key,
			LastFrameInPicture:        i+1 == frameNum,
			Info:                      control.GenericFrameInfo,
			Descriptor:                descriptor,
			Structure:                 structure,
			AttachDependencyStructure: control.AttachDependencyStructure,
		}
		if err := s.updateMono16ReferenceFrame(settings, enc); err != nil {
			return WebRTCEncodedPicture{}, err
		}
	}
	s.state = next
	return picture, nil
}

// EncodeHighBitDepth420Picture encodes one native 10/12-bit 4:2:0 WebRTC
// picture.
func (s *WebRTCStream) EncodeHighBitDepth420Picture(src SourceFrame42016, forceKey bool) (WebRTCEncodedPicture, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return WebRTCEncodedPicture{}, ErrInvalidConfig
	}
	if !webRTCStreamHighBitDepth420(s.config) {
		return WebRTCEncodedPicture{}, ErrUnsupported
	}
	return s.EncodeHighBitDepthColorPicture(src, forceKey)
}

// EncodeHighBitDepthColorPicture encodes one native 10/12-bit color WebRTC
// picture.
func (s *WebRTCStream) EncodeHighBitDepthColorPicture(src SourceFrame42016, forceKey bool) (WebRTCEncodedPicture, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return WebRTCEncodedPicture{}, ErrInvalidConfig
	}
	if !webRTCStreamHighBitDepthColor(s.config) {
		return WebRTCEncodedPicture{}, ErrUnsupported
	}
	if src.Width != int(s.config.Resolution.Width) || src.Height != int(s.config.Resolution.Height) {
		return WebRTCEncodedPicture{}, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, s.config.Resolution.Width, s.config.Resolution.Height)
	}
	if src.BitDepth != s.config.ColorConfig.BitDepth {
		return WebRTCEncodedPicture{}, fmt.Errorf("encoder: source bit depth %d does not match stream bit depth %d", src.BitDepth, s.config.ColorConfig.BitDepth)
	}
	unit, next, err := WebRTCNextTemporalUnitForState(s.config, s.state, forceKey)
	if err != nil {
		return WebRTCEncodedPicture{}, err
	}
	frameNum := webRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || frameNum > s.config.SpatialLayerCount {
		return WebRTCEncodedPicture{}, ErrInvalidFrame
	}

	var picture WebRTCEncodedPicture
	picture.FrameNum = frameNum
	picture.Keyframe = unit.Key
	picture.Unit = unit
	metadataDims := webRTCScalabilityMetadataDimensionsForConfig(s.config)
	for i := uint8(0); i < frameNum; i++ {
		settings, ok := webRTCPictureUnitFrameSettings(unit, i)
		if !ok {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		layerSrc, err := s.sourceForColor16Layer(src, settings.SpatialID, settings.Resolution)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		enc := s.color16Encoders[settings.SpatialID]
		if enc == nil {
			return WebRTCEncodedPicture{}, ErrInvalidConfig
		}
		tu, key, err := s.encode42016PictureLayer(enc, layerSrc, settings, unit.Key)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if key != webRTCStreamExpectedCodedKey(unit.Key, settings, s.config.Scalability) {
			return WebRTCEncodedPicture{}, ErrInvalidFrame
		}
		includeScalabilityMetadata := unit.Key && settings.SpatialID == 0
		layerSize, err := webRTCLayerTemporalUnitSize(tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if cap(s.layerScratch[i]) < layerSize {
			s.layerScratch[i] = make([]byte, 0, layerSize)
		}
		layerTU, err := appendWebRTCLayerTemporalUnit(s.layerScratch[i][:0], tu, settings.TemporalID, settings.SpatialID, s.config.Scalability, includeScalabilityMetadata, metadataDims)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		s.layerScratch[i] = layerTU
		control, structure, err := webRTCPictureTemporalUnitFrameControl(unit, s.state, i)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		descriptor, err := s.appendPictureDependencyDescriptor(i, structure, control.GenericFrameInfo, control.AttachDependencyStructure)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		picture.Frames[i] = WebRTCEncodedFrame{
			TU:                        layerTU,
			Keyframe:                  unit.Key,
			CodedKeyframe:             key,
			LastFrameInPicture:        i+1 == frameNum,
			Info:                      control.GenericFrameInfo,
			Descriptor:                descriptor,
			Structure:                 structure,
			AttachDependencyStructure: control.AttachDependencyStructure,
		}
		if err := s.updateColor16ReferenceFrame(settings, enc); err != nil {
			return WebRTCEncodedPicture{}, err
		}
	}
	s.state = next
	return picture, nil
}

func (s *WebRTCStream) encodePictureLayer(enc *VideoEncoder, layerSrc SourceFrame420, settings FrameEncodeSettings, keyPicture bool) ([]byte, bool, error) {
	if !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
		tu, key, err := enc.EncodeWithTemporalID(layerSrc, keyPicture, settings.TemporalID)
		return tu, key, err
	}
	maxWidth, maxHeight, err := s.sequenceMaxCodedSize()
	if err != nil {
		return nil, false, err
	}
	switch settings.Type {
	case FrameTypeKey:
		if !keyPicture {
			return nil, false, ErrInvalidFrame
		}
		if layerSrc.Width != enc.renderWidth || layerSrc.Height != enc.renderHeight {
			return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", layerSrc.Width, layerSrc.Height, enc.renderWidth, enc.renderHeight)
		}
		if enc.renderWidth != enc.width || enc.renderHeight != enc.height {
			layerSrc = enc.padSource(layerSrc)
		}
		tu, err := enc.encodeKeyWithSequenceMax(layerSrc, maxWidth, maxHeight)
		return tu, true, err
	case FrameTypeDelta:
		if settings.ReferenceCount == 0 {
			return nil, false, ErrInvalidFrame
		}
		refSlot, err := webRTCStreamCodedReferenceBuffer(settings, s.config.Scalability)
		if err != nil {
			return nil, false, err
		}
		if refSlot >= WebRTCReferenceBuffers || s.referenceFrames[refSlot].Y == nil {
			return nil, false, ErrInvalidFrame
		}
		tu, err := enc.encodeReferencePFrameWithSequenceMax(layerSrc, s.referenceFrames[refSlot], refSlot, settings, maxWidth, maxHeight)
		return tu, false, err
	default:
		return nil, false, ErrInvalidFrame
	}
}

func webRTCStreamUsesSharedReferenceSlotCoding(config Config) bool {
	return config.SpatialLayerCount > 1 &&
		!config.Scalability.IsSimulcast()
}

func (s *WebRTCStream) encodeMonoPictureLayer(enc *MonochromeVideoEncoder, layerSrc SourceFrameMono, settings FrameEncodeSettings, keyPicture bool) ([]byte, bool, error) {
	if !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
		tu, key, err := enc.EncodeWithTemporalID(layerSrc, keyPicture, settings.TemporalID)
		return tu, key, err
	}
	maxWidth, maxHeight, err := s.sequenceMaxCodedSizeMono()
	if err != nil {
		return nil, false, err
	}
	switch settings.Type {
	case FrameTypeKey:
		if !keyPicture {
			return nil, false, ErrInvalidFrame
		}
		if layerSrc.Width != enc.renderWidth || layerSrc.Height != enc.renderHeight {
			return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", layerSrc.Width, layerSrc.Height, enc.renderWidth, enc.renderHeight)
		}
		if enc.renderWidth != enc.width || enc.renderHeight != enc.height {
			layerSrc = enc.padSource(layerSrc)
		}
		tu, err := enc.encodeKeyWithSequenceMax(layerSrc, maxWidth, maxHeight)
		return tu, true, err
	case FrameTypeDelta:
		if settings.ReferenceCount == 0 {
			return nil, false, ErrInvalidFrame
		}
		refSlot, err := webRTCStreamCodedReferenceBuffer(settings, s.config.Scalability)
		if err != nil {
			return nil, false, err
		}
		if refSlot >= WebRTCReferenceBuffers || s.monoReferenceFrames[refSlot].Y == nil {
			return nil, false, ErrInvalidFrame
		}
		tu, err := enc.encodeReferencePFrameWithSequenceMax(layerSrc, s.monoReferenceFrames[refSlot], refSlot, settings, maxWidth, maxHeight)
		return tu, false, err
	default:
		return nil, false, ErrInvalidFrame
	}
}

func (s *WebRTCStream) encodeMono16PictureLayer(enc *HighBitDepthMonochromeVideoEncoder, layerSrc SourceFrameMono16, settings FrameEncodeSettings, keyPicture bool) ([]byte, bool, error) {
	if !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
		tu, key, err := enc.EncodeWithTemporalID(layerSrc, keyPicture, settings.TemporalID)
		return tu, key, err
	}
	maxWidth, maxHeight, err := s.sequenceMaxCodedSizeMono16()
	if err != nil {
		return nil, false, err
	}
	switch settings.Type {
	case FrameTypeKey:
		if !keyPicture {
			return nil, false, ErrInvalidFrame
		}
		if layerSrc.Width != enc.renderWidth || layerSrc.Height != enc.renderHeight {
			return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", layerSrc.Width, layerSrc.Height, enc.renderWidth, enc.renderHeight)
		}
		if enc.renderWidth != enc.width || enc.renderHeight != enc.height {
			layerSrc = enc.padSource(layerSrc)
		}
		tu, err := enc.encodeKeyWithSequenceMax(layerSrc, maxWidth, maxHeight)
		return tu, true, err
	case FrameTypeDelta:
		if settings.ReferenceCount == 0 {
			return nil, false, ErrInvalidFrame
		}
		refSlot, err := webRTCStreamCodedReferenceBuffer(settings, s.config.Scalability)
		if err != nil {
			return nil, false, err
		}
		if refSlot >= WebRTCReferenceBuffers || s.mono16ReferenceFrames[refSlot].Y == nil {
			return nil, false, ErrInvalidFrame
		}
		tu, err := enc.encodeReferencePFrameWithSequenceMax(layerSrc, s.mono16ReferenceFrames[refSlot], refSlot, settings, maxWidth, maxHeight)
		return tu, false, err
	default:
		return nil, false, ErrInvalidFrame
	}
}

func (s *WebRTCStream) encode42016PictureLayer(enc *HighBitDepth420VideoEncoder, layerSrc SourceFrame42016, settings FrameEncodeSettings, keyPicture bool) ([]byte, bool, error) {
	if !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
		tu, key, err := enc.EncodeWithTemporalID(layerSrc, keyPicture, settings.TemporalID)
		return tu, key, err
	}
	maxWidth, maxHeight, err := s.sequenceMaxCodedSize42016()
	if err != nil {
		return nil, false, err
	}
	switch settings.Type {
	case FrameTypeKey:
		if !keyPicture {
			return nil, false, ErrInvalidFrame
		}
		if layerSrc.Width != enc.renderWidth || layerSrc.Height != enc.renderHeight {
			return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", layerSrc.Width, layerSrc.Height, enc.renderWidth, enc.renderHeight)
		}
		if enc.renderWidth != enc.width || enc.renderHeight != enc.height {
			layerSrc = enc.padSource(layerSrc)
		}
		tu, err := enc.encodeKeyWithSequenceMax(layerSrc, maxWidth, maxHeight)
		return tu, true, err
	case FrameTypeDelta:
		if settings.ReferenceCount == 0 {
			return nil, false, ErrInvalidFrame
		}
		refSlot, err := webRTCStreamCodedReferenceBuffer(settings, s.config.Scalability)
		if err != nil {
			return nil, false, err
		}
		if refSlot >= WebRTCReferenceBuffers || s.color16ReferenceFrames[refSlot].Y == nil {
			return nil, false, ErrInvalidFrame
		}
		tu, err := enc.encodeReferencePFrameWithSequenceMax(layerSrc, s.color16ReferenceFrames[refSlot], refSlot, settings, maxWidth, maxHeight)
		return tu, false, err
	default:
		return nil, false, ErrInvalidFrame
	}
}

func webRTCStreamCodedReferenceBuffer(settings FrameEncodeSettings, mode ScalabilityMode) (uint8, error) {
	if settings.ReferenceCount == 0 || settings.ReferenceCount > WebRTCMaxFrameReferences {
		return 0, ErrInvalidFrame
	}
	index := uint8(0)
	if !mode.IsSimulcast() && !mode.UsesKeyFrameInterLayerDependency() && settings.SpatialID > 0 && settings.ReferenceCount > 1 {
		index = settings.ReferenceCount - 1
	}
	ref := settings.ReferenceBuffers[index]
	if ref >= WebRTCReferenceBuffers {
		return 0, ErrInvalidFrame
	}
	return ref, nil
}

func webRTCStreamExpectedCodedKey(keyPicture bool, settings FrameEncodeSettings, mode ScalabilityMode) bool {
	if keyPicture && !mode.IsSimulcast() && settings.Type == FrameTypeDelta {
		return false
	}
	return keyPicture
}

func (s *WebRTCStream) appendPictureDependencyDescriptor(frameIndex uint8, structure WebRTCFrameDependencyStructure, info WebRTCGenericFrameInfo, attachStructure bool) ([]byte, error) {
	if s == nil || frameIndex >= WebRTCMaxSpatialLayers {
		return nil, ErrInvalidFrame
	}
	size, err := WebRTCDependencyDescriptorSize(structure, info, attachStructure)
	if err != nil {
		return nil, err
	}
	if cap(s.descriptorScratch[frameIndex]) < size {
		s.descriptorScratch[frameIndex] = make([]byte, 0, size)
	}
	descriptor, err := AppendWebRTCDependencyDescriptor(s.descriptorScratch[frameIndex][:0], structure, info, true, true, attachStructure)
	if err != nil {
		return nil, err
	}
	s.descriptorScratch[frameIndex] = descriptor
	return descriptor, nil
}

func (s *WebRTCStream) sequenceMaxCodedSize() (int, int, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return 0, 0, ErrInvalidConfig
	}
	top := s.config.SpatialLayerCount - 1
	enc := s.encoders[top]
	if enc == nil {
		return 0, 0, ErrInvalidConfig
	}
	return enc.width, enc.height, nil
}

func (s *WebRTCStream) sequenceMaxCodedSizeMono() (int, int, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return 0, 0, ErrInvalidConfig
	}
	top := s.config.SpatialLayerCount - 1
	enc := s.monoEncoders[top]
	if enc == nil {
		return 0, 0, ErrInvalidConfig
	}
	return enc.width, enc.height, nil
}

func (s *WebRTCStream) sequenceMaxCodedSizeMono16() (int, int, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return 0, 0, ErrInvalidConfig
	}
	top := s.config.SpatialLayerCount - 1
	enc := s.mono16Encoders[top]
	if enc == nil {
		return 0, 0, ErrInvalidConfig
	}
	return enc.width, enc.height, nil
}

func (s *WebRTCStream) sequenceMaxCodedSize42016() (int, int, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return 0, 0, ErrInvalidConfig
	}
	top := s.config.SpatialLayerCount - 1
	enc := s.color16Encoders[top]
	if enc == nil {
		return 0, 0, ErrInvalidConfig
	}
	return enc.width, enc.height, nil
}

func (s *WebRTCStream) updateReferenceFrame(settings FrameEncodeSettings, enc *VideoEncoder) error {
	if !settings.UpdateBufferSet {
		return nil
	}
	if settings.UpdateBuffer >= WebRTCReferenceBuffers || enc == nil {
		return ErrInvalidFrame
	}
	recon, err := webRTCStreamLayerReconstruction(enc)
	if err != nil {
		return err
	}
	copyFrameInto(&s.referenceFrames[settings.UpdateBuffer], recon)
	return nil
}

func (s *WebRTCStream) updateMonoReferenceFrame(settings FrameEncodeSettings, enc *MonochromeVideoEncoder) error {
	if !settings.UpdateBufferSet {
		return nil
	}
	if settings.UpdateBuffer >= WebRTCReferenceBuffers || enc == nil {
		return ErrInvalidFrame
	}
	recon := enc.Recon()
	if recon.Y == nil {
		return ErrInvalidFrame
	}
	copyMonoFrameInto(&s.monoReferenceFrames[settings.UpdateBuffer], recon)
	return nil
}

func (s *WebRTCStream) updateMono16ReferenceFrame(settings FrameEncodeSettings, enc *HighBitDepthMonochromeVideoEncoder) error {
	if !settings.UpdateBufferSet {
		return nil
	}
	if settings.UpdateBuffer >= WebRTCReferenceBuffers || enc == nil {
		return ErrInvalidFrame
	}
	recon := enc.Recon()
	if recon.Y == nil {
		return ErrInvalidFrame
	}
	copyMono16FrameInto(&s.mono16ReferenceFrames[settings.UpdateBuffer], recon)
	return nil
}

func (s *WebRTCStream) update42016ReferenceFrame(settings FrameEncodeSettings, enc *HighBitDepth420VideoEncoder) error {
	return s.updateColor16ReferenceFrame(settings, enc)
}

func (s *WebRTCStream) updateColor16ReferenceFrame(settings FrameEncodeSettings, enc *HighBitDepth420VideoEncoder) error {
	if !settings.UpdateBufferSet {
		return nil
	}
	if settings.UpdateBuffer >= WebRTCReferenceBuffers || enc == nil {
		return ErrInvalidFrame
	}
	recon := enc.Recon()
	if recon.Y == nil {
		return ErrInvalidFrame
	}
	copyColor16FrameInto(&s.color16ReferenceFrames[settings.UpdateBuffer], recon, parserColorConfig(s.config.ColorConfig))
	return nil
}

func webRTCStreamLayerReconstruction(enc *VideoEncoder) (SourceFrame420, error) {
	if enc == nil {
		return SourceFrame420{}, ErrInvalidConfig
	}
	if err := enc.joinFilter(); err != nil {
		return SourceFrame420{}, err
	}
	recon := enc.lastRecon
	if recon.Y == nil {
		recon = enc.recon
	}
	if recon.Y == nil {
		return SourceFrame420{}, ErrInvalidFrame
	}
	return recon, nil
}

func (s *WebRTCStream) sourceForLayer(src SourceFrame420, spatialID uint8, resolution Resolution) (SourceFrame420, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrame420{}, ErrInvalidFrame
	}
	if src.Width == int(resolution.Width) && src.Height == int(resolution.Height) {
		return src, nil
	}
	return scaleSourceFrameColorNearest(&s.scaledFrames[spatialID], src, int(resolution.Width), int(resolution.Height), parserColorConfig(s.config.ColorConfig))
}

func (s *WebRTCStream) sourceForMonoLayer(src SourceFrameMono, spatialID uint8, resolution Resolution) (SourceFrameMono, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrameMono{}, ErrInvalidFrame
	}
	if src.Width == int(resolution.Width) && src.Height == int(resolution.Height) {
		return src, nil
	}
	return scaleSourceFrameMonoNearest(&s.scaledMono[spatialID], src, int(resolution.Width), int(resolution.Height))
}

func (s *WebRTCStream) sourceForMono16Layer(src SourceFrameMono16, spatialID uint8, resolution Resolution) (SourceFrameMono16, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrameMono16{}, ErrInvalidFrame
	}
	if src.Width == int(resolution.Width) && src.Height == int(resolution.Height) {
		return src, nil
	}
	return scaleSourceFrameMono16Nearest(&s.scaledMono16[spatialID], src, int(resolution.Width), int(resolution.Height))
}

func (s *WebRTCStream) sourceFor42016Layer(src SourceFrame42016, spatialID uint8, resolution Resolution) (SourceFrame42016, error) {
	return s.sourceForColor16Layer(src, spatialID, resolution)
}

func (s *WebRTCStream) sourceForColor16Layer(src SourceFrame42016, spatialID uint8, resolution Resolution) (SourceFrame42016, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrame42016{}, ErrInvalidFrame
	}
	if src.Width == int(resolution.Width) && src.Height == int(resolution.Height) {
		return src, nil
	}
	return scaleSourceFrameColor16Nearest(&s.scaledFrames16[spatialID], src, int(resolution.Width), int(resolution.Height), parserColorConfig(s.config.ColorConfig))
}

func scaleSourceFrameColorNearest(dst *SourceFrame420, src SourceFrame420, width, height int, color parser.ColorConfig) (SourceFrame420, error) {
	if width <= 0 || height <= 0 {
		return SourceFrame420{}, ErrInvalidFrame
	}
	cw, ch := chromaWidthForColor(width, color), chromaHeightForColor(height, color)
	srcCW, srcCH := chromaWidthForColor(src.Width, color), chromaHeightForColor(src.Height, color)
	if len(dst.Y) != width*height {
		dst.Y = make([]byte, width*height)
	}
	if len(dst.U) != cw*ch {
		dst.U = make([]byte, cw*ch)
	}
	if len(dst.V) != cw*ch {
		dst.V = make([]byte, cw*ch)
	}
	dst.YStride = width
	dst.ChromaStride = cw
	dst.Width = width
	dst.Height = height
	scalePlaneNearest(dst.Y, width, width, height, src.Y, src.YStride, src.Width, src.Height)
	scalePlaneNearest(dst.U, cw, cw, ch, src.U, src.ChromaStride, srcCW, srcCH)
	scalePlaneNearest(dst.V, cw, cw, ch, src.V, src.ChromaStride, srcCW, srcCH)
	return *dst, nil
}

func scaleSourceFrameMonoNearest(dst *SourceFrameMono, src SourceFrameMono, width, height int) (SourceFrameMono, error) {
	if width <= 0 || height <= 0 {
		return SourceFrameMono{}, ErrInvalidFrame
	}
	if len(dst.Y) != width*height {
		dst.Y = make([]byte, width*height)
	}
	dst.YStride = width
	dst.Width = width
	dst.Height = height
	scalePlaneNearest(dst.Y, width, width, height, src.Y, src.YStride, src.Width, src.Height)
	return *dst, nil
}

var scalePlaneNearestImpl = scalePlaneNearestPureGo

func scalePlaneNearest(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	scalePlaneNearestImpl(dst, dstStride, dstWidth, dstHeight, src, srcStride, srcWidth, srcHeight)
}

func scalePlaneNearestPureGo(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	for y := 0; y < dstHeight; y++ {
		sy := y * srcHeight / dstHeight
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[sy*srcStride:]
		for x := 0; x < dstWidth; x++ {
			drow[x] = srow[x*srcWidth/dstWidth]
		}
	}
}

func appendWebRTCLayerTemporalUnit(dst []byte, src []byte, temporalID uint8, spatialID uint8, scalability ScalabilityMode, includeScalabilityMetadata bool, metadataDims webRTCScalabilityMetadataDimensions) ([]byte, error) {
	it := obu.NewLowOverheadIterator(src)
	out := dst
	metadataInserted := false
	for {
		unit, ok, err := it.Next()
		if err != nil {
			return dst, err
		}
		if !ok {
			return out, nil
		}
		switch unit.Header.Type {
		case obu.TypeFrameHeader, obu.TypeFrame, obu.TypeTileGroup, obu.TypeRedundantFrameHeader:
			next, err := AppendLowOverheadOBU(out, OBU{
				Type:       unit.Header.Type,
				TemporalID: temporalID,
				SpatialID:  spatialID,
				Payload:    unit.Payload,
			})
			if err != nil {
				return dst, err
			}
			out = next
		default:
			next, err := AppendLowOverheadOBU(out, OBU{
				Type:    unit.Header.Type,
				Payload: unit.Payload,
			})
			if err != nil {
				return dst, err
			}
			out = next
		}
		if includeScalabilityMetadata && !metadataInserted && unit.Header.Type == obu.TypeSequenceHeader {
			next, _, err := appendLowOverheadWebRTCScalabilityMetadataOBU(out, scalability, metadataDims)
			if err != nil {
				return dst, err
			}
			out = next
			metadataInserted = true
		}
	}
}

func webRTCLayerTemporalUnitSize(src []byte, temporalID uint8, spatialID uint8, scalability ScalabilityMode, includeScalabilityMetadata bool, metadataDims webRTCScalabilityMetadataDimensions) (int, error) {
	it := obu.NewLowOverheadIterator(src)
	total := 0
	metadataInserted := false
	for {
		unit, ok, err := it.Next()
		if err != nil {
			return 0, err
		}
		if !ok {
			return total, nil
		}
		out := OBU{Type: unit.Header.Type, Payload: unit.Payload}
		switch unit.Header.Type {
		case obu.TypeFrameHeader, obu.TypeFrame, obu.TypeTileGroup, obu.TypeRedundantFrameHeader:
			out.TemporalID = temporalID
			out.SpatialID = spatialID
		}
		size, err := LowOverheadOBUSize(out)
		if err != nil {
			return 0, err
		}
		total += size
		if includeScalabilityMetadata && !metadataInserted && unit.Header.Type == obu.TypeSequenceHeader {
			metadataSize, ok, err := lowOverheadWebRTCScalabilityMetadataOBUSize(scalability, metadataDims)
			if err != nil {
				return 0, err
			}
			if ok {
				total += metadataSize
			}
			metadataInserted = true
		}
	}
}

func webRTCStreamFramesPerSecond(rate Rational) int {
	if !rate.Valid() {
		return 0
	}
	fps := int((int64(rate.Num) + int64(rate.Den)/2) / int64(rate.Den))
	if fps < 1 {
		return 1
	}
	return fps
}

func webRTCPictureTemporalUnitFrameNum(unit WebRTCPictureTemporalUnit) uint8 {
	if unit.Key {
		return unit.KeyUnit.FrameNum
	}
	if unit.Delta {
		return unit.DeltaUnit.FrameNum
	}
	return 0
}

func webRTCPictureUnitFrameSettings(unit WebRTCPictureTemporalUnit, frameIndex uint8) (FrameEncodeSettings, bool) {
	if unit.Key == unit.Delta {
		return FrameEncodeSettings{}, false
	}
	if unit.Key {
		if frameIndex >= unit.KeyUnit.FrameNum {
			return FrameEncodeSettings{}, false
		}
		return unit.KeyUnit.Frames[frameIndex], true
	}
	if frameIndex >= unit.DeltaUnit.FrameNum {
		return FrameEncodeSettings{}, false
	}
	return unit.DeltaUnit.Frames[frameIndex], true
}

func webRTCPictureTemporalUnitFrameControl(unit WebRTCPictureTemporalUnit, state WebRTCEncoderState, frameIndex uint8) (WebRTCFrameControl, WebRTCFrameDependencyStructure, error) {
	if unit.Key == unit.Delta {
		return WebRTCFrameControl{}, WebRTCFrameDependencyStructure{}, ErrInvalidFrame
	}
	if unit.Key {
		if frameIndex >= unit.KeyUnit.Control.FrameNum {
			return WebRTCFrameControl{}, WebRTCFrameDependencyStructure{}, ErrInvalidFrame
		}
		return unit.KeyUnit.Control.Frames[frameIndex], unit.KeyUnit.Control.DependencyStructure, nil
	}
	if frameIndex >= unit.DeltaUnit.Control.FrameNum || !state.DependencyStructureState.Valid {
		return WebRTCFrameControl{}, WebRTCFrameDependencyStructure{}, ErrInvalidFrame
	}
	return unit.DeltaUnit.Control.Frames[frameIndex], state.DependencyStructureState.Structure, nil
}
