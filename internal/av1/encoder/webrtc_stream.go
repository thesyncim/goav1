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

	encoders        [WebRTCMaxSpatialLayers]*VideoEncoder
	monoEncoders    [WebRTCMaxSpatialLayers]*MonochromeVideoEncoder
	mono16Encoders  [WebRTCMaxSpatialLayers]*HighBitDepthMonochromeVideoEncoder
	color16Encoders [WebRTCMaxSpatialLayers]*HighBitDepth420VideoEncoder
	scaledFrames    [WebRTCMaxSpatialLayers]SourceFrame420
	scaledMono      [WebRTCMaxSpatialLayers]SourceFrameMono
	scaledMono16    [WebRTCMaxSpatialLayers]SourceFrameMono16
	scaledFrames16  [WebRTCMaxSpatialLayers]SourceFrame42016
	// layerGen[k] == pictureGen marks layer k's scaled source as built for
	// the current picture (sourceForLayer).
	layerGen [WebRTCMaxSpatialLayers]uint64
	// layerMEGen[k] == pictureGen marks layer k's frame ME as run for the
	// current picture (webRTCMESeed).
	layerMEGen        [WebRTCMaxSpatialLayers]uint64
	pictureGen        uint64
	scaleScratch      boxScaleScratch
	layerScratch      [WebRTCMaxSpatialLayers][]byte
	descriptorScratch [WebRTCMaxSpatialLayers][]byte

	// refPool holds the 8-bit 4:2:0 shared reference pictures, SVT's
	// reference-picture pool: refSlot names the pool entry each valid slot
	// references and refPoolRefs counts the slots per entry. A layer's
	// fresh reconstruction is adopted by swapping buffers with a free entry
	// (see adoptReference), never copied.
	refPool     [webRTCRefPoolSize]SourceFrame420
	refPoolRefs [webRTCRefPoolSize]uint8
	refSlot     [WebRTCReferenceBuffers]uint8
	// refSize records each valid slot's picture geometry in every mode.
	refSize                [WebRTCReferenceBuffers][2]int32
	monoReferenceFrames    [WebRTCReferenceBuffers]SourceFrameMono
	mono16ReferenceFrames  [WebRTCReferenceBuffers]SourceFrameMono16
	color16ReferenceFrames [WebRTCReferenceBuffers]SourceFrame42016
	// referenceValid marks the slots holding a reference; dropping a slot
	// keeps its buffers for reuse.
	referenceValid [WebRTCReferenceBuffers]bool
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
	if _, err := rc.buffers().resolve(); err != nil {
		return nil, err
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
		BufferInitialMs:   int32(rc.BufferInitialMs),
		BufferOptimalMs:   int32(rc.BufferOptimalMs),
		BufferMaxMs:       int32(rc.BufferMaxMs),
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
	// SVT-AV1 CBR q range (min_qp_allowed 4, max_qp_allowed 63).
	stream.rcMinQ, stream.rcMaxQ = 16, 255
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
			BufferInitialMs:     int(config.BufferInitialMs),
			BufferOptimalMs:     int(config.BufferOptimalMs),
			BufferMaxMs:         int(config.BufferMaxMs),
		}, config.ColorConfig)
	case RateControlCQP:
		enc, err = newVideoEncoderWithColor(int(layer.Resolution.Width), int(layer.Resolution.Height), config.Quantizer, config.ColorConfig)
	default:
		return nil, ErrUnsupported
	}
	if err != nil {
		return nil, err
	}
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
			// The 8-bit encoders run SVT-AV1's controller, which takes the range
			// as its best and worst quality.
			if enc.svtRC.enabled {
				enc.svtRC.setQRange(minQ, maxQ)
			}
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
			s.referenceValid = [WebRTCReferenceBuffers]bool{}
			s.refPoolRefs = [webRTCRefPoolSize]uint8{}
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
	s.layerGen = [WebRTCMaxSpatialLayers]uint64{}
	s.layerMEGen = [WebRTCMaxSpatialLayers]uint64{}
	s.layerScratch = [WebRTCMaxSpatialLayers][]byte{}
	s.descriptorScratch = [WebRTCMaxSpatialLayers][]byte{}
	s.referenceValid = [WebRTCReferenceBuffers]bool{}
	s.refPoolRefs = [webRTCRefPoolSize]uint8{}
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
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.UpdateRateControlConfig(RateControlConfig{
				TargetBitsPerSecond: int(targetKbps) * 1000,
				FramesPerSecond:     fps,
				MinQIndex:           s.rcMinQ,
				MaxQIndex:           s.rcMaxQ,
				BufferInitialMs:     int(config.BufferInitialMs),
				BufferOptimalMs:     int(config.BufferOptimalMs),
				BufferMaxMs:         int(config.BufferMaxMs),
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
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.UpdateRateControlConfig(RateControlConfig{
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
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.UpdateRateControlConfig(RateControlConfig{
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
		if err := enc.SetContentHint(config.Content); err != nil {
			return err
		}
		if err := enc.SetEffortLevel(config.Speed); err != nil {
			return err
		}
		switch config.RateControl {
		case RateControlCBR:
			targetKbps := webRTCStreamLayerTargetKbps(config, i)
			if err := enc.UpdateRateControlConfig(RateControlConfig{
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
	s.pictureGen++
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
		enc.hme.svt.setHint(s.webRTCPyramidHint(layerSrc, settings.SpatialID))
		enc.hme.seed = s.webRTCMESeed(settings.SpatialID)
		runs := enc.hme.runs
		tu, key, err := s.encodePictureLayer(enc, layerSrc, settings, unit.Key)
		enc.hme.svt.setHint(svtPyramidHint{})
		enc.hme.seed = nil
		if enc.hme.runs != runs {
			s.layerMEGen[settings.SpatialID] = s.pictureGen
		}
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
	s.pictureGen++
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
	s.pictureGen++
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
	s.pictureGen++
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
		if refSlot >= WebRTCReferenceBuffers || !s.referenceValid[refSlot] {
			return nil, false, ErrInvalidFrame
		}
		tu, err := enc.encodeReferencePFrameWithSequenceMax(layerSrc, s.refPool[s.refSlot[refSlot]], refSlot, settings, maxWidth, maxHeight)
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

// updateReferenceFrame records the picture enc just coded into reference slot
// UpdateBuffer. Only shared-slot coding (spatial SVC, see
// webRTCStreamUsesSharedReferenceSlotCoding) reads slot pixels; it takes the
// reconstruction by ownership into the reference pool (adoptReference), as
// SVT's reference-picture pool does, instead of copying it. In every other
// mode each layer encoder owns its references, so the slot records only that
// it is held and its geometry, and the deferred CDEF is not forced. Entering
// shared-slot mode changes the dependency structure, which invalidates the
// slots, so a pixel-less slot is never read. The monochrome and
// high-bit-depth variants below still copy, and likewise only in shared-slot
// mode.
func (s *WebRTCStream) updateReferenceFrame(settings FrameEncodeSettings, enc *VideoEncoder) error {
	if !settings.UpdateBufferSet {
		return nil
	}
	if settings.UpdateBuffer >= WebRTCReferenceBuffers || enc == nil {
		return ErrInvalidFrame
	}
	slot := settings.UpdateBuffer
	if !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
		// The layer encoder owns this reference; the slot records only that
		// it is held and its geometry.
		s.refSize[slot] = [2]int32{int32(enc.width), int32(enc.height)}
		s.referenceValid[slot] = true
		return nil
	}
	recon, err := webRTCStreamLayerReconstruction(enc)
	if err != nil {
		return err
	}
	if s.referenceValid[slot] {
		s.refPoolRefs[s.refSlot[slot]]--
		s.referenceValid[slot] = false
	}
	idx, ok := s.adoptReference(enc, recon)
	if !ok {
		return ErrInvalidFrame
	}
	s.refSlot[slot] = idx
	s.refPoolRefs[idx]++
	s.refSize[slot] = [2]int32{int32(recon.Width), int32(recon.Height)}
	s.referenceValid[slot] = true
	return nil
}

// webRTCRefPoolSize bounds the shared reference pool: one entry per slot, a
// spare per spatial layer so a free entry of each layer's size exists, and
// one more.
const webRTCRefPoolSize = WebRTCReferenceBuffers + WebRTCMaxSpatialLayers + 1

// adoptReference moves recon, which enc just wrote into one of its own
// reconstruction buffers, into a free pool entry and returns that entry.
// Like SVT's reference-picture pool it hands the picture over by ownership:
// the encoder buffer and the free entry swap headers, so the encoder writes
// its next reconstruction into the entry's old memory. The free entry is
// preferably one of recon's size, so the swapped-in buffer fits the
// encoder's next write without reallocating. An entry no slot references
// is read by nobody, and an encoder buffer is never shared with a slot, so
// the swap is invisible to every later read. A reconstruction outside the
// encoder's own buffers falls back to a copy.
func (s *WebRTCStream) adoptReference(enc *VideoEncoder, recon SourceFrame420) (uint8, bool) {
	free, empty, any := -1, -1, -1
	for i := range s.refPool {
		if s.refPoolRefs[i] != 0 {
			continue
		}
		p := &s.refPool[i]
		if len(p.Y) == len(recon.Y) && len(p.U) == len(recon.U) && len(p.V) == len(recon.V) {
			free = i
			break
		}
		if p.Y == nil && empty < 0 {
			empty = i
		}
		if any < 0 {
			any = i
		}
	}
	if free < 0 {
		free = empty
	}
	if free < 0 {
		free = any
	}
	if free < 0 {
		return 0, false
	}
	if owner := enc.reconstructionOwner(recon); owner != nil {
		*owner, s.refPool[free] = s.refPool[free], *owner
	} else {
		copyFrameInto(&s.refPool[free], recon)
	}
	return uint8(free), true
}

func (s *WebRTCStream) updateMonoReferenceFrame(settings FrameEncodeSettings, enc *MonochromeVideoEncoder) error {
	if !settings.UpdateBufferSet || !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
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
	if !settings.UpdateBufferSet || !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
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
	if !settings.UpdateBufferSet || !webRTCStreamUsesSharedReferenceSlotCoding(s.config) {
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
	if err := enc.finishLazyCDEF(); err != nil {
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

// Spatial-layer sources. Each lower layer's picture is the box-filtered
// (scale_box.go) picture of the layer above it, built once per picture
// (layerGen == pictureGen) top-down: S1 from the input, S0 from S1. The
// cascade makes an exact 1/2 step's luma the upper layer's HME
// half-resolution pyramid plane (webRTCPyramidHint).

// cascadeParentLayer reports the layer the spatialID source is scaled from:
// the next layer up when it is a downscale of the input, else none (-1).
func (s *WebRTCStream) cascadeParentLayer(spatialID uint8, inputW, inputH int) int {
	up := spatialID + 1
	if up >= s.config.SpatialLayerCount || up >= WebRTCMaxSpatialLayers {
		return -1
	}
	r := s.config.SpatialLayers[up].Resolution
	if !r.Valid() || (int(r.Width) == inputW && int(r.Height) == inputH) {
		return -1
	}
	return int(up)
}

func (s *WebRTCStream) sourceForLayer(src SourceFrame420, spatialID uint8, resolution Resolution) (SourceFrame420, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrame420{}, ErrInvalidFrame
	}
	w, h := int(resolution.Width), int(resolution.Height)
	if src.Width == w && src.Height == h {
		return src, nil
	}
	if f := s.scaledFrames[spatialID]; s.layerGen[spatialID] == s.pictureGen && f.Width == w && f.Height == h {
		return f, nil
	}
	parent := src
	if up := s.cascadeParentLayer(spatialID, src.Width, src.Height); up >= 0 {
		p, err := s.sourceForLayer(src, uint8(up), s.config.SpatialLayers[up].Resolution)
		if err != nil {
			return SourceFrame420{}, err
		}
		parent = p
	}
	out, err := scaleSourceFrameColorBox(&s.scaledFrames[spatialID], parent, w, h, parserColorConfig(s.config.ColorConfig), &s.scaleScratch)
	if err == nil {
		s.layerGen[spatialID] = s.pictureGen
	}
	return out, err
}

// webRTCPyramidHint returns the lower layers' luma that equals the HME
// pyramid of the spatialID layer's source src: layer spatialID-1 when it is
// an exact 1/2 box step of src (the half plane), and layer spatialID-2 when
// it is one of that (the sixteenth plane).
func (s *WebRTCStream) webRTCPyramidHint(src SourceFrame420, spatialID uint8) svtPyramidHint {
	var hint svtPyramidHint
	if spatialID == 0 || src.Y == nil || src.Width%2 != 0 || src.Height%2 != 0 {
		return hint
	}
	half := s.scaledFrames[spatialID-1]
	if s.layerGen[spatialID-1] != s.pictureGen || half.Width*2 != src.Width || half.Height*2 != src.Height || half.Y == nil {
		return hint
	}
	hint.parent = &src.Y[0]
	hint.half = half
	if spatialID >= 2 && half.Width%2 == 0 && half.Height%2 == 0 {
		six := s.scaledFrames[spatialID-2]
		if s.layerGen[spatialID-2] == s.pictureGen && six.Width*2 == half.Width && six.Height*2 == half.Height && six.Y != nil {
			hint.six = six
		}
	}
	return hint
}

// webRTCMESeed returns the ME of the layer below spatialID when it ran for
// this picture; the upper layer's HME takes it as its level-0 result when
// the geometry and reference match (hmeState.seedUsable).
func (s *WebRTCStream) webRTCMESeed(spatialID uint8) *hmeState {
	// Simulcast only: measured on L3T3_KEY the seed moved BD-rate by up to
	// +0.75% on some layers, against within 0.16% on S3T3.
	if !s.config.Scalability.IsSimulcast() || spatialID == 0 || spatialID >= WebRTCMaxSpatialLayers || s.layerMEGen[spatialID-1] != s.pictureGen {
		return nil
	}
	lower := s.encoders[spatialID-1]
	if lower == nil {
		return nil
	}
	return &lower.hme
}

func (s *WebRTCStream) sourceForMonoLayer(src SourceFrameMono, spatialID uint8, resolution Resolution) (SourceFrameMono, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrameMono{}, ErrInvalidFrame
	}
	w, h := int(resolution.Width), int(resolution.Height)
	if src.Width == w && src.Height == h {
		return src, nil
	}
	if f := s.scaledMono[spatialID]; s.layerGen[spatialID] == s.pictureGen && f.Width == w && f.Height == h {
		return f, nil
	}
	parent := src
	if up := s.cascadeParentLayer(spatialID, src.Width, src.Height); up >= 0 {
		p, err := s.sourceForMonoLayer(src, uint8(up), s.config.SpatialLayers[up].Resolution)
		if err != nil {
			return SourceFrameMono{}, err
		}
		parent = p
	}
	out, err := scaleSourceFrameMonoBox(&s.scaledMono[spatialID], parent, w, h, &s.scaleScratch)
	if err == nil {
		s.layerGen[spatialID] = s.pictureGen
	}
	return out, err
}

func (s *WebRTCStream) sourceForMono16Layer(src SourceFrameMono16, spatialID uint8, resolution Resolution) (SourceFrameMono16, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrameMono16{}, ErrInvalidFrame
	}
	w, h := int(resolution.Width), int(resolution.Height)
	if src.Width == w && src.Height == h {
		return src, nil
	}
	if f := s.scaledMono16[spatialID]; s.layerGen[spatialID] == s.pictureGen && f.Width == w && f.Height == h {
		return f, nil
	}
	parent := src
	if up := s.cascadeParentLayer(spatialID, src.Width, src.Height); up >= 0 {
		p, err := s.sourceForMono16Layer(src, uint8(up), s.config.SpatialLayers[up].Resolution)
		if err != nil {
			return SourceFrameMono16{}, err
		}
		parent = p
	}
	out, err := scaleSourceFrameMono16Box(&s.scaledMono16[spatialID], parent, w, h, &s.scaleScratch)
	if err == nil {
		s.layerGen[spatialID] = s.pictureGen
	}
	return out, err
}

func (s *WebRTCStream) sourceFor42016Layer(src SourceFrame42016, spatialID uint8, resolution Resolution) (SourceFrame42016, error) {
	return s.sourceForColor16Layer(src, spatialID, resolution)
}

func (s *WebRTCStream) sourceForColor16Layer(src SourceFrame42016, spatialID uint8, resolution Resolution) (SourceFrame42016, error) {
	if spatialID >= WebRTCMaxSpatialLayers || !resolution.Valid() {
		return SourceFrame42016{}, ErrInvalidFrame
	}
	w, h := int(resolution.Width), int(resolution.Height)
	if src.Width == w && src.Height == h {
		return src, nil
	}
	if f := s.scaledFrames16[spatialID]; s.layerGen[spatialID] == s.pictureGen && f.Width == w && f.Height == h {
		return f, nil
	}
	parent := src
	if up := s.cascadeParentLayer(spatialID, src.Width, src.Height); up >= 0 {
		p, err := s.sourceForColor16Layer(src, uint8(up), s.config.SpatialLayers[up].Resolution)
		if err != nil {
			return SourceFrame42016{}, err
		}
		parent = p
	}
	out, err := scaleSourceFrameColor16Box(&s.scaledFrames16[spatialID], parent, w, h, parserColorConfig(s.config.ColorConfig), &s.scaleScratch)
	if err == nil {
		s.layerGen[spatialID] = s.pictureGen
	}
	return out, err
}

func scaleSourceFrameColorBox(dst *SourceFrame420, src SourceFrame420, width, height int, color parser.ColorConfig, scratch *boxScaleScratch) (SourceFrame420, error) {
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
	scalePlaneBox8(dst.Y, width, width, height, src.Y, src.YStride, src.Width, src.Height, scratch)
	scalePlaneBox8(dst.U, cw, cw, ch, src.U, src.ChromaStride, srcCW, srcCH, scratch)
	scalePlaneBox8(dst.V, cw, cw, ch, src.V, src.ChromaStride, srcCW, srcCH, scratch)
	return *dst, nil
}

func scaleSourceFrameMonoBox(dst *SourceFrameMono, src SourceFrameMono, width, height int, scratch *boxScaleScratch) (SourceFrameMono, error) {
	if width <= 0 || height <= 0 {
		return SourceFrameMono{}, ErrInvalidFrame
	}
	if len(dst.Y) != width*height {
		dst.Y = make([]byte, width*height)
	}
	dst.YStride = width
	dst.Width = width
	dst.Height = height
	scalePlaneBox8(dst.Y, width, width, height, src.Y, src.YStride, src.Width, src.Height, scratch)
	return *dst, nil
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

// reconstructionOwner returns the encoder field whose buffer holds recon: one
// of the reference reconstruction ring entries or the keyframe buffer. It
// returns nil when recon lives elsewhere.
func (e *VideoEncoder) reconstructionOwner(recon SourceFrame420) *SourceFrame420 {
	if len(recon.Y) == 0 {
		return nil
	}
	for _, f := range [...]*SourceFrame420{&e.reconBufs[0], &e.reconBufs[1], &e.keyRecon} {
		if len(f.Y) == len(recon.Y) && len(f.U) == len(recon.U) && len(f.V) == len(recon.V) &&
			&f.Y[0] == &recon.Y[0] && len(f.U) > 0 && &f.U[0] == &recon.U[0] && len(f.V) > 0 && &f.V[0] == &recon.V[0] {
			return f
		}
	}
	return nil
}
