package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/obu"
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
	goldenInterval    int
	goldenIntervalSet bool

	encoders     [WebRTCMaxSpatialLayers]*VideoEncoder
	scaledFrames [WebRTCMaxSpatialLayers]SourceFrame420
	layerScratch [WebRTCMaxSpatialLayers][]byte

	referenceFrames [WebRTCReferenceBuffers]SourceFrame420
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
// encoder config. The pixel encoder currently accepts 8-bit profile-0 I420
// under CBR. Multi-spatial pixel output is supported for WebRTC SVC and
// simulcast modes, including key and key-shift schedules.
func NewWebRTCStreamConfig(config Config) (*WebRTCStream, error) {
	normalized, fps, err := normalizeWebRTCStreamConfig(config)
	if err != nil {
		return nil, err
	}
	var stream WebRTCStream
	stream.config = normalized
	stream.rcMinQ, stream.rcMaxQ = 20, 200
	for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
		enc, err := newWebRTCStreamLayerEncoder(normalized, i, fps, stream.rcMinQ, stream.rcMaxQ)
		if err != nil {
			return nil, err
		}
		stream.encoders[i] = enc
	}
	return &stream, nil
}

func normalizeWebRTCStreamConfig(config Config) (Config, int, error) {
	normalized, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return Config{}, 0, err
	}
	if !webRTCPixelScalabilitySupported(normalized) {
		return Config{}, 0, ErrUnsupported
	}
	if normalized.Profile != Profile0 || normalized.BitDepth != 8 || normalized.RateControl != RateControlCBR {
		return Config{}, 0, ErrUnsupported
	}
	fps := webRTCStreamFramesPerSecond(normalized.MaxFramerate)
	if fps <= 0 {
		return Config{}, 0, ErrInvalidConfig
	}
	for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
		if webRTCStreamLayerTargetKbps(normalized, i) <= 0 {
			return Config{}, 0, ErrInvalidConfig
		}
	}
	return normalized, fps, nil
}

func webRTCPixelScalabilitySupported(config Config) bool {
	return config.SpatialLayerCount > 0 && config.SpatialLayerCount <= WebRTCMaxSpatialLayers
}

func newWebRTCStreamLayerEncoder(config Config, layerIndex uint8, fps int, minQ uint8, maxQ uint8) (*VideoEncoder, error) {
	layer := config.SpatialLayers[layerIndex]
	targetKbps := webRTCStreamLayerTargetKbps(config, layerIndex)
	enc, err := NewVideoEncoderCBR(int(layer.Resolution.Width), int(layer.Resolution.Height), RateControlConfig{
		TargetBitsPerSecond: int(targetKbps) * 1000,
		FramesPerSecond:     fps,
		MinQIndex:           minQ,
		MaxQIndex:           maxQ,
	})
	if err != nil {
		return nil, err
	}
	if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
		return nil, err
	}
	enc.SetSceneCutKeyframes(false)
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
		if enc == nil {
			continue
		}
		enc.rcMinQ = minQ
		enc.rcMaxQ = maxQ
		enc.qIndex = minQ/2 + maxQ/2
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
		s.encoders[i].SetGoldenInterval(n)
	}
}

// SetTileColumns forwards the tile-column override to every underlying spatial
// encoder.
func (s *WebRTCStream) SetTileColumns(cols int) {
	if s != nil {
		s.tileColumns = cols
	}
	for i := uint8(0); s != nil && i < s.config.SpatialLayerCount; i++ {
		s.encoders[i].SetTileColumns(cols)
	}
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

// SetConfig atomically updates bitrate, framerate, and supported scalability
// settings. Changes that alter layer geometry or dependency structure make the
// next encoded picture a key picture while preserving frame IDs.
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
	if sameGeometry {
		if err := s.updateLayerControls(normalized, fps); err != nil {
			return err
		}
		s.config = normalized
		if !sameStructure {
			s.state = webRTCEncoderStateForNextKey(s.state)
			s.referenceFrames = [WebRTCReferenceBuffers]SourceFrame420{}
		}
		return nil
	}

	nextEncoders, err := s.buildReplacementLayerEncoders(normalized, fps)
	if err != nil {
		return err
	}
	if err := s.closeEncoders(); err != nil {
		closeWebRTCStreamEncoders(nextEncoders)
		return err
	}
	s.encoders = nextEncoders
	s.config = normalized
	s.scaledFrames = [WebRTCMaxSpatialLayers]SourceFrame420{}
	s.layerScratch = [WebRTCMaxSpatialLayers][]byte{}
	s.referenceFrames = [WebRTCReferenceBuffers]SourceFrame420{}
	s.state = webRTCEncoderStateForNextKey(s.state)
	return nil
}

func (s *WebRTCStream) updateLayerControls(config Config, fps int) error {
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		enc := s.encoders[i]
		if enc == nil {
			return ErrInvalidConfig
		}
		if err := enc.SetTemporalLayers(int(config.TemporalLayerCount)); err != nil {
			return err
		}
		targetKbps := webRTCStreamLayerTargetKbps(config, i)
		if err := enc.SetRateControlConfig(RateControlConfig{
			TargetBitsPerSecond: int(targetKbps) * 1000,
			FramesPerSecond:     fps,
			MinQIndex:           s.rcMinQ,
			MaxQIndex:           s.rcMaxQ,
		}); err != nil {
			return err
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

func (s *WebRTCStream) closeEncoders() error {
	var firstErr error
	for i := uint8(0); i < WebRTCMaxSpatialLayers; i++ {
		if s.encoders[i] == nil {
			continue
		}
		if err := s.encoders[i].Close(); err != nil && firstErr == nil {
			firstErr = err
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

func webRTCEncoderStateForNextKey(state WebRTCEncoderState) WebRTCEncoderState {
	return WebRTCEncoderState{
		NextOrderHint: state.NextOrderHint,
		NextFrameID:   state.NextFrameID,
	}
}

// Prewarm sizes the underlying encoders' reusable buffers without advancing
// the WebRTC frame metadata.
func (s *WebRTCStream) Prewarm() error {
	for i := uint8(0); s != nil && i < s.config.SpatialLayerCount; i++ {
		if err := s.encoders[i].Prewarm(); err != nil {
			return err
		}
	}
	return nil
}

// Encode encodes one single-spatial frame and returns it with WebRTC packaging
// metadata.
func (s *WebRTCStream) Encode(src SourceFrame420, forceKey bool) (WebRTCEncodedFrame, error) {
	picture, err := s.EncodePicture(src, forceKey)
	if err != nil {
		return WebRTCEncodedFrame{}, err
	}
	if picture.FrameNum != 1 {
		return WebRTCEncodedFrame{}, ErrUnsupported
	}
	return picture.Frames[0], nil
}

// EncodePicture encodes one picture and returns one frame per active spatial
// layer with dependency descriptors matching the configured scalability mode.
func (s *WebRTCStream) EncodePicture(src SourceFrame420, forceKey bool) (WebRTCEncodedPicture, error) {
	if s == nil || s.config.SpatialLayerCount == 0 {
		return WebRTCEncodedPicture{}, ErrInvalidConfig
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
		layerSize, err := webRTCLayerTemporalUnitSize(tu, settings.TemporalID, settings.SpatialID)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if cap(s.layerScratch[i]) < layerSize {
			s.layerScratch[i] = make([]byte, 0, layerSize)
		}
		layerTU, err := appendWebRTCLayerTemporalUnit(s.layerScratch[i][:0], tu, settings.TemporalID, settings.SpatialID)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		s.layerScratch[i] = layerTU
		control, structure, err := webRTCPictureTemporalUnitFrameControl(unit, s.state, i)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		descSize, err := WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, control.AttachDependencyStructure)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		descriptor, err := AppendWebRTCDependencyDescriptor(make([]byte, 0, descSize), structure, control.GenericFrameInfo, true, true, control.AttachDependencyStructure)
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
	return scaleSourceFrame420Nearest(&s.scaledFrames[spatialID], src, int(resolution.Width), int(resolution.Height))
}

func scaleSourceFrame420Nearest(dst *SourceFrame420, src SourceFrame420, width, height int) (SourceFrame420, error) {
	if width <= 0 || height <= 0 {
		return SourceFrame420{}, ErrInvalidFrame
	}
	cw, ch := (width+1)/2, (height+1)/2
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
	scalePlaneNearest(dst.U, cw, cw, ch, src.U, src.ChromaStride, (src.Width+1)/2, (src.Height+1)/2)
	scalePlaneNearest(dst.V, cw, cw, ch, src.V, src.ChromaStride, (src.Width+1)/2, (src.Height+1)/2)
	return *dst, nil
}

func scalePlaneNearest(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	for y := 0; y < dstHeight; y++ {
		sy := y * srcHeight / dstHeight
		drow := dst[y*dstStride : y*dstStride+dstWidth]
		srow := src[sy*srcStride:]
		for x := 0; x < dstWidth; x++ {
			drow[x] = srow[x*srcWidth/dstWidth]
		}
	}
}

func appendWebRTCLayerTemporalUnit(dst []byte, src []byte, temporalID uint8, spatialID uint8) ([]byte, error) {
	it := obu.NewLowOverheadIterator(src)
	out := dst
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
	}
}

func webRTCLayerTemporalUnitSize(src []byte, temporalID uint8, spatialID uint8) (int, error) {
	it := obu.NewLowOverheadIterator(src)
	total := 0
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
