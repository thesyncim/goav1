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
	// Keyframe reports whether the frame resets the decode chain.
	Keyframe bool
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

	encoders     [WebRTCMaxSpatialLayers]*VideoEncoder
	scaledFrames [WebRTCMaxSpatialLayers]SourceFrame420
	layerScratch [WebRTCMaxSpatialLayers][]byte
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
	for i := uint8(0); i < stream.config.SpatialLayerCount; i++ {
		enc := stream.encoders[i]
		enc.rcMinQ = rc.MinQIndex
		enc.rcMaxQ = rc.MaxQIndex
		if enc.rcMinQ == 0 {
			enc.rcMinQ = 20
		}
		if enc.rcMaxQ == 0 || enc.rcMaxQ <= enc.rcMinQ {
			enc.rcMaxQ = 200
		}
		enc.qIndex = enc.rcMinQ/2 + enc.rcMaxQ/2
	}
	return stream, nil
}

// NewWebRTCStreamConfig creates a WebRTC stream from the lower-level WebRTC
// encoder config. The pixel encoder currently accepts 8-bit profile-0 I420
// under CBR. Multi-spatial pixel output is supported for modes whose delta
// frames do not require AV1 inter-layer prediction: simulcast and key-only SVC
// modes, including key-shift schedules.
func NewWebRTCStreamConfig(config Config) (*WebRTCStream, error) {
	normalized, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return nil, err
	}
	if !webRTCPixelScalabilitySupported(normalized) {
		return nil, ErrUnsupported
	}
	if normalized.Profile != Profile0 || normalized.BitDepth != 8 || normalized.RateControl != RateControlCBR {
		return nil, ErrUnsupported
	}
	fps := webRTCStreamFramesPerSecond(normalized.MaxFramerate)
	if fps <= 0 {
		return nil, ErrInvalidConfig
	}
	var stream WebRTCStream
	stream.config = normalized
	for i := uint8(0); i < normalized.SpatialLayerCount; i++ {
		layer := normalized.SpatialLayers[i]
		targetKbps := layer.TargetBitrateKbps
		if targetKbps <= 0 {
			targetKbps = normalized.TargetBitrateKbps
		}
		if targetKbps <= 0 {
			return nil, ErrInvalidConfig
		}
		enc, err := NewVideoEncoderCBR(int(layer.Resolution.Width), int(layer.Resolution.Height), RateControlConfig{
			TargetBitsPerSecond: int(targetKbps) * 1000,
			FramesPerSecond:     fps,
			MinQIndex:           20,
			MaxQIndex:           200,
		})
		if err != nil {
			return nil, err
		}
		if err := enc.SetTemporalLayers(int(normalized.TemporalLayerCount)); err != nil {
			return nil, err
		}
		enc.SetSceneCutKeyframes(false)
		stream.encoders[i] = enc
	}
	return &stream, nil
}

func webRTCPixelScalabilitySupported(config Config) bool {
	if config.SpatialLayerCount <= 1 {
		return true
	}
	return config.Scalability.IsSimulcast() || config.Scalability.UsesKeyFrameInterLayerDependency()
}

// SetGoldenInterval forwards the golden-reference refresh policy to every
// underlying spatial encoder. Zero disables golden references.
func (s *WebRTCStream) SetGoldenInterval(n int) {
	for i := uint8(0); s != nil && i < s.config.SpatialLayerCount; i++ {
		s.encoders[i].SetGoldenInterval(n)
	}
}

// SetTileColumns forwards the tile-column override to every underlying spatial
// encoder.
func (s *WebRTCStream) SetTileColumns(cols int) {
	for i := uint8(0); s != nil && i < s.config.SpatialLayerCount; i++ {
		s.encoders[i].SetTileColumns(cols)
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
		tu, key, err := enc.EncodeWithTemporalID(layerSrc, unit.Key, settings.TemporalID)
		if err != nil {
			return WebRTCEncodedPicture{}, err
		}
		if key != unit.Key {
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
			Keyframe:                  key,
			Info:                      control.GenericFrameInfo,
			Descriptor:                descriptor,
			Structure:                 structure,
			AttachDependencyStructure: control.AttachDependencyStructure,
		}
	}
	s.state = next
	return picture, nil
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
