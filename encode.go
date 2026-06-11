package goav1

// encode.go is the public realtime encoding surface. An Encoder turns 4:2:0
// frames into AV1 low-overhead temporal units (a keyframe first, then
// motion-compensated inter frames) under fixed quality or CBR rate control,
// with optional temporal layering and WebRTC dependency-descriptor packaging
// through RTCEncoder. Every emitted stream decodes bit-exactly to the
// encoder's own reconstruction in this package's Decoder and in the reference
// decoders.

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// I420Frame is one 8-bit 4:2:0 picture. Y holds Width x Height luma samples
// at YStride; U and V hold the half-resolution chroma planes at ChromaStride.
type I420Frame = encoder.SourceFrame420

// VideoEncoderConfig configures a realtime encoder.
type VideoEncoderConfig struct {
	// Width and Height are the frame dimensions in pixels; both must be
	// even and at least 16. Dimensions that are not multiples of 8 encode
	// at the next padded coded size with render_size carrying the true
	// dimensions; decoded surfaces are the coded size and display crops to
	// the render size.
	Width, Height int

	// QIndex selects fixed-quality encoding (1..255) when TargetBitrate is
	// zero.
	QIndex uint8

	// TargetBitrate, when positive, enables CBR rate control at this many
	// bits per second; Framerate must then also be positive. MinQIndex and
	// MaxQIndex bound the controller (defaults 20 and 200).
	TargetBitrate int
	Framerate     int
	MinQIndex     uint8
	MaxQIndex     uint8

	// TemporalLayers selects the layering mode: 0 or 1 for a flat stream,
	// 2 for L1T2 with droppable odd frames, 3 for L1T3 (T0/T2/T1/T2 groups
	// with a droppable top layer and a middle layer the trailing T2
	// references).
	TemporalLayers int

	// TileColumns overrides the tile-column count used for parallel inter
	// encoding (rounded down to a power of two, clamped to the legal range);
	// zero selects automatically from the frame width.
	TileColumns int

	// GoldenInterval is the number of base-layer frames between golden
	// anchor refreshes; zero keeps the default (32) and a negative value
	// disables golden references.
	GoldenInterval int
}

// EncodedFrame is one encoded picture as a low-overhead temporal unit.
type EncodedFrame struct {
	// Data is the temporal unit. It is freshly allocated per frame and owned
	// by the caller.
	Data []byte
	// Keyframe reports whether this frame restarts the decode chain.
	Keyframe bool
	// TemporalID is the frame's temporal layer (always 0 without layering).
	TemporalID uint8
}

// VideoEncoder encodes a stream of same-sized 4:2:0 frames.
type VideoEncoder struct {
	enc *encoder.VideoEncoder
}

// NewVideoEncoder creates an encoder from cfg.
func NewVideoEncoder(cfg VideoEncoderConfig) (*VideoEncoder, error) {
	enc, err := newVideoEncoder(cfg)
	if err != nil {
		return nil, err
	}
	return &VideoEncoder{enc: enc}, nil
}

func newVideoEncoder(cfg VideoEncoderConfig) (*encoder.VideoEncoder, error) {
	var enc *encoder.VideoEncoder
	var err error
	if cfg.TargetBitrate > 0 {
		rc := encoder.RateControlConfig{
			TargetBitsPerSecond: cfg.TargetBitrate,
			FramesPerSecond:     cfg.Framerate,
			MinQIndex:           cfg.MinQIndex,
			MaxQIndex:           cfg.MaxQIndex,
		}
		if rc.MinQIndex == 0 {
			rc.MinQIndex = 20
		}
		if rc.MaxQIndex == 0 {
			rc.MaxQIndex = 200
		}
		enc, err = encoder.NewVideoEncoderCBR(cfg.Width, cfg.Height, rc)
	} else {
		if cfg.QIndex == 0 {
			return nil, fmt.Errorf("goav1: VideoEncoderConfig needs QIndex or TargetBitrate")
		}
		enc, err = encoder.NewVideoEncoder(cfg.Width, cfg.Height, cfg.QIndex)
	}
	if err != nil {
		return nil, err
	}
	switch cfg.TemporalLayers {
	case 0, 1:
	default:
		if err := enc.SetTemporalLayers(cfg.TemporalLayers); err != nil {
			return nil, err
		}
	}
	if cfg.TileColumns > 0 {
		enc.SetTileColumns(cfg.TileColumns)
	}
	if cfg.GoldenInterval < 0 {
		enc.SetGoldenInterval(0)
	} else if cfg.GoldenInterval > 0 {
		enc.SetGoldenInterval(cfg.GoldenInterval)
	}
	return enc, nil
}

// Encode encodes one frame. forceKey restarts the stream with a keyframe.
// The returned Data aliases an encoder-owned buffer that is reused by the
// next Encode call - send or copy it before encoding the next frame, the
// same lifetime the Reconstruction planes have.
func (e *VideoEncoder) Encode(frame I420Frame, forceKey bool) (EncodedFrame, error) {
	tid := e.enc.TemporalID()
	tu, key, err := e.enc.Encode(frame, forceKey)
	if err != nil {
		return EncodedFrame{}, err
	}
	if key {
		tid = 0
	}
	return EncodedFrame{Data: tu, Keyframe: key, TemporalID: tid}, nil
}

// Reconstruction returns the most recent frame's reconstruction — exactly
// what a conformant decoder outputs for it. The planes alias encoder-owned
// buffers that are recycled two frames later; copy for longer-lived use.
func (e *VideoEncoder) Reconstruction() I420Frame {
	return e.enc.Recon()
}

// QIndex reports the current working quantizer index (the CBR controller
// moves it between frames).
func (e *VideoEncoder) QIndex() uint8 {
	return e.enc.QIndex()
}

// RTCFrame is one encoded frame with WebRTC packaging metadata.
type RTCFrame struct {
	// Data is the temporal unit (the RTP payload content).
	Data []byte
	// Keyframe reports whether this frame restarts the decode chain.
	Keyframe bool
	// TemporalID is the frame's temporal layer.
	TemporalID uint8
	// DependencyDescriptor is the serialized RTP dependency descriptor for a
	// single-packet frame; keyframes attach the dependency structure.
	DependencyDescriptor []byte
}

// RTCEncoder encodes an L1T1 or L1T2 WebRTC stream with per-frame dependency
// descriptors. It requires CBR configuration (TargetBitrate and Framerate).
type RTCEncoder struct {
	stream *encoder.WebRTCStream
}

// NewRTCEncoder creates a WebRTC encoder from cfg.
func NewRTCEncoder(cfg VideoEncoderConfig) (*RTCEncoder, error) {
	if cfg.TargetBitrate <= 0 || cfg.Framerate <= 0 {
		return nil, fmt.Errorf("goav1: RTCEncoder requires TargetBitrate and Framerate")
	}
	rc := encoder.RateControlConfig{
		TargetBitsPerSecond: cfg.TargetBitrate,
		FramesPerSecond:     cfg.Framerate,
		MinQIndex:           cfg.MinQIndex,
		MaxQIndex:           cfg.MaxQIndex,
	}
	if rc.MinQIndex == 0 {
		rc.MinQIndex = 20
	}
	if rc.MaxQIndex == 0 {
		rc.MaxQIndex = 200
	}
	layers := cfg.TemporalLayers
	if layers == 0 {
		layers = 1
	}
	stream, err := encoder.NewWebRTCStreamLayers(cfg.Width, cfg.Height, rc, layers)
	if err != nil {
		return nil, err
	}
	return &RTCEncoder{stream: stream}, nil
}

// Encode encodes one frame with its dependency descriptor.
func (e *RTCEncoder) Encode(frame I420Frame, forceKey bool) (RTCFrame, error) {
	out, err := e.stream.Encode(frame, forceKey)
	if err != nil {
		return RTCFrame{}, err
	}
	return RTCFrame{
		Data:                 out.TU,
		Keyframe:             out.Keyframe,
		TemporalID:           out.Info.TemporalID,
		DependencyDescriptor: out.Descriptor,
	}, nil
}
