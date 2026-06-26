package goav1

// encode.go is the public realtime encoding surface. An Encoder turns
// caller-owned pictures into AV1 low-overhead temporal units (a keyframe first,
// then motion-compensated inter frames) under fixed quality or CBR rate
// control, with optional temporal layering and WebRTC dependency-descriptor
// packaging through RTCEncoder. I420/NV12/NV21 preserve 4:2:0 chroma samples;
// I422 inputs are resampled; I444 inputs are preserved when an RTCEncoder is
// explicitly configured for profile-1 native 4:4:4 and otherwise resampled.
// I400 and monochrome Frame inputs fill neutral chroma unless an RTCEncoder is
// explicitly configured for native monochrome across WebRTC L*/S* modes;
// explicit high-bit-depth monochrome RTC configs preserve 10/12-bit I400 or
// monochrome Frame input for single-spatial and simulcast streams, while other
// 10/12-bit Frame inputs are downshifted before entering the current 8-bit
// realtime path. Native 10/12-bit monochrome keyframes and P-frames are also
// available through
// EncodeI400HighBitDepthKeyframe and EncodeI400HighBitDepthPFrame. Every
// emitted stream decodes bit-exactly to the encoder's own reconstruction in
// this package's Decoder and in the reference decoders.

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// I420Frame is one 8-bit 4:2:0 picture. Y holds Width x Height luma samples
// at YStride; U and V hold the half-resolution chroma planes at ChromaStride.
type I420Frame = encoder.SourceFrame420
type EncoderDecisionStats = encoder.EncoderDecisionStats

// I422Frame is one 8-bit 4:2:2 picture. Y holds Width x Height luma samples
// at YStride; U and V hold half-width, full-height chroma planes at
// ChromaStride. The friendly realtime encoder resamples this input to its
// current 4:2:0 profile-0 encode path.
type I422Frame struct {
	Y            []byte
	U            []byte
	V            []byte
	YStride      int
	ChromaStride int
	Width        int
	Height       int
}

// I444Frame is one 8-bit 4:4:4 picture. Y, U, and V all hold Width x Height
// samples at their respective strides. The friendly realtime encoder resamples
// this input to 4:2:0 unless RTCEncoder is configured for native profile-1
// 4:4:4.
type I444Frame struct {
	Y       []byte
	U       []byte
	V       []byte
	YStride int
	UStride int
	VStride int
	Width   int
	Height  int
}

// I400Frame is one 8-bit monochrome picture. Y holds Width x Height luma
// samples at YStride. The friendly realtime stream encoder fills neutral chroma
// for I400 inputs unless an RTCEncoder is explicitly configured for native
// monochrome. EncodeI400Keyframe, EncodeI400PFrame,
// EncodeI400HighBitDepthKeyframe, and EncodeI400HighBitDepthPFrame always emit
// native AV1 monochrome pictures.
type I400Frame struct {
	Y       []byte
	YStride int
	Width   int
	Height  int
}

// I400HighBitDepthFrame is one 10/12-bit monochrome picture. Y holds
// Width x Height samples at YStride, in uint16 values whose high bits above
// BitDepth must be zero.
type I400HighBitDepthFrame struct {
	Y        []uint16
	YStride  int
	Width    int
	Height   int
	BitDepth uint8
}

// EncodeI400Keyframe encodes one native AV1 monochrome keyframe.
//
// qIndex 0 emits a lossless keyframe. qIndex 1..255 emits a non-lossless
// keyframe and returns the encoder-side reconstruction a conformant decoder
// must reproduce exactly. Frame dimensions must be positive multiples of 8.
func EncodeI400Keyframe(frame I400Frame, qIndex uint8) ([]byte, I400Frame, error) {
	src := encoder.SourceFrameMono(frame)
	if qIndex == 0 {
		tu, err := encoder.EncodeLosslessMonochromeKeyframe(src)
		if err != nil {
			return nil, I400Frame{}, err
		}
		recon := I400Frame{
			Y:       make([]byte, len(frame.Y)),
			YStride: frame.YStride,
			Width:   frame.Width,
			Height:  frame.Height,
		}
		copy(recon.Y, frame.Y)
		return tu, recon, nil
	}
	tu, recon, err := encoder.EncodeMonochromeKeyframe(src, qIndex)
	if err != nil {
		return nil, I400Frame{}, err
	}
	return tu, I400Frame(recon), nil
}

// EncodeI400HighBitDepthKeyframe encodes one native 10/12-bit AV1 monochrome
// keyframe.
//
// qIndex 0 emits a lossless keyframe. qIndex 1..255 emits a non-lossless
// keyframe and returns the encoder-side reconstruction a conformant decoder
// must reproduce exactly. Frame dimensions must be positive multiples of 8.
func EncodeI400HighBitDepthKeyframe(frame I400HighBitDepthFrame, qIndex uint8) ([]byte, I400HighBitDepthFrame, error) {
	if qIndex == 0 {
		return EncodeI400HighBitDepthLosslessKeyframe(frame)
	}
	src := encoder.SourceFrameMono16(frame)
	tu, recon, err := encoder.EncodeHighBitDepthMonochromeKeyframe(src, qIndex)
	if err != nil {
		return nil, I400HighBitDepthFrame{}, err
	}
	return tu, I400HighBitDepthFrame(recon), nil
}

// EncodeI400HighBitDepthLosslessKeyframe encodes one native 10/12-bit AV1
// monochrome lossless keyframe. The returned reconstruction aliases a fresh
// copy of frame.Y because lossless output must reproduce the source exactly.
func EncodeI400HighBitDepthLosslessKeyframe(frame I400HighBitDepthFrame) ([]byte, I400HighBitDepthFrame, error) {
	src := encoder.SourceFrameMono16(frame)
	tu, err := encoder.EncodeLosslessHighBitDepthMonochromeKeyframe(src)
	if err != nil {
		return nil, I400HighBitDepthFrame{}, err
	}
	recon := I400HighBitDepthFrame{
		Y:        make([]uint16, len(frame.Y)),
		YStride:  frame.YStride,
		Width:    frame.Width,
		Height:   frame.Height,
		BitDepth: frame.BitDepth,
	}
	copy(recon.Y, frame.Y)
	return tu, recon, nil
}

// EncodeI400HighBitDepthPFrame encodes one native 10/12-bit AV1 monochrome
// inter frame.
//
// ref must be the previous native high-bit-depth monochrome reconstruction for
// the same coded geometry and bit depth, usually the reconstruction returned by
// EncodeI400HighBitDepthKeyframe or a prior EncodeI400HighBitDepthPFrame call.
// qIndex must be 1..255.
func EncodeI400HighBitDepthPFrame(frame I400HighBitDepthFrame, ref I400HighBitDepthFrame, qIndex uint8) ([]byte, I400HighBitDepthFrame, error) {
	tu, recon, err := encoder.EncodeHighBitDepthMonochromePFrame(encoder.SourceFrameMono16(frame), encoder.SourceFrameMono16(ref), qIndex)
	if err != nil {
		return nil, I400HighBitDepthFrame{}, err
	}
	return tu, I400HighBitDepthFrame(recon), nil
}

// EncodeI400PFrame encodes one native AV1 monochrome inter frame.
//
// ref must be the previous native monochrome reconstruction for the same coded
// geometry, usually the reconstruction returned by EncodeI400Keyframe or a prior
// EncodeI400PFrame call. qIndex must be 1..255.
func EncodeI400PFrame(frame I400Frame, ref I400Frame, qIndex uint8) ([]byte, I400Frame, error) {
	tu, recon, err := encoder.EncodeMonochromePFrame(encoder.SourceFrameMono(frame), encoder.SourceFrameMono(ref), qIndex)
	if err != nil {
		return nil, I400Frame{}, err
	}
	return tu, I400Frame(recon), nil
}

// NV12Frame is one 8-bit 4:2:0 picture in semi-planar NV12 layout. Y holds
// Width x Height luma samples at YStride; UV holds interleaved U,V pairs for
// the half-resolution chroma plane at UVStride bytes per chroma row.
type NV12Frame struct {
	Y        []byte
	UV       []byte
	YStride  int
	UVStride int
	Width    int
	Height   int
}

// NV21Frame is one 8-bit 4:2:0 picture in semi-planar NV21 layout. Y holds
// Width x Height luma samples at YStride; VU holds interleaved V,U pairs for
// the half-resolution chroma plane at VUStride bytes per chroma row.
type NV21Frame struct {
	Y        []byte
	VU       []byte
	YStride  int
	VUStride int
	Width    int
	Height   int
}

const (
	EncoderDecisionBlockLevelCount        = encoder.EncoderDecisionBlockLevelCount
	EncoderDecisionPartitionCount         = encoder.EncoderDecisionPartitionCount
	EncoderDecisionBlockSizeCount         = encoder.EncoderDecisionBlockSizeCount
	EncoderDecisionInterModeCount         = encoder.EncoderDecisionInterModeCount
	EncoderDecisionCompoundInterModeCount = encoder.EncoderDecisionCompoundInterModeCount
	EncoderDecisionReferenceFrameCount    = encoder.EncoderDecisionReferenceFrameCount
	EncoderDecisionTransformTypeCount     = encoder.EncoderDecisionTransformTypeCount

	EncoderDecisionBlockSize8x8   = encoder.EncoderDecisionBlockSize8x8
	EncoderDecisionBlockSize16x16 = encoder.EncoderDecisionBlockSize16x16
	EncoderDecisionBlockSize32x32 = encoder.EncoderDecisionBlockSize32x32
	EncoderDecisionBlockSize64x64 = encoder.EncoderDecisionBlockSize64x64
	EncoderDecisionBlockSize16x8  = encoder.EncoderDecisionBlockSize16x8
	EncoderDecisionBlockSize8x16  = encoder.EncoderDecisionBlockSize8x16
	EncoderDecisionBlockSize32x16 = encoder.EncoderDecisionBlockSize32x16
	EncoderDecisionBlockSize16x32 = encoder.EncoderDecisionBlockSize16x32

	EncoderDecisionReferenceLast   = encoder.EncoderDecisionReferenceLast
	EncoderDecisionReferenceGolden = encoder.EncoderDecisionReferenceGolden

	EncoderDecisionTransformDCTDCT   = encoder.EncoderDecisionTransformDCTDCT
	EncoderDecisionTransformADSTDCT  = encoder.EncoderDecisionTransformADSTDCT
	EncoderDecisionTransformDCTADST  = encoder.EncoderDecisionTransformDCTADST
	EncoderDecisionTransformADSTADST = encoder.EncoderDecisionTransformADSTADST
	EncoderDecisionTransformIDTX     = encoder.EncoderDecisionTransformIDTX
)

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
	// anchor refreshes; zero keeps the default (16) and a negative value
	// disables golden references.
	GoldenInterval int
}

// EncodedFrame is one encoded picture as a low-overhead temporal unit.
type EncodedFrame struct {
	// Data is the temporal unit. It aliases an encoder-owned buffer reused by
	// the next Encode call; copy it before retaining or sending asynchronously.
	Data []byte
	// Keyframe reports whether this frame restarts the decode chain.
	Keyframe bool
	// TemporalID is the frame's temporal layer (always 0 without layering).
	TemporalID uint8
}

// VideoEncoder encodes a stream of same-sized 4:2:0 frames.
type VideoEncoder struct {
	enc           *encoder.VideoEncoder
	yuv420Scratch I420Frame
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
			_ = enc.Close()
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
	// Every buffer, pool and per-coder scratch is sized now, so the first
	// real frame pays no initialization latency and steady-state encoding
	// allocates nothing.
	if err := enc.Prewarm(); err != nil {
		_ = enc.Close()
		return nil, err
	}
	return enc, nil
}

// SetGoldenInterval sets how many base-layer inter frames pass between golden
// reference refreshes. Zero disables golden references.
func (e *VideoEncoder) SetGoldenInterval(n int) {
	if e != nil && e.enc != nil {
		e.enc.SetGoldenInterval(n)
	}
}

// SetTileColumns sets the desired tile-column count for subsequent encoded
// frames. The encoder rounds down to a legal power-of-two tile layout.
func (e *VideoEncoder) SetTileColumns(cols int) {
	if e != nil && e.enc != nil {
		e.enc.SetTileColumns(cols)
	}
}

// Encode encodes one frame. forceKey restarts the stream with a keyframe.
// The returned Data aliases an encoder-owned buffer that is reused by the
// next Encode call - send or copy it before encoding the next frame, the
// same lifetime the Reconstruction planes have.
func (e *VideoEncoder) Encode(frame I420Frame, forceKey bool) (EncodedFrame, error) {
	if e == nil || e.enc == nil {
		return EncodedFrame{}, fmt.Errorf("goav1: VideoEncoder is not initialized")
	}
	if err := validateI420Frame(frame); err != nil {
		return EncodedFrame{}, err
	}
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

// EncodeI422 encodes one I422 frame after resampling chroma into the encoder's
// reusable I420 scratch.
func (e *VideoEncoder) EncodeI422(frame I422Frame, forceKey bool) (EncodedFrame, error) {
	if e == nil || e.enc == nil {
		return EncodedFrame{}, fmt.Errorf("goav1: VideoEncoder is not initialized")
	}
	i420, err := i422ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return EncodedFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeI444 encodes one I444 frame after resampling chroma into the encoder's
// reusable I420 scratch.
func (e *VideoEncoder) EncodeI444(frame I444Frame, forceKey bool) (EncodedFrame, error) {
	if e == nil || e.enc == nil {
		return EncodedFrame{}, fmt.Errorf("goav1: VideoEncoder is not initialized")
	}
	i420, err := i444ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return EncodedFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeI400 encodes one monochrome frame after filling neutral chroma into
// the encoder's reusable I420 scratch.
func (e *VideoEncoder) EncodeI400(frame I400Frame, forceKey bool) (EncodedFrame, error) {
	if e == nil || e.enc == nil {
		return EncodedFrame{}, fmt.Errorf("goav1: VideoEncoder is not initialized")
	}
	i420, err := i400ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return EncodedFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeFrame encodes one generic Frame after adapting 8/10/12-bit 4:2:0,
// 4:2:2, 4:4:4, or monochrome samples into the current 8-bit 4:2:0 encode
// path. 10/12-bit input is downshifted to the most significant 8 bits; the
// emitted bitstream is still an 8-bit profile-0 WebRTC stream.
func (e *VideoEncoder) EncodeFrame(frame Frame, forceKey bool) (EncodedFrame, error) {
	if e == nil || e.enc == nil {
		return EncodedFrame{}, fmt.Errorf("goav1: VideoEncoder is not initialized")
	}
	i420, err := frameToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return EncodedFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeNV12 encodes one NV12 frame. The input is converted into the
// encoder's reusable I420 scratch before entering the same encode path as
// Encode.
func (e *VideoEncoder) EncodeNV12(frame NV12Frame, forceKey bool) (EncodedFrame, error) {
	if e == nil || e.enc == nil {
		return EncodedFrame{}, fmt.Errorf("goav1: VideoEncoder is not initialized")
	}
	i420, err := nv12ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return EncodedFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeNV21 encodes one NV21 frame. The input is converted into the
// encoder's reusable I420 scratch before entering the same encode path as
// Encode.
func (e *VideoEncoder) EncodeNV21(frame NV21Frame, forceKey bool) (EncodedFrame, error) {
	if e == nil || e.enc == nil {
		return EncodedFrame{}, fmt.Errorf("goav1: VideoEncoder is not initialized")
	}
	i420, err := nv21ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return EncodedFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// Close waits for any background encoder work to finish and releases
// persistent workers. It is safe to call more than once.
func (e *VideoEncoder) Close() error {
	if e == nil || e.enc == nil {
		return nil
	}
	err := e.enc.Close()
	e.enc = nil
	return err
}

// Reconstruction returns the most recent frame's reconstruction — exactly
// what a conformant decoder outputs for it. The planes alias encoder-owned
// buffers that are recycled two frames later; copy for longer-lived use.
func (e *VideoEncoder) Reconstruction() I420Frame {
	if e == nil || e.enc == nil {
		return I420Frame{}
	}
	return e.enc.Recon()
}

// SetDecisionStatsEnabled toggles encoder-decision diagnostics. It is disabled
// by default; enable it only around measurement runs.
func (e *VideoEncoder) SetDecisionStatsEnabled(enabled bool) {
	if e == nil || e.enc == nil {
		return
	}
	e.enc.SetDecisionStatsEnabled(enabled)
}

// ResetDecisionStats clears the accumulated encoder-decision diagnostics.
func (e *VideoEncoder) ResetDecisionStats() {
	if e == nil || e.enc == nil {
		return
	}
	e.enc.ResetDecisionStats()
}

// DecisionStats returns a copy of the accumulated encoder-decision diagnostics.
func (e *VideoEncoder) DecisionStats() EncoderDecisionStats {
	if e == nil || e.enc == nil {
		return EncoderDecisionStats{}
	}
	return e.enc.DecisionStats()
}

// QIndex reports the current working quantizer index (the CBR controller
// moves it between frames).
func (e *VideoEncoder) QIndex() uint8 {
	if e == nil || e.enc == nil {
		return 0
	}
	return e.enc.QIndex()
}

// RTCFrame is one encoded frame with WebRTC packaging metadata.
type RTCFrame struct {
	// Data is the temporal unit (the RTP payload content). It aliases an
	// encoder-owned buffer reused by the next Encode call.
	Data []byte
	// Keyframe reports whether this frame belongs to a key picture.
	Keyframe bool
	// CodedKeyframe reports whether this frame is coded as an AV1 keyframe.
	// For multi-spatial SVC key pictures, enhancement layers can belong to the
	// key picture while still being coded as inter frames.
	CodedKeyframe bool
	// LastFrameInPicture reports whether this frame is the last frame in the
	// WebRTC picture. AppendRTPPackets uses it to set the RTP marker bit.
	LastFrameInPicture bool
	// TemporalID is the frame's temporal layer.
	TemporalID uint8
	// SpatialID is the frame's spatial layer.
	SpatialID uint8
	// FrameID is the dependency-descriptor frame number.
	FrameID uint64
	// DependencyDescriptor is the serialized RTP dependency descriptor for a
	// single-packet frame; keyframes attach the dependency structure. It is
	// freshly allocated and owned by the caller. Use AppendRTPPackets when the
	// frame is fragmented across multiple RTP payloads.
	DependencyDescriptor []byte

	frameInfo                 encoder.WebRTCGenericFrameInfo
	dependencyStructure       encoder.WebRTCFrameDependencyStructure
	attachDependencyStructure bool
}

// RTCPicture is one encoded WebRTC picture. Single-spatial streams have one
// frame; supported SVC and simulcast streams have one frame per active spatial
// layer.
type RTCPicture struct {
	Frames   [EncoderWebRTCMaxSpatialLayers]RTCFrame
	FrameNum int
	Keyframe bool
}

// AllDecodeTargetsMask returns a dependency-descriptor active decode target
// mask with every target in p enabled.
func (p RTCPicture) AllDecodeTargetsMask() (uint32, error) {
	structure, err := p.dependencyStructure()
	if err != nil {
		return 0, err
	}
	return encoder.WebRTCAllDecodeTargetsMask(structure)
}

// ActiveDecodeTargetsMask returns a dependency-descriptor active decode target
// mask that enables every target at or below the supplied spatial and temporal
// layer IDs.
func (p RTCPicture) ActiveDecodeTargetsMask(maxSpatialID uint8, maxTemporalID uint8) (uint32, error) {
	structure, err := p.dependencyStructure()
	if err != nil {
		return 0, err
	}
	return encoder.WebRTCActiveDecodeTargetsMask(structure, maxSpatialID, maxTemporalID)
}

// SpatialDecodeTargetsMask returns a dependency-descriptor active decode target
// mask that enables decode targets for one spatial layer up to maxTemporalID.
// This is useful when forwarding a browser-compatible simulcast stream or base
// SVC stream from a multi-spatial encoded picture.
func (p RTCPicture) SpatialDecodeTargetsMask(spatialID uint8, maxTemporalID uint8) (uint32, error) {
	structure, err := p.dependencyStructure()
	if err != nil {
		return 0, err
	}
	return encoder.WebRTCSpatialDecodeTargetsMask(structure, spatialID, maxTemporalID)
}

// ActiveDecodeTargetsRTPOptions returns packetization options that write the
// active decode-target mask for maxSpatialID/maxTemporalID on the first RTP
// packet of each frame in the picture.
func (p RTCPicture) ActiveDecodeTargetsRTPOptions(maxSpatialID uint8, maxTemporalID uint8) (EncoderWebRTCRTPPacketDependencyDescriptorOptions, error) {
	mask, err := p.ActiveDecodeTargetsMask(maxSpatialID, maxTemporalID)
	if err != nil {
		return EncoderWebRTCRTPPacketDependencyDescriptorOptions{}, err
	}
	return EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		ActiveDecodeTargetsPresentOnFirstPacket: true,
		ActiveDecodeTargetsMask:                 mask,
	}, nil
}

// SpatialDecodeTargetsRTPOptions returns packetization options that write an
// exact spatial-layer active decode-target mask on the first RTP packet of each
// frame in the picture.
func (p RTCPicture) SpatialDecodeTargetsRTPOptions(spatialID uint8, maxTemporalID uint8) (EncoderWebRTCRTPPacketDependencyDescriptorOptions, error) {
	mask, err := p.SpatialDecodeTargetsMask(spatialID, maxTemporalID)
	if err != nil {
		return EncoderWebRTCRTPPacketDependencyDescriptorOptions{}, err
	}
	return EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		ActiveDecodeTargetsPresentOnFirstPacket: true,
		ActiveDecodeTargetsMask:                 mask,
	}, nil
}

func (p RTCPicture) dependencyStructure() (encoder.WebRTCFrameDependencyStructure, error) {
	if p.FrameNum <= 0 || p.FrameNum > EncoderWebRTCMaxSpatialLayers {
		return encoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	structure := p.Frames[0].dependencyStructure
	for i := 1; i < p.FrameNum; i++ {
		if p.Frames[i].dependencyStructure != structure {
			return encoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
		}
	}
	return structure, nil
}

// RTCFrameRTPScratchSize reports caller-owned scratch needed to packetize one
// RTCFrame into AV1 RTP payload bodies and dependency descriptors.
type RTCFrameRTPScratchSize struct {
	Packetizer         RTPPacketizerScratchSize
	MaxPayloadBytes    int
	MaxDescriptorBytes int
}

// AllDecodeTargetsMask returns a dependency-descriptor active decode target
// mask with every target in f enabled.
func (f RTCFrame) AllDecodeTargetsMask() (uint32, error) {
	return encoder.WebRTCAllDecodeTargetsMask(f.dependencyStructure)
}

// ActiveDecodeTargetsMask returns a dependency-descriptor active decode target
// mask that enables every target at or below the supplied spatial and temporal
// layer IDs.
func (f RTCFrame) ActiveDecodeTargetsMask(maxSpatialID uint8, maxTemporalID uint8) (uint32, error) {
	return encoder.WebRTCActiveDecodeTargetsMask(f.dependencyStructure, maxSpatialID, maxTemporalID)
}

// RTPPacketScratchLen reports scratch sizes for AppendRTPPackets. Callers may
// first pass nil or short obuScratch to learn the OBU count, allocate that many
// RTPPacketizerOBU slots, then call again to learn packet/work-plan sizes.
func (f RTCFrame) RTPPacketScratchLen(limits RTPPayloadSizeLimits, obuScratch []RTPPacketizerOBU) (RTCFrameRTPScratchSize, error) {
	return f.RTPPacketScratchLenWithOptions(limits, obuScratch, EncoderWebRTCRTPPacketDependencyDescriptorOptions{})
}

// RTPPacketScratchLenWithOptions reports scratch sizes for AppendRTPPacketsWithOptions.
func (f RTCFrame) RTPPacketScratchLenWithOptions(limits RTPPayloadSizeLimits, obuScratch []RTPPacketizerOBU, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (RTCFrameRTPScratchSize, error) {
	packetizer, err := RTPPacketizerScratchLen(f.Data, limits, obuScratch)
	size := RTCFrameRTPScratchSize{Packetizer: packetizer}
	if err != nil {
		return size, err
	}
	if f.attachDependencyStructure {
		options.AttachStructureOnFirstPacket = true
	}
	firstFlags := RTPPacketDependencyDescriptorFlags{
		FirstPacketInFrame: true,
		LastPacketInFrame:  packetizer.Packets <= 1,
	}
	descriptor, err := encoder.WebRTCDependencyDescriptorSizeWithOptions(f.dependencyStructure, f.frameInfo, encoderWebRTCRTPPacketDependencyDescriptorOptions(firstFlags, options))
	if err != nil {
		return size, err
	}
	size.MaxDescriptorBytes = descriptor
	if packetizer.Packets > 1 {
		nextDescriptor, err := encoder.WebRTCDependencyDescriptorSizeWithOptions(f.dependencyStructure, f.frameInfo, encoderWebRTCRTPPacketDependencyDescriptorOptions(RTPPacketDependencyDescriptorFlags{}, options))
		if err != nil {
			return size, err
		}
		if nextDescriptor > size.MaxDescriptorBytes {
			size.MaxDescriptorBytes = nextDescriptor
		}
	}
	if packetizer.OBUs != 0 {
		size.MaxPayloadBytes = limits.MaxPayloadLen
	}
	return size, nil
}

// AppendRTPPackets packetizes f.Data into AV1 RTP payload bodies and appends the
// corresponding RTP dependency descriptor bytes. Packet and descriptor spans are
// written into spans; the caller owns RTP headers, header-extension IDs, SRTP,
// pacing, retransmission, and network transport.
func (f RTCFrame) AppendRTPPackets(payloadDst []byte, descriptorDst []byte, spans []EncoderWebRTCRTPPacketSpan, limits RTPPayloadSizeLimits, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (rtpPayloads []byte, descriptors []byte, packetCount int, err error) {
	return f.AppendRTPPacketsWithOptions(payloadDst, descriptorDst, spans, limits, obuScratch, packetScratch, workScratch, EncoderWebRTCRTPPacketDependencyDescriptorOptions{})
}

// AppendRTPPacketsWithOptions is AppendRTPPackets with dependency descriptor
// options for WebRTC control-plane events such as active decode target changes.
func (f RTCFrame) AppendRTPPacketsWithOptions(payloadDst []byte, descriptorDst []byte, spans []EncoderWebRTCRTPPacketSpan, limits RTPPayloadSizeLimits, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (rtpPayloads []byte, descriptors []byte, packetCount int, err error) {
	packetizer, err := NewRTPPacketizer(f.Data, limits, f.CodedKeyframe, f.LastFrameInPicture, obuScratch, packetScratch, workScratch)
	if err != nil {
		return payloadDst, descriptorDst, 0, err
	}
	control := EncoderWebRTCFrameControl{
		GenericFrameInfo:          f.frameInfo,
		AttachDependencyStructure: f.attachDependencyStructure,
	}
	if f.attachDependencyStructure {
		options.AttachStructureOnFirstPacket = true
	}
	rtpPayloads = payloadDst
	descriptors = descriptorDst
	for {
		if packetCount >= len(spans) {
			if packetizer.NumPackets() == 0 {
				return rtpPayloads, descriptors, packetCount, nil
			}
			return payloadDst, descriptorDst, 0, ErrRTPPacketPlanTooSmall
		}
		payloadStart := len(rtpPayloads)
		descriptorStart := len(descriptors)
		nextPayloads, nextDescriptors, marker, ok, err := AppendEncoderWebRTCFrameControlRTPPacketWithOptions(rtpPayloads, descriptors, &packetizer, control, f.dependencyStructure, options)
		if err != nil {
			return payloadDst, descriptorDst, 0, err
		}
		if !ok {
			return rtpPayloads, descriptors, packetCount, nil
		}
		spans[packetCount] = EncoderWebRTCRTPPacketSpan{
			PayloadOffset:    payloadStart,
			PayloadLength:    len(nextPayloads) - payloadStart,
			DescriptorOffset: descriptorStart,
			DescriptorLength: len(nextDescriptors) - descriptorStart,
			Marker:           marker,
		}
		rtpPayloads = nextPayloads
		descriptors = nextDescriptors
		packetCount++
	}
}

// RTCEncoder encodes WebRTC AV1 streams from I420, I422, I444, I400, NV12, or
// NV21 input with per-frame dependency descriptors. I422 inputs are adapted to
// 4:2:0. I444 inputs are adapted to 4:2:0 unless NewRTCEncoderWithConfig is
// given an explicit profile-1 4:4:4 color config, in which case EncodeI444 and
// EncodeI444Picture emit native AV1 4:4:4 streams. I400 inputs are adapted to
// 4:2:0 unless NewRTCEncoderWithConfig is given an explicit monochrome color
// config, in which case EncodeI400 and EncodeI400Picture emit native AV1
// monochrome streams for WebRTC L*/S* modes.
// NewRTCEncoder covers single-spatial L1T* temporal ladders;
// NewRTCEncoderWithConfig additionally covers supported multi-spatial
// WebRTC SVC and simulcast modes under CBR or CQP rate control. NewRTCEncoder
// is the CBR convenience constructor and still requires TargetBitrate and
// Framerate.
type RTCEncoder struct {
	stream                  *encoder.WebRTCStream
	yuv420Scratch           I420Frame
	i400Scratch             I400Frame
	i400HighBitDepthScratch I400HighBitDepthFrame
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
	if cfg.TileColumns > 0 {
		stream.SetTileColumns(cfg.TileColumns)
	}
	if cfg.GoldenInterval < 0 {
		stream.SetGoldenInterval(0)
	} else if cfg.GoldenInterval > 0 {
		stream.SetGoldenInterval(cfg.GoldenInterval)
	}
	if err := stream.Prewarm(); err != nil {
		_ = stream.Close()
		return nil, err
	}
	return &RTCEncoder{stream: stream}, nil
}

// NormalizeRTCEncoderConfig validates and normalizes cfg for the friendly
// realtime RTCEncoder pixel pipeline. It returns ErrEncoderUnsupported for
// lower-level WebRTC encoder configurations that are valid for control-plane
// helpers but not yet encodable by RTCEncoder's 8-bit profile-0 4:2:0 pipeline.
func NormalizeRTCEncoderConfig(cfg EncoderConfig) (EncoderConfig, error) {
	return encoder.NormalizeWebRTCStreamConfig(cfg)
}

// NewRTCEncoderWithConfig creates a WebRTC encoder from the lower-level WebRTC
// encoder config. Use EncodePicture when the selected scalability mode has more
// than one spatial layer.
func NewRTCEncoderWithConfig(cfg EncoderConfig) (*RTCEncoder, error) {
	stream, err := encoder.NewWebRTCStreamConfig(cfg)
	if err != nil {
		return nil, err
	}
	if err := stream.Prewarm(); err != nil {
		_ = stream.Close()
		return nil, err
	}
	return &RTCEncoder{stream: stream}, nil
}

// Config returns the normalized lower-level WebRTC encoder config.
func (e *RTCEncoder) Config() EncoderConfig {
	if e == nil || e.stream == nil {
		return EncoderConfig{}
	}
	return e.stream.Config()
}

// RTPFrameDuration returns the exact RTP timestamp duration of one encoded
// picture under the encoder's current normalized config.
func (e *RTCEncoder) RTPFrameDuration() (EncoderRational, error) {
	if e == nil || e.stream == nil {
		return EncoderRational{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	return EncoderWebRTCRTPFrameDuration(e.stream.Config())
}

// SetConfig atomically updates bitrate, framerate, rate-control mode, fixed
// quantizer, and supported scalability settings. Changes that alter layer
// geometry or dependency structure make the next encoded picture a key picture
// while preserving frame IDs.
func (e *RTCEncoder) SetConfig(cfg EncoderConfig) error {
	if e == nil || e.stream == nil {
		return fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	return e.stream.SetConfig(cfg)
}

// SetGoldenInterval sets how many base-layer inter frames pass between golden
// reference refreshes in every active spatial encoder. Zero disables golden
// references.
func (e *RTCEncoder) SetGoldenInterval(n int) {
	if e != nil && e.stream != nil {
		e.stream.SetGoldenInterval(n)
	}
}

// SetTileColumns sets the desired tile-column count in every active spatial
// encoder for subsequent encoded frames. The encoder rounds down to a legal
// power-of-two tile layout.
func (e *RTCEncoder) SetTileColumns(cols int) {
	if e != nil && e.stream != nil {
		e.stream.SetTileColumns(cols)
	}
}

// Close waits for any background encoder work to finish and releases
// persistent workers across all spatial encoders. It is safe to call more than
// once.
func (e *RTCEncoder) Close() error {
	if e == nil || e.stream == nil {
		return nil
	}
	err := e.stream.Close()
	e.stream = nil
	return err
}

// Encode encodes one frame with its dependency descriptor. The returned Data
// has the same lifetime as VideoEncoder.Encode; copy it before retaining or
// sending asynchronously.
func (e *RTCEncoder) Encode(frame I420Frame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if !rtcEncoderConfigIsNativeI420(e.stream.Config()) {
		return RTCFrame{}, ErrEncoderUnsupported
	}
	if err := validateI420Frame(frame); err != nil {
		return RTCFrame{}, err
	}
	out, err := e.stream.Encode(frame, forceKey)
	if err != nil {
		return RTCFrame{}, err
	}
	return rtcFrameFromInternal(out), nil
}

// EncodeI422 encodes one single-spatial I422 frame with its dependency
// descriptor after resampling to the current I420 encode path.
func (e *RTCEncoder) EncodeI422(frame I422Frame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	i420, err := i422ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeI444 encodes one single-spatial I444 frame with its dependency
// descriptor, preserving native 4:4:4 when the encoder config requests it and
// otherwise resampling to the current I420 encode path.
func (e *RTCEncoder) EncodeI444(frame I444Frame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if rtcEncoderConfigIsNativeI444(e.stream.Config()) {
		native, err := i444ToNativeScratch(&e.yuv420Scratch, frame)
		if err != nil {
			return RTCFrame{}, err
		}
		out, err := e.stream.Encode(native, forceKey)
		if err != nil {
			return RTCFrame{}, err
		}
		return rtcFrameFromInternal(out), nil
	}
	i420, err := i444ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeI400 encodes one single-spatial monochrome frame with its dependency
// descriptor after filling neutral chroma into the current I420 encode path.
func (e *RTCEncoder) EncodeI400(frame I400Frame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if e.stream.Config().ColorConfig.MonoChrome {
		if err := validateI400Frame(frame); err != nil {
			return RTCFrame{}, err
		}
		out, err := e.stream.EncodeMonochrome(encoder.SourceFrameMono(frame), forceKey)
		if err != nil {
			return RTCFrame{}, err
		}
		return rtcFrameFromInternal(out), nil
	}
	i420, err := i400ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeI400HighBitDepth encodes one single-spatial 10/12-bit monochrome frame
// with its dependency descriptor when the encoder is configured for native
// high-bit-depth monochrome.
func (e *RTCEncoder) EncodeI400HighBitDepth(frame I400HighBitDepthFrame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if !e.stream.Config().ColorConfig.MonoChrome || e.stream.Config().ColorConfig.BitDepth <= 8 {
		return RTCFrame{}, ErrEncoderUnsupported
	}
	out, err := e.stream.EncodeHighBitDepthMonochrome(encoder.SourceFrameMono16(frame), forceKey)
	if err != nil {
		return RTCFrame{}, err
	}
	return rtcFrameFromInternal(out), nil
}

// EncodeFrame encodes one generic Frame with its dependency descriptor. For
// explicit high-bit-depth monochrome RTC configs it preserves matching 10/12-bit
// monochrome input; otherwise it adapts 8/10/12-bit 4:2:0, 4:2:2, 4:4:4, or
// monochrome samples into the current 8-bit 4:2:0 encode path.
func (e *RTCEncoder) EncodeFrame(frame Frame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if e.stream.Config().ColorConfig.MonoChrome {
		if e.stream.Config().ColorConfig.BitDepth > 8 {
			i400, err := frameToI400HighBitDepthScratch(&e.i400HighBitDepthScratch, frame, e.stream.Config().ColorConfig.BitDepth)
			if err != nil {
				return RTCFrame{}, err
			}
			return e.EncodeI400HighBitDepth(i400, forceKey)
		}
		i400, err := frameToI400Scratch(&e.i400Scratch, frame)
		if err != nil {
			return RTCFrame{}, err
		}
		return e.EncodeI400(i400, forceKey)
	}
	if rtcEncoderConfigIsNativeI444(e.stream.Config()) {
		i444, err := frameToI444NativeScratch(&e.yuv420Scratch, frame)
		if err != nil {
			return RTCFrame{}, err
		}
		out, err := e.stream.Encode(i444, forceKey)
		if err != nil {
			return RTCFrame{}, err
		}
		return rtcFrameFromInternal(out), nil
	}
	i420, err := frameToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeNV12 encodes one single-spatial NV12 frame with its dependency
// descriptor. It uses the same output lifetime as Encode.
func (e *RTCEncoder) EncodeNV12(frame NV12Frame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	i420, err := nv12ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodeNV21 encodes one single-spatial NV21 frame with its dependency
// descriptor. It uses the same output lifetime as Encode.
func (e *RTCEncoder) EncodeNV21(frame NV21Frame, forceKey bool) (RTCFrame, error) {
	if e == nil || e.stream == nil {
		return RTCFrame{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	i420, err := nv21ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCFrame{}, err
	}
	return e.Encode(i420, forceKey)
}

// EncodePicture encodes one WebRTC picture. The returned frames have the same
// lifetime as VideoEncoder.Encode; copy frame Data before retaining or sending
// asynchronously.
func (e *RTCEncoder) EncodePicture(frame I420Frame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if !rtcEncoderConfigIsNativeI420(e.stream.Config()) {
		return RTCPicture{}, ErrEncoderUnsupported
	}
	if err := validateI420Frame(frame); err != nil {
		return RTCPicture{}, err
	}
	out, err := e.stream.EncodePicture(frame, forceKey)
	if err != nil {
		return RTCPicture{}, err
	}
	var picture RTCPicture
	picture.FrameNum = int(out.FrameNum)
	picture.Keyframe = out.Keyframe
	for i := 0; i < picture.FrameNum; i++ {
		picture.Frames[i] = rtcFrameFromInternal(out.Frames[i])
	}
	return picture, nil
}

// EncodeI422Picture encodes one I422 WebRTC picture after resampling to the
// current I420 encode path. The returned frames have the same lifetime as
// EncodePicture.
func (e *RTCEncoder) EncodeI422Picture(frame I422Frame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	i420, err := i422ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCPicture{}, err
	}
	return e.EncodePicture(i420, forceKey)
}

// EncodeI444Picture encodes one I444 WebRTC picture, preserving native 4:4:4
// when the encoder config requests it and otherwise resampling to the current
// I420 encode path. The returned frames have the same lifetime as EncodePicture.
func (e *RTCEncoder) EncodeI444Picture(frame I444Frame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if rtcEncoderConfigIsNativeI444(e.stream.Config()) {
		native, err := i444ToNativeScratch(&e.yuv420Scratch, frame)
		if err != nil {
			return RTCPicture{}, err
		}
		out, err := e.stream.EncodePicture(native, forceKey)
		if err != nil {
			return RTCPicture{}, err
		}
		return rtcPictureFromInternal(out), nil
	}
	i420, err := i444ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCPicture{}, err
	}
	return e.EncodePicture(i420, forceKey)
}

// EncodeI400Picture encodes one monochrome WebRTC picture after filling
// neutral chroma into the current I420 encode path. The returned frames have
// the same lifetime as EncodePicture.
func (e *RTCEncoder) EncodeI400Picture(frame I400Frame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if e.stream.Config().ColorConfig.MonoChrome {
		if err := validateI400Frame(frame); err != nil {
			return RTCPicture{}, err
		}
		out, err := e.stream.EncodeMonochromePicture(encoder.SourceFrameMono(frame), forceKey)
		if err != nil {
			return RTCPicture{}, err
		}
		return rtcPictureFromInternal(out), nil
	}
	i420, err := i400ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCPicture{}, err
	}
	return e.EncodePicture(i420, forceKey)
}

// EncodeI400HighBitDepthPicture encodes one 10/12-bit monochrome WebRTC
// picture when the encoder is configured for native high-bit-depth monochrome.
func (e *RTCEncoder) EncodeI400HighBitDepthPicture(frame I400HighBitDepthFrame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if !e.stream.Config().ColorConfig.MonoChrome || e.stream.Config().ColorConfig.BitDepth <= 8 {
		return RTCPicture{}, ErrEncoderUnsupported
	}
	out, err := e.stream.EncodeHighBitDepthMonochromePicture(encoder.SourceFrameMono16(frame), forceKey)
	if err != nil {
		return RTCPicture{}, err
	}
	return rtcPictureFromInternal(out), nil
}

// EncodeFramePicture encodes one generic Frame as a WebRTC picture. For
// explicit high-bit-depth monochrome RTC configs it preserves matching 10/12-bit
// monochrome input; otherwise it adapts the frame into the current 8-bit 4:2:0
// encode path. Multi-spatial output keeps the same dependency-descriptor
// behavior as EncodePicture.
func (e *RTCEncoder) EncodeFramePicture(frame Frame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	if e.stream.Config().ColorConfig.MonoChrome {
		if e.stream.Config().ColorConfig.BitDepth > 8 {
			i400, err := frameToI400HighBitDepthScratch(&e.i400HighBitDepthScratch, frame, e.stream.Config().ColorConfig.BitDepth)
			if err != nil {
				return RTCPicture{}, err
			}
			return e.EncodeI400HighBitDepthPicture(i400, forceKey)
		}
		i400, err := frameToI400Scratch(&e.i400Scratch, frame)
		if err != nil {
			return RTCPicture{}, err
		}
		return e.EncodeI400Picture(i400, forceKey)
	}
	if rtcEncoderConfigIsNativeI444(e.stream.Config()) {
		i444, err := frameToI444NativeScratch(&e.yuv420Scratch, frame)
		if err != nil {
			return RTCPicture{}, err
		}
		out, err := e.stream.EncodePicture(i444, forceKey)
		if err != nil {
			return RTCPicture{}, err
		}
		return rtcPictureFromInternal(out), nil
	}
	i420, err := frameToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCPicture{}, err
	}
	return e.EncodePicture(i420, forceKey)
}

// EncodeNV12Picture encodes one NV12 WebRTC picture. The returned frames have
// the same lifetime as EncodePicture; copy frame Data before retaining or
// sending asynchronously.
func (e *RTCEncoder) EncodeNV12Picture(frame NV12Frame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	i420, err := nv12ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCPicture{}, err
	}
	return e.EncodePicture(i420, forceKey)
}

// EncodeNV21Picture encodes one NV21 WebRTC picture. The returned frames have
// the same lifetime as EncodePicture; copy frame Data before retaining or
// sending asynchronously.
func (e *RTCEncoder) EncodeNV21Picture(frame NV21Frame, forceKey bool) (RTCPicture, error) {
	if e == nil || e.stream == nil {
		return RTCPicture{}, fmt.Errorf("goav1: RTCEncoder is not initialized")
	}
	i420, err := nv21ToI420Scratch(&e.yuv420Scratch, frame)
	if err != nil {
		return RTCPicture{}, err
	}
	return e.EncodePicture(i420, forceKey)
}

func rtcFrameFromInternal(out encoder.WebRTCEncodedFrame) RTCFrame {
	return RTCFrame{
		Data:                      out.TU,
		Keyframe:                  out.Keyframe,
		CodedKeyframe:             out.CodedKeyframe,
		LastFrameInPicture:        out.LastFrameInPicture,
		TemporalID:                out.Info.TemporalID,
		SpatialID:                 out.Info.SpatialID,
		FrameID:                   out.Info.FrameID,
		DependencyDescriptor:      out.Descriptor,
		frameInfo:                 out.Info,
		dependencyStructure:       out.Structure,
		attachDependencyStructure: out.AttachDependencyStructure,
	}
}

func rtcPictureFromInternal(out encoder.WebRTCEncodedPicture) RTCPicture {
	var picture RTCPicture
	picture.FrameNum = int(out.FrameNum)
	picture.Keyframe = out.Keyframe
	for i := 0; i < picture.FrameNum; i++ {
		picture.Frames[i] = rtcFrameFromInternal(out.Frames[i])
	}
	return picture
}

func validateI420Frame(frame I420Frame) error {
	if frame.Width <= 0 || frame.Height <= 0 || frame.Width%2 != 0 || frame.Height%2 != 0 {
		return fmt.Errorf("goav1: I420Frame dimensions must be positive even values, got %dx%d", frame.Width, frame.Height)
	}
	chromaWidth := frame.Width / 2
	chromaHeight := frame.Height / 2
	if frame.YStride < frame.Width {
		return fmt.Errorf("goav1: I420Frame YStride %d is smaller than width %d", frame.YStride, frame.Width)
	}
	if frame.ChromaStride < chromaWidth {
		return fmt.Errorf("goav1: I420Frame ChromaStride %d is smaller than chroma width %d", frame.ChromaStride, chromaWidth)
	}
	yLen, ok := i420PlaneLen(frame.YStride, frame.Width, frame.Height)
	if !ok {
		return fmt.Errorf("goav1: I420Frame Y plane dimensions overflow int")
	}
	chromaLen, ok := i420PlaneLen(frame.ChromaStride, chromaWidth, chromaHeight)
	if !ok {
		return fmt.Errorf("goav1: I420Frame chroma plane dimensions overflow int")
	}
	if len(frame.Y) < yLen {
		return fmt.Errorf("goav1: I420Frame Y plane is too short: got %d bytes, need %d", len(frame.Y), yLen)
	}
	if len(frame.U) < chromaLen {
		return fmt.Errorf("goav1: I420Frame U plane is too short: got %d bytes, need %d", len(frame.U), chromaLen)
	}
	if len(frame.V) < chromaLen {
		return fmt.Errorf("goav1: I420Frame V plane is too short: got %d bytes, need %d", len(frame.V), chromaLen)
	}
	return nil
}

func validateI422Frame(frame I422Frame) error {
	return validatePlanarFrame(
		"I422Frame",
		frame.Y, frame.U, frame.V,
		frame.YStride, frame.ChromaStride, frame.ChromaStride,
		frame.Width, frame.Height,
		frame.Width/2, frame.Height,
		true,
	)
}

func validateI444Frame(frame I444Frame) error {
	return validatePlanarFrame(
		"I444Frame",
		frame.Y, frame.U, frame.V,
		frame.YStride, frame.UStride, frame.VStride,
		frame.Width, frame.Height,
		frame.Width, frame.Height,
		true,
	)
}

func validateI400Frame(frame I400Frame) error {
	return validatePlanarFrame(
		"I400Frame",
		frame.Y, nil, nil,
		frame.YStride, 0, 0,
		frame.Width, frame.Height,
		0, 0,
		false,
	)
}

func validatePlanarFrame(
	name string,
	y []byte, u []byte, v []byte,
	yStride int, uStride int, vStride int,
	width int, height int,
	chromaWidth int, chromaHeight int,
	hasChroma bool,
) error {
	if width <= 0 || height <= 0 || width%2 != 0 || height%2 != 0 {
		return fmt.Errorf("goav1: %s dimensions must be positive even values, got %dx%d", name, width, height)
	}
	if yStride < width {
		return fmt.Errorf("goav1: %s YStride %d is smaller than width %d", name, yStride, width)
	}
	yLen, ok := i420PlaneLen(yStride, width, height)
	if !ok {
		return fmt.Errorf("goav1: %s Y plane dimensions overflow int", name)
	}
	if len(y) < yLen {
		return fmt.Errorf("goav1: %s Y plane is too short: got %d bytes, need %d", name, len(y), yLen)
	}
	if !hasChroma {
		return nil
	}
	if uStride < chromaWidth {
		return fmt.Errorf("goav1: %s UStride %d is smaller than chroma width %d", name, uStride, chromaWidth)
	}
	if vStride < chromaWidth {
		return fmt.Errorf("goav1: %s VStride %d is smaller than chroma width %d", name, vStride, chromaWidth)
	}
	uLen, ok := i420PlaneLen(uStride, chromaWidth, chromaHeight)
	if !ok {
		return fmt.Errorf("goav1: %s U plane dimensions overflow int", name)
	}
	vLen, ok := i420PlaneLen(vStride, chromaWidth, chromaHeight)
	if !ok {
		return fmt.Errorf("goav1: %s V plane dimensions overflow int", name)
	}
	if len(u) < uLen {
		return fmt.Errorf("goav1: %s U plane is too short: got %d bytes, need %d", name, len(u), uLen)
	}
	if len(v) < vLen {
		return fmt.Errorf("goav1: %s V plane is too short: got %d bytes, need %d", name, len(v), vLen)
	}
	return nil
}

func validateNV12Frame(frame NV12Frame) error {
	return validateSemiPlanar420Frame("NV12Frame", "UV", frame.Y, frame.UV, frame.YStride, frame.UVStride, frame.Width, frame.Height)
}

func validateNV21Frame(frame NV21Frame) error {
	return validateSemiPlanar420Frame("NV21Frame", "VU", frame.Y, frame.VU, frame.YStride, frame.VUStride, frame.Width, frame.Height)
}

func validateSemiPlanar420Frame(name string, chromaName string, y []byte, chroma []byte, yStride int, chromaStride int, width int, height int) error {
	if width <= 0 || height <= 0 || width%2 != 0 || height%2 != 0 {
		return fmt.Errorf("goav1: %s dimensions must be positive even values, got %dx%d", name, width, height)
	}
	if yStride < width {
		return fmt.Errorf("goav1: %s YStride %d is smaller than width %d", name, yStride, width)
	}
	if chromaStride < width {
		return fmt.Errorf("goav1: %s %sStride %d is smaller than width %d", name, chromaName, chromaStride, width)
	}
	yLen, ok := i420PlaneLen(yStride, width, height)
	if !ok {
		return fmt.Errorf("goav1: %s Y plane dimensions overflow int", name)
	}
	chromaLen, ok := i420PlaneLen(chromaStride, width, height/2)
	if !ok {
		return fmt.Errorf("goav1: %s %s plane dimensions overflow int", name, chromaName)
	}
	if len(y) < yLen {
		return fmt.Errorf("goav1: %s Y plane is too short: got %d bytes, need %d", name, len(y), yLen)
	}
	if len(chroma) < chromaLen {
		return fmt.Errorf("goav1: %s %s plane is too short: got %d bytes, need %d", name, chromaName, len(chroma), chromaLen)
	}
	return nil
}

func frameToI420Scratch(dst *I420Frame, frame Frame) (I420Frame, error) {
	format := frame.Format
	if format.BitDepth == 0 {
		format.BitDepth = 8
	}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		return I420Frame{}, err
	}
	if format.Width <= 0 || format.Height <= 0 || format.Width%2 != 0 || format.Height%2 != 0 {
		return I420Frame{}, fmt.Errorf("goav1: Frame dimensions must be positive even values, got %dx%d", format.Width, format.Height)
	}
	if format.BitDepth != 8 && format.BitDepth != 10 && format.BitDepth != 12 {
		return I420Frame{}, ErrFrameInvalidFormat
	}
	if !format.MonoChrome && !format.SubsamplingX && format.SubsamplingY {
		return I420Frame{}, ErrFrameInvalidFormat
	}
	if frame.Y.Width != format.Width || frame.Y.Height != format.Height || !encoderPlaneFits(frame.Y, layout.BytesPerSample) {
		return I420Frame{}, ErrFrameInvalidPlane
	}
	if !format.MonoChrome {
		if frame.U.Width != layout.ChromaWidth || frame.U.Height != layout.ChromaHeight ||
			frame.V.Width != layout.ChromaWidth || frame.V.Height != layout.ChromaHeight ||
			!encoderPlaneFits(frame.U, layout.BytesPerSample) ||
			!encoderPlaneFits(frame.V, layout.BytesPerSample) {
			return I420Frame{}, ErrFrameInvalidPlane
		}
		if format.BitDepth == 8 && format.SubsamplingX && format.SubsamplingY && frame.U.Stride == frame.V.Stride {
			return I420Frame{
				Y:            frame.Y.Pix,
				U:            frame.U.Pix,
				V:            frame.V.Pix,
				YStride:      frame.Y.Stride,
				ChromaStride: frame.U.Stride,
				Width:        format.Width,
				Height:       format.Height,
			}, nil
		}
	}

	frameI420ByteScratch(dst, format.Width, format.Height)
	copyFramePlaneTo8(dst.Y, dst.YStride, frame.Y, layout.BytesPerSample, format.BitDepth, format.Width, format.Height)
	switch {
	case format.MonoChrome:
		for i := range dst.U {
			dst.U[i] = 128
			dst.V[i] = 128
		}
	case format.SubsamplingX && format.SubsamplingY:
		copyFramePlaneTo8(dst.U, dst.ChromaStride, frame.U, layout.BytesPerSample, format.BitDepth, format.Width/2, format.Height/2)
		copyFramePlaneTo8(dst.V, dst.ChromaStride, frame.V, layout.BytesPerSample, format.BitDepth, format.Width/2, format.Height/2)
	case format.SubsamplingX:
		downsampleFramePlane422To420(dst.U, dst.ChromaStride, frame.U, layout.BytesPerSample, format.BitDepth, format.Width/2, format.Height/2)
		downsampleFramePlane422To420(dst.V, dst.ChromaStride, frame.V, layout.BytesPerSample, format.BitDepth, format.Width/2, format.Height/2)
	default:
		downsampleFramePlane444To420(dst.U, dst.ChromaStride, frame.U, layout.BytesPerSample, format.BitDepth, format.Width/2, format.Height/2)
		downsampleFramePlane444To420(dst.V, dst.ChromaStride, frame.V, layout.BytesPerSample, format.BitDepth, format.Width/2, format.Height/2)
	}
	return *dst, nil
}

func frameToI444NativeScratch(dst *I420Frame, frame Frame) (I420Frame, error) {
	format := frame.Format
	if format.BitDepth == 0 {
		format.BitDepth = 8
	}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		return I420Frame{}, err
	}
	if format.Width <= 0 || format.Height <= 0 || format.Width%2 != 0 || format.Height%2 != 0 {
		return I420Frame{}, fmt.Errorf("goav1: Frame dimensions must be positive even values, got %dx%d", format.Width, format.Height)
	}
	if format.BitDepth != 8 || format.MonoChrome || format.SubsamplingX || format.SubsamplingY {
		return I420Frame{}, ErrFrameInvalidFormat
	}
	if frame.Y.Width != format.Width || frame.Y.Height != format.Height ||
		frame.U.Width != layout.ChromaWidth || frame.U.Height != layout.ChromaHeight ||
		frame.V.Width != layout.ChromaWidth || frame.V.Height != layout.ChromaHeight ||
		!encoderPlaneFits(frame.Y, layout.BytesPerSample) ||
		!encoderPlaneFits(frame.U, layout.BytesPerSample) ||
		!encoderPlaneFits(frame.V, layout.BytesPerSample) {
		return I420Frame{}, ErrFrameInvalidPlane
	}
	if frame.U.Stride == frame.V.Stride {
		return I420Frame{
			Y:            frame.Y.Pix,
			U:            frame.U.Pix,
			V:            frame.V.Pix,
			YStride:      frame.Y.Stride,
			ChromaStride: frame.U.Stride,
			Width:        format.Width,
			Height:       format.Height,
		}, nil
	}
	frameI444ByteScratch(dst, format.Width, format.Height)
	dst.Y = frame.Y.Pix
	dst.YStride = frame.Y.Stride
	copyFramePlaneTo8(dst.U, dst.ChromaStride, frame.U, layout.BytesPerSample, format.BitDepth, format.Width, format.Height)
	copyFramePlaneTo8(dst.V, dst.ChromaStride, frame.V, layout.BytesPerSample, format.BitDepth, format.Width, format.Height)
	return *dst, nil
}

func frameToI400Scratch(dst *I400Frame, frame Frame) (I400Frame, error) {
	format := frame.Format
	if format.BitDepth == 0 {
		format.BitDepth = 8
	}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		return I400Frame{}, err
	}
	if format.Width <= 0 || format.Height <= 0 || format.Width%2 != 0 || format.Height%2 != 0 {
		return I400Frame{}, fmt.Errorf("goav1: Frame dimensions must be positive even values, got %dx%d", format.Width, format.Height)
	}
	if !format.MonoChrome {
		return I400Frame{}, ErrFrameInvalidFormat
	}
	if format.BitDepth != 8 && format.BitDepth != 10 && format.BitDepth != 12 {
		return I400Frame{}, ErrFrameInvalidFormat
	}
	if frame.Y.Width != format.Width || frame.Y.Height != format.Height || !encoderPlaneFits(frame.Y, layout.BytesPerSample) {
		return I400Frame{}, ErrFrameInvalidPlane
	}
	if format.BitDepth == 8 {
		return I400Frame{
			Y:       frame.Y.Pix,
			YStride: frame.Y.Stride,
			Width:   format.Width,
			Height:  format.Height,
		}, nil
	}
	frameI400ByteScratch(dst, format.Width, format.Height)
	copyFramePlaneTo8(dst.Y, dst.YStride, frame.Y, layout.BytesPerSample, format.BitDepth, format.Width, format.Height)
	return *dst, nil
}

func frameToI400HighBitDepthScratch(dst *I400HighBitDepthFrame, frame Frame, bitDepth uint8) (I400HighBitDepthFrame, error) {
	format := frame.Format
	if format.BitDepth == 0 {
		format.BitDepth = 8
	}
	layout, err := FrameRequiredSize(frame.Format)
	if err != nil {
		return I400HighBitDepthFrame{}, err
	}
	if format.Width <= 0 || format.Height <= 0 || format.Width%2 != 0 || format.Height%2 != 0 {
		return I400HighBitDepthFrame{}, fmt.Errorf("goav1: Frame dimensions must be positive even values, got %dx%d", format.Width, format.Height)
	}
	if !format.MonoChrome || format.BitDepth != bitDepth || (bitDepth != 10 && bitDepth != 12) {
		return I400HighBitDepthFrame{}, ErrFrameInvalidFormat
	}
	if frame.Y.Width != format.Width || frame.Y.Height != format.Height || !encoderPlaneFits(frame.Y, layout.BytesPerSample) {
		return I400HighBitDepthFrame{}, ErrFrameInvalidPlane
	}
	frameI400HighBitDepthScratch(dst, format.Width, format.Height, bitDepth)
	copyFramePlaneTo16(dst.Y, dst.YStride, frame.Y, layout.BytesPerSample, format.Width, format.Height)
	return *dst, nil
}

func frameI420ByteScratch(dst *I420Frame, width int, height int) {
	chromaWidth := width / 2
	chromaHeight := height / 2
	yLen := width * height
	chromaLen := chromaWidth * chromaHeight
	if cap(dst.Y) < yLen {
		dst.Y = make([]byte, yLen)
	} else {
		dst.Y = dst.Y[:yLen]
	}
	if cap(dst.U) < chromaLen {
		dst.U = make([]byte, chromaLen)
	} else {
		dst.U = dst.U[:chromaLen]
	}
	if cap(dst.V) < chromaLen {
		dst.V = make([]byte, chromaLen)
	} else {
		dst.V = dst.V[:chromaLen]
	}
	dst.YStride = width
	dst.ChromaStride = chromaWidth
	dst.Width = width
	dst.Height = height
}

func frameI444ByteScratch(dst *I420Frame, width int, height int) {
	yLen := width * height
	chromaLen := width * height
	if cap(dst.Y) < yLen {
		dst.Y = make([]byte, yLen)
	} else {
		dst.Y = dst.Y[:yLen]
	}
	if cap(dst.U) < chromaLen {
		dst.U = make([]byte, chromaLen)
	} else {
		dst.U = dst.U[:chromaLen]
	}
	if cap(dst.V) < chromaLen {
		dst.V = make([]byte, chromaLen)
	} else {
		dst.V = dst.V[:chromaLen]
	}
	dst.YStride = width
	dst.ChromaStride = width
	dst.Width = width
	dst.Height = height
}

func frameI400ByteScratch(dst *I400Frame, width int, height int) {
	yLen := width * height
	if cap(dst.Y) < yLen {
		dst.Y = make([]byte, yLen)
	} else {
		dst.Y = dst.Y[:yLen]
	}
	dst.YStride = width
	dst.Width = width
	dst.Height = height
}

func frameI400HighBitDepthScratch(dst *I400HighBitDepthFrame, width int, height int, bitDepth uint8) {
	yLen := width * height
	if cap(dst.Y) < yLen {
		dst.Y = make([]uint16, yLen)
	} else {
		dst.Y = dst.Y[:yLen]
	}
	dst.YStride = width
	dst.Width = width
	dst.Height = height
	dst.BitDepth = bitDepth
}

func copyFramePlaneTo8(dst []byte, dstStride int, src FramePlane, bytesPerSample int, bitDepth uint8, width int, height int) {
	for y := 0; y < height; y++ {
		row := dst[y*dstStride : y*dstStride+width]
		for x := 0; x < width; x++ {
			row[x] = frameSampleTo8(framePlaneSample(src, bytesPerSample, x, y), bitDepth)
		}
	}
}

func copyFramePlaneTo16(dst []uint16, dstStride int, src FramePlane, bytesPerSample int, width int, height int) {
	for y := 0; y < height; y++ {
		row := dst[y*dstStride : y*dstStride+width]
		for x := 0; x < width; x++ {
			row[x] = framePlaneSample(src, bytesPerSample, x, y)
		}
	}
}

func downsampleFramePlane422To420(dst []byte, dstStride int, src FramePlane, bytesPerSample int, bitDepth uint8, width int, height int) {
	for y := 0; y < height; y++ {
		row := dst[y*dstStride : y*dstStride+width]
		sy := y * 2
		for x := 0; x < width; x++ {
			sum := uint32(framePlaneSample(src, bytesPerSample, x, sy)) +
				uint32(framePlaneSample(src, bytesPerSample, x, sy+1))
			row[x] = frameSampleAverageTo8(sum, 2, bitDepth)
		}
	}
}

func downsampleFramePlane444To420(dst []byte, dstStride int, src FramePlane, bytesPerSample int, bitDepth uint8, width int, height int) {
	for y := 0; y < height; y++ {
		row := dst[y*dstStride : y*dstStride+width]
		sy := y * 2
		for x := 0; x < width; x++ {
			sx := x * 2
			sum := uint32(framePlaneSample(src, bytesPerSample, sx, sy)) +
				uint32(framePlaneSample(src, bytesPerSample, sx+1, sy)) +
				uint32(framePlaneSample(src, bytesPerSample, sx, sy+1)) +
				uint32(framePlaneSample(src, bytesPerSample, sx+1, sy+1))
			row[x] = frameSampleAverageTo8(sum, 4, bitDepth)
		}
	}
}

func framePlaneSample(plane FramePlane, bytesPerSample int, x int, y int) uint16 {
	off := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[off])
	}
	return uint16(plane.Pix[off]) | uint16(plane.Pix[off+1])<<8
}

func frameSampleAverageTo8(sum uint32, count uint32, bitDepth uint8) byte {
	if count == 0 {
		return 0
	}
	return frameSampleTo8(uint16((sum+count/2)/count), bitDepth)
}

func frameSampleTo8(sample uint16, bitDepth uint8) byte {
	if bitDepth <= 8 {
		if sample > 255 {
			return 255
		}
		return byte(sample)
	}
	v := uint32(sample) >> (bitDepth - 8)
	if v > 255 {
		v = 255
	}
	return byte(v)
}

func i422ToI420Scratch(dst *I420Frame, frame I422Frame) (I420Frame, error) {
	if err := validateI422Frame(frame); err != nil {
		return I420Frame{}, err
	}
	chromaWidth := frame.Width / 2
	chromaHeight := frame.Height / 2
	planar420Scratch(dst, frame.Y, frame.YStride, frame.Width, frame.Height, chromaWidth, chromaHeight)
	for y := 0; y < chromaHeight; y++ {
		srcY0 := (y * 2) * frame.ChromaStride
		srcY1 := srcY0 + frame.ChromaStride
		u0 := frame.U[srcY0 : srcY0+chromaWidth]
		u1 := frame.U[srcY1 : srcY1+chromaWidth]
		v0 := frame.V[srcY0 : srcY0+chromaWidth]
		v1 := frame.V[srcY1 : srcY1+chromaWidth]
		du := dst.U[y*chromaWidth : y*chromaWidth+chromaWidth]
		dv := dst.V[y*chromaWidth : y*chromaWidth+chromaWidth]
		for x := 0; x < chromaWidth; x++ {
			du[x] = uint8((uint16(u0[x]) + uint16(u1[x]) + 1) >> 1)
			dv[x] = uint8((uint16(v0[x]) + uint16(v1[x]) + 1) >> 1)
		}
	}
	return *dst, nil
}

func i444ToI420Scratch(dst *I420Frame, frame I444Frame) (I420Frame, error) {
	if err := validateI444Frame(frame); err != nil {
		return I420Frame{}, err
	}
	chromaWidth := frame.Width / 2
	chromaHeight := frame.Height / 2
	planar420Scratch(dst, frame.Y, frame.YStride, frame.Width, frame.Height, chromaWidth, chromaHeight)
	for y := 0; y < chromaHeight; y++ {
		uY0 := (y * 2) * frame.UStride
		uY1 := uY0 + frame.UStride
		vY0 := (y * 2) * frame.VStride
		vY1 := vY0 + frame.VStride
		for x := 0; x < chromaWidth; x++ {
			ux := x * 2
			vx := x * 2
			uSum := uint16(frame.U[uY0+ux]) + uint16(frame.U[uY0+ux+1]) +
				uint16(frame.U[uY1+ux]) + uint16(frame.U[uY1+ux+1])
			vSum := uint16(frame.V[vY0+vx]) + uint16(frame.V[vY0+vx+1]) +
				uint16(frame.V[vY1+vx]) + uint16(frame.V[vY1+vx+1])
			dst.U[y*chromaWidth+x] = uint8((uSum + 2) >> 2)
			dst.V[y*chromaWidth+x] = uint8((vSum + 2) >> 2)
		}
	}
	return *dst, nil
}

func i444ToNativeScratch(dst *I420Frame, frame I444Frame) (I420Frame, error) {
	if err := validateI444Frame(frame); err != nil {
		return I420Frame{}, err
	}
	if frame.UStride == frame.VStride {
		return I420Frame{
			Y:            frame.Y,
			U:            frame.U,
			V:            frame.V,
			YStride:      frame.YStride,
			ChromaStride: frame.UStride,
			Width:        frame.Width,
			Height:       frame.Height,
		}, nil
	}
	frameI444ByteScratch(dst, frame.Width, frame.Height)
	dst.Y = frame.Y
	dst.YStride = frame.YStride
	for y := 0; y < frame.Height; y++ {
		copy(dst.U[y*dst.ChromaStride:y*dst.ChromaStride+frame.Width], frame.U[y*frame.UStride:y*frame.UStride+frame.Width])
		copy(dst.V[y*dst.ChromaStride:y*dst.ChromaStride+frame.Width], frame.V[y*frame.VStride:y*frame.VStride+frame.Width])
	}
	return *dst, nil
}

func i400ToI420Scratch(dst *I420Frame, frame I400Frame) (I420Frame, error) {
	if err := validateI400Frame(frame); err != nil {
		return I420Frame{}, err
	}
	chromaWidth := frame.Width / 2
	chromaHeight := frame.Height / 2
	planar420Scratch(dst, frame.Y, frame.YStride, frame.Width, frame.Height, chromaWidth, chromaHeight)
	for i := range dst.U {
		dst.U[i] = 128
		dst.V[i] = 128
	}
	return *dst, nil
}

func nv12ToI420Scratch(dst *I420Frame, frame NV12Frame) (I420Frame, error) {
	if err := validateNV12Frame(frame); err != nil {
		return I420Frame{}, err
	}
	return semiPlanar420ToI420Scratch(dst, frame.Y, frame.UV, frame.YStride, frame.UVStride, frame.Width, frame.Height, 0, 1), nil
}

func nv21ToI420Scratch(dst *I420Frame, frame NV21Frame) (I420Frame, error) {
	if err := validateNV21Frame(frame); err != nil {
		return I420Frame{}, err
	}
	return semiPlanar420ToI420Scratch(dst, frame.Y, frame.VU, frame.YStride, frame.VUStride, frame.Width, frame.Height, 1, 0), nil
}

func planar420Scratch(dst *I420Frame, y []byte, yStride int, width int, height int, chromaWidth int, chromaHeight int) {
	chromaLen := chromaWidth * chromaHeight
	if cap(dst.U) < chromaLen {
		dst.U = make([]byte, chromaLen)
	} else {
		dst.U = dst.U[:chromaLen]
	}
	if cap(dst.V) < chromaLen {
		dst.V = make([]byte, chromaLen)
	} else {
		dst.V = dst.V[:chromaLen]
	}
	dst.Y = y
	dst.YStride = yStride
	dst.ChromaStride = chromaWidth
	dst.Width = width
	dst.Height = height
}

func semiPlanar420ToI420Scratch(dst *I420Frame, yPlane []byte, chromaPlane []byte, yStride int, chromaStride int, width int, height int, uOffset int, vOffset int) I420Frame {
	chromaWidth := width / 2
	chromaHeight := height / 2
	planar420Scratch(dst, yPlane, yStride, width, height, chromaWidth, chromaHeight)
	for y := 0; y < chromaHeight; y++ {
		chroma := chromaPlane[y*chromaStride : y*chromaStride+width]
		u := dst.U[y*chromaWidth : y*chromaWidth+chromaWidth]
		v := dst.V[y*chromaWidth : y*chromaWidth+chromaWidth]
		for x := 0; x < chromaWidth; x++ {
			u[x] = chroma[x*2+uOffset]
			v[x] = chroma[x*2+vOffset]
		}
	}
	return *dst
}

func i420PlaneLen(stride int, width int, height int) (int, bool) {
	maxInt := int(^uint(0) >> 1)
	rowsBeforeLast := height - 1
	if rowsBeforeLast > (maxInt-width)/stride {
		return 0, false
	}
	return rowsBeforeLast*stride + width, true
}

func rtcEncoderConfigIsNativeI420(cfg EncoderConfig) bool {
	return !cfg.ColorConfig.MonoChrome &&
		cfg.ColorConfig.BitDepth == 8 &&
		cfg.ColorConfig.SubsamplingX &&
		cfg.ColorConfig.SubsamplingY
}

func rtcEncoderConfigIsNativeI444(cfg EncoderConfig) bool {
	return cfg.Profile == EncoderProfile1 &&
		!cfg.ColorConfig.MonoChrome &&
		cfg.ColorConfig.BitDepth == 8 &&
		!cfg.ColorConfig.SubsamplingX &&
		!cfg.ColorConfig.SubsamplingY
}
