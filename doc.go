// Package goav1 is a pure-Go AV1 decoder and encoder with WebRTC RTP/RTCP
// helpers, using Go SIMD (GOEXPERIMENT=simd) with zero cgo and zero
// handwritten assembly.
//
// Decoding: NewDecoder, NewDecoderFromIVF, NewDecoderFromIVFReaderAt, and the
// RTP payload/packet constructors build a Decoder over a stream and size every
// buffer up front so steady-state decoding (DecodeNext, DecodePayload,
// DecodeRTPPayload) is allocation-free. Options configure tile/wavefront/frame
// threading (WithWorkers, WithMaxFrameDelay), allocation bounds
// (WithFrameSizeLimit), film-grain synthesis (WithFilmGrain), AV1 §7.5
// operating-point and layer selection (WithOperatingPoint, WithAllLayers), and
// resilient freeze-frame error concealment (WithErrorConcealment). Decoded
// frames expose ColorInfo, FrameMetadata (HDR CLL, HDR MDCV, ITU-T T.35,
// Timecode, Scalability), and Concealed state. LayeredDecoder handles
// multi-resolution WebRTC shared-reference SVC streams. NewIVFIterator walks
// IVF files.
//
// Encoding: NewVideoEncoder encodes 8-bit 4:2:0, native 4:2:2 (Profile 2),
// native 4:4:4 (Profile 1), and bit-exact lossless streams with fixed-quality
// CQP or CBR rate control, ColorInfo and OBU_METADATA signaling, periodic
// keyframes, and runtime SetQIndex/SetBitrate. NewHighBitDepthVideoEncoder,
// NewMonochromeVideoEncoder, and NewHighBitDepthMonochromeVideoEncoder provide
// native 10/12-bit and monochrome streaming encoders. NewRTCEncoderWithConfig
// adds WebRTC temporal/spatial scalability and RTP dependency descriptors.
//
// WebRTC transport helpers cover AV1 RTP payload packetization and assembly,
// RTP header and RFC 8285 extension framing, RFC 4588 retransmission, RTCP
// packets and feedback (NACK, PLI, FIR, REMB, Transport-CC, AV1 layer refresh),
// packet sequencing, and AV1 SDP negotiation. Container muxing beyond IVF and
// SRTP/ICE/network scheduling stay with the caller.
//
// Returned byte slices and frames alias decoder-, encoder-, or caller-owned
// buffers; each function documents how long they stay valid.
package goav1
