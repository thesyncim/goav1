// Package goav1 is a pure-Go AV1 decoder and realtime AV1 encoder for WebRTC.
//
// Decoding: NewDecoder, NewDecoderFromIVF, NewDecoderFromIVFReaderAt and the
// RTP payload and packet constructors build a Decoder over a complete stream
// and size every buffer from its headers; DecodeNext then returns frames
// without allocating. WithWorkers enables tile, wavefront and frame threading,
// WithMaxFrameDelay bounds frame threading's output delay and
// WithFrameSizeLimit rejects streams larger than the caller will allocate for.
// LayeredDecoder is the receive path for streams whose reference slots span
// several coded frame sizes, such as WebRTC shared-reference SVC.
// NewIVFIterator walks IVF files.
//
// Encoding: NewVideoEncoder and NewRTCEncoderWithConfig encode I420, NV12,
// NV21, I422, I444, I400 and high-bit-depth frames for realtime use, including
// temporal and spatial scalability, rate control and runtime reconfiguration.
// RTCEncoder emits RTP payloads with dependency descriptors.
//
// WebRTC transport helpers cover AV1 RTP payload packetization and assembly,
// RTP header and RFC 8285 extension framing, RFC 4588 retransmission, RTCP
// packets and feedback (NACK, PLI, FIR, REMB, Transport-CC, AV1 layer refresh),
// packet sequencing and AV1 SDP negotiation. SRTP, ICE, full SDP assembly and
// network scheduling stay with the caller.
//
// Returned byte slices and frames alias decoder- or caller-owned buffers;
// each function documents how long they stay valid.
package goav1
