// Package goav1 is a pure-Go AV1 codec package focused on realtime and
// WebRTC use.
//
// The package consumes and produces AV1 Open Bitstream Units, AV1 RTP payload
// bodies, and RTP packet/header syntax. SRTP, full SDP assembly, jitter
// buffering, packet-loss policy, and network scheduling stay caller-owned.
//
// The implementation is built from byte-exact transport and parser primitives
// inward. Public helpers expose IVF containers, low-overhead, temporal-unit,
// and Annex B OBU parsing, AV1 RTP fixed-header and RFC 8285 extension element
// framing and negotiated-extension lookup, complete RTP packet dependency
// descriptor extraction, AV1 RTP payload iteration, construction, caller-owned
// RTP sizing, complete RTP packet decode wrappers, generic and compound RTCP packet
// helpers, RTCP
// SR/RR/SDES/BYE and RTPFB/PSFB packet helpers, RTCP
// NACK/Transport-CC/PLI/FIR/REMB FCI helpers, AV1 RTCP Layer Refresh Request
// FCI list helpers, RTCP feedback force-key classification for single and
// compound packets, sequence-header parsing, caller-owned frame pools and
// sample-plane scratch helpers, DSP block/blend helpers, intra/inter
// prediction, decoder
// frame-work prediction bridges, residual reconstruction primitives,
// post-reconstruction filtering helpers, AV1 SDP/rtpmap/fmtp/rtcp-fb
// capability parsing and emission, tile block/transform
// geometry, transform/coefficient context replay helpers, tile coefficient
// replay, tile block coefficient decode, tile-level restoration orchestration,
// decoder residual-state setup and residual execution/retention bridges for
// jobs and worker batches, decoder block coefficient reconstruction,
// coefficient replay reconstruction adapters, one-block coefficient
// decode/reconstruct helpers, decoder postfilter scratch binding, superres
// upscaling, film-grain output helpers, a realtime 8-bit profile-0 AV1 encoder
// for WebRTC streams with I420/I422/I444/I400/NV12/NV21 and generic Frame
// input adapters, RTCEncoder config preflight
// normalization, RTP payload packetization with dependency descriptors, sized
// complete RTP packet wrapping, multi-spatial EncodePicture output for WebRTC
// SVC and simulcast modes, and runtime bitrate/framerate/scalability
// reconfiguration. Lower-level WebRTC encoder
// helpers expose W3C SVC mode metadata, temporal/spatial dependency
// structures, decode-target grids, active decode-target masks, key-shift
// scheduling, explicit sequence color config, sequence-matched sample-plane
// loading, and the supported scalabilityMode capability list. High-level
// decoder helpers cover low-overhead streams, ordered/live AV1 RTP payload
// bodies or complete packets, retained-fragment reset after RTP loss, and
// shared-reference SVC RTP receive loops with decoded AV1 layer metadata;
// frame-work/layer-pool APIs remain available when callers need explicit
// surface routing.
//
// Hot paths use caller-owned buffers and fixed storage. Returned byte slices
// alias caller-provided input or output buffers unless a future API explicitly
// documents ownership transfer.
package goav1
