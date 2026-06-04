// Package goav1 is a pure-Go AV1 codec package focused on realtime and
// WebRTC use.
//
// The package consumes and produces AV1 Open Bitstream Units and AV1 RTP
// payload bodies. RTP headers, SRTP, SDP, jitter buffering, packet-loss policy,
// and network scheduling stay caller-owned.
//
// The implementation is built from byte-exact transport and parser primitives
// inward. Public helpers expose IVF containers, low-overhead, temporal-unit,
// and Annex B OBU parsing, AV1 RTP payload iteration, construction,
// caller-owned RTP sizing, sequence-header parsing, caller-owned frame pools
// and sample-plane scratch helpers, DSP block/blend helpers, intra/inter
// prediction, decoder frame-work prediction bridges, residual reconstruction
// primitives, post-reconstruction filtering helpers, tile block/transform
// geometry, transform/coefficient context replay helpers, tile coefficient
// replay, tile block coefficient decode, tile-level restoration orchestration,
// decoder residual-state setup and residual execution/retention bridges for
// jobs and worker batches, decoder block coefficient reconstruction,
// coefficient replay reconstruction adapters, one-block coefficient
// decode/reconstruct helpers, decoder postfilter scratch binding, superres
// upscaling, film-grain output helpers, and the WebRTC AV1 encoder control
// foundation. The decoder and encoder APIs will grow at this top level as the
// internal pipeline stabilizes.
//
// Hot paths use caller-owned buffers and fixed storage. Returned byte slices
// alias caller-provided input or output buffers unless a future API explicitly
// documents ownership transfer.
package goav1
