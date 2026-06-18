# Changelog

Release-level changes to goav1 are tracked here. Detailed progress logs,
per-vector archaeology, and work-in-progress notes belong in commits, issues,
or pull requests, not in long-lived markdown.

The project has not published a release tag yet.

## [Unreleased]

### Decoder

- Pure-Go AV1 decode pipeline covers IVF, OBU, Annex B, RTP, frame headers,
  tile decode, prediction, transforms, reconstruction, loop filter, CDEF,
  super-resolution, loop restoration, film grain, SVC routing, and raw YUV
  output.
- Current committed decoder vector gates are strict-MD5 clean as documented in
  [CONFORMANCE.md](CONFORMANCE.md).
- Hot paths are designed around caller-owned buffers, reusable frame pools,
  explicit scratch sizing, and zero steady-state allocations.
- Performance work remains active, with dav1d/libaom source-shaped scalar code,
  compiler/allocation guardrails, and initial SIMD dispatch/kernels.

### Encoder Scope

- The encoder target is a realtime WebRTC AV1 encoder, not a general-purpose
  offline AV1 encoder.
- The intended surface is WebRTC-style control and transport: bitrate,
  framerate, resolution and keyframe controls; temporal and spatial layer/SVC
  configuration; camera and screen-content tuning; realtime speed/quality
  controls; low-overhead OBU/RTP output; and dependency/scalability metadata
  needed by WebRTC integrations.
- Encoder implementation is in scope. The public realtime WebRTC API is usable
  for 8-bit I420 streams, including temporal layering, multi-spatial SVC and
  simulcast pictures, RTP payload packetization, dependency descriptors, and
  runtime bitrate/framerate/scalability reconfiguration. New behavior must
  continue to be ported from pinned libaom/libwebrtc source, with pinned
  SVT-AV1 used as the speed-architecture reference. New/touched code must keep
  relevant upstream C type widths and signedness, and should land with oracle
  tests and allocation evidence appropriate to the path.

### Verification

- Local development gates: `go test ./...`, `make alloc`,
  `make compiler-reports`, and `make trace-zero`.
- Decoder vector gates: `make dryrun-fast`, `make dryrun-full`,
  `make dryrun-extended`, and `make dryrun-profiles`.
- Cross-decoder performance tracking: `make bench-cross` and the optional
  generated corpus lane documented in [README.md](README.md).
