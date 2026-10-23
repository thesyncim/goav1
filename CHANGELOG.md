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
- With more than one worker, low-overhead decoders (NewDecoder,
  NewDecoderFromIVF and the reader variants) run dav1d-style frame threading
  by default: up to ceil(sqrt(workers)) frames are in flight, so DecodeNext can
  return a frame a few calls after the payload that carried it, and the last
  calls drain the pipeline. Output order and pixels are unchanged.
  WithMaxFrameDelay(1) restores one-frame-per-call delivery (dav1d
  max_frame_delay). RTP decoders are realtime and never frame-thread.
  Frames with several tiles frame-thread too, as long as every tile arrives in
  one tile group: each tile's entropy pass runs on its own lane, and the
  frame's reconstruction and post-filter overlap the next frame's tiles.

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

### Encoder

- `VideoEncoder` no longer promotes delta frames to keyframes on detected
  scene cuts: the HME cut detector, `DisableSceneCutKeyframes`,
  `SetSceneCutKeyframes` and qualitybench `-goav1-scene-cut` are removed. This
  matches SVT-AV1 4.1.0 rtc, which forces scene-change detection off and codes
  cuts with intra blocks in an inter frame, and libwebrtc's libaom setup
  (`kf_mode = AOM_KF_DISABLED`).
- Screen content follows SVT-AV1 4.1.0 rtc. `VideoEncoderConfig.Content` and
  the WebRTC `Content` take `EncoderContentAuto` (the new zero value, SVT
  `--scm 2`), `EncoderContentCamera` or `EncoderContentScreen`. Auto runs
  SVT's `svt_aom_is_screen_content` on each keyframe (up to 1080p; inter
  frames inherit the class). Screen pictures use SVT's screen-content ME:
  pre-HME, HME level 2, the larger HME and ME search areas, and the early
  exits, plus the existing screen-content rate control and CDEF levels.
  Headers follow SVT: sequence headers always select screen-content tools
  and integer MV, `allow_screen_content_tools` is set on screen keyframes
  only, and inter frames keep fractional motion. A content change takes
  effect at the next keyframe. `SetScreenContentSelection` is removed.
  Screen keyframes run SVT's luma palette search (palette_level 3), and
  screen inter pictures add SVT's V/H/SMOOTH intra candidates
  (intra_level 6).

### Verification

- Local development gates: `go test ./...`, `make alloc`,
  `make compiler-reports`, and `make trace-zero`.
- Decoder vector gates: `make dryrun-fast`, `make dryrun-full`,
  `make dryrun-extended`, and `make dryrun-profiles`.
- Cross-decoder performance tracking: `make bench-cross` and the optional
  generated corpus lane documented in [README.md](README.md).
