# goav1 v0.1.0 (pre-release notes)

`v0.1.0` has not been tagged. These notes define the intended first
pre-release shape without duplicating the detailed conformance tables in
[CONFORMANCE.md](CONFORMANCE.md) or the package map in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Scope

- Pure-Go AV1 decoder and realtime bitstream toolkit.
- No cgo dependency.
- Caller-owned buffers and reusable decoder-owned arenas for steady decode.
- Byte-exact decoder gates against the committed libaom vectors and vendored
  profile corpus.
- CLI decode to raw YUV through `cmd/aom-go-dec`.

## Encoder

Encoder implementation is in project scope. The first encoder target is
deliberately narrow and real: a WebRTC-focused realtime AV1 encoder, not an
offline/general-purpose encoder. It should expose the controls WebRTC callers
need: bitrate, framerate, resolution, keyframe requests, temporal/spatial layer
configuration, SVC, camera/screen-content tuning, realtime speed/quality
controls, low-overhead OBU/RTP output, and dependency/scalability metadata.

Encoder behavior must be ported from pinned libaom/libwebrtc source and proved
with oracle tests before it is presented as supported.

## Verification

Before tagging, release evidence must include:

- commit SHA, Go version, platform, and upstream pins
- `go test ./...`
- `make alloc`
- `make compiler-reports`
- `make trace-zero`
- decoder vector gate summaries from [CONFORMANCE.md](CONFORMANCE.md)
- cross-decoder benchmark output when local reference decoders are available
- module inventory and license/notice review

## Known Gaps

- Tile-list OBU playback is parsed but not reconstructed/published end to end.
- Switch-frame parsing has targeted tests, but the pinned libaom vector set does
  not ship a dedicated `S_FRAME` IVF with MD5 goldens.
- Broader profile-2, 12-bit, malformed-stream, fuzz, real-world corpus, and
  SIMD coverage should keep expanding after the first tag.

## Licensing Reminder

goav1 is distributed under BSD 2-Clause. Portions are derived from libaom; the
Alliance for Open Media Patent License 1.0 is reproduced in [PATENTS](PATENTS)
and must be redistributed with source-form distributions. Upstream attribution
lives in [NOTICE](NOTICE).
