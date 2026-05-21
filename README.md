# goav1

Pure Go AV1 implementation focused on realtime/WebRTC decoding first.

This repository is intentionally built from the transport and bitstream edge
inward:

1. AV1 OBU syntax and LEB128 primitives.
2. AV1 RTP payload packetization and depacketization.
3. Incremental decode pipeline, frame pools, entropy, tiles, and DSP.
4. Realtime encoder foundation.
5. Architecture-specific SIMD/assembly acceleration.

## Principles

- Pure Go. No CGO wrappers around C codecs.
- Zero allocations in hot paths.
- Explicit memory ownership and caller-provided scratch buffers.
- Deterministic behavior across supported platforms.
- Modular package boundaries for bitstream, RTP, decode logic, memory, DSP, and
  future assembly backends.
- Generic Go reference implementations first, SIMD parity second.

## Current Safe Point

The initial implementation covers:

- AV1 LEB128 read/write helpers.
- Strict OBU header parsing and low-overhead OBU iteration.
- WebRTC-compatible OBU normalization that restores low-overhead size fields.
- AV1 RTP aggregation header parsing, payload iteration, payload building,
  single-OBU fragmentation, and fragment reassembly.
- WebRTC-compatible low-overhead OBU RTP packetization with caller-owned
  scratch.
- Caller-buffer AV1 RTP depacketization into complete OBU spans.
- WebRTC-compatible RTP frame assembly that restores low-overhead OBU size
  fields from fragmented packet payloads with caller-owned scratch.
- AV1 entropy inverse-CDF initialization, validation, adaptation,
  caller-owned CDF state wrappers, and allocation-free range reading for tile
  symbol, signed-delta, uniform, and finite-subexp decoding.
- AV1 sequence header parsing for decoder configuration.
- AV1 uncompressed frame-header prefix parsing for low-latency decoder routing.
- AV1 key/intra frame-size parsing for frame-pool sizing and render metadata.
- AV1 inter-frame reference routing and reference-size parsing with explicit
  caller-owned reference state.
- AV1 post-size motion controls and tile layout parsing with fixed arrays for
  decoder tile scheduling.
- AV1 quantization parameter parsing, including signed deltas and qmatrix
  levels.
- AV1 segmentation parameter parsing with explicit previous-state ownership and
  lossless/q-index derivation.
- AV1 delta-q and delta-loopfilter flag parsing after segmentation.
- AV1 loopfilter parameter parsing with explicit previous delta ownership.
- AV1 CDEF parameter parsing for frame-level deringing setup.
- AV1 loop-restoration parameter parsing with restoration type and unit-size
  derivation.
- AV1 transform-mode and reference-mode parsing after restoration.
- AV1 skip-mode eligibility/ref selection and frame-level enable parsing.
- AV1 frame-level warped-motion and reduced transform-set flag parsing.
- AV1 global-motion type and matrix parameter parsing with reference-state
  carryover.
- AV1 film-grain parameter parsing with fixed storage and reference-state
  carryover.
- AV1 tile-group header parsing and caller-buffer tile payload span splitting.
- Caller-buffer tile work-plan construction for deterministic decode scheduling.
- Zero-allocation tile-job payload range validation and slicing helpers.
- Zero-allocation tile entropy reader setup from scheduled tile jobs, including
  frame-level CDF-update control propagation into frame-work callbacks.
- Per-job AV1 context-update-tile marking so decode workers can retain only
  the designated tile's adapted frame entropy context.
- Caller-owned tile decode state that binds a scheduled job to its entropy
  reader, block delta-q/loopfilter state, and frame-context retention decision
  without hot-path allocation.
- Deterministic tile-job batch planning for bounded worker execution.
- Reusable bounded worker-pool dispatch for tile batches.
- Zero-allocation frame-work batch helpers for safe per-job tile payload,
  entropy-reader, sequence/superblock geometry, frame geometry,
  clipped reconstruction regions, writable output-plane windows, block-delta
  context, segmentation/filter setup, motion/film-grain context, and frame
  quantizer/delta decode-state access inside reconstruction callbacks.
- Decoder tile-work planning from parsed frame/tile-group events into
  caller-owned spans, jobs, and batches, including checked frame-work begin
  and tile-group continuation plans, bounded tile-work step execution, and
  payload-carrying frame-work batch callbacks and event-level run helpers, and
  caller-owned frame work lifecycle state with event-level orchestration,
  ordered run-and-finish helpers for final tile groups, abort release,
  show-existing output, and stream-boundary drop handling for temporal units,
  new sequences, and show-existing frames.
- Incremental decoder stream state over OBUs/RTP payloads.
- AV1 show-existing-frame decoder events with reference-state validation.
- Decoder reference-state carryover for segmentation data and loop-filter
  deltas.
- Self-contained tile-group decoder events with active frame-header context.
- Caller-buffer decoder surface-reference tracking for AV1 refresh and
  show-existing-frame updates, including event-aware frame completion and
  inter-reference lookup and reference-frame pointer resolution helpers with
  checked frame begin and atomic frame-pool release.
- Caller-buffer frame plane layout and binding primitives.
- Caller-owned deterministic frame pools for reusable and retained decode
  surfaces, including format-checked acquire and atomic batch release.
- AV1 header-derived frame formats, including monochrome surface layout.
- Allocation regression tests for the critical byte-level paths.

Public APIs live at the module root:

```go
import "github.com/thesyncim/goav1"
```

## References

- [AV1 Bitstream & Decoding Process Specification](https://aomediacodec.github.io/av1-spec/av1-spec.pdf)
- [RTP Payload Format For AV1](https://aomediacodec.github.io/av1-rtp-spec/)
- dav1d architecture and performance practices.
- libaom realtime encoder behavior.
- libwebrtc AV1 RTP behavior.

## Development

```sh
make test
make bench
make fuzz-smoke
```

CI is expected to fail on allocation regressions covered by unit tests. As the
decoder grows, benchmark thresholds should move into generated history files or
a dedicated comparison step.
