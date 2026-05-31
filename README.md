# goav1

[![ci](https://github.com/thesyncim/goav1/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/ci.yml)
[![lint](https://github.com/thesyncim/goav1/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/lint.yml)
[![testvectors](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml)

Pure Go AV1 implementation focused on realtime/WebRTC decoding first.

For an end-to-end map of the decoder - package layout, per-frame pipeline,
public API surface, threading model, testing strategy, and known limitations
- see [ARCHITECTURE.md](ARCHITECTURE.md). New contributors and integrators
should start there. For the spec-level feature inventory (which AV1 syntax,
prediction, transform, and post-filter rows are implemented, partial, or
missing) and the libaom fast-suite vector pass/fail table, see
[CONFORMANCE.md](CONFORMANCE.md). For an integrator-facing guide to AV1
scalable video coding (SVC) with goav1 - operating points, OBU extension
headers, the multi-pool surface routing pattern, and the
`GOAV1_SCALED_PRED=1` runtime flag - see [docs/svc.md](docs/svc.md). For
upstream pinning policy, see [UPSTREAM.md](UPSTREAM.md). For the threat
model, reporting process, and supported versions, see
[SECURITY.md](SECURITY.md).

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
- Public zero-allocation AV1 IVF test-vector container parsing for DKIF/AV01
  streams.
- Strict OBU header parsing and low-overhead OBU iteration.
- Public zero-allocation Section 5 temporal-unit iteration for `.obu`
  conformance streams.
- Public zero-allocation Annex B temporal/frame/OBU-unit iteration for
  length-prefixed AV1 streams.
- WebRTC-compatible OBU normalization that restores low-overhead size fields.
- AV1 RTP aggregation header parsing, payload iteration, payload building,
  single-OBU fragmentation, and fragment reassembly.
- WebRTC-compatible low-overhead OBU RTP packetization with caller-owned
  scratch, public two-pass scratch sizing, and exact next-packet output sizing.
- Caller-buffer AV1 RTP depacketization into complete OBU spans with
  allocation-free size preflight.
- WebRTC-compatible RTP frame assembly that restores low-overhead OBU size
  fields from fragmented packet payloads with caller-owned scratch and public
  output-size preflight.
- AV1 entropy inverse-CDF initialization, validation, adaptation,
  caller-owned CDF state wrappers, and allocation-free range reading for tile
  symbol, signed-delta, uniform, and finite-subexp decoding.
- AV1 sequence header parsing for decoder configuration.
- libaom-derived low-overhead and Annex B sequence-header OBU vectors for
  parser/oracle parity checks.
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
  clipped reconstruction regions, writable output-plane windows, resolved
  reference-frame and reference-plane lookup, job-aligned reference windows
  with clipped interpolation margins, block-delta context,
  segmentation/filter setup, motion/film-grain context, and frame
  quantizer/delta decode-state access inside reconstruction callbacks.
- Decoder tile-work planning from parsed frame/tile-group events into
  caller-owned spans, jobs, and batches, including checked frame-work begin
  and tile-group continuation plans, bounded tile-work step execution, and
  payload-carrying frame-work batch callbacks and event-level run helpers, and
  caller-owned frame work lifecycle state with event-level orchestration,
  ordered run/postfilter/finish helpers for final tile groups, abort release,
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
- Public caller-buffer frame plane layout, pool backing sizing, binding
  primitives, and sample-plane load/store helpers for postfilter scratch.
- Caller-owned deterministic frame pools for reusable and retained decode
  surfaces, including format-checked acquire and atomic batch release.
- Public zero-allocation DSP plane-block fill/copy/residual-add, A64 mask
  blending, and 8x8 min/max absolute-difference helpers.
- Pure-Go DSP plane-block fill, copy, and clipped residual-add primitives for
  reconstruction reference paths.
- Pure-Go intra plane prediction for DC, vertical, and horizontal
  reconstruction blocks with caller-owned edge buffers.
- Public zero-allocation intra, directional intra, filter-intra edge, and CFL
  prediction helpers for caller-owned reconstruction paths.
- Pure-Go full-pixel motion-vector helpers and inter plane prediction for
  nearest/reference-copy reconstruction paths.
- Public zero-allocation motion-vector, reference-origin, and inter-prediction
  helpers for full-pixel and fractional translational prediction.
- Public decoder frame-work prediction bridge for block, luma, intra, inter,
  CfL, OBMC, and compound prediction with caller-owned scratch.
- Public AV1 transform scan/scratch helpers and dequantization lookup tables
  with caller-buffer coefficient dequantization for residual reconstruction.
- Pure-Go inverse identity transform foundation with AV1 transform-size shifts
  and caller-buffer residual output.
- Pure-Go 4x4 and 8x8 inverse DCT residual output with caller-owned transform
  scratch.
- Zero-allocation inverse transform dispatch for DCT_DCT and IDTX residual
  paths with per-transform scratch sizing.
- Public zero-allocation residual reconstruction bridge for dequantization,
  inverse transform, clipped plane residual application, and max scratch sizing.
- Zero-allocation AV1 loop-filter level, delta, segmentation, and threshold
  derivation for post-reconstruction deblocking setup.
- Pure-Go narrow loop-filter edge deblocking for 8/10/12-bit reconstruction
  planes.
- Pure-Go six-sample loop-filter edge deblocking with flat filtering and
  narrow-filter fallback for 8/10/12-bit reconstruction planes.
- Pure-Go eight-sample loop-filter edge deblocking with AV1's luma flat filter
  and narrow-filter fallback for 8/10/12-bit reconstruction planes.
- Pure-Go fourteen-sample wide loop-filter edge deblocking with AV1's wide flat
  filter and eight-/four-sample fallback for 8/10/12-bit reconstruction planes.
- Public zero-allocation loop-filter level, threshold, direct-edge, and
  block-edge application helpers for deblocking paths.
- Public zero-allocation CDEF and loop-restoration primitive helpers with
  caller-owned scratch for post-reconstruction filtering paths.
- Public zero-allocation tile block, partition, transform-size, and extended
  transform-set geometry helpers for caller-owned tile decode planning.
- Public caller-owned transform context marking and luma transform-block replay
  helpers for tile coefficient/reconstruction planning.
- Public caller-owned coefficient entropy context helpers for TXB skip and
  DC-sign context replay during tile coefficient decode.
- Public zero-allocation tile block coefficient decode helper that combines
  transform-tree decode, luma/chroma TXB decode, and visitor replay.
- Public zero-allocation luma/chroma tile coefficient-tree decode helpers with
  caller-owned CDF, entropy-context, and TXB scratch storage.
- Public zero-allocation decoder postfilter request binding for loop filter,
  CDEF, superres, loop restoration, film grain, and loop-filter/CDEF side-map
  storage, aggregate request binding, flat scratch arena binding, decoder-level
  loop-restoration frame buffers, frame-work side-data arena binding,
  postfilter capability gates, header-derived scratch contexts, runner scratch
  sizing, dynamic supported-runner scratch binding, dynamic caller-owned scratch
  binding, and reusable max scratch sizing.
- Public decoder side-map reset/mark helpers for passing tile residual CDEF
  indices and loop-filter metadata into postfilter planning.
- Public decoder reference-MV and temporal motion-field storage binding for
  caller-owned inter-prediction side data.
- Public decoder tile residual CDF, decode-state, and block-loop request
  lifecycle helpers for caller-owned entropy adaptation.
- Public zero-allocation decoder frame-work block coefficient position,
  segment-aware quantizer, and reconstruction helpers for placing decoded TXBs
  into caller-owned output surfaces.
- Public zero-allocation luma/chroma coefficient replay adapters that feed
  decoded tile TXB blocks directly into decoder frame-work reconstruction.
- Public zero-allocation one-block coefficient decode/reconstruct helper for
  custom block-loop callers that want an entropy-to-output-surface step.
- Public fuzz-smoke coverage for exported coefficient decode, one-block
  decode/reconstruct, job residual decode/reconstruct, batch residual
  decode/retain, residual-runner side-data attachment, residual event-runner
  execution, supported postfilter scratch-runner integration, and frame-work
  side-data binding entry points.
- Public zero-allocation decoder tile residual decode/reconstruct bridge with
  caller-owned prediction and residual scratch, intra/inter transform dispatch,
  default frame-work transform selection, max residual scratch sizing, aggregate
  batch residual scratch sizing/binding, per-worker residual runner binding,
  default event-runner prediction scratch, residual-runner side-data attachment,
  event-level residual-runner scratch sizing, aggregate event scratch sizing,
  reusable max scratch sizing, event-list scratch sizing and execution,
  low-overhead OBU/RTP stream-runner scratch sizing, binding, single-payload
  and multi-payload RTP scratch sizing and execution, event side-data and
  composite event runner binding, reusable event runner state, direct postfilter
  runner dispatch, caller-owned completed-output sizing and collection,
  caller-owned aggregate residual stats, caller-owned superres display output,
  and execution, batch loop-context sizing, plus job and batch helpers that
  compose decode-state setup, residual decode/reconstruct, and CDF retention.
- Public tile-level loop-restoration frame planning, caller-owned
  record/boundary binding, boundary extension, and unit/frame apply helpers.
- Public zero-allocation superres upscaling and film-grain RNG, scaling,
  synthesis, sampling, and row-application helpers for output filtering paths.
- Optional IVF/OBU/RTP/test-vector oracle plumbing behind the `goav1_oracle`
  build tag, with a zero-size disabled oracle for ordinary builds and an
  ignored local cache for official libaom vectors downloaded on demand.
- Zero-allocation block-edge loop-filter resolution and application helpers
  that combine tile delta-lf state, segmentation, thresholds, and deblocking.
- AV1 header-derived frame formats, including monochrome surface layout.
- Allocation regression tests for the critical byte-level paths.
- Public hot-path benchmarks for OBU iteration, RTP packetization/assembly,
  frame sample round-trips, tile coefficient replay, tile block coefficient
  decode, decoder prediction, decoder block coefficient reconstruction,
  coefficient replay reconstruction, one-block coefficient decode/reconstruct,
  reconstruction/deblocking primitives, and output filters.

Public APIs live at the module root:

```go
import av1 "github.com/thesyncim/goav1"
```

The root package re-exports the long-lived caller-facing types and helpers in
groups that follow the decode pipeline:

- IVF/OBU/RTP transport: `NewIVFIterator`, `NewLowOverheadIterator`,
  `NewTemporalUnitIterator`, `NewAnnexBIterator`, `NewRTPPayloadIterator`,
  `AssembleRTPFrame`, plus the matching `Parse*` and `Put*` helpers.
- Sequence and frame headers: `SequenceHeader`, `FrameHeaderPrefix`,
  `FrameSize`, `TileInfo`, and the per-syntax parameter blocks
  (`QuantizationParams`, `LoopFilterParams`, `CDEFParams`,
  `RestorationParams`, `FilmGrainParams`, etc.).
- Decoder driving: `DecoderStream`, `DecoderEvent`, `DecoderFrameWorkState`,
  `DecoderSurfaceReferences`, `FramePool`, and the `BeginDecoderFrameWork` /
  `RunDecoderFrameWorkEventWithContext` lifecycle helpers.
- Worker pool and tile work: `NewTileWorkerPool`, `TileBatch`, `TileJob`,
  `PlanDecoderTileWork`, `ExecuteDecoderFrameWorkStep`.
- Residual decode + reconstruct: the
  `DecoderFrameWorkBatchResidualRunner` and the per-job
  `DecodeAndReconstructDecoderFrameWorkJobResiduals` /
  `DecodeAndRetainDecoderFrameWorkJobResiduals` helpers.
- Post-filter pipeline: `BindDecoderFrameWorkPostFilterRequest`
  composes loop-filter, CDEF, super-res, loop-restoration, and film-grain
  stages over caller-owned scratch.

See the executable
[Example](https://pkg.go.dev/github.com/thesyncim/goav1#example-package)
in `example_test.go` for the minimum read-side smoke path.

## CLI tool

The repository ships an end-to-end command-line decoder under
[`cmd/aom-go-dec`](cmd/aom-go-dec) that drives the same public residual stream
runner used by `BenchmarkDecodeFullVector`. It is the shortest path from "I
have an IVF" to "I have raw YUV bytes" and serves as executable documentation
for callers wiring goav1 into their own pipeline.

### Build and install

```sh
make build-cmd          # writes ./bin/aom-go-dec
make install-cmd        # go install into $GOBIN / $GOPATH/bin
```

`make build-cmd` accepts `CMDBIN=/path/to/binary` to override the output
path. The CLI is a single Go binary with no CGO and no external dependencies
beyond the standard library and goav1 itself.

### Usage

```
aom-go-dec [-o out.yuv] [-workers N] [-quiet] input.ivf
```

Decode the bundled libaom quantizer-00 vector into a raw I420 file:

```sh
./bin/aom-go-dec \
    -o /tmp/quantizer00.yuv \
    internal/av1/testdata/libaom/av1-1-b8-00-quantizer-00.ivf
```

`-o` defaults to standard output so the binary composes naturally with
`ffplay`, `ffmpeg`, or any other YUV-aware tool:

```sh
./bin/aom-go-dec internal/av1/testdata/libaom/av1-1-b8-00-quantizer-00.ivf \
    | ffplay -f rawvideo -pixel_format yuv420p -video_size 352x288 -framerate 30 -
```

Per-frame timing and aggregate throughput are logged on standard error so
the YUV stream on standard output stays clean. Use `-quiet` to suppress the
per-frame log line and `-workers N` to grow the tile worker goroutine pool.

10/12-bit streams are written as little-endian 16-bit samples per the
canonical `yuv420p10le` / `yuv420p12le` layout. Monochrome streams produce
I400 (Y plane only). The CLI exits with status 1 on any decode error and
prints the error text to standard error.

The CLI uses the high-level public decoder path and writes frames after the
normal loop-filter, CDEF, super-res, loop-restoration, and film-grain
post-filter chain has run. `cmd/aom-go-dec/main_test.go` keeps that boundary
covered with per-frame MD5 goldens for CDEF/restoration, film grain, and
inter super-res clips. Throughput benchmarks may still isolate lower-level
residual or post-filter stages when measuring specific hot paths.

A second example program under
[`cmd/dump_svc`](cmd/dump_svc) targets AV1 scalable video coding
streams: it inspects the parsed sequence header's operating-points
list, logs the per-(temporal, spatial) OBU fan-out, and dumps the
highest-spatial-layer YUV per temporal unit (libaom
`output_all_layers=0` behavior). The CLI auto-installs
`GOAV1_SCALED_PRED=1` into the process environment so SVC streams
take the scaled-inter prediction path by default; pass
`-scaled-pred=false` to disable. It also supports pinning a
specific spatial layer through `-spatial-layer=N` and selecting the
output sample format with `-format=i420|yuv420p10le` (or the
`-i420` / `-yuv420p10le` shorthand aliases); format mismatches
against the bitstream bit depth are rejected before any YUV bytes
are written. Use `-svc-info` to print the SVC structure without
decoding pixels:

```sh
go build -o ./bin/dump_svc ./cmd/dump_svc
./bin/dump_svc -svc-info \
    internal/av1/testdata/libaom/av1-1-b8-22-svc-L2T2.ivf
```

See [docs/svc.md](docs/svc.md) for the integrator-facing SVC guide
and the current support snapshot. SVC decode is in active
development; the L1T2 vector currently passes frame 0 lenient MD5
but mismatches later frames, and L2T1 / L2T2 require the multi-pool
surface routing path described in `docs/svc.md`. When a decode
error stops mid-stream the CLI exits non-zero and annotates the
failing payload with its parsed (TemporalID, SpatialID) layer
composition plus the proximate decoder error message.

## Status

**All 14 libaom `SuiteLevelFast` vectors PASS** strict per-frame MD5
through the framework dry-run. The broader committed libaom remote
manifest is also green (`make dryrun-full`: 240/240), and the vendored
profile corpus is green (`make dryrun-profiles`: 30/30), including
profile-1 4:4:4 8/10-bit all-intra, inter, palette, CDEF/restoration,
odd edge-size CDEF/restoration, edge-motion, film grain, all-key superres,
inter superres, profile-2 4:2:2 8/10-bit, and profile-2 4:2:0 12-bit
edge-size, film-grain, and superres clips.

Realtime decode is in active development. The supporting primitives -
transport, OBU/sequence/frame-header parsing, tile work scheduling, residual
decode, reconstruction, and the post-filter pipeline - all have public APIs
and unit-test coverage.

- `make testvectors-fast` exercises the committed test-vector suite and the
  oracle-tagged libaom frame-MD5 checks. It runs in CI on every push and is
  expected to stay green.
- `make dryrun-fast` runs strict per-frame MD5 against the 14
  `SuiteLevelFast` libaom vectors.
- `make testvectors-full` will download and execute the full libaom remote
  suite. It is not part of the default CI gate and is intended for local
  parity sweeps.
- `make dryrun-extended` opts into the `SuiteLevelExtended` cohort:
  226 strict-MD5 vectors covering 10-bit quantizer sweeps, odd and
  larger frame sizes, and the remaining libaom SVC permutations (L2T1,
  L2T2). This target is local diagnostic coverage, not part of default CI.

The encoder, SIMD acceleration, and platform-specific assembly backends are
not implemented yet.

## References

- [AV1 Bitstream & Decoding Process Specification](https://aomediacodec.github.io/av1-spec/av1-spec.pdf)
- [RTP Payload Format For AV1](https://aomediacodec.github.io/av1-rtp-spec/)
- [libaom AV1 test vectors](https://storage.googleapis.com/aom-test-data) -
  the upstream conformance corpus; the `make testvectors-fast` slice is a
  curated subset that ships with the repository (see
  `internal/av1/testdata/libaom`).
- dav1d architecture and performance practices.
- libaom realtime encoder behavior.
- libwebrtc AV1 RTP behavior.

## Performance

The repository ships an end-to-end decoder throughput benchmark and a wide
catalogue of per-stage micro-benchmarks. Both classes are pure Go testing
benchmarks driven through the public API, so they reflect what production
callers will measure when they wire the decoder into their own code.

### Quick start

```sh
make bench                 # end-to-end frames/sec + MB/sec on the bundled libaom IVF
make bench-all             # full micro-benchmark sweep across every package
make bench-public          # public-API hot-path micro-benchmarks
```

`make bench` runs the top-level decoder benchmarks defined in
`bench_test.go`:

- `BenchmarkDecodeFullVector` decodes every frame of the bundled
  `internal/av1/testdata/libaom/av1-1-b8-00-quantizer-00.ivf` vector through
  the public residual stream runner, reporting ns/op, MB/sec of decoded
  bitstream, `frames/op`, and `frames/s`.
- `BenchmarkDecodePostFilteredProfileClip` uses the high-level decoder on a
  Profile 1 CDEF/restoration clip, so the normal post-filtered output path is
  represented in the default sweep.
- `BenchmarkDecodeSuperResInterProfileClip` uses the high-level decoder on a
  Profile 1 super-res inter clip, covering the external output-pool reference
  publication path.
- `BenchmarkDecodeFirstFrameOnly` isolates the keyframe decode latency so
  the first-frame cost is visible separately from the steady-state per-frame
  number.
- `BenchmarkDecodeFullVectorAllocs` is a zero-allocation guardrail: it
  decodes the same vector and asserts that the steady-state hot path
  allocates zero bytes per iteration. New contributors should re-run it
  after any decoder change.

Example output on an Apple M4 Max with Go 1.26:

```
BenchmarkDecodeFullVector-16                   34   33393048 ns/op   3.86 MB/s   2.000 frames/op   59.89 frames/s     8542 B/op   0 allocs/op
BenchmarkDecodePostFilteredProfileClip-16       3  100165639 ns/op   0.23 MB/s   4.000 frames/op   39.93 frames/s       64 B/op   4 allocs/op
BenchmarkDecodeSuperResInterProfileClip-16      3   44682528 ns/op   0.15 MB/s   8.000 frames/op  179.10 frames/s      128 B/op   8 allocs/op
BenchmarkDecodeFullVectorAllocs-16             36   33660895 ns/op                                                 0 B/op   0 allocs/op
BenchmarkDecodeFirstFrameOnly-16               72   16451093 ns/op   4.53 MB/s                                      3464 B/op   0 allocs/op
```

The `MB/s` column is computed from the IVF bitstream byte count; the
`frames/s` metric is added via `b.ReportMetric` and is the most useful
single number for capacity planning.

### Post-filter performance

The frame-level post-filter chain (loop filter -> CDEF -> loop
restoration) dominates the cost of any frame that signals all three
stages. The benchmarks in
`internal/av1/decoder/postfilter_chain_bench_test.go` measure each
stage in isolation and as a chain on a shared synthetic 128x128 4:2:0
8-bit frame so the per-stage cost is directly comparable:

```sh
go test -bench='^BenchmarkApply(LoopFilter|CDEF|LoopRestoration|FullPostFilter)$' \
        -benchtime=5s -run=^$ ./internal/av1/decoder
```

Example output on an Apple M4 Max with Go 1.23 (lower is better; the
`MB/s` denominator is the decoded pixel byte count Y+U+V):

| Stage              | ns/op     | ms/frame | MB/s  | Share of full chain |
|--------------------|-----------|----------|-------|---------------------|
| Loop filter        |   727 570 |   0.728  | 33.78 |                ~29% |
| CDEF               | 1 480 415 |   1.480  | 16.60 |                ~59% |
| Loop restoration   |   341 944 |   0.342  | 71.87 |                ~14% |
| Full chain (LF+CDEF+LR) | 2 497 145 |   2.497  |  9.84 |              100% |

The shares slightly exceed 100% because each stage benchmark warms its
own caches in isolation; the chained benchmark amortises some of that
cost across all three stages.

Notes:
- The frame size is intentionally bounded to 128x128 luma because the
  loop-filter benchmark fixture packs every 16x16 block into a single
  super-block mode context (32 4x4-slot limit). Throughput scales
  approximately linearly with frame area, so the MB/s column is the
  best cross-size comparison metric.
- All scratch buffers are bound once during fixture setup; the
  per-iteration hot path stays at `0 B/op`, matching the rest of the
  decoder.
- Film grain and super-res are not in the supported frame-pool
  publication chain on this fixture, so the "Full chain" benchmark
  reflects the steady-state cost of the three stages that are always
  applied together on the publishable surface.

### Per-stage micro-benchmarks

The most expensive decode stages each have benchmarks in their own
packages. Run them individually when profiling a regression:

```sh
go test -bench=BenchmarkInverseDCTBlock        -benchmem ./internal/av1/transform
go test -bench=BenchmarkFilterFrameBlocks      -benchmem ./internal/av1/cdef
go test -bench=BenchmarkApplyWienerRestoration -benchmem ./internal/av1/restoration
go test -bench=BenchmarkApplySelfguidedRestoration -benchmem ./internal/av1/restoration
go test -bench=BenchmarkPredictInterPlaneBlock -benchmem ./internal/av1/motion
go test -bench=BenchmarkPublic                 -benchmem .
```

`make bench-all` chains all of these together. For a meaningful
benchmark run, pass `-benchtime=3s` (or larger) so each benchmark has
time to amortise its bound scratch and reach steady state. The default
`-benchtime=1s` is enough for smoke-testing.

## Development

```sh
make test                 # unit tests across all packages
make testvectors          # committed test-vector suite (with oracle)
make testvectors-fast     # fast slice including oracle MD5 checks
make dryrun-fast          # in-progress framework dry-run against fast vectors
make dryrun-extended      # opt-in extended cohort (10-bit q-sweep, larger sizes, extra SVC)
make bench                # end-to-end frames/sec + MB/sec
make bench-all            # full microbenchmark sweep across every package
make bench-public         # public-API benchmarks
make alloc                # zero-allocation regression coverage
make fuzz-smoke           # short fuzz harness sweep
make ci-local             # fmt-check + vet + test + alloc
```

CI is expected to fail on allocation regressions covered by unit tests. As the
decoder grows, benchmark thresholds should move into generated history files or
a dedicated comparison step.

## Changelog

The session-by-session change history lives in
[CHANGELOG.md](CHANGELOG.md) in keep-a-changelog format. It groups
work into correctness fixes against the libaom conformance vectors,
public API and CLI additions, performance work, hardening / fuzz
coverage, and CI infrastructure, and includes the current
`SuiteLevelFast` test-vector pass/fail table.

Pre-release notes for the upcoming `v0.1.0` tag live in
[RELEASE_NOTES_v0.1.0.md](RELEASE_NOTES_v0.1.0.md).

## License

goav1 is distributed under the BSD 2-Clause license. See [LICENSE](LICENSE)
for the full text.

Portions of this repository are derived from libaom, the AV1 reference
implementation released by the Alliance for Open Media. Those portions are
redistributed under the same BSD 2-Clause terms, and the Alliance for Open
Media Patent License 1.0 that accompanies libaom is reproduced verbatim in
[PATENTS](PATENTS), as required by section 1.2.1 of that grant. Callers and
redistributors must keep both LICENSE and PATENTS together with any
source-form distribution of goav1.

A complete list of upstream attributions covering libaom, dav1d, libwebrtc,
the AV1 bitstream and RTP specifications, and the bundled libaom test
vectors lives in [NOTICE](NOTICE). The pinned upstream commits and the
working-tree paths where the upstream LICENSE / PATENTS / COPYING files are
preserved are tracked in [third_party/upstream.lock](third_party/upstream.lock)
and produced on demand by `make sync-upstreams`.

## Acknowledgements

This project would not exist without the work of the AV1 community.

- The **Alliance for Open Media** and the contributors to **libaom** —
  the AV1 reference encoder/decoder at
  https://aomedia.googlesource.com/aom — whose bitstream syntax,
  reconstruction logic, default CDFs, dequantization scales, transform
  scan tables, loop-filter / CDEF / restoration / film-grain code, and
  conformance test vectors are the substrate on which goav1 is built.
  The pinned reference is `v3.14.0`.
- The **VideoLAN dav1d authors** — https://code.videolan.org/videolan/dav1d
  — whose decoder architecture, OBU parsing layout, tile/decode
  pipeline shape, and DSP organisation informed the package layout of
  this repository. The pinned reference is `1.5.3`.
- The **WebRTC project** at https://webrtc.googlesource.com/src for
  the AV1 RTP payload and depayload behavior that drives the
  realtime-focused RTP code in this repository.

All bugs in the Go port are ours; all good ideas above the bitstream
edge belong to the upstream authors named above.
