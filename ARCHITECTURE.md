# goav1 Architecture

This document is the contributor-facing map of the goav1 decoder and the
WebRTC encoder implementation scope. It explains the package layout, the
per-frame pipeline a caller drives, the public API surface, the testing
strategy, and the known limitations. Pair it with the per-package `doc.go`
comments and `README.md` for context.

Audience:

- New contributors orienting themselves before touching code.
- Integrators wiring goav1 into a player or RTP receiver who need to know
  who owns what memory.
- Anyone debugging a decoder mismatch and trying to localise the failing
  stage.

The decoder is intentionally built from the transport edge inward. The same
principle drives both the design and the file layout: lower-level packages
own byte framing and primitive math, higher-level packages own AV1 semantics
and orchestration, and the root package exports the caller-facing API only.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Repository Layout](#repository-layout)
3. [Package Dependency Graph](#package-dependency-graph)
4. [Package Reference](#package-reference)
5. [The Per-Frame Pipeline](#the-per-frame-pipeline)
6. [Public API Surface](#public-api-surface)
7. [Memory Ownership Model](#memory-ownership-model)
8. [Threading Model](#threading-model)
9. [Testing Strategy](#testing-strategy)
10. [Build Tags and Environment Variables](#build-tags-and-environment-variables)
11. [Scalable Video Coding (SVC)](#scalable-video-coding-svc)
12. [Known Limitations and Scope](#known-limitations-and-scope)
13. [Upstream References](#upstream-references)
    - [Licensing](#licensing)

---

## Design Principles

These principles drive every decision in the decoder. They are why the API
looks the way it does and why some parts that would normally be hidden are
exported.

- **Pure Go, no CGO.** The decoder is intended to ship in environments
  (WebRTC servers, mobile, sandboxed runtimes) where linking a C codec is
  expensive or impossible. Every path, including the post-filter chain and
  film grain, lives in regular Go.

- **Zero allocations in hot paths.** All scratch buffers, tile job slices,
  CDF tables, prediction edges, and residual scratch are caller-owned. Helpers
  expose `*ScratchSize` types that report the exact backing slice lengths the
  caller must provide; once bound, the helper allocates nothing. This is
  enforced by `BenchmarkPublic*` benchmarks with `-benchmem` and by a small
  set of `make alloc` smoke tests.

- **Explicit memory ownership.** Returned byte slices alias caller-provided
  input or output buffers. There is no hidden copying and there is no
  finalizer-driven lifetime. The package contract is "everything lives in the
  slice you supplied unless we explicitly say otherwise."

- **Deterministic behavior across platforms.** Tile batching, work-stealing,
  and reduction ordering are all deterministic functions of the bitstream
  inputs. The same frame produces the same job partition on every
  architecture. SIMD and assembly backends (future work) must preserve this
  by being bit-exact with the pure-Go reference.

- **Modular package boundaries.** Each package owns one syntactic or
  semantic layer (bitstream framing, OBU parsing, transforms, etc.). Cross-
  package coupling is one-directional: lower layers never import higher
  layers. The `threading` package sits above `tile` and `parser` to compose
  per-batch work; it does not reach upward into `decoder`.

- **Generic Go reference first, SIMD parity second.** Every transform and
  filter ships first as a portable Go implementation with byte-exact
  conformance tests. SIMD/assembly backends, when added, must clear the same
  tests before being wired into the dispatch.

- **Port from pinned upstream.** AV1 syntax, RTP behaviour, decoder
  control flow, and realtime encoder behavior are never implemented from spec
  summaries. `third_party/` pins exact libaom, dav1d, SVT-AV1, and libwebrtc
  commits and `make verify-upstreams` fails if the local clones drift. See
  `UPSTREAM.md`.

---

## Repository Layout

```
goav1/
+-- doc.go, decoder.go, tile.go, transform.go, ...   # root package (re-exports)
+-- example_test.go                                  # canonical caller usage
+-- cmd/                                             # CLI tools
|   +-- aom-go-dec/    IVF -> decoded-frame CLI (residual-stream runner)
|   +-- dump_svc/      SVC layer inspection CLI
+-- README.md                                        # overview and quick start
+-- UPSTREAM.md                                      # pinning policy
+-- ARCHITECTURE.md                                  # this file
+-- Makefile                                         # standard developer targets
+-- internal/av1/                                    # implementation
|   +-- bitstream/      OBU LEB128 and bit reader primitives
|   +-- obu/            OBU framing: header parse, low-overhead/AnnexB/temporal iterators
|   +-- rtp/            AV1 RTP payload format
|   +-- ivf/            IVF test-vector container
|   +-- entropy/        AV1 arithmetic coder + CDF tables and adaptation
|   +-- parser/         Sequence + frame header parsing (all OBU payload syntax)
|   +-- common/         Shared constants and value types
|   +-- tables/         Generated lookup tables
|   +-- memory/         Caller-owned scratch arena helpers
|   +-- frame/          Frame buffer + Pool + sample plane helpers
|   +-- dsp/            Plane block fill/copy/residual-add + blend primitives
|   +-- quantize/       Dequantization, QMatrix tables
|   +-- transform/      Inverse DCT/ADST/IDTX/WHT plus hybrid dispatch
|   +-- reconstruct/    Composes dequant + inverse transform + residual add
|   +-- prediction/     Intra/directional/CFL/filter-intra prediction
|   +-- motion/         Motion vector math, inter prediction, warp filter
|   +-- loopfilter/     Deblocking edge filters and threshold derivation
|   +-- cdef/           CDEF direction search and filter
|   +-- superres/       Super-resolution upscaling
|   +-- restoration/    Wiener + self-guided loop restoration
|   +-- filmgrain/      Film grain synthesis
|   +-- tile/           Tile-level decode: partition, block loop, coeff, transforms, modes
|   +-- work/           Caller-buffer work plans (TilePlan, FramePlan, etc.)
|   +-- threading/      Worker pool, batches, frame-work batch context
|   +-- decoder/        High-level stream + event + frame-work state machine
|   +-- encoder/        WebRTC realtime AV1 encoder target
|   +-- testvector/     Conformance vectors + libaom oracle harness
|   +-- testdata/       Committed libaom IVF test vectors
+-- third_party/        Pinned upstream clones (gitignored, populated by sync-upstreams)
+-- scripts/            Allocation checker + upstream sync/verify scripts
```

The root package contains only re-exports. Every operation a caller invokes
is implemented in `internal/av1/...`; the root files (`decoder.go`,
`tile.go`, `transform.go`, etc.) just promote internal types and functions
to the public namespace.

---

## Package Dependency Graph

Arrows point in the direction of `import`. The graph is layered: any cycle
would be a bug. Within each layer the packages are roughly independent.

```
            +------------------------------------------+
  L9        |             root package goav1           |
            +------------------------------------------+
                              |
            +------------------------------------------+
  L8        |          internal/av1/decoder            |
            +------------------------------------------+
                              |
            +------------------------------------------+
  L7        |    internal/av1/threading + work         |
            +------------------------------------------+
                              |
            +------------------------------------------+
  L6        |             internal/av1/tile            |
            +------------------------------------------+
              /        |       |       |        |     \
  L5        ...  reconstruct  predict  motion  loop   cdef
                              |
                              | predict, motion, loopfilter, cdef,
                              | restoration, superres, filmgrain all
                              | depend on the L4 utilities
            +------------------------------------------+
  L4        |  quantize, transform, frame, dsp, tables |
            +------------------------------------------+
                              |
            +------------------------------------------+
  L3        |   parser  (sequence + frame syntax)      |
            +------------------------------------------+
                              |
            +------------------------------------------+
  L2        |   obu  +  rtp  +  ivf  + entropy         |
            +------------------------------------------+
                              |
            +------------------------------------------+
  L1        |          bitstream (LEB128 + bit reader) |
            +------------------------------------------+
                              |
            +------------------------------------------+
  L0        |          common + memory                 |
            +------------------------------------------+
```

Notes on the layering:

- `entropy` lives at L2 alongside `obu`. It only depends on `common` and the
  shared bit primitives; CDF adaptation is a self-contained state machine
  driven by the tile-level decoder.
- `tile` is the largest package by source count. It depends on every L4
  helper, on `parser` for syntax structures, on `entropy` for the arithmetic
  reader, and on `prediction`/`motion`/`reconstruct` for in-tile
  reconstruction. It does not import `threading` — the relationship goes the
  other way.
- `threading` composes `tile` jobs into per-batch work and binds the
  caller-owned scratch a frame-work batch needs. It does not own the worker
  goroutines' lifetime beyond Close/Acquire/Release semantics on the `Pool`.
- `decoder` is the top of the internal stack. It owns the `Stream` (OBU and
  RTP intake), the `Event` enum, the `FrameWorkState` lifecycle, and the
  per-event run helpers. Everything else is composed from below.

---

## Package Reference

### L0: foundations

- **`internal/av1/common`** — shared constants (max tiles, ref frame counts,
  primary-ref sentinels), compact value types, bit-manipulation helpers.
- **`internal/av1/memory`** — small caller-owned scratch arena helpers used
  by the higher-layer scratch types.

### L1: byte primitives

- **`internal/av1/bitstream`** — AV1 LEB128 encode/decode and the bit reader
  shared across OBU, entropy, and container code. Deliberately tiny and
  allocation-free; higher-level parsers own all state.

### L2: framing

- **`internal/av1/obu`** — OBU header, optional extension header, optional
  OBU size field, and payload slicing. Provides three iterators:
  - `NewLowOverheadIterator` (WebRTC-style, no leading size when not needed)
  - `NewTemporalUnitIterator` (Section 5 `.obu` conformance files)
  - `NewAnnexBIterator` (length-prefixed temporal-unit/frame-unit framing)
  Plus normalization helpers that restore the low-overhead size field
  required by the RTP payload format.

- **`internal/av1/rtp`** — AV1 RTP payload aggregation header, payload
  iteration, packetization (single OBU, fragmented OBU), depacketization
  into complete OBU spans with caller-supplied scratch, and a `Depacketizer`
  state machine that survives packet loss boundaries.

- **`internal/av1/ivf`** — Zero-allocation DKIF/AV01 IVF container reader
  used by the conformance harness and the public example.

- **`internal/av1/entropy`** — AV1 arithmetic coder. Owns the CDF state
  type (`CDFState`), the boolean and multi-symbol readers, signed-delta and
  finite-subexp helpers, and the inverse-CDF init/adapt path. All buffers
  are caller-owned; the reader operates on a slice of payload bytes plus a
  small carry register.

### L3: syntax parsing

- **`internal/av1/parser`** — Every AV1 OBU payload syntax beyond the
  framing layer:
  - `ParseSequenceHeader`, `ParseFrameHeaderPrefix`
  - `ParseFrameSize`, `ParseTileInfo`
  - `ParseQuantizationParams`, `ParseSegmentationParams`,
    `ParseDeltaParams`
  - `ParseLoopFilterParams`, `ParseCDEFParams`, `ParseRestorationParams`
  - `ParseTransformReferenceParams`, `ParseSkipModeParams`,
    `ParseFrameModeParams`, `ParseGlobalMotionParams`, `ParseFilmGrainParams`
  - `ParseTileGroupHeader` and tile-payload span splitting
  - `ReferenceState`: long-lived caller-owned reference-slot record used to
    carry segmentation, loop-filter deltas, global motion and film grain
    forward between frames.

  Parsing is one-shot per OBU; the parser never decodes coefficients. Its
  output is consumed by `decoder.Stream` and packed into `decoder.Event`.

### L4: per-frame helpers

- **`internal/av1/frame`** — `Frame` (plane geometry + sample storage),
  `Format` (sequence-derived plane layout), `Pool` (caller-buffer
  acquire/release of `Frame` slots from a fixed `[]byte` backing store).
  This is the surface store the decoder reads from for inter prediction and
  writes into for reconstruction.

- **`internal/av1/dsp`** — Plane block fill/copy/clipped residual-add and
  the A64 mask blend used by inter prediction. Stable entry points so a
  future SIMD backend can swap implementations behind a dispatch.

- **`internal/av1/quantize`** — Dequantization, QMatrix lookup tables, and
  the `Quantizer`/`Plane` value types consumed by reconstruction.

- **`internal/av1/transform`** — Inverse DCT (sizes 4..64), ADST, flipped
  ADST, IDTX, WHT 4x4, and the hybrid transform dispatch (`DCT_DCT`,
  `ADST_DCT`, ...). Scan order tables and TXB helpers live here.

- **`internal/av1/tables`** — Generated and hand-checked AV1 lookup tables
  (intra mode contexts, partition contexts, etc.).

### L5: per-block helpers

- **`internal/av1/prediction`** — Intra prediction (DC, planar, vertical,
  horizontal, directional, paeth), the filter-intra mode, the CFL
  subsample+predict path, and intra edge filter / upsample. All paths take
  caller-owned edge and scratch buffers.

- **`internal/av1/motion`** — Motion vector arithmetic, fractional-pel
  interpolation (the libaom 8-tap convolve, with smooth/sharp variants), the
  full-pel and nearest-pel paths used in WebRTC realtime profiles, and the
  warped motion filter.

- **`internal/av1/loopfilter`** — Deblocking. Per-level threshold
  derivation and the 4/6/8/14-sample edge filters with the flat/narrow
  fallbacks AV1 mandates. Operates on 8/10/12-bit reconstruction planes.

- **`internal/av1/cdef`** — Constrained Directional Enhancement Filter:
  direction search, primary/secondary tap filter, and per-block filtering.

- **`internal/av1/restoration`** — Wiener and self-guided loop restoration
  for the loop-restoration stage of the post-filter chain.

- **`internal/av1/superres`** — Frame-level horizontal super-resolution
  upscaling between coded and display widths.

- **`internal/av1/filmgrain`** — Film grain synthesis: Gaussian RNG,
  scaling LUTs, luma/chroma grain blocks, per-row application.

- **`internal/av1/reconstruct`** — Composes dequant -> inverse transform ->
  clipped residual add over caller-owned scratch. Reports the maximum
  scratch needed across all transform sizes so the caller can size one
  arena per frame.

### L6: tile-level decode

- **`internal/av1/tile`** — The big one. Roughly fifty files. Includes:
  - `payload.go`, `plan.go` — caller-buffer tile-work planning from a
    tile-group payload; produces `[]Job` and `[]Span` slices.
  - `partition.go`, `partition_walk.go` — partition-tree decode and the
    block-walk visitor.
  - `mode.go`, `intra.go`, `inter.go`, `inter_mode.go`,
    `inter_compound.go`, `inter_filter.go` — per-block mode decode for
    intra, inter, compound, OBMC, inter-intra, filter-intra.
  - `palette.go` — palette mode decoding (color indices, cache).
  - `mv.go`, `ref_mv.go`, `ref_mv_frame.go`, `motion_field.go`,
    `motion_mode.go` — motion vector decode and reference-MV stack.
  - `coeff.go`, `coeff_tree.go`, `coeff_context.go`, `coeff_tables.go`,
    `block_coeff.go` — transform-block coefficient entropy decode.
  - `decode.go`, `decode_test.go` — per-tile decode state and the entropy
    reader binding.
  - `plane_block.go`, `block_loop.go` — block-loop visitor: the controller
    that walks superblocks and emits per-block callbacks for residual
    decode + reconstruction.

- **`internal/av1/work`** — Pure description types: `TilePlan`, `FramePlan`,
  `FrameStep`, `ShowExistingFramePlan`, etc. These are the caller-buffer
  plan values that flow from planning helpers through executors to
  callbacks; they carry no behaviour.

### L7: scheduling

- **`internal/av1/threading`** — Worker pool plus the rich
  `FrameWorkBatch` context that batch callbacks receive. Files:
  - `pool.go` — bounded goroutine `Pool` and `BatchFunc` dispatch.
  - `batch.go` — `Batch` value type and `BuildBatches` deterministic
    partition.
  - `predict.go`, `reconstruct.go`, `tile_residual.go` — per-batch helpers
    that compose `tile` decode with `prediction`, `motion`, and
    `reconstruct` over caller-owned scratch.
  - `pool.go` / `cdef_index.go` / `loop_filter_map.go` /
    `restoration_buffers.go` — frame-level side-data maps shared between
    reconstruction and the post-filter stages.
  - `ref_mv_frame.go`, `wedge.go`, `cdf_reset.go` — supporting state for
    motion field projection and CDF lifecycle.

### L8: high-level driver

- **`internal/av1/decoder`** — The state machine the user actually drives:
  - `Stream` (`stream.go`) — accepts OBUs, low-overhead bytes, or RTP
    payloads via `PushLowOverhead` / `PushOBUInto` / `PushRTPPayload` and
    produces `Event` values. Owns the in-progress `frameState`, the
    sequence header, the `ReferenceState`, and the RTP `Depacketizer`.
  - `Event` (`stream.go`) — a tagged union over OBU classification. Each
    `Event` carries the parsed metadata for one OBU (sequence header,
    frame header, tile group, etc.).
  - `FrameWorkState` (`work.go`) — caller-owned per-frame lifecycle. Hosts
    surface acquisition, tile-work planning, post-filter side data,
    per-frame CDF storage retained for the primary reference frame, and
    the `RunEvent*` family of helpers.
  - `SurfaceReferences` (`surface.go`) — long-lived mapping from the seven
    AV1 reference slots into `frame.Pool` indices. Updated atomically at
    frame finish or show-existing-frame.
  - `Postfilter*` (`postfilter*.go`) — the post-filter pipeline: loop
    filter, CDEF, super-res, loop restoration, film grain. Each stage has
    a `*ScratchSize`/`*Request`/`*Result` triple and a Bind helper that
    slices caller-owned scratch into the per-stage views.

- **`internal/av1/encoder`** — WebRTC realtime AV1 encoder implementation
  target. The control/configuration foundation is landed; bitstream emission is
  the next implementation step. The scope is WebRTC use cases and controls only.
  Correctness/control behavior is ported from pinned libaom/libwebrtc source,
  and speed-sensitive architecture is checked against pinned SVT-AV1 before
  local invention.

### Conformance and oracle

- **`internal/av1/testvector`** — Vector metadata (`Tag`, `Vector`,
  `Manifest`, `Suite`), the libaom remote manifest, the optional `Oracle`
  type (inert in default builds, MD5-comparing under the `goav1_oracle`
  build tag), and the per-frame MD5 digester. The committed libaom IVFs
  live in `internal/av1/testdata/libaom/`.

---

## The Per-Frame Pipeline

This section traces what happens when a caller pushes one frame's worth of
bytes through the decoder. The path is documented in the order events fire.

### Stage 0: Bytes in

The caller has a payload — either a raw OBU stream (WebRTC low-overhead),
an Annex B chunk, a temporal-unit blob from an IVF frame, or one RTP
packet payload. They call into `decoder.Stream`:

```
Stream.PushLowOverhead(src, eventsBuf)
       |
       v
obu.NewLowOverheadIterator(src)
       |
       v
for each unit:
    Stream.PushUnitInto(&event, unit, newCodedVideoSequence=false)
        |
        v
    switch unit.Header.Type {
    case TypeSequenceHeader:    parser.ParseSequenceHeader -> EventSequenceHeader
    case TypeTemporalDelimiter: -> EventTemporalDelimiter
    case TypeFrameHeader:       acceptFrameHeader -> parser.Parse* chain
    case TypeFrame:             ParseFrameHeaderPrefix + Parse* chain
    case TypeTileGroup:         ParseTileGroupHeader -> EventTileGroup
    ...
    }
```

`Stream` owns a small `frameState` between the `EventFrameHeader` and the
final `EventTileGroup` so continuation tile groups can re-use the parsed
frame-level metadata without re-parsing. The state is cleared either when
the final tile group is consumed or when a `NewCodedVideoSequence` /
`TemporalDelimiter` event drops the in-flight frame.

### Stage 1: Plan the frame work

The caller hands each `Event` to a `FrameWorkState`. For an `EventFrame` or
`EventFrameHeader` the helper:

```
RunDecoderFrameWorkEventWithContext(state, refs, pool, seq, event,
                                    align, refSurfaces, refFrames,
                                    workers, spans, jobs, batches,
                                    releases, workerPool, fn)
    |
    +-- planEventWithResolvedContext
    |     |
    |     +-- FrameWorkState.PlanEvent
    |     |     |
    |     |     +-- ShowExisting() if EventExistingFrame
    |     |     +-- AbortIfEventDropsFrameWork() if dropping
    |     |     +-- ResetFrameSurfaceReferences() if new sequence
    |     |     +-- Begin() -> BeginFrameWork():
    |     |             |
    |     |             +-- DecoderBeginFrameSurface  (acquire output surface from pool)
    |     |             +-- PlanDecoderTileWork       (build spans/jobs/batches)
    |     |             +-- record FrameWorkPlan + active=true
    |     |
    |     +-- ResolveFrameReferences (lookup *frame.Frame for each ref slot)
    |
    +-- RunStepWithPayloadContextAndPostFilter
          |
          ... (Stage 2)
```

After `Begin`, the `FrameWorkState` is "active". It holds:
- the acquired output surface index,
- the resolved reference count,
- a `FrameWorkSequenceContext` derived once from the sequence header,
- the initial CDF storage (cloned from the primary reference frame's
  retained CDFs if available, otherwise default-initialised from
  `BaseQIdx`),
- placeholders for the per-frame side-data maps (CDEF index, loop-filter
  record, restoration buffers) that the caller fills before tile work runs.

### Stage 2: Execute tile work

`RunStepWithPayloadContextAndPostFilter` walks the `FrameWorkStep`:

```
executeFrameWorkStepWithPayload(step, workerPool, output, refs, payload, ...,
                                 frameContext, sideData, jobs, batches, fn)
    |
    +-- threading.Pool.Dispatch
          |
          +-- for each Batch b:
                |
                +-- worker goroutine wraps b into a FrameWorkBatch:
                |     - sequence + frame context
                |     - resolved output and reference frames
                |     - per-job payload slice
                |     - per-job entropy reader scratch
                |     - per-job DecodeState scratch
                |     - side-data map handles (CDEF index, LF record, restoration)
                |     - initial CDF storage pointer
                |     - retained CDF storage pointer (for context-update tile)
                |
                +-- invoke fn(FrameWorkBatch)
```

The caller's `FrameWorkBatchFunc` typically calls one of the high-level
"runner" helpers exported under the
`DecoderFrameWorkBatchResidualRunner` family. Those helpers compose:

```
FrameWorkBatch.DecodeAndReconstructJobResiduals(index, state, cdfs, scratch, req)
    |
    +-- tile.BlockLoop walks partitions and superblocks for the job
    |
    +-- for each block:
    |     |
    |     +-- req.Predictor(visit)               (prediction package: intra/inter/CFL)
    |     +-- req.TransformSelector(visit)       (transform package: type/size)
    |     +-- DecodeBlockCoefficients            (tile.coeff + entropy package)
    |     +-- ReconstructBlockCoeff:
    |           - quantize.Dequantize
    |           - transform.Inverse(...)
    |           - dsp.AddResidual to output frame
    |
    +-- update tile-level CDFs (only retained if this is the
        context-update tile; see FrameHeader.ContextUpdateTileIdx)
    |
    +-- update the CDEF / loop-filter / motion-field side-data maps
```

Every batch reads and writes its own slice of `jobs` and its own scratch.
Batches do not share the entropy reader, the CDF storage view, or the
prediction scratch; they share the *output frame*, the *reference frames*,
the side-data maps (each map is partitioned by superblock so writes don't
overlap), and the retained-CDF storage (only the context-update tile
writes it).

### Stage 3: Post-filter

Once the final tile group of the frame is consumed,
`runFrameWorkPostFilter` (or the runner variant) fires the post-filter
callback before the output is published to reference slots:

```
post(FrameWorkPostFilterContext{
    Event, Step,
    Output, ReferenceCount, ExecutedTileWork,
    CDEFIndexMap, LoopFilterMap, RestorationFrameBuffers,
})
    |
    +-- typical implementation calls
    |   BindDecoderFrameWorkPostFilterRequest(...) once, then
    |   request.Apply()
    |
    +-- Apply walks the five stages in spec order:
          1. Loop filter (deblocking)
                - uses LoopFilterMap populated during reconstruction
                - applies 4/6/8/14-tap edge filters in-place on Output
          2. CDEF
                - uses CDEFIndexMap (per-block strength index)
                - direction search + tap filter, in-place
          3. Super-res
                - if FrameSize.SuperResUpscaledWidth != CodedWidth
                - bicubic horizontal upscale into per-plane scratch then
                  back into Output
          4. Loop restoration
                - per-unit Wiener or self-guided projection
                - uses RestorationFrameBuffers (records + stripe boundaries)
          5. Film grain
                - synthesises noise blocks from FilmGrainParams,
                  applies to Output (or detaches to a synth surface)
```

Each stage has caller-owned scratch sized via `*ScratchSize.Max(...)` so
one arena can satisfy all in-flight frames. The Result types report
per-plane statistics for diagnostics.

### Stage 4: Publish references

After the post-filter callback returns without error,
`FinishIfEventCompletesFrameWork` updates `SurfaceReferences` based on the
frame header's `RefreshFrameFlags`. Surfaces that were previously bound to
a slot being overwritten are released into the `releases` scratch
slice and returned to the `FramePool` atomically. The retained CDF
storage is captured into the slot's `tileResidualFrameContexts` so a
future `PrimaryRefFrame` lookup can inherit it.

```
FrameWorkState.Finish(refs, pool, event, releases)
    |
    +-- FinishFrameSurface (mutate refs, collect freed surface indices)
    +-- finishTileResidualCDFs (write retained CDFs into ref slot)
    +-- pool.ReleaseMany(releases)
    +-- state.resetActive()
```

For a show-existing-frame event the path is shorter: the `ShowExisting`
helper finds the referenced surface, returns it as the displayed frame,
aborts any in-flight frame work, and releases any surfaces freed by the
key-frame reference reset (if applicable).

### Stage 5: Output

The output `*Frame` returned in `FrameWorkEventResult.Output` is owned by
the `FramePool` until it is overwritten or explicitly released. The caller
can read its planes via `Frame.Plane*`, copy them out, or pass the surface
index back into a render path that respects the pool's lifetime.

### One-frame call sequence diagram

```
caller             Stream         FrameWorkState   Pool          workerPool        Frame
  |                   |                  |             |               |              |
  |--PushLow*-------->|                  |             |               |              |
  |  bytes            |--parse OBU------>|             |               |              |
  |                   |  (parser.Parse*) |             |               |              |
  |<------- Event ----|                  |             |               |              |
  |                   |                  |             |               |              |
  |--RunEventWithContext.................................>|             |              |
  |                   |                  |--Begin----->|--Acquire---->frame.Frame*    |
  |                   |                  |  surface    |               |              |
  |                   |                  |             |               |              |
  |                   |                  |--PlanTile-->|               |              |
  |                   |                  |  (spans,jobs,batches)       |              |
  |                   |                  |                             |              |
  |                   |                  |--Execute-->                 |              |
  |                   |                  |             FrameWorkBatch--+(fn)          |
  |                   |                  |                             | residual+    |
  |                   |                  |                             | reconstruct  |
  |                   |                  |                             |--write------>|
  |                   |                  |                             |              |
  |                   |                  |--PostFilter------------------------------->|
  |                   |                  |  (loopfilter,CDEF,SR,LR,FG) (in-place)
  |                   |                  |
  |                   |                  |--Finish---->refs            |              |
  |<--EventResult.Output......................................................frame*-|
```

---

## Public API Surface

Every type and function a caller needs lives at the root package:

```go
import av1 "github.com/thesyncim/goav1"
```

The exports group by pipeline stage. Each file in the root package contains
a per-file overview comment.

### Transport (read side)

- IVF: `NewIVFIterator`, `IVFHeader`, `IVFFrame`.
- OBU iteration: `NewLowOverheadIterator`, `NewTemporalUnitIterator`,
  `NewAnnexBIterator`, `ParseLowOverheadOBU`, `ParseOBUHeader`,
  `ParseOBUElement`.
- RTP read: `NewRTPPayloadIterator`, `AssembleRTPFrame`, the
  `Depacketizer` type.
- Normalisation: `PutLowOverheadSize`, `NormalizeOBU*`.

### Transport (write side)

- RTP write: `PutRTPSingleOBUPayload`, `PutRTPFragmentedOBUPayload`,
  caller-owned scratch sizing helpers.

### Syntax

- Sequence / frame: `ParseSequenceHeader`, `ParseFrameHeaderPrefix`,
  `ParseFrameSize`, `ParseTileInfo`, plus the `SequenceHeader`,
  `FrameHeaderPrefix`, `FrameSize`, `TileInfo` types.
- Per-block: `QuantizationParams`, `SegmentationParams`, `DeltaParams`,
  `LoopFilterParams`, `CDEFParams`, `RestorationParams`,
  `TransformReferenceParams`, `SkipModeParams`, `FrameModeParams`,
  `GlobalMotionParams`, `FilmGrainParams`, `TileGroup`, `TileSpan`,
  `ReferenceState`, `ReferenceFrame`.

### Frame storage

- `Frame`, `FramePlane`, `FrameLayout`, `FrameFormat`.
- `FramePool`, `BindFramePool`, `FramePoolFromBacking`.

### Entropy

- `CDFState`, `CDFReader`, `EntropyReader`.
- Adaptation, validation, finite-subexp and signed-delta helpers.

### Decoder driving

- `DecoderStream`, `DecoderEvent`, `DecoderEventKind`.
- `DecoderFrameWorkState`, `DecoderSurfaceReferences`.
- `BeginDecoderFrameWork`, `PlanDecoderTileWork`,
  `PlanDecoderFrameTileWork`, `ExecuteDecoderTileWork`,
  `ExecuteDecoderFrameWorkStep`, `ExecuteDecoderFrameWorkStepWithContext`,
  `ExecuteDecoderFrameWorkStepWithPayload`.
- `RunDecoderFrameWorkEventWithContext`,
  `RunDecoderFrameWorkEventWithContextAndPostFilter`.
- `DecoderEventDropsFrameWork`, `DecoderEventCompletesFrameWork`,
  `DecoderEventOutputsFrame`.
- Surface helpers: `DecoderAcquireFrameSurface`,
  `DecoderBeginFrameSurface`, `ResolveDecoderFrameReferences`,
  `DecoderFinishFrameSurface`, `DecoderShowExistingFrameSurface`.

### Worker pool + tile work

- `NewTileWorkerPool`, `TileWorkerPool`, `TileBatch`, `TileBatchFunc`,
  `TileJob`, `TileSpan`.

### Residual decode / reconstruct

- `DecoderFrameWorkBatch`, `DecoderFrameWorkBatchFunc`.
- `DecoderFrameWorkTileResidualCDFStorage`,
  `DecoderFrameWorkTileResidualScratch`,
  `DecoderFrameWorkTileResidualRequest`,
  `DecoderFrameWorkTileResidualStats`.
- `DecoderFrameWorkBatchResidualRunner`, plus per-job
  `DecodeAndReconstructDecoderFrameWorkJobResiduals` and
  `DecodeAndRetainDecoderFrameWorkJobResiduals`.
- `DecoderFrameWorkBlockCoeffReconstruction`,
  `DecoderFrameWorkBlockTransforms`,
  `DecoderFrameWorkBlockTransformSelector`,
  `DecoderFrameWorkBlockPredictor`.

### Post-filter

- `DecoderFrameWorkPostFilterStage`,
  `DecoderFrameWorkPostFilterRequest`,
  `DecoderFrameWorkPostFilterResult`,
  `DecoderFrameWorkPostFilterScratchSize`,
  `DecoderFrameWorkPostFilterScratch`,
  `BindDecoderFrameWorkPostFilterRequest`.
- Per-stage Bind/Apply triples for loop filter, CDEF, super-res,
  loop restoration, and film grain.
- `DecoderFrameWorkSupportedPostFilterRunner` (built-in chain) and
  `DecoderFrameWorkCallerPostFilterRunner` (custom stages).

### Standalone DSP / reconstruction primitives

- Intra/inter/CFL/OBMC prediction helpers.
- Inverse transform dispatch (`InverseBlock`, scan/scratch helpers).
- Loop filter edge primitives.
- Reconstruction bridge (`ReconstructPlaneBlock`).
- CDEF and loop-restoration primitive helpers.

A minimal read-side smoke path lives in `example_test.go`, runnable with
`go test ./... -run Example`. See also `ExampleParseSequenceHeader` for a
short walk-through that extracts the sequence header from an IVF fixture.

---

## Memory Ownership Model

Every public helper documents whether returned slices alias caller input or
caller scratch. The general rules:

- **Inputs alias.** `ParseLowOverheadOBU(payload)` returns a `Unit` whose
  `Payload` field aliases `payload`. The caller must keep `payload` alive
  as long as the parsed values are used.
- **Outputs go into caller-owned scratch.** Tile work uses
  `spans []TileSpan`, `jobs []TileJob`, `batches []TileBatch`,
  `releases []int` slices the caller pre-allocates. Helpers report the
  required length via `*ScratchSize` or via documented bounds (e.g. tile
  count is bounded by `parser.MaxTiles`).
- **Frame samples live in the pool.** `FramePool` owns the backing
  `[]byte`. `Frame.Plane*` returns a writable window into that storage.
  Frames are valid until `Pool.Release(index)` is called.
- **No finalizers, no goroutine ownership.** The only goroutines created
  by the decoder are the worker goroutines in `TileWorkerPool`, and they
  are bounded by the pool's `workers` parameter and torn down by
  `Pool.Close()`. Frame buffers and side-data maps are released when the
  pool releases them; there is no GC-driven lifetime.
- **CDF storage is per-frame and caller-owned for retention.** The
  `FrameWorkState` keeps one copy per AV1 reference slot so a frame whose
  `PrimaryRefFrame` points at it can seed its CDFs from the retained
  values.

When a helper says "caller-buffer" or "caller-owned" in its godoc, it
means the helper writes into or reads from the caller's slice and never
allocates a replacement.

---

## Threading Model

```
                   +----------------------------+
                   |  caller goroutine          |
                   |                            |
                   |  RunEventWithContext       |
                   +-----------+----------------+
                               |
                               v
                   +----------------------------+
                   |  threading.Pool            |
                   |                            |
                   |  bounded worker goroutines |
                   |  (workers parameter)       |
                   +-----------+----------------+
                               |
                  fan-out one Batch per worker
                               |
            +-------+-------+-------+-------+   ...
            v       v       v       v       v
         worker  worker  worker  worker  worker
         |       |       |       |       |
         |       |       |       |       |  Each worker:
         |       |       |       |       |   * reads a slice of jobs[]
         |       |       |       |       |   * uses its own entropy reader scratch
         |       |       |       |       |   * uses its own residual scratch
         |       |       |       |       |   * writes its own slice of output samples
         |       |       |       |       |   * writes its own slice of side-data maps
         |       |       |       |       |   * atomic adapts CDF only on the
         |       |       |       |       |     context-update tile
         v       v       v       v       v
         +---------------+---------------+
                         v
         +----------------------------+
         |  Caller goroutine joins    |  threading.Pool.Wait blocks for all batches
         +----------------------------+
                         v
         +----------------------------+
         |  Post-filter on the caller |  Loop filter / CDEF / SR / LR / FG run
         |  goroutine                 |  serially after tile work completes
         +----------------------------+
```

Determinism guarantees:

- `BuildBatches` partitions jobs by their `SBCols*SBRows` total so the
  same job list produces the same batch layout on every run.
- Workers never read from another worker's slice of `jobs[]` or
  `batches[]`. CDF retention is the only cross-batch write and is
  restricted to the tile flagged in `FrameHeader.ContextUpdateTileIdx`.
- Side-data maps (CDEF index, loop-filter record, restoration unit
  records) are written by superblock index, which is unique per job; no
  two workers touch the same map cell.
- The post-filter stage runs on the caller goroutine after `Pool.Wait`
  returns. It does not race with reconstruction.

The current pool dispatch is a straight fan-out: one batch per worker,
join, repeat. There is no work-stealing today. The deterministic batch
partitioner already accounts for uneven job sizes by grouping nearby
jobs whose unit counts sum close to the per-batch target.

---

## Testing Strategy

The decoder ships with five orthogonal layers of test coverage. They run
in the standard Makefile targets and the GitHub Actions workflows.

### 1. Unit tests

Every package has `_test.go` files covering its primitives. `make test`
runs `go test ./...`. CI runs this on every push.

### 2. Allocation regression tests

Files matching `*_public_test.go` use `testing.AllocsPerRun` to lock in
the zero-allocation contract of the public API. Examples:

- `postfilter_public_test.go`
- `prediction_public_test.go`
- `decoder_prediction_public_test.go`
- `tile_coeff_tree_public_test.go`
- `frame_public_test.go`
- `loopfilter_public_test.go`
- `tile_geometry_public_test.go`
- `output_filter_public_test.go`
- `motion_public_test.go`
- `reconstruct_public_test.go`

`make alloc` runs `./scripts/check_allocs.sh`, which runs only the
zero-allocation-critical packages. Any regression to allocations on a
hot path will fail this gate.

### 3. Fuzz harnesses

`make fuzz-smoke` runs short-time fuzzing across the byte primitives
(LEB128, OBU header, RTP payload, IVF iterator, AnnexB iterator),
the entropy reader and CDF state, every parser entry, the tile coefficient
decode/reconstruct family, the post-filter primitives, prediction
helpers, and the threading frame-work batch helpers. See the Makefile
for the complete list; the default is 250000 iterations per fuzzer with
8-way parallelism.

These fuzzers catch boundary bugs that would otherwise show up as MD5
mismatches halfway through a long vector.

### 4. Microbenchmarks

`make bench` runs every `Benchmark*` across the tree. `make bench-public`
runs only the public `BenchmarkPublic*` benchmarks at the root package;
these are the ones we treat as the throughput/contract baseline. Use
`-benchmem` to surface any allocation regressions the alloc tests don't
already cover.

### 5. Conformance via the libaom oracle

This is the most important and most subtle layer. The conformance harness
lives in `internal/av1/testvector/`.

The harness has two modes:

- **Default (inert oracle).** With no build tag, `Oracle` is a zero-size
  value and `OracleEnabled` is the constant `false`. Tests that gate on
  `OracleEnabled` compile away to nothing. This keeps `go test ./...`
  cheap and hermetic.

- **`-tags goav1_oracle`.** Activates the byte-comparing oracle. Vector
  bytes are sourced from `internal/av1/testdata/libaom/` (committed) or
  downloaded into a local cache directory when the test selects a
  remote-only vector.

The MD5 comparison itself runs through `Oracle.CheckMD5(tag, frameIndex,
md5)`, which compares against the per-frame digests parsed from the
libaom `.md5` companion files.

The framework dry-run test (`TestLibaomFastFrameWorkDryRun`) walks the
fast slice of the libaom remote manifest, drives every frame through
`Stream.PushLowOverhead -> FrameWorkState.RunEventWith*`, and checks the
MD5 of each decoded surface against the libaom reference. Two MD5
verification modes:

- **Lenient (default).** Only frame 0's MD5 must match the official
  digest. Subsequent frames log a per-frame match/mismatch line. This is
  the mode that `make testvectors-fast` / `make dryrun-fast` and CI use:
  it lets us track per-vector progress without blocking on the last
  unfixed feature.
- **Strict (`GOAV1_STRICT_MD5=1`).** Every frame's MD5 must match. Any
  mismatch fails. Use for diagnostic snapshots when surfaces are
  passing frame 0 by accident.

Each subtest logs a trailing summary line of the form:

```
vector=NAME frames=N md5_matches=M first_mismatch=F
```

`first_mismatch` is `-1` when every frame matched.

The committed fixture in `internal/av1/testdata/libaom/` is
`av1-1-b8-00-quantizer-00.ivf`, which `make testvectors-fast` runs
through the full pipeline. The committed `TestLibaomQuantizer00` test
catches any regression on that vector under the lenient mode; the
`TestLibaomQuantizer00FrameWorkDryRun` test catches regressions on the
single vector that currently passes strict mode.

`make testvectors` runs both the unbuilt and `goav1_oracle` builds.
`make testvectors-full` (gated by `GOAV1_FULL_LIBAOM_VECTORS=1`)
downloads and runs the full libaom manifest; not part of CI.

### 6. Conformance for specific primitives

Two targets exist for primitive-level conformance against libaom's
reference code:

- `make test-motion-conformance` — exercises the full libaom convolve
  conformance set when `GOAV1_FULL_LIBAOM_CONVOLVE=1` is set.
- `make test-transform-conformance` — exercises libaom-derived inverse
  DCT/ADST/WHT reference matchers.

These are useful when touching the motion or transform packages to
confirm bit-exact parity with libaom before chasing higher-level
mismatches.

---

## Build Tags and Environment Variables

| Tag/Variable                       | Effect                                                                                                    |
|------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `goav1_oracle` (build tag)         | Compiles in the byte-comparing oracle and remote vector helpers. Default off.                             |
| `goav1_scaled_pred` (build tag)    | Pins the SVC scaled-inter-prediction dispatcher live for the entire process. Compile-time twin of `GOAV1_SCALED_PRED=1`. Default off (same-size reference path stays bit-exact). |
| `GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1` | Enables `TestLibaomFastFrameWorkDryRun` (fast subset of libaom vectors through the full pipeline).     |
| `GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1` | Enables `TestLibaomExtendedFrameWorkDryRun` (10-bit q-sweep, larger sizes, SVC L2T1/L2T2; opt-in, never CI). |
| `GOAV1_SCALED_PRED=1`              | Runtime opt-in for the SVC scaled-inter-prediction path. Without it, mismatched-size references fail with `threading: invalid batch`. Required for L2T1 / L2T2 multi-resolution SVC. |
| `GOAV1_STRICT_MD5=1`               | Requires every frame's MD5 to match the libaom reference in the framework dry-run. Default is lenient.    |
| `GOAV1_FULL_LIBAOM_VECTORS=1`      | Allows `TestLibaomRemoteSuiteFullDownloads` to download and run the entire libaom remote manifest.        |
| `GOAV1_FULL_LIBAOM_CONVOLVE=1`     | Enables the full libaom convolve conformance sweep in `make test-motion-conformance`.                     |

The `Oracle` type and `OracleEnabled` constant collapse to no-ops in
default builds. Code paths that gate on `OracleEnabled` are eliminated by
the compiler in the absence of the `goav1_oracle` tag, so they cost
nothing in production builds.

---

## Scalable Video Coding (SVC)

The decoder parses every AV1 SVC bitstream feature and routes spatial /
temporal layer information through to callers, but the end-to-end
multi-spatial decode path is still in active development. This section
is the architectural summary; see [docs/svc.md](docs/svc.md) for the
integrator-facing guide and `cmd/dump_svc` for an executable example.

### How SVC is signalled

Two pieces of bitstream syntax carry layer information:

- **OBU extension header.** A 1-byte optional extension after the OBU
  header encodes `temporal_id` (3 bits) and `spatial_id` (2 bits) for
  the payload. Parsed into `obu.Header.TemporalID` /
  `obu.Header.SpatialID` and propagated to every `DecoderEvent` as
  `event.TemporalID` / `event.SpatialID`.
- **Sequence-header operating points.**
  `SequenceHeader.OperatingPoints[]` lists up to 32 operating-point
  descriptors. Each `OperatingPoint.IDC` is a 16-bit
  `operating_point_idc` whose low 8 bits select temporal layers and
  bits 8..11 select spatial layers. The optional scalability metadata
  OBU (`MetadataTypeScalability`, parsed into
  `Metadata.Scalability`) carries longer-form per-layer description.

### How SVC threads through the pipeline

1. **Stream layer.** `decoder.Stream` parses every OBU it sees,
   independent of layer. The (T, S) pair lands on every emitted
   `Event` so callers can drop or route by layer.
2. **Per-spatial-layer state.** Each spatial layer needs its own
   `DecoderFrameWorkState` because frame-level CDF retention is
   per-layer. Per-layer states co-exist behind a single
   `DecoderSurfaceReferences` (the seven AV1 reference slots are
   stream-global per the spec).
3. **Multi-pool surface routing.** Different spatial layers may have
   different `CodedWidth x Height` (L2T1: 640x360 + 1280x720).
   `frame.LayerPool` (root: `FrameLayerPool`) aggregates per-format
   `FramePool` instances behind a single global-surface-ID namespace.
4. **Cross-pool reference resolution.** When a layer-1 frame
   references a layer-0 surface, the lookup goes through a
   caller-supplied `DecoderFrameSurfaceProvider` (and a matching
   `DecoderFrameSurfaceReleaser` for refresh-time deallocation). The
   root `NewDecoderFrameLayerPool` adapter satisfies both interfaces
   directly when using `FrameLayerPool`.
5. **Scaled inter prediction.** When an enhancement-layer frame
   references a smaller base-layer surface, the decoder must apply
   the AV1 8-tap convolve at the appropriate scale factor
   (`internal/av1/motion/scaled.go`). The threading dispatcher gates
   the scaled path behind the `goav1_scaled_pred` build tag and the
   `GOAV1_SCALED_PRED=1` runtime environment variable; both routes
   exist so the path can be enabled without recompilation while still
   letting developer workflows pin it on.

### Where the code lives

- `internal/av1/decoder/svc.go` — `FrameSurfaceProvider`,
  `FrameSurfaceReleaser`, `TemporalMotionReferenceProvider`,
  `RunEventWithContextAndExternalReferences`, and tile-list external
  anchor-frame resolution. The root package exposes these through
  `DecoderFrameWorkExternalReferenceRuntime`, `DecoderFrameSurfaceProvider`,
  `DecoderFrameSurfaceReleaser`, `DecoderTemporalMotionReferenceProvider`,
  and the resolver helpers.
- `internal/av1/decoder/svc_layer_pool.go` — `FrameLayerPool`
  adapter that satisfies the provider/releaser pair against a
  `*frame.LayerPool`; root callers use `NewDecoderFrameLayerPool`,
  `DecoderAcquireLayerFrameSurface`, and
  `DecoderLayerPoolGlobalSurfaceID`.
- `internal/av1/frame/layer_pool.go` — `LayerPool` aggregator and
  `LayerFactory` interface. Exported at root as `FrameLayerPool` and
  `FrameLayerFactory`.
- `internal/av1/motion/scaled.go` — scaled 8-tap convolve and
  `NewScaleFactors` (libaom-derived `av1_setup_scale_factors_for_frame`
  parity).
- `internal/av1/threading/predict_scaled.go`,
  `predict_scaled_tag.go`, `predict_scaled_notag.go` — the
  dispatcher gate that decides when to route through the scaled
  convolver.
- `internal/av1/testvector/libaom_oracle_test.go` —
  `libaomSpatialLayers` is the reference SVC harness, the canonical
  implementation to copy when wiring multi-pool SVC into production
  code today.

### Limitations today

- SVC decode is covered by the framework/oracle path. L1T2, L2T1,
  and L2T2 libaom vectors pass the strict gates listed in
  CONFORMANCE.md, including multi-pool surface routing and scaled
  inter prediction.
- Public SVC ergonomics are still lower-level than the simple
  single-stream decoder path. Integrations that need explicit
  spatial-layer selection should follow `docs/svc.md` and the
  `cmd/dump_svc` / testvector harness patterns.
- Tile-list playback is wired for frame-context-carrying events. Tile-list
  OBUs parse and surface as events, anchors resolve through reference slots or
  a provider-backed surface namespace, raw entry payloads plan as single-tile
  residual jobs, entries reconstruct through the residual runner, and decoded
  rectangles blit into a published mosaic output. Context-less tile-list OBUs
  still fail early because no tile grid is available to validate or decode
  against.

---

## Known Limitations and Scope

### Current state (as of the README)

`make testvectors-fast` and `make dryrun-fast` are green in CI on every push.
The dry-run lane uses strict per-frame MD5 and covers the committed fast
libaom vectors, including the current L1T2 SVC fast vector.

The broader local gates in CONFORMANCE.md currently pass:

- `make dryrun-relevant-supported`
- `make dryrun-full`
- `make dryrun-extended`
- `make dryrun-profiles`

### Feature coverage status

For the full spec-level inventory (per-mode intra prediction, per-mode inter
prediction, per-transform type, per-restoration type, OBU type coverage,
container coverage, plus the libaom fast-suite vector pass/fail table), see
[CONFORMANCE.md](CONFORMANCE.md). The summary below is the high-level
roll-up.

| Area                                  | State                                            |
|---------------------------------------|--------------------------------------------------|
| OBU framing (low-overhead, AnnexB, TU)| Complete, zero-allocation, fuzzed.               |
| RTP payload format                    | Complete, fuzzed, includes assembler.            |
| Sequence + frame header parsing       | Complete (all parameter blocks).                 |
| Tile-group parsing + work planning    | Complete.                                        |
| Entropy decoder + CDF adapt           | Complete, fuzzed.                                |
| Intra prediction (DC, planar, dir)    | Complete; clip-to-frame-edge wired.              |
| Filter-intra and CFL                  | Complete with caller-owned scratch.              |
| Inter prediction (translational)      | Full + sub-pel + warped + OBMC + inter-intra.    |
| Compound and wedge prediction         | Complete with mask blends.                       |
| Motion field projection (MFMV)        | Complete; tracked via fast dry-run.              |
| Transforms (DCT, ADST, IDTX, WHT)     | Complete pure-Go; hybrid dispatch validated.     |
| Dequantization + QMatrix              | Complete.                                        |
| Block-level reconstruction            | Complete; zero-alloc residual + add.             |
| Loop filter (4/6/8/14-sample)         | Complete with flat + narrow fallbacks.           |
| CDEF                                  | Direction search + filter complete.              |
| Super-res                             | Complete.                                        |
| Loop restoration (Wiener + SGR-proj)  | Complete primitives + frame plan + apply.        |
| Film grain                            | Complete synthesis + per-row apply.              |
| Show-existing-frame                   | Complete (lifecycle, reference reset, release).  |
| `show_frame=0` / non-displayable      | Supported via reference slot tracking.           |
| Annex B / IVF / RTP intake            | Complete, including high-level `NewDecoderFromRTPPayloads` / `NewDecoderFromRTPPackets` plus live `DecodeRTPPayloadAfterLoss` / `DecodeRTPPacketAfterLoss` retained-fragment reset for AV1 RTP payload bodies and complete packets. |
| WebRTC signaling helpers              | Complete for AV1/90000 SDP/fmtp profile/level/tier checks, RTP fixed-header packet parse/build, RFC 8285 one-/two-byte header-extension element parse/build, RTP header-extension mapping checks, RTP MID/RID/RRID SDES payload helpers, AV1 RID receiver restrictions, AV1 simulcast RID groups, AV1 rtcp-fb checks, generic and compound RTCP packet parsing, RTCP SR/RR/SDES/BYE and RTPFB/PSFB helpers, NACK/Transport-CC/PLI/FIR/REMB helpers, AV1 Layer Refresh Request FCI entry/list parse/build/validation, and force-key classification. |
| SVC streams                           | Parsed and decoded through the framework path; L1T2/L2T1/L2T2 strict-MD5 gates pass with multi-pool surface routing and scaled inter prediction. See [docs/svc.md](docs/svc.md). |
| Realtime pixel encoder                | Functional for 8-bit profile-0 WebRTC streams from I420/I422/I444/I400/NV12/NV21 plus generic 8/10/12-bit `Frame` inputs adapted into the current 4:2:0 encode path, including fixed-quality/CBR, forced keyframes, temporal layering, runtime bitrate/framerate/rate-control/scalability reconfiguration, multi-spatial `RTCEncoder.EncodePicture` for W3C SVC and simulcast modes, tile columns, golden references, RTP payload packetization, sized complete RTP packet wrapping, dependency descriptors, active decode target signaling, and LRR layer-grid validation. |
| WebRTC encoder control/metadata       | W3C AV1 SVC mode vocabulary, temporal/spatial dependency structures, full decode-target grids, W3C key-shift temporal schedules, pinned-libwebrtc L2T2_KEY_SHIFT dependency templates, explicit sequence color config, exact RTP frame-duration helper, RTP packet spans and sized complete RTP packet wrapping for caller-supplied frame payloads, and sequence-matched `Frame` validation/loading for profile-0/1/2 8/10/12-bit 4:0:0, 4:2:0, 4:2:2, and 4:4:4 sample formats. |

### Not yet implemented

- **Native high-bit-depth and non-4:2:0 bitstream encoding.** The friendly
  realtime pixel encoder accepts broader caller inputs through adapters, but
  those inputs still enter the current 8-bit profile-0 4:2:0 encode path.
  Native 10/12-bit and true 4:2:2/4:4:4 bitstream emission remains open.
- **Full WebRTC media transport.** The package emits AV1 RTP payload bodies,
  dependency descriptors, sized complete RTP packet wrappers with fixed headers,
  RFC 8285 extension elements, raw
  MID/RID/RRID SDES payload helpers, and focused AV1
  SDP/fmtp/extmap/RID/simulcast plus RTCP LRR helper surfaces. SRTP, full SDP
  assembly, jitter buffering, loss policy, pacing, and retransmission remain
  integration-layer responsibilities.
- **Work-stealing scheduling.** The current `threading.Pool` does a
  deterministic fan-out per batch; there is no dynamic stealing across
  workers mid-frame. Determinism plus the batch unit-count balancing
  has been sufficient for realtime targets so far.

### Open Work

1. **Broaden the WebRTC realtime encoder.** Add native high-bit-depth and
   true non-4:2:0 bitstream emission, richer tuning controls, broader
   libaom/libwebrtc/SVT oracle coverage, and measured compression-efficiency
   tuning.
2. **Broaden decoder coverage.** Keep expanding profile-2, 12-bit,
   malformed/adversarial, fuzz, and real-world corpus coverage beyond the
   committed vector gates.
3. **Add SIMD backends.** Wire amd64 and arm64 kernels for the hot DSP entries:
   inverse transforms, deblocking edges, CDEF, restoration, and convolve
   filters. Each backend must clear `make alloc`,
   `make test-motion-conformance`, and `make test-transform-conformance`
   before dispatch.

---

## Upstream References

The decoder is ported from pinned upstream sources, never from spec
summaries. See `UPSTREAM.md` for the verification policy and
`third_party/upstream.lock` for the exact pinned references. Highlights:

- **dav1d 1.5.3** — decoder architecture, OBU parsing, tile/decode
  pipeline, DSP layout.
- **libaom v3.14.0** — reference AV1 bitstream behavior and realtime
  encoder behavior.
- **SVT-AV1 v4.1.0** — production realtime encoder architecture,
  threading, mode decision, and rate-control reference.
- **libwebrtc (branch-heads/7848)** — AV1 RTP payload/depayload behavior and
  WebRTC encoder control integration.
- **AV1 RTP spec v1.0.0** (`av1-rtp-spec`) — normative RTP payload
  format.
- **AV1 Bitstream & Decoding Process Specification** — normative
  bitstream and decoding process.

When adding behavior:

1. Inspect the relevant pinned source file.
2. Add or extend byte-level tests before broadening the implementation.
3. Cite the upstream path in the Go test name, comment, or commit
   message when useful.
4. Keep ports C-readable: flat structs, fixed arrays, explicit state,
   no reflection, no hidden allocation, no clever iterator abstraction
   in hot loops, and matching C integer widths/signedness in new or touched
   parity paths.

`make sync-upstreams` performs shallow, sparse clones of the pinned
upstreams under `third_party/upstream/` (gitignored).
`make verify-upstreams` checks each local clone is at the pinned commit.

### Licensing

goav1 is distributed under the BSD 2-Clause license recorded in the
root `LICENSE` file. The license text matches the upstream libaom
`LICENSE` (preserved verbatim at
`third_party/upstream/libaom/LICENSE`) so that derivative work from
libaom carries forward under identical terms.

The companion `PATENTS` file at the repository root is a byte-for-byte
copy of the Alliance for Open Media Patent License 1.0 published with
libaom (`third_party/upstream/libaom/PATENTS`). Section 1.2.1 of that
license requires the grant to be reproduced "in the root directory of
the source code with its Implementation" for source-form distribution;
keeping `LICENSE` and `PATENTS` together at the module root satisfies
that condition. Distributors who repackage goav1 in binary form must
include both files in their documentation, legal notices, or
equivalent materials per section 1.2.1(b).

Third-party attributions for libaom, dav1d, libwebrtc, the AV1
bitstream and RTP specifications, and the bundled libaom test vectors
are collected in the root `NOTICE` file. Each entry records the
upstream URL, the pinned reference, the upstream license, and a
verbatim copy of the upstream copyright notice.

Files in this repository that are direct ports of libaom (or dav1d /
libwebrtc) C code should cite their upstream source path with a
header comment of the form

```go
// Ported from libaom: av1/common/<file>.c
```

near the top of the file or function. This makes the derivation
visible during code review, keeps the BSD attribution requirement
satisfied at the source level, and helps diff against the upstream
when chasing bit-exact regressions. Numerical tables reproduced
verbatim (default CDFs, dequantization scales, transform scan orders,
qmatrix levels) carry the same citation in the table file header.

The reference checkouts under `third_party/upstream/<name>/` are
ignored by git but always re-materialised by `make sync-upstreams`;
they exist purely to make the upstream license text and source layout
available locally. They MUST NOT be edited.

---

## Onboarding Checklist

For a new contributor wiring something into goav1:

1. Read `README.md` and this `ARCHITECTURE.md` end to end.
2. Run `make test` to confirm the tree builds and the unit tests pass.
3. Run `make testvectors-fast` to confirm the committed conformance
   slice is green.
4. Run `make dryrun-fast` to see the current libaom fast-slice
   pass/fail summary.
5. Look at `example_test.go` for the minimal read-side caller pattern.
6. Look at `internal/av1/testvector/libaom_oracle_test.go` for the
   full caller pattern (intake, plan, run, post-filter, finish).
7. Pick a package, read its `doc.go`, scan its `*_test.go` files; the
   test names are the closest thing the project has to a user manual.

For an integrator:

1. Wire `Stream.PushLowOverhead` (or `PushRTPPayload`) into your bytes
   source. If you already have an ordered batch of AV1 RTP payload bodies for a
   single decode chain or independent simulcast layer, `NewDecoderFromRTPPayloads`
   wraps this runner setup for you; the same decoder can consume live payloads
   through `DecodeRTPPayload` and `DecodeRTPPayloadAfterLoss`.
2. Allocate the long-lived storage: `DecoderFrameWorkState`,
   `DecoderSurfaceReferences`, `FramePool`, `TileWorkerPool`, and the
   per-event scratch slices (`events`, `referenceSurfaces`,
   `referenceFrames`, `spans`, `jobs`, `batches`, `releases`).
3. Drive each `Event` through
   `RunDecoderFrameWorkEventWithContextAndPostFilter` with your batch
   callback (usually a wrapper around
   `FrameWorkBatch.DecodeAndReconstructJobResiduals`) and your
   post-filter callback (usually
   `BindDecoderFrameWorkPostFilterRequest` plus `request.Apply()`).
4. Inspect `FrameWorkEventResult.Output` when
   `DecoderEventOutputsFrame(event)` is true.

For a debugger chasing a bit-exact mismatch:

1. Reproduce under `GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1` with
   `GOAV1_STRICT_MD5=1` to fail fast.
2. Look at the trailing `vector=... first_mismatch=...` line to learn
   which frame to inspect.
3. Use the per-package `debug_*_dump_test.go` files (gitignored, not
   committed) as inspiration for adding scoped dumps in the failing
   stage.
4. Cross-reference the corresponding libaom path in
   `third_party/upstream/libaom` after a `make sync-upstreams`.
