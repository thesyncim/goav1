# Scalable Video Coding (SVC) with goav1

This document is the integrator-facing guide to decoding AV1 SVC streams
with the goav1 pure-Go decoder. It explains what SVC is, how AV1 carries
spatial and temporal layers, how to drive an SVC bitstream through the
goav1 public API, which integration patterns are supported today, and
what is still in development.

Pair it with [ARCHITECTURE.md](../ARCHITECTURE.md) for the broader
package map, [CONFORMANCE.md](../CONFORMANCE.md) for the feature
inventory, and the executable example at
[`cmd/dump_svc`](../cmd/dump_svc) for a working SVC driver that exits
with the selected spatial layer.

---

## Table of Contents

1. [What SVC is](#what-svc-is)
2. [How AV1 signals layers](#how-av1-signals-layers)
3. [The simple path: same-size spatial layers](#the-simple-path-same-size-spatial-layers)
4. [The full path: multi-pool, mixed-resolution SVC](#the-full-path-multi-pool-mixed-resolution-svc)
5. [The `DecoderFrameSurfaceProvider` / `DecoderFrameSurfaceReleaser` pattern](#the-decoderframesurfaceprovider--decoderframesurfacereleaser-pattern)
6. [Scaled inter prediction](#scaled-inter-prediction)
7. [Picking which layer to emit](#picking-which-layer-to-emit)
8. [Current support status](#current-support-status)
9. [Conformance vectors](#conformance-vectors)
10. [Driving SVC end to end: walkthrough](#driving-svc-end-to-end-walkthrough)

---

## What SVC is

Scalable Video Coding lets a single bitstream carry several encoded
representations of the same source — at different resolutions
(*spatial layers*), at different framerates (*temporal layers*), or
both. A subscriber chooses the layers it wants and decodes only those,
dropping packets that belong to higher layers it cannot afford.

Two scalability axes show up in AV1:

- **Spatial layers** widen and tall the picture. A typical L2T1
  configuration encodes layer 0 at, say, 640x360 and layer 1 at
  1280x720; layer 1 frames reference both their own previous frames
  and the corresponding base-layer surface at the smaller resolution.
- **Temporal layers** thin the frame rate. An L1T2 configuration
  encodes alternating T0/T1 frames; a receiver that only decodes T0
  sees half the framerate at the same picture size.

The combinations the encoder declared are listed in the AV1 sequence
header as *operating points*; a receiver typically picks one
operating-point IDC and decodes only the OBUs whose
(`temporal_id`, `spatial_id`) pair is selected by that IDC's bitmask.

WebRTC's AV1 RTP payload format propagates the same (T, S) pair on
every packet via the AV1 aggregation header descriptors, so realtime
SVC selection happens at the SFU long before any OBU reaches the
decoder.

---

## How AV1 signals layers

Two pieces of bitstream syntax matter for SVC integration: the OBU
extension header and the sequence-header operating-points list.

### OBU extension header

Every OBU optionally carries a 1-byte *extension header* after the
1-byte OBU header. When the extension flag is set, the extension byte
encodes:

- `temporal_id` (3 bits) — temporal layer this OBU belongs to.
- `spatial_id` (2 bits) — spatial layer this OBU belongs to.

goav1 surfaces these on every parsed OBU as fields on `obu.Header`:

```go
type Header struct {
    Type         Type
    Extension    bool
    HasSizeField bool
    TemporalID   uint8 // 0..7
    SpatialID    uint8 // 0..3
}
```

When `Extension == false`, both IDs default to 0 (the base layer). The
decoder's stream layer also threads the parsed (T, S) pair through to
every emitted `DecoderEvent`:

```go
type DecoderEvent struct {
    Kind       DecoderEventKind
    TemporalID uint8
    SpatialID  uint8
    // ... parsed OBU payload, frame header, tile group, etc.
}
```

This means a caller walking events does not need to peek at the OBU
header — `event.SpatialID` and `event.TemporalID` are authoritative
for the current frame in flight.

### Sequence-header operating points

`SequenceHeader.OperatingPoints[]` carries up to 32 operating-point
descriptors. Each descriptor exposes:

- `IDC` (16 bits) — `operating_point_idc`. Bit `i` (0..7) selects
  temporal layer `i`; bit `8+j` (j=0..3) selects spatial layer `j`.
  An IDC of 0 means *all layers* (the default single-layer profile).
- `SeqLevelIdx`, `SeqTier` — the AV1 level/tier that operating point
  was constrained to.
- Decoder-model and initial-display-delay parameters.

For example, an L2T2 stream declares four operating points:

```
operating_point[0] idc=0x0303 temporal_mask=0x03 spatial_mask=0x03   # all layers
operating_point[1] idc=0x0301 temporal_mask=0x01 spatial_mask=0x03   # spatial only
operating_point[2] idc=0x0103 temporal_mask=0x03 spatial_mask=0x01   # temporal only
operating_point[3] idc=0x0101 temporal_mask=0x01 spatial_mask=0x01   # base layer only
```

The receiver pins an IDC and skips OBUs whose
(`temporal_id`, `spatial_id`) pair is masked out. goav1 itself does
*not* drop OBUs by IDC — the stream layer parses everything it sees
and emits events for every layer; selection is a caller decision.

### Scalability metadata OBU

The optional `OBU_METADATA` payload of type
`METADATA_TYPE_SCALABILITY` carries a longer-form description of the
layer structure (per-layer description, refresh dependencies, picture
group). goav1 parses this through `ParseMetadata` and exposes it as
`Metadata.Scalability` of type `MetadataScalability` (root-package
type alias). It is informational only; the decoder does not consume
it.

---

## The simple path: same-size spatial layers

The shortest end-to-end pipeline through the public API uses a single
`FramePool` to back every output surface. This works for any SVC
stream whose spatial layers all share one `CodedWidth x Height`:

- L1T1 / L1T2 / L1T3 — pure temporal scalability at one resolution.
- "Quality" SVC where every spatial layer is the same size and only
  differs in quantizer or filter parameters.

The driving code is identical to the non-SVC fast path:

```go
runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(
    plan, &stream,
    av1.DecoderFrameWorkResidualEventRuntime{
        State:             &state,
        Refs:              &refs,
        FramePool:         &pool,   // one pool for every layer
        Align:             64,
        ReferenceSurfaces: refSurfaces,
        ReferenceFrames:   refFrames,
        Releases:          releases,
        WorkerPool:        workerPool,
        SideData:          &sideData,
        Stats:             &stats,
    }, scratch, &batchRunner)

for _, payload := range payloads {
    var result av1.DecoderFrameWorkResidualStreamResult
    if err := runner.RunLowOverheadInto(&result, payload, nil); err != nil {
        return err
    }
    for _, frame := range result.Run.Outputs {
        // frame.SpatialID / frame.TemporalID is reflected by the
        // corresponding event's IDs; result.Run.Outputs is emitted in
        // event order so the last non-nil entry is the highest-layer
        // surface published in this payload.
    }
}
```

`cmd/dump_svc/main.go` is exactly this pattern — see the walkthrough
below.

The single-pool path's hard limit: it can only host one
`FrameFormat`. If layer 0 is 640x360 and layer 1 is 1280x720, the
`AcquireFormat` call inside `BeginDecoderFrameWork` will reject the
layer-1 size and decoding fails. You need the multi-pool path for
that.

---

## The full path: multi-pool, mixed-resolution SVC

True spatial SVC (L2T1, L2T2, L3T1, ...) needs one frame pool per
spatial layer plus a way to *route* AV1 reference-slot lookups across
those pools. The seven AV1 reference slots are stream-global (per the
spec), so an enhancement-layer frame might reference a base-layer
surface that lives in a different pool than the layer-1 pool the
output is being acquired from.

goav1 ships two pieces of glue for this:

1. **`FrameLayerPool`** — a caller-owned aggregator over per-format
   `FramePool` instances. Sub-pools are bound lazily on first
   observation of a given `FrameFormat`. Surfaces have global IDs
   that encode `(subPoolIndex, localIndex)`, so a single integer can
   be exchanged across layers.

   ```go
   pools   := make([]av1.FramePool, 4)        // up to 4 spatial layers
   formats := make([]av1.FrameFormat, 4)
   bound   := make([]bool, 4)
   layerPool, err := av1.BindFrameLayerPool(pools, formats, bound, 256, factory)
   ```

   The `FrameLayerFactory` is invoked once per new format and is
   where the caller allocates the per-layer backing slice
   (`make([]byte, backingSize)`) and calls `BindFramePool`.

2. **`DecoderFrameSurfaceProvider` / `DecoderFrameSurfaceReleaser`**
   (and `DecoderTemporalMotionReferenceProvider`) — pluggable
   interfaces used by `DecoderFrameWorkExternalReferenceRuntime` to
   resolve reference surfaces against whichever pool actually holds
   them.

   The pattern is small enough to reproduce in full:

   ```go
   type DecoderFrameSurfaceProvider interface {
       FrameSurface(id int) (*av1.Frame, error)
   }

   type DecoderFrameSurfaceReleaser interface {
       ReleaseFrameSurfaces(ids []int) error
   }
   ```

   For callers using `FrameLayerPool`, the root package provides the
   adapter directly. `currentSubPool` is the `*FramePool` selected for
   the event's coded `FrameFormat`.

   ```go
   layerRefs := av1.NewDecoderFrameLayerPool(&layerPool)
   runtime.External = av1.DecoderFrameWorkExternalReferenceRuntime{
       Provider:      layerRefs,
       Releaser:      layerRefs,
       GlobalSurface: func(local int) int {
           return av1.DecoderLayerPoolGlobalSurfaceID(&layerPool, currentSubPool, local)
       },
       FrameContexts: &sharedFrameContexts,
   }
   ```

---

## The `DecoderFrameSurfaceProvider` / `DecoderFrameSurfaceReleaser` pattern

A working SVC provider needs to do four things:

1. **Mint a "global surface ID" namespace.** Each per-layer
   `FramePool` returns local pool indices starting at 0; those
   indices collide across pools. Use `FrameLayerPool`'s built-in
   `EncodeGlobalID(subPool, local)` to get a single int that encodes
   `(layer, local)`. The dry-run harness uses a simpler hand-rolled
   encoding (`spatialID*256 + local`) for the same purpose.

2. **Store global IDs in `DecoderSurfaceReferences`.** The
   per-frame `DecoderFrameWorkExternalReferenceRuntime.GlobalSurface`
   translator converts the output surface's local index (returned by
   the per-layer pool) into the global ID that gets refreshed into the
   reference slots. When the current coded layer comes from a
   `FrameLayerPool`, use
   `DecoderLayerPoolGlobalSurfaceID(&layerPool, currentSubPool, local)`.

3. **Implement `FrameSurface(id)` to route the lookup.** Decode the
   global ID back into `(layer, local)` and call the right per-layer
   pool's `Frame(local)` accessor:

   ```go
   func (s *Layers) FrameSurface(id int) (*av1.Frame, error) {
       layer, local := decodeGlobalID(id)
       return s.byID[layer].pool.Frame(local)
   }
   ```

4. **Implement `ReleaseFrameSurfaces(ids)`.** The decoder hands back
   *global* IDs at frame finish (the ones the refresh slot used to
   carry); the releaser is responsible for dispatching each release
   to the pool that owns it:

   ```go
   func (s *Layers) ReleaseFrameSurfaces(ids []int) error {
       for _, id := range ids {
           layer, local := decodeGlobalID(id)
           if err := s.byID[layer].pool.Release(local); err != nil {
               return err
           }
       }
       return nil
   }
   ```

For mid-frame motion-field projection across spatial layers
(`UseRefFrameMVS == true`), the same pattern applies via
`DecoderTemporalMotionReferenceProvider`. The per-layer state must also
carry its own temporal-MV scratch (one `mvStore` slot per local pool
index) since the projection metadata is keyed by surface. The root
helpers `PublishDecoderTemporalMotionReference`,
`ResolveDecoderTemporalMotionReferences`, and
`ResolveDecoderTemporalMotionReferencesWithProvider` cover the common
store/update operations.

`NewDecoderFrameLayerPool` is a thin adapter over `FrameLayerPool` that
already satisfies the provider and releaser interfaces, so callers using
`FrameLayerPool` get the routing for free.

---

## Scaled inter prediction

True spatial SVC needs *scaled inter prediction*: enhancement-layer
frames reference base-layer surfaces at a different resolution and the
decoder must apply the AV1 8-tap convolve at the appropriate scale factor. The
reference implementation lives at `internal/av1/motion/scaled.go`, with the
threading dispatcher in `internal/av1/threading/predict_scaled.go`. The SVC
strict gates enable this path and L2T1/L2T2 pass every-frame MD5.

To run the SVC extended dry-run directly:

```sh
GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 \
    go test -tags goav1_oracle ./internal/av1/testvector \
    -run TestLibaomExtendedFrameWorkDryRun -count=1 -timeout 1800s -v
```

---

## Picking which layer to emit

libaom's `aomdec` with the default `output_all_layers=0` emits exactly
one YUV frame per temporal unit, at the highest `SpatialID` the
temporal unit publishes. `cmd/dump_svc/main.go` follows the same
convention by default, and surfaces the choice through the
`-spatial-layer=N` flag: `-1` (the default) emits the highest layer
the temporal unit publishes, while a non-negative `N` pins emission
to a specific `SpatialID` (any temporal unit that did not publish at
that layer simply emits no bytes for that TU, mirroring an SFU that
drops everything above its operating point).

`cmd/dump_svc` also auto-installs `GOAV1_SCALED_PRED=1` into the
process environment unless the caller passed `-scaled-pred=false`,
which means out-of-the-box invocations of the CLI take the
scaled-inter path whenever an SVC stream needs it. The companion
output-format flags (`-format=i420 | yuv420p10le`, and the
`-i420` / `-yuv420p10le` shorthand aliases) reject mismatches against
the bitstream's bit depth so callers do not silently produce
half-resolution YUV from a misconfigured filter chain.

The selection rule applies *after* a temporal unit's events have all
been driven through the runner:

```go
for _, payload := range payloads {
    runner.RunLowOverheadInto(&result, payload, nil)

    var emit *av1.Frame
    for _, frame := range result.Run.Outputs {
        if frame != nil {
            emit = frame   // last non-nil entry wins
        }
    }
    if emit != nil {
        writeYUVFrame(out, emit)
    }
}
```

This works because the residual stream runner pushes outputs in event
order: each completed spatial layer's surface lands in
`result.Run.Outputs` in `SpatialID` ascending order, so the last
non-nil pointer corresponds to the highest spatial layer for that
temporal unit. Callers that want *all* layers (the
`output_all_layers=1` equivalent) should iterate the full slice and
write each non-nil frame separately, keyed by an event-side
`SpatialID` track.

For RTP receivers, the AV1 aggregation header gives you the
(T, S) pair *before* you decode the OBU. That is the right place to
implement layer drop policy — discard packets above the operating
point you chose, then feed only the surviving payloads to the
decoder. For a single decode chain, or for one independent simulcast
spatial layer at a time, `NewDecoderFromRTPPayloads` is the friendly
ordered-payload entry point. The returned decoder can also drive live
payloads with `DecodeRTPPayload`; call `DecodeRTPPayloadAfterLoss` after
the jitter buffer detects a packet gap to clear retained RTP fragments while
preserving parser sequence/reference state. For full SVC modes with shared
reference slots, use `NewLayeredDecoderFromRTPPayloads`; it owns the
layer-aware frame pool, per-spatial frame-work states, shared reference slots,
and shared frame contexts needed for references to resolve across
spatial-layer pools. Use `DecodeNextWithMetadata`,
`DecodeRTPPayloadWithMetadata`, or `DecodeRTPPayloadAfterLossWithMetadata` when
the receive loop needs each output paired with parsed AV1 spatial ID, temporal
ID, frame type, coded-keyframe flag, and frame-size metadata. Dependency
descriptor values such as active decode-target masks are RTP header-extension
metadata and are parsed separately with `ParseRTPDependencyDescriptor`. The
SDP helpers (`ParseAV1SDPFmtp`, `ParseAV1SDPExtmap`, `ParseAV1SDPRID`,
`ParseAV1SDPSimulcast`, `AV1SDPOffersReceiveParams`,
`AV1SDPOffersReceiveSequence`, and `AV1SDPOffersReceiveFrame`) cover the
`AV1/90000` payload binding, the profile/level/tier compatibility check, RTP
header-extension mapping checks for dependency descriptors and RID/MID SDES
values, AV1 RID receiver restrictions such as `max-width`, `max-height`,
`max-fps`, `max-fs`, `max-br`, `max-pps`, `max-bpp`, and `depend`, and AV1
simulcast RID groups before RTP starts flowing. RTP SDES helpers
(`PutRTPMIDHeaderExtension`, `PutRTPStreamIDHeaderExtension`, and
`PutRTPRepairedStreamIDHeaderExtension`) validate and write raw MID/RID/RRID
header-extension payload bytes after the caller has selected RTP extension
IDs; the `AV1SDP*RTCPFeedback` helpers verify payload-specific or wildcard
`rtcp-fb` support for NACK, PLI, FIR, LRR, Transport-CC, and REMB. Generic RTCP
packet helpers (`ParseRTCPPacket`, `ParseRTCPCompoundPackets`) cover common
packet parsing and compound-packet demux, including unknown packet types;
sender/receiver report helpers (`ParseRTCPSenderReportPacket`,
`ParseRTCPReceiverReportPacket`) cover SR/RR packet wrapping, report blocks,
and signed 24-bit cumulative-loss fields; SDES helpers (`ParseRTCPSDESPacket`)
cover source-description chunks such as CNAME and TOOL; BYE helpers
(`ParseRTCPByePacket`) cover source lists and reason text; feedback packet
helpers (`ParseRTCPFeedbackPacket`) cover complete RTPFB/PSFB packet wrapping
and parsing; RTCP Transport-CC helpers (`ParseRTCPTransportFeedbackFCI`) cover
status chunks, receive-delta ticks, and no-timestamp feedback; RTCP REMB helpers
(`ParseRTCPReceiverEstimatedMaximumBitrateFCI`) cover the legacy WebRTC
bitrate/SSRC FCI format; and RTCP LRR helpers
(`ParseAV1RTCPLayerRefreshRequestEntries`,
`EncoderWebRTCValidateLayerRefreshRequests`) cover the AV1 layer-index FCI
entry-list format and validate requested upgrades against the active
scalability grid. The
lower-level framework path with `FrameLayerPool`,
`NewDecoderFrameLayerPool`, and `DecoderFrameWorkExternalReferenceRuntime`
remains available when callers need custom arena ownership or event-level
control.

---

## Current support status

The SVC decode pipeline is green on the committed strict gates. As of this
writing:

| Capability                                          | State                                      |
|-----------------------------------------------------|--------------------------------------------|
| OBU extension header parse (`temporal_id`, `spatial_id`) | Complete; surfaced on every `DecoderEvent` and `obu.Unit`. |
| Sequence-header operating-points parse              | Complete; `SequenceHeader.OperatingPoints[]` populated. |
| Scalability metadata OBU parse                      | Complete; `Metadata.Scalability`.          |
| Per-spatial-layer `FrameWorkState`                  | Complete; one state instance per layer.    |
| Stream-global `DecoderSurfaceReferences` across layers | Complete; one reference set shared across all layer states. |
| Multi-pool reference routing (`DecoderFrameSurfaceProvider`) | Complete and exported at root through `DecoderFrameWorkExternalReferenceRuntime`, `NewDecoderFrameLayerPool`, and provider resolve helpers. |
| `FrameLayerPool` aggregator                         | Complete and exported at root.             |
| Scaled inter prediction (8-tap convolve at scale)   | Complete; exercised by L2T1/L2T2 strict gates. |
| Warp + scaled fallback                              | Complete; pairs with scaled inter when warped-motion mismatches the reference scale. |
| Inter-intra + scaled                                | Complete for the committed strict SVC vectors. |
| L1T2 single-pool decode                             | Strict every-frame MD5 pass in `make dryrun-relevant-supported`. |
| L2T1 / L2T2 multi-pool decode                       | Strict every-frame MD5 pass in `make dryrun-extended`. |
| WebRTC AV1 SVC control metadata                     | Complete for the W3C mode vocabulary (`L*T*`, `L*T*h`, `L*T*_KEY`, `L*T*_KEY_SHIFT`, `S*T*`, `S*T*h`) with dependency-descriptor decode targets over the full `(spatial, temporal)` grid, W3C key-shift temporal schedules, and pinned-libwebrtc `L2T2_KEY_SHIFT` dependency templates. |
| WebRTC AV1 SDP helpers                              | Complete for the registered `AV1/90000` payload binding, optional `profile` / `level-idx` / `tier` fmtp defaults and validation, RTP header-extension mapping checks, AV1 RID receiver restrictions, AV1 simulcast RID groups, offer receive frame checks, sequence-header compatibility checks, and AV1 rtcp-fb checks. |
| WebRTC AV1 RTCP generic/SR/RR/SDES/BYE/Transport-CC/REMB/LRR helpers | Complete for generic packet parse/build, compound-packet demux, sender/receiver report packet parse/build, source-description/CNAME packet parse/build, BYE packet parse/build, RTPFB/PSFB packet wrapping/parsing, transport-wide status chunks, delta ticks, no-timestamp feedback, legacy WebRTC REMB bitrate/SSRC FCI parse/build, AV1 layer-index and LRR FCI entry/list parse/build/validation, plus encoder-config temporal/spatial layer-grid checks. |
| High-level RTP payload decode                       | `NewDecoderFromRTPPayloads` covers ordered/live AV1 RTP payload bodies for single decode chains and independent simulcast layers; `NewLayeredDecoderFromRTPPayloads` covers shared-reference SVC RTP streams with frame-only and `WithMetadata` AV1 layer outputs; both include `DecodeRTPPayloadAfterLoss` retained-fragment reset after packet gaps. |
| Strict every-frame parity                           | Passing for the committed SVC vectors; broader SVC corpus expansion remains open. |

The WebRTC control row is metadata/control support for already-produced frame
payloads. The friendly `RTCEncoder` pixel path exposes `EncodePicture` for
multi-spatial WebRTC output in SVC and simulcast modes: it downscales the
8-bit I420 source into active spatial layers, emits one RTP-frame-ready AV1
output per layer, and stamps the dependency descriptors. Full SVC modes use
shared reference slots for inter-layer prediction; simulcast modes keep
independent per-spatial encoders. `RTCEncoder.SetConfig` applies bitrate,
framerate, and supported scalability changes atomically; changes that alter
layer geometry or dependency structure make the next picture a key picture.
`EncoderWebRTCRTPFrameDuration` returns the exact RTP timestamp duration for
the normalized encoder configuration, so callers can pace RTP timestamps from
the same framerate/timebase the encoder accepted. `RTCPicture` exposes
`AllDecodeTargetsMask`, `ActiveDecodeTargetsMask`, and
`ActiveDecodeTargetsRTPOptions` so layer-activation changes can be applied
consistently to every RTP frame in a multi-spatial picture.

The framework dry-run tests
(`internal/av1/testvector/libaom_oracle_test.go`) exercise the
multi-pool path against the committed L1T2/L2T1/L2T2 vectors and emit
per-vector `vector=NAME frames=N md5_matches=M first_mismatch=F`
summary lines. Run them with:

```sh
GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 \
    go test -tags goav1_oracle ./internal/av1/testvector \
    -run TestLibaomExtendedFrameWorkDryRun -count=1 -timeout 1800s -v
```

For the strict relevant gate, including the L1T2 SVC vector:

```sh
GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 \
    go test -tags goav1_oracle ./internal/av1/testvector \
    -run TestLibaomFastFrameWorkDryRun -count=1 -timeout 600s -v
```

The fast dry-run CI lane includes the current fast SVC vector. Full, extended,
and profile lanes are still heavier local/release gates.

---

## Conformance vectors

Three SVC vectors are committed under `internal/av1/testdata/libaom/`:

| Vector                                  | Layout | Resolution         | Frames | Operating points |
|-----------------------------------------|--------|--------------------|--------|------------------|
| `av1-1-b8-22-svc-L1T2.ivf`              | L1T2   | 640x360            | 8      | T-only + base     |
| `av1-1-b8-22-svc-L2T1.ivf`              | L2T1   | 640x360 + 1280x720 | 8      | S-only + base     |
| `av1-1-b8-22-svc-L2T2.ivf`              | L2T2   | 640x360 + 1280x720 | 8      | full + 3 subsets  |

L1T2 is single-pool friendly (one spatial layer); L2T1 and L2T2
require the multi-pool path. Each ships with the libaom `.md5`
companion so the framework dry-run can compare per-frame digests.

`cmd/dump_svc -svc-info` is the quickest way to inspect any of these
without running a decode:

```sh
./bin/dump_svc -svc-info internal/av1/testdata/libaom/av1-1-b8-22-svc-L2T2.ivf
```

prints the operating-points list and the per-(T,S) OBU fan-out.

---

## Driving SVC end to end: walkthrough

This section recaps the moving parts of an SVC integration in roughly
the order a caller wires them together. Each step references the type
or helper that owns that slice of state.

1. **Read the IVF / RTP container.** Use `NewIVFIterator` for files
   or the RTP `Depacketizer` for live streams. Surfaces emitted by
   `it.Next()` give you per-frame payloads plus the IVF timebase.

2. **Probe the sequence header.** Walk the first frame's OBUs with
   `ParseLowOverheadOBU` and feed the `OBUSequenceHeader` payload to
   `ParseSequenceHeader`. Inspect `OperatingPoints[]` to decide which
   operating-point IDC you want to decode. (For most SVC clients
   this is fixed at session setup, not negotiated mid-stream.)

3. **Decide the integration shape.**
   - **Single-pool:** if every spatial layer shares one
     `CodedWidth x Height`, bind one `FramePool` sized for
     `RefFrames + N_in_flight` surfaces and drive the public
     stream runner. This is the `cmd/dump_svc` and
     `cmd/aom-go-dec` shape.
   - **Multi-pool RTP:** for WebRTC-style shared-reference SVC RTP
     payloads, use `NewLayeredDecoderFromRTPPayloads` and feed ordered
     payload bodies with `DecodeNext` or live payload bodies with
     `DecodeRTPPayload`. Use the `WithMetadata` variants when routing or
     rendering needs the decoded frame's AV1 spatial/temporal IDs.
   - **Custom multi-pool:** if you need event-level control, bind a
     `FrameLayerPool`, wrap it with `NewDecoderFrameLayerPool`, and drive
     residual events with `DecoderFrameWorkExternalReferenceRuntime`. The
     current event's `FramePool` must be the sub-pool matching that event's
     coded `FrameFormat`; root helpers such as `FrameCodedFormatFromHeaders`,
     `FrameLayerPool.SubPool`, and `DecoderLayerPoolGlobalSurfaceID` provide
     the pieces. The dry-run harness (`libaomSpatialLayers` in
     `internal/av1/testvector/libaom_oracle_test.go`) remains the reference
     implementation for full mixed-resolution orchestration.

4. **Probe scratch sizes once.** Call
   `DecoderFrameWorkResidualLowOverheadStreamsPlan` (or the
   per-event sizing helpers) over the full payload sequence to
   compute the residual-runner scratch sizes. Allocate the
   `DecoderFrameWorkResidualStreamScratch` arena from the returned
   sizes; reuse it across frames.

5. **Set the scaled-prediction gate.** If any spatial layer
   references a smaller layer at a different resolution, set
   `GOAV1_SCALED_PRED=1` in the process environment or build with
   `-tags goav1_scaled_pred`. Without this gate, mismatched
   references fail with `threading: invalid batch`.

6. **Drive payloads.** Call `runner.RunLowOverheadInto(&result,
   payload, postFilter)` (or the matching per-payload helper for
   the multi-pool path) once per IVF frame. The runner internally
   parses every OBU in the payload, advances per-layer
   `FrameWorkState` as needed, and publishes completed surfaces
   into `result.Run.Outputs`.

7. **Pick the output for this temporal unit.** Walk the
   `result.Run.Outputs` slice and either emit the last non-nil
   entry (highest spatial layer, libaom `output_all_layers=0`
   behavior) or every non-nil entry keyed by event-side `SpatialID`
   (libaom `output_all_layers=1` behavior).

8. **Copy or stream the YUV samples out.** Surface pointers in
   `result.Run.Outputs` alias the runner's caller-owned output
   arena; they are valid until the next `Run*` call into the
   runner. Copy or stream the planes before the next iteration.

9. **Reset between sessions.** Call `pool.Reset()`,
   `refs.Reset()`, `state.Reset()`, and `runner.Reset()` before
   starting a new coded video sequence.

For a complete worked example, read
[`cmd/dump_svc/main.go`](../cmd/dump_svc/main.go) and
[`cmd/dump_svc/decoder.go`](../cmd/dump_svc/decoder.go) together;
they are the smallest end-to-end SVC driver in the repository and
deliberately mirror the shape of the bench harness in
`bench_test.go` so the same scratch-binding pattern carries from
microbenchmarks to integration.
