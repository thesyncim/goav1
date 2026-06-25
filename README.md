# goav1

[![ci](https://github.com/thesyncim/goav1/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/ci.yml)
[![lint](https://github.com/thesyncim/goav1/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/lint.yml)
[![testvectors](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml)

Pure-Go AV1 decoder and WebRTC-focused realtime encoder toolkit, built toward
dav1d-class decode performance with no cgo dependency.

The decoder path covers AV1 IVF/OBU/RTP input, sequence and frame headers, tile
decode, prediction, inverse transforms, reconstruction, loop filter, CDEF,
super-resolution, loop restoration, film grain, SVC surface routing, and raw
YUV output. Hot paths are designed around caller-owned buffers, reusable frame
pools, explicit scratch sizing, and zero steady-state allocations.

- **Pure Go, no cgo, no dependencies** - `go get` and cross-compile like a
  normal Go package.
- **Byte-exact conformance gates** - the committed libaom remote manifest,
  extended diagnostic vectors, and vendored profile corpus currently pass strict
  per-frame MD5; see [Conformance](#conformance).
- **Realtime-first ownership model** - reusable decoders, caller-owned scratch,
  bounded worker pools, and allocation/compiler guardrails are part of the
  normal development loop.
- **Performance work is active** - scalar Go is already source-shaped against
  libaom/dav1d where it matters, and SIMD/assembly dispatch exists for the first
  motion kernels, but matching dav1d throughput remains the main open target.
- **Encoder implementation is in scope** - the encoder target is WebRTC AV1
  realtime use cases and controls, not offline/general-purpose encoding.

Pre-v1: no release tag has been published yet.

## Install

```sh
go get github.com/thesyncim/goav1
```

Requires Go 1.26 or newer.

## Quick start

goav1 exposes both a friendly copy-returning API and the explicit ownership
surface used by realtime integrations. Pick the path by lifetime and allocation
needs:

| Use case | API | Ownership model |
| --- | --- | --- |
| One-shot decoded pixels | `DecodeIVF` | Copies visible planes into independent `DecodedFrame` values |
| AV1 WebRTC SDP/RTP control checks | `AV1SDPRTPMap.SDP` / `ParseAV1SDPRTPMap` / `AV1SDPFmtpAttribute.SDP` / `ParseAV1SDPFmtpAttribute` / `ParseAV1SDPExtmap` / `AV1SDPNegotiatesHeaderExtensionID` / `AV1SDPOffersReceiveHeaderExtensionID` / `AV1SDPAnswersSendHeaderExtensionID` / `AV1SDPRTCPFeedback.SDP` / `ParseAV1SDPRID` / `ParseAV1SDPSimulcast` / `PutRTPPacket` / `ParseRTPPacket` / `PutRTPHeaderExtensionElements` / `ParseRTPHeaderExtensionElements` / `FindRTPHeaderExtensionElement` / `ParseRTPPacketDependencyDescriptor` / `EncoderWebRTCScalabilityModeIDC` / `AppendEncoderLowOverheadWebRTCScalabilityMetadataOBU` / `EncoderWebRTCRTPPacketsWithHeadersSize` / `AppendEncoderWebRTCRTPPacketsWithHeaders` / `PutRTPMIDHeaderExtension` / `PutRTPTransportWideCCHeaderExtension` / `PutRTPTransportWideCC02HeaderExtension` / `PutRTPAbsoluteCaptureTimeHeaderExtension` / `PutRTPColorSpaceHeaderExtension` / `PutRTPCoordinationOfVideoOrientationHeaderExtension` / `PutRTPVideoTimingHeaderExtension` / `AV1SDPOffersReceive*` | Parses and emits `AV1/90000` payload bindings, profile/level/tier fmtp lines and values, RTP header-extension mappings and direction-aware extmap IDs, payload-specific or wildcard rtcp-fb lines, AV1 RID receiver restrictions, AV1 simulcast RID groups, RTP fixed headers and RFC 8285 one-/two-byte header-extension elements, zero-allocation negotiated-extension lookup, complete RTP packet dependency-descriptor extraction, WebRTC scalabilityMode to AV1 scalability_mode_idc mapping and predefined or explicit-SS scalability metadata OBUs, sized complete encoded-frame RTP packets with dependency-descriptor plus optional MID/RID/RRID and TWCC/TWCC-02 extensions, raw MID/RID/RRID SDES payloads, and raw CVO/playout-delay/TWCC/TWCC-02/absolute-send-time/absolute-capture-time/color-space/video-content-type/video-timing payloads; complete SDP assembly and media transport policy stay caller-owned |
| AV1 WebRTC RTCP packet checks | `ParseRTCPPacket` / `ParseRTCPCompoundPackets` / `PutRTCPPacket` / `ParseRTCPSenderReportPacket` / `PutRTCPSenderReportPacket` / `ParseRTCPReceiverReportPacket` / `PutRTCPReceiverReportPacket` / `ParseRTCPSDESPacket` / `PutRTCPSDESPacket` / `ParseRTCPByePacket` / `PutRTCPByePacket` / `ParseRTCPFeedbackPacket` / `PutRTCPFeedbackPacket` / `ParseRTCPGenericNACKPairs` / `AppendRTCPGenericNACKPairsForLostSequenceNumbers` / `ParseRTCPTransportFeedbackFCI` / `RTCPTransportFeedbackForReceivedPackets` / `AppendRTCPTransportFeedbackPacketsForReceivedPackets` / `EncoderWebRTCRTCPCompoundPacketsTransportFeedback` / `SummarizeRTCPTransportFeedback` / `AppendRTCPTransportFeedbackPacketReceptions` / `ParseRTCPPictureLossIndicationFCI` / `ParseRTCPFullIntraRequestEntries` / `ParseRTCPReceiverEstimatedMaximumBitrateFCI` / `EncoderWebRTCRTCPCompoundPacketsApplyReceiverEstimatedMaximumBitrate` / `ParseAV1RTCPLayerRefreshRequestEntries` / `EncoderWebRTCValidateLayerRefreshRequests` / `EncoderWebRTCRTCPFeedbackRequiresKeyFrame` / `EncoderWebRTCRTCPPacketsRequireKeyFrame` / `EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame` | Parses and serializes generic RTCP packets, compound RTCP packet streams, sender/receiver report packets, source-description/CNAME chunks, BYE source lists, complete RTPFB/PSFB feedback packets, generic NACK PID/BLP pairs, RTP sequence-list-to-NACK grouping, transport-wide congestion-control status chunks, receiver-side Transport-CC report construction, and receive-timeline reconstruction, empty PLI FCI payloads, FIR FCI entries, legacy WebRTC REMB bitrate/SSRC FCI payloads, and AV1 LRR FCI entry lists; Transport-CC feedback and REMB feedback can be extracted from single, parsed compound, or raw compound packets, with REMB applied to encoder bitrate config while pacing policy stays caller-owned; LRR entries can be validated against the active temporal/spatial layer grid, and single, parsed compound, or raw compound PLI/FIR/LRR feedback can be classified into an encoder `forceKey` decision; network transport stays caller-owned |
| Repeated in-memory IVF decode | `NewDecoderFromIVF` + `DecodeNext` | Copies IVF payloads once, then reuses decoder-owned frame and post-filter arenas |
| Large/file-backed IVF decode | `NewDecoderFromIVFReaderAt` + `DecodeNext` | Indexes IVF frame offsets and reads each payload into one reusable buffer |
| Already-demuxed temporal units | `NewDecoder(payloads)` + `DecodeNext` | Retains payload slices by reference and reuses all decode/output arenas |
| Ordered or live AV1 RTP payload bodies or packets | `NewDecoderFromRTPPayloads(payloads)` / `NewDecoderFromRTPPackets(packets)` + `DecodeNext` / `DecodeRTPPayload` / `DecodeRTPPacket` | Retains constructor payload/packet slices by reference; each RTP payload or packet may return zero frames for fragments |
| Shared-reference SVC RTP payload bodies or packets | `NewLayeredDecoderFromRTPPayloads(payloads)` / `NewLayeredDecoderFromRTPPackets(packets)` + `DecodeNext` / `DecodeRTPPayload` / `DecodeRTPPacket` / `DecodeNextWithMetadata` | Owns a layer-aware frame pool and shared reference state for multi-spatial WebRTC receive loops |
| Manual OBU or RTP run loop | `DecoderFrameWorkResidualStreamRunner` `Run*Into` methods | Caller-owned event, tile, frame, post-filter, and output-result scratch |
| IVF demux only | `NewIVFIterator` | Zero-allocation payload views into the source bytes |

The simplest helper decodes a complete IVF stream and returns independent frame
copies:

```go
package main

import (
	"fmt"
	"log"
	"os"

	av1 "github.com/thesyncim/goav1"
)

func main() {
	ivf, err := os.ReadFile("input.ivf")
	if err != nil {
		log.Fatal(err)
	}

	frames, err := av1.DecodeIVF(ivf, av1.WithWorkers(1))
	if err != nil {
		log.Fatal(err)
	}

	for i, frame := range frames {
		fmt.Printf("frame %d: %dx%d bytes/sample=%d\n",
			i, frame.Width, frame.Height, frame.BytesPerSample)
	}
}
```

For steady decode loops, build a reusable decoder. Returned `*Frame` values
alias decoder-owned frame-pool memory and stay valid until the next decode call:

```go
dec, err := av1.NewDecoderFromIVF(ivf, av1.WithWorkers(4))
if err != nil {
	log.Fatal(err)
}
defer dec.Close()

for {
	frames, ok, err := dec.DecodeNext()
	if err != nil {
		log.Fatal(err)
	}
	if !ok {
		break
	}
	for _, frame := range frames {
		_ = frame // consume or copy visible Y/U/V planes before the next call
	}
}
```

Lower-level package APIs expose the same pipeline pieces directly: IVF and OBU
iterators, RTP packetization/depacketization, parser structs, tile work
scheduling, residual decode/reconstruct helpers, post-filter scratch binding,
and DSP primitives. SDP helpers cover the registered `AV1/90000` rtpmap payload
binding parsing and emission, profile/level/tier fmtp line parsing and emission,
RTP header-extension mappings and negotiated extmap IDs for dependency descriptors, transport/video
metadata, and RID/MID SDES values, AV1 RID receiver
restrictions, AV1 simulcast RID groups, sequence-header compatibility checks,
offer receive checks, payload-specific or wildcard rtcp-fb line emission and checks, and the
dependency descriptor extmap URI. RTP SDES helpers validate and write raw
MID/RID/RRID header-extension payload bytes after the caller has selected RTP
extension IDs; RTP packet helpers parse and write fixed RTP headers plus
RFC 8285 one-/two-byte extension element envelopes; WebRTC RTP helpers also
parse and write raw CVO, playout-delay, transport-wide-cc, transport-wide-cc-02, absolute-send-time,
absolute-capture-time, color-space, video-content-type, and video-timing
payload bytes. Complete SDP generation, transceiver setup, SRTP, jitter
buffering, loss policy, pacing, and retransmission remain with the caller. RTCP helpers cover generic packet and compound
packet parsing for forward-compatible demux, sender/receiver report packets with
report-block parsing, source-description/CNAME chunks, BYE source lists and
reason text, complete RTPFB/PSFB feedback packet wrapping and parsing, generic
NACK PID/BLP pairs and RTP loss sequence grouping, Transport-CC feedback status
chunks, delta ticks, receiver-side report construction, packet reception
timelines, and compact summaries, empty
PLI FCI payloads, FIR FCI entries, legacy WebRTC REMB
bitrate/SSRC FCI parsing and serialization, REMB-to-encoder-bitrate config
helpers, and AV1 Layer Refresh Request FCI entry-list parsing,
serialization, and validation against an encoder config;
`EncoderWebRTCRTCPFeedbackRequiresKeyFrame`,
`EncoderWebRTCRTCPPacketsRequireKeyFrame`, and
`EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame` classify parsed single,
parsed compound, or raw compound PLI/FIR/LRR feedback into the existing
`forceKey` argument when a full refresh is the desired safe response. For
ordered RTP payload bodies,
`NewDecoderFromRTPPayloads` is the reusable high-level path for single decode
chains and independent simulcast layers. `NewLayeredDecoderFromRTPPayloads` is
the high-level receive path for shared-reference WebRTC SVC streams whose
reference slots span spatial layers or coded frame sizes. Use the `WithMetadata`
variants on `LayeredDecoder` when a receive loop needs the decoded output paired
with parsed AV1 spatial ID, temporal ID, frame type, keyframe flag, and
frame-size metadata. RTP dependency descriptor fields such as active
decode-target masks are RTP header-extension metadata; receive loops that keep
complete packets can extract them with `ParseRTPPacketDependencyDescriptor`,
while payload-only integrations can parse the extension value with
`ParseRTPDependencyDescriptor`. Both decoders can drive live payloads or packets
with `DecodeRTPPayload`, `DecodeRTPPacket`, and the matching `AfterLoss`
variants when the constructor payloads represent the stream shape and maximum
retained-fragment/event scratch needed.
`RTPPacketSequencer.AppendMissingSequenceNumbers` and
`RTPPacketSequencer.AppendMissingRTCPGenericNACKPairs` expose currently missing
packet gaps for caller-owned NACK timing policy. The `RunLowOverheadInto`,
`RunLowOverheadsInto`, `RunRTPPayloadInto`, `RunRTPPayloadsInto`, and
`RunRTPPayloadAfterLossInto` families remain the
allocation-aware entry points for callers that already own every arena and
result buffer directly. The executable examples in `example_test.go` and
`example_decode_simple_test.go` are kept in sync by `go test`.

## Features

| Area | Status |
| --- | --- |
| Containers and transport | IVF, AV1 low-overhead OBU, Annex B, Section 5 temporal units, RTP fixed-header and RFC 8285 one-/two-byte extension element parse/build, sized encoded-frame RTP packet wrapping with dependency-descriptor/MID/RID/RRID/TWCC/TWCC-02 extensions, AV1 RTP payload parse/build/fragment/reassemble, AV1 SDP/fmtp/profile/level/tier/extmap/RID/simulcast/rtcp-fb helpers, RTP MID/RID/CVO/playout-delay/TWCC/TWCC-02/absolute-send-time/absolute-capture-time/color-space/video-content-type/video-timing payload helpers, RTCP generic/compound/SR/RR/SDES/BYE/NACK/Transport-CC report construction and receive timelines/PLI/FIR/REMB, RTP gap-to-NACK, and AV1 LRR helpers |
| Decoder profiles | Profile 0 and Profile 1 pass committed/vendored strict-MD5 gates; Profile 2 has passing 4:2:2 8/10/12-bit plus 4:2:0 and 4:4:4 12-bit profile clips, with wider 12-bit breadth still expanding |
| Bit depths and formats | 8-bit and 10-bit covered broadly; 12-bit covered by targeted profile-2 4:2:0, 4:2:2, and 4:4:4 clips; 4:2:0, 4:2:2, 4:4:4, and monochrome surfaces |
| Prediction and residuals | Intra, directional intra, filter intra, CfL, palette, IntraBC, inter/compound, OBMC, warped motion, scaled motion, transforms, dequantization, and CDF adaptation |
| Post filters | Loop filter, CDEF, super-resolution, loop restoration, and film grain are wired into the high-level decode/output path |
| WebRTC RTP decode | `NewDecoderFromRTPPayloads` / `NewDecoderFromRTPPackets` cover ordered/live RTP payload bodies or complete packets for single decode chains and simulcast layers; `NewLayeredDecoderFromRTPPayloads` / `NewLayeredDecoderFromRTPPackets` cover shared-reference SVC RTP streams; `DecodeRTPPayloadAfterLoss` and `DecodeRTPPacketAfterLoss` reset retained fragments after packet gaps |
| SVC | L1T2/L2T1/L2T2 oracle vectors pass through the framework path; public integration guidance lives in [docs/svc.md](docs/svc.md) |
| Tile groups | Single and multi-tile groups pass current strict-MD5 gates; tile-list OBUs parse, validate source reference anchors/uniform decode layout, resolve external anchor frames through reference slots or provider-backed surface IDs, plan raw tile-list entry decode jobs, bind each entry's external anchor as LAST_FRAME for residual decode, reconstruct entries through the residual runner, blit decoded rectangles into a tile-list output mosaic, and publish tile-list outputs through low-level stream runners plus high-level low-overhead, IVF, RTP payload, RTP packet, and sequenced RTP packet decode when frame context is present |
| Encoder | Functional realtime 8-bit profile-0 WebRTC encoder with I420/I422/I444/I400/NV12/NV21 plus generic 8/10/12-bit `Frame` input adapters, fixed-quality/CBR, forced keyframes, temporal layering, runtime bitrate/framerate/rate-control/scalability reconfiguration, current RTP timestamp-duration helpers, multi-spatial `RTCEncoder.EncodePicture` for W3C SVC and simulcast modes, tile columns, golden references, RTP payload packetization, sized complete RTP packet wrapping with dependency descriptors, MID/RID/RRID, and TWCC/TWCC-02, active decode target signaling, and LRR layer-grid validation; non-4:2:0, monochrome, and high-bit-depth inputs adapt into the current 8-bit 4:2:0 encode path; lower-level WebRTC controls cover the W3C AV1 SVC mode vocabulary, temporal/spatial dependency structures, dependency-descriptor decode targets, W3C key-shift temporal schedules, pinned-libwebrtc L2T2_KEY_SHIFT templates, explicit sequence color config, and `Frame` validation/loading for profile-0/1/2 sample formats |
| SIMD/assembly | CPU-dispatch skeleton plus initial amd64/arm64 motion kernels; broader transform/CDEF/restoration kernels are still roadmap work |

The full feature matrix, status legend, vector coverage, and forward-looking
gaps live in [CONFORMANCE.md](CONFORMANCE.md). The package and lifecycle map
lives in [ARCHITECTURE.md](ARCHITECTURE.md). Upstream pins are tracked in
[UPSTREAM.md](UPSTREAM.md) and `third_party/upstream.lock`.

## Encoder Scope

Encoder implementation is scoped to realtime WebRTC AV1. It is not intended to
be an offline encoder, a full authoring tool, or a general replacement for
`aomenc`.

There are two public encoder surfaces:

- `VideoEncoder` / `RTCEncoder` is the friendly realtime pixel encoder. It
  accepts 8-bit I420, I422, I444, I400, NV12, and NV21 input plus generic
  8/10/12-bit `Frame` inputs for 4:2:0, 4:2:2, 4:4:4, and monochrome layouts.
  I422/I444 and generic non-4:2:0 chroma samples are resampled, monochrome
  fills neutral chroma, and 10/12-bit `Frame` samples are downshifted before
  entering the current 8-bit 4:2:0 profile-0 encode path.
  `RTCEncoder.Encode` emits one
  single-spatial-layer AV1 temporal unit per call; `RTCEncoder.EncodePicture`
  emits one RTP-frame-ready output per active spatial layer for WebRTC SVC and
  simulcast modes, including key and key-shift schedules.
  `RTCEncoder.SetConfig` applies
  bitrate, framerate, CBR/CQP rate-control, fixed quantizer, and supported
  scalability changes atomically; changes that alter layer geometry or
  dependency structure make the next picture a key picture.
  `RTCEncoder.RTPFrameDuration` reports the current RTP timestamp increment
  after those changes.
  `NormalizeRTCEncoderConfig` lets callers preflight whether a
  lower-level WebRTC config is supported by this friendly pixel pipeline before
  constructing or reconfiguring an encoder. `RTCFrame.AppendRTPPackets`
  packetizes each frame into AV1 RTP
  payload bodies and matching per-packet dependency descriptors using
  caller-owned buffers; `EncoderWebRTCRTPPacketsWithHeadersSize` and
  `AppendEncoderWebRTCRTPPacketsWithHeaders` size and wrap those bodies into
  complete RTP packets with fixed headers, an RFC 8285 dependency-descriptor
  header extension, and optional negotiated MID/RID/RRID plus TWCC/TWCC-02
  header extensions; `RTCFrame.AppendRTPPacketsWithOptions`,
  `RTCPicture.ActiveDecodeTargetsMask`, and
  `RTCPicture.ActiveDecodeTargetsRTPOptions` let integrations signal layer
  activation changes through the dependency descriptor extension.
  `EncoderWebRTCRTCPCompoundPacketsApplyReceiverEstimatedMaximumBitrate` applies
  legacy WebRTC REMB feedback to bounded target bitrate fields before callers
  pass the result to `RTCEncoder.SetConfig`;
  `EncoderWebRTCValidateLayerRefreshRequests` validates AV1 RTCP LRR feedback
  against the configured temporal/spatial grid before callers decide whether to
  force a full key picture. `SetTileColumns` and `SetGoldenInterval` let callers
  retune tile parallelism and golden-reference refresh policy between frames.
- `WebRTCEncoder` is the lower-level control/metadata surface for WebRTC
  picture scheduling. It validates the W3C AV1 SVC mode vocabulary
  (`L*T*`, `L*T*h`, `L*T*_KEY`, `L*T*_KEY_SHIFT`, and `S*T*`/`S*T*h`
  simulcast names), temporal/spatial layer structures, decode-target grids,
  dependency descriptors, active decode target masks, W3C key-shift temporal
  schedules, pinned-libwebrtc `L2T2_KEY_SHIFT` dependency templates, explicit
  sequence color config through `ColorConfigSet`/`ColorConfig`, RTP packet
  spans for already-produced frame payloads, and `Frame` validation /
  sample-plane loading for sequence-matched 8/10/12-bit 4:0:0, 4:2:0, 4:2:2,
  and 4:4:4 caller-owned buffers, including profile-2 12-bit 4:2:0 and
  4:4:4 control paths. `WebRTCEncoder.RTPFrameDuration` mirrors the same
  timestamp-duration query for control-plane-only callers.

Multi-spatial pixel output uses independent per-spatial encoders for simulcast
and shared-reference inter-layer prediction for full SVC. Native high-bit-depth
and true non-4:2:0 bitstream encoding in the friendly pixel encoder, broader
oracle coverage, and compression efficiency tuning remain open. Encoder correctness
and control behavior should
continue to be ported from pinned libaom/libwebrtc source;
speed-sensitive architecture should be checked against pinned SVT-AV1 before
local invention. New encoder code, and decoder code touched while optimizing
it, should preserve upstream C integer width/signedness where it affects layout,
overflow, shifts, or ABI-shaped state.

The realtime encoder is now functional. `goav1.VideoEncoder` turns 4:2:0
frames into AV1 temporal units under fixed quality or CBR rate control, with
forced keyframes, L1T2/L1T3 temporal layering, parallel tile columns, golden
reference anchors, and access to the exact reconstruction a conformant
decoder produces; `goav1.RTCEncoder` wraps the same engine with per-frame RTP
dependency descriptors, caller-owned RTP payload packetization, and runtime
WebRTC control reconfiguration. Every emitted stream decodes bit-exactly to the
encoder's own reconstruction in this package's decoder and in aomdec/dav1d
(enforced by the test gates), including single-spatial and simulcast WebRTC
settings cycles that change bitrate, framerate, rate control, and scalability;
steady-state encoding allocates a handful of objects per frame. See
`ExampleVideoEncoder` and `ExampleRTCEncoder` for
the round trip, and `cmd/encbench` for the standing 1080p60 performance
measurement against SVT-AV1 (throughput currently exceeds SVT preset 12 on
identical input; rate-distortion quality remains behind and is the active
work track).

## CLI

Build the command-line decoder:

```sh
make build-cmd          # writes ./bin/aom-go-dec
make install-cmd        # go install into $GOBIN / $GOPATH/bin
```

Decode an IVF to raw YUV:

```sh
./bin/aom-go-dec -o out.yuv input.ivf
```

or stream to a player:

```sh
./bin/aom-go-dec input.ivf \
    | ffplay -f rawvideo -pixel_format yuv420p -video_size 352x288 -framerate 30 -
```

The CLI writes visible frames after the normal post-filter chain has run. It
supports `-workers N`, writes 10/12-bit output as little-endian 16-bit samples,
emits monochrome as I400, and keeps the YUV byte stream on stdout clean while
timing information goes to stderr.

`cmd/dump_svc` is the companion SVC inspection/decode tool. It can print
operating-point/layer structure and dump the highest spatial layer or a selected
spatial layer:

```sh
go build -o ./bin/dump_svc ./cmd/dump_svc
./bin/dump_svc -svc-info internal/av1/testdata/libaom/av1-1-b8-22-svc-L2T2.ivf
```

## Conformance

goav1 is currently green on the committed AV1 decoder vectors that are part of
this repository's strict framework gates:

| Gate | Current coverage |
| --- | --- |
| `make dryrun-fast` | 14/14 libaom `SuiteLevelFast` vectors pass strict per-frame MD5 |
| `make dryrun-relevant-supported` | 14/14 relevant vectors pass, including 8/10-bit film grain and monochrome |
| `make dryrun-full` | 240/240 committed remote libaom vectors pass |
| `make dryrun-extended` | 226/226 opt-in diagnostic vectors pass, including quantizer sweeps, odd/larger sizes, SVC L1T2/L2T1/L2T2, and multi-tile coverage |
| `make dryrun-profiles` | 38/38 vendored profile clips pass, including profile-0 S_FRAME, profile-1 4:4:4, profile-2 4:2:2 8/10/12-bit, profile-2 4:4:4 12-bit, targeted 12-bit profile-2, CDEF/restoration, super-res, film grain, edge-motion, and non-SVC multi-tile clips |

Optional local corpus lanes are available for broader real-world coverage:

```sh
scripts/gen_bench_corpus.sh
make dryrun-corpus

GOAV1_EXTERNAL_CORPUS_DIR=/path/to/external/ivfs make dryrun-external-corpus
```

The external-corpus lane accepts stream-MD5 or frame-MD5 sidecars and is meant
for Argon Streams, dav1d-test-data, FATE, and local lab corpora without
vendoring large binary clips into this repository.

Known conformance gaps are explicit:

- Contextless tile-list OBUs without frame context or external anchors reject
  early; tile-list reconstruction and publish paths are covered when frame
  context is present.
- Broader profile-2 and real-world corpus breadth should keep expanding even
  though the current committed gates are green.

## Performance

The performance target is clear: close the gap to dav1d without giving up pure
Go portability or byte-exact output. The current work is focused on the same
places the matched profiles point to: coefficient decode shape, loop-filter
planning/application, post-filter scheduling, hot struct layout, zero-cost
tracing, and SIMD kernels for the highest-volume DSP paths.

Current goal order:

- Close the decoder throughput gap with measured, corpus-based goav1 vs dav1d
  and libaom comparisons: same thread count, discarded output, JSON/history
  friendly reports, hot-path profiles, allocation gates, escape/BCE reports,
  trace-zero proof, and hot-struct size guards.
- Spend optimization work where profiles say the gap lives: coefficient/bit
  reader shape, loop-filter edge planning, post-filter execution, large event
  and batch copies, zero-copy IVF payload views, reusable post-filter scratch,
  SIMD dispatch and kernels, and parallel post-filter scheduling.
- Keep quality expanding beyond the committed green vectors: real-world
  corpora, broader profile-2 and 12-bit edge combinations, switch-frame and
  additional tile-list edge coverage, malformed stream hardening, and fuzzing.
- Expand the realtime encoder beyond the current 8-bit profile-0 path:
  high-bit-depth and true non-4:2:0 bitstream encoding, richer WebRTC tuning
  controls, and
  broader libaom/libwebrtc/SVT oracle coverage before wider production claims.
- Preserve upstream C integer widths, signedness, overflow, shift behavior, and
  layout in all new code and touched parity paths. Do not churn untouched legacy
  code solely for type-width cleanup.

- **Zero-allocation steady decode.** `NewDecoder` probes once, sizes the frame
  pool and post-filter scratch up front, and then reuses those arenas through
  `DecodeNext`. `make alloc` enforces `0 B/op` and `0 allocs/op` across the
  post-filtered profile, super-res, restoration, film-grain, and full-vector
  decode benchmarks.
- **Fair cross-decoder timing.** `make bench-cross` compares the verified
  full-decode path against `aomdec`, `dav1d`, and `SvtAv1DecApp` when present,
  using single-thread decode and discarded output. The optional corpus lane
  extends that comparison to longer generated clips so startup overhead does not
  hide the real goav1/dav1d throughput ratio.
- **Compiler guardrails.** Escape-analysis, BCE, trace-zero, and hot-struct
  size checks are treated as perf gates because they catch hidden regressions
  before they turn into frame-time noise.

Run the local benchmarks:

```sh
make bench                 # end-to-end decode benchmarks with frames/sec + MB/sec
make bench-all             # all package benchmarks
make bench-public          # public API hot-path benchmarks
make gc-metrics            # decode GC scan/object-count benchmarks
```

Run the cross-decoder throughput tool when `aomdec`, `dav1d`, or
`SvtAv1DecApp` are on `PATH`:

```sh
make bench-cross
```

`bench-cross` uses the committed libaom vector set, the same output-discard/MD5
style, and the same single-thread baseline where possible. Missing external
decoders are skipped so the target can run on lightweight developer machines.

For the longer steady-state corpus comparison:

```sh
scripts/gen_bench_corpus.sh
GOAV1_BENCH_CORPUS=1 go test -tags goav1_oracle \
    -run TestCrossDecoderCorpus \
    -timeout 30m ./internal/av1/testvector -v -count=1
```

That report prints per-clip and aggregate fps, raw and startup-adjusted external
decoder timings, and the goav1/dav1d ratio that should drive optimization work.

Allocation and compiler guardrails are part of performance, not an afterthought:

```sh
make alloc
make compiler-reports
make trace-zero
```

`make alloc` runs benchmark allocation gates through `scripts/check_allocs.sh`;
`make compiler-reports` fails on new hot-package heap escapes and reports BCE
sites; `make trace-zero` proves disabled entropy tracing compiles out of release
hot paths.

## Verification

Fast edit loop:

```sh
go test ./...
make alloc
make compiler-reports
make trace-zero
```

Codec-safe local gate:

```sh
make webrtc-production
make ci-local
make testvectors-fast
make dryrun-fast
```

`make webrtc-production` requires `aomdec` and `dav1d` on `PATH`; it is the
strict realtime/WebRTC gate for encoder output, decoder RTP receive paths, SVC
scalability modes, RTP/RTCP helpers, loss recovery, control reconfiguration, and
external decoder parity.

Broader parity sweeps:

```sh
make dryrun-full
make dryrun-extended
make dryrun-profiles
make dryrun-corpus
```

Before changing SIMD/DSP behavior, also run the relevant conformance probes:

```sh
make test-motion-conformance
make test-transform-conformance
```

## Trust And Verification

Released version: none yet.

`v0.1.0` is not a release until the tag and GitHub Release are both published.
Pre-release notes live in [RELEASE_NOTES_v0.1.0.md](RELEASE_NOTES_v0.1.0.md).

Required branch checks:

<!-- required-checks:start -->
- `ci`
- `lint`
- `testvectors`
<!-- required-checks:end -->

Release checklist:

- confirm README, package docs, [CONFORMANCE.md](CONFORMANCE.md), and
  [ARCHITECTURE.md](ARCHITECTURE.md) agree
- run the verification commands above on a clean tree
- record conformance gate summaries and benchmark evidence
- publish the tag and GitHub Release together

Supply-chain controls:

- workflow permissions are expected to remain least-privilege
- upstream pins for libaom, dav1d, SVT-AV1, and libwebrtc are recorded in
  `third_party/upstream.lock`; run `make sync-upstreams` and
  `make verify-upstreams` to materialize and verify the local clones
- release evidence should record commit SHA, Go version, platform, pinned
  upstream versions, conformance summaries, allocation/compiler reports,
  cross-decoder benchmark output, and module inventory
- future binary releases need signed checksums, provenance, and an SPDX or
  CycloneDX SBOM

Security reports: [SECURITY.md](SECURITY.md).

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) - package map, pipeline, threading, memory
  model, and public API surfaces
- [CONFORMANCE.md](CONFORMANCE.md) - feature inventory, vector coverage, known
  gaps, and reproduction commands
- [docs/svc.md](docs/svc.md) - scalable video coding integration guide
- [docs/dsp.md](docs/dsp.md) - DSP and SIMD dispatch notes
- [UPSTREAM.md](UPSTREAM.md) - upstream pinning policy
- [CHANGELOG.md](CHANGELOG.md) - release-level change summary
- [SECURITY.md](SECURITY.md) - threat model, supported versions, and reporting
  process

## License

goav1 is distributed under the BSD 2-Clause license. See [LICENSE](LICENSE) for
the full text.

Portions of this repository are derived from libaom, the AV1 reference
implementation released by the Alliance for Open Media. Those portions are
redistributed under the same BSD 2-Clause terms, and the Alliance for Open Media
Patent License 1.0 that accompanies libaom is reproduced verbatim in
[PATENTS](PATENTS), as required by section 1.2.1 of that grant. Callers and
redistributors must keep both [LICENSE](LICENSE) and [PATENTS](PATENTS) together
with any source-form distribution of goav1.

A complete list of upstream attributions covering libaom, dav1d, SVT-AV1,
libwebrtc, the AV1 bitstream and RTP specifications, and the bundled libaom
test vectors lives in [NOTICE](NOTICE). The pinned upstream commits and local
upstream license paths are tracked in
[third_party/upstream.lock](third_party/upstream.lock) and produced on demand by
`make sync-upstreams`.

## Acknowledgements

This project would not exist without the work of the AV1 community.

- The **Alliance for Open Media** and the contributors to **libaom** - the AV1
  reference encoder/decoder at https://aomedia.googlesource.com/aom - whose
  bitstream syntax, reconstruction logic, default CDFs, dequantization scales,
  transform scan tables, loop-filter/CDEF/restoration/film-grain code, and
  conformance test vectors are the substrate on which goav1 is built. The pinned
  reference is `v3.14.0`.
- The **VideoLAN dav1d authors** - https://code.videolan.org/videolan/dav1d -
  whose decoder architecture, tile/decode pipeline shape, DSP organization, and
  performance practices guide the ongoing decoder optimization work. The pinned
  reference is `1.5.3`.
- The **WebRTC project** at https://webrtc.googlesource.com/src for the AV1 RTP
  payload and depayload behavior that drives the realtime-focused RTP code in
  this repository.

All bugs in the Go port are ours; all good ideas above the bitstream edge belong
to the upstream authors named above.
