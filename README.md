# goav1

[![CI](https://github.com/thesyncim/goav1/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/ci.yml)
[![Lint](https://github.com/thesyncim/goav1/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/lint.yml)
[![Test vectors](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/thesyncim/goav1.svg)](https://pkg.go.dev/github.com/thesyncim/goav1)

**goav1** is a pure-Go AV1 decoder, general-purpose & realtime encoder, and WebRTC RTP/RTCP toolkit.

- **Zero cgo, zero assembly (`.s`/`.S`), zero third-party dependencies** in the root module.
- **Go-native SIMD** (`GOEXPERIMENT=simd` via `simd/archsimd`) for arm64 and amd64, with portable pure-Go fallbacks.
- **Memory-safe & GC-free on hot paths** — `unsafe` is strictly isolated in `internal/unsafeutil` (`scripts/check_unsafe.sh`), with **0 allocs/frame** in steady-state decode and encode.
- **Bit-exact conformance** — strict MD5 parity against `dav1d --filmgrain 1` across 130+ AV1 feature vectors and real-content streams.

## Performance (Apple M4 Max, `GOEXPERIMENT=simd`)

| Workload | Metric | goav1 vs Reference | Steady-State Allocs |
| --- | --- | --- | --- |
| **Encoder** vs `SvtAv1EncApp` (720p 240f, preset 12 RTC CBR, 1T) | Wall / Cycles | **1.39×–1.82× faster** (up to **2.04× fewer cycles**) | **0 allocs/frame** |
| **Encoder** vs `SvtAv1EncApp` (720p 240f, preset 12 RTC CBR, MT) | Wall / Cycles | **2.68× faster** (**1.52× fewer cycles**) | **0 allocs/frame** |
| **Decoder** vs `dav1d` (5-stream 720p/1080p corpus, 1T) | Wall / Cycles | **1.13×** wall / **1.13×** cycles | **0 allocs/frame** |
| **Decoder** vs `dav1d` (5-stream 720p/1080p corpus, 8T) | Wall / Cycles | **1.04×–1.11×** wall / **1.03×–1.06×** cycles (beats `dav1d` on `svtra_mtbike` at **0.83×**) | **0 allocs/frame** |

## AV1 Feature Parity

| Feature | Decoder | Encoder |
| --- | --- | --- |
| **Profiles 0, 1, 2** (8/10/12-bit; 4:2:0, 4:2:2, 4:4:4, Monochrome) | Full | Full (`VideoEncoder`, `HighBitDepthVideoEncoder`, `MonochromeVideoEncoder`, `HighBitDepthMonochromeVideoEncoder`) |
| **Lossless** (qindex 0, WHT, no in-loop filters) | Full | Full (`Lossless: true`, `EncodeI420LosslessKeyframe`) |
| **Rate Control** (CQP, CRF, 1-pass CBR/VBR, 2-pass VBR, live `SetQIndex`/`SetBitrate`) | — | Full (`QIndex`, `CRF`, `TargetBitrate`, `VBR`, `Pass: 1/2`) |
| **GOP & Frame Types** (Key, Inter P, Hidden Alt-Ref, B-frame pyramid, `show_existing_frame`) | Full | Full (`BFrames`, `EncodeBatch`, `Flush`) |
| **Scalability (SVC)** (Temporal & spatial layers, `operating_point_idc`) | Full (`WithOperatingPoint`, `WithAllLayers`, `LayeredDecoder`) | Full (`TemporalLayers`, `RTCEncoder`) |
| **In-Loop Filtering** (Deblocking Loop Filter, CDEF) | Full | Full |
| **Loop Restoration** (Switchable Wiener + SGRProj) | Full | Full (`EnableLoopRestoration`, `EncodeI420LoopRestorationKeyframe`) |
| **Super-Resolution** (Horizontal scaling, denom 9..16) | Full | Full (`SuperResDenominator`, `EncodeI420SuperResKeyframe`) |
| **Film Grain** (Synthesis & noise modeling/signaling) | Full (`WithFilmGrain`) | Full (`FilmGrainLevel`, `EstimateFilmGrainParams`, `SetFilmGrain`) |
| **Screen Content** (Palette coding, IntraBC) | Full | Full (Palette, `EncodeI420IntraBCKeyframe`) |
| **Global Motion** (Header signaling & translation estimation) | Full | Full (`EnableGlobalMotion`, `EstimateGlobalMotionTranslation`) |
| **HDR & Metadata** (`ColorInfo`, `HDR_CLL`, `HDR_MDCV`, `ITUT_T35`, `TIMECODE`, `SCALABILITY`) | Full | Full (`Color`, `Metadata`, `SetMetadata`) |
| **Error Concealment & Live Push Streaming** | Full (`WithErrorConcealment`, `DecodePayload`) | Full |
| **Transport & Framing** (IVF, AV1 RTP, RFC 8285, RTX, RTCP, SDP) | Full | Full |

*Container muxing/demuxing beyond IVF (MP4/ISOBMFF, WebM/MKV) and SRTP/ICE network transport stay with the caller.*

## Install

Requires Go **1.27.4+** (`GOEXPERIMENT=simd` enables arm64/amd64 Go SIMD kernels; without it, portable pure-Go fallbacks are used).

```sh
go get github.com/thesyncim/goav1
GOEXPERIMENT=simd go build ./...
```

## Quick Start

### Decode an IVF stream

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

	frames, err := av1.DecodeIVF(ivf)
	if err != nil {
		log.Fatal(err)
	}
	for i, f := range frames {
		fmt.Printf("frame %d: %dx%d\n", i, f.Width, f.Height)
	}
}
```

For zero-alloc streaming decode, use `NewDecoderFromIVF` + `DecodeNext` or push live low-overhead temporal units with `Decoder.DecodePayload`. Options: `WithWorkers`, `WithMaxFrameDelay`, `WithFrameSizeLimit`, `WithFilmGrain`, `WithOperatingPoint`, `WithAllLayers`, and `WithErrorConcealment`. See [example_decode_simple_test.go](example_decode_simple_test.go).

### Encode an AV1 stream

```go
package main

import (
	"log"

	av1 "github.com/thesyncim/goav1"
)

func main() {
	const width, height = 64, 64
	enc, err := av1.NewVideoEncoder(av1.VideoEncoderConfig{
		Width: width, Height: height, CRF: 30,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer enc.Close()

	frame := av1.I420Frame{
		Y: make([]byte, width*height),
		U: make([]byte, width/2*height/2),
		V: make([]byte, width/2*height/2),
		YStride: width, ChromaStride: width / 2,
		Width: width, Height: height,
	}
	out, err := enc.Encode(frame, false)
	if err != nil {
		log.Fatal(err)
	}

	// Copy out.Data before the next Encode call if retaining it.
	temporalUnit := append([]byte(nil), out.Data...)
	_ = temporalUnit
}
```

For WebRTC RTP packetization, dependency descriptors, and SVC, see [`RTCEncoder`](example_encode_test.go).

## Buffer Ownership

| API | Returned Data |
| --- | --- |
| `DecodeIVF` | Independent copies of visible Y/U/V samples |
| `Decoder.DecodeNext` / `DecodePayload` | Decoder-owned frame-pool memory (valid until next decode call) |
| `VideoEncoder.Encode` | Encoder-owned temporal-unit bytes (valid until next encode call) |
| `NewIVFIterator` / OBU parsers | Views into caller-provided input bytes |

Copy aliased buffers before retaining them or passing them across goroutines. Always `Close()` reusable `Decoder`, `VideoEncoder`, and `RTCEncoder` instances.

## CLI Tools

```sh
make build-cmd
./bin/aom-go-dec -h
```

`aom-go-dec` decodes IVF to raw YUV with `-workers`, `-framedelay`, `-filmgrain`, `-oppoint`, `-conceal`, and `-cpuprofile`. For SVC inspection, see [cmd/dump_svc](cmd/dump_svc).

## Verification

```sh
./scripts/check_unsafe.sh
make test
make ci-local
make testvectors-fast
```

See [docs/quality-and-parity.md](docs/quality-and-parity.md) for the full conformance and benchmark gates, and [SIMD_PORT.md](SIMD_PORT.md) for Go SIMD kernel details.

## References & License

- [Changelog](CHANGELOG.md) · [Quality & Parity](docs/quality-and-parity.md) · [SIMD Port](SIMD_PORT.md) · [Upstream Attribution](UPSTREAM.md) · [Security](SECURITY.md)
- Licensed under **BSD 2-Clause** ([LICENSE](LICENSE)); libaom-derived portions also carry the **AOM Patent License 1.0** ([PATENTS](PATENTS)). See [NOTICE](NOTICE).
