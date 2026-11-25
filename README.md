# goav1

[![CI](https://github.com/thesyncim/goav1/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/ci.yml)
[![Lint](https://github.com/thesyncim/goav1/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/lint.yml)
[![Test vectors](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml)

goav1 is a pure-Go AV1 decoder and encoder with WebRTC RTP/RTCP helpers. The
root Go module has no third-party dependencies, uses zero cgo, and contains zero
handwritten assembly (`.s`/`.S`) — optimized arm64 and amd64 kernels are written
in Go using `simd/archsimd` (`GOEXPERIMENT=simd`), with audited memory safety
(`scripts/check_unsafe.sh`) and zero steady-state GC allocations on hot
decode/encode paths.

## Performance snapshot (Apple M4 Max, `GOEXPERIMENT=simd`)

- **Encoder vs `SvtAv1EncApp` (720p 240f, preset 12 RTC CBR)**: **1.39×–1.82× faster** single-threaded (up to **2.04× fewer cycles**) and **2.68× faster** multi-threaded, with **0.0 allocs/frame** in steady-state 1T and RTC streams.
- **Decoder vs `dav1d` (5-stream 720p/1080p real-content corpus)**: **1.13× wall / 1.13× cycles** single-threaded and **1.04×–1.12× wall / 1.03×–1.06× cycles** at 8 threads, with **0 allocs/frame** in steady state and bit-exact MD5 parity across 130+ AV1 feature vectors and real-content streams.

## Install

Requires the official Go distribution, version 1.27.4 or newer; no toolchain
is pinned. The arm64 and amd64 SIMD paths use the experimental
`GOEXPERIMENT=simd` setting. Build with `GOEXPERIMENT=simd go build ./...` to
enable them. Without it, normal Go fallbacks are used. Codec implementations,
including SIMD paths, are written in Go without handwritten assembly.
See [SIMD validation and measurements](SIMD_PORT.md) for the selected paths.

~~~sh
go get github.com/thesyncim/goav1
~~~

## Decode an IVF file

`DecodeIVF` returns independent, row-packed copies of each visible frame:

~~~go
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
	for i, frame := range frames {
		fmt.Printf("frame %d: %dx%d\n", i, frame.Width, frame.Height)
	}
}
~~~

For a reusable decode loop, use `NewDecoderFromIVF` and `DecodeNext`, or push
live low-overhead temporal units with `Decoder.DecodePayload`. Frames returned
by `DecodeNext` / `DecodePayload` alias decoder-owned memory and remain valid
until the next decode call. Options include `WithWorkers`, `WithMaxFrameDelay`,
`WithFrameSizeLimit`, `WithFilmGrain`, `WithOperatingPoint`, `WithAllLayers`,
and `WithErrorConcealment`. See the executable [decoder examples](example_decode_simple_test.go).

## Encode an AV1 stream

`VideoEncoder` encodes same-sized 8-bit 4:2:0, native 4:2:2 (Profile 2), or
native 4:4:4 (Profile 1) streams, as well as bit-exact 8-bit lossless streams
(`Lossless: true`). Rate control supports fixed quality (`QIndex`), Constant
Rate Factor (`CRF`), 1-pass CBR and VBR (`TargetBitrate`, `Framerate`, `VBR`),
and 2-pass VBR (`Pass: 1` / `Pass: 2`, `FirstPassStats`), plus lookahead
B-frame / hidden Alt-Ref pyramids (`BFrames`, `EncodeBatch`, `Flush`),
automatic film-grain modeling and signaling (`FilmGrainLevel`,
`EstimateFilmGrainParams`, `SetFilmGrain`), horizontal Super-Resolution
(`EncodeI420SuperResKeyframe`), in-loop Wiener Restoration
(`EncodeI420LoopRestorationKeyframe`), `Color` description, `Metadata` (HDR
CLL, HDR MDCV, ITU-T T.35, Timecode), `KeyFrameInterval`, and runtime
`SetQIndex` / `SetBitrate`. For native 10/12-bit or monochrome streams, use
`HighBitDepthVideoEncoder`, `MonochromeVideoEncoder`, or
`HighBitDepthMonochromeVideoEncoder`.

~~~go
package main

import (
	"log"

	av1 "github.com/thesyncim/goav1"
)

func main() {
	const width, height = 64, 64
	enc, err := av1.NewVideoEncoder(av1.VideoEncoderConfig{
		Width: width, Height: height, QIndex: 80,
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

	// Keep the temporal unit beyond the next Encode call.
	temporalUnit := append([]byte(nil), out.Data...)
	_ = temporalUnit
}
~~~

For RTP payloads, dependency descriptors, temporal layers, and WebRTC control,
see [`RTCEncoder` and the executable encoder examples](example_encode_test.go).

## Ownership

| API | Returned data |
| --- | --- |
| `DecodeIVF` | Independent copies of visible Y/U/V samples |
| `Decoder.DecodeNext` | Decoder-owned frame-pool memory, reused by a later decode call |
| `VideoEncoder.Encode` | Encoder-owned temporal-unit bytes, reused by the next encode call |
| `NewIVFIterator` and OBU parsers | Views into caller-provided input bytes |

Copy aliased data before retaining it or passing it to asynchronous work. Close
reusable `Decoder`, `VideoEncoder`, and `RTCEncoder` values when finished.

## Capabilities and scope

- **Decoder (Profiles 0, 1, 2; 8/10/12-bit; 4:2:0, 4:2:2, 4:4:4, monochrome)**:
  Decodes IVF, low-overhead OBU streams (batch or live push via `DecodePayload`),
  and ordered RTP payloads/packets (`DecodeRTPPayload`, `DecodeRTPPacket`,
  `LayeredDecoder` for multi-resolution SVC). Implements full AV1 prediction,
  transforms, reconstruction, loop filter, CDEF, super-resolution, loop
  restoration (Wiener/SGR), film-grain synthesis (toggleable via `WithFilmGrain`),
  operating-point & spatial-layer selection (`WithOperatingPoint`,
  `WithAllLayers`), `ColorInfo` & `FrameMetadata` exposure (HDR CLL, HDR MDCV,
  ITU-T T.35, Timecode, Scalability), and resilient freeze-frame error
  concealment (`WithErrorConcealment`).
- **Encoder**: Full pure-Go AV1 encoder (`VideoEncoder`,
  `HighBitDepthVideoEncoder`, `MonochromeVideoEncoder`,
  `HighBitDepthMonochromeVideoEncoder`, `KeyframeEncoder`, `RTCEncoder`)
  targeting both realtime/WebRTC and general-purpose AV1 encoding — covering
  8/10/12-bit 4:2:0, 4:2:2, 4:4:4, monochrome, and bit-exact lossless
  keyframes/streams; CQP, CRF, 1-pass CBR/VBR, and 2-pass VBR rate control;
  lookahead B-frame / hidden Alt-Ref pyramids (`EncodeBatch`/`Flush`);
  temporal/spatial SVC; screen-content palette coding; horizontal
  Super-Resolution; in-loop Wiener Restoration search & coding; Global Motion
  translation estimation; automatic film-grain modeling & signaling; CDEF,
  loop filter, and cyclic-refresh AQ; color description & `OBU_METADATA`
  emission; and WebRTC dependency descriptors.
- **Transport & Framing**: AV1 RTP payload packetization/depacketization, RFC
  8285 header extensions, RFC 4588 RTX, RTCP feedback (NACK, PLI, FIR, REMB,
  Transport-CC, AV1 layer refresh), packet sequencing, AV1 SDP capability
  helpers, and IVF iteration (`NewIVFIterator`).

The decoder's internal pipeline (frame pools, tile work, post-filter runners,
kernels) is kept under `internal/` and is not part of the public API. Container
demuxing/muxing beyond IVF (MP4/ISOBMFF, WebM/MKV) and WebRTC network/SRTP/ICE
transport stay with the caller.

## Verify changes

Run the commands below when changing decoder, encoder, or transport behavior:

~~~sh
make test
make ci-local
make testvectors-fast
make dryrun-fast
make dryrun-profiles
~~~

`make dryrun-full` runs the broader committed libaom vector suite;
`make dryrun-extended` runs an opt-in diagnostic cohort.
`make webrtc-production` is the larger encoder, WebRTC, and reference-decoder
lane.
Some lanes need reference tools or downloadable vectors. See the
[quality and parity guide](docs/quality-and-parity.md) for what each gate proves
and the performance commands.

## CLI decoder

~~~sh
make build-cmd
./bin/aom-go-dec -h
~~~

The command-line decoder reads IVF and writes decoded YUV. For SVC inspection,
see [cmd/dump_svc](cmd/dump_svc).

## Project references

- [Quality and parity gates](docs/quality-and-parity.md)
- [Go-native SIMD build and validation](SIMD_PORT.md)
- [Current end-to-end decoder and encoder gap versus dav1d and SVT-AV1](docs/e2e-codec-gap-20261124.md)
- [Pinned upstream sources and porting policy](UPSTREAM.md)
- [Executable public API examples](example_test.go), [decode examples](example_decode_simple_test.go),
  and [encoder examples](example_encode_test.go)
- [Change history](CHANGELOG.md)
- [Security policy](SECURITY.md)
- [Package API documentation](https://pkg.go.dev/github.com/thesyncim/goav1)

## License

The source distribution uses the BSD 2-Clause terms in [LICENSE](LICENSE).
The libaom-derived portions also carry the Alliance for Open Media patent
license reproduced in [PATENTS](PATENTS). Third-party attributions are in
[NOTICE](NOTICE).
