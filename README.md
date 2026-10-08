# goav1

[![CI](https://github.com/thesyncim/goav1/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/ci.yml)
[![Lint](https://github.com/thesyncim/goav1/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/lint.yml)
[![Test vectors](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml/badge.svg?branch=main)](https://github.com/thesyncim/goav1/actions/workflows/testvectors.yml)

goav1 is a pure-Go AV1 decoder with a realtime, WebRTC-oriented encoder and
RTP/RTCP helpers. The root Go module has no third-party module dependencies and
does not use cgo. The project is under active development: decoder parity,
encoder quality, and throughput coverage continue to expand.

## Install

Requires Go 1.27.0 or newer. The native arm64 SIMD paths use the experimental
`GOEXPERIMENT=simd` setting with the official Go 1.27 toolchain.

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

For a reusable decode loop, use `NewDecoderFromIVF` and `DecodeNext`. Frames
returned by `DecodeNext` alias decoder-owned memory and remain valid until the
next decode call. See the executable [decoder examples](example_decode_simple_test.go).

## Encode an I420 stream

`VideoEncoder` accepts same-sized 8-bit I420 frames. Set `QIndex` for fixed
quality, or set both `TargetBitrate` and `Framerate` for CBR rate control.

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

- Decode AV1 from IVF, low-overhead OBU streams, and ordered RTP payloads or
  packets. The decoder includes prediction, reconstruction, and AV1 post-filter
  stages.
- Encode realtime AV1 streams with fixed-quality or CBR control, with
  WebRTC-specific layering and RTP metadata through `RTCEncoder`.
- Parse and build AV1 RTP payloads, RTP extension elements, selected SDP
  capability values, and RTCP packets and feedback.
- Use lower-level IVF, OBU, bitstream, frame, and DSP helpers when an
  integration needs explicit ownership of buffers and work.

The checked-in decoder vectors cover profiles 0 and 1, plus selected profile 2
clips, and include 8-, 10-, and 12-bit content. This describes fixture coverage;
it does not claim every stream in those profiles is supported.

The encoder targets realtime WebRTC use. It is not an offline encoder or a
general replacement for `aomenc`. The package supplies RTP/RTCP primitives and
AV1 capability helpers; callers provide full SDP negotiation, SRTP, jitter
buffering, network I/O, packet timing, and retransmission policy.

Coverage is measured by the checked-in vector and integration gates. Passing
those gates does not establish complete AV1 conformance for every profile,
bit depth, malformed stream, or deployment.

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
