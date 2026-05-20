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
- AV1 RTP aggregation header parsing, payload iteration, payload building,
  single-OBU fragmentation, and fragment reassembly.
- Caller-buffer AV1 RTP depacketization into complete OBU spans.
- AV1 sequence header parsing for decoder configuration.
- Incremental decoder stream state over OBUs/RTP payloads.
- Caller-buffer frame plane layout and binding primitives.
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
