# Quality and parity

goav1 uses independent fixture and reference-decoder checks for behavior, and
separate benchmarks for speed. A passing decoder vector set is evidence for the
exercised inputs and modes; it does not certify every AV1 profile, bit depth,
malformed input, or deployment.

## Decoder gates

The strict dry-run lanes compare decoded frame output with libaom-derived
expected digests. Use focused cohorts while iterating, then run the broader
gates that cover the changed behavior before making a wider parity claim.

| Command | Coverage and role |
| --- | --- |
| `make test` | Run the root module's Go tests. |
| `make testvectors-fast` | Run the maintained fast test-vector slice, including its oracle-tagged checks. |
| `make dryrun-fast` | Strict per-frame MD5 dry-run for the fast libaom cohort. |
| `make dryrun-profiles` | Strict per-frame MD5 checks for the vendored profile clips. |
| `make dryrun-full` | Strict dry-run over the full committed remote libaom manifest; broader and slower. |
| `make dryrun-extended` | Opt-in diagnostic cohort with additional vectors, including larger and multi-quantizer cases. |
| `make dryrun-corpus` | Check the locally generated real-content corpus, when present. |
| `make dryrun-external-corpus` | Check caller-supplied IVF corpora with supported digest sidecars. |
| `make feature-sweep` | Encode about 180 small synthetic clips with `aomenc` and `SvtAv1EncApp`, one coding tool or header mode each, and compare the decoder's output at one and at eight workers, and the layered decoder's, with `dav1d`. Needs `aomenc`, `aomdec`, `dav1d` and `ffmpeg`; the SVT-AV1 cases run when its encoder is installed. |

The full and extended suites may need vector downloads and the oracle test
harness. The generated corpus lanes require local corpus data. A skipped lane
or missing fixture is not a pass.

The published libaom vectors do not use every coding tool. The feature sweep
exists because two bugs passed every vector gate: quantizer matrices were
applied in the wrong orientation, and inter frames after an intra block copy
key frame inherited the wrong entropy contexts. Run it after changing
coefficient decoding, dequantization, entropy-context handling, prediction or
post-filter code, and add a case when a tool turns out to be uncovered.

## Encoder and WebRTC evidence

Decoder MD5 gates do not establish encoder compression quality or WebRTC
interoperability. The encoder's output and WebRTC integration have separate
reference and production gates:

- `make webrtc-reference` requires both `aomdec` and `dav1d`. It checks internal
  encoder reconstructions against output from both decoders, then checks public
  encoder and WebRTC output with the same references. It has no browser
  dependency.
- `make webrtc-production` includes those reference checks plus broader
  encoder, decoder, RTP/RTCP, control, and loss-handling tests, the browser-push
  RTP integration, and live browser playback. It requires the reference
  decoders and browser setup.

Check target output for skipped or unavailable checks before treating a lane as
complete.

The encoder is scoped to realtime WebRTC AV1. It is not intended as a general
offline encoder or an `aomenc` replacement. Current tuning and broader
reference coverage remain active work.

## Go-native SIMD

The module minimum is Go 1.27.4. The arm64 and amd64 Go-native SIMD paths
require the experimental `GOEXPERIMENT=simd` setting with the official Go
toolchain. The native arm64 and amd64 CI lanes enable it for build, tests, race
and checkptr runs of the kernel packages, and the strict fast/profile decoder
gates. See the [SIMD port guide](../SIMD_PORT.md) for local commands and
kernel-level validation expectations.

## Development checks

`make ci-local` runs the gofmt check, vet, the `unsafe` allowlist check, Go
tests, allocation checks, compiler reports, and the trace-zero check.

Before changing a decoder DSP or transform path, run its focused conformance
probe:

~~~sh
make test-motion-conformance
make test-transform-conformance
~~~

Performance evidence is separate:

~~~sh
make alloc
make compiler-reports
make trace-zero
make bench
make bench-cross
~~~

The allocation and compiler targets guard known hot paths. The benchmarks
measure throughput; `make bench-cross` is a tracking tool, not a conformance
gate, and may skip external decoder columns when the tools are unavailable.
For publishable encoder or decoder comparisons, use the protocol and manifest
checks described by the relevant scripts and quality benchmark command targets
in the Makefile.

## Parity workflow

When changing codec behavior, use the exact pinned upstream source and the
existing independent fixture or oracle. Compare the relevant output bytes,
lengths, return values, and state. Keep the test focused on the boundary being
changed, then run the broader applicable gate. Do not regenerate expected
output from goav1 or widen a digest or numeric tolerance to hide a mismatch.

Record unsupported modes and missing evidence as gaps. A benchmark, successful
build, encoder-to-decoder round trip, or parser unit test does not replace an
independent parity check for a different behavior.

See [UPSTREAM.md](../UPSTREAM.md) for source pins, local clone commands, and
porting policy.
