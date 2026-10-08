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

The full and extended suites may need vector downloads and the oracle test
harness. The generated corpus lanes require local corpus data. A skipped lane
or missing fixture is not a pass.

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

## Development checks

`make ci-local` runs formatting checks, vet, Go tests, allocation checks,
compiler reports, and the trace-zero check. The formatting target reports
unformatted files; use `make fmt-check-strict` when formatting must fail the
command.

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
