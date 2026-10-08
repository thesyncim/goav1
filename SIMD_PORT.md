# Go-native SIMD port

The arm64 Go-native SIMD kernels use `simd/archsimd` behind Go's experimental
`GOEXPERIMENT=simd` build experiment. This document describes how to build and
validate those paths; it does not make a performance claim. Earlier measurements
used a forked toolchain and are not evidence for the official Go release.

## Toolchain and commands

The modules require Go 1.27.0 or newer. Use the official Go distribution. The
`scripts/simdtip.sh` filename is retained for compatibility, but the script now
selects `go` from `PATH` by default or an executable supplied with `GO`:

~~~sh
scripts/simdtip.sh build ./...
scripts/simdtip.sh test ./internal/av1/dsp -run 'SIMD'
scripts/simdtip.sh conformance
GOEXPERIMENT=simd make dryrun-profiles
~~~

The script forces `simd` on, preserves other explicit experiment selections,
and inherits the caller's `GOROOT`, `GOTOOLCHAIN`, and `GOCACHE`. For normal
builds, use the standard Go command without the experiment. SIMD files are
selected by the `goexperiment.simd` build tag.

## Validation

For each kernel change, establish byte-for-byte agreement with an independent
scalar or assembly implementation across edge cases and relevant input ranges.
Then prove that the public dispatch selects the intended path and run the
applicable end-to-end parity gates:

~~~sh
scripts/simdtip.sh build ./...
scripts/simdtip.sh test ./...
scripts/simdtip.sh conformance
GOEXPERIMENT=simd make dryrun-profiles
~~~

The native arm64 CI lane uses the official Go 1.27.x toolchain with
`GOEXPERIMENT=simd` and runs build, tests, and the strict fast and profile
decoder parity gates. Those decoder gates validate exercised decoder inputs;
they do not measure encoder quality or prove universal AV1 conformance.

Measure performance separately on the same official toolchain and hardware for
the SIMD, scalar, and assembly implementations, using representative sizes and
inputs. Confirm the SIMD implementation is live and check allocations before
making performance claims. See [quality and parity](docs/quality-and-parity.md)
for project-level checks.

## Porting guidelines

- Keep architecture and experiment build constraints explicit, and ensure the
  generic or assembly fallback remains available.
- Test output bytes and state with an independent oracle. Do not derive expected
  results from the implementation under test or widen tolerances to mask drift.
- Include tails, unaligned boundaries, extreme values, and supported bit depths
  in differential coverage.
- Benchmark through public dispatch as well as the isolated kernel so the
  measured path matches production use.
- Keep kernels on the existing implementation when SIMD is not byte-exact or
  does not show a repeatable benefit under the official toolchain.
