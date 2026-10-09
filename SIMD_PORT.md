# Go-native SIMD port

The arm64 and amd64 Go-native SIMD kernels use `simd/archsimd` behind Go's experimental
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
for project-level checks. The current arm64 kernel results, reproduction
commands, and claim limits are in the
[Go 1.27 SIMD measurement note](docs/go127-simd-performance.md).

## Remaining assembly (2026-10-09 snapshot)

This tree contains two assembly source files, both AMD64, with six `TEXT`
symbols. No ARM64 assembly source files remain. These are source counts, not
runtime shares or evidence that the Go SIMD replacements are faster.

| Source | Symbols | Remaining work |
| --- | ---: | --- |
| `internal/av1/transform/dct4lane_avx2_amd64.s` | 4 | Four-row/four-column inverse DCT32 and DCT64 kernels; replace the live adapters and preserve stage-range fallbacks |
| `internal/av1/dsp/cpu/cpuid_amd64.s` | 2 | CPUID/XGETBV feature detection; remove only after the remaining assembly dispatch no longer requires it |

The experiment build enables architecture-specific Go SIMD replacements.
Default and `purego` builds use scalar Go where assembly has been removed;
unsupported CPUs, shapes, and numeric ranges retain checked scalar fallbacks.
Removing assembly does not establish performance parity: whole-decoder and
encoder measurements must still demonstrate the result independently.

The recovered October 9 motion and loop-filter optimizations are evaluated
against the previous PR checkpoint, not against dav1d or SVT-AV1. See the
[measurement note](docs/go127-simd-performance.md) for the sampling protocol
and remaining comparison limits.

Do not delete an entire assembly file merely because one SIMD dispatch replaces
one of its symbols. Check default builds, experiment builds, other architectures,
assembly-to-assembly calls, and numeric/shape fallbacks. Remove replaced bodies
once those production references are gone, retaining an independent correctness
oracle without keeping otherwise-dead production code.

## Porting guidelines

- Keep architecture and experiment build constraints explicit. Standard builds
  use the Go fallback where SIMD replaces assembly, and retain assembly where
  SIMD has not shown a repeatable benefit.
- Test output bytes and state with an independent oracle. Do not derive expected
  results from the implementation under test or widen tolerances to mask drift.
- Include tails, unaligned boundaries, extreme values, and supported bit depths
  in differential coverage.
- Benchmark through public dispatch as well as the isolated kernel so the
  measured path matches production use.
- Keep kernels on the existing implementation when SIMD is not byte-exact or
  does not show a repeatable benefit under the official toolchain.
